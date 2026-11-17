"""从正式 ONNX 权重准备隔离 GGUF 编码器；不创建推理会话。

官方 mmproj 仅提供名称、布局和元数据模板。所有实际权重取自指定的
ONNX / external-data 文件，保留它们的 FP16 / FP32 精度，并逐张量复核。
位置编码只保留正式原子前端实际使用的 13 行，不引入其他位置值。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import prepare_external_weights as proto


OFFICIAL_SHA256 = "8882e9ddab3186f9aa71b1417c847177913e1466655ac944cf86e9b846735d62"
DTYPES = {1: np.dtype("<f4"), 7: np.dtype("<i8"), 10: np.dtype("<f2")}


def digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def graph_nodes(path, view):
    nodes = []
    with path.open("rb") as stream:
        for item in view["graph_fields"]:
            if item.number != 1:
                continue
            inner = list(proto.fields(stream, item.payload, item.end))
            values = {key: [proto.small_bytes(stream, part).decode("utf-8")
                            for part in inner if part.number == key and part.wire == 2]
                      for key in (1, 2, 3, 4)}
            nodes.append({"inputs": values[1], "outputs": values[2],
                          "name": values[3][0] if values[3] else "",
                          "op": values[4][0]})
    return nodes


def tensor_array(path, tensor):
    dtype = DTYPES[tensor["dtype"]]
    count = int(np.prod(tensor["dimensions"], dtype=np.int64))
    external = tensor["external"]
    if external:
        location = external["location"]
        relative = Path(location)
        if relative.is_absolute() or len(relative.parts) != 1 or ":" in location:
            raise ValueError("不允许外部权重越出指定模型目录")
        source = proto.checked_source(path.parent / relative)
        offset = int(external.get("offset", "0"))
        length = int(external["length"])
    else:
        raw = tensor["raw_field"]
        if raw is None:
            raise ValueError(f"张量缺少 raw_data：{tensor['name']}")
        source, offset, length = path, raw.payload, raw.end - raw.payload
    if offset < 0 or length != count * dtype.itemsize or offset + length > source.stat().st_size:
        raise ValueError(f"张量字节范围不一致：{tensor['name']}")
    return np.memmap(source, mode="r", dtype=dtype, offset=offset,
                     shape=tuple(tensor["dimensions"]))


def learned_name(name):
    direct = {"ln_post.weight": "a.post_ln.weight", "ln_post.bias": "a.post_ln.bias",
              "proj1.weight": "mm.a.mlp.1.weight", "proj1.bias": "mm.a.mlp.1.bias",
              "proj2.weight": "mm.a.mlp.2.weight", "proj2.bias": "mm.a.mlp.2.bias"}
    if name in direct:
        return direct[name]
    conv = re.fullmatch(r"conv2d([123])\.(weight|bias)", name)
    if conv:
        return f"a.conv2d.{conv[1]}.{conv[2]}"
    match = re.fullmatch(r"layers\.(\d+)\.(.+)\.(weight|bias)", name)
    if not match:
        raise ValueError(f"未知学习参数名：{name}")
    block, part, suffix = match.groups()
    parts = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
             "self_attn.v_proj": "attn_v", "self_attn.out_proj": "attn_out",
             "self_attn_layer_norm": "ln1", "final_layer_norm": "ln2",
             "fc1": "ffn_up", "fc2": "ffn_down"}
    return f"a.blk.{block}.{parts[part]}.{suffix}"


def build_mapping(model_root):
    mapping, model_views, ignored, nodes_by_file = {}, {}, [], {}
    for part in ("frontend", "backend"):
        path = proto.checked_source(model_root / f"qwen3_asr_encoder_{part}.onnx")
        view = proto.inspect(path)
        model_views[path] = view
        tensors = {tensor["name"]: tensor for tensor in view["tensors"]}
        nodes = graph_nodes(path, view)
        nodes_by_file[path.name] = nodes
        semantic = {name: name for name in tensors if not name.startswith(("val_", "unsqueeze_"))}
        transforms = {}
        for node in nodes:
            if node["op"] != "MatMul":
                continue
            weights = [name for name in node["inputs"] if name in tensors]
            if len(weights) != 1:
                continue
            weight = weights[0]
            consumers = [next_node for next_node in nodes if node["outputs"][0] in next_node["inputs"]
                         and next_node["op"] in ("Add", "BiasGelu")]
            biases = [name for next_node in consumers for name in next_node["inputs"]
                      if name in tensors and name.endswith(".bias")]
            if part == "frontend":
                if weight != "val_27":
                    raise ValueError("前端投影矩阵发生变化")
                semantic[weight] = "conv_out.weight"
            elif len(biases) == 1:
                semantic[weight] = biases[0][:-4] + "weight"
            else:
                raise ValueError(f"无法依据 MatMul→Add 确定矩阵用途：{weight}")
            transforms[weight] = "transpose_2d"
        if part == "frontend":
            final = nodes[-1]
            if final["op"] != "Add" or final["inputs"] != ["linear", "unsqueeze_1"]:
                raise ValueError("正式前端的位置编码 Add 发生变化")
            semantic["unsqueeze_1"] = "position"
        for name, tensor in tensors.items():
            if name not in semantic:
                controls = {("frontend", "val_26"): ([1, 13, 7680], "frontend reshape to batch/time/conv channels"),
                            ("frontend", "val_0"): ([1], "frontend add channel dimension at axis 1"),
                            ("backend", "val_13"): ([16], "attention head count"),
                            ("backend", "val_14"): ([64], "attention head dimension"),
                            ("backend", "val_32"): (0.125, "attention scale 1/sqrt(64)"),
                            ("backend", "val_37"): ([1024], "flatten attention output to hidden size")}
                if (part, name) not in controls:
                    raise ValueError(f"未映射的未知图常量：{name}")
                expected, purpose = controls[part, name]
                values = tensor_array(path, tensor)
                if not np.array_equal(values, np.asarray(expected)):
                    raise ValueError(f"图常量值发生变化：{name}")
                ignored.append({"model": path.name, "name": name, "dtype": tensor["dtype"],
                                "dimensions": tensor["dimensions"], "reason": "graph_control_constant",
                                "value": values.tolist(), "purpose": purpose, "payload_sha256": digest(values)})
                continue
            source_name = semantic[name]
            target = ("a.position_embd.weight" if source_name == "position" else
                      "a.conv_out.weight" if source_name == "conv_out.weight" else learned_name(source_name))
            if target in mapping:
                raise ValueError(f"重复 GGUF 张量：{target}")
            operation = transforms.get(name, "identity")
            if source_name == "position":
                operation = "remove_batch_dimension_13_positions"
            elif source_name.startswith("conv2d") and source_name.endswith(".bias"):
                operation = "reshape_conv_bias_480_1_1"
            mapping[target] = (path, tensor, operation, source_name)
    return mapping, model_views, ignored, nodes_by_file


def mapped_array(path, tensor, operation):
    array = tensor_array(path, tensor)
    if operation == "transpose_2d":
        array = array.T
    elif operation == "remove_batch_dimension_13_positions":
        if array.shape != (1, 13, 1024):
            raise ValueError("位置编码不符合正式 13×1024 前端")
        array = array[0]
    elif operation == "reshape_conv_bias_480_1_1":
        array = array.reshape(480, 1, 1)
    return np.ascontiguousarray(array)


def official_float(tensor, gguf):
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        words = tensor.data.view(np.uint16) if tensor.data.dtype == np.uint8 else tensor.data
        return (words.astype(np.uint32) << 16).view(np.float32)
    if tensor.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
        return tensor.data
    raise ValueError(f"模板出现不支持的量化张量：{tensor.name}")


def prepare(args):
    sys.path.insert(0, str(proto.assert_no_reparse(args.gguf_python)))
    import gguf
    model_root = proto.assert_no_reparse(args.model_root)
    template = proto.checked_source(args.template)
    output_root = proto.assert_no_reparse(args.output_root)
    output = proto.assert_no_reparse(args.output)
    receipt_path = output.with_suffix(".verification.json")
    if not output.is_relative_to(output_root) or output.suffix != ".gguf":
        raise ValueError("派生文件必须位于明确指定的隔离输出目录")
    if output_root.is_relative_to(model_root) or model_root.is_relative_to(output_root):
        raise ValueError("隔离输出目录不能与正式模型目录重叠")
    if output.exists() or receipt_path.exists():
        raise ValueError("拒绝覆盖现有模型或验证记录")
    if proto.sha(template) != OFFICIAL_SHA256:
        raise ValueError("官方模板 SHA256 与已核验文件不同")
    mapping, views, ignored, nodes = build_mapping(model_root)
    reader = gguf.GGUFReader(template)
    if set(mapping) != {tensor.name for tensor in reader.tensors}:
        raise ValueError("正式 ONNX 与官方 GGUF 的学习参数集合不一致")
    if len(mapping) != 398 or len(ignored) != 6:
        raise ValueError("正式编码器张量数量发生变化")
    source_files = set(views)
    for path, view in views.items():
        source_files.update(path.parent / tensor["external"]["location"]
                            for tensor in view["tensors"] if tensor["external"])
    source_hashes = {str(path): proto.sha(path) for path in sorted(source_files)}
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(output, arch="clip")
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key in ("general.architecture", "general.file_type"):
            continue
        writer.add_key_value(key, field.contents(), field.types[0],
                             field.types[-1] if len(field.types) > 1 else None)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    records = []
    for official in reader.tensors:
        path, tensor, operation, source_name = mapping[official.name]
        array = mapped_array(path, tensor, operation)
        reference = official_float(official, gguf)
        if official.name == "a.position_embd.weight":
            reference = reference[:13]
        if array.shape != reference.shape:
            raise ValueError(f"布局不匹配：{official.name}: {array.shape} / {reference.shape}")
        rounded_reference = reference.astype(array.dtype)
        equal = bool(np.array_equal(array, rounded_reference))
        unequal = rounded_reference != array
        difference = rounded_reference.astype(np.float32) - array.astype(np.float32)
        records.append({"gguf_name": official.name, "onnx_model": path.name,
                        "onnx_name": tensor["name"], "semantic_name": source_name,
                        "onnx_dtype": tensor["dtype"], "onnx_dimensions": tensor["dimensions"],
                        "transformation": operation, "output_dtype": str(array.dtype),
                        "output_shape": list(array.shape), "onnx_payload_sha256": digest(tensor_array(path, tensor)),
                        "mapped_payload_sha256": digest(array), "template_dtype": official.tensor_type.name,
                        "template_matches_after_local_dtype_rounding": equal,
                        "template_unequal_elements_after_local_dtype_rounding": int(np.count_nonzero(unequal)),
                        "template_max_abs_difference_after_local_dtype_rounding": float(np.max(np.abs(difference))),
                        "template_rounded_payload_sha256": digest(rounded_reference)})
        writer.add_tensor(official.name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    derived = gguf.GGUFReader(output)
    derived_by_name = {tensor.name: tensor for tensor in derived.tensors}
    for record in records:
        written = derived_by_name[record["gguf_name"]]
        if digest(written.data) != record["mapped_payload_sha256"]:
            raise ValueError(f"GGUF 写入后的权重不一致：{written.name}")
        if written.data.dtype != np.dtype(record["output_dtype"]):
            raise ValueError(f"GGUF 写入后的精度不一致：{written.name}")
        record["derived_payload_verified"] = True
    if source_hashes != {path: proto.sha(Path(path)) for path in source_hashes}:
        raise ValueError("准备期间正式模型文件发生变化")
    receipt = {"schema": 1, "execution_state": "prepared_only_no_runtime_session",
               "inference_calls": 0, "runtime_sessions": 0, "source_model_root": str(model_root),
               "source_file_sha256": source_hashes, "template": str(template),
               "template_sha256": OFFICIAL_SHA256, "output": str(output),
               "output_sha256": proto.sha(output), "output_bytes": output.stat().st_size,
               "tensor_count": len(records), "formal_initializer_count": sum(len(view["tensors"]) for view in views.values()),
               "covered_formal_initializer_count": len(mapping), "mapped_graph_constant_count": len(ignored),
               "dtype_counts": dict(Counter(row["output_dtype"] for row in records)),
               "all_derived_tensors_match_formal_onnx": True,
               "all_template_tensors_match_after_local_dtype_rounding": all(row["template_matches_after_local_dtype_rounding"] for row in records),
               "position_policy": "Exactly 13 FP16 positions from formal ONNX; reset for each 100-mel-frame chunk",
               "graph_control_initializers": ignored, "tensor_verification": records,
               "formal_graph_nodes": nodes}
    with proto.guarded_new_file(receipt_path, output_root, text=True) as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return {key: receipt[key] for key in ("execution_state", "output", "output_sha256", "output_bytes",
                                        "tensor_count", "dtype_counts", "all_template_tensors_match_after_local_dtype_rounding")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--gguf-python", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
