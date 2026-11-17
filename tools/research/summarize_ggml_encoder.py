"""汇总同模型 GGML/Vulkan 编码器的已完成正式轮；不启动应用或推理。

逐轮原始 JSON、早期失败诊断和后台快照均保持原样。统计只采用显式指定的
无 profile、无 embedding 写入轮次，显存按 LUID/physical/duplicate 分开。
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "evidence/ggml-vulkan-encoder-20261117"
FIELDS = ("duration", "text", "text_accu", "tokens", "timestamps")
RAW_FIELDS = ("text", "token_ids", "n_prefill", "n_generate", "is_aborted")
MEMORY = ("private_bytes", "private_working_set", "working_set")
TIMES = ("t_encode", "t_prefill", "t_generate", "t_context")
MIB = 1048576


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def stats(values):
    if not values or any(not finite(value) for value in values):
        return {"available": False, "median": None, "min": None, "max": None, "values": values}
    return {"available": True, "median": statistics.median(values), "min": min(values),
            "max": max(values), "mean": statistics.mean(values), "values": values}


def compare(old, new, unit):
    a, b = stats(old), stats(new)
    valid = a["available"] and b["available"]
    difference = a["median"] - b["median"] if valid else None
    return {"unit": unit, "baseline": a, "candidate": b,
            "absolute_reduction": difference,
            "reduction_percent": difference / a["median"] * 100 if valid and a["median"] else None}


def background(path, expected_pid, rounds, candidate_side="candidate"):
    rows = load(path)
    require(candidate_side != "baseline", "候选后台标签不能与 baseline 相同")
    mapping = {"baseline": "baseline", candidate_side: "candidate"}
    wanted = {(str(index), side, phase) for index in range(1, rounds + 1)
              for side in ("baseline", "candidate") for phase in ("before", "after")}
    seen, unexpected, source_labels = set(), [], []
    for row in rows:
        require(row["side"] in mapping, "后台进程快照含未配置的 side 标签")
        side = mapping[row["side"]]
        key = (str(row["round"]), side, row["phase"])
        source_labels.append({"round": str(row["round"]), "side_original": row["side"],
                              "side_normalized": side, "phase": row["phase"]})
        require(key not in seen, "后台进程快照标签重复")
        seen.add(key)
        processes = row.get("processes")
        require(isinstance(processes, list), "后台进程快照缺少进程数组")
        for process in processes:
            if process.get("ProcessId") != expected_pid:
                unexpected.append({"snapshot": key, "process": process})
        require(any(process.get("ProcessId") == expected_pid for process in processes), "快照缺少预期正常托盘进程")
    require(seen == wanted, "后台进程快照未完整覆盖所有正式轮开始和结束")
    require(not unexpected, "后台快照含预期托盘以外的正式应用进程")
    return {"checked": True, "path": str(path), "sha256": sha(path), "snapshots": len(rows),
            "expected_tray_pid": expected_pid, "unexpected_formal_processes": unexpected,
            "side_normalization": mapping, "source_snapshot_labels": source_labels,
            "formal_worker_seen": False,
            "scope": "逐轮开始/结束的既有后台快照；仅正常正式托盘进程，无正式模型 worker 记录"}


def adapter_key(row):
    return (row["adapter_luid"], row.get("physical_adapter_index"), row.get("duplicate_instance_index"))


def gpu_values(run, point):
    if point == "sampled_peak":
        rows = run["sampled_gpu_peak"]["adapters"]
    else:
        rows = next(stage for stage in run["stages"] if stage["stage"] == point)["native_gpu"]["gpu_process_memory"]["rows"]
        rows = [row for row in rows if row["pid"] == run["worker_pid"]]
    output = {}
    for row in rows:
        key = adapter_key(row)
        require(key not in output, "同快照存在重复 GPU 计数身份")
        require(not row.get("errors"), "GPU 行含采集错误")
        require(all(finite(row.get(name)) and row[name] >= 0 for name in ("dedicated_bytes", "shared_bytes")),
                "GPU 计数缺失或无效")
        output[key] = row
    return output


def validate_run(path, candidate):
    data = load(path)
    for flag in ("passed", "complete", "execution_passed", "measurement_passed", "gpu_measurement_passed", "cleanup_confirmed"):
        require(data.get(flag) is True, f"正式轮未通过 {flag}：{path.name}")
    require(not data.get("error") and not data.get("cleanup_error"), "正式轮含执行/清理错误")
    require(data.get("diagnostic") is False, "诊断轮不得混入正式统计")
    require(not data.get("encoder_embeddings") and not data.get("load_profiles") and not data.get("aligner_profiles"),
            "正式轮含 embedding 写入或 profile 记录")
    env = data.get("research_environment", {})
    require("YYUR_ENCODER_EMBEDDINGS_DIR" not in env and "YYUR_PROFILE_LOAD" not in env, "正式轮启用写盘诊断")
    require(not data.get("loading_policy_changed"), "本轮不得改变加载策略")
    require(data.get("vulkan_decoder_verified") is True, "解码器 Vulkan 路由未确认")
    if candidate:
        require(env.get("YYUR_ENCODER_RESEARCH_PROVIDER") == "ggml_vulkan" and data.get("gpu_encoder_verified") is True,
                "候选 GGML/Vulkan 编码器路由未确认")
    require(len(data["samples"]) == 5 and len(data["stages"]) == 6, "正式轮须有五条请求及六个阶段快照")
    require(not data.get("sampling_errors") and not data.get("gpu_sampling_errors"), "正式采样含错误")
    for row in data["samples"]:
        require(all(key in row["result"] for key in FIELDS), "识别结果缺少五个字段")
        require(all(key in row["metrics"] for key in RAW_FIELDS), "原生结果字段缺失")
        require(finite(row["latency_seconds"]) and row["latency_seconds"] > 0, "请求耗时无效")
        require(all(finite(row["metrics"].get(key)) and row["metrics"][key] >= 0 for key in TIMES), "原生耗时无效")
    require(abs(sum(row["latency_seconds"] for row in data["samples"]) - data["total_request_seconds"]) < 1e-9,
            "五条请求总耗时与逐条不符")
    require(abs(data["model_ready_seconds"] + data["first_request_seconds"]
                - data["model_ready_plus_first_request_seconds"]) < 1e-9, "ready+first 耗时不一致")
    for point in ("ready", data["samples"][-1]["phase"]):
        gpu_values(data, point)
    gpu_values(data, "sampled_peak")
    return data


def summarize(args):
    runs_dir = args.runs_dir.resolve(strict=True)
    paths = {"baseline": [runs_dir / f"{args.baseline_prefix}{index}.json" for index in range(1, args.rounds + 1)],
             "candidate": [runs_dir / f"{args.candidate_prefix}{index}.json" for index in range(1, args.rounds + 1)]}
    runs = {side: [validate_run(path, side == "candidate") for path in values] for side, values in paths.items()}
    all_runs = [run for values in runs.values() for run in values]
    for key in ("manifest_sha256", "workload_sha256", "host_sample_target_ms", "gpu_sample_target_ms"):
        require(len({run[key] for run in all_runs}) == 1, f"两端输入或采样口径不一致：{key}")
    settings = [{key: value for key, value in run["settings"].items() if key != "app_dir"} for run in all_runs]
    require(all(value == settings[0] for value in settings), "模型或配置在正式轮间变化")
    for side in runs:
        require(len({run["binary_sha256"] for run in runs[side]}) == 1, "同一端 EXE 身份变化")
    phases = [row["phase"] for row in runs["baseline"][0]["samples"]]
    require(phases[0] == "warmed_first_short" and "file_60s" in phases, "请求顺序或首句命名无效")
    result = {
        "label": args.label, "rounds_per_side": args.rounds, "passed": True,
        "formula": "reduction = (baseline_median - candidate_median) / baseline_median * 100; positive means lower",
        "model": settings[0]["asr_model"], "settings_without_app_path": settings[0],
        "background": background(args.background, args.expected_background_pid, args.rounds,
                                 args.background_candidate_side),
        "manifest_sha256": all_runs[0]["manifest_sha256"], "workload_sha256": all_runs[0]["workload_sha256"],
        "binary_sha256": {side: values[0]["binary_sha256"] for side, values in runs.items()},
        "evidence": [{"side": side, "path": str(path), "sha256": sha(path)} for side in paths for path in paths[side]],
        "timing": {}, "requests": [], "memory": {}, "gpu": [], "cpu": {}, "quality": {},
        "sampling_scope": "Host 25ms / WDDM GPU 100ms target; observed sampled maxima, process-local tree and PID/LUID GPU counters separately. Windows/driver caches retained.",
        "loading_policy_changed": False, "diagnostic_runs_included": False,
        "raw_transcripts_in_summary": False,
        "preserved_early_gpu_rejection": str(STUDY / "runs/baseline-diagnostic.json"),
    }
    if args.native_receipt:
        result["native_staging_receipt"] = {"path": str(args.native_receipt), "sha256": sha(args.native_receipt),
                                           "data": load(args.native_receipt)}
    for name in ("model_ready_seconds", "model_ready_plus_first_request_seconds", "total_request_seconds", "warmup_seconds"):
        result["timing"][name] = compare(*[[run[name] for run in runs[side]] for side in ("baseline", "candidate")], "seconds")
    for name in TIMES:
        result["timing"][name + "_sum"] = compare(*[[sum(row["metrics"][name] for row in run["samples"])
                                                       for run in runs[side]] for side in ("baseline", "candidate")], "seconds")
    quality_rows, public = [], {}
    for index, phase in enumerate(phases):
        original = runs["baseline"][0]["samples"][index]
        workload = {key: original[key] for key in ("phase", "id", "source", "seed", "pcm_sha256")}
        row = {**workload, "timing": {"latency_seconds": compare(
            *[[run["samples"][index]["latency_seconds"] for run in runs[side]] for side in ("baseline", "candidate")], "seconds")}}
        for name in TIMES:
            row["timing"][name] = compare(*[[run["samples"][index]["metrics"][name] for run in runs[side]]
                                             for side in ("baseline", "candidate")], "seconds")
        for a, b in zip(runs["baseline"], runs["candidate"]):
            left, right = a["samples"][index], b["samples"][index]
            require(all(left[key] == right[key] == workload[key] for key in workload), "输入/种子/顺序在正式轮间变化")
            require(all(left[key] == workload[key] for key in workload), "基线输入身份变化")
            differences = [key for key in FIELDS if left["result"][key] != right["result"][key]]
            raw = [key for key in RAW_FIELDS if left["metrics"][key] != right["metrics"][key]]
            quality_rows.append({"phase": phase, "differences": differences, "raw_differences": raw})
        public[phase] = {}
        for side in runs:
            annotations = [run["samples"][index].get("public_text_cer") for run in runs[side]]
            if all(value is not None for value in annotations):
                edits = sum(value["edits"] for value in annotations)
                chars = sum(value["reference_characters"] for value in annotations)
                public[phase][side] = {"edits": edits, "reference_characters": chars, "cer": edits / chars,
                                       "round_cer": [value["cer"] for value in annotations]}
            else:
                require(all(value is None for value in annotations), "公开参考覆盖在轮次之间变化")
                public[phase][side] = None
        result["requests"].append(row)
    result["quality"] = {"paired_requests": len(quality_rows), "five_fields": list(FIELDS), "raw_fields": list(RAW_FIELDS),
                         "all_exact": all(not row["differences"] and not row["raw_differences"] for row in quality_rows),
                         "different_requests": sum(bool(row["differences"] or row["raw_differences"]) for row in quality_rows),
                         "comparisons": quality_rows, "public_cer": public,
                         "file_60s_ground_truth": None,
                         "ground_truth_scope": "Three public short clips plus the repeated short clip; 60s unannotated slice is regression-only"}
    tail = phases[-1]
    for point in ("sampled_peak", "ready", tail):
        result["memory"][point] = {}
        for name in MEMORY:
            values = [[run["sampled_peak_totals"][name] if point == "sampled_peak" else
                       next(stage for stage in run["stages"] if stage["stage"] == point)["totals"][name]
                       for run in runs[side]] for side in ("baseline", "candidate")]
            result["memory"][point][name] = compare(*values, "bytes")
    adapters = set().union(*(set(gpu_values(run, "sampled_peak")) for run in all_runs))
    for key in sorted(adapters, key=str):
        item = {"adapter_luid": key[0], "physical_adapter_index": key[1], "duplicate_instance_index": key[2], "points": {}}
        for point in ("sampled_peak", "ready", tail):
            maps = {side: [gpu_values(run, point).get(key) for run in runs[side]] for side in runs}
            item["points"][point] = {
                name: compare(*[[value.get(name) if value else None for value in maps[side]]
                                for side in ("baseline", "candidate")], "bytes")
                for name in ("dedicated_bytes", "shared_bytes")}
        result["gpu"].append(item)
    result["gpu_missing_observations"] = {
        side: [{"worker_pid": run["worker_pid"], "worker_not_observed_sample_indices": run["sampled_gpu_peak"]["worker_not_observed_sample_indices"],
                "adapters": [{key: value for key, value in adapter.items() if key in (
                    "adapter_luid", "physical_adapter_index", "duplicate_instance_index", "not_observed_sample_indices", "valid_samples")}
                    for adapter in run["sampled_gpu_peak"]["adapters"]]} for run in runs[side]] for side in runs}
    result["cpu"]["job_cpu_seconds"] = compare(*[[run.get("job_cpu_seconds") for run in runs[side]]
                                                  for side in ("baseline", "candidate")], "cpu_seconds")
    result["cpu"]["job_cpu_note"] = "Formal wrapper did not collect Job accounting; null is retained. Alive-process stage CPU is available separately."
    for point in ("ready", tail, "ready_to_tail"):
        values = []
        for side in ("baseline", "candidate"):
            rows = []
            for run in runs[side]:
                stages = {stage["stage"]: stage for stage in run["stages"]}
                value = stages[tail]["cpu_seconds_total"] - stages["ready"]["cpu_seconds_total"] if point == "ready_to_tail" else stages[point]["cpu_seconds_total"]
                rows.append(value)
            values.append(rows)
        result["cpu"]["alive_tree_" + point] = compare(*values, "cpu_seconds")
    result["cpu"]["alive_tree_scope"] = "Summed lifetime user+kernel CPU for currently alive service/worker at phase snapshots. Ready-to-tail excludes load; includes in-between observer/IPC overhead."
    return result


def value_range(item, scale=1, digits=3):
    if not item["available"]:
        return "未采集"
    return f"{item['median']/scale:.{digits}f} [{item['min']/scale:.{digits}f}～{item['max']/scale:.{digits}f}]"


def row(label, metric, scale=1):
    percent = metric["reduction_percent"]
    change = f"{percent:+.2f}%" if percent is not None else "基线为0/未采集"
    return f"| {label} | {value_range(metric['baseline'], scale)} | {value_range(metric['candidate'], scale)} | {change} |"


def render(data):
    def direction(metric):
        percent = metric["reduction_percent"]
        return ("降低" if percent >= 0 else "增加") + f"{abs(percent):.2f}%" if percent is not None else "保持零值/未采集"
    active_gpu = max(data["gpu"], key=lambda item: item["points"]["sampled_peak"]["dedicated_bytes"]["baseline"]["median"] or 0)
    lines = [f"# 同模型 GGML/Vulkan 编码器研究草稿：{data['label']}", "",
             "本稿汇总已完成的隔离同机对照，供后续候选与主报告整合。正式安装未改动。",
             f"五条完整请求合计{direction(data['timing']['total_request_seconds'])}；峰私有提交{direction(data['memory']['sampled_peak']['private_bytes'])}，峰总工作集{direction(data['memory']['sampled_peak']['working_set'])}。",
             f"适配器{active_gpu['adapter_luid']}的dedicated显存峰值{direction(active_gpu['points']['sampled_peak']['dedicated_bytes'])}。速度收益与私有提交、显存代价应同时评价。",
             f"模型为 Qwen3-ASR {data['model']}；ASR 编码器由 CPU ORT 改为 GGML/Vulkan，解码 GGUF、Vulkan 解码器、ForcedAligner 与加载策略保留。",
             f"每端 {data['rounds_per_side']} 个新进程，顺序保存在后台快照中；每轮固定五条请求，含精确 60 秒文件。正式轮无 profile、无 embedding 写入。",
             "Windows/驱动缓存保留；主机内存目标25ms、GPU目标100ms采样。表格为中位数 [最小～最大]，减幅=(基线-候选)/基线；负号表示增加。", "",
             "## 速度（秒）", "",
             "| 指标 | CPU ORT 基线 | GGML/Vulkan 候选 | 减幅 |", "|---|---:|---:|---:|",
             row("启动至 model_ready", data["timing"]["model_ready_seconds"]),
             row("model_ready + 第一条完整请求", data["timing"]["model_ready_plus_first_request_seconds"]),
             row("五条完整请求合计", data["timing"]["total_request_seconds"])]
    for name in TIMES:
        lines.append(row(name + " 五请求合计", data["timing"][name + "_sum"]))
    lines += ["", "第一条请求已经预热，名称为 warmed_first_short。ready+first 排除了中间资源快照开销；它与请求本身的耗时分别列出。",
              "t_generate 只覆盖最后一次重试，完整 IPC 耗时保留所有工作；原生阶段合计不强制等于完整 IPC。", "",
              "| 请求/阶段（秒） | 基线 | 候选 | 减幅 |", "|---|---:|---:|---:|"]
    for request in data["requests"]:
        for name in ("latency_seconds", *TIMES):
            lines.append(row(f"{request['phase']} / {name}", request["timing"][name]))
    lines += ["", "## 主机内存（MiB）", "",
              "| 阶段/指标 | 基线 | 候选 | 减幅 |", "|---|---:|---:|---:|"]
    names = {"private_bytes": "私有提交", "private_working_set": "私有工作集", "working_set": "总工作集"}
    for point, metrics in data["memory"].items():
        for name, metric in metrics.items():
            lines.append(row(f"{point} / {names[name]}", metric, MIB))
    lines += ["", "私有提交与驻留工作集是不同计数，分别评价。工作集下降与私有提交增加在本轮同时出现，不能将单一工作集下降写成整体内存下降。",
              "sampled_peak 是每轮采样峰值的统计，并非瞬时真峰。ready 与 tail_short 为实际阶段快照。", "",
              "## 按适配器的 GPU 显存（MiB）", "",
              "按 LUID/physical/duplicate 逐项列出，dedicated 与 shared 分开，保留真实0，不推导跨适配器总和。缺测索引保存在 JSON。", "",
              "| 适配器/阶段/计数 | 基线 | 候选 | 减幅 |", "|---|---:|---:|---:|"]
    for gpu in data["gpu"]:
        label = f"{gpu['adapter_luid']}/p{gpu['physical_adapter_index']}/d{gpu['duplicate_instance_index']}"
        for point, metrics in gpu["points"].items():
            for name, metric in metrics.items():
                lines.append(row(f"{label} / {point} / {name}", metric, MIB))
    lines += ["", "## CPU 与输出一致性", "",
              "Job 累计 CPU 本轮未采集，JSON保持 null。下表为当时存活的 service/worker 用户态+内核态 CPU；ready_to_tail 是活跃请求阶段差值。", "",
              "| CPU秒 | 基线 | 候选 | 减幅 |", "|---|---:|---:|---:|"]
    for name, metric in data["cpu"].items():
        if name.startswith("alive_tree_") and isinstance(metric, dict):
            lines.append(row(name, metric))
    quality = data["quality"]
    lines += ["", f"{quality['paired_requests']} 对正式请求：五字段 duration/text/text_accu/tokens/timestamps，以及原生 text/token_ids/n_prefill/n_generate/is_aborted，"
              f"差异请求数为 {quality['different_requests']}。这项是固定输入回归，识别错误率另外使用公开文本参考。", "",
              "| 请求 | 基线字符编辑数/参考字符数 | 候选字符编辑数/参考字符数 |", "|---|---:|---:|"]
    for phase, sides in quality["public_cer"].items():
        values = []
        for side in ("baseline", "candidate"):
            item = sides[side]
            values.append(f"{item['edits']}/{item['reference_characters']} ({item['cer']*100:.2f}%)" if item else "无配套逐字稿")
        lines.append(f"| {phase} | {values[0]} | {values[1]} |")
    lines += ["", "字符归一化采用NFKC、大小写折叠并去掉空白/Unicode标点；英文同样按字符计算。60秒切片无配套逐字稿，仅做文本、token和时间戳回归。",
              "候选实际 GPU/CPU 节点数保存在逐轮原始报告；CPU view 等节点计数按原值保留，不直接归类为核心算子回退。", "",
              "## 并发及证据", "",
              f"{data['background']['snapshots']} 张逐轮开始/结束后台快照仅列正常正式托盘 PID {data['background']['expected_tray_pid']}，未记录正式模型 worker。",
              "最早 baseline-diagnostic.json 的GPU门槛误拒记录原样保留；原因是同worker同时有两个合法LUID，后续按适配器分别统计。它和任何带embedding/profile的诊断均未计入本稿。",
              f"基线 EXE SHA256：{data['binary_sha256']['baseline']}",
              f"候选 EXE SHA256：{data['binary_sha256']['candidate']}",
              f"输入 manifest SHA256：{data['manifest_sha256']}",
              f"工作负载 SHA256：{data['workload_sha256']}",
              "原生DLL的阶段收据、逐轮JSON及其SHA256均列于汇总JSON。本稿只汇总这一组隔离正式轮；最终部署选择由主报告结合其他候选统一给出。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=STUDY / "runs")
    parser.add_argument("--baseline-prefix", default="baseline-formal")
    parser.add_argument("--candidate-prefix", default="candidate-formal")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--background", type=Path, default=STUDY / "runs/formal1-background.json")
    parser.add_argument("--background-candidate-side", default="candidate",
                        help="后台快照中候选的原始 side 标签；规范化为 candidate 并保留原标签")
    parser.add_argument("--expected-background-pid", type=int, default=7320)
    parser.add_argument("--label", default="f4fa")
    parser.add_argument("--native-receipt", type=Path, default=STUDY / "prepared/staging-native-audio-refresh.json")
    parser.add_argument("--output", type=Path, default=STUDY / "summary-f4fa.json")
    parser.add_argument("--report", type=Path, default=STUDY / "analysis.md")
    args = parser.parse_args()
    require(1 <= args.rounds <= 20, "轮次数无效")
    for output in (args.output, args.report):
        require(output.resolve().is_relative_to(STUDY.resolve()) and not output.exists(), "输出必须是本轮目录中的新文件")
    result = summarize(args)
    text = render(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({"summary": str(args.output), "report": str(args.report), "paired_requests": result["quality"]["paired_requests"],
                      "all_exact": result["quality"]["all_exact"],
                      "five_request_reduction_percent": result["timing"]["total_request_seconds"]["reduction_percent"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
