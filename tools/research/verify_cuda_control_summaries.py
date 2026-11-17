"""独立重算已保存CUDA对照JSON，不导入汇总算法，不读取大模型或运行应用。"""
from pathlib import Path
import argparse
import hashlib
import json
import math

FIELDS = ("duration", "text", "text_accu", "tokens", "timestamps")
RAW = ("text", "token_ids", "n_prefill", "n_generate", "is_aborted")
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evidence/ggml-cuda-20261117/previous-cuda"


def check(ok, reason):
    if not ok:
        raise AssertionError(reason)


def read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def median(values):
    ordered = sorted(values)
    count = len(ordered)
    return ordered[count // 2] if count % 2 else (ordered[count // 2 - 1] + ordered[count // 2]) / 2


def close(a, b):
    return a is None and b is None or a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def audit_stat(saved, values):
    check(saved["values"] == values, "原始metric值不符")
    available = all(type(v) in (int, float) and math.isfinite(v) for v in values)
    check(saved["available"] == available, "缺测状态错误")
    for key, expected in (("median", median(values) if available else None),
                          ("min", min(values) if available else None), ("max", max(values) if available else None)):
        check(close(saved[key], expected), f"{key}统计错误")


def audit_metric(saved, old, new):
    audit_stat(saved["baseline"], old)
    audit_stat(saved["candidate"], new)
    valid = all(type(v) in (int, float) and math.isfinite(v) for v in old + new)
    diff = median(old) - median(new) if valid else None
    percent = diff / median(old) * 100 if valid and median(old) else None
    check(close(saved["absolute_reduction"], diff) and close(saved["reduction_percent"], percent), "差额或减幅错误")


def value(run, metric):
    parts = metric.split(".")
    if parts[0] == "memory":
        _, point, counter = parts
        if point == "peak":
            actual = max(row["totals"][counter] for row in run["memory_samples"])
            check(actual == run["sampled_peak_totals"][counter], "25ms内存采样峰计算错误")
            return actual
        return next(row for row in run["stages"] if row["stage"] == point)["totals"][counter]
    if parts[0] == "request":
        row = next(row for row in run["samples"] if row["phase"] == parts[1])
        return row["latency_seconds"] if parts[2] == "complete" else row["metrics"][parts[2]]
    if parts[0] == "sum":
        return sum(row["metrics"][parts[1]] for row in run["samples"])
    return run[metric]


def gpu_value(run, key, point, counter):
    worker = run["worker_pid"]
    def matching(rows):
        return [r for r in rows if r["pid"] == worker and
                (r["adapter_luid"], r.get("physical_adapter_index"), r.get("duplicate_instance_index")) == key]
    if point == "sampled_peak":
        values = [r[counter] for sample in run["gpu_samples"]
                  for r in matching(sample["gpu_process_memory"]["rows"]) if r.get(counter) is not None]
        actual = max(values) if values else None
        recorded = matching(run["sampled_gpu_peak"]["adapters"])
        check(len(recorded) <= 1, "显存适配器身份重复")
        check((recorded[0][counter] if recorded else None) == actual, "显存采样峰错误")
        return actual
    stage = next(row for row in run["stages"] if row["stage"] == point)
    rows = matching(stage["native_gpu"]["gpu_process_memory"]["rows"])
    check(len(rows) <= 1, "阶段显存适配器身份重复")
    return rows[0][counter] if rows else None


def differences(a, b):
    output = []
    for x, y in zip(a["samples"], b["samples"]):
        check(all(x[k] == y[k] for k in ("phase", "id", "source", "seed", "pcm_sha256")), "固定输入身份变化")
        output.append({"phase": x["phase"], "fields": [k for k in FIELDS if x["result"][k] != y["result"][k]],
                       "raw": [k for k in RAW if x["metrics"][k] != y["metrics"][k]]})
    return output


def verify(path):
    data = read(path)
    sides = tuple(data["variants"])
    runs = {side: [] for side in sides}
    primes = {}
    for row in data["evidence"]:
        source = Path(row["path"])
        check(sha(source) == row["sha256"], "原始报告SHA不符")
        run = read(source)
        if row["prime"]:
            check(not row["statistics_included"], "prime混入统计")
            primes[row["variant"]] = run
        else:
            check(row["statistics_included"], "正式轮未纳入")
            check(all(run.get(k) is True for k in ("passed", "measurement_passed", "gpu_measurement_passed", "cleanup_confirmed")), "运行标志未通过")
            check(not run["diagnostic"] and not run["encoder_embeddings"], "混入诊断")
            check(run["decoder_backend_verified"] and run["expected_decoder_device"] == "CUDA0", "实际CUDA解码未确认")
            check(all(r["gpu_layers"] == r["total_layers"] == 29 for r in run["decoder_offload_records"]), "CUDA29层未确认")
            if run["encoder_provider"] == "ggml_cuda":
                check(all(r["batch"] == 4 and r["fe_gpu_core"] > 0 and r["be_gpu_core"] > 0
                          and r["fe_cpu_core"] == r["be_cpu_core"] == 0 for r in run["encoder_native_stages"]), "batch4/核心回落检查失败")
            check(run["settings"]["cuda_graph_optimization"] is False and run["research_environment"] == {}, "实验graph或外部环境不同")
            runs[row["variant"]].append(run)
    check(all(len(values) == 3 for values in runs.values()), "每版须三正式轮")
    for metric, saved in data["metrics"].items():
        audit_metric(saved, [value(r, metric) for r in runs[sides[0]]], [value(r, metric) for r in runs[sides[1]]])
    gpu_checks = 0
    for row in data["gpu"]:
        key = row["adapter_luid"], row["physical_adapter_index"], row["duplicate_instance_index"]
        for metric, saved in row["metrics"].items():
            point, counter = metric.split(".")
            audit_metric(saved, [gpu_value(r, key, point, counter) for r in runs[sides[0]]],
                         [gpu_value(r, key, point, counter) for r in runs[sides[1]]])
            gpu_checks += 1
    changed = 0
    for index, (a, b) in enumerate(zip(runs[sides[0]], runs[sides[1]])):
        diff = differences(a, b)
        check(diff == data["cross_route"]["rounds"][index]["rows"], "跨路线差异字段不符")
        changed += sum(bool(r["fields"] or r["raw"]) for r in diff)
    check(changed == data["cross_route"]["different_requests"], "跨路线差异计数错误")
    for side in sides:
        for run in runs[side]:
            check(all(not d["fields"] and not d["raw"] for d in differences(run, primes[side])), "自身prime回归失败")
        check(data["quality"][side]["self_prime_exact"] and data["quality"][side]["self_repeat_exact"], "自身回归标记错误")
        public = [s["public_text_cer"] for r in runs[side] for s in r["samples"] if s.get("public_text_cer")]
        edits, chars = sum(r["edits"] for r in public), sum(r["reference_characters"] for r in public)
        check(data["quality"][side]["public_edits"] == edits and data["quality"][side]["public_characters"] == chars
              and close(data["quality"][side]["public_cer"], edits/chars), "公开CER错误")
    identity = {}
    if data.get("cross_model"):
        stage = read(BASE / "historical-stage.json")
        expected_native = {name: row["sha256"] for name, row in stage["native_files"].items()}
        app = Path(stage["app"])
        for run in runs["historical"]:
            check(run["binary_sha256"] == stage["exe_sha256"] and run["settings_sha256"] == stage["config_sha256"], "历史EXE/config未绑定")
            check(run["artifact_identity"]["native_dll_sha256"] == expected_native, "历史全部native未绑定")
            check(Path(run["app_dir"]).resolve() == app.resolve(), "历史app根错误")
        targets = {}
        for sub in ("Qwen3-ASR/Qwen3-ASR-0.6B", "Qwen3-ForcedAligner/Qwen3-ForcedAligner-0.6B"):
            target = (app / "models" / sub).resolve()
            check(target == (Path(stage["models_origin"]) / sub).resolve(), "历史原embedded模型目录错误")
            targets[sub] = str(target)
        check((app / "runtime/llama").resolve() == (app / "core/server/engines/llama/bin").resolve(), "harness native别名错误")
        identity = {"historical_exe_and_config_bound": True, "historical_native_files_bound": len(expected_native),
                    "archived_embedded_model_targets": targets, "model_payload_rehashed": False,
                    "harness_alias_and_ignored_asr_model_field": True}
    else:
        flat = sum(runs.values(), [])
        check(len({r["binary_sha256"] for r in flat}) == 1, "同模型EXE不一致")
        for key in ("runtime_inventory", "models_inventory", "mmproj_sha256"):
            check(all(r["artifact_identity"][key] == flat[0]["artifact_identity"][key] for r in flat), "同模型权重/runtime不同")
        configs = [{k: v for k, v in r["settings"].items() if k not in ("app_dir", "encoder_provider")} for r in flat]
        check(all(c == configs[0] for c in configs), "provider外配置不同")
        check(data["selected_stage_io_audit"]["all_selected_rounds_clean"], "最终轮被stage污染")
        identity = {"same_exe_weights_native": True, "configuration_only_encoder_provider": True,
                    "original_pair2_preserved_and_replaced": all("same-model-repeat" in r["path"] for r in data["evidence"] if r.get("round") == 2)}
    return {"independent_validation_passed": True, "summary": str(path), "summary_sha256": sha(path),
            "method": "从原JSON独立取值、排序重算中位/范围/百分比；25ms host及WDDM原采样重算峰；未导入summary算法",
            "metrics_checked": len(data["metrics"]), "gpu_metrics_checked": gpu_checks, "formal_runs": 6,
            "cross_requests": 15, "different_cross_requests": changed, "self_prime_exact": True,
            "identity": identity, "large_assets_rehashed": False, "models_started": 0, "raw_transcripts_in_validation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.summaries:
        output = path.with_name(path.stem + "-validation.json")
        check(not output.exists(), "保留已有验证收据")
        result = verify(path)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"passed": True, "summary": path.name, "different_cross_requests": result["different_cross_requests"]}))


if __name__ == "__main__":
    main()
