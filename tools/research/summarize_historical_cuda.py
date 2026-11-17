"""只读汇总原0.4.3 CUDA/0.6B与GGML CUDA/1.7B跨模型整体对照。"""
from pathlib import Path
import argparse
import json

from summarize_cuda_controls import read, require, digest, difference
from summarize_ggml_encoder import compare, gpu_values, FIELDS, RAW_FIELDS, TIMES, MEMORY, MIB
from summarize_ggml_three_routes import metric_values

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "evidence/ggml-cuda-20261117/previous-cuda"
SIDES = ("historical", "pioneer17")
FLAGS = ("passed", "complete", "execution_passed", "measurement_passed",
         "gpu_measurement_passed", "cleanup_confirmed", "decoder_backend_verified")


def summarize(args):
    runs, primes, evidence = {}, {}, []
    stage = read(STUDY / "historical-stage.json")
    for side in SIDES:
        primepath = args.runs_dir / f"{side}-prime.json"
        prime = read(primepath)
        require(prime.get("execution_passed") and prime.get("cleanup_confirmed"), "prime未完成")
        primes[side] = prime
        evidence.append({"variant": side, "prime": True, "statistics_included": False,
                         "path": str(primepath), "sha256": digest(primepath)})
        runs[side] = []
        for index in range(1, 4):
            path = args.runs_dir / f"{side}-formal{index}.json"
            run = read(path)
            require(all(run.get(flag) is True for flag in FLAGS), f"正式轮未完成：{path.name}")
            require(not run["diagnostic"] and not any(run.get(k) for k in
                    ("encoder_embeddings", "load_profiles", "aligner_profiles", "sampling_errors", "gpu_sampling_errors")), "诊断或采样失败")
            require(run["outputs_exact"] is True and difference(run, prime)["all_exact"], "自身prime回归不精确")
            require(len(run["samples"]) == 5 and len(run["stages"]) == 6, "五请求六快照未齐")
            require(run["settings"]["asr_model"] == ("0.6b" if side == "historical" else "1.7b"), "模型身份错误")
            require(run["settings"]["gpu_backend"] == "cuda" and run["expected_decoder_device"] == "CUDA0", "解码后端错误")
            require(run["research_environment"] == {} and not run["loading_policy_changed"], "环境或加载策略改变")
            require(run["settings"]["cuda_graph_optimization"] is False, "实验多流Graph开启")
            require(all(r["gpu_layers"] == r["total_layers"] == 29 for r in run["decoder_offload_records"]), "非29层CUDA")
            expected = stage["exe_sha256"] if side == "historical" else "2c1302bd54399c456afabf3a469cda23beafdb0f302f3f205d6f878c05029f40"
            require(run["binary_sha256"] == prime["binary_sha256"] == expected, "最终EXE错误")
            bgpath = path.with_suffix(".background.json")
            bg = read(bgpath)
            require(not any(p["worker"] for p in bg["before"] + bg["after"]), "正式worker并发")
            if side == "historical":
                require(run["artifact_identity"]["source_commit"] == stage["source_commit"], "历史提交错误")
                require(run["settings_sha256"] == stage["config_sha256"]
                        and run["artifact_identity"]["settings_sha256"] == stage["config_sha256"], "历史配置身份错误")
                require(Path(run["app_dir"]).resolve() == Path(stage["app"]).resolve(), "历史应用根目录错误")
                expected_native = {name: row["sha256"] for name, row in stage["native_files"].items()}
                require(run["artifact_identity"]["native_dll_sha256"] == expected_native, "原native身份错误")
                require(run["encoder_provider"] == "cpu" and not run["encoder_route"], "原CPU ORT被替换")
            else:
                require(run["encoder_provider"] == "ggml_cuda" and run["gpu_encoder_verified"]
                        and run["encoder_zero_cpu_core"], "先行编码路由或CPU核心回落错误")
            require(abs(sum(r["latency_seconds"] for r in run["samples"]) - run["total_request_seconds"]) < 1e-9, "请求时间总和错误")
            runs[side].append(run)
            evidence.append({"variant": side, "round": index, "prime": False, "statistics_included": True,
                             "path": str(path), "sha256": digest(path), "background_sha256": digest(bgpath)})
    flat = [r for values in runs.values() for r in values]
    for key in ("manifest_sha256", "workload_sha256", "host_sample_target_ms", "gpu_sample_target_ms"):
        require(len({r[key] for r in flat}) == 1, f"输入或采样身份不一：{key}")
    phases = [r["phase"] for r in flat[0]["samples"]]
    require(phases[0] == "warmed_first_short" and len({r["id"] for r in flat[0]["samples"]}) == 4, "五请求四PCM异常")
    units = {"model_ready_seconds": "seconds", "model_ready_plus_first_request_seconds": "seconds",
             "total_request_seconds": "seconds", "job_cpu_seconds": "cpu_seconds"}
    units.update({f"sum.{t}": "seconds" for t in TIMES})
    units.update({f"memory.{point}.{counter}": "bytes" for point in ("peak", "ready", "tail_short") for counter in MEMORY})
    units.update({f"request.{phase}.{counter}": "seconds" for phase in phases for counter in ("complete", *TIMES)})
    metrics = {key: compare(metric_values(runs[SIDES[0]], key), metric_values(runs[SIDES[1]], key), unit) for key, unit in units.items()}
    quality = {}
    for side, values in runs.items():
        public = [s["public_text_cer"] for r in values for s in r["samples"] if s.get("public_text_cer")]
        edits, chars = sum(p["edits"] for p in public), sum(p["reference_characters"] for p in public)
        quality[side] = {"requests": 15, "self_prime_exact": all(difference(r, primes[side])["all_exact"] for r in values),
                         "self_repeat_exact": all(difference(r, values[0])["all_exact"] for r in values),
                         "public_annotated_requests": len(public), "public_edits": edits, "public_characters": chars,
                         "public_cer": edits/chars if chars else None, "unannotated_60s_cer": None}
    cross = [difference(a, b) for a, b in zip(runs[SIDES[0]], runs[SIDES[1]])]
    gpu = []
    keys = set().union(*(set(gpu_values(r, "sampled_peak")) for r in flat))
    for key in sorted(keys, key=str):
        row = {"adapter_luid": key[0], "physical_adapter_index": key[1], "duplicate_instance_index": key[2], "metrics": {}}
        for point in ("sampled_peak", "ready", "tail_short"):
            for counter in ("dedicated_bytes", "shared_bytes"):
                data = [[gpu_values(r, point).get(key, {}).get(counter) for r in runs[side]] for side in SIDES]
                row["metrics"][f"{point}.{counter}"] = compare(*data, "bytes")
        gpu.append(row)
    return {"schema": 1, "complete": True, "label": "原0.4.3 CUDA/0.6B与先行GGML CUDA/1.7B整体方案对照",
            "cross_model": True, "rounds_per_variant": 3, "prime_in_statistics": False,
            "variants": {s: {"exe_sha256": v[0]["binary_sha256"], "artifact_identity": v[0]["artifact_identity"],
                             "settings": v[0]["settings"]} for s, v in runs.items()}, "evidence": evidence,
            "metrics": metrics, "gpu": gpu, "quality": quality,
            "cross_route": {"requests": 15, "all_exact": all(r["all_exact"] for r in cross),
                            "different_requests": sum(r["different_requests"] for r in cross), "rounds": cross},
            "formula": "(historical0.6B median - pioneer1.7B median)/historical0.6B median*100; positive means lower",
            "comparison_boundary": "跨版本、跨模型、跨量化及上下文策略的整体体验比较；差额不归因于GGML/CUDA单一编码器改动",
            "historical_identity": "3b7c123原源码使用Rust1.95重建；旧归档native组件重组、原ORT1.24和原embedded ONNX/Q4 GGUF，当前热词；非旧EXE原二进制整包",
            "configuration_changes": "旧example static ASR context、vulkan cache false；新1.7B dynamic context；其余差异在完整settings及stage receipt中列出",
            "scope": "AB/BA/AB三对新进程；五请求/四PCM；60秒文件为PCM IPC含aligner，不含FFmpeg/导出。Windows/驱动缓存保留。",
            "sampling": "主机25ms/WDDM100ms目标的观测峰；commit、private WS、total WS及各LUID dedicated/shared分开；缺测null",
            "cpu_scope": "Job累计user+kernel CPU秒，含service/worker载入、5请求和退出；不含Python驱动/正常GUI，不是CPU百分比",
            "quality_scope": "固定样本/种子；公开短句CER单列，60秒无逐字真值；五字段/raw变化数只表示输出差异，不是CER",
            "raw_transcripts_in_summary": False}


def render(data):
    lines = ["# 原GitHub CUDA分支与先行版整体对照", "", data["comparison_boundary"], data["historical_identity"], "",
             "每版三轮、每轮五请求/四段PCM；prime不计入。中位数[最小～最大]；减幅(old-new)/old。", "",
             "| 指标 | 历史0.6B | 先行1.7B | 减幅 |", "|---|---:|---:|---:|"]
    names = {"model_ready_seconds": "载入至model_ready秒", "model_ready_plus_first_request_seconds": "ready＋首句秒",
             "total_request_seconds": "五请求合计秒", "request.file_60s.complete": "60秒文件IPC秒", "job_cpu_seconds": "Job累计CPU秒"}
    names.update({f"memory.{point}.{counter}": f"{label}{description} MiB" for point, label in
                  (("peak", "采样峰"), ("ready", "ready"), ("tail_short", "末段")) for counter, description in
                  (("private_bytes", "私有提交"), ("private_working_set", "私有WS"), ("working_set", "总WS"))})
    for key, name in names.items():
        metric = data["metrics"][key]
        scale = MIB if metric["unit"] == "bytes" else 1
        def value(side):
            s = metric[side]
            return f'{s["median"]/scale:.3f} [{s["min"]/scale:.3f}～{s["max"]/scale:.3f}]'
        lines.append(f"| {name} | {value('baseline')} | {value('candidate')} | {metric['reduction_percent']:+.2f}% |")
    lines += ["", f"跨模型15条中{data['cross_route']['different_requests']}条存在五字段/raw差异；这些是输出变化数。", "",
              "| 路线 | 对自身prime精确 | 跨轮精确 | 公开短句CER |", "|---|---|---|---:|"]
    for side, name in (("historical", "历史0.6B"), ("pioneer17", "先行1.7B")):
        q = data["quality"][side]
        lines.append(f"| {name} | {q['self_prime_exact']} | {q['self_repeat_exact']} | {q['public_cer']*100:.2f}% |")
    lines += ["", "逐请求编码/prefill/generate、峰/ready/末段三类主机内存、逐LUID专用/共享显存详见JSON。",
              data["scope"], data["sampling"], data["cpu_scope"], data["quality_scope"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=STUDY / "branch-paired")
    parser.add_argument("--output", type=Path, default=STUDY / "branch-summary.json")
    parser.add_argument("--report", type=Path, default=STUDY / "branch-analysis.md")
    args = parser.parse_args()
    require(not args.output.exists() and not args.report.exists(), "保留已有汇总，使用新输出")
    data = summarize(args)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.report.write_text(render(data), encoding="utf-8")
    print(json.dumps({"summarized": True, "cross_model": True, "different_requests": data["cross_route"]["different_requests"]}))


if __name__ == "__main__":
    main()
