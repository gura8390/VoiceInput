"""只读汇总同1.7B CPU ORT/CUDA与GGML/CUDA对照，保留污染轮及补测映射。"""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone

from summarize_ggml_encoder import compare, gpu_values, FIELDS, RAW_FIELDS, TIMES, MEMORY, MIB
from summarize_ggml_three_routes import metric_values

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "evidence/ggml-cuda-20261117/previous-cuda"
SIDES = ("cpu_cuda17", "pioneer17")
FLAGS = ("passed", "complete", "execution_passed", "measurement_passed",
         "gpu_measurement_passed", "cleanup_confirmed", "decoder_backend_verified")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def birth(path):
    stat = Path(path).stat()
    return getattr(stat, "st_birthtime", stat.st_ctime)


def difference(a, b):
    require(len(a["samples"]) == len(b["samples"]) == 5, "须为五请求")
    rows = []
    for x, y in zip(a["samples"], b["samples"]):
        require(all(x[k] == y[k] for k in ("phase", "id", "source", "seed", "pcm_sha256")), "输入或种子不同")
        rows.append({"phase": x["phase"],
                     "fields": [k for k in FIELDS if x["result"][k] != y["result"][k]],
                     "raw": [k for k in RAW_FIELDS if x["metrics"][k] != y["metrics"][k]]})
    return {"requests": len(rows), "all_exact": all(not r["fields"] and not r["raw"] for r in rows),
            "different_requests": sum(bool(r["fields"] or r["raw"]) for r in rows), "rows": rows}


def timeline(args, paths):
    start = birth(args.stage_app)
    finish = args.stage_receipt.stat().st_mtime
    rows = []
    for side in SIDES:
        for index, path in enumerate(paths[side], 1):
            run_start, run_end = birth(path.with_suffix(".stderr.txt")), path.stat().st_mtime
            overlap = max(0, min(finish, run_end) - max(start, run_start))
            rows.append({"variant": side, "round": index, "path": str(path),
                         "start_utc": utc(run_start), "end_utc": utc(run_end),
                         "stage_io_overlap_seconds": overlap, "clean_of_known_stage_io": overlap == 0})
    return {"stage_start_utc": utc(start), "stage_finish_utc": utc(finish),
            "basis": "隔离app目录创建时间至最后stage receipt写入；每轮stderr创建至结果JSON落盘，采用保守包含区间",
            "rounds": rows, "all_selected_rounds_clean": all(r["clean_of_known_stage_io"] for r in rows)}


def collect(args):
    replacements = read(args.replacements_json) if args.replacements_json else {}
    paths = {side: [Path(replacements.get(side, {}).get(str(i), args.runs_dir / f"{side}-formal{i}.json"))
                    for i in range(1, 4)] for side in SIDES}
    runs, primes, evidence = {}, {}, []
    for side in SIDES:
        primepath = args.runs_dir / f"{side}-prime.json"
        prime = read(primepath)
        require(prime.get("execution_passed") and prime.get("cleanup_confirmed"), "自身prime未完成")
        primes[side] = prime
        evidence.append({"variant": side, "prime": True, "statistics_included": False,
                         "path": str(primepath), "sha256": digest(primepath)})
        runs[side] = []
        for index, path in enumerate(paths[side], 1):
            run = read(path)
            require(all(run.get(flag) is True for flag in FLAGS), f"正式轮标志未通过：{path.name}")
            require(not run.get("diagnostic") and not any(run.get(k) for k in
                    ("load_profiles", "aligner_profiles", "encoder_embeddings", "sampling_errors", "gpu_sampling_errors")), "诊断或错误不能混入")
            require(run["research_environment"] == {} and not run["loading_policy_changed"], "环境或加载策略不同")
            require(run["encoder_provider"] == ("cpu" if side == SIDES[0] else "ggml_cuda"), "编码路由错误")
            require(run["expected_decoder_device"] == "CUDA0" and run["settings"]["gpu_backend"] == "cuda", "CUDA解码路由错误")
            require(all(row["gpu_layers"] == row["total_layers"] == 29 for row in run["decoder_offload_records"]), "解码未29层GPU")
            if side == SIDES[1]:
                require(run.get("gpu_encoder_verified") and run.get("encoder_zero_cpu_core"), "GGML编码GPU核心回落检查失败")
                require(len(run["encoder_graph_records"]) == 6 and all(row["gpu_nodes"] > 0 for row in run["encoder_graph_records"]), "GGML图记录缺失")
            else:
                require(not run["encoder_route"] and not run["encoder_graph_records"], "CPU路线隐含使用GGML")
            require(run["settings"]["asr_model"] == "1.7b" and run["settings"]["cuda_graph_optimization"] is False, "模型或实验多流配置错误")
            require(run.get("outputs_exact") is True and difference(run, prime)["all_exact"], "自身prime回归不精确")
            bgpath = path.with_suffix(".background.json")
            bg = read(bgpath)
            require(bg["formal_worker_seen"] is False and not any(p["worker"] for p in bg["before"] + bg["after"]), "正式worker并发")
            strictpath = path.with_suffix(".strict-validation.json")
            strict = read(strictpath)
            require(strict["all_strict_checks_passed"] is True and strict["report_sha256"] == digest(path), "严格复核收据不符")
            require(len(run["samples"]) == 5 and len(run["stages"]) == 6, "正式样本或阶段数异常")
            require(abs(sum(r["latency_seconds"] for r in run["samples"]) - run["total_request_seconds"]) < 1e-9, "耗时总和错误")
            require(difference(run, prime)["all_exact"], "参考结果不符")
            runs[side].append(run)
            evidence.append({"variant": side, "round": index, "prime": False, "statistics_included": True,
                             "path": str(path), "sha256": digest(path), "background_sha256": digest(bgpath),
                             "strict_validation_sha256": digest(strictpath)})
    flat = [r for values in runs.values() for r in values]
    for key in ("binary_sha256", "manifest_sha256", "workload_sha256", "host_sample_target_ms", "gpu_sample_target_ms"):
        require(len({r[key] for r in flat}) == 1, f"身份不同：{key}")
    allowed = {"app_dir", "encoder_provider"}
    settings = [{k: v for k, v in r["settings"].items() if k not in allowed} for r in flat]
    require(all(s == settings[0] for s in settings), "只有app_dir/encoder_provider可以不同")
    for key in ("runtime_inventory", "models_inventory", "mmproj_sha256", "staging_sha256"):
        require(all(r["artifact_identity"][key] == flat[0]["artifact_identity"][key] for r in flat), f"同权重或runtime身份不同：{key}")
    return paths, runs, primes, evidence


def summarize(args):
    paths, runs, primes, evidence = collect(args)
    phases = [r["phase"] for r in runs[SIDES[0]][0]["samples"]]
    require(phases[0] == "warmed_first_short" and len({r["id"] for r in runs[SIDES[0]][0]["samples"]}) == 4, "五请求四PCM口径异常")
    require(next(r for r in runs[SIDES[0]][0]["samples"] if r["phase"] == "file_60s")["result"]["duration"] == 60, "文件非60秒")
    units = {"model_ready_seconds": "seconds", "model_ready_plus_first_request_seconds": "seconds",
             "total_request_seconds": "seconds", "job_cpu_seconds": "cpu_seconds"}
    units.update({f"sum.{t}": "seconds" for t in TIMES})
    units.update({f"memory.{point}.{counter}": "bytes" for point in ("peak", "ready", "tail_short") for counter in MEMORY})
    units.update({f"request.{phase}.{counter}": "seconds" for phase in phases for counter in ("complete", *TIMES)})
    metrics = {key: compare(metric_values(runs[SIDES[0]], key), metric_values(runs[SIDES[1]], key), unit) for key, unit in units.items()}
    cross = [difference(a, b) for a, b in zip(runs[SIDES[0]], runs[SIDES[1]])]
    quality = {}
    for side, values in runs.items():
        public = [s["public_text_cer"] for r in values for s in r["samples"] if s.get("public_text_cer")]
        edits = sum(p["edits"] for p in public)
        chars = sum(p["reference_characters"] for p in public)
        quality[side] = {"requests": 15, "self_prime_exact": all(difference(r, primes[side])["all_exact"] for r in values),
                         "self_repeat_exact": all(difference(r, values[0])["all_exact"] for r in values),
                         "public_annotated_requests": len(public), "public_edits": edits, "public_characters": chars,
                         "public_cer": edits/chars if chars else None, "unannotated_60s_cer": None}
    gpu = []
    keys = set().union(*(set(gpu_values(r, "sampled_peak")) for values in runs.values() for r in values))
    for key in sorted(keys, key=str):
        row = {"adapter_luid": key[0], "physical_adapter_index": key[1], "duplicate_instance_index": key[2], "metrics": {}}
        for point in ("sampled_peak", "ready", "tail_short"):
            for counter in ("dedicated_bytes", "shared_bytes"):
                data = [[gpu_values(r, point).get(key, {}).get(counter) for r in runs[side]] for side in SIDES]
                row["metrics"][f"{point}.{counter}"] = compare(*data, "bytes")
        gpu.append(row)
    io = timeline(args, paths)
    require(io["all_selected_rounds_clean"] or args.provisional, "已选轮含stage重I/O，必须补测或显式provisional")
    old_paths = {side: [args.runs_dir / f"{side}-formal{i}.json" for i in range(1, 4)] for side in SIDES}
    original_timeline = timeline(args, old_paths)
    return {"schema": 1, "label": "同1.7B CPU ORT＋CUDA与GGML/CUDA编码对照，非原GitHub分支",
            "complete": io["all_selected_rounds_clean"], "provisional": not io["all_selected_rounds_clean"],
            "rounds_per_variant": 3, "prime_in_statistics": False, "evidence": evidence,
            "variants": {side: {"exe_sha256": values[0]["binary_sha256"], "artifact_identity": values[0]["artifact_identity"],
                                "settings": values[0]["settings"]} for side, values in runs.items()},
            "identity": {"same_exe": True, "same_models_inventory": True, "same_native_inventory": True,
                         "configuration_difference_only": ["app_dir", "encoder_provider"], "no_experimental_multistream_graph": True},
            "metrics": metrics, "gpu": gpu, "quality": quality,
            "cross_route": {"requests": 15, "all_exact": all(x["all_exact"] for x in cross),
                            "different_requests": sum(x["different_requests"] for x in cross), "rounds": cross},
            "selected_stage_io_audit": io, "original_stage_io_audit": original_timeline,
            "formula": "(CPU ORT+CUDA median - GGML/CUDA median) / CPU ORT+CUDA median *100; positive means lower",
            "scope": "每版3新进程×5请求/4PCM；精确60秒file为PCM IPC含对齐，不含FFmpeg/导出；Windows/驱动缓存保留",
            "sampling": "主机25ms、WDDM100ms目标；三种主机计数与逐PID/LUID/physical/duplicate专用共享显存独立，缺测null",
            "cpu_scope": "Windows Job user+kernel累计CPU秒，包含serve/worker模型载入、五请求、清理；不含Python采集器/正常GUI；不是百分比或wall time",
            "quality_scope": "固定样本/固定种子精确回归；公开短句CER单列，60秒无逐字真值；输出变化率不作CER",
            "encoder_scope": "GGML FE/BE核心GPU节点>0且CPU核心回落0，metadata/view CPU节点保留；Mel/文字处理在CPU",
            "raw_transcripts_in_summary": False}


def render(result):
    lines = ["# 同1.7B CUDA编码路线对照", "", result["label"], "",
             "CPU ORT编码与GGML/CUDA batch4编码使用同一最终EXE、同1.7B权重及相同decoder/ORT运行库；配置只有隔离app路径和encoder_provider不同。",
             "每版三轮，每轮五请求、四段不同PCM；prime不计入统计。中位数[最小～最大]，减幅为(old-new)/old。", ""]
    if result["provisional"]:
        lines += ["本表为污染轮保留统计：已发现stage复制/哈希与原formal2重叠，等待整对补测后生成正式表。", ""]
    lines += ["| 指标 | CPU ORT＋CUDA | GGML/CUDA | 减幅 |", "|---|---:|---:|---:|"]
    names = {"model_ready_seconds": "载入至model_ready秒", "model_ready_plus_first_request_seconds": "ready＋首句秒",
             "total_request_seconds": "五请求合计秒", "request.file_60s.complete": "60秒文件IPC秒", "job_cpu_seconds": "Job累计CPU秒"}
    names.update({f"memory.{point}.{counter}": f"{label}{description} MiB" for point, label in
                  (("peak", "采样峰"), ("ready", "ready"), ("tail_short", "末段")) for counter, description in
                  (("private_bytes", "私有提交"), ("private_working_set", "私有WS"), ("working_set", "总WS"))})
    for key, name in names.items():
        metric = result["metrics"][key]
        scale = MIB if metric["unit"] == "bytes" else 1
        def value(side):
            s = metric[side]
            return f'{s["median"]/scale:.3f} [{s["min"]/scale:.3f}～{s["max"]/scale:.3f}]'
        percent = metric["reduction_percent"]
        lines.append(f"| {name} | {value('baseline')} | {value('candidate')} | {percent:+.2f}% |")
    lines += ["", f"跨路线15条请求中，{result['cross_route']['different_requests']}条存在五结果字段或raw差异；全部精确：{result['cross_route']['all_exact']}。",
              "两路线各自对prime及跨轮精确性、公开短句CER、逐请求编码/prefill/generate、各LUID专用/共享显存的中位和范围详见JSON。",
              "GGML核心算子在GPU，CPU核心回落0；CPU metadata/view和图外Mel/文字处理保留。解码29/29层CUDA，实验多流Graph未开启。",
              "", result["scope"], result["sampling"], result["cpu_scope"], result["quality_scope"], "",
              "原formal2与stage I/O重叠的JSON保持原样；所选最终轮与原轮时间审计同时保留在汇总JSON。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=STUDY / "runs")
    parser.add_argument("--replacements-json", type=Path)
    parser.add_argument("--stage-app", type=Path, default=STUDY / "historical-original")
    parser.add_argument("--stage-receipt", type=Path, default=STUDY / "historical-stage.json")
    parser.add_argument("--output", type=Path, default=STUDY / "same-model-summary.json")
    parser.add_argument("--report", type=Path, default=STUDY / "same-model-analysis.md")
    parser.add_argument("--provisional", action="store_true")
    args = parser.parse_args()
    require(not args.output.exists() and not args.report.exists(), "保留已有输出，使用新路径")
    result = summarize(args)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.report.write_text(render(result), encoding="utf-8")
    print(json.dumps({"summarized": True, "complete": result["complete"], "cross_exact": result["cross_route"]["all_exact"]}))


if __name__ == "__main__":
    main()
