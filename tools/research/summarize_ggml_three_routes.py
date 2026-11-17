"""汇总三版Latin square已完成正式轮，不加载模型、不修改原始报告。"""
from pathlib import Path
import argparse
import json

from bench_ggml_three_routes import ROOT, STUDY, NAMES, PROVIDERS, BACKENDS, read, sha, save, require
from summarize_ggml_encoder import compare, gpu_values, value_range, FIELDS, RAW_FIELDS, TIMES, MEMORY, MIB


def collect(args):
    folder = args.runs_dir.resolve(strict=True)
    variants, primes, evidence = {}, {}, []
    for variant in NAMES:
        prime_path = folder / f"{variant}-prime.json"
        prime = read(prime_path)
        require(prime.get("complete") and prime.get("execution_passed") and prime.get("cleanup_confirmed")
                and not prime.get("diagnostic") and not prime.get("encoder_embeddings"), "自身prime参考未完整完成")
        require(prime["encoder_provider"] == PROVIDERS[variant], "prime路由身份错误")
        primes[variant] = prime
        evidence.append({"variant": variant, "round": 0, "prime": True,
                         "path": str(prime_path), "sha256": sha(prime_path), "statistics_included": False})
        values = []
        for index in range(1, args.rounds + 1):
            path = folder / f"{variant}-formal{index}.json"
            run = read(path)
            require(all(run.get(flag) is True for flag in (
                "passed", "complete", "execution_passed", "measurement_passed", "gpu_measurement_passed", "cleanup_confirmed")),
                f"正式轮未完成：{path.name}")
            require(not run.get("diagnostic") and not run.get("encoder_embeddings") and not run.get("load_profiles"), "诊断不能混入正式轮")
            require(run["encoder_provider"] == PROVIDERS[variant] and run["expected_decoder_device"] == (
                "CUDA0" if BACKENDS[variant] == "cuda" else "Vulkan0"), "三版路由身份不一致")
            require(run.get("decoder_backend_verified") is True, "解码后端和29层卸载未确认")
            if variant != "baseline":
                require(run.get("gpu_encoder_verified") is True and run.get("encoder_zero_cpu_core") is True,
                        "GGML编码器后端或零CPU核心回落未确认")
            require(len(run["samples"]) == 5 and len(run["stages"]) == 6, "请求或快照数量异常")
            identity = run.get("artifact_identity")
            require(identity and identity["variant"] == variant, "产物身份缺失")
            require(run["binary_sha256"] == prime["binary_sha256"]
                    and run["manifest_sha256"] == prime["manifest_sha256"]
                    and run["workload_sha256"] == prime["workload_sha256"]
                    and identity["staging_sha256"] == prime["artifact_identity"]["staging_sha256"], "prime与正式轮身份不一致")
            background = path.with_suffix(".background.json")
            bg = read(background)
            require(bg["formal_worker_seen"] is False and not any(p["worker"] for p in bg["before"] + bg["after"]), "正式worker并发")
            cache = path.with_suffix(".cache.json")
            require(cache.is_file(), "cache身份收据缺失")
            evidence.append({"variant": variant, "round": index, "path": str(path), "sha256": sha(path),
                             "background_sha256": sha(background), "cache_receipt_sha256": sha(cache)})
            values.append(run)
        variants[variant] = values
    all_runs = [run for values in variants.values() for run in values]
    for key in ("manifest_sha256", "workload_sha256"):
        require(len({run[key] for run in all_runs}) == 1, f"输入身份变化：{key}")
    require(len({run["artifact_identity"]["staging_sha256"] for run in all_runs}) == 1, "运行期间stage身份发生变化")
    require(variants["preview"][0]["binary_sha256"] == variants["pioneer"][0]["binary_sha256"], "预览/先行并非同一新EXE")
    for variant, values in variants.items():
        require(len({run["binary_sha256"] for run in values}) == 1, "同一路由EXE变化")
    allowed = {"app_dir", "encoder_provider", "gpu_backend", "ggml_encoder_model"}
    settings = [{key: value for key, value in run["settings"].items() if key not in allowed} for run in all_runs]
    require(all(item == settings[0] for item in settings), "路由以外配置发生变化")
    require(settings[0]["asr_model"] == "1.7b", "本轮不能更换ASR模型")
    for dll in ("llama.dll", "ggml.dll", "ggml-base.dll"):
        require(len({run["artifact_identity"]["native_dll_sha256"][dll] for run in all_runs}) == 1,
                f"后端以外核心运行库发生变化：{dll}")
    mmproj = {run["artifact_identity"]["mmproj_sha256"] for run in all_runs}
    require(len(mmproj) == 1, "编码权重发生变化")
    phases = [row["phase"] for row in all_runs[0]["samples"]]
    require(phases[0] == "warmed_first_short" and len({row["id"] for row in all_runs[0]["samples"]}) == 4, "五请求/四PCM口径异常")
    require(next(row for row in all_runs[0]["samples"] if row["phase"] == "file_60s")["result"]["duration"] == 60.0,
            "文件样本不是60秒")
    return variants, primes, evidence, phases


def metric_values(values, key):
    if key.startswith("memory."):
        _, point, counter = key.split(".")
        if point == "peak":
            return [run["sampled_peak_totals"][counter] for run in values]
        return [next(s for s in run["stages"] if s["stage"] == point)["totals"][counter] for run in values]
    if key.startswith("request."):
        _, phase, counter = key.split(".")
        rows = [next(row for row in run["samples"] if row["phase"] == phase) for run in values]
        return [row["latency_seconds"] if counter == "complete" else row["metrics"][counter] for row in rows]
    if key.startswith("sum."):
        counter = key.split(".")[1]
        return [sum(row["metrics"][counter] for row in run["samples"]) for run in values]
    return [run.get(key) for run in values]


def summarize(args):
    variants, primes, evidence, phases = collect(args)
    metrics = {"model_ready_seconds": "seconds", "model_ready_plus_first_request_seconds": "seconds",
               "total_request_seconds": "seconds", "job_cpu_seconds": "cpu_seconds"}
    metrics.update({"sum." + key: "seconds" for key in TIMES})
    metrics.update({f"memory.{point}.{counter}": "bytes" for point in ("peak", "ready", phases[-1]) for counter in MEMORY})
    metrics.update({f"request.{phase}.{counter}": "seconds" for phase in phases for counter in ("complete", *TIMES)})
    pairs = (("baseline", "preview"), ("baseline", "pioneer"), ("preview", "pioneer"))
    result = {"schema": 1, "rounds": args.rounds, "order": "三轮Latin square；prime不计入正式统计",
        "variants": {v: {"name": NAMES[v], "exe_sha256": values[0]["binary_sha256"],
                        "artifact_identity": values[0]["artifact_identity"],
                        "encoder_zero_cpu_core": [run.get("encoder_zero_cpu_core") for run in values]}
                     for v, values in variants.items()},
        "evidence": evidence, "comparisons": {}, "gpu": [], "quality": {}, "cross_route_regression": {},
        "scope": "五请求/四PCM，同一1.7B与量化；完整IPC含file对齐，不含FFmpeg或导出。Windows/driver缓存保留。",
        "sampling": "主机25ms、WDDM100ms目标，均为观测峰；逐适配器，不与主机commit相加。",
        "formula": "(left median - right median) / left median *100; positive=lower",
        "passed": True, "passed_scope": "产物身份/测量/路由完整；输出精确性单列，不由passed推断",
        "raw_transcripts_in_summary": False}
    for left, right in pairs:
        label = f"{right}_vs_{left}"
        result["comparisons"][label] = {key: compare(metric_values(variants[left], key), metric_values(variants[right], key), unit)
                                         for key, unit in metrics.items()}
    adapters = set().union(*(set(gpu_values(run, "sampled_peak")) for values in variants.values() for run in values))
    for key in sorted(adapters, key=str):
        gpu = {"adapter_luid": key[0], "physical_adapter_index": key[1], "duplicate_instance_index": key[2], "comparisons": {}}
        for left, right in pairs:
            output = {}
            for point in ("sampled_peak", "ready", phases[-1]):
                for counter in ("dedicated_bytes", "shared_bytes"):
                    data = [[gpu_values(run, point).get(key, {}).get(counter) for run in variants[variant]]
                            for variant in (left, right)]
                    output[f"{point}.{counter}"] = compare(*data, "bytes")
            gpu["comparisons"][f"{right}_vs_{left}"] = output
        result["gpu"].append(gpu)
    for variant, values in variants.items():
        differences, own_differences, prime_differences, public = [], [], [], []
        for run_index, run in enumerate(values):
            for index, sample in enumerate(run["samples"]):
                base = variants["baseline"][run_index]["samples"][index]
                own = values[0]["samples"][index]
                prime = primes[variant]["samples"][index]
                keys = ("phase", "id", "source", "seed", "pcm_sha256")
                require(all(sample[key] == base[key] == own[key] == prime[key] for key in keys), "输入/种子/顺序变化")
                fields = [key for key in FIELDS if sample["result"][key] != base["result"][key]]
                raw = [key for key in RAW_FIELDS if sample["metrics"][key] != base["metrics"][key]]
                differences.append({"round": run_index + 1, "phase": sample["phase"], "fields": fields, "raw": raw})
                own_fields = [key for key in FIELDS if sample["result"][key] != own["result"][key]]
                own_raw = [key for key in RAW_FIELDS if sample["metrics"][key] != own["metrics"][key]]
                own_differences.append({"round": run_index + 1, "phase": sample["phase"], "fields": own_fields, "raw": own_raw})
                prime_fields = [key for key in FIELDS if sample["result"][key] != prime["result"][key]]
                prime_raw = [key for key in RAW_FIELDS if sample["metrics"][key] != prime["metrics"][key]]
                prime_differences.append({"round": run_index + 1, "phase": sample["phase"], "fields": prime_fields, "raw": prime_raw})
                if sample.get("public_text_cer"):
                    public.append(sample["public_text_cer"])
        edits = sum(row["edits"] for row in public)
        characters = sum(row["reference_characters"] for row in public)
        result["quality"][variant] = {"requests": len(differences), "vs_baseline": differences,
            "different_requests": sum(bool(row["fields"] or row["raw"]) for row in differences),
            "same_variant_repeat_differences": own_differences,
            "repeat_exact": all(not row["fields"] and not row["raw"] for row in own_differences),
            "same_variant_prime_differences": prime_differences,
            "prime_exact": all(not row["fields"] and not row["raw"] for row in prime_differences),
            "public_annotated_requests": len(public), "public_edits": edits, "public_characters": characters,
            "public_cer": edits / characters if characters else None, "file_60s_ground_truth": None}
    for left, right in pairs:
        rows = []
        for round_index, (a, b) in enumerate(zip(variants[left], variants[right]), 1):
            for x, y in zip(a["samples"], b["samples"]):
                rows.append({"round": round_index, "phase": x["phase"],
                    "fields": [key for key in FIELDS if x["result"][key] != y["result"][key]],
                    "raw": [key for key in RAW_FIELDS if x["metrics"][key] != y["metrics"][key]]})
        result["cross_route_regression"][f"{right}_vs_{left}"] = {"requests": len(rows),
            "different_requests": sum(bool(row["fields"] or row["raw"]) for row in rows),
            "all_exact": all(not row["fields"] and not row["raw"] for row in rows), "comparisons": rows}
    return result


def fmt(metric, side, scale=1):
    return value_range(metric[side], scale)


def render(data):
    lines = ["# 基准版、预览版与先行版统一对照", "",
             "三版均使用同一Qwen3-ASR 1.7B与原精度。预览/先行使用同一研究EXE，分别采用GGML/Vulkan和GGML/CUDA batch4；基准为原0.6.0 CPU ORT/Vulkan。",
             "每版三个新进程、每轮五请求/四PCM，prime单列排除。保持加载策略，正式安装未切换。",
             "数值为中位数[最小～最大]；百分比按未舍入值(left-right)/left计算，正数表示下降。", ""]
    names = {"model_ready_seconds": "model_ready秒", "model_ready_plus_first_request_seconds": "ready+first秒",
             "total_request_seconds": "五请求合计秒", "job_cpu_seconds": "Job累计CPU秒",
             "memory.peak.private_bytes": "峰私有提交MiB", "memory.peak.working_set": "峰总WS MiB",
             "memory.tail_short.private_bytes": "末段私有提交MiB", "request.file_60s.complete": "60秒文件完整IPC秒"}
    for label, metrics in data["comparisons"].items():
        right, left = label.split("_vs_")
        lines += [f"## {NAMES[right]} 对 {NAMES[left]}", "",
                  f"| 指标 | {NAMES[left]} | {NAMES[right]} | 减幅 |", "|---|---:|---:|---:|"]
        for key, name in names.items():
            metric = metrics[key]
            scale = MIB if metric["unit"] == "bytes" else 1
            percent = metric["reduction_percent"]
            text = f"{percent:+.2f}%" if percent is not None else "未采集/基线为0"
            lines.append(f"| {name} | {fmt(metric, 'baseline', scale)} | {fmt(metric, 'candidate', scale)} | {text} |")
    lines += ["", "完整IPC输入为已解码PCM，文件模式含对齐及合并；媒体解码、TXT/JSON/SRT导出另测。ready+first排除中间观测快照开销。",
              "主机commit、私有WS、总WS分别列出，不能相互代替；GPU逐LUID的dedicated/shared和全部逐请求编码/prefill/generate计时均见JSON，缺测保持null而不补零。", "",
              "## 固定样本结果", "", "| 版本 | 对基准差异请求数 | 对自身prime精确 | 同版跨轮精确 | 公开短句CER |", "|---|---:|---|---|---:|"]
    for variant, item in data["quality"].items():
        cer = f"{item['public_cer']*100:.2f}%" if item["public_cer"] is not None else "无参考"
        lines.append(f"| {NAMES[variant]} | {item['different_requests']}/{item['requests']} | {item['prime_exact']} | {item['repeat_exact']} | {cer} |")
    lines += ["", "| 两版交叉回归 | 差异请求数 | 全字段精确 |", "|---|---:|---|"]
    for pair, item in data["cross_route_regression"].items():
        right, left = pair.split("_vs_")
        lines.append(f"| {NAMES[right]} 对 {NAMES[left]} | {item['different_requests']}/{item['requests']} | {item['all_exact']} |")
    lines += ["", "仅公开短句有逐字稿；60秒切片的文本/token/时间戳按固定输入回归。跨后端差异率是输出变化率，不是识别错误率。",
              "GGML两版的FE/BE GPU/CPU核心节点、CPU metadata/view节点原值保留；前者要求CPU核心回落为0，图外Mel/文字处理仍在CPU。",
              "Vulkan cache和CUDA graph配置、DLL哈希、同一新EXE哈希、模型引用及每轮后台快照由产物身份和收据关联。此文仅报告当前三版测量，不自动部署。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=STUDY / "runs")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=STUDY / "summary.json")
    parser.add_argument("--report", type=Path, default=STUDY / "analysis.md")
    args = parser.parse_args()
    require(1 <= args.rounds <= 20, "轮次数无效")
    require(not args.output.exists() and not args.report.exists(), "汇总输出已存在，保留旧证据")
    require(args.output.resolve().is_relative_to(STUDY.resolve()) and args.report.resolve().is_relative_to(STUDY.resolve()),
            "汇总及报告须保存在本轮隔离目录")
    result = summarize(args)
    text = render(result)
    save(args.output, result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({"summarized": True, "variants": 3, "formal_rounds": args.rounds * 3,
                      "summary": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
