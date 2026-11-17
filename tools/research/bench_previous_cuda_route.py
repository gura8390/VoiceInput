"""隔离对照旧CUDA技术路线复现与GGML CUDA先行版；不代表旧GitHub分支。

两路线固定相同最终2c EXE、1.7B模型、量化和全部运行库，仅改变编码器。
stage/plan/preflight不加载模型；只有显式run才启动本工具拥有的worker进程树。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import bench_ggml_vulkan_encoder as shared
from bench_ggml_three_routes import background, job_cpu
from bench_common import CREATE_NO_WINDOW

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "evidence/ggml-cuda-20261117/previous-cuda"
SOURCE = ROOT / "evidence/ggml-cuda-20261117/prepared-final"
FINAL_EXE = "2c1302bd54399c456afabf3a469cda23beafdb0f302f3f205d6f878c05029f40"
NAMES = {"cpu_cuda17": "旧CUDA路线复现，非原分支", "pioneer17": "GGML CUDA先行版，同1.7B"}
PROVIDERS = {"cpu_cuda17": "cpu", "pioneer17": "ggml_cuda"}
RUNTIME_GROUPS = ("llama", "onnx", "audio", "media")
MODEL_GROUPS = ("Qwen3-ASR/Qwen3-ASR-1.7B", "Qwen3-ForcedAligner/Qwen3-ForcedAligner-0.6B")
GRAPH_ENV = ("CUDA_MODULE_LOADING", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")


def require(ok, message):
    shared.require(ok, message)


def read(path):
    return shared.read(path)


def sha(path):
    return shared.sha(path)


def contained(path):
    path = Path(path).resolve()
    require(path != STUDY.resolve() and path.is_relative_to(STUDY.resolve()), "输出必须位于previous-cuda隔离目录")
    return path


def save(path, value):
    path = contained(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def tree(folder):
    folder = Path(folder).resolve(strict=True)
    return {path.relative_to(folder).as_posix(): {"bytes": path.stat().st_size, "sha256": sha(path)}
            for path in sorted(folder.rglob("*")) if path.is_file()}


def shared_inventories(app):
    runtime = {group: tree(app / "runtime" / group) for group in RUNTIME_GROUPS}
    models = {group: tree(app / "models" / group) for group in MODEL_GROUPS}
    require(all(runtime.values()) and all(models.values()), "运行库或所选模型目录为空")
    return runtime, models


def junction(source, target):
    source = Path(source).resolve(strict=True)
    target = contained(target)
    require(not target.exists(), "目录引用已存在，拒绝覆盖")
    target.parent.mkdir(parents=True, exist_ok=True)
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    command = "New-Item -ItemType Junction -Path " + quote(target) + " -Target " + quote(source) + " | Out-Null"
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                   check=True, capture_output=True, creationflags=CREATE_NO_WINDOW, timeout=30)
    require(target.resolve(strict=True) == source, "目录引用目标不匹配")


def stage(args):
    prepared = contained(args.prepared)
    require(not prepared.exists(), "prepared已存在，拒绝覆盖；旧证据保持")
    source = args.source_prepared.resolve(strict=True)
    app = source / "pioneer"
    source_stage = read(source / "staging.json")
    source_prep = read(source / "preparation.json")
    settings = read(app / "config/settings.json")
    require(sha(app / "语音输入.exe") == source_stage["variants"]["pioneer"]["exe_sha256"] == FINAL_EXE,
            "源EXE不是已确认的最终2c候选")
    require(sha(app / "config/settings.json") == source_stage["variants"]["pioneer"]["settings_sha256"], "源配置身份变化")
    require(settings.get("asr_model") == "1.7b" and settings.get("gpu_backend") == "cuda"
            and settings.get("encoder_provider") == "ggml_cuda" and settings.get("gpu") is True,
            "源配置必须是同1.7B GGML CUDA先行版")
    require(not settings.get("cuda_graph_optimization"), "本对照保留既有CUDA graph策略")
    require({p.name: sha(p) for p in (app / "runtime/llama").glob("*.dll")}
            == source_stage["variants"]["pioneer"]["native_dll_sha256"], "源原生DLL身份变化")
    projector = Path(settings["ggml_encoder_model"]).resolve(strict=True)
    require(sha(projector) == source_stage["mmproj_sha256"], "源编码器GGUF身份变化")
    manifest, work, references = (read(source / name) for name in
                                  ("manifest.json", "workload.json", "public-text-reference.json"))
    require(len(work) == 5 and len(manifest) == 4 and len({row["id"] for row in work}) == 4,
            "本对照固定五请求、四PCM")
    shared.validate_inputs(source / "manifest.json", source / "workload.json")
    runtime, models = shared_inventories(app)
    source_hashes = {str(source / name): sha(source / name) for name in
                     ("staging.json", "preparation.json", "manifest.json", "workload.json", "public-text-reference.json")}
    prepared.mkdir(parents=True)
    (prepared / "inputs").mkdir()
    copied_inputs = []
    for row in manifest:
        pcm = Path(row["pcm"])
        require(sha(pcm) == row["pcm_sha256"], "PCM身份不匹配")
        target = prepared / "inputs" / pcm.name
        shutil.copy2(pcm, target)
        copied = {**row, "pcm": str(target)}
        if row.get("wav"):
            wav = Path(row["wav"])
            require(sha(wav) == row["wav_sha256"], "WAV身份不匹配")
            target_wav = prepared / "inputs" / wav.name
            shutil.copy2(wav, target_wav)
            copied["wav"] = str(target_wav)
        copied_inputs.append(copied)
    save(prepared / "manifest.json", copied_inputs)
    save(prepared / "workload.json", work)
    save(prepared / "public-text-reference.json", references)
    variants = {}
    for variant in NAMES:
        target = prepared / variant
        (target / "config").mkdir(parents=True)
        shutil.copy2(app / "语音输入.exe", target / "语音输入.exe")
        for dll in app.glob("*.dll"):
            shutil.copy2(dll, target / dll.name)
        for filename in ("hot.txt", "hot-rule.txt"):
            if (app / "config" / filename).is_file():
                shutil.copy2(app / "config" / filename, target / "config" / filename)
        config = {**settings, "app_dir": "..", "encoder_provider": PROVIDERS[variant]}
        save(target / "config/settings.json", config)
        junction(app / "models", target / "models")
        for group in RUNTIME_GROUPS:
            junction(app / "runtime" / group, target / "runtime" / group)
        save(prepared / f"{variant}-env.json", {})
        variants[variant] = {"name": NAMES[variant], "app": str(target), "exe_sha256": sha(target / "语音输入.exe"),
                             "settings_sha256": sha(target / "config/settings.json"),
                             "root_dll_sha256": {p.name: sha(p) for p in target.glob("*.dll")},
                             "hotword_files": {p.name: sha(p) for p in (target / "config").glob("hot*.txt")},
                             "models_target": str((target / "models").resolve()),
                             "runtime_targets": {group: str((target / "runtime" / group).resolve()) for group in RUNTIME_GROUPS}}
    configs = [read(prepared / name / "config/settings.json") for name in NAMES]
    require({key for key in configs[0] if configs[0][key] != configs[1][key]} == {"encoder_provider"},
            "两份配置必须仅encoder_provider不同")
    require(all(sha(Path(path)) == expected for path, expected in source_hashes.items()), "stage期间源收据变化")
    save(prepared / "staging.json", {"staged": True, "models_loaded": 0, "model_processes_started": 0,
        "source_prepared": str(source), "source_app": str(app), "source_receipt_sha256": source_hashes,
        "formal_app": source_prep["formal_app"], "formal_application_modified": False,
        "comparison_scope": "同1.7B旧CUDA技术路线复现，非GitHub旧分支或旧EXE测量",
        "unsupported_historical_branch": "0.4.3-cuda.1只0.6B，单独测量，不能用本报告冒充",
        "model": "1.7b", "only_configuration_difference": ["encoder_provider"], "variants": variants,
        "runtime_file_inventory": runtime, "selected_model_file_inventory": models,
        "mmproj": str(projector), "mmproj_sha256": sha(projector),
        "manifest_sha256": sha(prepared / "manifest.json"), "workload_sha256": sha(prepared / "workload.json"),
        "public_reference_sha256": sha(prepared / "public-text-reference.json"),
        "runtime_and_model_references": "只读使用junction；不修改源目录或ACL；缓存/日志写各自隔离应用根",
        "hash_preflight_scope": "stage和run在计时外读回运行库及所选模型SHA256；保留Windows/驱动缓存，prime排除统计",
        "loading_policy_changed": False, "frontend_batch": 4,
        "raw_units": "保留bytes、秒、原始结果及公开逐字稿CER；不在原始报告中换算MiB或舍入"})
    shared.progress("staged", variants=2, unique_pcm=4, requests=5, models_loaded=0, output=str(prepared))


def validate(prepared):
    prepared = contained(prepared).resolve(strict=True)
    receipt = read(prepared / "staging.json")
    require(receipt["model"] == "1.7b" and receipt["only_configuration_difference"] == ["encoder_provider"], "stage范围异常")
    for name, expected in (("manifest.json", "manifest_sha256"), ("workload.json", "workload_sha256"),
                           ("public-text-reference.json", "public_reference_sha256")):
        require(sha(prepared / name) == receipt[expected], "输入或逐字稿清单变化")
    shared.validate_inputs(prepared / "manifest.json", prepared / "workload.json")
    configs = []
    for variant, identity in receipt["variants"].items():
        app = prepared / variant
        require(sha(app / "语音输入.exe") == identity["exe_sha256"] == FINAL_EXE, "最终EXE身份变化")
        config = read(app / "config/settings.json")
        require(sha(app / "config/settings.json") == identity["settings_sha256"], "隔离配置身份变化")
        require(config["asr_model"] == "1.7b" and config["gpu_backend"] == "cuda"
                and config["encoder_provider"] == PROVIDERS[variant], "路由标签与配置不同")
        require(str((app / "models").resolve(strict=True)) == identity["models_target"], "模型目录引用变化")
        require({group: str((app / "runtime" / group).resolve(strict=True)) for group in RUNTIME_GROUPS}
                == identity["runtime_targets"], "运行库组引用变化")
        require({p.name: sha(p) for p in app.glob("*.dll")} == identity["root_dll_sha256"], "根目录DLL变化")
        require({p.name: sha(p) for p in (app / "config").glob("hot*.txt")} == identity["hotword_files"], "热词变化")
        require(read(prepared / f"{variant}-env.json") == {}, "本对照不能启用额外研究环境覆盖")
        configs.append(config)
    require({key for key in configs[0] if configs[0][key] != configs[1][key]} == {"encoder_provider"}, "配置差异超范围")
    app = prepared / "pioneer17"
    runtime, models = shared_inventories(app)
    require(runtime == receipt["runtime_file_inventory"] and models == receipt["selected_model_file_inventory"],
            "原生运行库或模型字节身份变化")
    require(sha(receipt["mmproj"]) == receipt["mmproj_sha256"], "GGUF编码器变化")
    return prepared, receipt


def plan(args):
    prepared = contained(args.prepared).resolve(strict=True)
    require((prepared / "staging.json").is_file(), "请先stage")
    output = contained(args.output_dir)
    entries = [{"phase": "prime", "round": 0, "variant": variant} for variant in NAMES]
    for index, order in enumerate((("cpu_cuda17", "pioneer17"), ("pioneer17", "cpu_cuda17"),
                                    ("cpu_cuda17", "pioneer17")), 1):
        entries += [{"phase": "formal", "round": index, "variant": variant} for variant in order]
    for entry in entries:
        tag = f"{entry['variant']}-prime" if entry["phase"] == "prime" else f"{entry['variant']}-formal{entry['round']}"
        command = [sys.executable, str(Path(__file__).resolve()), "run", "--prepared", str(prepared),
                   "--variant", entry["variant"], "--output", str(output / (tag + ".json"))]
        if entry["phase"] == "formal":
            command += ["--reference", str(output / f"{entry['variant']}-prime.json")]
        entry.update(tag=tag, command=command)
    save(output / "run-plan.json", {"prepared": str(prepared), "prime_excluded_from_statistics": True,
         "order": "prime A,B; formal AB BA AB", "routes": NAMES, "entries": entries, "models_started_by_plan": 0})
    path = contained(output / "run-serial.ps1")
    script = ["$ErrorActionPreference='Stop'", "$taskPlan=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'run-plan.json') -Raw -Encoding UTF8 | ConvertFrom-Json",
              "foreach($taskEntry in $taskPlan.entries){", "  $taskCommand=@($taskEntry.command)",
              "  $taskExe=$taskCommand[0]", "  $taskArgs=@($taskCommand | Select-Object -Skip 1)",
              "  & $taskExe @taskArgs", "  if($LASTEXITCODE -ne 0){throw ('对照失败，证据已保留：'+$taskEntry.tag)}", "}"]
    with path.open("x", encoding="utf-8-sig", newline="\n") as stream:
        stream.write("\n".join(script) + "\n")
    shared.progress("planned", prime=2, formal=6, models_loaded=0, plan=str(output / "run-plan.json"))


def run(args):
    prepared, receipt = validate(args.prepared)
    app = prepared / args.variant
    output = contained(args.output)
    require(not output.exists() and not output.with_suffix(".logs").exists(), "输出或日志目录已存在")
    quality_only = args.quality_prepared is not None
    inputs = Path(args.quality_prepared).resolve(strict=True) if quality_only else prepared
    if quality_only:
        quality_root = (ROOT / "evidence/ggml-cuda-20261117").resolve()
        require(inputs.is_relative_to(quality_root), "质量输入须来自本轮隔离研究目录")
        require(output.is_relative_to((STUDY / "quality").resolve()) and "formal" not in output.stem,
                "质量输出须位于previous-cuda/quality且不能命名为formal")
    workload = inputs / ("workload-multiseed.json" if quality_only else "workload.json")
    expected_requests = len(read(workload))
    if args.reference:
        reference_path = contained(args.reference).resolve(strict=True)
        reference = read(reference_path)
        require(reference.get("artifact_identity", {}).get("variant") == args.variant,
                "正式轮须与本路线自身prime比较；跨路线差异另外分析")
    else:
        reference_path = None
    shared.STUDY = STUDY
    identity = {"variant": args.variant, "name": NAMES[args.variant], "model": "1.7b",
                "historical_branch_executable": False, "comparison_scope": receipt["comparison_scope"],
                "staging_sha256": sha(prepared / "staging.json"), "exe_sha256": FINAL_EXE,
                "runtime_inventory": receipt["runtime_file_inventory"], "models_inventory": receipt["selected_model_file_inventory"],
                "mmproj_sha256": receipt["mmproj_sha256"], "only_configuration_difference": ["encoder_provider"],
                "native_provider": PROVIDERS[args.variant], "decoder": "cuda", "frontend_batch": 4,
                "parent_cuda_environment": {key: os.environ.get(key) for key in GRAPH_ENV}, "loading_policy_changed": False,
                "benchmark_scope": "quality_only" if quality_only else "performance",
                "quality_prepared": str(inputs) if quality_only else None}
    options = SimpleNamespace(exe=app / "语音输入.exe", app_dir=app, config=app / "config/settings.json",
        manifest=inputs / "manifest.json", workload=workload, env_json=prepared / f"{args.variant}-env.json",
        public_reference=inputs / "public-text-reference.json", reference=reference_path, match="exact", profile=args.profile,
        preflight_only=args.preflight_only, output=output, host_ms=25.0, gpu_ms=100.0, timeout=180.0,
        expected_decoder_backend="cuda", strict_core_nodes=True, job_cpu_reader=job_cpu, artifact_identity=identity,
        quality_only=quality_only)
    if args.preflight_only:
        shared.run(options)
        return
    formal = Path(receipt["formal_app"])
    before = background(formal)
    try:
        shared.run(options)
    finally:
        after = background(formal)
        save(output.with_suffix(".background.json"), {"variant": args.variant, "before": before, "after": after,
             "formal_worker_seen": False, "scope": "只查询正式应用，不关闭或控制它"})
    report = read(output)
    log = output.with_suffix(".logs") / "rust-worker.log"
    text = log.read_text("utf-8", errors="replace")
    validation = {"variant": args.variant, "report_sha256": sha(output), "worker_log_sha256": sha(log),
        "shared_runner_passed": report.get("passed") is True,
        "final_exe_verified": report.get("binary_sha256") == FINAL_EXE,
        "stable_expected_request_count": len(report["samples"]) == expected_requests and len(report["stages"]) == expected_requests + 1,
        "job_cpu_valid": type(report.get("job_cpu_seconds")) in (int, float) and math.isfinite(report["job_cpu_seconds"]) and report["job_cpu_seconds"] > 0,
        "cleanup_confirmed": report.get("cleanup_confirmed") is True,
        "CUDA29_confirmed": report.get("decoder_backend_verified") is True
            and bool(re.search(r"using device CUDA0", text))
            and bool(re.search(r"CUDA0\s+model buffer size\s*=\s*[1-9]", text))
            and report.get("decoder_offload_records", [{}])[0] == {"gpu_layers": 29, "total_layers": 29},
        "route_confirmed": report.get("encoder_provider") == PROVIDERS[args.variant],
        "no_implicit_GGML_for_CPU_route": args.variant != "cpu_cuda17" or not report.get("encoder_route"),
        "GGML_CUDA_batch4_no_CPU_core_fallback": args.variant != "pioneer17" or (
             report.get("gpu_encoder_verified") is True and report.get("encoder_zero_cpu_core") is True
             and all(row.get("batch") == 4 for row in report.get("encoder_native_stages", []))),
        "full_result_fields_retained": all(all(key in row["result"] for key in shared.FIELDS)
                                            and all(key in row["metrics"] for key in shared.RAW_FIELDS) for row in report["samples"]),
        "raw_bytes_retained": all(type(report["sampled_peak_totals"].get(key)) is int for key in shared.MEMORY),
        "public_CER_retained": all("public_text_cer" in row for row in report["samples"])}
    if not quality_only:
        validation["stable_five_requests"] = validation.pop("stable_expected_request_count")
    checks = {key: value for key, value in validation.items() if type(value) is bool}
    validation["all_strict_checks_passed"] = all(checks.values())
    save(output.with_suffix(".strict-validation.json"), validation)
    require(validation["all_strict_checks_passed"], "附加严格路由/身份检查失败，原始报告及拒绝证据已保留")
    shared.progress("strict_validation", variant=args.variant, passed=True, output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("stage", help="准备同1.7B两路线，不加载模型")
    p.add_argument("--source-prepared", type=Path, default=SOURCE)
    p.add_argument("--prepared", type=Path, default=STUDY / "prepared")
    p.set_defaults(function=stage)
    p = commands.add_parser("plan", help="生成prime及AB BA AB计划，不执行")
    p.add_argument("--prepared", type=Path, default=STUDY / "prepared")
    p.add_argument("--output-dir", type=Path, default=STUDY / "runs")
    p.set_defaults(function=plan)
    p = commands.add_parser("run", help="仅此子命令启动明确选择的隔离worker")
    p.add_argument("--prepared", type=Path, default=STUDY / "prepared")
    p.add_argument("--variant", choices=NAMES, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--quality-prepared", type=Path,
                   help="改用外部manifest/workload-multiseed/public-reference；只写previous-cuda/quality，不纳入性能统计")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.set_defaults(function=run)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
