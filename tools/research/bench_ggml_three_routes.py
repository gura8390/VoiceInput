"""准备并测量基准版、GGML/Vulkan预览版、GGML/CUDA先行版；默认不执行推理。

复用既有 Job/IPC/主机及WDDM采集循环。prepare、stage、plan只准备资产和命令；
run 才执行明确指定的隔离应用，正式安装不参与。旧研究证据保持原样。
"""
from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes as w
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import bench_ggml_vulkan_encoder as shared
from bench_common import CREATE_NO_WINDOW, k

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "evidence/ggml-cuda-20261117"
NAMES = {"baseline": "基准版", "preview": "预览版", "pioneer": "先行版"}
BACKENDS = {"baseline": "vulkan", "preview": "vulkan", "pioneer": "cuda"}
PROVIDERS = {"baseline": "cpu", "preview": "ggml_vulkan", "pioneer": "ggml_cuda"}
POLICY_KEYS = ("vulkan_pipeline_cache", "cuda_graph_optimization", "gpu_backend", "encoder_provider")
GRAPH_ENV = ("GGML_CUDA_GRAPH_OPT", "GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_NO_PINNED",
             "GGML_CUDA_FORCE_MMQ", "GGML_CUDA_ENABLE_UNIFIED_MEMORY", "CUDA_MODULE_LOADING")


def require(ok, message):
    shared.require(ok, message)


def read(path):
    return shared.read(path)


def sha(path):
    return shared.sha(path)


def contained(path):
    path = Path(path).resolve()
    require(path.is_relative_to(STUDY.resolve()) and path != STUDY.resolve(), "输出必须位于本轮隔离目录")
    return path


def save(path, value):
    path = contained(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def tree(folder):
    folder = Path(folder)
    if not folder.exists():
        return {}
    return {path.relative_to(folder).as_posix(): sha(path) for path in sorted(folder.rglob("*")) if path.is_file()}


def prepare(args):
    out = contained(args.prepared)
    require(not out.exists(), "准备目录已存在，请保留旧证据并使用新目录")
    source = args.source_prepared.resolve(strict=True)
    formal = args.formal_app.resolve(strict=True)
    config = read(formal / "config/settings.json")
    require(config.get("encoder_provider", "cpu") == "cpu" and config.get("gpu_backend") == "vulkan", "基准配置须为CPU ORT/Vulkan")
    require(config["asr_model"] == "1.7b", "本轮只测既有1.7B，不更换模型")
    old_manifest, work = read(source / "manifest.json"), read(source / "workload.json")
    require(len(work) == 5 and len({item["id"] for item in work}) == 4, "本轮须为五请求、四段不同PCM")
    ids = {item["id"] for item in work}
    out.mkdir(parents=True)
    (out / "inputs").mkdir()
    manifest = []
    for item in old_manifest:
        if item["id"] not in ids:
            continue
        source_pcm = Path(item["pcm"])
        require(sha(source_pcm) == item["pcm_sha256"], "源PCM身份不匹配")
        target = out / "inputs" / source_pcm.name
        shutil.copy2(source_pcm, target)
        row = {**item, "pcm": str(target)}
        if item.get("wav"):
            wav = Path(item["wav"])
            require(sha(wav) == item["wav_sha256"], "源WAV身份不匹配")
            shutil.copy2(wav, out / "inputs" / wav.name)
            row["wav"] = str(out / "inputs" / wav.name)
        manifest.append(row)
    require(len(manifest) == 4 and next(row for row in manifest if row["id"] == "file_60s")["samples"] == 960000,
            "精确60秒输入缺失")
    references = [row for row in read(source / "public-text-reference.json") if row["id"] in ids]
    save(out / "manifest.json", manifest)
    save(out / "workload.json", work)
    save(out / "public-text-reference.json", references)
    # Model path will be supplied by stage; all saved configs are separate.
    for variant in NAMES:
        folder = out / variant / "config"
        folder.mkdir(parents=True)
        settings = {**config, "app_dir": ".."}
        if variant != "baseline":
            settings.update(encoder_provider=PROVIDERS[variant], gpu_backend=BACKENDS[variant])
        save(folder / "settings.json", settings)
    save(out / "preparation.json", {
        "prepared": True, "formal_app": str(formal), "formal_settings_sha256": sha(formal / "config/settings.json"),
        "source_prepared": str(source), "model": "1.7b", "unique_pcm": 4, "requests": 5,
        "manifest_sha256": sha(out / "manifest.json"), "workload_sha256": sha(out / "workload.json"),
        "formal_application_modified": False, "models_loaded": 0,
        "stage_pending": True, "loading_policy_changed": False,
        "contract": "原0.6.0 EXE作为基准；预览/先行使用同一新EXE与同一mmproj/Q5 decoder，只改变显式运行后端。ForcedAligner仍CPU ORT。"})
    shared.progress("prepared", variants=3, requests=5, unique_pcm=4, models_loaded=0, output=str(out))


def junction(source, target):
    require(not target.exists(), "引用目录已存在，拒绝覆盖")
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    command = "New-Item -ItemType Junction -Path " + quote(target) + " -Target " + quote(source) + " | Out-Null"
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                   capture_output=True, check=True, creationflags=CREATE_NO_WINDOW, timeout=30)
    require(target.resolve() == source, "目录引用目标不一致")


def stage(args):
    prepared = contained(args.prepared).resolve(strict=True)
    prep = read(prepared / "preparation.json")
    formal = Path(prep["formal_app"])
    require(sha(formal / "config/settings.json") == prep["formal_settings_sha256"], "正式配置身份变化")
    prototype = args.research_exe.resolve(strict=True)
    projector = args.encoder_model.resolve(strict=True)
    native = {"preview": args.preview_native.resolve(strict=True), "pioneer": args.cuda_native.resolve(strict=True)}
    for variant in NAMES:
        require(not (prepared / variant / "语音输入.exe").exists(), "隔离应用已存在，拒绝覆盖")
    for folder in native.values():
        require(all((folder / name).is_file() for name in ("llama.dll", "ggml.dll", "ggml-base.dll", "caps_encoder.dll")),
                "候选原生目录缺少必要DLL")
    require((native["preview"] / "ggml-vulkan.dll").is_file(), "预览版缺少Vulkan插件")
    require((native["pioneer"] / "ggml-cuda.dll").is_file(), "先行版缺少CUDA插件")
    cache_seed = prepared / "cache-seed/vulkan"
    if (formal / "cache/vulkan").is_dir():
        shutil.copytree(formal / "cache/vulkan", cache_seed)
    staged = {}
    for variant in NAMES:
        app = prepared / variant
        source_exe = formal / "语音输入.exe" if variant == "baseline" else prototype
        shutil.copy2(source_exe, app / "语音输入.exe")
        for source in formal.glob("*.dll"):
            shutil.copy2(source, app / source.name)
        for name in ("hot.txt", "hot-rule.txt"):
            source = formal / "config" / name
            if source.is_file():
                shutil.copy2(source, app / "config" / name)
        junction(formal / "models", app / "models")
        (app / "runtime").mkdir()
        if variant == "baseline":
            junction(formal / "runtime/llama", app / "runtime/llama")
        else:
            shutil.copytree(native[variant], app / "runtime/llama")
            config_path = app / "config/settings.json"
            config = read(config_path)
            config["ggml_encoder_model"] = str(projector)
            # Only this newly prepared, never-executed candidate config changes.
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), "utf-8")
        for group in ("onnx", "audio", "media"):
            junction(formal / "runtime" / group, app / "runtime" / group)
        if cache_seed.exists():
            shutil.copytree(cache_seed, app / "cache/vulkan")
        save(prepared / f"{variant}-env.json", {})
        staged[variant] = {"name": NAMES[variant], "app": str(app), "exe_sha256": sha(app / "语音输入.exe"),
                           "settings_sha256": sha(app / "config/settings.json"),
                           "native_dll_sha256": {path.name: sha(path) for path in (app / "runtime/llama").glob("*.dll")},
                           "initial_vulkan_cache": tree(app / "cache/vulkan"),
                           "model_source": str((app / "models").resolve())}
    require(staged["preview"]["exe_sha256"] == staged["pioneer"]["exe_sha256"], "预览与先行必须使用同一EXE")
    require(len({value["model_source"] for value in staged.values()}) == 1, "三版模型必须指向同一只读源")
    require(sha(formal / "config/settings.json") == prep["formal_settings_sha256"], "正式配置被改动")
    save(prepared / "staging.json", {"staged": True, "models_loaded": 0, "formal_application_modified": False,
        "variants": staged, "mmproj": str(projector), "mmproj_sha256": sha(projector),
        "cache_policy": "三版从同一Vulkan cache种子独立复制；Windows/驱动缓存保留，CUDA模块及graph策略单独记录。",
        "loading_policy_changed": False})
    shared.progress("staged", variants=3, models_loaded=0)


class JobAccounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in ("user", "kernel", "period_user", "period_kernel")]
    _fields_ += [(name, w.DWORD) for name in ("faults", "processes", "active", "terminated")]


def job_cpu(owned):
    value = JobAccounting()
    if not k.QueryInformationJobObject(owned.handle, 1, ctypes.byref(value), ctypes.sizeof(value), None):
        raise ctypes.WinError(ctypes.get_last_error())
    return (value.user + value.kernel) / 1e7


def background(formal):
    quoted = "'" + str((formal / "语音输入.exe").resolve()).replace("'", "''") + "'"
    command = ("[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
               "@(Get-CimInstance Win32_Process | Where-Object {$_.ExecutablePath -eq " + quoted +
               "} | ForEach-Object {[pscustomobject]@{pid=$_.ProcessId;parent=$_.ParentProcessId;"
               "worker=($_.CommandLine -match '(^| )worker( |$)')}}) | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, check=True, creationflags=CREATE_NO_WINDOW, timeout=30)
    text = result.stdout.decode("utf-8-sig").strip()
    rows = json.loads(text) if text else []
    rows = rows if isinstance(rows, list) else [rows]
    require(not any(row["worker"] for row in rows), "正式模型worker正在运行，请保留它并稍后测量")
    return rows


def run(args):
    prepared = contained(args.prepared).resolve(strict=True)
    prep = read(prepared / "preparation.json")
    staging = read(prepared / "staging.json")
    app = prepared / args.variant
    formal = Path(prep["formal_app"])
    settings = read(app / "config/settings.json")
    require(settings["encoder_provider"] == PROVIDERS[args.variant] and settings["gpu_backend"] == BACKENDS[args.variant],
            "配置后端与三版标签不一致")
    require(sha(app / "语音输入.exe") == staging["variants"][args.variant]["exe_sha256"], "EXE身份在stage后变化")
    require(sha(app / "config/settings.json") == staging["variants"][args.variant]["settings_sha256"], "配置在stage后变化")
    current_native = {p.name: sha(p) for p in (app / "runtime/llama").glob("*.dll")}
    require(current_native == staging["variants"][args.variant]["native_dll_sha256"], "运行库在stage后变化")
    out = contained(args.output)
    require(not out.exists(), "输出已存在，拒绝覆盖")
    inputs = contained(args.quality_prepared).resolve(strict=True) if args.quality_prepared else prepared
    quality_only = args.quality_prepared is not None
    if quality_only:
        require(out.is_relative_to((STUDY / "quality").resolve()) and "-formal" not in out.stem,
                "质量运行须写入本轮quality目录并使用独立名称，不混入正式性能轮")
    workload = inputs / ("workload-multiseed.json" if quality_only else "workload.json")
    identity = {"variant": args.variant, "name": NAMES[args.variant],
                "staging_sha256": sha(prepared / "staging.json"),
                "native_dll_sha256": current_native,
                "configuration_policy": {key: settings.get(key) for key in POLICY_KEYS},
                "child_graph_environment": {key: os.environ.get(key) if not key.startswith("GGML_") else None for key in GRAPH_ENV},
                "vulkan_cache_before": tree(app / "cache/vulkan"), "mmproj_sha256": staging["mmproj_sha256"],
                "shared_model_source": str((app / "models").resolve()),
                "native_provider": settings["encoder_provider"], "decoder": BACKENDS[args.variant],
                "benchmark_scope": "quality_only" if quality_only else "performance",
                "quality_prepared": str(inputs) if quality_only else None}
    # The shared runner uses its own process-only globals; no old file is edited.
    shared.STUDY = STUDY
    options = SimpleNamespace(exe=app / "语音输入.exe", app_dir=app, config=app / "config/settings.json",
        manifest=inputs / "manifest.json", workload=workload,
        env_json=prepared / f"{args.variant}-env.json", reference=args.reference,
        public_reference=inputs / "public-text-reference.json", match="compare", profile=args.profile,
        preflight_only=args.preflight_only, output=out, host_ms=25.0, gpu_ms=100.0, timeout=180.0,
        expected_decoder_backend=BACKENDS[args.variant], strict_core_nodes=True,
        job_cpu_reader=job_cpu, artifact_identity=identity, quality_only=quality_only)
    before = background(formal) if not args.preflight_only else None
    try:
        shared.run(options)
    finally:
        if not args.preflight_only:
            after = background(formal)
            save(out.with_suffix(".background.json"), {"variant": args.variant, "before": before, "after": after,
                 "scope": "正式应用开始/结束快照；不控制正常托盘，不发送消息", "formal_worker_seen": False})
            save(out.with_suffix(".cache.json"), {"variant": args.variant, "before": identity["vulkan_cache_before"],
                 "after": tree(app / "cache/vulkan"), "policy": identity["configuration_policy"]})


def plan(args):
    prepared = contained(args.prepared).resolve(strict=True)
    output = contained(args.output_dir)
    entries = []
    for variant in NAMES:
        entries.append({"phase": "prime", "round": 0, "variant": variant})
    for index, order in enumerate((("baseline", "preview", "pioneer"), ("preview", "pioneer", "baseline"),
                                    ("pioneer", "baseline", "preview")), 1):
        entries += [{"phase": "formal", "round": index, "variant": variant} for variant in order]
    for entry in entries:
        tag = f"{entry['variant']}-prime" if entry["phase"] == "prime" else f"{entry['variant']}-formal{entry['round']}"
        command = [sys.executable, str(Path(__file__).resolve()), "run", "--prepared", str(prepared),
                   "--variant", entry["variant"], "--output", str(output / (tag + ".json"))]
        if entry["phase"] == "formal":
            command += ["--reference", str(output / f"{entry['variant']}-prime.json")]
        entry.update(tag=tag, command=command)
    save(output / "run-plan.json", {"prepared": str(prepared), "prime_excluded_from_statistics": True,
        "order": "Latin square, three independent process rounds per variant", "entries": entries,
        "models_started_by_plan": 0, "loading_policy_changed": False})
    script = ["$ErrorActionPreference='Stop'", "$taskPlan=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'run-plan.json') -Raw -Encoding UTF8 | ConvertFrom-Json",
              "foreach($taskEntry in $taskPlan.entries){", "    $taskCommand=@($taskEntry.command)",
              "    $taskExe=$taskCommand[0]", "    $taskArgs=@($taskCommand | Select-Object -Skip 1)",
              "    & $taskExe @taskArgs",
              "    if($LASTEXITCODE -ne 0){throw ('测量失败，证据已保留：'+$taskEntry.tag)}", "}"]
    path = contained(output / "run-serial.ps1")
    with path.open("x", encoding="utf-8-sig") as stream:
        stream.write("\n".join(script) + "\n")
    shared.progress("planned", prime=3, formal=9, models_loaded=0, plan=str(output / "run-plan.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="复制既有固定输入和三版配置，不加载模型")
    prepare_parser.add_argument("--formal-app", type=Path, required=True)
    prepare_parser.add_argument("--source-prepared", type=Path, required=True)
    prepare_parser.add_argument("--prepared", type=Path, default=STUDY / "prepared")
    stage_parser = sub.add_parser("stage", help="复制三版运行资产，原目录只读")
    stage_parser.add_argument("--prepared", type=Path, default=STUDY / "prepared")
    for name in ("research-exe", "preview-native", "cuda-native", "encoder-model"):
        stage_parser.add_argument("--" + name, type=Path, required=True)
    plan_parser = sub.add_parser("plan", help="生成三轮Latin square串行命令，暂不执行")
    plan_parser.add_argument("--prepared", type=Path, default=STUDY / "prepared")
    plan_parser.add_argument("--output-dir", type=Path, default=STUDY / "runs")
    run_parser = sub.add_parser("run", help="执行一版隔离worker；旧报告保持原样")
    run_parser.add_argument("--prepared", type=Path, default=STUDY / "prepared")
    run_parser.add_argument("--variant", choices=NAMES, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--reference", type=Path)
    run_parser.add_argument("--quality-prepared", type=Path,
                            help="只替换此轮manifest/多种子workload/公开文本参考；质量输出放quality目录")
    run_parser.add_argument("--profile", action="store_true")
    run_parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "stage": stage, "plan": plan, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
