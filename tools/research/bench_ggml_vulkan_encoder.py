"""同模型 CPU ORT / GGML Vulkan 编码器研究：准备输入及隔离配置，观测 worker。

默认只读预检；run 才启动明确指定的隔离 worker。保持原有模型加载顺序，
不设置分阶段加载开关，不操作正式桌面、自启动或正式配置。
"""
from pathlib import Path
import argparse
import array
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
import wave

from bench_common import OwnedTree, Sampler, call, receive, timed_io, result_fields, FIELDS, CREATE_NO_WINDOW
from bench_cuda_route import host_snapshot_valid, gpu_worker_covered
from cuda_metrics import Collector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_layout import ROOT, SETTINGS

STUDY = ROOT / "evidence/ggml-vulkan-encoder-20261117"
MEMORY = ("private_bytes", "private_working_set", "working_set")
RAW_FIELDS = ("text", "token_ids", "n_prefill", "n_generate", "is_aborted")
TIMES = ("t_encode", "t_prefill", "t_generate", "t_context")
WORK = [("warmed_first_short", "aishell_zh", "mic"), ("warm_zh", "hello_zh", "mic"),
        ("warm_en", "hello_en", "mic"), ("file_60s", "file_60s", "file"),
        ("tail_short", "aishell_zh", "mic")]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def output_path(path):
    path = Path(path).resolve()
    require(path.is_relative_to(STUDY.resolve()) and path != STUDY.resolve(), "研究输出必须位于本轮隔离目录")
    return path


def save_new(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def progress(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def cer(actual, reference):
    actual, reference = normalize_text(actual), normalize_text(reference)
    if not reference:
        return None
    row = list(range(len(actual) + 1))
    for i, left in enumerate(reference, 1):
        new = [i]
        for j, right in enumerate(actual, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = new
    return {"edits": row[-1], "reference_characters": len(reference), "cer": row[-1] / len(reference),
            "normalization": "NFKC + casefold; remove whitespace and Unicode punctuation; character metric"}


def prepare(args):
    out = output_path(args.output_dir)
    require(not out.exists(), "准备目录已存在，请使用新目录，保留原有证据")
    formal = args.formal_app.resolve(strict=True)
    source_manifest = args.source_manifest.resolve(strict=True)
    originals = read(source_manifest)
    by_id = {item["id"]: item for item in originals}
    settings_path = formal / SETTINGS
    settings = read(settings_path)
    require(settings.get("gpu_backend") == "vulkan" and settings.get("encoder_provider", "cpu") == "cpu",
            "基线须为现用 CPU ORT 编码器与 Vulkan 解码器")
    require(settings.get("asr_model") in ("0.6b", "1.7b"), "基线模型选择无效")
    corpus_path = args.public_corpus_manifest.resolve(strict=True)
    public = {item["id"]: item for item in read(corpus_path)["clips"]}
    wav = args.long_wav.resolve(strict=True)
    with wave.open(str(wav), "rb") as reader:
        require((reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) == (1, 2, 16000),
                "长源音频须为 16kHz 单声道 PCM16")
        require(reader.getnframes() >= 960000, "长源音频不足精确 60 秒")
        frames = reader.readframes(960000)
        long_frames = reader.getnframes()
    # Validate every copied short input before creating the output directory.
    for name in ("aishell_zh", "hello_zh", "hello_en", "long_mixed"):
        source = Path(by_id[name]["pcm"])
        require(sha(source) == by_id[name]["pcm_sha256"], f"固定 PCM 身份不匹配：{name}")
        require(source.stat().st_size % 4 == 0, f"PCM 长度无效：{name}")
    out.mkdir(parents=True)
    inputs = out / "inputs"
    inputs.mkdir()
    manifest, references = [], []
    for name in ("aishell_zh", "hello_zh", "hello_en", "long_mixed"):
        source = Path(by_id[name]["pcm"])
        target = inputs / (name + ".f32")
        shutil.copy2(source, target)
        sample = {"id": name, "pcm": str(target), "pcm_sha256": sha(target),
                  "sample_rate": 16000, "channels": 1, "encoding": "f32le",
                  "samples": target.stat().st_size // 4, "duration_seconds": target.stat().st_size / 64000}
        manifest.append(sample)
        item = public.get(name)
        matched = False
        if item and item.get("reference") and item.get("sha256"):
            sibling = source.with_suffix(".wav")
            matched = sibling.is_file() and sha(sibling) == item["sha256"]
        references.append({"id": name, "pcm_sha256": sample["pcm_sha256"],
                           "ground_truth": item["reference"] if matched else None,
                           "reference_verified_by_source_wav_sha256": matched,
                           "language": item.get("language") if item else None,
                           "source_url": item.get("source_url") if item else None,
                           "reference_url": item.get("reference_url") if item else None,
                           "source_manifest": str(corpus_path), "source_manifest_sha256": sha(corpus_path)})
    target_wav = inputs / "file_60s.wav"
    with wave.open(str(target_wav), "wb") as writer:
        writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        writer.writeframes(frames)
    integers = array.array("h", frames)
    if sys.byteorder != "little":
        integers.byteswap()
    floats = array.array("f", (value / 32768.0 for value in integers))
    if sys.byteorder != "little":
        floats.byteswap()
    target = inputs / "file_60s.f32"
    with target.open("xb") as stream:
        stream.write(floats.tobytes())
    require(target.stat().st_size == 960000 * 4, "60 秒 PCM 样本数不匹配")
    manifest.append({"id": "file_60s", "pcm": str(target), "pcm_sha256": sha(target),
                     "sample_rate": 16000, "channels": 1, "encoding": "f32le", "samples": 960000,
                     "duration_seconds": 60.0, "wav": str(target_wav), "wav_sha256": sha(target_wav),
                     "source_wav": str(wav), "source_wav_sha256": sha(wav), "source_total_frames": long_frames,
                     "cut_start_frame": 0, "cut_end_frame_exclusive": 960000,
                     "conversion": "PCM16 signed sample / 32768.0 -> little-endian float32; no resampling"})
    references.append({"id": "file_60s", "pcm_sha256": sha(target), "ground_truth": None,
                       "reason": "固定 60 秒切片未配套逐字稿，不把 ASR 输出作为人工真值"})
    baseline = out / "baseline"
    (baseline / "config").mkdir(parents=True)
    # This is a configuration template. Stage runtime/models before execution;
    # no junction is created and no formal runtime/cache is written here.
    config = {**settings, "app_dir": ".."}
    save_new(baseline / SETTINGS, config)
    save_new(out / "manifest.json", manifest)
    save_new(out / "public-text-reference.json", references)
    save_new(out / "workload.json", [{"phase": phase, "id": name, "source": source, "seed": 42}
                                      for phase, name, source in WORK])
    save_new(out / "preparation.json", {
        "prepared": True, "models_loaded": 0, "model_processes_started": 0,
        "formal_application_modified": False, "formal_app": str(formal),
        "formal_settings_sha256": sha(settings_path), "formal_settings": settings,
        "selected_asr_model": settings["asr_model"], "baseline_configuration": str(baseline / SETTINGS),
        "runtime_reference_read_only": str(formal / "runtime"), "model_reference_read_only": str(formal / "models"),
        "runtime_staging_pending": True,
        "staging_contract": {
            "baseline": "Isolated app root/config/cache; formal runtime/models are read-only inputs. Do not execute against formal app root.",
            "candidate": "Copy the accepted b9090 runtime into an isolated app, add caps encoder DLL; keep the selected ASR model and decoder GGUF unchanged. ForcedAligner stays CPU ORT.",
            "cache": "Both sides use their own app/cache; prime and synchronize policy before formal A/B.",
            "loading_policy": "Preserve current loading policy; no YYUR_ASR_INIT or staged loading experiments."},
        "source_manifest": str(source_manifest), "source_manifest_sha256": sha(source_manifest),
        "public_corpus_manifest_sha256": sha(corpus_path), "manifest_sha256": sha(out / "manifest.json"),
        "workload_sha256": sha(out / "workload.json"), "sample_count": len(manifest),
        "exact_file_seconds": 60.0, "seed": 42,
        "quality_policy": "Fixed-source regression and verified public text references; unannotated slice has no CER verdict."})
    progress("prepared", inputs=len(manifest), exact_file_seconds=60, models_loaded=0,
             public_text_references=sum(bool(item.get("ground_truth")) for item in references), output=str(out))


def validate_inputs(manifest_path, workload_path, require_sixty=True):
    manifest, work = read(manifest_path), read(workload_path)
    require(isinstance(manifest, list) and manifest and isinstance(work, list) and work, "输入清单或请求顺序为空")
    by_id = {}
    for item in manifest:
        require(item["id"] not in by_id, "输入 ID 重复")
        path = Path(item["pcm"])
        require(path.is_file() and sha(path) == item["pcm_sha256"], f"PCM 身份不匹配：{item['id']}")
        require(path.stat().st_size % 4 == 0 and 0 < path.stat().st_size <= 80 * 64000, "PCM 超过模型窗口或字节数无效")
        require(item.get("encoding") == "f32le" and item.get("sample_rate") == 16000, "PCM 格式无效")
        by_id[item["id"]] = item
    if require_sixty:
        require(any(item["id"] == "file_60s" and item["samples"] == 960000 for item in manifest), "清单缺少精确 60 秒样本")
    for item in work:
        require(item["id"] in by_id and item["source"] in ("mic", "file"), "工作负载引用无效")
        require(type(item["seed"]) is int and 0 <= item["seed"] <= 0xFFFFFFFF, "种子无效")
    return by_id, work


def stage(args):
    prepared = output_path(args.prepared).resolve(strict=True)
    receipt = read(prepared / "preparation.json")
    formal = Path(receipt["formal_app"]).resolve(strict=True)
    require(sha(formal / SETTINGS) == receipt["formal_settings_sha256"], "正式配置在准备后发生变化")
    prototype = args.candidate_exe.resolve(strict=True)
    native = args.encoder_dll.resolve(strict=True)
    model = args.encoder_model.resolve(strict=True)
    baseline, candidate = prepared / "baseline", prepared / "candidate"
    require(not (baseline / "语音输入.exe").exists() and not candidate.exists(), "隔离应用已存在，拒绝覆盖")

    def link(source, target):
        source = source.resolve(strict=True)
        require(not target.exists(), "目录引用目标已存在")
        target.parent.mkdir(parents=True, exist_ok=True)
        quote = lambda value: "'" + str(value).replace("'", "''") + "'"
        command = "New-Item -ItemType Junction -Path " + quote(target) + " -Target " + quote(source) + " | Out-Null"
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                       check=True, capture_output=True, creationflags=CREATE_NO_WINDOW, timeout=30)
        require(target.resolve() == source, "只读资产引用目标不一致")

    (candidate / "config").mkdir(parents=True)
    save_new(candidate / SETTINGS, read(baseline / SETTINGS))
    for app, executable in ((baseline, formal / "语音输入.exe"), (candidate, prototype)):
        shutil.copy2(executable, app / "语音输入.exe")
        for source in formal.glob("*.dll"):
            shutil.copy2(source, app / source.name)
        for name in ("hot.txt", "hot-rule.txt"):
            source = formal / "config" / name
            if source.is_file():
                shutil.copy2(source, app / "config" / name)
        link(formal / "models", app / "models")
        if (formal / "cache/vulkan").is_dir():
            shutil.copytree(formal / "cache/vulkan", app / "cache/vulkan")
    link(formal / "runtime", baseline / "runtime")
    (candidate / "runtime").mkdir()
    for source in (formal / "runtime").iterdir():
        if source.name == "llama":
            shutil.copytree(source, candidate / "runtime/llama")
        elif source.is_dir():
            link(source, candidate / "runtime" / source.name)
        else:
            shutil.copy2(source, candidate / "runtime" / source.name)
    shutil.copy2(native, candidate / "runtime/llama/caps_encoder.dll")
    env = {"YYUR_ENCODER_RESEARCH_PROVIDER": "ggml_vulkan",
           "YYUR_GGML_NATIVE_DIR": str(candidate / "runtime/llama"),
           "YYUR_GGML_ENCODER_MODEL": str(model)}
    save_new(prepared / "baseline-env.json", {})
    save_new(prepared / "candidate-env.json", env)
    save_new(prepared / "candidate-embedding-env.json", {
        **env, "YYUR_ENCODER_EMBEDDINGS_DIR": str(STUDY / "diagnostics/candidate-embeddings")})
    save_new(prepared / "ort-embedding-env.json", {
        "YYUR_ENCODER_RESEARCH_PROVIDER": "ort",
        "YYUR_ENCODER_EMBEDDINGS_DIR": str(STUDY / "diagnostics/ort-embeddings")})
    originals = {p.name: sha(p) for p in (formal / "runtime/llama").glob("*.dll")}
    copied = {name: sha(candidate / "runtime/llama" / name) for name in originals}
    require(copied == originals, "候选既有解码运行库身份发生变化")
    require((baseline / "models").resolve() == (candidate / "models").resolve(), "两端模型源必须相同")
    require(sha(formal / SETTINGS) == receipt["formal_settings_sha256"], "正式配置被改动")
    save_new(prepared / "staging.json", {
        "staged": True, "models_loaded": 0, "formal_application_modified": False,
        "baseline_app": str(baseline), "candidate_app": str(candidate),
        "baseline_exe_sha256": sha(baseline / "语音输入.exe"),
        "candidate_exe_sha256": sha(candidate / "语音输入.exe"),
        "encoder_dll_sha256": sha(native), "encoder_model_sha256": sha(model),
        "unchanged_decoder_runtime": originals, "candidate_decoder_runtime_exact": True,
        "shared_model_source": str((baseline / "models").resolve()),
        "runtime_and_models_policy": "Formal assets are shared read-only; candidate llama DLLs are independent copies. All writable app/config/cache/log outputs stay in this study.",
        "loading_policy_changed": False,
        "cache_policy": "Both apps start with identical copies of the current formal Vulkan cache; no shared writable cache."})
    progress("staged", models_loaded=0, baseline=str(baseline), candidate=str(candidate))


class GpuSampler:
    """Dedicated/shared WDDM samples, kept separate from host commitment."""
    def __init__(self, owned, started, interval):
        self.owned, self.started, self.interval = owned, started, interval
        self.collector = Collector()
        self.stop = threading.Event()
        self.samples, self.errors = [], []
        self.thread = threading.Thread(target=self.run, name="ggml-gpu-memory", daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                value = self.collector.snapshot(self.owned.ids())
                value["elapsed_seconds"] = time.perf_counter() - self.started
                self.samples.append(value)
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")
            self.stop.wait(self.interval)

    def finish(self):
        self.stop.set()
        self.thread.join()


def json_markers(text, marker):
    records, position = [], 0
    while True:
        found = text.find(marker, position)
        if found < 0:
            return records
        position = found + len(marker)
        try:
            record, count = json.JSONDecoder().raw_decode(text[position:])
            records.append(record)
            position += count
        except json.JSONDecodeError:
            position += 1


def preflight(args):
    app, exe = args.app_dir.resolve(strict=True), args.exe.resolve(strict=True)
    require(app.is_relative_to(STUDY.resolve()), "worker 必须使用本轮 evidence 中的隔离应用根目录")
    config_path = (args.config or app / SETTINGS).resolve(strict=True)
    require(config_path.is_relative_to(STUDY.resolve()), "配置必须位于本轮研究目录")
    config = read(config_path)
    expected_decoder = getattr(args, "expected_decoder_backend", "vulkan")
    require(config.get("gpu_backend") == expected_decoder, f"解码器配置须为 {expected_decoder}")
    require(config.get("asr_model") in ("0.6b", "1.7b"), "模型选择无效")
    require((app / "runtime/llama/llama.dll").is_file(), "隔离应用未准备 llama 运行库")
    require((app / "models").is_dir(), "隔离应用未准备只读模型视图")
    require((app / "runtime/onnx/onnxruntime.dll").is_file(), "须保留原有 CPU ORT/ForcedAligner 运行库")
    by_id, work = validate_inputs(args.manifest, args.workload, require_sixty=not getattr(args, "quality_only", False))
    # Explicit app override ensures cache/log paths use the isolated root.
    effective = {**config, "app_dir": str(app)}
    env_values = read(args.env_json) if args.env_json else {}
    require(isinstance(env_values, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env_values.items()),
            "环境配置必须为字符串键值 JSON")
    require("YYUR_ASR_INIT" not in env_values, "本轮不使用分阶段加载开关")
    reference = read(args.reference) if args.reference else None
    if reference:
        require(reference.get("execution_passed") and reference.get("cleanup_confirmed"), "参考运行未成功完成")
        require(reference["manifest_sha256"] == sha(args.manifest)
                and reference["workload_sha256"] == sha(args.workload), "参考输入或顺序身份不一致")
        require(len(reference["samples"]) == len(work), "参考请求数不一致")
        require(reference["settings"]["asr_model"] == config["asr_model"], "必须比较相同 Qwen 模型")
    return app, exe, config_path, effective, by_id, work, env_values, reference


def gpu_peaks(samples, worker):
    # Hybrid-GPU machines can expose both dGPU and iGPU instances for the same
    # worker. Keep every PID/LUID/physical/duplicate identity separately. A valid
    # measured zero on one adapter must not invalidate the other adapter, and
    # absent samples must not silently become zero or a cross-adapter total.
    groups, invalid, missing = {}, [], []
    worker_samples = 0
    for index, sample in enumerate(samples):
        gpu = sample.get("gpu_process_memory", {})
        rows = [row for row in gpu.get("rows", []) if row.get("pid") == worker]
        if not rows:
            missing.append(index)
            continue
        worker_samples += 1
        if gpu.get("sampling_ok") is not True:
            invalid.append({"sample": index, "reason": "WDDM 计数采集不完整"})
        for row in rows:
            valid = row.get("adapter_luid") and not row.get("errors") and all(
                type(row.get(key)) is int and row[key] >= 0 for key in ("dedicated_bytes", "shared_bytes"))
            if not valid:
                invalid.append({"sample": index, "adapter_luid": row.get("adapter_luid"), "errors": row.get("errors")})
                continue
            key = (row["pid"], row["adapter_luid"], row.get("physical_adapter_index"), row.get("duplicate_instance_index"))
            groups.setdefault(key, []).append((index, row))
    adapters = []
    for key, values in sorted(groups.items(), key=lambda entry: str(entry[0])):
        seen = {index for index, row in values}
        adapters.append({"pid": key[0], "adapter_luid": key[1], "physical_adapter_index": key[2],
                         "duplicate_instance_index": key[3], "valid_samples": len(values),
                         "not_observed_sample_indices": [index for index in range(len(samples)) if index not in seen],
                         "dedicated_bytes": max(row["dedicated_bytes"] for index, row in values),
                         "shared_bytes": max(row["shared_bytes"] for index, row in values)})
    single = adapters[0] if len(adapters) == 1 else None
    return {"available": bool(adapters) and not invalid, "adapters": adapters,
            "single_adapter": len(adapters) == 1,
            "adapter_luid": single["adapter_luid"] if single else None,
            "dedicated_bytes": single["dedicated_bytes"] if single else None,
            "shared_bytes": single["shared_bytes"] if single else None,
            "worker_sample_count": worker_samples, "worker_not_observed_sample_indices": missing,
            "invalid_worker_samples": invalid, "cross_adapter_aggregate": None,
            "scope": "逐 PID/LUID/physical/duplicate 的采样观测峰值；不推导跨适配器总值；缺测保持显式索引，真实零值保留"}


def run(args):
    app, exe, config_path, settings, by_id, work, extra_env, reference = preflight(args)
    encoder_provider = extra_env.get("YYUR_ENCODER_RESEARCH_PROVIDER", settings.get("encoder_provider", "cpu"))
    encoder_device = {"ggml_vulkan": "Vulkan0", "ggml_cuda": "CUDA0"}.get(encoder_provider)
    decoder_device = "CUDA0" if getattr(args, "expected_decoder_backend", "vulkan") == "cuda" else "Vulkan0"
    job_cpu_reader = getattr(args, "job_cpu_reader", None)
    if args.preflight_only:
        progress("preflight", passed=True, requests=len(work), models_loaded=0,
                 asr_model=settings["asr_model"], exe_sha256=sha(exe))
        return
    out = output_path(args.output)
    require(not out.exists() and not out.with_suffix(".logs").exists(), "输出或日志目录已存在")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": 1, "complete": False, "execution_passed": False, "measurement_passed": False,
              "passed": False, "executable": str(exe), "binary_sha256": sha(exe), "app_dir": str(app),
              "settings": settings, "settings_sha256": sha(config_path),
              "manifest_sha256": sha(args.manifest), "workload_sha256": sha(args.workload),
              "reference_sha256": sha(args.reference) if args.reference else None,
              "quality_checked": reference is not None, "outputs_exact": None,
              "match_policy": args.match,
              "diagnostic": args.profile or bool(extra_env.get("YYUR_ENCODER_EMBEDDINGS_DIR")),
              "research_environment": extra_env,
              "encoder_provider": encoder_provider, "expected_decoder_device": decoder_device,
              "artifact_identity": getattr(args, "artifact_identity", None),
              "host_sample_target_ms": args.host_ms, "gpu_sample_target_ms": args.gpu_ms,
              "loading_policy_changed": False, "external_audio_upload": False,
              "samples": [], "stages": [], "error": None,
              "memory_scope": "新建 service/worker Job 树；主机 private/WS 与 WDDM dedicated/shared 分开记录；均为采样观测值",
              "timing_scope": "同机新进程，保留 Windows/驱动缓存；完整 IPC 为主计时，原生 t_generate 只覆盖最后一次重试",
              "quality_scope": "固定样本/种子五字段回归；有配套公开逐字稿的输入另外计算字符错误率"}
    public = {item["id"]: item for item in read(args.public_reference)} if args.public_reference else {}
    owned, proc, host, gpu = OwnedTree(), None, None, None
    started = time.perf_counter()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("YYUR_", "GGML_"))
           and k not in ("VK_INSTANCE_LAYERS", "VK_LAYER_PATH")}
    env.update(extra_env)
    logs = out.with_suffix(".logs")
    env["YYUR_LOG_DIR"] = str(logs)
    if args.profile:
        env.update(YYUR_PROFILE_LOAD="1", YYUR_ENCODER_TIMING="1", YYUR_ALIGNER_TIMING="1")
    worker = None

    def snapshot(name):
        nonlocal worker
        stage = {"stage": name, "elapsed_seconds": time.perf_counter() - started, **owned.snapshot()}
        if job_cpu_reader:
            stage["job_cpu_seconds"] = job_cpu_reader(owned)
        require(host_snapshot_valid(stage), "主机内存快照不完整")
        children = [pid for pid in owned.ids() if pid != proc.pid]
        require(len(children) == 1 and (worker is None or children == [worker]), "须保持单一稳定模型 worker")
        worker = children[0]
        stage["native_gpu"] = gpu.collector.snapshot([worker])
        stage["gpu_worker_covered"] = gpu_worker_covered(stage["native_gpu"], worker)
        report["stages"].append(stage)

    try:
        gpu = GpuSampler(owned, started, args.gpu_ms / 1000)
        with out.with_suffix(".stderr.txt").open("xb") as stderr:
            proc = subprocess.Popen([str(exe), "--app-dir", str(app), "--config", str(config_path), "serve"],
                                    env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                    creationflags=CREATE_NO_WINDOW)
            owned.assign(proc)
            report["service_pid"] = proc.pid
            host = Sampler(owned, started, args.host_ms / 1000)
            host.thread.start()
            gpu.thread.start()
            require(timed_io(owned, 30, lambda: receive(proc))["kind"] == "boot", "服务启动握手失败")
            report["boot_seconds"] = time.perf_counter() - started
            tick = time.perf_counter()
            ready = call(proc, owned, args.timeout, {"command": "warmup", "id": 1})
            report["warmup_seconds"] = time.perf_counter() - tick
            report["model_ready_seconds"] = time.perf_counter() - started
            report["warmup_metrics"] = ready.get("metrics")
            snapshot("ready")
            for index, item in enumerate(work):
                sample = by_id[item["id"]]
                pcm = Path(sample["pcm"]).read_bytes()
                request = {"command": "segment", "id": index + 2, "task_id": f"ggml-ab-{index}",
                           "source": item["source"], "language": "auto", "context": "", "offset": 0.0,
                           "overlap": 4.0, "is_final": True, "time_start": 1700000000.0,
                           "time_submit": 1700000000.0, "seed": item["seed"]}
                tick = time.perf_counter()
                reply = call(proc, owned, args.timeout, request, pcm)
                latency = time.perf_counter() - tick
                if index == 0:
                    report["first_request_seconds"] = latency
                    report["model_ready_plus_first_request_seconds"] = report["model_ready_seconds"] + latency
                    report["observed_first_complete_seconds"] = time.perf_counter() - started
                    report["first_complete_timing_scope"] = "ready+first excludes intervening measurement snapshots; observed wall includes those snapshots"
                value = result_fields(reply)
                differences, raw_differences = [], []
                if reference:
                    old = reference["samples"][index]
                    require(all(old[key] == item[key] for key in ("phase", "id", "source", "seed"))
                            and old["pcm_sha256"] == sample["pcm_sha256"], "参考请求身份不一致")
                    differences = [key for key in FIELDS if old["result"][key] != value[key]]
                    raw_differences = [key for key in RAW_FIELDS if old["metrics"].get(key) != reply["metrics"].get(key)]
                for key in TIMES:
                    require(type(reply["metrics"].get(key)) in (int, float)
                            and math.isfinite(reply["metrics"][key]) and reply["metrics"][key] >= 0,
                            f"原生阶段计时缺失：{key}")
                ground = public.get(item["id"], {})
                if ground:
                    require(ground["pcm_sha256"] == sample["pcm_sha256"], "公开逐字稿 PCM 身份不一致")
                report["samples"].append({**item, "pcm_sha256": sample["pcm_sha256"],
                    "result": value, "metrics": reply["metrics"], "latency_seconds": latency,
                    "differences": differences, "raw_differences": raw_differences,
                    "public_text_cer": cer(reply["metrics"]["text"], ground["ground_truth"])
                    if ground.get("ground_truth") else None})
                snapshot(item["phase"])
                progress("request_complete", index=index, phase=item["phase"], seconds=round(latency, 6),
                         changed_fields=len(differences), changed_raw_fields=len(raw_differences))
            report["worker_pid"] = worker
            report["total_request_seconds"] = sum(row["latency_seconds"] for row in report["samples"])
            report["stage_seconds_sum"] = {key: sum(row["metrics"][key] for row in report["samples"]) for key in TIMES}
            report["outputs_exact"] = (not any(row["differences"] or row["raw_differences"]
                                               for row in report["samples"])) if reference else None
            report["complete"] = True
            call(proc, owned, 15, {"command": "shutdown", "id": 10000})
            proc.stdin.close()
            proc.wait(timeout=15)
            report["execution_passed"] = proc.returncode == 0
            if job_cpu_reader:
                report["job_cpu_seconds"] = job_cpu_reader(owned)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            if host:
                host.finish()
                report["memory_samples"] = host.samples
                report["sampling_errors"] = host.errors
                valid = bool(host.samples) and not host.errors and all(host_snapshot_valid(s) for s in host.samples)
                report["sampled_peak_totals"] = {key: max(s["totals"][key] for s in host.samples) if valid else None for key in MEMORY}
                report["measurement_passed"] = valid and bool(report["stages"])
            if gpu:
                if gpu.thread.ident is not None:
                    gpu.finish()
                report["gpu_samples"] = gpu.samples
                report["gpu_sampling_errors"] = gpu.errors
                report["sampled_gpu_peak"] = gpu_peaks(gpu.samples, worker)
                report["gpu_measurement_passed"] = bool(report["stages"]) and not gpu.errors and all(
                    stage["gpu_worker_covered"] for stage in report["stages"]) and report["sampled_gpu_peak"]["available"]
                report["gpu_collector_cleanup"] = gpu.collector.close()
            owned.kill()
            if proc is not None:
                proc.wait(timeout=15)
            deadline = time.monotonic() + 5
            while owned.ids() and time.monotonic() < deadline:
                time.sleep(.02)
            report["cleanup_confirmed"] = not owned.ids()
        except BaseException as error:
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
        finally:
            owned.close()
        log = logs / "rust-worker.log"
        if log.is_file():
            text = log.read_text("utf-8", errors="replace")
            report["worker_log_sha256"] = sha(log)
            report["load_profiles"] = json_markers(text, "YYUR_LOAD_PROFILE ")
            report["aligner_profiles"] = json_markers(text, "YYUR_ALIGNER_TIMING ")
            report["encoder_embeddings"] = json_markers(text, "YYUR_ENCODER_EMBEDDING ")
            report["encoder_route"] = json_markers(text, "YYUR_GGML_ENCODER ")
            report["encoder_native_stages"] = json_markers(text, "YYUR_GGML_NATIVE_STAGES ")
            encode = json_markers(text, "YYUR_GGML_ENCODE ")
            report["encoder_graph_records"] = encode
            if encoder_device:
                # The adapter performs its ordinary two-second silence warmup
                # before the five actual calls. Keep that work distinct.
                route_ok = bool(report["encoder_route"]) and all(
                    encoder_device.lower() == str(record.get("backend", "")).lower() for record in report["encoder_route"])
                count_ok = len(encode) == len(report["samples"]) + 1
                node_ok = bool(encode) and all(type(record.get("gpu_nodes")) is int and record["gpu_nodes"] > 0
                    and type(record.get("cpu_nodes")) is int and record["cpu_nodes"] >= 0
                    and type(record.get("total_nodes")) is int and record["total_nodes"] > 0
                    for record in encode)
                report["gpu_encoder_verified"] = route_ok and count_ok and node_ok
                native = report["encoder_native_stages"]
                core_keys = ("fe_gpu_core", "fe_cpu_core", "be_gpu_core", "be_cpu_core")
                report["encoder_core_metrics_complete"] = len(native) == len(report["samples"]) + 1 and all(
                    all(type(record.get(key)) is int and record[key] >= 0 for key in core_keys) for record in native)
                report["encoder_zero_cpu_core"] = report["encoder_core_metrics_complete"] and all(
                    record["fe_gpu_core"] > 0 and record["be_gpu_core"] > 0
                    and record["fe_cpu_core"] == 0 and record["be_cpu_core"] == 0 for record in native)
                if getattr(args, "strict_core_nodes", False):
                    report["gpu_encoder_verified"] &= report["encoder_zero_cpu_core"]
                if count_ok:
                    report["encoder_load_warmup_graph"] = encode[0]
                    for sample, metrics in zip(report["samples"], encode[1:]):
                        sample["encoder_graph"] = metrics
                if report["encoder_core_metrics_complete"]:
                    report["encoder_load_warmup_native"] = native[0]
                    for sample, metrics in zip(report["samples"], native[1:]):
                        sample["encoder_native_stages"] = metrics
            layers = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)
            report["vulkan_decoder_verified"] = "using device Vulkan0" in text and bool(layers) and all(
                int(gpu_layers) == int(total) and int(total) > 0 for gpu_layers, total in layers)
            report["decoder_backend_verified"] = f"using device {decoder_device}" in text and ("29", "29") in layers and all(
                int(gpu_layers) == int(total) for gpu_layers, total in layers)
            report["decoder_offload_records"] = [{"gpu_layers": int(a), "total_layers": int(b)} for a, b in layers]
        report["passed"] = bool(report["execution_passed"] and report["measurement_passed"]
            and report.get("gpu_measurement_passed") and report.get("cleanup_confirmed") and not report["error"]
            and not report.get("cleanup_error") and report.get("decoder_backend_verified")
            and (not encoder_device or report.get("gpu_encoder_verified"))
            and (reference is None or args.match == "compare" or report["outputs_exact"] is True))
        report["elapsed_seconds"] = time.perf_counter() - started
        save_new(out, report)
    progress("complete", passed=report["passed"], requests=len(report["samples"]),
             outputs_exact=report["outputs_exact"], report=str(out))
    if not report["passed"]:
        raise SystemExit(1)


def check(args):
    by_id, work = validate_inputs(args.manifest, args.workload)
    # No model/session/GPU collector initialization in this command.
    require(normalize_text("Ａ B，c。") == "abc", "文字归一化自检失败")
    require(cer("你好", "你好")["edits"] == 0 and cer("你", "你好")["edits"] == 1, "字符错误率自检失败")
    require(json_markers('prefix YYUR_LOAD_PROFILE {"stage":"x"}\n', "YYUR_LOAD_PROFILE ") == [{"stage": "x"}],
            "诊断 JSON 解析自检失败")
    row = lambda luid, dedicated: {"pid": 7, "adapter_luid": luid, "physical_adapter_index": 0,
                                  "duplicate_instance_index": None, "dedicated_bytes": dedicated,
                                  "shared_bytes": 0, "errors": []}
    test = gpu_peaks([{"gpu_process_memory": {"sampling_ok": True, "rows": []}},
                      {"gpu_process_memory": {"sampling_ok": True, "rows": [row("dgpu", 100), row("igpu", 0)]}}], 7)
    require(test["available"] and len(test["adapters"]) == 2 and test["dedicated_bytes"] is None
            and test["worker_not_observed_sample_indices"] == [0], "双适配器与缺测显式记录自检失败")
    progress("checked", passed=True, unique_inputs=len(by_id), requests=len(work), models_loaded=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare", help="只准备固定输入及隔离基线配置，不加载模型")
    prep.add_argument("--formal-app", type=Path, required=True)
    prep.add_argument("--source-manifest", type=Path, required=True)
    prep.add_argument("--public-corpus-manifest", type=Path, required=True)
    prep.add_argument("--long-wav", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    verify = subs.add_parser("check", help="验证固定 PCM、工作负载和纯函数，不启动模型")
    staging = subs.add_parser("stage", help="准备隔离应用和候选环境；只复制资产，不加载模型")
    staging.add_argument("--prepared", type=Path, required=True)
    staging.add_argument("--candidate-exe", type=Path, required=True)
    staging.add_argument("--encoder-dll", type=Path, required=True)
    staging.add_argument("--encoder-model", type=Path, required=True)
    execute = subs.add_parser("run", help="启动指定隔离 worker；正式安装不参与")
    for sub in (verify, execute):
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--workload", type=Path, required=True)
    execute.add_argument("--exe", type=Path, required=True)
    execute.add_argument("--app-dir", type=Path, required=True)
    execute.add_argument("--config", type=Path)
    execute.add_argument("--env-json", type=Path)
    execute.add_argument("--reference", type=Path)
    execute.add_argument("--public-reference", type=Path)
    execute.add_argument("--match", choices=("exact", "compare"), default="exact")
    execute.add_argument("--profile", action="store_true", help="另行诊断轮启用原生加载/编码/对齐计时")
    execute.add_argument("--preflight-only", action="store_true", help="只检查，不启动模型")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--host-ms", type=float, default=25)
    execute.add_argument("--gpu-ms", type=float, default=100)
    execute.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "check":
        check(args)
    elif args.command == "stage":
        stage(args)
    else:
        require(5 <= args.host_ms <= 1000 and 25 <= args.gpu_ms <= 2000 and 1 <= args.timeout <= 600,
                "采样周期或超时无效")
        run(args)


if __name__ == "__main__":
    main()
