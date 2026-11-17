"""准备/分析三路线质量对照；只有显式 cli-run --execute 才启动隔离 CLI。

不把与基准输出不同等同于识别变差。原始文本、格式化文本、Unicode规范化
词文、人工逐字稿CER、导出结构和时间戳分别报告。默认1.7B、batch4、三种子。
"""
from __future__ import annotations

import argparse
import array
import difflib
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import wave

from bench_common import OwnedTree, CREATE_NO_WINDOW

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "evidence/ggml-cuda-20261117"
PRIOR = ROOT / "evidence/ggml-vulkan-encoder-20261117/prepared"
SEEDS = (42, 314159, 20261117)
LABELS = {"baseline": "基准版0.6.0", "preview": "预览版GGML/Vulkan", "pioneer": "先行版GGML/CUDA"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_output(path):
    path = Path(path).resolve()
    require(path.is_relative_to(STUDY.resolve()) and path != STUDY.resolve(), "输出必须位于本轮隔离研究目录")
    require(not path.exists(), "拒绝覆盖已有证据")
    return path


def save(path, value):
    path = checked_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def lexical(text):
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold()
                   if not char.isspace() and not unicodedata.category(char).startswith("P"))


def edit_distance(actual, reference):
    row = list(range(len(actual) + 1))
    for index, left in enumerate(reference, 1):
        new = [index]
        for column, right in enumerate(actual, 1):
            new.append(min(new[-1] + 1, row[column] + 1, row[column - 1] + (left != right)))
        row = new
    return row[-1]


def cer(actual, reference):
    if reference is None:
        return None
    actual, reference = lexical(actual), lexical(reference)
    require(bool(reference), "逐字稿规范化后为空")
    edits = edit_distance(actual, reference)
    return {"edits": edits, "reference_characters": len(reference), "cer": edits / len(reference)}


def wav_samples(path):
    with wave.open(str(path), "rb") as stream:
        require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (1, 2, 16000),
                "公开输入须为16kHz单声道PCM16")
        return stream.readframes(stream.getnframes())


def prepare(args):
    output = checked_output(args.output_dir)
    require(not output.exists(), "质量准备目录已存在")
    corpus_manifest = args.corpus / "manifest.json"
    clips = read(corpus_manifest)["clips"]
    by_id = {row["id"]: row for row in clips}
    prior = {row["id"]: row for row in read(PRIOR / "manifest.json")}
    raw_by_id, originals = {}, {}
    for row in clips:
        path = Path(row["path"])
        require(sha(path) == row["sha256"], f"公开WAV哈希不匹配：{row['id']}")
        raw_by_id[row["id"]] = wav_samples(path)
        original = args.corpus / "originals" / path.name
        if row.get("source_sha256"):
            require(sha(original) == row["source_sha256"], "原始下载文件身份不匹配")
            require(wav_samples(original) == raw_by_id[row["id"]], "规范化WAV改变了实际采样")
            originals[row["id"]] = {"source_file": str(original), "sha256": sha(original)}
    mixed = by_id["long_mixed"]
    joined = bytes(round(mixed["gap_seconds"] * 16000) * 2).join(raw_by_id[name] for name in mixed["sequence"])
    require(joined == raw_by_id["long_mixed"], "长输入不是清单声明的逐字节拼接")
    expected_text = " ".join(by_id[name]["reference"] for name in mixed["sequence"])
    require(lexical(expected_text) == lexical(mixed["reference"]), "拼接逐字稿与组件顺序不一致")
    long_unscored = prior["file_60s"]
    require(sha(long_unscored["pcm"]) == long_unscored["pcm_sha256"], "60秒旧证据PCM改变")
    require(sha(long_unscored["wav"]) == long_unscored["wav_sha256"], "60秒旧证据WAV改变")
    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    manifest, references = [], []
    for row in clips:
        name = row["id"]
        integers = array.array("h", raw_by_id[name])
        if sys.byteorder != "little":
            integers.byteswap()
        values = array.array("f", (value / 32768.0 for value in integers))
        if sys.byteorder != "little":
            values.byteswap()
        target = inputs / (name + ".f32")
        with target.open("xb") as stream:
            stream.write(values.tobytes())
        if name in prior:
            require(sha(target) == prior[name]["pcm_sha256"], "新转换与已核验旧PCM不一致")
        wav = inputs / (name + ".wav")
        shutil.copyfile(row["path"], wav)
        manifest.append({"id": name, "pcm": str(target), "pcm_sha256": sha(target),
                         "sample_rate": 16000, "channels": 1, "encoding": "f32le",
                         "samples": len(integers), "duration_seconds": len(integers) / 16000,
                         "wav": str(wav), "wav_sha256": sha(wav)})
        references.append({"id": name, "pcm_sha256": sha(target), "ground_truth": row["reference"],
                           "language": row["language"], "reference_verified_by_source_wav_sha256": True,
                           "source_url": row.get("source_url"), "reference_url": row.get("reference_url"),
                           "original_download": originals.get(name), "composed_from": row.get("sequence"),
                           "reference_scope": "公开参考逐字稿；长拼接逐字节及逐字稿组件顺序已核验"})
    for suffix, field in ((".f32", "pcm"), (".wav", "wav")):
        shutil.copyfile(long_unscored[field], inputs / ("file_60s" + suffix))
    manifest.append({**long_unscored, "pcm": str(inputs / "file_60s.f32"), "wav": str(inputs / "file_60s.wav")})
    references.append({"id": "file_60s", "pcm_sha256": long_unscored["pcm_sha256"], "ground_truth": None,
                       "reason": "未配套人工逐字稿；仅输出一致性、结构及性能，不计算CER"})
    order = ("aishell_zh", "hello_en", "hello_zh", "song_cmd_zh", "conversation_zh", "meeting_zh", "long_mixed", "file_60s")
    workload = [{"phase": f"quality_seed_{seed}_{name}", "id": name,
                 "source": "file" if name == "file_60s" else "mic", "seed": seed}
                for seed in SEEDS for name in order]
    save(output / "manifest.json", manifest)
    save(output / "public-text-reference.json", references)
    save(output / "workload-multiseed.json", workload)
    cli_plan = [{"id": name, "seed": seed, "wav": str(inputs / (name + ".wav")),
                 "wav_sha256": sha(inputs / (name + ".wav"))}
                for seed in (42, 314159) for name in ("long_mixed", "file_60s")]
    save(output / "cli-plan.json", cli_plan)
    save(output / "preparation.json", {"complete": True, "models_loaded": 0, "inference_calls": 0,
         "formal_program_modified": False, "model": "Qwen3-ASR-1.7B", "frontend_batch": 4,
         "seeds": SEEDS, "requests_per_route": len(workload), "route_labels": LABELS,
         "corpus_manifest": str(corpus_manifest), "corpus_manifest_sha256": sha(corpus_manifest),
         "manifest_sha256": sha(output / "manifest.json"), "workload_sha256": sha(output / "workload-multiseed.json"),
         "scored_inputs": 7, "unscored_inputs": 1, "three_route_worker_requests": len(workload) * 3,
         "cli_checks_per_route": len(cli_plan), "normalization": "NFKC + casefold; remove Unicode punctuation and whitespace",
         "quality_rules": ["Exact output is a regression observation, not an accuracy verdict.",
                           "CER uses only human/public source references, never another ASR result.",
                           "Raw and formatted text are scored separately; tokens/timestamps are separate structure evidence.",
                           "Seeds are paired across routes; repeated/composed examples are not independent corpus evidence.",
                           "Seven public clips are a small smoke corpus; no domain-wide superiority claim."]})
    print(json.dumps({"prepared": True, "inputs": len(manifest), "scored_inputs": 7, "requests_per_route": len(workload), "models_loaded": 0}))


def named_paths(values, *, file_prefix=False):
    output = {}
    for value in values:
        name, separator, path = value.partition("=")
        require(separator and name and name not in output, "须使用不重复的标签=路径")
        output[name] = Path(path).resolve(strict=not file_prefix)
    return output


def compare_text(left, right, reference):
    a, b = cer(left, reference), cer(right, reference)
    changed = lexical(left) != lexical(right)
    return {"exact": left == right, "normalized_lexical_equal": not changed,
            "difference_class": "exact" if left == right else "lexical_changed" if changed else "format_only",
            "left_cer": a, "right_cer": b,
            "delta_edits": b["edits"] - a["edits"] if a and b else None,
            "delta_cer_percentage_points": (b["cer"] - a["cer"]) * 100 if a and b else None,
            "accuracy_verdict": ("unscored_no_ground_truth" if reference is None else
                                 "equal_CER" if a["edits"] == b["edits"] else
                                 "right_more_errors" if b["edits"] > a["edits"] else "right_fewer_errors")}


def timestamp_pairs(result):
    tokens, times = result.get("tokens", []), result.get("timestamps", [])
    require(len(tokens) == len(times), "token与timestamp数目不同")
    require(all(isinstance(token, str) for token in tokens), "token必须为文本")
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in times), "时间戳无效")
    return [(lexical(token), value) for token, value in zip(tokens, times) if lexical(token)]


def compare_timestamps(left, right):
    a, b = timestamp_pairs(left), timestamp_pairs(right)
    aligned = [value[0] for value in a] == [value[0] for value in b]
    deltas = [abs(x[1] - y[1]) for x, y in zip(a, b)] if aligned else []
    matcher = difflib.SequenceMatcher(None, [value[0] for value in a], [value[0] for value in b], autojunk=False)
    common, changes = [], []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            i, j = block.a + offset, block.b + offset
            delta = b[j][1] - a[i][1]
            common.append(abs(delta))
            if delta:
                changes.append({"normalized_token": a[i][0], "left_normalized_token_index": i,
                                "right_normalized_token_index": j, "left_time": a[i][1],
                                "right_time": b[j][1], "signed_delta_seconds": delta})
    common_result = {"method": "SequenceMatcher normalized lexical tokens, autojunk=False; pair only equal matching blocks, never zip different token arrays",
                     "matched_tokens": len(common), "changed_matched_timestamps": len(changes),
                     "max_matched_timestamp_delta": max(common, default=0.0), "timestamp_changes": changes,
                     "unmatched_changes": [{"operation": op, "left_range": [i, j], "right_range": [k, l],
                                             "left_tokens": [value[0] for value in a[i:j]],
                                             "right_tokens": [value[0] for value in b[k:l]]}
                                            for op, i, j, k, l in matcher.get_opcodes() if op != "equal"]}
    return {"tokens_exact": left.get("tokens") == right.get("tokens"),
            "timestamps_exact": left.get("timestamps") == right.get("timestamps"),
            "normalized_lexical_token_sequence_equal": aligned,
            "matched_lexical_tokens": len(a) if aligned else 0,
            "max_matched_token_timestamp_delta": max(deltas, default=0.0) if aligned else None,
            "changed_matched_timestamps": sum(value != 0 for value in deltas) if aligned else None,
            "common_subsequence_timestamps": common_result,
            "scope": "Only compare matching normalized lexical token sequences; no timestamp accuracy claim without timed human labels."}


def analyze(args):
    references = {row["id"]: row for row in read(args.references)}
    paths = named_paths(args.run)
    require(len(paths) >= 2 and args.baseline in paths, "至少两路线并包含指定基准")
    runs = {label: read(path) for label, path in paths.items()}
    for label, run in runs.items():
        require(run.get("execution_passed") and run.get("cleanup_confirmed"), "只分析执行成功并已清理的运行")
        require(run["settings"]["asr_model"] == "1.7b", "质量对照必须为同1.7B模型")
    for key in ("manifest_sha256", "workload_sha256"):
        require(len({run[key] for run in runs.values()}) == 1, "输入或工作顺序身份不同")
    route_policy = {"app_dir", "gpu_backend", "encoder_provider", "ggml_encoder_model",
                    "vulkan_pipeline_cache", "cuda_graph_optimization"}
    settings = [{key: value for key, value in run["settings"].items() if key not in route_policy}
                for run in runs.values()]
    require(all(value == settings[0] for value in settings), "模型以外的识别/格式/对齐参数改变，不能视为同条件质量对照")
    def key(row):
        return row["phase"], row["id"], row["source"], row["seed"], row["pcm_sha256"]
    indexed = {label: {key(row): row for row in run["samples"]} for label, run in runs.items()}
    require(all(len(indexed[label]) == len(runs[label]["samples"]) for label in runs), "请求身份重复")
    require(all(set(rows) == set(indexed[args.baseline]) for rows in indexed.values()), "三方请求不成对")
    scored, comparisons = {}, []
    for label, rows in indexed.items():
        output = []
        for identity, row in rows.items():
            reference = references[row["id"]]
            require(reference["pcm_sha256"] == row["pcm_sha256"], "人工逐字稿与PCM身份不一致")
            ground = reference.get("ground_truth")
            require(ground is None or reference.get("reference_verified_by_source_wav_sha256") is True,
                    "有逐字稿的输入必须有来源WAV身份核验")
            output.append({"phase": row["phase"], "id": row["id"], "seed": row["seed"], "source": row["source"],
                           "raw_cer": cer(row["metrics"]["text"], ground),
                           "formatted_cer": cer(row["result"]["text"], ground),
                           "raw_text_sha256": hashlib.sha256(row["metrics"]["text"].encode()).hexdigest()})
        scored[label] = output
    for left, right in itertools.combinations(runs, 2):
        rows = []
        for identity, a in indexed[left].items():
            b = indexed[right][identity]
            ground = references[a["id"]].get("ground_truth")
            rows.append({"phase": a["phase"], "id": a["id"], "seed": a["seed"],
                         "raw": compare_text(a["metrics"]["text"], b["metrics"]["text"], ground),
                         "formatted": compare_text(a["result"]["text"], b["result"]["text"], ground),
                         "alignment": compare_timestamps(a["result"], b["result"])})
        comparisons.append({"left": left, "right": right, "samples": rows})
    aggregates = {}
    for label, rows in scored.items():
        aggregates[label] = {}
        for seed in sorted({row["seed"] for row in rows}):
            selected = [row["raw_cer"] for row in rows if row["seed"] == seed and row["raw_cer"]]
            edits = sum(row["edits"] for row in selected)
            characters = sum(row["reference_characters"] for row in selected)
            aggregates[label][str(seed)] = {"scored_requests": len(selected), "edits": edits,
                "reference_characters": characters, "micro_cer": edits / characters if characters else None,
                "scope": "Includes repeated/composed examples; this is the fixed smoke workload, not an independent-corpus estimate."}
    save(args.output, {"complete": True, "analysis_only": True, "inference_calls": 0,
         "source_reports": {label: {"path": str(path), "sha256": sha(path)} for label, path in paths.items()},
         "reference_manifest_sha256": sha(args.references), "per_route": scored, "per_seed_aggregate": aggregates,
         "pairwise": comparisons, "normalization": "NFKC + casefold; remove Unicode punctuation and whitespace",
         "accuracy_policy": "Output differences are not automatic failures. CER compares each route to independent references; unlabelled cases remain unscored."})
    print(json.dumps({"complete": True, "routes": len(runs), "paired_requests": len(indexed[args.baseline]), "inference_calls": 0}))


def read_exports(prefix):
    paths = {suffix: Path(str(prefix) + suffix) for suffix in (".txt", ".json", ".srt")}
    require(all(path.is_file() for path in paths.values()), "TXT/JSON/SRT三种导出必须齐全")
    text, data = paths[".txt"].read_text("utf-8-sig"), read(paths[".json"])
    timestamp_pairs(data)
    srt = paths[".srt"].read_text("utf-8-sig").replace("\r\n", "\n")
    chunks = re.split(r"\n\s*\n", srt.strip())
    bodies, segments = [], []
    for index, block in enumerate(chunks, 1):
        lines = block.splitlines()
        require(len(lines) >= 3 and lines[0] == str(index), "SRT编号或结构无效")
        match = re.fullmatch(r"(\d{2,}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2,}):(\d{2}):(\d{2}),(\d{3})", lines[1])
        require(match is not None, "SRT时间格式无效")
        values = list(map(int, match.groups()))
        start, end = [3600 * values[k] + 60 * values[k + 1] + values[k + 2] + values[k + 3] / 1000 for k in (0, 4)]
        require(start <= end and all(values[k] < 60 for k in (1, 2, 5, 6)), "SRT时间范围无效")
        bodies.append("\n".join(lines[2:]))
        segments.append({"index": index, "start": start, "end": end, "text": bodies[-1], "lexical": lexical(bodies[-1])})
    return {"text": text, "json": data, "srt_text": srt, "srt_segments": segments,
            "structure": {"token_timestamp_count": len(data["tokens"]), "srt_segments": len(chunks),
                          "txt_matches_json_lexically": lexical(text) == lexical("".join(data["tokens"])),
                          "txt_matches_srt_lexically": lexical(text) == lexical("".join(bodies))},
            "files": {suffix: {"sha256": sha(path), "bytes": path.stat().st_size} for suffix, path in paths.items()}}


def compare_srt(left, right):
    a, b = left["srt_segments"], right["srt_segments"]
    def boundaries(rows):
        position, result = 0, []
        for row in rows:
            position += len(row["lexical"])
            result.append(position)
        return result
    same = "".join(row["lexical"] for row in a) == "".join(row["lexical"] for row in b)
    matcher = difflib.SequenceMatcher(None, [row["lexical"] for row in a], [row["lexical"] for row in b], autojunk=False)
    matched, changed = 0, []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            x, y = a[block.a + offset], b[block.b + offset]
            matched += 1
            if x["start"] != y["start"] or x["end"] != y["end"]:
                changed.append({"left_segment": x["index"], "right_segment": y["index"], "lexical": x["lexical"],
                                "left_times": [x["start"], x["end"]], "right_times": [y["start"], y["end"]]})
    return {"left_segments": len(a), "right_segments": len(b), "normalized_document_equal": same,
            "lexical_cut_positions_equal": boundaries(a) == boundaries(b) if same else None,
            "left_lexical_cut_positions": boundaries(a) if same else None,
            "right_lexical_cut_positions": boundaries(b) if same else None,
            "matched_segment_texts": matched, "changed_matched_segment_times": changed,
            "scope": "Match identical normalized segment texts in order; do not zip different segment arrays."}


def exports(args):
    references = {row["id"]: row for row in read(args.references)}
    require(args.id in references, "导出样本不在逐字稿清单")
    paths = named_paths(args.exports, file_prefix=True)
    values = {label: read_exports(prefix) for label, prefix in paths.items()}
    rows = []
    for left, right in itertools.combinations(values, 2):
        a, b = values[left], values[right]
        rows.append({"left": left, "right": right,
                     "export_file_hashes_exact": all(a["files"][suffix] == b["files"][suffix] for suffix in (".txt", ".json", ".srt")),
                     "text": compare_text(a["text"], b["text"], references[args.id].get("ground_truth")),
                     "alignment": compare_timestamps(a["json"], b["json"]), "srt_structure": compare_srt(a, b)})
    save(args.output, {"complete": True, "analysis_only": True, "inference_calls": 0, "id": args.id,
         "declared_seed": args.seed, "seed_provenance": "Declared for grouping; verify the separately retained CLI execution receipt.",
         "routes": {label: {"prefix": str(paths[label]), "files": value["files"], "structure": value["structure"]} for label, value in values.items()},
         "pairwise": rows, "format_policy": "Byte equality, normalized lexical text, CER and lexical-token timing are separate observations."})
    print(json.dumps({"complete": True, "routes": len(values), "id": args.id, "inference_calls": 0}))


def cli_run(args):
    checked_output(args.output)
    routes, plan = read(args.routes), read(args.plan)
    require(set(routes) == set(LABELS), "CLI计划须包含baseline/preview/pioneer三条路线")
    jobs = []
    for label, route in routes.items():
        app = Path(route["app_dir"]).resolve(strict=True)
        require(app.is_relative_to(STUDY.resolve()), "CLI只能启动本轮evidence中的隔离应用")
        config = app / "config/settings.json"
        settings = read(config)
        require(settings["asr_model"] == "1.7b", "CLI须使用同1.7B模型")
        exe = app / "语音输入.exe"
        require(exe.is_file(), "隔离EXE不存在")
        extra = read(route["env_json"]) if route.get("env_json") else {}
        require(isinstance(extra, dict) and all(isinstance(key, str) and isinstance(value, str) for key, value in extra.items()),
                "CLI环境配置须为字符串映射")
        expected_backend = "cuda" if label == "pioneer" else "vulkan"
        expected_encoder = {"baseline": "cpu", "preview": "ggml_vulkan", "pioneer": "ggml_cuda"}[label]
        actual_encoder = extra.get("YYUR_ENCODER_RESEARCH_PROVIDER", settings.get("encoder_provider", "cpu"))
        if actual_encoder == "ort":
            actual_encoder = "cpu"
        require(settings.get("gpu") is True and settings.get("gpu_backend") == expected_backend,
                "CLI解码后端与路线标签不同")
        require(actual_encoder == expected_encoder, "CLI编码后端与路线标签不同")
        require(extra.get("YYUR_GGML_ENCODER_FE_BATCH", "4") == "4", "质量比较固定前端batch4")
        require("YYUR_ASR_INIT" not in extra, "不得改变加载顺序")
        require("YYUR_ENCODER_EMBEDDINGS_DIR" not in extra, "CLI质量轮不混入embedding写盘")
        for sample in plan:
            require(re.fullmatch(r"[A-Za-z0-9_]+", sample["id"]) is not None
                    and type(sample["seed"]) is int and 0 <= sample["seed"] <= 0xFFFFFFFF,
                    "CLI计划ID或seed无效")
            require(sha(sample["wav"]) == sample["wav_sha256"], "CLI音频身份改变")
            folder = checked_output(args.run_dir / label / f"seed-{sample['seed']}" / sample["id"])
            command = [str(exe), "--app-dir", str(app), "--config", str(config), "--asr-model", "1.7b",
                       "--seed", str(sample["seed"]), "transcribe", sample["wav"], "--output-dir", str(folder)]
            jobs.append({"label": label, "sample": sample, "output_dir": str(folder), "argv": command,
                         "exe_sha256": sha(exe), "config_sha256": sha(config), "research_environment": extra})
    if not args.execute:
        print(json.dumps({"preflight_only": True, "jobs": len(jobs), "models_loaded": 0, "argv": [job["argv"] for job in jobs]}, ensure_ascii=True))
        return
    results = []
    for job in jobs:
        folder = Path(job["output_dir"])
        folder.mkdir(parents=True)
        env = {key: value for key, value in os.environ.items() if not key.startswith(("YYUR_", "GGML_"))}
        env.update(job["research_environment"])
        env["YYUR_LOG_DIR"] = str(folder / "logs")
        owned = OwnedTree()
        process, error, cleanup_ok = None, None, False
        result = {**job, "complete": False}
        try:
            with (folder / "stdout.txt").open("xb") as stdout, (folder / "stderr.txt").open("xb") as stderr:
                process = subprocess.Popen(job["argv"], env=env, stdout=stdout, stderr=stderr, creationflags=CREATE_NO_WINDOW)
                owned.assign(process)
                process.wait(timeout=args.timeout)
            require(process.returncode == 0, "CLI失败，请查看保留的独立日志")
            prefix = folder / Path(job["sample"]["wav"]).stem
            value = read_exports(prefix)
            result.update(complete=True, exit_code=0, export_prefix=str(prefix), exports=value["files"], structure=value["structure"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                owned.kill()
                deadline = time.monotonic() + 10
                while owned.ids() and time.monotonic() < deadline:
                    time.sleep(0.05)
                cleanup_ok = not owned.ids()
                if process is not None:
                    process.wait(timeout=10)
            finally:
                owned.close()
                result.update(cleanup_confirmed=cleanup_ok, error=error)
                save(folder / "invocation.json", result)
        require(cleanup_ok, "CLI进程树清理后仍有进程")
        require(error is None, error)
        results.append(result)
    save(args.output, {"complete": True, "formal_application_modified": False, "scope": "Actual isolated CLI quality checks; excluded from performance medians.", "runs": results})
    print(json.dumps({"complete": True, "cli_runs": len(results)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=STUDY / "quality/prepared")
    p.set_defaults(function=prepare)
    p = commands.add_parser("analyze")
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--baseline", default="baseline")
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(function=analyze)
    p = commands.add_parser("exports")
    p.add_argument("--exports", action="append", required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(function=exports)
    p = commands.add_parser("cli-run")
    p.add_argument("--routes", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--run-dir", type=Path, default=STUDY / "quality/cli")
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(function=cli_run)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
