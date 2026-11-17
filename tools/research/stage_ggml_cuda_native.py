"""Stage the pinned b9090 CUDA backend beside the installed decoder, without running a model."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

EXPECTED = {
    "llama.dll": "8378440b27a7a1074261037a6ba33865aee59ef77f0b66c9a00458c82a14073a",
    "ggml.dll": "31ab732d47e93c8a101a795889d2bc1084e963713aef1dacb7ea7da4989ae25a",
    "ggml-base.dll": "f7e5c323d9a52888f0690271f206cd80ae23f657f867f3558759d7f5dfff2c16",
    "caps_encoder.dll": "848fcb250bb06d41ff6a0f662aa180bdfea9810d33ea62fc383a2edb69cfa038",
    "ggml-cuda.dll": "680b659b0792c9f0699ca9a3c581ba871df0a8bb1acbfc59b8008536a281926c",
    "cublas64_12.dll": "9513540e4ec4c51ee9e7304138c2cc255c29a8c181f9e80c38efa25738becd99",
    "cublasLt64_12.dll": "b199d1ff892a81b7fd3d57ba1781549609b41500b36008fef326038393ad46c7",
    "cudart64_12.dll": "c2c9a9c22a9bcba90e261825968836787b331038047a26770cffb7a583c28344",
}
BLAS_SYMBOLS = {
    "cublasCreate_v2", "cublasDestroy_v2", "cublasGemmBatchedEx", "cublasGemmEx",
    "cublasGemmStridedBatchedEx", "cublasGetStatusString", "cublasSetMathMode",
    "cublasSetStream_v2", "cublasSgemmStridedBatched", "cublasSgemm_v2", "cublasStrsmBatched",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def import_sections(text: str) -> dict[str, dict[str, set[str]]]:
    # Same dumpbin table format used by the existing delay-cuBLAS build receipt.
    result: dict[str, dict[str, set[str]]] = {"normal": {}, "delay": {}}
    phase, current = "normal", None
    for line in text.splitlines():
        if line.strip() == "Summary":
            current = None
        if "delay load imports" in line.lower():
            phase, current = "delay", None
        module = re.fullmatch(r"\s{4}(\S+\.dll)\s*", line, re.I)
        if module:
            current = module[1].lower()
            result[phase][current] = set()
        pattern = r"\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)\s*" if phase == "delay" else r"\s+[0-9A-F]+\s+(\S+)\s*"
        symbol = re.fullmatch(pattern, line)
        if current and symbol:
            result[phase][current].add(symbol[1])
    return result


def exports(text: str) -> set[str]:
    return set(re.findall(r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)", text, re.M))


def probe(runtime: Path, output: Path) -> None:
    if os.name != "nt":
        raise RuntimeError("Windows is required")
    os.environ["PATH"] = str(runtime) + ";" + os.environ.get("PATH", "")
    with os.add_dll_directory(str(runtime)):
        loaded = {}
        for name in ("caps_encoder.dll", "ggml-cuda.dll", "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"):
            loaded[name] = ctypes.CDLL(str(runtime / name), winmode=8)
        encoder = loaded["caps_encoder.dll"]
        encoder.caps_ggml_encoder_abi_version.restype = ctypes.c_uint32
        encoder.caps_ggml_encoder_output_rows.argtypes = [ctypes.c_size_t]
        encoder.caps_ggml_encoder_output_rows.restype = ctypes.c_size_t
        if encoder.caps_ggml_encoder_abi_version() != 1:
            raise ValueError("Unexpected encoder ABI")
        rows = {str(n): int(encoder.caps_ggml_encoder_output_rows(n))
                for n in (0, 1, 8, 9, 99, 100, 101, 200, 800, 5260, 6001)}
        if any(value != int(key) // 100 * 13 + (int(key) % 100 + 7) // 8 for key, value in rows.items()):
            raise ValueError("Unexpected output row count")
        for symbol in BLAS_SYMBOLS:
            getattr(loaded["cublas64_12.dll"], symbol)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetModuleHandleW.argtypes, kernel.GetModuleHandleW.restype = [ctypes.c_wchar_p], ctypes.c_void_p
        kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
        kernel.GetModuleFileNameW.restype = ctypes.c_uint32
        paths = {}
        for name in (*EXPECTED, "nvcuda.dll"):
            handle = kernel.GetModuleHandleW(name)
            buffer = ctypes.create_unicode_buffer(32768)
            if not handle or not kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
                raise ctypes.WinError(ctypes.get_last_error())
            path = Path(buffer.value).resolve()
            if name != "nvcuda.dll" and path.parent != runtime:
                raise ValueError(f"Module escaped the isolated runtime: {name}: {path}")
            paths[name] = str(path)
        report = {
            "passed": True, "encoder_abi": 1, "output_rows": rows,
            "module_paths": paths, "cublas_symbols_resolved": sorted(BLAS_SYMBOLS),
            "models_loaded": 0, "backend_init_called": False,
            "cublas_create_or_gemm_called": False,
            "fallback_validation": "DLL dependencies and all 11 symbols resolve; numerical cuBLAS execution is left to the serial model benchmark",
        }
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-native", type=Path)
    parser.add_argument("--cuda-plugin", type=Path)
    parser.add_argument("--vendor-dir", type=Path)
    parser.add_argument("--encoder-plugin", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--dumpbin", type=Path)
    parser.add_argument("--child-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    runtime, evidence = args.output_dir.resolve(), args.evidence_dir.resolve()
    if args.child_probe:
        probe(runtime, evidence / "loader-abi-cublas-probe.json")
        return
    if not all((args.installed_native, args.cuda_plugin, args.vendor_dir, args.encoder_plugin, args.dumpbin)):
        parser.error("All source directories/plugins and --dumpbin are required")
    if runtime.exists():
        raise ValueError("Refusing to replace an existing runtime")
    installed = args.installed_native.resolve(strict=True)
    encoder = args.encoder_plugin.resolve(strict=True)
    cuda = args.cuda_plugin.resolve(strict=True)
    vendor = args.vendor_dir.resolve(strict=True)
    reference_core = {}
    for name in ("ggml-base.dll", "ggml.dll"):
        reference = vendor / name
        if reference.is_file():
            reference_core[name] = digest(reference)
            if reference_core[name] != EXPECTED[name]:
                raise ValueError(f"Archived CUDA reference core differs: {name}")
    sources = {p.name: p for p in installed.glob("*.dll") if p.name.lower() != "ggml-vulkan.dll"}
    sources.update({"caps_encoder.dll": encoder, "ggml-cuda.dll": cuda})
    sources.update({name: vendor / name for name in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")})
    before = {name: digest(path) for name, path in sources.items()}
    for name, expected in EXPECTED.items():
        if before.get(name) != expected:
            raise ValueError(f"Pinned input differs: {name}: {before.get(name)}")
    if installed == runtime or installed in runtime.parents or runtime in installed.parents:
        raise ValueError("Runtime must be isolated from the installed native directory")
    runtime.mkdir(parents=True)
    evidence.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        shutil.copy2(source, runtime / name)
    after = {p.name: digest(p) for p in runtime.glob("*.dll")}
    if before != after:
        raise ValueError("Staged file hashes differ")
    dumpbin = args.dumpbin.resolve(strict=True)
    cache = {}
    def dump(option: str, name: str) -> str:
        key = option, name
        if key not in cache:
            cache[key] = subprocess.check_output([str(dumpbin), "/nologo", option, str(runtime / name)],
                creationflags=0x08000000).decode("utf-8", "replace").replace("\r\n", "\n")
            (evidence / (name + "-" + option.strip("/") + ".txt")).write_text(cache[key], "utf-8")
        return cache[key]
    plugin_imports = import_sections(dump("/imports", "ggml-cuda.dll"))
    if set(plugin_imports["delay"]) != {"cublas64_12.dll"} or plugin_imports["delay"]["cublas64_12.dll"] != BLAS_SYMBOLS:
        raise ValueError("Delayed cuBLAS import table differs")
    if "cublas64_12.dll" in plugin_imports["normal"]:
        raise ValueError("cuBLAS was eagerly imported")
    checked = []
    for consumer in ("ggml-cuda.dll", "caps_encoder.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"):
        tables = import_sections(dump("/imports", consumer))
        for phase, modules in tables.items():
            for module, symbols in modules.items():
                actual = next((name for name in sources if name.lower() == module), None)
                if actual:
                    missing = symbols - exports(dump("/exports", actual))
                    if missing:
                        raise ValueError(f"Unsatisfied {consumer} -> {actual}: {sorted(missing)}")
                    checked.append({"consumer": consumer, "provider": actual, "phase": phase, "symbols": len(symbols)})
    command = [sys.executable, str(Path(__file__).resolve()), "--output-dir", str(runtime),
               "--evidence-dir", str(evidence), "--child-probe"]
    subprocess.run(command, check=True, creationflags=0x08000000,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    receipt = {
        "passed": True, "runtime": str(runtime), "source_paths": {name: str(p) for name, p in sources.items()},
        "sha256": after, "logical_bytes": sum((runtime / name).stat().st_size for name in after),
        "local_import_edges_checked": checked,
        "decoder_and_ggml_core_byte_identical": True,
        "archived_cuda_reference_core_sha256": reference_core,
        "cuda_backend": "b9090 / SM120a, original 136 CUDA objects, delayed cuBLAS relink",
        "encoder": "same fixed FE batch4 mixed-F16/F32-weight GGML implementation as the Vulkan preview",
        "omitted": ["ggml-vulkan.dll", "ORT CUDA", "cuDNN", "cuFFT"],
        "models_loaded": 0, "backend_init_called": False, "production_deployed": False,
    }
    (evidence / "native-stage-receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps({key: receipt[key] for key in ("passed", "runtime", "logical_bytes", "decoder_and_ggml_core_byte_identical", "models_loaded")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
