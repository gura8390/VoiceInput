"""Check an isolated Windows encoder DLL and its matching dependencies, without loading a model."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("This ABI probe requires Windows")
    runtime = args.runtime_dir.resolve(strict=True)
    with os.add_dll_directory(str(runtime)):
        library = ctypes.CDLL(str(runtime / "caps_encoder.dll"))
        library.caps_ggml_encoder_abi_version.restype = ctypes.c_uint32
        library.caps_ggml_encoder_output_rows.argtypes = [ctypes.c_size_t]
        library.caps_ggml_encoder_output_rows.restype = ctypes.c_size_t
        abi = int(library.caps_ggml_encoder_abi_version())
        if abi != 1:
            raise ValueError(f"Unexpected ABI: {abi}")
        rows = {}
        for frames in (0, 1, 8, 9, 99, 100, 101, 200, 800, 5260, 6001):
            actual = int(library.caps_ggml_encoder_output_rows(frames))
            expected = frames // 100 * 13 + (frames % 100 + 7) // 8
            if actual != expected:
                raise ValueError(f"Row count differs: {frames}: {actual} != {expected}")
            rows[str(frames)] = actual
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel.GetModuleHandleW.restype = ctypes.c_void_p
        kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
        kernel.GetModuleFileNameW.restype = ctypes.c_uint32
        module_paths = {}
        for name in ("caps_encoder.dll", "llama.dll", "ggml.dll", "ggml-base.dll"):
            handle = kernel.GetModuleHandleW(name)
            buffer = ctypes.create_unicode_buffer(32768)
            if not handle or not kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
                raise ctypes.WinError(ctypes.get_last_error())
            path = Path(buffer.value).resolve()
            if path.parent != runtime:
                raise ValueError(f"Dependency escaped isolated runtime: {name}: {path}")
            module_paths[name] = str(path)
        result = {
            "abi_version": abi,
            "output_rows": rows,
            "loaded_modules": module_paths,
            "dll_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(runtime.glob("*.dll"))},
            "model_loaded": False,
            "inference_executed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", "utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "dll_sha256"},
                         indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
