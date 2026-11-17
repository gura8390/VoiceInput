"""在隔离目录测量原样重建的 0.4.3-CUDA 分支；默认只做预检。"""
from pathlib import Path
import argparse
import json
from types import SimpleNamespace

import bench_ggml_vulkan_encoder as shared
from bench_ggml_three_routes import background, job_cpu

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'evidence/ggml-cuda-20261117'
COMMIT = '3b7c123df6fe8cb8e0c220f65258652297419fa5'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', type=Path, required=True)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--expected-exe-sha256', required=True)
    parser.add_argument('--stage-receipt', type=Path, default=STUDY / 'previous-cuda/historical-stage.json')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=STUDY / 'prepared-final')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--quality', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    app, exe, config = (p.resolve(strict=True) for p in (args.app,args.exe,args.config))
    output = args.output.resolve()
    allowed = (STUDY / 'previous-cuda').resolve()
    shared.require(app.is_relative_to(allowed) and app != allowed, '应用须位于历史 CUDA 隔离目录')
    shared.require(exe.is_relative_to(app) and config.is_relative_to(app), 'EXE 和配置须属于指定应用')
    shared.require(shared.sha(exe) == args.expected_exe_sha256.lower(), 'EXE 与原分支构建收据不符')
    shared.require(output.is_relative_to(allowed) and not output.exists(), '输出须为新的历史 CUDA 证据文件')
    settings = shared.read(config)
    shared.require(settings.get('encoder_provider') == 'cpu' and settings.get('gpu_backend') == 'cuda', '原分支应为 CPU ORT 编码和 CUDA 解码')
    shared.require(settings.get('asr_model') == '0.6b', '原分支固定 0.6B；该审计字段被原程序忽略')
    native = app / 'core/server/engines/llama/bin'
    shared.require((app/'runtime/llama').resolve() == native.resolve(), '采集器目录别名须指向同一旧原生库')
    receipt_path=args.stage_receipt.resolve(strict=True)
    receipt=shared.read(receipt_path)
    native_hashes={p.name:shared.sha(p) for p in native.glob('*.dll')}
    shared.require(receipt.get('prepared') and receipt['source_commit'] == COMMIT
        and receipt.get('source_modified') is False, '收据须绑定未改动的原分支源码')
    shared.require(Path(receipt['app']).resolve() == app and Path(receipt['exe']).resolve() == exe
        and Path(receipt['config']).resolve() == config, '收据应用目录不匹配')
    shared.require(receipt['exe_sha256'] == args.expected_exe_sha256.lower()
        and receipt['config_sha256'] == shared.sha(config), '程序或配置偏离原分支准备收据')
    shared.require(native_hashes == {name:row['sha256'] for name,row in receipt['native_files'].items()},
        '原生 DLL 偏离历史准备收据')
    ort=app/'internal/onnxruntime/capi'
    shared.require(ort.resolve() == Path(receipt['ort']['source']).resolve()
        and (app/'runtime/onnx').resolve() == ort.resolve()
        and shared.sha(ort/'onnxruntime.dll') == receipt['ort']['sha256'], '旧 ORT 身份不匹配')
    for relative,record in receipt['model_files'].items():
        model=app/'models'/relative
        shared.require(model.resolve(strict=True) == (Path(receipt['models_origin'])/relative).resolve(strict=True),
            '旧模型文件目标不匹配')
        shared.require(model.stat().st_size == record['bytes'], '旧模型文件大小变化')
    identity = {'variant':'historical_original', 'source_commit':COMMIT,
        'label':'GitHub 原分支源码重建，0.6B，原 CUDA 原生库',
        'model':'0.6b', 'cross_model_comparison':True,
        'exe_sha256':shared.sha(exe), 'settings_sha256':shared.sha(config),
        'native_dll_sha256':native_hashes, 'stage_receipt_sha256':shared.sha(receipt_path),
        'layout':'旧程序目录；runtime 别名和 asr_model 字段仅供统一采集器审计',
        'model_storage':'归档原内嵌 ONNX 与原 Q4_K GGUF；原 ORT 1.24 运行库'}
    inputs = args.inputs.resolve(strict=True)
    options = SimpleNamespace(exe=exe,app_dir=app,config=config,
        manifest=inputs/'manifest.json',
        workload=inputs/('workload-multiseed.json' if args.quality else 'workload.json'),
        env_json=None,reference=args.reference,public_reference=inputs/'public-text-reference.json',
        match='compare',profile=False,preflight_only=not args.execute,output=output,
        host_ms=25.0,gpu_ms=100.0,timeout=180.0,expected_decoder_backend='cuda',
        strict_core_nodes=True,job_cpu_reader=job_cpu,artifact_identity=identity,quality_only=args.quality)
    shared.STUDY = STUDY
    formal = Path(shared.read(STUDY/'prepared-final/preparation.json')['formal_app'])
    before = background(formal) if args.execute else None
    try:
        shared.run(options)
    finally:
        if args.execute:
            after=background(formal)
            with output.with_suffix('.background.json').open('x',encoding='utf8') as stream:
                json.dump({'before':before,'after':after,'formal_application_modified':False},stream,ensure_ascii=False,indent=2)


if __name__ == '__main__':
    main()
