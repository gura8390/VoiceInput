"""Stage only the encoder adapter in an existing detached research worktree.

The production checkout and installation are never modified. Run the normal
build/tests in the staged checkout. Select the native backend and model through
the isolated application's configuration; process-only research overrides remain
available for diagnostic runs.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ENCODER_SHA256 = 'da5f6436c653fb3b4ddce42dfa788637cc8600d969386a7fcc27e23f2817f5a9'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worktree', type=Path, required=True)
    args = parser.parse_args()
    worktree = args.worktree.resolve(strict=True)
    allowed = [(ROOT / '.build' / name).resolve() for name in ('ggml-vulkan-research','ggml-cuda-research')]
    if not any(worktree.is_relative_to(base) and worktree != base for base in allowed):
        parser.error('Candidate must be under an isolated GGML research build directory')
    detached = subprocess.run(['git', '-C', str(worktree), 'symbolic-ref', '-q', 'HEAD'],
                              capture_output=True, text=True)
    if detached.returncode != 1:
        parser.error('Candidate must be a detached worktree')
    encoder = worktree / 'src/engine/encoder.rs'
    original = encoder.read_bytes()
    if hashlib.sha256(original).hexdigest() != ENCODER_SHA256:
        parser.error('Encoder base identity mismatch; refusing to reapply or overwrite changes')
    source = original.decode('utf-8')
    if source.count('AudioEncoder') != 3:
        parser.error('Unexpected encoder type declarations')
    source = source.replace('AudioEncoder', 'OrtAudioEncoder')
    source += '\n' + (ROOT / 'tools/research/ggml_encoder_candidate.rs').read_text('utf-8')
    encoder.write_text(source, encoding='utf-8', newline='\n')
    def patch_once(path, old, new):
        source=path.read_text('utf-8')
        if source.count(old)!=1:
            parser.error(f'Unexpected base source at {path.name}')
        path.write_text(source.replace(old,new,1),encoding='utf-8',newline='\n')
    config=worktree/'src/config.rs'
    patch_once(config,'    CudaAll,\n}', '    CudaAll,\n    GgmlVulkan,\n    GgmlCuda,\n}')
    for path in (config,worktree/'src/engine.rs'):
        patch_once(path,'    pub encoder_provider: EncoderProvider,',
            '    pub encoder_provider: EncoderProvider,\n    pub ggml_encoder_model: Option<PathBuf>,')
        patch_once(path,'            encoder_provider: EncoderProvider::Cpu,',
            '            encoder_provider: EncoderProvider::Cpu,\n            ggml_encoder_model: None,')
    patch_once(config,'        validate_encoder_cpu_tuning(',
        '        if matches!(value.encoder_provider, EncoderProvider::GgmlVulkan | EncoderProvider::GgmlCuda) {\n'
        '            let expected = if value.encoder_provider == EncoderProvider::GgmlCuda { GpuBackend::Cuda } else { GpuBackend::Vulkan };\n'
        '            ensure!(value.gpu && value.gpu_backend == expected, "GGML 编解码器的 GPU 后端配置不一致");\n'
        '        }\n'
        '        if let Some(model) = value.ggml_encoder_model.as_ref() {\n'
        '            let resolved = if model.is_absolute() { model.clone() } else {\n'
        '                path.and_then(Path::parent).unwrap_or(&value.app_dir).join(model)\n'
        '            };\n'
        '            // Model existence belongs to the selected engine preflight, after CLI overrides.\n'
        '            value.ggml_encoder_model = Some(resolved);\n'
        '        }\n'
        '        validate_encoder_cpu_tuning(')
    patch_once(worktree/'src/pipeline.rs','        encoder_provider: settings.encoder_provider,',
        '        encoder_provider: settings.encoder_provider,\n'
        '        ggml_encoder_model: settings.ggml_encoder_model.clone(),\n'
        '        llama_dir: Some(crate::paths::AppPaths::new(&settings.app_dir).llama()),')
    patch_once(worktree/'src/engine.rs','        for name in [spec.frontend, spec.backend, spec.decoder] {',
        '        let uses_ggml = AudioEncoder::uses_ggml(spec.frontend, options)?;\n'
        '        let required = if uses_ggml { vec![spec.decoder] } else { vec![spec.frontend, spec.backend, spec.decoder] };\n'
        '        for name in required {')
    patch_once(worktree/'src/engine.rs','        require_encoder_sidecars(&dir, spec.frontend, spec.backend)?;',
        '        if uses_ggml {\n'
        '            let model = AudioEncoder::model_path(&dir, options);\n'
        '            let metadata = model.metadata().with_context(|| format!("读取 GGML 编码模型失败：{}", model.display()))?;\n'
        '            ensure!(metadata.is_file() && metadata.len() > 0, "GGML 编码模型为空或不是文件：{}", model.display());\n'
        '        } else { require_encoder_sidecars(&dir, spec.frontend, spec.backend)?; }')
    probe = worktree / 'examples/encoder_probe.rs'
    source = probe.read_text('utf-8').replace(
        'pub use yuybuuru::engine::{EncoderProvider, EngineOptions};',
        'pub use yuybuuru::engine::{EncoderProvider, EngineOptions, GpuBackend};\n'
        '#[path = "../src/engine/llama.rs"]\n#[allow(dead_code)]\nmod llama;')
    source = source.replace('    let args = Args::parse();',
        '    ensure!(!matches!(std::env::var("YYUR_ENCODER_RESEARCH_PROVIDER").as_deref(), Ok("ggml_vulkan"|"ggml_cuda")),\n'
        '        "This CPU-only probe cannot label a GGML/Vulkan run; use bench_ggml_vulkan_encoder.py");\n'
        '    let args = Args::parse();', 1)
    probe.write_text(source, encoding='utf-8', newline='\n')
    (worktree/'tests/ggml_route_config.rs').write_text(
        (ROOT/'tools/research/ggml_route_config_tests.rs').read_text('utf-8'),encoding='utf-8',newline='\n')
    print(f'Staged GGML encoder research adapter: {worktree}')


if __name__ == '__main__':
    main()
