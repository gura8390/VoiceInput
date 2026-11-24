use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Packaged model choices retain their original precision and file names.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum AsrModel {
    #[default]
    #[serde(rename = "0.6b")]
    #[value(name = "0.6b")]
    Qwen3Asr06B,
    #[serde(rename = "1.7b")]
    #[value(name = "1.7b")]
    Qwen3Asr17B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AsrModelSpec {
    pub relative_dir: &'static str,
    pub frontend: &'static str,
    pub backend: &'static str,
    pub decoder: &'static str,
}

impl AsrModel {
    pub fn spec(self) -> AsrModelSpec {
        match self {
            Self::Qwen3Asr06B => AsrModelSpec {
                relative_dir: "models/Qwen3-ASR/Qwen3-ASR-0.6B",
                frontend: "qwen3_asr_encoder_frontend.int4.onnx",
                backend: "qwen3_asr_encoder_backend.int4.onnx",
                decoder: "qwen3_asr_llm.q4_k.gguf",
            },
            Self::Qwen3Asr17B => AsrModelSpec {
                relative_dir: "models/Qwen3-ASR/Qwen3-ASR-1.7B",
                frontend: "qwen3_asr_encoder_frontend.onnx",
                backend: "qwen3_asr_encoder_backend.onnx",
                decoder: "qwen3_asr_llm.gguf",
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuBackend {
    Cuda,
    #[default]
    Vulkan,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncoderProvider {
    #[default]
    Cpu,
    CudaBackend,
    CudaAll,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Relative paths are resolved from the configuration file's directory.
    /// The packaged config/settings.json therefore uses ".." for the app root.
    pub app_dir: PathBuf,
    pub asr_model: AsrModel,
    pub onnx_dll: Option<PathBuf>,
    pub encoder_provider: EncoderProvider,
    /// Per-session CUDA EP arena limit; excludes libraries and other sessions.
    pub encoder_cuda_mem_limit_mb: usize,
    /// None preserves the shared intra-op setting; Some(0) requests ORT auto.
    pub encoder_frontend_threads: Option<usize>,
    pub encoder_frontend_arena: bool,
    pub encoder_backend_arena: bool,
    pub encoder_parallel_execution: bool,
    pub encoder_spin_wait: bool,
    pub encoder_dynamic_block_base: usize,
    pub language: String,
    pub asr_idle_seconds: u64,
    pub aligner_idle_seconds: u64,
    pub worker_timeout_seconds: u64,
    pub trim_after_load: bool,
    pub onnx_intra_threads: usize,
    pub onnx_inter_threads: usize,
    pub llm_threads: i32,
    pub llm_batch_threads: i32,
    pub onnx_memory_pattern: bool,
    pub onnx_arena_shrink: bool,
    pub onnx_device_initializers: bool,
    pub onnx_prepacking: bool,
    pub asr_dynamic_context: bool,
    pub asr_single_output: bool,
    pub aligner_cpu_mem_arena: bool,
    pub aligner_dynamic_context: bool,
    pub aligner_logits_limit: u32,
    pub file_context_handoff: bool,
    pub vulkan_pipeline_cache: bool,
    pub gpu: bool,
    pub gpu_backend: GpuBackend,
    /// Parallelize independent Q/K/V branches in the CUDA graph only.
    pub cuda_graph_optimization: bool,
    pub hotwords: bool,
    pub hotword_rules: bool,
    pub hot_threshold: f64,
    pub hot_similar_threshold: f64,
    pub format_num: bool,
    pub format_spell: bool,
    pub segment_seconds: usize,
    pub overlap_seconds: usize,
    /// Run one silent encoder+decoder pass after explicit desktop warmup. Default off.
    pub asr_warmup: bool,
    /// Drop no-speech clips and trim silence after hold-to-dictate. Default off.
    pub vad: bool,
    pub vad_threshold: f64,
    pub vad_min_speech_ms: u64,
    pub vad_speech_pad_ms: u64,
    /// Optional prefix such as `https://mirror.example`. Empty uses the catalog URL.
    pub model_mirror_base: String,
    pub model_download_retries: u32,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            app_dir: PathBuf::from("."),
            asr_model: AsrModel::default(),
            onnx_dll: None,
            encoder_provider: EncoderProvider::Cpu,
            encoder_cuda_mem_limit_mb: 512,
            encoder_frontend_threads: None,
            encoder_frontend_arena: false,
            encoder_backend_arena: false,
            encoder_parallel_execution: false,
            encoder_spin_wait: false,
            encoder_dynamic_block_base: 0,
            language: "auto".into(),
            asr_idle_seconds: 120,
            aligner_idle_seconds: 10,
            worker_timeout_seconds: 120,
            trim_after_load: true,
            onnx_intra_threads: 16,
            onnx_inter_threads: 1,
            llm_threads: 1,
            llm_batch_threads: 1,
            onnx_memory_pattern: true,
            onnx_arena_shrink: false,
            onnx_device_initializers: false,
            onnx_prepacking: true,
            asr_dynamic_context: false,
            asr_single_output: true,
            aligner_cpu_mem_arena: false,
            aligner_dynamic_context: true,
            aligner_logits_limit: 4000,
            file_context_handoff: true,
            vulkan_pipeline_cache: true,
            gpu: true,
            gpu_backend: GpuBackend::Vulkan,
            cuda_graph_optimization: false,
            hotwords: true,
            hotword_rules: true,
            hot_threshold: 0.85,
            hot_similar_threshold: 0.6,
            format_num: true,
            format_spell: true,
            segment_seconds: 60,
            overlap_seconds: 4,
            asr_warmup: false,
            vad: false,
            vad_threshold: 0.5,
            vad_min_speech_ms: 250,
            vad_speech_pad_ms: 30,
            model_mirror_base: String::new(),
            model_download_retries: 3,
        }
    }
}

pub(crate) fn validate_model_mirror(value: &str) -> Result<()> {
    let value = value.trim();
    if value.is_empty() {
        return Ok(());
    }
    ensure!(value.len() <= 200, "模型镜像地址过长");
    ensure!(
        value.starts_with("https://") || value.starts_with("http://"),
        "模型镜像只接受 http 或 https"
    );
    ensure!(
        !value.contains('@')
            && !value.contains('\\')
            && !value.contains(' ')
            && !value.contains(".."),
        "模型镜像地址无效"
    );
    let host = value.split_once("://").map(|(_, rest)| rest).unwrap_or("");
    ensure!(
        !host.is_empty() && !host.starts_with('/'),
        "模型镜像缺少主机名"
    );
    Ok(())
}

pub(crate) fn resolve_model_url(catalog_url: &str, mirror_base: &str) -> Result<String> {
    validate_model_mirror(mirror_base)?;
    let mirror = mirror_base.trim().trim_end_matches('/');
    if mirror.is_empty() {
        return Ok(catalog_url.to_owned());
    }
    Ok(format!("{mirror}/{catalog_url}"))
}
impl Settings {
    pub fn load(path: Option<&Path>, app_override: Option<&Path>) -> Result<Self> {
        let mut value = if let Some(path) = path {
            serde_json::from_slice::<Self>(
                &std::fs::read(path).with_context(|| format!("Read {}", path.display()))?,
            )?
        } else {
            Self::default()
        };
        if let Some(app) = app_override {
            value.app_dir = app.to_owned();
        } else if value.app_dir.is_relative() {
            let base = path.and_then(Path::parent).map(Path::to_owned).unwrap_or(
                std::env::current_exe()?
                    .parent()
                    .context("Executable directory")?
                    .to_owned(),
            );
            value.app_dir = base.join(&value.app_dir);
        }
        value.app_dir = value
            .app_dir
            .canonicalize()
            .with_context(|| format!("Application directory {}", value.app_dir.display()))?;
        ensure!(
            value.segment_seconds > 0 && value.segment_seconds + 2 * value.overlap_seconds <= 80,
            "Segments must fit the 80-second model window"
        );
        ensure!(
            (0.0..=1.0).contains(&value.hot_threshold),
            "Invalid hotword threshold"
        );
        ensure!(
            matches!(value.aligner_logits_limit, 0 | 4000),
            "Aligner logits limit must be 0 (full vocabulary) or 4000 (timestamps)"
        );
        ensure!(
            value.encoder_cuda_mem_limit_mb > 0
                && value
                    .encoder_cuda_mem_limit_mb
                    .checked_mul(1024 * 1024)
                    .is_some(),
            "Encoder CUDA arena limit must be a positive MiB value"
        );
        validate_encoder_cpu_tuning(
            value.encoder_frontend_threads,
            value.encoder_dynamic_block_base,
        )?;
        ensure!(
            (0.0..=1.0).contains(&value.vad_threshold),
            "VAD threshold must be between 0 and 1"
        );
        ensure!(
            value.vad_min_speech_ms <= 10_000,
            "VAD minimum speech must be at most 10000 ms"
        );
        ensure!(
            value.vad_speech_pad_ms <= 2_000,
            "VAD speech padding must be at most 2000 ms"
        );
        ensure!(
            value.model_download_retries <= 8,
            "Model download retries must be at most 8"
        );
        validate_model_mirror(&value.model_mirror_base)?;
        Ok(value)
    }
}

pub(crate) fn validate_encoder_cpu_tuning(
    frontend_threads: Option<usize>,
    dynamic_block_base: usize,
) -> Result<()> {
    ensure!(
        frontend_threads.is_none_or(|threads| threads <= 64),
        "Encoder frontend threads must be 0 (automatic) through 64"
    );
    ensure!(
        dynamic_block_base <= 64,
        "Encoder dynamic block base must be 0 (disabled) through 64"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_choice_defaults_to_small_and_survives_worker_serialization() {
        let old: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(old.asr_model, AsrModel::Qwen3Asr06B);
        for (name, expected) in [
            ("0.6b", AsrModel::Qwen3Asr06B),
            ("1.7b", AsrModel::Qwen3Asr17B),
        ] {
            let settings: Settings = serde_json::from_value(serde_json::json!({
                "asr_model": name, "encoder_frontend_arena": true, "onnx_intra_threads": 8
            }))
            .unwrap();
            let serialized = serde_json::to_string(&settings).unwrap();
            let worker: Settings = serde_json::from_str(&serialized).unwrap();
            assert_eq!(worker.asr_model, expected);
            assert!(worker.encoder_frontend_arena);
            assert_eq!(worker.onnx_intra_threads, 8);
            assert_eq!(serde_json::to_value(worker).unwrap()["asr_model"], name);
        }
        for invalid in ["1.7B", "large", "", "7b"] {
            assert!(
                serde_json::from_value::<Settings>(serde_json::json!({
                    "asr_model": invalid
                }))
                .is_err()
            );
        }
    }

    #[test]
    fn presets_keep_the_original_model_formats_distinct() {
        let small = AsrModel::Qwen3Asr06B.spec();
        let large = AsrModel::Qwen3Asr17B.spec();
        assert!(small.relative_dir.ends_with("Qwen3-ASR-0.6B"));
        assert_eq!(small.frontend, "qwen3_asr_encoder_frontend.int4.onnx");
        assert_eq!(small.backend, "qwen3_asr_encoder_backend.int4.onnx");
        assert_eq!(small.decoder, "qwen3_asr_llm.q4_k.gguf");
        assert!(large.relative_dir.ends_with("Qwen3-ASR-1.7B"));
        assert_eq!(large.frontend, "qwen3_asr_encoder_frontend.onnx");
        assert_eq!(large.backend, "qwen3_asr_encoder_backend.onnx");
        assert_eq!(large.decoder, "qwen3_asr_llm.gguf");
    }

    #[test]
    fn old_settings_keep_encoder_cpu_tuning_disabled() {
        let old: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(old.encoder_frontend_threads, None);
        assert!(!old.encoder_frontend_arena);
        assert!(!old.encoder_backend_arena);
        assert!(!old.encoder_parallel_execution);
        assert!(!old.encoder_spin_wait);
        assert_eq!(old.encoder_dynamic_block_base, 0);
        let configured: Settings = serde_json::from_str(
            r#"{"encoder_frontend_threads":4,"encoder_frontend_arena":true,"encoder_backend_arena":true,"encoder_parallel_execution":true,"encoder_spin_wait":true,"encoder_dynamic_block_base":4}"#,
        ).unwrap();
        let roundtrip: Settings =
            serde_json::from_value(serde_json::to_value(configured).unwrap()).unwrap();
        assert_eq!(roundtrip.encoder_frontend_threads, Some(4));
        assert!(roundtrip.encoder_frontend_arena);
        assert!(roundtrip.encoder_backend_arena);
        assert!(roundtrip.encoder_parallel_execution);
        assert!(roundtrip.encoder_spin_wait);
        assert_eq!(roundtrip.encoder_dynamic_block_base, 4);
    }

    #[test]
    fn encoder_cpu_tuning_accepts_auto_and_bounds_resources() {
        for threads in [None, Some(0), Some(1), Some(64)] {
            for block in [0, 1, 2, 3, 4, 8, 64] {
                assert!(validate_encoder_cpu_tuning(threads, block).is_ok());
            }
        }
        assert!(validate_encoder_cpu_tuning(Some(65), 0).is_err());
        assert!(validate_encoder_cpu_tuning(None, 65).is_err());
        assert!(validate_encoder_cpu_tuning(Some(usize::MAX), usize::MAX).is_err());
        assert!(serde_json::from_str::<Settings>(r#"{"encoder_frontend_threads":-1}"#).is_err());
    }

    #[test]
    fn encoder_provider_is_explicit_and_cpu_by_default() {
        let old: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(old.encoder_provider, EncoderProvider::Cpu);
        assert!(old.onnx_dll.is_none());
        assert_eq!(old.encoder_cuda_mem_limit_mb, 512);
        for (name, expected) in [
            ("cpu", EncoderProvider::Cpu),
            ("cuda_backend", EncoderProvider::CudaBackend),
            ("cuda_all", EncoderProvider::CudaAll),
        ] {
            let settings: Settings = serde_json::from_value(serde_json::json!({
                "encoder_provider": name,
                "onnx_dll": "runtime-ort-gpu/capi/onnxruntime.dll"
            }))
            .unwrap();
            assert_eq!(settings.encoder_provider, expected);
        }
        assert!(serde_json::from_str::<Settings>(r#"{"encoder_provider":"cuda"}"#).is_err());
    }

    #[test]
    fn old_settings_keep_warmup_and_vad_off() {
        let old: Settings = serde_json::from_str("{}").unwrap();
        assert!(!old.asr_warmup);
        assert!(!old.vad);
        assert_eq!(old.vad_threshold, 0.5);
        assert_eq!(old.vad_min_speech_ms, 250);
        assert_eq!(old.vad_speech_pad_ms, 30);
        assert!(old.model_mirror_base.is_empty());
        assert_eq!(old.model_download_retries, 3);
        let configured: Settings = serde_json::from_str(
            r#"{"asr_warmup":true,"vad":true,"vad_threshold":0.6,"vad_min_speech_ms":400,"vad_speech_pad_ms":60,"model_mirror_base":"https://mirror.example/github","model_download_retries":2}"#,
        )
        .unwrap();
        assert!(configured.asr_warmup);
        assert!(configured.vad);
        assert_eq!(configured.vad_threshold, 0.6);
        assert_eq!(configured.vad_min_speech_ms, 400);
        assert_eq!(configured.model_download_retries, 2);
        let roundtrip: Settings =
            serde_json::from_value(serde_json::to_value(&configured).unwrap()).unwrap();
        assert_eq!(roundtrip.model_mirror_base, configured.model_mirror_base);
        assert!(roundtrip.asr_warmup);
        assert!(resolve_model_url(
            "https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/Qwen3-ASR-0.6B-gguf.zip",
            &configured.model_mirror_base
        )
        .unwrap()
        .starts_with("https://mirror.example/github/https://github.com/"));
        assert!(
            resolve_model_url("https://example.invalid/model", "")
                .unwrap()
                .starts_with("https://")
        );
        let directory =
            std::env::temp_dir().join(format!("yyur-settings-{}-{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        for invalid in [
            r#"{"model_mirror_base":"ftp://mirror.example"}"#,
            r#"{"model_mirror_base":"https://user@mirror.example"}"#,
            r#"{"model_mirror_base":"http://"}"#,
            r#"{"model_mirror_base":"https://mirror.example/../secret"}"#,
            r#"{"vad_threshold":1.5}"#,
            r#"{"model_download_retries":9}"#,
            r#"{"vad_min_speech_ms":10001}"#,
        ] {
            let path = directory.join("settings.json");
            std::fs::write(&path, invalid).unwrap();
            assert!(Settings::load(Some(&path), None).is_err(), "{invalid}");
        }
        let path = directory.join("settings.json");
        std::fs::write(&path, "{}").unwrap();
        let loaded = Settings::load(Some(&path), None).unwrap();
        assert!(!loaded.asr_warmup && !loaded.vad);
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn asr_dynamic_context_is_opt_in_for_old_settings() {
        assert!(!Settings::default().asr_dynamic_context);
        let old: Settings = serde_json::from_str("{}").unwrap();
        assert!(!old.asr_dynamic_context);
        let enabled: Settings = serde_json::from_str(r#"{"asr_dynamic_context":true}"#).unwrap();
        assert!(enabled.asr_dynamic_context);
    }
}
