//! The installed application's directory layout. No legacy directory fallback.
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
pub struct AppPaths<'a> {
    root: &'a Path,
}

impl<'a> AppPaths<'a> {
    pub fn new(root: &'a Path) -> Self {
        Self { root }
    }

    pub fn config(self) -> PathBuf {
        self.root.join("config")
    }

    pub fn settings(self) -> PathBuf {
        self.config().join("settings.json")
    }

    pub fn hotwords(self) -> PathBuf {
        self.config().join("hot.txt")
    }

    pub fn hotword_rules(self) -> PathBuf {
        self.config().join("hot-rule.txt")
    }

    pub fn llama(self) -> PathBuf {
        self.root.join("runtime/llama")
    }

    pub fn onnx(self) -> PathBuf {
        self.root.join("runtime/onnx/onnxruntime.dll")
    }

    pub fn portaudio(self) -> PathBuf {
        self.root.join("runtime/audio/libportaudio64bit.dll")
    }

    pub fn ffmpeg(self) -> PathBuf {
        self.root.join("runtime/media/ffmpeg.exe")
    }

    pub fn logs(self) -> PathBuf {
        self.root.join("logs")
    }

    pub fn vulkan_cache(self) -> PathBuf {
        self.root.join("cache/vulkan")
    }

    pub fn aligner_model(self) -> PathBuf {
        self.root.join("models/Qwen3-ForcedAligner/Qwen3-ForcedAligner-0.6B")
    }
}
