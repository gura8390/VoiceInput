//! Model catalog, verified download and external-weight preparation.
mod catalog;
mod external;
mod fetch;
mod install;
mod vad;

pub use vad::{Decision, Silero, SpeechSpan, VadSession, trim_speech};

use crate::config::Settings;
use std::sync::atomic::AtomicBool;

pub fn models_missing(settings: &Settings) -> bool {
    !catalog::startup_missing(settings).is_empty()
}

pub fn install_selected(
    settings: &Settings,
    all_asr: bool,
    cancel: &AtomicBool,
    progress: &dyn Fn(&str),
) -> Result<String> {
    let packages = catalog::download_selection(settings, all_asr);
    install::install_packages(&settings.app_dir, settings, &packages, cancel, progress)
}

pub(crate) use fetch::download_verified;
pub(crate) use install::package_files_present;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("读取 {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0xf) as usize] as char);
    }
    out
}

pub(crate) fn parse_sha256(value: &str) -> Result<String> {
    let value = value.trim().to_ascii_lowercase();
    anyhow::ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "SHA256 必须是 64 位十六进制"
    );
    Ok(value)
}
