//! Pinned model packages. Sizes and SHA256 were measured from the published files.
use crate::config::{AsrModel, Settings};
#[cfg(test)]
use crate::paths::AppPaths;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PackageKind {
    Zip,
    File,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelFile {
    /// Path inside the zip. For a single-file package this is the file name.
    pub zip_path: &'static str,
    pub install_name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    /// ONNX files are converted to external weights after extraction.
    pub externalize: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Package {
    pub id: &'static str,
    pub title: &'static str,
    pub url: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    pub install_dir: &'static str,
    pub kind: PackageKind,
    pub files: &'static [ModelFile],
}

const ASR_06: &[ModelFile] = &[
    ModelFile {
        zip_path: "qwen3_asr_encoder_frontend.int4.onnx",
        install_name: "qwen3_asr_encoder_frontend.int4.onnx",
        size: 20_343_991,
        sha256: "8c29d210b934f377983c0daa28d276f36cbc9a8a823d59d3afc5da2fd8eda2f6",
        externalize: true,
    },
    ModelFile {
        zip_path: "qwen3_asr_encoder_backend.int4.onnx",
        install_name: "qwen3_asr_encoder_backend.int4.onnx",
        size: 94_750_816,
        sha256: "95b8ed080ef99e28de62523ea0d70fb5b1383f17d4238464a265b3454eaaaa55",
        externalize: true,
    },
    ModelFile {
        zip_path: "qwen3_asr_llm.q4_k.gguf",
        install_name: "qwen3_asr_llm.q4_k.gguf",
        size: 484_215_360,
        sha256: "38a5492b88292dcb14c48fb939212da8223ec8d1e19b29d6e2751f26c0754cdd",
        externalize: false,
    },
];

const ASR_17: &[ModelFile] = &[
    ModelFile {
        zip_path: "Qwen3-ASR-1.7B/qwen3_asr_encoder_frontend.onnx",
        install_name: "qwen3_asr_encoder_frontend.onnx",
        size: 24_063_978,
        sha256: "fc288ba9a0ca732251b33628c803cd5188d3dd77a2cd7336f0f0fd37969ed4cd",
        externalize: true,
    },
    ModelFile {
        zip_path: "Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.onnx",
        install_name: "qwen3_asr_encoder_backend.onnx",
        size: 611_276_553,
        sha256: "4b0b1e34e8393f29222006aae0719283b0ff28b61f289b6ff8377b3ddbda27e3",
        externalize: true,
    },
    ModelFile {
        zip_path: "Qwen3-ASR-1.7B/qwen3_asr_llm.gguf",
        install_name: "qwen3_asr_llm.gguf",
        size: 1_471_800_896,
        sha256: "89e198470f41d487ab5b6b8a36a35400b3b1534451c4499236983050fe9193e5",
        externalize: false,
    },
];

const ALIGNER: &[ModelFile] = &[
    ModelFile {
        zip_path: "qwen3_aligner_encoder_frontend.int4.onnx",
        install_name: "qwen3_aligner_encoder_frontend.int4.onnx",
        size: 20_876_727,
        sha256: "858ea960c7feaf48a8787cb4a375e70216f99899cc229b174a6ebfdebd4f6e59",
        externalize: true,
    },
    ModelFile {
        zip_path: "qwen3_aligner_encoder_backend.int4.onnx",
        install_name: "qwen3_aligner_encoder_backend.int4.onnx",
        size: 164_176_179,
        sha256: "7c8931be70a64cd75be1671d714e9497d0bc2ef7943ffc9024807f71d24855b1",
        externalize: true,
    },
    ModelFile {
        zip_path: "qwen3_aligner_llm.q4_k.gguf",
        install_name: "qwen3_aligner_llm.q4_k.gguf",
        size: 484_399_552,
        sha256: "c2608b44866ac790fb0f9f542ae4a473c49fde03a7a2eb5b5370cc095b78dc8b",
        externalize: false,
    },
];

const SILERO: &[ModelFile] = &[ModelFile {
    zip_path: "silero_vad.onnx",
    install_name: "silero_vad.onnx",
    size: 2_204_103,
    sha256: "f6f4407b41e05e277e7f5cfd9b9ae957b1dfe763c6076a934734c0b448e09b31",
    externalize: false,
}];

/// 0.6B and the aligner are the INT4/Q4_K packages from HaujetZhao/Qwen3-ASR-GGUF.
/// 1.7B stays the original FP16 ONNX + Q5_K_M GGUF. That layout matches
/// CapsWriter-Offline `Qwen3-ASR-1.7B-q5_k.zip`, not the GGUF repo's INT4 zip.
const PACKAGES: &[Package] = &[
    Package {
        id: "asr-0.6b",
        title: "Qwen3-ASR 0.6B",
        url: "https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/Qwen3-ASR-0.6B-gguf.zip",
        size: 563_976_147,
        sha256: "3684b9cac9978a93c82ceeb93448bb5027c0b2beaf2e0affa35190ba20e545ff",
        install_dir: "models/Qwen3-ASR/Qwen3-ASR-0.6B",
        kind: PackageKind::Zip,
        files: ASR_06,
    },
    Package {
        id: "asr-1.7b",
        title: "Qwen3-ASR 1.7B",
        url: "https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/Qwen3-ASR-1.7B-q5_k.zip",
        size: 1_951_570_656,
        sha256: "f40040fe62a5ef0c09f8699fdbcb30f18bb8ae2bcd515ed4954e1f62b8b0e88f",
        install_dir: "models/Qwen3-ASR/Qwen3-ASR-1.7B",
        kind: PackageKind::Zip,
        files: ASR_17,
    },
    Package {
        id: "aligner-0.6b",
        title: "Qwen3-ForcedAligner 0.6B",
        url: "https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/Qwen3-ForceAligner-0.6B-gguf.zip",
        size: 504_734_961,
        sha256: "d9c16a2db3293c0c4120b0291d5d0fcfa208d842577092ca49cecc08a4690fb7",
        install_dir: "models/Qwen3-ForcedAligner/Qwen3-ForcedAligner-0.6B",
        kind: PackageKind::Zip,
        files: ALIGNER,
    },
    Package {
        id: "silero-vad",
        title: "Silero VAD",
        url: "https://raw.githubusercontent.com/snakers4/silero-vad/57053ec6ef95ab4704d380e3ff5b98c2f093586c/src/silero_vad/data/silero_vad.onnx",
        size: 2_204_103,
        sha256: "f6f4407b41e05e277e7f5cfd9b9ae957b1dfe763c6076a934734c0b448e09b31",
        install_dir: "models/silero",
        kind: PackageKind::File,
        files: SILERO,
    },
];

#[cfg(test)]
pub(crate) fn packages() -> &'static [Package] {
    PACKAGES
}

pub(crate) fn asr_package(model: AsrModel) -> &'static Package {
    match model {
        AsrModel::Qwen3Asr06B => &PACKAGES[0],
        AsrModel::Qwen3Asr17B => &PACKAGES[1],
    }
}

pub(crate) fn aligner_package() -> &'static Package {
    &PACKAGES[2]
}

pub(crate) fn vad_package() -> &'static Package {
    &PACKAGES[3]
}

/// Packages the running configuration needs. Silero is included only when VAD is on.
pub(crate) fn startup_missing(settings: &Settings) -> Vec<&'static Package> {
    let mut needed = vec![asr_package(settings.asr_model), aligner_package()];
    if settings.vad {
        needed.push(vad_package());
    }
    needed
        .into_iter()
        .filter(|package| !super::package_files_present(&settings.app_dir, package))
        .collect()
}

/// Download set for the tray action and `download-models`.
/// Silero is always included; it is about 2 MiB and skipped when already valid.
pub(crate) fn download_selection(settings: &Settings, all_asr: bool) -> Vec<&'static Package> {
    let mut selected = Vec::new();
    if all_asr {
        selected.push(asr_package(AsrModel::Qwen3Asr06B));
        selected.push(asr_package(AsrModel::Qwen3Asr17B));
    } else {
        selected.push(asr_package(settings.asr_model));
    }
    selected.push(aligner_package());
    selected.push(vad_package());
    selected
}

pub(crate) fn weights_name(onnx_name: &str) -> String {
    let path = std::path::Path::new(onnx_name);
    path.with_extension("weights.bin")
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| format!("{onnx_name}.weights.bin"))
}

#[cfg(test)]
fn install_layout_matches_runtime() -> bool {
    let small = AsrModel::Qwen3Asr06B.spec();
    let large = AsrModel::Qwen3Asr17B.spec();
    let aligner = AppPaths::new(std::path::Path::new("."))
        .aligner_model()
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>()
        .join("/");
    asr_package(AsrModel::Qwen3Asr06B).install_dir == small.relative_dir
        && file_named(ASR_06, small.frontend)
        && file_named(ASR_06, small.backend)
        && file_named(ASR_06, small.decoder)
        && asr_package(AsrModel::Qwen3Asr17B).install_dir == large.relative_dir
        && file_named(ASR_17, large.frontend)
        && file_named(ASR_17, large.backend)
        && file_named(ASR_17, large.decoder)
        && aligner_package().install_dir == aligner.trim_start_matches("./")
        && vad_package().install_dir == "models/silero"
}

#[cfg(test)]
fn file_named(files: &[ModelFile], name: &str) -> bool {
    files.iter().any(|file| file.install_name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_hashes_sizes_and_urls_are_pinned() {
        for package in packages() {
            assert!(package.url.starts_with("https://"), "{}", package.id);
            assert_eq!(package.sha256.len(), 64);
            assert!(package.size > 0);
            assert!(!package.files.is_empty());
            let mut file_bytes = 0u64;
            for file in package.files {
                assert_eq!(file.sha256.len(), 64, "{}", file.install_name);
                assert!(file.size > 0);
                assert!(!file.install_name.contains('/') && !file.install_name.contains('\\'));
                file_bytes += file.size;
                if package.kind == PackageKind::File {
                    assert_eq!(file.sha256, package.sha256);
                    assert_eq!(file.size, package.size);
                }
            }
            if package.kind == PackageKind::Zip {
                assert!(file_bytes > package.size, "{}", package.id);
            }
        }
        assert!(install_layout_matches_runtime());
        assert_eq!(
            ASR_17[0].sha256,
            "fc288ba9a0ca732251b33628c803cd5188d3dd77a2cd7336f0f0fd37969ed4cd"
        );
        assert_eq!(
            ASR_17[2].sha256,
            "89e198470f41d487ab5b6b8a36a35400b3b1534451c4499236983050fe9193e5"
        );
        assert!(
            asr_package(AsrModel::Qwen3Asr17B)
                .url
                .contains("Qwen3-ASR-1.7B-q5_k.zip")
        );
        assert!(
            !asr_package(AsrModel::Qwen3Asr17B)
                .url
                .contains("Qwen3-ASR-GGUF")
        );
        assert!(
            asr_package(AsrModel::Qwen3Asr06B)
                .url
                .contains("Qwen3-ASR-0.6B-gguf.zip")
        );
        assert!(
            aligner_package()
                .url
                .contains("Qwen3-ForceAligner-0.6B-gguf.zip")
        );
    }

    #[test]
    fn selection_follows_the_active_model_and_vad_switch() {
        let mut settings = Settings::default();
        settings.asr_model = AsrModel::Qwen3Asr17B;
        settings.vad = false;
        let missing = startup_missing(&settings);
        assert!(missing.iter().any(|package| package.id == "asr-1.7b"));
        assert!(missing.iter().any(|package| package.id == "aligner-0.6b"));
        assert!(missing.iter().all(|package| package.id != "silero-vad"));
        settings.vad = true;
        assert!(
            startup_missing(&settings)
                .iter()
                .any(|package| package.id == "silero-vad")
        );
        let selected = download_selection(&settings, false);
        assert!(selected.iter().any(|package| package.id == "asr-1.7b"));
        assert!(selected.iter().all(|package| package.id != "asr-0.6b"));
        assert!(
            download_selection(&settings, true)
                .iter()
                .any(|package| package.id == "asr-0.6b")
        );
    }
}
