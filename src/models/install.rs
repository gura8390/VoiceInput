//! Download, verify, extract and externalize the pinned model packages.
use super::{
    catalog::{self, Package, PackageKind, weights_name},
    download_verified,
    external::{PrepareOptions, onnx_external_sidecar_ready, prepare},
    sha256_file,
};
use crate::config::{Settings, resolve_model_url};
use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

static PREPARE_SEQ: AtomicU64 = AtomicU64::new(1);

pub(crate) fn install_packages(
    app: &Path,
    settings: &Settings,
    packages: &[&Package],
    cancel: &AtomicBool,
    progress: &dyn Fn(&str),
) -> Result<String> {
    let mut lines = Vec::new();
    for package in packages {
        if cancel.load(Ordering::Acquire) {
            bail!("已取消下载");
        }
        progress(&format!("正在准备 {}", package.title));
        ensure_package(app, settings, package, cancel, progress)?;
        lines.push(format!("{} 已就绪", package.title));
    }
    if lines.is_empty() {
        Ok("没有需要下载的模型".to_owned())
    } else {
        Ok(lines.join("\n"))
    }
}

pub(crate) fn package_files_present(app: &Path, package: &Package) -> bool {
    package.files.iter().all(|file| {
        let path = app.join(package.install_dir).join(file.install_name);
        let present = fs::metadata(&path).is_ok_and(|meta| meta.is_file() && meta.len() > 0);
        if !present {
            return false;
        }
        if file.externalize {
            let weights = path.with_extension("weights.bin");
            return fs::metadata(&weights).is_ok_and(|meta| meta.is_file() && meta.len() > 0);
        }
        true
    })
}

fn ensure_package(
    app: &Path,
    settings: &Settings,
    package: &Package,
    cancel: &AtomicBool,
    progress: &dyn Fn(&str),
) -> Result<()> {
    if package
        .files
        .iter()
        .all(|file| match file_ready(app, package, file) {
            Ok(ready) => ready,
            Err(_) => false,
        })
    {
        progress(&format!("{} 已校验通过，跳过", package.title));
        return Ok(());
    }
    let archive = if package.kind == PackageKind::Zip && needs_archive(app, package)? {
        Some(cached_archive(app, settings, package, cancel, progress)?)
    } else {
        None
    };
    for file in package.files {
        if cancel.load(Ordering::Acquire) {
            bail!("已取消下载");
        }
        let dest_dir = app.join(package.install_dir);
        fs::create_dir_all(&dest_dir)?;
        let dest = dest_dir.join(file.install_name);
        if file.externalize {
            ensure_onnx(
                app,
                package,
                file,
                archive.as_deref(),
                &dest,
                cancel,
                progress,
            )?;
        } else if package.kind == PackageKind::File {
            let url = resolve_model_url(package.url, &settings.model_mirror_base)?;
            progress(&format!("正在下载 {}", package.title));
            download_verified(
                &url,
                &dest,
                file.size,
                file.sha256,
                settings.model_download_retries,
                cancel,
                progress,
            )?;
        } else if file_is_valid(&dest, file.size, file.sha256)? {
            progress(&format!("{} 校验通过，跳过", file.install_name));
        } else {
            let archive_path = archive.as_deref().context("缺少压缩包")?;
            let source = extract_member(archive_path, file, &dest_dir, cancel, progress)?;
            replace_file(&source, &dest)?;
        }
    }
    // Remove temporary zip archives and staging files once every target file is verified in place.
    if package.kind == PackageKind::Zip {
        if let Some(archive_path) = archive.as_deref() {
            let _ = fs::remove_file(archive_path);
        }
        let staging_dir = app.join("cache/downloads/staging").join(package.id);
        if staging_dir.exists() {
            let _ = fs::remove_dir_all(&staging_dir);
        }
        let prepare_dir = app.join("cache/prepare").join(package.id);
        if prepare_dir.exists() {
            let _ = fs::remove_dir(&prepare_dir);
        }
    }
    Ok(())
}

fn needs_archive(app: &Path, package: &Package) -> Result<bool> {
    for file in package.files {
        if file_ready(app, package, file)? {
            continue;
        }
        let dest = app.join(package.install_dir).join(file.install_name);
        // A verified inline ONNX can be externalized without downloading again.
        if file.externalize && file_is_valid(&dest, file.size, file.sha256)? {
            continue;
        }
        return Ok(true);
    }
    Ok(false)
}

fn ensure_onnx(
    app: &Path,
    package: &Package,
    file: &catalog::ModelFile,
    archive: Option<&Path>,
    dest: &Path,
    cancel: &AtomicBool,
    progress: &dyn Fn(&str),
) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        bail!("已取消下载");
    }
    let weights_name = weights_name(file.install_name);
    let weights = dest.with_file_name(&weights_name);
    if sidecar_is_valid(dest, &weights, &weights_name, file.sha256)? {
        progress(&format!("{} 的外置权重已就绪，跳过", file.install_name));
        return Ok(());
    }
    let source = if file_is_valid(dest, file.size, file.sha256)? {
        progress(&format!(
            "{} 已是内嵌权重，正在生成外置文件",
            file.install_name
        ));
        dest.to_path_buf()
    } else {
        let archive = archive.context("缺少压缩包")?;
        extract_member(
            archive,
            file,
            &app.join("cache/downloads/staging").join(package.id),
            cancel,
            progress,
        )?
    };
    let temp = app.join("cache/prepare").join(package.id).join(format!(
        "{}-{}",
        std::process::id(),
        PREPARE_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let prepared = temp.join(file.install_name);
    let options = PrepareOptions {
        minimum: 4096,
        alignment: 4096,
        expected_source_sha256: Some(file.sha256.to_owned()),
    };
    let prepared_result = prepare(&source, &prepared, &temp, &options);
    if let Err(error) = prepared_result {
        let _ = fs::remove_dir_all(&temp);
        return Err(error);
    }
    if let Err(error) = (|| -> Result<()> {
        ensure!(
            onnx_external_sidecar_ready(&prepared, &weights_name)?,
            "外置权重生成后未能通过结构检查"
        );
        let prepared_weights = prepared.with_extension("weights.bin");
        let prepared_receipt = prepared.with_extension("external.json");
        let weights_backup = weights.with_file_name(format!("{weights_name}.rollback"));
        let had_old_weights = weights.is_file();
        if had_old_weights {
            let _ = fs::remove_file(&weights_backup);
            fs::copy(&weights, &weights_backup)?;
        }
        replace_file(&prepared_weights, &weights)?;
        if let Err(error) = replace_file(&prepared, dest) {
            if had_old_weights {
                let _ = replace_file(&weights_backup, &weights);
            } else {
                let _ = fs::remove_file(&weights);
            }
            return Err(error);
        }
        let _ = fs::remove_file(&weights_backup);
        replace_file(&prepared_receipt, &dest.with_extension("external.json"))?;
        Ok(())
    })() {
        let _ = fs::remove_dir_all(&temp);
        return Err(error);
    }
    let _ = fs::remove_dir_all(&temp);
    progress(&format!("已生成 {}", weights_name));
    Ok(())
}

fn sidecar_is_valid(
    onnx: &Path,
    weights: &Path,
    weights_name: &str,
    source_sha: &str,
) -> Result<bool> {
    if receipt_matches(onnx, weights, source_sha)? {
        return Ok(true);
    }
    if !onnx_external_sidecar_ready(onnx, weights_name)? {
        return Ok(false);
    }
    // An externalized file no longer matches the inline source hash.
    Ok(!file_is_valid(onnx, fs::metadata(onnx)?.len(), source_sha)?)
}

fn receipt_matches(onnx: &Path, weights: &Path, source_sha: &str) -> Result<bool> {
    let receipt_path = onnx.with_extension("external.json");
    let text = match fs::read_to_string(&receipt_path) {
        Ok(text) => text,
        Err(_) => return Ok(false),
    };
    let value: serde_json::Value = match serde_json::from_str(&text) {
        Ok(value) => value,
        Err(_) => return Ok(false),
    };
    if value.get("source_sha256").and_then(|item| item.as_str()) != Some(source_sha)
        || value
            .get("payloads_byte_identical")
            .and_then(|item| item.as_bool())
            != Some(true)
    {
        return Ok(false);
    }
    let Some(candidate) = value.get("candidate_sha256").and_then(|item| item.as_str()) else {
        return Ok(false);
    };
    let Some(external) = value
        .get("external_weights_sha256")
        .and_then(|item| item.as_str())
    else {
        return Ok(false);
    };
    Ok(sha256_file(onnx).ok().as_deref() == Some(candidate)
        && sha256_file(weights).ok().as_deref() == Some(external))
}

fn cached_archive(
    app: &Path,
    settings: &Settings,
    package: &Package,
    cancel: &AtomicBool,
    progress: &dyn Fn(&str),
) -> Result<PathBuf> {
    let name = package.url.rsplit('/').next().context("压缩包文件名")?;
    let destination = crate::paths::AppPaths::new(app).downloads().join(name);
    let url = resolve_model_url(package.url, &settings.model_mirror_base)?;
    progress(&format!("正在下载 {}", package.title));
    download_verified(
        &url,
        &destination,
        package.size,
        package.sha256,
        settings.model_download_retries,
        cancel,
        progress,
    )?;
    Ok(destination)
}

fn extract_member(
    archive_path: &Path,
    file: &catalog::ModelFile,
    directory: &Path,
    cancel: &AtomicBool,
    progress: &dyn Fn(&str),
) -> Result<PathBuf> {
    fs::create_dir_all(directory)?;
    let destination = directory.join(file.install_name);
    if file_is_valid(&destination, file.size, file.sha256)? {
        progress(&format!("{} 已解压且校验通过，跳过", file.install_name));
        return Ok(destination);
    }
    progress(&format!("正在解压 {}", file.install_name));
    let mut archive = zip::ZipArchive::new(File::open(archive_path)?)?;
    let mut entry = archive
        .by_name(file.zip_path)
        .with_context(|| format!("压缩包缺少 {}", file.zip_path))?;
    ensure!(
        entry.size() == file.size,
        "{} 的解压大小与目录不一致",
        file.install_name
    );
    let temp = directory.join(format!("{}.extracting", file.install_name));
    let extracted = (|| -> Result<String> {
        let mut output = File::create(&temp)?;
        let mut hasher = Sha256::new();
        // Heap-allocate the 1 MiB copy buffer to avoid overflowing the 1 MiB Windows main thread stack.
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut read_bytes = 0u64;
        loop {
            if cancel.load(Ordering::Acquire) {
                bail!("已取消下载");
            }
            let n = entry.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            read_bytes += n as u64;
            ensure!(read_bytes <= file.size, "解压超出目录中的大小");
            sha2::Digest::update(&mut hasher, &buffer[..n]);
            output.write_all(&buffer[..n])?;
        }
        output.flush()?;
        drop(output);
        ensure!(read_bytes == file.size, "解压结果不完整");
        Ok(super::hex(&hasher.finalize()))
    })();
    let actual = match extracted {
        Ok(hash) => hash,
        Err(error) => {
            let _ = fs::remove_file(&temp);
            return Err(error);
        }
    };
    if actual != file.sha256 {
        let _ = fs::remove_file(&temp);
        bail!("{} 解压后的 SHA256 不一致", file.install_name);
    }
    replace_file(&temp, &destination)?;
    Ok(destination)
}

fn file_ready(app: &Path, package: &Package, file: &catalog::ModelFile) -> Result<bool> {
    let dest = app.join(package.install_dir).join(file.install_name);
    if file.externalize {
        let weights_name = weights_name(file.install_name);
        let weights = dest.with_file_name(&weights_name);
        sidecar_is_valid(&dest, &weights, &weights_name, file.sha256)
    } else {
        file_is_valid(&dest, file.size, file.sha256)
    }
}

fn file_is_valid(path: &Path, size: u64, sha256: &str) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => {
            bail!("拒绝使用符号链接：{}", path.display())
        }
        Ok(meta) if meta.is_file() && meta.len() == size => Ok(sha256_file(path)? == sha256),
        Ok(_) => Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn replace_file(source: &Path, destination: &Path) -> Result<()> {
    if source == destination {
        return Ok(());
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    if fs::symlink_metadata(destination).is_ok() {
        ensure!(
            !fs::symlink_metadata(destination)?.file_type().is_symlink(),
            "拒绝替换符号链接：{}",
            destination.display()
        );
        let backup = destination.with_file_name(format!(
            "{}.replaced",
            destination.file_name().context("文件名")?.to_string_lossy()
        ));
        let _ = fs::remove_file(&backup);
        fs::rename(destination, &backup)?;
        if let Err(error) = fs::rename(source, destination) {
            let _ = fs::rename(&backup, destination);
            return Err(error).with_context(|| format!("替换 {}", destination.display()));
        }
        let _ = fs::remove_file(&backup);
    } else {
        fs::rename(source, destination)?;
    }
    Ok(())
}
