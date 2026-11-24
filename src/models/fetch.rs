//! Resumable HTTP download with SHA256 verification.
use super::{parse_sha256, sha256_file};
use anyhow::{Context, Result, bail, ensure};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

pub(crate) fn download_verified(
    url: &str,
    destination: &Path,
    expected_size: u64,
    expected_sha256: &str,
    retries: u32,
    cancel: &AtomicBool,
    progress: &dyn Fn(&str),
) -> Result<()> {
    let expected_sha256 = parse_sha256(expected_sha256)?;
    if file_matches(destination, expected_size, &expected_sha256)? {
        progress("已有文件校验通过，跳过下载");
        return Ok(());
    }
    if destination.exists() {
        ensure!(
            !is_symlink(destination)?,
            "目标是符号链接，拒绝覆盖：{}",
            destination.display()
        );
        fs::remove_file(destination)?;
    }
    let partial = partial_path(destination)?;
    let attempts = retries.saturating_add(1);
    let mut last_error = None;
    for attempt in 1..=attempts {
        if cancel.load(Ordering::Acquire) {
            bail!("已取消下载");
        }
        match download_once(
            url,
            &partial,
            expected_size,
            &expected_sha256,
            cancel,
            progress,
        ) {
            Ok(()) => {
                if let Some(parent) = destination.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::rename(&partial, destination)?;
                return Ok(());
            }
            Err(error) => {
                let message = format!("{error:#}");
                if message.contains("已取消下载") || attempt == attempts {
                    return Err(error);
                }
                progress(&format!("下载中断，正在重试（{attempt}/{attempts}）"));
                last_error = Some(error);
                std::thread::sleep(Duration::from_millis((400 * u64::from(attempt)).min(8_000)));
            }
        }
    }
    Err(last_error.context("下载失败")?)
}

fn download_once(
    url: &str,
    partial: &Path,
    expected_size: u64,
    expected_sha256: &str,
    cancel: &AtomicBool,
    progress: &dyn Fn(&str),
) -> Result<()> {
    if let Some(parent) = partial.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut have = match fs::symlink_metadata(partial) {
        Ok(meta) => {
            ensure!(!meta.file_type().is_symlink(), "临时文件不能是符号链接");
            meta.len()
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error.into()),
    };
    if have > expected_size {
        fs::remove_file(partial)?;
        have = 0;
    }
    if have == expected_size {
        return finish_partial(partial, expected_size, expected_sha256);
    }
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(120))
        .redirects(0)
        .build();
    let (status, mut reader, total) = open_with_redirects(&agent, url, have, expected_size)?;
    if status == 200 && have != 0 {
        have = 0;
    }
    if status == 416 {
        return finish_partial(partial, expected_size, expected_sha256);
    }
    if let Some(total) = total {
        ensure!(total == expected_size, "服务器报告的大小与目录不一致");
    }
    let mut file = if have == 0 {
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(partial)?
    } else {
        OpenOptions::new().append(true).open(partial)?
    };
    let mut buffer = [0u8; 256 * 1024];
    let mut written = have;
    loop {
        if cancel.load(Ordering::Acquire) {
            file.flush()?;
            bail!("已取消下载");
        }
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])?;
        written += n as u64;
        ensure!(written <= expected_size, "下载超出目录中的大小");
        if written == expected_size
            || written / (8 * 1024 * 1024) != (written - n as u64) / (8 * 1024 * 1024)
        {
            progress(&format!("已下载 {written} / {expected_size} 字节"));
        }
    }
    file.flush()?;
    drop(file);
    ensure!(
        written == expected_size,
        "下载不完整：{written}/{expected_size} 字节"
    );
    finish_partial(partial, expected_size, expected_sha256)
}

fn finish_partial(partial: &Path, expected_size: u64, expected_sha256: &str) -> Result<()> {
    let meta = fs::symlink_metadata(partial)?;
    ensure!(meta.len() == expected_size, "文件大小与目录不一致");
    let actual = sha256_file(partial)?;
    if actual != expected_sha256 {
        let _ = fs::remove_file(partial);
        bail!("SHA256 不一致，已丢弃损坏文件");
    }
    Ok(())
}

fn open_with_redirects(
    agent: &ureq::Agent,
    url: &str,
    offset: u64,
    expected_size: u64,
) -> Result<(u16, Box<dyn Read + Send>, Option<u64>)> {
    let mut current = url.to_owned();
    for _ in 0..8 {
        let range = format!("bytes={offset}-");
        let response = agent
            .get(&current)
            .set("Range", &range)
            .set("User-Agent", "YuYbUuRu")
            .call();
        match response {
            Ok(response) => {
                let status = response.status();
                let total = content_total(response.header("content-range"), expected_size);
                let reader = response.into_reader();
                return Ok((status, Box::new(reader), total));
            }
            Err(ureq::Error::Status(code, response))
                if matches!(code, 301 | 302 | 303 | 307 | 308) =>
            {
                let location = response
                    .header("location")
                    .context("重定向缺少 Location")?
                    .to_owned();
                current = join_location(&current, &location)?;
            }
            Err(ureq::Error::Status(416, _)) => {
                return Ok((416, Box::new(std::io::empty()), Some(expected_size)));
            }
            Err(ureq::Error::Status(code, response)) => {
                bail!("下载失败，HTTP {code} {}", response.status_text());
            }
            Err(error) => return Err(error).context("下载连接失败"),
        }
    }
    bail!("重定向次数过多")
}

fn content_total(header: Option<&str>, _expected: u64) -> Option<u64> {
    let header = header?;
    let total = header.rsplit('/').next()?;
    if total == "*" {
        return None;
    }
    total.parse().ok()
}

fn join_location(current: &str, location: &str) -> Result<String> {
    if location.starts_with("https://") || location.starts_with("http://") {
        return Ok(location.to_owned());
    }
    ensure!(location.starts_with('/'), "无法解析重定向地址");
    let (scheme, rest) = current.split_once("://").context("下载地址无效")?;
    let host = rest.split('/').next().context("下载地址缺少主机")?;
    Ok(format!("{scheme}://{host}{location}"))
}

fn file_matches(path: &Path, expected_size: u64, expected_sha256: &str) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => bail!("拒绝使用符号链接：{}", path.display()),
        Ok(meta) if meta.is_file() && meta.len() == expected_size => {
            Ok(sha256_file(path)? == expected_sha256)
        }
        Ok(_) => Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn is_symlink(path: &Path) -> Result<bool> {
    Ok(fs::symlink_metadata(path)?.file_type().is_symlink())
}

fn partial_path(destination: &Path) -> Result<std::path::PathBuf> {
    let name = destination
        .file_name()
        .context("下载文件名")?
        .to_string_lossy();
    Ok(destination.with_file_name(format!("{name}.partial")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::hex;
    use sha2::Digest;
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::thread;

    fn serve(body: Vec<u8>) -> (String, Arc<AtomicBool>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        thread::spawn(move || {
            while !flag.load(Ordering::Relaxed) {
                listener.set_nonblocking(true).unwrap();
                match listener.accept() {
                    Ok((mut socket, _)) => {
                        let _ = socket.set_read_timeout(Some(Duration::from_secs(2)));
                        let _ = socket.set_write_timeout(Some(Duration::from_secs(2)));
                        let _ = handle(&mut socket, &body);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });
        (format!("http://{address}/file.bin"), stop)
    }

    fn handle(socket: &mut std::net::TcpStream, body: &[u8]) -> std::io::Result<()> {
        let mut request = Vec::new();
        let mut byte = [0u8; 1];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") && request.len() < 8192 {
            if socket.read(&mut byte)? == 0 {
                break;
            }
            request.push(byte[0]);
        }
        let text = String::from_utf8_lossy(&request);
        let mut start = 0usize;
        for line in text.lines() {
            if let Some(value) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                start = value.split('-').next().unwrap_or("0").parse().unwrap_or(0);
            }
        }
        let start = start.min(body.len());
        let slice = &body[start..];
        let header = if start == 0 {
            format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
        } else if start >= body.len() {
            "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                .to_owned()
        } else {
            format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{}/*\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len() - 1,
                slice.len()
            )
        };
        socket.write_all(header.as_bytes())?;
        if !header.starts_with("HTTP/1.1 416") {
            socket.write_all(slice)?;
        }
        Ok(())
    }

    #[test]
    fn resume_downloads_and_existing_hash_is_skipped() {
        let body = b"hello world, this payload is resumed".to_vec();
        let (url, stop) = serve(body.clone());
        let dir =
            std::env::temp_dir().join(format!("yyur-fetch-{}-{}", std::process::id(), line!()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let destination = dir.join("file.bin");
        let partial = dir.join("file.bin.partial");
        fs::write(&partial, &body[..6]).unwrap();
        let sha = {
            let mut hasher = sha2::Sha256::new();
            sha2::Digest::update(&mut hasher, &body);
            hex(&hasher.finalize())
        };
        let cancel = AtomicBool::new(false);
        download_verified(
            &url,
            &destination,
            body.len() as u64,
            &sha,
            0,
            &cancel,
            &|_| {},
        )
        .unwrap();
        assert_eq!(fs::read(&destination).unwrap(), body);
        assert!(!partial.exists());
        // A refused port must not be contacted when the verified file is already present.
        download_verified(
            "http://127.0.0.1:1/absent",
            &destination,
            body.len() as u64,
            &sha,
            0,
            &cancel,
            &|_| {},
        )
        .unwrap();
        fs::write(&destination, b"corrupt").unwrap();
        let error = download_verified(
            &url,
            &destination,
            body.len() as u64,
            &sha,
            0,
            &cancel,
            &|_| {},
        );
        assert!(error.is_ok(), "{error:?}");
        assert_eq!(fs::read(&destination).unwrap(), body);
        stop.store(true, Ordering::Relaxed);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn hash_mismatch_removes_the_partial_file() {
        let body = b"not-the-expected-bytes".to_vec();
        let (url, stop) = serve(body.clone());
        let dir =
            std::env::temp_dir().join(format!("yyur-fetch-bad-{}-{}", std::process::id(), line!()));
        fs::create_dir_all(&dir).unwrap();
        let destination = dir.join("file.bin");
        let cancel = AtomicBool::new(false);
        let error = download_verified(
            &url,
            &destination,
            body.len() as u64,
            &"ab".repeat(32),
            0,
            &cancel,
            &|_| {},
        );
        assert!(error.is_err());
        assert!(!destination.exists());
        assert!(!dir.join("file.bin.partial").exists());
        stop.store(true, Ordering::Relaxed);
        let _ = fs::remove_dir_all(dir);
    }
}
