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
    let agent = build_agent(url)?;
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
            progress(&format_progress(written, expected_size));
        }
    }
    file.flush()?;
    drop(file);
    ensure!(
        written == expected_size,
        "下载不完整：{}",
        format_progress(written, expected_size)
    );
    finish_partial(partial, expected_size, expected_sha256)
}

fn format_progress(written: u64, expected: u64) -> String {
    let pct = if expected == 0 {
        100.0
    } else {
        (written as f64 / expected as f64 * 100.0).min(100.0)
    };
    if expected >= 1024 * 1024 {
        format!(
            "已下载 {:.1} / {:.1} MiB ({:.0}%)",
            written as f64 / (1024.0 * 1024.0),
            expected as f64 / (1024.0 * 1024.0),
            pct
        )
    } else {
        format!(
            "已下载 {:.1} / {:.1} KiB ({:.0}%)",
            written as f64 / 1024.0,
            expected as f64 / 1024.0,
            pct
        )
    }
}

fn build_agent(url: &str) -> Result<ureq::Agent> {
    let tls = native_tls::TlsConnector::new().context("初始化系统 TLS 失败")?;
    let mut builder = ureq::AgentBuilder::new()
        .tls_connector(std::sync::Arc::new(tls))
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(120))
        .redirects(0)
        .try_proxy_from_env(true);
    if !has_env_proxy() && !is_loopback_url(url) {
        if let Some(proxy_url) = system_proxy_for_url(url) {
            if let Ok(proxy) = ureq::Proxy::new(&proxy_url) {
                builder = builder.proxy(proxy);
            }
        }
    }
    Ok(builder.build())
}

fn has_env_proxy() -> bool {
    [
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ]
    .iter()
    .any(|name| std::env::var(name).is_ok_and(|value| !value.trim().is_empty()))
}

fn extract_host(url: &str) -> Option<&str> {
    let (_, rest) = url.split_once("://")?;
    let authority = rest.split('/').next()?;
    let host_port = authority.rsplit('@').next()?;
    if let Some(bracketed) = host_port.strip_prefix('[') {
        bracketed.split(']').next()
    } else {
        host_port.split(':').next()
    }
}

fn is_loopback_url(url: &str) -> bool {
    let Some(host) = extract_host(url) else {
        return false;
    };
    host.eq_ignore_ascii_case("localhost") || host == "127.0.0.1" || host == "::1"
}

fn select_windows_proxy(proxy_server: &str, proxy_override: Option<&str>, url: &str) -> Option<String> {
    let raw = proxy_server.trim();
    if raw.is_empty() {
        return None;
    }
    let host = extract_host(url).unwrap_or("");
    if let Some(overrides) = proxy_override {
        for rule in overrides.split(';').map(str::trim).filter(|s| !s.is_empty()) {
            if rule.eq_ignore_ascii_case("<local>") && !host.contains('.') {
                return None;
            }
            if wildcard_match(rule, host) {
                return None;
            }
        }
    }
    let scheme = url.split_once("://").map(|(s, _)| s).unwrap_or("https");
    let chosen = if raw.contains('=') {
        let mut by_scheme = None;
        let mut fallback = None;
        for entry in raw.split(';').map(str::trim).filter(|s| !s.is_empty()) {
            if let Some((k, v)) = entry.split_once('=') {
                let k = k.trim();
                let v = v.trim();
                if v.is_empty() {
                    continue;
                }
                if k.eq_ignore_ascii_case(scheme) {
                    by_scheme = Some(v);
                } else if fallback.is_none() && (k.eq_ignore_ascii_case("http") || k.eq_ignore_ascii_case("https")) {
                    fallback = Some(v);
                }
            }
        }
        by_scheme.or(fallback)?
    } else {
        raw
    };
    if chosen.contains("://") {
        Some(chosen.to_owned())
    } else {
        Some(format!("http://{chosen}"))
    }
}

fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p = pattern.to_ascii_lowercase();
    let t = text.to_ascii_lowercase();
    if p == t {
        return true;
    }
    if let Some(suffix) = p.strip_prefix('*') {
        return t.ends_with(suffix);
    }
    if let Some(prefix) = p.strip_suffix('*') {
        return t.starts_with(prefix);
    }
    false
}

#[cfg(windows)]
fn system_proxy_for_url(url: &str) -> Option<String> {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::Registry::{
        HKEY_CURRENT_USER, RRF_RT_REG_DWORD, RRF_RT_REG_SZ, RegGetValueW,
    };

    fn to_wide(value: &str) -> Vec<u16> {
        OsStr::new(value).encode_wide().chain(Some(0)).collect()
    }

    let subkey = to_wide(r"Software\Microsoft\Windows\CurrentVersion\Internet Settings");
    let enable_name = to_wide("ProxyEnable");
    let mut enabled: u32 = 0;
    let mut size = std::mem::size_of::<u32>() as u32;
    let status = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            subkey.as_ptr(),
            enable_name.as_ptr(),
            RRF_RT_REG_DWORD,
            std::ptr::null_mut(),
            (&mut enabled as *mut u32).cast(),
            &mut size,
        )
    };
    if status != 0 || enabled == 0 {
        return None;
    }

    fn read_reg_string(subkey: &[u16], name: &str) -> Option<String> {
        let wide_name = to_wide(name);
        let mut bytes: u32 = 0;
        let st = unsafe {
            RegGetValueW(
                HKEY_CURRENT_USER,
                subkey.as_ptr(),
                wide_name.as_ptr(),
                RRF_RT_REG_SZ,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut bytes,
            )
        };
        if st != 0 || bytes < 2 {
            return None;
        }
        let mut buf = vec![0u16; (bytes as usize).div_ceil(2)];
        let st = unsafe {
            RegGetValueW(
                HKEY_CURRENT_USER,
                subkey.as_ptr(),
                wide_name.as_ptr(),
                RRF_RT_REG_SZ,
                std::ptr::null_mut(),
                buf.as_mut_ptr().cast(),
                &mut bytes,
            )
        };
        if st != 0 {
            return None;
        }
        let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        Some(String::from_utf16_lossy(&buf[..len]))
    }

    let server = read_reg_string(&subkey, "ProxyServer")?;
    let overrides = read_reg_string(&subkey, "ProxyOverride");
    select_windows_proxy(&server, overrides.as_deref(), url)
}

#[cfg(not(windows))]
fn system_proxy_for_url(_url: &str) -> Option<String> {
    None
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
                if matches!(status, 301 | 302 | 303 | 307 | 308) {
                    let location = response
                        .header("location")
                        .context("重定向缺少 Location")?
                        .to_owned();
                    current = join_location(&current, &location)?;
                    continue;
                }
                ensure!(
                    matches!(status, 200 | 206),
                    "下载失败，HTTP {status} {}",
                    response.status_text()
                );
                let total = content_total(response.header("content-range"), expected_size);
                let reader = response.into_reader();
                return Ok((status, Box::new(reader), total));
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
        ensure!(
            !(current.starts_with("https://") && location.starts_with("http://")),
            "拒绝从 HTTPS 重定向降级到 HTTP"
        );
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

    #[test]
    fn windows_proxy_parser_and_https_downgrade_guard() {
        assert_eq!(
            select_windows_proxy("127.0.0.1:7897", Some("localhost;127.*;<local>"), "https://github.com/a/b").as_deref(),
            Some("http://127.0.0.1:7897")
        );
        assert_eq!(
            select_windows_proxy(
                "http=127.0.0.1:8080;https=127.0.0.1:8443",
                None,
                "https://github.com/a/b"
            )
            .as_deref(),
            Some("http://127.0.0.1:8443")
        );
        assert_eq!(
            select_windows_proxy("127.0.0.1:7897", Some("localhost;127.*;*.local"), "http://127.0.0.1:9000/f"),
            None
        );
        assert!(join_location("https://github.com/a", "http://evil.example/b").is_err());
        assert_eq!(
            join_location("https://github.com/a", "https://objects.githubusercontent.com/b").unwrap(),
            "https://objects.githubusercontent.com/b"
        );
        assert!(format_progress(64 * 1024 * 1024, 128 * 1024 * 1024).contains("64.0 / 128.0 MiB (50%)"));
        assert!(format_progress(512 * 1024, 1024 * 1024).contains("MiB"));
        assert!(format_progress(1024, 2048).contains("1.0 / 2.0 KiB (50%)"));
    }

    #[test]
    fn follows_302_redirect_and_downloads_payload() {
        let body = b"redirected-payload-bytes-ok".to_vec();
        let (target_url, stop_target) = serve(body.clone());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let redir_addr = listener.local_addr().unwrap();
        let stop_redir = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop_redir);
        let loc = target_url.clone();
        thread::spawn(move || {
            listener.set_nonblocking(true).unwrap();
            while !flag.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut socket, _)) => {
                        let _ = socket.set_read_timeout(Some(Duration::from_secs(2)));
                        let mut buf = [0u8; 1024];
                        let _ = socket.read(&mut buf);
                        let resp = format!(
                            "HTTP/1.1 302 Found\r\nLocation: {loc}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        );
                        let _ = socket.write_all(resp.as_bytes());
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });
        let dir = std::env::temp_dir().join(format!("yyur-fetch-302-{}-{}", std::process::id(), line!()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("file.bin");
        let sha = {
            let mut hasher = sha2::Sha256::new();
            sha2::Digest::update(&mut hasher, &body);
            hex(&hasher.finalize())
        };
        let cancel = AtomicBool::new(false);
        download_verified(
            &format!("http://{redir_addr}/start"),
            &dest,
            body.len() as u64,
            &sha,
            0,
            &cancel,
            &|_| {},
        )
        .unwrap();
        assert_eq!(fs::read(&dest).unwrap(), body);
        stop_redir.store(true, Ordering::Relaxed);
        stop_target.store(true, Ordering::Relaxed);
        let _ = fs::remove_dir_all(dir);
    }
}
