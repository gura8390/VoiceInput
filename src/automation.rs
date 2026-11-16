//! Local JSON automation for the tray application. No model, hook or microphone is opened.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::{
    config::{AsrModel, Settings},
    text::HotwordEngine,
};
use anyhow::{Context, Result, bail, ensure};
use clap::Subcommand;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    os::windows::ffi::OsStrExt,
    path::{Path, PathBuf},
    ptr,
    sync::atomic::{AtomicU64, Ordering},
};
use windows_sys::Win32::{
    Foundation::*,
    Storage::FileSystem::{MOVEFILE_WRITE_THROUGH, MoveFileExW, ReplaceFileW},
    System::{Registry::*, Threading::*},
    UI::WindowsAndMessaging::*,
};

const RUN_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";
const RUN_VALUE: &str = "语音输入";
static TEMP_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Subcommand)]
pub enum Command {
    /// 读取托盘进程、保存配置和登录启动注册状态。
    Status,
    /// 请求当前程序的托盘实例正常退出。
    Quit,
    /// 使用当前热词和规则修正给定文字，不录音、不输入到其他窗口。
    Correct { text: String },
    /// 读取或修改保存配置。运行中的实例需要退出后重新启动。
    Config {
        #[arg(long, value_parser = clap::value_parser!(bool))]
        hotwords: Option<bool>,
        #[arg(long, value_parser = clap::value_parser!(bool))]
        hotword_rules: Option<bool>,
    },
    /// 读取或设置本程序管理的单一登录启动项。
    Startup {
        #[arg(long, value_parser = clap::value_parser!(bool))]
        enabled: Option<bool>,
    },
}

impl Command {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Status => "status",
            Self::Quit => "quit",
            Self::Correct { .. } => "correct",
            Self::Config { .. } => "config",
            Self::Startup { .. } => "startup",
        }
    }
}

fn wide(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(Some(0)).collect()
}

fn wide_path(value: &Path) -> Vec<u16> {
    value.as_os_str().encode_wide().chain(Some(0)).collect()
}

fn response(action: &str, data: Value) -> Value {
    json!({"schema": 1, "ok": true, "action": action, "data": data})
}

pub fn failure(action: &str, code: &str, message: &str, details: &str) -> Value {
    json!({"schema": 1, "ok": false, "action": action,
           "error": {"code": code, "message": message, "details": details}})
}

fn saved_configuration(settings: &Settings) -> Value {
    json!({"asr_model": settings.asr_model, "hotwords": settings.hotwords,
           "hotword_rules": settings.hotword_rules, "app_dir": settings.app_dir,
           "settings_scope": "saved_configuration", "active_settings_verified": false})
}

fn load_settings(path: &Path, app_override: Option<&Path>) -> Result<Settings> {
    let present = path.try_exists().context("无法检查配置文件")?;
    Settings::load(present.then_some(path), app_override).context("读取语音输入配置失败")
}

pub fn execute(
    command: &Command,
    config_path: &Path,
    app_override: Option<&Path>,
    model: Option<AsrModel>,
) -> Result<Value> {
    let absolute_config = std::path::absolute(config_path).context("解析配置路径失败")?;
    let config_path = absolute_config.as_path();
    ensure!(
        matches!(command, Command::Config { .. }) || model.is_none(),
        "--asr-model 只能与 automation config 一起使用"
    );
    let executable = std::env::current_exe()?
        .canonicalize()
        .context("无法定位当前程序")?;
    let instances = instances(&executable)?;
    if matches!(command, Command::Quit) {
        return Ok(response("quit", quit_instances(&executable, &instances)?));
    }
    if let Command::Startup { enabled } = command {
        let changed = match enabled {
            Some(enabled) => set_startup(&executable, *enabled)?,
            None => false,
        };
        let mut status = startup_status(&executable)?;
        status["changed"] = json!(changed);
        return Ok(response("startup", status));
    }
    let settings = load_settings(config_path, app_override)?;
    let data = match command {
        Command::Status => json!({
            "running": !instances.is_empty(), "desktop_instances": instances,
            "instance_scope": "current_executable_path", "executable": executable,
            "config_path": config_path, "saved_config": saved_configuration(&settings),
            "startup": startup_status(&executable)?,
        }),
        Command::Quit => unreachable!(),
        Command::Correct { text } => {
            let read = |path: &Path, enabled: bool| -> Result<String> {
                if !enabled {
                    return Ok(String::new());
                }
                match fs::read_to_string(path) {
                    Ok(value) => Ok(value),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
                    Err(error) => Err(error).with_context(|| format!("读取热词文件 {} 失败", path.display())),
                }
            };
            let paths = crate::paths::AppPaths::new(&settings.app_dir);
            let hot = read(&paths.hotwords(), settings.hotwords)?;
            let rules = read(&paths.hotword_rules(), settings.hotword_rules)?;
            let engine = HotwordEngine::from_contents(
                &hot,
                &rules,
                settings.hot_threshold,
                settings.hot_similar_threshold,
            )?;
            let corrected = engine.correct_with_k(text, 0);
            json!({"input": text, "corrected": engine.substitute_rules(&corrected.text),
                   "matches": corrected.matches, "hotwords": settings.hotwords,
                   "hotword_rules": settings.hotword_rules, "scope": "text_correction_only"})
        }
        Command::Config {
            hotwords,
            hotword_rules,
        } => {
            let changed = patch_configuration(
                config_path,
                app_override,
                &ConfigPatch {
                    model,
                    hotwords: *hotwords,
                    hotword_rules: *hotword_rules,
                },
            )?;
            let saved = load_settings(config_path, app_override)?;
            json!({"changed": changed, "config_path": config_path,
                   "restart_required": changed && !instances.is_empty(),
                   "saved_config": saved_configuration(&saved)})
        }
        Command::Startup { .. } => unreachable!(),
    };
    Ok(response(command.name(), data))
}

struct MutexGuard(HANDLE);
impl MutexGuard {
    fn acquire(key: &str) -> Result<Self> {
        let digest = Sha256::digest(key.to_ascii_lowercase().as_bytes());
        let name = wide(&format!("Local\\YuYbUuRu.Automation.{digest:x}"));
        let handle = unsafe { CreateMutexW(ptr::null(), 0, name.as_ptr()) };
        ensure!(
            !handle.is_null(),
            "无法创建配置操作锁：{}",
            std::io::Error::last_os_error()
        );
        let waited = unsafe { WaitForSingleObject(handle, 5_000) };
        if waited != WAIT_OBJECT_0 && waited != WAIT_ABANDONED {
            unsafe {
                CloseHandle(handle);
            }
            bail!("等待配置操作锁失败，请稍后重试（{waited}）");
        }
        Ok(Self(handle))
    }
}
impl Drop for MutexGuard {
    fn drop(&mut self) {
        unsafe {
            ReleaseMutex(self.0);
            CloseHandle(self.0);
        }
    }
}

#[derive(Default)]
struct ConfigPatch {
    model: Option<AsrModel>,
    hotwords: Option<bool>,
    hotword_rules: Option<bool>,
}

fn patch_configuration(
    path: &Path,
    app_override: Option<&Path>,
    patch: &ConfigPatch,
) -> Result<bool> {
    if patch.model.is_none() && patch.hotwords.is_none() && patch.hotword_rules.is_none() {
        return Ok(false);
    }
    let parent = path
        .parent()
        .context("配置文件缺少父目录")?;
    fs::create_dir_all(parent).context("无法创建配置目录")?;
    let parent = parent
        .canonicalize()
        .context("无法解析配置目录")?;
    let path = parent.join(path.file_name().context("配置文件名无效")?);
    let _guard = MutexGuard::acquire(&path.to_string_lossy())?;
    let original = match fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error).context("无法读取原配置文件"),
    };
    let mut value: Value = match &original {
        Some(bytes) => serde_json::from_slice(bytes).context("配置文件不是有效 JSON")?,
        // A new file must keep the same application root as the effective
        // defaults. Relative app_dir values are always relative to this file.
        None => {
            let mut defaults = Settings::load(None, app_override)?;
            let executable = std::env::current_exe()?;
            let package_config =
                crate::paths::AppPaths::new(executable.parent().context("程序缺少父目录")?)
                    .config();
            if app_override.is_none()
                && package_config
                    .canonicalize()
                    .is_ok_and(|directory| directory == parent)
            {
                defaults.app_dir = PathBuf::from("..");
            }
            serde_json::to_value(defaults)?
        }
    };
    let before = value.clone();
    let object = value.as_object_mut().context("配置 JSON 必须是对象")?;
    if let Some(model) = patch.model {
        object.insert("asr_model".into(), serde_json::to_value(model)?);
    }
    if let Some(enabled) = patch.hotwords {
        object.insert("hotwords".into(), json!(enabled));
    }
    if let Some(enabled) = patch.hotword_rules {
        object.insert("hotword_rules".into(), json!(enabled));
    }
    if value == before {
        return Ok(false);
    }
    let mut bytes = serde_json::to_vec_pretty(&value)?;
    bytes.push(b'\n');
    let temporary = parent.join(format!(
        ".settings-{}-{}.tmp",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .context("无法创建临时配置文件")?;
        file.write_all(&bytes).context("写入临时配置失败")?;
        file.sync_all().context("保存临时配置失败")?;
        drop(file);
        Settings::load(Some(&temporary), app_override).context("新配置校验失败")?;
        let current = match fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error).context("核对原配置失败"),
        };
        ensure!(current == original, "配置已被其他程序修改，请重试");
        let target_name = wide_path(&path);
        let temp_name = wide_path(&temporary);
        let saved = unsafe {
            if original.is_some() {
                ReplaceFileW(
                    target_name.as_ptr(),
                    temp_name.as_ptr(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                )
            } else {
                MoveFileExW(
                    temp_name.as_ptr(),
                    target_name.as_ptr(),
                    MOVEFILE_WRITE_THROUGH,
                )
            }
        };
        ensure!(
            saved != 0,
            "原子保存配置失败：{}",
            std::io::Error::last_os_error()
        );
        Ok(())
    })();
    if temporary.exists() {
        let _ = fs::remove_file(&temporary);
    }
    result?;
    Ok(true)
}

#[derive(Serialize)]
struct TrayInstance {
    pid: u32,
    window: usize,
}
struct Enumeration<'a> {
    executable: &'a Path,
    found: Vec<TrayInstance>,
}

unsafe fn process_image(process: HANDLE) -> Result<PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    let mut buffer = vec![0u16; 32_768];
    let mut length = buffer.len() as u32;
    ensure!(
        QueryFullProcessImageNameW(process, 0, buffer.as_mut_ptr(), &mut length) != 0,
        "读取进程程序路径失败：{}",
        std::io::Error::last_os_error()
    );
    PathBuf::from(std::ffi::OsString::from_wide(&buffer[..length as usize]))
        .canonicalize()
        .context("无法核对进程程序路径")
}

fn same_image(a: &Path, b: &Path) -> bool {
    a.to_string_lossy()
        .eq_ignore_ascii_case(&b.to_string_lossy())
}

unsafe extern "system" fn enumerate_window(
    hwnd: HWND,
    parameter: LPARAM,
) -> windows_sys::core::BOOL {
    let state = &mut *(parameter as *mut Enumeration<'_>);
    let mut class = [0u16; 256];
    let length = GetClassNameW(hwnd, class.as_mut_ptr(), class.len() as i32);
    if length <= 0 {
        return 1;
    }
    let class = String::from_utf16_lossy(&class[..length as usize]);
    let mut pid = 0;
    GetWindowThreadProcessId(hwnd, &mut pid);
    if class != format!("YuYbUuRuTray_{pid}") {
        return 1;
    }
    let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
    if process.is_null() {
        return 1;
    }
    let matched = process_image(process).is_ok_and(|image| same_image(&image, state.executable));
    CloseHandle(process);
    if matched {
        state.found.push(TrayInstance {
            pid,
            window: hwnd as usize,
        });
    }
    1
}

fn instances(executable: &Path) -> Result<Vec<TrayInstance>> {
    let mut state = Enumeration {
        executable,
        found: Vec::new(),
    };
    ensure!(
        unsafe {
            EnumWindows(
                Some(enumerate_window),
                (&mut state as *mut Enumeration<'_>) as LPARAM,
            )
        } != 0,
        "读取托盘实例失败：{}",
        std::io::Error::last_os_error()
    );
    Ok(state.found)
}

fn quit_instances(executable: &Path, instances: &[TrayInstance]) -> Result<Value> {
    let mut exited = Vec::new();
    for instance in instances {
        let process = unsafe {
            OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SYNCHRONIZE,
                0,
                instance.pid,
            )
        };
        if process.is_null() {
            continue;
        }
        let outcome = (|| -> Result<()> {
            ensure!(
                same_image(&unsafe { process_image(process) }?, executable),
                "退出目标的程序路径已变化"
            );
            let mut current_pid = 0;
            unsafe {
                GetWindowThreadProcessId(instance.window as HWND, &mut current_pid);
            }
            ensure!(current_pid == instance.pid, "退出目标窗口已变化");
            ensure!(
                unsafe { PostMessageW(instance.window as HWND, WM_CLOSE, 0, 0) } != 0,
                "无法向托盘实例发送退出请求：{}",
                std::io::Error::last_os_error()
            );
            ensure!(
                unsafe { WaitForSingleObject(process, 10_000) } == WAIT_OBJECT_0,
                "托盘实例未在十秒内完成退出，请等待当前任务结束后重试"
            );
            Ok(())
        })();
        unsafe {
            CloseHandle(process);
        }
        outcome?;
        exited.push(instance.pid);
    }
    ensure!(
        self::instances(executable)?.is_empty(),
        "仍有语音输入托盘实例未退出，请稍后重试"
    );
    Ok(json!({"requested": instances.len(), "exited_pids": exited, "running": false}))
}

struct RegistryKey(HKEY);
impl Drop for RegistryKey {
    fn drop(&mut self) {
        unsafe {
            RegCloseKey(self.0);
        }
    }
}

fn run_value() -> Result<Option<String>> {
    let mut key = ptr::null_mut();
    let opened = unsafe {
        RegOpenKeyExW(
            HKEY_CURRENT_USER,
            wide(RUN_KEY).as_ptr(),
            0,
            KEY_QUERY_VALUE,
            &mut key,
        )
    };
    if opened == ERROR_FILE_NOT_FOUND {
        return Ok(None);
    }
    ensure!(
        opened == ERROR_SUCCESS,
        "读取登录启动注册项失败（{opened}）"
    );
    let key = RegistryKey(key);
    let mut kind = 0;
    let mut bytes = 0;
    let status = unsafe {
        RegQueryValueExW(
            key.0,
            wide(RUN_VALUE).as_ptr(),
            ptr::null(),
            &mut kind,
            ptr::null_mut(),
            &mut bytes,
        )
    };
    if status == ERROR_FILE_NOT_FOUND {
        return Ok(None);
    }
    ensure!(
        status == ERROR_SUCCESS && kind == REG_SZ && bytes <= 65_536 && bytes % 2 == 0,
        "登录启动项格式无效或无法读取（{status}）"
    );
    let mut text = vec![0u16; bytes as usize / 2 + 1];
    let status = unsafe {
        RegQueryValueExW(
            key.0,
            wide(RUN_VALUE).as_ptr(),
            ptr::null(),
            &mut kind,
            text.as_mut_ptr().cast(),
            &mut bytes,
        )
    };
    ensure!(
        status == ERROR_SUCCESS && kind == REG_SZ,
        "读取登录启动命令失败（{status}）"
    );
    let end = text
        .iter()
        .position(|unit| *unit == 0)
        .unwrap_or(text.len());
    Ok(Some(
        String::from_utf16(&text[..end]).context("登录启动命令编码无效")?,
    ))
}

fn startup_command(executable: &Path) -> String {
    let path = executable.to_string_lossy();
    let path = path.strip_prefix(r"\\?\").unwrap_or(&path);
    format!("\"{path}\"")
}

fn startup_status(executable: &Path) -> Result<Value> {
    let expected = startup_command(executable);
    let registered = run_value()?;
    let enabled = registered.as_deref() == Some(expected.as_str());
    Ok(json!({"enabled": enabled, "value_name": RUN_VALUE,
              "registered_command": registered, "expected_command": expected,
              "conflict": registered.is_some() && !enabled,
              "validation_scope": "registration_only", "login_display": "tray"}))
}

fn set_startup(executable: &Path, enabled: bool) -> Result<bool> {
    let _guard = MutexGuard::acquire("YuYbUuRu.Startup.RunValue")?;
    let expected = startup_command(executable);
    let registered = run_value()?;
    ensure!(
        registered.as_deref().is_none_or(|value| value == expected),
        "同名登录启动项指向其他程序，无法自动覆盖或删除"
    );
    if enabled == registered.is_some() {
        return Ok(false);
    }
    let mut key = ptr::null_mut();
    let status = unsafe {
        RegCreateKeyExW(
            HKEY_CURRENT_USER,
            wide(RUN_KEY).as_ptr(),
            0,
            ptr::null(),
            REG_OPTION_NON_VOLATILE,
            KEY_SET_VALUE,
            ptr::null(),
            &mut key,
            ptr::null_mut(),
        )
    };
    ensure!(
        status == ERROR_SUCCESS,
        "打开登录启动注册项失败（{status}）"
    );
    let key = RegistryKey(key);
    let status = unsafe {
        if enabled {
            let value = wide(&expected);
            RegSetValueExW(
                key.0,
                wide(RUN_VALUE).as_ptr(),
                0,
                REG_SZ,
                value.as_ptr().cast(),
                (value.len() * 2) as u32,
            )
        } else {
            RegDeleteValueW(key.0, wide(RUN_VALUE).as_ptr())
        }
    };
    ensure!(
        status == ERROR_SUCCESS,
        "更新登录启动注册项失败（{status}）"
    );
    ensure!(
        run_value()?.as_deref() == enabled.then_some(expected.as_str()),
        "登录启动状态读回不一致"
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);
    impl TestDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "yyur-automation-{}-{}",
                std::process::id(),
                TEMP_ID.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn configuration_preserves_unknown_fields_and_repeated_patch_is_idempotent() {
        let directory = TestDirectory::new();
        let path = directory.0.join("settings.json");
        fs::write(
            &path,
            br#"{"app_dir":".","asr_model":"0.6b","future":{"value":42},"language":"auto"}"#,
        )
        .unwrap();
        let patch = ConfigPatch {
            model: Some(AsrModel::Qwen3Asr17B),
            hotwords: Some(false),
            hotword_rules: None,
        };
        assert!(patch_configuration(&path, None, &patch).unwrap());
        let saved = fs::read(&path).unwrap();
        let value: Value = serde_json::from_slice(&saved).unwrap();
        assert_eq!(value["future"]["value"], 42);
        assert_eq!(value["language"], "auto");
        assert_eq!(value["app_dir"], ".");
        assert_eq!(value["asr_model"], "1.7b");
        assert_eq!(value["hotwords"], false);
        assert!(!patch_configuration(&path, None, &patch).unwrap());
        assert_eq!(fs::read(&path).unwrap(), saved);
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
    }

    #[test]
    fn invalid_configuration_is_preserved() {
        let directory = TestDirectory::new();
        let path = directory.0.join("settings.json");
        let original = br#"{"app_dir":".","segment_seconds":999}"#;
        fs::write(&path, original).unwrap();
        assert!(
            patch_configuration(
                &path,
                None,
                &ConfigPatch {
                    hotwords: Some(false),
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
    }

    #[test]
    fn creating_custom_configuration_keeps_the_effective_application_root() {
        let directory = TestDirectory::new();
        let path = directory.0.join("settings.json");
        let effective = Settings::load(None, None).unwrap();
        assert!(patch_configuration(
            &path,
            None,
            &ConfigPatch { hotwords: Some(false), ..Default::default() },
        ).unwrap());
        let saved = Settings::load(Some(&path), None).unwrap();
        assert_eq!(saved.app_dir, effective.app_dir);
        assert!(!saved.hotwords);
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
    }
}
