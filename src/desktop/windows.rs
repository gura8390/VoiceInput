#![allow(unsafe_op_in_unsafe_fn)]
use super::{
    BudgetEventSender, DesktopEvent, DesktopOptions, DesktopOutput, HoldAction, HoldTracker,
    Shortcut, input,
};
use crate::capture::Capture;
use anyhow::{Result, bail};
#[cfg(test)]
use std::sync::mpsc::Sender;
use std::{
    cell::{Cell, RefCell},
    ffi::OsString,
    mem::size_of,
    os::windows::ffi::{OsStrExt, OsStringExt},
    path::PathBuf,
    ptr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::Receiver,
    },
    thread,
    time::Instant,
};
use windows_sys::Win32::{
    Foundation::*,
    System::LibraryLoader::GetModuleHandleW,
    UI::{
        Controls::Dialogs::*, Input::KeyboardAndMouse::VK_CAPITAL, Shell::*, WindowsAndMessaging::*,
    },
};

const WM_TRAY: u32 = WM_APP + 1;
const WM_OUTPUT: u32 = WM_APP + 2;
const WM_REPLAY: u32 = WM_APP + 3;
const NIN_KEYSELECT: u32 = 0x0401;
const TIMER_HOLD: usize = 1;
const TIMER_HEALTH: usize = 2;
const TIMER_SHUTDOWN: usize = 3;
const COMMAND_TRANSCRIBE: u32 = 1;
const COMMAND_HOTWORDS: u32 = 2;
const COMMAND_QUIT: u32 = 3;
const COMMAND_DOWNLOAD: u32 = 4;

thread_local! { static HOOK_CONTEXT: Cell<*const Context> = const { Cell::new(ptr::null()) }; }

#[cfg(test)]
enum PassiveSmokeEvent {
    Ready {
        hwnd: usize,
        thread_id: u32,
        tray_added: bool,
        capture_enabled: bool,
        hooks_enabled: bool,
    },
    State {
        text: String,
        icon_modified: bool,
    },
}
#[cfg(test)]
thread_local! {
    static PASSIVE_SMOKE: RefCell<Option<Sender<PassiveSmokeEvent>>> = const { RefCell::new(None) };
}
#[cfg(test)]
fn observe_passive_smoke(event: PassiveSmokeEvent) {
    PASSIVE_SMOKE.with(|slot| {
        if let Some(sender) = &*slot.borrow() {
            let _ = sender.send(event);
        }
    });
}

fn wide(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(Some(0)).collect()
}
fn wide_path(value: &std::path::Path) -> Vec<u16> {
    value.as_os_str().encode_wide().chain(Some(0)).collect()
}
fn tip(value: &str) -> [u16; 128] {
    fill_wide(value)
}

fn fill_wide<const N: usize>(value: &str) -> [u16; N] {
    let mut output = [0u16; N];
    if N == 0 {
        return output;
    }
    let mut length = 0;
    for unit in value.encode_utf16() {
        if length + 1 >= N {
            break;
        }
        output[length] = unit;
        length += 1;
    }
    if length > 0 && (0xd800..=0xdbff).contains(&output[length - 1]) {
        output[length - 1] = 0;
    }
    output
}

struct ApplicationIcons {
    large: HICON,
    small: HICON,
}

impl ApplicationIcons {
    unsafe fn load(instance: HINSTANCE) -> Result<Self> {
        let large = LoadImageW(
            instance,
            1usize as *const u16,
            IMAGE_ICON,
            GetSystemMetrics(SM_CXICON),
            GetSystemMetrics(SM_CYICON),
            LR_SHARED,
        ) as HICON;
        let small = LoadImageW(
            instance,
            1usize as *const u16,
            IMAGE_ICON,
            GetSystemMetrics(SM_CXSMICON),
            GetSystemMetrics(SM_CYSMICON),
            LR_SHARED,
        ) as HICON;
        if !large.is_null() && !small.is_null() {
            return Ok(Self { large, small });
        }
        bail!("无法加载语音输入图标：{}", std::io::Error::last_os_error());
    }
}

struct Context {
    options: DesktopOptions,
    events: BudgetEventSender,
    capture: Capture,
    holds: RefCell<HoldTracker>,
    hwnd: Cell<HWND>,
    icons: ApplicationIcons,
    status: RefCell<String>,
    taskbar_created: u32,
    added: Cell<bool>,
    retries: Cell<u32>,
    closing: Cell<bool>,
    alive: Arc<AtomicBool>,
    busy: Arc<AtomicBool>,
    models_missing: Cell<bool>,
    model_notice_shown: Cell<bool>,
    downloading: Arc<AtomicBool>,
}

impl Context {
    unsafe fn notification(&self) -> NOTIFYICONDATAW {
        let mut data = NOTIFYICONDATAW::default();
        data.cbSize = size_of::<NOTIFYICONDATAW>() as u32;
        data.hWnd = self.hwnd.get();
        data.uID = 1;
        data.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP | NIF_SHOWTIP;
        data.uCallbackMessage = WM_TRAY;
        data.hIcon = self.icons.small;
        data.szTip = tip(&format!("语音输入 · {}", self.status.borrow()));
        data
    }

    unsafe fn add_icon(&self) -> bool {
        let mut data = self.notification();
        if Shell_NotifyIconW(NIM_ADD, &mut data) == 0 {
            return false;
        }
        self.added.set(true);
        data.Anonymous.uVersion = NOTIFYICON_VERSION_4;
        Shell_NotifyIconW(NIM_SETVERSION, &mut data);
        self.show_model_notice();
        true
    }

    unsafe fn show_model_notice(&self) {
        if !self.models_missing.get() || !self.added.get() || self.model_notice_shown.get() {
            return;
        }
        let mut data = self.notification();
        data.uFlags |= NIF_INFO;
        data.szInfoTitle = fill_wide("语音输入");
        data.szInfo = fill_wide("缺少语音模型。请在托盘菜单选择“下载模型”。");
        data.dwInfoFlags = NIIF_INFO;
        if Shell_NotifyIconW(NIM_MODIFY, &mut data) != 0 {
            self.model_notice_shown.set(true);
        }
    }

    unsafe fn set_status(&self, value: String) {
        *self.status.borrow_mut() = value;
        let modified =
            self.added.get() && Shell_NotifyIconW(NIM_MODIFY, &mut self.notification()) != 0;
        #[cfg(test)]
        observe_passive_smoke(PassiveSmokeEvent::State {
            text: self.status.borrow().clone(),
            icon_modified: modified,
        });
        #[cfg(not(test))]
        let _ = modified;
    }

    unsafe fn key(&self, key: Shortcut, down: bool) {
        if self.closing.get() {
            return;
        }
        let actions = {
            let mut holds = self.holds.borrow_mut();
            if down {
                holds.press(key, Instant::now())
            } else {
                holds.release(key, Instant::now())
            }
        };
        for action in actions {
            match action {
                HoldAction::Begin => {
                    self.capture.begin_press();
                    SetTimer(self.hwnd.get(), TIMER_HOLD, 300, None);
                }
                HoldAction::Finish => {
                    KillTimer(self.hwnd.get(), TIMER_HOLD);
                    self.capture.finish_press();
                    self.set_status("正在识别".into());
                }
                HoldAction::Cancel => {
                    KillTimer(self.hwnd.get(), TIMER_HOLD);
                    self.capture.cancel_press();
                    self.set_status("待命".into());
                }
                HoldAction::Replay(key) => {
                    PostMessageW(self.hwnd.get(), WM_REPLAY, key.index(), 0);
                }
            }
        }
    }

    unsafe fn start_shutdown(&self) {
        if self.closing.replace(true) {
            return;
        }
        self.alive.store(false, Ordering::Release);
        self.options.cancel_download.store(true, Ordering::Release);
        self.capture.cancel_press();
        let _ = self.events.send(DesktopEvent::Quit);
        KillTimer(self.hwnd.get(), TIMER_HOLD);
        // Keep pumping the owner window while an in-flight paste restores its
        // clipboard. Idle output receivers need no join and cannot inject later.
        if self.busy.load(Ordering::Acquire) {
            SetTimer(self.hwnd.get(), TIMER_SHUTDOWN, 20, None);
        } else {
            PostQuitMessage(0);
        }
    }

    unsafe fn choose_file(&self) {
        let mut file = vec![0u16; 32_768];
        let filter = wide(
            "音频与视频\0*.wav;*.mp3;*.m4a;*.flac;*.ogg;*.aac;*.wma;*.mp4;*.mkv;*.webm;*.mov\0所有文件\0*.*\0",
        );
        let title = wide("选择需要转录的音频或视频");
        let mut dialog = OPENFILENAMEW::default();
        dialog.lStructSize = size_of::<OPENFILENAMEW>() as u32;
        dialog.hwndOwner = self.hwnd.get();
        dialog.lpstrFilter = filter.as_ptr();
        dialog.lpstrTitle = title.as_ptr();
        dialog.lpstrFile = file.as_mut_ptr();
        dialog.nMaxFile = file.len() as u32;
        dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
        if GetOpenFileNameW(&mut dialog) != 0 {
            let end = file
                .iter()
                .position(|unit| *unit == 0)
                .unwrap_or(file.len());
            let path = PathBuf::from(OsString::from_wide(&file[..end]));
            let _ = self.events.send(DesktopEvent::Transcribe(path));
        } else {
            let error = CommDlgExtendedError();
            if error != 0 {
                self.set_status(format!("打开文件失败: {error}"));
            }
        }
    }

    unsafe fn start_download(&self) {
        if self.downloading.swap(true, Ordering::AcqRel) {
            self.set_status("正在下载模型".into());
            return;
        }
        self.options.cancel_download.store(false, Ordering::Release);
        self.set_status("正在下载模型".into());
        if self.events.send(DesktopEvent::DownloadModels).is_err() {
            self.downloading.store(false, Ordering::Release);
            self.set_status("无法开始下载模型".into());
        }
    }

    unsafe fn edit_hotwords(&self) {
        let file = crate::paths::AppPaths::new(&self.options.base_dir).hotwords();
        if !file.exists() {
            if let Err(error) = std::fs::create_dir_all(file.parent().expect("Hotword directory"))
                .and_then(|_| std::fs::write(&file, b""))
            {
                self.set_status(error.to_string());
                return;
            }
        }
        let verb = wide("open");
        let path = wide_path(&file);
        let result = ShellExecuteW(
            self.hwnd.get(),
            verb.as_ptr(),
            path.as_ptr(),
            ptr::null(),
            ptr::null(),
            SW_SHOWNORMAL,
        );
        if result as usize <= 32 {
            self.set_status("无法打开热词文件".into());
        }
    }

    unsafe fn menu(&self) {
        let menu = CreatePopupMenu();
        if menu.is_null() {
            return;
        }
        let state = wide(&self.status.borrow());
        AppendMenuW(menu, MF_STRING | MF_GRAYED, 0, state.as_ptr());
        AppendMenuW(menu, MF_SEPARATOR, 0, ptr::null());
        let mut entries = Vec::new();
        if self.models_missing.get() {
            entries.push((COMMAND_DOWNLOAD, "下载模型"));
        }
        entries.extend([
            (COMMAND_TRANSCRIBE, "转录音频 / 视频…"),
            (COMMAND_HOTWORDS, "编辑热词"),
            (COMMAND_QUIT, "退出"),
        ]);
        for (id, label) in entries {
            let text = wide(label);
            AppendMenuW(menu, MF_STRING, id as usize, text.as_ptr());
        }
        let mut point = POINT::default();
        GetCursorPos(&mut point);
        SetForegroundWindow(self.hwnd.get());
        let command = TrackPopupMenuEx(
            menu,
            TPM_RETURNCMD | TPM_NONOTIFY | TPM_RIGHTBUTTON,
            point.x,
            point.y,
            self.hwnd.get(),
            ptr::null(),
        ) as u32;
        PostMessageW(self.hwnd.get(), WM_NULL, 0, 0);
        DestroyMenu(menu);
        match command {
            COMMAND_DOWNLOAD => self.start_download(),
            COMMAND_TRANSCRIBE => self.choose_file(),
            COMMAND_HOTWORDS => self.edit_hotwords(),
            COMMAND_QUIT => self.start_shutdown(),
            _ => {}
        }
    }
}

unsafe extern "system" fn keyboard_hook(code: i32, message: WPARAM, data: LPARAM) -> LRESULT {
    if code >= 0 {
        let event = &*(data as *const KBDLLHOOKSTRUCT);
        if event.vkCode == VK_CAPITAL as u32
            && event.flags & LLKHF_INJECTED == 0
            && event.dwExtraInfo != input::INPUT_TAG
        {
            let handled = HOOK_CONTEXT.with(|slot| {
                let context = slot.get();
                if context.is_null() {
                    return false;
                }
                let down = matches!(message as u32, WM_KEYDOWN | WM_SYSKEYDOWN);
                let up = matches!(message as u32, WM_KEYUP | WM_SYSKEYUP);
                if down || up {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        (&*context).key(Shortcut::CapsLock, down)
                    }));
                    true
                } else {
                    false
                }
            });
            if handled {
                return 1;
            }
        }
    }
    CallNextHookEx(ptr::null_mut(), code, message, data)
}

unsafe extern "system" fn mouse_hook(code: i32, message: WPARAM, data: LPARAM) -> LRESULT {
    if code >= 0 && matches!(message as u32, WM_XBUTTONDOWN | WM_XBUTTONUP) {
        let event = &*(data as *const MSLLHOOKSTRUCT);
        if event.mouseData >> 16 == 2
            && event.flags & LLMHF_INJECTED == 0
            && event.dwExtraInfo != input::INPUT_TAG
        {
            let handled = HOOK_CONTEXT.with(|slot| {
                let context = slot.get();
                if context.is_null() {
                    return false;
                }
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (&*context).key(Shortcut::X2, message as u32 == WM_XBUTTONDOWN)
                }));
                true
            });
            if handled {
                return 1;
            }
        }
    }
    CallNextHookEx(ptr::null_mut(), code, message, data)
}

unsafe fn dispatch(
    context: &Context,
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> Option<LRESULT> {
    if message == context.taskbar_created && message != 0 {
        context.added.set(false);
        context.retries.set(0);
        context.add_icon();
        return Some(0);
    }
    match message {
        WM_TRAY => {
            match lparam as u32 & 0xffff {
                WM_CONTEXTMENU | WM_RBUTTONUP => context.menu(),
                WM_LBUTTONDBLCLK | NIN_KEYSELECT => context.choose_file(),
                _ => {}
            }
            Some(0)
        }
        WM_TIMER if wparam == TIMER_HOLD => {
            if context.holds.borrow().should_confirm(Instant::now()) {
                context.capture.confirm_hold();
                context.set_status("正在录音".into());
                KillTimer(hwnd, TIMER_HOLD);
            }
            Some(0)
        }
        WM_TIMER if wparam == TIMER_HEALTH => {
            if let Some(error) = context.capture.take_error() {
                context.set_status(error);
            }
            if !context.added.get() && context.retries.get() < 10 {
                context.retries.set(context.retries.get() + 1);
                context.add_icon();
            }
            Some(0)
        }
        WM_TIMER if wparam == TIMER_SHUTDOWN => {
            if !context.busy.load(Ordering::Acquire) {
                PostQuitMessage(0);
            }
            Some(0)
        }
        WM_REPLAY => {
            if !context.closing.get() {
                let key = if wparam == 0 {
                    Shortcut::CapsLock
                } else {
                    Shortcut::X2
                };
                if let Err(error) = input::replay(key) {
                    context.set_status(error.to_string());
                }
            }
            Some(0)
        }
        WM_OUTPUT => {
            let output = Box::from_raw(lparam as *mut DesktopOutput);
            match *output {
                DesktopOutput::State(text) | DesktopOutput::Error(text) => context.set_status(text),
                DesktopOutput::Text(_) => {}
                DesktopOutput::DownloadFinished { ok, message } => {
                    context.downloading.store(false, Ordering::Release);
                    if ok {
                        context.models_missing.set(false);
                    }
                    context.set_status(message);
                }
            }
            Some(0)
        }
        WM_CLOSE => {
            context.start_shutdown();
            Some(0)
        }
        WM_DESTROY => {
            PostQuitMessage(0);
            Some(0)
        }
        _ => None,
    }
}

unsafe extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    let pointer = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const Context;
    if !pointer.is_null() {
        if let Ok(Some(result)) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            dispatch(&*pointer, hwnd, message, wparam, lparam)
        })) {
            return result;
        }
    }
    DefWindowProcW(hwnd, message, wparam, lparam)
}

fn post_output(hwnd: usize, output: DesktopOutput) {
    let pointer = Box::into_raw(Box::new(output));
    if unsafe { PostMessageW(hwnd as HWND, WM_OUTPUT, 0, pointer as LPARAM) } == 0 {
        unsafe {
            drop(Box::from_raw(pointer));
        }
    }
}

pub(super) fn run(
    options: DesktopOptions,
    events: BudgetEventSender,
    output: Receiver<DesktopOutput>,
) -> Result<()> {
    // No API below runs in unit tests. Construction is explicit and opt-in.
    unsafe {
        let capture = if options.enable_capture {
            Capture::open(&options.portaudio_dll, events.clone())?
        } else {
            Capture::disabled(events.clone())
        };
        let instance = GetModuleHandleW(ptr::null());
        let icons = ApplicationIcons::load(instance)?;
        let class_name = wide(&format!("YuYbUuRuTray_{}", std::process::id()));
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance,
            hIcon: icons.large,
            hIconSm: icons.small,
            lpszClassName: class_name.as_ptr(),
            ..WNDCLASSEXW::default()
        };
        if RegisterClassExW(&class) == 0 {
            bail!("RegisterClassExW: {}", std::io::Error::last_os_error());
        }
        let taskbar_created = RegisterWindowMessageW(wide("TaskbarCreated").as_ptr());
        let alive = Arc::new(AtomicBool::new(true));
        let busy = Arc::new(AtomicBool::new(false));
        let models_missing = options.models_missing;
        let context = Box::new(Context {
            options,
            events,
            capture,
            holds: RefCell::default(),
            hwnd: Cell::new(ptr::null_mut()),
            icons,
            status: RefCell::new(if models_missing {
                "缺少语音模型".to_owned()
            } else {
                "待命".to_owned()
            }),
            taskbar_created,
            added: Cell::new(false),
            retries: Cell::new(0),
            closing: Cell::new(false),
            alive: Arc::clone(&alive),
            busy: Arc::clone(&busy),
            models_missing: Cell::new(models_missing),
            model_notice_shown: Cell::new(false),
            downloading: Arc::new(AtomicBool::new(false)),
        });
        let hwnd = CreateWindowExW(
            0,
            class_name.as_ptr(),
            wide("语音输入").as_ptr(),
            0,
            0,
            0,
            0,
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            instance,
            ptr::null(),
        );
        if hwnd.is_null() {
            UnregisterClassW(class_name.as_ptr(), instance);
            bail!("CreateWindowExW: {}", std::io::Error::last_os_error());
        }
        context.hwnd.set(hwnd);
        SendMessageW(
            hwnd,
            WM_SETICON,
            ICON_BIG as WPARAM,
            context.icons.large as LPARAM,
        );
        SendMessageW(
            hwnd,
            WM_SETICON,
            ICON_SMALL as WPARAM,
            context.icons.small as LPARAM,
        );
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, (&*context as *const Context) as isize);
        if taskbar_created != 0 {
            ChangeWindowMessageFilterEx(hwnd, taskbar_created, MSGFLT_ALLOW, ptr::null_mut());
        }
        context.add_icon();
        SetTimer(hwnd, TIMER_HEALTH, 1_000, None);
        let mut keyboard = ptr::null_mut();
        let mut mouse = ptr::null_mut();
        let hooks_ok = if context.options.enable_hooks {
            HOOK_CONTEXT.with(|slot| slot.set(&*context));
            keyboard = SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_hook), instance, 0);
            mouse = SetWindowsHookExW(WH_MOUSE_LL, Some(mouse_hook), instance, 0);
            !keyboard.is_null() && !mouse.is_null()
        } else {
            true
        };
        let worker_hwnd = hwnd as usize;
        let restore = context.options.restore_clipboard;
        let worker_alive = Arc::clone(&alive);
        let worker_busy = Arc::clone(&busy);
        let output_worker = thread::Builder::new()
            .name("desktop-output".into())
            .spawn(move || {
                while let Ok(message) = output.recv() {
                    if !worker_alive.load(Ordering::Acquire) {
                        break;
                    }
                    match message {
                        DesktopOutput::Text(text) => {
                            worker_busy.store(true, Ordering::Release);
                            if worker_alive.load(Ordering::Acquire) {
                                if let Err(error) = input::output_text(
                                    worker_hwnd as HWND,
                                    &text,
                                    restore,
                                    &worker_alive,
                                ) {
                                    post_output(
                                        worker_hwnd,
                                        DesktopOutput::Error(error.to_string()),
                                    );
                                }
                            }
                            worker_busy.store(false, Ordering::Release);
                        }
                        message => post_output(worker_hwnd, message),
                    }
                }
            });
        let mut outcome = Ok(());
        #[cfg(test)]
        observe_passive_smoke(PassiveSmokeEvent::Ready {
            hwnd: hwnd as usize,
            thread_id: windows_sys::Win32::System::Threading::GetCurrentThreadId(),
            tray_added: context.added.get(),
            capture_enabled: context.options.enable_capture,
            hooks_enabled: context.options.enable_hooks,
        });
        if !hooks_ok {
            outcome = Err(anyhow::anyhow!(
                "Unable to install CapsLock/X2 hooks: {}",
                std::io::Error::last_os_error()
            ));
        } else if let Err(error) = &output_worker {
            outcome = Err(anyhow::anyhow!("Unable to create output worker: {error}"));
        } else {
            let mut message = MSG::default();
            loop {
                let status = GetMessageW(&mut message, ptr::null_mut(), 0, 0);
                if status == 0 {
                    break;
                }
                if status == -1 {
                    outcome = Err(anyhow::anyhow!(
                        "GetMessageW: {}",
                        std::io::Error::last_os_error()
                    ));
                    break;
                }
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }
        alive.store(false, Ordering::Release);
        context.capture.cancel_press();
        if !keyboard.is_null() {
            UnhookWindowsHookEx(keyboard);
        }
        if !mouse.is_null() {
            UnhookWindowsHookEx(mouse);
        }
        HOOK_CONTEXT.with(|slot| slot.set(ptr::null()));
        for id in [TIMER_HOLD, TIMER_HEALTH, TIMER_SHUTDOWN] {
            KillTimer(hwnd, id);
        }
        if context.added.get() {
            Shell_NotifyIconW(NIM_DELETE, &mut context.notification());
        }
        // All queued output boxes are ours; reclaim those preceding WM_QUIT.
        let mut pending = MSG::default();
        while PeekMessageW(&mut pending, hwnd, WM_OUTPUT, WM_OUTPUT, PM_REMOVE) != 0 {
            drop(Box::from_raw(pending.lParam as *mut DesktopOutput));
        }
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
        DestroyWindow(hwnd);
        UnregisterClassW(class_name.as_ptr(), instance);
        if let Ok(worker) = output_worker {
            if worker.is_finished() {
                let _ = worker.join();
            }
        }
        outcome
    }
}

#[cfg(test)]
pub(super) fn test_install_uninstall_hooks() -> Result<()> {
    if HOOK_CONTEXT.with(|slot| !slot.get().is_null()) {
        bail!("Hook smoke requires a thread without an active capture context");
    }
    struct Hooks {
        keyboard: HHOOK,
        mouse: HHOOK,
    }
    impl Drop for Hooks {
        fn drop(&mut self) {
            unsafe {
                if !self.keyboard.is_null() {
                    UnhookWindowsHookEx(self.keyboard);
                }
                if !self.mouse.is_null() {
                    UnhookWindowsHookEx(self.mouse);
                }
            }
        }
    }
    unsafe {
        let instance = GetModuleHandleW(ptr::null());
        let mut hooks = Hooks {
            keyboard: ptr::null_mut(),
            mouse: ptr::null_mut(),
        };
        hooks.keyboard = SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_hook), instance, 0);
        if hooks.keyboard.is_null() {
            bail!(
                "Keyboard hook installation failed: {}",
                std::io::Error::last_os_error()
            );
        }
        hooks.mouse = SetWindowsHookExW(WH_MOUSE_LL, Some(mouse_hook), instance, 0);
        if hooks.mouse.is_null() {
            bail!(
                "Mouse hook installation failed: {}",
                std::io::Error::last_os_error()
            );
        }
        if UnhookWindowsHookEx(hooks.mouse) == 0 {
            bail!("Mouse hook removal failed");
        }
        hooks.mouse = ptr::null_mut();
        if UnhookWindowsHookEx(hooks.keyboard) == 0 {
            bail!("Keyboard hook removal failed");
        }
        hooks.keyboard = ptr::null_mut();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    struct OwnWindowGuard {
        hwnd: usize,
        thread: u32,
    }
    impl Drop for OwnWindowGuard {
        fn drop(&mut self) {
            unsafe {
                if self.hwnd != 0
                    && GetWindowThreadProcessId(self.hwnd as HWND, ptr::null_mut()) == self.thread
                {
                    PostMessageW(self.hwnd as HWND, WM_CLOSE, 0, 0);
                }
            }
        }
    }
    #[test]
    fn tooltip_truncation_never_leaves_a_surrogate_half() {
        let text = format!("{}😀", "中".repeat(126));
        let units = tip(&text);
        let end = units.iter().position(|unit| *unit == 0).unwrap();
        assert_eq!(end, 126);
        assert!(String::from_utf16(&units[..end]).is_ok());
    }

    #[test]
    #[ignore = "Coordinated passive tray smoke only; capture/hooks/SendInput are disabled"]
    fn native_passive_desktop_smoke() {
        use std::{sync::mpsc, time::Duration};
        let (observe, observed) = mpsc::channel();
        let (events, received_events) = crate::desktop::event_channel(1024 * 1024);
        let (output, received_output) = mpsc::channel();
        let native = thread::spawn(move || {
            PASSIVE_SMOKE.with(|slot| *slot.borrow_mut() = Some(observe));
            let mut options = DesktopOptions::new(env!("CARGO_MANIFEST_DIR"));
            options.enable_hooks = false;
            options.enable_capture = false;
            let result = run(options, events, received_output);
            PASSIVE_SMOKE.with(|slot| {
                slot.borrow_mut().take();
            });
            result
        });
        let ready = observed
            .recv_timeout(Duration::from_secs(5))
            .expect("Own tray window did not start");
        let PassiveSmokeEvent::Ready {
            hwnd,
            thread_id,
            tray_added,
            capture_enabled,
            hooks_enabled,
        } = ready
        else {
            panic!("Unexpected first smoke event");
        };
        let mut window = OwnWindowGuard {
            hwnd,
            thread: thread_id,
        };
        assert!(!capture_enabled && !hooks_enabled);
        // The HWND is passed directly from this test's UI thread. Never discover
        // or close another app instance by process name or FindWindow.
        let owner_thread = unsafe { GetWindowThreadProcessId(hwnd as HWND, ptr::null_mut()) };
        assert_eq!(owner_thread, thread_id);
        assert!(tray_added, "Shell_NotifyIconW did not add our passive icon");
        output
            .send(DesktopOutput::State("被动桌面烟测 · 状态更新成功".into()))
            .unwrap();
        let changed = observed
            .recv_timeout(Duration::from_secs(5))
            .expect("State message was not processed");
        let PassiveSmokeEvent::State {
            text,
            icon_modified,
        } = changed
        else {
            panic!("Missing State update");
        };
        assert_eq!(text, "被动桌面烟测 · 状态更新成功");
        assert!(
            icon_modified,
            "Shell_NotifyIconW did not modify our tooltip"
        );
        assert_ne!(unsafe { PostMessageW(hwnd as HWND, WM_CLOSE, 0, 0) }, 0);
        window.hwnd = 0;
        assert!(matches!(
            received_events
                .recv_timeout(Duration::from_secs(5))
                .unwrap(),
            DesktopEvent::Quit
        ));
        drop(output);
        native.join().unwrap().unwrap();
        assert_eq!(unsafe { IsWindow(hwnd as HWND) }, 0);
        assert!(
            received_events.try_recv().is_err(),
            "Passive test emitted an unexpected active event"
        );
        let report = serde_json::json!({"own_hwnd": hwnd, "own_thread_id": thread_id,
            "tray_icon_added": true, "state_message_processed": true, "tooltip_modified": true,
            "closed_by_own_wm_close": true, "own_window_destroyed": true, "quit_event_received": true,
            "capture_enabled": false, "hooks_enabled": false, "sendinput_called": false,
            "worker_or_ffmpeg_started": false, "network_port_opened": false});
        let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("evidence/desktop-passive-smoke.json");
        std::fs::write(file, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        println!("{report}");
    }

    #[test]
    #[ignore = "Explicit passive application lifecycle smoke; no models/devices/hooks/Text"]
    fn application_passive_lifecycle_smoke() {
        use std::{
            sync::mpsc,
            time::{Duration, Instant},
        };
        let (observe, observed) = mpsc::channel();
        let (done, finished) = mpsc::channel();
        let started = Instant::now();
        let app = thread::spawn(move || {
            PASSIVE_SMOKE.with(|slot| *slot.borrow_mut() = Some(observe));
            let settings = crate::config::Settings {
                app_dir: env!("CARGO_MANIFEST_DIR").into(),
                ..Default::default()
            };
            let result = crate::application::desktop(settings, None, true);
            PASSIVE_SMOKE.with(|slot| {
                slot.borrow_mut().take();
            });
            let _ = done.send(result.map_err(|error| format!("{error:#}")));
        });
        let ready = observed
            .recv_timeout(Duration::from_secs(5))
            .expect("Passive application did not create its own tray");
        let PassiveSmokeEvent::Ready {
            hwnd,
            thread_id,
            tray_added,
            capture_enabled,
            hooks_enabled,
        } = ready
        else {
            panic!("Unexpected passive application event");
        };
        let mut window = OwnWindowGuard {
            hwnd,
            thread: thread_id,
        };
        assert!(tray_added);
        assert!(!capture_enabled && !hooks_enabled);
        assert_eq!(
            unsafe { GetWindowThreadProcessId(hwnd as HWND, ptr::null_mut()) },
            thread_id
        );
        assert_ne!(unsafe { PostMessageW(hwnd as HWND, WM_CLOSE, 0, 0) }, 0);
        window.hwnd = 0;
        finished
            .recv_timeout(Duration::from_secs(5))
            .expect("Passive application failed to join its dispatch/file/model threads")
            .unwrap();
        app.join().unwrap();
        assert_eq!(unsafe { IsWindow(hwnd as HWND) }, 0);
        let report = serde_json::json!({"application_desktop_passive": true, "own_hwnd": hwnd,
            "own_thread_id": thread_id, "tray_icon_added": true, "closed_by_own_wm_close": true,
            "application_returned_cleanly": true, "lifecycle_seconds": started.elapsed().as_secs_f64(),
            "capture_enabled": false, "hooks_enabled": false, "text_output_sent": false,
            "models_loaded": false, "ffmpeg_started": false, "network_port_opened": false});
        let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("evidence/application-passive-smoke.json");
        std::fs::write(file, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        println!("{report}");
    }
}
