//! Regression coverage for the config/runtime layout migration.
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
use yuybuuru::{application::TextProcessor, config::Settings, paths::AppPaths};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct TestApp(PathBuf);
impl TestApp {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "yyur-layout-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(root.join("config")).unwrap();
        Self(root.canonicalize().unwrap())
    }
}
impl Drop for TestApp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn relocated_config_resolves_models_at_app_root_and_honors_override() {
    let app = TestApp::new();
    let other = TestApp::new();
    let paths = AppPaths::new(&app.0);
    fs::write(paths.settings(), br#"{"app_dir":"..","asr_model":"1.7b"}"#).unwrap();
    let settings = Settings::load(Some(&paths.settings()), None).unwrap();
    assert_eq!(settings.app_dir, app.0);
    let engine = yuybuuru::pipeline::engine_options(&settings);
    assert_eq!(
        engine.asr_model_directory(&settings.app_dir),
        app.0.join("models/Qwen3-ASR/Qwen3-ASR-1.7B")
    );
    let overridden = Settings::load(Some(&paths.settings()), Some(&other.0)).unwrap();
    assert_eq!(overridden.app_dir, other.0);
    // A manually selected config keeps its documented relative-path meaning.
    fs::write(paths.settings(), br#"{"app_dir":"."}"#).unwrap();
    assert_eq!(
        Settings::load(Some(&paths.settings()), None).unwrap().app_dir,
        paths.config()
    );
}

#[test]
fn correction_reads_and_reloads_config_rules_without_using_legacy_root_files() {
    let app = TestApp::new();
    let paths = AppPaths::new(&app.0);
    fs::write(app.0.join("hot-rule.txt"), "^placeholder$ = Legacy").unwrap();
    fs::write(paths.hotword_rules(), "^placeholder$ = Fresh").unwrap();
    let settings = Settings {
        app_dir: app.0.clone(),
        ..Default::default()
    };
    let mut processor = TextProcessor::new(&settings).unwrap();
    assert_eq!(processor.mic("placeholder", false).unwrap(), "Fresh");
    fs::write(paths.hotword_rules(), "^placeholder$ = Updated").unwrap();
    assert_eq!(processor.mic("placeholder", false).unwrap(), "Updated");
    fs::remove_file(paths.hotword_rules()).unwrap();
    assert_eq!(processor.mic("placeholder", false).unwrap(), "placeholder");
}

#[test]
fn fresh_application_reads_defaults_and_creates_config_without_legacy_fallback() {
    let app = TestApp::new();
    let override_app = TestApp::new();
    let paths = AppPaths::new(&app.0);
    fs::remove_dir(paths.config()).unwrap();
    let executable = app.0.join("语音输入.exe");
    fs::copy(env!("CARGO_BIN_EXE_yuybuuru"), &executable).unwrap();
    let legacy = b"deliberately invalid legacy settings: must never be read";
    fs::write(app.0.join("settings.json"), legacy).unwrap();

    let execute = |arguments: &[&str]| {
        let output = Command::new(&executable)
            .args(arguments)
            .current_dir(std::env::temp_dir())
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stdout));
        let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(value["ok"], true);
        value
    };
    let status = execute(&["automation", "status"]);
    assert_eq!(
        PathBuf::from(status["data"]["saved_config"]["app_dir"].as_str().unwrap()),
        app.0
    );
    assert!(!paths.config().exists(), "A read-only query must not create configuration");
    let overridden = execute(&[
        "--app-dir", override_app.0.to_str().unwrap(), "automation", "status",
    ]);
    assert_eq!(
        PathBuf::from(overridden["data"]["saved_config"]["app_dir"].as_str().unwrap()),
        override_app.0
    );
    assert!(!paths.config().exists());

    execute(&["automation", "config", "--hotwords", "false"]);
    let saved: serde_json::Value =
        serde_json::from_slice(&fs::read(paths.settings()).unwrap()).unwrap();
    assert_eq!(saved["app_dir"], "..");
    assert_eq!(saved["hotwords"], false);
    let loaded = Settings::load(Some(&paths.settings()), None).unwrap();
    assert_eq!(loaded.app_dir, app.0);
    assert!(!loaded.hotwords);
    assert_eq!(fs::read(app.0.join("settings.json")).unwrap(), legacy);
}
