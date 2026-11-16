#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::{io::Write, path::PathBuf};
use yuybuuru::{
    application, automation,
    config::{AsrModel, Settings},
    engine::{Engine, EngineOptions},
    media, paths::AppPaths, worker,
};

#[derive(Parser)]
#[command(
    version,
    about = "本机离线听写、热词修正与音视频转录"
)]
struct Cli {
    #[arg(long, global = true)]
    app_dir: Option<PathBuf>,
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    /// Select the packaged ASR model; overrides the JSON setting.
    #[arg(long, global = true, value_enum)]
    asr_model: Option<AsrModel>,
    #[arg(long, global = true)]
    seed: Option<u32>,
    #[arg(long, global = true)]
    asr_idle_seconds: Option<u64>,
    #[arg(long, global = true)]
    aligner_idle_seconds: Option<u64>,
    #[command(subcommand)]
    command: Option<Mode>,
}
#[derive(Subcommand)]
enum Mode {
    /// 供本机自动化工具使用的 JSON 接口，不显示新窗口。
    Automation {
        #[command(subcommand)]
        command: automation::Command,
    },
    Desktop {
        #[arg(long)]
        passive: bool,
    },
    Transcribe {
        #[arg(required = true)]
        files: Vec<PathBuf>,
        #[arg(long)]
        output_dir: Option<PathBuf>,
        #[arg(long)]
        overwrite: bool,
    },
    /// Internal local protocol endpoint; no TCP listener.
    Serve,
    #[command(hide = true)]
    Worker {
        #[arg(long)]
        settings_json: String,
    },
    /// Offline native diagnostic, bypassing the desktop and input hooks.
    Infer {
        file: PathBuf,
        #[arg(long)]
        native_seed: Option<u32>,
        #[arg(long, default_value = "auto")]
        language: String,
        #[arg(long, default_value = "")]
        context: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}
impl Cli {
    fn apply_overrides(&self, settings: &mut Settings) {
        if let Some(value) = self.asr_model {
            settings.asr_model = value;
        }
        if let Some(value) = self.asr_idle_seconds {
            settings.asr_idle_seconds = value;
        }
        if let Some(value) = self.aligner_idle_seconds {
            settings.aligner_idle_seconds = value;
        }
    }
}
fn main() {
    let automation_requested = std::env::args_os().any(|value| value == "automation");
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error)
            if automation_requested
                && matches!(
                    error.kind(),
                    clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
                ) =>
        {
            let result = serde_json::json!({
                "schema": 1, "ok": true, "action": "help",
                "data": {"help": error.to_string()}
            });
            let _ = write_json(&result);
            return;
        }
        Err(error) if automation_requested => {
            let result = automation::failure(
                "arguments",
                "invalid_arguments",
                "自动化命令参数无效",
                &error.to_string(),
            );
            let _ = write_json(&result);
            std::process::exit(2);
        }
        Err(error) => error.exit(),
    };
    let action = match &cli.command {
        Some(Mode::Automation { command }) => Some(command.name()),
        _ => None,
    };
    if let Err(error) = run(cli) {
        if let Some(action) = action {
            let result = automation::failure(
                action,
                "operation_failed",
                "自动化操作失败",
                &format!("{error:#}"),
            );
            let _ = write_json(&result);
        } else {
            eprintln!("{error:#}");
        }
        std::process::exit(1);
    }
}
fn write_json(value: &serde_json::Value) -> Result<()> {
    let mut output = std::io::stdout().lock();
    serde_json::to_writer(&mut output, value)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}
fn run(cli: Cli) -> Result<()> {
    if let Some(Mode::Automation { command }) = &cli.command {
        let config = cli
            .config
            .clone()
            .unwrap_or(
                AppPaths::new(
                    std::env::current_exe()?.parent().context("Executable directory")?,
                ).settings(),
            );
        let config = if config.is_absolute() {
            config
        } else {
            std::env::current_dir()?.join(config)
        };
        let response =
            automation::execute(command, &config, cli.app_dir.as_deref(), cli.asr_model)?;
        return write_json(&response);
    }
    if let Some(Mode::Worker { settings_json }) = &cli.command {
        let mut settings = serde_json::from_str(settings_json)?;
        cli.apply_overrides(&mut settings);
        return worker::run(settings);
    }
    let default_config = std::env::current_exe()?
        .parent()
        .map(|dir| AppPaths::new(dir).settings());
    let config_path = cli
        .config
        .as_deref()
        .or_else(|| default_config.as_deref().filter(|path| path.is_file()));
    let mut settings = Settings::load(config_path, cli.app_dir.as_deref())?;
    cli.apply_overrides(&mut settings);
    match cli.command.unwrap_or(Mode::Desktop { passive: false }) {
        Mode::Automation { .. } => unreachable!(),
        Mode::Desktop { passive } => application::desktop(settings, cli.seed, passive),
        Mode::Serve => worker::serve(settings),
        Mode::Worker { .. } => unreachable!(),
        Mode::Transcribe {
            files,
            output_dir,
            overwrite,
        } => {
            let service = worker::WorkerService::start(settings.clone())?;
            let client = application::RuntimeClient::new(service.client(), settings, cli.seed)?;
            for file in files {
                let (result, outputs) =
                    client.transcribe(&file, output_dir.as_deref(), overwrite)?;
                println!(
                    "{}",
                    serde_json::to_string(
                        &serde_json::json!({"file":file,"text":result.text_accu,"outputs":outputs})
                    )?
                );
            }
            drop(service);
            Ok(())
        }
        Mode::Infer {
            file,
            native_seed,
            language,
            context,
            output,
        } => {
            let mut options: EngineOptions = application_options(&settings);
            options.seed = native_seed;
            let audio = media::read_audio(&settings.app_dir, &file)?;
            let mut engine = Engine::load(&settings.app_dir, &options)?;
            let result = engine.transcribe(&audio, &language, &context)?;
            let data = serde_json::to_vec_pretty(&result)?;
            if let Some(file) = output {
                std::fs::write(file, data)?;
            } else {
                std::io::stdout().write_all(&data)?;
            }
            Ok(())
        }
    }
}
fn application_options(settings: &Settings) -> EngineOptions {
    yuybuuru::pipeline::engine_options(settings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automation_config_accepts_global_model_and_explicit_boolean_values() {
        let cli = Cli::try_parse_from([
            "语音输入",
            "automation",
            "config",
            "--asr-model",
            "1.7b",
            "--hotwords",
            "false",
            "--hotword-rules",
            "true",
        ])
        .unwrap();
        assert_eq!(cli.asr_model, Some(AsrModel::Qwen3Asr17B));
        match cli.command.unwrap() {
            Mode::Automation {
                command:
                    automation::Command::Config {
                        hotwords,
                        hotword_rules,
                    },
            } => {
                assert_eq!(hotwords, Some(false));
                assert_eq!(hotword_rules, Some(true));
            }
            _ => panic!("Expected automation configuration command"),
        }
        assert!(
            Cli::try_parse_from(["语音输入", "automation", "startup", "--enabled", "invalid"])
                .is_err()
        );
    }

    #[test]
    fn model_option_is_global_and_cli_overrides_config() {
        for args in [
            vec!["yuybuuru", "--asr-model", "1.7b", "serve"],
            vec!["yuybuuru", "serve", "--asr-model", "1.7b"],
            vec!["yuybuuru", "transcribe", "audio.wav", "--asr-model", "1.7b"],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            let mut settings: Settings = serde_json::from_str(r#"{"asr_model":"0.6b"}"#).unwrap();
            cli.apply_overrides(&mut settings);
            assert_eq!(settings.asr_model, AsrModel::Qwen3Asr17B);
        }
        let cli = Cli::try_parse_from(["yuybuuru", "serve", "--asr-model", "0.6b"]).unwrap();
        let mut settings: Settings = serde_json::from_str(r#"{"asr_model":"1.7b"}"#).unwrap();
        cli.apply_overrides(&mut settings);
        assert_eq!(settings.asr_model, AsrModel::Qwen3Asr06B);
    }

    #[test]
    fn omitted_model_keeps_config_and_unknown_names_are_rejected() {
        let cli = Cli::try_parse_from(["yuybuuru", "serve"]).unwrap();
        let mut settings: Settings = serde_json::from_str(r#"{"asr_model":"1.7b"}"#).unwrap();
        cli.apply_overrides(&mut settings);
        assert_eq!(settings.asr_model, AsrModel::Qwen3Asr17B);
        assert!(Cli::try_parse_from(["yuybuuru", "--asr-model", "unknown", "serve"]).is_err());
        assert!(Cli::try_parse_from(["yuybuuru", "--asr-model", "1.7B"]).is_err());
    }
}
