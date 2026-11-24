//! Optional Silero VAD. Disabled unless settings.vad is true.
//! Gating is independent of the recognizer and does not alter text for clips that are kept,
//! other than removing leading and trailing silence.
use crate::{config::Settings, engine::ensure_onnx_runtime, paths::AppPaths};
use anyhow::{Context, Result, ensure};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};

const WINDOW: usize = 512;
const CONTEXT: usize = 64;
const STATE_LEN: usize = 2 * 1 * 128;
const RATE: u64 = 16_000;

pub enum Decision {
    Drop,
    Keep(Vec<f32>),
}

pub struct Silero {
    session: Session,
}

impl Silero {
    pub fn load(dll: &std::path::Path, model: &std::path::Path) -> Result<Self> {
        ensure_onnx_runtime(dll)?;
        ensure!(
            model.is_file(),
            "未找到 Silero VAD 模型：{}",
            model.display()
        );
        let session = Session::builder()
            .map_err(|error| anyhow::anyhow!("{error}"))?
            .with_intra_threads(1)
            .map_err(|error| anyhow::anyhow!("{error}"))?
            .with_inter_threads(1)
            .map_err(|error| anyhow::anyhow!("{error}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| anyhow::anyhow!("{error}"))?
            .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])
            .map_err(|error| anyhow::anyhow!("{error}"))?
            .commit_from_file(model)
            .with_context(|| format!("加载 Silero VAD 失败：{}", model.display()))?;
        Ok(Self { session })
    }

    pub fn probabilities(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            audio.iter().all(|sample| sample.is_finite()),
            "音频包含非有限采样"
        );
        let mut state = vec![0f32; STATE_LEN];
        let mut context = vec![0f32; CONTEXT];
        let mut probs = Vec::new();
        let mut offset = 0;
        while offset < audio.len() {
            let mut window = vec![0f32; WINDOW];
            let n = (audio.len() - offset).min(WINDOW);
            window[..n].copy_from_slice(&audio[offset..offset + n]);
            let mut input = Vec::with_capacity(CONTEXT + WINDOW);
            input.extend_from_slice(&context);
            input.extend_from_slice(&window);
            let input_tensor = Tensor::from_array(([1usize, input.len()], input.clone()))?;
            let state_tensor = Tensor::from_array(([2usize, 1, 128], state.clone()))?;
            let sample_rate = Tensor::from_array((Vec::<i64>::new(), vec![RATE as i64]))?;
            let outputs = self.session.run(ort::inputs![
                "input" => input_tensor,
                "state" => state_tensor,
                "sr" => sample_rate
            ])?;
            let (_, probability) = outputs
                .get("output")
                .context("Silero VAD 缺少 output")?
                .try_extract_tensor::<f32>()
                .map_err(|error| anyhow::anyhow!("{error}"))?;
            ensure!(probability.len() == 1, "Silero VAD 输出形状异常");
            let (_, next_state) = outputs
                .get("stateN")
                .context("Silero VAD 缺少 state")?
                .try_extract_tensor::<f32>()
                .map_err(|error| anyhow::anyhow!("{error}"))?;
            ensure!(next_state.len() == STATE_LEN, "Silero VAD 状态形状异常");
            state = next_state.to_vec();
            let context_start = input.len() - CONTEXT;
            context.copy_from_slice(&input[context_start..]);
            probs.push(probability[0]);
            offset += WINDOW;
        }
        Ok(probs)
    }
}

pub fn classify(slot: &mut Option<Silero>, settings: &Settings, audio: &[f32]) -> Result<Decision> {
    if audio.is_empty() {
        return Ok(Decision::Drop);
    }
    if slot.is_none() {
        let paths = AppPaths::new(&settings.app_dir);
        let dll = match settings.onnx_dll.as_ref() {
            Some(path) if path.is_absolute() => path.clone(),
            Some(path) => settings.app_dir.join(path),
            None => paths.onnx(),
        };
        *slot = Some(Silero::load(&dll, &paths.vad_model())?);
    }
    let model = slot.as_mut().context("VAD 未加载")?;
    let probs = model.probabilities(audio)?;
    Ok(trim_speech(
        audio,
        &probs,
        settings.vad_threshold,
        settings.vad_min_speech_ms,
        settings.vad_speech_pad_ms,
    ))
}

/// Drop clips whose detected speech is shorter than `min_speech_ms`.
/// Otherwise keep the span from the first speech window to the last, plus padding.
pub fn trim_speech(
    audio: &[f32],
    probs: &[f32],
    threshold: f64,
    min_speech_ms: u64,
    pad_ms: u64,
) -> Decision {
    if audio.is_empty() {
        return Decision::Drop;
    }
    let mut speech_samples = 0usize;
    let mut first = None;
    let mut last = 0usize;
    for (index, probability) in probs.iter().enumerate() {
        let start = index.saturating_mul(WINDOW);
        if start >= audio.len() {
            break;
        }
        if probability.is_finite() && f64::from(*probability) >= threshold {
            let end = (start + WINDOW).min(audio.len());
            speech_samples += end - start;
            if first.is_none() {
                first = Some(start);
            }
            last = end;
        }
    }
    let Some(first) = first else {
        return Decision::Drop;
    };
    let minimum = (RATE.saturating_mul(min_speech_ms) / 1000) as usize;
    if speech_samples < minimum {
        return Decision::Drop;
    }
    let pad = (RATE.saturating_mul(pad_ms) / 1000) as usize;
    let start = first.saturating_sub(pad);
    let end = last.saturating_add(pad).min(audio.len());
    if start >= end {
        return Decision::Drop;
    }
    Decision::Keep(audio[start..end].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speech_pattern(windows: &[(usize, f32)], total_windows: usize) -> Vec<f32> {
        let mut probs = vec![0.0; total_windows];
        for (index, value) in windows {
            probs[*index] = *value;
        }
        probs
    }

    #[test]
    fn silence_and_short_noise_are_dropped() {
        let audio = vec![0.0; WINDOW * 40];
        assert!(matches!(
            trim_speech(&audio, &vec![0.0; 40], 0.5, 250, 30),
            Decision::Drop
        ));
        let click = speech_pattern(&[(3, 0.95)], 40);
        assert!(matches!(
            trim_speech(&audio, &click, 0.5, 250, 30),
            Decision::Drop
        ));
        assert!(matches!(
            trim_speech(&[], &[], 0.5, 250, 30),
            Decision::Drop
        ));
    }

    #[test]
    fn leading_and_trailing_silence_is_trimmed_with_pad() {
        let windows = 40;
        let audio: Vec<f32> = (0..windows * WINDOW).map(|i| i as f32).collect();
        let mut probs = vec![0.1; windows];
        for item in probs.iter_mut().take(30).skip(10) {
            *item = 0.9;
        }
        let Decision::Keep(trimmed) = trim_speech(&audio, &probs, 0.5, 250, 30) else {
            panic!("speech should be kept");
        };
        let pad = 16_000 * 30 / 1000;
        let start = 10 * WINDOW - pad;
        let end = 30 * WINDOW + pad;
        assert_eq!(trimmed, audio[start..end]);
        // Twenty 32 ms windows exceed the 250 ms minimum.
        assert!(20 * WINDOW > 16_000 * 250 / 1000);
    }

    #[test]
    fn threshold_is_inclusive_and_pad_does_not_leave_the_clip() {
        let audio = vec![1.0; WINDOW * 4];
        let probs = vec![0.5, 0.5, 0.5, 0.5];
        let Decision::Keep(trimmed) = trim_speech(&audio, &probs, 0.5, 0, 10_000) else {
            panic!("threshold match should count");
        };
        assert_eq!(trimmed, audio);
        let edge = vec![0.49, 0.5];
        let audio = vec![1.0; WINDOW * 2];
        let Decision::Keep(trimmed) = trim_speech(&audio, &edge, 0.5, 0, 0) else {
            panic!("second window is speech");
        };
        assert_eq!(trimmed, audio[WINDOW..]);
    }
}
