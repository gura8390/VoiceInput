//! Optional Silero VAD running inside the worker process. Disabled unless settings.vad is true.
//! Gating is independent of the recognizer and does not alter text for clips that are kept,
//! other than removing leading and trailing silence.
use crate::engine::ensure_onnx_runtime;
use anyhow::{Context, Result, ensure};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::{Tensor, TensorRef},
};

const WINDOW: usize = 512;
const CONTEXT: usize = 64;
const STATE_LEN: usize = 2 * 1 * 128;
const RATE: u64 = 16_000;

#[derive(Debug, PartialEq)]
pub enum Decision {
    Drop,
    Keep(Vec<f32>),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeechSpan {
    pub speech_samples: usize,
    pub first: Option<usize>,
    pub last: usize,
    pub total_samples: usize,
}

impl SpeechSpan {
    /// Resolve trimmed `[start, end)` sample indices for an audio buffer of `total_samples`.
    pub fn resolve_bounds(self, min_speech_ms: u64, pad_ms: u64) -> Option<(usize, usize)> {
        let first = self.first?;
        if self.total_samples == 0 {
            return None;
        }
        let minimum = (RATE.saturating_mul(min_speech_ms) / 1000) as usize;
        if self.speech_samples < minimum {
            return None;
        }
        let pad = (RATE.saturating_mul(pad_ms) / 1000) as usize;
        let start = first.saturating_sub(pad);
        let end = self.last.saturating_add(pad).min(self.total_samples);
        (start < end).then_some((start, end))
    }
}

/// Stateful streaming accumulator across chunked worker VAD calls.
pub struct VadSession {
    state: Vec<f32>,
    context: [f32; CONTEXT],
    tail: Vec<f32>,
    processed_samples: usize,
    speech_samples: usize,
    first: Option<usize>,
    last: usize,
}

impl Default for VadSession {
    fn default() -> Self {
        Self::new()
    }
}

impl VadSession {
    pub fn new() -> Self {
        Self {
            state: vec![0f32; STATE_LEN],
            context: [0f32; CONTEXT],
            tail: Vec::new(),
            processed_samples: 0,
            speech_samples: 0,
            first: None,
            last: 0,
        }
    }

    pub fn record_window(&mut self, start: usize, len: usize, probability: f32, threshold: f64) {
        if probability.is_finite() && f64::from(probability) >= threshold {
            self.speech_samples += len;
            if self.first.is_none() {
                self.first = Some(start);
            }
            self.last = start + len;
        }
    }

    pub fn summary(&self) -> SpeechSpan {
        SpeechSpan {
            speech_samples: self.speech_samples,
            first: self.first,
            last: self.last,
            total_samples: self.processed_samples,
        }
    }
}

pub struct Silero {
    session: Session,
    sample_rate: Tensor<i64>,
    input_buf: Vec<f32>,
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
        let sample_rate = Tensor::from_array((Vec::<i64>::new(), vec![RATE as i64]))?;
        Ok(Self {
            session,
            sample_rate,
            input_buf: vec![0f32; CONTEXT + WINDOW],
        })
    }

    fn run_window(
        &mut self,
        window: &[f32],
        state: &mut [f32],
        context: &mut [f32; CONTEXT],
    ) -> Result<f32> {
        self.input_buf[..CONTEXT].copy_from_slice(context);
        let n = window.len().min(WINDOW);
        self.input_buf[CONTEXT..CONTEXT + n].copy_from_slice(&window[..n]);
        if n < WINDOW {
            self.input_buf[CONTEXT + n..CONTEXT + WINDOW].fill(0.0);
        }
        let input_ref = TensorRef::from_array_view(([1usize, CONTEXT + WINDOW], &self.input_buf[..]))?;
        let state_ref = TensorRef::from_array_view(([2usize, 1, 128], &*state))?;
        let outputs = self.session.run(ort::inputs![
            "input" => input_ref,
            "state" => state_ref,
            "sr" => &self.sample_rate
        ])?;
        let (_, probability) = outputs
            .get("output")
            .context("Silero VAD 缺少 output")?
            .try_extract_tensor::<f32>()
            .map_err(|error| anyhow::anyhow!("{error}"))?;
        ensure!(probability.len() == 1, "Silero VAD 输出形状异常");
        let prob = probability[0];
        let (_, next_state) = outputs
            .get("stateN")
            .context("Silero VAD 缺少 state")?
            .try_extract_tensor::<f32>()
            .map_err(|error| anyhow::anyhow!("{error}"))?;
        ensure!(next_state.len() == STATE_LEN, "Silero VAD 状态形状异常");
        state.copy_from_slice(next_state);
        context.copy_from_slice(&self.input_buf[WINDOW..CONTEXT + WINDOW]);
        Ok(prob)
    }

    pub fn probabilities(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            audio.iter().all(|sample| sample.is_finite()),
            "音频包含非有限采样"
        );
        let mut state = vec![0f32; STATE_LEN];
        let mut context = [0f32; CONTEXT];
        let mut probs = Vec::with_capacity(audio.len().div_ceil(WINDOW));
        let mut offset = 0;
        while offset < audio.len() {
            let n = (audio.len() - offset).min(WINDOW);
            let prob = self.run_window(&audio[offset..offset + n], &mut state, &mut context)?;
            probs.push(prob);
            offset += WINDOW;
        }
        Ok(probs)
    }

    /// Feed a chunk into a streaming VAD session, preserving Silero RNN state and
    /// unaligned tail samples across chunk boundaries.
    pub fn scan_chunk(
        &mut self,
        session: &mut VadSession,
        chunk: &[f32],
        threshold: f64,
        is_final: bool,
    ) -> Result<SpeechSpan> {
        ensure!(
            chunk.iter().all(|sample| sample.is_finite()),
            "音频包含非有限采样"
        );
        let mut combined_buf;
        let data: &[f32] = if session.tail.is_empty() {
            chunk
        } else {
            combined_buf = std::mem::take(&mut session.tail);
            combined_buf.extend_from_slice(chunk);
            &combined_buf
        };
        let mut offset = 0;
        while offset + WINDOW <= data.len() {
            let prob = self.run_window(
                &data[offset..offset + WINDOW],
                &mut session.state,
                &mut session.context,
            )?;
            let start = session.processed_samples;
            session.record_window(start, WINDOW, prob, threshold);
            session.processed_samples += WINDOW;
            offset += WINDOW;
        }
        let remaining = &data[offset..];
        if is_final {
            if !remaining.is_empty() {
                let prob =
                    self.run_window(remaining, &mut session.state, &mut session.context)?;
                let start = session.processed_samples;
                session.record_window(start, remaining.len(), prob, threshold);
                session.processed_samples += remaining.len();
            }
        } else if !remaining.is_empty() {
            session.tail.extend_from_slice(remaining);
        }
        Ok(session.summary())
    }
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
    let mut session = VadSession::new();
    for (index, probability) in probs.iter().enumerate() {
        let start = index.saturating_mul(WINDOW);
        if start >= audio.len() {
            break;
        }
        let len = (audio.len() - start).min(WINDOW);
        session.record_window(start, len, *probability, threshold);
    }
    session.processed_samples = audio.len();
    match session.summary().resolve_bounds(min_speech_ms, pad_ms) {
        Some((start, end)) => Decision::Keep(audio[start..end].to_vec()),
        None => Decision::Drop,
    }
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

    #[test]
    fn speech_span_bounds_match_trim_speech() {
        let span = SpeechSpan {
            speech_samples: 20 * WINDOW,
            first: Some(10 * WINDOW),
            last: 30 * WINDOW,
            total_samples: 40 * WINDOW,
        };
        let pad = 16_000 * 30 / 1000;
        assert_eq!(
            span.resolve_bounds(250, 30),
            Some((10 * WINDOW - pad, 30 * WINDOW + pad))
        );
        assert_eq!(span.resolve_bounds(1000, 30), None);
    }
}
