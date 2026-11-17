// Appended only to the detached research worktree's encoder.rs by the staging
// script. Production settings, executable and model files remain unchanged.
type EncoderHandle = *mut std::ffi::c_void;
struct GgmlEncoderApi {
    _library: libloading::Library,
    load: unsafe extern "C" fn(*const std::ffi::c_char, *const std::ffi::c_char, i32, i32) -> EncoderHandle,
    dimension: unsafe extern "C" fn(EncoderHandle) -> i32,
    output_rows: unsafe extern "C" fn(usize) -> usize,
    encode: unsafe extern "C" fn(EncoderHandle, *const f32, usize, usize, *mut f32, usize) -> i32,
    backend_name: unsafe extern "C" fn(EncoderHandle) -> *const std::ffi::c_char,
    last_error: unsafe extern "C" fn() -> *const std::ffi::c_char,
    stats: unsafe extern "C" fn(EncoderHandle, *mut u64, *mut u64, *mut u64) -> i32,
    free: unsafe extern "C" fn(EncoderHandle),
}
impl GgmlEncoderApi {
    fn load_library(path: &Path) -> Result<Self> {
        let library: libloading::Library = unsafe {
            libloading::os::windows::Library::load_with_flags(path, 0x00000008)?
        }.into();
        unsafe {
            let version = *library.get::<unsafe extern "C" fn() -> u32>(b"caps_ggml_encoder_abi_version\0")?;
            ensure!(version() == 1, "研究编码器 ABI 版本不匹配");
            Ok(Self {
                load: *library.get(b"caps_ggml_encoder_load\0")?,
                dimension: *library.get(b"caps_ggml_encoder_dimension\0")?,
                output_rows: *library.get(b"caps_ggml_encoder_output_rows\0")?,
                encode: *library.get(b"caps_ggml_encoder_encode\0")?,
                backend_name: *library.get(b"caps_ggml_encoder_backend_name\0")?,
                last_error: *library.get(b"caps_ggml_encoder_last_error\0")?,
                stats: *library.get(b"caps_ggml_encoder_stats\0")?,
                free: *library.get(b"caps_ggml_encoder_free\0")?,
                _library: library,
            })
        }
    }
    fn error(&self) -> String {
        let raw = unsafe { (self.last_error)() };
        if raw.is_null() { "原生编码器没有提供错误详情".into() }
        else { unsafe { std::ffi::CStr::from_ptr(raw) }.to_string_lossy().into_owned() }
    }
}
struct GgmlAudioEncoder {
    handle: EncoderHandle,
    api: GgmlEncoderApi,
    // Keeps the exact installed decoder ABI and backend DLLs alive. The normal
    // loader mutex also serializes plugin initialization with decoder loading.
    _native_api: std::sync::Arc<super::llama::Api>,
    mel: MelExtractor,
    dim: usize,
    last_metrics: EncoderMetrics,
}
// The handle has exclusive ownership and is used by one mutable worker. The
// existing loader may create it on the load thread and then transfer ownership.
unsafe impl Send for GgmlAudioEncoder {}
impl GgmlAudioEncoder {
    fn load(dir: &Path, options: &EngineOptions, provider: &str) -> Result<Self> {
        let (expected_backend, device_name) = match provider {
            "ggml_vulkan" => (super::GpuBackend::Vulkan, "Vulkan0"),
            "ggml_cuda" => (super::GpuBackend::Cuda, "CUDA0"),
            _ => anyhow::bail!("未知 GGML 编码后端：{provider}"),
        };
        ensure!(options.use_gpu && options.gpu_backend == expected_backend,
            "GGML 编码器与解码器必须使用配置指定的同一 GPU 后端");
        let native_dir = std::env::var_os("YYUR_GGML_NATIVE_DIR")
            .map(PathBuf::from).or_else(|| options.llama_dir.clone())
            .context("缺少 GGML 运行库目录")?;
        let mmproj = AudioEncoder::model_path(dir, options);
        let native_api = super::llama::Api::get(&native_dir)?;
        let api = GgmlEncoderApi::load_library(&native_dir.join("caps_encoder.dll"))?;
        let model_path = mmproj.canonicalize()?;
        let model_string = model_path.to_str().context("研究模型路径不是 Unicode")?;
        let model_string = model_string.strip_prefix(r"\\?\").unwrap_or(model_string);
        let model = std::ffi::CString::new(model_string.replace('\\', "/"))?;
        let device = std::ffi::CString::new(device_name)?;
        let handle = unsafe { (api.load)(model.as_ptr(), device.as_ptr(), options.onnx_intra_threads.try_into()?, 1) };
        ensure!(!handle.is_null(), "加载 GGML 编码器失败：{}", api.error());
        let dim = unsafe { (api.dimension)(handle) };
        if !(1..=8192).contains(&dim) {
            unsafe { (api.free)(handle); }
            anyhow::bail!("GGML 编码器输出维度非法：{dim}");
        }
        let backend = unsafe { (api.backend_name)(handle) };
        let backend = if backend.is_null() { String::new() }
            else { unsafe { std::ffi::CStr::from_ptr(backend) }.to_string_lossy().into_owned() };
        if backend != device_name {
            unsafe { (api.free)(handle); }
            anyhow::bail!("编码器实际后端 {backend} 与请求 {device_name} 不符");
        }
        eprintln!("YYUR_GGML_ENCODER {}", serde_json::json!({"model":mmproj,"asr_directory":dir,
            "backend":backend,"dimension":dim,"input_f16_roundtrip":true,"output_f16_roundtrip":true}));
        let mut encoder = Self { handle, api, _native_api:native_api, mel:MelExtractor::new(),
            dim:dim as usize,last_metrics:EncoderMetrics::default() };
        // Identical two-second silence workload to the production ORT loader.
        encoder.encode(&vec![0.0; 32000])?;
        Ok(encoder)
    }
    fn encode(&mut self, audio: &[f32]) -> Result<Features> {
        let total = std::time::Instant::now();
        let mel_started = std::time::Instant::now();
        let (mut mel, frames) = self.mel.extract(audio)?;
        for value in &mut mel { *value = half::f16::from_f32(*value).to_f32(); }
        let mel_seconds = mel_started.elapsed().as_secs_f64();
        let valid = output_length(frames);
        ensure!(unsafe { (self.api.output_rows)(frames) } == valid,
            "GGML 与现有编码器的有效行数不一致");
        let count = valid.checked_mul(self.dim).context("编码器输出大小溢出")?;
        let mut data = vec![0.0; count];
        let tick = std::time::Instant::now();
        let code = unsafe { (self.api.encode)(self.handle,mel.as_ptr(),frames,valid,data.as_mut_ptr(),data.len()) };
        ensure!(code == 0, "GGML 编码失败（{code}）：{}",self.api.error());
        let encode_seconds = tick.elapsed().as_secs_f64();
        for value in &mut data { *value = half::f16::from_f32(*value).to_f32(); }
        ensure!(data.iter().all(|v|v.is_finite()),"GGML 编码器输出含非有限值");
        let (mut gpu,mut cpu,mut nodes) = (0,0,0);
        ensure!(unsafe { (self.api.stats)(self.handle,&mut gpu,&mut cpu,&mut nodes) } == 0,
            "读取 GGML 调度统计失败：{}",self.api.error());
        eprintln!("YYUR_GGML_ENCODE {}",serde_json::json!({"audio_samples":audio.len(),"frames":frames,
            "valid_rows":valid,"dimension":self.dim,"mel_seconds":mel_seconds,"graph_seconds":encode_seconds,
            "total_seconds":total.elapsed().as_secs_f64(),"gpu_nodes":gpu,"cpu_nodes":cpu,"total_nodes":nodes}));
        // The research adapter stores the complete graph in this legacy slot;
        // it is FE+BE, not comparable to the ORT BE-only slot. IPC t_encode and
        // total_seconds are the common full-encoder comparison metrics.
        self.last_metrics=EncoderMetrics {audio_samples:audio.len(),frames,chunks:frames.div_ceil(100),
            valid_rows:valid,output_dim:self.dim,fp16:true,mel_seconds,backend_run_seconds:encode_seconds,
            total_seconds:total.elapsed().as_secs_f64(),..Default::default()};
        Ok(Features {data,dim:self.dim})
    }
}
impl Drop for GgmlAudioEncoder {
    fn drop(&mut self) { unsafe { (self.api.free)(self.handle); } }
}
enum ResearchEncoderBackend { Ort(OrtAudioEncoder), Ggml(GgmlAudioEncoder) }
pub(super) struct AudioEncoder { backend:ResearchEncoderBackend, asr:bool, calls:usize, provider:String }
impl AudioEncoder {
    fn provider(options: &EngineOptions) -> Result<String> {
        let provider=std::env::var("YYUR_ENCODER_RESEARCH_PROVIDER").unwrap_or_else(|_|match options.encoder_provider {
            EncoderProvider::GgmlVulkan => "ggml_vulkan".into(),
            EncoderProvider::GgmlCuda => "ggml_cuda".into(),
            _ => "ort".into(),
        });
        ensure!(matches!(provider.as_str(),"ort"|"ggml_vulkan"|"ggml_cuda"),"未知研究编码器：{provider}");
        Ok(provider)
    }
    #[allow(dead_code)] // The CPU-only standalone probe does not load Engine.
    pub fn uses_ggml(frontend:&str,options:&EngineOptions)->Result<bool> {
        Ok(frontend=="qwen3_asr_encoder_frontend.onnx" && Self::provider(options)?!="ort")
    }
    pub fn model_path(dir:&Path,options:&EngineOptions)->PathBuf {
        std::env::var_os("YYUR_GGML_ENCODER_MODEL").map(PathBuf::from)
            .or_else(||options.ggml_encoder_model.clone())
            .unwrap_or_else(||dir.join("qwen3_asr_encoder.gguf"))
    }
    pub fn load(dll:&Path,dir:&Path,frontend:&str,backend:&str,options:&EngineOptions)->Result<Self> {
        let provider=Self::provider(options)?;
        // Never route the independent INT4 forced aligner or the optional 0.6B
        // encoder through the 1.7B projector. No loading-order changes.
        let asr=frontend=="qwen3_asr_encoder_frontend.onnx";
        let use_ggml=asr && provider!="ort";
        let backend=if use_ggml {
            ResearchEncoderBackend::Ggml(GgmlAudioEncoder::load(dir,options,&provider)?)
        } else {
            let mut ort_options=options.clone();
            if matches!(ort_options.encoder_provider,EncoderProvider::GgmlVulkan|EncoderProvider::GgmlCuda) {
                // The optional 0.6B pack and independent aligner retain CPU ORT.
                ort_options.encoder_provider=EncoderProvider::Cpu;
            }
            ResearchEncoderBackend::Ort(OrtAudioEncoder::load(dll,dir,frontend,backend,&ort_options)?)
        };
        let actual_provider=if use_ggml {provider}else{"ort".into()};
        Ok(Self {backend,asr,calls:0,provider:actual_provider})
    }
    pub fn encode(&mut self,audio:&[f32])->Result<Features> {
        let feature=match &mut self.backend {
            ResearchEncoderBackend::Ort(value)=>value.encode(audio)?,
            ResearchEncoderBackend::Ggml(value)=>value.encode(audio)?,
        };
        if self.asr {
            if let Some(directory)=std::env::var_os("YYUR_ENCODER_EMBEDDINGS_DIR") {
                use std::io::Write;
                let directory=PathBuf::from(directory);
                std::fs::create_dir_all(&directory)?;
                let path=directory.join(format!("{}-{}.f32",std::process::id(),self.calls));
                let mut output=std::io::BufWriter::new(std::fs::OpenOptions::new().write(true).create_new(true).open(&path)?);
                for value in &feature.data { output.write_all(&value.to_le_bytes())?; }
                output.flush()?;
                let mut hash=sha2::Sha256::new();
                use sha2::Digest;
                for sample in audio {hash.update(sample.to_le_bytes());}
                eprintln!("YYUR_ENCODER_EMBEDDING {}",serde_json::json!({"path":path,
                    "pcm_sha256":format!("{:x}",hash.finalize()),"rows":feature.data.len()/feature.dim,"dimension":feature.dim,
                    "provider":self.provider}));
            }
        }
        self.calls+=1;
        Ok(feature)
    }
    #[allow(dead_code)]
    pub fn last_metrics(&self)->&EncoderMetrics {
        match &self.backend { ResearchEncoderBackend::Ort(value)=>value.last_metrics(),
            ResearchEncoderBackend::Ggml(value)=>&value.last_metrics }
    }
}
