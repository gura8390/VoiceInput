//! Installed only in the isolated research worktree; never starts a runtime.
use std::{fs, path::PathBuf, time::{SystemTime, UNIX_EPOCH}};
use yuybuuru::{config::{Settings, EncoderProvider}, engine::Engine, pipeline::engine_options};

struct Fixture { root:PathBuf, app:PathBuf, config:PathBuf, model:PathBuf }
impl Fixture {
    fn new()->Self {
        let root=std::env::temp_dir().join(format!("yyur-ggml-route-{}-{}",std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir(&root).unwrap();
        let app=root.join("app");fs::create_dir(&app).unwrap();
        fs::create_dir(root.join("config")).unwrap();
        fs::create_dir(root.join("weights")).unwrap();
        let config=root.join("config/settings.json");
        let model=root.join("weights/encoder.gguf");fs::write(&model,b"path-fixture-only").unwrap();
        Self {root,app,config,model}
    }
    fn load(&self, provider:&str,backend:&str,model:&str,asr:&str)->anyhow::Result<Settings> {
        fs::write(&self.config,serde_json::to_vec(&serde_json::json!({"app_dir":self.app,
            "encoder_provider":provider,"gpu_backend":backend,"ggml_encoder_model":model,"asr_model":asr})).unwrap()).unwrap();
        Settings::load(Some(&self.config),None)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        // Remove only the exact files/directories created by this fixture.
        let _=fs::remove_file(&self.config);let _=fs::remove_file(&self.model);
        let _=fs::remove_file(self.app.join("models/Qwen3-ASR/Qwen3-ASR-1.7B/qwen3_asr_llm.gguf"));
        for relative in ["models/Qwen3-ASR/Qwen3-ASR-1.7B","models/Qwen3-ASR","models"] {
            let _=fs::remove_dir(self.app.join(relative));
        }
        for path in [&self.app,&self.root.join("config"),&self.root.join("weights")] {let _=fs::remove_dir(path);}
        let _=fs::remove_dir(&self.root);
    }
}
#[test]
fn ggml_settings_resolve_from_config_and_survive_worker_serialization() {
    let f=Fixture::new();
    for (provider,backend,expected) in [("ggml_vulkan","vulkan",EncoderProvider::GgmlVulkan),
                                      ("ggml_cuda","cuda",EncoderProvider::GgmlCuda)] {
        let settings=f.load(provider,backend,"../weights/encoder.gguf","1.7b").unwrap();
        assert_eq!(settings.ggml_encoder_model.as_ref().unwrap().canonicalize().unwrap(),f.model.canonicalize().unwrap());
        let worker:Settings=serde_json::from_slice(&serde_json::to_vec(&settings).unwrap()).unwrap();
        assert_eq!(worker.encoder_provider,expected);
        let options=engine_options(&worker);
        assert_eq!(options.ggml_encoder_model,settings.ggml_encoder_model);
        assert_eq!(options.llama_dir.unwrap(),worker.app_dir.join("runtime/llama"));
    }
    assert!(f.load("ggml_cuda","vulkan","../weights/encoder.gguf","1.7b").is_err());
    assert!(f.load("ggml_vulkan","cuda","../weights/encoder.gguf","1.7b").is_err());
}
#[test]
fn unused_ggml_model_does_not_break_optional_small_cpu_pack() {
    let f=Fixture::new();
    let settings=f.load("ggml_cuda","cuda","../weights/missing.gguf","1.7b").unwrap();
    let pack=f.app.join("models/Qwen3-ASR/Qwen3-ASR-1.7B");
    fs::create_dir_all(&pack).unwrap();
    fs::write(pack.join("qwen3_asr_llm.gguf"),b"preflight-only").unwrap();
    let error=Engine::load(&f.app,&engine_options(&settings)).err().expect("Missing selected encoder must fail");
    assert!(format!("{error:#}").contains("读取 GGML 编码模型失败"));
    assert!(f.load("ggml_cuda","cuda","../weights/missing.gguf","0.6b").is_ok());
    assert!(f.load("cpu","vulkan","../weights/missing.gguf","0.6b").is_ok());
}
