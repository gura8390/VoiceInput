# 从零完整复现手册（灾备与自动化重建）

本手册面向**本地运行目录完全丢失**的场景：任何开发者或 AI 仅凭本 GitHub 仓库（源码 + 补丁 + 本清单中的精确版本与哈希），即可从零重建完整的 Windows x64 可运行程序、原生推理库与模型权重。

---

## 1. 目标运行目录总览

```text
语音输入/
  语音输入.exe                                  # Rust 应用层单一入口（含图标与版本资源）
  config/
    settings.json                               # 从仓库 settings.example.json 复制
    hot.txt                                     # 从仓库 assets/text/hot_default.txt 复制
    hot-rule.txt                                # 从仓库 assets/text/rules_default.txt 复制
  runtime/
    llama/
      llama.dll                                 # llama.cpp b9090 + 4 个原生补丁
      ggml.dll                                  # 匹配 b9090
      ggml-base.dll                             # 匹配 b9090
      ggml-cpu-*.dll                            # 匹配 b9090 的 CPU 动态后端插件
      ggml-vulkan.dll                           # 匹配 b9090 + Vulkan 管线缓存补丁
      libomp140.x86_64.dll                      # CPU 插件所需 OpenMP 运行库（启动时预加载）
    onnx/
      onnxruntime.dll                           # Microsoft ONNX Runtime 1.24.4 (x64 CPU)
    audio/
      libportaudio64bit.dll                     # PortAudio v19 (Windows x64)
    media/
      ffmpeg.exe                                # Windows x64 FFmpeg（及配套 DLL，若非静态构建）
  models/
    Qwen3-ASR/Qwen3-ASR-1.7B/                   # 默认 ASR 模型（FP16 ONNX + 外置权重 + Q5_K_M GGUF）
    Qwen3-ASR/Qwen3-ASR-0.6B/                   # 可选小模型（INT4 ONNX + 外置权重 + Q4_K GGUF）
    Qwen3-ForcedAligner/Qwen3-ForcedAligner-0.6B/ # 文件转录时间戳对齐器（INT4 ONNX + 外置权重 + Q4_K GGUF）
    silero/silero_vad.onnx                      # 可选 Silero VAD（仅 vad=true 时必需）
  cache/                                        # 运行时自动创建（vulkan 缓存 / downloads）
  logs/                                         # 运行时自动创建（rust-worker.log）
```

> 根目录另需系统已安装 Microsoft Visual C++ Redistributable (x64)，或在 `语音输入.exe` 同级放置便携版 MSVC/UCRT 运行库。

---

## 2. 编译 Rust 应用层

前置工具：Windows x64、Rust stable MSVC（按 `rust-toolchain.toml`）、Visual Studio Build Tools (C++ x64)、Windows SDK。依赖版本由 `Cargo.lock` 锁定（`ort = "=2.0.0-rc.12"` 对应 ONNX Runtime `api-24`）。

```powershell
# 如缺少 Windows SDK 导入库，可先自动准备到共享构建目录（不改全局环境）
python .\tools\build\bootstrap_sdk.py

# 编译发布版 EXE 并运行完整单元/集成测试
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1 -Tests
```

产物位于 `dist\语音输入.exe`。初始化运行目录：

```powershell
New-Item -ItemType Directory -Path '.\dist\app\config' -Force
Copy-Item '.\dist\语音输入.exe' '.\dist\app\语音输入.exe'
Copy-Item '.\settings.example.json' '.\dist\app\config\settings.json'
Copy-Item '.\assets\text\hot_default.txt' '.\dist\app\config\hot.txt'
Copy-Item '.\assets\text\rules_default.txt' '.\dist\app\config\hot-rule.txt'
```

---

## 3. 模型权重复现（双层保障）

### 方案 A（首选）：使用内置下载器一键获取并校验

`语音输入.exe` 内置了断点续传下载、SHA256 校验、解压和 Rust 原生外置权重转换（自动支持 `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` 及 Windows 系统代理；也可在 `config/settings.json` 填写 `model_mirror_base` 镜像前缀）：

```powershell
# 下载当前配置所选 ASR 模型（默认 1.7B）+ 对齐器 + Silero VAD
.\dist\app\语音输入.exe download-models

# 同时下载 0.6B 与 1.7B 两档 ASR + 对齐器 + Silero VAD
.\dist\app\语音输入.exe download-models --all
```

#### 固定模型包目录与校验参数（与 `src/models/catalog.rs` 严格一致）

| 包 ID | 模型名称 | 官方直链 URL | 压缩包/单文件大小 (B) | 压缩包/单文件 SHA-256 | 目标目录 |
| --- | --- | --- | ---: | --- | --- |
| `asr-0.6b` | Qwen3-ASR 0.6B (INT4/Q4_K) | `https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/Qwen3-ASR-0.6B-gguf.zip` | `563976147` | `3684b9cac9978a93c82ceeb93448bb5027c0b2beaf2e0affa35190ba20e545ff` | `models/Qwen3-ASR/Qwen3-ASR-0.6B` |
| `asr-1.7b` | Qwen3-ASR 1.7B (FP16/Q5_K_M，默认) | `https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/Qwen3-ASR-1.7B-q5_k.zip` | `1951570656` | `f40040fe62a5ef0c09f8699fdbcb30f18bb8ae2bcd515ed4954e1f62b8b0e88f` | `models/Qwen3-ASR/Qwen3-ASR-1.7B` |
| `aligner-0.6b` | Qwen3-ForcedAligner 0.6B (INT4/Q4_K) | `https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/Qwen3-ForceAligner-0.6B-gguf.zip` | `504734961` | `d9c16a2db3293c0c4120b0291d5d0fcfa208d842577092ca49cecc08a4690fb7` | `models/Qwen3-ForcedAligner/Qwen3-ForcedAligner-0.6B` |
| `silero-vad` | Silero VAD (可选) | `https://raw.githubusercontent.com/snakers4/silero-vad/57053ec6ef95ab4704d380e3ff5b98c2f093586c/src/silero_vad/data/silero_vad.onnx` | `2204103` | `f6f4407b41e05e277e7f5cfd9b9ae957b1dfe763c6076a934734c0b448e09b31` | `models/silero` |

> **注意**：`asr-1.7b` 对应的是 `CapsWriter-Offline` 的 `Qwen3-ASR-1.7B-q5_k.zip`（原 FP16 ONNX + Q5_K_M GGUF），**不是** `Qwen3-ASR-GGUF` 的 `Qwen3-ASR-1.7B-gguf.zip`（那套是 INT4/Q4_K 且文件名不同）。

#### 解压成员明细与外置权重规则

所有标记“需外置”的 ONNX 文件在落位时均以固定参数 `minimum=4096, alignment=4096` 转换为外置权重，生成同前缀 `.weights.bin` 和 `.external.json` 收据（与 `tools/models/prepare_external_weights.py` 逐字节一致）：

| 包 ID | 压缩包内路径 | 安装文件名 | 原始大小 (B) | 原始文件 SHA-256 | 需外置 |
| --- | --- | --- | ---: | --- | :---: |
| `asr-0.6b` | `qwen3_asr_encoder_frontend.int4.onnx` | `qwen3_asr_encoder_frontend.int4.onnx` | `20343991` | `8c29d210b934f377983c0daa28d276f36cbc9a8a823d59d3afc5da2fd8eda2f6` | 是 |
| `asr-0.6b` | `qwen3_asr_encoder_backend.int4.onnx` | `qwen3_asr_encoder_backend.int4.onnx` | `94750816` | `95b8ed080ef99e28de62523ea0d70fb5b1383f17d4238464a265b3454eaaaa55` | 是 |
| `asr-0.6b` | `qwen3_asr_llm.q4_k.gguf` | `qwen3_asr_llm.q4_k.gguf` | `484215360` | `38a5492b88292dcb14c48fb939212da8223ec8d1e19b29d6e2751f26c0754cdd` | 否 |
| `asr-1.7b` | `Qwen3-ASR-1.7B/qwen3_asr_encoder_frontend.onnx` | `qwen3_asr_encoder_frontend.onnx` | `24063978` | `fc288ba9a0ca732251b33628c803cd5188d3dd77a2cd7336f0f0fd37969ed4cd` | 是 |
| `asr-1.7b` | `Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.onnx` | `qwen3_asr_encoder_backend.onnx` | `611276553` | `4b0b1e34e8393f29222006aae0719283b0ff28b61f289b6ff8377b3ddbda27e3` | 是 |
| `asr-1.7b` | `Qwen3-ASR-1.7B/qwen3_asr_llm.gguf` | `qwen3_asr_llm.gguf` | `1471800896` | `89e198470f41d487ab5b6b8a36a35400b3b1534451c4499236983050fe9193e5` | 否 |
| `aligner-0.6b` | `qwen3_aligner_encoder_frontend.int4.onnx` | `qwen3_aligner_encoder_frontend.int4.onnx` | `20876727` | `858ea960c7feaf48a8787cb4a375e70216f99899cc229b174a6ebfdebd4f6e59` | 是 |
| `aligner-0.6b` | `qwen3_aligner_encoder_backend.int4.onnx` | `qwen3_aligner_encoder_backend.int4.onnx` | `164176179` | `7c8931be70a64cd75be1671d714e9497d0bc2ef7943ffc9024807f71d24855b1` | 是 |
| `aligner-0.6b` | `qwen3_aligner_llm.q4_k.gguf` | `qwen3_aligner_llm.q4_k.gguf` | `484399552` | `c2608b44866ac790fb0f9f542ae4a473c49fde03a7a2eb5b5370cc095b78dc8b` | 否 |
| `silero-vad` | `silero_vad.onnx` | `silero_vad.onnx` | `2204103` | `f6f4407b41e05e277e7f5cfd9b9ae957b1dfe763c6076a934734c0b448e09b31` | 否 |

### 方案 B（终极兜底）：若上游预转换 Release 失效，从 Hugging Face 原始权重重建

1. 从 Hugging Face 下载官方原始权重：`Qwen/Qwen3-ASR-0.6B`、`Qwen/Qwen3-ASR-1.7B`、`Qwen/Qwen3-ForcedAligner-0.6B`。
2. 克隆上游导出与量化脚本仓库：`https://github.com/HaujetZhao/Qwen3-ASR-GGUF`。
   - **ASR 导出链**：依次运行 `01-Export-ASR-Encoder-Frontend.py` → `02-Export_ASR-Encoder-Backend.py` → `03-Optimize-ASR-Encoder.py` →（0.6B 需 `04-Quantize-ASR-Encoder.py` 转 INT4；1.7B 保持 FP16）→ `05-Export-ASR-Decoder-HF.py` → `06-Convert-ASR-Decoder-GGUF.py` → `07-Quantize-ASR-Decoder-GGUF.py`（0.6B 用 `Q4_K`，1.7B 用 `Q5_K_M`）。
   - **Aligner 导出链**：依次运行 `11-Export-Aligner-Encoder-Frontend.py` → `12-Export-Aligner-Encoder-Backend.py` → `13-Optimize-Aligner-Encoder.py` → `14-Quantize-Aligner-Encoder.py` → `15-Export-Aligner-Decoder-HF.py` → `16-Convert-Aligner-Decoder-GGUF.py` → `17-Quantize-Aligner-Decoder-GGUF.py`。
3. 对所有导出后的编码器 `.onnx` 运行本仓库脚本生成外置权重：
   ```powershell
   python .\tools\models\prepare_external_weights.py `
     --source <内嵌权重ONNX> `
     --output <输出目录>\<目标ONNX文件名> `
     --output-root <输出目录> `
     --minimum 4096 --alignment 4096
   ```

---

## 4. 原生运行库（`runtime/`）从零复现

### 4.1 `runtime/llama/`（llama.cpp b9090 + 4 个补丁一站式构建）

- **上游基线**：`https://github.com/ggml-org/llama.cpp`，tag **`b9090`**，commit **`5757c4dcb178a01c85234a6db7503b19c9598873`**。
- **必需补丁与精确应用顺序**（均位于本仓库 `native/`）：
  1. `native/aligner-logits.patch`（导出 `caps_llama_init_from_model_with_logits_limit`、`caps_llama_logits_width`）
  2. `native/ggml-vulkan-cache.patch`（Windows Vulkan 持久管线缓存）
  3. `native/asr-single-output.patch`（导出 `caps_llama_init_from_model_with_output_limit`、`caps_llama_output_limit`）
  4. `native/windows-vulkan-bounded-upload.patch`（Windows Vulkan 4×1 MiB 环有界上传；应用前 `src/llama-model-loader.cpp` LF SHA256 为 `e487eeb7849bcb5ff1be2c90c360cded0dad3af34b844150c711d6b779a8c4a6`，应用后为 `6fc9e68b6645385e11816307462120fd7f10d81aa8a016ac404c2fbb8a24f178`）

在安装了 **Visual Studio Build Tools (C++ x64)**、**Windows SDK**、**CMake**、**Ninja** 和 **[Windows Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows)** 的 **x64 Developer PowerShell** 中，于本仓库根目录执行：

```powershell
$src = Join-Path (Get-Location) '.build/llama-b9090-full'
$bld = Join-Path (Get-Location) '.build/llama-b9090-full-build'

git clone --branch b9090 --depth 1 https://github.com/ggml-org/llama.cpp.git $src
if ((git -C $src rev-parse HEAD).Trim() -ne '5757c4dcb178a01c85234a6db7503b19c9598873') {
    throw 'llama.cpp 基线提交不匹配'
}
foreach ($patch in @(
    'native/aligner-logits.patch',
    'native/ggml-vulkan-cache.patch',
    'native/asr-single-output.patch',
    'native/windows-vulkan-bounded-upload.patch'
)) {
    $patchPath = (Resolve-Path $patch).Path
    git -C $src apply --check $patchPath
    if ($LASTEXITCODE -ne 0) { throw "补丁检查失败: $patch" }
    git -C $src apply $patchPath
    if ($LASTEXITCODE -ne 0) { throw "补丁应用失败: $patch" }
}

# 从零一次性构建核心库 + CPU 全变体插件 + Vulkan 后端
cmake -S $src -B $bld -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON `
  -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF `
  -DGGML_CPU=ON -DGGML_CPU_ALL_VARIANTS=ON `
  -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_OPENCL=OFF -DGGML_SYCL=OFF `
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF `
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build $bld --parallel 2
```

将 `$bld\bin\` 中生成的 `llama.dll`、`ggml.dll`、`ggml-base.dll`、`ggml-cpu-*.dll`、`ggml-vulkan.dll` 复制到 `runtime/llama/`，并放入匹配的 `libomp140.x86_64.dll`（来自 Visual Studio/LLVM OpenMP x64 运行库或兼容 CapsWriter 运行库；注意 `src/engine/llama.rs` 启动时会显式预加载 `runtime/llama/libomp140.x86_64.dll`）。

- **校验导出符号**：使用 `dumpbin /EXPORTS runtime\llama\llama.dll` 或 `python tools\validation\verify_asr_single_output.py` 确认以下 4 个私有符号均存在：
  - `caps_llama_init_from_model_with_logits_limit`
  - `caps_llama_logits_width`
  - `caps_llama_init_from_model_with_output_limit`
  - `caps_llama_output_limit`
- **降级兼容说明**：如仅用未打补丁的原版 `b9090` DLL 临时测试，须在 `config/settings.json` 中设置 `"aligner_logits_limit": 0, "asr_single_output": false`（将失去 0.4.3/0.4.4 内存优化）。

### 4.2 `runtime/onnx/onnxruntime.dll`

- **版本要求**：Microsoft ONNX Runtime **1.24.4**（Windows x64 CPU 版），需与 `Cargo.toml` 锁定的 `ort = "=2.0.0-rc.12"`（`api-24`）ABI 匹配。
- **获取方式**：
  - 从 PyPI 官方包 `onnxruntime==1.24.4` 的 `win_amd64` wheel 中提取 `onnxruntime/capi/onnxruntime.dll`（及同目录 `onnxruntime_providers_shared.dll` 如有），放置到 `runtime/onnx/`；
  - 或从 Microsoft ONNX Runtime 官方发布页下载 `1.24.4` Windows x64 归档并提取 `lib/onnxruntime.dll`。

### 4.3 `runtime/audio/libportaudio64bit.dll`

- **版本要求**：PortAudio v19 Windows x64 动态库，需导出标准 C API（`Pa_Initialize`、`Pa_Terminate`、`Pa_GetDefaultInputDevice`、`Pa_GetDeviceInfo`、`Pa_GetErrorText`、`Pa_OpenStream`、`Pa_StartStream`、`Pa_StopStream`、`Pa_AbortStream`、`Pa_CloseStream`、`Pa_SetStreamFinishedCallback`）。
- **获取方式**：从 PortAudio 官方源码编译 Windows x64 DLL（建议启用 WASAPI），或从 `python-sounddevice`（`_sounddevice_data/portaudio-binaries/libportaudio64bit.dll`）/ 合法取得的 `CapsWriter-Offline` 运行库复制并保留 MIT 许可。

### 4.4 `runtime/media/ffmpeg.exe`

- **版本要求**：任意现代 Windows x64 FFmpeg（BtbN / Gyan 官方发布构建或 `CapsWriter-Offline` 附带版本），程序仅通过以下命令调用：
  ```text
  ffmpeg.exe -nostdin -v error -i <input> -f f32le -ac 1 -ar 16000 -
  ```
- 仅在音视频文件转录时调用；实时麦克风听写不依赖 FFmpeg。若使用动态链接版 FFmpeg，须将其配套 `av*.dll` / `sw*.dll` 一并放在 `runtime/media/`。

---

## 5. 端到端重建验收命令

全部文件就位后，在应用目录执行：

```powershell
# 1. 验证配置解析与单实例自动化接口（只读，不加载模型）
.\dist\app\语音输入.exe automation status

# 2. 复核全部模型文件存在性与 SHA256 校验（已通过的文件会自动跳过）
.\dist\app\语音输入.exe download-models

# 3. 端到端离线文件转录验证（需准备任意简短 wav/mp3 样本）
.\dist\app\语音输入.exe transcribe '.\sample.wav' --output-dir '.\verify-out'
```

验收通过标准：命令退出码为 `0`，`.\verify-out\` 下生成非空 `.txt`、`.json`、`.srt` 文件，且 `logs\rust-worker.log` 无加载或 ABI 报错。
