# GGML 技术分支使用说明

2026 年 11 月 18 日按用户要求发布为两个独立源码技术分支：

| 分支 | 名称 | ASR 编码 / 解码 |
| --- | --- | --- |
| `0.6.0-GGML` | 预览版 | GGML / Vulkan 编码，Vulkan 解码 |
| `0.6.0-GGML-CUDA` | 先行版 | GGML / CUDA 编码，CUDA 解码 |

两者均从正式 `0.6.0` 的 `748cd7dc12acd095f63c28dc8b44ce2693ce2a22` 分出，应用源码相同，示例配置分别选择对应技术路线。内部 Cargo / 程序资源版本保留 `0.6.0`，分支名标识技术路线。本次只发布源码技术分支，不创建应用 Release 或上传运行包，不更改 main、既有标签、旧 CUDA 技术分支及本机安装。

## 构建和配置

在分支根目录按 [README](../README.md) 的工具链要求执行 `build.ps1 -Tests`、`build.ps1`。适配器已经接入 `src/`，直接构建即可；`stage_ggml_encoder_candidate.py` 是回溯旧基准源码的研究工具，勿在已经接入的分支重复应用。

运行目录和原 0.6.0 一致，另需准备：

1. 匹配 b9090 的 batch4 `caps_encoder.dll`，放入 `runtime/llama/`。构建和接口见 [小批编码器说明](../native/ggml-vulkan-encoder-batched.md)。保留经过验证的原 decoder 与 ggml core，不随意混用 ABI。
2. 从当前 1.7B FP16 / FP32 ONNX 等值转换的编码 GGUF，存为 `models/Qwen3-ASR/Qwen3-ASR-1.7B/qwen3_asr_encoder.gguf`。方法见 [编码研究报告](ggml-vulkan-encoder-research.md) 及 `tools/research/prepare_ggml_encoder.py`；仍使用原 Q5_K_M 解码器，不混用其它精度模板值。
3. 对应 Vulkan 或 CUDA 原生后端；CUDA 还需匹配的 cudart / cuBLAS / Lt。见 [CUDA 原生说明](../native/ggml-cuda-encoder.md) 和 [CUDA 延迟加载构建](../native/cuda-delayload.md)。当前归档 CUDA 构建针对 RTX 5060 Laptop 的 SM120a，其它架构需重新构建及测试。NVIDIA vendor 库受其自己的许可约束，本仓库未上传这些库。
4. CPU ONNX Runtime 与原 ForcedAligner 模型，供文件对齐；可选 0.6B ASR 编码也继续使用 CPU ORT。Mel、热词和文字处理仍在 CPU。

**先将本分支 `settings.example.json` 复制到运行目录的 `config/settings.json`，再启动。** 示例 `app_dir=..` 指向程序根目录；`ggml_encoder_model=../models/Qwen3-ASR/Qwen3-ASR-1.7B/qwen3_asr_encoder.gguf` 以配置文件目录为基准。不要把这份相对路径配置留在程序根目录直接用。

底层缺配置默认仍是 CPU ORT；本分支通过示例配置选择 GGML，不假设仅切换 Git 分支即可自动改动本机配置。已有个人设置先保留，手工合入 `encoder_provider`、`gpu_backend`、`ggml_encoder_model` 三项，保留个人词表与其它选项。选中 GGML 但模型缺失、后端不一致或设备不可用时明确报错。

没有新增主窗或设置窗；沿用实时听写、热词与文件转录。没有采用分阶段加载模型。本次不提供模型、DLL、个人配置、个人词表、录音或本机 evidence。

## 测量与源代码身份

[三版对照](ggml-three-versions-comparison.md) 和 [原 CUDA 分支对照](previous-cuda-comparison.md) 保留完整资源、速度和固定样本差异。报告数字绑定 2026-11-17 的 `2c1302bd…` 研究 EXE，不把源码上传描述成新二进制重新测量。两个公开分支应用代码来自这份最终研究源码，示例配置改为便携相对路径；发布前在分支中另做编译和配置路径检查，收据留本机。

先行比预览五请求时间少 11.24%，但峰私有提交多约 723 MiB，专用显存多约 154 MiB。参考小集合的 CER 相同；其它种子及文件存在字词、标点、字幕切分与时间戳差异，详见报告。正式基准版的提交和显存优势保留，没有宣称 GGML 全面省内存。

## 许可与隐私

项目和原生补丁遵循各自的 MIT 等已有许可，原作者、Rust 依赖与词典声明保留。见 [第三方声明](../THIRD-PARTY-NOTICES.txt) 及 `assets/licenses/`。不得将 NVIDIA vendor 运行库称作 MIT。公开提交不含本机绝对用户路径、个人热词、原始转录和凭据，研究语料工具通过显式参数接收输入。
