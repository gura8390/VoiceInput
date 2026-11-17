# GGML / Vulkan 编码器隔离研究

本页保留第一候选的接入和构建记录。后续固定前端 batch4 的工作区优化见 [小批研究记录](ggml-vulkan-encoder-batched.md)；其保留同一模型和整段 BE 注意力。

本研究只接入现有 Qwen3-ASR 1.7B 编码器，不更换模型、解码器权重、热词、界面或文件对齐器。正式程序的 Rust 托盘和三项功能保持现有组织。原生候选工程已完成；编码数值、转写质量、速度和占用由独立候选程序测量后决定是否采用。

## 原生接入方案

采用已验证的 llama.cpp **b9090 / `5757c4dcb178a01c85234a6db7503b19c9598873`** 源码及现有对齐器、单输出预留补丁。新增 `caps_encoder.dll`，运行时继续使用正式版本同一组 `llama.dll`、`ggml.dll`、`ggml-base.dll`、CPU 和 Vulkan 后端。

构建目录产生的其它 DLL 不纳入候选运行目录。仅新增 `caps_encoder.dll`，因此原有 decoder 的自定义优化和实际二进制保持不变。正式 decoder 已含有界 Vulkan 上传优化；本次原生源码底座的模型加载器尚未带此补丁，严禁用构建目录的新 llama.dll 替换正式 DLL。

[上游 Qwen3A 修复 72e60f500d73](https://github.com/ggml-org/llama.cpp/commit/72e60f500d739eb9dc26e51cda5fdfe062faf766) 修正了 100 帧卷积分块、特征排列和每块位置编码重置。这里只回移其编码图；上游 PCM 预处理会按 800 帧再分窗，而现有程序使用整段后端注意力，因此不采用该 PCM 入口。

数据流如下：

1. Rust 保留当前 MelExtractor，得到按频带排列的 `[128, frames]` mel。
2. Rust 把 mel 值做与现有 FE 输入相同的 F16 往返转换，再作为 F32 传入窄接口。
3. C++ 补齐最后一个 100 帧块，执行分块卷积和逐块位置编码；进入 BE 之前按现有 `output_length` 删去尾部补齐 token，并做 FE 输出的 F16 往返转换。
4. BE 保留整段双向注意力。C++ 输出 F32 `[valid_rows, dimension]`；Rust 将结果做最终 F16 往返转换后交给现有 decoder。

存储权重只接受 F16/F32；F16 bias 和位置值在图中精确提升到 F32，保证 F32 激活及 CPU 回退路径可处理。没有新量化。ggml 内部运算和 ORT 仍有浮点实现差异，需记录 embedding 误差和转写结果。

## C ABI

声明见 [caps-encoder.h](caps-encoder.h)，Windows x64 使用 C 调用约定和 opaque handle。

| 函数 | 行为 |
| --- | --- |
| `abi_version()` | 返回 1 |
| `load(mmproj, device_name, threads, use_gpu)` | 加载音频编码器；返回 handle，失败返回 NULL |
| `dimension(handle)` | 返回投影输出维度，1.7B 预期 2048 |
| `output_rows(frames)` | `13*(frames/100)+ceil((frames%100)/8)` |
| `encode(handle, mel, frames, valid_rows, out, capacity_floats)` | 成功返回 0，参数或推理错误返回 -1 |
| `backend_name(handle)` | 实际主后端名称 |
| `last_error()` | 最近一次 load/encode 的线程局部错误文本 |
| `stats(handle, gpu, cpu, total)` | 最近一次图分配的各后端节点数量；成功 0 |
| `stats_detail(handle, gpu_core, cpu_core, gpu_metadata, cpu_metadata, unknown)` | 分开统计计算节点和 NONE/VIEW/RESHAPE/PERMUTE/TRANSPOSE 节点；成功 0 |
| `free(handle)` | 释放全部编码器资源 |

调用者先通过现有 llama API 注册匹配 b9090 的后端，再加载窄接口。接口不重新注册后端。GPU 模式默认严格选择 `Vulkan0`；指定设备不存在或初始化失败则失败，不静默改用 CPU。个别算子仍由 ggml 调度器按实际支持情况安排到 CPU；每次编码日志和 stats 会记录其位置。一个 handle 同时只允许一个编码调用。

旧版 `ggml_conv_2d` 展开为 `IM2COL` 和 `MUL_MAT`，已有 Vulkan 实现；不要求引入新版原生 CONV2D 算子。对于实际输入形状的后端分配，使用图分配后的节点统计。

## 复现

1. 准备包含对齐器和单输出预留补丁的精确 b9090 源码副本。模型加载器源码是有界上传修改前的版本，具体哈希以 manifest 为准。`ggml-vulkan-encoder-base.json` 固定检查 14 个关键源文件；不接受不同基础源码。
2. 从仓库运行：

```powershell
python tools/research/prepare_ggml_encoder_poc.py `
  --base-source <现有已验证原生源码> `
  --output-dir <新的隔离源码目录>

./tools/research/build_ggml_encoder_poc.ps1 `
  -NativeSource <隔离源码目录> -BuildDir <隔离构建目录> `
  -SdkInclude <Windows-SDK版本Include目录> `
  -SdkLib <包含um/x64和ucrt/x64的目录> `
  -SdkBin <Windows-SDK版本bin/x64目录> `
  -VcVars <vcvars64.bat> -CMake <cmake.exe> -Ninja <ninja.exe>
```

准备脚本校验基础文件，复制到新目录，先 `git apply --check` 再应用 [研究补丁](ggml-vulkan-encoder.patch)，生成 receipt。它不下载模型、不运行推理、不改正式运行目录。构建仅编译 `mtmd` 目标，输出名为 `caps_encoder.dll`。补丁含窄 ABI 源码、分块图修复和节点统计，沿用 llama.cpp MIT 许可。

3. 复制正式版本的整组匹配 native runtime 到隔离候选目录，仅新增 `caps_encoder.dll`。通过 dumpbin 检查依赖、导出，通过无模型 ABI 探针检查动态装载与行数函数，再开展编码和转写比较。无模型探针可运行 `python tools/research/probe_ggml_encoder_abi.py --runtime-dir <隔离运行目录> --output <证据JSON路径>`。

## 本机工程验证

- MSVC 14.44.35207、Windows SDK 10.0.28000.0、Release/Ninja 构建成功。
- DLL 的额外依赖仅现有 `llama.dll`、`ggml.dll`、`ggml-base.dll` 和系统 MSVC/UCRT；没有 CUDA、ORT 或新增 vendor DLL。
- 将正式 native runtime 原样复制到隔离目录、只新增本 DLL，动态装载成功；读回的四个模块都来自该隔离目录。
- ABI 1 和 0、1、8、9、99、100、101、200、800、5260、6001 帧的行数结果正确。
- 最终研究 DLL 为 873472 bytes，SHA256 `f4fa35166080a9a8a13fdb963d40b525e07a824d108532f686ad9c564eeb319f`。
- 首次完整候选诊断发现旧 clip 的 narrow ifstream 无法打开中文路径；权重读取现改为 `std::filesystem::u8path`，保留中文目录。
- 第二次完整候选诊断完成模型加载和 GPU 图分配后，发现 audio batch 标识遗漏；已补 `batch.is_audio=true`，避免走三通道图像输入处理。两个失败 run 都由研究报告保留，不计为成功推理。
- 无模型 ABI 探针没有加载模型或启动推理。完整数值、文本和资源对比记录由研究报告补充。

## 需要测量的项目

使用相同正式模型权重、相同输入音频、热词、seed 和 decoder，先比较 ORT CPU 与 ggml CPU / Vulkan 的 embedding 行数及数值误差，再比较最终识别文本、文件时间戳、冷暖耗时、峰值和稳态私有提交、工作集、独立显存与共享 GPU 内存。文件对齐器继续使用现有 ORT 实现。

当前统一的是编码器与解码器的 ggml/Vulkan 技术栈。窄接口仍将编码结果读回 CPU float buffer，再由现有 decoder 输入，尚未实施跨图零拷贝。
