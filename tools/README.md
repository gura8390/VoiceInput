# 开发工具目录

这些脚本用于构建、验证、模型准备和复查研究，不参与日常听写。正式程序只从应用目录运行。工具的项目根目录及当前运行布局集中在 `project_layout.py`；研究输出保存在本项目被 Git 忽略的 `evidence/` 中，历史测量文件保持原样。

## 构建与部署

`build/` 保存 SDK、原生补丁准备与事务部署工具：

| 文件 | 用途 |
| --- | --- |
| `bootstrap_sdk.py` | 下载并校验固定版本的官方 SDK 导入库，保存到共享构建环境；可用 `YYUR_SDK_LIB_DIR` 指定其它位置 |
| `prepare_asr_single_output.py` | 从已接受的原生源码准备 ASR 单输出补丁与源码校验记录 |
| `build_asr_single_output.ps1` | 使用显式指定的 SDK/C++ 工具链构建该原生补丁 |
| `deployment_transaction.py` | 逐项核对、备份、部署及回退；必须提供具体计划，不自动发现或改动正式安装 |

应用构建仍使用项目根目录的 `build.ps1`。原生补丁命令见 [原生库说明](../native/README.md) 及该目录其它说明。

## 验证

`validation/` 保存沙盒测试与需要明确参数的原生探针：

- `test_*.py`：事务部署、资源测量门槛、线程调度及基准编排的模拟/沙盒回归。
- `verify_asr_single_output.py`：检查原生 ABI、符号和无模型边界。
- `asr_output_guard_probe.py`：真实模型的输出行数边界及回归探针。
- `cuda_blas_smoke.py`：隔离 CUDA/cuBLAS 正确性烟测。
- `_test_paths.py`：供测试发现导入实际工具组；不复制实现。

从项目根目录执行：

```powershell
python -m unittest discover -s tools/validation -p 'test_*.py'
.\build.ps1 -Tests
```

普通 Python 测试覆盖沙盒或模拟，不加载语音模型。原生探针使用明确指定的模型、候选 DLL 和输出路径，分别记录真实推理结果。

## 模型准备

`models/prepare_external_weights.py` 使用 Python 标准库将内嵌 ONNX 权重按原字节转存到外部 sidecar。它不加载 ONNX Runtime、不创建 Session、不修改精度，也不执行音频推理。元数据扫描会跳过大权重内容；转换时按块复制和校验。

```powershell
python tools/models/prepare_external_weights.py --source 输入目录/encoder.onnx
python tools/models/prepare_external_weights.py --source 输入目录/encoder.onnx --output 输出目录/encoder.onnx --output-root 输出目录 --expected-source-sha256 已核对的源文件SHA256
```

转换必须显式指定 `--output-root`，源目录与输出树相互分离。工具拒绝覆盖任何既有 ONNX、sidecar 或收据，拒绝父目录穿越、符号链接和目录联接等重解析路径。输出分别为原文件名的 ONNX、同前缀 `.weights.bin` 及 `.external.json` 收据。中途失败时不会生成成功收据；已经创建的部分输出保留供检查，下一次转换需使用新输出位置。

收据逐 tensor 记录 dtype、维度、payload SHA256，并核对其余模型字段、节点、属性和原始文件前后身份。工具不假定固定 tensor 数量，可用于 ASR 前后端及文件对齐器。量化权重、scale 和零点按原始序列化字节保留。执行输出与资源变化由独立编码和完整转写对照记录。

`validation/test_external_weights.py` 覆盖合成权重转换、输入缺失、输出碰撞、源目录交叉、路径边界、重解析点和并发独占写入。

## 研究

`research/` 保存已做过的 CPU、Vulkan、CUDA 与模型对照工具：

| 工具组 | 文件 |
| --- | --- |
| 进程树、GPU 观测与共同测量 | `bench_common.py`、`cuda_metrics.py` |
| CPU 编码、模型与后端对照 | `bench_encoder.py`、`bench_asr_models.py`、`bench_cuda_route.py`、`cpu_qos.py` |
| 前端批处理候选准备 | `prepare_batched_frontend.py` |
| 隔离 CUDA 运行目录准备 | `stage_cuda_research.py` |
| ORT CUDA 下载、依赖检查与准备 | `prepare_ort_cuda_runtime.py`、`prepare_ort_cuda_dependencies.py`、`inspect_ort_cuda_closure.py`、`stage_ort_cuda_encoder.py` |
| ORT profile 汇总 | `summarize_ort_profile.py` |
| GGML 编码器等值权重与隔离接入 | `prepare_ggml_encoder.py`、`stage_ggml_encoder_candidate.py`、`ggml_encoder_candidate.rs`、`ggml_route_config_tests.rs` |
| GGML 原生构建、ABI 与浮点检查 | `build_ggml_encoder_poc.ps1`、`probe_ggml_encoder_abi.py`、`compare_ggml_embeddings.py` |
| GGML Vulkan / CUDA 三版对照 | `stage_ggml_cuda_native.py`、`bench_ggml_three_routes.py`、`summarize_ggml_three_routes.py`、`quality_ggml_routes.py` |
| GitHub 原 CUDA 分支与同模型路线对照 | `bench_historical_cuda_branch.py`、`bench_previous_cuda_route.py`、`summarize_cuda_controls.py`、`summarize_historical_cuda.py`、`verify_cuda_control_summaries.py` |

研究工具现按 `config/` 和 `runtime/` 目录运行。CUDA 候选 ORT 使用独立的 `runtime/onnx-cuda/`，CPU ONNX 目录仍为 `runtime/onnx/`。已有旧版测量记录只作为历史证据；重新测量需准备符合当前布局的隔离运行目录、候选二进制和匹配参考，不能将旧记录路径当作当前默认路径。

模型、私人音频、热词、转写结果、运行库与本机测量文件均不属于源码发布载荷。[优化历史](../docs/optimization-history.md) 记录采用或停止的技术路线。

GGML 研究工具只把适配器放入指定的新 detached worktree，在本地 `evidence/` 中准备候选。正式程序保持原样。三版名称、同权重条件、采样口径和实际文件质量差异见 [三版报告](../docs/ggml-three-versions-comparison.md)；CUDA 原生载荷身份和重现方式见 [原生说明](../native/ggml-cuda-encoder.md)。
