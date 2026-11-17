# GGML / CUDA 先行版原生运行库

本轮三个名称固定如下：基准版是正式 0.6.0 的 CPU ORT 编码和 Vulkan 解码；预览版是固定 FE batch4 的 GGML / Vulkan 编码与 Vulkan 解码；先行版使用相同 batch4 GGML 编码图及模型权重，将编码和解码都交给 CUDA。

本页记录先行版原生运行库的准备方式。实时听写、热词、文件转录和托盘范围保持原样，不改模型、量化、文件对齐器、正式部署或 GitHub。

## 复用范围

先行版不是把历史 CUDA 安装包整包覆盖到新项目。正式 decoder 和 ggml core 保持逐字节一致，只新增已归档的 CUDA 插件及其必要依赖，并复用预览版的通用编码器 C ABI。

| 文件 | SHA256 |
| --- | --- |
| `llama.dll` | `8378440b27a7a1074261037a6ba33865aee59ef77f0b66c9a00458c82a14073a` |
| `ggml.dll` | `31ab732d47e93c8a101a795889d2bc1084e963713aef1dacb7ea7da4989ae25a` |
| `ggml-base.dll` | `f7e5c323d9a52888f0690271f206cd80ae23f657f867f3558759d7f5dfff2c16` |
| `caps_encoder.dll` | `848fcb250bb06d41ff6a0f662aa180bdfea9810d33ea62fc383a2edb69cfa038` |
| `ggml-cuda.dll` | `680b659b0792c9f0699ca9a3c581ba871df0a8bb1acbfc59b8008536a281926c` |
| `cudart64_12.dll` | `c2c9a9c22a9bcba90e261825968836787b331038047a26770cffb7a583c28344` |
| `cublas64_12.dll` | `9513540e4ec4c51ee9e7304138c2cc255c29a8c181f9e80c38efa25738becd99` |
| `cublasLt64_12.dll` | `b199d1ff892a81b7fd3d57ba1781549609b41500b36008fef326038393ad46c7` |

CUDA 插件仍是 b9090 的 SM120a 构建，原 136 个 CUDA 对象只做了 cuBLAS 延迟链接，没有改变内核或公开 ABI。旧 `build.ninja` 中含 `compute_120a` 和 `sm_120a`，归档 relink receipt 与当前插件哈希一致。构建针对本机 RTX 5060 Laptop；其它 GPU 架构需重新确认。

归档 CUDA 运行库原先使用的 `ggml.dll`、`ggml-base.dll` 也与本次正式 core 哈希完全相同，进一步核实了复用关系；这两个归档文件没有覆盖正式版本。

`caps_encoder.dll` 没有为先行版重编。它按调用方传入的 `device_name` 选择 ggml 后端，CUDA 使用 `CUDA0`。100 帧分块、固定 batch4、F16 边界和整段 BE 注意力均不改变；实际数值和识别文本由新后端比较单独记录。

CUDA-only 运行目录保留匹配的 CPU 后端和 OpenMP，用于现有文件对齐器，省略未使用的 `ggml-vulkan.dll`。此轮没有引入 ORT CUDA、cuDNN 或 cuFFT。

## 依赖与接口检查

已核对的本地导入边：CUDA 插件向正式 `ggml-base.dll` 导入 41 个符号，向 cudart 导入 56 个，向 cuBLAS 延迟导入 11 个；编码器 DLL 向 llama、ggml、base 分别导入 15、4、149 个；cuBLAS 向 Lt 导入 75 个。这些符号在目标 DLL 中全部存在。

Vulkan 预览版使用的主要计算图操作在 b9090 CUDA 源码中已有实现，包括矩阵乘法、IM2COL、GELU_ERF、F16/F32 转换、归一化及张量排列。具体输入图的 GPU/CPU 分配仍由先行版实测日志记录。

独立无模型子进程已成功装载编码器、CUDA 插件、cudart、cuBLAS 和 Lt；ABI 1、11 组有效行数和 11 个 cuBLAS 符号地址检查通过。此检查没有调用后端初始化、cuBLAS Create/GEMM 或加载模型；实际 CUDA 推理及 cuBLAS 数值路径由统一串行基准执行。当前没有发现缺少的 DLL 或导出符号。

FP16 编码器可能使用 cuBLAS 路径，因此即使历史量化 decoder 没有装载 cuBLAS，本次也保留完整 cuBLAS/Lt 依赖。

## 复现与搜索路径

使用 [stage 工具](../tools/research/stage_ggml_cuda_native.py) 准备新的隔离目录：

```powershell
python tools/research/stage_ggml_cuda_native.py `
  --installed-native <正式0.6.0的runtime/llama> `
  --cuda-plugin <归档delay-cublas/ggml-cuda.dll> `
  --vendor-dir <含固定cudart和cuBLAS/Lt的归档目录> `
  --encoder-plugin <预览版batch4的caps_encoder.dll> `
  --output-dir <新的先行版native目录> `
  --evidence-dir <原生准备证据目录> `
  --dumpbin <dumpbin.exe>
```

工具固定核对上述关键哈希，拒绝覆盖已有运行目录，检查导入/导出，并在独立子进程做无模型装载探针。它不编译、不加载模型、不运行推理，也不改正式路径。

运行时应将这些 DLL 放入先行候选 `app_dir/runtime/llama`，并保留 worker 的私有 PATH 前置逻辑。`LoadLibraryEx` 的一次性依赖搜索不自动覆盖之后的 delayimp 搜索；只把 `YYUR_GGML_NATIVE_DIR` 指向另一个目录而不处理 worker PATH，可能导致首次实际使用 cuBLAS 时找不到依赖。不能同时装载两套不同目录的 ggml core，现有 `Api::get` 会拒绝这种情况。

本次原生准备逻辑总载荷为 843788856 bytes。模型不在该目录中；模型和量化保持与基准版、预览版一致。
