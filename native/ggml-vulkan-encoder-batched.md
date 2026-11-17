# GGML 编码器固定前端小批研究

## 方案与当前结论

第一候选 `f4fa3516…` 已完成 GGML / Vulkan 编码器接入，但整段前端卷积保留了较大的 GPU 计算池。第二候选保留同一编码器、解码器、权重精度和整段 BE 注意力，只将 FE 改为固定 **4 个 100 帧块**的小批计算。

编码器模型、权重和后端始终常驻。FE 和 BE 使用同一 `clip_ctx`、同一 Vulkan backend 和同一 ggml scheduler；没有复制两套权重或计算池，也没有改成模型分阶段加载。

**实际建议只使用 batch4。** `YYUR_GGML_ENCODER_FE_BATCH=1` 是保留失败证据用的诊断开关：60 秒样本中四个输出字段及 token 发生变化，峰值私有提交只比 batch4 再少约 36 MiB，已拒绝采用。默认值为 4；未知值严格拒绝。

父端诊断已记录：5 组 batch4 embedding 与第一候选逐 float 完全一致，五项输出字段和 raw 文本也一致。结论限于本次输入集。第一候选作为对照保留。

## 内存来源与调整

旧版 `ggml_conv_2d` 展开为 `IM2COL + MUL_MAT`。对约 60 个块同时计算时，第二层 FP16 im2col 约 395.5 MiB，首层 FP32 激活约 351.6 MiB，还需要后续排列、输出和其它活跃张量。这是根据形状计算的单个张量大小，不是进程占用测量值。

`ggml-alloc.c` 的分配器只在新图需要更大缓冲时重分配，较小图不缩容。因此整段 FE 的高水位会留在编码器 scheduler 中。第二候选把这一高水位变成 `max(FE固定小批, 整段BE)`。

具体数据流：

1. FE 每次固定接收 `[128, 400]` mel；不足的尾批补零。
2. 保留逐 100 帧卷积、逐块位置重置、有效 token 裁剪和 FE 输出的 F16 往返边界。
3. 每批结果读回一个约几 MiB 的 host hidden buffer，按原顺序拼接。
4. BE 对全部有效 hidden 一次计算，保持原有全局双向注意力，再交给现有 decoder。

这一步仍有 FE 输出及 BE 输入的 CPU 往返。避免额外跨图 GPU 资源接口，使本次变更只涉及计算图工作区。

## 诊断观察

下列为父端诊断数据，用来定位结构内存来源；正式多轮速度和内存比较由总体研究报告汇总。

| 项目 | 第一候选：全部块 FE | 第二候选：batch4 |
| --- | ---: | ---: |
| 峰值私有提交 | 约 4265 MiB | 约 3218.6 MiB |
| 尾部私有提交 | 约 3986 MiB | 约 2940.4 MiB |
| 尾部工作集 | 约 850 MiB | 约 834.4 MiB |
| 独立显存峰值 | 约 3158 MiB | 约 2119 MiB |
| 60 秒编码图耗时 | 约 0.326 秒 | 约 0.1218 秒 |
| batch4 调度计算池 | — | 约 77.44 MB |
| batch4 FE / BE 计算节点 | — | FE 405 GPU / 0 CPU；BE 778 GPU / 0 CPU |

相对 ORT CPU 基线，batch4 的工作集下降，但私有提交仍增加约 700 MiB。clip loader 按 tensor 读取临时 `read_buf` 并在加载结束销毁，没有保留整份 600 MiB CPU 权重副本。剩余差额与约 605.8 MiB GPU 编码器权重、计算池及驱动资源同量级；这解释了优先调查方向，具体驱动 host commitment 的组成还需专门跟踪才能定量拆分。

b9090 Vulkan 的设备能力将 `buffer_from_host_ptr` 标为 false，当前没有可直接启用的稳定文件映射 GPU 权重入口。单纯增加 clip mmap 主要改变加载和文件页行为，不能直接消除 GPU 常驻资源。权重流式上传或自行维护 external-memory 导入会增加 PCIe 传输和原生实现成本，本轮没有引入。

## 复现

先按 [首个候选说明](ggml-vulkan-encoder.md) 准备精确基础源码，再增加 `--frontend-batched`：

```powershell
python tools/research/prepare_ggml_encoder_poc.py `
  --base-source <固定b9090源码及原有对齐和单输出补丁> `
  --output-dir <新的batch4隔离源码目录> `
  --frontend-batched
```

该操作先应用首个候选补丁，再核对 [batch4 基础文件哈希](ggml-vulkan-encoder-batched-base.json)，应用 [小批补丁](ggml-vulkan-encoder-batched.patch)。构建仍使用 `tools/research/build_ggml_encoder_poc.ps1`，输出目录必须与首个候选分开。

运行目录仍复制正式匹配 native runtime，只新增第二候选的 `caps_encoder.dll`。不替换 decoder、ggml 或 Vulkan 后端 DLL。正式应用和主源码没有因此改动。

本次构建：878080 bytes，SHA256 `848fcb250bb06d41ff6a0f662aa180bdfea9810d33ea62fc383a2edb69cfa038`。无模型 ABI、依赖动态装载、11 组行数检查和非法 batch2 拒绝检查通过。原生分工没有运行模型；模型诊断和正式多轮比较由父端顺序执行。

## 诊断接口

原 ABI 1 的函数签名保持不变；节点统计是一次语句中所有 FE 调用和 BE 调用的执行次数合计。附加定义见 [caps-encoder-batched.h](caps-encoder-batched.h)。

每条语句输出一条 `YYUR_GGML_NATIVE_STAGES ` 前缀 JSON 日志，包含：

- `batch`、`frames`、`rows`、`fe_calls`；
- `fe_seconds`、`be_seconds`、`total_seconds`，单位秒；
- `gpu_compute_bytes`、`cpu_compute_bytes`，同一 scheduler 的计算池高水位字节数；
- `fe_gpu_core`、`fe_cpu_core`、`be_gpu_core`、`be_cpu_core`。

另有可选 `caps_ggml_encoder_get_stage_metrics(handle, stage, out)`，stage 1 是 FE，2 是 BE；返回计算、元数据及未分类节点、图调用次数、计算池字节和阶段秒数。不要求修改现有 Rust EXE 即可从日志提取这些信息。
