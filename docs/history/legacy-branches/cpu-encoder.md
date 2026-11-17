# CPU 编码器研究归档

`perf/cpu-encoder` 的最终提交为 [`b05102b2d90d20a34737c51cb2ef0490479e0115`](https://github.com/gura8390/YuYbUuRu/commit/b05102b2d90d20a34737c51cb2ef0490479e0115)，对应 0.4.4。该提交已经是当前 `main` 的祖先；本节使用不可变提交链接，分支名称删除后仍可查阅原实现。本轮采用 CPU 前端固定形状 arena，其余调参保留为研究入口。它属于 2026 年 10 月 27 日的 **Qwen3-ASR 0.6B CPU ORT 编码＋Vulkan 解码**研究，与后来的 1.7B、GGML/Vulkan、GGML/CUDA 实验分别统计。

## 采用方案：只复用固定形状前端缓冲

前端每块输入固定为 `[1,128,100]`，输出为 `[1,13,896]`。`encoder_frontend_arena=true` 让 ONNX Runtime 复用这些固定形状的 CPU 分配。动态长度后端继续 arena-off，文件对齐器也保留独立的 arena-off；对齐器创建时显式清除从 ASR 继承的六项 CPU 实验参数。模型、int4 权重、FP32 激活、F16 KV、采样和音频切分均保持原样。

采用配置为 ORT intra/inter 线程 **8/1**、llama 线程 **1/1**，`encoder_frontend_threads=null` 继承 8，`encoder_backend_arena=false`、`encoder_parallel_execution=false`、`encoder_spin_wait=false`、`encoder_dynamic_block_base=0`。产品保留系统调度，不设置 HighQoS、亲和性、进程优先级或系统电源计划。发行示例显式启用前端 arena；旧配置缺少该字段时默认仍为 `false`，没有默默改变旧配置。

实现见 [CPU session 配置](https://github.com/gura8390/YuYbUuRu/blob/b05102b2d90d20a34737c51cb2ef0490479e0115/src/engine/encoder.rs#L114)、[对齐器参数隔离](https://github.com/gura8390/YuYbUuRu/blob/b05102b2d90d20a34737c51cb2ef0490479e0115/src/engine/aligner.rs#L70)、[发行示例](https://github.com/gura8390/YuYbUuRu/blob/b05102b2d90d20a34737c51cb2ef0490479e0115/settings.example.json#L5)和[原提交说明](https://github.com/gura8390/YuYbUuRu/blob/b05102b2d90d20a34737c51cb2ef0490479e0115/README.md#L58)。

## 最终三轮完整服务对照

同一最终 0.4.4 EXE（SHA256 `0561d50c100b21d4b0e19ac33e50ae98dbbb2b029a198aaee8ad4be21c342374`）、同机 i9-14900HX / RTX 5060 Laptop 8 GB / 驱动 573.24、CPU ORT 1.24.4，相同 39 请求、每组 3 轮交错。每轮音频合计 551.25 秒，含 30 条标准样本、4 次短句重复、两条 68 秒及三条 80 秒压力样本。最终资源表关闭分段计时和 ORT profiling。

| 指标：每轮统计量的三轮中位数 | 前端 arena 关闭 | 前端 arena 开启 | 变化 |
|---|---:|---:|---:|
| 累计编码墙钟秒 | 9.442584 | 8.503537 | 减少 0.939047 秒，9.94% |
| 39 次完整请求累计秒 | 16.647778 | 15.715232 | 减少 0.932546 秒，5.60% |
| ready 至最后请求完成的进程树 CPU 秒 | 66.156250 | 61.406250 | 减少 4.750000 秒，7.18% |
| 私有提交采样峰值 MiB | 1209.8125 | 1243.4023 | 增加 33.5898 MiB，2.78% |
| 队列尾私有提交 MiB | 1002.5000 | 1035.8594 | 增加 33.3594 MiB，3.33% |
| 私有驻留采样峰值 MiB | 372.7930 | 374.0391 | 增加 1.2461 MiB，0.33% |

编码三轮原值为 `9.4076324 / 9.4425836 / 9.6536685` 秒，对照采用组 `8.4234856 / 8.5035370 / 8.5917243` 秒。收益来自减少重复分配，代价是保留约 33.6 MiB 额外私有提交；这轮是有内存成本的提速，并非内存下降方案。

`t_encode` 是完整 `AudioEncoder::encode` 的墙钟累计，包含 Mel、前端、后端和复制，排除加载静音 warmup、prompt、decoder、IPC、文件对齐器编码。完整响应是 39 次 IPC 的累计，排除请求间采集器观察开销。这里的 CPU 是 `ready` 与 `after_38` 两个阶段点的 service/worker 树 CPU 累计差值，区别于后续新工具的全生命周期 Job CPU。主机内存按目标 25 ms 采样，私有提交、私有驻留和总工作集各自统计；显存仅在请求完成阶段按 PID/适配器读取，尾部专用显存两组均为 647.6016 MiB，没有把阶段值写成瞬时峰值，也没有与主机提交相加。

六份最终报告各 39 条，合计 234 条；两组各 117/117 条的五个最终字段及五个 raw 字段回归精确。独立 CPU 探针另核完整 float32 embedding 的 SHA 和形状。文件组各 7 条结果精确，正式路径两份媒体的 TXT/JSON/SRT 六个哈希一致；12 秒对齐器释放及 123 秒整 worker 退出/重载记录通过。质量结论是固定输入、现有量化模型的回归一致；该轮没有新增人工 WER 测评。

## 筛选过但没有采用的方案

以下调参筛选使用同一 0.6B 的 39 条 CPU-only 编码请求，通常每候选单轮；它不含 decoder。基线筛选本身为 8.9482～9.4964 秒，不能把单轮最快值当作最终三轮完整服务收益。所有列出的筛选候选均保留完整 embedding 精确回归记录。

| 方向 | 实际试法与旧记录 | 未采用原因及保留状态 |
|---|---|---|
| 前端线程 | 前端 1/2/4 线程：25.9924 / 16.3523 / 11.6232 秒；12/16：9.3354 / 9.5251 秒；加前端 arena 后 12/16：8.1791 / 8.2119 秒 | 低线程明显变慢；更高线程增加 CPU 工作量/保留分配，没有形成更好的资源与速度组合。保持产品 8/1，独立前端线程覆盖只保留为选项。 |
| 后端线程 | 固定前端 8，后端 12/16 线程：9.0978 / 8.9349 秒；探针全进程采样 CPU 为 77.6563 / 84.8125 秒 | 单轮编码收益小于增加的 CPU 工作量，未采用。这里的探针 CPU 包含载入、输入预检和输出哈希，是采样下界，未与服务表 CPU 混算。 |
| Run 内自旋 | `allow_spinning=1`，同时 `force_spinning_stop=1`，只在 Run 内保持线程活跃：9.5743 秒、采样 CPU 77.5 秒 | 首轮编码与 CPU 都更差，保持关闭。没有通过持续忙等换响应。 |
| 动态任务分块 | `session.dynamic_block_base=4`：9.4631 秒；叠加自旋：9.8567 秒、采样 CPU 80.1406 秒 | 与基线筛选相比没有形成值得采用的收益，保持 0。它是 ORT 线程任务分块，区别于 ASR 动态上下文。 |
| 后端 arena / 每次 Run 后 shrink | 后端 arena：9.1089 秒、峰提交 764.86 MiB；后端 arena＋shrink：9.4432 秒、733.32 MiB；前后端 arena＋shrink：8.5248 秒、766.11 MiB；基线约 379～381 MiB | 动态后端长音频分配保留量明显增加；shrink 没有恢复原提交优势。只给固定形状前端开 arena。收缩只接入动态后端 RunOptions，没有让每个固定前端块额外扫描。 |
| 后端图分支并行 | ORT_PARALLEL，后端 intra/inter 为 4/2、4/3、8/2；前端保持 8。编码 14.6058 / 14.2867 / 11.8723 秒；并行＋arena/shrink 为 11.3067 秒 | Q/K/V 有独立分支，但此模式需要关闭 ORT memory pattern。实测更慢，保留串行执行；没有把初始化阶段的并行载入与此处图内并行混为一谈。 |
| 自有进程 HighQoS | CPU-only 三组各 3 轮：base 编码中位 9.384919 秒；arena 8.273627 秒；HighQoS＋arena 8.133873 秒 | arena 以外额外仅减少 1.69%，不足以引入产品调度策略，产品接入撤回。诊断工具仅操作自己创建的 Popen 句柄，保留其它 power-throttling 位；不枚举并修改外部进程。三轮读回均从 ControlMask/StateMask 的 0/0 变为 1/0；0/0 表示系统管理，不是已观测到 EcoQoS 限频。 |

前端 CPU 批处理另用**单条 80 秒压力音频**进行专用探针，不属于上述 39 请求三轮服务表。独立模型副本只改公共 batch 维度和 Reshape batch 常量，权重与 11 个算子保持原字节；B1/B2/B4/B8 的逐 float 位结果均精确。前端热 Run 时间分别为 **0.5254296 / 0.5116563 / 0.5355561 / 0.6091622 秒**。B2 只让该前端子阶段少 2.62%，B4/B8 更慢，因此没有为此替换产品 ONNX 图，也没有宣称完整编码或完整响应快 2.62%。这是旧 CPU ONNX 动态 batch 实验，与后来 GGML 前端 batch4 属于不同实现和测量组。

另一个排除依据来自定位诊断：Mel 约 0.121 秒，前端 Run 约 3.866 秒，后端 Run 约 4.966 秒，打包/收集/复制合计不足 0.02 秒。该诊断用于定位，不作为最终关闭计时组的分母。没有为了小占比重写 f64 FFT、每 band 累加顺序或完整音频 global max。新版 ORT/不同精度路径在该轮只做源码评估；没有本轮完整 A/B 数字，归档应标为**未量化**，不写成已实测收益。

## 可复查的代码与本地证据

不可变提交中的研究入口：

- [encoder_probe.rs](https://github.com/gura8390/YuYbUuRu/blob/b05102b2d90d20a34737c51cb2ef0490479e0115/examples/encoder_probe.rs)：CPU-only、同模型/ORT/PCM 身份、完整 embedding SHA/形状。
- [bench_encoder.py 的口径](https://github.com/gura8390/YuYbUuRu/blob/b05102b2d90d20a34737c51cb2ef0490479e0115/tools/bench_encoder.py#L138)：Owned Job、25 ms 采样、超时和清理；探针 CPU 与完整流水线 CPU 分开。
- [CPU session 的线程/arena/parallel 配置](https://github.com/gura8390/YuYbUuRu/blob/b05102b2d90d20a34737c51cb2ef0490479e0115/src/engine/encoder.rs#L114)、[自旋与 dynamic_block](https://github.com/gura8390/YuYbUuRu/blob/b05102b2d90d20a34737c51cb2ef0490479e0115/src/engine/encoder.rs#L152)、[arena shrink](https://github.com/gura8390/YuYbUuRu/blob/b05102b2d90d20a34737c51cb2ef0490479e0115/src/engine/encoder.rs#L267)。
- [prepare_batched_frontend.py](https://github.com/gura8390/YuYbUuRu/blob/b05102b2d90d20a34737c51cb2ef0490479e0115/tools/prepare_batched_frontend.py)、[frontend_batch_probe.rs](https://github.com/gura8390/YuYbUuRu/blob/b05102b2d90d20a34737c51cb2ef0490479e0115/examples/frontend_batch_probe.rs)：修改独立副本并逐 float 位比较，不接产品推理路径。
- [cpu_qos.py](https://github.com/gura8390/YuYbUuRu/blob/b05102b2d90d20a34737c51cb2ef0490479e0115/tools/cpu_qos.py#L39)：只操作自有进程的 EXECUTION_SPEED，并读回状态。

本地证据入口相对项目根目录：

- `evidence/final/CPU编码器测量汇总.json`：`primary` 为最终两组三轮；`cpu_only_paired` 为三组探针三轮；`screening` 为 28 个筛选报告；`frontend_batch` 为 80 秒批处理结果。
- `evidence/final/CPU编码器优化报告.txt`：保留采用原因、全部未采用方案及其测量口径。
- `evidence/bench/final-v044-{control,frontend-arena}-r{1,2,3}.json`：六份原始完整流水线报告。本次逐份复核其 SHA 与汇总一致，各 39 请求，`passed/complete/outputs_exact/cleanup_confirmed` 均 true；线程与其它实验配置固定，前端 arena 是两组唯一算法配置差异。
- `evidence/bench/screen-*.json` 与对应 `.probe.json`：筛选配置、阶段耗时、完整 embedding 回归。
- `evidence/bench/paired-encoder-{base,arena,qos-arena}-r{1,2,3}.json`：HighQoS 三组各三轮记录，区分基线系统管理状态与显式 HighQoS。
- `evidence/bench/frontend-batch-stress.json`：80 秒输入、原模型/修改副本身份、B1/2/4/8 分阶段耗时及浮点一致性。

本页整理原有源码与测量记录，本次归档没有重新运行性能测试。
