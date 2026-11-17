# CPU 编码器与 Vulkan 调度研究归档

研究日期：2026-10-27。归档核对日期：2026-11-18。

这份记录保存已停止的低收益研究及其不利结果。该研究基于旧版 0.4.4、Qwen3-ASR 0.6B、CPU ORT 编码器（INT4 权重、FP32 激活）和 Vulkan 解码器；与后来 1.7B 模型、GGML GPU 编码器的研究分开统计。本轮新增采用项为 **0 项**，实验候选未部署，继续使用当时的正式 0.4.4。

## 1. 分支与源码身份

- 分支：`perf/cpu-vulkan-scheduling`。
- 研究提交：`5d2c228d1e7237073d5af7541ca93bef8a75530f`，原始差异见 [归档补丁](cpu-vulkan-scheduling.patch)，提交说明为 `Record CPU and Vulkan scheduling experiments without changing deployment`。
- 父提交及与 main 的共同基点：[`b05102b2d90d20a34737c51cb2ef0490479e0115`](https://github.com/gura8390/YuYbUuRu/commit/b05102b2d90d20a34737c51cb2ef0490479e0115)，即 `Reuse CPU frontend buffers on the Vulkan route`。
- 分支差异共 5 个文件、112 行新增，没有生产 Rust 源码改动：`README.md`、`native/README.md`、`native/vulkan-fence-schedule-research.patch`、`tools/bench_cuda_route.py`、`tools/test_vulkan_schedule.py`。
- 原生补丁目标为 llama.cpp b9090，并接在已有 Vulkan 持久化管线缓存补丁之后；没有修改 shader、模型、采样或数值计算规则。

原分支的完整 5 文件差异已作为 [归档补丁](cpu-vulkan-scheduling.patch) 保存，并验证可从父版本重建原源码树。分支名用于说明历史来源，后续查阅以本文、归档补丁与静态源码快照为准。

## 2. 尝试清单与停止理由

| 方向 | 实际实施 | 结果与取舍 |
| --- | --- | --- |
| CPU 前端 FE 双会话并行 | 独立 FE 会话轮转处理 100 帧块，总 intra 线程预算仍为 8；BE 保持 8 线程，等待所有 FE 后按原序合并 | 同原型三轮中位累计编码慢 4.21%，私有提交峰值增加 49.36 MiB；未采用 |
| CPU 前端 FE 四会话并行 | 四会话各 2 线程，只有一轮筛查 | 累计编码 8.7292 秒，私有提交峰值 565.50 MiB，均高于单会话三轮中位；没有继续扩大测试 |
| Vulkan 独立 transfer queue | 使用 b9090 已有开关；验证实际初始化策略 | 完整 IPC 慢 1.59%，未采用 |
| Vulkan graphics-capable compute queue | 使用 b9090 已有开关；实际 compute queue family 从 2 改为 0 | 两轮中位 IPC 慢 2.96%，未采用；轮数低于其他候选 |
| Vulkan 关闭 async | 使用 b9090 已有开关；验证 async 实际关闭 | 生成阶段快 3.44%，但 prefill 慢 102.89%，完整 IPC 慢 2.32%，未采用 |
| Vulkan 最终 fence 阻塞等待 | 研究 DLL 改用无限超时 `waitForFences`，保持队列顺序及 fence reset | 完整流水线 CPU 秒减少 1.38%，但 IPC 慢 2.58%，未采用 |
| Vulkan 最终 fence 短自旋 | 研究 DLL 把每轮轮询的 pause 从 1000 次缩为 10 次 | 生成阶段快 1.04%，但 IPC 慢 0.16%、CPU 秒增 1.52%，未采用 |
| 固定 decoder token 重放 | 不运行编码器与随机采样，比较固定输入、固定 token 序列 | 仅为诊断补充；热重放中的小变化未形成完整 ASR 响应收益 |

上表百分比采用对应实验阶段的独立轮次中位值，不按最快一轮挑选。研究保留慢结果；不是把所有未来调度方案判为无效。

## 3. 780、351 与 35 的准确含义

- **完整 ASR 780 次**：V2 为 14 组 × 39 = 546 次，V3 为 6 组 × 39 = 234 次。每组固定 39 条、551.25 秒音频；统计单位是独立运行组，39 条内部输入按固定顺序与种子执行。
- V2 包括 control、transfer、blocking、sync 各 3 组，graphics 2 组。V3 包括 control 与 short-spin 各 3 组。V2/V3 属于不同构建与实验阶段，分别设 control，不能混为同一基线。
- 完整 ASR 每组最终输出的 5 个字段与原始输出的 5 个字段逐项精确一致；同时检查实际 GPU offload/compute、GPU 采样、实际 queue/fence 策略与进程清理。
- **独立 CPU 编码 351 次**：9 组 × 39；workers1/2 各 3 组，workers4 1 组，旧正式 product 探针 2 组。每条编码器输出形状与完整 float32 embedding SHA 一致。该探针不运行 decoder 或采样，不计入 780 次完整 ASR。
- **固定 decoder 35 次重放**：5 个独立进程组，各 7 次内部重复。只有两个 control 进程，其他实验各一个；同进程内部重复相关，不能把它们写成 35 次独立统计实验。warm6 是去除首轮后的描述性中位值，全部 7 次原值仍保留。
- 上述精确输出回归只覆盖固定输入及种子，未评价未见语音的普遍 WER。
- `evidence/scheduling/control-r1.json` 是早期探索运行，原生编译与其重叠或时间接近，已按 `first-run-note.json` 排除在正式配对统计和 780 次计数之外。

## 4. 完整流水线结果：保留所有不利数值

以下为每个候选独立运行组的中位数。IPC、编码、prefill、生成均为 39 条请求累计墙钟秒。CPU 为服务与 worker 从 ready 到最后回复的 user+kernel CPU 秒差；私有提交是 25 ms 采样峰值，单位 MiB，并非工作集。显存为阶段采样的最大 dedicated 值，不是连续瞬时峰值，也不与主机私有提交相加。

### V2：队列与阻塞等待

| 候选 | 轮数 | IPC 秒 | 编码秒 | Prefill 秒 | 生成秒 | CPU 秒 | 私有提交峰 MiB | GPU 阶段最大 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 3 | 15.7219 | 8.5159 | 0.4340 | 6.6989 | 62.500 | 1243.32 | 647.60 |
| transfer | 3 | 15.9717 | 8.7006 | 0.4301 | 6.7719 | 63.422 | 1244.20 | 647.60 |
| graphics | 2 | 16.1873 | 8.6832 | 0.4373 | 6.9893 | 63.195 | 1243.82 | 644.57 |
| blocking | 3 | 16.1281 | 8.7595 | 0.4325 | 6.8735 | 61.641 | 1242.89 | 647.60 |
| sync | 3 | 16.0868 | 8.6681 | 0.8805 | 6.4682 | 63.250 | 1243.55 | 647.60 |

相对同阶段 control：

| 候选 | IPC 变化 | Prefill 变化 | 生成变化 | CPU 变化 |
| --- | ---: | ---: | ---: | ---: |
| transfer | +1.59% | −0.89% | +1.09% | +1.48% |
| graphics | +2.96% | +0.76% | +4.34% | +1.11% |
| blocking | +2.58% | −0.34% | +2.61% | −1.38% |
| sync | +2.32% | +102.89% | −3.44% | +1.20% |

负值表示耗时或 CPU 秒减少，正值表示增加。不能把单个阶段变快直接记作完整响应变快。

实际日志：默认、transfer、blocking、sync 的 compute queue family 为 2，graphics 为 0；transfer queue family 均为 1，`single_queue=false`。transfer 确实启用 async transfer，sync 确实关闭 async，blocking 确实使用最终 fence 阻塞等待。显存测量逐阶段核对 owned worker 仅一个非零 adapter 实例，其他实例实测为零。

### V3：最终 fence 短自旋

| 候选 | 轮数 | IPC 秒 | 编码秒 | Prefill 秒 | 生成秒 | CPU 秒 | 私有提交峰 MiB | GPU 阶段最大 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 3 | 15.8835 | 8.6441 | 0.4317 | 6.7282 | 62.734 | 1243.74 | 647.60 |
| short-spin | 3 | 15.9089 | 8.7279 | 0.4287 | 6.6580 | 63.688 | 1243.50 | 647.60 |

short-spin 相对 V3 control：IPC **+0.16%**、prefill **−0.70%**、生成 **−1.04%**、CPU **+1.52%**。私有提交中位峰差仅 −0.24 MiB，显存阶段最大相同，没有形成值得部署的资源或完整响应改善。

## 5. CPU FE 并行原型结果

编码为每条 `encoder.encode` 外围墙钟的累计，排除模型加载、warmup、输入与输出哈希和 JSON 写入。CPU 为 25 ms 采样得到的生命周期 CPU 下界，含输入预检、哈希、加载、warmup 与退出；此处 CPU 秒不能与上节 ready→最后回复的 CPU 秒直接横比。内存含探针预加载的测试 PCM，属于独立探针，不是完整程序占用。

| 原型 | 轮数 | 编码秒 | 生命周期 CPU 下界秒 | 私有提交峰 MiB |
| --- | ---: | ---: | ---: | ---: |
| workers1 | 3 | 8.3394 | 60.203 | 412.34 |
| workers2 | 3 | 8.6904 | 62.641 | 461.70 |
| workers4 | 1 | 8.7292 | 62.688 | 565.50 |
| product | 2 | 8.2398 | 59.602 | 412.05 |

同并行原型的双 FE 与单 FE 相比：累计编码 **8.339360 → 8.690386 秒，慢 4.21%**；私有提交峰 **412.34 → 461.70 MiB，增加 49.36 MiB**。workers4 仅一轮，product 仅两轮，不能同 workers1/2 拼成一个三轮组。

同轮标签的三对实际累计编码结果如下。这里是编码墙钟而不是完整 ASR 请求延迟；每行各自运行了 39 条，已经计入上述 351 次编码探针，不再另加一次“哈希回归”计数。

| 轮标签 | 单 FE 编码秒 | 双 FE 编码秒 | 双 FE 变化 | 单 FE 私有峰 MiB | 双 FE 私有峰 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| r1 | 8.0349888 | 8.3858874 | +4.37% | 412.32812 | 461.69531 |
| r2 | 8.4263680 | 8.6903864 | +3.13% | 412.39062 | 460.80078 |
| r3 | 8.3393598 | 8.6956875 | +4.27% | 412.33984 | 461.82031 |

运行次序有记录：首批为 workers1-r1、workers2-r1、workers4-r1、workers2-r2、workers1-r2；最后一批为 product-r1、workers2-r3、workers1-r3、product-r2。没有把 r 标签解释为随机化顺序。

每 FE 会话 intra 线程分别为 8/4/2，总预算保持 8；BE 仍 8 线程，FE arena 开启、BE arena 关闭。各 FE 会话按块索引轮转，全部 join 后按原序收集；只有一个块的短句在双 FE 时只会用到一个 4 线程会话。BE 未复制，音频切分及每块 100 帧不变。

按音频长度统计，同原型 workers1→workers2：

| 桶 | 输入条数 | 单 FE 编码秒 | 双 FE 编码秒 | 变化 |
| --- | ---: | ---: | ---: | ---: |
| <2 秒 | 11 | 0.336856 | 0.365315 | +8.45% |
| 2–10 秒 | 20 | 1.549034 | 1.587251 | +2.47% |
| 10–60 秒 | 3 | 0.674263 | 0.680129 | +0.87% |
| ≥60 秒 | 5 | 5.769790 | 6.020834 | +4.35% |

每桶是三轮桶内累计值的中位数，桶中位相加不必等于总体中位数。个别长输入仅快 0.09%–0.56%，同时有约 5% 的变慢，未形成按长度选择并行的采用理由。

## 6. 固定 decoder 重放：仅作描述性诊断

固定输入为 1060 行 prompt embedding、255 个既定 token；不运行音频编码或 sampler。每个进程重复 7 次，完整词表 151936 个 logits 的指定 checkpoint finite 与 SHA 检查一致。各组的 warm6 中位如下：

| 进程组 | Prefill 秒 | 生成秒 | tokens/s | 生命周期 CPU 下界秒 |
| --- | ---: | ---: | ---: | ---: |
| control-a | 0.082110 | 0.918430 | 277.648 | 4.4375 |
| blocking | 0.082900 | 0.934730 | 272.806 | 2.5469 |
| sync | 0.084511 | 0.953199 | 267.520 | 4.4688 |
| short-spin | 0.081697 | 0.905294 | 281.677 | 4.1562 |
| control-b | 0.081847 | 0.911251 | 279.835 | 4.1406 |

这里的 CPU 下界包含 launcher、输入/模型哈希、加载、所有 7 次重放、全词表 finite 扫描与退出，不是纯 decoder CPU。短自旋热重放的小增益未在完整 ASR 累计 IPC 中成立，故没有据此部署。

## 7. 未实施方向与当时的投入判断

下列方向属于当时的分析与停止依据，不能写成已经实测失败：

1. **SameAsRequested CPU arena**：未实现、未测量。此前 BE 默认 arena 没有稳定提速且增加内存；更改扩容策略需处理 C API allocator 生命周期并做隔离回归，当时缺少值得投入的吞吐证据。
2. **ORT 全局线程池**：未实施。主要减少闲置线程/栈提交，不会直接减少串行 FE→BE 的活跃算力预算，还会触及对齐器独立线程策略，预期收益小。
3. **继续压缩 FE 打包/收集/复制**：先前 39 条累计不足 0.02 秒，即便全消除也仅约编码总时长的 0.24%；Mel 约 0.121 秒，完全消除的理论上限约 1.4%，实际只会节省其中一部分。重写浮点预处理需付出精度回归成本，这些是上限估算，不是新测出的收益。
4. **同请求阶段重叠**：当时 BE 依赖完整 FE，Vulkan prefill 又依赖完整编码，直接重叠受依赖约束。跨请求流水线仍未测；实时听写通常没有下一段积压，长文件则需重构同步 RPC、取消、顺序与错误语义，留作另一个文件吞吐问题。

以上判断针对 2026-10-27 的 0.4.4 CPU 编码路线；不用于否定后续 GGML GPU 编码器或其他模型版本。

## 8. 复现开关与源码保存范围

### Vulkan 调度

在父版本源码副本应用随文归档补丁，可恢复 `tools/bench_cuda_route.py`、`tools/test_vulkan_schedule.py` 和 `native/vulkan-fence-schedule-research.patch`；使用匹配的旧版 b9090 Vulkan 缓存源码与隔离运行目录。blocking/short-spin 必须用研究补丁构建的插件；其余三种队列选择是 b9090 的已有开关。

| `--vulkan-schedule` 值 | 子进程中设置的环境变量 |
| --- | --- |
| `control` | 清除下面 5 个候选变量 |
| `transfer` | `GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1` |
| `sync` | `GGML_VK_DISABLE_ASYNC=1` |
| `graphics` | `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` |
| `blocking` | `GGML_VK_FENCE_WAIT_BLOCKING=1` |
| `short-spin` | `GGML_VK_FENCE_SHORT_SPIN=1` |

这些变量以“存在”为开启条件：值为 `0` 仍会开启，关闭应删除变量。若同时设置 blocking 与 short-spin，blocking 优先。`configure_vulkan_schedule` 先复制环境，再清除所有候选变量，仅设置本次选中的一项；不修改系统环境或正式配置。它要求配置为 Vulkan，并把实际候选和环境写入报告，同时要求 GPU 测量门禁通过。

旧版命令参数模板如下，所有占位路径应指向隔离材料；本归档没有重新执行：

```text
python tools/bench_cuda_route.py --exe <旧版候选EXE> --app-dir <隔离应用目录> --manifest <私有测试清单> --reference <既有精确参考JSON> --output <新结果JSON> --mode asr --match exact --vulkan-schedule <control|transfer|sync|graphics|blocking|short-spin>
```

应检查实际 `YYUR_VULKAN_QUEUE` JSON 和 `final fence wait policy` 日志，不只检查请求的环境变量。源码测试覆盖继承环境中的 5 个 `0` 值被正确清除、PATH 保留、原环境不变，以及未知候选拒绝。

### CPU 并行 FE

CPU 探针的私有研究开关是 `YYUR_FE_WORKERS=1|2|4`；未知数值拒绝。探针参数为 `--app-dir`、`--manifest`、`--output`，可选 `--config`、`--reference`、`--repeats`。它要求 CPU encoder，输出完整 embedding SHA/形状用于比较，不进行 decoder 或采样。

**CPU 并行原型不在 5d2c228 的 5 文件差异中。** 已将原文件逐字节保存为 [编码器静态快照](parallel-encoder-snapshot.txt) 与 [探针静态快照](parallel-encoder-probe-snapshot.txt)。这是静态历史源码，不接入当前 `src` 或构建入口。

原型对应 b05102b/0.4.4 的旧工程结构与 Cargo 依赖：探针通过 `yuybuuru::engine::{EncoderProvider, EngineOptions}` 和 `yuybuuru::pipeline::engine_options` 使用原配置路径；相对引用 `../src/engine/mel.rs` 与 `../.build/parallel_encoder.rs`，后者还依赖父模块的 `EncoderProvider`、`EngineOptions` 和 `mel`。重建时应恢复该基点、旧 Cargo 依赖及 example/.build 相对布局，再引入快照，而不是把 `.txt` 直接放进当前重组后的源码中编译。测试输入与模型保持为本地材料。

| 本地材料 | SHA256 |
| --- | --- |
| `.build/parallel_encoder.rs` | `08cd7000c9ee08e1dff575219c0a20c4eeeb171992ca189f485e97f5023acb30` |
| `.build/parallel_encoder_probe.source.rs` | `cd7bf2011b153c2918bb801fd100e33abab2745ac2c3f5313f0f2a370f2991ef` |

### 原报告重算

`python evidence/scheduling/write_report.py` 会读取明确列出的既有 JSON/日志，重新核对 passed、exact、GPU、实际策略和清理，计算中位数与长度分桶并写出报告；它不运行模型。该生成器属于本地证据材料，并非 5d2c228 新增文件。为避免覆盖原归档，应在独立副本中使用。

## 9. 构建与正式版本保护记录

- V2/V3 compile 与 link 均为零错误，各有 5 条既有警告，导入/导出一致；153 个已编译 shader 对象哈希一致，没有重新生成 SPIR-V。
- CPU 隔离 Rust 原型 Release 构建零错误。Python 检查共 18 项通过，其中 16 项为已有 CUDA 门禁，2 项为新增调度隔离测试。
- 当时生产 `src` 和正式 EXE 未修改。报告中的 69 项 Rust 测试属于先前 0.4.4 的记录，未在本轮重跑；本轮也没有重新操作实体按键、粘贴或真实媒体入口。
- 正式 0.4.4 的 4 项文件哈希和 62 项模型/热词/配套文件保持，原服务未重启；候选插件和调度开关未部署。
- 上一轮已经采用的编码 −9.94% 属于 b05102b/0.4.4 的先前收益，不能重复算作本轮调度改善。本轮新增采用项为 0 项。

## 10. 可复核资料与公开范围

本次窄查确认：原始调度研究材料实际保留在当前工程的 `evidence/scheduling/`；旧归档 `rust-legacy/evidence/` 中没有同名 scheduling 目录。旧归档里的 `tools/bench_cpu_schedule.py` 是较早 CPU 线程/亲和性工具，不作为本次 351 次 FE 并行探针的来源。

主要证据：

- `evidence/scheduling/final/CPU与Vulkan调度研究报告.txt`：当时完整结论与口径；SHA256 `9ee8d3fcb19fe10ba142abaa347f418bb0f012614b62fe253c1900b99a3785e4`。
- `evidence/scheduling/final/测量汇总.json`：分组中位、逐轮脱敏统计与构建/保护记录；SHA256 `3b583494415237f1fd927520bc9a50f0ec4b10179766dfa68b68fce0ba7ca7c1`。
- `evidence/scheduling/write_report.py`：统计与门禁校验；SHA256 `7d03200b95ffeb54ef7dd9d0d1ac567e078bcdbd4e62299621aa1badb32b8f75`。
- `evidence/scheduling/v2-*.json`、`v3-*.json`：20 组完整 ASR 原始结果。
- `evidence/scheduling/fe-workers{1,2,4}-r*.json`、`fe-product-r*.json` 及相应 `.probe.json`：9 组独立 CPU 编码探针。
- `evidence/scheduling/fixed-summary.json`、`run_fixed_replay.py`、`fixed-*.json`/`.host.json`：5 个固定重放进程、35 次内部重复及采样口径。
- `evidence/scheduling/first-run-note.json`：排除探索轮的原因。
- `evidence/scheduling/retained-deployment-verification.json`：正式部署保护记录。
- Git 提交内的 `README.md`、`native/README.md`、原生补丁、benchmark 与测试：公开源码侧的研究记录。

GitHub 文档只保存方法、聚合数值、失败理由和经过检查的源码文本。原始录音、识别文本、token ID、PCM、个人配置/热词、个人绝对目录、完整进程日志和邮件标识不随归档上传。固定重放需要本地私有 fixture，公开源码归档不包含该输入。
