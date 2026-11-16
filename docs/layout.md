# 文件组织与维护

0.6.0 采用下列组织方式。内部测量版本 1.0.0／1.0.1 已在 2026 年 11 月 16 日起验证此布局。应用保留实时听写、热词修正、音视频文件转录三项功能，操作入口为托盘。编码 ONNX 与配套 `.weights.bin` 放在相同模型目录。

## 正式运行目录

```text
语音输入.exe
必需的启动 DLL
config/
  settings.json
  hot.txt
  hot-rule.txt
runtime/
  llama/       原生文本生成库、CPU 插件与 Vulkan 后端
  onnx/        ONNX Runtime 及其依赖
  audio/       PortAudio
  media/       FFmpeg 及独立启动所需的运行库
models/        ASR 和时间戳对齐模型
cache/         Vulkan 管线缓存
logs/          当前 Rust 进程日志
docs/          当前说明、优化索引和许可证
```

根目录保留 Windows 在程序入口执行前就需要解析的 CRT/MSVC DLL；将这部分直接搬到子目录会破坏启动。FFmpeg 的运行库随 `runtime/media` 独立保存。模型文件、热词内容、识别参数和个人历史文件均不因目录整理而删除。

新程序只使用上述新路径，不搜索旧 `core/server`、`internal` 或根目录配置。升级时由显式迁移完成文件搬运，避免持续维护两套布局。`src/paths.rs` 集中定义默认路径。`settings.json` 中相对 `app_dir` 始终以该配置文件所在目录为基准，因此安装在 `config/` 下的配置使用 `"app_dir": ".."`。显式指定的 `--config` 路径规则保持不变。

一个正式 EXE 负责桌面入口，并按需以同一 EXE 启动模型工作进程。工作进程独立退出仍用于释放模型内存，不代表恢复客户端和服务端两个启动器，也没有为此启用网络服务。

## 源码、构建和归档

仓库根目录保存活跃源码。`src/` 保留现有功能模块；`tools/build/`、`tools/validation/` 和 `tools/research/` 分别容纳构建部署、验证和研究脚本；`dist/` 保存生成的发布程序。

共享构建 SDK 由 `YYUR_SHARED_BUILD_DIR` 或相对工作区布局指定，不经过旧 Rust 或 CUDA 工程的目录联接。旧 Rust、Python、CUDA 工作树及原 Git 主仓在本机独立归档。活跃源码具有完整的 `.git` 目录，不依赖旧档；迁移记录保留原对象与引用，历史提交保持可查。

正式安装中已过时的 Python 说明、研究运行副本、旧日志和 FFmpeg 源码另存于安装目录同级的 `语音输入-历史归档-20261116`。年份目录可能含有用户内容，继续原地保留，不按缓存清空。

历史报告中的路径、版本和测量数据保留原貌。迁移前后的路径映射、文件校验、Git 核对和当前候选验收记录位于活跃源码的 `evidence/layout-20261116/`。历史目录联接按照迁移映射修正，避免同名新工程误接旧证据。

## 相关说明

- [使用说明](使用说明.md)
- [AI 自动化接口](ai-automation.md)
- [历次优化与测量](optimization-history.md)

目录整理改善维护与部署结构，本身不作为内存或识别速度优化百分比的测量依据。
