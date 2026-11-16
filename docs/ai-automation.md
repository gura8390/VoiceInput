# 面向 AI 的本机操作接口

适用于语音输入 0.6.0，接口 schema 仍为 1。日常界面保持托盘菜单，自动化工具通过命令返回的 UTF-8 JSON 读写状态。普通命令执行后退出，桌面听写由托盘实例持续运行。

## 查询与文字修正

```powershell
.\语音输入.exe automation status
.\语音输入.exe automation correct "No doubt"
.\语音输入.exe automation correct "Copywriter"
```

成功结果包含 `schema=1`、`ok=true`、`action` 和 `data`。操作失败时 `ok=false`，`error` 提供错误代码、中文消息和诊断详情，同时返回非零退出码。参数帮助可用 `automation --help`。

`status` 通过窗口类、进程 ID 和实际 EXE 路径确认本程序的托盘实例。`saved_config` 描述磁盘保存值，`active_settings_verified=false` 明确当前接口读取的范围。热词修正会读取当前词表和规则，返回 `corrected`。独立说出的 No doubt 与 Copywriter 按当前本机规则分别修正为 Godot 和 CapsWriter，普通完整英文句子仍按规则语境处理。

## 配置与登录启动

```powershell
.\语音输入.exe automation config
.\语音输入.exe automation config --asr-model 1.7b --hotwords true --hotword-rules true
.\语音输入.exe automation startup
.\语音输入.exe automation startup --enabled true
.\语音输入.exe automation startup --enabled false
```

配置更新保留其它 JSON 字段，以同目录临时文件校验后原子保存。相同值的重复更新返回 `changed=false`。运行中修改模型或热词开关后，`restart_required=true`，需要正常退出并重开应用。

登录启动使用当前用户 Run 项下名为“语音输入”的单一值，命令指向当前 EXE。登录时显示托盘。重复启用和停用具有幂等性，同名值若属于其它程序会明确报错。结果中的 `validation_scope=registration_only` 描述注册核验范围，实际 Windows 登录需独立验收。

## 文件转录与正常退出

```powershell
.\语音输入.exe transcribe ".\录音.wav" --output-dir ".\输出"
.\语音输入.exe automation quit
.\语音输入.exe
```

文件转录输出 TXT、JSON、SRT，默认保留已有同名输出。需要覆盖时明确传入 `--overwrite`。`quit` 核对窗口与程序路径后发送正常退出消息并等待，避免中断剪贴板恢复。

使用 `--config` 和 `--app-dir` 可以指定隔离测试配置与运行目录。生产默认读取 EXE 旁的 `config/settings.json`，热词和规则分别位于程序根目录下的 `config/hot.txt`、`config/hot-rule.txt`。相对 `app_dir` 仍以配置文件的父目录为基准，正式布局使用 `"app_dir": ".."`。显式传入的相对 `--config` 以当前工作目录为基准；不要把它误认为相对 EXE。路径约定见 [目录说明](layout.md)。

## 托盘菜单的系统语义入口

现有原生托盘菜单通过 Windows MSAA 暴露菜单项目名称、角色、启用状态和“执行”动作。自动化应按可访问名称查找“转录音频 / 视频…”、“编辑热词”、“退出”，再调用其默认动作。状态行可读取且处于禁用状态。

该原生弹出菜单在本机 UI Automation Control View / Raw View 中仅显示根 Pane。等效系统无障碍入口为 `AccessibleObjectFromWindow` 的 `OBJID_CLIENT` / `IAccessible`，其子节点可按名称读取，并通过 `accDoDefaultAction` 执行。文件选择对话框使用系统提供的语义控件。验证和操作使用这些语义接口，避免依赖屏幕坐标。

图标与 DPI 声明嵌入 EXE。PerMonitorV2 使原生菜单按显示器缩放绘制。每次交付应分别记录命令接口、菜单语义动作、实际转录和退出的验证结果。
