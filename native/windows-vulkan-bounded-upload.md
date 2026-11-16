# Windows Vulkan 有界模型上传

[windows-vulkan-bounded-upload.patch](windows-vulkan-bounded-upload.patch) 应用于既有、已验证的 llama.cpp **b9090**（提交 `5757c4dcb178a01c85234a6db7503b19c9598873`）源码，已经包含 [对齐器补丁](aligner-logits.patch) 和 [ASR 单输出补丁](asr-single-output.patch)。本补丁只修改 `src/llama-model-loader.cpp`，不改变公开参数结构、导出接口、模型权重、量化或推理计算。

补丁前该文件的 SHA-256：`e487eeb7849bcb5ff1be2c90c360cded0dad3af34b844150c711d6b779a8c4a6`（UTF-8/LF）。

## 加载策略

在 **Windows、`use_mmap=true`、Vulkan registry、GPU/IGPU 默认 buffer** 同时满足，且后端具备 async、host buffer、events 能力并成功初始化上传器时，GPU 权重复用原有 **4×1 MiB** pinned 缓冲环分块读取、异步上传。CPU 权重继续使用原 mmap 指针，保留嵌入表的零拷贝映射。

CPU、CUDA、其他平台与其他后端继续原路径；`use_mmap=false` 的原分块路径保持不变。策略默认生效，不读取研究阶段的 `YYUR_LLAMA_HYBRID_GPU_LOAD` 环境变量。上传偏移、对齐、字节数、缓冲同步、进度计数都沿用原代码。

后端能力或上传器初始化失败时，保留原映射加载路径。上传开始后的驱动错误、异常及原取消回调清理行为沿用上游实现；本补丁没有增加中途失败自动重试或取消安全保证。

该方式避免通过整个 GPU 权重文件映射触页，也避免为最大完整 tensor 保留同步 staging。预取与系统文件缓存仍保留，进程工作集下降不能直接等同于物理 RAM 同比例释放。内存和延迟结果应使用相同模型、后端与输出进行实测。

## 复现构建

在已具备 MSVC、Windows SDK、CMake、Ninja 的 x64 Developer PowerShell 中，从本项目根目录操作。先按 [原生构建说明](README.md) 准备精确 b9090 源码，再依次应用 `aligner-logits.patch`、`asr-single-output.patch` 和本补丁。已有前两个补丁的源码只应用最后一个：

```powershell
$taskRepo = (Get-Location).Path
$taskNativeRelative = '.build/llama-b9090-single-output'
$taskNative = (Resolve-Path $taskNativeRelative).Path
$taskPatch = (Resolve-Path 'native/windows-vulkan-bounded-upload.patch').Path
$taskLoader = Join-Path $taskNative 'src/llama-model-loader.cpp'
if ((Get-FileHash -LiteralPath $taskLoader -Algorithm SHA256).Hash -ne 'E487EEB7849BCB5FF1BE2C90C360CDED0DAD3AF34B844150C711D6B779A8C4A6') {
  throw '补丁前源码哈希不匹配。'
}
git -C $taskRepo apply --directory=$taskNativeRelative --check $taskPatch
if ($LASTEXITCODE -ne 0) { throw '补丁检查失败，请核对源码版本及已有补丁。' }
git -C $taskRepo apply --directory=$taskNativeRelative $taskPatch
if ($LASTEXITCODE -ne 0) { throw '应用补丁失败。' }
if ((Get-FileHash -LiteralPath $taskLoader -Algorithm SHA256).Hash -ne '6FC9E68B6645385E11816307462120FD7F10D81AA8A016AC404C2FBB8A24F178') {
  throw '补丁没有得到本轮验证过的源码。'
}

$taskSdkVersion = $env:WindowsSDKVersion.TrimEnd('\')
$taskSdkRoot = $env:WindowsSdkDir
& ./tools/build/build_asr_single_output.ps1 `
  -NativeSource $taskNative `
  -BuildDir '.build/windows-vulkan-bounded-upload' `
  -SdkInclude (Join-Path $taskSdkRoot "Include/$taskSdkVersion") `
  -SdkLib (Join-Path $taskSdkRoot "Lib/$taskSdkVersion") `
  -SdkBin (Join-Path $taskSdkRoot "bin/$taskSdkVersion/x64")
```

现有脚本只构建 `llama` 目标并关闭后端构建。仅在隔离候选运行目录替换生成的 `bin/llama.dll`，保留原有匹配的 ggml、CPU、Vulkan、OpenMP 依赖，先验证 imports、exports、ABI 和真实转写结果，再部署。不能一并覆盖构建目录生成的其他 DLL。

`git apply` 从项目根运行并显式给出目标子目录，避免嵌套源码目录中 Git 路径前缀导致跳过补丁；前后源码哈希也需同时通过。共享 SDK 后备环境可把三个 SDK 参数改为对应的已验证目录，不能将旧构建缓存当作新构建目录复用。

符合条件且上传器成功建立时会出现：

```text
load_all_data: Windows Vulkan bounded GPU upload enabled; CPU mmap preserved, GPU ring 4 x 1048576 bytes
```

日志只证明选择了该加载路径；识别一致性、加载延迟、工作集和提交量需要独立验证。补丁沿用 llama.cpp 的 MIT 许可，保留 [原许可](llama.cpp-LICENSE)。
