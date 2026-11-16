param([switch]$Tests,[switch]$Check)
$ErrorActionPreference='Stop'
$taskCargo=Join-Path $env:USERPROFILE '.cargo/bin/cargo.exe'
$taskVcVars='C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
if(-not(Test-Path -LiteralPath $taskVcVars)){
    $taskVsWhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if(Test-Path -LiteralPath $taskVsWhere){
        $taskVsInstall=& $taskVsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        $taskVcVars=Join-Path $taskVsInstall 'VC/Auxiliary/Build/vcvars64.bat'
    }
}
if(-not(Test-Path -LiteralPath $taskVcVars)){throw 'MSVC C++ Build Tools and Windows SDK are required.'}
# Load build environment only into this PowerShell process. No user/global PATH changes.
$taskCommand='call "'+$taskVcVars+'" >nul && set'
& cmd.exe /d /c $taskCommand | ForEach-Object {
    if($_ -match '^([^=]+)=(.*)$'){[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}
}
$taskSharedBuild=if($env:YYUR_SHARED_BUILD_DIR){[IO.Path]::GetFullPath($env:YYUR_SHARED_BUILD_DIR)}else{[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../05-环境/YuYbUuRu-build'))}
$taskSdkPackage=if($env:YYUR_SDK_LIB_DIR){$env:YYUR_SDK_LIB_DIR}else{Join-Path $taskSharedBuild 'windows-sdk-x64'}
$taskSdkCppPackage=if($env:YYUR_SDK_CPP_DIR){$env:YYUR_SDK_CPP_DIR}else{Join-Path $taskSharedBuild 'windows-sdk-cpp'}
$taskInstalledSdk=if($env:WindowsSdkDir -and $env:WindowsSDKVersion){Join-Path $env:WindowsSdkDir ('Lib/'+$env:WindowsSDKVersion.TrimEnd('\'))}else{$null}
$taskSdkBase=Join-Path $taskSdkPackage 'c'
# Prefer the installed Windows SDK initialized by vcvars. The shared fallback
# is independent of archived checkouts and never changes user/global settings.
if(-not($taskInstalledSdk -and (Test-Path -LiteralPath (Join-Path $taskInstalledSdk 'um/x64/kernel32.lib')))){
    if(-not(Test-Path -LiteralPath (Join-Path $taskSdkBase 'um/x64/kernel32.lib'))){throw '缺少 Windows SDK 导入库，请安装 SDK 或运行 tools/build/bootstrap_sdk.py。'}
    $env:LIB=(Join-Path $taskSdkBase 'um/x64')+';'+(Join-Path $taskSdkBase 'ucrt/x64')+';'+$env:LIB
}
if(-not $env:RC -and -not(Get-Command rc.exe -ErrorAction SilentlyContinue)){
    $taskResourceCompiler=Join-Path $taskSdkCppPackage 'c/bin/10.0.28000.0/x64/rc.exe'
    if(-not(Test-Path -LiteralPath $taskResourceCompiler)){throw '缺少资源编译器 rc.exe，请安装 Windows SDK 或设置 YYUR_SDK_CPP_DIR。'}
    $env:RC=$taskResourceCompiler
}
Push-Location $PSScriptRoot
try{
    if($Tests){& $taskCargo test --release}
    elseif($Check){& $taskCargo check}
    else{& $taskCargo build --release}
    if($LASTEXITCODE -ne 0){throw "cargo failed with exit $LASTEXITCODE"}
    if(-not $Tests -and -not $Check){
        $taskDist=Join-Path $PSScriptRoot 'dist'
        New-Item -ItemType Directory -Path $taskDist -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'target/release/yuybuuru.exe') -Destination (Join-Path $taskDist '语音输入.exe') -Force
        Write-Output (Join-Path $taskDist '语音输入.exe')
    }
}finally{Pop-Location}
