param(
    [Parameter(Mandatory=$true)][string]$NativeSource,
    [Parameter(Mandatory=$true)][string]$BuildDir,
    [Parameter(Mandatory=$true)][string]$SdkInclude,
    [Parameter(Mandatory=$true)][string]$SdkLib,
    [Parameter(Mandatory=$true)][string]$SdkBin,
    [string]$VcVars,
    [string]$CMake,
    [string]$Ninja,
    [ValidateRange(1,4)][int]$Jobs=2
)
$ErrorActionPreference='Stop'
if(-not $CMake){$CMake=(Get-Command cmake.exe -ErrorAction Stop).Source}
if(-not $Ninja){$Ninja=(Get-Command ninja.exe -ErrorAction Stop).Source}
if($VcVars){
    & cmd.exe /d /c ('call "'+$VcVars+'" >nul && set') | ForEach-Object {
        if($_ -match '^([^=]+)=(.*)$'){[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}
    }
    if($LASTEXITCODE -ne 0){throw 'MSVC environment initialization failed'}
}
if(-not $env:VCToolsInstallDir){throw 'Use an MSVC developer shell or pass -VcVars'}
$taskVc=[IO.Path]::GetFullPath($env:VCToolsInstallDir)
foreach($taskRequired in @($CMake,$Ninja,(Join-Path $NativeSource 'tools/mtmd/caps-encoder.h'),
    (Join-Path $SdkInclude 'um/Windows.h'),(Join-Path $SdkInclude 'ucrt/stdlib.h'),
    (Join-Path $SdkLib 'um/x64/kernel32.lib'),(Join-Path $SdkLib 'ucrt/x64/ucrt.lib'),(Join-Path $SdkBin 'rc.exe'))){
    if(-not(Test-Path -LiteralPath $taskRequired -PathType Leaf)){throw "Missing build dependency: $taskRequired"}
}
$taskIncludes=@('ucrt','shared','um','winrt','cppwinrt') | ForEach-Object {Join-Path $SdkInclude $_}
$env:INCLUDE=(Join-Path $taskVc 'include')+';'+($taskIncludes -join ';')
$env:LIB=(Join-Path $SdkLib 'um/x64')+';'+(Join-Path $SdkLib 'ucrt/x64')+';'+(Join-Path $taskVc 'lib/x64')
$env:PATH=$SdkBin+';'+$env:PATH
$env:VSLANG='1033'
New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
$taskConfigureLog=Join-Path $BuildDir 'caps-encoder-configure.log'
$taskBuildLog=Join-Path $BuildDir 'caps-encoder-build.log'
$taskArgs=@('-S',$NativeSource,'-B',$BuildDir,'-G','Ninja',
    "-DCMAKE_MAKE_PROGRAM=$Ninja",'-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=ON','-DGGML_BACKEND_DL=ON','-DGGML_NATIVE=OFF',
    '-DGGML_CPU=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF','-DGGML_OPENCL=OFF','-DGGML_SYCL=OFF',
    '-DLLAMA_BUILD_COMMON=ON','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF',
    '-DLLAMA_BUILD_TOOLS=ON','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_OPENSSL=OFF')
& $CMake @taskArgs *> $taskConfigureLog
if($LASTEXITCODE -ne 0){Get-Content -LiteralPath $taskConfigureLog -Tail 60; throw 'GGML encoder configure failed'}
& $CMake --build $BuildDir --target mtmd --parallel $Jobs *> $taskBuildLog
if($LASTEXITCODE -ne 0){Get-Content -LiteralPath $taskBuildLog -Tail 70; throw 'GGML encoder build failed'}
$taskDll=Join-Path $BuildDir 'bin/caps_encoder.dll'
Get-Item -LiteralPath $taskDll | Select-Object FullName,Length
Get-FileHash -LiteralPath $taskDll,$taskConfigureLog,$taskBuildLog -Algorithm SHA256 | ForEach-Object {
    Write-Output ("SHA256 {0} {1}" -f $_.Hash,$_.Path)
}
