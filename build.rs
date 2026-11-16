use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    println!("cargo:rerun-if-changed=assets/application.manifest");
    println!("cargo:rerun-if-env-changed=RC");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let version = env::var("CARGO_PKG_VERSION").expect("package version");
    let mut numbers: Vec<_> = version
        .split(['.', '-'])
        .take(3)
        .map(|part| part.parse::<u16>().expect("numeric package version"))
        .collect();
    numbers.resize(4, 0);
    let numeric_version = numbers
        .iter()
        .map(u16::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let icon = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"))
        .join("assets/icon.ico")
        .to_string_lossy()
        .replace('\\', "/");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"))
        .join("assets/application.manifest")
        .to_string_lossy()
        .replace('\\', "/");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("build output directory"));
    let source = output.join("application.rc");
    let resource = output.join("application.res");
    fs::write(
        &source,
        format!(
            r#"#pragma code_page(65001)
1 ICON "{icon}"
1 24 "{manifest}"
1 VERSIONINFO
FILEVERSION {numeric_version}
PRODUCTVERSION {numeric_version}
FILEFLAGSMASK 0x3fL
FILEFLAGS 0x0L
FILEOS 0x40004L
FILETYPE 0x1L
FILESUBTYPE 0x0L
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "080404b0"
        BEGIN
            VALUE "FileDescription", "语音输入"
            VALUE "FileVersion", "{version}"
            VALUE "InternalName", "yuybuuru"
            VALUE "OriginalFilename", "语音输入.exe"
            VALUE "ProductName", "语音输入"
            VALUE "ProductVersion", "{version}"
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x0804, 1200
    END
END
"#
        ),
    )
    .expect("write Windows application resources");

    let compiler = env::var_os("RC").unwrap_or_else(|| "rc.exe".into());
    let status = Command::new(compiler)
        .arg("/nologo")
        .arg("/fo")
        .arg(&resource)
        .arg(&source)
        .status()
        .expect("启动 Windows 资源编译器失败，请安装 Windows SDK 或设置 RC 路径");
    assert!(status.success(), "Windows 应用图标和版本资源编译失败");
    println!("cargo:rustc-link-arg={}", resource.display());
}
