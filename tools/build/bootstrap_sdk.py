"""Fetch pinned Microsoft SDK import libraries into the shared build environment."""
from pathlib import Path, PurePosixPath
import sys
import hashlib
import json
import os
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_layout import ROOT
SHARED_BUILD = Path(os.environ.get('YYUR_SHARED_BUILD_DIR', str(ROOT.parents[1] / '05-环境/YuYbUuRu-build')))
SDK = Path(os.environ.get('YYUR_SDK_LIB_DIR', str(SHARED_BUILD / 'windows-sdk-x64'))).resolve()
URL = 'https://api.nuget.org/v3-flatcontainer/microsoft.windows.sdk.cpp.x64/10.0.28000.2705/microsoft.windows.sdk.cpp.x64.10.0.28000.2705.nupkg'
EXPECTED_SHA256 = 'bf892c1048059984850a1ed79bb81c0243478c6d7472d637c90565ab5c836013'
PACKAGE = SDK.parent / 'windows-sdk-x64.nupkg'
PACKAGE.parent.mkdir(exist_ok=True, parents=True)
if not PACKAGE.exists():
    urllib.request.urlretrieve(URL, PACKAGE)
with PACKAGE.open('rb') as stream:
    package_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
if package_sha256 != EXPECTED_SHA256:
    raise RuntimeError('SDK package SHA-256 differs from the pinned official package; nothing was extracted')
with zipfile.ZipFile(PACKAGE) as archive:
    extracted = []
    for info in archive.infolist():
        rel = PurePosixPath(info.filename)
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError('Unsafe archive member')
        if info.filename.lower().endswith('.lib') and ('x64' in rel.parts):
            target = SDK / Path(*rel.parts)
            assert target.resolve().is_relative_to(SDK.resolve())
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_bytes(archive.read(info))
            extracted.append(str(target))
        elif rel.name.lower().startswith(('license', 'copyright')):
            SDK.mkdir(exist_ok=True, parents=True)
            (SDK / rel.name).write_bytes(archive.read(info))
receipt = {'url': URL, 'sha256': package_sha256,
           'package_bytes': PACKAGE.stat().st_size, 'extracted_libraries': len(extracted),
           'sdk_directory': str(SDK),
           'library_directories': sorted({str(Path(file).parent) for file in extracted})}
(ROOT / 'evidence').mkdir(exist_ok=True)
(ROOT / 'evidence/sdk-bootstrap.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), 'utf-8')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
