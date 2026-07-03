# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs


ROOT = Path.cwd()


def add_tree(folder, exclude_suffixes=()):
    root = ROOT / folder
    entries = []
    if not root.exists():
        return entries
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in exclude_suffixes:
            continue
        entries.append((str(path), str(path.parent.relative_to(ROOT))))
    return entries


datas = []
for folder in ("settings", "templates", "modules", "models", "translations"):
    datas += add_tree(folder)
datas += add_tree("drivers", exclude_suffixes=(".pdb", ".lib", ".exp"))

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=collect_dynamic_libs("onnxruntime"),
    datas=datas,
    hiddenimports=[
        "sklearn.preprocessing._polynomial",
        "sklearn.linear_model._base",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchvision",
        "matplotlib",
        "PIL",
        "networkx",
        "sympy",
        "pandas",
        "fsspec",
        "tracker.face.tongue_model",
        "tracker.hand.hand_depth_model",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ExVR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    uac_admin=True,
    icon=['logo\\logo.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ExVR',
)
