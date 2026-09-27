# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Linux (x86_64) builds.
#
# PyInstaller cannot cross-compile, so this spec must be run on Linux, e.g.
# from macOS with:
#   docker run --rm --platform linux/amd64 -v "$PWD":/src -w /src python:3.11 bash build-linux.sh
# See build-linux.sh for the full recipe.

from pathlib import Path

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
# Note: drivers/ is intentionally excluded. On Linux the upstream VRto3D
# driver (vrto3d install.sh) and the standalone VMT driver are installed
# and registered separately; ExVR does not ship or manage them.

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "sklearn.preprocessing._polynomial",
        "sklearn.linear_model._base",
        # pynput resolves its X11 backend dynamically at import time, so
        # PyInstaller's static analysis misses these modules.
        "pynput.keyboard._xorg",
        "pynput.mouse._xorg",
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
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ExVR',
)
