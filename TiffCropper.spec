# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

block_cipher = None

# ============================================================
# Paths
# ============================================================

openslide_path = os.path.abspath("openslide_bin")
assets_path = os.path.abspath("assets")

# ============================================================
# Collect compiled dependencies
# ============================================================

imagecodecs_datas, imagecodecs_binaries, imagecodecs_hiddenimports = collect_all("imagecodecs")
numcodecs_datas, numcodecs_binaries, numcodecs_hiddenimports = collect_all("numcodecs")
zarr_datas, zarr_binaries, zarr_hiddenimports = collect_all("zarr")

# OpenSlide DLLs, if available as a Python package
openslide_binaries = collect_dynamic_libs("openslide")

# ============================================================
# Data files
# ============================================================

datas = []

if os.path.isdir(openslide_path):
    datas.append((openslide_path, "openslide_bin"))

if os.path.isdir(assets_path):
    datas.append((assets_path, "assets"))

datas += imagecodecs_datas
datas += numcodecs_datas
datas += zarr_datas

# ============================================================
# Binaries
# ============================================================

binaries = []
binaries += imagecodecs_binaries
binaries += numcodecs_binaries
binaries += zarr_binaries
binaries += openslide_binaries

# ============================================================
# Hidden imports
# ============================================================

hiddenimports = [
    "numpy",
    "PIL",
    "PIL.Image",
    "tifffile",
    "zarr",
    "numcodecs",
    "imagecodecs",
    "openslide",

    # Extra safety for imagecodecs compiled modules
    "imagecodecs._shared",
    "imagecodecs._imcd",
]

hiddenimports += imagecodecs_hiddenimports
hiddenimports += numcodecs_hiddenimports
hiddenimports += zarr_hiddenimports

# Remove duplicates while preserving order
hiddenimports = list(dict.fromkeys(hiddenimports))

# ============================================================
# Analysis
# ============================================================

a = Analysis(
    ["src/tiffcropper/app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ============================================================
# Single-file executable
# ============================================================

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="TiffCropper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon="assets/icon/cropper.ico",
)