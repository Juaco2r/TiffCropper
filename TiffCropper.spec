# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

block_cipher = None

# ============================================================
# Paths
# ============================================================

project_root = os.path.abspath(".")
openslide_path = os.path.join(project_root, "openslide_bin")
assets_path = os.path.join(project_root, "assets")
icon_path = os.path.join(assets_path, "icon", "cropper.ico")

# ============================================================
# Collect compiled dependencies
# ============================================================

imagecodecs_datas, imagecodecs_binaries, imagecodecs_hiddenimports = collect_all("imagecodecs")
numcodecs_datas, numcodecs_binaries, numcodecs_hiddenimports = collect_all("numcodecs")
zarr_datas, zarr_binaries, zarr_hiddenimports = collect_all("zarr")

# OpenSlide dynamic libraries if available as package binaries
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
    "imagecodecs._aec",
    "imagecodecs._bitshuffle",
    "imagecodecs._brotli",
    "imagecodecs._deflate",
    "imagecodecs._jpeg2k",
    "imagecodecs._jpeg8",
    "imagecodecs._jpegsof3",
    "imagecodecs._lz4",
    "imagecodecs._lzf",
    "imagecodecs._lzma",
    "imagecodecs._png",
    "imagecodecs._tiff",
    "imagecodecs._webp",
    "imagecodecs._zlib",
    "imagecodecs._zopfli",
    "imagecodecs._zstd",
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
    pathex=[project_root],
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
    icon=icon_path if os.path.exists(icon_path) else None,
)