# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import (
    collect_all,
    collect_dynamic_libs,
    collect_submodules,
    collect_data_files,
)

block_cipher = None

# ============================================================
# Paths
# ============================================================

project_root = os.path.abspath(".")
openslide_path = os.path.join(project_root, "openslide_bin")
assets_path = os.path.join(project_root, "assets")
icon_path = os.path.join(assets_path, "icon", "cropper.ico")

# ============================================================
# Collect compiled / package dependencies
# ============================================================

imagecodecs_datas, imagecodecs_binaries, imagecodecs_hiddenimports = collect_all("imagecodecs")
numcodecs_datas, numcodecs_binaries, numcodecs_hiddenimports = collect_all("numcodecs")
zarr_datas, zarr_binaries, zarr_hiddenimports = collect_all("zarr")

# Leica LIF support.
# readlif is imported lazily in the app, so PyInstaller may not detect it
# unless it is explicitly collected here.
readlif_hiddenimports = collect_submodules("readlif")
bs4_hiddenimports = collect_submodules("bs4")
soupsieve_hiddenimports = collect_submodules("soupsieve")

readlif_datas = collect_data_files("readlif")
bs4_datas = collect_data_files("bs4")
soupsieve_datas = collect_data_files("soupsieve")

# OpenSlide dynamic libraries if available as package binaries.
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
datas += readlif_datas
datas += bs4_datas
datas += soupsieve_datas

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

    # Leica LIF support
    "readlif",
    "readlif.reader",
    "bs4",
    "bs4.builder",
    "bs4.builder._htmlparser",
    "soupsieve",

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
hiddenimports += readlif_hiddenimports
hiddenimports += bs4_hiddenimports
hiddenimports += soupsieve_hiddenimports

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
