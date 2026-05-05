# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_dynamic_libs

block_cipher = None

# Ruta a OpenSlide binaries
openslide_path = os.path.abspath("openslide_bin")

a = Analysis(
    ['src/tiffcropper/app.py'],
    pathex=[],
    
    # 🔥 IMPORTANTE: incluir DLLs de imagecodecs
    binaries=collect_dynamic_libs('imagecodecs'),
    
    datas=[
        (openslide_path, 'openslide_bin'),
        ('assets', 'assets'),
    ],
    
    hiddenimports=[
        'zarr',
        'numcodecs',
        'imagecodecs',
        'imagecodecs._imagecodecs',
        'tifffile',
        'PIL',
        'numpy',
    ],
    
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TiffCropper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # 🔥 mejor evitar UPX con estas libs
    console=False,
    icon='assets/icon/cropper.ico',
)