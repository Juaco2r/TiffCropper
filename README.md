# TiffCropper

### WSI Crop, Tile and Merge Tool for Digital Pathology Images

TiffCropper is a standalone Windows application for working with large digital pathology and microscopy images. It supports ROI cropping, square tiling for efficient downstream processing, and reconstruction of tiled images.

The Windows release is designed to run without a separate Python installation.

![WSI Cropper Concept Screenshot](assets/screenshots/WSICropper_concept.png)

---

## Main Features

- Crop high-resolution ROIs from large WSI files.
- Select the full image area automatically for whole-slide export/downsampling.
- Downsample crops before saving.
- Generate square tiles with optional overlap.
- Bulk tile multiple images with the same parameters.
- Add black or white padding to incomplete border tiles.
- Save outputs as TIFF or JPEG.
- Merge tiles automatically from names such as `ImageName_A1.tif`, `ImageName_B1.tif`, `ImageName_A2.tif`.
- Merge tiles manually using a row/column grid when filenames do not encode tile position.
- Preserve calibration metadata when available.
- Support BigTIFF output and lossless DEFLATE compression.

---

## Supported Input Formats

| Format | Backend | Notes |
|---|---|---|
| TIFF / pyramidal TIFF | tifffile / OpenSlide | Pyramid support when available |
| OME-TIFF | tifffile | Physical pixel size preserved when available |
| NDPI | OpenSlide | Requires OpenSlide binaries bundled in the Windows executable |
| JPEG / PNG | Pillow | Mainly useful for tiles or simple raster inputs |

---

## Quick Start

1. Launch `TiffCropper.exe`.
2. Select a mode from the top menu:
   - `Crop`
   - `Tiles`
   - `Merge Tiles`
3. Load one or more images.
4. Configure parameters.
5. Preview when needed.
6. Save the result.

---

## Crop Mode

Crop mode extracts one ROI from the selected image.

Parameters:

- `X`: left coordinate in pixels.
- `Y`: top coordinate in pixels.
- `Width`: ROI width in pixels.
- `Height`: ROI height in pixels.
- `Select full area`: automatically sets `X=0`, `Y=0`, `Width=image width`, `Height=image height`.
- `Downsample`: default `1`. A value of `2` saves the crop at half width and half height.
- Output format: TIFF, OME-TIFF or JPEG.

Output examples:

```text
ImageName_crop_final.tif
ImageName_crop_finalDS2.tif
ImageName_crop_ROI1.ome.tif
```

---

## Tiles Mode

Tiles mode creates square tiles from one image or multiple selected images.

Parameters:

- `Square tile size px`: default `1024`.
- `Overlap`: percentage overlap between neighboring tiles. Default `0%`.
- `Downsample`: default `1`. Applied after the original-resolution tile is extracted.
- `Padding`: black or white padding for border tiles.
- Output format: TIFF or JPEG.

Tiles are saved inside a subfolder with the same name as the image, created beside the input file.

Naming convention:

```text
ImageName_A1.tif
ImageName_B1.tif
ImageName_A2.tif
ImageName_A1Ov10.tif
ImageName_A1Ov10DS2.tif
```

Columns use letters (`A`, `B`, `C`, ...), rows use numbers (`1`, `2`, `3`, ...).

---

## Merge Tiles Mode

Merge mode reconstructs a tiled image.

### Auto mode

Use when tiles follow the naming convention:

```text
ImageName_A1.tif
ImageName_B1.tif
ImageName_A2.tif
```

If the tile names include overlap, for example `Ov10`, the program can read it automatically when overlap is set to `auto`.

### Manual grid mode

Use when tile names do not encode their position.

Workflow:

1. Select tile files or a folder.
2. Choose `Manual grid` mode.
3. Click `Preview Reconstruction`.
4. Enter number of rows and columns.
5. Assign tiles to grid cells manually, or use auto-fill by selected order.
6. Save the merged image.

Manual mode assumes all tiles have the same square size.

---

## Important Note About Overlap

The merge operation is geometric. It does not perform image registration or intelligent stitching.

For overlap:

```text
stride = tile_size - overlap_px
```

Neighboring tiles are placed using this stride. In overlapping regions, the later tile overwrites the previous tile. This works correctly when tiles were generated from the same source image using the same tile size, overlap and downsample.

---

## Development Setup

Clone the repository:

```bash
git clone https://github.com/Juaco2r/TiffCropper.git
cd TiffCropper
```

Create and activate a virtual environment:

```bat
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the app:

```bat
python src\tiffcropper\app.py
```

---

## Windows Executable Build

Recommended: build on Windows 10/11 using Python 3.10 or 3.11.

From the repository root:

```bat
venv\Scripts\activate
build_windows.bat
```

The executable will be created at:

```text
dist\TiffCropper.exe
```

The build script bundles the OpenSlide DLLs from the `openslide-bin` wheel into the executable under `openslide_bin`, which matches the runtime DLL-loading logic in the app.

Alternative using the spec file:

```bat
pyinstaller --noconfirm --clean TiffCropper.spec
```

---

## Recommended Repository Files

```text
requirements.txt
build_windows.bat
TiffCropper.spec
README.md
src/tiffcropper/app.py
assets/icon/cropper.ico
```

---

## System Requirements

For the Windows executable:

- Windows 10 or 11.
- 8 GB RAM minimum; 16 GB or more recommended for large WSI.
- Sufficient disk space for large crops or tile sets.
- No internet connection required after download.

For development/building:

- Windows 10 or 11.
- Python 3.10 or 3.11.
- Visual C++ runtime normally included with Windows or installed by common scientific Python packages.

---

## License

MIT License  
© 2026 Jose Rodriguez-Rojas

See `LICENSE` for details.

---

## Third-Party Dependencies

This project depends on:

- NumPy
- tifffile
- imagecodecs
- Pillow
- zarr
- PyQt5
- OpenSlide / openslide-python / openslide-bin
- PyInstaller for building the executable

See `THIRD_PARTY_NOTICES.md` for additional information.
