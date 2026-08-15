# PathoImage Toolkit Protocol
Version: 0.1.0  
Author: Jose Rodriguez-Rojas  
License: MIT  
Last updated: 2026  

---

# 1. Overview

PathoImage Toolkit is a standalone Windows application designed for digital pathology and microscopy workflows to extract high-resolution Regions of Interest (ROIs) from large Whole Slide Images (WSI).

Supported input formats:

- TIFF (.tif, .tiff)
- OME-TIFF (.ome.tif, .ome.tiff)
- NDPI (Hamamatsu)

The application preserves calibration metadata when available and supports robust recovery strategies for partially corrupted pyramid tiles.

The Windows release requires no Python installation.

---

# 2. Key Features

- Graphical coordinate input (X, Y, Width, Height)
- TIFF and OME-TIFF output
- Optional lossless compression (DEFLATE)
- Metadata preservation (resolution and physical pixel size)
- NDPI support via OpenSlide
- Automatic pyramid-aware preview
- Robust fallback reading for corrupted tiles
- BigTIFF support for large outputs
- Fully offline operation

---

# 3. System Requirements

- Windows 10 or Windows 11 (64-bit recommended)
- ≥ 4 GB RAM (≥ 16 GB recommended for large WSIs)
- Sufficient disk space for output files
- No internet connection required

---

# 4. Supported File Formats

## 4.1 TIFF / OME-TIFF

Backend:
- `tifffile`
- OpenSlide (when available for pyramidal TIFF)

Capabilities:
- Multi-resolution pyramid support (when present)
- Metadata extraction from TIFF tags
- Physical resolution inference (DPI ↔ µm per pixel)

## 4.2 NDPI

Backend:
- OpenSlide

Capabilities:
- Multi-level pyramid access
- Extraction at full resolution (level 0)
- Access to OpenSlide metadata properties

---

# 5. Coordinate System

The image is treated as a pixel matrix:

- X → Column index (horizontal axis, left = 0)
- Y → Row index (vertical axis, top = 0)
- Width → Number of columns extracted
- Height → Number of rows extracted

ROI bounds are automatically clipped to image dimensions to prevent out-of-range errors.

---

# 6. Output Options

## 6.1 Output Format

The user may select:

- TIFF (.tif)
- OME-TIFF (.ome.tif)

Output file naming:
{original_name}crop{suffix}.tif
{original_name}crop{suffix}.ome.tif


If no suffix is provided, `"final"` is used.

---

## 6.2 Lossless Compression

When enabled:

- Compression: DEFLATE
- Predictor: enabled

This reduces file size without data loss.

---

## 6.3 BigTIFF Support

All outputs are written using:

- `bigtiff=True`
- 256×256 tiling
- RGB photometric interpretation

This ensures compatibility with large ROIs (>4GB).

---

# 7. Metadata Preservation

## 7.1 Resolution

When available, the following are preserved:

- XResolution
- YResolution
- ResolutionUnit

If physical pixel size (µm per pixel) is available:

- PhysicalSizeX and PhysicalSizeY are embedded in OME-TIFF output.

---

## 7.2 NDPI Metadata (OME-TIFF mode)

When cropping NDPI and exporting as OME-TIFF:

- OpenSlide properties are embedded as OME MapAnnotations.

This preserves scanner-specific metadata.

---

# 8. Preview System

The preview system:

- Uses the most appropriate pyramid level
- Automatically selects a downsampled level when ROI is large
- Limits preview resolution to improve responsiveness
- Avoids decoding full-resolution data unnecessarily

This enables interactive cropping even for very large WSIs.

---

# 9. Robust Cropping Strategy

For OpenSlide-supported formats (NDPI and pyramidal TIFF):

Primary strategy:
- Attempt block-based reading at level 0 (full resolution).

If decoding fails (e.g., corrupted JPEG tiles):

- Slide handle is reopened.
- Fallback to alternative pyramid levels.
- Regions are upscaled if needed.
- User is notified of partial recovery.

This improves robustness for partially corrupted slides.

---

# 10. Memory and Performance Considerations

- ROI extraction is block-based (default block size: 1024 px).
- Output is written in tiled format (256×256).
- Large ROIs require proportional RAM.
- Preview avoids loading full WSI into memory.

Performance depends on:
- Disk speed
- Pyramid structure
- Compression
- Available RAM

---

# 11. Error Handling

The application includes safeguards for:

- Out-of-bounds coordinates
- Corrupted pyramid levels
- Missing metadata
- File access permissions
- OpenSlide decoding failures

If fallback recovery is used, the user receives a warning dialog.

---

# 12. Frequently Asked Questions

Q: Does cropping modify the original file?  
A: No. A new file is created.

Q: Can multiple ROIs be cropped simultaneously?  
A: No. Each ROI requires a separate action.

Q: What is the maximum supported size?  
A: Limited by available RAM and disk space.

Q: Are special characters allowed in suffix?  
A: Avoid: < > : " / \ | ? *  
Recommended: letters, numbers, underscores, hyphens.

Q: Is internet required?  
A: No.

---

# 13. Development Notes

Core dependencies:

- numpy
- tifffile
- PyQt5
- Pillow
- OpenSlide
- zarr (optional for Zarr-backed TIFF access)

The Windows release bundles required OpenSlide binaries.

---

# 14. License

MIT License  
© 2026 Jose Rodriguez-Rojas