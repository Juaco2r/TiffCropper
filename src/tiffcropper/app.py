
import os
import sys
import re
import math
import csv
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import tifffile

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout,
    QWidget, QPushButton, QLineEdit, QLabel, QFileDialog,
    QSpinBox, QMessageBox, QComboBox, QCheckBox, QGroupBox,
    QStackedWidget, QDoubleSpinBox, QProgressBar, QDialog,
    QTableWidget, QTableWidgetItem, QTextBrowser, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor, QIcon

# ============================================================
# App metadata
# ============================================================

APP_NAME = "TiffCropper"
APP_VERSION = "1.2"
APP_TITLE = "TiffCropper: WSI Crop, Tile and Merge Tool for Digital Pathology Images"
APP_DOI = "10.5281/zenodo.20316535"
APP_GITHUB = "https://github.com/Juaco2r/TiffCropper"
APP_AUTHOR = "José Rodriguez-Rojas"
APP_YEAR = "2026"
APP_LICENSE = "MIT License"
APP_ICON_PATH = "assets/icon/cropper.ico"


def resource_path(relative_path):
    """
    Get absolute path to a bundled resource.

    Works both during normal Python execution and inside a PyInstaller executable.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parents[2]

    return str(base_path / relative_path)
APP_CITATION = (
    f"Rodriguez-Rojas J. {APP_TITLE}. "
    f"Version {APP_VERSION}. Zenodo; {APP_YEAR}. "
    f"doi:{APP_DOI}"
)


# ============================================================
# OpenSlide DLL helper
# ============================================================

def _setup_openslide_dll_path():
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        dll_dir = os.path.join(sys._MEIPASS, "openslide_bin")
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")


_setup_openslide_dll_path()


# ============================================================
# Supported formats
# ============================================================

SUPPORTED_EXTENSIONS = (
    ".tif", ".tiff", ".ome.tif", ".ome.tiff",
    ".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu",
    ".bif", ".svslide", ".dcm",
    ".jpg", ".jpeg", ".png"
)

OPENSLIDE_EXTENSIONS = (
    ".tif", ".tiff", ".ome.tif", ".ome.tiff",
    ".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu",
    ".bif", ".svslide", ".dcm"
)

TIFF_EXTENSIONS = (
    ".tif", ".tiff", ".ome.tif", ".ome.tiff", ".svs"
)

RASTER_EXTENSIONS = (
    ".jpg", ".jpeg", ".png"
)


def _has_ext(path_or_name, exts):
    name = str(path_or_name).lower()
    return any(name.endswith(e) for e in exts)


def _image_file_filter():
    exts = " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS)
    return f"Image Files ({exts});;All Files (*)"


def _tile_file_filter():
    return "Tile Files (*.tif *.tiff *.ome.tif *.ome.tiff *.jpg *.jpeg *.png);;All Files (*)"


# ============================================================
# LIF splitter filters
# ============================================================

LIF_EXTENSIONS = (".lif",)


def _lif_file_filter():
    return "Leica LIF Files (*.lif);;All Files (*)"


def _lif_output_folder_for(lif_path: Path, output_base: Optional[str] = None) -> Path:
    """Return the output folder for one .lif file."""
    if output_base:
        return Path(output_base).expanduser() / f"{lif_path.stem}_split_ome_tiff"
    return lif_path.parent / f"{lif_path.stem}_split_ome_tiff"


# ============================================================
# Optional imports
# ============================================================

def _try_import_openslide():
    try:
        import openslide
        return openslide
    except Exception:
        return None


def _try_import_pil():
    try:
        from PIL import Image
        return Image
    except Exception:
        return None


# ============================================================
# Metadata helpers
# ============================================================

def _mpp_to_dpi(mpp: float) -> float:
    return 25400.0 / float(mpp)


def _dpi_to_mpp(dpi: float) -> float:
    return 25400.0 / float(dpi)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _ascii_safe(s: str) -> str:
    if s is None:
        return ""
    return str(s).encode("ascii", errors="ignore").decode("ascii")


def _tag_to_float(tag):
    if tag is None:
        return None
    v = tag.value
    try:
        if isinstance(v, tuple) and len(v) == 2 and v[1] != 0:
            return float(v[0]) / float(v[1])
        return float(v)
    except Exception:
        return None


def _resolution_to_mpp(xres: float, yres: float, unit: str):
    """Convert TIFF resolution tags to micrometers per pixel.

    TIFF stores resolution as pixels per unit. The physical pixel size is
    therefore the size of that unit divided by the resolution value.
    """
    if not xres or not yres or not unit:
        return None
    unit = str(unit).upper()
    try:
        if unit == "INCH":
            return 25400.0 / float(xres), 25400.0 / float(yres)
        if unit == "CENTIMETER":
            return 10000.0 / float(xres), 10000.0 / float(yres)
    except Exception:
        return None
    return None


def _mpp_to_resolution_tuple(mpp_x: float, mpp_y: float):
    """Return a TIFF resolution tuple in pixels per inch from µm/px."""
    if not mpp_x or not mpp_y:
        return None
    try:
        return (_mpp_to_dpi(float(mpp_x)), _mpp_to_dpi(float(mpp_y)), "INCH")
    except Exception:
        return None


def _convert_physical_size_to_um(value, unit):
    """Convert OME physical size units to micrometers."""
    try:
        value = float(value)
    except Exception:
        return None

    unit = str(unit or "um").strip().lower().replace("µ", "u")

    if unit in ("um", "micrometer", "micrometre", "micrometers", "micrometres"):
        return value
    if unit in ("nm", "nanometer", "nanometre", "nanometers", "nanometres"):
        return value / 1000.0
    if unit in ("mm", "millimeter", "millimetre", "millimeters", "millimetres"):
        return value * 1000.0
    if unit in ("cm", "centimeter", "centimetre", "centimeters", "centimetres"):
        return value * 10000.0
    if unit in ("m", "meter", "metre", "meters", "metres"):
        return value * 1000000.0

    # If the unit is absent or unusual, assume the common OME default: micrometers.
    return value


def _extract_ome_physical_size_um(ome_xml):
    """Extract PhysicalSizeX/Y from OME-XML, returned as (µm/px X, µm/px Y)."""
    if not ome_xml:
        return None

    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(ome_xml)

        # Namespace-aware search first, then a namespace-agnostic fallback.
        pixels = None
        for elem in root.iter():
            if elem.tag.endswith("Pixels"):
                pixels = elem
                break

        if pixels is None:
            return None

        psx = pixels.attrib.get("PhysicalSizeX")
        psy = pixels.attrib.get("PhysicalSizeY")
        if psx is None or psy is None:
            return None

        unit_x = pixels.attrib.get("PhysicalSizeXUnit", "um")
        unit_y = pixels.attrib.get("PhysicalSizeYUnit", "um")

        mpp_x = _convert_physical_size_to_um(psx, unit_x)
        mpp_y = _convert_physical_size_to_um(psy, unit_y)

        if mpp_x is None or mpp_y is None or mpp_x <= 0 or mpp_y <= 0:
            return None
        return float(mpp_x), float(mpp_y)
    except Exception:
        return None


def _scale_resolution_and_mpp(source_resolution=None, source_mpp=None, pixel_scale=1.0):
    """Adjust calibration metadata after image downsampling.

    pixel_scale is the linear pixel-size scale factor. For example, downsample=2
    means output pixels are physically twice as large, so µm/px doubles and
    pixels-per-inch/centimeter resolution halves.
    """
    try:
        pixel_scale = float(pixel_scale)
    except Exception:
        pixel_scale = 1.0
    if pixel_scale <= 0:
        pixel_scale = 1.0

    scaled_resolution = None
    if source_resolution is not None:
        try:
            xres, yres, unit = source_resolution
            if xres and yres and unit:
                scaled_resolution = (
                    float(xres) / pixel_scale,
                    float(yres) / pixel_scale,
                    unit
                )
        except Exception:
            scaled_resolution = None

    scaled_mpp = None
    if source_mpp is not None:
        try:
            mpp_x, mpp_y = source_mpp
            if mpp_x and mpp_y:
                scaled_mpp = (
                    float(mpp_x) * pixel_scale,
                    float(mpp_y) * pixel_scale
                )
        except Exception:
            scaled_mpp = None

    # If only one representation exists, derive the other so TIFF tags and OME XML agree.
    if scaled_mpp is None and scaled_resolution is not None:
        try:
            xres, yres, unit = scaled_resolution
            scaled_mpp = _resolution_to_mpp(xres, yres, unit)
        except Exception:
            scaled_mpp = None

    if scaled_resolution is None and scaled_mpp is not None:
        try:
            scaled_resolution = _mpp_to_resolution_tuple(scaled_mpp[0], scaled_mpp[1])
        except Exception:
            scaled_resolution = None

    return scaled_resolution, scaled_mpp


def _ome_map_annotation_xml(kv: dict, ann_id: str = "Annotation:0") -> str:
    items = []
    for k, v in kv.items():
        k = xml_escape(_ascii_safe(k))
        v = xml_escape(_ascii_safe(v))
        items.append(f'<M K="{k}" V="{v}"/>')
    items_xml = "\n            ".join(items) if items else ""
    return f"""
    <StructuredAnnotations>
      <MapAnnotation ID="{ann_id}">
        <Value>
          <Map>
            {items_xml}
          </Map>
        </Value>
      </MapAnnotation>
    </StructuredAnnotations>
    """.strip()


def _build_ome_xml_rgb(size_x, size_y, physical_size_x_um, physical_size_y_um, image_name, annotation_kv=None):
    psx = f' PhysicalSizeX="{physical_size_x_um:.6f}" PhysicalSizeXUnit="um"' if physical_size_x_um else ""
    psy = f' PhysicalSizeY="{physical_size_y_um:.6f}" PhysicalSizeYUnit="um"' if physical_size_y_um else ""
    ann_xml = _ome_map_annotation_xml(annotation_kv, ann_id="Annotation:0") if annotation_kv else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="{xml_escape(_ascii_safe(image_name))}">
    <Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint8"
            SizeX="{size_x}" SizeY="{size_y}" SizeC="3" SizeZ="1" SizeT="1"{psx}{psy}>
      <Channel ID="Channel:0" SamplesPerPixel="3"/>
      <TiffData IFD="0" PlaneCount="1"/>
    </Pixels>
  </Image>
  {ann_xml}
</OME>
"""


# ============================================================
# Image helpers
# ============================================================

def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255

    arr_float = arr.astype(np.float32, copy=False)
    if arr_float.size == 0:
        return arr_float.astype(np.uint8)

    finite = np.isfinite(arr_float)
    if not np.any(finite):
        return np.zeros(arr_float.shape, dtype=np.uint8)

    valid = arr_float[finite]
    vmin = float(np.percentile(valid, 1))
    vmax = float(np.percentile(valid, 99))

    if vmax <= vmin:
        vmax = float(np.max(valid))
        vmin = float(np.min(valid))
    if vmax <= vmin:
        return np.zeros(arr_float.shape, dtype=np.uint8)

    arr_float = np.clip((arr_float - vmin) / (vmax - vmin), 0, 1)
    return (arr_float * 255).astype(np.uint8)


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 2:
        arr = _normalize_to_uint8(arr)
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = _normalize_to_uint8(arr[:, :, 0])
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 2:
            arr = _normalize_to_uint8(arr[:, :, 0])
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 3:
            arr = _normalize_to_uint8(arr)
        elif arr.shape[2] == 4:
            arr = _normalize_to_uint8(arr[:, :, :3])
        elif arr.shape[2] > 4:
            arr = _normalize_to_uint8(arr[:, :, :3])
        else:
            raise ValueError(f"Unsupported image array shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported image array shape after squeeze: {arr.shape}")

    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def _array_to_rgb_preview(arr: np.ndarray, axes: str = None) -> np.ndarray:
    """Return a display-only RGB preview from grayscale, RGB, or scientific multichannel data.

    This function is intentionally only for visualization. It may normalize and
    select a representative T/Z plane, but it never controls how scientific data
    are saved. Saving raw multichannel crops is handled separately.
    """
    arr = np.asarray(arr)
    axes = (axes or "").strip()

    if axes and len(axes) == arr.ndim and "Y" in axes and "X" in axes:
        slicer = []
        kept_axes = []
        for ax in axes:
            if ax in ("Y", "X", "C", "S"):
                slicer.append(slice(None))
                kept_axes.append(ax)
            else:
                # For preview only, show the first timepoint/Z-plane/etc.
                slicer.append(0)
        arr = arr[tuple(slicer)]
        axes = "".join(kept_axes)

        if "Y" in axes and "X" in axes:
            order = [axes.index("Y"), axes.index("X")]
            if "C" in axes:
                order.append(axes.index("C"))
            elif "S" in axes:
                order.append(axes.index("S"))
            arr = np.transpose(arr, order)

    return _to_uint8_rgb(arr)


def _resize_spatial_array(arr: np.ndarray, axes: str = None, downsample: float = 1.0):
    """Resize only Y/X dimensions while preserving channels, Z, T and dtype."""
    arr = np.asarray(arr)
    axes = (axes or "").strip()
    try:
        downsample = float(downsample)
    except Exception:
        downsample = 1.0
    if downsample == 1.0:
        return arr, axes
    if downsample <= 0:
        raise ValueError("Downsample must be > 0.")

    if axes and len(axes) == arr.ndim and "Y" in axes and "X" in axes:
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
    else:
        if arr.ndim < 2:
            return arr, axes
        y_axis = arr.ndim - 2
        x_axis = arr.ndim - 1

    y_size = int(arr.shape[y_axis])
    x_size = int(arr.shape[x_axis])
    new_y = max(1, int(round(y_size / downsample)))
    new_x = max(1, int(round(x_size / downsample)))

    arr_moved = np.moveaxis(arr, (y_axis, x_axis), (-2, -1))
    prefix_shape = arr_moved.shape[:-2]
    planes = arr_moved.reshape((-1, y_size, x_size))
    resized = np.empty((planes.shape[0], new_y, new_x), dtype=arr.dtype)

    from PIL import Image
    resample = Image.Resampling.LANCZOS if downsample > 1 else Image.Resampling.BILINEAR
    for i, plane in enumerate(planes):
        im = Image.fromarray(np.asarray(plane))
        resized_plane = np.asarray(im.resize((new_x, new_y), resample=resample))
        if resized_plane.dtype != arr.dtype:
            resized_plane = np.clip(resized_plane, np.iinfo(arr.dtype).min, np.iinfo(arr.dtype).max).astype(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else resized_plane.astype(arr.dtype)
        resized[i] = resized_plane

    resized = resized.reshape(prefix_shape + (new_y, new_x))
    resized = np.moveaxis(resized, (-2, -1), (y_axis, x_axis))
    return np.ascontiguousarray(resized), axes


def _guess_axes_for_array(arr: np.ndarray, axes: str = None) -> str:
    """Return a reasonable axes string for tifffile OME writing."""
    arr = np.asarray(arr)
    axes = (axes or "").strip()
    if axes and len(axes) == arr.ndim:
        return axes
    if arr.ndim == 2:
        return "YX"
    if arr.ndim == 3:
        if arr.shape[0] <= 16 and arr.shape[-1] not in (3, 4):
            return "CYX"
        if arr.shape[-1] <= 16:
            return "YXC"
        return "ZYX"
    if arr.ndim == 4:
        return "CZYX"
    if arr.ndim == 5:
        return "TCZYX"
    return ""


def _downsample_for_preview(rgb: np.ndarray, max_side: int = 512) -> np.ndarray:
    rgb = _to_uint8_rgb(rgb)
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    step = int(np.ceil(m / max_side))
    return rgb[::step, ::step, :]


def _numpy_rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(_to_uint8_rgb(rgb))
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ============================================================
# Leica LIF splitter helpers
# ============================================================

def _require_readlif():
    """Import readlif only when the LIF splitter is used."""
    try:
        from readlif.reader import LifFile
        return LifFile
    except Exception as exc:
        raise RuntimeError(
            "The package 'readlif' is not installed in this Python environment.\n\n"
            "Install it in the same environment used to run this app:\n\n"
            "    pip install readlif --no-deps\n\n"
            "If that still fails, run:\n\n"
            "    pip install readlif pillow beautifulsoup4\n\n"
            f"Original import error: {exc}"
        ) from exc


def _lif_safe_name(text: Any, max_len: int = 90) -> str:
    s = str(text or "scene").strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    if not s:
        s = "scene"
    return s[:max_len]


def _lif_human_bytes(n: Optional[float]) -> str:
    if n is None:
        return "unknown"
    try:
        n = float(n)
    except Exception:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"


def _lif_get_int(value: Any, default: int = 1) -> int:
    try:
        if value is None:
            return default
        value = int(value)
        return value if value > 0 else default
    except Exception:
        return default


def _lif_get_image_dims(img: Any) -> Tuple[int, int, int, int, int, int]:
    """
    Return size_x, size_y, size_z, size_t, size_m, size_c.

    readlif commonly exposes dims as (x, y, z, t, m), while channels are
    available through the image.channels attribute.
    """
    dims = getattr(img, "dims", None) or ()
    size_x = _lif_get_int(dims[0] if len(dims) > 0 else None)
    size_y = _lif_get_int(dims[1] if len(dims) > 1 else None)
    size_z = _lif_get_int(dims[2] if len(dims) > 2 else getattr(img, "nz", 1))
    size_t = _lif_get_int(dims[3] if len(dims) > 3 else getattr(img, "nt", 1))
    size_m = _lif_get_int(dims[4] if len(dims) > 4 else 1)
    size_c = _lif_get_int(getattr(img, "channels", 1))
    size_z = _lif_get_int(getattr(img, "nz", size_z), size_z)
    size_t = _lif_get_int(getattr(img, "nt", size_t), size_t)
    return size_x, size_y, size_z, size_t, size_m, size_c


def _lif_physical_sizes_um(img: Any) -> Dict[str, Optional[float]]:
    """
    Extract approximate physical pixel size from readlif scale metadata.

    readlif scale is usually expressed as pixels per micrometer, so the physical
    size in micrometers per pixel is 1 / scale.
    """
    out = {"PhysicalSizeX": None, "PhysicalSizeY": None, "PhysicalSizeZ": None}
    scale = getattr(img, "scale", None)
    if not scale:
        return out
    for i, key in enumerate(["PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ"]):
        try:
            value = float(scale[i])
            if value > 0:
                out[key] = 1.0 / value
        except Exception:
            pass
    return out


def _lif_channel_names(img: Any, size_c: int) -> List[str]:
    candidates = []
    for attr_name in ["channel_names", "channels_names", "channel_name"]:
        try:
            value = getattr(img, attr_name)
            if value:
                candidates = list(value)
                break
        except Exception:
            pass
    names = []
    for i in range(size_c):
        if i < len(candidates) and candidates[i]:
            names.append(str(candidates[i]))
        else:
            names.append(f"Channel_{i}")
    return names


def _lif_json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_lif_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _lif_json_safe(v) for k, v in obj.items()}
    return str(obj)


def _lif_scene_metadata_dict(img: Any, scene_index: int) -> Dict[str, Any]:
    size_x, size_y, size_z, size_t, size_m, size_c = _lif_get_image_dims(img)
    metadata = {
        "scene_index": scene_index,
        "name": getattr(img, "name", ""),
        "path": getattr(img, "path", ""),
        "dims_readlif_xyztm": getattr(img, "dims", None),
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "size_t": size_t,
        "size_m": size_m,
        "size_c": size_c,
        "channels": getattr(img, "channels", None),
        "nz": getattr(img, "nz", None),
        "nt": getattr(img, "nt", None),
        "scale_px_per_um": getattr(img, "scale", None),
        "scale_n": getattr(img, "scale_n", None),
        "bit_depth": getattr(img, "bit_depth", None),
        "mosaic_position": getattr(img, "mosaic_position", None),
        "settings": getattr(img, "settings", None),
        "info": getattr(img, "info", None),
    }
    metadata.update(_lif_physical_sizes_um(img))
    return _lif_json_safe(metadata)



def _lif_first_frame_array(img: Any) -> np.ndarray:
    frame = img.get_frame(z=0, t=0, c=0, m=0)
    arr = np.asarray(frame)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported first frame shape from readlif: {arr.shape}")
    return arr


def _lif_bit_depth_text(bit_depth: Any) -> str:
    """Format readlif bit-depth metadata for display.

    Some Leica files expose this as a tuple, for example (12, 12, 12, 12),
    which means 12-bit acquisition for each channel. Showing the raw tuple in
    the table is confusing, so this function makes it readable.
    """
    if bit_depth is None or bit_depth == "":
        return "unknown"
    try:
        if isinstance(bit_depth, (list, tuple)):
            values = [int(v) for v in bit_depth if v is not None]
            if not values:
                return "unknown"
            unique = sorted(set(values))
            if len(unique) == 1:
                return f"{unique[0]}-bit × {len(values)} ch"
            return ", ".join(f"C{i}:{v}-bit" for i, v in enumerate(values))
        return f"{int(bit_depth)}-bit"
    except Exception:
        return str(bit_depth)


def _lif_max_bit_depth(bit_depth: Any) -> Optional[int]:
    try:
        if isinstance(bit_depth, (list, tuple)):
            vals = [int(v) for v in bit_depth if v is not None]
            return max(vals) if vals else None
        if bit_depth is not None and bit_depth != "":
            return int(bit_depth)
    except Exception:
        return None
    return None


def _lif_infer_storage_dtype_text(img: Any, frame_mode: Optional[str] = None) -> str:
    """Infer the likely stored dtype for the table without reading a full frame."""
    max_bit = _lif_max_bit_depth(getattr(img, "bit_depth", None))
    if max_bit is not None:
        if max_bit <= 8:
            return "uint8"
        if max_bit <= 16:
            return "uint16"
        return f">16-bit"

    mode = str(frame_mode or getattr(img, "mode", "") or "")
    mode_lower = mode.lower()
    if "16" in mode_lower:
        return "uint16"
    if mode in ("L", "P", "RGB", "RGBA"):
        return "uint8"
    if mode in ("I", "I;32"):
        return "int32"
    if mode == "F":
        return "float32"
    return "unknown"


def _lif_preview_plane_from_frame(frame: Any, max_side: int) -> Tuple[np.ndarray, str]:
    """Return a small 2D preview plane and source mode text.

    The previous preview path used PIL.thumbnail with bilinear resampling. Some
    Leica 12/16-bit frames are opened by Pillow in modes such as I;16, where
    bilinear thumbnailing can raise "image has wrong mode". This function uses
    a safer NEAREST resize first, then falls back to conservative conversions.
    Export is unaffected because this is only for GUI thumbnails.
    """
    mode_text = str(getattr(frame, "mode", ""))

    if hasattr(frame, "size") and hasattr(frame, "resize"):
        try:
            from PIL import Image
            nearest = Image.Resampling.NEAREST
            bilinear = Image.Resampling.BILINEAR
        except Exception:
            nearest = 0
            bilinear = 2

        try:
            w, h = frame.size
            scale = min(float(max_side) / max(float(w), float(h)), 1.0)
            new_w = max(1, int(round(float(w) * scale)))
            new_h = max(1, int(round(float(h) * scale)))
        except Exception:
            new_w = max_side
            new_h = max_side

        # Try direct NEAREST resize first. This usually works for 16-bit modes
        # and avoids the "wrong mode" error caused by bilinear resampling.
        for resample in (nearest, bilinear):
            try:
                small = frame.resize((new_w, new_h), resample=resample)
                arr = np.asarray(small)
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                return np.asarray(arr), mode_text
            except Exception:
                pass

        # Last-resort preview fallback: convert only for display.
        # This may reduce the visual bit depth, but does not affect export.
        for convert_mode in ("I", "F", "L"):
            try:
                small = frame.convert(convert_mode).resize((new_w, new_h), resample=nearest)
                arr = np.asarray(small)
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                return np.asarray(arr), f"{mode_text}->{convert_mode}"
            except Exception:
                pass

    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = _downsample_for_preview(arr, max_side=max_side)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return np.asarray(arr), mode_text


def _lif_match_plane_shape(plane: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize a small preview plane if channels produced slightly different sizes."""
    plane = np.asarray(plane)
    if plane.shape[:2] == target_shape:
        return plane
    try:
        from PIL import Image
        pil = Image.fromarray(_normalize_to_uint8(plane))
        pil = pil.resize((int(target_shape[1]), int(target_shape[0])), resample=Image.Resampling.NEAREST)
        return np.asarray(pil)
    except Exception:
        out = np.zeros(target_shape, dtype=plane.dtype if hasattr(plane, "dtype") else np.uint8)
        h = min(out.shape[0], plane.shape[0])
        w = min(out.shape[1], plane.shape[1])
        out[:h, :w] = plane[:h, :w]
        return out


def _lif_false_color_preview(planes_u8: List[np.ndarray]) -> np.ndarray:
    """Combine LIF channels into a simple IF-style false-color RGB preview."""
    if not planes_u8:
        raise ValueError("No planes available for preview.")
    if len(planes_u8) == 1:
        return np.stack([planes_u8[0], planes_u8[0], planes_u8[0]], axis=-1).astype(np.uint8)

    h, w = planes_u8[0].shape[:2]
    acc = np.zeros((h, w, 3), dtype=np.float32)

    # Common IF visualization: C0 often DAPI -> blue, then green, red, magenta.
    colors = [
        (0.0, 0.0, 1.0),  # channel 0: blue
        (0.0, 1.0, 0.0),  # channel 1: green
        (1.0, 0.0, 0.0),  # channel 2: red
        (1.0, 0.0, 1.0),  # channel 3: magenta
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0),
    ]
    for i, plane in enumerate(planes_u8):
        color = colors[i % len(colors)]
        p = plane.astype(np.float32) / 255.0
        acc[:, :, 0] += p * color[0]
        acc[:, :, 1] += p * color[1]
        acc[:, :, 2] += p * color[2]

    acc = np.clip(acc, 0.0, 1.0)
    return np.ascontiguousarray((acc * 255.0).astype(np.uint8))


def _lif_thumbnail_rgb(img: Any, max_side: int = 170, max_channels: int = 4) -> Tuple[np.ndarray, str]:
    """
    Create a lightweight visual preview for one LIF scene.

    For IF, it combines the first Z/T/M plane of up to four channels using a
    false-color display. This is only for display; export still writes the
    original raw data without RGB conversion or intensity normalization.
    """
    size_x, size_y, size_z, size_t, size_m, size_c = _lif_get_image_dims(img)
    n = max(1, min(size_c, max_channels))
    planes = []
    mode_texts = []
    dtype_text = _lif_infer_storage_dtype_text(img)

    target_shape = None
    for c in range(n):
        frame = img.get_frame(z=0, t=0, c=c, m=0)
        try:
            arr, mode_text = _lif_preview_plane_from_frame(frame, max_side=max_side)
            mode_texts.append(mode_text)
            if dtype_text == "unknown":
                dtype_text = str(arr.dtype)
            if target_shape is None:
                target_shape = arr.shape[:2]
            else:
                arr = _lif_match_plane_shape(arr, target_shape)
            planes.append(_normalize_to_uint8(arr))
        finally:
            try:
                del frame
            except Exception:
                pass

    if not planes:
        raise ValueError("Could not create thumbnail.")

    rgb = _lif_false_color_preview(planes)
    mode_texts = [m for m in mode_texts if m]
    if mode_texts:
        unique_modes = []
        for m in mode_texts:
            if m not in unique_modes:
                unique_modes.append(m)
        dtype_text = f"{dtype_text}; mode {'/'.join(unique_modes)}"
    return rgb, dtype_text


def _lif_scene_geometry_from_first_frame(img: Any, first_arr: np.ndarray) -> Dict[str, Any]:
    size_x, size_y, size_z, size_t, size_m, size_c = _lif_get_image_dims(img)

    if first_arr.ndim == 2:
        frame_y, frame_x = first_arr.shape
        sample_count = 1
    elif first_arr.ndim == 3:
        frame_y, frame_x, sample_count = first_arr.shape
    else:
        raise ValueError(f"Unsupported first frame shape: {first_arr.shape}")

    if frame_y != size_y or frame_x != size_x:
        size_y, size_x = int(frame_y), int(frame_x)

    out_t = int(size_t) * int(size_m)
    out_c = int(size_c) * int(sample_count)
    out_z = int(size_z)
    shape_tczyx = (out_t, out_c, out_z, int(size_y), int(size_x))
    total_planes = int(out_t * out_c * out_z)
    estimated_bytes = int(np.prod(shape_tczyx, dtype=np.int64)) * int(first_arr.dtype.itemsize)
    per_plane_bytes = int(size_y) * int(size_x) * int(first_arr.dtype.itemsize)

    return {
        "shape_tczyx": shape_tczyx,
        "dtype": first_arr.dtype,
        "sample_count_per_frame": int(sample_count),
        "readlif_size_t": int(size_t),
        "readlif_size_m": int(size_m),
        "readlif_size_c": int(size_c),
        "size_z": int(size_z),
        "size_y": int(size_y),
        "size_x": int(size_x),
        "total_planes": total_planes,
        "estimated_bytes": estimated_bytes,
        "per_plane_bytes": per_plane_bytes,
        "mosaic_flattened_into_t": bool(size_m > 1),
        "rgb_samples_folded_into_c": bool(sample_count > 1),
    }


def _lif_plane_iterator(
    img: Any,
    first_arr: np.ndarray,
    geom: Dict[str, Any],
    progress_callback=None,
) -> Iterable[np.ndarray]:
    """Yield 2D planes in TCZYX order without building the full stack in RAM."""
    size_t = geom["readlif_size_t"]
    size_m = geom["readlif_size_m"]
    size_c = geom["readlif_size_c"]
    size_z = geom["size_z"]
    sample_count = geom["sample_count_per_frame"]
    total_planes = geom["total_planes"]

    done = 0
    first_used = False

    for t in range(size_t):
        for m in range(size_m):
            for c in range(size_c):
                for z in range(size_z):
                    if not first_used and t == 0 and m == 0 and c == 0 and z == 0:
                        frame_arr = first_arr
                        first_used = True
                    else:
                        frame = img.get_frame(z=z, t=t, c=c, m=m)
                        frame_arr = np.asarray(frame)

                    if frame_arr.ndim == 2:
                        done += 1
                        if progress_callback is not None:
                            progress_callback(done, total_planes)
                        yield np.ascontiguousarray(frame_arr)
                    elif frame_arr.ndim == 3:
                        for s in range(sample_count):
                            done += 1
                            if progress_callback is not None:
                                progress_callback(done, total_planes)
                            yield np.ascontiguousarray(frame_arr[:, :, s])
                    else:
                        raise ValueError(f"Unsupported frame shape while reading: {frame_arr.shape}")

                    if first_used and frame_arr is not first_arr:
                        try:
                            del frame_arr
                        except Exception:
                            pass


def _lif_save_scene_ome_tiff_lowmem(
    img: Any,
    scene_index: int,
    out_path: Path,
    overwrite: bool = False,
    skip_existing: bool = True,
    compression: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """Export one readlif scene/page as one OME-TIFF using low-memory streaming."""
    out_path = Path(out_path)
    if out_path.exists():
        if skip_existing and not overwrite:
            return {
                "output_path": str(out_path),
                "output_size_bytes": out_path.stat().st_size,
                "skipped_existing": True,
            }
        if not overwrite:
            raise FileExistsError(f"Output exists and overwrite is disabled: {out_path}")

    size_x, size_y, size_z, size_t, size_m, size_c = _lif_get_image_dims(img)
    channel_names = _lif_channel_names(img, size_c)
    physical = _lif_physical_sizes_um(img)

    first_arr = _lif_first_frame_array(img)
    geom = _lif_scene_geometry_from_first_frame(img, first_arr)

    if geom["sample_count_per_frame"] > 1:
        expanded = []
        sample_labels = ["R", "G", "B", "A"]
        for base in channel_names:
            for s in range(geom["sample_count_per_frame"]):
                label = sample_labels[s] if s < len(sample_labels) else f"S{s}"
                expanded.append(f"{base}_{label}")
        channel_names = expanded

    metadata = {
        "axes": "TCZYX",
        "Channel": {"Name": channel_names},
    }
    for key, value in physical.items():
        if value is not None:
            metadata[key] = float(value)
            metadata[f"{key}Unit"] = "µm"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".part")
    if tmp_path.exists():
        if overwrite:
            tmp_path.unlink()
        else:
            raise FileExistsError(f"Temporary output exists. Delete it or enable overwrite: {tmp_path}")

    imwrite_kwargs = {
        "bigtiff": True,
        "ome": True,
        "metadata": metadata,
        "software": f"{APP_NAME} v{APP_VERSION} LIF splitter",
        "shape": geom["shape_tczyx"],
        "dtype": geom["dtype"],
    }
    if compression and str(compression).lower() not in ("none", ""):
        imwrite_kwargs["compression"] = compression

    try:
        plane_iter = _lif_plane_iterator(img, first_arr, geom, progress_callback=progress_callback)
        tifffile.imwrite(str(tmp_path), plane_iter, **imwrite_kwargs)
        if out_path.exists() and overwrite:
            out_path.unlink()
        os.replace(str(tmp_path), str(out_path))
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise
    finally:
        try:
            del first_arr
        except Exception:
            pass

    result = {
        "output_path": str(out_path),
        "output_shape_tczyx": "x".join(str(x) for x in geom["shape_tczyx"]),
        "output_dtype": str(geom["dtype"]),
        "output_size_bytes": out_path.stat().st_size if out_path.exists() else None,
        "estimated_raw_bytes": geom["estimated_bytes"],
        "per_plane_bytes": geom["per_plane_bytes"],
        "channel_names": ";".join(channel_names),
        "skipped_existing": False,
        "readlif_size_t": geom["readlif_size_t"],
        "readlif_size_m": geom["readlif_size_m"],
        "readlif_size_c": geom["readlif_size_c"],
        "sample_count_per_frame": geom["sample_count_per_frame"],
        "mosaic_flattened_into_t": geom["mosaic_flattened_into_t"],
        "rgb_samples_folded_into_c": geom["rgb_samples_folded_into_c"],
    }
    result.update(physical)
    return result


def _lif_write_xml_header(lif_obj: Any, lif_path: Path, out_dir: Path) -> Optional[Path]:
    try:
        xml_text = getattr(lif_obj, "xml_header", None)
        if callable(xml_text):
            xml_text = xml_text()
        if not xml_text:
            return None
        xml_path = out_dir / f"{_lif_safe_name(lif_path.stem)}_Leica_XML_header.xml"
        xml_path.write_text(str(xml_text), encoding="utf-8", errors="replace")
        return xml_path
    except Exception:
        return None


def _lif_write_scene_json(metadata: Dict[str, Any], out_dir: Path, scene_index: int, scene_name: str) -> Optional[Path]:
    try:
        json_dir = out_dir / "metadata_json"
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / f"scene_{scene_index:03d}_{_lif_safe_name(scene_name)}_metadata.json"
        json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return json_path
    except Exception:
        return None


def _lif_write_manifest(manifest_path: Path, rows: List[Dict[str, Any]]) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    return manifest_path


# ============================================================
# Crop rectangle widget
# ============================================================

class CropSelectionLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self._pixmap_original = None
        self._thumb_w = None
        self._thumb_h = None
        self._full_w = None
        self._full_h = None
        self._dragging = False
        self._start = QPoint()
        self._end = QPoint()
        self._selection_rect_widget = None
        self.selection_callback = None
        self.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")

    def set_image(self, rgb, full_w, full_h, callback=None):
        rgb = _to_uint8_rgb(rgb)
        self._thumb_h, self._thumb_w = rgb.shape[:2]
        self._full_w = int(full_w)
        self._full_h = int(full_h)
        self.selection_callback = callback
        self._pixmap_original = _numpy_rgb_to_qpixmap(rgb)
        self._selection_rect_widget = None
        self.update()

    def has_image(self):
        return self._pixmap_original is not None and self._full_w is not None and self._full_h is not None

    def _display_rect(self):
        if self._pixmap_original is None:
            return QRect(0, 0, 0, 0)
        label_w = self.width()
        label_h = self.height()
        img_w = self._pixmap_original.width()
        img_h = self._pixmap_original.height()
        if img_w <= 0 or img_h <= 0 or label_w <= 0 or label_h <= 0:
            return QRect(0, 0, 0, 0)
        scale = min(label_w / img_w, label_h / img_h)
        disp_w = int(round(img_w * scale))
        disp_h = int(round(img_h * scale))
        x0 = int(round((label_w - disp_w) / 2))
        y0 = int(round((label_h - disp_h) / 2))
        return QRect(x0, y0, disp_w, disp_h)

    def _clamp_point_to_display_rect(self, p: QPoint):
        r = self._display_rect()
        x = max(r.left(), min(p.x(), r.right()))
        y = max(r.top(), min(p.y(), r.bottom()))
        return QPoint(x, y)

    def _widget_rect_to_full_coords(self, rect: QRect):
        if not self.has_image():
            return None
        disp = self._display_rect()
        if disp.width() <= 0 or disp.height() <= 0:
            return None
        inter = rect.normalized().intersected(disp)
        if inter.width() <= 1 or inter.height() <= 1:
            return None

        x0_img = (inter.left() - disp.left()) / disp.width() * self._thumb_w
        y0_img = (inter.top() - disp.top()) / disp.height() * self._thumb_h
        x1_img = (inter.right() - disp.left()) / disp.width() * self._thumb_w
        y1_img = (inter.bottom() - disp.top()) / disp.height() * self._thumb_h

        x0_full = int(round(x0_img / self._thumb_w * self._full_w))
        y0_full = int(round(y0_img / self._thumb_h * self._full_h))
        x1_full = int(round(x1_img / self._thumb_w * self._full_w))
        y1_full = int(round(y1_img / self._thumb_h * self._full_h))

        x0_full = max(0, min(x0_full, self._full_w - 1))
        y0_full = max(0, min(y0_full, self._full_h - 1))
        x1_full = max(1, min(x1_full, self._full_w))
        y1_full = max(1, min(y1_full, self._full_h))

        x = min(x0_full, x1_full)
        y = min(y0_full, y1_full)
        w = abs(x1_full - x0_full)
        h = abs(y1_full - y0_full)
        return x, y, max(1, w), max(1, h)

    def set_selection_from_full_coords(self, x, y, w, h):
        if not self.has_image():
            return
        disp = self._display_rect()
        if disp.width() <= 0 or disp.height() <= 0:
            return
        x0 = float(x)
        y0 = float(y)
        x1 = float(x + w)
        y1 = float(y + h)
        px0 = disp.left() + (x0 / self._full_w) * disp.width()
        py0 = disp.top() + (y0 / self._full_h) * disp.height()
        px1 = disp.left() + (x1 / self._full_w) * disp.width()
        py1 = disp.top() + (y1 / self._full_h) * disp.height()
        self._selection_rect_widget = QRect(
            int(round(px0)),
            int(round(py0)),
            int(round(px1 - px0)),
            int(round(py1 - py0))
        ).normalized()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.has_image():
            disp = self._display_rect()
            if disp.contains(event.pos()):
                self._dragging = True
                self._start = self._clamp_point_to_display_rect(event.pos())
                self._end = self._start
                self._selection_rect_widget = QRect(self._start, self._end)
                self.update()

    def mouseMoveEvent(self, event):
        if self._dragging and self.has_image():
            self._end = self._clamp_point_to_display_rect(event.pos())
            self._selection_rect_widget = QRect(self._start, self._end).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self._end = self._clamp_point_to_display_rect(event.pos())
            self._selection_rect_widget = QRect(self._start, self._end).normalized()
            coords = self._widget_rect_to_full_coords(self._selection_rect_widget)
            if coords is not None and self.selection_callback is not None:
                x, y, w, h = coords
                self.selection_callback(x, y, w, h)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))
        if self._pixmap_original is not None:
            disp = self._display_rect()
            scaled = self._pixmap_original.scaled(disp.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(disp.topLeft(), scaled)
        if self._selection_rect_widget is not None:
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.drawRect(self._selection_rect_widget.normalized())
            painter.fillRect(self._selection_rect_widget.normalized(), QColor(255, 0, 0, 35))
        painter.end()


# ============================================================
# Tile helpers
# ============================================================

def _col_to_letters(col_index: int) -> str:
    letters = ""
    col_index += 1
    while col_index > 0:
        col_index, rem = divmod(col_index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _letters_to_col(letters: str) -> int:
    col = 0
    for ch in letters.upper():
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return col - 1


def _suffix_for_tile(overlap_percent: float, downsample: float) -> str:
    suffix = ""
    if float(overlap_percent) != 0:
        suffix += f"Ov{int(overlap_percent)}" if float(overlap_percent).is_integer() else f"Ov{overlap_percent:g}"
    if float(downsample) != 1:
        suffix += f"DS{int(downsample)}" if float(downsample).is_integer() else f"DS{downsample:g}"
    return suffix


def _suffix_for_division_tile(rows: int, cols: int, downsample: float) -> str:
    suffix = f"Div{rows}x{cols}"
    if float(downsample) != 1:
        suffix += f"DS{int(downsample)}" if float(downsample).is_integer() else f"DS{downsample:g}"
    return suffix


def _extension_from_combo(text: str) -> str:
    return ".jpg" if "JPEG" in text.upper() or "JPG" in text.upper() else ".tif"


def _write_format_from_combo(text: str) -> str:
    return "jpeg" if "JPEG" in text.upper() or "JPG" in text.upper() else "tiff"


def _parse_tile_name(path: Path):
    stem = path.stem
    if stem.endswith(".ome"):
        stem = Path(stem).stem

    pattern_fixed = re.compile(
        r"^(?P<base>.+?)_(?P<col>[A-Z]+)(?P<row>\d+)"
        r"(?:Ov(?P<ov>\d+(?:\.\d+)?))?"
        r"(?:DS(?P<ds>\d+(?:\.\d+)?))?"
        r"(?:_X(?P<x>\d+)_Y(?P<y>\d+)_W(?P<w>\d+)_H(?P<h>\d+))?$"
    )
    m = pattern_fixed.match(stem)
    if m:
        return {
            "path": path,
            "base": m.group("base"),
            "col": _letters_to_col(m.group("col")),
            "row": int(m.group("row")) - 1,
            "overlap": float(m.group("ov")) if m.group("ov") else 0.0,
            "downsample": float(m.group("ds")) if m.group("ds") else 1.0,
            "x": int(m.group("x")) if m.group("x") else None,
            "y": int(m.group("y")) if m.group("y") else None,
            "w": int(m.group("w")) if m.group("w") else None,
            "h": int(m.group("h")) if m.group("h") else None,
        }

    pattern_div = re.compile(
        r"^(?P<base>.+?)_R(?P<row>\d+)_C(?P<col>\d+)"
        r"(?:Div(?P<rows>\d+)x(?P<cols>\d+))?"
        r"(?:DS(?P<ds>\d+(?:\.\d+)?))?"
        r"(?:_X(?P<x>\d+)_Y(?P<y>\d+)_W(?P<w>\d+)_H(?P<h>\d+))?$"
    )
    m = pattern_div.match(stem)
    if m:
        return {
            "path": path,
            "base": m.group("base"),
            "col": int(m.group("col")) - 1,
            "row": int(m.group("row")) - 1,
            "overlap": 0.0,
            "downsample": float(m.group("ds")) if m.group("ds") else 1.0,
            "x": int(m.group("x")) if m.group("x") else None,
            "y": int(m.group("y")) if m.group("y") else None,
            "w": int(m.group("w")) if m.group("w") else None,
            "h": int(m.group("h")) if m.group("h") else None,
        }

    return None


def _compute_tile_grid(width: int, height: int, tile_size: int, overlap_percent: float):
    overlap_px = int(round(tile_size * overlap_percent / 100.0))
    stride = tile_size - overlap_px
    if stride <= 0:
        raise ValueError("Overlap must be lower than 100%.")

    x_positions = list(range(0, width, stride))
    y_positions = list(range(0, height, stride))

    return x_positions, y_positions, stride, overlap_px


def _division_bounds(base_x, base_y, base_w, base_h, rows, cols, row_idx, col_idx):
    x0 = base_x + int(math.floor(col_idx * base_w / cols))
    x1 = base_x + int(math.floor((col_idx + 1) * base_w / cols))
    y0 = base_y + int(math.floor(row_idx * base_h / rows))
    y1 = base_y + int(math.floor((row_idx + 1) * base_h / rows))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _crop_external_padding(rgb: np.ndarray, padding_color: str = "black") -> np.ndarray:
    """
    Crop uniform external padding without allocating coordinate arrays.

    Previous versions used np.where(mask), which can allocate huge int64
    coordinate arrays for large merged images. Here we reduce the mask to
    occupied rows/columns first, which is much more memory efficient.
    """
    rgb = _to_uint8_rgb(rgb)
    if rgb.size == 0:
        return rgb

    if padding_color.lower() == "white":
        content_mask = np.any(rgb < 250, axis=2)
    else:
        content_mask = np.any(rgb > 5, axis=2)

    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)

    if not rows.any() or not cols.any():
        return rgb

    y0 = int(np.argmax(rows))
    y1 = int(len(rows) - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(len(cols) - np.argmax(cols[::-1]))

    return rgb[y0:y1, x0:x1, :]


# ============================================================
# Image backend
# ============================================================

class ImageBackend:
    def __init__(self):
        self.path = None
        self.path_obj = None
        self.reader = None
        self.file_kind = None
        self.slide_dims = None
        self.source_resolution = None
        self.source_mpp = None
        self.openslide_props = {}
        self._fail_log = []
        self._openslide_obj = None
        self._os_level_count = None
        self._os_downsamples = None
        self._os_level_dimensions = None
        self._tif_obj = None
        self._tif_series = None
        self._tif_axes = None
        self._zarr_array = None
        self._zarr_error = None

    def load(self, path: str):
        # Close any cached reader before loading a new file.
        self.close()
        self.path = path
        self.path_obj = Path(path)
        self.reader = None
        self.file_kind = None
        self.slide_dims = None
        self.source_resolution = None
        self.source_mpp = None
        self.openslide_props = {}
        self._fail_log = []
        self._openslide_obj = None
        self._os_level_count = None
        self._os_downsamples = None
        self._os_level_dimensions = None
        self._tif_obj = None
        self._tif_series = None
        self._tif_axes = None
        self._zarr_array = None
        self._zarr_error = None

        lower_name = self.path_obj.name.lower()

        if _has_ext(lower_name, OPENSLIDE_EXTENSIONS):
            try:
                w, h, res, mpp, props = self._probe_openslide(path)
                self.reader = "openslide"
                self.file_kind = "wsi"
                self.slide_dims = (int(w), int(h))
                self.source_resolution = res
                self.source_mpp = mpp
                self.openslide_props = props or {}
                return self
            except Exception as e:
                self._fail_log.append(("OpenSlide", str(e)))

        if _has_ext(lower_name, TIFF_EXTENSIONS):
            try:
                w, h, res, mpp = self._probe_tifffile(path)
                self.reader = "tifffile"
                self.file_kind = "tiff"
                self.slide_dims = (int(w), int(h))
                self.source_resolution = res
                self.source_mpp = mpp
                self.openslide_props = {}
                return self
            except Exception as e:
                self._fail_log.append(("tifffile", str(e)))

        if _has_ext(lower_name, RASTER_EXTENSIONS):
            try:
                arr = self._read_with_pil(path)
                h, w = arr.shape[:2]
                self.reader = "pil"
                self.file_kind = "raster"
                self.slide_dims = (int(w), int(h))
                self.source_resolution = None
                self.source_mpp = None
                self.openslide_props = {}
                return self
            except Exception as e:
                self._fail_log.append(("PIL", str(e)))

        msg = [f"Could not open image:\n{path}\n"]
        msg.append(f"Detected extension: {self.path_obj.suffix}")
        msg.append("\nTried the following readers:")
        for reader_name, err in self._fail_log:
            msg.append(f"\n- {reader_name} failed:\n  {err}")
        msg.append(
            "\n\nSupported extensions:\n"
            f"{SUPPORTED_EXTENSIONS}\n\n"
            "Suggestions:\n"
            "- Install OpenSlide support:\n"
            "  pip install openslide-python openslide-bin\n\n"
            "- For MRXS, make sure the .mrxs file is beside its associated data folder.\n"
            "- For TIFF/OME-TIFF, make sure tifffile and zarr are installed:\n"
            "  pip install tifffile zarr\n"
        )
        raise RuntimeError("\n".join(msg))

    def close(self):
        """Close any cached image reader handles.

        Keeping an OpenSlide handle open is much faster for repeated tile reads,
        but it should be closed when switching files or after batch jobs.
        """
        if getattr(self, "_openslide_obj", None) is not None:
            try:
                self._openslide_obj.close()
            except Exception:
                pass
        self._openslide_obj = None
        if getattr(self, "_tif_obj", None) is not None:
            try:
                self._tif_obj.close()
            except Exception:
                pass
        self._tif_obj = None
        self._tif_series = None
        self._tif_axes = None
        self._zarr_array = None
        self._zarr_error = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_openslide(self):
        openslide = _try_import_openslide()
        if openslide is None:
            raise RuntimeError("OpenSlide not available.")
        if self._openslide_obj is None:
            self._openslide_obj = openslide.OpenSlide(self.path)
            self._os_level_count = self._openslide_obj.level_count
            self._os_downsamples = list(self._openslide_obj.level_downsamples)
            self._os_level_dimensions = list(self._openslide_obj.level_dimensions)
        return self._openslide_obj

    def _reopen_openslide(self):
        self.close()
        return self._get_openslide()

    def _get_tiff_zarr(self):
        """Open a TIFF/OME-TIFF once and reuse its zarr view for repeated crops."""
        if self._tif_obj is None:
            self._tif_obj = tifffile.TiffFile(self.path)
            self._tif_series = self._tif_obj.series[0]
            self._tif_axes = getattr(self._tif_series, "axes", "")
            try:
                z = self._tif_series.aszarr()
                import zarr
                self._zarr_array = zarr.open(z, mode="r")
                self._zarr_error = None
            except Exception as e:
                self._zarr_array = None
                self._zarr_error = e
        return self._zarr_array, self._tif_series, self._tif_axes, self._zarr_error

    def _read_with_pil(self, path: str) -> np.ndarray:
        Image = _try_import_pil()
        if Image is None:
            raise RuntimeError("PIL/Pillow is not installed. Install with: pip install pillow")
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    def _probe_openslide(self, path: str):
        openslide = _try_import_openslide()
        if openslide is None:
            raise RuntimeError(
                "OpenSlide is not installed or cannot be imported.\n"
                "Install with:\n"
                "pip install openslide-python openslide-bin"
            )
        slide = openslide.OpenSlide(path)
        try:
            w, h = slide.dimensions
            props = dict(slide.properties or {})
            mpp_x = _safe_float(props.get("openslide.mpp-x"))
            mpp_y = _safe_float(props.get("openslide.mpp-y"))
            res_tuple = None
            mpp = None
            if mpp_x and mpp_y:
                res_tuple = (_mpp_to_dpi(mpp_x), _mpp_to_dpi(mpp_y), "INCH")
                mpp = (mpp_x, mpp_y)
            return int(w), int(h), res_tuple, mpp, props
        finally:
            try:
                slide.close()
            except Exception:
                pass

    def _probe_tifffile(self, path: str):
        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError("No image series found in TIFF/OME-TIFF.")

            # Prefer OME-XML physical pixel size when present because many OME-TIFF
            # files store calibration there rather than in classic TIFF resolution tags.
            ome_mpp = None
            try:
                ome_mpp = _extract_ome_physical_size_um(tif.ome_metadata)
            except Exception:
                ome_mpp = None

            s0 = tif.series[0]
            shape0 = s0.shape
            axes = getattr(s0, "axes", "")
            if "X" in axes and "Y" in axes:
                w = int(shape0[axes.index("X")])
                h = int(shape0[axes.index("Y")])
            else:
                h, w = int(shape0[0]), int(shape0[1])

            page0 = s0.pages[0] if getattr(s0, "pages", None) else tif.pages[0]
            tags = page0.tags
            xres_f = _tag_to_float(tags.get("XResolution"))
            yres_f = _tag_to_float(tags.get("YResolution"))
            unit_str = None
            ru = tags.get("ResolutionUnit")
            if ru is not None:
                try:
                    u = int(ru.value)
                    if u == 2:
                        unit_str = "INCH"
                    elif u == 3:
                        unit_str = "CENTIMETER"
                except Exception:
                    unit_str = None

            res_tuple = (xres_f, yres_f, unit_str) if (xres_f and yres_f and unit_str) else None

            if ome_mpp is not None:
                mpp = ome_mpp
            else:
                mpp = _resolution_to_mpp(xres_f, yres_f, unit_str) if res_tuple else None

            # If OME metadata provided the pixel size but TIFF resolution tags were
            # absent/unusable, create a matching pixels-per-inch resolution tuple.
            if res_tuple is None and mpp is not None:
                res_tuple = _mpp_to_resolution_tuple(mpp[0], mpp[1])

            return int(w), int(h), res_tuple, mpp

    @staticmethod
    def clip_roi(x, y, w, h, full_w, full_h):
        x = max(0, min(int(x), int(full_w) - 1))
        y = max(0, min(int(y), int(full_h) - 1))
        w = max(1, min(int(w), int(full_w) - x))
        h = max(1, min(int(h), int(full_h) - y))
        return x, y, w, h

    def crop(self, x: int, y: int, w: int, h: int, fill: int = 255):
        if not self.path or not self.reader or not self.slide_dims:
            raise RuntimeError("No image loaded.")
        full_w, full_h = self.slide_dims
        x, y, w, h = self.clip_roi(x, y, w, h, full_w, full_h)
        if self.reader == "openslide":
            return self._crop_openslide_robust(x, y, w, h, fill=fill)
        if self.reader == "tifffile":
            return self._crop_tifffile(self.path, x, y, w, h)
        if self.reader == "pil":
            arr = self._read_with_pil(self.path)
            return _to_uint8_rgb(arr[y:y+h, x:x+w, :]), {"used": False}
        raise RuntimeError(f"Unknown reader: {self.reader}")

    def crop_raw(self, x: int, y: int, w: int, h: int):
        """Crop without forcing RGB/uint8 conversion.

        This is mainly for IF / OME-TIFF data where channels and original
        intensities must be preserved. Returns (array, axes, info).
        """
        if not self.path or not self.reader or not self.slide_dims:
            raise RuntimeError("No image loaded.")
        full_w, full_h = self.slide_dims
        x, y, w, h = self.clip_roi(x, y, w, h, full_w, full_h)
        if self.reader == "tifffile":
            return self._crop_tifffile_raw(self.path, x, y, w, h)
        # OpenSlide and PIL are visual/RGB readers in this application.
        rgb, info = self.crop(x, y, w, h, fill=255)
        return rgb, "YXS", info

    def read_tile_with_padding(self, x: int, y: int, size: int, padding_color: str = "black") -> np.ndarray:
        if not self.slide_dims:
            raise RuntimeError("No image loaded.")
        full_w, full_h = self.slide_dims
        fill = 0 if padding_color.lower() == "black" else 255
        out = np.full((size, size, 3), fill, dtype=np.uint8)
        crop_w = max(0, min(size, full_w - x))
        crop_h = max(0, min(size, full_h - y))
        if crop_w <= 0 or crop_h <= 0:
            return out
        roi, _ = self.crop(x, y, crop_w, crop_h, fill=fill)
        out[:crop_h, :crop_w, :] = _to_uint8_rgb(roi)
        return out

    def _crop_openslide_robust(self, x0, y0, w, h, block=1024, fill=255, prefer_levels=(2, 3)):
        """
        Robust OpenSlide crop using a cached OpenSlide handle.

        Important behavior:
        - Edge padding is handled outside this function by read_tile_with_padding().
        - This function should NOT silently fill failed internal blocks with black/white.
        - If OpenSlide cannot read a block at level 0 and also cannot recover it from a fallback level,
          it raises an error instead of creating artificial padding in the middle of the tile.
        - For performance, the OpenSlide handle is kept open across repeated crops/tiles.
          If OpenSlide enters a latched error state, the handle is reopened automatically.
        """
        s = self._get_openslide()
        level_count = self._os_level_count
        downsamples = self._os_downsamples
        W0, H0 = self._os_level_dimensions[0]

        if x0 < 0 or y0 < 0 or x0 + w > W0 or y0 + h > H0:
            raise ValueError(f"ROI out of bounds. Slide=({W0},{H0}), ROI=({x0},{y0},{w},{h})")

        fallback_levels = []
        for lv in prefer_levels:
            if 1 <= int(lv) <= level_count - 1 and int(lv) not in fallback_levels:
                fallback_levels.append(int(lv))
        for lv in range(1, level_count):
            if lv not in fallback_levels:
                fallback_levels.append(lv)

        out = np.full((h, w, 3), fill, dtype=np.uint8)
        failed0 = 0
        recovered = 0
        min_lvl_used = None

        try:
            for by in range(0, h, block):
                bh = min(block, h - by)
                for bx in range(0, w, block):
                    bw = min(block, w - bx)
                    sx, sy = x0 + bx, y0 + by

                    try:
                        im0 = s.read_region((sx, sy), 0, (bw, bh)).convert("RGB")
                        out[by:by+bh, bx:bx+bw, :] = np.asarray(im0, dtype=np.uint8)
                        continue
                    except Exception as level0_error:
                        failed0 += 1
                        s = self._reopen_openslide()

                    block_recovered = False
                    last_recovery_error = None

                    for lvl in fallback_levels:
                        ds = float(downsamples[lvl])
                        lw = max(1, int(math.ceil(bw / ds)))
                        lh = max(1, int(math.ceil(bh / ds)))
                        try:
                            iml = s.read_region((sx, sy), lvl, (lw, lh)).convert("RGB")
                            if (lw, lh) != (bw, bh):
                                from PIL import Image
                                iml = iml.resize((bw, bh), resample=Image.BILINEAR)
                            out[by:by+bh, bx:bx+bw, :] = np.asarray(iml, dtype=np.uint8)
                            recovered += 1
                            min_lvl_used = lvl if min_lvl_used is None else min(min_lvl_used, lvl)
                            block_recovered = True
                            break
                        except Exception as recovery_error:
                            last_recovery_error = recovery_error
                            s = self._reopen_openslide()

                    if not block_recovered:
                        raise RuntimeError(
                            "OpenSlide failed to read an internal block and fallback recovery also failed.\n"
                            "To avoid artificial black/white padding inside the image, the tile was not saved.\n\n"
                            f"Slide: {self.path}\n"
                            f"Requested crop: X={x0}, Y={y0}, W={w}, H={h}\n"
                            f"Failed block inside crop: local X={bx}, local Y={by}, W={bw}, H={bh}\n"
                            f"Slide-level block origin: X={sx}, Y={sy}\n\n"
                            f"Level 0 error: {level0_error}\n"
                            f"Last fallback error: {last_recovery_error}"
                        )
        finally:
            # Keep the cached OpenSlide handle open for repeated tile reads.
            pass

        return out, {"used": failed0 > 0, "failed0": failed0, "recovered": recovered, "fallback_level": min_lvl_used}

    def _crop_tifffile(self, path, x, y, w, h):
        arr, axes, fallback_info = self._crop_tifffile_raw(path, x, y, w, h)
        return _array_to_rgb_preview(arr, axes), fallback_info

    def _crop_tifffile_raw(self, path, x, y, w, h):
        fallback_info = {"used": False}
        za, series, axes, zarr_error = self._get_tiff_zarr()
        axes = axes or getattr(series, "axes", "") or ""
        if za is not None:
            slicer = self._build_spatial_slicer_raw(za.ndim, axes, x, y, w, h)
            arr = np.asarray(za[tuple(slicer)])
            return arr, axes if len(axes) == arr.ndim else _guess_axes_for_array(arr, axes), fallback_info

        # Fallback: read the full series only when tiled/zarr access is unavailable.
        # This is less efficient for very large files, so the reason is returned.
        fallback_info = {
            "used": True,
            "reason": f"zarr crop failed, used full read fallback: {zarr_error}"
        }
        arr = series.asarray()
        slicer = self._build_spatial_slicer_raw(arr.ndim, axes, x, y, w, h)
        arr = np.asarray(arr[tuple(slicer)])
        return arr, axes if len(axes) == arr.ndim else _guess_axes_for_array(arr, axes), fallback_info

    def _build_spatial_slicer_raw(self, ndim, axes, x, y, w, h):
        slicer = []
        if axes and len(axes) == ndim and "Y" in axes and "X" in axes:
            for ax in axes:
                if ax == "Y":
                    slicer.append(slice(y, y + h))
                elif ax == "X":
                    slicer.append(slice(x, x + w))
                else:
                    # Preserve C, Z, T and any other non-spatial dimensions.
                    slicer.append(slice(None))
        else:
            # Conservative fallback: assume the last two axes are Y/X.
            for i in range(ndim):
                if i == ndim - 2:
                    slicer.append(slice(y, y + h))
                elif i == ndim - 1:
                    slicer.append(slice(x, x + w))
                else:
                    slicer.append(slice(None))
        return slicer

    def _build_spatial_slicer(self, ndim, axes, x, y, w, h):
        # Kept for backwards compatibility with older code paths.
        return self._build_spatial_slicer_raw(ndim, axes, x, y, w, h)

    def input_thumbnail(self, max_side=512):
        if not self.path or not self.reader:
            raise RuntimeError("No image loaded.")

        if self.reader == "openslide":
            openslide = _try_import_openslide()
            if openslide is None:
                raise RuntimeError("OpenSlide not available.")
            slide = openslide.OpenSlide(self.path)
            try:
                try:
                    img = slide.get_thumbnail((max_side, max_side)).convert("RGB")
                    return _to_uint8_rgb(np.asarray(img, dtype=np.uint8))
                except Exception:
                    lvl = slide.level_count - 1
                    w, h = slide.level_dimensions[lvl]
                    img = slide.read_region((0, 0), lvl, (int(w), int(h))).convert("RGB")
                    return _downsample_for_preview(_to_uint8_rgb(np.asarray(img)), max_side=max_side)
            finally:
                try:
                    slide.close()
                except Exception:
                    pass

        if self.reader == "pil":
            return _downsample_for_preview(_to_uint8_rgb(self._read_with_pil(self.path)), max_side=max_side)

        if self.reader == "tifffile":
            with tifffile.TiffFile(self.path) as tif:
                s0 = tif.series[0]
                try:
                    if hasattr(s0, "levels") and s0.levels:
                        arr = s0.levels[-1].asarray()
                    else:
                        arr = s0.asarray()
                except Exception:
                    arr = tif.pages[0].asarray()
            return _downsample_for_preview(_to_uint8_rgb(arr), max_side=max_side)

        raise RuntimeError(f"Unknown reader: {self.reader}")


# ============================================================
# Save helper
# ============================================================

def save_rgb_image(output_path, rgb, output_format="tiff", write_ome=False, lossless=True,
                   source_resolution=None, source_mpp=None, image_name=None, annotation_kv=None,
                   pixel_scale=1.0):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _to_uint8_rgb(rgb)

    # Apply calibration scaling after downsampling.
    # Example: original 0.25 µm/px, downsample=2 -> output 0.50 µm/px.
    source_resolution, source_mpp = _scale_resolution_and_mpp(
        source_resolution=source_resolution,
        source_mpp=source_mpp,
        pixel_scale=pixel_scale
    )

    resolution = None
    resolutionunit = None
    if source_resolution is not None:
        xres, yres, unit = source_resolution
        if xres and yres and unit:
            resolution = (float(xres), float(yres))
            resolutionunit = unit

    mpp_x_um = None
    mpp_y_um = None
    if source_mpp:
        try:
            mpp_x_um, mpp_y_um = float(source_mpp[0]), float(source_mpp[1])
        except Exception:
            mpp_x_um = None
            mpp_y_um = None

    if output_format == "jpeg":
        from PIL import Image
        save_kwargs = {"quality": 95}
        # JPEG can store DPI. It cannot store full OME physical-size metadata,
        # but adding DPI is still better than dropping calibration completely.
        if resolution is not None and resolutionunit == "INCH":
            save_kwargs["dpi"] = (float(resolution[0]), float(resolution[1]))
        Image.fromarray(rgb).save(str(output_path), **save_kwargs)
        return

    compression_kwargs = {"compression": "deflate", "predictor": True} if lossless else {}

    if write_ome:
        ome_xml = _build_ome_xml_rgb(
            size_x=int(rgb.shape[1]),
            size_y=int(rgb.shape[0]),
            physical_size_x_um=mpp_x_um,
            physical_size_y_um=mpp_y_um,
            image_name=image_name or output_path.stem,
            annotation_kv=annotation_kv,
        )
        tifffile.imwrite(
            str(output_path), rgb, bigtiff=True, tile=(256, 256), photometric="rgb",
            description=_ascii_safe(ome_xml), software=f"{APP_NAME} v{APP_VERSION}",
            resolution=resolution, resolutionunit=resolutionunit, **compression_kwargs
        )
    else:
        tifffile.imwrite(
            str(output_path), rgb, bigtiff=True, tile=(256, 256), photometric="rgb",
            description=_ascii_safe(f"Generated by {APP_NAME} v{APP_VERSION}"),
            software=f"{APP_NAME} v{APP_VERSION}", resolution=resolution,
            resolutionunit=resolutionunit, **compression_kwargs
        )


def save_multichannel_image(output_path, arr, axes=None, write_ome=True, lossless=True,
                            source_resolution=None, source_mpp=None, image_name=None,
                            pixel_scale=1.0):
    """Save scientific crop data without RGB conversion or intensity normalization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(np.asarray(arr))
    axes = _guess_axes_for_array(arr, axes)

    source_resolution, source_mpp = _scale_resolution_and_mpp(
        source_resolution=source_resolution,
        source_mpp=source_mpp,
        pixel_scale=pixel_scale
    )

    resolution = None
    resolutionunit = None
    if source_resolution is not None:
        try:
            xres, yres, unit = source_resolution
            if xres and yres and unit:
                resolution = (float(xres), float(yres))
                resolutionunit = unit
        except Exception:
            resolution = None
            resolutionunit = None

    metadata = {}
    if axes:
        metadata["axes"] = axes
    if image_name:
        metadata["Name"] = str(image_name)
    if source_mpp:
        try:
            metadata["PhysicalSizeX"] = float(source_mpp[0])
            metadata["PhysicalSizeY"] = float(source_mpp[1])
            metadata["PhysicalSizeXUnit"] = "µm"
            metadata["PhysicalSizeYUnit"] = "µm"
        except Exception:
            pass

    compression_kwargs = {"compression": "deflate", "predictor": True} if lossless else {}

    tifffile.imwrite(
        str(output_path),
        arr,
        bigtiff=True,
        ome=bool(write_ome),
        metadata=metadata if metadata else None,
        photometric="minisblack",
        software=f"{APP_NAME} v{APP_VERSION}",
        resolution=resolution,
        resolutionunit=resolutionunit,
        **compression_kwargs
    )

# ============================================================
# Manual merge grid dialog
# ============================================================

class ManualGridDialog(QDialog):
    def __init__(self, tile_paths, read_tile_func, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Tile Layout")
        self.resize(980, 680)
        self.tile_paths = list(tile_paths)
        self.read_tile_func = read_tile_func
        self.mapping = {}

        layout = QVBoxLayout(self)
        top = QHBoxLayout()

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 200)
        self.rows_spin.setValue(2)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 200)
        self.cols_spin.setValue(2)

        top.addWidget(QLabel("Rows:"))
        top.addWidget(self.rows_spin)
        top.addWidget(QLabel("Columns:"))
        top.addWidget(self.cols_spin)

        build_btn = QPushButton("Build / Reset Grid")
        build_btn.clicked.connect(self.build_grid)
        top.addWidget(build_btn)

        top.addWidget(QLabel("Tile:"))
        self.tile_combo = QComboBox()
        self.tile_combo.addItems([p.name for p in self.tile_paths])
        top.addWidget(self.tile_combo, 1)

        assign_btn = QPushButton("Assign to selected cell")
        assign_btn.clicked.connect(self.assign_selected_tile)
        top.addWidget(assign_btn)

        auto_btn = QPushButton("Auto-fill by selected order")
        auto_btn.clicked.connect(self.autofill_by_order)
        top.addWidget(auto_btn)

        layout.addLayout(top)

        note = QLabel("Select one cell, choose a tile, and click Assign. Auto-fill fills rows left-to-right using the selected file order.")
        note.setStyleSheet("color: #555; padding: 4px;")
        layout.addWidget(note)

        self.table = QTableWidget()
        layout.addWidget(self.table, 1)

        bottom = QHBoxLayout()
        clear_btn = QPushButton("Clear selected cell")
        clear_btn.clicked.connect(self.clear_selected_cell)
        ok_btn = QPushButton("Use Layout")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        bottom.addWidget(clear_btn)
        bottom.addStretch()
        bottom.addWidget(ok_btn)
        bottom.addWidget(cancel_btn)
        layout.addLayout(bottom)

        self.build_grid()

    def build_grid(self):
        self.mapping = {}
        rows = self.rows_spin.value()
        cols = self.cols_spin.value()
        self.table.setRowCount(rows)
        self.table.setColumnCount(cols)
        self.table.setHorizontalHeaderLabels([str(c + 1) for c in range(cols)])
        self.table.setVerticalHeaderLabels([str(r + 1) for r in range(rows)])
        for r in range(rows):
            self.table.setRowHeight(r, 95)
            for c in range(cols):
                item = QTableWidgetItem("Empty")
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)
        for c in range(cols):
            self.table.setColumnWidth(c, 150)

    def assign_selected_tile(self):
        selected = self.table.selectedIndexes()
        if not selected:
            QMessageBox.warning(self, "No cell selected", "Select a grid cell first.")
            return
        idx = self.tile_combo.currentIndex()
        if idx < 0 or idx >= len(self.tile_paths):
            return
        tile_path = self.tile_paths[idx]
        cell = selected[0]
        r, c = cell.row(), cell.column()
        self.mapping[(r, c)] = tile_path
        item = QTableWidgetItem(tile_path.name)
        item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(r, c, item)

    def clear_selected_cell(self):
        selected = self.table.selectedIndexes()
        if not selected:
            return
        cell = selected[0]
        r, c = cell.row(), cell.column()
        self.mapping.pop((r, c), None)
        item = QTableWidgetItem("Empty")
        item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(r, c, item)

    def autofill_by_order(self):
        self.mapping = {}
        rows = self.rows_spin.value()
        cols = self.cols_spin.value()
        k = 0
        for r in range(rows):
            for c in range(cols):
                if k >= len(self.tile_paths):
                    item = QTableWidgetItem("Empty")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(r, c, item)
                    continue
                tile_path = self.tile_paths[k]
                self.mapping[(r, c)] = tile_path
                item = QTableWidgetItem(tile_path.name)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)
                k += 1

    def get_layout(self):
        return {"rows": self.rows_spin.value(), "cols": self.cols_spin.value(), "mapping": dict(self.mapping)}


# ============================================================
# Main GUI
# ============================================================

class WSICropTileMergeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(resource_path(APP_ICON_PATH)))
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} - WSI Crop / Tile / Merge")
        self.setGeometry(100, 80, 1180, 820)
        self.setStyleSheet("background-color: #f0f0f0;")

        self.backend = ImageBackend()
        self.tile_files = []
        self.bulk_paths = []
        self.manual_layout = None
        self.lif_single_path = None
        self.lif_single_obj = None
        self.lif_single_images = []
        self.lif_bulk_paths = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        title = QLabel(f"{APP_NAME} v{APP_VERSION} - WSI Crop / Tile / Merge Tool")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 12px;")
        root.addWidget(title)

        menu_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Crop", "Tiles", "Merge Tiles", "Split LIF"])
        self.mode_combo.setMaximumWidth(180)

        menu_row.addWidget(QLabel("Mode:"))
        menu_row.addWidget(self.mode_combo)
        menu_row.addStretch()

        help_btn = QPushButton("Help / About")
        help_btn.setToolTip("Show app information, citation, DOI, and usage notes.")
        help_btn.clicked.connect(self.show_help_about)
        menu_row.addWidget(help_btn)

        root.addLayout(menu_row)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #27ae60; padding: 8px;")

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.stack.addWidget(self._build_crop_page())
        self.stack.addWidget(self._build_tiles_page())
        self.stack.addWidget(self._build_merge_page())
        self.stack.addWidget(self._build_lif_page())
        self.mode_combo.currentIndexChanged.connect(lambda: self.stack.setCurrentIndex(self.mode_combo.currentIndex()))

        root.addWidget(self.info_label)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

    def _set_label_pixmap(self, label, rgb):
        pm = _numpy_rgb_to_qpixmap(_to_uint8_rgb(rgb))
        label.setPixmap(pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _set_widgets_visible(self, widgets, visible: bool):
        for w in widgets:
            w.setVisible(visible)
            w.setEnabled(visible)

    def _write_batch_log(self, output_folder: Path, rows):
        """Write a CSV log for batch tiling operations."""
        if not rows:
            return None
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = output_folder / f"TiffCropper_batch_log_{stamp}.csv"
        fieldnames = [
            "timestamp", "operation", "status", "image", "reader",
            "output_folder", "tiles_expected", "tiles_written", "message"
        ]
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        return log_path

    def _draw_fixed_grid_on_thumb(self, thumb_rgb, full_w, full_h, tile_size, overlap):
        pm = _numpy_rgb_to_qpixmap(thumb_rgb)
        painter = QPainter(pm)
        pen_grid = QPen(QColor(220, 0, 0), 1)
        pen_pad = QPen(QColor(230, 190, 0), 2)
        sx = pm.width() / float(full_w)
        sy = pm.height() / float(full_h)
        xs, ys, stride, _ = _compute_tile_grid(full_w, full_h, tile_size, overlap)

        painter.setPen(pen_grid)
        for y in ys:
            for x in xs:
                painter.drawRect(int(x * sx), int(y * sy), int(tile_size * sx), int(tile_size * sy))

        if xs and ys:
            painter.setPen(pen_pad)
            painter.drawRect(0, 0, int((xs[-1] + tile_size) * sx), int((ys[-1] + tile_size) * sy))

        painter.end()
        return pm

    def _draw_division_grid_on_thumb(self, thumb_rgb, full_w, full_h, rows, cols):
        pm = _numpy_rgb_to_qpixmap(thumb_rgb)
        painter = QPainter(pm)
        painter.setPen(QPen(QColor(220, 0, 0), 1))
        sx = pm.width() / float(full_w)
        sy = pm.height() / float(full_h)
        for r in range(1, rows):
            y = int(round((r * full_h / rows) * sy))
            painter.drawLine(0, y, pm.width(), y)
        for c in range(1, cols):
            x = int(round((c * full_w / cols) * sx))
            painter.drawLine(x, 0, x, pm.height())
        painter.end()
        return pm

    def _mk_spin(self, label, mn, mx, val, parent_layout):
        box = QVBoxLayout()
        box.addWidget(QLabel(label))
        sp = QSpinBox()
        sp.setRange(mn, mx)
        sp.setValue(val)
        box.addWidget(sp)
        parent_layout.addLayout(box)
        return sp

    def _help_about_html(self):
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; font-size: 10pt;">
          <h2>{APP_NAME} v{APP_VERSION}</h2>
          <p><b>{APP_TITLE}</b></p>

          <p>
            TiffCropper is a standalone Windows application for cropping, tiling,
            and reconstructing large digital pathology and microscopy images.
          </p>

          <h3>Main features</h3>
          <ul>
            <li>ROI cropping from large WSI and microscopy images.</li>
            <li>Full-area selection for whole-slide export or downsampling.</li>
            <li>Fixed-size square tile generation with optional overlap.</li>
            <li>Row/column-based image division for structured tiling.</li>
            <li>Bulk tiling of multiple images using the same parameters.</li>
            <li>Automatic tile merging from encoded tile names.</li>
            <li>Manual grid-based tile merging when filenames do not encode position.</li>
            <li>OME-TIFF export with physical pixel size preservation when available.</li>
            <li>Leica LIF splitting into one OME-TIFF per internal scene/page, preserving IF channels.</li>
            <li>BigTIFF output and optional lossless DEFLATE compression.</li>
          </ul>

          <h3>Supported input formats</h3>
          <p>{supported}</p>

          <h3>Important tiling note</h3>
          <p>
            In fixed square tile mode, padding is applied only to true border tiles.
            Internal tiles are extracted as direct crops to avoid introducing artificial
            black or white padding inside the image.
          </p>

          <h3>Merge note</h3>
          <p>
            Tile merging is geometric. It does not perform image registration or
            intelligent stitching. For best results, merge tiles generated from the
            same source image using the same tile size, overlap, and downsample settings.
          </p>

          <h3>Performance and logs</h3>
          <p>
            For repeated WSI tiling, TiffCropper keeps OpenSlide and TIFF/zarr readers
            open during each image job to reduce repeated file-opening overhead. Batch
            tiling also writes a CSV log with status, expected tile counts, written tile
            counts, output folders, and error messages when failures occur.
          </p>

          <h3>Citation</h3>
          <p>{APP_CITATION}</p>

          <p>
            <b>DOI:</b> <a href="https://doi.org/{APP_DOI}">https://doi.org/{APP_DOI}</a><br>
            <b>GitHub:</b> <a href="{APP_GITHUB}">{APP_GITHUB}</a><br>
            <b>License:</b> {APP_LICENSE}<br>
            <b>Author:</b> {APP_AUTHOR}
          </p>
        </body>
        </html>
        """

    def show_help_about(self):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"About {APP_NAME}")
        dlg.resize(720, 620)

        layout = QVBoxLayout(dlg)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(self._help_about_html())
        layout.addWidget(browser)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)

        dlg.exec_()

    # ========================================================
    # LIF splitter page
    # ========================================================

    def _build_lif_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        file_box = QGroupBox("Leica LIF input")
        file_layout = QVBoxLayout(file_box)

        row1 = QHBoxLayout()
        one_btn = QPushButton("Load one LIF + previews")
        one_btn.clicked.connect(self.load_lif_single_file)
        bulk_btn = QPushButton("Bulk select LIF files")
        bulk_btn.clicked.connect(self.load_lif_bulk_files)
        folder_btn = QPushButton("Bulk folder")
        folder_btn.clicked.connect(self.load_lif_bulk_folder)
        self.lif_file_label = QLabel("No LIF file selected")
        self.lif_file_label.setStyleSheet("padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px;")
        row1.addWidget(one_btn)
        row1.addWidget(bulk_btn)
        row1.addWidget(folder_btn)
        row1.addWidget(self.lif_file_label, 1)
        file_layout.addLayout(row1)

        row2 = QHBoxLayout()
        out_btn = QPushButton("Output folder optional")
        out_btn.clicked.connect(self.browse_lif_output_folder)
        clear_out_btn = QPushButton("Clear")
        clear_out_btn.clicked.connect(lambda: self.lif_output_edit.setText(""))
        self.lif_output_edit = QLineEdit("")
        self.lif_output_edit.setPlaceholderText("Leave empty to save beside each .lif file")
        row2.addWidget(out_btn)
        row2.addWidget(self.lif_output_edit, 1)
        row2.addWidget(clear_out_btn)
        file_layout.addLayout(row2)

        layout.addWidget(file_box)

        opt = QGroupBox("Export options")
        opt_layout = QHBoxLayout(opt)
        self.lif_skip_existing_chk = QCheckBox("Skip existing")
        self.lif_skip_existing_chk.setChecked(True)
        self.lif_overwrite_chk = QCheckBox("Overwrite")
        self.lif_xml_chk = QCheckBox("Save Leica XML header")
        self.lif_xml_chk.setChecked(True)
        self.lif_json_chk = QCheckBox("Save scene metadata JSON")
        self.lif_json_chk.setChecked(True)
        self.lif_recursive_chk = QCheckBox("Recursive folder search")
        self.lif_stop_on_error_chk = QCheckBox("Stop on error")
        self.lif_compression_combo = QComboBox()
        self.lif_compression_combo.addItems(["None", "deflate"])
        opt_layout.addWidget(self.lif_skip_existing_chk)
        opt_layout.addWidget(self.lif_overwrite_chk)
        opt_layout.addWidget(self.lif_xml_chk)
        opt_layout.addWidget(self.lif_json_chk)
        opt_layout.addWidget(self.lif_recursive_chk)
        opt_layout.addWidget(self.lif_stop_on_error_chk)
        opt_layout.addWidget(QLabel("Compression:"))
        opt_layout.addWidget(self.lif_compression_combo)
        opt_layout.addStretch()
        layout.addWidget(opt)

        note = QLabel(
            "Single-file mode loads false-color scene thumbnails so you can select which pages to save. "
            "Bulk mode exports all scenes from each selected .lif file without preview. "
            "Export is one OME-TIFF per LIF scene/page, preserving channels as raw IF data."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555; padding: 4px;")
        layout.addWidget(note)

        self.lif_scene_table = QTableWidget()
        self.lif_scene_table.setColumnCount(7)
        self.lif_scene_table.setHorizontalHeaderLabels([
            "Save", "Scene", "Name", "Size X×Y", "C/Z/T/M", "Dtype / mode / bit depth", "Thumbnail"
        ])
        self.lif_scene_table.setColumnWidth(0, 60)
        self.lif_scene_table.setColumnWidth(1, 70)
        self.lif_scene_table.setColumnWidth(2, 230)
        self.lif_scene_table.setColumnWidth(3, 130)
        self.lif_scene_table.setColumnWidth(4, 130)
        self.lif_scene_table.setColumnWidth(5, 210)
        self.lif_scene_table.setColumnWidth(6, 190)
        layout.addWidget(self.lif_scene_table, 1)

        btn_row = QHBoxLayout()
        select_all_btn = QPushButton("Select all")
        select_all_btn.clicked.connect(lambda: self.set_lif_scene_checks(True))
        clear_btn = QPushButton("Clear selection")
        clear_btn.clicked.connect(lambda: self.set_lif_scene_checks(False))
        save_selected_btn = QPushButton("SAVE SELECTED SCENES")
        save_selected_btn.clicked.connect(self.save_lif_selected_scenes)
        save_all_single_btn = QPushButton("Save all scenes from loaded LIF")
        save_all_single_btn.clicked.connect(self.save_lif_all_single_scenes)
        bulk_save_btn = QPushButton("RUN BULK EXPORT")
        bulk_save_btn.clicked.connect(self.run_lif_bulk_export)
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(save_selected_btn)
        btn_row.addWidget(save_all_single_btn)
        btn_row.addWidget(bulk_save_btn)
        layout.addLayout(btn_row)

        return page

    def _close_lif_single(self):
        try:
            if self.lif_single_obj is not None:
                self.lif_single_obj.close()
        except Exception:
            pass
        self.lif_single_obj = None
        self.lif_single_images = []
        self.lif_single_path = None

    def browse_lif_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.lif_output_edit.setText(folder)

    def set_lif_scene_checks(self, checked: bool):
        for r in range(self.lif_scene_table.rowCount()):
            widget = self.lif_scene_table.cellWidget(r, 0)
            if isinstance(widget, QCheckBox):
                widget.setChecked(checked)

    def selected_lif_scene_indices(self):
        indices = []
        for r in range(self.lif_scene_table.rowCount()):
            widget = self.lif_scene_table.cellWidget(r, 0)
            if isinstance(widget, QCheckBox) and widget.isChecked():
                item = self.lif_scene_table.item(r, 1)
                if item is not None:
                    try:
                        indices.append(int(item.text()))
                    except Exception:
                        pass
        return indices

    def load_lif_single_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Leica LIF file", "", _lif_file_filter())
        if not path:
            return
        self._load_lif_single_path(Path(path))

    def _load_lif_single_path(self, lif_path: Path):
        try:
            LifFile = _require_readlif()
            self._close_lif_single()
            self.lif_scene_table.setRowCount(0)
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            QApplication.processEvents()

            lif = LifFile(str(lif_path))
            images = list(lif.get_iter_image())
            self.lif_single_obj = lif
            self.lif_single_path = lif_path
            self.lif_single_images = images
            self.lif_file_label.setText(f"Single LIF loaded: {lif_path.name} | {len(images)} scene(s)")

            self.progress.setRange(0, max(1, len(images)))
            self.progress.setValue(0)
            self.lif_scene_table.setRowCount(len(images))

            for row, img in enumerate(images):
                scene_name = str(getattr(img, "name", f"scene_{row}"))
                size_x, size_y, size_z, size_t, size_m, size_c = _lif_get_image_dims(img)
                bit_depth = getattr(img, "bit_depth", "")
                bit_depth_text = _lif_bit_depth_text(bit_depth)
                dtype_text = _lif_infer_storage_dtype_text(img)

                chk = QCheckBox()
                chk.setChecked(True)
                chk.setStyleSheet("margin-left: 16px;")
                self.lif_scene_table.setCellWidget(row, 0, chk)
                self.lif_scene_table.setItem(row, 1, QTableWidgetItem(str(row)))
                self.lif_scene_table.setItem(row, 2, QTableWidgetItem(scene_name))
                self.lif_scene_table.setItem(row, 3, QTableWidgetItem(f"{size_x} × {size_y}"))
                self.lif_scene_table.setItem(row, 4, QTableWidgetItem(f"C={size_c}, Z={size_z}, T={size_t}, M={size_m}"))

                thumb_label = QLabel("preview failed")
                thumb_label.setAlignment(Qt.AlignCenter)
                thumb_label.setFixedSize(180, 120)
                thumb_label.setStyleSheet("background: white; border: 1px solid #bdc3c7;")
                try:
                    thumb_rgb, dtype_text = _lif_thumbnail_rgb(img, max_side=170, max_channels=4)
                    pm = _numpy_rgb_to_qpixmap(thumb_rgb)
                    thumb_label.setPixmap(pm.scaled(thumb_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                except Exception as thumb_error:
                    thumb_label.setText(f"No preview\n{thumb_error}")

                self.lif_scene_table.setItem(row, 5, QTableWidgetItem(f"{dtype_text} / {bit_depth_text}"))
                self.lif_scene_table.setCellWidget(row, 6, thumb_label)
                self.lif_scene_table.setRowHeight(row, 128)

                self.progress.setValue(row + 1)
                self.info_label.setText(f"Loaded LIF preview: {lif_path.name} | scene {row + 1}/{len(images)}")
                QApplication.processEvents()

            self.progress.setVisible(False)
            self.info_label.setText(f"Ready: select scenes to export from {lif_path.name}")
        except Exception as e:
            self.progress.setVisible(False)
            QMessageBox.critical(self, "LIF Load Error", str(e))

    def load_lif_bulk_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Leica LIF files", "", _lif_file_filter())
        if not paths:
            return
        self.lif_bulk_paths = [Path(p) for p in paths]
        self.lif_file_label.setText(f"Bulk LIF selection: {len(self.lif_bulk_paths)} file(s). Preview not loaded.")
        self.info_label.setText("Bulk LIF mode ready. Click RUN BULK EXPORT to save all scenes from each file.")

    def load_lif_bulk_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder containing LIF files")
        if not folder:
            return
        folder = Path(folder)
        pattern = "**/*.lif" if self.lif_recursive_chk.isChecked() else "*.lif"
        self.lif_bulk_paths = sorted(folder.glob(pattern))
        self.lif_file_label.setText(f"Bulk folder: {folder} | {len(self.lif_bulk_paths)} .lif file(s)")
        self.info_label.setText("Bulk LIF folder loaded. Preview not loaded for bulk mode.")

    def _lif_output_base(self):
        text = self.lif_output_edit.text().strip()
        return text if text else None

    def _lif_compression(self):
        text = self.lif_compression_combo.currentText().strip()
        return None if text.lower() == "none" else text

    def save_lif_selected_scenes(self):
        if self.lif_single_path is None or self.lif_single_obj is None:
            QMessageBox.warning(self, "No LIF loaded", "Load one LIF file first.")
            return
        selected = self.selected_lif_scene_indices()
        if not selected:
            QMessageBox.warning(self, "No scenes selected", "Select at least one scene/page to save.")
            return
        try:
            manifest = self._export_lif_file(
                lif_path=self.lif_single_path,
                scene_indices=selected,
                lif_obj=self.lif_single_obj,
                images=self.lif_single_images,
            )
            QMessageBox.information(self, "LIF export complete", f"Export finished.\n\nManifest:\n{manifest}")
        except Exception as e:
            self.progress.setVisible(False)
            QMessageBox.critical(self, "LIF Export Error", str(e))

    def save_lif_all_single_scenes(self):
        if self.lif_single_path is None or self.lif_single_obj is None:
            QMessageBox.warning(self, "No LIF loaded", "Load one LIF file first.")
            return
        try:
            manifest = self._export_lif_file(
                lif_path=self.lif_single_path,
                scene_indices=None,
                lif_obj=self.lif_single_obj,
                images=self.lif_single_images,
            )
            QMessageBox.information(self, "LIF export complete", f"Export finished.\n\nManifest:\n{manifest}")
        except Exception as e:
            self.progress.setVisible(False)
            QMessageBox.critical(self, "LIF Export Error", str(e))

    def run_lif_bulk_export(self):
        if not self.lif_bulk_paths:
            QMessageBox.warning(self, "No bulk LIF files", "Select LIF files or a folder first.")
            return
        manifests = []
        failures = []
        self.progress.setVisible(True)
        self.progress.setRange(0, len(self.lif_bulk_paths))
        self.progress.setValue(0)
        QApplication.processEvents()

        for i, lif_path in enumerate(self.lif_bulk_paths, start=1):
            try:
                self.info_label.setText(f"Bulk LIF export: {lif_path.name} ({i}/{len(self.lif_bulk_paths)})")
                QApplication.processEvents()
                manifests.append(self._export_lif_file(lif_path=lif_path, scene_indices=None))
            except Exception as e:
                failures.append((lif_path, e))
                if self.lif_stop_on_error_chk.isChecked():
                    break
            self.progress.setRange(0, len(self.lif_bulk_paths))
            self.progress.setValue(i)
            QApplication.processEvents()

        self.progress.setVisible(False)
        msg = f"Bulk export finished.\n\nManifest files: {len(manifests)}"
        if manifests:
            msg += "\n" + "\n".join(str(p) for p in manifests[:10])
            if len(manifests) > 10:
                msg += f"\n... and {len(manifests) - 10} more"
        if failures:
            msg += f"\n\nFailures: {len(failures)}"
            for p, e in failures[:5]:
                msg += f"\n{p.name}: {e}"
        QMessageBox.information(self, "Bulk LIF export", msg)
        self.info_label.setText(msg.replace("\n", " | "))

    def _export_lif_file(self, lif_path: Path, scene_indices=None, lif_obj=None, images=None):
        LifFile = _require_readlif()
        own_lif = False
        if lif_obj is None or images is None:
            lif_obj = LifFile(str(lif_path))
            images = list(lif_obj.get_iter_image())
            own_lif = True

        out_dir = _lif_output_folder_for(lif_path, self._lif_output_base())
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / f"{_lif_safe_name(lif_path.stem)}_manifest.csv"
        rows = []

        try:
            xml_path = _lif_write_xml_header(lif_obj, lif_path, out_dir) if self.lif_xml_chk.isChecked() else None
            selected = set(scene_indices) if scene_indices is not None else set(range(len(images)))

            for scene_index, img in enumerate(images):
                if scene_index not in selected:
                    continue

                scene_name = str(getattr(img, "name", f"scene_{scene_index}"))
                safe_scene = _lif_safe_name(scene_name)
                out_path = out_dir / f"scene_{scene_index:03d}_{safe_scene}.ome.tif"
                base_meta = _lif_scene_metadata_dict(img, scene_index)
                json_path = _lif_write_scene_json(base_meta, out_dir, scene_index, scene_name) if self.lif_json_chk.isChecked() else None

                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "lif_file": str(lif_path),
                    "scene_index": scene_index,
                    "scene_name": scene_name,
                    "readlif_path": getattr(img, "path", ""),
                    "size_x": base_meta.get("size_x"),
                    "size_y": base_meta.get("size_y"),
                    "size_z": base_meta.get("size_z"),
                    "size_t": base_meta.get("size_t"),
                    "size_m": base_meta.get("size_m"),
                    "size_c": base_meta.get("size_c"),
                    "bit_depth": base_meta.get("bit_depth"),
                    "scale_px_per_um": base_meta.get("scale_px_per_um"),
                    "PhysicalSizeX_um_per_px": base_meta.get("PhysicalSizeX"),
                    "PhysicalSizeY_um_per_px": base_meta.get("PhysicalSizeY"),
                    "PhysicalSizeZ_um_per_px": base_meta.get("PhysicalSizeZ"),
                    "xml_header_path": str(xml_path) if xml_path else "",
                    "scene_metadata_json": str(json_path) if json_path else "",
                    "output_path": str(out_path),
                    "status": "pending",
                    "error": "",
                }

                def progress_cb(done, total, scene_index=scene_index, scene_name=scene_name):
                    self.progress.setVisible(True)
                    self.progress.setRange(0, max(1, total))
                    self.progress.setValue(done)
                    self.info_label.setText(
                        f"Writing LIF scene {scene_index}: {scene_name} | plane {done}/{total}"
                    )
                    QApplication.processEvents()

                try:
                    result = _lif_save_scene_ome_tiff_lowmem(
                        img=img,
                        scene_index=scene_index,
                        out_path=out_path,
                        overwrite=self.lif_overwrite_chk.isChecked(),
                        skip_existing=self.lif_skip_existing_chk.isChecked(),
                        compression=self._lif_compression(),
                        progress_callback=progress_cb,
                    )
                    row.update(result)
                    row["status"] = "skipped_existing" if result.get("skipped_existing") else "success"
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = f"{exc}\n{traceback.format_exc()}"
                    if self.lif_stop_on_error_chk.isChecked():
                        rows.append(row)
                        raise

                rows.append(row)
                _lif_write_manifest(manifest_path, rows)
                QApplication.processEvents()

        finally:
            if own_lif:
                try:
                    lif_obj.close()
                except Exception:
                    pass

        _lif_write_manifest(manifest_path, rows)
        self.progress.setVisible(False)
        self.info_label.setText(f"LIF export manifest saved: {manifest_path}")
        return manifest_path

    # ========================================================
    # Crop page
    # ========================================================

    def _build_crop_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        file_layout = QHBoxLayout()
        self.crop_file_label = QLabel("No file selected")
        self.crop_file_label.setStyleSheet("padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px;")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_crop_file)
        file_layout.addWidget(browse_btn)
        file_layout.addWidget(self.crop_file_label, 1)
        layout.addLayout(file_layout)

        coords_box = QGroupBox("Crop by exact pixels")
        coords_layout = QHBoxLayout(coords_box)
        self.x_spin = self._mk_spin("X:", 0, 10_000_000, 0, coords_layout)
        self.y_spin = self._mk_spin("Y:", 0, 10_000_000, 0, coords_layout)
        self.w_spin = self._mk_spin("Width:", 1, 10_000_000, 1000, coords_layout)
        self.h_spin = self._mk_spin("Height:", 1, 10_000_000, 1000, coords_layout)
        self.full_area_btn = QPushButton("Select full area")
        self.full_area_btn.clicked.connect(self.select_full_crop_area)
        update_rect_btn = QPushButton("Update rectangle from pixels")
        update_rect_btn.clicked.connect(self.update_crop_rectangle_from_spinboxes)
        coords_layout.addWidget(self.full_area_btn)
        coords_layout.addWidget(update_rect_btn)
        layout.addWidget(coords_box)

        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Output:"))
        self.crop_out_combo = QComboBox()
        self.crop_out_combo.addItems(["TIFF (.tif)", "OME-TIFF (.ome.tif)", "JPEG (.jpg)"])
        # OME-TIFF is the safest default for microscopy/IF multichannel crops.
        self.crop_out_combo.setCurrentIndex(1)
        self.crop_out_combo.setMaximumWidth(220)
        out_layout.addWidget(self.crop_out_combo)
        out_layout.addWidget(QLabel("Suffix after _crop_:"))
        self.crop_suffix_edit = QLineEdit("final")
        self.crop_suffix_edit.setMaximumWidth(180)
        out_layout.addWidget(self.crop_suffix_edit)
        out_layout.addWidget(QLabel("Downsample:"))
        self.crop_downsample_spin = QDoubleSpinBox()
        self.crop_downsample_spin.setRange(0.01, 1000.0)
        self.crop_downsample_spin.setDecimals(2)
        self.crop_downsample_spin.setValue(1.0)
        self.crop_downsample_spin.setMaximumWidth(100)
        out_layout.addWidget(self.crop_downsample_spin)
        out_layout.addStretch()
        layout.addLayout(out_layout)

        opt = QGroupBox("Options")
        opt_layout = QHBoxLayout(opt)
        self.crop_lossless_chk = QCheckBox("Lossless compression DEFLATE")
        self.crop_lossless_chk.setChecked(True)
        self.crop_preserve_channels_chk = QCheckBox("Preserve raw multichannel data for TIFF/OME-TIFF")
        self.crop_preserve_channels_chk.setChecked(True)
        self.crop_preview_chk = QCheckBox("Preview panel")
        self.crop_preview_chk.setChecked(True)
        opt_layout.addWidget(self.crop_lossless_chk)
        opt_layout.addWidget(self.crop_preserve_channels_chk)
        opt_layout.addWidget(self.crop_preview_chk)
        opt_layout.addStretch()
        layout.addWidget(opt)

        prev = QGroupBox("Preview - drag a rectangle on the input thumbnail")
        prev_layout = QHBoxLayout(prev)
        self.crop_thumb_in = CropSelectionLabel()
        self.crop_thumb_in.setFixedSize(540, 330)
        self.crop_thumb_out = QLabel("Crop preview")
        self.crop_thumb_out.setAlignment(Qt.AlignCenter)
        self.crop_thumb_out.setFixedSize(540, 330)
        self.crop_thumb_out.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")
        prev_layout.addWidget(self.crop_thumb_in)
        prev_layout.addWidget(self.crop_thumb_out)
        layout.addWidget(prev)

        btn_row = QHBoxLayout()
        preview_btn = QPushButton("Preview Crop")
        preview_btn.clicked.connect(self.preview_crop)
        crop_btn = QPushButton("CROP & SAVE")
        crop_btn.clicked.connect(self.crop_image)
        btn_row.addWidget(preview_btn)
        btn_row.addWidget(crop_btn)
        layout.addLayout(btn_row)

        return page

    def _on_rectangle_selected(self, x, y, w, h):
        self.x_spin.setValue(x)
        self.y_spin.setValue(y)
        self.w_spin.setValue(w)
        self.h_spin.setValue(h)
        self.info_label.setText(f"Rectangle selected: X={x}, Y={y}, W={w}, H={h}")

    def update_crop_rectangle_from_spinboxes(self):
        if not self.backend.path or not self.backend.slide_dims:
            QMessageBox.warning(self, "Error", "Please select a file first.")
            return
        self.crop_thumb_in.set_selection_from_full_coords(
            self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value()
        )
        self.info_label.setText(
            f"Rectangle updated from pixels: X={self.x_spin.value()}, Y={self.y_spin.value()}, "
            f"W={self.w_spin.value()}, H={self.h_spin.value()}"
        )

    def browse_crop_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select WSI", "", _image_file_filter())
        if not file_path:
            return
        try:
            self.backend = ImageBackend().load(file_path)
            w, h = self.backend.slide_dims
            self.crop_file_label.setText(os.path.basename(file_path))
            self.x_spin.setMaximum(max(0, w - 1))
            self.y_spin.setMaximum(max(0, h - 1))
            self.w_spin.setMaximum(max(1, w))
            self.h_spin.setMaximum(max(1, h))
            print("Loaded with:", self.backend.reader, self.backend.file_kind, self.backend.path_obj.suffix)
            self.info_label.setText(f"Loaded: {Path(file_path).name} | Size: {w} x {h} px | Reader: {self.backend.reader}")
            if self.crop_preview_chk.isChecked():
                thumb = self.backend.input_thumbnail(max_side=900)
                self.crop_thumb_in.set_image(thumb, full_w=w, full_h=h, callback=self._on_rectangle_selected)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def select_full_crop_area(self):
        if not self.backend.path or not self.backend.slide_dims:
            QMessageBox.warning(self, "Error", "Please select a file first.")
            return
        full_w, full_h = self.backend.slide_dims
        self.x_spin.setValue(0)
        self.y_spin.setValue(0)
        self.w_spin.setValue(full_w)
        self.h_spin.setValue(full_h)
        self.crop_thumb_in.set_selection_from_full_coords(0, 0, full_w, full_h)
        self.info_label.setText(f"Full area selected: 0, 0, {full_w}, {full_h}")

    def preview_crop(self):
        if not self.backend.path:
            QMessageBox.warning(self, "Error", "Please select a file first.")
            return
        try:
            preserve_raw = (
                hasattr(self, "crop_preserve_channels_chk")
                and self.crop_preserve_channels_chk.isChecked()
                and self.backend.reader == "tifffile"
            )
            if preserve_raw:
                raw, axes, _ = self.backend.crop_raw(
                    self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value()
                )
                downsample = float(self.crop_downsample_spin.value())
                raw_preview, axes_preview = _resize_spatial_array(raw, axes, downsample) if downsample != 1.0 else (raw, axes)
                roi_preview = _array_to_rgb_preview(raw_preview, axes_preview)
                shape_text = f"raw shape={tuple(raw.shape)} | axes={axes or 'unknown'} | dtype={raw.dtype}"
            else:
                roi_preview, _ = self.backend.crop(
                    self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value(), fill=255
                )
                downsample = float(self.crop_downsample_spin.value())
                if downsample != 1.0:
                    from PIL import Image
                    new_w = max(1, int(round(roi_preview.shape[1] / downsample)))
                    new_h = max(1, int(round(roi_preview.shape[0] / downsample)))
                    roi_preview = np.asarray(Image.fromarray(roi_preview).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)
                shape_text = f"preview={roi_preview.shape[1]} x {roi_preview.shape[0]} px"

            if self.crop_preview_chk.isChecked():
                if self.backend.slide_dims:
                    full_w, full_h = self.backend.slide_dims
                    thumb = self.backend.input_thumbnail(max_side=900)
                    self.crop_thumb_in.set_image(thumb, full_w=full_w, full_h=full_h, callback=self._on_rectangle_selected)
                    self.crop_thumb_in.set_selection_from_full_coords(
                        self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value()
                    )
                self._set_label_pixmap(self.crop_thumb_out, _downsample_for_preview(roi_preview, 512))
            self.info_label.setText(f"Preview ready: {shape_text} | Reader: {self.backend.reader}")
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", str(e))

    def _crop_output_settings(self):
        suffix = self.crop_suffix_edit.text().strip() or "final"
        downsample = float(self.crop_downsample_spin.value())
        if downsample != 1.0:
            ds_txt = f"DS{int(downsample)}" if downsample.is_integer() else f"DS{downsample:g}"
            suffix = f"{suffix}{ds_txt}"
        combo = self.crop_out_combo.currentText()
        output_format = _write_format_from_combo(combo)
        write_ome = combo.startswith("OME-TIFF")
        ext = ".ome.tif" if write_ome else _extension_from_combo(combo)
        return suffix, downsample, combo, output_format, write_ome, ext

    def crop_image(self):
        if not self.backend.path or not self.backend.path_obj:
            QMessageBox.warning(self, "Error", "Please select a file first.")
            return
        try:
            suffix, raw_downsample, combo, output_format, write_ome, ext = self._crop_output_settings()
            out_path = self.backend.path_obj.parent / f"{self.backend.path_obj.stem}_crop_{suffix}{ext}"

            preserve_raw = (
                hasattr(self, "crop_preserve_channels_chk")
                and self.crop_preserve_channels_chk.isChecked()
                and self.backend.reader == "tifffile"
                and output_format != "jpeg"
            )

            if preserve_raw:
                roi, axes, _ = self.backend.crop_raw(
                    self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value()
                )
                if raw_downsample != 1.0:
                    roi, axes = _resize_spatial_array(roi, axes, raw_downsample)
                save_multichannel_image(
                    out_path, roi, axes=axes, write_ome=write_ome,
                    lossless=self.crop_lossless_chk.isChecked(),
                    source_resolution=self.backend.source_resolution,
                    source_mpp=self.backend.source_mpp,
                    image_name=out_path.stem,
                    pixel_scale=raw_downsample
                )
                preview_rgb = _array_to_rgb_preview(roi, axes)
                saved_text = f"Saved raw multichannel crop: {out_path} | shape={tuple(roi.shape)} | axes={axes or 'unknown'} | dtype={roi.dtype}"
            else:
                roi, _ = self.backend.crop(
                    self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value(), fill=255
                )
                if raw_downsample != 1.0:
                    from PIL import Image
                    new_w = max(1, int(round(roi.shape[1] / raw_downsample)))
                    new_h = max(1, int(round(roi.shape[0] / raw_downsample)))
                    roi = np.asarray(Image.fromarray(roi).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)
                save_rgb_image(
                    out_path, roi, output_format, write_ome, self.crop_lossless_chk.isChecked(),
                    self.backend.source_resolution, self.backend.source_mpp, out_path.stem,
                    self.backend.openslide_props if write_ome else None,
                    pixel_scale=raw_downsample
                )
                preview_rgb = roi
                saved_text = f"Saved RGB crop: {out_path} | Reader: {self.backend.reader}"

            if self.crop_preview_chk.isChecked():
                self._set_label_pixmap(self.crop_thumb_out, _downsample_for_preview(preview_rgb, 512))
            self.info_label.setText(saved_text)
            QMessageBox.information(self, "Success", saved_text)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ========================================================
    # Tiles page
    # ========================================================

    def _build_tiles_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        file_layout = QHBoxLayout()
        self.tiles_file_label = QLabel("No image selected")
        self.tiles_file_label.setStyleSheet("padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px;")
        one_btn = QPushButton("Load one image")
        one_btn.clicked.connect(self.load_one_tile_image)
        bulk_btn = QPushButton("Bulk select images")
        bulk_btn.clicked.connect(self.load_bulk_tile_images)
        file_layout.addWidget(one_btn)
        file_layout.addWidget(bulk_btn)
        file_layout.addWidget(self.tiles_file_label, 1)
        layout.addLayout(file_layout)

        params = QGroupBox("Tiling parameters")
        grid = QGridLayout(params)

        self.tile_mode_label = QLabel("Tiling method:")
        self.tile_mode_combo = QComboBox()
        self.tile_mode_combo.addItems(["Fixed square tile size", "Divide image by rows/columns"])
        self.tile_mode_combo.currentIndexChanged.connect(self.update_tile_mode_controls)

        self.tile_fixed_info_label = QLabel(
            "Fixed square tiles. True border tiles are padded to tile size. Middle tiles are direct crops."
        )
        self.tile_fixed_info_label.setStyleSheet("color: #555;")
        self.tile_division_info_label = QLabel(
            "Divide image into a fixed grid. Tiles may be rectangular; no artificial padding is added."
        )
        self.tile_division_info_label.setStyleSheet("color: #555;")

        self.tile_size_label = QLabel("Tile size px:")
        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(16, 100000)
        self.tile_size_spin.setValue(1024)

        self.tile_overlap_label = QLabel("Overlap:")
        self.tile_overlap_spin = QDoubleSpinBox()
        self.tile_overlap_spin.setRange(0, 99.9)
        self.tile_overlap_spin.setDecimals(2)
        self.tile_overlap_spin.setValue(0.0)
        self.tile_overlap_spin.setSuffix(" %")


        self.tile_padding_label = QLabel("Padding color:")
        self.tile_padding_combo = QComboBox()
        self.tile_padding_combo.addItems(["black", "white"])

        self.tile_rows_label = QLabel("Rows:")
        self.tile_rows_spin = QSpinBox()
        self.tile_rows_spin.setRange(1, 10000)
        self.tile_rows_spin.setValue(2)

        self.tile_cols_label = QLabel("Columns:")
        self.tile_cols_spin = QSpinBox()
        self.tile_cols_spin.setRange(1, 10000)
        self.tile_cols_spin.setValue(2)

        self.tile_downsample_label = QLabel("Downsample:")
        self.tile_downsample_spin = QDoubleSpinBox()
        self.tile_downsample_spin.setRange(0.01, 1000.0)
        self.tile_downsample_spin.setDecimals(2)
        self.tile_downsample_spin.setValue(1.0)

        self.tile_format_label = QLabel("Format:")
        self.tile_out_combo = QComboBox()
        self.tile_out_combo.addItems(["TIFF (.tif)", "JPEG (.jpg)"])

        self.tile_lossless_chk = QCheckBox("Lossless TIFF compression")
        self.tile_lossless_chk.setChecked(True)

        grid.addWidget(self.tile_mode_label, 0, 0)
        grid.addWidget(self.tile_mode_combo, 0, 1, 1, 2)
        grid.addWidget(self.tile_fixed_info_label, 1, 0, 1, 4)
        grid.addWidget(self.tile_division_info_label, 1, 0, 1, 4)

        grid.addWidget(self.tile_size_label, 2, 0)
        grid.addWidget(self.tile_size_spin, 2, 1)
        grid.addWidget(self.tile_overlap_label, 2, 2)
        grid.addWidget(self.tile_overlap_spin, 2, 3)
        grid.addWidget(self.tile_padding_label, 3, 0)
        grid.addWidget(self.tile_padding_combo, 3, 1)

        grid.addWidget(self.tile_rows_label, 2, 0)
        grid.addWidget(self.tile_rows_spin, 2, 1)
        grid.addWidget(self.tile_cols_label, 2, 2)
        grid.addWidget(self.tile_cols_spin, 2, 3)

        grid.addWidget(self.tile_downsample_label, 4, 0)
        grid.addWidget(self.tile_downsample_spin, 4, 1)
        grid.addWidget(self.tile_format_label, 4, 2)
        grid.addWidget(self.tile_out_combo, 4, 3)
        grid.addWidget(self.tile_lossless_chk, 5, 0, 1, 4)

        layout.addWidget(params)

        self.tile_fixed_widgets = [
            self.tile_fixed_info_label,
            self.tile_size_label,
            self.tile_size_spin,
            self.tile_overlap_label,
            self.tile_overlap_spin,
            self.tile_padding_label,
            self.tile_padding_combo,
        ]
        self.tile_division_widgets = [
            self.tile_division_info_label,
            self.tile_rows_label,
            self.tile_rows_spin,
            self.tile_cols_label,
            self.tile_cols_spin,
        ]

        prev = QGroupBox("Thumbnail / Grid Preview")
        prev_layout = QHBoxLayout(prev)
        self.tiles_thumb = QLabel("Thumbnail")
        self.tiles_thumb.setAlignment(Qt.AlignCenter)
        self.tiles_thumb.setFixedSize(760, 420)
        self.tiles_thumb.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")
        prev_layout.addWidget(self.tiles_thumb)
        layout.addWidget(prev)

        btn_row = QHBoxLayout()
        preview_btn = QPushButton("Preview Grid")
        preview_btn.clicked.connect(self.preview_tiles_grid)
        save_btn = QPushButton("SAVE TILES")
        save_btn.clicked.connect(self.save_tiles_bulk)
        btn_row.addWidget(preview_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        self.update_tile_mode_controls()
        return page

    def update_tile_mode_controls(self):
        if not hasattr(self, "tile_mode_combo"):
            return
        is_fixed = self.tile_mode_combo.currentText().startswith("Fixed")
        self._set_widgets_visible(self.tile_fixed_widgets, is_fixed)
        self._set_widgets_visible(self.tile_division_widgets, not is_fixed)


        if hasattr(self, "info_label"):
            if is_fixed:
                self.info_label.setText(
                    "Tiles mode: fixed square tiles. True border tiles are padded; middle tiles are direct crops."
                )
            else:
                self.info_label.setText("Tiles mode: divide image by rows and columns. Tiles may be rectangular.")

    def load_one_tile_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", _image_file_filter())
        if path:
            self.bulk_paths = [Path(path)]
            self._load_tiles_preview_image(self.bulk_paths[0])

    def load_bulk_tile_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select images for bulk tiling", "", _image_file_filter())
        if paths:
            self.bulk_paths = [Path(p) for p in paths]
            self._load_tiles_preview_image(self.bulk_paths[0])

    def _load_tiles_preview_image(self, path):
        self.backend = ImageBackend().load(str(path))
        w, h = self.backend.slide_dims
        print("Loaded for tiling with:", self.backend.reader, self.backend.file_kind, self.backend.path_obj.suffix)
        self.tiles_file_label.setText(
            f"Selected: {len(self.bulk_paths)} file(s) | Preview: {path.name} | {w} x {h} px | Reader: {self.backend.reader}"
        )
        self._set_label_pixmap(self.tiles_thumb, self.backend.input_thumbnail(max_side=768))
        self.info_label.setText(f"Loaded for tiling: {path.name} | Reader: {self.backend.reader}")

    def _tile_params(self):
        return {
            "mode": self.tile_mode_combo.currentText(),
            "tile_size": int(self.tile_size_spin.value()),
            "overlap": float(self.tile_overlap_spin.value()),
            "rows": int(self.tile_rows_spin.value()),
            "cols": int(self.tile_cols_spin.value()),
            "downsample": float(self.tile_downsample_spin.value()),
            "padding": self.tile_padding_combo.currentText(),
            "output_format": _write_format_from_combo(self.tile_out_combo.currentText()),
            "ext": _extension_from_combo(self.tile_out_combo.currentText()),
        }

    def preview_tiles_grid(self):
        if not self.backend.path or not self.backend.slide_dims:
            QMessageBox.warning(self, "Error", "Load one or more images first.")
            return
        try:
            params = self._tile_params()
            full_w, full_h = self.backend.slide_dims
            thumb = self.backend.input_thumbnail(max_side=768)

            if params["mode"].startswith("Fixed"):
                tile_size = params["tile_size"]
                overlap = params["overlap"]
                pm = self._draw_fixed_grid_on_thumb(thumb, full_w, full_h, tile_size, overlap)
                xs, ys, stride, ovpx = _compute_tile_grid(full_w, full_h, tile_size, overlap)
                self.info_label.setText(
                    f"Fixed grid: {len(xs)} columns x {len(ys)} rows = {len(xs) * len(ys)} tiles | "
                    f"Overlap {overlap:g}% ({ovpx}px) | Stride {stride}px | Reader: {self.backend.reader}"
                )
            else:
                rows = params["rows"]
                cols = params["cols"]
                pm = self._draw_division_grid_on_thumb(thumb, full_w, full_h, rows, cols)
                approx_w = int(math.ceil(full_w / cols))
                approx_h = int(math.ceil(full_h / rows))
                self.info_label.setText(
                    f"Division grid: {rows} rows x {cols} columns = {rows * cols} tiles | "
                    f"Approx tile size: {approx_w} x {approx_h} px | Reader: {self.backend.reader}"
                )

            self.tiles_thumb.setPixmap(pm.scaled(self.tiles_thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", str(e))

    def save_tiles_bulk(self):
        if not self.bulk_paths:
            QMessageBox.warning(self, "Error", "Load one or more images first.")
            return
        log_rows = []
        try:
            params = self._tile_params()
            jobs = []
            total_tiles = 0
            for p in self.bulk_paths:
                b = ImageBackend().load(str(p))
                try:
                    w, h = b.slide_dims
                    if params["mode"].startswith("Fixed"):
                        xs, ys, _, _ = _compute_tile_grid(w, h, params["tile_size"], params["overlap"])
                        n_tiles = len(xs) * len(ys)
                    else:
                        n_tiles = params["rows"] * params["cols"]
                    jobs.append((p, b.reader, n_tiles))
                    total_tiles += n_tiles
                finally:
                    b.close()

            if total_tiles == 0:
                QMessageBox.warning(self, "No tiles", "No tiles would be generated with the current settings.")
                return

            self.progress.setVisible(True)
            self.progress.setRange(0, total_tiles)
            self.progress.setValue(0)
            QApplication.processEvents()

            written = 0
            failed = 0
            log_base = self.bulk_paths[0].parent if self.bulk_paths else Path.cwd()

            for p, reader_name, expected_tiles in jobs:
                try:
                    if params["mode"].startswith("Fixed"):
                        count = self._save_fixed_tiles_one_image(p, params, written)
                        operation = "fixed_tiles"
                        out_folder = p.parent / p.stem
                    else:
                        count = self._save_division_tiles_one_image(p, params, written)
                        operation = "division_tiles"
                        out_folder = p.parent / f"{p.stem}_{_suffix_for_division_tile(params['rows'], params['cols'], params['downsample'])}"
                    written += count
                    log_rows.append({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "operation": operation,
                        "status": "success",
                        "image": str(p),
                        "reader": reader_name,
                        "output_folder": str(out_folder),
                        "tiles_expected": expected_tiles,
                        "tiles_written": count,
                        "message": "",
                    })
                except Exception as image_error:
                    failed += 1
                    message = f"{image_error}\n{traceback.format_exc()}"
                    log_rows.append({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "operation": "fixed_tiles" if params["mode"].startswith("Fixed") else "division_tiles",
                        "status": "failed",
                        "image": str(p),
                        "reader": reader_name,
                        "output_folder": "",
                        "tiles_expected": expected_tiles,
                        "tiles_written": 0,
                        "message": message,
                    })
                    self.info_label.setText(f"Failed: {p.name}. Continuing with remaining files...")
                    QApplication.processEvents()

            log_path = self._write_batch_log(log_base, log_rows)
            self.progress.setVisible(False)

            if failed:
                self.info_label.setText(f"Done with warnings: saved {written} tiles; {failed} image(s) failed. Log: {log_path}")
                QMessageBox.warning(
                    self,
                    "Done with warnings",
                    f"Saved {written} tiles from {len(self.bulk_paths) - failed} image(s).\n"
                    f"Failed images: {failed}\n\nLog saved to:\n{log_path}"
                )
            else:
                self.info_label.setText(f"Done: saved {written} tiles from {len(self.bulk_paths)} image(s). Log: {log_path}")
                QMessageBox.information(
                    self,
                    "Done",
                    f"Saved {written} tiles from {len(self.bulk_paths)} image(s).\n\nLog saved to:\n{log_path}"
                )
        except Exception as e:
            self.progress.setVisible(False)
            if log_rows:
                try:
                    log_path = self._write_batch_log(self.bulk_paths[0].parent, log_rows)
                    QMessageBox.critical(self, "Tiling Error", f"{e}\n\nPartial log saved to:\n{log_path}")
                    return
                except Exception:
                    pass
            QMessageBox.critical(self, "Tiling Error", str(e))

    def _save_fixed_tiles_one_image(self, image_path, params, start_progress):
        backend = ImageBackend().load(str(image_path))
        full_w, full_h = backend.slide_dims
        tile_size = params["tile_size"]
        overlap = params["overlap"]
        downsample = params["downsample"]
        padding = params["padding"]
        output_format = params["output_format"]
        ext = params["ext"]

        suffix = _suffix_for_tile(overlap, downsample)

        xs, ys, _, _ = _compute_tile_grid(full_w, full_h, tile_size, overlap)
        out_dir = image_path.parent / image_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        total = len(xs) * len(ys)

        for row_idx, y in enumerate(ys):
            for col_idx, x in enumerate(xs):
                crop_w = min(tile_size, full_w - x)
                crop_h = min(tile_size, full_h - y)
                if crop_w <= 0 or crop_h <= 0:
                    continue

                # Only use padding when the tile truly touches the image boundary.
                # For internal/middle tiles, use a direct crop. This avoids accidental padding
                # appearing in the middle of the image.
                is_edge_tile = (x + tile_size > full_w) or (y + tile_size > full_h)

                if is_edge_tile:
                    tile = backend.read_tile_with_padding(x=x, y=y, size=tile_size, padding_color=padding)
                    actual_w, actual_h = crop_w, crop_h
                else:
                    tile, _ = backend.crop(x, y, tile_size, tile_size, fill=255)
                    actual_w, actual_h = tile_size, tile_size

                if float(downsample) != 1.0:
                    from PIL import Image
                    new_w = max(1, int(round(tile.shape[1] / downsample)))
                    new_h = max(1, int(round(tile.shape[0] / downsample)))
                    tile = np.asarray(Image.fromarray(tile).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)

                out_name = (
                    f"{image_path.stem}_{_col_to_letters(col_idx)}{row_idx + 1}{suffix}"
                    f"_X{x}_Y{y}_W{actual_w}_H{actual_h}{ext}"
                )
                out_path = out_dir / out_name
                save_rgb_image(
                    out_path, tile, output_format, False, self.tile_lossless_chk.isChecked(),
                    backend.source_resolution, backend.source_mpp, out_path.stem,
                    pixel_scale=downsample
                )
                count += 1
                self.progress.setValue(start_progress + count)
                if count % 10 == 0:
                    self.info_label.setText(
                        f"Saving fixed tiles: {image_path.name} | {count}/{total} | Reader: {backend.reader}"
                    )
                    QApplication.processEvents()

        backend.close()
        QApplication.processEvents()
        return count

    def _save_division_tiles_one_image(self, image_path, params, start_progress):
        backend = ImageBackend().load(str(image_path))
        full_w, full_h = backend.slide_dims
        rows = params["rows"]
        cols = params["cols"]
        downsample = params["downsample"]
        output_format = params["output_format"]
        ext = params["ext"]
        suffix = _suffix_for_division_tile(rows, cols, downsample)
        out_dir = image_path.parent / f"{image_path.stem}_{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        total = rows * cols
        count = 0

        for r in range(rows):
            for c in range(cols):
                x, y, w, h = _division_bounds(0, 0, full_w, full_h, rows, cols, r, c)
                tile, _ = backend.crop(x, y, w, h, fill=255)

                if float(downsample) != 1.0:
                    from PIL import Image
                    new_w = max(1, int(round(tile.shape[1] / downsample)))
                    new_h = max(1, int(round(tile.shape[0] / downsample)))
                    tile = np.asarray(Image.fromarray(tile).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)

                out_name = f"{image_path.stem}_R{r + 1:03d}_C{c + 1:03d}{suffix}_X{x}_Y{y}_W{w}_H{h}{ext}"
                out_path = out_dir / out_name
                save_rgb_image(
                    out_path, tile, output_format, False, self.tile_lossless_chk.isChecked(),
                    backend.source_resolution, backend.source_mpp, out_path.stem,
                    pixel_scale=downsample
                )
                count += 1
                self.progress.setValue(start_progress + count)
                if count % 5 == 0:
                    self.info_label.setText(f"Saving division tiles: {image_path.name} | {count}/{total} | Reader: {backend.reader}")
                    QApplication.processEvents()

        backend.close()
        QApplication.processEvents()
        return count

    # ========================================================
    # Merge page
    # ========================================================

    def _build_merge_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        file_layout = QHBoxLayout()
        folder_btn = QPushButton("Select tile folder")
        folder_btn.clicked.connect(self.select_tile_folder)
        files_btn = QPushButton("Select tile files")
        files_btn.clicked.connect(self.select_tile_files)
        self.merge_file_label = QLabel("No tiles selected")
        self.merge_file_label.setStyleSheet("padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px;")
        file_layout.addWidget(folder_btn)
        file_layout.addWidget(files_btn)
        file_layout.addWidget(self.merge_file_label, 1)
        layout.addLayout(file_layout)

        params = QGroupBox("Merge parameters")
        grid = QGridLayout(params)
        self.merge_mode_combo = QComboBox()
        self.merge_mode_combo.addItems(["Auto (from names)", "Manual grid"])
        self.merge_overlap_edit = QLineEdit("auto")
        self.merge_overlap_edit.setMaximumWidth(100)
        self.merge_padding_combo = QComboBox()
        self.merge_padding_combo.addItems(["black", "white"])
        self.merge_crop_padding_chk = QCheckBox("Crop external padding after merge")
        self.merge_crop_padding_chk.setChecked(True)
        self.merge_out_combo = QComboBox()
        self.merge_out_combo.addItems(["TIFF (.tif)", "JPEG (.jpg)"])
        self.merge_lossless_chk = QCheckBox("Lossless TIFF compression")
        self.merge_lossless_chk.setChecked(True)
        grid.addWidget(QLabel("Mode:"), 0, 0)
        grid.addWidget(self.merge_mode_combo, 0, 1)
        grid.addWidget(QLabel("Overlap %:"), 1, 0)
        grid.addWidget(self.merge_overlap_edit, 1, 1)
        grid.addWidget(QLabel("auto reads Ov# from names; for manual, enter a number"), 1, 2)
        grid.addWidget(QLabel("Padding color to crop:"), 2, 0)
        grid.addWidget(self.merge_padding_combo, 2, 1)
        grid.addWidget(self.merge_crop_padding_chk, 2, 2)
        grid.addWidget(QLabel("Format:"), 3, 0)
        grid.addWidget(self.merge_out_combo, 3, 1)
        grid.addWidget(self.merge_lossless_chk, 3, 2)
        layout.addWidget(params)

        prev = QGroupBox("Reconstruction Preview")
        prev_layout = QHBoxLayout(prev)
        self.merge_thumb = QLabel("Reconstruction thumbnail")
        self.merge_thumb.setAlignment(Qt.AlignCenter)
        self.merge_thumb.setFixedSize(820, 430)
        self.merge_thumb.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")
        prev_layout.addWidget(self.merge_thumb)
        layout.addWidget(prev)

        btn_row = QHBoxLayout()
        prev_btn = QPushButton("Preview Reconstruction")
        prev_btn.clicked.connect(self.preview_merge)
        save_btn = QPushButton("SAVE MERGED IMAGE")
        save_btn.clicked.connect(self.save_merged)
        btn_row.addWidget(prev_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)
        return page

    def select_tile_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder containing tiles")
        if folder:
            folder = Path(folder)
            self.tile_files = sorted([
                p for p in folder.iterdir()
                if p.is_file() and _has_ext(p.name, (".tif", ".tiff", ".ome.tif", ".ome.tiff", ".jpg", ".jpeg", ".png"))
            ])
            self.merge_file_label.setText(f"Loaded {len(self.tile_files)} tile file(s) from {folder}")

    def select_tile_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select tile files", "", _tile_file_filter())
        if paths:
            self.tile_files = sorted([Path(p) for p in paths])
            self.merge_file_label.setText(f"Loaded {len(self.tile_files)} selected tile file(s)")

    def _collect_tile_metadata(self):
        metas = []
        for p in self.tile_files:
            meta = _parse_tile_name(p)
            if meta is not None:
                metas.append(meta)
        if not metas:
            raise ValueError(
                "No valid tiles found. Expected ImageName_A1.tif, ImageName_B2Ov10DS2.tif, "
                "or ImageName_R001_C001Div2x2.tif"
            )
        bases = sorted(set(m["base"] for m in metas))
        if len(bases) != 1:
            raise ValueError(f"Multiple image bases detected: {bases}. Select one tile set at a time.")
        return metas, bases[0]

    def _merge_overlap_value(self, metas):
        txt = self.merge_overlap_edit.text().strip().lower()
        if txt == "auto":
            overlaps = sorted(set(float(m["overlap"]) for m in metas))
            if len(overlaps) > 1:
                raise ValueError(f"Multiple overlap values found: {overlaps}. Enter overlap manually.")
            return overlaps[0]
        return float(txt)

    def _read_tile_file(self, path):
        if _has_ext(path.name, (".jpg", ".jpeg", ".png")):
            from PIL import Image
            return _to_uint8_rgb(np.asarray(Image.open(path).convert("RGB")))
        return _to_uint8_rgb(tifffile.imread(str(path)))

    def _manual_overlap_value(self):
        txt = self.merge_overlap_edit.text().strip().lower()
        if txt == "auto" or txt == "":
            return 0.0
        return float(txt)

    def _open_manual_layout_dialog(self):
        if not self.tile_files:
            QMessageBox.warning(self, "Error", "Select a tile folder or tile files first.")
            return False
        dlg = ManualGridDialog(self.tile_files, self._read_tile_file, self)
        if self.manual_layout:
            try:
                dlg.rows_spin.setValue(int(self.manual_layout.get("rows", 2)))
                dlg.cols_spin.setValue(int(self.manual_layout.get("cols", 2)))
                dlg.build_grid()
                for (r, c), path in self.manual_layout.get("mapping", {}).items():
                    if path in self.tile_files:
                        dlg.mapping[(r, c)] = path
                        item = QTableWidgetItem(path.name)
                        item.setTextAlignment(Qt.AlignCenter)
                        dlg.table.setItem(r, c, item)
            except Exception:
                pass
        if dlg.exec_() != QDialog.Accepted:
            return False
        self.manual_layout = dlg.get_layout()
        return True

    def _tile_canvas_geometry(self, metas, overlap):
        """
        Compute merged canvas size and tile placements.

        When tiles include _X#_Y# coordinate metadata generated by TiffCropper,
        the merge uses those coordinates. This is safer than relying only on
        row/column labels because it handles partial edge tiles and downsampled
        tiles more robustly. If coordinates are missing, it falls back to the
        classic row/column + overlap geometry.

        Returns:
            out_w, out_h, stride_x, stride_y, placements

        placements[path] = (tile_width_px, tile_height_px, x_px, y_px)
        """
        placements = {}
        max_right = 0
        max_bottom = 0

        use_coordinates = all(m.get("x") is not None and m.get("y") is not None for m in metas)

        if use_coordinates:
            # Normalize coordinates so a crop/tile set that starts away from (0, 0)
            # still reconstructs into a compact canvas. Coordinates are stored in
            # original-resolution pixels, so divide by the tile downsample factor.
            scaled_xs = []
            scaled_ys = []
            for m in metas:
                ds = float(m.get("downsample") or 1.0)
                if ds <= 0:
                    ds = 1.0
                scaled_xs.append(int(round(float(m["x"]) / ds)))
                scaled_ys.append(int(round(float(m["y"]) / ds)))

            min_x = min(scaled_xs) if scaled_xs else 0
            min_y = min(scaled_ys) if scaled_ys else 0

            for m, sx, sy in zip(metas, scaled_xs, scaled_ys):
                tile = self._read_tile_file(m["path"])
                th, tw = tile.shape[:2]
                x = sx - min_x
                y = sy - min_y
                placements[m["path"]] = (tw, th, x, y)
                max_right = max(max_right, x + tw)
                max_bottom = max(max_bottom, y + th)

            # Informational only in coordinate mode.
            return int(max_right), int(max_bottom), 0, 0, placements

        # Fallback for older tile names without coordinate metadata.
        first = self._read_tile_file(metas[0]["path"])
        base_h, base_w = first.shape[:2]
        stride_x = base_w - int(round(base_w * overlap / 100.0))
        stride_y = base_h - int(round(base_h * overlap / 100.0))
        if stride_x <= 0 or stride_y <= 0:
            raise ValueError("Overlap must be lower than 100%.")

        for m in metas:
            tile = self._read_tile_file(m["path"])
            th, tw = tile.shape[:2]
            _, _, x, y = placements[m["path"]]
            placements[m["path"]] = (tw, th, x, y)
            max_right = max(max_right, x + tw)
            max_bottom = max(max_bottom, y + th)

        return int(max_right), int(max_bottom), stride_x, stride_y, placements

    def _reconstruct_tiles_manual(self, preview=False):
        if not self.manual_layout:
            if not self._open_manual_layout_dialog():
                raise RuntimeError("Manual layout was cancelled.")

        rows = int(self.manual_layout["rows"])
        cols = int(self.manual_layout["cols"])
        mapping = self.manual_layout["mapping"]
        if not mapping:
            raise ValueError("Manual layout is empty. Assign at least one tile.")

        first_path = next(iter(mapping.values()))
        first = self._read_tile_file(first_path)
        tile_h, tile_w = first.shape[:2]
        overlap = self._manual_overlap_value()
        stride_x = tile_w - int(round(tile_w * overlap / 100.0))
        stride_y = tile_h - int(round(tile_h * overlap / 100.0))
        if stride_x <= 0 or stride_y <= 0:
            raise ValueError("Overlap must be lower than 100%.")

        out_w = 0
        out_h = 0
        tile_cache = {}
        for (r, c), tile_path in mapping.items():
            tile = self._read_tile_file(tile_path)
            tile_cache[(r, c)] = tile
            th, tw = tile.shape[:2]
            out_w = max(out_w, c * stride_x + tw)
            out_h = max(out_h, r * stride_y + th)

        base = Path(first_path).parent.name or "manual_layout"

        if preview:
            scale = min(820 / out_w, 430 / out_h, 1.0)
            prev_w = max(1, int(round(out_w * scale)))
            prev_h = max(1, int(round(out_h * scale)))
            canvas = np.zeros((prev_h, prev_w, 3), dtype=np.uint8)
            from PIL import Image
            for (r, c), tile in tile_cache.items():
                small_w = max(1, int(round(tile.shape[1] * scale)))
                small_h = max(1, int(round(tile.shape[0] * scale)))
                small = np.asarray(Image.fromarray(tile).resize((small_w, small_h), Image.Resampling.BILINEAR), dtype=np.uint8)
                x = int(round(c * stride_x * scale))
                y = int(round(r * stride_y * scale))
                hh, ww = small.shape[:2]
                canvas[y:min(y+hh, prev_h), x:min(x+ww, prev_w), :] = small[:max(0, min(hh, prev_h-y)), :max(0, min(ww, prev_w-x)), :]
            if self.merge_crop_padding_chk.isChecked():
                canvas = _crop_external_padding(canvas, self.merge_padding_combo.currentText())
            return canvas, base, overlap, min(stride_x, stride_y) if stride_x and stride_y else 0

        merged = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(tile_cache))
        self.progress.setValue(0)
        for i, ((r, c), tile) in enumerate(tile_cache.items(), start=1):
            x = c * stride_x
            y = r * stride_y
            th, tw = tile.shape[:2]
            merged[y:y+th, x:x+tw, :] = tile
            self.progress.setValue(i)
            if i % 10 == 0:
                QApplication.processEvents()
        self.progress.setVisible(False)
        if self.merge_crop_padding_chk.isChecked():
            merged = _crop_external_padding(merged, self.merge_padding_combo.currentText())
        return merged, base, overlap, min(stride_x, stride_y) if stride_x and stride_y else 0

    def _reconstruct_tiles(self, preview=False):
        if self.merge_mode_combo.currentText().startswith("Manual"):
            return self._reconstruct_tiles_manual(preview=preview)

        metas, base = self._collect_tile_metadata()
        overlap = self._merge_overlap_value(metas)
        out_w, out_h, stride_x, stride_y, placements = self._tile_canvas_geometry(metas, overlap)

        if preview:
            scale = min(820 / out_w, 430 / out_h, 1.0)
            prev_w = max(1, int(round(out_w * scale)))
            prev_h = max(1, int(round(out_h * scale)))
            canvas = np.zeros((prev_h, prev_w, 3), dtype=np.uint8)
            from PIL import Image
            for m in metas:
                tile = self._read_tile_file(m["path"])
                small_w = max(1, int(round(tile.shape[1] * scale)))
                small_h = max(1, int(round(tile.shape[0] * scale)))
                small = np.asarray(Image.fromarray(tile).resize((small_w, small_h), Image.Resampling.BILINEAR), dtype=np.uint8)
                _, _, x0, y0 = placements[m["path"]]
                x = int(round(x0 * scale))
                y = int(round(y0 * scale))
                hh, ww = small.shape[:2]
                canvas[y:min(y+hh, prev_h), x:min(x+ww, prev_w), :] = small[:max(0, min(hh, prev_h-y)), :max(0, min(ww, prev_w-x)), :]
            if self.merge_crop_padding_chk.isChecked():
                canvas = _crop_external_padding(canvas, self.merge_padding_combo.currentText())
            return canvas, base, overlap, min(stride_x, stride_y) if stride_x and stride_y else 0

        merged = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(metas))
        for i, m in enumerate(metas, start=1):
            tile = self._read_tile_file(m["path"])
            _, _, x, y = placements[m["path"]]
            th, tw = tile.shape[:2]
            merged[y:y+th, x:x+tw, :] = tile
            self.progress.setValue(i)
            if i % 10 == 0:
                QApplication.processEvents()
        self.progress.setVisible(False)
        if self.merge_crop_padding_chk.isChecked():
            merged = _crop_external_padding(merged, self.merge_padding_combo.currentText())
        return merged, base, overlap, min(stride_x, stride_y) if stride_x and stride_y else 0

    def preview_merge(self):
        if not self.tile_files:
            QMessageBox.warning(self, "Error", "Select a tile folder or tile files first.")
            return
        try:
            if self.merge_mode_combo.currentText().startswith("Manual"):
                if not self._open_manual_layout_dialog():
                    return
            rgb, base, overlap, stride = self._reconstruct_tiles(preview=True)
            self._set_label_pixmap(self.merge_thumb, rgb)
            self.info_label.setText(f"Preview ready: {base} | overlap {overlap:g}% | stride {stride}px")
        except Exception as e:
            self.progress.setVisible(False)
            QMessageBox.critical(self, "Merge Preview Error", str(e))

    def save_merged(self):
        if not self.tile_files:
            QMessageBox.warning(self, "Error", "Select a tile folder or tile files first.")
            return
        try:
            rgb, base, overlap, stride = self._reconstruct_tiles(preview=False)
            combo = self.merge_out_combo.currentText()
            output_format = _write_format_from_combo(combo)
            ext = _extension_from_combo(combo)
            suffix = _suffix_for_tile(overlap, 1)
            out_path = self.tile_files[0].parent / f"{base}_merged{suffix}{ext}"

            # Reuse calibration from the first tile. The tiles already contain the
            # final pixel size, so no additional scaling is applied during merge.
            source_resolution = None
            source_mpp = None
            try:
                metadata_backend = ImageBackend().load(str(self.tile_files[0]))
                source_resolution = metadata_backend.source_resolution
                source_mpp = metadata_backend.source_mpp
                metadata_backend.close()
            except Exception:
                source_resolution = None
                source_mpp = None

            save_rgb_image(
                out_path, rgb, output_format, False, self.merge_lossless_chk.isChecked(),
                source_resolution=source_resolution, source_mpp=source_mpp,
                image_name=out_path.stem, pixel_scale=1.0
            )
            self._set_label_pixmap(self.merge_thumb, _downsample_for_preview(rgb, 900))
            self.info_label.setText(f"Saved merged image: {out_path}")
            QMessageBox.information(self, "Done", f"Saved merged image:\n{out_path}")
        except Exception as e:
            self.progress.setVisible(False)
            QMessageBox.critical(self, "Merge Error", str(e))


# ============================================================
# Main
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = WSICropTileMergeGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setWindowIcon(QIcon(resource_path(APP_ICON_PATH)))

    window = WSICropTileMergeGUI()
    window.show()
    sys.exit(app.exec_())

