
import os
import sys
import re
import math
import csv
import json
import traceback
import threading
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    QTableWidget, QTableWidgetItem, QTextBrowser, QDialogButtonBox,
    QScrollArea, QShortcut, QSlider, QColorDialog, QSizePolicy,
    QListWidget, QListWidgetItem, QAbstractItemView
)
from PyQt5.QtCore import Qt, QRect, QPoint, QPointF, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor, QIcon, QKeySequence, QPainterPath, QBrush

# ============================================================
# App metadata
# ============================================================

APP_NAME = "TiffCropper"
APP_VERSION = "1.3"
APP_TITLE = "PathoImage Toolkit: WSI, IF, OME-TIFF and LIF Image Utility"
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


def _is_ome_tiff_name(path_or_name) -> bool:
    """Return True for OME-TIFF names.

    These should be opened with tifffile first because OpenSlide can sometimes
    open them as generic pyramidal TIFFs but not expose OME physical calibration
    or multichannel axes.
    """
    name = str(path_or_name).lower()
    return name.endswith(('.ome.tif', '.ome.tiff'))


def _format_mpp_text(mpp) -> str:
    try:
        if mpp and mpp[0] and mpp[1]:
            return f"{float(mpp[0]):.6g} x {float(mpp[1]):.6g} µm/px"
    except Exception:
        pass
    return "unknown"


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
# Channel preview / downsample helpers
# ============================================================

def _safe_imresize_2d(arr: np.ndarray, new_w: int, new_h: int, resample=None) -> np.ndarray:
    """Resize one 2D plane preserving dtype as much as possible."""
    from PIL import Image
    arr = np.asarray(arr)
    if resample is None:
        resample = Image.Resampling.LANCZOS
    im = Image.fromarray(arr)
    out = np.asarray(im.resize((int(new_w), int(new_h)), resample=resample))
    if out.dtype != arr.dtype:
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            out = np.clip(out, info.min, info.max).astype(arr.dtype)
        else:
            out = out.astype(arr.dtype)
    return out


COLOR_MAPS = {
    "gray": (1.0, 1.0, 1.0),
    "white": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "orange": (1.0, 0.45, 0.0),
}
DEFAULT_CHANNEL_COLORS = ["blue", "green", "red", "magenta", "cyan", "yellow", "orange", "white"]


def _normalize_channel_float(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    """Normalize one channel to float 0..1 for display/exported previews only."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.float32)
    vals = arr[finite]
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def _representative_yx_or_yxc(arr: np.ndarray, axes: str = None) -> Tuple[np.ndarray, str]:
    """Keep Y/X and channel/sample axes, taking first Z/T/M/etc. for display."""
    arr = np.asarray(arr)
    axes = (axes or "").strip()
    if axes and len(axes) == arr.ndim and "Y" in axes and "X" in axes:
        slicer = []
        kept = []
        for ax in axes:
            if ax in ("Y", "X", "C", "S"):
                slicer.append(slice(None))
                kept.append(ax)
            else:
                slicer.append(0)
        arr = arr[tuple(slicer)]
        axes = "".join(kept)
        order = [axes.index("Y"), axes.index("X")]
        out_axes = "YX"
        if "C" in axes:
            order.append(axes.index("C"))
            out_axes += "C"
        elif "S" in axes:
            order.append(axes.index("S"))
            out_axes += "S"
        arr = np.transpose(arr, order)
        return np.asarray(arr), out_axes
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr, "YX"
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            return arr, "YXS"
        if arr.shape[0] <= 16:
            return np.moveaxis(arr, 0, -1), "YXC"
    return _to_uint8_rgb(arr), "YXS"


def _count_display_channels(arr: np.ndarray, axes: str = None) -> int:
    arr2, ax2 = _representative_yx_or_yxc(arr, axes)
    if arr2.ndim == 2:
        return 1
    if arr2.ndim == 3 and ax2.endswith(("C", "S")):
        return int(arr2.shape[-1])
    return 1


def render_channel_composite(arr: np.ndarray, axes: str, channel_settings: List[Dict[str, Any]],
                             p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    """Render scientific multichannel data or preserve true RGB for display.

    Important: RGB/H&E images must not be percentile-normalized channel-by-channel,
    because that changes the colour balance and can create a strange appearance
    when zooming out. If the source has an S/samples axis or is a normal RGB/RGBA
    image, the preview preserves the original RGB values by default.
    """
    arr2, ax2 = _representative_yx_or_yxc(arr, axes)

    # True RGB/RGBA image: preserve colour balance. Channel checkboxes can still
    # hide individual RGB channels, but values are not re-normalized.
    if arr2.ndim == 3 and ax2 == "YXS" and arr2.shape[-1] in (3, 4):
        rgb = _to_uint8_rgb(arr2)
        if not channel_settings:
            return rgb
        out = np.zeros_like(rgb)
        default_rgb_colors = ["red", "green", "blue"]
        for st in channel_settings:
            if not st.get("visible", True):
                continue
            c = int(st.get("channel", 0))
            if c < 0 or c >= min(3, rgb.shape[-1]):
                continue
            color_name = str(st.get("color", default_rgb_colors[c] if c < 3 else "gray")).lower()
            # Default RGB mapping keeps true RGB. Non-default choices are allowed
            # for visual inspection, but still use raw 8-bit values rather than
            # percentile normalization.
            if c < 3 and color_name == default_rgb_colors[c]:
                out[:, :, c] = rgb[:, :, c]
            else:
                color_vec = np.asarray(COLOR_MAPS.get(color_name, COLOR_MAPS["gray"]), dtype=np.float32)
                ch = rgb[:, :, c].astype(np.float32) / 255.0
                out = np.clip(out.astype(np.float32) + ch[:, :, None] * color_vec[None, None, :] * 255.0, 0, 255).astype(np.uint8)
        return out

    if arr2.ndim == 2:
        ch = _normalize_channel_float(arr2, p_low, p_high)
        return np.clip(np.stack([ch, ch, ch], axis=-1) * 255, 0, 255).astype(np.uint8)

    h, w = arr2.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    if not channel_settings:
        n = arr2.shape[-1]
        channel_settings = [
            {"channel": i, "visible": True, "color": DEFAULT_CHANNEL_COLORS[i % len(DEFAULT_CHANNEL_COLORS)]}
            for i in range(n)
        ]
    # Draw in natural channel order. Colors and visibility are user-controlled;
    # order was intentionally removed from the GUI because it did not add much
    # value for additive IF overlays.
    for st in channel_settings:
        if not st.get("visible", True):
            continue
        c = int(st.get("channel", 0))
        if c < 0 or c >= arr2.shape[-1]:
            continue
        color_name = str(st.get("color", "gray")).lower()
        color_vec = np.asarray(COLOR_MAPS.get(color_name, COLOR_MAPS["gray"]), dtype=np.float32)
        ch = _normalize_channel_float(arr2[:, :, c], p_low, p_high)
        rgb += ch[:, :, None] * color_vec[None, None, :]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _placeholder_rgb(message: str, width: int = 900, height: int = 600) -> np.ndarray:
    """Small display-only placeholder used when a full-resolution preview would be unsafe."""
    arr = np.full((int(height), int(width), 3), 238, dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(arr)
        draw = ImageDraw.Draw(im)
        lines = []
        for line in str(message).split("\n"):
            while len(line) > 70:
                lines.append(line[:70])
                line = line[70:]
            lines.append(line)
        y = 25
        for line in lines[:12]:
            draw.text((25, y), line, fill=(40, 40, 40))
            y += 24
        return np.asarray(im, dtype=np.uint8)
    except Exception:
        return arr




def _human_bytes(n: float) -> str:
    """Human readable byte count used in memory-safety messages."""
    try:
        n = float(n)
    except Exception:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.1f} {units[i]}"


MAX_INTERACTIVE_CROP_BYTES = 2.5 * 1024 ** 3
MAX_PREVIEW_CROP_BYTES = 512 * 1024 ** 2


def _exception_text(exc: Exception) -> str:
    """Return a useful error message even for exceptions with empty str(exc)."""
    try:
        msg = str(exc)
    except Exception:
        msg = ""
    if not msg:
        msg = f"{type(exc).__name__} raised with no message."
    try:
        tb = traceback.format_exc()
        if tb and "NoneType: None" not in tb:
            msg = f"{msg}\n\nTraceback:\n{tb}"
    except Exception:
        pass
    return msg


def _is_safe_placeholder_meta(meta: Optional[Dict[str, Any]]) -> bool:
    """Return True when a preview function returned a display-only placeholder.

    Placeholder images are useful inside the GUI, but they must never be used as
    crop/tile output data. This prevents text such as "Preview skipped to keep
    GUI responsive" from being saved as if it were the real image content.
    """
    try:
        reader = str((meta or {}).get("reader", "")).lower()
        if "placeholder" in reader or "safe-placeholder" in reader:
            return True
        if "skipped" in str((meta or {}).get("error", "")).lower():
            return True
    except Exception:
        pass
    return False


def _resize_rgb_to_exact(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize a visual RGB array to an exact output size."""
    rgb = _to_uint8_rgb(rgb)
    out_w = max(1, int(out_w))
    out_h = max(1, int(out_h))
    if rgb.shape[1] == out_w and rgb.shape[0] == out_h:
        return rgb
    from PIL import Image
    return np.asarray(Image.fromarray(rgb).resize((out_w, out_h), Image.Resampling.LANCZOS), dtype=np.uint8)


def _read_visual_crop_for_save(backend, x: int, y: int, w: int, h: int, downsample: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a crop for visual RGB saving without confusing preview levels with full-res coordinates.

    For downsample=1, this asks the backend for an exact full-resolution crop.
    For downsample>1, it deliberately uses a reduced pyramid/overview region and
    then resizes to the exact requested output dimensions. This avoids reading a
    huge full-resolution ROI only to downsample it immediately.
    """
    ds = max(1.0, float(downsample or 1.0))
    out_w = max(1, int(round(float(w) / ds)))
    out_h = max(1, int(round(float(h) / ds)))
    if ds > 1.0:
        # max_side is the desired output size, so read_roi_region_from_backend will
        # choose a pyramid/overview level close to that size when available.
        arr, axes, meta = read_roi_region_from_backend(backend, (int(x), int(y), int(w), int(h)), max_side=max(out_w, out_h))
        if _is_safe_placeholder_meta(meta):
            raise RuntimeError(
                "Save aborted because the ROI preview returned a display-only placeholder instead of pixels. "
                "The placeholder is allowed in the GUI, but it must not be saved as image content. "
                "Use the low-memory crop path, a TIFF/OME-TIFF output, or convert the source to a tiled/pyramidal OME-TIFF with region access."
            )
        rgb = _array_to_rgb_preview(arr, axes)
        rgb = _resize_rgb_to_exact(rgb, out_w, out_h)
        return rgb, {**(meta or {}), "visual_downsample": ds, "target_output_shape": (out_h, out_w, 3)}

    # Exact visual crop. This can still use OpenSlide fallback inside backend.crop()
    # for TIFF/OME-TIFF files that tifffile cannot expose by zarr/memmap.
    rgb, info = backend.crop(int(x), int(y), int(w), int(h), fill=255)
    return _to_uint8_rgb(rgb), {"reader": getattr(backend, "reader", "unknown"), "exact_visual_crop": True, **(info or {})}


def _try_openslide_thumbnail(path: str, max_side: int = 256):
    """Try a small OpenSlide thumbnail even if tifffile could not expose a fast preview.

    This is intended for Explorer thumbnails only. It can help with pyramidal TIFFs
    whose reduced levels are accessible through OpenSlide but not through tifffile.
    """
    openslide = _try_import_openslide()
    if openslide is None:
        return None
    try:
        slide = openslide.OpenSlide(str(path))
        try:
            img = slide.get_thumbnail((int(max_side), int(max_side))).convert("RGB")
            return np.asarray(img, dtype=np.uint8), "YXS", {"reader": "openslide-thumbnail-fallback"}
        finally:
            try:
                slide.close()
            except Exception:
                pass
    except Exception:
        return None


def _safe_int_product(values) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return int(out)


def _estimate_rgb_crop_bytes(width: int, height: int, downsample: float = 1.0, channels: int = 3, dtype=np.uint8) -> int:
    ds = max(1.0, float(downsample or 1.0))
    w = max(1, int(round(float(width) / ds)))
    h = max(1, int(round(float(height) / ds)))
    return _safe_int_product([h, w, int(channels)]) * np.dtype(dtype).itemsize


def _tiff_output_shape_for_crop(backend, x: int, y: int, w: int, h: int, downsample: float = 1.0):
    """Return output shape/axes/dtype for a raw tifffile crop without reading pixels."""
    if backend is None or getattr(backend, "reader", None) != "tifffile":
        return None, None, None
    try:
        za, series, axes, _ = backend._get_tiff_zarr()
        axes = axes or getattr(series, "axes", "") or ""
        shape = tuple(getattr(series, "shape", ()) or ())
        dtype = getattr(series, "dtype", None)
        if dtype is None and za is not None:
            dtype = za.dtype
        dtype = np.dtype(dtype or np.uint16)
        ds = max(1.0, float(downsample or 1.0))
        out_w = max(1, int(round(float(w) / ds)))
        out_h = max(1, int(round(float(h) / ds)))
        if axes and len(axes) == len(shape) and "Y" in axes and "X" in axes:
            out_shape = list(shape)
            out_shape[axes.index("Y")] = out_h
            out_shape[axes.index("X")] = out_w
            return tuple(int(v) for v in out_shape), axes, dtype
        # Conservative fallback assumes the final two axes are Y/X.
        out_shape = list(shape)
        if len(out_shape) >= 2:
            out_shape[-2] = out_h
            out_shape[-1] = out_w
        else:
            out_shape = [out_h, out_w]
        return tuple(int(v) for v in out_shape), _guess_axes_for_array(np.empty((1, 1), dtype=dtype), axes), dtype
    except Exception:
        return None, None, None


def _estimate_tiff_raw_crop_bytes(backend, x: int, y: int, w: int, h: int, downsample: float = 1.0) -> Tuple[Optional[int], Optional[Tuple[int, ...]], Optional[str], Optional[np.dtype]]:
    shape, axes, dtype = _tiff_output_shape_for_crop(backend, x, y, w, h, downsample=downsample)
    if shape is None or dtype is None:
        return None, shape, axes, dtype
    return _safe_int_product(shape) * np.dtype(dtype).itemsize, shape, axes, np.dtype(dtype)


def _normalize_axes_for_tiff_write(arr_shape, axes: str) -> str:
    axes = (axes or "").strip()
    if axes and len(axes) == len(arr_shape):
        return axes
    if len(arr_shape) == 2:
        return "YX"
    if len(arr_shape) == 3:
        if arr_shape[-1] in (3, 4):
            return "YXS"
        if arr_shape[0] <= 16:
            return "CYX"
    return _guess_axes_for_array(np.empty(tuple(1 for _ in arr_shape)), axes)


def save_tiff_raw_crop_lowmem(backend, output_path: Path, x: int, y: int, w: int, h: int,
                              write_ome: bool = True, lossless: bool = True,
                              image_name: Optional[str] = None):
    """Save a raw TIFF/OME-TIFF crop without materializing the full ROI in RAM.

    This path is intentionally conservative: it supports no-downsample crops and
    non-Sample axes such as CYX, ZCYX, TCYX, TCZYX. RGB/Sample-axis images are
    not streamed here because tifffile may collapse singleton T/Z axes for RGB
    samples; those should be saved as visual crops with a downsample or tiles.
    """
    if backend is None or getattr(backend, "reader", None) != "tifffile":
        raise RuntimeError("Low-memory raw crop requires a tifffile-backed image.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_w, full_h = backend.slide_dims
    x, y, w, h = backend.clip_roi(int(x), int(y), int(w), int(h), full_w, full_h)
    za, series, axes, zarr_error = backend._get_tiff_zarr()
    axes = axes or getattr(series, "axes", "") or ""
    shape = tuple(getattr(series, "shape", ()) or ())
    dtype = np.dtype(getattr(series, "dtype", None) or (za.dtype if za is not None else np.uint16))
    if not axes or len(axes) != len(shape) or "Y" not in axes or "X" not in axes:
        raise RuntimeError("Low-memory raw crop requires known Y/X axes in the TIFF series.")
    if "S" in axes:
        raise RuntimeError(
            "Low-memory raw crop does not stream RGB/Sample-axis images safely. "
            "Use a downsampled visual crop or tile the image instead."
        )
    if za is None:
        try:
            za = tifffile.memmap(backend.path, series=0)
        except Exception as exc:
            raise RuntimeError(f"Could not access TIFF pixels by zarr or memmap for low-memory crop: {zarr_error}; {exc}")

    y_axis = axes.index("Y")
    x_axis = axes.index("X")
    out_shape = list(shape)
    out_shape[y_axis] = int(h)
    out_shape[x_axis] = int(w)
    out_shape = tuple(int(v) for v in out_shape)
    non_spatial_axes = [i for i, ax in enumerate(axes) if ax not in ("Y", "X")]
    non_spatial_shape = tuple(int(shape[i]) for i in non_spatial_axes)
    non_spatial_pos = {axis_index: pos for pos, axis_index in enumerate(non_spatial_axes)}

    def gen_planes():
        iterator_shape = non_spatial_shape if non_spatial_shape else (1,)
        for idx in np.ndindex(iterator_shape):
            slicer = []
            for dim_i, ax in enumerate(axes):
                if ax == "Y":
                    slicer.append(slice(int(y), int(y + h)))
                elif ax == "X":
                    slicer.append(slice(int(x), int(x + w)))
                else:
                    slicer.append(idx[non_spatial_pos[dim_i]] if non_spatial_shape else 0)
            plane = np.asarray(za[tuple(slicer)])
            yield np.ascontiguousarray(plane)

    metadata = {"axes": axes, "Name": image_name or output_path.stem}
    if backend.source_mpp:
        try:
            metadata.update({
                "PhysicalSizeX": float(backend.source_mpp[0]),
                "PhysicalSizeY": float(backend.source_mpp[1]),
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeYUnit": "µm",
            })
        except Exception:
            pass
    resolution = None
    resolutionunit = None
    if backend.source_resolution:
        try:
            xres, yres, unit = backend.source_resolution
            if xres and yres and unit:
                resolution = (float(xres), float(yres))
                resolutionunit = unit
        except Exception:
            pass
    compression_kwargs = {"compression": "deflate", "predictor": True} if lossless else {}
    tifffile.imwrite(
        str(output_path), gen_planes(), shape=out_shape, dtype=dtype,
        bigtiff=True, ome=bool(write_ome), metadata=metadata,
        photometric="minisblack", software=f"{APP_NAME} v{APP_VERSION}",
        resolution=resolution, resolutionunit=resolutionunit, **compression_kwargs
    )
    return {"shape": out_shape, "axes": axes, "dtype": str(dtype), "lowmem": True}
def read_preview_array_from_file(path: str, max_side: int = 1200, allow_full_fallback: bool = False) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """Read a memory-light representative preview array from RGB, TIFF, or OME-TIFF.

    This function deliberately avoids full reads of very large TIFF/OME-TIFF files.
    If zarr/levels/memmap access is unavailable and the image is large, it returns a
    placeholder instead of freezing the GUI.
    """
    path = str(path)
    p = Path(path)
    meta = {"path": path, "reader": "unknown"}
    if _has_ext(p.name, RASTER_EXTENSIONS):
        from PIL import Image
        im = Image.open(path)
        im.thumbnail((max_side, max_side))
        arr = np.asarray(im.convert("RGB"))
        return arr, "YXS", {**meta, "reader": "PIL", "shape": tuple(arr.shape), "axes": "YXS"}

    if _has_ext(p.name, TIFF_EXTENSIONS) and not p.name.lower().endswith(".svs"):
        with tifffile.TiffFile(path) as tif:
            s = tif.series[0]
            axes = getattr(s, "axes", "") or ""
            shape = tuple(s.shape)
            try:
                if hasattr(s, "levels") and len(s.levels) > 1:
                    best = s.levels[-1]
                    arr = best.asarray()
                    axes2 = getattr(best, "axes", axes) or axes
                    return arr, axes2, {**meta, "reader": "tifffile-level", "shape": shape, "axes": axes}
            except Exception:
                pass

            # Fast thumbnail path: try available overview/secondary pyramid levels
            # before touching full-resolution zarr. This keeps initial thumbnails
            # responsive for large compressed OME-TIFF files.
            try:
                if axes and len(axes) == len(shape) and "Y" in axes and "X" in axes:
                    full_h0 = int(shape[axes.index("Y")]); full_w0 = int(shape[axes.index("X")])
                elif len(shape) >= 2:
                    full_h0, full_w0 = int(shape[-2]), int(shape[-1])
                else:
                    full_w0 = full_h0 = 0
                if full_w0 > 0 and full_h0 > 0:
                    overview = _read_best_tiff_overview_region(
                        s, tif, full_w0, full_h0, (0, 0, full_w0, full_h0),
                        target_ds=max(1.0, max(full_w0, full_h0) / float(max(1, max_side))),
                        max_side=max_side,
                        primary_axes=axes,
                        prefer_second_level=True,
                    )
                    if overview is not None:
                        arr, axes2, ometa = overview
                        return arr, axes2, {**meta, **ometa, "shape": shape, "axes": axes, "fast_overview_first": True}
            except Exception as overview_error:
                last_error = overview_error

            # Preferred path for tiled/compressed OME-TIFF: zarr region stepping.
            try:
                z = s.aszarr()
                import zarr
                za = zarr.open(z, mode="r")
                if axes and len(axes) == za.ndim and "Y" in axes and "X" in axes:
                    y_axis = axes.index("Y")
                    x_axis = axes.index("X")
                    y = int(za.shape[y_axis])
                    x = int(za.shape[x_axis])
                    step = max(1, int(math.ceil(max(y, x) / float(max_side))))
                    slicer = []
                    kept = []
                    for ax in axes:
                        if ax == "Y":
                            slicer.append(slice(0, y, step)); kept.append("Y")
                        elif ax == "X":
                            slicer.append(slice(0, x, step)); kept.append("X")
                        elif ax in ("C", "S"):
                            slicer.append(slice(None)); kept.append(ax)
                        else:
                            slicer.append(0)
                    arr = np.asarray(za[tuple(slicer)])
                    return arr, "".join(kept), {**meta, "reader": "tifffile-zarr", "shape": shape, "axes": axes, "step": step}
            except Exception as zarr_error:
                last_error = zarr_error

            # Try memory mapping for uncompressed/non-tiled TIFFs.
            try:
                mm = tifffile.memmap(path, series=0)
                mm_axes = axes if axes and len(axes) == mm.ndim else _guess_axes_for_array(mm, axes)
                if mm_axes and len(mm_axes) == mm.ndim and "Y" in mm_axes and "X" in mm_axes:
                    y_axis = mm_axes.index("Y")
                    x_axis = mm_axes.index("X")
                    y = int(mm.shape[y_axis])
                    x = int(mm.shape[x_axis])
                    step = max(1, int(math.ceil(max(y, x) / float(max_side))))
                    slicer = []
                    kept = []
                    for ax in mm_axes:
                        if ax == "Y":
                            slicer.append(slice(0, y, step)); kept.append("Y")
                        elif ax == "X":
                            slicer.append(slice(0, x, step)); kept.append("X")
                        elif ax in ("C", "S"):
                            slicer.append(slice(None)); kept.append(ax)
                        else:
                            slicer.append(0)
                    arr = np.asarray(mm[tuple(slicer)])
                    return arr, "".join(kept), {**meta, "reader": "tifffile-memmap", "shape": shape, "axes": mm_axes, "step": step}
            except Exception as mmap_error:
                last_error = mmap_error

            # If full-res zarr/memmap is unavailable, try a reduced-resolution
            # TIFF/OME-TIFF pyramid level or secondary overview series before
            # giving up. This is especially useful for the Crop preview.
            try:
                if axes and len(axes) == len(shape) and "Y" in axes and "X" in axes:
                    full_h0 = int(shape[axes.index("Y")]); full_w0 = int(shape[axes.index("X")])
                elif len(shape) >= 2:
                    full_h0, full_w0 = int(shape[-2]), int(shape[-1])
                else:
                    full_w0 = full_h0 = 0
                if full_w0 > 0 and full_h0 > 0:
                    overview = _read_best_tiff_overview_region(
                        s, tif, full_w0, full_h0, (0, 0, full_w0, full_h0),
                        target_ds=max(1.0, max(full_w0, full_h0) / float(max(1, max_side))),
                        max_side=max_side,
                        primary_axes=axes,
                        prefer_second_level=True,
                    )
                    if overview is not None:
                        arr, axes2, ometa = overview
                        return arr, axes2, {**meta, **ometa, "shape": shape, "axes": axes}
            except Exception as overview_error:
                last_error = overview_error

            # Only small files are allowed to fall back to full read.
            spatial_pixels = 0
            try:
                if axes and len(axes) == len(shape) and "Y" in axes and "X" in axes:
                    spatial_pixels = int(shape[axes.index("Y")]) * int(shape[axes.index("X")])
                elif len(shape) >= 2:
                    spatial_pixels = int(shape[-2]) * int(shape[-1])
            except Exception:
                spatial_pixels = 0
            if allow_full_fallback or spatial_pixels <= 25_000_000:
                arr = s.asarray()
                arr2, axes2 = _representative_yx_or_yxc(arr, axes)
                rgb = _downsample_for_preview(_array_to_rgb_preview(arr2, axes2), max_side=max_side)
                return rgb, "YXS", {**meta, "reader": "tifffile-full-fallback", "shape": shape, "axes": axes}

            msg = (
                "Preview skipped to keep GUI responsive.\n"
                f"File: {p.name}\nShape: {shape} axes={axes}\n"
                "This TIFF could not expose a tiled/zarr or memmap preview.\n"
                "Saving/cropping may still work; preview would require a full read."
            )
            return _placeholder_rgb(msg), "YXS", {**meta, "reader": "safe-placeholder", "shape": shape, "axes": axes, "error": str(last_error)}

    b = ImageBackend().load(path)
    try:
        arr = b.input_thumbnail(max_side=max_side)
        return arr, "YXS", {**meta, "reader": b.reader, "shape": tuple(arr.shape), "axes": "YXS"}
    finally:
        b.close()

def save_preview_jpg(path: Path, rgb: np.ndarray, quality: int = 95):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8_rgb(rgb)).save(str(path), quality=int(quality))


def _downsample_tiff_raw_lowmem(input_path: Path, output_path: Path, downsample: float,
                                write_ome: bool = True, lossless: bool = True,
                                source_resolution=None, source_mpp=None):
    """Downsample a TIFF/OME-TIFF plane-by-plane to reduce RAM usage."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffFile(str(input_path)) as tif:
        s = tif.series[0]
        axes = getattr(s, "axes", "") or ""
        if not axes or len(axes) != len(s.shape) or "Y" not in axes or "X" not in axes:
            arr = s.asarray()
            arr, axes = _resize_spatial_array(arr, axes, downsample)
            save_multichannel_image(output_path, arr, axes=axes, write_ome=write_ome, lossless=lossless,
                                    source_resolution=source_resolution, source_mpp=source_mpp,
                                    image_name=output_path.stem, pixel_scale=downsample)
            return
        try:
            z = s.aszarr()
            import zarr
            za = zarr.open(z, mode="r")
        except Exception:
            arr = s.asarray()
            arr, axes = _resize_spatial_array(arr, axes, downsample)
            save_multichannel_image(output_path, arr, axes=axes, write_ome=write_ome, lossless=lossless,
                                    source_resolution=source_resolution, source_mpp=source_mpp,
                                    image_name=output_path.stem, pixel_scale=downsample)
            return

        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        y_size = int(za.shape[y_axis])
        x_size = int(za.shape[x_axis])
        new_y = max(1, int(round(y_size / float(downsample))))
        new_x = max(1, int(round(x_size / float(downsample))))
        out_shape = list(za.shape)
        out_shape[y_axis] = new_y
        out_shape[x_axis] = new_x
        out_shape = tuple(out_shape)
        prefix_axes = [i for i, ax in enumerate(axes) if ax not in ("Y", "X")]
        prefix_shape = tuple(za.shape[i] for i in prefix_axes)
        inv = {axis: k for k, axis in enumerate(prefix_axes)}

        def gen_planes():
            iterator_shape = prefix_shape if prefix_shape else (1,)
            for idx in np.ndindex(iterator_shape):
                slicer = []
                for dim_i, ax in enumerate(axes):
                    if ax == "Y":
                        slicer.append(slice(None))
                    elif ax == "X":
                        slicer.append(slice(None))
                    else:
                        slicer.append(idx[inv[dim_i]] if prefix_shape else 0)
                plane = np.asarray(za[tuple(slicer)])
                yield _safe_imresize_2d(plane, new_x, new_y)

        source_resolution, source_mpp = _scale_resolution_and_mpp(
            source_resolution=source_resolution, source_mpp=source_mpp, pixel_scale=downsample
        )
        metadata = {"axes": axes, "Name": output_path.stem}
        if source_mpp:
            metadata.update({
                "PhysicalSizeX": float(source_mpp[0]), "PhysicalSizeY": float(source_mpp[1]),
                "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
            })
        resolution = None
        resolutionunit = None
        if source_resolution:
            try:
                xres, yres, unit = source_resolution
                resolution = (float(xres), float(yres))
                resolutionunit = unit
            except Exception:
                pass
        compression_kwargs = {"compression": "deflate", "predictor": True} if lossless else {}
        tifffile.imwrite(str(output_path), gen_planes(), shape=out_shape, dtype=za.dtype,
                         bigtiff=True, ome=bool(write_ome), metadata=metadata,
                         photometric="minisblack", software=f"{APP_NAME} v{APP_VERSION}",
                         resolution=resolution, resolutionunit=resolutionunit, **compression_kwargs)


def downsample_image_file(input_path: Path, output_dir: Optional[Path], downsample: float,
                          output_kind: str = "OME-TIFF (.ome.tif)", preserve_raw: bool = True,
                          lossless: bool = True, overwrite: bool = False) -> Path:
    """Downsample one supported image file."""
    input_path = Path(input_path)
    output_dir = input_path.parent if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds_txt = f"DS{int(downsample)}" if float(downsample).is_integer() else f"DS{downsample:g}"
    if "OME" in output_kind.upper():
        ext, output_format, write_ome = ".ome.tif", "tiff", True
    elif "JPEG" in output_kind.upper() or "JPG" in output_kind.upper():
        ext, output_format, write_ome = ".jpg", "jpeg", False
    else:
        ext, output_format, write_ome = ".tif", "tiff", False
    out_path = output_dir / f"{input_path.stem}_downsample_{ds_txt}{ext}"
    if out_path.exists() and not overwrite:
        return out_path
    if preserve_raw and output_format == "tiff" and _has_ext(input_path.name, TIFF_EXTENSIONS):
        # Preserve raw data for scientific multichannel TIFF/OME-TIFF.
        # If the file is a true RGB image stored with a Samples axis, use the visual RGB path instead.
        try:
            with tifffile.TiffFile(str(input_path)) as _tf_check:
                _axes_check = getattr(_tf_check.series[0], "axes", "") or ""
            _is_rgb_samples = "S" in _axes_check
        except Exception:
            _is_rgb_samples = False
        if not _is_rgb_samples:
            b = ImageBackend().load(str(input_path))
            try:
                _downsample_tiff_raw_lowmem(input_path, out_path, downsample, write_ome=write_ome,
                                            lossless=lossless, source_resolution=b.source_resolution,
                                            source_mpp=b.source_mpp)
            finally:
                b.close()
            return out_path

    b = ImageBackend().load(str(input_path))
    try:
        if b.reader == "openslide":
            slide = b._get_openslide()
            best_level = slide.get_best_level_for_downsample(float(downsample))
            level_w, level_h = slide.level_dimensions[best_level]
            img = slide.read_region((0, 0), best_level, (int(level_w), int(level_h))).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
            actual_ds = float(slide.level_downsamples[best_level])
            if abs(actual_ds - float(downsample)) / max(float(downsample), 1.0) > 0.05:
                from PIL import Image
                full_w, full_h = b.slide_dims
                new_w = max(1, int(round(full_w / float(downsample))))
                new_h = max(1, int(round(full_h / float(downsample))))
                arr = np.asarray(Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)
            save_rgb_image(out_path, arr, output_format=output_format, write_ome=write_ome, lossless=lossless,
                           source_resolution=b.source_resolution, source_mpp=b.source_mpp,
                           image_name=out_path.stem, pixel_scale=downsample)
            return out_path
        if b.reader == "pil":
            from PIL import Image
            im = Image.open(str(input_path)).convert("RGB")
            new_w = max(1, int(round(im.width / float(downsample))))
            new_h = max(1, int(round(im.height / float(downsample))))
            arr = np.asarray(im.resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)
            save_rgb_image(out_path, arr, output_format=output_format, write_ome=False, lossless=lossless,
                           image_name=out_path.stem)
            return out_path
        raw, axes, _ = b.crop_raw(0, 0, b.slide_dims[0], b.slide_dims[1])
        raw, axes = _resize_spatial_array(raw, axes, downsample)
        rgb = _array_to_rgb_preview(raw, axes)
        save_rgb_image(out_path, rgb, output_format=output_format, write_ome=False, lossless=lossless,
                       source_resolution=b.source_resolution, source_mpp=b.source_mpp,
                       image_name=out_path.stem, pixel_scale=downsample)
        return out_path
    finally:
        b.close()


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

def _clean_annotation_display_name(value: Any) -> str:
    """Make GeoJSON feature ids/classes readable in the annotation menu.

    Some annotation exporters encode the real class in Feature.id using helper
    prefixes, for example:
        parent_Neoplastic_Tissue
        merged_Neoplastic_Tissue
        child_Stroma
    The menu should show the biological/pathology class, not the exporter role.
    """
    text = str(value or "annotation").strip()
    if not text:
        return "annotation"

    # Remove repeated exporter/helper prefixes while preserving the real class.
    # Keep this list conservative: these words describe annotation grouping/role,
    # not biological classes. The user's ArtidisNet-like files use parent_* and
    # merged_* ids; QuPath-like files may use annotation/object prefixes.
    removable_prefixes = (
        "merged", "merge", "parent", "child", "annotation", "annotations",
        "object", "objects", "feature", "roi", "polygon", "multipolygon"
    )
    previous = None
    while previous != text:
        previous = text
        for prefix in removable_prefixes:
            pattern = rf"^{prefix}[_:\-\s]+(.+)$"
            m = re.match(pattern, text, flags=re.I)
            if m:
                text = m.group(1).strip()
                break

    text = re.sub(r"[_\s]+", " ", text).strip()
    return text or "annotation"


def _is_generic_geojson_name(value: Any) -> bool:
    """Return True for metadata words that are not useful class names."""
    if value in (None, ""):
        return True
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text in {
        "annotation", "annotations", "object", "objects", "feature", "features",
        "detection", "detections", "polygon", "multipolygon", "geometry",
        "geojson", "pathobject", "path object"
    }


def _geojson_feature_class_name(feature: Dict[str, Any], properties: Dict[str, Any]) -> str:
    """Return a readable class/name from QuPath, ArtidisNet, or generic GeoJSON.

    Priority is important:
    1. QuPath classification.name / class_name / label.
    2. Explicit class/name/label properties.
    3. Top-level Feature id/name/label, because ArtidisNet-like exports often
       store class names only there, e.g. id="merged_Neoplastic_Tissue".
    4. Generic objectType/type metadata only as a last resort.

    This avoids classifying everything as simply "annotation" when a file has
    properties.objectType="annotation" but the real class is in Feature.id.
    """
    props = properties or {}

    # QuPath classification block is the most reliable source when present.
    classification = props.get("classification")
    if isinstance(classification, dict):
        for key in ("name", "class_name", "className", "label", "class"):
            value = classification.get(key)
            if not _is_generic_geojson_name(value):
                return _clean_annotation_display_name(value)
    elif not _is_generic_geojson_name(classification):
        return _clean_annotation_display_name(classification)

    # Explicit class-like properties. Avoid generic object/type fields here.
    for key in (
        "class_name", "className", "class", "label", "name", "annotation_name",
        "annotationName", "category", "category_name", "categoryName"
    ):
        value = props.get(key)
        if not _is_generic_geojson_name(value):
            return _clean_annotation_display_name(value)

    # ArtidisNet / parent-child / merged style: the class can be top-level id.
    if isinstance(feature, dict):
        for key in ("id", "name", "label"):
            value = feature.get(key)
            if not _is_generic_geojson_name(value):
                return _clean_annotation_display_name(value)

    # Last resort: only use object/type metadata if it is not a generic word.
    for key in ("object_type", "objectType", "type"):
        value = props.get(key)
        if not _is_generic_geojson_name(value):
            return _clean_annotation_display_name(value)

    return "annotation"


# Backward compatible name for any older internal calls.
def _geojson_properties_class_name(properties: Dict[str, Any]) -> str:
    return _geojson_feature_class_name({}, properties or {})


def _qcolor_from_any(value: Any) -> Optional[QColor]:
    """Convert many GeoJSON/QuPath colour encodings to QColor.

    Supported examples:
    - "#RRGGBB", "RRGGBB", "#AARRGGBB"
    - signed/unsigned QuPath colorRGB integers
    - [R, G, B], [A, R, G, B], or 0..1 float triples
    - {"r": 255, "g": 0, "b": 0} / {"red": ..., "green": ..., "blue": ...}
    """
    try:
        if value is None:
            return None
        if isinstance(value, QColor):
            return QColor(value) if value.isValid() else None
        if isinstance(value, dict):
            low = {str(k).lower(): v for k, v in value.items()}
            r = low.get("r", low.get("red"))
            g = low.get("g", low.get("green"))
            b = low.get("b", low.get("blue"))
            if r is not None and g is not None and b is not None:
                vals = [float(r), float(g), float(b)]
                if max(vals) <= 1.0:
                    vals = [v * 255.0 for v in vals]
                return QColor(int(round(vals[0])), int(round(vals[1])), int(round(vals[2])))
            return None
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            vals = [float(v) for v in value]
            if len(vals) >= 4:
                # Most JSON colour arrays are RGBA, so preserve the first three
                # values. QuPath normally uses colorRGB integer rather than ARGB
                # arrays, so this default avoids converting red RGBA to blue.
                vals = vals[:3]
            else:
                vals = vals[:3]
            if max(vals) <= 1.0:
                vals = [v * 255.0 for v in vals]
            return QColor(int(round(vals[0])), int(round(vals[1])), int(round(vals[2])))
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            # CSS-like rgb(r,g,b)
            m = re.match(r"rgba?\s*\(([^)]+)\)", text, flags=re.I)
            if m:
                parts = [p.strip() for p in m.group(1).split(",")]
                if len(parts) >= 3:
                    return _qcolor_from_any([float(parts[0]), float(parts[1]), float(parts[2])])
            if not text.startswith("#") and re.fullmatch(r"[0-9A-Fa-f]{6,8}", text):
                text = "#" + text[-6:]
            qc = QColor(text)
            if qc.isValid():
                return qc
            if text.lstrip("+-").isdigit():
                value = int(text)
            else:
                return None
        if isinstance(value, (int, float)):
            # QuPath classification.colorRGB is often a signed Java int. Treat as
            # ARGB/RGB and keep the lower 24 bits.
            intval = int(value)
            if intval < 0:
                intval &= 0xFFFFFFFF
            r = (intval >> 16) & 255
            g = (intval >> 8) & 255
            b = intval & 255
            return QColor(r, g, b)
    except Exception:
        return None
    return None


def _deterministic_qcolor_for_text(text: str) -> QColor:
    """Stable fallback colour so classes without explicit colours are not all red."""
    palette = [
        QColor(230, 25, 75), QColor(60, 180, 75), QColor(0, 130, 200),
        QColor(245, 130, 48), QColor(145, 30, 180), QColor(70, 240, 240),
        QColor(240, 50, 230), QColor(210, 245, 60), QColor(250, 190, 190),
        QColor(0, 128, 128), QColor(230, 190, 255), QColor(170, 110, 40),
    ]
    s = str(text or "annotation")
    idx = sum((i + 1) * ord(ch) for i, ch in enumerate(s)) % len(palette)
    return QColor(palette[idx])


def _qcolor_from_geojson_feature(feature: Dict[str, Any], properties: Dict[str, Any],
                                 default: QColor = QColor(255, 0, 0), class_name: str = "annotation") -> QColor:
    """Extract annotation colour from QuPath / ArtidisNet / generic GeoJSON."""
    props = properties or {}
    candidates = []

    # Direct properties used by generic annotation tools.
    for key in (
        "class_color_hex", "classColor", "class_color", "color", "colour",
        "stroke", "fill", "strokeColor", "fillColor", "lineColor", "borderColor",
        "colorRGB", "rgb"
    ):
        if key in props:
            candidates.append(props.get(key))

    # QuPath classification block.
    classification = props.get("classification")
    if isinstance(classification, dict):
        for key in (
            "colorRGB", "color", "colour", "class_color_hex", "classColor",
            "stroke", "fill", "strokeColor", "fillColor", "rgb"
        ):
            if key in classification:
                candidates.append(classification.get(key))

    # Some exporters place colour fields at the Feature level.
    if isinstance(feature, dict):
        for key in ("color", "colour", "stroke", "fill", "strokeColor", "fillColor", "colorRGB", "rgb"):
            if key in feature:
                candidates.append(feature.get(key))

    for value in candidates:
        qc = _qcolor_from_any(value)
        if qc is not None and qc.isValid():
            return qc

    # If the file has class names but no explicit colour, use a stable per-class
    # fallback instead of one red colour for everything.
    if class_name and class_name != "annotation":
        return _deterministic_qcolor_for_text(class_name)
    return QColor(default)


def _qcolor_from_geojson_properties(properties: Dict[str, Any], default: QColor = QColor(255, 0, 0)) -> QColor:
    # Backward-compatible wrapper.
    class_name = _geojson_properties_class_name(properties or {})
    return _qcolor_from_geojson_feature({}, properties or {}, default=default, class_name=class_name)


def _clean_geojson_ring(raw_ring: Any) -> List[Tuple[float, float]]:
    ring = []
    if not isinstance(raw_ring, (list, tuple)):
        return ring
    for pt in raw_ring:
        try:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x = float(pt[0])
                y = float(pt[1])
                if np.isfinite(x) and np.isfinite(y):
                    ring.append((x, y))
        except Exception:
            continue
    return ring


def _geojson_geometry_to_rings(geometry: Dict[str, Any]) -> List[List[Tuple[float, float]]]:
    """Return polygon rings from Polygon, MultiPolygon, or GeometryCollection GeoJSON."""
    if not isinstance(geometry, dict):
        return []
    gtype = str(geometry.get("type", "")).lower()
    coords = geometry.get("coordinates")
    rings: List[List[Tuple[float, float]]] = []

    if gtype == "polygon":
        for raw_ring in coords or []:
            ring = _clean_geojson_ring(raw_ring)
            if len(ring) >= 2:
                rings.append(ring)
    elif gtype == "multipolygon":
        for polygon in coords or []:
            for raw_ring in polygon or []:
                ring = _clean_geojson_ring(raw_ring)
                if len(ring) >= 2:
                    rings.append(ring)
    elif gtype == "linestring":
        ring = _clean_geojson_ring(coords)
        if len(ring) >= 2:
            rings.append(ring)
    elif gtype == "multilinestring":
        for raw_ring in coords or []:
            ring = _clean_geojson_ring(raw_ring)
            if len(ring) >= 2:
                rings.append(ring)
    elif gtype == "geometrycollection":
        for child in geometry.get("geometries", []) or []:
            rings.extend(_geojson_geometry_to_rings(child))
    return rings


def load_geojson_annotations(path: str) -> List[Dict[str, Any]]:
    """Load QuPath / ArtidisNet / generic GeoJSON polygon annotations.

    Coordinates are assumed to be in image pixel coordinates, which is the usual
    convention for QuPath GeoJSON exports and ArtidisNet-style image annotation files.
    """
    path = str(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = []
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        features = data.get("features", []) or []
    elif isinstance(data, dict) and data.get("type") == "Feature":
        features = [data]
    elif isinstance(data, dict) and "geometry" in data:
        features = [{"type": "Feature", "geometry": data.get("geometry"), "properties": data.get("properties", {})}]
    elif isinstance(data, dict) and "coordinates" in data and "type" in data:
        features = [{"type": "Feature", "geometry": data, "properties": {}}]
    elif isinstance(data, list):
        features = data

    annotations: List[Dict[str, Any]] = []
    for i, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties", {}) or {}
        geom = feature.get("geometry") if "geometry" in feature else feature
        rings = _geojson_geometry_to_rings(geom)
        if not rings:
            continue
        class_name = _geojson_feature_class_name(feature, props)
        annotations.append({
            "id": feature.get("id", props.get("id", props.get("object_id", i))),
            "class_name": class_name,
            "rings": rings,
            "color": _qcolor_from_geojson_feature(feature, props, class_name=class_name),
            "properties": props,
            "feature_id": feature.get("id", None),
        })
    return annotations



def _qcolor_to_qupath_color_rgb(color: Any) -> int:
    """Return QuPath-style signed colorRGB integer from QColor-like input."""
    q = QColor(color if color is not None else QColor(255, 0, 0))
    value = (255 << 24) | (q.red() << 16) | (q.green() << 8) | q.blue()
    if value >= 2 ** 31:
        value -= 2 ** 32
    return int(value)


def save_geojson_annotations(path: str, annotations: List[Dict[str, Any]], image_path: Optional[str] = None) -> int:
    """Save editable/loaded annotations as a GeoJSON FeatureCollection.

    Coordinates are written in full-resolution image pixel coordinates. Rings are
    explicitly closed in the exported GeoJSON, which is what QuPath expects for
    polygon annotations.
    """
    features = []
    for i, ann in enumerate(annotations or []):
        cls = str(ann.get("class_name", "annotation") or "annotation")
        color = QColor(ann.get("color", _deterministic_qcolor_for_text(cls)))
        rings_out = []
        for ring in ann.get("rings", []) or []:
            clean = []
            for pt in ring or []:
                try:
                    x, y = float(pt[0]), float(pt[1])
                    if np.isfinite(x) and np.isfinite(y):
                        clean.append([x, y])
                except Exception:
                    continue
            if len(clean) >= 3:
                if clean[0] != clean[-1]:
                    clean.append(list(clean[0]))
                rings_out.append(clean)
        if not rings_out:
            continue
        if len(rings_out) == 1:
            geometry = {"type": "Polygon", "coordinates": [rings_out[0]]}
        else:
            geometry = {"type": "MultiPolygon", "coordinates": [[r] for r in rings_out]}
        props = dict(ann.get("properties", {}) or {})
        props.update({
            "objectType": "annotation",
            "name": cls,
            "classification": {
                "name": cls,
                "colorRGB": _qcolor_to_qupath_color_rgb(color),
            },
            "source": props.get("source", "TiffCropper manual annotation"),
        })
        feature_id = ann.get("feature_id", ann.get("id", f"annotation_{i + 1:04d}"))
        features.append({
            "type": "Feature",
            "id": str(feature_id),
            "geometry": geometry,
            "properties": props,
        })
    data = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "software": f"{APP_NAME} v{APP_VERSION}",
            "coordinate_space": "full-resolution image pixels",
        },
    }
    if image_path:
        data["properties"]["image"] = str(image_path)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(features)


def _apply_display_adjustments(rgb: np.ndarray, brightness: int = 0, negative: bool = False) -> np.ndarray:
    """Apply display-only brightness and negative mode. Does not modify saved scientific data."""
    arr = _to_uint8_rgb(rgb).astype(np.int16, copy=False)
    try:
        brightness = int(brightness)
    except Exception:
        brightness = 0
    if brightness != 0:
        arr = arr + int(round(brightness * 2.55))
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if negative:
        arr = 255 - arr
    return np.ascontiguousarray(arr)


class CropSelectionLabel(QLabel):
    """Interactive crop preview with zoom-region coordinates.

    Left-drag selects the crop rectangle.
    Mouse wheel zooms in/out around the cursor if a zoom callback is connected.
    Right-drag or middle-drag pans the preview using natural grab-and-move behavior.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self._rgb_original = None
        self._pixmap_original = None
        self._thumb_w = None
        self._thumb_h = None
        self._full_w = None
        self._full_h = None
        self._roi_full = None
        self._dragging = False
        self._panning = False
        self._start = QPoint()
        self._end = QPoint()
        self._selection_full = None
        self._drag_rect_widget = None
        self.selection_callback = None
        self.center_callback = None
        self.zoom_callback = None
        self.pan_callback = None
        self._pan_start_full = None
        self._negative = False
        self._brightness = 0
        self.setToolTip(
            "Left-drag: select crop rectangle. Mouse wheel: zoom. "
            "Right/middle-drag: pan/recenter when zoomed."
        )
        self.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")

    def set_image(self, rgb, full_w, full_h, callback=None, roi_full=None,
                  center_callback=None, zoom_callback=None, pan_callback=None):
        rgb = _to_uint8_rgb(rgb)
        self._rgb_original = rgb
        self._thumb_h, self._thumb_w = rgb.shape[:2]
        self._full_w = int(full_w)
        self._full_h = int(full_h)
        if roi_full is None:
            roi_full = (0, 0, self._full_w, self._full_h)
        self._roi_full = tuple(int(round(float(v))) for v in roi_full)
        self.selection_callback = callback
        self.center_callback = center_callback
        self.zoom_callback = zoom_callback
        self.pan_callback = pan_callback
        self._drag_rect_widget = None
        self._update_pixmap_from_display_settings()
        self.update()

    def _update_pixmap_from_display_settings(self):
        if self._rgb_original is None:
            self._pixmap_original = None
            return
        adjusted = _apply_display_adjustments(
            self._rgb_original,
            brightness=self._brightness,
            negative=self._negative,
        )
        self._pixmap_original = _numpy_rgb_to_qpixmap(adjusted)

    def set_negative(self, enabled: bool):
        self._negative = bool(enabled)
        self._update_pixmap_from_display_settings()
        self.update()

    def set_brightness(self, value: int):
        self._brightness = int(value)
        self._update_pixmap_from_display_settings()
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

    def _roi(self):
        if self._roi_full is not None:
            rx, ry, rw, rh = self._roi_full
            return float(rx), float(ry), max(1.0, float(rw)), max(1.0, float(rh))
        return 0.0, 0.0, max(1.0, float(self._full_w or 1)), max(1.0, float(self._full_h or 1))

    def _clamp_point_to_display_rect(self, p: QPoint):
        r = self._display_rect()
        x = max(r.left(), min(p.x(), r.right()))
        y = max(r.top(), min(p.y(), r.bottom()))
        return QPoint(x, y)

    def _widget_to_full_xy(self, p: QPoint):
        if not self.has_image():
            return None
        disp = self._display_rect()
        if disp.width() <= 0 or disp.height() <= 0:
            return None
        p = self._clamp_point_to_display_rect(p)
        rx, ry, rw, rh = self._roi()
        fx = (p.x() - disp.left()) / max(1.0, float(disp.width()))
        fy = (p.y() - disp.top()) / max(1.0, float(disp.height()))
        x = rx + fx * rw
        y = ry + fy * rh
        x = max(0.0, min(x, float(self._full_w or x)))
        y = max(0.0, min(y, float(self._full_h or y)))
        return x, y

    def _widget_to_roi_fraction(self, p: QPoint):
        disp = self._display_rect()
        if disp.width() <= 0 or disp.height() <= 0:
            return None
        p = self._clamp_point_to_display_rect(p)
        fx = (p.x() - disp.left()) / max(1.0, float(disp.width()))
        fy = (p.y() - disp.top()) / max(1.0, float(disp.height()))
        return max(0.0, min(1.0, float(fx))), max(0.0, min(1.0, float(fy)))

    def _full_to_widget_xy(self, x: float, y: float) -> QPointF:
        disp = self._display_rect()
        rx, ry, rw, rh = self._roi()
        px = disp.left() + ((float(x) - rx) / rw) * disp.width()
        py = disp.top() + ((float(y) - ry) / rh) * disp.height()
        return QPointF(float(px), float(py))

    def _widget_rect_to_full_coords(self, rect: QRect):
        if not self.has_image():
            return None
        disp = self._display_rect()
        if disp.width() <= 0 or disp.height() <= 0:
            return None
        inter = rect.normalized().intersected(disp)
        if inter.width() <= 1 or inter.height() <= 1:
            return None

        p0 = self._widget_to_full_xy(inter.topLeft())
        p1 = self._widget_to_full_xy(inter.bottomRight())
        if p0 is None or p1 is None:
            return None
        x0_full, y0_full = p0
        x1_full, y1_full = p1

        x0_full = max(0, min(int(round(x0_full)), self._full_w - 1))
        y0_full = max(0, min(int(round(y0_full)), self._full_h - 1))
        x1_full = max(1, min(int(round(x1_full)), self._full_w))
        y1_full = max(1, min(int(round(y1_full)), self._full_h))

        x = min(x0_full, x1_full)
        y = min(y0_full, y1_full)
        w = abs(x1_full - x0_full)
        h = abs(y1_full - y0_full)
        return x, y, max(1, w), max(1, h)

    def set_selection_from_full_coords(self, x, y, w, h):
        if not self.has_image():
            return
        x = max(0, min(int(x), max(0, self._full_w - 1)))
        y = max(0, min(int(y), max(0, self._full_h - 1)))
        w = max(1, min(int(w), self._full_w - x))
        h = max(1, min(int(h), self._full_h - y))
        self._selection_full = (x, y, w, h)
        self._drag_rect_widget = None
        self.update()

    def _selection_widget_rect_for_current_roi(self):
        if not self._selection_full or not self.has_image():
            return None
        sx, sy, sw, sh = self._selection_full
        rx, ry, rw, rh = self._roi()
        ix0 = max(float(sx), rx)
        iy0 = max(float(sy), ry)
        ix1 = min(float(sx + sw), rx + rw)
        iy1 = min(float(sy + sh), ry + rh)
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        p0 = self._full_to_widget_xy(ix0, iy0)
        p1 = self._full_to_widget_xy(ix1, iy1)
        return QRect(
            int(round(p0.x())),
            int(round(p0.y())),
            int(round(p1.x() - p0.x())),
            int(round(p1.y() - p0.y()))
        ).normalized().intersected(self._display_rect())

    def wheelEvent(self, event):
        if self.zoom_callback is None or not self.has_image():
            super().wheelEvent(event)
            return
        xy = self._pos_to_full_xy(event.pos())
        if xy is None:
            return
        delta = event.angleDelta().y()
        factor = 1.25 if delta > 0 else 0.8
        self.zoom_callback(float(factor), (float(xy[0]), float(xy[1])))
        event.accept()

    def mousePressEvent(self, event):
        if not self.has_image() or not self._display_rect().contains(event.pos()):
            return
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._start = self._clamp_point_to_display_rect(event.pos())
            self._end = self._start
            self._drag_rect_widget = QRect(self._start, self._end)
            self.update()
        elif event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._panning = True
            self._start = self._clamp_point_to_display_rect(event.pos())
            self._end = self._start
            self._pan_start_full = self._widget_to_full_xy(self._start)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._dragging and self.has_image():
            self._end = self._clamp_point_to_display_rect(event.pos())
            self._drag_rect_widget = QRect(self._start, self._end).normalized()
            self.update()
        elif self._panning and self.has_image():
            self._end = self._clamp_point_to_display_rect(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self._end = self._clamp_point_to_display_rect(event.pos())
            rect = QRect(self._start, self._end).normalized()
            coords = self._widget_rect_to_full_coords(rect)
            if coords is not None:
                x, y, w, h = coords
                self._selection_full = (x, y, w, h)
                if self.selection_callback is not None:
                    self.selection_callback(x, y, w, h)
            self._drag_rect_widget = None
            self.update()
        elif event.button() in (Qt.RightButton, Qt.MiddleButton) and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            frac = self._widget_to_roi_fraction(event.pos())
            if self.pan_callback is not None and self._pan_start_full is not None and frac is not None:
                self.pan_callback((float(self._pan_start_full[0]), float(self._pan_start_full[1])), frac)
            elif self.center_callback is not None:
                xy = self._pos_to_full_xy(event.pos())
                if xy is not None:
                    self.center_callback(float(xy[0]), float(xy[1]))
            self._pan_start_full = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))
        if self._pixmap_original is not None:
            disp = self._display_rect()
            scaled = self._pixmap_original.scaled(disp.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(disp.topLeft(), scaled)
        rect_to_draw = self._drag_rect_widget if self._drag_rect_widget is not None else self._selection_widget_rect_for_current_roi()
        if rect_to_draw is not None:
            rect_to_draw = rect_to_draw.normalized().intersected(self._display_rect())
            if rect_to_draw.width() > 0 and rect_to_draw.height() > 0:
                pen = QPen(QColor(255, 0, 0), 2)
                painter.setPen(pen)
                painter.drawRect(rect_to_draw)
                painter.fillRect(rect_to_draw, QColor(255, 0, 0, 35))
        painter.end()


class FixedSquarePreviewLabel(QLabel):
    """Thumbnail label with a fixed-size draggable square ROI."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self._pixmap_original = None
        self._thumb_w = None
        self._thumb_h = None
        self._full_w = None
        self._full_h = None
        self.square_size_full = 1024
        self._center_full = None
        self._dragging = False
        self.selection_callback = None
        self.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")

    def set_image(self, rgb, full_w, full_h, square_size_full=1024, callback=None):
        rgb = _to_uint8_rgb(rgb)
        self._thumb_h, self._thumb_w = rgb.shape[:2]
        self._full_w = int(full_w)
        self._full_h = int(full_h)
        self.square_size_full = int(square_size_full)
        self.selection_callback = callback
        self._pixmap_original = _numpy_rgb_to_qpixmap(rgb)
        if self._center_full is None:
            self._center_full = (self._full_w // 2, self._full_h // 2)
        self._emit_selection()
        self.update()

    def has_image(self):
        return self._pixmap_original is not None and self._full_w and self._full_h

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

    def _widget_to_full(self, p: QPoint):
        disp = self._display_rect()
        x = max(disp.left(), min(p.x(), disp.right()))
        y = max(disp.top(), min(p.y(), disp.bottom()))
        fx = int(round((x - disp.left()) / max(1, disp.width()) * self._full_w))
        fy = int(round((y - disp.top()) / max(1, disp.height()) * self._full_h))
        return max(0, min(fx, self._full_w - 1)), max(0, min(fy, self._full_h - 1))

    def _selection_full(self):
        if not self.has_image():
            return None
        cx, cy = self._center_full or (self._full_w // 2, self._full_h // 2)
        s = max(1, min(int(self.square_size_full), self._full_w, self._full_h))
        x = max(0, min(int(cx - s // 2), self._full_w - s))
        y = max(0, min(int(cy - s // 2), self._full_h - s))
        return x, y, s, s

    def _selection_widget_rect(self):
        sel = self._selection_full()
        if sel is None:
            return QRect(0, 0, 0, 0)
        x, y, w, h = sel
        disp = self._display_rect()
        px = disp.left() + x / self._full_w * disp.width()
        py = disp.top() + y / self._full_h * disp.height()
        pw = w / self._full_w * disp.width()
        ph = h / self._full_h * disp.height()
        return QRect(int(round(px)), int(round(py)), int(round(pw)), int(round(ph)))

    def _emit_selection(self):
        sel = self._selection_full()
        if sel and self.selection_callback:
            self.selection_callback(*sel)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.has_image() and self._display_rect().contains(event.pos()):
            self._dragging = True
            self._center_full = self._widget_to_full(event.pos())
            self._emit_selection()
            self.update()

    def mouseMoveEvent(self, event):
        if self._dragging and self.has_image():
            self._center_full = self._widget_to_full(event.pos())
            self._emit_selection()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self._center_full = self._widget_to_full(event.pos())
            self._emit_selection()
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))
        if self._pixmap_original is not None:
            disp = self._display_rect()
            scaled = self._pixmap_original.scaled(disp.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(disp.topLeft(), scaled)
            rect = self._selection_widget_rect()
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawRect(rect)
            painter.fillRect(rect, QColor(255, 0, 0, 35))
        painter.end()


class ZoomRegionPreviewLabel(QLabel):
    """Preview label that supports panning, rectangle zoom, GeoJSON overlays, and tile capture overlay.

    Normal mode:
        right-drag pans/moves the view. Mouse wheel zooms in/out.
    Rectangle-zoom mode:
        left-drag a rectangle; on release, the app zooms to that full-resolution ROI.
    Tile mode:
        left-drag places the square tile capture box on the desired image position.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setMinimumSize(420, 300)
        self.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")
        self._pixmap_original = None
        self._roi_full = None
        self._full_dims = None
        self._dragging = False
        self._selecting = False
        self._rect_zoom_enabled = False
        self._tile_mode_enabled = False
        self._sel_start = QPoint()
        self._sel_end = QPoint()
        self.center_callback = None
        self.rectangle_callback = None
        self.tile_callback = None
        self.zoom_callback = None
        self._pan_start_pos = QPoint()
        self._pan_start_center = None

        self._annotations = []
        self._show_annotations = True
        self._annotation_color = QColor(255, 0, 0)
        self._annotation_opacity = 110
        self._annotation_fill = True
        self._annotation_boundary_width = 2
        self._annotation_class_styles = {}

        # Lightweight editable annotation state. Coordinates are stored in
        # full-resolution image pixels, matching QuPath-style GeoJSON exports.
        self._annotation_draw_mode = "none"
        self._drawing_annotation = False
        self._active_annotation_ring = []
        self._annotation_created_callback = None
        self._annotation_preview_callback = None

        self._tile_center_full = None
        self._tile_size_full = 1024
        self._show_tile = False
        self.setToolTip(
            "Mouse wheel: zoom in/out. Right-drag: pan/move the view. "
            "In annotation mode, left-click/drag draws polygon, freehand, or rectangle objects."
        )

    def set_preview(self, rgb, roi_full=None, full_dims=None, center_callback=None, rectangle_callback=None, zoom_callback=None):
        self._pixmap_original = _numpy_rgb_to_qpixmap(_to_uint8_rgb(rgb))
        self._roi_full = roi_full
        self._full_dims = full_dims
        self.center_callback = center_callback
        if rectangle_callback is not None:
            self.rectangle_callback = rectangle_callback
        if zoom_callback is not None:
            self.zoom_callback = zoom_callback
        self.update()

    def set_annotations(self, annotations: Optional[List[Dict[str, Any]]]):
        self._annotations = list(annotations or [])
        self.update()

    def clear_annotations(self):
        self._annotations = []
        self.update()

    def set_annotations_visible(self, visible: bool):
        self._show_annotations = bool(visible)
        self.update()

    def set_annotation_style(self, color: Optional[QColor] = None, opacity: Optional[int] = None,
                             fill: Optional[bool] = None, boundary_width: Optional[int] = None):
        if color is not None:
            self._annotation_color = QColor(color)
        if opacity is not None:
            self._annotation_opacity = max(0, min(255, int(opacity)))
        if fill is not None:
            self._annotation_fill = bool(fill)
        if boundary_width is not None:
            self._annotation_boundary_width = max(1, int(boundary_width))
        self.update()

    def set_annotation_class_styles(self, class_styles: Optional[Dict[str, Dict[str, Any]]]):
        """Set per-class visibility and colour for GeoJSON overlays."""
        cleaned = {}
        for name, st in (class_styles or {}).items():
            cls = str(name or "annotation")
            cleaned[cls] = {
                "visible": bool(st.get("visible", True)),
                "color": QColor(st.get("color", self._annotation_color)),
            }
        self._annotation_class_styles = cleaned
        self.update()

    def set_annotation_draw_callbacks(self, created_callback=None, preview_callback=None):
        """Register callbacks used by the GUI annotation tools."""
        self._annotation_created_callback = created_callback
        self._annotation_preview_callback = preview_callback

    def set_annotation_draw_mode(self, mode: str = "none"):
        """Enable an editable annotation drawing mode.

        Supported modes are: none, polygon, freehand, rectangle.
        """
        mode = str(mode or "none").strip().lower()
        if mode not in {"none", "polygon", "freehand", "rectangle"}:
            mode = "none"
        if mode != self._annotation_draw_mode:
            self.cancel_active_annotation(emit=False)
        self._annotation_draw_mode = mode
        if mode != "none":
            self._tile_mode_enabled = False
            self._rect_zoom_enabled = False
        self.setCursor(Qt.CrossCursor if mode != "none" else Qt.ArrowCursor)
        self.update()

    def cancel_active_annotation(self, emit: bool = True):
        self._drawing_annotation = False
        self._active_annotation_ring = []
        self._selecting = False
        try:
            self.releaseMouse()
        except Exception:
            pass
        if emit and self._annotation_preview_callback:
            self._annotation_preview_callback([])
        self.update()

    def finish_polygon_annotation(self):
        if self._annotation_draw_mode != "polygon":
            return False
        return self._create_annotation_from_ring(self._active_annotation_ring, "polygon")

    def _notify_annotation_preview(self):
        if self._annotation_preview_callback:
            self._annotation_preview_callback(list(self._active_annotation_ring))

    def _append_active_point_from_pos(self, pos, min_dist_full: float = 2.0):
        xy = self._pos_to_full_xy(pos)
        if xy is None:
            return False
        x, y = float(xy[0]), float(xy[1])
        if self._active_annotation_ring:
            lx, ly = self._active_annotation_ring[-1]
            if math.hypot(float(x) - float(lx), float(y) - float(ly)) < float(min_dist_full):
                return False
        self._active_annotation_ring.append((x, y))
        self._notify_annotation_preview()
        self.update()
        return True

    def _create_annotation_from_ring(self, ring, source: str = "manual"):
        pts = []
        for pt in ring or []:
            try:
                x, y = float(pt[0]), float(pt[1])
                if np.isfinite(x) and np.isfinite(y):
                    pts.append((x, y))
            except Exception:
                continue
        if len(pts) < 3:
            self.cancel_active_annotation()
            return False
        # Remove duplicated closing point; export will close the ring explicitly.
        if len(pts) > 3 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-6:
            pts = pts[:-1]
        self._drawing_annotation = False
        self._active_annotation_ring = []
        try:
            self.releaseMouse()
        except Exception:
            pass
        if self._annotation_created_callback:
            self._annotation_created_callback(list(pts), source)
        if self._annotation_preview_callback:
            self._annotation_preview_callback([])
        self.update()
        return True

    def _emit_draw_rectangle_from_widget_rect(self, rect):
        if not self._roi_full:
            return False
        disp = self._display_rect()
        rect = rect.normalized().intersected(disp)
        if rect.width() < 3 or rect.height() < 3:
            self.cancel_active_annotation()
            return False
        p0 = self._pos_to_full_xy(rect.topLeft())
        p1 = self._pos_to_full_xy(rect.bottomRight())
        if p0 is None or p1 is None:
            self.cancel_active_annotation()
            return False
        x0, y0 = p0
        x1, y1 = p1
        x0, x1 = sorted([float(x0), float(x1)])
        y0, y1 = sorted([float(y0), float(y1)])
        ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return self._create_annotation_from_ring(ring, "rectangle")

    def _draw_active_annotation(self, painter: QPainter):
        if not self._active_annotation_ring or not self._roi_full:
            return
        disp = self._display_rect()
        if disp.width() <= 0 or disp.height() <= 0:
            return
        painter.save()
        painter.setClipRect(disp)
        painter.setPen(QPen(QColor(255, 210, 0, 230), max(2, int(self._annotation_boundary_width))))
        painter.setBrush(QBrush(QColor(255, 210, 0, 45)) if self._annotation_draw_mode in ("rectangle", "polygon") else Qt.NoBrush)
        path = QPainterPath()
        for i, (x, y) in enumerate(self._active_annotation_ring):
            pt = self._full_to_widget_xy(x, y)
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        if self._annotation_draw_mode == "rectangle" and len(self._active_annotation_ring) >= 4:
            path.closeSubpath()
        painter.drawPath(path)
        for x, y in self._active_annotation_ring:
            pt = self._full_to_widget_xy(x, y)
            painter.drawEllipse(pt, 3.0, 3.0)
        painter.restore()

    def set_tile_overlay(self, center_xy=None, size_full: int = 1024, visible: bool = False, tile_callback=None):
        self._tile_center_full = center_xy
        self._tile_size_full = max(1, int(size_full))
        self._show_tile = bool(visible)
        if tile_callback is not None:
            self.tile_callback = tile_callback
        self.update()

    def enable_tile_mode(self, enabled: bool = True):
        self._tile_mode_enabled = bool(enabled)
        if enabled:
            self.set_annotation_draw_mode("none")
            self.enable_rectangle_zoom(False)
        self.setCursor(Qt.CrossCursor if self._tile_mode_enabled else Qt.ArrowCursor)
        self.update()

    def enable_rectangle_zoom(self, enabled: bool = True):
        self._rect_zoom_enabled = bool(enabled)
        if enabled:
            self.set_annotation_draw_mode("none")
            self._tile_mode_enabled = False
        self._selecting = False
        self._dragging = False
        self._sel_start = QPoint()
        self._sel_end = QPoint()
        self.setCursor(Qt.CrossCursor if self._rect_zoom_enabled else Qt.ArrowCursor)
        self.update()

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

    def _clamp_to_display(self, pos):
        disp = self._display_rect()
        x = max(disp.left(), min(pos.x(), disp.right()))
        y = max(disp.top(), min(pos.y(), disp.bottom()))
        return QPoint(x, y)

    def _pos_to_full_xy(self, pos):
        if not self._roi_full:
            return None
        disp = self._display_rect()
        if disp.width() <= 0 or disp.height() <= 0:
            return None
        p = self._clamp_to_display(pos)
        rx, ry, rw, rh = self._roi_full
        fx = (p.x() - disp.left()) / max(1, disp.width())
        fy = (p.y() - disp.top()) / max(1, disp.height())
        return rx + fx * rw, ry + fy * rh

    def _full_to_widget_xy(self, x: float, y: float) -> QPointF:
        disp = self._display_rect()
        if not self._roi_full:
            return QPointF(float(disp.center().x()), float(disp.center().y()))
        rx, ry, rw, rh = self._roi_full
        px = disp.left() + ((float(x) - float(rx)) / max(1.0, float(rw))) * disp.width()
        py = disp.top() + ((float(y) - float(ry)) / max(1.0, float(rh))) * disp.height()
        return QPointF(float(px), float(py))

    def wheelEvent(self, event):
        if self.zoom_callback is None or self._pixmap_original is None:
            super().wheelEvent(event)
            return
        disp = self._display_rect()
        if not disp.contains(event.pos()):
            super().wheelEvent(event)
            return
        xy = self._pos_to_full_xy(event.pos())
        if xy is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        factor = 1.25 if delta > 0 else 0.8
        self.zoom_callback(float(factor), (float(xy[0]), float(xy[1])))
        event.accept()

    def _current_roi_center(self):
        if not self._roi_full:
            return None
        rx, ry, rw, rh = self._roi_full
        return float(rx) + float(rw) / 2.0, float(ry) + float(rh) / 2.0

    def _emit_center_from_pos(self, pos):
        if not self.center_callback:
            return
        xy = self._pos_to_full_xy(pos)
        if xy is None:
            return
        self.center_callback(float(xy[0]), float(xy[1]))

    def _emit_pan_from_pos(self, pos):
        """Natural grab-and-move pan: the grabbed image point follows the cursor."""
        if not self.center_callback or not self._roi_full or self._pan_start_center is None:
            return
        disp = self._display_rect()
        if disp.width() <= 0 or disp.height() <= 0:
            return
        rx, ry, rw, rh = self._roi_full
        dx = float(pos.x() - self._pan_start_pos.x())
        dy = float(pos.y() - self._pan_start_pos.y())
        cx = float(self._pan_start_center[0]) - dx / max(1.0, float(disp.width())) * float(rw)
        cy = float(self._pan_start_center[1]) - dy / max(1.0, float(disp.height())) * float(rh)
        if self._full_dims:
            full_w, full_h = self._full_dims
            cx = max(0.0, min(float(cx), float(full_w)))
            cy = max(0.0, min(float(cy), float(full_h)))
        self.center_callback(float(cx), float(cy))

    def _emit_tile_from_pos(self, pos):
        xy = self._pos_to_full_xy(pos)
        if xy is None:
            return
        self._tile_center_full = (float(xy[0]), float(xy[1]))
        if self.tile_callback:
            self.tile_callback(float(xy[0]), float(xy[1]))
        self.update()

    def _emit_rectangle_from_widget_rect(self, rect):
        if not self.rectangle_callback or not self._roi_full:
            return
        disp = self._display_rect()
        rect = rect.normalized().intersected(disp)
        if rect.width() < 8 or rect.height() < 8:
            return
        p0 = self._pos_to_full_xy(rect.topLeft())
        p1 = self._pos_to_full_xy(rect.bottomRight())
        if p0 is None or p1 is None:
            return
        x0, y0 = p0
        x1, y1 = p1
        x = max(0.0, min(x0, x1))
        y = max(0.0, min(y0, y1))
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        if w >= 2 and h >= 2:
            self.rectangle_callback(float(x), float(y), float(w), float(h))

    def mousePressEvent(self, event):
        if self._pixmap_original is None:
            super().mousePressEvent(event)
            return

        draw_mode = getattr(self, "_annotation_draw_mode", "none")
        if event.button() == Qt.LeftButton and draw_mode != "none":
            if draw_mode == "polygon":
                self._append_active_point_from_pos(event.pos(), min_dist_full=0.0)
                event.accept()
                return
            if draw_mode == "freehand":
                self._drawing_annotation = True
                self._active_annotation_ring = []
                self.grabMouse()
                self._append_active_point_from_pos(event.pos(), min_dist_full=0.0)
                event.accept()
                return
            if draw_mode == "rectangle":
                self._drawing_annotation = True
                self.grabMouse()
                self._sel_start = self._clamp_to_display(event.pos())
                self._sel_end = self._sel_start
                xy = self._pos_to_full_xy(self._sel_start)
                self._active_annotation_ring = [(float(xy[0]), float(xy[1]))] if xy else []
                self.update()
                event.accept()
                return

        if event.button() == Qt.LeftButton:
            if self._tile_mode_enabled:
                self._dragging = True
                self._drag_button = Qt.LeftButton
                self.grabMouse()
                self._emit_tile_from_pos(event.pos())
                event.accept()
                return
            if self._rect_zoom_enabled:
                self._selecting = True
                self.grabMouse()
                self._sel_start = self._clamp_to_display(event.pos())
                self._sel_end = self._sel_start
                self.update()
                event.accept()
                return
            # In normal Image Preview mode, left-drag is intentionally not used
            # for panning. Right-drag is used to move the view.
            event.accept()
            return

        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._dragging = True
            self._drag_button = event.button()
            self.grabMouse()
            self._pan_start_pos = self._clamp_to_display(event.pos())
            self._pan_start_center = self._current_roi_center()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        draw_mode = getattr(self, "_annotation_draw_mode", "none")
        if self._drawing_annotation and draw_mode == "freehand":
            # Increase minimum distance slightly with zoomed-out views to avoid
            # thousands of redundant points from a single hand stroke.
            min_dist = 2.0
            try:
                if self._roi_full and self._display_rect().width() > 0:
                    min_dist = max(2.0, float(self._roi_full[2]) / max(1.0, float(self._display_rect().width())) * 1.5)
            except Exception:
                pass
            self._append_active_point_from_pos(event.pos(), min_dist_full=min_dist)
            event.accept()
            return
        if self._drawing_annotation and draw_mode == "rectangle":
            self._sel_end = self._clamp_to_display(event.pos())
            p0 = self._pos_to_full_xy(self._sel_start)
            p1 = self._pos_to_full_xy(self._sel_end)
            if p0 is not None and p1 is not None:
                x0, y0 = p0
                x1, y1 = p1
                x0, x1 = sorted([float(x0), float(x1)])
                y0, y1 = sorted([float(y0), float(y1)])
                self._active_annotation_ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            self.update()
            event.accept()
            return
        if self._tile_mode_enabled and self._dragging and getattr(self, "_drag_button", None) == Qt.LeftButton:
            self._emit_tile_from_pos(event.pos())
            event.accept()
            return
        if self._rect_zoom_enabled and self._selecting:
            self._sel_end = self._clamp_to_display(event.pos())
            self.update()
            event.accept()
            return
        if self._dragging and getattr(self, "_drag_button", None) in (Qt.RightButton, Qt.MiddleButton):
            self._emit_pan_from_pos(event.pos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        draw_mode = getattr(self, "_annotation_draw_mode", "none")
        if event.button() == Qt.LeftButton and self._drawing_annotation and draw_mode == "freehand":
            self._append_active_point_from_pos(event.pos(), min_dist_full=0.0)
            self._create_annotation_from_ring(self._active_annotation_ring, "freehand")
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._drawing_annotation and draw_mode == "rectangle":
            self._sel_end = self._clamp_to_display(event.pos())
            rect = QRect(self._sel_start, self._sel_end).normalized()
            self._emit_draw_rectangle_from_widget_rect(rect)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._tile_mode_enabled and self._dragging:
            self._dragging = False
            self._drag_button = None
            try:
                self.releaseMouse()
            except Exception:
                pass
            self._emit_tile_from_pos(event.pos())
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._rect_zoom_enabled and self._selecting:
            self._selecting = False
            try:
                self.releaseMouse()
            except Exception:
                pass
            self._sel_end = self._clamp_to_display(event.pos())
            rect = QRect(self._sel_start, self._sel_end).normalized()
            self.enable_rectangle_zoom(False)
            self._emit_rectangle_from_widget_rect(rect)
            self.update()
            event.accept()
            return
        if event.button() in (Qt.RightButton, Qt.MiddleButton) and self._dragging:
            self._dragging = False
            self._drag_button = None
            try:
                self.releaseMouse()
            except Exception:
                pass
            self.setCursor(Qt.ArrowCursor)
            self._emit_pan_from_pos(event.pos())
            self._pan_start_center = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if getattr(self, "_annotation_draw_mode", "none") == "polygon" and event.button() == Qt.LeftButton:
            self.finish_polygon_annotation()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _draw_annotations(self, painter: QPainter):
        if not self._show_annotations or not self._annotations or not self._roi_full:
            return
        disp = self._display_rect()
        if disp.width() <= 0 or disp.height() <= 0:
            return
        painter.save()
        painter.setClipRect(disp)
        alpha = max(0, min(255, int(self._annotation_opacity)))
        for ann in self._annotations:
            cls = str(ann.get("class_name", "annotation") or "annotation")
            st = self._annotation_class_styles.get(cls, {})
            if not bool(st.get("visible", True)):
                continue
            base = QColor(st.get("color", ann.get("color", self._annotation_color)))
            pen_color = QColor(base.red(), base.green(), base.blue(), max(1, alpha))
            fill_color = QColor(base.red(), base.green(), base.blue(), alpha)
            painter.setPen(QPen(pen_color, max(1, int(self._annotation_boundary_width))))
            painter.setBrush(QBrush(fill_color) if self._annotation_fill else Qt.NoBrush)
            for ring in ann.get("rings", []) or []:
                if len(ring) < 2:
                    continue
                path = QPainterPath()
                first = True
                for x, y in ring:
                    pt = self._full_to_widget_xy(x, y)
                    if first:
                        path.moveTo(pt)
                        first = False
                    else:
                        path.lineTo(pt)
                if len(ring) >= 3:
                    path.closeSubpath()
                painter.drawPath(path)
        painter.restore()

    def _draw_tile_overlay(self, painter: QPainter):
        if not self._show_tile or not self._tile_center_full or not self._roi_full:
            return
        cx, cy = self._tile_center_full
        s = float(self._tile_size_full)
        x0 = float(cx) - s / 2.0
        y0 = float(cy) - s / 2.0
        x1 = x0 + s
        y1 = y0 + s
        p0 = self._full_to_widget_xy(x0, y0)
        p1 = self._full_to_widget_xy(x1, y1)
        rect = QRect(int(round(p0.x())), int(round(p0.y())), int(round(p1.x() - p0.x())), int(round(p1.y() - p0.y()))).normalized()
        rect = rect.intersected(self._display_rect())
        if rect.width() <= 0 or rect.height() <= 0:
            return
        painter.save()
        painter.setClipRect(self._display_rect())
        painter.setPen(QPen(QColor(20, 120, 255), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        painter.fillRect(rect, QColor(20, 120, 255, 25))
        painter.restore()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))
        if self._pixmap_original is not None:
            disp = self._display_rect()
            scaled = self._pixmap_original.scaled(disp.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(disp.topLeft(), scaled)
            self._draw_annotations(painter)
            self._draw_active_annotation(painter)
            self._draw_tile_overlay(painter)
        if self._rect_zoom_enabled and (self._selecting or not self._sel_start.isNull()):
            rect = QRect(self._sel_start, self._sel_end).normalized().intersected(self._display_rect())
            if rect.width() > 0 and rect.height() > 0:
                painter.setPen(QPen(QColor(255, 80, 0), 2))
                painter.fillRect(rect, QColor(255, 120, 0, 35))
                painter.drawRect(rect)
        painter.end()


def _compute_zoom_roi(full_w: int, full_h: int, center_xy, zoom: float, viewport_w: int, viewport_h: int):
    """Compute an original-resolution ROI for a fixed zoom and preview aspect ratio."""
    full_w = int(full_w)
    full_h = int(full_h)
    zoom = max(1.0, float(zoom))
    viewport_w = max(1, int(viewport_w))
    viewport_h = max(1, int(viewport_h))
    aspect = viewport_w / float(viewport_h)

    roi_w = full_w / zoom
    roi_h = roi_w / aspect
    max_h_by_zoom = full_h / zoom
    if roi_h > max_h_by_zoom:
        roi_h = max_h_by_zoom
        roi_w = roi_h * aspect
    roi_w = max(1, min(int(round(roi_w)), full_w))
    roi_h = max(1, min(int(round(roi_h)), full_h))

    if center_xy is None:
        cx, cy = full_w / 2.0, full_h / 2.0
    else:
        cx, cy = float(center_xy[0]), float(center_xy[1])
    x = int(round(cx - roi_w / 2.0))
    y = int(round(cy - roi_h / 2.0))
    x = max(0, min(x, full_w - roi_w))
    y = max(0, min(y, full_h - roi_h))
    return int(x), int(y), int(roi_w), int(roi_h)


def read_zoom_region_from_file(path: str, center_xy=None, zoom: float = 1.0,
                               viewport_size=(900, 600), max_side: int = 1800):
    """Read only the visible preview region, with stride/downsample to keep RAM stable."""
    path = str(path)
    p = Path(path)
    vw, vh = viewport_size
    meta = {"path": path, "reader": "unknown", "roi": None}

    # Raster images are safe to crop with PIL.
    if _has_ext(p.name, RASTER_EXTENSIONS):
        from PIL import Image
        im = Image.open(path).convert("RGB")
        full_w, full_h = im.size
        roi = _compute_zoom_roi(full_w, full_h, center_xy, zoom, vw, vh)
        x, y, w, h = roi
        crop = im.crop((x, y, x + w, y + h))
        crop.thumbnail((max_side, max_side))
        arr = np.asarray(crop, dtype=np.uint8)
        return arr, "YXS", {**meta, "reader": "PIL-region", "shape": (full_h, full_w), "axes": "YXS", "roi": roi, "full_dims": (full_w, full_h)}

    if _has_ext(p.name, TIFF_EXTENSIONS) and not p.name.lower().endswith(".svs"):
        with tifffile.TiffFile(path) as tif:
            s0 = tif.series[0]
            axes = getattr(s0, "axes", "") or ""
            shape = tuple(s0.shape)
            if axes and len(axes) == len(shape) and "Y" in axes and "X" in axes:
                full_h = int(shape[axes.index("Y")])
                full_w = int(shape[axes.index("X")])
            else:
                # Fallback assumes first two dimensions are Y/X.
                full_h, full_w = int(shape[0]), int(shape[1])
            roi = _compute_zoom_roi(full_w, full_h, center_xy, zoom, vw, vh)
            x, y, w, h = roi
            target_ds = max(1.0, max(float(w), float(h)) / float(max(1, max_side)))
            step = max(1, int(math.ceil(max(w, h) / float(max_side))))
            try:
                z = s0.aszarr()
                import zarr
                za = zarr.open(z, mode="r")
                if axes and len(axes) == za.ndim and "Y" in axes and "X" in axes:
                    slicer = []
                    kept = []
                    for ax in axes:
                        if ax == "Y":
                            slicer.append(slice(y, y + h, step)); kept.append("Y")
                        elif ax == "X":
                            slicer.append(slice(x, x + w, step)); kept.append("X")
                        elif ax in ("C", "S"):
                            slicer.append(slice(None)); kept.append(ax)
                        else:
                            slicer.append(0)
                    arr = np.asarray(za[tuple(slicer)])
                    return arr, "".join(kept), {**meta, "reader": "tifffile-zarr-region", "shape": shape, "axes": axes, "roi": roi, "step": step, "full_dims": (full_w, full_h)}
            except Exception as zarr_error:
                last_error = zarr_error

            # Try memory mapping for uncompressed/non-tiled TIFFs.
            try:
                mm = tifffile.memmap(path, series=0)
                mm_axes = axes if axes and len(axes) == mm.ndim else _guess_axes_for_array(mm, axes)
                if mm_axes and len(mm_axes) == mm.ndim and "Y" in mm_axes and "X" in mm_axes:
                    slicer = []
                    kept = []
                    for ax in mm_axes:
                        if ax == "Y":
                            slicer.append(slice(y, y + h, step)); kept.append("Y")
                        elif ax == "X":
                            slicer.append(slice(x, x + w, step)); kept.append("X")
                        elif ax in ("C", "S"):
                            slicer.append(slice(None)); kept.append(ax)
                        else:
                            slicer.append(0)
                    arr = np.asarray(mm[tuple(slicer)])
                    return arr, "".join(kept), {**meta, "reader": "tifffile-memmap-region", "shape": shape, "axes": mm_axes, "roi": roi, "step": step, "full_dims": (full_w, full_h)}
            except Exception as mmap_error:
                last_error = mmap_error

            # If direct full-resolution region access is unavailable, try a
            # reduced-resolution TIFF pyramid/overview level. This avoids the
            # placeholder in crop thumbnails for pyramidal TIFF/OME-TIFF files.
            try:
                overview = _read_best_tiff_overview_region(
                    s0, tif, full_w, full_h, roi,
                    target_ds=target_ds,
                    max_side=max_side,
                    primary_axes=axes,
                    prefer_second_level=True,
                )
                if overview is not None:
                    arr, out_axes, ometa = overview
                    return arr, out_axes, {**meta, **ometa, "shape": shape, "axes": axes, "roi": roi, "full_dims": (full_w, full_h)}
            except Exception as overview_error:
                last_error = overview_error

            # Avoid full reading huge files in the GUI thread.
            spatial_pixels = int(full_w) * int(full_h)
            if spatial_pixels <= 25_000_000:
                arr = s0.asarray()
                if axes and len(axes) == arr.ndim and "Y" in axes and "X" in axes:
                    slicer = []
                    kept = []
                    for ax in axes:
                        if ax == "Y":
                            slicer.append(slice(y, y + h, step)); kept.append("Y")
                        elif ax == "X":
                            slicer.append(slice(x, x + w, step)); kept.append("X")
                        elif ax in ("C", "S"):
                            slicer.append(slice(None)); kept.append(ax)
                        else:
                            slicer.append(0)
                    arr = np.asarray(arr[tuple(slicer)])
                    return arr, "".join(kept), {**meta, "reader": "tifffile-full-region", "shape": shape, "axes": axes, "roi": roi, "step": step, "full_dims": (full_w, full_h)}
                arr = np.asarray(arr[y:y+h:step, x:x+w:step])
                return arr, _guess_axes_for_array(arr, axes), {**meta, "reader": "tifffile-simple-region", "shape": shape, "axes": axes, "roi": roi, "step": step, "full_dims": (full_w, full_h)}

            msg = (
                "Zoom preview skipped to keep GUI responsive.\n"
                f"File: {p.name}\nShape: {shape} axes={axes}\n"
                "This TIFF could not expose zarr or memmap region access.\n"
                "Use saved previews or convert to tiled OME-TIFF for fast interactive viewing."
            )
            return _placeholder_rgb(msg, width=max(600, vw), height=max(400, vh)), "YXS", {**meta, "reader": "safe-placeholder-region", "shape": shape, "axes": axes, "roi": roi, "full_dims": (full_w, full_h), "error": str(last_error)}

    # WSI path through OpenSlide/PIL backend.
    b = ImageBackend().load(path)
    try:
        full_w, full_h = b.slide_dims
        roi = _compute_zoom_roi(full_w, full_h, center_xy, zoom, vw, vh)
        x, y, w, h = roi
        if b.reader == "openslide":
            slide = b._get_openslide()
            target_ds = max(1.0, max(w, h) / float(max_side))
            level = slide.get_best_level_for_downsample(target_ds)
            ds = float(slide.level_downsamples[level])
            lw = max(1, int(round(w / ds)))
            lh = max(1, int(round(h / ds)))
            arr = np.asarray(slide.read_region((x, y), level, (lw, lh)).convert("RGB"), dtype=np.uint8)
            return arr, "YXS", {**meta, "reader": "openslide-region", "roi": roi, "step": ds, "full_dims": (full_w, full_h)}
        arr, _ = b.crop(x, y, w, h)
        arr = _downsample_for_preview(arr, max_side=max_side)
        return arr, "YXS", {**meta, "reader": f"{b.reader}-region", "roi": roi, "full_dims": (full_w, full_h)}
    finally:
        b.close()





def _tiff_level_yx_shape(level, fallback_axes: str = ""):
    """Return (width, height, axes, shape) for a tifffile series/level-like object."""
    try:
        shape = tuple(getattr(level, "shape", ()) or ())
        axes = (getattr(level, "axes", "") or fallback_axes or "").strip()
        if shape and axes and len(axes) == len(shape) and "Y" in axes and "X" in axes:
            return int(shape[axes.index("X")]), int(shape[axes.index("Y")]), axes, shape
        if len(shape) >= 2:
            # Conservative fallback: most tifffile arrays put spatial Y/X in the last two axes.
            return int(shape[-1]), int(shape[-2]), axes, shape
    except Exception:
        pass
    return None


def _tiff_collect_overview_levels(primary_series, tif_obj=None, full_w: int = None, full_h: int = None,
                                  primary_axes: str = ""):
    """Collect possible TIFF pyramid/overview levels from series.levels and fallback series.

    Some TIFF/OME-TIFF files expose reduced-resolution images as ``series.levels``.
    Others expose them as additional ``tif.series`` entries. This helper supports
    both, so the GUI can use a lower-resolution page instead of showing the
    "Zoom preview skipped" placeholder when the full-resolution image is too
    large to read directly.
    """
    candidates = []
    seen = set()

    def add(level, source_label):
        info = _tiff_level_yx_shape(level, fallback_axes=primary_axes)
        if info is None:
            return
        lw, lh, lax, lshape = info
        if lw <= 0 or lh <= 0:
            return
        key = (id(level), tuple(lshape), lax)
        if key in seen:
            return
        seen.add(key)
        if full_w and full_h:
            ds_x = float(full_w) / max(1.0, float(lw))
            ds_y = float(full_h) / max(1.0, float(lh))
            ds = max(1.0, (ds_x + ds_y) / 2.0)
        else:
            ds = 1.0
        # Skip the full-resolution level here. The normal zarr/memmap path handles it.
        if ds <= 1.01:
            return
        candidates.append({
            "level": level,
            "axes": lax,
            "shape": tuple(lshape),
            "w": int(lw),
            "h": int(lh),
            "downsample": float(ds),
            "source": str(source_label),
        })

    try:
        for i, lv in enumerate(list(getattr(primary_series, "levels", []) or [])):
            add(lv, f"series.levels[{i}]")
    except Exception:
        pass

    try:
        if tif_obj is not None:
            for i, s in enumerate(list(getattr(tif_obj, "series", []) or [])[1:], start=1):
                add(s, f"tif.series[{i}]")
    except Exception:
        pass

    # Prefer lowest memory / highest downsample for very zoomed-out views, but the
    # final selection is done by target downsample score.
    candidates.sort(key=lambda c: (c["downsample"], c["w"] * c["h"]))
    return candidates


def _read_tiff_level_region_any(level, level_axes: str, roi, full_w: int, full_h: int,
                                level_downsample: float, max_side: int = 1800,
                                max_full_level_elements: int = 120_000_000):
    """Read a visible region from a TIFF overview level.

    First tries zarr region access. If that is not available, it safely reads the
    whole overview level only when that level is small enough. This is the key
    fallback for compressed pyramidal TIFFs where full-res zarr/memmap access is
    unavailable but a low-res pyramid page can still be used for preview.
    """
    level_downsample = max(1.0, float(level_downsample or 1.0))
    x, y, w, h = [int(v) for v in roi]
    lxw = max(1, int(round(float(w) / level_downsample)))
    lxh = max(1, int(round(float(h) / level_downsample)))
    step = max(1, int(math.ceil(max(float(lxw), float(lxh)) / float(max(1, max_side)))))

    # Fast path: if the overview can expose zarr chunks, read only the visible part.
    try:
        import zarr
        lza = zarr.open(level.aszarr(), mode="r")
        arr, out_axes = _slice_preview_zarr_region(
            lza, level_axes, roi, step=step, level_downsample=level_downsample
        )
        return arr, out_axes, {"method": "zarr", "step": step}
    except Exception as zarr_error:
        last_error = zarr_error

    # Fallback path: read the entire overview level only if it is reasonably small.
    try:
        shape = tuple(getattr(level, "shape", ()) or ())
        n_elem = int(np.prod(shape, dtype=np.int64)) if shape else 0
        if n_elem <= 0 or n_elem > int(max_full_level_elements):
            raise RuntimeError(
                f"Overview level is too large for safe full read: shape={shape}, elements={n_elem}. "
                f"zarr error: {last_error}"
            )
        arr_full = np.asarray(level.asarray())
        axes = level_axes if level_axes and len(level_axes) == arr_full.ndim else _guess_axes_for_array(arr_full, level_axes)
        arr, out_axes = _slice_preview_zarr_region(
            arr_full, axes, roi, step=step, level_downsample=level_downsample
        )
        return arr, out_axes, {"method": "asarray-overview", "step": step, "overview_shape": shape}
    except Exception as full_error:
        raise RuntimeError(f"Could not read TIFF overview level. zarr={last_error}; full={full_error}")


def _read_best_tiff_overview_region(primary_series, tif_obj, full_w: int, full_h: int, roi,
                                    target_ds: float, max_side: int, primary_axes: str = "",
                                    prefer_second_level: bool = False):
    """Try to read the ROI from the best available reduced-resolution TIFF level.

    Returns ``(arr, axes, meta)`` or ``None``.  ``prefer_second_level`` is useful
    for the crop tab: when several levels are available and the target is very
    zoomed out, the function may choose the second/overview level rather than
    failing or attempting full-res access.
    """
    candidates = _tiff_collect_overview_levels(
        primary_series, tif_obj=tif_obj, full_w=full_w, full_h=full_h, primary_axes=primary_axes
    )
    if not candidates:
        return None

    target_ds = max(1.0, float(target_ds or 1.0))
    if prefer_second_level and len(candidates) >= 1 and target_ds <= 1.5:
        # At low zoom, still prefer the first overview level for responsiveness.
        selected = candidates[0]
    else:
        selected = min(
            candidates,
            key=lambda c: abs(math.log(max(c["downsample"], 1.0001) / max(target_ds, 1.0001)))
        )
    arr, out_axes, read_meta = _read_tiff_level_region_any(
        selected["level"], selected["axes"], roi, full_w, full_h,
        selected["downsample"], max_side=max_side
    )
    meta = {
        "reader": f"tifffile-overview-{read_meta.get('method', 'unknown')}",
        "overview_source": selected["source"],
        "overview_shape": selected["shape"],
        "level_downsample": selected["downsample"],
        "step": read_meta.get("step", 1),
    }
    return arr, out_axes, meta

def _slice_preview_zarr_region(za, axes: str, roi, step: int = 1, level_downsample: float = 1.0):
    """Return a Y/X(/C/S) preview slice from a zarr/memmap-like TIFF array."""
    x, y, w, h = roi
    ds = max(1.0, float(level_downsample))
    lx = int(round(float(x) / ds))
    ly = int(round(float(y) / ds))
    lw = max(1, int(round(float(w) / ds)))
    lh = max(1, int(round(float(h) / ds)))
    step = max(1, int(step))
    slicer = []
    kept = []
    if axes and len(axes) == za.ndim and "Y" in axes and "X" in axes:
        for ax in axes:
            if ax == "Y":
                slicer.append(slice(ly, ly + lh, step)); kept.append("Y")
            elif ax == "X":
                slicer.append(slice(lx, lx + lw, step)); kept.append("X")
            elif ax in ("C", "S"):
                slicer.append(slice(None)); kept.append(ax)
            else:
                slicer.append(0)
        return np.asarray(za[tuple(slicer)]), "".join(kept)

    # Conservative fallback: assume last two dimensions are Y/X.
    for dim_i in range(za.ndim):
        if dim_i == za.ndim - 2:
            slicer.append(slice(ly, ly + lh, step)); kept.append("Y")
        elif dim_i == za.ndim - 1:
            slicer.append(slice(lx, lx + lw, step)); kept.append("X")
        else:
            slicer.append(0)
    return np.asarray(za[tuple(slicer)]), "".join(kept)


def read_zoom_region_from_backend(backend, center_xy=None, zoom: float = 1.0,
                                  viewport_size=(900, 600), max_side: int = 1800):
    """Read only the visible Image Preview region from an already-open backend.

    This is the efficient path used by Image Preview. It keeps OpenSlide/TIFF
    handles cached through ImageBackend, prefers existing pyramid levels when
    possible, and never loads a large full-resolution image just to display the
    current screen region.
    """
    if backend is None or not getattr(backend, "path", None) or not getattr(backend, "slide_dims", None):
        raise RuntimeError("No cached preview backend is loaded.")

    vw, vh = viewport_size
    full_w, full_h = backend.slide_dims
    roi = _compute_zoom_roi(full_w, full_h, center_xy, zoom, vw, vh)
    x, y, w, h = roi
    target_ds = max(1.0, max(float(w), float(h)) / float(max(1, max_side)))
    meta = {
        "path": backend.path,
        "reader": f"{backend.reader}-cached-region",
        "roi": roi,
        "full_dims": (int(full_w), int(full_h)),
    }

    if backend.reader == "openslide":
        slide = backend._get_openslide()
        level = slide.get_best_level_for_downsample(target_ds)
        ds = float(slide.level_downsamples[level])
        lw = max(1, int(round(float(w) / ds)))
        lh = max(1, int(round(float(h) / ds)))
        arr = np.asarray(slide.read_region((int(x), int(y)), int(level), (lw, lh)).convert("RGB"), dtype=np.uint8)
        return arr, "YXS", {**meta, "reader": "openslide-cached-region", "level": int(level), "step": ds}

    if backend.reader == "tifffile":
        # Fast preview path for OME-TIFF/TIFF: open the TIFF directory only,
        # then try pyramid/overview levels BEFORE constructing a full-resolution
        # zarr view.  Some compressed OME-TIFF files are slow when aszarr() is
        # called on the full-resolution level, while level 1/2/overview pages are
        # immediately usable for thumbnails and low-zoom previews.
        if getattr(backend, "_tif_obj", None) is None:
            backend._tif_obj = tifffile.TiffFile(backend.path)
            backend._tif_series = backend._tif_obj.series[0]
            backend._tif_axes = getattr(backend._tif_series, "axes", "") or ""
            backend._zarr_array = None
            backend._zarr_error = None
        series = backend._tif_series
        axes = backend._tif_axes or getattr(series, "axes", "") or ""
        shape = tuple(getattr(series, "shape", ()) or ())
        zarr_error = getattr(backend, "_zarr_error", None)

        # 1) Prefer a real pyramid/overview level for low zoom or whole-slide previews.
        # This is the equivalent of asking for "level 2" when available, but it
        # chooses the closest available downsample to the current viewport.
        try:
            overview = _read_best_tiff_overview_region(
                series, getattr(backend, "_tif_obj", None), full_w, full_h, roi,
                target_ds=target_ds, max_side=max_side, primary_axes=axes,
                prefer_second_level=True,
            )
            if overview is not None:
                arr, out_axes, ometa = overview
                return arr, out_axes, {**meta, **ometa, "shape": shape, "axes": axes, "fast_overview_first": True}
        except Exception as overview_error:
            meta["overview_error"] = str(overview_error)

        # 2) Only if no overview exists, try full-resolution zarr region access.
        try:
            za, series, axes, zarr_error = backend._get_tiff_zarr()
            axes = axes or getattr(series, "axes", "") or ""
            shape = tuple(getattr(series, "shape", ()) or ())
        except Exception as e:
            za = None
            zarr_error = e

        step = max(1, int(math.ceil(max(float(w), float(h)) / float(max_side))))
        if za is not None:
            arr, out_axes = _slice_preview_zarr_region(za, axes, roi, step=step, level_downsample=1.0)
            return arr, out_axes, {**meta, "reader": "tifffile-zarr-cached-region", "shape": shape, "axes": axes, "step": step}

        # 3) Try memory mapping for uncompressed/non-tiled TIFFs without reading the full image.
        try:
            mm = tifffile.memmap(backend.path, series=0)
            mm_axes = axes if axes and len(axes) == mm.ndim else _guess_axes_for_array(mm, axes)
            arr, out_axes = _slice_preview_zarr_region(mm, mm_axes, roi, step=step, level_downsample=1.0)
            return arr, out_axes, {**meta, "reader": "tifffile-memmap-cached-region", "shape": shape, "axes": mm_axes, "step": step}
        except Exception as mmap_error:
            pass

        # 4) Last safe fallback only for genuinely small images.
        spatial_pixels = int(full_w) * int(full_h)
        if spatial_pixels <= 25_000_000:
            arr = series.asarray()
            arr, out_axes = _slice_preview_zarr_region(arr, axes if axes and len(axes) == arr.ndim else _guess_axes_for_array(arr, axes), roi, step=step, level_downsample=1.0)
            return arr, out_axes, {**meta, "reader": "tifffile-small-full-cached-region", "shape": shape, "axes": axes, "step": step}

        msg = (
            "Zoom preview skipped to keep GUI responsive.\n"
            f"File: {Path(backend.path).name}\nShape: {shape} axes={axes}\n"
            "No fast pyramid/overview, zarr, or memmap region access was available.\n"
            "Convert to a tiled/pyramidal OME-TIFF for fast interactive viewing."
        )
        return _placeholder_rgb(msg, width=max(600, int(vw)), height=max(400, int(vh))), "YXS", {**meta, "reader": "safe-placeholder-cached-region", "shape": shape, "axes": axes, "error": str(zarr_error)}

    if backend.reader == "pil":
        from PIL import Image
        im = Image.open(backend.path).convert("RGB")
        crop = im.crop((x, y, x + w, y + h))
        crop.thumbnail((max_side, max_side))
        return np.asarray(crop, dtype=np.uint8), "YXS", {**meta, "reader": "PIL-cached-region", "shape": (full_h, full_w), "axes": "YXS"}

    # Generic fallback through ImageBackend crop only for the selected display region.
    arr, _ = backend.crop(x, y, w, h)
    arr = _downsample_for_preview(arr, max_side=max_side)
    return arr, "YXS", {**meta, "reader": f"{backend.reader}-cached-region-fallback"}


def read_roi_region_from_backend(backend, roi_full: Tuple[int, int, int, int], max_side: int = 1200):
    """Read a specific full-resolution ROI for display without loading it at full size.

    The ROI and all returned metadata are always expressed in full-resolution
    source coordinates. Reduced pyramid/overview levels are used only as an
    internal display source. This prevents preview level selection from changing
    crop coordinates or target output size.
    """
    if backend is None or not getattr(backend, "path", None) or not getattr(backend, "slide_dims", None):
        raise RuntimeError("No cached preview backend is loaded.")
    full_w, full_h = backend.slide_dims
    x, y, w, h = backend.clip_roi(int(roi_full[0]), int(roi_full[1]), int(roi_full[2]), int(roi_full[3]), full_w, full_h)
    roi = (int(x), int(y), int(w), int(h))
    target_ds = max(1.0, max(float(w), float(h)) / float(max(1, max_side)))
    meta = {
        "path": backend.path,
        "reader": f"{backend.reader}-cached-roi-preview",
        "roi": roi,
        "full_dims": (int(full_w), int(full_h)),
        "coordinate_space": "full-resolution image pixels",
    }

    if backend.reader == "openslide":
        slide = backend._get_openslide()
        level = slide.get_best_level_for_downsample(target_ds)
        ds = float(slide.level_downsamples[level])
        lw = max(1, int(round(float(w) / ds)))
        lh = max(1, int(round(float(h) / ds)))
        arr = np.asarray(slide.read_region((int(x), int(y)), int(level), (lw, lh)).convert("RGB"), dtype=np.uint8)
        return arr, "YXS", {**meta, "reader": "openslide-cached-roi-preview", "level": int(level), "step": ds}

    if backend.reader == "tifffile":
        # Do not call _get_tiff_zarr first. That can touch the full-resolution
        # image and be very slow for large compressed OME-TIFFs. Open the TIFF
        # directory, try overview/pyramid levels first, then fall back to zarr.
        if getattr(backend, "_tif_obj", None) is None:
            backend._tif_obj = tifffile.TiffFile(backend.path)
            backend._tif_series = backend._tif_obj.series[0]
            backend._tif_axes = getattr(backend._tif_series, "axes", "") or ""
            backend._zarr_array = None
            backend._zarr_error = None
        series = backend._tif_series
        axes = backend._tif_axes or getattr(series, "axes", "") or ""
        shape = tuple(getattr(series, "shape", ()) or ())
        zarr_error = getattr(backend, "_zarr_error", None)

        # 1) Prefer pyramid/overview levels for display. Coordinates remain full-res.
        try:
            overview = _read_best_tiff_overview_region(
                series, getattr(backend, "_tif_obj", None), full_w, full_h, roi,
                target_ds=target_ds, max_side=max_side, primary_axes=axes,
                prefer_second_level=True,
            )
            if overview is not None:
                arr, out_axes, ometa = overview
                return arr, out_axes, {**meta, **ometa, "shape": shape, "axes": axes, "preview_from_overview": True}
        except Exception as overview_error:
            meta["overview_error"] = str(overview_error)

        # 2) Full-resolution zarr/memmap only when no overview can serve the preview.
        try:
            za, series, axes, zarr_error = backend._get_tiff_zarr()
            axes = axes or getattr(series, "axes", "") or ""
            shape = tuple(getattr(series, "shape", ()) or ())
        except Exception as e:
            za = None
            zarr_error = e

        step = max(1, int(math.ceil(max(float(w), float(h)) / float(max_side))))
        if za is not None:
            arr, out_axes = _slice_preview_zarr_region(za, axes, roi, step=step, level_downsample=1.0)
            return arr, out_axes, {**meta, "reader": "tifffile-zarr-cached-roi-preview", "shape": shape, "axes": axes, "step": step}

        try:
            mm = tifffile.memmap(backend.path, series=0)
            mm_axes = axes if axes and len(axes) == mm.ndim else _guess_axes_for_array(mm, axes)
            arr, out_axes = _slice_preview_zarr_region(mm, mm_axes, roi, step=step, level_downsample=1.0)
            return arr, out_axes, {**meta, "reader": "tifffile-memmap-cached-roi-preview", "shape": shape, "axes": mm_axes, "step": step}
        except Exception:
            pass

        if int(full_w) * int(full_h) <= 25_000_000:
            arr = series.asarray()
            arr, out_axes = _slice_preview_zarr_region(arr, axes if axes and len(axes) == arr.ndim else _guess_axes_for_array(arr, axes), roi, step=step, level_downsample=1.0)
            return arr, out_axes, {**meta, "reader": "tifffile-small-full-cached-roi-preview", "shape": shape, "axes": axes, "step": step}

        msg = (
            "Crop/tile preview skipped to keep GUI responsive.\n"
            f"File: {Path(backend.path).name}\nShape: {shape} axes={axes}\n"
            "No fast pyramid/overview, zarr, or memmap region access was available.\n"
            "The selected crop coordinates are still full-resolution coordinates."
        )
        return _placeholder_rgb(msg, width=700, height=500), "YXS", {**meta, "reader": "safe-placeholder-cached-roi-preview", "shape": shape, "axes": axes, "error": str(zarr_error)}

    if backend.reader == "pil":
        from PIL import Image
        im = Image.open(backend.path).convert("RGB")
        crop = im.crop((x, y, x + w, y + h))
        crop.thumbnail((max_side, max_side))
        return np.asarray(crop, dtype=np.uint8), "YXS", {**meta, "reader": "PIL-cached-roi-preview", "shape": (full_h, full_w), "axes": "YXS"}

    arr, _ = backend.crop(x, y, w, h)
    arr = _downsample_for_preview(arr, max_side=max_side)
    return arr, "YXS", {**meta, "reader": f"{backend.reader}-cached-roi-preview-fallback"}

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
    text_u = str(text).upper()
    if "JPEG" in text_u or "JPG" in text_u:
        return ".jpg"
    if "OME-TIFF" in text_u or "OME.TIF" in text_u:
        return ".ome.tif"
    return ".tif"


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


def _compute_tile_positions(length: int, tile_size: int, stride: int, edge_mode: str = "edge_aligned"):
    """Compute tile start positions along one dimension.

    edge_mode="edge_aligned" avoids tiny last sliver tiles by adding a final
    tile whose right/bottom edge coincides with the image boundary. This can
    create extra overlap at the border, but it keeps the last tiles comparable
    in size to the other tiles and avoids artificial padding for raw IF data.

    edge_mode="partial" preserves the older behavior: positions are generated
    by regular stride until the image end, so the last tile may be very small.
    """
    length = int(length)
    tile_size = int(tile_size)
    stride = int(stride)
    if length <= 0:
        return []
    if tile_size <= 0 or stride <= 0:
        raise ValueError("Tile size and stride must be positive.")
    if length <= tile_size:
        return [0]

    edge_mode = str(edge_mode or "edge_aligned").lower().replace("-", "_")
    if edge_mode in ("partial", "partial_edges", "allow_partial"):
        return list(range(0, length, stride))

    max_start = max(0, length - tile_size)
    positions = list(range(0, max_start + 1, stride))
    if not positions or positions[-1] != max_start:
        positions.append(max_start)
    return sorted(set(int(v) for v in positions))


def _compute_tile_grid(width: int, height: int, tile_size: int, overlap_percent: float, edge_mode: str = "edge_aligned"):
    overlap_px = int(round(tile_size * overlap_percent / 100.0))
    stride = tile_size - overlap_px
    if stride <= 0:
        raise ValueError("Overlap must be lower than 100%.")

    x_positions = _compute_tile_positions(width, tile_size, stride, edge_mode=edge_mode)
    y_positions = _compute_tile_positions(height, tile_size, stride, edge_mode=edge_mode)

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

        # OME-TIFF and scientific multichannel TIFFs should prefer tifffile.
        # OpenSlide may open some OME-TIFFs as generic TIFFs but then lose the
        # OME axes and physical pixel size; crops saved afterwards may appear as
        # 1 µm/px.  For .ome.tif/.ome.tiff, preserve scientific metadata first.
        if _is_ome_tiff_name(lower_name) and _has_ext(lower_name, TIFF_EXTENSIONS):
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
        """Open a TIFF/OME-TIFF once and reuse its zarr view for repeated crops.

        Important: some preview paths open ``self._tif_obj`` first only to inspect
        pyramid/overview levels.  In that case ``_zarr_array`` is still None and
        ``_zarr_error`` is also None.  Older versions returned immediately in that
        state, so exact cropping thought there was no zarr/memmap access and raised
        a false error.  This method now creates the zarr view whenever it has not
        actually been attempted yet.
        """
        if self._tif_obj is None:
            self._tif_obj = tifffile.TiffFile(self.path)
            self._tif_series = self._tif_obj.series[0]
            self._tif_axes = getattr(self._tif_series, "axes", "") or ""
            self._zarr_array = None
            self._zarr_error = None

        if self._zarr_array is None and self._zarr_error is None:
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

            # Calibration fallback for TIFF/OME-TIFF files opened through OpenSlide.
            # Some pyramidal OME-TIFFs can be read by OpenSlide but OpenSlide does
            # not expose PhysicalSizeX/Y; tifffile can still read the OME metadata.
            if (mpp is None or res_tuple is None) and _has_ext(path, TIFF_EXTENSIONS):
                try:
                    _tw, _th, tif_res, tif_mpp = self._probe_tifffile(path)
                    if tif_mpp is not None:
                        mpp = tif_mpp
                    if tif_res is not None:
                        res_tuple = tif_res
                    elif mpp is not None:
                        res_tuple = _mpp_to_resolution_tuple(mpp[0], mpp[1])
                except Exception:
                    pass
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
                    u_val = ru.value
                    # tifffile can expose enums, ints, or strings depending on file.
                    if hasattr(u_val, "name"):
                        name = str(u_val.name).upper()
                        if "INCH" in name:
                            unit_str = "INCH"
                        elif "CENTIMETER" in name or "CENTIMETRE" in name:
                            unit_str = "CENTIMETER"
                    if unit_str is None:
                        u = int(u_val)
                        if u == 2:
                            unit_str = "INCH"
                        elif u == 3:
                            unit_str = "CENTIMETER"
                except Exception:
                    try:
                        txt = str(ru.value).upper()
                        if "INCH" in txt:
                            unit_str = "INCH"
                        elif "CENTIMETER" in txt or "CENTIMETRE" in txt:
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
        """Return an RGB visual crop from a TIFF/OME-TIFF.

        The preferred path is exact full-resolution tifffile/zarr access.  If a
        compressed TIFF cannot expose zarr/memmap access, try OpenSlide as a
        visual-only fallback before failing.  This keeps normal RGB crop export
        usable for very large TIFFs while raw IF export still requires tifffile
        zarr/memmap access so that scientific channel intensities are preserved.
        """
        try:
            arr, axes, fallback_info = self._crop_tifffile_raw(path, x, y, w, h)
            return _array_to_rgb_preview(arr, axes), fallback_info
        except Exception as tif_error:
            openslide = _try_import_openslide()
            if openslide is not None:
                try:
                    slide = openslide.OpenSlide(str(path))
                    try:
                        img = slide.read_region((int(x), int(y)), 0, (int(w), int(h))).convert("RGB")
                        return np.asarray(img, dtype=np.uint8), {
                            "used": True,
                            "reason": f"visual OpenSlide fallback after tifffile crop failed: {tif_error}",
                            "fallback_reader": "openslide-visual",
                        }
                    finally:
                        try:
                            slide.close()
                        except Exception:
                            pass
                except Exception as os_error:
                    raise RuntimeError(
                        "Could not crop this TIFF/OME-TIFF through tifffile/zarr/memmap, "
                        "and the visual OpenSlide fallback also failed.\n\n"
                        f"tifffile error: {tif_error}\nOpenSlide fallback error: {os_error}"
                    ) from tif_error
            raise

    def _crop_tifffile_raw(self, path, x, y, w, h):
        fallback_info = {"used": False}
        za, series, axes, zarr_error = self._get_tiff_zarr()
        axes = axes or getattr(series, "axes", "") or ""
        if za is not None:
            slicer = self._build_spatial_slicer_raw(za.ndim, axes, x, y, w, h)
            arr = np.asarray(za[tuple(slicer)])
            return arr, axes if len(axes) == arr.ndim else _guess_axes_for_array(arr, axes), fallback_info

        # Fallback: read the full series only for genuinely small images.
        # For large compressed TIFF/OME-TIFF images this would allocate tens of GiB.
        try:
            spatial_pixels = int(self.slide_dims[0]) * int(self.slide_dims[1]) if self.slide_dims else 0
        except Exception:
            spatial_pixels = 0
        if spatial_pixels > 25_000_000:
            raise RuntimeError(
                "This TIFF/OME-TIFF does not expose zarr/memmap region access for exact cropping, "
                "and the full image is too large to read safely. Use a tiled/pyramidal OME-TIFF, "
                "increase downsample for a visual crop, or use Tiles mode. "
                f"Original zarr error: {zarr_error}"
            )
        fallback_info = {
            "used": True,
            "reason": f"zarr crop failed, used small full read fallback: {zarr_error}"
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
            # Memory-light thumbnail: use TIFF/zarr region stepping instead of
            # reading the full 20k x 20k x C image into RAM. This prevents the
            # GUI from becoming unresponsive while loading IF/OME-TIFF files.
            arr, axes, _meta = read_preview_array_from_file(self.path, max_side=max_side)
            return _array_to_rgb_preview(arr, axes)

        raise RuntimeError(f"Unknown reader: {self.reader}")


# ============================================================
# Save helper
# ============================================================

def _photometric_for_axes_shape(axes: str, shape) -> str:
    """Return a safe tifffile photometric mode for a given axes/shape.

    Tifffile raises "shape does not match stored shape" when an array with a
    Samples axis (for example YXS or TZYXS) is written as minisblack OME.
    Those arrays must be written as RGB/RGBA. Non-sample scientific IF arrays
    such as CYX, ZCYX or TCZYX remain minisblack.
    """
    axes = (axes or "").strip()
    try:
        if axes and "S" in axes and len(axes) == len(shape):
            s = int(shape[axes.index("S")])
            if s in (3, 4):
                return "rgb"
    except Exception:
        pass
    try:
        # Fallback for plain RGB arrays without axes metadata.
        if len(shape) >= 3 and int(shape[-1]) in (3, 4):
            return "rgb"
    except Exception:
        pass
    return "minisblack"


def _is_sample_axis_array(arr_shape, axes: str) -> bool:
    axes = (axes or "").strip()
    try:
        return bool(axes and "S" in axes and len(axes) == len(arr_shape) and int(arr_shape[axes.index("S")]) in (3, 4))
    except Exception:
        return bool(len(arr_shape) >= 3 and int(arr_shape[-1]) in (3, 4))


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
            if resolution is None:
                derived = _mpp_to_resolution_tuple(mpp_x_um, mpp_y_um)
                if derived is not None:
                    resolution = (float(derived[0]), float(derived[1]))
                    resolutionunit = derived[2]
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



def save_rgb_crop_lowmem(backend, output_path: Path, x: int, y: int, w: int, h: int,
                         downsample: float = 1.0, output_format: str = "tiff",
                         write_ome: bool = False, lossless: bool = True,
                         source_resolution=None, source_mpp=None, image_name: Optional[str] = None,
                         annotation_kv: Optional[Dict[str, Any]] = None,
                         chunk_size: int = 2048, progress_callback=None) -> Dict[str, Any]:
    """Save a large visual RGB crop using disk-backed chunks instead of one huge RAM array.

    This is intended for large H&E/RGB-style crops where a single array could be
    many GiB. It reads the source in manageable chunks, writes those chunks into a
    temporary memory-mapped array on disk, and then writes the final TIFF/OME-TIFF.

    The function deliberately does not use preview arrays. Therefore a GUI
    placeholder such as "Preview skipped to keep GUI responsive" can never become
    saved image content.
    """
    if backend is None or not getattr(backend, "path", None):
        raise RuntimeError("No image backend is loaded.")

    output_format = str(output_format or "tiff").lower()
    if output_format == "jpeg":
        raise RuntimeError(
            "Large low-memory crop saving is supported for TIFF/OME-TIFF output. "
            "JPEG requires a complete in-memory RGB image; increase downsample or save as TIFF/OME-TIFF."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = max(1.0, float(downsample or 1.0))
    full_w, full_h = backend.slide_dims
    x, y, w, h = backend.clip_roi(int(x), int(y), int(w), int(h), int(full_w), int(full_h))
    out_w = max(1, int(round(float(w) / ds)))
    out_h = max(1, int(round(float(h) / ds)))
    chunk_size = max(256, int(chunk_size or 2048))

    # Calibration metadata must describe the output pixels after downsampling.
    source_resolution, source_mpp = _scale_resolution_and_mpp(
        source_resolution=source_resolution,
        source_mpp=source_mpp,
        pixel_scale=ds,
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

    mpp_x_um = None
    mpp_y_um = None
    if source_mpp:
        try:
            mpp_x_um, mpp_y_um = float(source_mpp[0]), float(source_mpp[1])
            if resolution is None:
                derived = _mpp_to_resolution_tuple(mpp_x_um, mpp_y_um)
                if derived is not None:
                    resolution = (float(derived[0]), float(derived[1]))
                    resolutionunit = derived[2]
        except Exception:
            mpp_x_um = None
            mpp_y_um = None

    temp_dir = tempfile.mkdtemp(prefix="tiffcropper_rgb_crop_")
    temp_path = Path(temp_dir) / "rgb_crop_memmap.npy"
    mm = None
    try:
        mm = np.lib.format.open_memmap(str(temp_path), mode="w+", dtype=np.uint8, shape=(out_h, out_w, 3))
        total_blocks = int(math.ceil(out_h / chunk_size)) * int(math.ceil(out_w / chunk_size))
        done = 0

        from PIL import Image
        for oy0 in range(0, out_h, chunk_size):
            oy1 = min(out_h, oy0 + chunk_size)
            for ox0 in range(0, out_w, chunk_size):
                ox1 = min(out_w, ox0 + chunk_size)

                if ds == 1.0:
                    sx0 = int(x + ox0)
                    sy0 = int(y + oy0)
                    sw = int(ox1 - ox0)
                    sh = int(oy1 - oy0)
                else:
                    sx0 = int(x + math.floor(ox0 * ds))
                    sy0 = int(y + math.floor(oy0 * ds))
                    sx1 = int(x + math.ceil(ox1 * ds))
                    sy1 = int(y + math.ceil(oy1 * ds))
                    sx1 = min(int(x + w), sx1)
                    sy1 = min(int(y + h), sy1)
                    sw = max(1, sx1 - sx0)
                    sh = max(1, sy1 - sy0)

                chunk, info = backend.crop(sx0, sy0, sw, sh, fill=255)
                chunk = _to_uint8_rgb(chunk)
                target_w = int(ox1 - ox0)
                target_h = int(oy1 - oy0)
                if chunk.shape[1] != target_w or chunk.shape[0] != target_h:
                    chunk = np.asarray(
                        Image.fromarray(chunk).resize((target_w, target_h), Image.Resampling.LANCZOS),
                        dtype=np.uint8,
                    )
                mm[oy0:oy1, ox0:ox1, :] = chunk
                done += 1
                if progress_callback is not None:
                    try:
                        progress_callback(done, total_blocks)
                        QApplication.processEvents()
                    except Exception:
                        pass

        mm.flush()
        compression_kwargs = {"compression": "deflate", "predictor": True} if lossless else {}
        if write_ome:
            ome_xml = _build_ome_xml_rgb(
                size_x=int(out_w),
                size_y=int(out_h),
                physical_size_x_um=mpp_x_um,
                physical_size_y_um=mpp_y_um,
                image_name=image_name or output_path.stem,
                annotation_kv=annotation_kv,
            )
            tifffile.imwrite(
                str(output_path), mm, bigtiff=True, tile=(256, 256), photometric="rgb",
                description=_ascii_safe(ome_xml), software=f"{APP_NAME} v{APP_VERSION}",
                resolution=resolution, resolutionunit=resolutionunit, **compression_kwargs,
            )
        else:
            tifffile.imwrite(
                str(output_path), mm, bigtiff=True, tile=(256, 256), photometric="rgb",
                description=_ascii_safe(f"Generated by {APP_NAME} v{APP_VERSION}"),
                software=f"{APP_NAME} v{APP_VERSION}", resolution=resolution,
                resolutionunit=resolutionunit, **compression_kwargs,
            )

        return {
            "lowmem": True,
            "shape": (int(out_h), int(out_w), 3),
            "dtype": "uint8",
            "downsample": float(ds),
            "chunks": int(total_blocks),
            "temp_bytes": int(out_h) * int(out_w) * 3,
        }
    finally:
        try:
            del mm
        except Exception:
            pass
        try:
            if temp_path.exists():
                temp_path.unlink()
            Path(temp_dir).rmdir()
        except Exception:
            pass

def save_multichannel_image(output_path, arr, axes=None, write_ome=True, lossless=True,
                            source_resolution=None, source_mpp=None, image_name=None,
                            pixel_scale=1.0):
    """Save scientific crop data without RGB conversion or intensity normalization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    # Avoid unnecessary copies. This matters for disk-backed memmap arrays used
    # by exact raw merge of large multichannel tiles.
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
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
            # Use ASCII "um" to maximize compatibility with QuPath/ImageJ readers.
            metadata["PhysicalSizeXUnit"] = "um"
            metadata["PhysicalSizeYUnit"] = "um"
            # Ensure classic TIFF resolution tags are also written. Some viewers
            # prioritize TIFF tags over OME-XML; this prevents defaulting to 1 µm/px.
            if resolution is None:
                derived = _mpp_to_resolution_tuple(float(source_mpp[0]), float(source_mpp[1]))
                if derived is not None:
                    resolution = (float(derived[0]), float(derived[1]))
                    resolutionunit = derived[2]
        except Exception:
            pass

    compression_kwargs = {"compression": "deflate", "predictor": True} if lossless else {}

    photometric = _photometric_for_axes_shape(axes, arr.shape)

    tifffile.imwrite(
        str(output_path),
        arr,
        bigtiff=True,
        ome=bool(write_ome),
        metadata=metadata if metadata else None,
        photometric=photometric,
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
# Background workers and batch job helpers
# ============================================================

class AppWorker(QThread):
    """Run long jobs outside the Qt GUI thread.

    This keeps Windows from showing "Not Responding" while tiles, downsampled
    files, or LIF scenes are being written. Progress is reported through Qt
    signals, so all GUI changes still happen safely in the main thread.
    """
    message = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, job_func, *args, **kwargs):
        super().__init__()
        self.job_func = job_func
        self.args = args
        self.kwargs = kwargs
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()
        self.message.emit("Cancellation requested. Waiting for current file/plane to finish safely...")

    def run(self):
        try:
            result = self.job_func(
                self.cancel_event,
                self.progress.emit,
                self.message.emit,
                *self.args,
                **self.kwargs,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled by user.")


def _save_fixed_tiles_one_image_worker(image_path, params, lossless=True, cancel_event=None, tile_progress_cb=None):
    image_path = Path(image_path)
    backend = ImageBackend().load(str(image_path))
    try:
        full_w, full_h = backend.slide_dims
        tile_size = int(params["tile_size"])
        overlap = float(params["overlap"])
        downsample = float(params["downsample"])
        padding = params["padding"]
        output_format = params["output_format"]
        write_ome = bool(params.get("write_ome", False))
        preserve_raw = bool(params.get("preserve_raw", False)) and backend.reader == "tifffile" and output_format != "jpeg"
        ext = params["ext"]
        suffix = _suffix_for_tile(overlap, downsample)
        edge_mode = params.get("edge_mode", "edge_aligned")
        xs, ys, _, _ = _compute_tile_grid(full_w, full_h, tile_size, overlap, edge_mode=edge_mode)
        out_dir = image_path.parent / image_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        total = len(xs) * len(ys)
        count = 0
        for row_idx, y in enumerate(ys):
            for col_idx, x in enumerate(xs):
                _check_cancel(cancel_event)
                crop_w = min(tile_size, full_w - x)
                crop_h = min(tile_size, full_h - y)
                if crop_w <= 0 or crop_h <= 0:
                    continue
                if preserve_raw:
                    # Exact scientific tile: no RGB conversion, no per-tile intensity normalization.
                    # Border tiles are saved as their true smaller size rather than padded,
                    # because padding would create artificial pixels in multichannel data.
                    actual_w, actual_h = crop_w, crop_h
                    tile, axes, _ = backend.crop_raw(x, y, actual_w, actual_h)
                    if downsample != 1.0:
                        tile, axes = _resize_spatial_array(tile, axes, downsample)
                    out_name = (
                        f"{image_path.stem}_{_col_to_letters(col_idx)}{row_idx + 1}{suffix}"
                        f"_X{x}_Y{y}_W{actual_w}_H{actual_h}{ext}"
                    )
                    out_path = out_dir / out_name
                    save_multichannel_image(
                        out_path, tile, axes=axes, write_ome=write_ome, lossless=lossless,
                        source_resolution=backend.source_resolution, source_mpp=backend.source_mpp,
                        image_name=out_path.stem, pixel_scale=downsample,
                    )
                else:
                    # Visual tile path: converts to RGB uint8 for normal image/JPG workflows.
                    # This is not intensity-preserving for IF data. Use preserve_raw for IF.
                    is_edge_tile = (x + tile_size > full_w) or (y + tile_size > full_h)
                    if is_edge_tile:
                        tile = backend.read_tile_with_padding(x=x, y=y, size=tile_size, padding_color=padding)
                        actual_w, actual_h = crop_w, crop_h
                    else:
                        tile, _ = backend.crop(x, y, tile_size, tile_size, fill=255)
                        actual_w, actual_h = tile_size, tile_size
                    if downsample != 1.0:
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
                        out_path, tile, output_format, write_ome, lossless,
                        backend.source_resolution, backend.source_mpp, out_path.stem,
                        pixel_scale=downsample,
                    )
                count += 1
                if tile_progress_cb is not None:
                    tile_progress_cb(count, total)
        return {"image": str(image_path), "output_folder": str(out_dir), "tiles_written": count, "reader": backend.reader}
    finally:
        backend.close()


def _save_division_tiles_one_image_worker(image_path, params, lossless=True, cancel_event=None, tile_progress_cb=None):
    image_path = Path(image_path)
    backend = ImageBackend().load(str(image_path))
    try:
        full_w, full_h = backend.slide_dims
        rows = int(params["rows"])
        cols = int(params["cols"])
        downsample = float(params["downsample"])
        output_format = params["output_format"]
        write_ome = bool(params.get("write_ome", False))
        preserve_raw = bool(params.get("preserve_raw", False)) and backend.reader == "tifffile" and output_format != "jpeg"
        ext = params["ext"]
        suffix = _suffix_for_division_tile(rows, cols, downsample)
        out_dir = image_path.parent / f"{image_path.stem}_{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        total = rows * cols
        count = 0
        for r in range(rows):
            for c in range(cols):
                _check_cancel(cancel_event)
                x, y, w, h = _division_bounds(0, 0, full_w, full_h, rows, cols, r, c)
                out_name = f"{image_path.stem}_R{r + 1:03d}_C{c + 1:03d}{suffix}_X{x}_Y{y}_W{w}_H{h}{ext}"
                out_path = out_dir / out_name
                if preserve_raw:
                    tile, axes, _ = backend.crop_raw(x, y, w, h)
                    if downsample != 1.0:
                        tile, axes = _resize_spatial_array(tile, axes, downsample)
                    save_multichannel_image(
                        out_path, tile, axes=axes, write_ome=write_ome, lossless=lossless,
                        source_resolution=backend.source_resolution, source_mpp=backend.source_mpp,
                        image_name=out_path.stem, pixel_scale=downsample,
                    )
                else:
                    tile, _ = backend.crop(x, y, w, h, fill=255)
                    if downsample != 1.0:
                        from PIL import Image
                        new_w = max(1, int(round(tile.shape[1] / downsample)))
                        new_h = max(1, int(round(tile.shape[0] / downsample)))
                        tile = np.asarray(Image.fromarray(tile).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)
                    save_rgb_image(
                        out_path, tile, output_format, write_ome, lossless,
                        backend.source_resolution, backend.source_mpp, out_path.stem,
                        pixel_scale=downsample,
                    )
                count += 1
                if tile_progress_cb is not None:
                    tile_progress_cb(count, total)
        return {"image": str(image_path), "output_folder": str(out_dir), "tiles_written": count, "reader": backend.reader}
    finally:
        backend.close()


def _tile_bulk_job(cancel_event, progress_cb, message_cb, image_paths, params, lossless=True, max_workers=1):
    image_paths = [Path(p) for p in image_paths]
    max_workers = max(1, int(max_workers or 1))
    rows_log = []
    ok = failed = 0
    progress_cb(0, len(image_paths))

    def run_one(p):
        if str(params["mode"]).startswith("Fixed"):
            return _save_fixed_tiles_one_image_worker(p, params, lossless=lossless, cancel_event=cancel_event)
        return _save_division_tiles_one_image_worker(p, params, lossless=lossless, cancel_event=cancel_event)

    if max_workers <= 1 or len(image_paths) <= 1:
        for i, p in enumerate(image_paths, start=1):
            _check_cancel(cancel_event)
            try:
                message_cb(f"Saving tiles: {p.name} ({i}/{len(image_paths)})")
                result = run_one(p)
                rows_log.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "operation": "fixed_tiles" if str(params["mode"]).startswith("Fixed") else "division_tiles",
                    "status": "success",
                    "image": str(p),
                    "reader": result.get("reader", ""),
                    "output_folder": result.get("output_folder", ""),
                    "tiles_expected": "",
                    "tiles_written": result.get("tiles_written", ""),
                    "message": "",
                })
                ok += 1
            except Exception as exc:
                failed += 1
                rows_log.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "operation": "fixed_tiles" if str(params["mode"]).startswith("Fixed") else "division_tiles",
                    "status": "failed",
                    "image": str(p),
                    "reader": "",
                    "output_folder": "",
                    "tiles_expected": "",
                    "tiles_written": 0,
                    "message": f"{exc}\n{traceback.format_exc()}",
                })
            progress_cb(i, len(image_paths))
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(image_paths))) as ex:
            future_to_path = {ex.submit(run_one, p): p for p in image_paths}
            done = 0
            for fut in as_completed(future_to_path):
                _check_cancel(cancel_event)
                p = future_to_path[fut]
                done += 1
                try:
                    result = fut.result()
                    ok += 1
                    rows_log.append({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "operation": "fixed_tiles" if str(params["mode"]).startswith("Fixed") else "division_tiles",
                        "status": "success",
                        "image": str(p),
                        "reader": result.get("reader", ""),
                        "output_folder": result.get("output_folder", ""),
                        "tiles_expected": "",
                        "tiles_written": result.get("tiles_written", ""),
                        "message": "",
                    })
                    message_cb(f"Finished tiles: {p.name} ({done}/{len(image_paths)})")
                except Exception as exc:
                    failed += 1
                    rows_log.append({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "operation": "fixed_tiles" if str(params["mode"]).startswith("Fixed") else "division_tiles",
                        "status": "failed",
                        "image": str(p),
                        "reader": "",
                        "output_folder": "",
                        "tiles_expected": "",
                        "tiles_written": 0,
                        "message": f"{exc}\n{traceback.format_exc()}",
                    })
                progress_cb(done, len(image_paths))
    log_base = image_paths[0].parent if image_paths else Path.cwd()
    log_path = log_base / f"TiffCropper_batch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = ["timestamp", "operation", "status", "image", "reader", "output_folder", "tiles_expected", "tiles_written", "message"]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_log:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return {"ok": ok, "failed": failed, "log_path": str(log_path), "task": "tiles"}


def _downsample_bulk_job(cancel_event, progress_cb, message_cb, image_paths, output_dir, factor, output_kind, preserve_raw, lossless, overwrite, max_workers=1):
    paths = [Path(p) for p in image_paths]
    out_dir = Path(output_dir) if output_dir else None
    max_workers = max(1, int(max_workers or 1))
    rows, ok, failed = [], 0, 0
    progress_cb(0, len(paths))

    def run_one(p):
        _check_cancel(cancel_event)
        return downsample_image_file(p, out_dir, factor, output_kind, preserve_raw, lossless, overwrite)

    if max_workers <= 1 or len(paths) <= 1:
        for i, p in enumerate(paths, start=1):
            _check_cancel(cancel_event)
            try:
                message_cb(f"Downsampling {p.name} ({i}/{len(paths)})")
                out = run_one(p)
                rows.append({"input": str(p), "output": str(out), "status": "success", "message": ""})
                ok += 1
            except Exception as exc:
                rows.append({"input": str(p), "output": "", "status": "failed", "message": f"{exc}\n{traceback.format_exc()}"})
                failed += 1
            progress_cb(i, len(paths))
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(paths))) as ex:
            future_to_path = {ex.submit(run_one, p): p for p in paths}
            done = 0
            for fut in as_completed(future_to_path):
                _check_cancel(cancel_event)
                p = future_to_path[fut]
                done += 1
                try:
                    out = fut.result()
                    rows.append({"input": str(p), "output": str(out), "status": "success", "message": ""})
                    ok += 1
                    message_cb(f"Finished downsample: {p.name} ({done}/{len(paths)})")
                except Exception as exc:
                    rows.append({"input": str(p), "output": "", "status": "failed", "message": f"{exc}\n{traceback.format_exc()}"})
                    failed += 1
                progress_cb(done, len(paths))

    log_dir = out_dir or (paths[0].parent if paths else Path.cwd())
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"TiffCropper_downsample_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output", "status", "message"])
        writer.writeheader()
        writer.writerows(rows)
    return {"ok": ok, "failed": failed, "log_path": str(log_path), "task": "downsample"}


def _export_lif_file_headless(lif_path, scene_indices, options, cancel_event=None, progress_cb=None, message_cb=None):
    LifFile = _require_readlif()
    lif_path = Path(lif_path)
    lif_obj = LifFile(str(lif_path))
    images = list(lif_obj.get_iter_image())
    out_dir = _lif_output_folder_for(lif_path, options.get("output_base"))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{_lif_safe_name(lif_path.stem)}_manifest.csv"
    rows = []
    xml_path = _lif_write_xml_header(lif_obj, lif_path, out_dir) if options.get("save_xml", True) else None
    selected = set(scene_indices) if scene_indices is not None else set(range(len(images)))
    total_scenes = len(selected)
    done_scenes = 0

    for scene_index, img in enumerate(images):
        if scene_index not in selected:
            continue
        _check_cancel(cancel_event)
        scene_name = str(getattr(img, "name", f"scene_{scene_index}"))
        safe_scene = _lif_safe_name(scene_name)
        out_path = out_dir / f"scene_{scene_index:03d}_{safe_scene}.ome.tif"
        base_meta = _lif_scene_metadata_dict(img, scene_index)
        json_path = _lif_write_scene_json(base_meta, out_dir, scene_index, scene_name) if options.get("save_json", True) else None
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

        def plane_progress(done, total, scene_index=scene_index, scene_name=scene_name):
            _check_cancel(cancel_event)
            if progress_cb is not None:
                progress_cb(done, max(1, total))
            if message_cb is not None:
                message_cb(f"Writing LIF {lif_path.name} | scene {scene_index}: {scene_name} | plane {done}/{total}")

        try:
            result = _lif_save_scene_ome_tiff_lowmem(
                img=img,
                scene_index=scene_index,
                out_path=out_path,
                overwrite=options.get("overwrite", False),
                skip_existing=options.get("skip_existing", True),
                compression=options.get("compression"),
                progress_callback=plane_progress,
            )
            row.update(result)
            row["status"] = "skipped_existing" if result.get("skipped_existing") else "success"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{exc}\n{traceback.format_exc()}"
            rows.append(row)
            _lif_write_manifest(manifest_path, rows)
            if options.get("stop_on_error", False):
                raise
        else:
            rows.append(row)
            _lif_write_manifest(manifest_path, rows)
        done_scenes += 1
        if progress_cb is not None:
            progress_cb(done_scenes, max(1, total_scenes))
        if message_cb is not None:
            message_cb(f"Finished LIF scene {scene_index}: {scene_name} ({done_scenes}/{total_scenes})")
    return manifest_path


def _lif_export_job(cancel_event, progress_cb, message_cb, lif_paths, scene_indices_by_file, options, max_workers=1):
    paths = [Path(p) for p in lif_paths]
    max_workers = max(1, int(max_workers or 1))
    manifests, failures = [], []
    progress_cb(0, len(paths))

    def run_one(p):
        indices = None
        if scene_indices_by_file:
            indices = scene_indices_by_file.get(str(p), None)
        return _export_lif_file_headless(p, indices, options, cancel_event=cancel_event, progress_cb=None, message_cb=None)

    # LIF export is memory- and disk-heavy. Parallelism is kept per file, not per scene.
    if max_workers <= 1 or len(paths) <= 1:
        for i, p in enumerate(paths, start=1):
            _check_cancel(cancel_event)
            try:
                message_cb(f"Exporting LIF: {p.name} ({i}/{len(paths)})")
                indices = scene_indices_by_file.get(str(p), None) if scene_indices_by_file else None
                manifest = _export_lif_file_headless(p, indices, options, cancel_event=cancel_event, progress_cb=progress_cb, message_cb=message_cb)
                manifests.append(str(manifest))
            except Exception as exc:
                failures.append({"file": str(p), "error": f"{exc}\n{traceback.format_exc()}"})
                if options.get("stop_on_error", False):
                    raise
            progress_cb(i, len(paths))
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(paths))) as ex:
            future_to_path = {ex.submit(run_one, p): p for p in paths}
            done = 0
            for fut in as_completed(future_to_path):
                _check_cancel(cancel_event)
                p = future_to_path[fut]
                done += 1
                try:
                    manifests.append(str(fut.result()))
                    message_cb(f"Finished LIF: {p.name} ({done}/{len(paths)})")
                except Exception as exc:
                    failures.append({"file": str(p), "error": f"{exc}\n{traceback.format_exc()}"})
                    if options.get("stop_on_error", False):
                        raise
                progress_cb(done, len(paths))
    return {"task": "lif", "manifests": manifests, "failures": failures}





# ============================================================
# IF Cell Threshold Explorer helpers
# ============================================================

IF_MAX_CHANNELS = 8
IF_DEFAULT_THRESHOLDS = [0.0, 2005.0, 610.0, 465.0, 0.0, 0.0, 0.0, 0.0]
# Very large polygons in QuPath exports are usually tissue/annotation ROIs, not single cells.
# Keep them out of the threshold overlay by default because filled overlays can hide the IF image.
IF_DEFAULT_MAX_CELL_AREA_PX = 50000.0
IF_STATUS_COLORS = {
    "C1+": QColor(0, 90, 255),
    "C2+": QColor(0, 180, 80),
    "C3+": QColor(230, 40, 40),
    "C4+": QColor(220, 40, 220),
    "C5+": QColor(255, 180, 0),
    "C6+": QColor(0, 210, 210),
    "C7+": QColor(255, 120, 0),
    "C8+": QColor(190, 190, 190),
    "Multi+": QColor(255, 220, 0),
    "Negative": QColor(150, 150, 150),
}


def _if_default_cache_path(image_path: Optional[str], geojson_path: Optional[str]) -> Path:
    """Return the default SQLite cache path for one image + GeoJSON pair."""
    if geojson_path:
        g = Path(geojson_path)
        return g.with_name(f"{g.stem}_IF_cells_cache.sqlite")
    if image_path:
        im = Path(image_path)
        return im.with_name(f"{im.stem}_IF_cells_cache.sqlite")
    return Path.cwd() / "IF_cells_cache.sqlite"


def _if_json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _iter_geojson_features_stream(path: str, chunk_size: int = 1024 * 1024):
    """Yield GeoJSON features without loading the whole file into memory.

    The common QuPath export is a FeatureCollection with a large features array.
    This parser finds the features array and decodes one feature object at a time.
    It intentionally avoids ijson so the executable does not need an extra dependency.
    """
    path = str(path)
    decoder = json.JSONDecoder()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        buf = ""
        found_features = False
        # Find FeatureCollection.features array.
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buf += chunk
            m = re.search(r'"features"\s*:\s*\[', buf)
            if m:
                buf = buf[m.end():]
                found_features = True
                break
            # Keep a tail in case the pattern is split across chunks.
            if len(buf) > 200_000:
                buf = buf[-200_000:]

        if not found_features:
            # Fallback for smaller/simpler GeoJSON files. This is not intended
            # for 900 MB files, but keeps compatibility with single Feature or list files.
            f.seek(0)
            data = json.load(f)
            if isinstance(data, dict) and data.get("type") == "Feature":
                yield data
            elif isinstance(data, dict) and "geometry" in data:
                yield {"type": "Feature", "geometry": data.get("geometry"), "properties": data.get("properties", {})}
            elif isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        yield obj
            return

        while True:
            # Ensure we have a non-empty buffer unless EOF.
            while not buf:
                chunk = f.read(chunk_size)
                if not chunk:
                    return
                buf += chunk

            # Skip whitespace and comma separators.
            stripped = buf.lstrip()
            if len(stripped) != len(buf):
                buf = stripped
            while buf.startswith(','):
                buf = buf[1:].lstrip()

            if buf.startswith(']'):
                return

            # Decode one full feature. If incomplete, read more chunks.
            while True:
                try:
                    feature, idx = decoder.raw_decode(buf)
                    buf = buf[idx:]
                    if isinstance(feature, dict):
                        yield feature
                    break
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buf += chunk


def _if_measurement_pairs_from_props(props: Any):
    """Yield (name, value) numeric measurement pairs from QuPath/generic properties."""
    if props is None:
        return
    if isinstance(props, dict):
        # QuPath commonly exports measurements as a list of {name, value}.
        measurements = props.get("measurements") or props.get("Measurements")
        if isinstance(measurements, list):
            for m in measurements:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("Name") or m.get("key") or m.get("measurement")
                    value = m.get("value") if "value" in m else m.get("Value")
                    if name is not None:
                        try:
                            yield str(name), float(value)
                        except Exception:
                            pass
        # Direct numeric properties, including keys like Cell: FAP (C2): Mean.
        for k, v in props.items():
            if k in ("geometry", "classification", "measurements", "Measurements"):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                try:
                    yield str(k), float(v)
                except Exception:
                    pass
            elif isinstance(v, dict):
                for kk, vv in _if_measurement_pairs_from_props(v):
                    yield f"{k}.{kk}", vv
    elif isinstance(props, list):
        for item in props:
            for kk, vv in _if_measurement_pairs_from_props(item):
                yield kk, vv


def _if_extract_channel_means(properties: Dict[str, Any], max_channels: int = IF_MAX_CHANNELS) -> List[Optional[float]]:
    """Extract channel mean values from QuPath cell GeoJSON properties.

    It detects names such as:
        Cell: FAP (C2): Mean
        Cell: aSMA (C3): Mean
        C4 Mean
    and stores them as zero-based C0..C7.
    """
    values: List[Optional[float]] = [None] * int(max_channels)
    for name, value in _if_measurement_pairs_from_props(properties or {}):
        lname = str(name).lower()
        if "mean" not in lname:
            continue
        m = re.search(r"\(\s*c\s*(\d+)\s*\)", str(name), flags=re.I)
        if not m:
            m = re.search(r"\bc\s*(\d+)\b", str(name), flags=re.I)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < max_channels:
            values[idx] = float(value)
    return values




def _if_raw_crop_to_yxc(arr: np.ndarray, axes: str = None) -> Tuple[np.ndarray, str]:
    """Return a raw crop in YX or YXC form without intensity normalization.

    This is different from display helpers: it preserves the original numeric
    values so threshold exploration uses scientific IF intensities.
    For Z/T stacks, the first Z/T plane is used. Channels are preserved.
    """
    arr = np.asarray(arr)
    axes = (axes or "").strip()

    if axes and len(axes) == arr.ndim and "Y" in axes and "X" in axes:
        slicer = []
        kept = []
        for ax in axes:
            if ax in ("Y", "X", "C", "S"):
                slicer.append(slice(None))
                kept.append(ax)
            else:
                slicer.append(0)
        arr = np.asarray(arr[tuple(slicer)])
        axes2 = "".join(kept)
        order = [axes2.index("Y"), axes2.index("X")]
        out_axes = "YX"
        if "C" in axes2:
            order.append(axes2.index("C"))
            out_axes += "C"
        elif "S" in axes2:
            order.append(axes2.index("S"))
            out_axes += "C"
        arr = np.transpose(arr, order)
        return np.asarray(arr), out_axes

    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr, "YX"
    if arr.ndim == 3:
        # Common scientific layouts: CYX or YXC.
        if arr.shape[-1] <= IF_MAX_CHANNELS:
            return arr, "YXC"
        if arr.shape[0] <= IF_MAX_CHANNELS:
            return np.moveaxis(arr, 0, -1), "YXC"
    raise ValueError(f"Unsupported raw IF crop shape for measurement: shape={arr.shape}, axes={axes}")


def _if_mask_from_rings(rings: List[List[Tuple[float, float]]], x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """Rasterize cell rings into a boolean mask in local crop coordinates."""
    from PIL import Image, ImageDraw
    w = int(w); h = int(h)
    if w <= 0 or h <= 0:
        return np.zeros((0, 0), dtype=bool)
    im = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(im)
    for ring in rings or []:
        pts = []
        for pt in ring or []:
            try:
                x, y = float(pt[0]) - float(x0), float(pt[1]) - float(y0)
                if np.isfinite(x) and np.isfinite(y):
                    pts.append((x, y))
            except Exception:
                continue
        if len(pts) >= 3:
            draw.polygon(pts, outline=1, fill=1)
    return np.asarray(im, dtype=np.uint8) > 0



def _if_load_measurement_backend(image_path: str) -> "ImageBackend":
    """Load image backend for IF measurement, preferring tifffile for OME-TIFF.

    OpenSlide is excellent for RGB WSI viewing, but IF OME-TIFF measurement needs
    raw channel arrays. Therefore TIFF/OME-TIFF files are opened with tifffile
    first whenever possible.
    """
    path = str(image_path)
    b = ImageBackend()
    lower_name = Path(path).name.lower()
    if _has_ext(lower_name, TIFF_EXTENSIONS):
        try:
            w, h, res, mpp = b._probe_tifffile(path)
            b.path = path
            b.path_obj = Path(path)
            b.reader = "tifffile"
            b.file_kind = "tiff"
            b.slide_dims = (int(w), int(h))
            b.source_resolution = res
            b.source_mpp = mpp
            b.openslide_props = {}
            return b
        except Exception:
            b.close()
    return b.load(path)


def _if_measure_cell_channel_means(backend: "ImageBackend", rings: List[List[Tuple[float, float]]],
                                   bbox: Tuple[float, float, float, float],
                                   max_channels: int = IF_MAX_CHANNELS) -> List[Optional[float]]:
    """Measure mean intensity inside one cell polygon for each IF channel.

    Only the cell bounding box is read from the source image. No RGB conversion
    or display normalization is applied. Values therefore correspond to the
    raw mean intensity inside the cell mask.
    """
    values: List[Optional[float]] = [None] * int(max_channels)
    if backend is None or not getattr(backend, "slide_dims", None):
        return values
    minx, miny, maxx, maxy = [float(v) for v in bbox]
    full_w, full_h = backend.slide_dims
    x0 = max(0, int(math.floor(minx)))
    y0 = max(0, int(math.floor(miny)))
    x1 = min(int(full_w), int(math.ceil(maxx)) + 1)
    y1 = min(int(full_h), int(math.ceil(maxy)) + 1)
    w = max(1, int(x1 - x0))
    h = max(1, int(y1 - y0))
    if w <= 1 or h <= 1:
        return values

    raw, axes, _info = backend.crop_raw(x0, y0, w, h)
    arr, arr_axes = _if_raw_crop_to_yxc(raw, axes)
    mask = _if_mask_from_rings(rings, x0, y0, arr.shape[1], arr.shape[0])
    if mask.size == 0 or not np.any(mask):
        return values

    if arr.ndim == 2:
        vals = np.asarray(arr)[mask]
        if vals.size:
            values[0] = float(np.mean(vals, dtype=np.float64))
        return values

    if arr.ndim == 3:
        n_channels = min(int(arr.shape[-1]), int(max_channels))
        for c in range(n_channels):
            vals = np.asarray(arr[:, :, c])[mask]
            if vals.size:
                values[c] = float(np.mean(vals, dtype=np.float64))
        return values

    return values


def _if_ring_bbox(rings: List[List[Tuple[float, float]]]):
    xs, ys = [], []
    for ring in rings or []:
        for x, y in ring or []:
            xs.append(float(x)); ys.append(float(y))
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)



def _if_scale_rings(rings: List[List[Tuple[float, float]]], scale_x: float = 1.0, scale_y: float = 1.0) -> List[List[Tuple[float, float]]]:
    """Scale GeoJSON coordinates into the loaded image coordinate system.

    QuPath/OME workflows sometimes export cell GeoJSON at a different pyramid
    level than the image opened in the viewer. The cache must store and measure
    cells in the same coordinate system as the loaded IF image. Scaling is
    applied directly around image origin (0, 0); no min-coordinate shift is used.
    """
    try:
        sx = float(scale_x)
        sy = float(scale_y)
    except Exception:
        sx = sy = 1.0
    if not np.isfinite(sx) or sx <= 0:
        sx = 1.0
    if not np.isfinite(sy) or sy <= 0:
        sy = 1.0
    if abs(sx - 1.0) < 1e-12 and abs(sy - 1.0) < 1e-12:
        return rings
    out = []
    for ring in rings or []:
        rr = []
        for pt in ring or []:
            try:
                x, y = float(pt[0]), float(pt[1])
                if np.isfinite(x) and np.isfinite(y):
                    rr.append((x * sx, y * sy))
            except Exception:
                continue
        if len(rr) >= 2:
            out.append(rr)
    return out


def _if_probe_geojson_extent(geojson_path: str, cancel_event=None, progress_cb=None, message_cb=None) -> Dict[str, Any]:
    """Stream the GeoJSON once to estimate its coordinate extent without storing geometries."""
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    count = 0
    skipped = 0
    for feature in _iter_geojson_features_stream(geojson_path):
        _check_cancel(cancel_event)
        if not isinstance(feature, dict):
            skipped += 1
            continue
        geom = feature.get("geometry") if "geometry" in feature else feature
        rings = _geojson_geometry_to_rings(geom)
        bbox = _if_ring_bbox(rings)
        if bbox is None:
            skipped += 1
            continue
        bx0, by0, bx1, by1 = bbox
        minx = min(minx, float(bx0))
        miny = min(miny, float(by0))
        maxx = max(maxx, float(bx1))
        maxy = max(maxy, float(by1))
        count += 1
        if count % 5000 == 0:
            if message_cb is not None:
                message_cb(f"Scanning GeoJSON extent: {count:,} cells")
            if progress_cb is not None:
                progress_cb(0, 0)
    if count <= 0 or not np.isfinite(minx) or not np.isfinite(maxx):
        return {"cell_count_probe": count, "skipped_probe": skipped, "valid": False}
    return {
        "cell_count_probe": count,
        "skipped_probe": skipped,
        "valid": True,
        "minx": float(minx), "miny": float(miny), "maxx": float(maxx), "maxy": float(maxy),
        "width": float(maxx - minx), "height": float(maxy - miny),
    }


def _if_auto_coordinate_scale(image_dims: Tuple[int, int], geo_extent: Dict[str, Any]) -> Dict[str, Any]:
    """Conservatively detect common GeoJSON-to-image pyramid scale mismatches.

    Automatic scaling is applied only when GeoJSON coordinates are clearly larger
    than the image and close to a common factor such as 2x, 4x, 8x, etc. If the
    GeoJSON extent is smaller than the image, no automatic upscaling is applied
    because this may simply mean the cells occupy only part of the image.
    """
    full_w, full_h = [float(v) for v in image_dims]
    if not geo_extent or not geo_extent.get("valid") or full_w <= 0 or full_h <= 0:
        return {"scale_x": 1.0, "scale_y": 1.0, "ratio_x": 1.0, "ratio_y": 1.0, "mode": "no_extent"}

    maxx = max(abs(float(geo_extent.get("maxx", 0.0))), abs(float(geo_extent.get("minx", 0.0))))
    maxy = max(abs(float(geo_extent.get("maxy", 0.0))), abs(float(geo_extent.get("miny", 0.0))))
    rx = maxx / full_w if full_w else 1.0
    ry = maxy / full_h if full_h else 1.0
    candidates = [2.0, 4.0, 8.0, 16.0, 32.0]

    def choose_scale(ratio: float) -> Tuple[float, str]:
        try:
            ratio = float(ratio)
        except Exception:
            return 1.0, "invalid"
        if not np.isfinite(ratio) or ratio <= 0:
            return 1.0, "invalid"
        if ratio <= 1.15:
            return 1.0, "same_or_partial_extent"
        best = min(candidates, key=lambda c: abs(c - ratio) / c)
        rel_err = abs(best - ratio) / best
        if rel_err <= 0.30:
            return 1.0 / best, f"common_downsample_{best:g}x"
        return 1.0, f"uncertain_ratio_{ratio:.3g}"

    sx, mode_x = choose_scale(rx)
    sy, mode_y = choose_scale(ry)

    # If only one axis was confidently detected but both ratios are close,
    # use the same scale for both axes to avoid small anisotropic artifacts.
    if sx != 1.0 and sy == 1.0 and abs(rx - ry) / max(rx, ry, 1.0) <= 0.20:
        sy = sx
        mode_y = mode_x + "_matched"
    if sy != 1.0 and sx == 1.0 and abs(rx - ry) / max(rx, ry, 1.0) <= 0.20:
        sx = sy
        mode_x = mode_y + "_matched"

    mode = "scaled" if (abs(sx - 1.0) > 1e-9 or abs(sy - 1.0) > 1e-9) else "unscaled"
    return {
        "scale_x": float(sx), "scale_y": float(sy),
        "ratio_x": float(rx), "ratio_y": float(ry),
        "mode": mode, "mode_x": mode_x, "mode_y": mode_y,
    }


def _if_polygon_area_centroid(ring: List[Tuple[float, float]]):
    """Return signed area and centroid for one ring."""
    if not ring or len(ring) < 3:
        return 0.0, None
    pts = list(ring)
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    a2 = 0.0
    cx = 0.0
    cy = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        cross = float(x0) * float(y1) - float(x1) * float(y0)
        a2 += cross
        cx += (float(x0) + float(x1)) * cross
        cy += (float(y0) + float(y1)) * cross
    if abs(a2) < 1e-9:
        return 0.0, None
    area = a2 / 2.0
    return area, (cx / (3.0 * a2), cy / (3.0 * a2))


def _if_rings_area_centroid(rings: List[List[Tuple[float, float]]]):
    total_area = 0.0
    sx = 0.0
    sy = 0.0
    for ring in rings or []:
        area, cen = _if_polygon_area_centroid(ring)
        if cen is None:
            continue
        weight = abs(float(area))
        total_area += weight
        sx += cen[0] * weight
        sy += cen[1] * weight
    bbox = _if_ring_bbox(rings)
    if total_area > 0:
        return total_area, sx / total_area, sy / total_area
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        return 0.0, (minx + maxx) / 2.0, (miny + maxy) / 2.0
    return 0.0, 0.0, 0.0


def _if_connect_cache(cache_path: str):
    import sqlite3
    # timeout avoids transient OneDrive / antivirus / previous-reader locks.
    con = sqlite3.connect(str(cache_path), timeout=30)
    con.row_factory = sqlite3.Row
    return con


def _if_initialize_cache_schema(con):
    channel_cols = ",\n            ".join([f"c{i}_mean REAL" for i in range(IF_MAX_CHANNELS)])
    con.executescript(f"""
        -- DELETE journal avoids persistent -wal/-shm files, which are often locked by
        -- OneDrive/antivirus on Windows while rebuilding the cache.
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS cells (
            id INTEGER PRIMARY KEY,
            source_id TEXT,
            class_name TEXT,
            minx REAL, miny REAL, maxx REAL, maxy REAL,
            cx REAL, cy REAL,
            area REAL,
            {channel_cols},
            geom_z BLOB
        );
        CREATE INDEX IF NOT EXISTS idx_cells_bbox ON cells(minx, maxx, miny, maxy);
        CREATE INDEX IF NOT EXISTS idx_cells_centroid ON cells(cx, cy);
        CREATE TABLE IF NOT EXISTS large_rois (
            id INTEGER PRIMARY KEY,
            source_id TEXT,
            class_name TEXT,
            minx REAL, miny REAL, maxx REAL, maxy REAL,
            cx REAL, cy REAL,
            area REAL,
            geom_z BLOB
        );
        CREATE INDEX IF NOT EXISTS idx_large_rois_bbox ON large_rois(minx, maxx, miny, maxy);
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    con.commit()


def _if_build_cell_cache(image_path: str, geojson_path: str, cache_path: str,
                         cancel_event=None, progress_cb=None, message_cb=None,
                         max_cell_area_px: float = IF_DEFAULT_MAX_CELL_AREA_PX) -> Dict[str, Any]:
    """Build a measured SQLite cache from a huge cell GeoJSON + IF image.

    The GeoJSON is streamed feature-by-feature, so the full 900 MB file is never
    loaded into RAM. For each cell, the code reads only the cell bounding box
    from the raw multichannel image, rasterizes the polygon as a mask, and saves
    mean_C1..mean_C8 in SQLite. Later threshold changes only compare cached
    means against slider values.
    """
    import sqlite3
    import zlib
    image_path = str(image_path or "")
    geojson_path = str(geojson_path or "")
    cache_path = str(cache_path or _if_default_cache_path(image_path, geojson_path))
    try:
        max_cell_area_px = float(max_cell_area_px)
    except Exception:
        max_cell_area_px = float(IF_DEFAULT_MAX_CELL_AREA_PX)
    if max_cell_area_px <= 0:
        max_cell_area_px = float('inf')

    if not image_path or not Path(image_path).exists():
        raise FileNotFoundError("Select a valid IF image before building the measured cache.")
    if not geojson_path or not Path(geojson_path).exists():
        raise FileNotFoundError("Select a valid cell GeoJSON before building the measured cache.")

    requested_cache_path = Path(cache_path)
    requested_cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Always build into a NEW timestamped SQLite file and only switch the GUI to
    # that file after the job finishes. This avoids two common Windows problems:
    #   1) deleting/overwriting a cache that is still locked by OneDrive/AV/SQLite;
    #   2) the GUI reading a partially-built cache and reporting only the cells
    #      inserted so far (for example "ROI cells: 7,000" while the job is still running).
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    build_cache_path = requested_cache_path.with_name(
        f"{requested_cache_path.stem}_measured_{stamp}{requested_cache_path.suffix}"
    )
    cache_note = f"Building new measured cache: {build_cache_path.name}"
    cache_path = str(build_cache_path)

    if message_cb is not None:
        message_cb(cache_note)

    backend = _if_load_measurement_backend(image_path)

    if message_cb is not None:
        message_cb("Scanning GeoJSON coordinate extent for image/GeoJSON scale check...")
    geo_extent = _if_probe_geojson_extent(
        geojson_path, cancel_event=cancel_event, progress_cb=progress_cb, message_cb=message_cb
    )
    scale_info = _if_auto_coordinate_scale(getattr(backend, "slide_dims", (1, 1)), geo_extent)
    geo_scale_x = float(scale_info.get("scale_x", 1.0))
    geo_scale_y = float(scale_info.get("scale_y", 1.0))
    total_expected = int(geo_extent.get("cell_count_probe", 0) or 0)
    if message_cb is not None:
        message_cb(
            "GeoJSON scale check: "
            f"image={getattr(backend, 'slide_dims', None)} | "
            f"geo max=({geo_extent.get('maxx', '?')}, {geo_extent.get('maxy', '?')}) | "
            f"ratio=({float(scale_info.get('ratio_x', 1)):.3g}, {float(scale_info.get('ratio_y', 1)):.3g}) | "
            f"scale=({geo_scale_x:.6g}, {geo_scale_y:.6g}) | {scale_info.get('mode')} | "
            f"valid cells to measure≈{total_expected:,}"
        )

    con = sqlite3.connect(cache_path)
    try:
        _if_initialize_cache_schema(con)
        con.execute("DELETE FROM cells")
        con.execute("DELETE FROM meta")
        con.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", [
            ("image_path", image_path),
            ("geojson_path", geojson_path),
            ("created", datetime.now().isoformat(timespec="seconds")),
            ("software", f"{APP_NAME} v{APP_VERSION}"),
            ("format", "IF Cell Threshold Explorer measured cache v3"),
            ("build_complete", "0"),
            ("requested_cache_path", str(requested_cache_path)),
            ("actual_cache_path", str(build_cache_path)),
            ("cache_note", cache_note),
            ("measurement", "mean intensity measured from raw image pixels inside each GeoJSON cell polygon"),
            ("max_cell_area_px", str(max_cell_area_px)),
            ("large_area_filter", "features with area_px > max_cell_area_px are treated as large annotations/ROIs and skipped"),
            ("image_reader", str(getattr(backend, "reader", "unknown"))),
            ("image_dims", "x".join(str(v) for v in (getattr(backend, "slide_dims", None) or ()))),
            ("geojson_extent_minx", str(geo_extent.get("minx", ""))),
            ("geojson_extent_miny", str(geo_extent.get("miny", ""))),
            ("geojson_extent_maxx", str(geo_extent.get("maxx", ""))),
            ("geojson_extent_maxy", str(geo_extent.get("maxy", ""))),
            ("geojson_coord_ratio_x", str(scale_info.get("ratio_x", ""))),
            ("geojson_coord_ratio_y", str(scale_info.get("ratio_y", ""))),
            ("geojson_to_image_scale_x", str(geo_scale_x)),
            ("geojson_to_image_scale_y", str(geo_scale_y)),
            ("geojson_scale_mode", str(scale_info.get("mode", ""))),
            ("geojson_scale_mode_x", str(scale_info.get("mode_x", ""))),
            ("geojson_scale_mode_y", str(scale_info.get("mode_y", ""))),
        ])

        insert_sql = """
            INSERT INTO cells(
                source_id, class_name, minx, miny, maxx, maxy, cx, cy, area,
                c0_mean, c1_mean, c2_mean, c3_mean, c4_mean, c5_mean, c6_mean, c7_mean, geom_z
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        large_roi_sql = """
            INSERT INTO large_rois(source_id, class_name, minx, miny, maxx, maxy, cx, cy, area, geom_z)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        batch = []
        count = 0
        skipped = 0
        measured = 0
        measurement_failed = 0
        skipped_large_area = 0
        last_message_t = datetime.now()

        if message_cb is not None:
            message_cb(f"Starting raw IF measurement for approximately {total_expected:,} cells...")
        if progress_cb is not None:
            progress_cb(0, total_expected if total_expected > 0 else 0)

        for feature in _iter_geojson_features_stream(geojson_path):
            _check_cancel(cancel_event)
            if not isinstance(feature, dict):
                skipped += 1
                continue
            props = feature.get("properties", {}) or {}
            geom = feature.get("geometry") if "geometry" in feature else feature
            rings = _geojson_geometry_to_rings(geom)
            if not rings:
                skipped += 1
                continue
            # Store and measure in the loaded image coordinate system.
            rings = _if_scale_rings(rings, geo_scale_x, geo_scale_y)
            bbox = _if_ring_bbox(rings)
            if bbox is None:
                skipped += 1
                continue
            minx, miny, maxx, maxy = bbox
            area, cx, cy = _if_rings_area_centroid(rings)

            # Large polygons are usually tissue/annotation ROIs, not individual cells.
            # Store them separately so tissue-level pixel thresholding can be limited
            # inside the general tissue annotation without treating it as a cell.
            try:
                if float(area) > float(max_cell_area_px):
                    skipped_large_area += 1
                    skipped += 1
                    cls_large = _geojson_feature_class_name(feature, props)
                    source_id_large = str(feature.get("id", props.get("id", props.get("object_id", f"large_roi_{skipped_large_area}"))))
                    geom_z_large = sqlite3.Binary(zlib.compress(_if_json_dumps_compact(rings).encode("utf-8"), level=1))
                    con.execute(large_roi_sql, (source_id_large, cls_large, minx, miny, maxx, maxy, cx, cy, area, geom_z_large))
                    if skipped_large_area % 10 == 0:
                        con.commit()
                    continue
            except Exception:
                pass

            # Primary path: calculate means directly from the raw IF image.
            # Fallback only supports GeoJSONs that already contain measurements.
            try:
                means = _if_measure_cell_channel_means(backend, rings, bbox, IF_MAX_CHANNELS)
                if any(v is not None for v in means):
                    measured += 1
                else:
                    props_means = _if_extract_channel_means(props, IF_MAX_CHANNELS)
                    if any(v is not None for v in props_means):
                        means = props_means
                    else:
                        measurement_failed += 1
            except Exception:
                props_means = _if_extract_channel_means(props, IF_MAX_CHANNELS)
                means = props_means
                measurement_failed += 1

            cls = _geojson_feature_class_name(feature, props)
            source_id = str(feature.get("id", props.get("id", props.get("object_id", count + 1))))
            geom_z = sqlite3.Binary(zlib.compress(_if_json_dumps_compact(rings).encode("utf-8"), level=1))
            row = [source_id, cls, minx, miny, maxx, maxy, cx, cy, area] + means + [geom_z]
            batch.append(row)
            count += 1

            # Lightweight heartbeat: calculating means from raw IF pixels can be
            # slow, especially when each cell requires a compressed OME-TIFF
            # region read. Emit progress more often than SQLite commits so the
            # user can see the job is alive.
            if count % 50 == 0:
                if message_cb is not None:
                    message_cb(
                        f"Measuring IF cells: {count:,}/{total_expected:,} processed | "
                        f"measured {measured:,} | failed/no mask {measurement_failed:,} | skipped {skipped:,} | large ROIs skipped {skipped_large_area:,}"
                    )
                if progress_cb is not None:
                    progress_cb(count, total_expected if total_expected > 0 else 0)

            if len(batch) >= 500:
                con.executemany(insert_sql, batch)
                con.commit()
                batch.clear()
                if message_cb is not None:
                    message_cb(
                        f"Measuring IF cells: {count:,} cached | measured {measured:,} | "
                        f"failed/no mask {measurement_failed:,} | skipped {skipped:,} | large ROIs skipped {skipped_large_area:,}"
                    )
                if progress_cb is not None:
                    progress_cb(count, total_expected if total_expected > 0 else 0)

        if batch:
            con.executemany(insert_sql, batch)
            con.commit()
        con.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", [
            ("cell_count", str(count)),
            ("skipped_count", str(skipped)),
            ("measured_count", str(measured)),
            ("measurement_failed_count", str(measurement_failed)),
            ("skipped_large_area_count", str(skipped_large_area)),
            ("large_roi_count", str(skipped_large_area)),
            ("build_complete", "1"),
            ("completed", datetime.now().isoformat(timespec="seconds")),
        ])
        con.commit()
        try:
            con.execute("ANALYZE")
            con.commit()
        except Exception:
            pass
        if progress_cb is not None:
            progress_cb(count, total_expected if total_expected > 0 else count)
        return {
            "cache_path": cache_path,
            "cell_count": count,
            "skipped_count": skipped,
            "measured_count": measured,
            "measurement_failed_count": measurement_failed,
            "skipped_large_area_count": skipped_large_area,
            "large_roi_count": skipped_large_area,
            "max_cell_area_px": max_cell_area_px,
            "geo_scale_x": geo_scale_x,
            "geo_scale_y": geo_scale_y,
            "geo_ratio_x": scale_info.get("ratio_x", 1.0),
            "geo_ratio_y": scale_info.get("ratio_y", 1.0),
            "geo_scale_mode": scale_info.get("mode", ""),
            "task": "if_cache",
        }
    finally:
        try:
            backend.close()
        except Exception:
            pass
        con.close()



# ============================================================
# Fast CSV-backed IF cache builder
# ============================================================

def _if_norm_text_id(value: Any) -> str:
    """Normalize object identifiers for robust GeoJSON/CSV matching."""
    if value is None:
        return ""
    return str(value).strip().strip('"').strip().lower()


def _if_safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            return None
        if "," in text and "." not in text:
            text = text.replace(",", ".")
        v = float(text)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _if_csv_column(fieldnames: Sequence[str], exact_names: Sequence[str] = (), regexes: Sequence[str] = ()) -> Optional[str]:
    fields = list(fieldnames or [])
    low_map = {str(c).strip().lower(): c for c in fields}
    for name in exact_names or []:
        key = str(name).strip().lower()
        if key in low_map:
            return low_map[key]
    for pattern in regexes or []:
        rx = re.compile(pattern, flags=re.I)
        for c in fields:
            if rx.search(str(c)):
                return c
    return None


def _if_extract_csv_cell_means(row: Dict[str, Any], max_channels: int = IF_MAX_CHANNELS) -> List[Optional[float]]:
    """Extract whole-cell channel means from a QuPath measurement CSV row."""
    values: List[Optional[float]] = [None] * int(max_channels)
    fallback: List[Optional[float]] = [None] * int(max_channels)
    for key, value in (row or {}).items():
        name = str(key or "")
        lname = name.lower()
        if "mean" not in lname:
            continue
        m = re.search(r"\(\s*c\s*(\d+)\s*\)", name, flags=re.I)
        if not m:
            m = re.search(r"\bc\s*(\d+)\b", name, flags=re.I)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if not (0 <= idx < max_channels):
            continue
        v = _if_safe_float(value)
        if v is None:
            continue
        # Use whole-cell measurements only. Do not use Nucleus/Cytoplasm/Membrane for thresholding.
        if re.search(r"(^|[^a-z])cell\s*:", lname, flags=re.I):
            values[idx] = float(v)
        elif not any(comp in lname for comp in ("nucleus", "cytoplasm", "membrane")):
            fallback[idx] = float(v)
    for i in range(int(max_channels)):
        if values[i] is None and fallback[i] is not None:
            values[i] = fallback[i]
    return values


class _IFCsvMeasurementIndex:
    def __init__(self, rows: List[Dict[str, Any]], by_id: Dict[str, int], grid: Dict[Tuple[int, int], List[int]], grid_size: float = 25.0):
        self.rows = rows
        self.by_id = by_id
        self.grid = grid
        self.grid_size = float(grid_size)
        self.used = set()
        self.id_matches = 0
        self.centroid_matches = 0
        self.order_matches = 0
        self.unmatched = 0

    def _grid_key(self, x: float, y: float) -> Tuple[int, int]:
        gs = max(1.0, self.grid_size)
        return int(math.floor(float(x) / gs)), int(math.floor(float(y) / gs))

    def match(self, source_id: str, cx: float, cy: float, order_index: int, max_dist_px: float = 35.0) -> Optional[Dict[str, Any]]:
        sid = _if_norm_text_id(source_id)
        if sid and sid in self.by_id:
            idx = self.by_id[sid]
            if idx not in self.used:
                self.used.add(idx)
                self.id_matches += 1
                return self.rows[idx]
        best_idx = None
        best_d2 = None
        try:
            gx, gy = self._grid_key(float(cx), float(cy))
            search_r = max(1, int(math.ceil(float(max_dist_px) / max(1.0, self.grid_size))) + 1)
            for yy in range(gy - search_r, gy + search_r + 1):
                for xx in range(gx - search_r, gx + search_r + 1):
                    for idx in self.grid.get((xx, yy), []):
                        if idx in self.used:
                            continue
                        item = self.rows[idx]
                        for xcol, ycol in (("cx_px", "cy_px"), ("cx_raw", "cy_raw")):
                            xval = item.get(xcol)
                            yval = item.get(ycol)
                            if xval is None or yval is None:
                                continue
                            dx = float(xval) - float(cx)
                            dy = float(yval) - float(cy)
                            d2 = dx * dx + dy * dy
                            if best_d2 is None or d2 < best_d2:
                                best_d2 = d2
                                best_idx = idx
            if best_idx is not None and best_d2 is not None and best_d2 <= float(max_dist_px) ** 2:
                self.used.add(best_idx)
                self.centroid_matches += 1
                return self.rows[best_idx]
        except Exception:
            pass
        try:
            idx = int(order_index)
            if 0 <= idx < len(self.rows) and idx not in self.used:
                self.used.add(idx)
                self.order_matches += 1
                return self.rows[idx]
        except Exception:
            pass
        self.unmatched += 1
        return None


def _if_load_csv_measurement_index(csv_path: str, backend: Optional["ImageBackend"] = None, message_cb=None) -> _IFCsvMeasurementIndex:
    csv_path = str(csv_path or "")
    if not csv_path or not Path(csv_path).exists():
        raise FileNotFoundError("Select a valid QuPath measurements CSV.")
    mpp_x = mpp_y = None
    try:
        if backend is not None and getattr(backend, "source_mpp", None):
            mpp_x, mpp_y = backend.source_mpp
            mpp_x = float(mpp_x) if mpp_x else None
            mpp_y = float(mpp_y) if mpp_y else None
    except Exception:
        mpp_x = mpp_y = None
    rows: List[Dict[str, Any]] = []
    by_id: Dict[str, int] = {}
    grid: Dict[Tuple[int, int], List[int]] = {}
    grid_size = 25.0
    with open(csv_path, "r", newline="", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(20000)
        f.seek(0)
        delimiter = "\t" if sample.count("\t") > sample.count(",") * 2 else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        id_col = _if_csv_column(fieldnames, exact_names=("Object ID", "object_id", "ID", "id"), regexes=(r"object\s*id",))
        class_col = _if_csv_column(fieldnames, exact_names=("Classification", "Class", "class"), regexes=(r"classification",))
        name_col = _if_csv_column(fieldnames, exact_names=("Name", "name"), regexes=(r"^name$",))
        x_col = _if_csv_column(fieldnames, exact_names=("Centroid X µm", "Centroid X um", "Centroid X", "Centroid X px"), regexes=(r"centroid\s*x",))
        y_col = _if_csv_column(fieldnames, exact_names=("Centroid Y µm", "Centroid Y um", "Centroid Y", "Centroid Y px"), regexes=(r"centroid\s*y",))
        if message_cb is not None:
            message_cb(f"Loading QuPath measurements CSV: id_col={id_col or 'none'}, x_col={x_col or 'none'}, y_col={y_col or 'none'}, mpp=({mpp_x or 'unknown'}, {mpp_y or 'unknown'})")
        for row in reader:
            means = _if_extract_csv_cell_means(row, IF_MAX_CHANNELS)
            oid = row.get(id_col, "") if id_col else ""
            cx_raw = _if_safe_float(row.get(x_col)) if x_col else None
            cy_raw = _if_safe_float(row.get(y_col)) if y_col else None
            cx_px = cy_px = None
            if cx_raw is not None and cy_raw is not None and mpp_x and mpp_y and mpp_x > 0 and mpp_y > 0:
                cx_px = float(cx_raw) / float(mpp_x)
                cy_px = float(cy_raw) / float(mpp_y)
            elif cx_raw is not None and cy_raw is not None:
                cx_px = float(cx_raw)
                cy_px = float(cy_raw)
            cls = ""
            if class_col and row.get(class_col):
                cls = str(row.get(class_col))
            elif name_col and row.get(name_col):
                cls = str(row.get(name_col))
            idx = len(rows)
            item = {"object_id": str(oid or ""), "class_name": cls, "cx_raw": cx_raw, "cy_raw": cy_raw, "cx_px": cx_px, "cy_px": cy_px, "means": means}
            rows.append(item)
            nid = _if_norm_text_id(oid)
            if nid and nid not in by_id:
                by_id[nid] = idx
            for xv, yv in ((cx_px, cy_px), (cx_raw, cy_raw)):
                if xv is None or yv is None:
                    continue
                key = (int(math.floor(float(xv) / grid_size)), int(math.floor(float(yv) / grid_size)))
                grid.setdefault(key, []).append(idx)
            if message_cb is not None and idx > 0 and idx % 50000 == 0:
                message_cb(f"Loaded {idx:,} measurement rows from CSV...")
    if message_cb is not None:
        n_means = sum(1 for item in rows if any(v is not None for v in item.get("means", [])))
        message_cb(f"CSV measurement index ready: {len(rows):,} rows | rows with Cell mean C#={n_means:,} | object IDs={len(by_id):,}")
    return _IFCsvMeasurementIndex(rows, by_id, grid, grid_size=grid_size)


def _if_build_cell_cache_from_csv(image_path: str, geojson_path: str, csv_path: str, cache_path: str,
                                  cancel_event=None, progress_cb=None, message_cb=None,
                                  max_cell_area_px: float = IF_DEFAULT_MAX_CELL_AREA_PX) -> Dict[str, Any]:
    import sqlite3
    import zlib
    image_path = str(image_path or "")
    geojson_path = str(geojson_path or "")
    csv_path = str(csv_path or "")
    cache_path = str(cache_path or _if_default_cache_path(image_path, geojson_path))
    try:
        max_cell_area_px = float(max_cell_area_px)
    except Exception:
        max_cell_area_px = float(IF_DEFAULT_MAX_CELL_AREA_PX)
    if max_cell_area_px <= 0:
        max_cell_area_px = float('inf')
    if not image_path or not Path(image_path).exists():
        raise FileNotFoundError("Select a valid IF image before building the cache.")
    if not geojson_path or not Path(geojson_path).exists():
        raise FileNotFoundError("Select a valid cell GeoJSON before building the cache.")
    if not csv_path or not Path(csv_path).exists():
        raise FileNotFoundError("Select a valid QuPath measurements CSV.")
    requested_cache_path = Path(cache_path)
    requested_cache_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    build_cache_path = requested_cache_path.with_name(f"{requested_cache_path.stem}_CSVfast_{stamp}{requested_cache_path.suffix}")
    cache_path = str(build_cache_path)
    if message_cb is not None:
        message_cb(f"Building fast CSV-backed cache: {build_cache_path.name}")
    backend = _if_load_measurement_backend(image_path)
    try:
        if message_cb is not None:
            message_cb("Scanning GeoJSON coordinate extent for image/GeoJSON scale check...")
        geo_extent = _if_probe_geojson_extent(geojson_path, cancel_event=cancel_event, progress_cb=progress_cb, message_cb=message_cb)
        scale_info = _if_auto_coordinate_scale(getattr(backend, "slide_dims", (1, 1)), geo_extent)
        geo_scale_x = float(scale_info.get("scale_x", 1.0))
        geo_scale_y = float(scale_info.get("scale_y", 1.0))
        total_expected = int(geo_extent.get("cell_count_probe", 0) or 0)
        if message_cb is not None:
            message_cb(f"GeoJSON scale check: image={getattr(backend, 'slide_dims', None)} | ratio=({float(scale_info.get('ratio_x', 1)):.3g}, {float(scale_info.get('ratio_y', 1)):.3g}) | scale=({geo_scale_x:.6g}, {geo_scale_y:.6g}) | valid objects≈{total_expected:,}")
        csv_index = _if_load_csv_measurement_index(csv_path, backend=backend, message_cb=message_cb)
        con = sqlite3.connect(cache_path)
        try:
            _if_initialize_cache_schema(con)
            con.execute("DELETE FROM cells")
            con.execute("DELETE FROM large_rois")
            con.execute("DELETE FROM meta")
            con.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", [
                ("image_path", image_path), ("geojson_path", geojson_path), ("csv_path", csv_path),
                ("created", datetime.now().isoformat(timespec="seconds")), ("software", f"{APP_NAME} v{APP_VERSION}"),
                ("format", "IF Cell Threshold Explorer CSV-backed cache v1"), ("build_complete", "0"),
                ("requested_cache_path", str(requested_cache_path)), ("actual_cache_path", str(build_cache_path)),
                ("measurement", "mean intensity imported from QuPath measurements CSV Cell: ... (C#): Mean columns"),
                ("measurement_source", "csv"), ("max_cell_area_px", str(max_cell_area_px)),
                ("large_area_filter", "features with area_px > max_cell_area_px are treated as large annotations/ROIs and skipped"),
                ("image_reader", str(getattr(backend, "reader", "unknown"))),
                ("image_dims", "x".join(str(v) for v in (getattr(backend, "slide_dims", None) or ()))),
                ("image_mpp", "x".join(str(v) for v in (getattr(backend, "source_mpp", None) or ()))),
                ("geojson_coord_ratio_x", str(scale_info.get("ratio_x", ""))),
                ("geojson_coord_ratio_y", str(scale_info.get("ratio_y", ""))),
                ("geojson_to_image_scale_x", str(geo_scale_x)), ("geojson_to_image_scale_y", str(geo_scale_y)),
                ("geojson_scale_mode", str(scale_info.get("mode", ""))),
            ])
            insert_sql = """
                INSERT INTO cells(
                    source_id, class_name, minx, miny, maxx, maxy, cx, cy, area,
                    c0_mean, c1_mean, c2_mean, c3_mean, c4_mean, c5_mean, c6_mean, c7_mean, geom_z
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            large_roi_sql = """
                INSERT INTO large_rois(source_id, class_name, minx, miny, maxx, maxy, cx, cy, area, geom_z)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            batch = []
            count = 0
            feature_index = 0
            skipped = 0
            skipped_large_area = 0
            csv_matched = 0
            csv_with_means = 0
            csv_missing = 0
            if message_cb is not None:
                message_cb(f"Building cache from CSV measurements for approximately {total_expected:,} GeoJSON objects...")
            if progress_cb is not None:
                progress_cb(0, total_expected if total_expected > 0 else 0)
            for feature in _iter_geojson_features_stream(geojson_path):
                _check_cancel(cancel_event)
                if not isinstance(feature, dict):
                    skipped += 1; feature_index += 1; continue
                props = feature.get("properties", {}) or {}
                geom = feature.get("geometry") if "geometry" in feature else feature
                rings = _geojson_geometry_to_rings(geom)
                if not rings:
                    skipped += 1; feature_index += 1; continue
                rings = _if_scale_rings(rings, geo_scale_x, geo_scale_y)
                bbox = _if_ring_bbox(rings)
                if bbox is None:
                    skipped += 1; feature_index += 1; continue
                minx, miny, maxx, maxy = bbox
                area, cx, cy = _if_rings_area_centroid(rings)
                try:
                    if float(area) > float(max_cell_area_px):
                        skipped_large_area += 1; skipped += 1
                        cls_large = _geojson_feature_class_name(feature, props)
                        source_id_large = str(feature.get("id", props.get("id", props.get("object_id", f"large_roi_{skipped_large_area}"))))
                        geom_z_large = sqlite3.Binary(zlib.compress(_if_json_dumps_compact(rings).encode("utf-8"), level=1))
                        con.execute(large_roi_sql, (source_id_large, cls_large, minx, miny, maxx, maxy, cx, cy, area, geom_z_large))
                        if skipped_large_area % 10 == 0:
                            con.commit()
                        feature_index += 1; continue
                except Exception:
                    pass
                source_id = str(feature.get("id", props.get("id", props.get("object_id", feature_index + 1))))
                csv_item = csv_index.match(source_id, cx, cy, feature_index)
                if csv_item is not None:
                    csv_matched += 1
                    means = list(csv_item.get("means", [None] * IF_MAX_CHANNELS))[:IF_MAX_CHANNELS]
                    if any(v is not None for v in means):
                        csv_with_means += 1
                    else:
                        csv_missing += 1
                else:
                    means = [None] * IF_MAX_CHANNELS
                    csv_missing += 1
                cls_geo = _geojson_feature_class_name(feature, props)
                cls_csv = str(csv_item.get("class_name", "")) if csv_item else ""
                cls = cls_geo if cls_geo and cls_geo != "annotation" else (cls_csv or cls_geo or "Cell")
                geom_z = sqlite3.Binary(zlib.compress(_if_json_dumps_compact(rings).encode("utf-8"), level=1))
                batch.append([source_id, cls, minx, miny, maxx, maxy, cx, cy, area] + means + [geom_z])
                count += 1
                feature_index += 1
                if len(batch) >= 2000:
                    con.executemany(insert_sql, batch); con.commit(); batch.clear()
                    if message_cb is not None:
                        message_cb(f"CSV-backed cache: {count:,}/{total_expected:,} objects cached | CSV matched {csv_matched:,} | with means {csv_with_means:,} | missing {csv_missing:,} | large ROIs skipped {skipped_large_area:,}")
                    if progress_cb is not None:
                        progress_cb(feature_index, total_expected if total_expected > 0 else 0)
            if batch:
                con.executemany(insert_sql, batch); con.commit()
            con.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", [
                ("cell_count", str(count)), ("skipped_count", str(skipped)),
                ("measured_count", str(csv_with_means)), ("measurement_failed_count", str(csv_missing)),
                ("csv_row_count", str(len(csv_index.rows))), ("csv_matched_count", str(csv_matched)),
                ("csv_with_means_count", str(csv_with_means)), ("csv_missing_count", str(csv_missing)),
                ("csv_id_matches", str(csv_index.id_matches)), ("csv_centroid_matches", str(csv_index.centroid_matches)),
                ("csv_order_matches", str(csv_index.order_matches)), ("csv_unmatched", str(csv_index.unmatched)),
                ("skipped_large_area_count", str(skipped_large_area)), ("large_roi_count", str(skipped_large_area)), ("build_complete", "1"),
                ("completed", datetime.now().isoformat(timespec="seconds")),
            ])
            con.commit()
            try:
                con.execute("ANALYZE"); con.commit()
            except Exception:
                pass
            if progress_cb is not None:
                progress_cb(count, total_expected if total_expected > 0 else count)
            return {
                "cache_path": cache_path, "cell_count": count, "skipped_count": skipped,
                "measured_count": csv_with_means, "measurement_failed_count": csv_missing,
                "skipped_large_area_count": skipped_large_area, "large_roi_count": skipped_large_area, "max_cell_area_px": max_cell_area_px,
                "geo_scale_x": geo_scale_x, "geo_scale_y": geo_scale_y,
                "geo_ratio_x": scale_info.get("ratio_x", 1.0), "geo_ratio_y": scale_info.get("ratio_y", 1.0),
                "geo_scale_mode": scale_info.get("mode", ""), "measurement_source": "csv",
                "csv_path": csv_path, "csv_row_count": len(csv_index.rows), "csv_matched_count": csv_matched,
                "csv_with_means_count": csv_with_means, "csv_missing_count": csv_missing,
                "csv_id_matches": csv_index.id_matches, "csv_centroid_matches": csv_index.centroid_matches,
                "csv_order_matches": csv_index.order_matches, "task": "if_cache",
            }
        finally:
            con.close()
    finally:
        try:
            backend.close()
        except Exception:
            pass

def _if_build_cache_job(cancel_event, progress_cb, message_cb, image_path, geojson_path, cache_path, max_cell_area_px=IF_DEFAULT_MAX_CELL_AREA_PX, csv_path=""):
    if csv_path and Path(str(csv_path)).exists():
        return _if_build_cell_cache_from_csv(
            image_path=image_path,
            geojson_path=geojson_path,
            csv_path=str(csv_path),
            cache_path=cache_path,
            cancel_event=cancel_event,
            progress_cb=progress_cb,
            message_cb=message_cb,
            max_cell_area_px=max_cell_area_px,
        )
    return _if_build_cell_cache(
        image_path=image_path,
        geojson_path=geojson_path,
        cache_path=cache_path,
        cancel_event=cancel_event,
        progress_cb=progress_cb,
        message_cb=message_cb,
        max_cell_area_px=max_cell_area_px,
    )


def _if_cache_meta(cache_path: str) -> Dict[str, str]:
    """Read cache metadata and provide safe fallbacks for older/partial caches."""
    if not cache_path or not Path(cache_path).exists():
        return {}
    con = _if_connect_cache(cache_path)
    try:
        meta = {}
        try:
            rows = con.execute("SELECT key, value FROM meta").fetchall()
            meta = {str(r["key"]): str(r["value"]) for r in rows}
        except Exception:
            meta = {}
        # Older caches did not store build_complete. Treat caches with cell_count
        # as complete, but caches without cell_count as partial/incomplete.
        if "build_complete" not in meta:
            meta["build_complete"] = "1" if meta.get("cell_count") else "0"
        # If cell_count is missing, count current rows only for diagnostics.
        if not meta.get("cell_count"):
            try:
                meta["partial_row_count"] = str(int(con.execute("SELECT COUNT(*) FROM cells").fetchone()[0]))
            except Exception:
                meta["partial_row_count"] = "?"
        return meta
    finally:
        con.close()

def _if_row_positive_channels(row, thresholds: Sequence[float], enabled_channels: Sequence[bool]) -> List[int]:
    pos = []
    for i in range(IF_MAX_CHANNELS):
        if i >= len(thresholds) or i >= len(enabled_channels) or not enabled_channels[i]:
            continue
        value = row[f"c{i}_mean"]
        if value is None:
            continue
        try:
            if float(value) >= float(thresholds[i]):
                pos.append(i)
        except Exception:
            pass
    return pos


def _if_centroid_ring(cx: float, cy: float, radius: float = 4.0, n: int = 10):
    pts = []
    r = max(1.0, float(radius))
    for k in range(max(6, int(n))):
        a = 2.0 * math.pi * k / max(6, int(n))
        pts.append((float(cx) + math.cos(a) * r, float(cy) + math.sin(a) * r))
    return pts


def _if_query_visible_cells(cache_path: str, roi_full: Tuple[int, int, int, int], limit: int = 50000,
                            sample_if_needed: bool = True, return_info: bool = False,
                            max_area_px: Optional[float] = None):
    """Return cells intersecting the current ROI without overwhelming the GUI.

    If the ROI contains more cells than the display limit, a deterministic
    id-modulo sample is returned instead of the first N rows. This gives a much
    better low-zoom overview while keeping drawing light.
    """
    if not cache_path or not Path(cache_path).exists() or roi_full is None:
        if return_info:
            return [], {"total": 0, "sampled": False, "sample_step": 1, "limit": int(limit or 0), "incomplete": False}
        return []
    meta = _if_cache_meta(cache_path)
    # Do not use a cache that is still being built or was interrupted before
    # the final cell_count/build_complete metadata was written.
    if str(meta.get("build_complete", "0")) != "1":
        if return_info:
            return [], {"total": 0, "sampled": False, "sample_step": 1, "limit": int(limit or 0), "incomplete": True, "partial_rows": meta.get("partial_row_count", "?")}
        return []
    x, y, w, h = [float(v) for v in roi_full]
    try:
        max_area_px = float(max_area_px) if max_area_px not in (None, '') else None
    except Exception:
        max_area_px = None
    if max_area_px is not None and max_area_px <= 0:
        max_area_px = None
    x2 = x + w
    y2 = y + h
    limit = max(1, int(limit or 1))
    con = _if_connect_cache(cache_path)
    try:
        where = "maxx >= ? AND minx <= ? AND maxy >= ? AND miny <= ?"
        params = (x, x2, y, y2)
        if max_area_px is not None:
            where += " AND area <= ?"
            params = params + (float(max_area_px),)
        try:
            total = int(con.execute(f"SELECT COUNT(*) FROM cells WHERE {where}", params).fetchone()[0])
        except Exception:
            total = 0
        sample_step = 1
        sampled = False
        extra_where = ""
        extra_params = []
        if sample_if_needed and total > limit:
            sample_step = max(1, int(math.ceil(float(total) / float(limit))))
            sampled = True
            extra_where = " AND (id % ?) = 0"
            extra_params.append(sample_step)
        sql = f"""
            SELECT id, source_id, class_name, minx, miny, maxx, maxy, cx, cy, area,
                   {', '.join([f'c{i}_mean' for i in range(IF_MAX_CHANNELS)])}, geom_z
            FROM cells
            WHERE {where}{extra_where}
            LIMIT ?
        """
        rows = con.execute(sql, params + tuple(extra_params) + (limit,)).fetchall()
        info = {"total": total, "sampled": sampled, "sample_step": sample_step, "limit": limit, "returned": len(rows)}
        return (rows, info) if return_info else rows
    finally:
        con.close()


def _if_query_visible_large_rois(cache_path: str, roi_full: Tuple[int, int, int, int], limit: int = 50):
    """Return large tissue/annotation ROIs intersecting the current view.

    These are stored when objects exceed Max cell area during cache build. They
    are not painted as cell overlays, but can be used as a tissue mask for
    pixel-level threshold preview.
    """
    if not cache_path or not Path(cache_path).exists() or roi_full is None:
        return []
    x, y, w, h = [float(v) for v in roi_full]
    x2 = x + w
    y2 = y + h
    con = _if_connect_cache(cache_path)
    try:
        try:
            rows = con.execute(
                """
                SELECT id, source_id, class_name, minx, miny, maxx, maxy, cx, cy, area, geom_z
                FROM large_rois
                WHERE maxx >= ? AND minx <= ? AND maxy >= ? AND miny <= ?
                ORDER BY area DESC
                LIMIT ?
                """,
                (x, x2, y, y2, max(1, int(limit or 1))),
            ).fetchall()
            return rows
        except Exception:
            return []
    finally:
        con.close()


def _if_rasterize_roi_rows_to_mask(roi_rows, roi_full: Tuple[int, int, int, int], out_shape_hw: Tuple[int, int]) -> np.ndarray:
    """Rasterize large ROI rows into the current preview pixel grid."""
    import zlib
    from PIL import Image, ImageDraw
    h, w = int(out_shape_hw[0]), int(out_shape_hw[1])
    if h <= 0 or w <= 0:
        return np.zeros((0, 0), dtype=bool)
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    rx, ry, rw, rh = [float(v) for v in roi_full]
    if rw <= 0 or rh <= 0:
        return np.zeros((h, w), dtype=bool)

    def map_pt(x, y):
        px = (float(x) - rx) / rw * float(w)
        py = (float(y) - ry) / rh * float(h)
        return (px, py)

    for row in roi_rows or []:
        try:
            blob = row["geom_z"]
            rings = json.loads(zlib.decompress(blob).decode("utf-8")) if blob is not None else []
        except Exception:
            rings = []
        for ring in rings or []:
            pts = []
            for pt in ring or []:
                try:
                    x, y = float(pt[0]), float(pt[1])
                    if np.isfinite(x) and np.isfinite(y):
                        pts.append(map_pt(x, y))
                except Exception:
                    continue
            if len(pts) >= 3:
                draw.polygon(pts, outline=1, fill=1)
    return np.asarray(mask_img, dtype=np.uint8) > 0


def _if_alpha_blend_mask(rgb: np.ndarray, mask: np.ndarray, color: QColor, alpha: float):
    """Alpha blend a single QColor over rgb where mask is True."""
    if mask is None or not np.any(mask):
        return rgb
    a = max(0.0, min(1.0, float(alpha)))
    if a <= 0:
        return rgb
    c = np.array([color.red(), color.green(), color.blue()], dtype=np.float32)
    out = rgb.astype(np.float32, copy=False)
    out[mask] = out[mask] * (1.0 - a) + c * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _if_apply_tissue_pixel_threshold_overlay(
    rgb: np.ndarray,
    raw_arr: np.ndarray,
    axes: str,
    roi_full: Tuple[int, int, int, int],
    cache_path: Optional[str],
    thresholds: Sequence[float],
    enabled_channels: Sequence[bool],
    show_positive: Sequence[bool],
    show_negative: bool,
    show_multi: bool,
    opacity: int = 120,
    restrict_to_large_roi: bool = True,
):
    """Apply a fast pixel-level threshold overlay to the current displayed ROI.

    This uses the already-loaded preview/ROI array, not the full-resolution image,
    so it remains fast for panning/zooming. If restrict_to_large_roi is enabled,
    pixels are classified only inside large tissue annotation polygons stored in
    the cache. At low zoom this is an overview; at higher zoom it becomes closer
    to the native data.
    """
    out = _to_uint8_rgb(rgb).copy()
    stats = {"tissue_pixels": 0, "shown_pixels": 0, "large_rois": 0, "masked": False}
    for i in range(IF_MAX_CHANNELS):
        stats[f"C{i + 1}+"] = 0
    stats["multi"] = 0
    stats["negative"] = 0
    if raw_arr is None or roi_full is None:
        return out, stats
    try:
        raw_yxc, raw_axes = _if_raw_crop_to_yxc(raw_arr, axes)
    except Exception:
        return out, stats
    raw_yxc = np.asarray(raw_yxc)
    if raw_yxc.ndim == 2:
        raw_yxc = raw_yxc[:, :, None]
    h, w = out.shape[:2]
    if raw_yxc.shape[0] != h or raw_yxc.shape[1] != w:
        # Conservative fallback: skip if raw and RGB are not aligned.
        return out, stats

    if restrict_to_large_roi:
        large_rows = _if_query_visible_large_rois(str(cache_path), roi_full, limit=50) if cache_path else []
        stats["large_rois"] = len(large_rows)
        tissue_mask = _if_rasterize_roi_rows_to_mask(large_rows, roi_full, (h, w)) if large_rows else np.zeros((h, w), dtype=bool)
        stats["masked"] = True
    else:
        tissue_mask = np.ones((h, w), dtype=bool)
        stats["masked"] = False
    if not np.any(tissue_mask):
        return out, stats
    stats["tissue_pixels"] = int(np.count_nonzero(tissue_mask))

    n_ch = min(raw_yxc.shape[2], IF_MAX_CHANNELS)
    pos_count = np.zeros((h, w), dtype=np.uint8)
    first_pos = np.full((h, w), -1, dtype=np.int16)
    for ch in range(n_ch):
        if ch >= len(enabled_channels) or not enabled_channels[ch]:
            continue
        try:
            mask = (raw_yxc[:, :, ch].astype(np.float32, copy=False) >= float(thresholds[ch])) & tissue_mask
        except Exception:
            continue
        stats[f"C{ch + 1}+"] = int(np.count_nonzero(mask))
        first_pos[(first_pos < 0) & mask] = ch
        pos_count[mask] += 1

    alpha = max(0, min(255, int(opacity))) / 255.0
    # Negative first, then single positives, then multi-positive on top.
    if show_negative:
        neg_mask = (pos_count == 0) & tissue_mask
        stats["negative"] = int(np.count_nonzero(neg_mask))
        out = _if_alpha_blend_mask(out, neg_mask, IF_STATUS_COLORS.get("Negative", QColor(150, 150, 150)), alpha)
    else:
        stats["negative"] = int(np.count_nonzero((pos_count == 0) & tissue_mask))

    for ch in range(n_ch):
        if ch >= len(show_positive) or not show_positive[ch]:
            continue
        single_mask = (pos_count == 1) & (first_pos == ch) & tissue_mask
        if np.any(single_mask):
            out = _if_alpha_blend_mask(out, single_mask, IF_STATUS_COLORS.get(f"C{ch + 1}+", _deterministic_qcolor_for_text(f"C{ch + 1}+")), alpha)

    multi_mask = (pos_count >= 2) & tissue_mask
    stats["multi"] = int(np.count_nonzero(multi_mask))
    if show_multi and np.any(multi_mask):
        out = _if_alpha_blend_mask(out, multi_mask, IF_STATUS_COLORS.get("Multi+", QColor(255, 220, 0)), alpha)
    elif not show_multi:
        # If multi overlay is disabled, show multi-positive pixels as the first
        # enabled positive channel that passes, when that channel is visible.
        for ch in range(n_ch):
            if ch >= len(show_positive) or not show_positive[ch]:
                continue
            m = multi_mask & (first_pos == ch)
            if np.any(m):
                out = _if_alpha_blend_mask(out, m, IF_STATUS_COLORS.get(f"C{ch + 1}+", _deterministic_qcolor_for_text(f"C{ch + 1}+")), alpha)

    shown = np.zeros((h, w), dtype=bool)
    if show_negative:
        shown |= (pos_count == 0) & tissue_mask
    for ch in range(n_ch):
        if ch < len(show_positive) and show_positive[ch]:
            shown |= (pos_count == 1) & (first_pos == ch) & tissue_mask
    if show_multi:
        shown |= multi_mask
    stats["shown_pixels"] = int(np.count_nonzero(shown))
    return out, stats


def _if_visible_rows_to_annotations(rows, thresholds: Sequence[float], enabled_channels: Sequence[bool],
                                    show_positive: Sequence[bool], show_negative: bool,
                                    show_multi: bool, use_polygons: bool,
                                    centroid_radius: float = 4.0) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    import zlib
    annotations = []
    stats = {"visible": 0, "shown": 0, "negative": 0, "multi": 0}
    for i in range(IF_MAX_CHANNELS):
        stats[f"C{i + 1}+"] = 0
    for row in rows or []:
        stats["visible"] += 1
        pos = _if_row_positive_channels(row, thresholds, enabled_channels)
        for ch in pos:
            stats[f"C{ch + 1}+"] += 1
        if len(pos) >= 2:
            stats["multi"] += 1
        if not pos:
            stats["negative"] += 1

        # Display filtering.
        if len(pos) >= 2 and show_multi:
            cls = "Multi+"
        elif len(pos) == 1:
            ch = pos[0]
            if ch >= len(show_positive) or not show_positive[ch]:
                continue
            cls = f"C{ch + 1}+"
        elif not pos and show_negative:
            cls = "Negative"
        else:
            continue

        if len(pos) >= 2 and not show_multi:
            # If multi checkbox is off, still allow display under individual channels.
            shown_individual = False
            for ch in pos:
                if ch < len(show_positive) and show_positive[ch]:
                    cls = f"C{ch + 1}+"
                    shown_individual = True
                    break
            if not shown_individual:
                continue

        color = IF_STATUS_COLORS.get(cls, _deterministic_qcolor_for_text(cls))
        rings = None
        if use_polygons:
            try:
                blob = row["geom_z"]
                if blob is not None:
                    rings = json.loads(zlib.decompress(blob).decode("utf-8"))
            except Exception:
                rings = None
        if not rings:
            rings = [_if_centroid_ring(row["cx"], row["cy"], radius=centroid_radius)]
        annotations.append({
            "id": row["id"],
            "class_name": cls,
            "rings": rings,
            "color": color,
            "properties": {
                "source_id": row["source_id"],
                "area": row["area"],
                "positive_channels": ",".join([f"C{ch + 1}" for ch in pos]),
            },
        })
        stats["shown"] += 1
    return annotations, stats


def _if_export_rows_to_csv(cache_path: str, roi_full: Tuple[int, int, int, int], out_path: str,
                           thresholds: Sequence[float], enabled_channels: Sequence[bool],
                           max_area_px: Optional[float] = IF_DEFAULT_MAX_CELL_AREA_PX):
    rows = _if_query_visible_cells(cache_path, roi_full, limit=1_000_000, max_area_px=max_area_px)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "source_id", "class_name", "cx", "cy", "area"] + [f"c{i}_mean" for i in range(IF_MAX_CHANNELS)] + ["positive_channels"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            pos = _if_row_positive_channels(row, thresholds, enabled_channels)
            writer.writerow({
                "id": row["id"],
                "source_id": row["source_id"],
                "class_name": row["class_name"],
                "cx": row["cx"],
                "cy": row["cy"],
                "area": row["area"],
                **{f"c{i}_mean": row[f"c{i}_mean"] for i in range(IF_MAX_CHANNELS)},
                "positive_channels": ";".join([f"C{ch + 1}" for ch in pos]),
            })
    return out_path, len(rows)


def draw_geojson_annotations_on_rgb(rgb: np.ndarray, annotations: List[Dict[str, Any]], roi_full: Tuple[int, int, int, int],
                                    color: Optional[QColor] = None, opacity: int = 110,
                                    fill: bool = True, boundary_width: int = 2,
                                    class_styles: Optional[Dict[str, Dict[str, Any]]] = None) -> np.ndarray:
    """Draw GeoJSON annotation overlays onto an RGB array for exported preview/tile captures.

    class_styles optionally maps annotation class/name to {visible: bool, color: QColor}.
    """
    if not annotations or roi_full is None:
        return _to_uint8_rgb(rgb)
    out = _to_uint8_rgb(rgb)
    h, w = out.shape[:2]
    rx, ry, rw, rh = [float(v) for v in roi_full]
    if rw <= 0 or rh <= 0:
        return out
    qimg = QImage(out.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    painter = QPainter(qimg)
    fallback = QColor(color or QColor(255, 0, 0))
    alpha = max(0, min(255, int(opacity)))
    class_styles = class_styles or {}

    def map_xy(x, y):
        px = (float(x) - rx) / rw * w
        py = (float(y) - ry) / rh * h
        return QPointF(float(px), float(py))

    for ann in annotations:
        cls = str(ann.get("class_name", "annotation") or "annotation")
        st = class_styles.get(cls, {})
        if not bool(st.get("visible", True)):
            continue
        base = QColor(st.get("color", ann.get("color", fallback)))
        pen_color = QColor(base.red(), base.green(), base.blue(), max(1, alpha))
        fill_color = QColor(base.red(), base.green(), base.blue(), alpha)
        painter.setPen(QPen(pen_color, max(1, int(boundary_width))))
        painter.setBrush(QBrush(fill_color) if fill else Qt.NoBrush)
        for ring in ann.get("rings", []) or []:
            if len(ring) < 2:
                continue
            path = QPainterPath()
            first = True
            for x, y in ring:
                pt = map_xy(x, y)
                if first:
                    path.moveTo(pt)
                    first = False
                else:
                    path.lineTo(pt)
            if len(ring) >= 3:
                path.closeSubpath()
            painter.drawPath(path)
    painter.end()
    qimg = qimg.convertToFormat(QImage.Format_RGB888)
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((qimg.height(), qimg.width(), 3)).copy()
    return np.ascontiguousarray(arr)



class TilePopupImageLabel(QLabel):
    """Scalable image label used by the non-modal tile capture popup."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")
        self._pixmap = None

    def set_rgb(self, rgb: Optional[np.ndarray]):
        self._pixmap = _numpy_rgb_to_qpixmap(_to_uint8_rgb(rgb)) if rgb is not None else None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))
        if self._pixmap is not None and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = int((self.width() - scaled.width()) / 2)
            y = int((self.height() - scaled.height()) / 2)
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QPen(QColor(80, 80, 80), 1))
            painter.drawText(self.rect(), Qt.AlignCenter, "Tile preview")
        painter.end()


class TileCapturePopup(QDialog):
    """Non-modal movable popup that shows the currently selected square tile."""
    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.owner = owner
        self.setWindowTitle("Tile capture preview")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(650, 720)

        layout = QVBoxLayout(self)
        self.image_label = TilePopupImageLabel()
        layout.addWidget(self.image_label, 1)

        self.info_label = QLabel("Tile: not set")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #555; padding: 4px;")
        layout.addWidget(self.info_label)

        controls = QHBoxLayout()
        self.include_geojson_chk = QCheckBox("Include visible GeoJSON")
        self.include_geojson_chk.setChecked(True)
        self.include_geojson_chk.stateChanged.connect(lambda *_: self.owner.schedule_preview_tile_popup_update())
        controls.addWidget(self.include_geojson_chk)
        controls.addStretch()
        self.save_btn = QPushButton("Save Tile JPG")
        self.save_btn.clicked.connect(self.owner.save_preview_tile_capture_jpg)
        controls.addWidget(self.save_btn)
        layout.addLayout(controls)




    def closeEvent(self, event):
        # Keep the dialog reusable and synchronize the checkbox in the main panel.
        event.ignore()
        self.hide()
        if hasattr(self.owner, "preview_tile_capture_chk"):
            self.owner.preview_tile_capture_chk.blockSignals(True)
            self.owner.preview_tile_capture_chk.setChecked(False)
            self.owner.preview_tile_capture_chk.blockSignals(False)
        self.owner.preview_tile_mode = False
        if hasattr(self.owner, "preview_image_label"):
            self.owner.preview_image_label.enable_tile_mode(False)
        self.owner.update_preview_tile_overlay()

    def set_tile(self, rgb: Optional[np.ndarray], info: str):
        self.image_label.set_rgb(rgb)
        self.info_label.setText(info)


# ============================================================
# Explorer thumbnail worker
# ============================================================

class ThumbnailExplorerWorker(QThread):
    """Build file-explorer thumbnails outside the GUI thread.

    QPixmap/QIcon are created in the main thread; this worker only reads a small
    RGB numpy preview using the same safe preview reader used elsewhere in the app.
    """
    item_ready = pyqtSignal(str, object, str)
    item_failed = pyqtSignal(str, str)
    progress = pyqtSignal(int, int)
    finished_count = pyqtSignal(int)

    def __init__(self, paths, max_side: int = 180, parent=None):
        super().__init__(parent)
        self.paths = [str(p) for p in paths]
        self.max_side = int(max_side)
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        total = len(self.paths)
        done = 0
        for path in self.paths:
            if self._cancel_requested:
                break
            try:
                arr, axes, meta = read_preview_array_from_file(
                    path, max_side=self.max_side, allow_full_fallback=False
                )
                # If tifffile could only return a placeholder, try OpenSlide's
                # thumbnail path as a final Explorer-only fallback. This is useful
                # for large pyramidal TIFFs where an overview is available through
                # OpenSlide but not through tifffile/zarr.
                if str(meta.get("reader", "")).startswith("safe-placeholder"):
                    os_thumb = _try_openslide_thumbnail(path, max_side=self.max_side)
                    if os_thumb is not None:
                        arr, axes, meta = os_thumb
                rgb = _array_to_rgb_preview(arr, axes)
                rgb = _downsample_for_preview(rgb, max_side=self.max_side)
                self.item_ready.emit(path, rgb, str(meta.get("reader", "")))
            except Exception as exc:
                self.item_failed.emit(path, str(exc))
            done += 1
            self.progress.emit(done, total)
        self.finished_count.emit(done)

# ============================================================
# Main GUI
# ============================================================

class WSICropTileMergeGUI(QMainWindow):
    # ========================================================
    # IF Threshold Explorer page
    # ========================================================

    def _build_if_threshold_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)

        file_box = QGroupBox("IF Cell Threshold Explorer - image + cell GeoJSON + optional QuPath CSV")
        file_grid = QGridLayout(file_box)
        img_btn = QPushButton("Load IF image")
        img_btn.clicked.connect(self.load_if_image_file)
        geo_btn = QPushButton("Load cell GeoJSON")
        geo_btn.clicked.connect(self.load_if_geojson_file)
        csv_btn = QPushButton("Load measurements CSV")
        csv_btn.clicked.connect(self.load_if_measurements_csv_file)
        cache_btn = QPushButton("Open existing cache")
        cache_btn.clicked.connect(self.open_if_cache_file)
        build_btn = QPushButton("Build / rebuild cache")
        build_btn.setToolTip("If a QuPath measurements CSV is loaded, build is fast and imports Cell: ... (C#): Mean. Otherwise it measures means from raw image pixels.")
        build_btn.clicked.connect(self.build_if_cache)
        self.if_image_label = QLabel("Image: none")
        self.if_geojson_label = QLabel("GeoJSON: none")
        self.if_csv_label = QLabel("CSV: none (optional but much faster)")
        self.if_cache_label = QLabel("Cache: none")
        for lbl in (self.if_image_label, self.if_geojson_label, self.if_csv_label, self.if_cache_label):
            lbl.setStyleSheet("padding: 6px; border: 1px solid #bdc3c7; border-radius: 5px;")
            lbl.setWordWrap(True)
        file_grid.addWidget(img_btn, 0, 0)
        file_grid.addWidget(self.if_image_label, 0, 1, 1, 4)
        file_grid.addWidget(geo_btn, 1, 0)
        file_grid.addWidget(self.if_geojson_label, 1, 1, 1, 4)
        file_grid.addWidget(csv_btn, 2, 0)
        file_grid.addWidget(self.if_csv_label, 2, 1, 1, 4)
        file_grid.addWidget(cache_btn, 3, 0)
        file_grid.addWidget(build_btn, 3, 1)
        file_grid.addWidget(self.if_cache_label, 3, 2, 1, 3)
        layout.addWidget(file_box)

        main = QHBoxLayout()
        main.setSpacing(10)

        left_panel = QWidget()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(380)
        left_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)

        self.if_channel_table = QTableWidget()
        self.if_channel_table.setColumnCount(3)
        self.if_channel_table.setHorizontalHeaderLabels(["On", "Channel", "Color"])
        self.if_channel_table.setColumnWidth(0, 45)
        self.if_channel_table.setColumnWidth(1, 75)
        self.if_channel_table.setColumnWidth(2, 115)
        self.if_channel_table.verticalHeader().setVisible(False)
        self.if_channel_table.verticalHeader().setDefaultSectionSize(24)
        self.if_channel_table.setMaximumHeight(150)
        left.addWidget(QLabel("IF display channels"))
        left.addWidget(self.if_channel_table)

        threshold_box = QGroupBox("Thresholds and visible positivity")
        th_grid = QGridLayout(threshold_box)
        self.if_threshold_spins = []
        self.if_show_pos_checks = []
        for i in range(4):
            th_grid.addWidget(QLabel(f"C{i + 1}"), i, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-1_000_000_000, 1_000_000_000)
            spin.setDecimals(3)
            spin.setSingleStep(10.0)
            spin.setValue(float(IF_DEFAULT_THRESHOLDS[i]))
            spin.valueChanged.connect(lambda *_: self.schedule_if_threshold_update())
            self.if_threshold_spins.append(spin)
            th_grid.addWidget(spin, i, 1)
            chk = QCheckBox(f"show C{i + 1}+")
            chk.setChecked(i in (1, 2, 3))
            chk.stateChanged.connect(lambda *_: self.schedule_if_threshold_update())
            self.if_show_pos_checks.append(chk)
            th_grid.addWidget(chk, i, 2)
        self.if_show_multi_chk = QCheckBox("show multi-positive")
        self.if_show_multi_chk.setChecked(True)
        self.if_show_multi_chk.stateChanged.connect(lambda *_: self.schedule_if_threshold_update())
        self.if_show_negative_chk = QCheckBox("show negative")
        self.if_show_negative_chk.setChecked(False)
        self.if_show_negative_chk.stateChanged.connect(lambda *_: self.schedule_if_threshold_update())
        th_grid.addWidget(self.if_show_multi_chk, 4, 0, 1, 2)
        th_grid.addWidget(self.if_show_negative_chk, 4, 2)
        left.addWidget(threshold_box)

        overlay_box = QGroupBox("Overlay mode")
        overlay_grid = QGridLayout(overlay_box)
        self.if_overlay_enabled_chk = QCheckBox("Show overlay / annotations")
        self.if_overlay_enabled_chk.setChecked(True)
        self.if_overlay_enabled_chk.setToolTip("Fast toggle to compare the IF image with/without threshold overlays.")
        self.if_overlay_enabled_chk.stateChanged.connect(lambda *_: self.schedule_if_threshold_update(delay_ms=0))
        self.if_analysis_level_combo = QComboBox()
        self.if_analysis_level_combo.addItems(["Cell level", "Tissue pixel level", "Cell + tissue"])
        self.if_analysis_level_combo.setToolTip("Cell level uses GeoJSON/CSV cells. Tissue pixel level thresholds pixels inside the large tissue ROI using the current image view.")
        self.if_analysis_level_combo.currentIndexChanged.connect(lambda *_: self.schedule_if_threshold_update())
        self.if_overlay_mode_combo = QComboBox()
        self.if_overlay_mode_combo.addItems(["Centroids only (fastest)", "Polygons when zoom >= 800%", "Polygons always"])
        self.if_overlay_mode_combo.currentIndexChanged.connect(lambda *_: self.schedule_if_threshold_update())
        self.if_opacity_slider = QSlider(Qt.Horizontal)
        self.if_opacity_slider.setRange(0, 100)
        self.if_opacity_slider.setValue(55)
        self.if_opacity_slider.valueChanged.connect(lambda *_: self.schedule_if_threshold_update())
        self.if_opacity_value_label = QLabel("55%")
        self.if_fill_chk = QCheckBox("Fill overlay")
        self.if_fill_chk.setChecked(True)
        self.if_fill_chk.stateChanged.connect(lambda *_: self.schedule_if_threshold_update())
        self.if_limit_spin = QSpinBox()
        self.if_limit_spin.setRange(500, 500000)
        self.if_limit_spin.setValue(50000)
        self.if_limit_spin.setSingleStep(5000)
        self.if_limit_spin.valueChanged.connect(lambda *_: self.schedule_if_threshold_update())
        self.if_max_area_spin = QDoubleSpinBox()
        self.if_max_area_spin.setRange(0, 1_000_000_000)
        self.if_max_area_spin.setDecimals(0)
        self.if_max_area_spin.setSingleStep(5000)
        self.if_max_area_spin.setValue(float(IF_DEFAULT_MAX_CELL_AREA_PX))
        self.if_max_area_spin.setToolTip("Maximum object area in pixels to treat as a cell. Large tissue/annotation polygons above this value are saved as tissue ROIs, not cell overlays. Set 0 to disable.")
        self.if_max_area_spin.valueChanged.connect(lambda *_: self.schedule_if_threshold_update())
        self.if_tissue_large_roi_chk = QCheckBox("Tissue pixels only inside large ROI")
        self.if_tissue_large_roi_chk.setChecked(True)
        self.if_tissue_large_roi_chk.setToolTip("When using Tissue pixel level, restrict pixel thresholding to the large tissue annotation stored in the cache.")
        self.if_tissue_large_roi_chk.stateChanged.connect(lambda *_: self.schedule_if_threshold_update())
        overlay_grid.addWidget(self.if_overlay_enabled_chk, 0, 0, 1, 3)
        overlay_grid.addWidget(QLabel("Analysis:"), 1, 0)
        overlay_grid.addWidget(self.if_analysis_level_combo, 1, 1, 1, 2)
        overlay_grid.addWidget(QLabel("Mode:"), 2, 0)
        overlay_grid.addWidget(self.if_overlay_mode_combo, 2, 1, 1, 2)
        overlay_grid.addWidget(QLabel("Opacity:"), 3, 0)
        overlay_grid.addWidget(self.if_opacity_slider, 3, 1)
        overlay_grid.addWidget(self.if_opacity_value_label, 3, 2)
        overlay_grid.addWidget(self.if_fill_chk, 4, 0, 1, 2)
        overlay_grid.addWidget(QLabel("Display/sample limit:"), 5, 0)
        overlay_grid.addWidget(self.if_limit_spin, 5, 1)
        overlay_grid.addWidget(QLabel("Max cell area px²:"), 6, 0)
        overlay_grid.addWidget(self.if_max_area_spin, 6, 1)
        overlay_grid.addWidget(self.if_tissue_large_roi_chk, 7, 0, 1, 3)
        left.addWidget(overlay_box)

        nav_box = QGroupBox("Navigation / export")
        nav_grid = QGridLayout(nav_box)
        zoom_in_btn = QPushButton("Zoom +")
        zoom_in_btn.clicked.connect(lambda: self.change_if_zoom(1.25))
        zoom_out_btn = QPushButton("Zoom -")
        zoom_out_btn.clicked.connect(lambda: self.change_if_zoom(0.8))
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(lambda: self.set_if_zoom(1.0))
        update_btn = QPushButton("Update")
        update_btn.clicked.connect(self.update_if_threshold_preview)
        export_btn = QPushButton("Export visible CSV")
        export_btn.clicked.connect(self.export_if_visible_csv)
        self.if_zoom_label = QLabel("Zoom: 100%")
        self.if_stats_label = QLabel("No cache loaded")
        self.if_stats_label.setWordWrap(True)
        nav_grid.addWidget(zoom_out_btn, 0, 0)
        nav_grid.addWidget(zoom_in_btn, 0, 1)
        nav_grid.addWidget(fit_btn, 0, 2)
        nav_grid.addWidget(update_btn, 1, 0)
        nav_grid.addWidget(export_btn, 1, 1, 1, 2)
        nav_grid.addWidget(self.if_zoom_label, 2, 0, 1, 3)
        nav_grid.addWidget(self.if_stats_label, 3, 0, 1, 3)
        left.addWidget(nav_box)

        left.addStretch(1)

        main.addWidget(left_panel, 0)

        self.if_preview_label = ZoomRegionPreviewLabel()
        self.if_preview_label.setText("IF Threshold Explorer")
        self.if_preview_label.setAlignment(Qt.AlignCenter)
        self.if_preview_label.setMinimumSize(460, 300)
        self.if_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main.addWidget(self.if_preview_label, 1)
        layout.addLayout(main, 1)
        return page

    def load_if_image_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select IF image", "", _image_file_filter())
        if path:
            self._load_if_image_from_path(Path(path))

    def _load_if_image_from_path(self, path: Path):
        path = Path(path)
        try:
            if getattr(self, "if_backend", None) is not None:
                try:
                    self.if_backend.close()
                except Exception:
                    pass
            self.if_image_path = path
            self.if_backend = _if_load_measurement_backend(str(path))
            self.if_zoom = 1.0
            arr, axes, meta = read_zoom_region_from_backend(
                self.if_backend,
                center_xy=None,
                zoom=1.0,
                viewport_size=(max(640, self.if_preview_label.width()), max(420, self.if_preview_label.height())) if hasattr(self, "if_preview_label") else (900, 600),
                max_side=900,
            )
            self.if_arr, self.if_axes, self.if_meta = arr, axes, meta
            if meta.get("full_dims"):
                self.if_full_dims = tuple(meta["full_dims"])
            else:
                h, w = np.asarray(arr).shape[:2]
                self.if_full_dims = (w, h)
            self.if_center = (self.if_full_dims[0] / 2.0, self.if_full_dims[1] / 2.0)
            self.if_image_label.setText(f"Image: {path.name} | axes={axes} | reader={meta.get('reader')} | full={self.if_full_dims[0]} × {self.if_full_dims[1]}")
            self.populate_if_channel_table()
            # Suggest/use default cache if it already exists.
            if self.if_geojson_path and not self.if_cache_path:
                cp = _if_default_cache_path(str(self.if_image_path), str(self.if_geojson_path))
                if cp.exists():
                    self.if_cache_path = cp
                    self._refresh_if_cache_label()
            self.update_if_threshold_preview()
        except Exception as e:
            QMessageBox.critical(self, "IF image load error", str(e))

    def load_if_geojson_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select cell GeoJSON",
            "",
            "GeoJSON files (*.geojson *.json);;All Files (*)",
        )
        if not path:
            return
        self.if_geojson_path = Path(path)
        self.if_geojson_label.setText(f"GeoJSON: {self.if_geojson_path.name}")
        cp = _if_default_cache_path(str(self.if_image_path) if self.if_image_path else None, str(self.if_geojson_path))
        self.if_cache_path = cp if cp.exists() else None
        self._refresh_if_cache_label()
        if self.if_cache_path:
            self.update_if_threshold_preview()
        else:
            self.info_label.setText("GeoJSON selected. Build the measured SQLite cache once; it will calculate mean_C# from the image inside each cell polygon.")

    def load_if_measurements_csv_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select QuPath measurements CSV",
            "",
            "CSV / TSV files (*.csv *.tsv *.txt);;All Files (*)",
        )
        if not path:
            return
        self.if_csv_path = Path(path)
        if hasattr(self, "if_csv_label"):
            self.if_csv_label.setText(f"CSV: {self.if_csv_path.name} | will import Cell: ... (C#): Mean")
        self.info_label.setText("Measurements CSV selected. Build/rebuild cache will use the fast CSV path instead of measuring every cell from the image.")

    def open_if_cache_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open IF cells SQLite cache",
            "",
            "SQLite cache (*.sqlite *.db);;All Files (*)",
        )
        if not path:
            return
        self.if_cache_path = Path(path)
        self._refresh_if_cache_label()
        self.update_if_threshold_preview()

    def _refresh_if_cache_label(self):
        if not getattr(self, "if_cache_label", None):
            return
        if not self.if_cache_path:
            self.if_cache_label.setText("Cache: none")
            return
        meta = _if_cache_meta(str(self.if_cache_path))
        complete = str(meta.get("build_complete", "0")) == "1"
        n = meta.get("cell_count", "?")
        if not complete and n == "?":
            n = f"partial {meta.get('partial_row_count', '?')}"
        measured = meta.get("measured_count", "?")
        failed = meta.get("measurement_failed_count", "0")
        skipped_large = meta.get("skipped_large_area_count", "")
        max_area_meta = meta.get("max_cell_area_px", "")
        reader = meta.get("image_reader", "")
        source = meta.get("measurement_source", "raw")
        incomplete_txt = " | INCOMPLETE / still building or interrupted" if not complete else ""
        extra = f" | source={source} | usable means={measured} | missing/no mean={failed}" if measured != "?" else f" | source={source}"
        if source == "csv":
            extra += f" | CSV matched={meta.get('csv_matched_count', '?')}"
        if skipped_large not in ("", "0"):
            extra += f" | large ROIs={meta.get('large_roi_count', skipped_large)}"
        try:
            if max_area_meta:
                extra += f" | max area={float(max_area_meta):.0f}px²"
        except Exception:
            pass
        reader_txt = f" | reader={reader}" if reader else ""
        scale_x = meta.get("geojson_to_image_scale_x", "")
        scale_y = meta.get("geojson_to_image_scale_y", "")
        ratio_x = meta.get("geojson_coord_ratio_x", "")
        ratio_y = meta.get("geojson_coord_ratio_y", "")
        scale_txt = ""
        try:
            sx = float(scale_x); sy = float(scale_y)
            rx = float(ratio_x); ry = float(ratio_y)
            scale_txt = f" | geo→img scale=({sx:.4g},{sy:.4g}) ratio=({rx:.3g},{ry:.3g})"
        except Exception:
            pass
        self.if_cache_label.setText(f"Cache: {Path(self.if_cache_path).name} | cells={n}{extra}{reader_txt}{scale_txt}{incomplete_txt}")
        if meta.get("geojson_path") and not self.if_geojson_path:
            self.if_geojson_path = Path(meta.get("geojson_path"))
            self.if_geojson_label.setText(f"GeoJSON: {self.if_geojson_path.name}")
        if meta.get("csv_path") and hasattr(self, "if_csv_label") and not getattr(self, "if_csv_path", None):
            self.if_csv_path = Path(meta.get("csv_path"))
            self.if_csv_label.setText(f"CSV: {self.if_csv_path.name}")

    def build_if_cache(self):
        if not self.if_image_path:
            QMessageBox.warning(self, "No image", "Select the IF multichannel image first. The cache measures mean_C# from this image.")
            return
        if not self.if_geojson_path:
            QMessageBox.warning(self, "No GeoJSON", "Select the cell GeoJSON first.")
            return
        cache_path = _if_default_cache_path(str(self.if_image_path) if self.if_image_path else None, str(self.if_geojson_path))
        # Important: do NOT point the viewer to the cache being created. The
        # builder writes a new timestamped file and the GUI switches to it only
        # when the job has fully completed and build_complete=1 is stored.
        self.if_cache_building = True
        using_csv = bool(getattr(self, "if_csv_path", None) and Path(self.if_csv_path).exists())
        self.if_cache_label.setText("Cache: building CSV-backed cache... viewer will switch when finished" if using_csv else "Cache: building measured cache... viewer will switch when finished")
        self.if_stats_label.setText("Building cache from QuPath CSV means. Overlay is paused." if using_csv else "Building measured cache. Overlay is paused to avoid reading a partial SQLite file.")
        self._start_background_job(
            "IF cell cache build",
            _if_build_cache_job,
            self._on_if_cache_built,
            str(self.if_image_path or ""),
            str(self.if_geojson_path),
            str(cache_path),
            float(self.if_max_area_spin.value()) if hasattr(self, 'if_max_area_spin') else float(IF_DEFAULT_MAX_CELL_AREA_PX),
            str(self.if_csv_path) if using_csv else "",
        )

    def _on_if_cache_built(self, result):
        self.if_cache_building = False
        self.if_cache_path = Path(result.get("cache_path"))
        self._refresh_if_cache_label()
        source = result.get("measurement_source", "raw image")
        extra = ""
        if source == "csv":
            extra = (
                f"\nCSV rows: {result.get('csv_row_count', 0):,}. "
                f"CSV matched: {result.get('csv_matched_count', 0):,}. "
                f"With Cell mean values: {result.get('csv_with_means_count', 0):,}. "
                f"Missing: {result.get('csv_missing_count', 0):,}.\n"
                f"Match mode: ID={result.get('csv_id_matches', 0):,}, "
                f"centroid={result.get('csv_centroid_matches', 0):,}, "
                f"order fallback={result.get('csv_order_matches', 0):,}."
            )
        QMessageBox.information(
            self,
            "IF cache built",
            f"Cached {result.get('cell_count', 0):,} cells. "
            f"Mean source: {source}. "
            f"Usable means: {result.get('measured_count', 0):,}. "
            f"Missing/no mean: {result.get('measurement_failed_count', 0):,}. "
            f"Skipped {result.get('skipped_count', 0):,}. Large tissue ROIs stored {result.get('large_roi_count', result.get('skipped_large_area_count', 0)):,}."
            f"{extra}\n"
            f"GeoJSON→image scale: X={float(result.get('geo_scale_x', 1.0)):.6g}, "
            f"Y={float(result.get('geo_scale_y', 1.0)):.6g} "
            f"(ratio X={float(result.get('geo_ratio_x', 1.0)):.3g}, "
            f"Y={float(result.get('geo_ratio_y', 1.0)):.3g}; {result.get('geo_scale_mode', '')}).\n\n"
            f"{self.if_cache_path}",
        )
        self.info_label.setText(f"IF cache ready: {self.if_cache_path}")
        self.update_if_threshold_preview()

    def populate_if_channel_table(self, settings: Optional[List[Dict[str, Any]]] = None):
        if self.if_arr is None or not hasattr(self, "if_channel_table"):
            return
        n_total = _count_display_channels(self.if_arr, self.if_axes)
        n = min(int(n_total), 6)
        self.if_channel_table.setRowCount(n)
        for i in range(n):
            st = settings[i] if settings and i < len(settings) else {}
            chk = QCheckBox()
            chk.setChecked(bool(st.get("visible", True)))
            chk.stateChanged.connect(lambda *_: self.schedule_if_threshold_update())
            self.if_channel_table.setCellWidget(i, 0, chk)
            item = QTableWidgetItem(f"C{i + 1}")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.if_channel_table.setItem(i, 1, item)
            color_combo = QComboBox()
            color_combo.addItems(list(COLOR_MAPS.keys()))
            default_color = st.get("color", DEFAULT_CHANNEL_COLORS[i % len(DEFAULT_CHANNEL_COLORS)])
            color_combo.setCurrentText(default_color if default_color in COLOR_MAPS else "gray")
            color_combo.currentIndexChanged.connect(lambda *_: self.schedule_if_threshold_update())
            self.if_channel_table.setCellWidget(i, 2, color_combo)

    def get_if_channel_settings_from_table(self) -> List[Dict[str, Any]]:
        settings = []
        if not hasattr(self, "if_channel_table"):
            return settings
        for r in range(self.if_channel_table.rowCount()):
            chk = self.if_channel_table.cellWidget(r, 0)
            color_combo = self.if_channel_table.cellWidget(r, 2)
            settings.append({
                "channel": r,
                "visible": chk.isChecked() if chk else True,
                "color": color_combo.currentText() if color_combo else DEFAULT_CHANNEL_COLORS[r % len(DEFAULT_CHANNEL_COLORS)],
            })
        return settings

    def if_threshold_values(self) -> List[float]:
        vals = [0.0] * IF_MAX_CHANNELS
        for i, spin in enumerate(getattr(self, "if_threshold_spins", []) or []):
            if i < IF_MAX_CHANNELS:
                vals[i] = float(spin.value())
        return vals

    def if_enabled_threshold_channels(self) -> List[bool]:
        enabled = [False] * IF_MAX_CHANNELS
        for i, chk in enumerate(getattr(self, "if_show_pos_checks", []) or []):
            if i < IF_MAX_CHANNELS:
                # Only checked channels participate in positivity, multi-positive,
                # and negative calculations. This avoids DAPI/C1 affecting C2/C3/C4 logic.
                enabled[i] = bool(chk.isChecked())
        return enabled

    def schedule_if_threshold_update(self, delay_ms: int = 40):
        if hasattr(self, "_if_render_timer"):
            self._if_render_timer.start(max(0, int(delay_ms)))
        else:
            self.update_if_threshold_preview()

    def update_if_threshold_preview(self):
        if self.if_image_path is None:
            return
        try:
            if getattr(self, "if_backend", None) is None:
                self.if_backend = _if_load_measurement_backend(str(self.if_image_path))
            viewport = (max(640, self.if_preview_label.width()), max(420, self.if_preview_label.height()))
            arr, axes, meta = read_zoom_region_from_backend(
                self.if_backend,
                center_xy=getattr(self, "if_center", None),
                zoom=max(1.0, float(self.if_zoom)),
                viewport_size=viewport,
                max_side=max(1400, int(max(viewport) * 2)),
            )
            self.if_arr, self.if_axes, self.if_meta = arr, axes, meta
            if meta.get("full_dims"):
                self.if_full_dims = tuple(meta["full_dims"])
            self.if_last_rgb = render_channel_composite(arr, axes, self.get_if_channel_settings_from_table())
            self._update_if_threshold_pixmap()
        except Exception as e:
            self.info_label.setText(f"IF preview error: {e}")

    def _if_overlay_use_polygons(self) -> bool:
        text = self.if_overlay_mode_combo.currentText() if hasattr(self, "if_overlay_mode_combo") else "Centroids only"
        if text.startswith("Polygons always"):
            return True
        if text.startswith("Polygons when"):
            return float(getattr(self, "if_zoom", 1.0)) >= 8.0
        return False

    def _update_if_threshold_pixmap(self):
        if self.if_last_rgb is None:
            return
        roi = self.if_meta.get("roi") if isinstance(self.if_meta, dict) else None
        full_dims = getattr(self, "if_full_dims", None)
        overlay_enabled = bool(self.if_overlay_enabled_chk.isChecked()) if hasattr(self, "if_overlay_enabled_chk") else True
        analysis_mode = self.if_analysis_level_combo.currentText() if hasattr(self, "if_analysis_level_combo") else "Cell level"
        do_cells = overlay_enabled and analysis_mode in ("Cell level", "Cell + tissue")
        do_tissue = overlay_enabled and analysis_mode in ("Tissue pixel level", "Cell + tissue")
        opacity_percent = int(self.if_opacity_slider.value()) if hasattr(self, "if_opacity_slider") else 55
        if hasattr(self, "if_opacity_value_label"):
            self.if_opacity_value_label.setText(f"{opacity_percent}%")

        # Prepare positivity display choices once; both cell and tissue modes use them.
        show_pos = [False] * IF_MAX_CHANNELS
        for i, chk in enumerate(getattr(self, "if_show_pos_checks", []) or []):
            show_pos[i] = bool(chk.isChecked())
        thresholds = self.if_threshold_values()
        enabled_channels = self.if_enabled_threshold_channels()
        show_negative = bool(self.if_show_negative_chk.isChecked()) if hasattr(self, "if_show_negative_chk") else False
        show_multi = bool(self.if_show_multi_chk.isChecked()) if hasattr(self, "if_show_multi_chk") else True

        display_rgb = _to_uint8_rgb(self.if_last_rgb)
        tissue_stats = None
        if do_tissue and roi is not None:
            restrict = bool(self.if_tissue_large_roi_chk.isChecked()) if hasattr(self, "if_tissue_large_roi_chk") else True
            display_rgb, tissue_stats = _if_apply_tissue_pixel_threshold_overlay(
                display_rgb,
                getattr(self, "if_arr", None),
                getattr(self, "if_axes", ""),
                roi,
                str(self.if_cache_path) if self.if_cache_path else None,
                thresholds=thresholds,
                enabled_channels=enabled_channels,
                show_positive=show_pos,
                show_negative=show_negative,
                show_multi=show_multi,
                opacity=int(round(opacity_percent / 100.0 * 255)),
                restrict_to_large_roi=restrict,
            )

        self.if_preview_label.set_preview(
            display_rgb,
            roi_full=roi,
            full_dims=full_dims,
            center_callback=self._on_if_center_changed,
            rectangle_callback=None,
            zoom_callback=self._on_if_view_zoom,
        )

        annotations = []
        stats = None
        if do_cells and self.if_cache_path and Path(self.if_cache_path).exists() and roi is not None:
            limit = int(self.if_limit_spin.value()) if hasattr(self, "if_limit_spin") else 50000
            max_area = float(self.if_max_area_spin.value()) if hasattr(self, 'if_max_area_spin') else float(IF_DEFAULT_MAX_CELL_AREA_PX)
            rows, query_info = _if_query_visible_cells(str(self.if_cache_path), roi, limit=limit, return_info=True, max_area_px=max_area)
            # Radius is specified in screen-ish pixels, converted back to full-resolution pixels.
            try:
                display_w = max(1, int(self.if_preview_label._display_rect().width()))
                radius_full = max(2.0, float(roi[2]) / float(display_w) * 4.0)
            except Exception:
                radius_full = 4.0
            annotations, stats = _if_visible_rows_to_annotations(
                rows,
                thresholds=thresholds,
                enabled_channels=enabled_channels,
                show_positive=show_pos,
                show_negative=show_negative,
                show_multi=show_multi,
                use_polygons=self._if_overlay_use_polygons(),
                centroid_radius=radius_full,
            )
            if stats is not None and query_info is not None:
                stats["visible_total"] = int(query_info.get("total", stats.get("visible", 0)))
                stats["sampled"] = bool(query_info.get("sampled", False))
                stats["sample_step"] = int(query_info.get("sample_step", 1))
                stats["returned"] = int(query_info.get("returned", len(rows)))
                stats["incomplete"] = bool(query_info.get("incomplete", False))
                stats["partial_rows"] = query_info.get("partial_rows", "?")

        self.if_preview_label.set_annotations(annotations if overlay_enabled else [])
        self.if_preview_label.set_annotation_style(
            opacity=int(round(opacity_percent / 100.0 * 255)),
            fill=bool(self.if_fill_chk.isChecked()) if hasattr(self, "if_fill_chk") else True,
            boundary_width=1,
        )
        class_styles = {name: {"visible": True, "color": QColor(color)} for name, color in IF_STATUS_COLORS.items()}
        self.if_preview_label.set_annotation_class_styles(class_styles)
        if hasattr(self, "if_zoom_label"):
            self.if_zoom_label.setText(f"Zoom: {int(self.if_zoom * 100)}%")
        if hasattr(self, "if_stats_label"):
            if not overlay_enabled:
                self.if_stats_label.setText("Overlay OFF. Image only.")
            elif stats:
                if stats.get("incomplete"):
                    self.if_stats_label.setText(
                        f"Cache incomplete/partial rows={stats.get('partial_rows', '?')}. "
                        "Wait for Build / rebuild cache to finish, then the viewer will switch automatically."
                    )
                else:
                    sample_txt = ""
                    if stats.get("sampled"):
                        sample_txt = f" | sampled 1/{stats.get('sample_step', 1)} returned={stats.get('returned', stats.get('visible', 0)):,}"
                    txt = (
                        f"Cell ROI: {stats.get('visible_total', stats.get('visible', 0)):,}{sample_txt} | shown: {stats.get('shown', 0):,} | "
                        f"C2+: {stats.get('C2+', 0):,} | C3+: {stats.get('C3+', 0):,} | C4+: {stats.get('C4+', 0):,} | "
                        f"multi+: {stats.get('multi', 0):,} | neg: {stats.get('negative', 0):,}"
                    )
                    if tissue_stats:
                        mask_txt = f"large ROI={tissue_stats.get('large_rois', 0)}" if tissue_stats.get("masked") else "full ROI"
                        txt += (
                            f"\nTissue pixels ({mask_txt}): shown={tissue_stats.get('shown_pixels', 0):,} / "
                            f"mask={tissue_stats.get('tissue_pixels', 0):,} | C2+={tissue_stats.get('C2+', 0):,} | "
                            f"C3+={tissue_stats.get('C3+', 0):,} | C4+={tissue_stats.get('C4+', 0):,} | "
                            f"multi+={tissue_stats.get('multi', 0):,}"
                        )
                    self.if_stats_label.setText(txt)
            elif tissue_stats:
                mask_txt = f"large ROI={tissue_stats.get('large_rois', 0)}" if tissue_stats.get("masked") else "full ROI"
                self.if_stats_label.setText(
                    f"Tissue pixels ({mask_txt}): shown={tissue_stats.get('shown_pixels', 0):,} / "
                    f"mask={tissue_stats.get('tissue_pixels', 0):,} | "
                    f"C2+={tissue_stats.get('C2+', 0):,} | C3+={tissue_stats.get('C3+', 0):,} | "
                    f"C4+={tissue_stats.get('C4+', 0):,} | multi+={tissue_stats.get('multi', 0):,}"
                )
            elif self.if_cache_path:
                if do_tissue:
                    self.if_stats_label.setText("Tissue mode active, but no large tissue ROI is available in this cache/view. Rebuild cache with Max cell area so large annotations are stored, or uncheck 'Tissue pixels only inside large ROI'.")
                else:
                    self.if_stats_label.setText("Cache loaded. No cells in current ROI or current overlay filters hide them.")
            else:
                self.if_stats_label.setText("No cache loaded. Build/open cache to overlay cell positivity or tissue ROI masks.")
        self.info_label.setText(f"IF Threshold Explorer | mode={analysis_mode} | ROI={roi} | cell overlays={len(annotations) if overlay_enabled else 0:,}" )

    def _on_if_center_changed(self, cx, cy):
        self.if_center = (float(cx), float(cy))
        self.schedule_if_threshold_update(delay_ms=35)

    def _on_if_view_zoom(self, factor: float, center_xy=None):
        self.change_if_zoom(factor, center_xy=center_xy)

    def change_if_zoom(self, factor: float, center_xy=None):
        if not getattr(self, "if_full_dims", None):
            return
        if center_xy is not None:
            self.if_center = (float(center_xy[0]), float(center_xy[1]))
        self.if_zoom = max(1.0, min(512.0, float(self.if_zoom) * float(factor)))
        self.schedule_if_threshold_update(delay_ms=10)

    def set_if_zoom(self, zoom: float):
        self.if_zoom = max(1.0, float(zoom))
        if self.if_full_dims:
            self.if_center = (self.if_full_dims[0] / 2.0, self.if_full_dims[1] / 2.0)
        self.schedule_if_threshold_update(delay_ms=10)

    def export_if_visible_csv(self):
        if not self.if_cache_path or not Path(self.if_cache_path).exists():
            QMessageBox.warning(self, "No cache", "Build or open an IF cell cache first.")
            return
        roi = self.if_meta.get("roi") if isinstance(self.if_meta, dict) else None
        if roi is None:
            QMessageBox.warning(self, "No ROI", "Load the image preview first.")
            return
        default_dir = self.if_image_path.parent if self.if_image_path else Path(self.if_cache_path).parent
        x, y, w, h = [int(round(float(v))) for v in roi]
        default_name = f"IF_visible_cells_x{x}_y{y}_w{w}_h{h}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export visible IF cells CSV",
            str(default_dir / default_name),
            "CSV files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            out_path, n = _if_export_rows_to_csv(
                str(self.if_cache_path),
                roi,
                path,
                thresholds=self.if_threshold_values(),
                enabled_channels=self.if_enabled_threshold_channels(),
                max_area_px=float(self.if_max_area_spin.value()) if hasattr(self, 'if_max_area_spin') else float(IF_DEFAULT_MAX_CELL_AREA_PX),
            )
            QMessageBox.information(self, "Exported", f"Exported {n:,} visible cell rows:\n{out_path}")
            self.info_label.setText(f"Exported IF visible CSV: {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "IF export error", str(e))


    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(resource_path(APP_ICON_PATH)))
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} - WSI Crop / Preview / Tile / Merge")
        self.setGeometry(40, 40, 980, 640)
        self.setMinimumSize(600, 420)
        self.setStyleSheet("background-color: #f0f0f0;")

        self.backend = ImageBackend()
        self.tile_files = []
        self.bulk_paths = []
        self.manual_layout = None
        self.lif_single_path = None
        self.lif_single_obj = None
        self.lif_single_images = []
        self.lif_bulk_paths = []
        self.downsample_paths = []
        self.downsample_preview_path = None
        self.downsample_roi = None
        self.preview_path = None
        self.preview_backend = None  # cached reader for fast repeated Image Preview region reads
        self.preview_arr = None
        self.preview_axes = None
        self.preview_meta = {}
        self.preview_full_dims = None
        self.preview_center = None
        self.preview_zoom = 1.0
        self.preview_last_rgb = None
        self.preview_annotations = []
        self.preview_annotation_color = QColor(255, 0, 0)
        self.preview_annotation_styles = {}
        self.preview_new_annotation_color = QColor(255, 0, 0)
        self.preview_annotation_draw_mode = "none"
        self._preview_annotation_counter = 1
        self.preview_tile_center = None
        self.preview_tile_mode = False
        self.preview_tile_popup = None
        self._preview_tile_popup_timer = QTimer(self)
        self._preview_tile_popup_timer.setSingleShot(True)
        self._preview_tile_popup_timer.timeout.connect(self.update_preview_tile_popup_now)
        self._preview_render_timer = QTimer(self)
        self._preview_render_timer.setSingleShot(True)
        self._preview_render_timer.timeout.connect(self.update_channel_preview)
        self.crop_zoom = 1.0
        self.crop_center = None
        self.crop_preview_meta = {}
        self.batch_channel_paths = []
        self.if_image_path = None
        self.if_geojson_path = None
        self.if_csv_path = None
        self.if_cache_path = None
        self.if_backend = None
        self.if_arr = None
        self.if_axes = None
        self.if_meta = {}
        self.if_full_dims = None
        self.if_center = None
        self.if_zoom = 1.0
        self.if_last_rgb = None
        self._if_render_timer = QTimer(self)
        self._if_render_timer.setSingleShot(True)
        self._if_render_timer.timeout.connect(self.update_if_threshold_preview)
        self.explorer_paths = []
        self.explorer_current_folder = None
        self.explorer_thumb_worker = None
        self.active_worker = None
        self.setFocusPolicy(Qt.StrongFocus)
        self._preview_channel_shortcuts = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        title = QLabel(f"{APP_NAME} v{APP_VERSION} - WSI Crop / Preview / Tile / Merge")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 2px;")
        root.addWidget(title)

        menu_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Crop", "Tiles", "Merge Tiles", "Split LIF",
            "Downsample", "Image Preview", "IF Threshold Explorer", "Explorer"
        ])
        self.mode_combo.setMaximumWidth(180)

        menu_row.addWidget(QLabel("Mode:"))
        menu_row.addWidget(self.mode_combo)
        menu_row.addStretch()

        self.worker_spin = QSpinBox()
        self.worker_spin.setRange(1, max(1, min(8, (os.cpu_count() or 2))))
        self.worker_spin.setValue(1)
        self.worker_spin.setToolTip("Number of parallel files to process in bulk jobs. Use 1–2 for very large IF/LIF files.")
        self.cancel_job_btn = QPushButton("Cancel job")
        self.cancel_job_btn.setEnabled(False)
        self.cancel_job_btn.clicked.connect(self.cancel_background_job)
        menu_row.addWidget(QLabel("Workers:"))
        menu_row.addWidget(self.worker_spin)
        menu_row.addWidget(self.cancel_job_btn)

        help_btn = QPushButton("Help / About")
        help_btn.setToolTip("Show app information, citation, DOI, and usage notes.")
        help_btn.clicked.connect(self.show_help_about)
        menu_row.addWidget(help_btn)

        root.addLayout(menu_row)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #27ae60; padding: 3px;")
        self.info_label.setMaximumHeight(42)
        self.info_label.setWordWrap(True)

        self.stack = QStackedWidget()
        # Keep the main pages directly in the window.  A global QScrollArea made
        # the Linux UI feel cramped because users had to scroll inside the app to
        # reach bottom controls.  Individual widgets are now compacted instead.
        root.addWidget(self.stack, 1)
        self.stack.addWidget(self._build_crop_page())
        self.stack.addWidget(self._build_tiles_page())
        self.stack.addWidget(self._build_merge_page())
        self.stack.addWidget(self._build_lif_page())
        self.stack.addWidget(self._build_downsample_page())
        self.stack.addWidget(self._build_preview_page())
        self.stack.addWidget(self._build_if_threshold_page())
        self.stack.addWidget(self._build_explorer_page())
        self.mode_combo.currentIndexChanged.connect(lambda: self.stack.setCurrentIndex(self.mode_combo.currentIndex()))

        root.addWidget(self.info_label)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self._install_preview_channel_shortcuts()

    def _set_label_pixmap(self, label, rgb):
        pm = _numpy_rgb_to_qpixmap(_to_uint8_rgb(rgb))
        label.setPixmap(pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _install_preview_channel_shortcuts(self):
        """Number keys 1..9 toggle channels in Image Preview mode."""
        self._preview_channel_shortcuts = []
        for i in range(1, 10):
            shortcut = QShortcut(QKeySequence(str(i)), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(lambda idx=i: self.toggle_preview_channel_by_number(idx))
            self._preview_channel_shortcuts.append(shortcut)

    def toggle_preview_channel_by_number(self, number: int):
        """Toggle channel N with key N while Image Preview is active.

        Example: pressing 1 toggles C0, pressing 2 toggles C1.
        The shortcut is ignored while editing text/numeric fields.
        """
        try:
            if not hasattr(self, "mode_combo") or self.mode_combo.currentText() != "Image Preview":
                return
            fw = QApplication.focusWidget()
            if isinstance(fw, (QLineEdit, QSpinBox, QDoubleSpinBox)):
                return
            row = int(number) - 1
            if not hasattr(self, "channel_table") or row < 0 or row >= self.channel_table.rowCount():
                return
            chk = self.channel_table.cellWidget(row, 0)
            if chk is None:
                return
            chk.setChecked(not chk.isChecked())
            self.info_label.setText(f"Channel {number} toggled {'ON' if chk.isChecked() else 'OFF'}")
            self.update_channel_preview()
        except Exception as exc:
            self.info_label.setText(f"Channel shortcut error: {exc}")

    def keyPressEvent(self, event):
        # Fallback in case the QShortcut is not active on a platform/widget.
        key = event.key()
        if Qt.Key_1 <= key <= Qt.Key_9 and hasattr(self, "mode_combo") and self.mode_combo.currentText() == "Image Preview":
            self.toggle_preview_channel_by_number(key - Qt.Key_0)
            event.accept()
            return
        super().keyPressEvent(event)

    def _set_widgets_visible(self, widgets, visible: bool):
        for w in widgets:
            w.setVisible(visible)
            w.setEnabled(visible)

    def _worker_count(self):
        try:
            return int(self.worker_spin.value())
        except Exception:
            return 1

    def _set_busy_state(self, busy: bool):
        self.mode_combo.setEnabled(not busy)
        self.worker_spin.setEnabled(not busy)
        self.cancel_job_btn.setEnabled(busy)
        if busy:
            self.progress.setVisible(True)
        else:
            self.progress.setVisible(False)

    def _on_background_progress(self, value, total):
        try:
            value = int(value)
        except Exception:
            value = 0
        try:
            total = int(total)
        except Exception:
            total = 0
        if total <= 0:
            # Unknown-length stage, e.g. streaming a huge GeoJSON before the
            # exact cell count is known. Keep the progress bar animated rather
            # than showing a misleading frozen 0%.
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, max(1, total))
            self.progress.setValue(max(0, min(value, total)))

    def _start_background_job(self, title, job_func, done_callback, *args, **kwargs):
        if self.active_worker is not None and self.active_worker.isRunning():
            QMessageBox.warning(self, "Job already running", "Wait for the current job to finish or click Cancel job.")
            return
        self.info_label.setText(f"Started: {title}")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._set_busy_state(True)
        worker = AppWorker(job_func, *args, **kwargs)
        self.active_worker = worker
        worker.message.connect(self.info_label.setText)
        worker.progress.connect(self._on_background_progress)
        worker.finished_ok.connect(lambda result: self._background_job_done(title, result, done_callback))
        worker.failed.connect(lambda text: self._background_job_failed(title, text))
        worker.finished.connect(lambda: self._set_busy_state(False))
        worker.start()

    def _background_job_done(self, title, result, done_callback):
        self.active_worker = None
        self.progress.setVisible(False)
        try:
            done_callback(result)
        except Exception as exc:
            QMessageBox.critical(self, f"{title} finished, but finalization failed", str(exc))

    def _background_job_failed(self, title, text):
        self.active_worker = None
        self.progress.setVisible(False)
        QMessageBox.critical(self, f"{title} error", text)
        self.info_label.setText(f"{title} failed.")

    def cancel_background_job(self):
        if self.active_worker is not None and self.active_worker.isRunning():
            self.active_worker.cancel()

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

    def _draw_fixed_grid_on_thumb(self, thumb_rgb, full_w, full_h, tile_size, overlap, edge_mode="edge_aligned"):
        pm = _numpy_rgb_to_qpixmap(thumb_rgb)
        painter = QPainter(pm)
        pen_grid = QPen(QColor(220, 0, 0), 1)
        pen_pad = QPen(QColor(230, 190, 0), 2)
        sx = pm.width() / float(full_w)
        sy = pm.height() / float(full_h)
        xs, ys, stride, _ = _compute_tile_grid(full_w, full_h, tile_size, overlap, edge_mode=edge_mode)

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
            <li>Bulk downsampling with raw multichannel OME-TIFF preservation when possible.</li>
            <li>Interactive image preview with channel on/off checkboxes, color assignment, region-based zoom, and JPG capture.</li>
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

    # ========================================================
    # Downsample page
    # ========================================================

    def _build_downsample_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        file_box = QGroupBox("Input images for downsampling")
        file_layout = QVBoxLayout(file_box)
        row1 = QHBoxLayout()
        one_btn = QPushButton("Load one image + preview")
        one_btn.clicked.connect(self.load_downsample_preview_file)
        bulk_btn = QPushButton("Bulk select images")
        bulk_btn.clicked.connect(self.load_downsample_bulk_files)
        folder_btn = QPushButton("Bulk folder")
        folder_btn.clicked.connect(self.load_downsample_bulk_folder)
        self.downsample_file_label = QLabel("No image selected")
        self.downsample_file_label.setStyleSheet("padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px;")
        row1.addWidget(one_btn)
        row1.addWidget(bulk_btn)
        row1.addWidget(folder_btn)
        row1.addWidget(self.downsample_file_label, 1)
        file_layout.addLayout(row1)
        row2 = QHBoxLayout()
        out_btn = QPushButton("Output folder optional")
        out_btn.clicked.connect(self.browse_downsample_output_folder)
        self.downsample_output_edit = QLineEdit("")
        self.downsample_output_edit.setPlaceholderText("Leave empty to save beside each input file")
        row2.addWidget(out_btn)
        row2.addWidget(self.downsample_output_edit, 1)
        file_layout.addLayout(row2)
        layout.addWidget(file_box)

        opt = QGroupBox("Downsample options")
        grid = QGridLayout(opt)
        self.downsample_factor_spin = QDoubleSpinBox()
        self.downsample_factor_spin.setRange(1.01, 256.0)
        self.downsample_factor_spin.setDecimals(2)
        self.downsample_factor_spin.setValue(4.0)
        self.downsample_factor_spin.valueChanged.connect(self.update_downsample_preview)
        self.downsample_output_combo = QComboBox()
        self.downsample_output_combo.addItems(["OME-TIFF (.ome.tif)", "TIFF (.tif)", "JPEG (.jpg)"])
        self.downsample_preserve_chk = QCheckBox("Preserve raw multichannel data for TIFF/OME-TIFF")
        self.downsample_preserve_chk.setChecked(True)
        self.downsample_lossless_chk = QCheckBox("Lossless TIFF compression")
        self.downsample_lossless_chk.setChecked(True)
        self.downsample_overwrite_chk = QCheckBox("Overwrite existing")
        self.downsample_square_spin = QSpinBox()
        self.downsample_square_spin.setRange(64, 100000)
        self.downsample_square_spin.setValue(650)
        self.downsample_square_spin.valueChanged.connect(self.update_downsample_square_size)
        grid.addWidget(QLabel("Downsample factor:"), 0, 0)
        grid.addWidget(self.downsample_factor_spin, 0, 1)
        grid.addWidget(QLabel("Output format:"), 0, 2)
        grid.addWidget(self.downsample_output_combo, 0, 3)
        grid.addWidget(QLabel("Preview square size at original resolution:"), 1, 0)
        grid.addWidget(self.downsample_square_spin, 1, 1)
        grid.addWidget(self.downsample_preserve_chk, 2, 0, 1, 2)
        grid.addWidget(self.downsample_lossless_chk, 2, 2)
        grid.addWidget(self.downsample_overwrite_chk, 2, 3)
        layout.addWidget(opt)

        prev = QGroupBox("Preview: drag the square over the whole image")
        prev_layout = QHBoxLayout(prev)
        self.downsample_whole_label = FixedSquarePreviewLabel()
        self.downsample_whole_label.setFixedSize(520, 380)
        self.downsample_original_label = QLabel("Original-resolution square")
        self.downsample_original_label.setAlignment(Qt.AlignCenter)
        self.downsample_original_label.setFixedSize(260, 260)
        self.downsample_original_label.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")
        self.downsample_result_label = QLabel("Downsampled square")
        self.downsample_result_label.setAlignment(Qt.AlignCenter)
        self.downsample_result_label.setFixedSize(260, 260)
        self.downsample_result_label.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")
        prev_layout.addWidget(self.downsample_whole_label)
        prev_layout.addWidget(self.downsample_original_label)
        prev_layout.addWidget(self.downsample_result_label)
        layout.addWidget(prev, 1)

        btn_row = QHBoxLayout()
        preview_btn = QPushButton("Update preview")
        preview_btn.clicked.connect(self.update_downsample_preview)
        run_btn = QPushButton("RUN BULK DOWNSAMPLE")
        run_btn.clicked.connect(self.run_downsample_bulk)
        btn_row.addWidget(preview_btn)
        btn_row.addWidget(run_btn)
        layout.addLayout(btn_row)
        return page

    def browse_downsample_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder for downsampled images")
        if folder:
            self.downsample_output_edit.setText(folder)

    def load_downsample_preview_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", _image_file_filter())
        if not path:
            return
        self.downsample_preview_path = Path(path)
        self.downsample_paths = [Path(path)]
        try:
            self.backend = ImageBackend().load(path)
            w, h = self.backend.slide_dims
            thumb = self.backend.input_thumbnail(max_side=900)
            self.downsample_whole_label.set_image(thumb, full_w=w, full_h=h,
                square_size_full=self.downsample_square_spin.value(), callback=self._on_downsample_square_changed)
            self.downsample_file_label.setText(f"Preview: {Path(path).name} | {w} × {h} px | Reader: {self.backend.reader}")
            self.info_label.setText("Downsample preview loaded. Drag the square to compare original detail versus selected downsample.")
            self.update_downsample_preview()
        except Exception as e:
            QMessageBox.critical(self, "Downsample load error", str(e))

    def load_downsample_bulk_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select images for bulk downsample", "", _image_file_filter())
        if paths:
            self.downsample_paths = [Path(p) for p in paths]
            self.downsample_file_label.setText(f"Bulk downsample: {len(self.downsample_paths)} file(s) selected")
            self.info_label.setText("Bulk downsample ready. Preview is only loaded when using single-image preview mode.")

    def load_downsample_bulk_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with images to downsample")
        if not folder:
            return
        folder = Path(folder)
        self.downsample_paths = sorted([p for p in folder.iterdir() if p.is_file() and _has_ext(p.name, SUPPORTED_EXTENSIONS)])
        self.downsample_file_label.setText(f"Bulk downsample folder: {len(self.downsample_paths)} file(s)")
        self.info_label.setText("Bulk downsample folder loaded.")

    def update_downsample_square_size(self):
        if hasattr(self, "downsample_whole_label"):
            self.downsample_whole_label.square_size_full = int(self.downsample_square_spin.value())
            self.downsample_whole_label._emit_selection()
            self.downsample_whole_label.update()
        self.update_downsample_preview()

    def _on_downsample_square_changed(self, x, y, w, h):
        self.downsample_roi = (x, y, w, h)

    def update_downsample_preview(self):
        if not getattr(self, "downsample_preview_path", None) or not getattr(self, "backend", None) or not self.backend.path:
            return
        if not self.downsample_roi:
            return
        try:
            x, y, w, h = self.downsample_roi
            preserve_raw = self.downsample_preserve_chk.isChecked() and _has_ext(self.downsample_preview_path.name, TIFF_EXTENSIONS)
            if preserve_raw and hasattr(self.backend, "crop_raw"):
                arr, axes, _ = self.backend.crop_raw(x, y, w, h)
                rgb_original = _array_to_rgb_preview(arr, axes)
                ds_arr, ds_axes = _resize_spatial_array(arr, axes, self.downsample_factor_spin.value())
                rgb_ds = _array_to_rgb_preview(ds_arr, ds_axes)
            else:
                rgb_original, _ = self.backend.crop(x, y, w, h)
                from PIL import Image
                new_w = max(1, int(round(rgb_original.shape[1] / self.downsample_factor_spin.value())))
                new_h = max(1, int(round(rgb_original.shape[0] / self.downsample_factor_spin.value())))
                rgb_ds = np.asarray(Image.fromarray(_to_uint8_rgb(rgb_original)).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)
            self._set_label_pixmap(self.downsample_original_label, _downsample_for_preview(rgb_original, 250))
            self._set_label_pixmap(self.downsample_result_label, _downsample_for_preview(rgb_ds, 250))
            self.info_label.setText(f"Downsample preview ROI X={x}, Y={y}, W={w}, H={h} | DS={self.downsample_factor_spin.value():g}")
        except Exception as e:
            self.info_label.setText(f"Downsample preview error: {e}")

    def run_downsample_bulk(self):
        if not self.downsample_paths:
            QMessageBox.warning(self, "No images", "Select one or more images first.")
            return
        out_dir = self.downsample_output_edit.text().strip()
        factor = float(self.downsample_factor_spin.value())
        output_kind = self.downsample_output_combo.currentText()
        preserve_raw = self.downsample_preserve_chk.isChecked()
        lossless = self.downsample_lossless_chk.isChecked()
        overwrite = self.downsample_overwrite_chk.isChecked()
        self._start_background_job(
            "Bulk downsample",
            _downsample_bulk_job,
            self._on_downsample_done,
            [str(p) for p in self.downsample_paths],
            out_dir,
            factor,
            output_kind,
            preserve_raw,
            lossless,
            overwrite,
            self._worker_count(),
        )

    def _on_downsample_done(self, result):
        QMessageBox.information(
            self,
            "Downsample complete",
            f"Done. Success: {result.get('ok', 0)}. Failed: {result.get('failed', 0)}.\n\nLog:\n{result.get('log_path', '')}",
        )
        self.info_label.setText(f"Downsample complete. Log: {result.get('log_path', '')}")

    # ========================================================
    # Image Preview page

    # ========================================================
    # File Explorer page with thumbnails
    # ========================================================

    def _build_explorer_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)

        folder_box = QGroupBox("Image file explorer")
        folder_layout = QGridLayout(folder_box)
        browse_btn = QPushButton("Open folder")
        browse_btn.clicked.connect(self.browse_explorer_folder)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.scan_explorer_folder)
        self.explorer_folder_edit = QLineEdit()
        self.explorer_folder_edit.setPlaceholderText("Folder containing image files")
        self.explorer_folder_edit.returnPressed.connect(self.scan_explorer_folder)
        self.explorer_thumb_size_spin = QSpinBox()
        self.explorer_thumb_size_spin.setRange(80, 384)
        self.explorer_thumb_size_spin.setValue(112)
        self.explorer_thumb_size_spin.setSuffix(" px")
        folder_layout.addWidget(browse_btn, 0, 0)
        folder_layout.addWidget(self.explorer_folder_edit, 0, 1, 1, 4)
        folder_layout.addWidget(refresh_btn, 0, 5)
        folder_layout.addWidget(QLabel("Thumbnail size:"), 1, 0)
        folder_layout.addWidget(self.explorer_thumb_size_spin, 1, 1)
        folder_layout.addWidget(QLabel("Current folder only. Use the left folder panel to move through the hierarchy."), 1, 2, 1, 4)
        layout.addWidget(folder_box)

        action_box = QGroupBox("Use selected images")
        action_layout = QHBoxLayout(action_box)
        self.explorer_target_combo = QComboBox()
        self.explorer_target_combo.addItems([
            "Image Preview", "IF Threshold Explorer", "Crop", "Tiles", "Downsample", "Batch JPG Preview"
        ])
        send_btn = QPushButton("Send selected to mode")
        send_btn.clicked.connect(self.send_explorer_selection_to_mode)
        preview_btn = QPushButton("Open first selected in Image Preview")
        preview_btn.clicked.connect(lambda: self.send_explorer_selection_to_mode(force_target="Image Preview"))
        select_all_btn = QPushButton("Select all")
        select_all_btn.clicked.connect(self.explorer_select_all)
        clear_btn = QPushButton("Clear selection")
        clear_btn.clicked.connect(self.explorer_clear_selection)
        self.explorer_selection_label = QLabel("0 selected")
        action_layout.addWidget(QLabel("Target:"))
        action_layout.addWidget(self.explorer_target_combo)
        action_layout.addWidget(send_btn)
        action_layout.addWidget(preview_btn)
        action_layout.addStretch()
        action_layout.addWidget(select_all_btn)
        action_layout.addWidget(clear_btn)
        action_layout.addWidget(self.explorer_selection_label)
        layout.addWidget(action_box)

        browser_row = QHBoxLayout()
        browser_row.setSpacing(10)

        folder_nav_box = QGroupBox("Folders")
        folder_nav_layout = QVBoxLayout(folder_nav_box)
        self.explorer_folder_nav = QListWidget()
        self.explorer_folder_nav.setMinimumWidth(170)
        self.explorer_folder_nav.setMaximumWidth(260)
        self.explorer_folder_nav.setMaximumHeight(150)
        self.explorer_folder_nav.itemDoubleClicked.connect(self.navigate_explorer_folder_item)
        self.explorer_folder_nav.itemClicked.connect(self.navigate_explorer_folder_item)
        folder_nav_layout.addWidget(QLabel("Parents and subfolders"))
        folder_nav_layout.addWidget(self.explorer_folder_nav, 1)
        browser_row.addWidget(folder_nav_box, 0)

        self.explorer_list = QListWidget()
        self.explorer_list.setViewMode(QListWidget.IconMode)
        self.explorer_list.setResizeMode(QListWidget.Adjust)
        self.explorer_list.setMovement(QListWidget.Static)
        self.explorer_list.setSpacing(6)
        self.explorer_list.setIconSize(QSize(112, 88))
        self.explorer_list.setGridSize(QSize(150, 132))
        self.explorer_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.explorer_list.itemSelectionChanged.connect(self.update_explorer_selection_label)
        self.explorer_list.itemDoubleClicked.connect(lambda *_: self.send_explorer_selection_to_mode(force_target="Image Preview"))
        browser_row.addWidget(self.explorer_list, 1)
        layout.addLayout(browser_row, 1)

        self.explorer_status_label = QLabel("No folder loaded")
        self.explorer_status_label.setStyleSheet("color: #555; padding: 4px;")
        layout.addWidget(self.explorer_status_label)
        return page

    def browse_explorer_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return
        self.explorer_folder_edit.setText(folder)
        self.scan_explorer_folder()

    def _stop_explorer_worker(self):
        worker = getattr(self, "explorer_thumb_worker", None)
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(1500)
        self.explorer_thumb_worker = None

    def _populate_explorer_folder_nav(self, folder: Path):
        if not hasattr(self, "explorer_folder_nav"):
            return
        self.explorer_folder_nav.clear()
        folder = Path(folder)

        def add_folder_item(label: str, path: Path, kind: str):
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, str(path))
            item.setData(Qt.UserRole + 1, kind)
            item.setToolTip(str(path))
            self.explorer_folder_nav.addItem(item)

        if folder.parent and folder.parent != folder:
            add_folder_item(f"↑ Parent: {folder.parent.name or str(folder.parent)}", folder.parent, "parent")

        ancestors = list(folder.parents)
        ancestors.reverse()
        # Keep the list readable even for very deep HPA folders.
        if len(ancestors) > 8:
            ancestors = ancestors[-8:]
        for anc in ancestors:
            if anc == folder.parent:
                continue
            add_folder_item(f"↰ {anc.name or str(anc)}", anc, "ancestor")

        add_folder_item(f"● Current: {folder.name or str(folder)}", folder, "current")

        try:
            subfolders = sorted([p for p in folder.iterdir() if p.is_dir()], key=lambda x: x.name.lower())
        except Exception:
            subfolders = []
        for sub in subfolders:
            add_folder_item(f"📁 {sub.name}", sub, "child")

    def navigate_explorer_folder_item(self, item):
        path = item.data(Qt.UserRole) if item is not None else None
        kind = item.data(Qt.UserRole + 1) if item is not None else None
        if not path or kind == "current":
            return
        self.explorer_folder_edit.setText(str(path))
        self.scan_explorer_folder()

    def scan_explorer_folder(self):
        folder_text = self.explorer_folder_edit.text().strip() if hasattr(self, "explorer_folder_edit") else ""
        if not folder_text:
            QMessageBox.warning(self, "No folder", "Choose a folder first.")
            return
        folder = Path(folder_text).expanduser()
        if not folder.is_dir():
            QMessageBox.warning(self, "Folder not found", f"This folder does not exist:\n{folder}")
            return

        self.explorer_current_folder = folder
        self._stop_explorer_worker()
        self._populate_explorer_folder_nav(folder)
        try:
            iterator = folder.iterdir()
            self.explorer_paths = sorted([
                p for p in iterator
                if p.is_file() and _has_ext(p.name, SUPPORTED_EXTENSIONS)
            ], key=lambda x: x.name.lower())
        except Exception as exc:
            QMessageBox.critical(self, "Explorer error", str(exc))
            return
        self.explorer_list.clear()

        thumb_size = int(self.explorer_thumb_size_spin.value())
        self.explorer_list.setIconSize(QSize(thumb_size, int(thumb_size * 0.78)))
        self.explorer_list.setGridSize(QSize(max(170, thumb_size + 50), max(170, int(thumb_size * 1.15) + 70)))

        loading_icon = QIcon(_numpy_rgb_to_qpixmap(_placeholder_rgb("Loading", width=thumb_size, height=max(96, int(thumb_size * 0.78)))))
        for p in self.explorer_paths:
            item = QListWidgetItem(loading_icon, f"{p.name}\nloading...")
            item.setData(Qt.UserRole, str(p))
            item.setToolTip(str(p))
            self.explorer_list.addItem(item)

        self.update_explorer_selection_label()
        self.explorer_status_label.setText(f"Found {len(self.explorer_paths)} image file(s) in current folder. Building thumbnails...")
        self.info_label.setText(f"Explorer loaded: {len(self.explorer_paths)} image file(s) in {folder.name or folder}.")

        if not self.explorer_paths:
            self.progress.setVisible(False)
            self.explorer_status_label.setText("Explorer ready: no image files in this folder.")
            return

        self.progress.setVisible(True)
        self.progress.setRange(0, len(self.explorer_paths))
        self.progress.setValue(0)
        self.explorer_thumb_worker = ThumbnailExplorerWorker(self.explorer_paths, max_side=thumb_size, parent=self)
        self.explorer_thumb_worker.item_ready.connect(self._on_explorer_thumbnail_ready)
        self.explorer_thumb_worker.item_failed.connect(self._on_explorer_thumbnail_failed)
        self.explorer_thumb_worker.progress.connect(self._on_explorer_thumbnail_progress)
        self.explorer_thumb_worker.finished_count.connect(self._on_explorer_thumbnail_finished)
        self.explorer_thumb_worker.start()

    def _find_explorer_item_by_path(self, path: str):
        path = str(path)
        for i in range(self.explorer_list.count()):
            item = self.explorer_list.item(i)
            if item.data(Qt.UserRole) == path:
                return item
        return None

    def _on_explorer_thumbnail_ready(self, path: str, rgb, reader: str):
        item = self._find_explorer_item_by_path(path)
        if item is None:
            return
        thumb_size = int(self.explorer_thumb_size_spin.value())
        pm = _numpy_rgb_to_qpixmap(_to_uint8_rgb(rgb))
        pm = pm.scaled(QSize(thumb_size, int(thumb_size * 0.78)), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        item.setIcon(QIcon(pm))
        item.setText(f"{Path(path).name}\n{reader}")
        item.setData(Qt.UserRole + 1, reader)

    def _on_explorer_thumbnail_failed(self, path: str, error: str):
        item = self._find_explorer_item_by_path(path)
        if item is None:
            return
        item.setText(f"{Path(path).name}\npreview failed")
        item.setToolTip(f"{path}\n\nThumbnail error:\n{error}")

    def _on_explorer_thumbnail_progress(self, done: int, total: int):
        if hasattr(self, "progress"):
            self.progress.setRange(0, max(1, total))
            self.progress.setValue(done)
        if hasattr(self, "explorer_status_label"):
            self.explorer_status_label.setText(f"Building thumbnails: {done}/{total}")

    def _on_explorer_thumbnail_finished(self, done: int):
        self.progress.setVisible(False)
        self.explorer_status_label.setText(f"Explorer ready: {len(self.explorer_paths)} image file(s).")

    def explorer_selected_paths(self) -> List[Path]:
        if not hasattr(self, "explorer_list"):
            return []
        paths = []
        for item in self.explorer_list.selectedItems():
            p = item.data(Qt.UserRole)
            if p:
                paths.append(Path(p))
        # Preserve the visual/list order instead of the arbitrary selection order.
        selected = set(str(p) for p in paths)
        return [Path(self.explorer_list.item(i).data(Qt.UserRole)) for i in range(self.explorer_list.count()) if self.explorer_list.item(i).data(Qt.UserRole) in selected]

    def update_explorer_selection_label(self):
        n = len(self.explorer_selected_paths()) if hasattr(self, "explorer_list") else 0
        if hasattr(self, "explorer_selection_label"):
            self.explorer_selection_label.setText(f"{n} selected")

    def explorer_select_all(self):
        if hasattr(self, "explorer_list"):
            self.explorer_list.selectAll()

    def explorer_clear_selection(self):
        if hasattr(self, "explorer_list"):
            self.explorer_list.clearSelection()

    def _load_crop_from_path(self, path: Path):
        path = Path(path)
        self.backend = ImageBackend().load(str(path))
        w, h = self.backend.slide_dims
        self.crop_file_label.setText(path.name)
        self.x_spin.setMaximum(max(0, w - 1))
        self.y_spin.setMaximum(max(0, h - 1))
        self.w_spin.setMaximum(max(1, w))
        self.h_spin.setMaximum(max(1, h))
        self.crop_zoom = 1.0
        self.crop_center = (w / 2.0, h / 2.0)
        self.crop_preview_meta = {}
        self.info_label.setText(f"Loaded from Explorer: {path.name} | Size: {w} x {h} px | Reader: {self.backend.reader}")
        if self.crop_preview_chk.isChecked():
            self.refresh_crop_input_preview()

    def _load_preview_from_path(self, path: Path):
        path = Path(path)
        try:
            if getattr(self, "preview_backend", None) is not None:
                try:
                    self.preview_backend.close()
                except Exception:
                    pass
            self.preview_path = path
            self.preview_backend = ImageBackend().load(str(path))
            self.preview_zoom = 1.0
            self.preview_arr, self.preview_axes, self.preview_meta = read_zoom_region_from_backend(
                self.preview_backend, center_xy=None, zoom=1.0,
                viewport_size=(max(640, self.preview_image_label.width()), max(420, self.preview_image_label.height())),
                max_side=900,
            )
            full_dims = self.preview_meta.get("full_dims")
            if full_dims:
                self.preview_full_dims = tuple(full_dims)
                self.preview_center = (self.preview_full_dims[0] / 2.0, self.preview_full_dims[1] / 2.0)
                self.preview_tile_center = self.preview_center
            else:
                h, w = np.asarray(self.preview_arr).shape[:2]
                self.preview_full_dims = (w, h)
                self.preview_center = (w / 2.0, h / 2.0)
                self.preview_tile_center = self.preview_center
            self.preview_file_label.setText(
                f"{self.preview_path.name} | axes={self.preview_axes} | reader={self.preview_meta.get('reader')} | full={self.preview_full_dims[0]} × {self.preview_full_dims[1]}"
            )
            self.populate_channel_table()
            self.update_channel_preview()
            self.info_label.setText(f"Loaded from Explorer into Image Preview: {path.name}")
        except Exception as e:
            QMessageBox.critical(self, "Preview load error", str(e))

    def _load_downsample_from_paths(self, paths: List[Path]):
        paths = [Path(p) for p in paths]
        self.downsample_paths = paths
        if len(paths) == 1:
            path = paths[0]
            self.downsample_preview_path = path
            self.backend = ImageBackend().load(str(path))
            w, h = self.backend.slide_dims
            thumb = self.backend.input_thumbnail(max_side=900)
            self.downsample_whole_label.set_image(
                thumb, full_w=w, full_h=h,
                square_size_full=self.downsample_square_spin.value(),
                callback=self._on_downsample_square_changed,
            )
            self.downsample_file_label.setText(f"Preview: {path.name} | {w} × {h} px | Reader: {self.backend.reader}")
            self.update_downsample_preview()
        else:
            self.downsample_preview_path = None
            self.downsample_file_label.setText(f"Bulk downsample from Explorer: {len(paths)} file(s) selected")
        self.info_label.setText(f"Explorer selection linked to Downsample: {len(paths)} file(s).")

    def send_explorer_selection_to_mode(self, force_target: Optional[str] = None):
        paths = self.explorer_selected_paths()
        if not paths:
            QMessageBox.warning(self, "No selection", "Select one or more images in the Explorer first.")
            return
        target = force_target or self.explorer_target_combo.currentText()
        try:
            if target == "Image Preview":
                self._load_preview_from_path(paths[0])
                self.mode_combo.setCurrentText("Image Preview")
                if len(paths) > 1:
                    self.info_label.setText(f"Image Preview uses one image; opened first selected: {paths[0].name}")
            elif target == "IF Threshold Explorer":
                self._load_if_image_from_path(paths[0])
                self.mode_combo.setCurrentText("IF Threshold Explorer")
                if len(paths) > 1:
                    self.info_label.setText(f"IF Threshold Explorer uses one image; loaded first selected: {paths[0].name}")
            elif target == "Crop":
                self._load_crop_from_path(paths[0])
                self.mode_combo.setCurrentText("Crop")
                if len(paths) > 1:
                    self.info_label.setText(f"Crop uses one image; loaded first selected: {paths[0].name}")
            elif target == "Tiles":
                self.bulk_paths = paths
                self._load_tiles_preview_image(paths[0])
                self.mode_combo.setCurrentText("Tiles")
            elif target == "Downsample":
                self._load_downsample_from_paths(paths)
                self.mode_combo.setCurrentText("Downsample")
            elif target == "Batch JPG Preview":
                self.batch_channel_paths = paths
                if hasattr(self, "batch_channel_label"):
                    self.batch_channel_label.setText(f"Selected from Explorer: {len(paths)} image file(s)")
                self.mode_combo.setCurrentText("Image Preview")
                self.info_label.setText("Explorer selection linked to Batch JPG Preview. Open the Batch JPG Preview section in Image Preview and run export.")
        except Exception as e:
            QMessageBox.critical(self, "Explorer link error", str(e))

    # ========================================================

    def _build_preview_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        load_btn = QPushButton("Load image")
        load_btn.clicked.connect(self.load_preview_file)
        self.preview_file_label = QLabel("No file loaded")
        self.preview_file_label.setStyleSheet("padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px;")
        top.addWidget(load_btn)
        top.addWidget(self.preview_file_label, 1)
        layout.addLayout(top)

        save_box = QGroupBox("Image Preview display tools")
        save_row = QHBoxLayout(save_box)
        self.preview_suffix_edit = QLineEdit("preview")
        self.preview_suffix_edit.setPlaceholderText("Suffix for JPG capture")
        save_btn = QPushButton("Save current JPG capture")
        save_btn.clicked.connect(self.save_channel_preview_jpg)
        zoom_in_btn = QPushButton("Zoom +")
        zoom_in_btn.clicked.connect(lambda: self.change_preview_zoom(1.25))
        zoom_out_btn = QPushButton("Zoom -")
        zoom_out_btn.clicked.connect(lambda: self.change_preview_zoom(0.8))
        zoom_fit_btn = QPushButton("Fit")
        zoom_fit_btn.clicked.connect(lambda: self.set_preview_zoom(1.0))
        rect_zoom_btn = QPushButton("Zoom to rectangle")
        rect_zoom_btn.setToolTip("Click this, then drag a rectangle on the current preview to zoom to that region.")
        rect_zoom_btn.clicked.connect(self.start_preview_rectangle_zoom)
        self.preview_zoom_label = QLabel("Zoom: 100%")
        save_row.addWidget(QLabel("Suffix:"))
        save_row.addWidget(self.preview_suffix_edit)
        save_row.addWidget(save_btn)
        save_row.addStretch()
        save_row.addWidget(zoom_out_btn)
        save_row.addWidget(zoom_in_btn)
        save_row.addWidget(zoom_fit_btn)
        save_row.addWidget(rect_zoom_btn)
        save_row.addWidget(self.preview_zoom_label)
        layout.addWidget(save_box)

        main = QHBoxLayout()
        main.setSpacing(10)

        left_panel = QWidget()
        left_panel.setMinimumWidth(340)
        left_panel.setMaximumWidth(410)
        left_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)

        self.channel_table = QTableWidget()
        self.channel_table.setColumnCount(3)
        self.channel_table.setHorizontalHeaderLabels(["On", "Channel", "Color"])
        self.channel_table.setColumnWidth(0, 45)
        self.channel_table.setColumnWidth(1, 80)
        self.channel_table.setColumnWidth(2, 140)
        self.channel_table.setMinimumWidth(300)
        self.channel_table.setMaximumHeight(190)
        self.channel_table.verticalHeader().setVisible(False)
        self.channel_table.verticalHeader().setDefaultSectionSize(24)
        self.channel_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.channel_table.setAlternatingRowColors(True)
        left.addWidget(QLabel("Channels / display mapping (max. 6 shown)"))
        left.addWidget(self.channel_table, 0)
        update_btn = QPushButton("Update preview")
        update_btn.clicked.connect(self.update_channel_preview)
        left.addWidget(update_btn)

        geo_box = QGroupBox("GeoJSON annotations / lightweight annotator")
        geo_layout = QGridLayout(geo_box)
        load_geo_btn = QPushButton("Load GeoJSON")
        load_geo_btn.clicked.connect(self.load_preview_geojson)
        save_geo_btn = QPushButton("Save GeoJSON")
        save_geo_btn.clicked.connect(self.save_preview_geojson)
        clear_geo_btn = QPushButton("Clear")
        clear_geo_btn.clicked.connect(self.clear_preview_geojson)
        self.preview_show_annotations_chk = QCheckBox("Show all")
        self.preview_show_annotations_chk.setChecked(True)
        self.preview_show_annotations_chk.stateChanged.connect(self.update_preview_annotation_style)
        geo_layout.addWidget(load_geo_btn, 0, 0)
        geo_layout.addWidget(save_geo_btn, 0, 1)
        geo_layout.addWidget(clear_geo_btn, 0, 2)
        geo_layout.addWidget(self.preview_show_annotations_chk, 0, 3)

        self.annotation_table = QTableWidget()
        self.annotation_table.setColumnCount(3)
        self.annotation_table.setHorizontalHeaderLabels(["On", "Name", "Color"])
        self.annotation_table.setColumnWidth(0, 42)
        self.annotation_table.setColumnWidth(1, 165)
        self.annotation_table.setColumnWidth(2, 72)
        self.annotation_table.verticalHeader().setVisible(False)
        self.annotation_table.verticalHeader().setDefaultSectionSize(24)
        self.annotation_table.setMinimumHeight(210)
        self.annotation_table.setMaximumHeight(330)
        self.annotation_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.annotation_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.annotation_table.setAlternatingRowColors(True)
        geo_layout.addWidget(self.annotation_table, 1, 0, 1, 4)

        self.preview_annotation_fill_chk = QCheckBox("Fill polygons")
        self.preview_annotation_fill_chk.setChecked(True)
        self.preview_annotation_fill_chk.stateChanged.connect(self.update_preview_annotation_style)
        geo_layout.addWidget(self.preview_annotation_fill_chk, 2, 0, 1, 2)
        geo_layout.addWidget(QLabel("Opacity:"), 3, 0)
        self.preview_annotation_opacity_slider = QSlider(Qt.Horizontal)
        self.preview_annotation_opacity_slider.setRange(0, 100)
        self.preview_annotation_opacity_slider.setValue(45)
        self.preview_annotation_opacity_slider.valueChanged.connect(self.update_preview_annotation_style)
        self.preview_annotation_opacity_value_label = QLabel("45%")
        geo_layout.addWidget(self.preview_annotation_opacity_slider, 3, 1, 1, 2)
        geo_layout.addWidget(self.preview_annotation_opacity_value_label, 3, 3)
        geo_layout.addWidget(QLabel("Boundary:"), 4, 0)
        self.preview_annotation_boundary_spin = QSpinBox()
        self.preview_annotation_boundary_spin.setRange(1, 25)
        self.preview_annotation_boundary_spin.setValue(2)
        self.preview_annotation_boundary_spin.valueChanged.connect(self.update_preview_annotation_style)
        geo_layout.addWidget(self.preview_annotation_boundary_spin, 4, 1)

        geo_layout.addWidget(QLabel("Draw class:"), 5, 0)
        self.preview_draw_class_combo = QComboBox()
        self.preview_draw_class_combo.setEditable(True)
        self.preview_draw_class_combo.addItems(["Tissue", "Tumor", "Stroma", "Necrosis", "Immune", "Other"])
        self.preview_draw_class_combo.setCurrentText("Tissue")
        geo_layout.addWidget(self.preview_draw_class_combo, 5, 1, 1, 2)
        self.preview_draw_color_btn = QPushButton(self.preview_new_annotation_color.name())
        self.preview_draw_color_btn.setStyleSheet(f"background-color: {self.preview_new_annotation_color.name()}; color: white;")
        self.preview_draw_color_btn.clicked.connect(self.choose_preview_draw_color)
        geo_layout.addWidget(self.preview_draw_color_btn, 5, 3)

        self.preview_draw_mode_combo = QComboBox()
        self.preview_draw_mode_combo.addItems(["Pan / select", "Polygon", "Freehand", "Rectangle"])
        self.preview_draw_mode_combo.currentIndexChanged.connect(self.set_preview_annotation_draw_mode)
        geo_layout.addWidget(QLabel("Draw mode:"), 6, 0)
        geo_layout.addWidget(self.preview_draw_mode_combo, 6, 1, 1, 3)

        finish_poly_btn = QPushButton("Finish polygon")
        finish_poly_btn.clicked.connect(self.finish_preview_polygon_annotation)
        cancel_draw_btn = QPushButton("Cancel draw")
        cancel_draw_btn.clicked.connect(self.cancel_preview_annotation_drawing)
        undo_ann_btn = QPushButton("Undo last")
        undo_ann_btn.clicked.connect(self.undo_last_preview_annotation)
        geo_layout.addWidget(finish_poly_btn, 7, 0)
        geo_layout.addWidget(cancel_draw_btn, 7, 1)
        geo_layout.addWidget(undo_ann_btn, 7, 2)
        note_draw = QLabel("Polygon: left-click points, double-click or Finish polygon. Freehand/Rectangle: left-drag. Right-drag still pans.")
        note_draw.setWordWrap(True)
        note_draw.setStyleSheet("color: #555;")
        geo_layout.addWidget(note_draw, 8, 0, 1, 4)
        left.addWidget(geo_box, 1)

        tile_box = QGroupBox("Specific square tile capture")
        tile_layout = QGridLayout(tile_box)
        tile_layout.addWidget(QLabel("Tile size px:"), 0, 0)
        self.preview_tile_size_spin = QSpinBox()
        self.preview_tile_size_spin.setRange(16, 100000)
        self.preview_tile_size_spin.setValue(1024)
        self.preview_tile_size_spin.valueChanged.connect(self.update_preview_tile_overlay)
        tile_layout.addWidget(self.preview_tile_size_spin, 0, 1)
        self.preview_tile_capture_chk = QCheckBox("Tile capture pop-up")
        self.preview_tile_capture_chk.setToolTip("Shows a movable non-modal window with the selected square tile. Left-drag on the image moves the tile. Right-drag pans the view.")
        self.preview_tile_capture_chk.stateChanged.connect(self.toggle_preview_tile_mode)
        tile_layout.addWidget(self.preview_tile_capture_chk, 1, 0, 1, 2)
        self.preview_tile_info_label = QLabel("Tile: not set")
        self.preview_tile_info_label.setWordWrap(True)
        tile_layout.addWidget(self.preview_tile_info_label, 2, 0, 1, 2)
        note = QLabel("When enabled, left-drag on the main preview moves the blue square. Right-drag pans the view. The pop-up updates and contains the Save Tile JPG button.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #555;")
        tile_layout.addWidget(note, 3, 0, 1, 2)
        left.addWidget(tile_box)
        left.addStretch(1)

        main.addWidget(left_panel, 0)

        self.preview_image_label = ZoomRegionPreviewLabel()
        self.preview_image_label.setText("Preview")
        self.preview_image_label.setAlignment(Qt.AlignCenter)
        self.preview_image_label.setMinimumSize(460, 300)
        self.preview_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_image_label.set_annotation_draw_callbacks(
            created_callback=self._on_preview_annotation_drawn,
            preview_callback=self._on_preview_annotation_preview_changed,
        )
        # No maximum size: in full-screen the preview uses all available space.
        main.addWidget(self.preview_image_label, 1)
        layout.addLayout(main, 1)
        return page

    def load_preview_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image to preview", "", _image_file_filter())
        if not path:
            return
        try:
            if getattr(self, "preview_backend", None) is not None:
                try:
                    self.preview_backend.close()
                except Exception:
                    pass
            self.preview_path = Path(path)
            self.preview_backend = ImageBackend().load(path)
            self.preview_zoom = 1.0
            # Read a memory-light visible-region preview first to detect channels.
            self.preview_arr, self.preview_axes, self.preview_meta = read_zoom_region_from_backend(
                self.preview_backend, center_xy=None, zoom=1.0,
                viewport_size=(max(640, self.preview_image_label.width()), max(420, self.preview_image_label.height())),
                max_side=900,
            )
            full_dims = self.preview_meta.get("full_dims")
            if full_dims:
                self.preview_full_dims = tuple(full_dims)
                self.preview_center = (self.preview_full_dims[0] / 2.0, self.preview_full_dims[1] / 2.0)
                self.preview_tile_center = self.preview_center
            else:
                h, w = np.asarray(self.preview_arr).shape[:2]
                self.preview_full_dims = (w, h)
                self.preview_center = (w / 2.0, h / 2.0)
                self.preview_tile_center = self.preview_center
            self.preview_file_label.setText(
                f"{self.preview_path.name} | axes={self.preview_axes} | reader={self.preview_meta.get('reader')} | full={self.preview_full_dims[0]} × {self.preview_full_dims[1]}"
            )
            self.populate_channel_table()
            self.update_channel_preview()
        except Exception as e:
            QMessageBox.critical(self, "Preview load error", str(e))

    def populate_channel_table(self, settings: Optional[List[Dict[str, Any]]] = None):
        if self.preview_arr is None:
            return
        n_total = _count_display_channels(self.preview_arr, self.preview_axes)
        n = min(int(n_total), 6)
        self.channel_table.setRowCount(n)
        try:
            h = self.channel_table.horizontalHeader().height() + n * self.channel_table.verticalHeader().defaultSectionSize() + 8
            self.channel_table.setMaximumHeight(max(90, min(190, h)))
        except Exception:
            pass

        # For normal RGB images, default mapping should preserve RGB appearance.
        _, ax2 = _representative_yx_or_yxc(self.preview_arr, self.preview_axes)
        rgb_sample = (ax2 == "YXS" and np.asarray(self.preview_arr).ndim == 3 and np.asarray(self.preview_arr).shape[-1] in (3, 4))
        rgb_names = ["R", "G", "B", "A"]
        rgb_colors = ["red", "green", "blue", "gray"]

        for i in range(n):
            st = settings[i] if settings and i < len(settings) else {}
            chk = QCheckBox()
            chk.setChecked(bool(st.get("visible", True)))
            chk.stateChanged.connect(self.update_channel_preview)
            self.channel_table.setCellWidget(i, 0, chk)

            label = rgb_names[i] if rgb_sample and i < len(rgb_names) else f"C{i}"
            item = QTableWidgetItem(label)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.channel_table.setItem(i, 1, item)

            color_combo = QComboBox()
            color_combo.addItems(list(COLOR_MAPS.keys()))
            default_color = st.get("color", rgb_colors[i] if rgb_sample and i < len(rgb_colors) else DEFAULT_CHANNEL_COLORS[i % len(DEFAULT_CHANNEL_COLORS)])
            color_combo.setCurrentText(default_color if default_color in COLOR_MAPS else "gray")
            color_combo.currentIndexChanged.connect(self.update_channel_preview)
            self.channel_table.setCellWidget(i, 2, color_combo)

    def get_channel_settings_from_table(self) -> List[Dict[str, Any]]:
        settings = []
        for r in range(self.channel_table.rowCount()):
            chk = self.channel_table.cellWidget(r, 0)
            color_combo = self.channel_table.cellWidget(r, 2)
            settings.append({
                "channel": r,
                "visible": chk.isChecked() if chk else True,
                "color": color_combo.currentText() if color_combo else DEFAULT_CHANNEL_COLORS[r % len(DEFAULT_CHANNEL_COLORS)],
            })
        return settings

    def update_channel_preview(self):
        if self.preview_path is None:
            return
        try:
            viewport = (max(640, self.preview_image_label.width()), max(420, self.preview_image_label.height()))
            if getattr(self, "preview_backend", None) is None:
                self.preview_backend = ImageBackend().load(str(self.preview_path))
            arr, axes, meta = read_zoom_region_from_backend(
                self.preview_backend,
                center_xy=getattr(self, "preview_center", None),
                zoom=max(1.0, float(self.preview_zoom)),
                viewport_size=viewport,
                max_side=max(1400, int(max(viewport) * 2)),
            )
            self.preview_arr = arr
            self.preview_axes = axes
            self.preview_meta = meta
            if meta.get("full_dims"):
                self.preview_full_dims = tuple(meta["full_dims"])
            self.preview_last_rgb = render_channel_composite(arr, axes, self.get_channel_settings_from_table())
            self._update_preview_pixmap()
            roi = meta.get("roi")
            self.info_label.setText(
                f"Preview loaded at {int(self.preview_zoom * 100)}% | ROI={roi} | reader={meta.get('reader')}"
            )
        except Exception as e:
            self.info_label.setText(f"Preview render error: {e}")

    def _update_preview_pixmap(self):
        if self.preview_last_rgb is None:
            return
        roi = self.preview_meta.get("roi") if isinstance(self.preview_meta, dict) else None
        full_dims = getattr(self, "preview_full_dims", None)
        self.preview_image_label.set_preview(
            _to_uint8_rgb(self.preview_last_rgb),
            roi_full=roi,
            full_dims=full_dims,
            center_callback=self._on_preview_center_changed,
            rectangle_callback=self._on_preview_rectangle_zoom,
            zoom_callback=self._on_preview_view_zoom,
        )
        self.preview_image_label.set_annotations(getattr(self, "preview_annotations", []))
        self.update_preview_annotation_style()
        self.update_preview_tile_overlay()
        self.preview_zoom_label.setText(f"Zoom: {int(self.preview_zoom * 100)}%")

    def schedule_preview_region_update(self, delay_ms: int = 35):
        if hasattr(self, "_preview_render_timer"):
            self._preview_render_timer.start(max(0, int(delay_ms)))
        else:
            self.update_channel_preview()

    def _on_preview_center_changed(self, cx, cy):
        self.preview_center = (float(cx), float(cy))
        # Debounce left-drag panning so very large slides do not queue many
        # expensive region reads while the mouse is moving.
        self.schedule_preview_region_update(delay_ms=35)

    def _on_preview_view_zoom(self, factor: float, center_xy=None):
        self.change_preview_zoom(factor, center_xy=center_xy)

    def start_preview_rectangle_zoom(self):
        if self.preview_path is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            return
        self.preview_image_label.enable_rectangle_zoom(True)
        self.info_label.setText("Rectangle zoom: drag a rectangle on the current preview.")

    def _on_preview_rectangle_zoom(self, x, y, w, h):
        if not getattr(self, "preview_full_dims", None):
            return
        full_w, full_h = self.preview_full_dims
        if w <= 1 or h <= 1:
            return
        self.preview_center = (x + w / 2.0, y + h / 2.0)
        # Add a small margin so the selected rectangle is visible inside the viewport.
        zoom_w = float(full_w) / max(1.0, float(w))
        zoom_h = float(full_h) / max(1.0, float(h))
        self.preview_zoom = max(1.0, min(64.0, 0.90 * min(zoom_w, zoom_h)))
        self.update_channel_preview()

    def change_preview_zoom(self, factor: float, center_xy=None):
        if center_xy is not None:
            self.preview_center = (float(center_xy[0]), float(center_xy[1]))
        self.preview_zoom = max(1.0, min(32.0, self.preview_zoom * float(factor)))
        if self.preview_zoom == 1.0 and getattr(self, "preview_full_dims", None):
            self.preview_center = (self.preview_full_dims[0] / 2.0, self.preview_full_dims[1] / 2.0)
        self.schedule_preview_region_update(delay_ms=20)

    def set_preview_zoom(self, value: float):
        self.preview_zoom = max(1.0, float(value))
        # Re-center on the full image when returning to fit.
        if self.preview_zoom == 1.0 and getattr(self, "preview_full_dims", None):
            self.preview_center = (self.preview_full_dims[0] / 2.0, self.preview_full_dims[1] / 2.0)
        self.update_channel_preview()

    def save_channel_preview_jpg(self):
        if self.preview_path is None or self.preview_last_rgb is None:
            QMessageBox.warning(self, "No preview", "Load and render an image first.")
            return
        suffix = self.preview_suffix_edit.text().strip() or "preview"
        out_path = self.preview_path.parent / f"{self.preview_path.stem}_{suffix}.jpg"
        try:
            pixmap = self.preview_image_label.grab()
            ok = pixmap.save(str(out_path), "JPG", 95)
            if not ok:
                raise RuntimeError("Qt could not save the displayed preview capture.")
            QMessageBox.information(self, "Saved", f"Saved displayed JPG capture:\n{out_path}")
            self.info_label.setText(f"Saved displayed preview capture: {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save preview error", str(e))



    def _current_preview_draw_class(self) -> str:
        if hasattr(self, "preview_draw_class_combo"):
            text = self.preview_draw_class_combo.currentText().strip()
            if text:
                return text
        return "Annotation"

    def choose_preview_draw_color(self):
        current = QColor(getattr(self, "preview_new_annotation_color", QColor(255, 0, 0)))
        color = QColorDialog.getColor(current, self, "Choose drawing annotation colour")
        if color.isValid():
            self.preview_new_annotation_color = QColor(color)
            if hasattr(self, "preview_draw_color_btn"):
                self.preview_draw_color_btn.setText(color.name())
                self.preview_draw_color_btn.setStyleSheet(
                    f"background-color: {color.name()}; "
                    f"color: {'white' if color.lightness() < 140 else 'black'};"
                )

    def set_preview_annotation_draw_mode(self, *args):
        text = self.preview_draw_mode_combo.currentText() if hasattr(self, "preview_draw_mode_combo") else "Pan / select"
        mapping = {
            "Pan / select": "none",
            "Polygon": "polygon",
            "Freehand": "freehand",
            "Rectangle": "rectangle",
        }
        mode = mapping.get(text, "none")
        self.preview_annotation_draw_mode = mode
        if hasattr(self, "preview_image_label"):
            self.preview_image_label.set_annotation_draw_mode(mode)
        if mode == "none":
            self.info_label.setText("Annotation drawing disabled. Right-drag pans; wheel zooms.")
        elif mode == "polygon":
            self.info_label.setText("Polygon mode: left-click points; double-click or Finish polygon to save.")
        elif mode == "freehand":
            self.info_label.setText("Freehand mode: left-drag to draw. Release to save annotation.")
        elif mode == "rectangle":
            self.info_label.setText("Rectangle annotation mode: left-drag to draw. Release to save annotation.")

    def finish_preview_polygon_annotation(self):
        if not hasattr(self, "preview_image_label"):
            return
        if not self.preview_image_label.finish_polygon_annotation():
            self.info_label.setText("Polygon was not saved. Add at least 3 points first.")

    def cancel_preview_annotation_drawing(self):
        if hasattr(self, "preview_image_label"):
            self.preview_image_label.cancel_active_annotation()
        self.info_label.setText("Current annotation drawing cancelled.")

    def _on_preview_annotation_preview_changed(self, ring):
        # Reserved for future coordinate readout. Keeping it light avoids UI lag.
        pass

    def _on_preview_annotation_drawn(self, ring, source="manual"):
        if self.preview_path is None:
            return
        class_name = self._current_preview_draw_class()
        color = QColor(getattr(self, "preview_new_annotation_color", QColor(255, 0, 0)))
        ann_id = f"manual_{self._preview_annotation_counter:04d}"
        self._preview_annotation_counter += 1
        annotation = {
            "id": ann_id,
            "feature_id": ann_id,
            "class_name": class_name,
            "rings": [list(ring)],
            "color": QColor(color),
            "properties": {
                "objectType": "annotation",
                "name": class_name,
                "source": f"TiffCropper {source}",
                "classification": {
                    "name": class_name,
                    "colorRGB": _qcolor_to_qupath_color_rgb(color),
                },
            },
        }
        self.preview_annotations.append(annotation)
        if hasattr(self, "preview_image_label"):
            self.preview_image_label.set_annotations(self.preview_annotations)
        # Preserve user visibility choices for existing classes and add the new class.
        styles = getattr(self, "preview_annotation_styles", {}) or {}
        if class_name not in styles:
            styles[class_name] = {"visible": True, "color": QColor(color)}
            self.preview_annotation_styles = styles
        self.populate_annotation_table()
        self.update_preview_annotation_style()
        self.info_label.setText(f"Added {source} annotation: {class_name} ({len(ring)} points).")

    def undo_last_preview_annotation(self):
        if not getattr(self, "preview_annotations", None):
            self.info_label.setText("No annotation to undo.")
            return
        removed = self.preview_annotations.pop()
        if hasattr(self, "preview_image_label"):
            self.preview_image_label.set_annotations(self.preview_annotations)
        self.populate_annotation_table()
        self.update_preview_annotation_style()
        self.info_label.setText(f"Removed last annotation: {removed.get('class_name', 'annotation')}.")

    def save_preview_geojson(self):
        if not getattr(self, "preview_annotations", None):
            QMessageBox.warning(self, "No annotations", "There are no annotations to save.")
            return
        default_dir = str(self.preview_path.parent) if self.preview_path is not None else ""
        default_name = f"{self.preview_path.stem}_annotations.geojson" if self.preview_path is not None else "annotations.geojson"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save GeoJSON annotations",
            str(Path(default_dir) / default_name) if default_dir else default_name,
            "GeoJSON files (*.geojson);;JSON files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            styles = self.get_preview_annotation_class_styles_from_table()
            annotations_to_save = []
            for ann in self.preview_annotations:
                ann_copy = dict(ann)
                cls = str(ann_copy.get("class_name", "annotation") or "annotation")
                if cls in styles and styles[cls].get("color") is not None:
                    ann_copy["color"] = QColor(styles[cls].get("color"))
                    props = dict(ann_copy.get("properties", {}) or {})
                    props["classification"] = {
                        "name": cls,
                        "colorRGB": _qcolor_to_qupath_color_rgb(ann_copy["color"]),
                    }
                    ann_copy["properties"] = props
                annotations_to_save.append(ann_copy)
            n = save_geojson_annotations(path, annotations_to_save, image_path=str(self.preview_path) if self.preview_path else None)
            QMessageBox.information(self, "Saved", f"Saved {n} annotation object(s):\n{path}")
            self.info_label.setText(f"Saved GeoJSON annotations: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save GeoJSON error", str(e))

    def _annotation_default_color_for_class(self, class_name: str) -> QColor:
        for ann in getattr(self, "preview_annotations", []) or []:
            if str(ann.get("class_name", "annotation") or "annotation") == str(class_name):
                return QColor(ann.get("color", QColor(255, 0, 0)))
        return QColor(255, 0, 0)

    def populate_annotation_table(self):
        if not hasattr(self, "annotation_table"):
            return
        annotations = getattr(self, "preview_annotations", []) or []
        classes = []
        for ann in annotations:
            cls = str(ann.get("class_name", "annotation") or "annotation")
            if cls not in classes:
                classes.append(cls)
        classes.sort(key=lambda x: x.lower())
        self.annotation_table.setRowCount(len(classes))

        # Preserve previous user edits when loading/repopulating.
        old_styles = getattr(self, "preview_annotation_styles", {}) or {}
        new_styles = {}
        for row, cls in enumerate(classes):
            old = old_styles.get(cls, {})
            default_color = QColor(old.get("color", self._annotation_default_color_for_class(cls)))
            visible = bool(old.get("visible", True))
            new_styles[cls] = {"visible": visible, "color": default_color}

            chk = QCheckBox()
            chk.setChecked(visible)
            chk.stateChanged.connect(self.update_preview_annotation_style)
            self.annotation_table.setCellWidget(row, 0, chk)

            item = QTableWidgetItem(cls)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.annotation_table.setItem(row, 1, item)

            btn = QPushButton(default_color.name())
            btn.setStyleSheet(
                f"background-color: {default_color.name()}; "
                f"color: {'white' if default_color.lightness() < 140 else 'black'};"
            )
            btn.clicked.connect(lambda _, name=cls: self.choose_preview_annotation_class_color(name))
            self.annotation_table.setCellWidget(row, 2, btn)

        self.preview_annotation_styles = new_styles
        try:
            n = min(12, max(1, len(classes)))
            h = self.annotation_table.horizontalHeader().height() + n * self.annotation_table.verticalHeader().defaultSectionSize() + 12
            self.annotation_table.setMaximumHeight(max(230, min(360, h)))
        except Exception:
            pass
        self.update_preview_annotation_style()

    def get_preview_annotation_class_styles_from_table(self) -> Dict[str, Dict[str, Any]]:
        styles = {}
        if not hasattr(self, "annotation_table"):
            return getattr(self, "preview_annotation_styles", {}) or {}
        for row in range(self.annotation_table.rowCount()):
            item = self.annotation_table.item(row, 1)
            if item is None:
                continue
            cls = item.text()
            chk = self.annotation_table.cellWidget(row, 0)
            old = (getattr(self, "preview_annotation_styles", {}) or {}).get(cls, {})
            styles[cls] = {
                "visible": chk.isChecked() if chk else True,
                "color": QColor(old.get("color", self._annotation_default_color_for_class(cls))),
            }
        self.preview_annotation_styles = styles
        return styles

    def update_preview_annotation_style(self, *args):
        if not hasattr(self, "preview_image_label"):
            return
        opacity_percent = int(self.preview_annotation_opacity_slider.value()) if hasattr(self, "preview_annotation_opacity_slider") else 45
        opacity_255 = int(round(opacity_percent / 100.0 * 255))
        if hasattr(self, "preview_annotation_opacity_value_label"):
            self.preview_annotation_opacity_value_label.setText(f"{opacity_percent}%")
        visible = bool(getattr(self, "preview_show_annotations_chk", None) and self.preview_show_annotations_chk.isChecked())
        fill = bool(getattr(self, "preview_annotation_fill_chk", None) and self.preview_annotation_fill_chk.isChecked())
        boundary = int(self.preview_annotation_boundary_spin.value()) if hasattr(self, "preview_annotation_boundary_spin") else 2
        styles = self.get_preview_annotation_class_styles_from_table()
        self.preview_image_label.set_annotations_visible(visible)
        self.preview_image_label.set_annotation_style(
            color=QColor(getattr(self, "preview_annotation_color", QColor(255, 0, 0))),
            opacity=opacity_255,
            fill=fill,
            boundary_width=boundary,
        )
        self.preview_image_label.set_annotation_class_styles(styles)
        self.schedule_preview_tile_popup_update()

    def choose_preview_annotation_class_color(self, class_name: str):
        styles = getattr(self, "preview_annotation_styles", {}) or {}
        current = QColor(styles.get(class_name, {}).get("color", self._annotation_default_color_for_class(class_name)))
        color = QColorDialog.getColor(current, self, f"Choose colour for {class_name}")
        if color.isValid():
            if class_name not in styles:
                styles[class_name] = {"visible": True, "color": color}
            else:
                styles[class_name]["color"] = color
            self.preview_annotation_styles = styles
            # Update just the button visual, then refresh overlays.
            if hasattr(self, "annotation_table"):
                for row in range(self.annotation_table.rowCount()):
                    item = self.annotation_table.item(row, 1)
                    if item and item.text() == class_name:
                        btn = self.annotation_table.cellWidget(row, 2)
                        if btn:
                            btn.setText(color.name())
                            btn.setStyleSheet(
                                f"background-color: {color.name()}; "
                                f"color: {'white' if color.lightness() < 140 else 'black'};"
                            )
                        break
            self.update_preview_annotation_style()

    def choose_preview_annotation_color(self):
        # Backward-compatible global colour picker retained for old calls.
        current = QColor(getattr(self, "preview_annotation_color", QColor(255, 0, 0)))
        color = QColorDialog.getColor(current, self, "Choose annotation overlay colour")
        if color.isValid():
            self.preview_annotation_color = color
            for cls in list((getattr(self, "preview_annotation_styles", {}) or {}).keys()):
                self.preview_annotation_styles[cls]["color"] = QColor(color)
            self.populate_annotation_table()
            self.update_preview_annotation_style()

    def load_preview_geojson(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load GeoJSON annotations for Image Preview",
            "",
            "GeoJSON files (*.geojson *.json);;All Files (*)",
        )
        if not path:
            return
        try:
            annotations = load_geojson_annotations(path)
            self.preview_annotations = annotations
            self.preview_image_label.set_annotations(annotations)
            self.preview_annotation_styles = {}
            self.populate_annotation_table()
            classes = sorted({a.get("class_name", "annotation") for a in annotations})
            class_text = ", ".join(classes[:8]) + ("..." if len(classes) > 8 else "")
            self.info_label.setText(
                f"Loaded {len(annotations)} GeoJSON annotation object(s) for Image Preview from {Path(path).name}"
                + (f" | names/classes: {class_text}" if class_text else "")
            )
            self.schedule_preview_tile_popup_update()
        except Exception as e:
            QMessageBox.critical(self, "GeoJSON error", str(e))

    def clear_preview_geojson(self):
        self.preview_annotations = []
        self.preview_annotation_styles = {}
        if hasattr(self, "annotation_table"):
            self.annotation_table.setRowCount(0)
        if hasattr(self, "preview_image_label"):
            self.preview_image_label.clear_annotations()
            self.preview_image_label.set_annotation_class_styles({})
        self.schedule_preview_tile_popup_update()
        self.info_label.setText("GeoJSON annotations cleared from Image Preview.")

    def _current_preview_tile_roi(self):
        if not getattr(self, "preview_full_dims", None):
            return None
        full_w, full_h = self.preview_full_dims
        size = int(self.preview_tile_size_spin.value()) if hasattr(self, "preview_tile_size_spin") else 1024
        size = max(1, min(size, max(int(full_w), int(full_h))))
        if self.preview_tile_center is None:
            self.preview_tile_center = self.preview_center or (full_w / 2.0, full_h / 2.0)
        cx, cy = self.preview_tile_center
        crop_w = min(size, int(full_w))
        crop_h = min(size, int(full_h))
        x = int(round(float(cx) - crop_w / 2.0))
        y = int(round(float(cy) - crop_h / 2.0))
        x = max(0, min(x, int(full_w) - crop_w))
        y = max(0, min(y, int(full_h) - crop_h))
        cx = x + crop_w / 2.0
        cy = y + crop_h / 2.0
        self.preview_tile_center = (cx, cy)
        return int(x), int(y), int(crop_w), int(crop_h)

    def update_preview_tile_overlay(self, *args):
        if not hasattr(self, "preview_image_label"):
            return
        roi = self._current_preview_tile_roi() if getattr(self, "preview_full_dims", None) else None
        visible = bool(getattr(self, "preview_tile_capture_chk", None) and self.preview_tile_capture_chk.isChecked())
        if roi:
            x, y, w, h = roi
            center = (x + w / 2.0, y + h / 2.0)
            size = int(self.preview_tile_size_spin.value()) if hasattr(self, "preview_tile_size_spin") else w
            self.preview_image_label.set_tile_overlay(center, size_full=size, visible=visible, tile_callback=self._on_preview_tile_center_changed)
            if hasattr(self, "preview_tile_info_label"):
                self.preview_tile_info_label.setText(f"Tile: X={x}, Y={y}, size={w} × {h} px")
        else:
            self.preview_image_label.set_tile_overlay(None, visible=False, tile_callback=self._on_preview_tile_center_changed)
        self.schedule_preview_tile_popup_update()

    def ensure_preview_tile_popup(self):
        if self.preview_tile_popup is None:
            self.preview_tile_popup = TileCapturePopup(self, self)
        return self.preview_tile_popup

    def toggle_preview_tile_mode(self, *args):
        enabled = bool(getattr(self, "preview_tile_capture_chk", None) and self.preview_tile_capture_chk.isChecked())
        self.preview_tile_mode = enabled
        if hasattr(self, "preview_image_label"):
            self.preview_image_label.enable_tile_mode(enabled)
        self.update_preview_tile_overlay()
        if enabled:
            if self.preview_path is None:
                QMessageBox.warning(self, "No image", "Load an image first.")
                self.preview_tile_capture_chk.blockSignals(True)
                self.preview_tile_capture_chk.setChecked(False)
                self.preview_tile_capture_chk.blockSignals(False)
                self.preview_image_label.enable_tile_mode(False)
                return
            popup = self.ensure_preview_tile_popup()
            popup.show()
            popup.raise_()
            self.schedule_preview_tile_popup_update(delay_ms=10)
            self.info_label.setText("Tile capture mode: left-drag the blue square to move the tile. Right-drag the main preview to pan the view. The pop-up updates with the tile section.")
        else:
            if self.preview_tile_popup is not None:
                self.preview_tile_popup.hide()

    def _on_preview_tile_center_changed(self, cx, cy):
        self.preview_tile_center = (float(cx), float(cy))
        self.update_preview_tile_overlay()

    def schedule_preview_tile_popup_update(self, delay_ms: int = 120):
        if not hasattr(self, "_preview_tile_popup_timer"):
            return
        popup = getattr(self, "preview_tile_popup", None)
        if popup is None or not popup.isVisible():
            return
        self._preview_tile_popup_timer.start(max(0, int(delay_ms)))

    def _tile_popup_include_geojson(self) -> bool:
        popup = getattr(self, "preview_tile_popup", None)
        if popup is not None and hasattr(popup, "include_geojson_chk"):
            return popup.include_geojson_chk.isChecked()
        return True

    def _render_current_preview_tile_rgb(self, include_geojson: bool = True, preview_max_side: Optional[int] = None) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        if self.preview_path is None:
            raise RuntimeError("No image loaded.")
        roi = self._current_preview_tile_roi()
        if roi is None:
            raise RuntimeError("Tile ROI is not defined.")
        x, y, w, h = roi
        b = getattr(self, "preview_backend", None)
        close_after = False
        if b is None:
            b = ImageBackend().load(str(self.preview_path))
            close_after = True
        try:
            if preview_max_side is not None:
                raw, axes, _ = read_roi_region_from_backend(b, (x, y, w, h), max_side=int(preview_max_side))
            else:
                raw, axes, _ = b.crop_raw(x, y, w, h)
        finally:
            if close_after:
                b.close()
        rgb = render_channel_composite(raw, axes, self.get_channel_settings_from_table())
        if include_geojson:
            if bool(getattr(self, "preview_show_annotations_chk", None) and self.preview_show_annotations_chk.isChecked()):
                opacity_percent = int(self.preview_annotation_opacity_slider.value()) if hasattr(self, "preview_annotation_opacity_slider") else 45
                opacity_255 = int(round(opacity_percent / 100.0 * 255))
                rgb = draw_geojson_annotations_on_rgb(
                    rgb,
                    getattr(self, "preview_annotations", []),
                    roi_full=(x, y, w, h),
                    color=QColor(getattr(self, "preview_annotation_color", QColor(255, 0, 0))),
                    opacity=opacity_255,
                    fill=bool(getattr(self, "preview_annotation_fill_chk", None) and self.preview_annotation_fill_chk.isChecked()),
                    boundary_width=int(self.preview_annotation_boundary_spin.value()) if hasattr(self, "preview_annotation_boundary_spin") else 2,
                    class_styles=self.get_preview_annotation_class_styles_from_table(),
                )
        return rgb, roi

    def update_preview_tile_popup_now(self):
        popup = getattr(self, "preview_tile_popup", None)
        if popup is None or not popup.isVisible():
            return
        try:
            rgb, roi = self._render_current_preview_tile_rgb(include_geojson=self._tile_popup_include_geojson(), preview_max_side=1200)
            # The popup uses a display-resolution ROI read, not a full-resolution tile read.
            rgb_small = _downsample_for_preview(rgb, max_side=1200)
            x, y, w, h = roi
            popup.set_tile(rgb_small, f"Tile: X={x}, Y={y}, size={w} × {h} px | GeoJSON: {'on' if self._tile_popup_include_geojson() else 'off'}")
        except Exception as e:
            popup.set_tile(None, f"Tile preview error: {e}")

    def save_preview_tile_capture_jpg(self):
        if self.preview_path is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            return
        try:
            rgb, roi = self._render_current_preview_tile_rgb(include_geojson=self._tile_popup_include_geojson())
            x, y, w, h = roi
            suffix = self.preview_suffix_edit.text().strip() or "tile"
            out_path = self.preview_path.parent / f"{self.preview_path.stem}_tile_x{x}_y{y}_s{w}_{suffix}.jpg"
            i = 2
            while out_path.exists():
                out_path = self.preview_path.parent / f"{self.preview_path.stem}_tile_x{x}_y{y}_s{w}_{suffix}_{i}.jpg"
                i += 1
            save_preview_jpg(out_path, rgb)
            self.info_label.setText(f"Saved tile capture JPG: {out_path}")
            QMessageBox.information(self, "Tile saved", f"Saved tile capture JPG:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Tile capture error", str(e))


    def closeEvent(self, event):
        try:
            if getattr(self, "preview_backend", None) is not None:
                self.preview_backend.close()
        except Exception:
            pass
        try:
            if getattr(self, "backend", None) is not None:
                self.backend.close()
        except Exception:
            pass
        try:
            if getattr(self, "if_backend", None) is not None:
                self.if_backend.close()
        except Exception:
            pass
        super().closeEvent(event)

    # ========================================================
    # Batch JPG Preview Export page
    # ========================================================

    def _build_batch_channel_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        file_box = QGroupBox("Batch export displayed-style JPG previews")
        file_layout = QVBoxLayout(file_box)
        row = QHBoxLayout()
        files_btn = QPushButton("Select image files")
        files_btn.clicked.connect(self.load_batch_channel_files)
        folder_btn = QPushButton("Select folder")
        folder_btn.clicked.connect(self.load_batch_channel_folder)
        self.batch_channel_label = QLabel("No files selected. Configure checkboxes/colors in Image Preview first, or use defaults.")
        self.batch_channel_label.setStyleSheet("padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px;")
        row.addWidget(files_btn)
        row.addWidget(folder_btn)
        row.addWidget(self.batch_channel_label, 1)
        file_layout.addLayout(row)
        row2 = QHBoxLayout()
        out_btn = QPushButton("Output folder optional")
        out_btn.clicked.connect(self.browse_batch_channel_output_folder)
        self.batch_channel_output_edit = QLineEdit("")
        self.batch_channel_output_edit.setPlaceholderText("Leave empty to create a channel_previews folder beside first input")
        row2.addWidget(out_btn)
        row2.addWidget(self.batch_channel_output_edit, 1)
        file_layout.addLayout(row2)
        layout.addWidget(file_box)
        opts = QGroupBox("Batch preview options")
        opt = QHBoxLayout(opts)
        self.batch_channel_suffix_edit = QLineEdit("channel_preview")
        self.batch_channel_max_side_spin = QSpinBox()
        self.batch_channel_max_side_spin.setRange(128, 10000)
        self.batch_channel_max_side_spin.setValue(1600)
        self.batch_channel_recursive_chk = QCheckBox("Recursive folder search")
        opt.addWidget(QLabel("Suffix:"))
        opt.addWidget(self.batch_channel_suffix_edit)
        opt.addWidget(QLabel("Max output side:"))
        opt.addWidget(self.batch_channel_max_side_spin)
        opt.addWidget(self.batch_channel_recursive_chk)
        opt.addStretch()
        layout.addWidget(opts)
        note = QLabel("This applies the Image Preview channel checkboxes/colors to many files and saves JPG previews only. It does not modify the original OME-TIFF/IF data.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #555; padding: 4px;")
        layout.addWidget(note)
        run_btn = QPushButton("RUN BATCH JPG PREVIEW EXPORT")
        run_btn.clicked.connect(self.run_batch_channel_convert)
        layout.addWidget(run_btn)
        layout.addStretch()
        return page

    def load_batch_channel_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select images", "", _image_file_filter())
        if paths:
            self.batch_channel_paths = [Path(p) for p in paths]
            self.batch_channel_label.setText(f"Selected {len(self.batch_channel_paths)} image file(s)")

    def load_batch_channel_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with images")
        if not folder:
            return
        folder = Path(folder)
        it = folder.rglob("*") if self.batch_channel_recursive_chk.isChecked() else folder.iterdir()
        self.batch_channel_paths = sorted([p for p in it if p.is_file() and _has_ext(p.name, SUPPORTED_EXTENSIONS)])
        self.batch_channel_label.setText(f"Folder loaded: {len(self.batch_channel_paths)} image file(s)")

    def browse_batch_channel_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder for JPG previews")
        if folder:
            self.batch_channel_output_edit.setText(folder)

    def run_batch_channel_convert(self):
        if not self.batch_channel_paths:
            QMessageBox.warning(self, "No files", "Select files or a folder first.")
            return
        settings = self.get_channel_settings_from_table() if hasattr(self, "channel_table") and self.channel_table.rowCount() else []
        suffix = self.batch_channel_suffix_edit.text().strip() or "channel_preview"
        max_side = int(self.batch_channel_max_side_spin.value())
        out_dir = Path(self.batch_channel_output_edit.text().strip()) if self.batch_channel_output_edit.text().strip() else self.batch_channel_paths[0].parent / "channel_previews"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(self.batch_channel_paths))
        rows, ok, failed = [], 0, 0
        for i, p in enumerate(self.batch_channel_paths, start=1):
            try:
                self.info_label.setText(f"Rendering channel preview {p.name} ({i}/{len(self.batch_channel_paths)})")
                QApplication.processEvents()
                arr, axes, meta = read_preview_array_from_file(str(p), max_side=max_side)
                n = _count_display_channels(arr, axes)
                st = settings if settings and max(s.get("channel", 0) for s in settings) < n else []
                rgb = render_channel_composite(arr, axes, st)
                out_path = out_dir / f"{p.stem}_{suffix}.jpg"
                save_preview_jpg(out_path, rgb)
                rows.append({"input": str(p), "output": str(out_path), "status": "success", "message": ""})
                ok += 1
            except Exception as e:
                failed += 1
                rows.append({"input": str(p), "output": "", "status": "failed", "message": f"{e}\n{traceback.format_exc()}"})
            self.progress.setValue(i)
            QApplication.processEvents()
        log_path = out_dir / f"TiffCropper_channel_preview_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["input", "output", "status", "message"])
            writer.writeheader()
            writer.writerows(rows)
        self.progress.setVisible(False)
        QMessageBox.information(self, "Batch channel convert", f"Done. Success: {ok}. Failed: {failed}.\n\nLog:\n{log_path}")
        self.info_label.setText(f"Batch channel convert complete. Log: {log_path}")

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

    def _lif_options(self):
        return {
            "output_base": self._lif_output_base(),
            "save_xml": self.lif_xml_chk.isChecked(),
            "save_json": self.lif_json_chk.isChecked(),
            "overwrite": self.lif_overwrite_chk.isChecked(),
            "skip_existing": self.lif_skip_existing_chk.isChecked(),
            "compression": self._lif_compression(),
            "stop_on_error": self.lif_stop_on_error_chk.isChecked(),
        }

    def save_lif_selected_scenes(self):
        if self.lif_single_path is None or self.lif_single_obj is None:
            QMessageBox.warning(self, "No LIF loaded", "Load one LIF file first.")
            return
        selected = self.selected_lif_scene_indices()
        if not selected:
            QMessageBox.warning(self, "No scenes selected", "Select at least one scene/page to save.")
            return
        scene_map = {str(self.lif_single_path): selected}
        self._start_background_job(
            "LIF selected scene export",
            _lif_export_job,
            self._on_lif_export_done,
            [str(self.lif_single_path)],
            scene_map,
            self._lif_options(),
            1,
        )

    def save_lif_all_single_scenes(self):
        if self.lif_single_path is None or self.lif_single_obj is None:
            QMessageBox.warning(self, "No LIF loaded", "Load one LIF file first.")
            return
        self._start_background_job(
            "LIF export",
            _lif_export_job,
            self._on_lif_export_done,
            [str(self.lif_single_path)],
            {},
            self._lif_options(),
            1,
        )

    def run_lif_bulk_export(self):
        if not self.lif_bulk_paths:
            QMessageBox.warning(self, "No bulk LIF files", "Select LIF files or a folder first.")
            return
        # LIF files are usually very large. More than 2 workers can saturate disk/RAM.
        workers = min(self._worker_count(), 2)
        self._start_background_job(
            "Bulk LIF export",
            _lif_export_job,
            self._on_lif_export_done,
            [str(p) for p in self.lif_bulk_paths],
            {},
            self._lif_options(),
            workers,
        )

    def _on_lif_export_done(self, result):
        manifests = result.get("manifests", []) if isinstance(result, dict) else []
        failures = result.get("failures", []) if isinstance(result, dict) else []
        msg = f"LIF export finished. Manifest files: {len(manifests)}"
        if manifests:
            msg += "\n" + "\n".join(str(p) for p in manifests[:8])
            if len(manifests) > 8:
                msg += f"\n... and {len(manifests) - 8} more"
        if failures:
            msg += f"\n\nFailures: {len(failures)}"
            for f in failures[:5]:
                msg += f"\n{Path(f.get('file', '')).name}: {str(f.get('error', ''))[:300]}"
        QMessageBox.information(self, "LIF export", msg)
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
        self.crop_preview_chk.stateChanged.connect(self.refresh_crop_input_preview)
        opt_layout.addWidget(self.crop_lossless_chk)
        opt_layout.addWidget(self.crop_preserve_channels_chk)
        opt_layout.addWidget(self.crop_preview_chk)
        opt_layout.addStretch()
        layout.addWidget(opt)

        display_box = QGroupBox("Crop display tools - zoom, brightness and negative")
        display_layout = QGridLayout(display_box)
        zoom_out_btn = QPushButton("Zoom -")
        zoom_out_btn.clicked.connect(lambda: self.change_crop_zoom(0.8))
        zoom_in_btn = QPushButton("Zoom +")
        zoom_in_btn.clicked.connect(lambda: self.change_crop_zoom(1.25))
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(lambda: self.set_crop_zoom(1.0))
        self.crop_zoom_label = QLabel("Zoom: 100%")
        display_layout.addWidget(zoom_out_btn, 0, 0)
        display_layout.addWidget(zoom_in_btn, 0, 1)
        display_layout.addWidget(fit_btn, 0, 2)
        display_layout.addWidget(self.crop_zoom_label, 0, 3)

        self.crop_negative_chk = QCheckBox("Negative")
        self.crop_negative_chk.stateChanged.connect(self.update_crop_display_adjustments)
        display_layout.addWidget(self.crop_negative_chk, 0, 4)

        display_layout.addWidget(QLabel("Brightness:"), 1, 0)
        self.crop_brightness_slider = QSlider(Qt.Horizontal)
        self.crop_brightness_slider.setRange(-100, 100)
        self.crop_brightness_slider.setValue(0)
        self.crop_brightness_slider.valueChanged.connect(self.update_crop_display_adjustments)
        self.crop_brightness_value_label = QLabel("0")
        display_layout.addWidget(self.crop_brightness_slider, 1, 1, 1, 3)
        display_layout.addWidget(self.crop_brightness_value_label, 1, 4)
        display_layout.setColumnStretch(3, 1)
        layout.addWidget(display_box)

        prev = QGroupBox("Preview - left-drag a rectangle; wheel zooms; right/middle-drag pans")
        prev_layout = QHBoxLayout(prev)
        self.crop_thumb_in = CropSelectionLabel()
        self.crop_thumb_in.setMinimumSize(300, 170)
        self.crop_thumb_in.setMaximumHeight(260)
        self.crop_thumb_in.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.crop_thumb_out = QLabel("Crop preview")
        self.crop_thumb_out.setAlignment(Qt.AlignCenter)
        self.crop_thumb_out.setMinimumSize(300, 170)
        self.crop_thumb_out.setMaximumHeight(260)
        self.crop_thumb_out.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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


    def refresh_crop_input_preview(self, *args):
        """Render the crop input preview using the current zoom, brightness and GeoJSON state."""
        if not hasattr(self, "crop_thumb_in"):
            return
        if not self.backend.path or not self.backend.slide_dims:
            return
        if hasattr(self, "crop_preview_chk") and not self.crop_preview_chk.isChecked():
            return
        full_w, full_h = self.backend.slide_dims
        if self.crop_center is None:
            self.crop_center = (full_w / 2.0, full_h / 2.0)
        try:
            viewport = (
                max(480, int(self.crop_thumb_in.width())),
                max(300, int(self.crop_thumb_in.height())),
            )
            # Use the already-open backend so crop preview can reuse OpenSlide/TIFF
            # handles and access pyramid/overview levels. The previous path opened
            # the file again with read_zoom_region_from_file(), which could fall
            # back to the "Zoom preview skipped" placeholder for some pyramidal TIFFs.
            arr, axes, meta = read_zoom_region_from_backend(
                self.backend,
                center_xy=self.crop_center,
                zoom=max(1.0, float(self.crop_zoom)),
                viewport_size=viewport,
                max_side=max(1200, int(max(viewport) * 2)),
            )
            rgb = _array_to_rgb_preview(arr, axes)
            roi = meta.get("roi") or (0, 0, full_w, full_h)
            self.crop_preview_meta = meta
            if meta.get("full_dims"):
                full_w, full_h = tuple(meta["full_dims"])
            self.crop_thumb_in.set_image(
                rgb,
                full_w=full_w,
                full_h=full_h,
                callback=self._on_rectangle_selected,
                roi_full=roi,
                center_callback=self._on_crop_view_center_changed,
                zoom_callback=self._on_crop_view_zoom,
                pan_callback=self._on_crop_view_pan,
            )
        except Exception as e:
            # Safe fallback: use the existing full-image thumbnail behavior.
            thumb = self.backend.input_thumbnail(max_side=900)
            self.crop_preview_meta = {"reader": "input_thumbnail-fallback", "roi": (0, 0, full_w, full_h), "error": str(e)}
            self.crop_thumb_in.set_image(
                thumb,
                full_w=full_w,
                full_h=full_h,
                callback=self._on_rectangle_selected,
                roi_full=(0, 0, full_w, full_h),
                center_callback=self._on_crop_view_center_changed,
                zoom_callback=self._on_crop_view_zoom,
                pan_callback=self._on_crop_view_pan,
            )
        self.update_crop_display_adjustments()
        self.crop_thumb_in.set_selection_from_full_coords(
            self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value()
        )
        if hasattr(self, "crop_zoom_label"):
            roi = self.crop_preview_meta.get("roi") if isinstance(self.crop_preview_meta, dict) else None
            self.crop_zoom_label.setText(f"Zoom: {int(self.crop_zoom * 100)}%")
            if roi:
                self.info_label.setText(f"Crop preview updated | zoom={int(self.crop_zoom * 100)}% | ROI={roi}")

    def update_crop_display_adjustments(self, *args):
        if not hasattr(self, "crop_thumb_in"):
            return
        negative = bool(getattr(self, "crop_negative_chk", None) and self.crop_negative_chk.isChecked())
        brightness = int(self.crop_brightness_slider.value()) if hasattr(self, "crop_brightness_slider") else 0
        if hasattr(self, "crop_brightness_value_label"):
            self.crop_brightness_value_label.setText(str(brightness))
        self.crop_thumb_in.set_negative(negative)
        self.crop_thumb_in.set_brightness(brightness)

    def _on_crop_view_zoom(self, factor: float, center_xy=None):
        self.change_crop_zoom(factor, center_xy=center_xy)

    def _on_crop_view_center_changed(self, cx, cy):
        self.crop_center = (float(cx), float(cy))
        self.refresh_crop_input_preview()

    def _on_crop_view_pan(self, start_full_xy, dest_fraction):
        """Natural grab-and-move pan: the point clicked at mouse-down follows the cursor."""
        if not self.backend.path or not self.backend.slide_dims:
            return
        roi = self.crop_preview_meta.get("roi") if isinstance(self.crop_preview_meta, dict) else None
        if not roi:
            return
        rx, ry, rw, rh = roi
        fx, fy = dest_fraction
        start_x, start_y = start_full_xy
        new_cx = float(start_x) + (0.5 - float(fx)) * float(rw)
        new_cy = float(start_y) + (0.5 - float(fy)) * float(rh)
        full_w, full_h = self.backend.slide_dims
        self.crop_center = (max(0.0, min(float(full_w), new_cx)), max(0.0, min(float(full_h), new_cy)))
        self.refresh_crop_input_preview()

    def change_crop_zoom(self, factor: float, center_xy=None):
        if not self.backend.path or not self.backend.slide_dims:
            return
        if center_xy is not None:
            self.crop_center = (float(center_xy[0]), float(center_xy[1]))
        self.crop_zoom = max(1.0, min(64.0, float(self.crop_zoom) * float(factor)))
        if self.crop_zoom == 1.0 and self.backend.slide_dims:
            full_w, full_h = self.backend.slide_dims
            self.crop_center = (full_w / 2.0, full_h / 2.0)
        self.refresh_crop_input_preview()

    def set_crop_zoom(self, value: float):
        if not self.backend.path or not self.backend.slide_dims:
            return
        self.crop_zoom = max(1.0, min(64.0, float(value)))
        if self.crop_zoom == 1.0:
            full_w, full_h = self.backend.slide_dims
            self.crop_center = (full_w / 2.0, full_h / 2.0)
        self.refresh_crop_input_preview()

    def save_crop_display_capture(self):
        if not hasattr(self, "crop_thumb_in") or not self.crop_thumb_in.has_image():
            QMessageBox.warning(self, "No preview", "Load an image first.")
            return
        if not self.backend.path_obj:
            QMessageBox.warning(self, "No image", "Load an image first.")
            return
        suffix = self.crop_suffix_edit.text().strip() or "display"
        out_path = self.backend.path_obj.parent / f"{self.backend.path_obj.stem}_crop_display_{suffix}.jpg"
        if out_path.exists():
            i = 2
            while True:
                candidate = self.backend.path_obj.parent / f"{self.backend.path_obj.stem}_crop_display_{suffix}_{i}.jpg"
                if not candidate.exists():
                    out_path = candidate
                    break
                i += 1
        pixmap = self.crop_thumb_in.grab()
        ok = pixmap.save(str(out_path), "JPG", 95)
        if ok:
            self.info_label.setText(f"Saved displayed crop view: {out_path}")
            QMessageBox.information(self, "Display saved", f"Saved displayed crop view:\n{out_path}")
        else:
            QMessageBox.critical(self, "Save error", f"Could not save display capture:\n{out_path}")

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
            self.crop_zoom = 1.0
            self.crop_center = (w / 2.0, h / 2.0)
            self.crop_preview_meta = {}
            print("Loaded with:", self.backend.reader, self.backend.file_kind, self.backend.path_obj.suffix)
            self.info_label.setText(f"Loaded: {Path(file_path).name} | Size: {w} x {h} px | Reader: {self.backend.reader} | MPP: {_format_mpp_text(self.backend.source_mpp)}")
            if self.crop_preview_chk.isChecked():
                self.refresh_crop_input_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error", _exception_text(e))

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
            x, y, w, h = self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value()
            roi_full = (int(x), int(y), int(w), int(h))
            # Preview must never read the full selected crop into RAM.  For very
            # large crops, read a pyramid/overview or strided region only.
            arr, axes, meta = read_roi_region_from_backend(
                self.backend, roi_full, max_side=1200
            )
            roi_preview = _array_to_rgb_preview(arr, axes)
            if self.crop_preview_chk.isChecked():
                if self.backend.slide_dims:
                    self.refresh_crop_input_preview()
                self._set_label_pixmap(self.crop_thumb_out, _downsample_for_preview(roi_preview, 512))
            source_shape = meta.get("shape", "") if isinstance(meta, dict) else ""
            self.info_label.setText(
                f"Preview ready from reduced region | ROI={roi_full} | reader={meta.get('reader', '') if isinstance(meta, dict) else ''} | source_shape={source_shape}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", _exception_text(e))

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
            x, y, w, h = self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value()
            visual_lowmem = False
            visual_est_rgb = None

            preserve_raw = (
                hasattr(self, "crop_preserve_channels_chk")
                and self.crop_preserve_channels_chk.isChecked()
                and self.backend.reader == "tifffile"
                and output_format != "jpeg"
            )

            # RGB/RGBA pyramidal TIFFs often have a Samples axis (YXS or TZYXS).
            # They are visual images, not raw IF channel stacks. Treat these as
            # visual RGB crops even if the Preserve raw checkbox is enabled; this
            # prevents tifffile OME writing errors such as "shape does not match
            # stored shape" and allows OpenSlide/pyramid fallbacks for cropping.
            if preserve_raw:
                _est_bytes0, _est_shape0, _est_axes0, _est_dtype0 = _estimate_tiff_raw_crop_bytes(
                    self.backend, x, y, w, h, downsample=raw_downsample
                )
                if _est_shape0 is not None and _est_axes0 and _is_sample_axis_array(_est_shape0, _est_axes0):
                    preserve_raw = False
                    self.info_label.setText(
                        f"Source uses RGB/RGBA Samples axis ({_est_axes0}); saving as visual RGB crop instead of raw IF planes."
                    )
                    QApplication.processEvents()

            # Preflight memory guard.  This prevents NumPy from attempting to
            # allocate tens of GiB for very large crops such as whole-slide RGB
            # regions.  The preview path remains available because it uses a
            # reduced pyramid/overview region.
            if preserve_raw:
                est_bytes, est_shape, est_axes, est_dtype = _estimate_tiff_raw_crop_bytes(
                    self.backend, x, y, w, h, downsample=raw_downsample
                )
                # If the source has a Samples axis (YXS/TZYXS), it is an RGB/RGBA
                # image rather than scientific IF channel data. Saving this as
                # minisblack OME causes tifffile's "shape does not match stored shape".
                # Keep the crop in the normal in-memory path if small enough;
                # save_multichannel_image will write it with photometric=rgb.
                if est_shape is not None and est_axes and _is_sample_axis_array(est_shape, est_axes):
                    if est_bytes is not None and est_bytes > MAX_INTERACTIVE_CROP_BYTES:
                        QMessageBox.warning(
                            self,
                            "RGB/Sample-axis crop too large",
                            "This source is stored with a Samples axis (for example YXS/TZYXS). "
                            "It cannot be streamed safely as raw IF planes.\n\n"
                            f"Estimated output: {_human_bytes(est_bytes)}\n"
                            f"Shape: {est_shape}\nAxes: {est_axes}\n\n"
                            "Use a downsampled visual crop, reduce the ROI, or use Tiles mode."
                        )
                        return
                if est_bytes is not None and est_bytes > MAX_INTERACTIVE_CROP_BYTES:
                    if raw_downsample == 1.0 and est_axes and "S" not in est_axes:
                        self.info_label.setText(
                            f"Large raw crop detected ({_human_bytes(est_bytes)}). Saving with low-memory streaming..."
                        )
                        QApplication.processEvents()
                        result = save_tiff_raw_crop_lowmem(
                            self.backend, out_path, x, y, w, h,
                            write_ome=write_ome, lossless=self.crop_lossless_chk.isChecked(),
                            image_name=out_path.stem,
                        )
                        # Show only a reduced preview after saving.
                        try:
                            arr, axes, meta = read_roi_region_from_backend(self.backend, (x, y, w, h), max_side=1000)
                            preview_rgb = _array_to_rgb_preview(arr, axes)
                            if self.crop_preview_chk.isChecked():
                                self._set_label_pixmap(self.crop_thumb_out, _downsample_for_preview(preview_rgb, 512))
                        except Exception:
                            pass
                        cal_text = f" | MPP inherited: {_format_mpp_text(self.backend.source_mpp)}" if self.backend.source_mpp else " | WARNING: source MPP unknown; output viewer may default to 1 µm/px"
                        saved_text = (
                            f"Saved large raw crop using low-memory streaming: {out_path} | "
                            f"shape={result.get('shape')} | axes={result.get('axes')} | dtype={result.get('dtype')}{cal_text}"
                        )
                        self.info_label.setText(saved_text)
                        QMessageBox.information(self, "Success", saved_text)
                        return
                    QMessageBox.warning(
                        self,
                        "Crop too large for memory",
                        "This raw crop is too large for the current in-memory/downsample path.\n\n"
                        f"Estimated output: {_human_bytes(est_bytes)}\n"
                        f"Shape: {est_shape}\nAxes: {est_axes}\n\n"
                        "Use downsampled tiles, reduce the ROI, or save an undownsampled non-RGB raw crop when possible."
                    )
                    return
            else:
                visual_est_rgb = _estimate_rgb_crop_bytes(w, h, downsample=raw_downsample, channels=3, dtype=np.uint8)
                if visual_est_rgb > MAX_INTERACTIVE_CROP_BYTES:
                    if output_format == "jpeg":
                        QMessageBox.warning(
                            self,
                            "JPEG crop too large",
                            "This crop is too large to build as a single JPEG array.\n\n"
                            f"Estimated RGB output: {_human_bytes(visual_est_rgb)}\n"
                            f"ROI: X={x}, Y={y}, W={w}, H={h}\n\n"
                            "Please save as TIFF/OME-TIFF for low-memory chunked export, "
                            "increase Downsample, or use Tiles mode."
                        )
                        return
                    visual_lowmem = True
                    self.info_label.setText(
                        f"Large RGB crop detected ({_human_bytes(visual_est_rgb)}). Saving with low-memory tiled export..."
                    )
                    QApplication.processEvents()

            if preserve_raw:
                roi, axes, _ = self.backend.crop_raw(x, y, w, h)
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
                cal_text = f" | MPP inherited: {_format_mpp_text(self.backend.source_mpp)}" if self.backend.source_mpp else " | WARNING: source MPP unknown; output viewer may default to 1 µm/px"
                saved_text = f"Saved raw multichannel crop: {out_path} | shape={tuple(roi.shape)} | axes={axes or 'unknown'} | dtype={roi.dtype}{cal_text}"
            else:
                if visual_lowmem:
                    result = save_rgb_crop_lowmem(
                        self.backend, out_path, x, y, w, h,
                        downsample=raw_downsample, output_format=output_format,
                        write_ome=write_ome, lossless=self.crop_lossless_chk.isChecked(),
                        source_resolution=self.backend.source_resolution,
                        source_mpp=self.backend.source_mpp,
                        image_name=out_path.stem,
                        annotation_kv=self.backend.openslide_props if write_ome else None,
                        progress_callback=lambda done, total: self.info_label.setText(
                            f"Saving large RGB crop by chunks: {done}/{total} blocks"
                        ),
                    )
                    cal_text = f" | MPP inherited: {_format_mpp_text(self.backend.source_mpp)}" if self.backend.source_mpp else " | WARNING: source MPP unknown; output viewer may default to 1 µm/px"
                    saved_text = (
                        f"Saved large RGB crop using low-memory tiled export: {out_path} | "
                        f"shape={result.get('shape')} | downsample={result.get('downsample')}{cal_text}"
                    )
                else:
                    roi, crop_meta = _read_visual_crop_for_save(self.backend, x, y, w, h, downsample=raw_downsample)
                    if _is_safe_placeholder_meta(crop_meta):
                        raise RuntimeError(
                            "The crop reader returned a display placeholder instead of real pixels. "
                            "The crop was not saved to avoid writing the preview-warning text into the output image."
                        )
                    save_rgb_image(
                        out_path, roi, output_format, write_ome, self.crop_lossless_chk.isChecked(),
                        self.backend.source_resolution, self.backend.source_mpp, out_path.stem,
                        self.backend.openslide_props if write_ome else None,
                        pixel_scale=raw_downsample
                    )
                    cal_text = f" | MPP inherited: {_format_mpp_text(self.backend.source_mpp)}" if self.backend.source_mpp else " | WARNING: source MPP unknown; output viewer may default to 1 µm/px"
                    reader_text = crop_meta.get("reader", self.backend.reader) if isinstance(crop_meta, dict) else self.backend.reader
                    saved_text = f"Saved RGB crop: {out_path} | Reader: {reader_text} | output_shape={tuple(roi.shape)}{cal_text}"

            # Display a reduced preview instead of converting the full saved crop.
            if self.crop_preview_chk.isChecked():
                try:
                    arr, axes_p, meta = read_roi_region_from_backend(self.backend, (x, y, w, h), max_side=1000)
                    if not _is_safe_placeholder_meta(meta):
                        preview_rgb = _array_to_rgb_preview(arr, axes_p)
                        self._set_label_pixmap(self.crop_thumb_out, _downsample_for_preview(preview_rgb, 512))
                except Exception:
                    pass
            self.info_label.setText(saved_text)
            QMessageBox.information(self, "Success", saved_text)
        except Exception as e:
            QMessageBox.critical(self, "Error", _exception_text(e))

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
            "Fixed square tiles. Edge-aligned avoids tiny final sliver tiles. Raw IF tiles preserve original dtype/channels."
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

        self.tile_edge_label = QLabel("Edge handling:")
        self.tile_edge_combo = QComboBox()
        self.tile_edge_combo.addItems(["Edge-aligned full tiles", "Partial edge tiles"])
        self.tile_edge_combo.setToolTip(
            "Edge-aligned avoids tiny last sliver tiles by shifting the final tile to the image boundary. "
            "This is recommended for IF/OME-TIFF raw tiles. Partial edge tiles keeps the older behavior."
        )

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
        self.tile_out_combo.addItems(["OME-TIFF (.ome.tif)", "TIFF (.tif)", "JPEG (.jpg)"])

        self.tile_lossless_chk = QCheckBox("Lossless TIFF compression")
        self.tile_lossless_chk.setChecked(True)
        self.tile_preserve_raw_chk = QCheckBox("Preserve raw multichannel data for TIFF/OME-TIFF tiles")
        self.tile_preserve_raw_chk.setChecked(True)
        self.tile_preserve_raw_chk.setToolTip("For IF/OME-TIFF inputs, save tile pixels with original channels and dtype. No RGB conversion or intensity normalization.")

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
        grid.addWidget(self.tile_edge_label, 3, 2)
        grid.addWidget(self.tile_edge_combo, 3, 3)

        grid.addWidget(self.tile_rows_label, 2, 0)
        grid.addWidget(self.tile_rows_spin, 2, 1)
        grid.addWidget(self.tile_cols_label, 2, 2)
        grid.addWidget(self.tile_cols_spin, 2, 3)

        grid.addWidget(self.tile_downsample_label, 4, 0)
        grid.addWidget(self.tile_downsample_spin, 4, 1)
        grid.addWidget(self.tile_format_label, 4, 2)
        grid.addWidget(self.tile_out_combo, 4, 3)
        grid.addWidget(self.tile_lossless_chk, 5, 0, 1, 4)
        grid.addWidget(self.tile_preserve_raw_chk, 6, 0, 1, 4)

        layout.addWidget(params)

        self.tile_fixed_widgets = [
            self.tile_fixed_info_label,
            self.tile_size_label,
            self.tile_size_spin,
            self.tile_overlap_label,
            self.tile_overlap_spin,
            self.tile_padding_label,
            self.tile_padding_combo,
            self.tile_edge_label,
            self.tile_edge_combo,
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
                    "Tiles mode: fixed square tiles. Edge-aligned mode avoids tiny last sliver tiles; raw IF tiles are not normalized."
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
        arr_prev, axes_prev, meta_prev = read_preview_array_from_file(str(path), max_side=768)
        self._set_label_pixmap(self.tiles_thumb, _array_to_rgb_preview(arr_prev, axes_prev))
        self.info_label.setText(f"Loaded for tiling: {path.name} | Reader: {self.backend.reader} | preview={meta_prev.get('reader', '')}")

    def _tile_params(self):
        out_text = self.tile_out_combo.currentText()
        preserve_raw = (
            hasattr(self, "tile_preserve_raw_chk")
            and self.tile_preserve_raw_chk.isChecked()
            and _write_format_from_combo(out_text) != "jpeg"
        )
        edge_text = self.tile_edge_combo.currentText() if hasattr(self, "tile_edge_combo") else "Edge-aligned full tiles"
        edge_mode = "partial" if "Partial" in edge_text else "edge_aligned"
        return {
            "mode": self.tile_mode_combo.currentText(),
            "edge_mode": edge_mode,
            "tile_size": int(self.tile_size_spin.value()),
            "overlap": float(self.tile_overlap_spin.value()),
            "rows": int(self.tile_rows_spin.value()),
            "cols": int(self.tile_cols_spin.value()),
            "downsample": float(self.tile_downsample_spin.value()),
            "padding": self.tile_padding_combo.currentText(),
            "output_format": _write_format_from_combo(out_text),
            "write_ome": str(out_text).startswith("OME-TIFF"),
            "preserve_raw": preserve_raw,
            "ext": _extension_from_combo(out_text),
        }

    def preview_tiles_grid(self):
        if not self.backend.path or not self.backend.slide_dims:
            QMessageBox.warning(self, "Error", "Load one or more images first.")
            return
        try:
            params = self._tile_params()
            full_w, full_h = self.backend.slide_dims
            arr_prev, axes_prev, meta_prev = read_preview_array_from_file(str(self.backend.path), max_side=768)
            thumb = _array_to_rgb_preview(arr_prev, axes_prev)

            if params["mode"].startswith("Fixed"):
                tile_size = params["tile_size"]
                overlap = params["overlap"]
                pm = self._draw_fixed_grid_on_thumb(thumb, full_w, full_h, tile_size, overlap, edge_mode=params.get("edge_mode", "edge_aligned"))
                xs, ys, stride, ovpx = _compute_tile_grid(full_w, full_h, tile_size, overlap, edge_mode=params.get("edge_mode", "edge_aligned"))
                self.info_label.setText(
                    f"Fixed grid: {len(xs)} columns x {len(ys)} rows = {len(xs) * len(ys)} tiles | "
                    f"Overlap {overlap:g}% ({ovpx}px) | Stride {stride}px | Edge: {params.get('edge_mode', 'edge_aligned')} | Reader: {self.backend.reader}"
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
            QMessageBox.critical(self, "Preview Error", _exception_text(e))

    def save_tiles_bulk(self):
        if not self.bulk_paths:
            QMessageBox.warning(self, "Error", "Load one or more images first.")
            return
        params = self._tile_params()
        lossless = self.tile_lossless_chk.isChecked()
        if params.get("preserve_raw"):
            self.info_label.setText("Raw multichannel tiling enabled: tiles preserve dtype/channels and are not normalized.")
        self._start_background_job(
            "Bulk tiling",
            _tile_bulk_job,
            self._on_tiles_done,
            [str(p) for p in self.bulk_paths],
            params,
            lossless,
            self._worker_count(),
        )

    def _on_tiles_done(self, result):
        QMessageBox.information(
            self,
            "Tiles complete",
            f"Done. Success: {result.get('ok', 0)}. Failed: {result.get('failed', 0)}.\n\nLog:\n{result.get('log_path', '')}",
        )
        self.info_label.setText(f"Tiles complete. Log: {result.get('log_path', '')}")

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

        edge_mode = params.get("edge_mode", "edge_aligned")
        xs, ys, _, _ = _compute_tile_grid(full_w, full_h, tile_size, overlap, edge_mode=edge_mode)
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
        self.merge_out_combo.addItems(["OME-TIFF (.ome.tif)", "TIFF (.tif)", "JPEG (.jpg)"])
        self.merge_lossless_chk = QCheckBox("Lossless TIFF compression")
        self.merge_lossless_chk.setChecked(True)
        self.merge_preserve_raw_chk = QCheckBox("Preserve raw multichannel data for TIFF/OME-TIFF merge")
        self.merge_preserve_raw_chk.setChecked(True)
        self.merge_preserve_raw_chk.setToolTip("For IF tiles generated as raw OME-TIFF/TIFF, merge without RGB conversion or intensity normalization.")
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
        grid.addWidget(self.merge_preserve_raw_chk, 4, 0, 1, 3)
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

    def _read_tile_file(self, path, display_limits=None):
        """Read a tile for visual merge preview/output.

        This path is display-oriented and converts to RGB/uint8. For IF data,
        display_limits should be computed once from the full tile set and reused
        for every tile. This avoids the visible seams caused by normalizing each
        tile independently. Use the raw merge option for intensity-preserving
        multichannel merges.
        """
        path = Path(path)
        if _has_ext(path.name, (".jpg", ".jpeg", ".png")):
            from PIL import Image
            return _to_uint8_rgb(np.asarray(Image.open(path).convert("RGB")))

        if display_limits is not None:
            arr, axes = self._read_tile_raw_file(path)
            return self._render_tile_with_display_limits(arr, axes, display_limits)

        # Fallback used only for small/manual previews. It may normalize one tile
        # independently, so it should not be used for final stitched IF previews.
        arr, axes = self._read_tile_raw_file(path)
        return _array_to_rgb_preview(arr, axes)

    def _read_tile_raw_file(self, path):
        """Read a tile preserving dtype/channels/axes when possible."""
        path = Path(path)
        if _has_ext(path.name, (".jpg", ".jpeg", ".png")):
            from PIL import Image
            arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            return arr, "YXS"
        with tifffile.TiffFile(str(path)) as tif:
            s = tif.series[0]
            axes = getattr(s, "axes", "") or ""
            arr = s.asarray()
        axes = _guess_axes_for_array(arr, axes)
        return np.asarray(arr), axes

    def _sample_values_for_limits(self, arr2, max_samples_per_tile=12000):
        """Return a small deterministic sample from a YX or YXC tile preview array."""
        arr2 = np.asarray(arr2)
        if arr2.ndim < 2 or arr2.size == 0:
            return arr2
        h, w = int(arr2.shape[0]), int(arr2.shape[1])
        pixels = max(1, h * w)
        step = max(1, int(math.ceil(math.sqrt(pixels / float(max_samples_per_tile)))))
        if arr2.ndim == 2:
            return arr2[::step, ::step]
        return arr2[::step, ::step, :]

    def _compute_tile_display_limits(self, metas, p_low=1.0, p_high=99.8):
        """Compute global display limits for a tile set.

        The previous preview path normalized each tile separately. That makes
        tile borders visible even when raw pixels are correct. This function
        samples all selected tiles and computes one low/high range per channel.
        """
        channel_samples = []
        for m in metas:
            try:
                arr, axes = self._read_tile_raw_file(m["path"])
                arr2, ax2 = _representative_yx_or_yxc(arr, axes)
                arr2 = self._sample_values_for_limits(arr2)
                if arr2.ndim == 2:
                    channels = [arr2]
                elif arr2.ndim == 3 and ax2.endswith(("C", "S")):
                    channels = [arr2[:, :, i] for i in range(arr2.shape[-1])]
                else:
                    channels = [arr2]
                while len(channel_samples) < len(channels):
                    channel_samples.append([])
                for i, ch in enumerate(channels):
                    vals = np.asarray(ch).reshape(-1)
                    if vals.size:
                        channel_samples[i].append(vals)
            except Exception:
                continue

        limits = []
        for parts in channel_samples:
            if not parts:
                limits.append((0.0, 1.0))
                continue
            vals = np.concatenate(parts).astype(np.float32, copy=False)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                limits.append((0.0, 1.0))
                continue
            lo = float(np.percentile(vals, p_low))
            hi = float(np.percentile(vals, p_high))
            if hi <= lo:
                lo = float(vals.min())
                hi = float(vals.max())
            if hi <= lo:
                # Keep a non-zero range so the renderer remains stable.
                hi = lo + 1.0
            limits.append((lo, hi))
        return limits or None

    def _normalize_with_limit(self, plane, lo, hi):
        plane = np.asarray(plane, dtype=np.float32)
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((plane - float(lo)) / (float(hi) - float(lo)), 0, 1)

    def _render_tile_with_display_limits(self, arr, axes, display_limits):
        """Render one raw tile using global limits shared by the entire tile set."""
        arr2, ax2 = _representative_yx_or_yxc(arr, axes)
        if arr2.ndim == 2:
            lo, hi = display_limits[0] if display_limits else (float(np.min(arr2)), float(np.max(arr2)))
            ch = self._normalize_with_limit(arr2, lo, hi)
            return np.clip(np.stack([ch, ch, ch], axis=-1) * 255, 0, 255).astype(np.uint8)

        if arr2.ndim == 3 and ax2 == "YXS" and arr2.shape[-1] in (3, 4):
            # RGB-like data: apply global per-sample limits only when available.
            rgb = np.zeros(arr2.shape[:2] + (3,), dtype=np.float32)
            for c in range(min(3, arr2.shape[-1])):
                if display_limits and c < len(display_limits):
                    lo, hi = display_limits[c]
                    rgb[:, :, c] = self._normalize_with_limit(arr2[:, :, c], lo, hi)
                else:
                    rgb[:, :, c] = _normalize_channel_float(arr2[:, :, c])
            return np.clip(rgb * 255, 0, 255).astype(np.uint8)

        if arr2.ndim == 3 and ax2.endswith(("C", "S")):
            h, w = arr2.shape[:2]
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            # Use the same default IF colors as Image Preview.
            for c in range(arr2.shape[-1]):
                color_name = DEFAULT_CHANNEL_COLORS[c % len(DEFAULT_CHANNEL_COLORS)]
                color_vec = np.asarray(COLOR_MAPS.get(color_name, COLOR_MAPS["gray"]), dtype=np.float32)
                if display_limits and c < len(display_limits):
                    lo, hi = display_limits[c]
                    ch = self._normalize_with_limit(arr2[:, :, c], lo, hi)
                else:
                    ch = _normalize_channel_float(arr2[:, :, c])
                rgb += ch[:, :, None] * color_vec[None, None, :]
            return np.clip(rgb, 0, 1).astype(np.float32).__mul__(255).astype(np.uint8)

        return _array_to_rgb_preview(arr, axes)

    def _spatial_shape_from_axes(self, arr, axes):
        arr = np.asarray(arr)
        axes = _guess_axes_for_array(arr, axes)
        if axes and len(axes) == arr.ndim and "Y" in axes and "X" in axes:
            return int(arr.shape[axes.index("Y")]), int(arr.shape[axes.index("X")])
        if arr.ndim < 2:
            raise ValueError(f"Cannot determine Y/X shape from tile with shape {arr.shape} and axes {axes}")
        return int(arr.shape[-2]), int(arr.shape[-1])

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
                tile_raw, tile_axes = self._read_tile_raw_file(m["path"])
                th, tw = self._spatial_shape_from_axes(tile_raw, tile_axes)
                x = sx - min_x
                y = sy - min_y
                placements[m["path"]] = (tw, th, x, y)
                max_right = max(max_right, x + tw)
                max_bottom = max(max_bottom, y + th)

            # Informational only in coordinate mode.
            return int(max_right), int(max_bottom), 0, 0, placements

        # Fallback for older tile names without coordinate metadata.
        first_raw, first_axes = self._read_tile_raw_file(metas[0]["path"])
        base_h, base_w = self._spatial_shape_from_axes(first_raw, first_axes)
        stride_x = base_w - int(round(base_w * overlap / 100.0))
        stride_y = base_h - int(round(base_h * overlap / 100.0))
        if stride_x <= 0 or stride_y <= 0:
            raise ValueError("Overlap must be lower than 100%.")

        for m in metas:
            tile_raw, tile_axes = self._read_tile_raw_file(m["path"])
            th, tw = self._spatial_shape_from_axes(tile_raw, tile_axes)
            x = int(m.get("col", 0)) * stride_x
            y = int(m.get("row", 0)) * stride_y
            placements[m["path"]] = (tw, th, x, y)
            max_right = max(max_right, x + tw)
            max_bottom = max(max_bottom, y + th)

        return int(max_right), int(max_bottom), stride_x, stride_y, placements

    def _tile_canvas_geometry_raw(self, metas, overlap):
        """Compute canvas geometry using raw tile shapes without RGB conversion."""
        placements = {}
        max_right = 0
        max_bottom = 0
        first_arr, first_axes = self._read_tile_raw_file(metas[0]["path"])
        base_h, base_w = self._spatial_shape_from_axes(first_arr, first_axes)
        use_coordinates = all(m.get("x") is not None and m.get("y") is not None for m in metas)

        if use_coordinates:
            scaled_xs, scaled_ys = [], []
            for m in metas:
                ds = float(m.get("downsample") or 1.0)
                if ds <= 0:
                    ds = 1.0
                scaled_xs.append(int(round(float(m["x"]) / ds)))
                scaled_ys.append(int(round(float(m["y"]) / ds)))
            min_x = min(scaled_xs) if scaled_xs else 0
            min_y = min(scaled_ys) if scaled_ys else 0
            for m, sx, sy in zip(metas, scaled_xs, scaled_ys):
                arr, axes = self._read_tile_raw_file(m["path"])
                th, tw = self._spatial_shape_from_axes(arr, axes)
                x = sx - min_x
                y = sy - min_y
                placements[m["path"]] = (tw, th, x, y)
                max_right = max(max_right, x + tw)
                max_bottom = max(max_bottom, y + th)
            return int(max_right), int(max_bottom), 0, 0, placements, first_axes, first_arr.shape, first_arr.dtype

        stride_x = base_w - int(round(base_w * overlap / 100.0))
        stride_y = base_h - int(round(base_h * overlap / 100.0))
        if stride_x <= 0 or stride_y <= 0:
            raise ValueError("Overlap must be lower than 100%.")
        for m in metas:
            arr, axes = self._read_tile_raw_file(m["path"])
            th, tw = self._spatial_shape_from_axes(arr, axes)
            x = int(m.get("col", 0)) * stride_x
            y = int(m.get("row", 0)) * stride_y
            placements[m["path"]] = (tw, th, x, y)
            max_right = max(max_right, x + tw)
            max_bottom = max(max_bottom, y + th)
        return int(max_right), int(max_bottom), stride_x, stride_y, placements, first_axes, first_arr.shape, first_arr.dtype

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
        # Use one shared display normalization for all tiles. Without this, each
        # tile gets normalized independently and seams become visible in the
        # stitched preview/JPG even when raw pixels are correct.
        display_limits = self._compute_tile_display_limits(metas)
        out_w, out_h, stride_x, stride_y, placements = self._tile_canvas_geometry(metas, overlap)

        if preview:
            scale = min(820 / out_w, 430 / out_h, 1.0)
            prev_w = max(1, int(round(out_w * scale)))
            prev_h = max(1, int(round(out_h * scale)))
            canvas = np.zeros((prev_h, prev_w, 3), dtype=np.uint8)
            from PIL import Image
            for m in metas:
                tile = self._read_tile_file(m["path"], display_limits=display_limits)
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
            tile = self._read_tile_file(m["path"], display_limits=display_limits)
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

    def _paste_raw_tile_into_canvas(self, canvas, canvas_axes, tile, tile_axes, x, y):
        canvas_axes = _guess_axes_for_array(canvas, canvas_axes)
        tile_axes = _guess_axes_for_array(tile, tile_axes)
        if canvas_axes != tile_axes:
            raise ValueError(f"Tile axes mismatch. Canvas axes={canvas_axes}, tile axes={tile_axes}")
        y_axis = canvas_axes.index("Y")
        x_axis = canvas_axes.index("X")
        for i, ax in enumerate(canvas_axes):
            if ax not in ("Y", "X") and canvas.shape[i] != tile.shape[i]:
                raise ValueError(
                    f"Non-spatial dimension mismatch for axis {ax}: canvas={canvas.shape[i]}, tile={tile.shape[i]}"
                )
        slicer = [slice(None)] * canvas.ndim
        th = int(tile.shape[y_axis])
        tw = int(tile.shape[x_axis])
        slicer[y_axis] = slice(int(y), int(y) + th)
        slicer[x_axis] = slice(int(x), int(x) + tw)
        canvas[tuple(slicer)] = tile

    def _reconstruct_tiles_raw(self, output_hint: Path):
        """Merge raw TIFF/OME-TIFF tiles without RGB conversion or intensity normalization.

        Uses a disk-backed memmap for the merged canvas to avoid holding the full
        multichannel image only in RAM. This is intended for tiles generated by
        this app's raw multichannel tiling mode.
        """
        if self.merge_mode_combo.currentText().startswith("Manual"):
            raise ValueError("Raw multichannel merge is currently supported for Auto-from-names mode only.")
        metas, base = self._collect_tile_metadata()
        overlap = self._merge_overlap_value(metas)
        out_w, out_h, stride_x, stride_y, placements, axes, first_shape, dtype = self._tile_canvas_geometry_raw(metas, overlap)
        axes = _guess_axes_for_array(np.empty(first_shape, dtype=dtype), axes)
        if not axes or "Y" not in axes or "X" not in axes:
            raise ValueError(f"Cannot raw-merge tiles without valid Y/X axes. Got axes={axes!r}")
        out_shape = list(first_shape)
        out_shape[axes.index("Y")] = int(out_h)
        out_shape[axes.index("X")] = int(out_w)
        out_shape = tuple(out_shape)
        tmp_path = Path(output_hint).with_suffix(Path(output_hint).suffix + ".rawmerge.tmp.dat")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        canvas = np.memmap(str(tmp_path), mode="w+", dtype=dtype, shape=out_shape)
        canvas[:] = 0
        self.progress.setVisible(True)
        self.progress.setRange(0, len(metas))
        try:
            for i, m in enumerate(metas, start=1):
                tile, tile_axes = self._read_tile_raw_file(m["path"])
                _, _, x, y = placements[m["path"]]
                self._paste_raw_tile_into_canvas(canvas, axes, tile, tile_axes, x, y)
                self.progress.setValue(i)
                if i % 3 == 0:
                    self.info_label.setText(f"Raw merge: {i}/{len(metas)} tiles")
                    QApplication.processEvents()
            canvas.flush()
            return canvas, axes, base, overlap, min(stride_x, stride_y) if stride_x and stride_y else 0, tmp_path
        except Exception:
            try:
                del canvas
            except Exception:
                pass
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise
        finally:
            self.progress.setVisible(False)

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
            combo = self.merge_out_combo.currentText()
            output_format = _write_format_from_combo(combo)
            ext = _extension_from_combo(combo)
            write_ome = str(combo).startswith("OME-TIFF")
            raw_merge = (
                hasattr(self, "merge_preserve_raw_chk")
                and self.merge_preserve_raw_chk.isChecked()
                and output_format != "jpeg"
            )

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

            if raw_merge:
                metas, base_for_name = self._collect_tile_metadata()
                overlap_for_name = self._merge_overlap_value(metas)
                suffix = _suffix_for_tile(overlap_for_name, 1)
                out_path = self.tile_files[0].parent / f"{base_for_name}_merged_raw{suffix}{ext}"
                canvas = None
                tmp_path = None
                try:
                    canvas, axes, base, overlap, stride, tmp_path = self._reconstruct_tiles_raw(out_path)
                    save_multichannel_image(
                        out_path, canvas, axes=axes, write_ome=write_ome,
                        lossless=self.merge_lossless_chk.isChecked(),
                        source_resolution=source_resolution, source_mpp=source_mpp,
                        image_name=out_path.stem, pixel_scale=1.0,
                    )
                    # Display a safe preview from the saved output rather than rendering the full memmap.
                    arr_prev, axes_prev, _ = read_preview_array_from_file(str(out_path), max_side=900)
                    self._set_label_pixmap(self.merge_thumb, _array_to_rgb_preview(arr_prev, axes_prev))
                    self.info_label.setText(f"Saved raw multichannel merged image: {out_path}")
                    QMessageBox.information(self, "Done", f"Saved raw multichannel merged image:\n{out_path}")
                finally:
                    try:
                        del canvas
                    except Exception:
                        pass
                    if tmp_path is not None:
                        try:
                            Path(tmp_path).unlink(missing_ok=True)
                        except Exception:
                            pass
                return

            rgb, base, overlap, stride = self._reconstruct_tiles(preview=False)
            suffix = _suffix_for_tile(overlap, 1)
            out_path = self.tile_files[0].parent / f"{base}_merged{suffix}{ext}"
            save_rgb_image(
                out_path, rgb, output_format, write_ome, self.merge_lossless_chk.isChecked(),
                source_resolution=source_resolution, source_mpp=source_mpp,
                image_name=out_path.stem, pixel_scale=1.0,
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