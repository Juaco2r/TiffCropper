import os
import sys
import re
import math
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape


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

import numpy as np
import tifffile

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout,
    QWidget, QPushButton, QLineEdit, QLabel, QFileDialog,
    QSpinBox, QMessageBox, QComboBox, QCheckBox, QGroupBox,
    QStackedWidget, QDoubleSpinBox, QProgressBar, QDialog,
    QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor


SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".ome.tif", ".ome.tiff", ".ndpi", ".jpg", ".jpeg", ".png")


def _try_import_openslide():
    try:
        import openslide  # type: ignore
        return openslide
    except Exception:
        return None


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


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported image array shape: {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(arr)


def _downsample_for_preview(rgb: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    step = int(np.ceil(m / max_side))
    return rgb[::step, ::step, :]


def _numpy_rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


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


def _extension_from_combo(text: str) -> str:
    return ".jpg" if "JPEG" in text.upper() or "JPG" in text.upper() else ".tif"


def _write_format_from_combo(text: str) -> str:
    return "jpeg" if "JPEG" in text.upper() or "JPG" in text.upper() else "tiff"


def _parse_tile_name(path: Path):
    stem = path.stem
    if stem.endswith(".ome"):
        stem = Path(stem).stem
    pattern = re.compile(
        r"^(?P<base>.+)_(?P<col>[A-Z]+)(?P<row>\d+)"
        r"(?:Ov(?P<ov>\d+(?:\.\d+)?))?"
        r"(?:DS(?P<ds>\d+(?:\.\d+)?))?$"
    )
    m = pattern.match(stem)
    if not m:
        return None
    return {
        "path": path,
        "base": m.group("base"),
        "col": _letters_to_col(m.group("col")),
        "row": int(m.group("row")) - 1,
        "overlap": float(m.group("ov")) if m.group("ov") else 0.0,
        "downsample": float(m.group("ds")) if m.group("ds") else 1.0,
    }


def _compute_tile_grid(width: int, height: int, tile_size: int, overlap_percent: float):
    overlap_px = int(round(tile_size * overlap_percent / 100.0))
    stride = tile_size - overlap_px
    if stride <= 0:
        raise ValueError("Overlap must be lower than 100%.")
    x_positions = list(range(0, width, stride))
    y_positions = list(range(0, height, stride))
    return x_positions, y_positions, stride, overlap_px


def _crop_external_padding(rgb: np.ndarray, padding_color: str = "black") -> np.ndarray:
    rgb = _to_uint8_rgb(rgb)
    if padding_color.lower() == "white":
        mask = np.any(rgb < 250, axis=2)
    else:
        mask = np.any(rgb > 5, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return rgb
    return rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1, :]


class ImageBackend:
    def __init__(self):
        self.path = None
        self.path_obj = None
        self.file_kind = None
        self.slide_dims = None
        self.source_resolution = None
        self.source_mpp = None
        self.openslide_props = {}
        self.tiff_via_openslide = False

    def load(self, path: str):
        self.path = path
        self.path_obj = Path(path)
        lower_name = self.path_obj.name.lower()
        ext = self.path_obj.suffix.lower()
        is_ome = lower_name.endswith(".ome.tif") or lower_name.endswith(".ome.tiff")

        if ext in [".tif", ".tiff"] or is_ome:
            self.file_kind = "tiff"
            w, h, res, mpp = self._probe_tiff(path)
            if not self.tiff_via_openslide:
                self.openslide_props = {}
        elif ext == ".ndpi":
            self.file_kind = "ndpi"
            w, h, res, mpp, props = self._probe_ndpi(path)
            self.openslide_props = props or {}
        elif ext in [".jpg", ".jpeg", ".png"]:
            self.file_kind = "raster"
            arr = self._read_with_pil(path)
            h, w = arr.shape[:2]
            res, mpp = None, None
            self.tiff_via_openslide = False
        else:
            raise ValueError(f"Unsupported extension: {ext}")

        self.slide_dims = (int(w), int(h))
        self.source_resolution = res
        self.source_mpp = mpp
        return self

    def _read_with_pil(self, path: str) -> np.ndarray:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    def _probe_tiff(self, path: str):
        openslide = _try_import_openslide()
        if openslide is not None:
            try:
                slide = openslide.OpenSlide(path)
                w, h = slide.dimensions
                props = dict(slide.properties or {})
                mpp_x = _safe_float(props.get("openslide.mpp-x"))
                mpp_y = _safe_float(props.get("openslide.mpp-y"))
                res_tuple = None
                mpp = None
                if mpp_x and mpp_y:
                    res_tuple = (_mpp_to_dpi(mpp_x), _mpp_to_dpi(mpp_y), "INCH")
                    mpp = (mpp_x, mpp_y)
                slide.close()
                self.openslide_props = props
                self.tiff_via_openslide = True
                return int(w), int(h), res_tuple, mpp
            except Exception:
                self.tiff_via_openslide = False

        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError("No image series found in TIFF/OME-TIFF.")
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
            mpp = (_dpi_to_mpp(xres_f), _dpi_to_mpp(yres_f)) if res_tuple and unit_str == "INCH" else None
            return w, h, res_tuple, mpp

    def _probe_ndpi(self, path: str):
        openslide = _try_import_openslide()
        if openslide is None:
            raise RuntimeError("OpenSlide is required for .ndpi. For dev: pip install openslide-python openslide-bin")
        slide = openslide.OpenSlide(path)
        w, h = slide.dimensions
        props = dict(slide.properties or {})
        mpp_x = _safe_float(props.get("openslide.mpp-x"))
        mpp_y = _safe_float(props.get("openslide.mpp-y"))
        res_tuple = None
        mpp = None
        if mpp_x and mpp_y:
            res_tuple = (_mpp_to_dpi(mpp_x), _mpp_to_dpi(mpp_y), "INCH")
            mpp = (mpp_x, mpp_y)
        slide.close()
        return int(w), int(h), res_tuple, mpp, props

    @staticmethod
    def clip_roi(x, y, w, h, full_w, full_h):
        x = max(0, min(int(x), int(full_w) - 1))
        y = max(0, min(int(y), int(full_h) - 1))
        w = max(1, min(int(w), int(full_w) - x))
        h = max(1, min(int(h), int(full_h) - y))
        return x, y, w, h

    def crop(self, x: int, y: int, w: int, h: int, fill: int = 255):
        if not self.path or not self.file_kind or not self.slide_dims:
            raise RuntimeError("No image loaded.")
        full_w, full_h = self.slide_dims
        x, y, w, h = self.clip_roi(x, y, w, h, full_w, full_h)
        if self.file_kind == "ndpi" or self.tiff_via_openslide:
            return self._crop_openslide_robust(self.path, x, y, w, h, fill=fill)
        if self.file_kind == "raster":
            arr = self._read_with_pil(self.path)
            return _to_uint8_rgb(arr[y:y+h, x:x+w, :]), {"used": False}
        return self._crop_tiff(self.path, x, y, w, h)

    def read_tile_with_padding(self, x: int, y: int, size: int, padding_color: str = "black") -> np.ndarray:
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

    def _crop_openslide_robust(self, path, x0, y0, w, h, block=1024, fill=255, prefer_levels=(2, 3)):
        openslide = _try_import_openslide()
        if openslide is None:
            raise RuntimeError("OpenSlide not available.")

        s0 = openslide.OpenSlide(path)
        level_count = s0.level_count
        downsamples = list(s0.level_downsamples)
        W0, H0 = s0.level_dimensions[0]
        s0.close()
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
        s = openslide.OpenSlide(path)

        for by in range(0, h, block):
            bh = min(block, h - by)
            for bx in range(0, w, block):
                bw = min(block, w - bx)
                sx, sy = x0 + bx, y0 + by
                try:
                    im0 = s.read_region((sx, sy), 0, (bw, bh)).convert("RGB")
                    out[by:by+bh, bx:bx+bw, :] = np.asarray(im0, dtype=np.uint8)
                    continue
                except Exception:
                    failed0 += 1
                    try:
                        s.close()
                    except Exception:
                        pass
                    s = openslide.OpenSlide(path)

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
                        break
                    except Exception:
                        try:
                            s.close()
                        except Exception:
                            pass
                        s = openslide.OpenSlide(path)
        try:
            s.close()
        except Exception:
            pass
        return out, {"used": failed0 > 0, "failed0": failed0, "recovered": recovered, "fallback_level": min_lvl_used}

    def _crop_tiff(self, path, x, y, w, h):
        fallback_info = {"used": False}
        with tifffile.TiffFile(path) as tif:
            s0 = tif.series[0]
            try:
                z = s0.aszarr()
                import zarr  # type: ignore
                za = zarr.open(z, mode="r")
                arr = za[y:y+h, x:x+w, ...]
                return _to_uint8_rgb(np.asarray(arr)), fallback_info
            except Exception:
                full = tif.pages[0].asarray()
                return _to_uint8_rgb(full[y:y+h, x:x+w, ...]), fallback_info

    def input_thumbnail(self, max_side=512):
        openslide = _try_import_openslide()
        if openslide is not None:
            try:
                slide = openslide.OpenSlide(self.path)
                lvl = slide.level_count - 1
                w, h = slide.level_dimensions[lvl]
                img = slide.read_region((0, 0), lvl, (int(w), int(h))).convert("RGB")
                slide.close()
                return _downsample_for_preview(_to_uint8_rgb(np.asarray(img)), max_side=max_side)
            except Exception:
                pass
        if self.file_kind == "raster":
            return _downsample_for_preview(_to_uint8_rgb(self._read_with_pil(self.path)), max_side=max_side)
        with tifffile.TiffFile(self.path) as tif:
            s0 = tif.series[0]
            if hasattr(s0, "levels") and s0.levels:
                arr = s0.levels[-1].asarray()
            else:
                arr = s0.asarray()
        return _downsample_for_preview(_to_uint8_rgb(arr), max_side=max_side)


def save_rgb_image(output_path, rgb, output_format="tiff", write_ome=False, lossless=True,
                   source_resolution=None, source_mpp=None, image_name=None, annotation_kv=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _to_uint8_rgb(rgb)

    if output_format == "jpeg":
        from PIL import Image
        Image.fromarray(rgb).save(str(output_path), quality=95)
        return

    resolution = None
    resolutionunit = None
    if source_resolution is not None:
        xres, yres, unit = source_resolution
        if xres and yres and unit:
            resolution = (float(xres), float(yres))
            resolutionunit = unit

    mpp_x_um = mpp_y_um = None
    if source_mpp:
        mpp_x_um, mpp_y_um = source_mpp

    compression_kwargs = {"compression": "deflate"} if lossless else {}

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
            description=_ascii_safe(ome_xml), software="WSI-Crop-Tile-Merge-GUI",
            resolution=resolution, resolutionunit=resolutionunit, **compression_kwargs
        )
    else:
        tifffile.imwrite(
            str(output_path), rgb, bigtiff=True, tile=(256, 256), photometric="rgb",
            description=_ascii_safe("Generated by WSI Crop / Tile / Merge GUI"),
            software="WSI-Crop-Tile-Merge-GUI", resolution=resolution,
            resolutionunit=resolutionunit, **compression_kwargs
        )


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
        return {
            "rows": self.rows_spin.value(),
            "cols": self.cols_spin.value(),
            "mapping": dict(self.mapping),
        }


class WSICropTileMergeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WSI Crop / Tile / Merge - TIFF / OME-TIFF / NDPI")
        self.setGeometry(100, 80, 1120, 760)
        self.setStyleSheet("background-color: #f0f0f0;")
        self.backend = ImageBackend()
        self.tile_files = []
        self.bulk_paths = []
        self.manual_layout = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        title = QLabel("WSI Crop / Tile / Merge Tool")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 12px;")
        root.addWidget(title)

        menu_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Crop", "Tiles", "Merge Tiles"])
        self.mode_combo.currentIndexChanged.connect(lambda: self.stack.setCurrentIndex(self.mode_combo.currentIndex()))
        self.mode_combo.setMaximumWidth(180)
        menu_row.addWidget(QLabel("Mode:"))
        menu_row.addWidget(self.mode_combo)
        menu_row.addStretch()
        root.addLayout(menu_row)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.stack.addWidget(self._build_crop_page())
        self.stack.addWidget(self._build_tiles_page())
        self.stack.addWidget(self._build_merge_page())

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #27ae60; padding: 8px;")
        root.addWidget(self.info_label)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

    def _set_label_pixmap(self, label, rgb):
        pm = _numpy_rgb_to_qpixmap(_to_uint8_rgb(rgb))
        label.setPixmap(pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _draw_grid_on_thumb(self, thumb_rgb, full_w, full_h, tile_size, overlap):
        pm = _numpy_rgb_to_qpixmap(thumb_rgb)
        painter = QPainter(pm)
        pen_grid = QPen(QColor(220, 0, 0), 1)
        pen_pad = QPen(QColor(230, 190, 0), 2)
        sx = pm.width() / float(full_w)
        sy = pm.height() / float(full_h)
        x_positions, y_positions, stride, _ = _compute_tile_grid(full_w, full_h, tile_size, overlap)
        painter.setPen(pen_grid)
        for y in y_positions:
            for x in x_positions:
                painter.drawRect(int(x * sx), int(y * sy), int(tile_size * sx), int(tile_size * sy))
        painter.setPen(pen_pad)
        painter.drawRect(0, 0, int((x_positions[-1] + tile_size) * sx), int((y_positions[-1] + tile_size) * sy))
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

        coords_layout = QHBoxLayout()
        self.x_spin = self._mk_spin("X:", 0, 10_000_000, 0, coords_layout)
        self.y_spin = self._mk_spin("Y:", 0, 10_000_000, 0, coords_layout)
        self.w_spin = self._mk_spin("Width:", 1, 10_000_000, 1000, coords_layout)
        self.h_spin = self._mk_spin("Height:", 1, 10_000_000, 1000, coords_layout)
        self.full_area_btn = QPushButton("Select full area")
        self.full_area_btn.clicked.connect(self.select_full_crop_area)
        coords_layout.addWidget(self.full_area_btn)
        layout.addLayout(coords_layout)

        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Output:"))
        self.crop_out_combo = QComboBox()
        self.crop_out_combo.addItems(["TIFF (.tif)", "OME-TIFF (.ome.tif)", "JPEG (.jpg)"])
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
        self.crop_preview_chk = QCheckBox("Preview panel")
        self.crop_preview_chk.setChecked(True)
        opt_layout.addWidget(self.crop_lossless_chk)
        opt_layout.addWidget(self.crop_preview_chk)
        opt_layout.addStretch()
        layout.addWidget(opt)

        prev = QGroupBox("Preview")
        prev_layout = QHBoxLayout(prev)
        self.crop_thumb_in = QLabel("Input thumbnail")
        self.crop_thumb_in.setAlignment(Qt.AlignCenter)
        self.crop_thumb_in.setFixedSize(520, 300)
        self.crop_thumb_in.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")
        self.crop_thumb_out = QLabel("Crop preview")
        self.crop_thumb_out.setAlignment(Qt.AlignCenter)
        self.crop_thumb_out.setFixedSize(520, 300)
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

    def browse_crop_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select WSI", "", "Image Files (*.tif *.tiff *.ome.tif *.ome.tiff *.ndpi *.jpg *.jpeg *.png);;All Files (*)")
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
            self.info_label.setText(f"Loaded: {Path(file_path).name} | Size: {w} x {h} px")
            if self.crop_preview_chk.isChecked():
                self._set_label_pixmap(self.crop_thumb_in, self.backend.input_thumbnail())
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
        self.info_label.setText(f"Full area selected: 0, 0, {full_w}, {full_h}")

    def preview_crop(self):
        if not self.backend.path:
            QMessageBox.warning(self, "Error", "Please select a file first.")
            return
        try:
            roi, _ = self.backend.crop(self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value(), fill=255)
            downsample = float(self.crop_downsample_spin.value())
            if downsample != 1.0:
                from PIL import Image
                new_w = max(1, int(round(roi.shape[1] / downsample)))
                new_h = max(1, int(round(roi.shape[0] / downsample)))
                roi = np.asarray(Image.fromarray(roi).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)
            if self.crop_preview_chk.isChecked():
                self._set_label_pixmap(self.crop_thumb_in, self.backend.input_thumbnail())
                self._set_label_pixmap(self.crop_thumb_out, _downsample_for_preview(roi, 512))
            self.info_label.setText(f"Preview ready: {roi.shape[1]} x {roi.shape[0]} px")
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", str(e))

    def crop_image(self):
        if not self.backend.path or not self.backend.path_obj:
            QMessageBox.warning(self, "Error", "Please select a file first.")
            return
        try:
            suffix = self.crop_suffix_edit.text().strip() or "final"
            downsample = float(self.crop_downsample_spin.value())
            if downsample != 1.0:
                ds_txt = f"DS{int(downsample)}" if downsample.is_integer() else f"DS{downsample:g}"
                suffix = f"{suffix}{ds_txt}"
            combo = self.crop_out_combo.currentText()
            output_format = _write_format_from_combo(combo)
            write_ome = combo.startswith("OME-TIFF")
            ext = ".ome.tif" if write_ome else _extension_from_combo(combo)
            out_path = self.backend.path_obj.parent / f"{self.backend.path_obj.stem}_crop_{suffix}{ext}"
            roi, fallback_info = self.backend.crop(self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value(), fill=255)
            raw_downsample = float(self.crop_downsample_spin.value())
            if raw_downsample != 1.0:
                from PIL import Image
                new_w = max(1, int(round(roi.shape[1] / raw_downsample)))
                new_h = max(1, int(round(roi.shape[0] / raw_downsample)))
                roi = np.asarray(Image.fromarray(roi).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)
            save_rgb_image(out_path, roi, output_format, write_ome, self.crop_lossless_chk.isChecked(), self.backend.source_resolution, self.backend.source_mpp, out_path.stem, self.backend.openslide_props if write_ome else None)
            if self.crop_preview_chk.isChecked():
                self._set_label_pixmap(self.crop_thumb_out, _downsample_for_preview(roi, 512))
            self.info_label.setText(f"Saved: {out_path}")
            QMessageBox.information(self, "Success", f"Saved:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

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
        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(16, 100000)
        self.tile_size_spin.setValue(1024)
        self.tile_overlap_spin = QDoubleSpinBox()
        self.tile_overlap_spin.setRange(0, 99.9)
        self.tile_overlap_spin.setDecimals(2)
        self.tile_overlap_spin.setValue(0.0)
        self.tile_overlap_spin.setSuffix(" %")
        self.tile_downsample_spin = QDoubleSpinBox()
        self.tile_downsample_spin.setRange(0.01, 1000.0)
        self.tile_downsample_spin.setDecimals(2)
        self.tile_downsample_spin.setValue(1.0)
        self.tile_padding_combo = QComboBox()
        self.tile_padding_combo.addItems(["black", "white"])
        self.tile_out_combo = QComboBox()
        self.tile_out_combo.addItems(["TIFF (.tif)", "JPEG (.jpg)"])
        self.tile_lossless_chk = QCheckBox("Lossless TIFF compression")
        self.tile_lossless_chk.setChecked(True)
        grid.addWidget(QLabel("Square tile size px:"), 0, 0)
        grid.addWidget(self.tile_size_spin, 0, 1)
        grid.addWidget(QLabel("Overlap:"), 0, 2)
        grid.addWidget(self.tile_overlap_spin, 0, 3)
        grid.addWidget(QLabel("Downsample:"), 1, 0)
        grid.addWidget(self.tile_downsample_spin, 1, 1)
        grid.addWidget(QLabel("Padding:"), 1, 2)
        grid.addWidget(self.tile_padding_combo, 1, 3)
        grid.addWidget(QLabel("Format:"), 2, 0)
        grid.addWidget(self.tile_out_combo, 2, 1)
        grid.addWidget(self.tile_lossless_chk, 2, 2, 1, 2)
        layout.addWidget(params)

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
        return page

    def load_one_tile_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Image Files (*.tif *.tiff *.ome.tif *.ome.tiff *.ndpi *.jpg *.jpeg *.png);;All Files (*)")
        if path:
            self.bulk_paths = [Path(path)]
            self._load_tiles_preview_image(self.bulk_paths[0])

    def load_bulk_tile_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select images for bulk tiling", "", "Image Files (*.tif *.tiff *.ome.tif *.ome.tiff *.ndpi *.jpg *.jpeg *.png);;All Files (*)")
        if paths:
            self.bulk_paths = [Path(p) for p in paths]
            self._load_tiles_preview_image(self.bulk_paths[0])

    def _load_tiles_preview_image(self, path):
        self.backend = ImageBackend().load(str(path))
        w, h = self.backend.slide_dims
        self.tiles_file_label.setText(f"Selected: {len(self.bulk_paths)} file(s) | Preview: {path.name} | {w} x {h} px")
        self._set_label_pixmap(self.tiles_thumb, self.backend.input_thumbnail(max_side=768))
        self.info_label.setText(f"Loaded for tiling: {path.name}")

    def _tile_params(self):
        return (
            int(self.tile_size_spin.value()),
            float(self.tile_overlap_spin.value()),
            float(self.tile_downsample_spin.value()),
            self.tile_padding_combo.currentText(),
            _write_format_from_combo(self.tile_out_combo.currentText()),
            _extension_from_combo(self.tile_out_combo.currentText()),
        )

    def preview_tiles_grid(self):
        if not self.backend.path or not self.backend.slide_dims:
            QMessageBox.warning(self, "Error", "Load one or more images first.")
            return
        try:
            tile_size, overlap, downsample, padding, output_format, ext = self._tile_params()
            full_w, full_h = self.backend.slide_dims
            thumb = self.backend.input_thumbnail(max_side=768)
            pm = self._draw_grid_on_thumb(thumb, full_w, full_h, tile_size, overlap)
            self.tiles_thumb.setPixmap(pm.scaled(self.tiles_thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            xs, ys, stride, ovpx = _compute_tile_grid(full_w, full_h, tile_size, overlap)
            self.info_label.setText(f"Grid: {len(xs)} columns x {len(ys)} rows = {len(xs)*len(ys)} tiles | Overlap {overlap:g}% ({ovpx}px) | Stride {stride}px")
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", str(e))

    def save_tiles_bulk(self):
        if not self.bulk_paths:
            QMessageBox.warning(self, "Error", "Load one or more images first.")
            return
        try:
            tile_size, overlap, downsample, padding, output_format, ext = self._tile_params()
            suffix = _suffix_for_tile(overlap, downsample)
            jobs = []
            total_tiles = 0
            for p in self.bulk_paths:
                b = ImageBackend().load(str(p))
                w, h = b.slide_dims
                xs, ys, _, _ = _compute_tile_grid(w, h, tile_size, overlap)
                jobs.append((p, len(xs), len(ys)))
                total_tiles += len(xs) * len(ys)
            self.progress.setVisible(True)
            self.progress.setRange(0, total_tiles)
            self.progress.setValue(0)
            QApplication.processEvents()
            written = 0
            for p, _, _ in jobs:
                written += self._save_tiles_one_image(p, tile_size, overlap, downsample, padding, output_format, ext, suffix, written)
            self.progress.setVisible(False)
            self.info_label.setText(f"Done: saved {written} tiles from {len(self.bulk_paths)} image(s).")
            QMessageBox.information(self, "Done", f"Saved {written} tiles from {len(self.bulk_paths)} image(s).")
        except Exception as e:
            self.progress.setVisible(False)
            QMessageBox.critical(self, "Tiling Error", str(e))

    def _save_tiles_one_image(self, image_path, tile_size, overlap, downsample, padding, output_format, ext, suffix, start_progress):
        backend = ImageBackend().load(str(image_path))
        full_w, full_h = backend.slide_dims
        xs, ys, _, _ = _compute_tile_grid(full_w, full_h, tile_size, overlap)
        out_dir = image_path.parent / image_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for row_idx, y in enumerate(ys):
            for col_idx, x in enumerate(xs):
                tile = backend.read_tile_with_padding(x=x, y=y, size=tile_size, padding_color=padding)
                if float(downsample) != 1.0:
                    from PIL import Image
                    new_size = max(1, int(round(tile_size / downsample)))
                    tile = np.asarray(Image.fromarray(tile).resize((new_size, new_size), Image.Resampling.LANCZOS), dtype=np.uint8)
                out_name = f"{image_path.stem}_{_col_to_letters(col_idx)}{row_idx + 1}{suffix}{ext}"
                out_path = out_dir / out_name
                save_rgb_image(out_path, tile, output_format, False, self.tile_lossless_chk.isChecked(), backend.source_resolution, backend.source_mpp, out_path.stem)
                count += 1
                self.progress.setValue(start_progress + count)
                if count % 10 == 0:
                    self.info_label.setText(f"Saving tiles: {image_path.name} | {count}/{len(xs)*len(ys)}")
                    QApplication.processEvents()
        QApplication.processEvents()
        return count

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
            self.tile_files = sorted([p for p in folder.iterdir() if p.is_file() and p.name.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png"))])
            self.merge_file_label.setText(f"Loaded {len(self.tile_files)} tile file(s) from {folder}")

    def select_tile_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select tile files", "", "Image Files (*.tif *.tiff *.jpg *.jpeg *.png);;All Files (*)")
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
            raise ValueError("No valid tiles found. Expected ImageName_A1.tif or ImageName_B2Ov10DS2.tif")
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
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
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
        if tile_w != tile_h:
            raise ValueError(f"Tiles must be square. First tile is {tile_w} x {tile_h}.")

        tile_size = tile_w
        overlap = self._manual_overlap_value()
        overlap_px = int(round(tile_size * overlap / 100.0))
        stride = tile_size - overlap_px
        if stride <= 0:
            raise ValueError("Overlap must be lower than 100%.")

        out_w = (cols - 1) * stride + tile_size
        out_h = (rows - 1) * stride + tile_size
        base = Path(first_path).parent.name or "manual_layout"

        if preview:
            scale = min(820 / out_w, 430 / out_h, 1.0)
            prev_w = max(1, int(round(out_w * scale)))
            prev_h = max(1, int(round(out_h * scale)))
            canvas = np.zeros((prev_h, prev_w, 3), dtype=np.uint8)
            from PIL import Image
            for (r, c), tile_path in mapping.items():
                tile = self._read_tile_file(tile_path)
                small_size = max(1, int(round(tile_size * scale)))
                small = Image.fromarray(tile).resize((small_size, small_size), Image.Resampling.BILINEAR)
                small = np.asarray(small, dtype=np.uint8)
                x = int(round(c * stride * scale))
                y = int(round(r * stride * scale))
                hh, ww = small.shape[:2]
                canvas[y:min(y+hh, prev_h), x:min(x+ww, prev_w), :] = small[:max(0, min(hh, prev_h-y)), :max(0, min(ww, prev_w-x)), :]
            if self.merge_crop_padding_chk.isChecked():
                canvas = _crop_external_padding(canvas, self.merge_padding_combo.currentText())
            return canvas, base, overlap, stride

        merged = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        total = len(mapping)
        self.progress.setVisible(True)
        self.progress.setRange(0, total)
        self.progress.setValue(0)
        for i, ((r, c), tile_path) in enumerate(mapping.items(), start=1):
            tile = self._read_tile_file(tile_path)
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                raise ValueError(f"All tiles must have same square size. Problem: {tile_path.name}")
            x = c * stride
            y = r * stride
            merged[y:y+tile_size, x:x+tile_size, :] = tile
            self.progress.setValue(i)
            if i % 10 == 0:
                QApplication.processEvents()
        self.progress.setVisible(False)
        if self.merge_crop_padding_chk.isChecked():
            merged = _crop_external_padding(merged, self.merge_padding_combo.currentText())
        return merged, base, overlap, stride

    def _reconstruct_tiles(self, preview=False):
        if self.merge_mode_combo.currentText().startswith("Manual"):
            return self._reconstruct_tiles_manual(preview=preview)
        metas, base = self._collect_tile_metadata()
        overlap = self._merge_overlap_value(metas)
        first = self._read_tile_file(metas[0]["path"])
        tile_h, tile_w = first.shape[:2]
        if tile_w != tile_h:
            raise ValueError(f"Tiles must be square. First tile is {tile_w} x {tile_h}.")
        tile_size = tile_w
        overlap_px = int(round(tile_size * overlap / 100.0))
        stride = tile_size - overlap_px
        if stride <= 0:
            raise ValueError("Overlap must be lower than 100%.")
        max_col = max(m["col"] for m in metas)
        max_row = max(m["row"] for m in metas)
        out_w = max_col * stride + tile_size
        out_h = max_row * stride + tile_size

        if preview:
            scale = min(820 / out_w, 430 / out_h, 1.0)
            prev_w = max(1, int(round(out_w * scale)))
            prev_h = max(1, int(round(out_h * scale)))
            canvas = np.zeros((prev_h, prev_w, 3), dtype=np.uint8)
            from PIL import Image
            for m in metas:
                tile = self._read_tile_file(m["path"])
                small = Image.fromarray(tile).resize((max(1, int(round(tile_size * scale))), max(1, int(round(tile_size * scale)))), Image.Resampling.BILINEAR)
                small = np.asarray(small, dtype=np.uint8)
                x = int(round(m["col"] * stride * scale))
                y = int(round(m["row"] * stride * scale))
                hh, ww = small.shape[:2]
                canvas[y:min(y+hh, prev_h), x:min(x+ww, prev_w), :] = small[:max(0, min(hh, prev_h-y)), :max(0, min(ww, prev_w-x)), :]
            if self.merge_crop_padding_chk.isChecked():
                canvas = _crop_external_padding(canvas, self.merge_padding_combo.currentText())
            return canvas, base, overlap, stride

        merged = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(metas))
        for i, m in enumerate(metas, start=1):
            tile = self._read_tile_file(m["path"])
            x = m["col"] * stride
            y = m["row"] * stride
            merged[y:y+tile_size, x:x+tile_size, :] = tile
            self.progress.setValue(i)
            if i % 10 == 0:
                QApplication.processEvents()
        self.progress.setVisible(False)
        if self.merge_crop_padding_chk.isChecked():
            merged = _crop_external_padding(merged, self.merge_padding_combo.currentText())
        return merged, base, overlap, stride

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
            save_rgb_image(out_path, rgb, output_format, False, self.merge_lossless_chk.isChecked(), image_name=out_path.stem)
            self._set_label_pixmap(self.merge_thumb, _downsample_for_preview(rgb, 900))
            self.info_label.setText(f"Saved merged image: {out_path}")
            QMessageBox.information(self, "Done", f"Saved merged image:\n{out_path}")
        except Exception as e:
            self.progress.setVisible(False)
            QMessageBox.critical(self, "Merge Error", str(e))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = WSICropTileMergeGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
