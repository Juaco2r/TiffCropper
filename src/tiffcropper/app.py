
import os
import sys

def _setup_openslide_dll_path():
    # When frozen with PyInstaller, binaries are unpacked under sys._MEIPASS
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        dll_dir = os.path.join(sys._MEIPASS, "openslide_bin")
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)  # Python 3.8+
            except Exception:
                pass
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

_setup_openslide_dll_path()


import math
import numpy as np
import tifffile
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QLineEdit, QLabel, QFileDialog,
    QSpinBox, QMessageBox, QComboBox, QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QImage


# -----------------------------
# Helpers
# -----------------------------
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


def _build_ome_xml_rgb(
    size_x: int,
    size_y: int,
    physical_size_x_um: float | None,
    physical_size_y_um: float | None,
    image_name: str,
    annotation_kv: dict | None = None,
) -> str:
    psx = f' PhysicalSizeX="{physical_size_x_um:.6f}" PhysicalSizeXUnit="um"' if physical_size_x_um else ""
    psy = f' PhysicalSizeY="{physical_size_y_um:.6f}" PhysicalSizeYUnit="um"' if physical_size_y_um else ""

    ann_xml = ""
    if annotation_kv:
        ann_xml = _ome_map_annotation_xml(annotation_kv, ann_id="Annotation:0")

    ome = f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="{xml_escape(_ascii_safe(image_name))}">
    <Pixels ID="Pixels:0"
            DimensionOrder="XYCZT"
            Type="uint8"
            SizeX="{size_x}"
            SizeY="{size_y}"
            SizeC="3"
            SizeZ="1"
            SizeT="1"{psx}{psy}>
      <Channel ID="Channel:0" SamplesPerPixel="3"/>
      <TiffData IFD="0" PlaneCount="1"/>
    </Pixels>
  </Image>
  {ann_xml}
</OME>
"""
    return ome


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Ensure array is uint8 RGB (H,W,3)."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim == 3 and arr.shape[2] != 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    return arr


def _downsample_for_preview(rgb: np.ndarray, max_side: int = 512) -> np.ndarray:
    """Fast downsample via striding to avoid heavy resizing libs."""
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    step = int(np.ceil(m / max_side))
    return rgb[::step, ::step, :]


def _numpy_rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    """Convert uint8 RGB array to QPixmap."""
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    bytes_per_line = 3 * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def crop_openslide_multilevel_robust(
    path: str, x0: int, y0: int, w: int, h: int,
    block: int = 1024, fill: int = 255, prefer_levels=(2, 3)
):
    """
    Robust ROI crop for OpenSlide-readable WSIs (NDPI or pyramidal TIFF).
    - Tries level 0 per block.
    - On failure, reopens slide handle (avoids sticky error state) and fills from prefer_levels then others.
    Returns: (roi_rgb_uint8, failed_level0_blocks, recovered_blocks, min_fallback_level_used)
    """
    openslide = _try_import_openslide()
    if openslide is None:
        raise RuntimeError("OpenSlide not available.")

    s0 = openslide.OpenSlide(path)
    level_count = s0.level_count
    downsamples = list(s0.level_downsamples)
    W0, H0 = s0.level_dimensions[0]
    s0.close()

    if x0 < 0 or y0 < 0 or x0 + w > W0 or y0 + h > H0:
        raise ValueError(
            f"ROI out of bounds.\nSlide (W,H)=({W0},{H0})\nROI x={x0}, y={y0}, w={w}, h={h}"
        )

    max_lvl = level_count - 1
    fallback_levels = []
    for lv in prefer_levels:
        lv = int(lv)
        if 1 <= lv <= max_lvl and lv not in fallback_levels:
            fallback_levels.append(lv)
    for lv in range(1, max_lvl + 1):
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

            filled = False
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
                    if min_lvl_used is None or lvl < min_lvl_used:
                        min_lvl_used = lvl
                    filled = True
                    break
                except Exception:
                    try:
                        s.close()
                    except Exception:
                        pass
                    s = openslide.OpenSlide(path)

            if not filled:
                pass

    try:
        s.close()
    except Exception:
        pass

    return out, failed0, recovered, min_lvl_used


# -----------------------------
# GUI
# -----------------------------
class CropWSIGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔬 WSI Cropper - TIFF / OME-TIFF / NDPI")
        self.setGeometry(100, 100, 980, 620)
        self.setStyleSheet("background-color: #f0f0f0;")
        self._tiff_via_openslide = False

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        title = QLabel("WSI Cropper (TIFF / OME-TIFF / NDPI)")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 15px;")
        layout.addWidget(title)

        # File selector
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px;")
        self.browse_btn = QPushButton("📁 Browse")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_btn)
        file_layout.addWidget(self.file_label, 1)
        layout.addLayout(file_layout)

        # Coordinates
        coords_layout = QHBoxLayout()
        self.x_spin = self._mk_spin("X:", 0, 10_000_000, 0, coords_layout)
        self.y_spin = self._mk_spin("Y:", 0, 10_000_000, 0, coords_layout)
        self.w_spin = self._mk_spin("Width:", 1, 10_000_000, 1000, coords_layout)
        self.h_spin = self._mk_spin("Height:", 1, 10_000_000, 1000, coords_layout)
        layout.addLayout(coords_layout)

        # Output format
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Output:"))
        self.out_combo = QComboBox()
        self.out_combo.addItems(["TIFF (.tif)", "OME-TIFF (.ome.tif)"])
        self.out_combo.setMaximumWidth(220)
        out_layout.addWidget(self.out_combo)
        out_layout.addStretch()
        layout.addLayout(out_layout)

        # Suffix
        suffix_layout = QHBoxLayout()
        suffix_layout.addWidget(QLabel("Suffix (after _crop_):"))
        self.suffix_edit = QLineEdit()
        self.suffix_edit.setPlaceholderText("final (default)")
        self.suffix_edit.setText("final")
        self.suffix_edit.setMaximumWidth(180)
        suffix_layout.addWidget(self.suffix_edit)
        suffix_layout.addStretch()
        layout.addLayout(suffix_layout)

        example = QLabel('Example: "1" → name_crop_1.tif | "ROI_A" → name_crop_ROI_A.tif | empty → name_crop_final.tif')
        example.setStyleSheet("color: #7f8c8d; font-size: 10px; padding: 5px;")
        example.setWordWrap(True)
        layout.addWidget(example)

        # Options
        opt_group = QGroupBox("Options")
        opt_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        opt_layout = QHBoxLayout(opt_group)

        self.lossless_chk = QCheckBox("Lossless compression (DEFLATE)")
        self.lossless_chk.setChecked(True)

        self.preview_chk = QCheckBox("Preview panel (thumbnails)")
        self.preview_chk.setChecked(True)
        self.preview_chk.stateChanged.connect(self._on_preview_toggle)

        opt_layout.addWidget(self.lossless_chk)
        opt_layout.addWidget(self.preview_chk)
        opt_layout.addStretch()
        layout.addWidget(opt_group)

        # Preview panel
        self.preview_group = QGroupBox("Preview")
        self.preview_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        prev_layout = QHBoxLayout(self.preview_group)

        self.thumb_in = QLabel("Input thumbnail")
        self.thumb_in.setAlignment(Qt.AlignCenter)
        self.thumb_in.setFixedSize(460, 260)
        self.thumb_in.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")

        self.thumb_out = QLabel("Crop preview")
        self.thumb_out.setAlignment(Qt.AlignCenter)
        self.thumb_out.setFixedSize(460, 260)
        self.thumb_out.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")

        prev_layout.addWidget(self.thumb_in)
        prev_layout.addWidget(self.thumb_out)
        layout.addWidget(self.preview_group)

        # Buttons row: Preview + Crop
        btn_row = QHBoxLayout()

        self.preview_btn = QPushButton("👁 Preview Crop")
        self.preview_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 10px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #7f8c8d; }
            QPushButton:pressed { background-color: #707b7c; }
        """)
        self.preview_btn.clicked.connect(self.preview_crop)

        self.crop_btn = QPushButton("✂️ CROP & SAVE")
        self.crop_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.crop_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 14px;
                border-radius: 10px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:pressed { background-color: #1f618d; }
        """)
        self.crop_btn.clicked.connect(self.crop_image)

        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.crop_btn)
        layout.addLayout(btn_row)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #27ae60; padding: 10px;")
        layout.addWidget(self.info_label)

        # Runtime state
        self.file_path = None
        self.file_kind = None     # "tiff" or "ndpi"
        self.slide_dims = None    # (w,h)
        self.source_resolution = None  # (xres, yres, unit_str)
        self.source_mpp = None         # (mpp_x_um, mpp_y_um)
        self.openslide_props = {}
        self._input_thumb_rgb = None  # cached preview RGB

    def _mk_spin(self, label, mn, mx, val, parent_layout):
        box = QVBoxLayout()
        box.addWidget(QLabel(label))
        sp = QSpinBox()
        sp.setRange(mn, mx)
        sp.setValue(val)
        box.addWidget(sp)
        parent_layout.addLayout(box)
        return sp

    def get_suffix(self):
        s = self.suffix_edit.text().strip()
        return s if s else "final"

    def _on_preview_toggle(self):
        self.preview_group.setVisible(self.preview_chk.isChecked())

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select WSI",
            "",
            "WSI Files (*.tif *.tiff *.ome.tif *.ome.tiff *.ndpi);;All Files (*)"
        )
        if not file_path:
            return

        self.file_path = file_path
        self.file_label.setText(os.path.basename(file_path))
        self._input_thumb_rgb = None
        self.thumb_in.clear()
        self.thumb_in.setText("Input thumbnail")
        self.thumb_out.clear()
        self.thumb_out.setText("Crop preview")

        try:
            ext = Path(file_path).suffix.lower()
            lower_name = Path(file_path).name.lower()
            is_ome = lower_name.endswith(".ome.tif") or lower_name.endswith(".ome.tiff")

            if ext in [".tif", ".tiff"] or is_ome:
                self.file_kind = "tiff"
                w, h, res, mpp = self._probe_tiff(file_path)
                if not getattr(self, "_tiff_via_openslide", False):
                    self.openslide_props = {}
            elif ext == ".ndpi":
                self.file_kind = "ndpi"
                w, h, res, mpp, props = self._probe_ndpi(file_path)
                self.openslide_props = props or {}
            else:
                raise ValueError(f"Unsupported extension: {ext}")

            self.slide_dims = (w, h)
            self.source_resolution = res
            self.source_mpp = mpp

            self.info_label.setText(f"✅ Loaded ({self.file_kind.upper()}), Size: {w} x {h} px")

            self.x_spin.setMaximum(max(0, w - 1))
            self.y_spin.setMaximum(max(0, h - 1))
            self.w_spin.setMaximum(max(1, w))
            self.h_spin.setMaximum(max(1, h))

            if self.preview_chk.isChecked():
                self._update_input_thumbnail(force=True)

        except Exception as e:
            self.info_label.setText(f"❌ Cannot read file: {e}")
            QMessageBox.critical(self, "Error", str(e))

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
                    dpi_x = _mpp_to_dpi(mpp_x)
                    dpi_y = _mpp_to_dpi(mpp_y)
                    res_tuple = (dpi_x, dpi_y, "INCH")
                    mpp = (mpp_x, mpp_y)

                slide.close()

                self.openslide_props = props
                self._tiff_via_openslide = True
                return int(w), int(h), res_tuple, mpp
            except Exception:
                self._tiff_via_openslide = False
        else:
            self._tiff_via_openslide = False

        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError("No image series found in TIFF/OME-TIFF.")

            series0 = tif.series[0]
            shape0 = series0.shape
            if len(shape0) < 2:
                raise ValueError(f"Unexpected shape: {shape0}")

            h, w = int(shape0[0]), int(shape0[1])

            page0 = series0.pages[0] if getattr(series0, "pages", None) else tif.pages[0]
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

            mpp = None
            if res_tuple and unit_str == "INCH":
                mpp = (_dpi_to_mpp(xres_f), _dpi_to_mpp(yres_f))

            return w, h, res_tuple, mpp

    def _probe_ndpi(self, path: str):
        openslide = _try_import_openslide()
        if openslide is None:
            raise RuntimeError(
                "OpenSlide is required for .ndpi.\n"
                "For dev: pip install openslide-python openslide-bin"
            )

        slide = openslide.OpenSlide(path)
        w, h = slide.dimensions
        props = dict(slide.properties or {})

        mpp_x = _safe_float(props.get("openslide.mpp-x"))
        mpp_y = _safe_float(props.get("openslide.mpp-y"))

        res_tuple = None
        mpp = None
        if mpp_x and mpp_y:
            dpi_x = _mpp_to_dpi(mpp_x)
            dpi_y = _mpp_to_dpi(mpp_y)
            res_tuple = (dpi_x, dpi_y, "INCH")
            mpp = (mpp_x, mpp_y)

        slide.close()
        return int(w), int(h), res_tuple, mpp, props

    def _clip_roi(self, x, y, w, h, full_w, full_h):
        x = max(0, min(int(x), int(full_w) - 1))
        y = max(0, min(int(y), int(full_h) - 1))
        w = max(1, min(int(w), int(full_w) - x))
        h = max(1, min(int(h), int(full_h) - y))
        return x, y, w, h

    @staticmethod
    def _crop_openslide_strict(path: str, x: int, y: int, w: int, h: int) -> np.ndarray:
        openslide = _try_import_openslide()
        if openslide is None:
            raise RuntimeError("OpenSlide not available.")
        slide = openslide.OpenSlide(path)
        try:
            img = slide.read_region((int(x), int(y)), 0, (int(w), int(h))).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
        finally:
            try:
                slide.close()
            except Exception:
                pass
        return arr

    def _crop_tiff(self, path: str, x: int, y: int, w: int, h: int):
        fallback_info = {"used": False}

        if getattr(self, "_tiff_via_openslide", False):
            try:
                roi = self._crop_openslide_strict(path, x, y, w, h)
                return roi, fallback_info
            except Exception as e:
                roi, failed0, recovered, min_lvl = crop_openslide_multilevel_robust(
                    path, int(x), int(y), int(w), int(h),
                    block=1024, fill=255, prefer_levels=(2, 3)
                )
                fallback_info = {
                    "used": True,
                    "failed0": failed0,
                    "recovered": recovered,
                    "fallback_level": min_lvl,
                    "original_error": str(e),
                }
                return roi, fallback_info

        # Non-OpenSlide TIFF
        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError("No image series found in TIFF/OME-TIFF.")
            s0 = tif.series[0]
            try:
                z = s0.aszarr()
                import zarr  # type: ignore
                za = zarr.open(z, mode="r")
                arr = za[y:y+h, x:x+w, ...]
                return np.asarray(arr), fallback_info
            except Exception:
                page = tif.pages[0]
                full = page.asarray()
                if full.ndim >= 3:
                    return full[y:y+h, x:x+w, ...], fallback_info
                return full[y:y+h, x:x+w], fallback_info

    def _crop_ndpi(self, path: str, x: int, y: int, w: int, h: int):
        fallback_info = {"used": False}
        try:
            roi = self._crop_openslide_strict(path, x, y, w, h)
            return roi, fallback_info
        except Exception as e:
            roi, failed0, recovered, min_lvl = crop_openslide_multilevel_robust(
                path, int(x), int(y), int(w), int(h),
                block=1024, fill=255, prefer_levels=(2, 3)
            )
            fallback_info = {
                "used": True,
                "failed0": failed0,
                "recovered": recovered,
                "fallback_level": min_lvl,
                "original_error": str(e),
            }
            return roi, fallback_info

    # -----------------------------
    # Preview Crop button
    # -----------------------------
    def _choose_preview_level(self, slide, w: int, h: int, target: int = 900) -> int:
        long_side = max(w, h)
        if long_side <= target:
            return 0
        for lvl in range(slide.level_count):
            ds = float(slide.level_downsamples[lvl])
            if long_side / ds <= target:
                return lvl
        return slide.level_count - 1

    def preview_crop(self):
        if not self.file_path or not self.file_kind or not self.slide_dims:
            QMessageBox.warning(self, "Error", "Please select a file first!")
            return

        try:
            x = self.x_spin.value()
            y = self.y_spin.value()
            w = self.w_spin.value()
            h = self.h_spin.value()

            full_w, full_h = self.slide_dims
            x, y, w, h = self._clip_roi(x, y, w, h, full_w, full_h)

            self.info_label.setText("👁 Generating preview...")
            QApplication.processEvents()

            openslide = _try_import_openslide()
            use_openslide = openslide is not None and (
                self.file_kind == "ndpi" or getattr(self, "_tiff_via_openslide", False)
            )

            if use_openslide:
                slide = openslide.OpenSlide(self.file_path)
                lvl = self._choose_preview_level(slide, w, h, target=900)
                ds = float(slide.level_downsamples[lvl])
                pw = max(1, int(math.ceil(w / ds)))
                ph = max(1, int(math.ceil(h / ds)))

                try:
                    img = slide.read_region((x, y), lvl, (pw, ph)).convert("RGB")
                except Exception:
                    # reopen + try more downsampled level once
                    try:
                        slide.close()
                    except Exception:
                        pass
                    slide = openslide.OpenSlide(self.file_path)
                    lvl = min(lvl + 1, slide.level_count - 1)
                    ds = float(slide.level_downsamples[lvl])
                    pw = max(1, int(math.ceil(w / ds)))
                    ph = max(1, int(math.ceil(h / ds)))
                    img = slide.read_region((x, y), lvl, (pw, ph)).convert("RGB")

                try:
                    slide.close()
                except Exception:
                    pass

                roi_prev = _to_uint8_rgb(np.asarray(img, dtype=np.uint8))
                level_txt = f"level {lvl} (ds≈{int(round(ds))}x)"
            else:
                # Non-OpenSlide TIFF: this may be heavy if huge; we try to avoid full decode but there is no pyramid.
                with tifffile.TiffFile(self.file_path) as tif:
                    s0 = tif.series[0]
                    arr = s0.asarray()
                    arr = arr[y:y+h, x:x+w, ...]
                roi_prev = _to_uint8_rgb(np.asarray(arr))
                level_txt = "direct"

            if self.preview_chk.isChecked():
                self._update_input_thumbnail(force=True)
                self._update_crop_thumbnail(roi_prev)

            self.info_label.setText(f"👁 Preview ready ({w}×{h}) [{level_txt}]")

        except Exception as e:
            self.info_label.setText(f"❌ Preview error: {str(e)}")
            QMessageBox.critical(self, "Preview Error", str(e))

    # -----------------------------
    # Thumbnails
    # -----------------------------
    def _update_input_thumbnail(self, force: bool = False):
        if not self.file_path:
            return

        if (not force) and (self._input_thumb_rgb is not None):
            pm = _numpy_rgb_to_qpixmap(self._input_thumb_rgb)
            self.thumb_in.setPixmap(pm.scaled(self.thumb_in.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            return

        rgb = None

        openslide = _try_import_openslide()
        if openslide is not None:
            try:
                slide = openslide.OpenSlide(self.file_path)
                lvl = slide.level_count - 1
                w, h = slide.level_dimensions[lvl]
                img = slide.read_region((0, 0), lvl, (int(w), int(h))).convert("RGB")
                slide.close()
                rgb = np.asarray(img)
            except Exception:
                rgb = None

        if rgb is None:
            try:
                with tifffile.TiffFile(self.file_path) as tif:
                    if not tif.series:
                        raise ValueError("No TIFF series found.")
                    s0 = tif.series[0]
                    if hasattr(s0, "levels") and s0.levels:
                        arr = s0.levels[-1].asarray()
                    else:
                        arr = s0.asarray()
                    rgb = _to_uint8_rgb(arr)
            except Exception as e:
                self.thumb_in.setText(f"Input thumbnail\n(unavailable)\n{e}")
                return

        rgb = _to_uint8_rgb(rgb)
        rgb = _downsample_for_preview(rgb, max_side=512)
        self._input_thumb_rgb = rgb

        pm = _numpy_rgb_to_qpixmap(rgb)
        self.thumb_in.setPixmap(pm.scaled(self.thumb_in.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _update_crop_thumbnail(self, roi: np.ndarray):
        rgb = _to_uint8_rgb(roi)
        rgb = _downsample_for_preview(rgb, max_side=512)
        pm = _numpy_rgb_to_qpixmap(rgb)
        self.thumb_out.setPixmap(pm.scaled(self.thumb_out.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # -----------------------------
    # Main action
    # -----------------------------
    def crop_image(self):
        if not self.file_path or not self.file_kind or not self.slide_dims:
            QMessageBox.warning(self, "Error", "Please select a file first!")
            return

        try:
            x = self.x_spin.value()
            y = self.y_spin.value()
            w = self.w_spin.value()
            h = self.h_spin.value()

            full_w, full_h = self.slide_dims
            x, y, w, h = self._clip_roi(x, y, w, h, full_w, full_h)

            suffix = self.get_suffix()
            base_path = Path(self.file_path)

            write_ome = self.out_combo.currentText().startswith("OME-TIFF")

            if write_ome:
                output_path = base_path.parent / f"{base_path.stem}_crop_{suffix}.ome.tif"
            else:
                output_path = base_path.parent / f"{base_path.stem}_crop_{suffix}.tif"

            self.info_label.setText("🔄 Cropping...")
            QApplication.processEvents()

            if self.file_kind == "tiff":
                roi, fallback_info = self._crop_tiff(self.file_path, x, y, w, h)
            else:
                roi, fallback_info = self._crop_ndpi(self.file_path, x, y, w, h)

            roi = _to_uint8_rgb(roi)

            if self.preview_chk.isChecked():
                self._update_input_thumbnail(force=True)
                self._update_crop_thumbnail(roi)

            self.info_label.setText("💾 Saving...")
            QApplication.processEvents()

            use_lossless = self.lossless_chk.isChecked()
            self._save(output_path=output_path, roi=roi, suffix=suffix, write_ome=write_ome, lossless=use_lossless)

            msg = (
                f"Crop saved!\n\n📁 {output_path.name}\n📐 {roi.shape}\n"
                f"🧭 X,Y,W,H = {x},{y},{w},{h}\n"
                f"🗜️ Lossless compression: {'ON' if use_lossless else 'OFF'}"
            )

            self.info_label.setText(f"✅ Saved: {output_path.name} ({roi.shape})")

            if fallback_info.get("used", False):
                QMessageBox.warning(
                    self,
                    "⚠️ Saved with partial recovery",
                    msg + "\n\n⚠️ Note:\nA partial decoding error was detected.\n"
                          "To complete the crop, some patches were read from a downsampled pyramid level "
                          f"(minimum fallback level used: {fallback_info.get('fallback_level')}).\n\n"
                          "Recommendation:\n"
                          "• Review the input file (it may contain corrupted JPEG tiles).\n"
                          "• If you can re-copy/re-export the slide, consider cropping again for full-detail output."
                )
            else:
                QMessageBox.information(self, "Success!", msg)

        except Exception as e:
            self.info_label.setText(f"❌ Error: {str(e)}")
            QMessageBox.critical(self, "Error", str(e))

    def _save(self, output_path: Path, roi: np.ndarray, suffix: str, write_ome: bool, lossless: bool):
        resolution = None
        resolutionunit = None
        if self.source_resolution is not None:
            xres, yres, unit = self.source_resolution
            if xres and yres and unit:
                resolution = (float(xres), float(yres))
                resolutionunit = unit

        mpp_x_um = mpp_y_um = None
        if self.source_mpp:
            mpp_x_um, mpp_y_um = self.source_mpp

        ome_kv = None
        if write_ome and self.file_kind == "ndpi" and self.openslide_props:
            ome_kv = {_ascii_safe(k): _ascii_safe(v) for k, v in self.openslide_props.items()}

        compression_kwargs = {}
        if lossless:
            compression_kwargs = {
                "compression": "deflate",
                "predictor": True,
            }

        roi = _to_uint8_rgb(roi)

        if write_ome:
            ome_xml = _build_ome_xml_rgb(
                size_x=int(roi.shape[1]),
                size_y=int(roi.shape[0]),
                physical_size_x_um=mpp_x_um,
                physical_size_y_um=mpp_y_um,
                image_name=output_path.stem,
                annotation_kv=ome_kv
            )

            tifffile.imwrite(
                str(output_path),
                np.ascontiguousarray(roi),
                bigtiff=True,
                tile=(256, 256),
                photometric="rgb",
                description=_ascii_safe(ome_xml),
                software="CropWSI-GUI",
                resolution=resolution,
                resolutionunit=resolutionunit,
                **compression_kwargs
            )
        else:
            desc = f"Cropped ROI [{suffix}]"
            tifffile.imwrite(
                str(output_path),
                np.ascontiguousarray(roi),
                bigtiff=True,
                tile=(256, 256),
                photometric="rgb",
                description=_ascii_safe(desc),
                software="CropWSI-GUI",
                resolution=resolution,
                resolutionunit=resolutionunit,
                **compression_kwargs
            )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CropWSIGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()