import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from pydicom.multival import MultiValue
from PySide6.QtCore import QPoint, QPointF, QProcess, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QProgressDialog,
    QPushButton, QSlider, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QToolBar, QVBoxLayout, QWidget,
)

try:
    from scipy import ndimage
except Exception:  # scipy необязателен
    ndimage = None

AX, COR, SAG = 0, 1, 2           # совпадает с осями массива (z, y, x)
PLANE_NAMES = {AX: "Аксиальная", COR: "Корональная", SAG: "Сагиттальная"}
LAYOUTS = ["Аксиальный", "Корональный", "Сагиттальный", "MPR", "MPR + 3D", "3D"]

CT_PRESETS = OrderedDict([
    ("Мягкие ткани", (40, 400)), ("Мозг", (40, 80)), ("Лёгкие", (-600, 1500)),
    ("Кости", (400, 1800)), ("Средостение", (50, 350)), ("Печень", (60, 160)),
    ("Сосуды (КТА)", (300, 700)),
])


@dataclass
class Series:
    uid: str
    patient: str
    desc: str
    number: int
    modality: str
    items: list = field(default_factory=list)  # [(path, frame_index)]
    date: str = ""

    def label(self):
        d = self.date
        if len(d) == 8:
            d = f"{d[6:]}.{d[4:6]}.{d[:4]}"
        s = f"#{self.number} {self.modality} {self.desc} ({len(self.items)})"
        return f"{s} · {d}" if d else s


def slice_pos(ds):
    """Позиция среза вдоль нормали (мм) или None."""
    try:
        ipp = np.array(ds.ImagePositionPatient, dtype=float)
        iop = np.array(ds.ImageOrientationPatient, dtype=float)
        return float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
    except Exception:
        return None


def slice_key(ds):
    p = slice_pos(ds)
    return p if p is not None else float(ds.get("InstanceNumber", 0) or 0)


def scan_folder(folder):
    """Рекурсивно сканирует папку и группирует файлы по SeriesInstanceUID."""
    tags = ["SeriesInstanceUID", "PatientName", "SeriesDescription",
            "SeriesNumber", "Modality", "ImagePositionPatient",
            "ImageOrientationPatient", "InstanceNumber", "NumberOfFrames",
            "StudyDate"]
    groups, raw = {}, {}
    for p in Path(folder).rglob("*"):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, specific_tags=tags)
        except Exception:
            continue
        uid = ds.get("SeriesInstanceUID")
        if not uid:
            continue
        uid = str(uid)
        if uid not in groups:
            groups[uid] = Series(
                uid=uid,
                patient=str(ds.get("PatientName", "")),
                desc=str(ds.get("SeriesDescription", "")),
                number=int(ds.get("SeriesNumber", 0) or 0),
                modality=str(ds.get("Modality", "")),
                date=str(ds.get("StudyDate", "") or ""),
            )
            raw[uid] = []
        key = slice_key(ds)
        for f in range(int(ds.get("NumberOfFrames", 1) or 1)):
            raw[uid].append((key, f, str(p)))
    for uid, s in groups.items():
        s.items = [(path, f) for _, f, path in sorted(raw[uid])]
    return sorted(groups.values(), key=lambda s: s.number)


def decode_with_dcmtk(path):
    """Фолбэк: распаковка JPEG / JPEG-LS средствами DCMTK."""
    for tool in ("dcmdjpeg", "dcmdjpls"):
        exe = shutil.which(tool)
        if not exe:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.dcm")
            r = subprocess.run([exe, str(path), out], capture_output=True)
            if r.returncode == 0:
                ds = pydicom.dcmread(out)
                return ds, ds.pixel_array
    raise RuntimeError("Не удалось декодировать пиксели (нужен DCMTK, "
                       "либо pylibjpeg / python-gdcm для JPEG2000)")


def _first(v):
    if isinstance(v, (list, MultiValue)):
        v = v[0]
    return float(v)


def read_pixels(path):
    """Читает файл → (массив [кадры, h, w] float32 в реальных единицах, info)."""
    ds = pydicom.dcmread(path)
    try:
        arr = ds.pixel_array
    except Exception:
        ds, arr = decode_with_dcmtk(path)
    arr = np.asarray(arr)
    if int(ds.get("SamplesPerPixel", 1)) == 3:          # RGB -> яркость
        if arr.ndim == 3:
            arr = arr[None]
        arr = arr.astype(np.float32).mean(-1)
    elif arr.ndim == 2:
        arr = arr[None]
    slope = float(ds.get("RescaleSlope", 1) or 1)
    inter = float(ds.get("RescaleIntercept", 0) or 0)
    arr = arr.astype(np.float32) * slope + inter
    wc = ww = None
    try:
        wc, ww = _first(ds.WindowCenter), _first(ds.WindowWidth)
    except Exception:
        pass
    ps = ds.get("PixelSpacing") or ds.get("ImagerPixelSpacing")
    try:
        ps = (float(ps[0]), float(ps[1]))           # (между строк, между столбцов)
    except Exception:
        ps = None
    th = ds.get("SpacingBetweenSlices") or ds.get("SliceThickness")
    try:
        th = float(th)
    except Exception:
        th = None
    info = {"wc": wc, "ww": ww, "ps": ps, "th": th, "pos": slice_pos(ds),
            "modality": str(ds.get("Modality", "")),
            "mono1": ds.get("PhotometricInterpretation") == "MONOCHROME1"}
    return arr, info


# ───────────────────────────────── Объём ─────────────────────────────────
class Volume:
    """3D-массив (Z, Y, X) float32 + геометрия."""

    def __init__(self, data, spacing, z_pos, has_pos, calibrated, modality,
                 dicom_wl, mono1, warn=""):
        self.data = data
        self.calibrated = calibrated
        self.sz, self.sy, self.sx = spacing if calibrated else (1.0, 1.0, 1.0)
        self.z_pos = z_pos
        self.has_pos = has_pos
        self.modality = modality
        self.dicom_wl = dicom_wl
        self.mono1 = mono1
        self.warn = warn
        self.mask = None
        keys = [0.1, 1, 5, 95, 99, 99.9, 99.99]
        q = np.percentile(data[:, ::4, ::4], keys)
        self.pct = dict(zip(keys, map(float, q)))
        self.vmin, self.vmax = float(data.min()), float(data.max())

    def count(self, plane):
        return self.data.shape[plane]

    def spacing2d(self, plane):
        return {AX: (self.sy, self.sx), COR: (self.sz, self.sx),
                SAG: (self.sz, self.sy)}[plane]

    @staticmethod
    def cut(arr, plane, i):
        if plane == AX:
            return arr[i]
        if plane == COR:
            return arr[::-1, i, :]       # верх изображения = голова
        return arr[::-1, :, i]


def load_volume(series, progress=None):
    slices, pos, first = [], [], None
    shape, skipped = None, 0
    last_path, arr = None, None
    for n, (path, f) in enumerate(series.items):
        if progress and not progress(n):
            return None
        if path != last_path:
            arr, info = read_pixels(path)
            last_path = path
        sl = arr[min(f, arr.shape[0] - 1)]
        if shape is None:
            shape, first = sl.shape, info
        if sl.shape != shape:
            skipped += 1
            continue
        slices.append(sl)
        pos.append(info["pos"] if arr.shape[0] == 1 else None)
    if not slices:
        raise RuntimeError("В серии нет пригодных срезов")
    data = np.stack(slices).astype(np.float32, copy=False)
    n = len(slices)

    th = first["th"] or 1.0
    has_pos = n > 1 and all(p is not None for p in pos)
    if has_pos:
        z = np.array(pos, dtype=float)
        dz = float(np.median(np.abs(np.diff(z))))
        if dz < 1e-3:
            has_pos, dz, z = False, th, np.arange(n) * th
    else:
        dz, z = th, np.arange(n) * th

    ps = first["ps"]
    calibrated = ps is not None
    warn = []
    if skipped:
        warn.append(f"Пропущено срезов другого размера: {skipped}")
    if not calibrated:
        warn.append("Нет PixelSpacing — измерения в пикселях")
    dicom_wl = None
    if first["wc"] is not None and first["ww"]:
        dicom_wl = (first["wc"], first["ww"])
    return Volume(data, (dz, *(ps or (1.0, 1.0))), z, has_pos, calibrated,
                  first["modality"], dicom_wl, first["mono1"], "; ".join(warn))


def make_presets(vol):
    p = OrderedDict()
    if vol.dicom_wl:
        p["Из DICOM"] = vol.dicom_wl
    q = vol.pct

    def rng(a, b):
        return ((q[a] + q[b]) / 2, max(q[b] - q[a], 1.0))

    p["Авто (1–99%)"] = rng(1, 99)
    if vol.modality == "CT":
        p.update(CT_PRESETS)
    else:
        p["Контраст (5–95%)"] = rng(5, 95)
        p["Широкое (0.1–99.9%)"] = rng(0.1, 99.9)
    return p


# ─────────────────────────────── Измерения ───────────────────────────────
@dataclass
class Measure:
    kind: str
    pts: list
    text: list


def measure_text(kind, pts, arr, sp, calibrated, modality):
    """Текст измерения. pts — в координатах изображения (столбец, строка)."""
    sy, sx = sp
    unit = "мм" if calibrated else "px"
    ct = modality == "CT"
    h, w = arr.shape
    if kind == "dist":
        (x0, y0), (x1, y1) = pts
        return [f"{math.hypot((x1 - x0) * sx, (y1 - y0) * sy):.1f} {unit}"]
    if kind == "angle":
        a, b, c = pts
        v1 = np.array([(a[0] - b[0]) * sx, (a[1] - b[1]) * sy])
        v2 = np.array([(c[0] - b[0]) * sx, (c[1] - b[1]) * sy])
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        ang = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1, 1)))) if n > 0 else 0.0
        return [f"{ang:.1f}°"]
    if kind == "probe":
        x = min(max(int(pts[0][0]), 0), w - 1)
        y = min(max(int(pts[0][1]), 0), h - 1)
        v = float(arr[y, x])
        return [f"{v:.0f} HU" if ct else f"{v:.1f}"]
    if kind == "ellipse":
        (x0, y0), (x1, y1) = pts
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a, b = abs(x1 - x0) / 2, abs(y1 - y0) / 2
        if a < 0.5 or b < 0.5:
            return []
        xa, xb = max(int(cx - a), 0), min(int(cx + a) + 1, w)
        ya, yb = max(int(cy - b), 0), min(int(cy + b) + 1, h)
        yy, xx = np.mgrid[ya:yb, xa:xb]
        m = ((xx + 0.5 - cx) / a) ** 2 + ((yy + 0.5 - cy) / b) ** 2 <= 1
        vals = arr[ya:yb, xa:xb][m]
        if vals.size == 0:
            return []
        area = math.pi * a * sx * b * sy
        u = " HU" if ct else ""
        return [f"S = {area:.1f} {unit}²",
                f"μ = {vals.mean():.1f} ± {vals.std():.1f}{u}",
                f"min {vals.min():.0f} / max {vals.max():.0f}"]
    return []


def draw_text(p, x, y, s, color="#ffd54a"):
    p.setPen(QColor(0, 0, 0, 200))
    p.drawText(int(x) + 1, int(y) + 1, s)
    p.setPen(QColor(color))
    p.drawText(int(x), int(y), s)


# ───────────────────────────── 2D-вид среза ─────────────────────────────
class SliceView(QWidget):
    def __init__(self, panel, plane):
        super().__init__()
        self.panel, self.plane = panel, plane
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        self.zoom, self.pan = 1.0, QPoint(0, 0)
        self._img = self._buf = self._arr = None
        self._sp = (1.0, 1.0)
        self._draft, self._hover = [], None
        self._last = QPoint()

    def reset_view(self):
        self.zoom, self.pan = 1.0, QPoint(0, 0)
        self.update()

    # --- отрисовка среза с окном W/L
    def render(self):
        if self.isHidden():
            return
        p, vol = self.panel, self.panel.vol
        if vol is None:
            self._img = self._arr = None
            return self.update()
        idx = p.pos[self.plane]
        arr = Volume.cut(vol.data, self.plane, idx)
        self._arr = arr
        self._sp = vol.spacing2d(self.plane)
        lo = p.wc - p.ww / 2
        img = (np.clip((arr - lo) / max(p.ww, 1e-6), 0, 1) * 255).astype(np.uint8)
        if vol.mono1 != p.invert:
            img = 255 - img
        h, w = img.shape
        mk = None
        if vol.mask is not None and p.show_mask:
            mk = Volume.cut(vol.mask, self.plane, idx)
        if mk is not None and mk.any():
            rgb = np.stack([img] * 3, -1).astype(np.float32)
            rgb[mk] = rgb[mk] * 0.55 + np.array([255, 40, 40], np.float32) * 0.45
            self._buf = np.ascontiguousarray(rgb.astype(np.uint8))
            self._img = QImage(self._buf.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        else:
            self._buf = np.ascontiguousarray(img)
            self._img = QImage(self._buf.data, w, h, w, QImage.Format.Format_Grayscale8)
        self.update()

    def _geom(self):
        if self._img is None or self._arr is None:
            return None
        h, w = self._arr.shape
        sy, sx = self._sp
        pw, ph = w * sx, h * sy
        fit = min(self.width() / pw, self.height() / ph) * self.zoom   # px на мм
        dw, dh = pw * fit, ph * fit
        x0 = (self.width() - dw) / 2 + self.pan.x()
        y0 = (self.height() - dh) / 2 + self.pan.y()
        return x0, y0, dw, dh, fit

    def to_img(self, pt):
        x0, y0, dw, dh, _ = self._geom()
        h, w = self._arr.shape
        return (pt.x() - x0) / dw * w, (pt.y() - y0) / dh * h

    def to_wid(self, u, v):
        x0, y0, dw, dh, _ = self._geom()
        h, w = self._arr.shape
        return x0 + u / w * dw, y0 + v / h * dh

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        g = self._geom()
        if g is None:
            p.setPen(QColor("#888"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.panel.msg)
            return
        x0, y0, dw, dh, fit = g
        pan, vol = self.panel, self.panel.vol
        idx = pan.pos[self.plane]
        h, w = self._arr.shape
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(QRectF(x0, y0, dw, dh), self._img)

        if pan.multi:                                   # перекрестие MPR
            u, v = pan.cross_uv(self.plane)
            cx, cy = self.to_wid(u, v)
            p.setPen(QPen(QColor(255, 213, 74, 170), 1))
            p.drawLine(QPointF(x0, cy), QPointF(x0 + dw, cy))
            p.drawLine(QPointF(cx, y0), QPointF(cx, y0 + dh))

        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for m in pan.meas.get((self.plane, idx), []):
            self._draw_measure(p, m.kind, m.pts, m.text)
        if self._draft:
            kind = pan.tool
            pts = list(self._draft)
            if kind in ("dist", "angle") and self._hover:
                pts.append(self._hover)
            need = {"dist": 2, "angle": 3}.get(kind, len(pts))
            text = []
            if len(pts) == need and kind != "region":
                text = measure_text(kind, pts, self._arr, self._sp,
                                    vol.calibrated, vol.modality)
            self._draw_measure(p, kind, pts, text)

        # подписи
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        draw_text(p, 10, 20, PLANE_NAMES[self.plane])
        draw_text(p, 10, 36, pan.title, "#ffffff")
        n = vol.count(self.plane)
        draw_text(p, 10, self.height() - 48, f"Срез {idx + 1}/{n}")
        draw_text(p, 10, self.height() - 30,
                  f"W {pan.ww:.0f}  L {pan.wc:.0f}  zoom {self.zoom:.2f}")
        if self._hover:
            i, j = int(self._hover[1]), int(self._hover[0])
            if 0 <= i < h and 0 <= j < w:
                val = float(self._arr[i, j])
                unit = " HU" if vol.modality == "CT" else ""
                draw_text(p, 10, self.height() - 12, f"x={j} y={i}  {val:.1f}{unit}")
        if vol.calibrated:                              # масштабная линейка
            mm = 100 / fit
            nice = max([c for c in (1, 2, 5, 10, 20, 50, 100, 200) if c <= mm] or [1])
            L = nice * fit
            bx, by = self.width() - 20 - L, self.height() - 20
            p.setPen(QPen(QColor("#ffd54a"), 2))
            p.drawLine(QPointF(bx, by), QPointF(bx + L, by))
            draw_text(p, bx, by - 6, f"{nice} мм")

    def _draw_measure(self, p, kind, pts, text):
        if not pts:
            return
        wp = [QPointF(*self.to_wid(u, v)) for u, v in pts]
        p.setPen(QPen(QColor("#00e5ff"), 1.6))
        if kind in ("dist", "angle"):
            for a, b in zip(wp, wp[1:]):
                p.drawLine(a, b)
            for q in wp:
                p.drawEllipse(q, 3, 3)
            ax, ay = wp[-1].x() + 8, wp[-1].y() - 8
        elif kind == "ellipse" and len(wp) >= 2:
            r = QRectF(wp[0], wp[1]).normalized()
            p.drawEllipse(r)
            ax, ay = r.right() + 6, r.top() + 12
        else:                                           # probe / region
            q = wp[0]
            p.drawLine(QPointF(q.x() - 5, q.y()), QPointF(q.x() + 5, q.y()))
            p.drawLine(QPointF(q.x(), q.y() - 5), QPointF(q.x(), q.y() + 5))
            ax, ay = q.x() + 8, q.y() - 8
        for k, line in enumerate(text):
            draw_text(p, ax, ay + 14 * k, line, "#00e5ff")

    # --- ввод
    def wheelEvent(self, e):
        dy = e.angleDelta().y()
        if self.panel.vol is None or dy == 0:
            return
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            k = 1.1 if dy > 0 else 1 / 1.1
            self.zoom = min(20.0, max(0.1, self.zoom * k))
            self.update()
            self.panel.viewChanged.emit(self.plane)
        else:
            self.panel.scroll(self.plane, -1 if dy > 0 else 1)

    def keyPressEvent(self, e):
        k, pan = e.key(), self.panel
        t = e.text()
        if k in (Qt.Key.Key_Up, Qt.Key.Key_Left):
            pan.scroll(self.plane, -1)
        elif k in (Qt.Key.Key_Down, Qt.Key.Key_Right):
            pan.scroll(self.plane, 1)
        elif k == Qt.Key.Key_PageUp:
            pan.scroll(self.plane, -10)
        elif k == Qt.Key.Key_PageDown:
            pan.scroll(self.plane, 10)
        elif k in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            lst = pan.meas.get((self.plane, pan.pos[self.plane]))
            if lst:
                lst.pop()
                self.update()
        elif k == Qt.Key.Key_Escape:
            self._draft = []
            self.update()
        elif t and t in "123456789":
            pan.apply_preset(int(t) - 1)
        elif t.lower() == "i":
            pan.set_invert(not pan.invert)

    def _commit(self, kind, pts):
        vol = self.panel.vol
        text = measure_text(kind, pts, self._arr, self._sp, vol.calibrated, vol.modality)
        if text:
            self.panel.add_measure(self.plane, Measure(kind, list(pts), text))
        self.update()

    def mousePressEvent(self, e):
        self.setFocus()
        self.panel.activated.emit(self.panel)
        self._last = e.position().toPoint()
        if self._img is None:
            return
        if e.button() == Qt.MouseButton.RightButton and self._draft:
            self._draft = []
            return self.update()
        if e.button() != Qt.MouseButton.LeftButton:
            return
        pt = self.to_img(e.position())
        t, pan = self.panel.tool, self.panel
        if t == "cross":
            pan.set_cursor_from(self.plane, *pt)
        elif t in ("dist", "angle"):
            self._draft.append(pt)
            if len(self._draft) == (2 if t == "dist" else 3):
                self._commit(t, self._draft)
                self._draft = []
        elif t == "ellipse":
            self._draft = [pt, pt]
        elif t == "probe":
            self._commit("probe", [pt])
        elif t == "region":
            pan.region_grow(self.plane, pan.pos[self.plane], *pt)
        self.update()

    def mouseMoveEvent(self, e):
        pt = e.position()
        p = pt.toPoint()
        d = p - self._last
        self._last = p
        if self._img is None:
            return
        self._hover = self.to_img(pt)
        pan, btn = self.panel, e.buttons()
        if btn & Qt.MouseButton.LeftButton:
            t = pan.tool
            if t == "wl":
                k = max(pan.ww / 400.0, 0.05)
                pan.set_wl(pan.wc + d.y() * k, max(1.0, pan.ww + d.x() * k))
            elif t == "cross":
                pan.set_cursor_from(self.plane, *self._hover)
            elif t == "ellipse" and len(self._draft) == 2:
                self._draft[1] = self._hover
        elif btn & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
            self.pan += d
            pan.viewChanged.emit(self.plane)
        self.update()

    def mouseReleaseEvent(self, e):
        if (self.panel.tool == "ellipse" and len(self._draft) == 2
                and e.button() == Qt.MouseButton.LeftButton):
            (x0, y0), (x1, y1) = self._draft
            g = self._geom()
            if g and abs(x1 - x0) / self._arr.shape[1] * g[2] > 3:
                self._commit("ellipse", self._draft)
            self._draft = []
            self.update()

    def leaveEvent(self, _):
        self._hover = None
        self.update()

    def mouseDoubleClickEvent(self, _):
        self.panel.apply_preset(0)
        self.reset_view()
        self.panel.viewChanged.emit(self.plane)


# ─────────────────── 3D-визуализация (программный рейкастинг) ───────────────────
class Volume3D(QWidget):
    paramsChanged = Signal()

    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.setMinimumSize(200, 200)
        self.az, self.el, self.zoom = 25.0, 15.0, 1.0
        self.lo_f, self.hi_f, self.opacity, self.mode = 0.25, 1.0, 0.35, "Объём"
        self.rng = (0.0, 1.0)
        self.fast, self.hq = 72, 144
        self.presets = OrderedDict()
        self._img = self._buf = None
        self._last = QPoint()

    def set_volume(self, vol):
        r0, r1 = vol.pct[0.1], vol.pct[99.99]
        if r1 <= r0:
            r0, r1 = vol.vmin, vol.vmax + 1
        self.rng = (r0, r1)

        def f(hu):
            return float(np.clip((hu - r0) / (r1 - r0), 0, 1))

        if vol.modality == "CT":
            raw = [("Кости", 250, 1800, "Объём", 0.6),
                   ("Мягкие ткани", -100, 300, "Объём", 0.3),
                   ("Сосуды (КТА)", 150, 600, "Объём", 0.5),
                   ("Лёгкие", -900, -400, "Объём", 0.3)]
            self.presets = OrderedDict((n, (f(a), f(b), m, o)) for n, a, b, m, o in raw)
        else:
            self.presets = OrderedDict([
                ("Ткани", (0.25, 1.0, "Объём", 0.35)),
                ("Яркие структуры / MIP", (0.5, 1.0, "MIP", 1.0)),
                ("Мягкий порог", (0.12, 1.0, "Объём", 0.25))])
        self.az, self.el, self.zoom = 25.0, 15.0, 1.0
        self.apply_preset(next(iter(self.presets)))

    def apply_preset(self, name):
        if name not in self.presets:
            return
        self.lo_f, self.hi_f, self.mode, self.opacity = self.presets[name]
        self.paramsChanged.emit()
        self.render(self.hq)

    def render(self, N=None):
        vol = self.panel.vol
        if vol is None or self.isHidden():
            return
        N = N or self.hq
        Z, Y, X = vol.data.shape
        ext = np.array([X * vol.sx, Y * vol.sy, Z * vol.sz])        # x, y, z (мм)
        L = float(ext.max()) * 1.15 / self.zoom
        g = ((np.arange(N, dtype=np.float32) + 0.5) / N * L - L / 2)
        az, el = math.radians(self.az), math.radians(self.el)
        ca, sa, ce, se = math.cos(az), math.sin(az), math.cos(el), math.sin(el)
        Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
        B = np.column_stack([(1, 0, 0), (0, 0, -1), (0, 1, 0)])    # u вправо, v вниз, w вглубь
        A = Rz @ Rx @ B
        c = ext / 2

        def coord(k):
            return (float(c[k]) + g[None, None, :] * float(A[k, 0])
                    + g[None, :, None] * float(A[k, 1])
                    + g[:, None, None] * float(A[k, 2]))

        px, py, pz = coord(0), coord(1), coord(2)                   # массивы (w, v, u)
        valid = ((px >= 0) & (px < ext[0]) & (py >= 0) & (py < ext[1])
                 & (pz >= 0) & (pz < ext[2]))
        ix = np.clip(px / vol.sx, 0, X - 1).astype(np.int32)
        iy = np.clip(py / vol.sy, 0, Y - 1).astype(np.int32)
        iz = np.clip(pz / vol.sz, 0, Z - 1).astype(np.int32)
        del px, py, pz
        vals = vol.data[iz, iy, ix]
        vals[~valid] = vol.vmin
        del ix, iy, iz, valid

        r0, r1 = self.rng
        lo = r0 + self.lo_f * (r1 - r0)
        hi = r0 + max(self.hi_f, self.lo_f + 0.01) * (r1 - r0)
        t = np.clip((vals - lo) / max(hi - lo, 1e-6), 0, 1)
        del vals
        tint = np.array([1.0, 0.93, 0.82], np.float32)
        if self.mode == "MIP":
            m = t.max(axis=0)
            rgb = np.stack([m] * 3, -1)
        else:
            a = np.clip((t ** 1.5) * self.opacity, 0, 0.98)
            T = np.cumprod(1 - a, axis=0)
            Tp = np.empty_like(T)
            Tp[0] = 1
            Tp[1:] = T[:-1]
            wgt = Tp * a
            shade = (1 - 0.5 * np.arange(N, dtype=np.float32) / N)[:, None, None]
            C = (wgt * (0.35 + 0.65 * t) * shade).sum(0)
            rgb = C[..., None] * tint
        self._buf = np.ascontiguousarray(np.clip(rgb * 255, 0, 255).astype(np.uint8))
        self._img = QImage(self._buf.data, N, N, 3 * N, QImage.Format.Format_RGB888)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._img is None:
            p.setPen(QColor("#888"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "3D: загрузите серию")
            return
        s = min(self.width(), self.height())
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(QRectF((self.width() - s) / 2, (self.height() - s) / 2, s, s), self._img)
        draw_text(p, 10, 20, "3D — перетаскивание: вращение, колесо: масштаб, двойной клик: сброс")

    def mousePressEvent(self, e):
        self.panel.activated.emit(self.panel)
        self._last = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if not (e.buttons() & Qt.MouseButton.LeftButton) or self.panel.vol is None:
            return
        p = e.position().toPoint()
        d = p - self._last
        self._last = p
        self.az = (self.az + d.x() * 0.6) % 360
        self.el = max(-90.0, min(90.0, self.el + d.y() * 0.6))
        self.render(self.fast)

    def mouseReleaseEvent(self, _):
        self.render(self.hq)

    def wheelEvent(self, e):
        dy = e.angleDelta().y()
        if dy:
            self.zoom = min(4.0, max(0.3, self.zoom * (1.1 if dy > 0 else 1 / 1.1)))
            self.render(self.fast)
            self.render(self.hq)

    def mouseDoubleClickEvent(self, _):
        self.az, self.el, self.zoom = 25.0, 15.0, 1.0
        self.render(self.hq)


# ─────────────────────── Панель: один набор видов (A или B) ───────────────────────
class Panel(QFrame):
    sliceChanged = Signal()
    viewChanged = Signal(int)
    status = Signal(str)
    activated = Signal(object)
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        self.vol = None
        self.pos = [0, 0, 0]                       # (z, y, x)
        self.wc = self.ww = 1.0
        self.invert = False
        self.tool = "wl"
        self.meas = {}                             # (plane, idx) -> [Measure]
        self.title = ""
        self.presets = OrderedDict()
        self.show_mask, self.tol_pct, self.multi = True, 15, False
        self.mode = LAYOUTS[0]
        self.msg = "Откройте папку с DICOM или загрузите исследование из PACS"
        self.views = [SliceView(self, pl) for pl in (AX, COR, SAG)]
        self.v3d = Volume3D(self)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(2, 2, 2, 2)
        self.grid.setSpacing(2)
        for w in (*self.views, self.v3d):
            w.hide()
        self.set_active(False)
        self.set_layout(LAYOUTS[0])

    def set_active(self, flag):
        self.setStyleSheet("#panel{border:2px solid %s;}" % ("#ffd54a" if flag else "#222"))

    def set_layout(self, mode):
        self.mode = mode
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.hide()
        ax, co, sa = self.views

        def add(w, *a):
            self.grid.addWidget(w, *a)
            w.show()

        if mode == "Аксиальный":
            add(ax, 0, 0)
        elif mode == "Корональный":
            add(co, 0, 0)
        elif mode == "Сагиттальный":
            add(sa, 0, 0)
        elif mode == "MPR":
            add(ax, 0, 0, 2, 1); add(co, 0, 1); add(sa, 1, 1)
        elif mode == "MPR + 3D":
            add(ax, 0, 0); add(co, 0, 1); add(sa, 1, 0); add(self.v3d, 1, 1)
        else:
            add(self.v3d, 0, 0)
        for i in range(2):
            self.grid.setColumnStretch(i, 1)
            self.grid.setRowStretch(i, 1)
        self.multi = mode in ("MPR", "MPR + 3D")
        self.refresh()
        self.v3d.render()

    def set_volume(self, vol, title):
        self.vol, self.title = vol, title
        Z, Y, X = vol.data.shape
        self.pos = [Z // 2, Y // 2, X // 2]
        self.meas.clear()
        self.presets = make_presets(vol)
        self.invert = False
        self.apply_preset(0)
        for v in self.views:
            v.reset_view()
            v._draft = []
        self.v3d.set_volume(vol)
        self.refresh()
        self.changed.emit()
        self.sliceChanged.emit()

    def refresh(self):
        for v in self.views:
            v.render()

    # --- окно W/L
    def apply_preset(self, i):
        names = list(self.presets)
        if not (0 <= i < len(names)):
            return
        self.wc, self.ww = self.presets[names[i]]
        self.preset_index = i
        self.refresh()
        self.changed.emit()

    def set_wl(self, wc, ww):
        self.wc, self.ww = wc, ww
        self.refresh()

    def set_invert(self, flag):
        self.invert = flag
        self.refresh()
        self.changed.emit()

    # --- позиция курсора / срезы
    def set_pos(self, z=None, y=None, x=None):
        if self.vol is None:
            return
        Z, Y, X = self.vol.data.shape
        old = list(self.pos)
        for k, (v, n) in enumerate(((z, Z), (y, Y), (x, X))):
            if v is not None:
                self.pos[k] = max(0, min(n - 1, int(v)))
        if self.pos == old:
            return
        self.refresh()
        if self.pos[0] != old[0]:
            self.sliceChanged.emit()

    def scroll(self, plane, d):
        if self.vol is None:
            return
        kw = {("z", "y", "x")[plane]: self.pos[plane] + d}
        self.set_pos(**kw)

    def set_cursor_from(self, plane, u, v):
        Z = self.vol.data.shape[0]
        c, r = int(u), int(v)
        if plane == AX:
            self.set_pos(y=r, x=c)
        elif plane == COR:
            self.set_pos(z=Z - 1 - r, x=c)
        else:
            self.set_pos(z=Z - 1 - r, y=c)

    def cross_uv(self, plane):
        z, y, x = self.pos
        Z = self.vol.data.shape[0]
        if plane == AX:
            return x + 0.5, y + 0.5
        if plane == COR:
            return x + 0.5, Z - 1 - z + 0.5
        return y + 0.5, Z - 1 - z + 0.5

    # --- измерения и сегментация
    def add_measure(self, plane, m):
        self.meas.setdefault((plane, self.pos[plane]), []).append(m)

    def clear_measures(self):
        self.meas.clear()
        for v in self.views:
            v._draft = []
            v.update()

    def clear_mask(self):
        if self.vol is not None:
            self.vol.mask = None
            self.refresh()

    def region_grow(self, plane, idx, u, v):
        """Объём структуры: 3D-связная область с интенсивностью seed ± допуск."""
        if ndimage is None:
            return QMessageBox.warning(self, "Объём", "Для измерения объёма нужен scipy:\npip install scipy")
        vol = self.vol
        Z, Y, X = vol.data.shape
        c, r = int(u), int(v)
        z, y, x = {AX: (idx, r, c), COR: (Z - 1 - r, idx, c), SAG: (Z - 1 - r, c, idx)}[plane]
        if not (0 <= z < Z and 0 <= y < Y and 0 <= x < X):
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            nb = vol.data[max(z - 1, 0):z + 2, max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
            ref = float(np.median(nb))
            tol = self.ww * self.tol_pct / 100.0
            m = (vol.data >= ref - tol) & (vol.data <= ref + tol)
            lab, _ = ndimage.label(m)
            lbl = lab[z, y, x]
            if lbl == 0:
                return self.status.emit("Точка вне допуска — увеличьте допуск")
            comp = lab == lbl
            cnt = int(comp.sum())
        finally:
            QApplication.restoreOverrideCursor()
        vol.mask = comp
        self.show_mask = True
        if vol.calibrated:
            ml = cnt * vol.sz * vol.sy * vol.sx / 1000.0
            text = [f"V = {ml:.2f} мл", f"{cnt} вокс."]
        else:
            text = [f"V = {cnt} вокс."]
        self.add_measure(plane, Measure("region", [(u, v)], text))
        self.status.emit("Объём: " + ", ".join(text))
        self.refresh()


# ─────────────────────────────────── Окно ───────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI Viewer — DICOM / PACS / MPR / 3D")
        self.series, self.proc, self.find_dir = [], None, None
        self._lock = self._loading3d = False

        self.pA, self.pB = Panel(), Panel()
        self.panels = [self.pA, self.pB]
        self.active = self.pA
        for p in self.panels:
            p.activated.connect(self.set_active)
            p.status.connect(lambda s: self.statusBar().showMessage(s, 8000))
            p.changed.connect(lambda p=p: p is self.active and self._sync_toolbar())
            p.sliceChanged.connect(lambda p=p: self.sync_slice(p))
            p.viewChanged.connect(lambda pl, p=p: self.sync_view(p, pl))
            p.v3d.paramsChanged.connect(lambda p=p: p is self.active and self._load_3d_controls())
        self.pB.hide()
        self.pA.set_active(True)

        self._build_toolbar()

        tabs = QTabWidget()
        tabs.addTab(self._local_tab(), "Локально")
        tabs.addTab(self._pacs_tab(), "PACS")
        tabs.addTab(self._analysis_tab(), "3D / Анализ")

        self.series_list = QListWidget()
        self.series_list.itemActivated.connect(
            lambda _: self.load_series_into(self.active, self.series_list.currentRow()))
        row = QHBoxLayout()
        for text, panel in (("→ Окно A", self.pA), ("→ Окно B (сравнение)", self.pB)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, p=panel: self.send_to(p))
            row.addWidget(b)

        left = QWidget()
        lay = QVBoxLayout(left)
        lay.addWidget(tabs)
        lay.addWidget(QLabel("Серии (двойной клик — открыть в активном окне):"))
        lay.addWidget(self.series_list, 1)
        lay.addLayout(row)

        self.views_split = QSplitter()
        self.views_split.addWidget(self.pA)
        self.views_split.addWidget(self.pB)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.views_split)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 1000])
        self.setCentralWidget(split)
        self._sync_toolbar()

    # --- панель инструментов
    def _build_toolbar(self):
        tb = QToolBar("Инструменты")
        tb.setMovable(False)
        self.addToolBar(tb)
        grp = QActionGroup(self)
        grp.setExclusive(True)
        tools = [
            ("wl", "W/L", "Яркость/контраст: ЛКМ + движение (по X — ширина, по Y — центр)"),
            ("cross", "✚ Курсор", "Позиция перекрестия MPR"),
            ("dist", "Линейка", "Расстояние: два клика"),
            ("angle", "Угол", "Угол: три клика (вершина — второй)"),
            ("ellipse", "ROI", "Эллипс: площадь, среднее/СКО, min/max (HU для КТ)"),
            ("probe", "Значение", "Значение в точке (HU для КТ)"),
            ("region", "Объём", "Клик по структуре: 3D-сегментация и объём (нужен scipy)"),
        ]
        for key, text, tip in tools:
            a = QAction(text, self)
            a.setCheckable(True)
            a.setToolTip(tip)
            a.triggered.connect(lambda _=False, k=key: self.set_tool(k))
            grp.addAction(a)
            tb.addAction(a)
            if key == "wl":
                a.setChecked(True)
        tb.addSeparator()

        tb.addWidget(QLabel(" Раскладка: "))
        self.cmb_layout = QComboBox()
        self.cmb_layout.addItems(LAYOUTS)
        self.cmb_layout.currentTextChanged.connect(
            lambda t: [p.set_layout(t) for p in self.panels])
        tb.addWidget(self.cmb_layout)

        tb.addWidget(QLabel("  Окно: "))
        self.cmb_preset = QComboBox()
        self.cmb_preset.setMinimumWidth(150)
        self.cmb_preset.activated.connect(lambda i: self.active.apply_preset(i))
        tb.addWidget(self.cmb_preset)

        self.act_inv = QAction("Инверсия", self)
        self.act_inv.setCheckable(True)
        self.act_inv.triggered.connect(lambda c: self.active.set_invert(bool(c)))
        tb.addAction(self.act_inv)

        act_clear = QAction("Очистить измерения", self)
        act_clear.triggered.connect(lambda: [p.clear_measures() for p in self.panels])
        tb.addAction(act_clear)
        tb.addSeparator()

        self.act_cmp = QAction("Сравнение", self)
        self.act_cmp.setCheckable(True)
        self.act_cmp.setToolTip("Два окна рядом (например, МРТ за прошлый год и текущая)")
        self.act_cmp.triggered.connect(self.toggle_compare)
        tb.addAction(self.act_cmp)
        self.act_sync = QAction("Синхронизация", self)
        self.act_sync.setCheckable(True)
        self.act_sync.setChecked(True)
        tb.addAction(self.act_sync)
        self.cmb_sync = QComboBox()
        self.cmb_sync.addItems(["По положению (мм)", "Пропорционально"])
        tb.addWidget(self.cmb_sync)

    def set_tool(self, key):
        for p in self.panels:
            p.tool = key
            for v in p.views:
                v._draft = []
                v.update()

    def set_active(self, panel):
        if panel is self.active:
            return
        self.active = panel
        for p in self.panels:
            p.set_active(p is panel)
        self._sync_toolbar()

    def _sync_toolbar(self):
        p = self.active
        self.cmb_preset.blockSignals(True)
        self.cmb_preset.clear()
        self.cmb_preset.addItems(
            [f"{i + 1}. {n}" for i, n in enumerate(p.presets)])
        self.cmb_preset.setCurrentIndex(getattr(p, "preset_index", 0) if p.presets else -1)
        self.cmb_preset.blockSignals(False)
        self.act_inv.setChecked(p.invert)
        self._load_3d_controls()

    def toggle_compare(self, on):
        self.pB.setVisible(bool(on))
        if on:
            self.statusBar().showMessage(
                "Выберите серию и нажмите «→ Окно B» (папку для сравнения можно добавить во вкладке «Локально»)", 8000)
        elif self.active is self.pB:
            self.set_active(self.pA)

    def send_to(self, panel):
        if panel is self.pB and not self.act_cmp.isChecked():
            self.act_cmp.setChecked(True)
            self.toggle_compare(True)
        self.load_series_into(panel, self.series_list.currentRow())
        self.set_active(panel)

    # --- синхронизация сравниваемых окон
    def sync_slice(self, src):
        if self._lock or not self.act_sync.isChecked() or not self.pB.isVisible():
            return
        dst = self.pB if src is self.pA else self.pA
        sv, dv = src.vol, dst.vol
        if sv is None or dv is None:
            return
        i = src.pos[0]
        if self.cmb_sync.currentIndex() == 0 and sv.has_pos and dv.has_pos:
            j = int(np.argmin(np.abs(dv.z_pos - sv.z_pos[i])))
        else:
            j = round(i / max(len(sv.z_pos) - 1, 1) * (len(dv.z_pos) - 1))
        self._lock = True
        try:
            dst.set_pos(z=j)
        finally:
            self._lock = False

    def sync_view(self, src, plane):
        if self._lock or not self.act_sync.isChecked() or not self.pB.isVisible():
            return
        dst = self.pB if src is self.pA else self.pA
        s, d = src.views[plane], dst.views[plane]
        self._lock = True
        try:
            d.zoom, d.pan = s.zoom, QPoint(s.pan)
            d.update()
        finally:
            self._lock = False

    # --- вкладка «Локально»
    def _local_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        b1 = QPushButton("Открыть папку с DICOM…")
        b1.clicked.connect(self.open_folder)
        b2 = QPushButton("Добавить папку (для сравнения)…")
        b2.clicked.connect(lambda: self.open_folder(append=True))
        lay.addWidget(b1)
        lay.addWidget(b2)
        lay.addStretch(1)
        return w

    def open_folder(self, append=False):
        d = QFileDialog.getExistingDirectory(self, "Папка с DICOM")
        if d:
            self.load_folder(d, append=bool(append))

    def load_folder(self, folder, append=False):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            found = scan_folder(folder)
        finally:
            QApplication.restoreOverrideCursor()
        if not found:
            return QMessageBox.information(self, "DICOM", "DICOM-файлы не найдены")
        if append:
            self.series += found
        else:
            self.series = found
        self.series_list.clear()
        for s in self.series:
            self.series_list.addItem(s.label())
        row = len(self.series) - len(found)
        self.series_list.setCurrentRow(row)
        if not append:
            self.load_series_into(self.pA, row)

    def load_series_into(self, panel, row):
        if not (0 <= row < len(self.series)):
            return
        s = self.series[row]
        dlg = QProgressDialog("Загрузка серии…", "Отмена", 0, len(s.items), self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(300)

        def prog(n):
            dlg.setValue(n)
            QApplication.processEvents()
            return not dlg.wasCanceled()

        try:
            vol = load_volume(s, prog)
        except Exception as ex:
            dlg.close()
            return QMessageBox.critical(self, "DICOM", f"Ошибка: {ex}")
        dlg.close()
        if vol is None:
            return
        panel.set_volume(vol, f"{s.patient} | {s.label()}")
        if vol.warn:
            self.statusBar().showMessage(vol.warn, 10000)

    # --- вкладка «3D / Анализ»
    def _analysis_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.sp_tol = QSpinBox()
        self.sp_tol.setRange(1, 100)
        self.sp_tol.setValue(15)
        self.sp_tol.setSuffix(" % от W")
        self.sp_tol.valueChanged.connect(lambda v: [setattr(p, "tol_pct", v) for p in self.panels])
        form.addRow("Допуск сегментации", self.sp_tol)
        lay.addLayout(form)
        chk = QCheckBox("Показывать сегментацию")
        chk.setChecked(True)
        chk.toggled.connect(self._toggle_mask)
        lay.addWidget(chk)
        b = QPushButton("Сбросить сегментацию")
        b.clicked.connect(lambda: [p.clear_mask() for p in self.panels])
        lay.addWidget(b)
        lay.addWidget(QLabel("<b>3D-визуализация</b>"))

        f3 = QFormLayout()
        self.cmb3_preset = QComboBox()
        self.cmb3_preset.activated.connect(
            lambda i: self.active.v3d.apply_preset(self.cmb3_preset.itemText(i)))
        self.cmb3_mode = QComboBox()
        self.cmb3_mode.addItems(["Объём", "MIP"])
        self.cmb3_mode.currentTextChanged.connect(lambda _: self._3d_changed(False))

        def mk(a, b_, v):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(a, b_)
            s.setValue(v)
            s.valueChanged.connect(lambda _: self._3d_changed(True))
            s.sliderReleased.connect(lambda: self._3d_changed(False))
            return s

        self.sl_lo, self.sl_hi, self.sl_op = mk(0, 1000, 250), mk(0, 1000, 1000), mk(1, 100, 35)
        f3.addRow("Пресет", self.cmb3_preset)
        f3.addRow("Режим", self.cmb3_mode)
        f3.addRow("Нижний порог", self.sl_lo)
        f3.addRow("Верхний порог", self.sl_hi)
        f3.addRow("Непрозрачность", self.sl_op)
        lay.addLayout(f3)
        lay.addStretch(1)
        return w

    def _toggle_mask(self, on):
        for p in self.panels:
            p.show_mask = on
            p.refresh()

    def _load_3d_controls(self):
        v = self.active.v3d
        self._loading3d = True
        try:
            self.cmb3_preset.clear()
            self.cmb3_preset.addItems(list(v.presets))
            self.cmb3_mode.setCurrentText(v.mode)
            self.sl_lo.setValue(int(v.lo_f * 1000))
            self.sl_hi.setValue(int(v.hi_f * 1000))
            self.sl_op.setValue(int(v.opacity * 100))
        finally:
            self._loading3d = False

    def _3d_changed(self, fast):
        if self._loading3d:
            return
        v = self.active.v3d
        v.lo_f = self.sl_lo.value() / 1000
        v.hi_f = self.sl_hi.value() / 1000
        v.opacity = self.sl_op.value() / 100
        v.mode = self.cmb3_mode.currentText()
        v.render(v.fast if fast else v.hq)

    # --- вкладка PACS
    def _pacs_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.host = QLineEdit("127.0.0.1")
        self.port = QSpinBox(); self.port.setRange(1, 65535); self.port.setValue(104)
        self.called = QLineEdit("ORTHANC")        # AE Title PACS
        self.mine = QLineEdit("MRIVIEWER")        # наш AE Title
        self.rport = QSpinBox(); self.rport.setRange(1, 65535); self.rport.setValue(11112)
        self.f_name = QLineEdit("*")
        self.f_id = QLineEdit()
        self.f_mod = QLineEdit("MR")
        form.addRow("Хост PACS", self.host)
        form.addRow("Порт PACS", self.port)
        form.addRow("AE PACS (called)", self.called)
        form.addRow("Наш AE (calling)", self.mine)
        form.addRow("Наш порт приёма", self.rport)
        form.addRow("Имя пациента", self.f_name)
        form.addRow("ID пациента", self.f_id)
        form.addRow("Модальность", self.f_mod)
        lay.addLayout(form)

        row = QHBoxLayout()
        b1 = QPushButton("Найти (C-FIND)")
        b2 = QPushButton("Загрузить (C-MOVE)")
        b1.clicked.connect(self.pacs_find)
        b2.clicked.connect(self.pacs_get)
        row.addWidget(b1)
        row.addWidget(b2)
        lay.addLayout(row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Пациент", "ID", "Дата", "Описание", "Мод."])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table, 1)

        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumHeight(120)
        lay.addWidget(self.log)
        return w

    def run_dcmtk(self, tool, args, on_done):
        if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
            return QMessageBox.information(self, "DCMTK", "Предыдущая операция ещё идёт")
        exe = shutil.which(tool)
        if not exe:
            return QMessageBox.critical(
                self, "DCMTK", f"Не найден «{tool}». Установите DCMTK и добавьте в PATH.")
        self.log.appendPlainText(f"$ {tool} {' '.join(args)}")
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyRead.connect(lambda: self.log.appendPlainText(
            bytes(self.proc.readAll()).decode(errors="replace").rstrip()))
        self.proc.finished.connect(lambda code, _s: on_done(code))
        self.proc.start(exe, args)

    def _assoc_args(self):
        return ["-aet", self.mine.text(), "-aec", self.called.text()]

    def pacs_find(self):
        if self.find_dir:
            shutil.rmtree(self.find_dir, ignore_errors=True)
        self.find_dir = tempfile.mkdtemp(prefix="findscu_")
        keys = ["QueryRetrieveLevel=STUDY",
                f"PatientName={self.f_name.text() or '*'}",
                f"PatientID={self.f_id.text()}" if self.f_id.text() else "PatientID",
                "StudyDate", "StudyDescription",
                f"ModalitiesInStudy={self.f_mod.text()}" if self.f_mod.text()
                else "ModalitiesInStudy",
                "StudyInstanceUID"]
        args = ["-S", *self._assoc_args()]
        for k in keys:
            args += ["-k", k]
        args += ["-X", "-od", self.find_dir, self.host.text(), str(self.port.value())]
        self.run_dcmtk("findscu", args, self._find_done)

    def _find_done(self, code):
        self.table.setRowCount(0)
        for f in sorted(Path(self.find_dir).glob("rsp*.dcm")):
            try:
                ds = pydicom.dcmread(f, force=True)
            except Exception:
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            vals = [str(ds.get("PatientName", "")), str(ds.get("PatientID", "")),
                    str(ds.get("StudyDate", "")), str(ds.get("StudyDescription", "")),
                    str(ds.get("ModalitiesInStudy", ""))]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 0:
                    it.setData(Qt.ItemDataRole.UserRole, str(ds.get("StudyInstanceUID", "")))
                self.table.setItem(r, c, it)
        self.log.appendPlainText(f"C-FIND завершён (код {code}), найдено: {self.table.rowCount()}")

    def pacs_get(self):
        row = self.table.currentRow()
        if row < 0:
            return QMessageBox.information(self, "PACS", "Выберите исследование в таблице")
        uid = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        out = Path.home() / "dicom_cache" / uid
        out.mkdir(parents=True, exist_ok=True)
        args = ["-S", *self._assoc_args(), "-aem", self.mine.text(),
                "+P", str(self.rport.value()),
                "-k", "QueryRetrieveLevel=STUDY", "-k", f"StudyInstanceUID={uid}",
                "-od", str(out), self.host.text(), str(self.port.value())]
        self.run_dcmtk("movescu", args, lambda code: self._move_done(code, out))

    def _move_done(self, code, out):
        self.log.appendPlainText(f"C-MOVE завершён (код {code}) -> {out}")
        if code == 0:
            self.load_folder(out, append=self.act_cmp.isChecked() and bool(self.series))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.resize(1500, 900)
    w.show()
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):   # python mri_viewer.py <папка>
        w.load_folder(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()