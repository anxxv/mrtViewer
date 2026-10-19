import atexit
import csv
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
from pydicom.multival import MultiValue
from PySide6.QtCore import QMarginsF, QPoint, QPointF, QProcess, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction, QActionGroup, QColor, QFont, QImage, QPageLayout, QPageSize,
    QPainter, QPdfWriter, QPen, QPolygonF,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
    QFileDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressDialog, QPushButton, QSlider, QSpinBox, QSplitter, QTableWidget,
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

# ───────────── Параметры памяти и выравнивания ─────────────
# Серии крупнее этого порога хранятся не в оперативной памяти, а во
# временном файле на диске (np.memmap) и подгружаются по мере надобности.
MEM_LIMIT = int(os.environ.get("MRT_MEM_LIMIT_MB", "256")) * 1024 * 1024
# Максимальный размер выровненного объёма (в вокселях).
MAX_ALIGN_VOX = int(float(os.environ.get("MRT_ALIGN_MAX_MVOX", "160")) * 1_000_000)
# Срезы считаются наклонными, если отклонение от осей пациента больше (градусов).
OBLIQUE_DEG = 1.0
CACHE_DIR = Path(tempfile.gettempdir()) / "mrtviewer_cache"
_cache_ids = itertools.count()
_live_cache = set()


def _drop_cache_file(path):
    if path is None:
        return
    try:
        Path(path).unlink()
    except Exception:
        pass
    _live_cache.discard(str(path))


def _cleanup_cache():
    for p in list(_live_cache):
        _drop_cache_file(p)


def _purge_stale_cache(max_age=24 * 3600):
    """Удаляет забытые временные файлы прошлых запусков."""
    try:
        now = time.time()
        for f in CACHE_DIR.glob("*.npy"):
            if now - f.stat().st_mtime > max_age:
                f.unlink()
    except Exception:
        pass


atexit.register(_cleanup_cache)


def alloc_array(shape, dtype=np.float32, force_disk=False):
    """Выделяет массив: в памяти (малый) или на диске через memmap (большой)."""
    nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if force_disk or nbytes > MEM_LIMIT:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{os.getpid()}_{next(_cache_ids)}.npy"
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=tuple(shape))
        _live_cache.add(str(path))
        return arr, path
    return np.empty(shape, dtype), None


# ──────────────────────────── Серии и файлы DICOM ────────────────────────────
@dataclass
class Item:
    """Один кадр (срез) серии."""
    path: str
    frame: int = 0
    sop: str = ""
    pos: float = None          # положение вдоль нормали, мм
    ipp: tuple = None          # ImagePositionPatient
    iop: tuple = None          # ImageOrientationPatient
    key: float = 0.0           # ключ сортировки
    inst: int = 0
    temporal: int = None       # TemporalPositionIdentifier / Index
    acq: int = None
    trig: float = None         # TriggerTime, мс
    atime: str = ""


@dataclass
class Series:
    uid: str
    patient: str
    desc: str
    number: int
    modality: str
    items: list = field(default_factory=list)   # [Item], отсортированы по положению
    date: str = ""
    phases: list = field(default_factory=list)  # [[Item]] — динамические серии
    phase_names: list = field(default_factory=list)

    @property
    def nphases(self):
        return max(len(self.phases), 1)

    def label(self):
        d = self.date
        if len(d) == 8:
            d = f"{d[6:]}.{d[4:6]}.{d[:4]}"
        if self.nphases > 1:
            cnt = f"{len(self.phases[0])} ср., {self.nphases} фаз"
        else:
            cnt = str(len(self.items))
        s = f"#{self.number} {self.modality} {self.desc} ({cnt})"
        return f"{s} · {d}" if d else s


@dataclass
class DicomObject:
    """Объект без пикселей изображения: SR, PR (аннотации), SEG (сегментация)."""
    path: str
    kind: str                  # "SR" | "PR" | "SEG"
    desc: str = ""
    number: int = 0
    date: str = ""
    sop: str = ""
    info: dict = None          # результат разбора (кэш)

    def label(self):
        names = {"SR": "Отчёт SR", "PR": "Аннотации PR", "SEG": "Сегментация SEG"}
        d = self.date
        if len(d) == 8:
            d = f"{d[6:]}.{d[4:6]}.{d[:4]}"
        s = f"[{names.get(self.kind, self.kind)}] {self.desc or Path(self.path).name}"
        return f"{s} · {d}" if d else s


def _normal_pos(iop, ipp):
    try:
        iop = np.asarray(iop, float)
        return float(np.dot(np.asarray(ipp, float), np.cross(iop[:3], iop[3:])))
    except Exception:
        return None


def slice_pos(ds):
    """Позиция среза вдоль нормали (мм) или None."""
    try:
        return _normal_pos(ds.ImageOrientationPatient, ds.ImagePositionPatient)
    except Exception:
        return None


def slice_key(ds):
    p = slice_pos(ds)
    return p if p is not None else float(ds.get("InstanceNumber", 0) or 0)


def _fg(pf, sh, seq, attr):
    """Значение из функциональных групп многокадрового файла."""
    for grp in (pf, sh):
        try:
            if grp is not None and seq in grp:
                return grp[seq][0][attr].value
        except Exception:
            continue
    return None


def _vec(v):
    try:
        return tuple(float(x) for x in v)
    except Exception:
        return None


def frame_geometry(ds, f, nf):
    """(IOP, IPP, временной индекс) кадра f."""
    if nf <= 1 or "PerFrameFunctionalGroupsSequence" not in ds:
        temporal = ds.get("TemporalPositionIdentifier")
        return (_vec(ds.get("ImageOrientationPatient")),
                _vec(ds.get("ImagePositionPatient")),
                int(temporal) if temporal not in (None, "") else None)
    try:
        pf = ds.PerFrameFunctionalGroupsSequence[f]
    except Exception:
        return None, None, None
    shared = ds.get("SharedFunctionalGroupsSequence")
    sh = shared[0] if shared else None
    iop = _vec(_fg(pf, sh, "PlaneOrientationSequence", "ImageOrientationPatient"))
    ipp = _vec(_fg(pf, sh, "PlanePositionSequence", "ImagePositionPatient"))
    t = _fg(pf, sh, "FrameContentSequence", "TemporalPositionIndex")
    return iop, ipp, (int(t) if t not in (None, "") else None)


def _to_int(v):
    try:
        return int(v)
    except Exception:
        return None


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None


SCAN_TAGS = ["SeriesInstanceUID", "SOPInstanceUID", "SOPClassUID", "PatientName",
             "SeriesDescription", "SeriesNumber", "Modality",
             "ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber",
             "NumberOfFrames", "StudyDate", "Rows", "TemporalPositionIdentifier",
             "AcquisitionNumber", "TriggerTime", "AcquisitionTime", "ContentTime",
             "PerFrameFunctionalGroupsSequence", "SharedFunctionalGroupsSequence"]


def _kind_of(ds):
    """Тип объекта без пикселей: SR / PR / SEG, иначе None (изображение)."""
    mod = str(ds.get("Modality", ""))
    cls = str(ds.get("SOPClassUID", ""))
    if mod == "SR" or cls.startswith("1.2.840.10008.5.1.4.1.1.88."):
        return "SR"
    if mod == "PR" or cls.startswith("1.2.840.10008.5.1.4.1.1.11."):
        return "PR"
    if mod == "SEG" or cls == "1.2.840.10008.5.1.4.1.1.66.4":
        return "SEG"
    return None


def split_phases(items):
    """Делит серию на фазы (динамика, несколько временных точек, эхо и т. п.).

    1) по TemporalPositionIdentifier / TemporalPositionIndex;
    2) если на каждом положении среза лежит одинаковое число (>1) снимков —
       по порядку (время запуска, время съёмки, номер съёмки, номер снимка).
    """
    def order(it):
        return (it.key, it.frame, it.path)

    temporal = sorted({it.temporal for it in items if it.temporal is not None})
    if len(temporal) > 1 and all(it.temporal is not None for it in items):
        phases = [sorted([it for it in items if it.temporal == t], key=order)
                  for t in temporal]
        return phases, _phase_names(phases)

    if items and all(it.pos is not None for it in items):
        groups = {}
        for it in items:
            groups.setdefault(round(it.pos, 2), []).append(it)
        mult = {len(g) for g in groups.values()}
        if len(mult) == 1 and next(iter(mult)) > 1:
            m = next(iter(mult))
            phases = [[] for _ in range(m)]

            def tkey(it):
                return (it.trig if it.trig is not None else 0.0, it.atime,
                        it.acq if it.acq is not None else 0, it.inst, it.path, it.frame)

            for g in groups.values():
                for rank, it in enumerate(sorted(g, key=tkey)):
                    phases[rank].append(it)
            phases = [sorted(ph, key=order) for ph in phases]
            return phases, _phase_names(phases)
    return [sorted(items, key=order)], [""]


def _phase_names(phases):
    names = []
    for i, ph in enumerate(phases):
        it = ph[0]
        if it.trig is not None:
            names.append(f"{it.trig:.0f} мс")
        elif it.atime:
            t = str(it.atime).split(".")[0].ljust(6, "0")
            names.append(f"{t[:2]}:{t[2:4]}:{t[4:6]}")
        else:
            names.append("")
    return names


def scan_folder(folder):
    """Рекурсивно сканирует папку. Возвращает (серии изображений, объекты SR/PR/SEG)."""
    groups, raw, objects = {}, {}, []
    for p in Path(folder).rglob("*"):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, specific_tags=SCAN_TAGS)
        except Exception:
            continue
        uid = ds.get("SeriesInstanceUID")
        if not uid:
            continue
        uid = str(uid)
        kind = _kind_of(ds)
        if kind:
            objects.append(DicomObject(
                path=str(p), kind=kind, desc=str(ds.get("SeriesDescription", "") or ""),
                number=int(ds.get("SeriesNumber", 0) or 0),
                date=str(ds.get("StudyDate", "") or ""),
                sop=str(ds.get("SOPInstanceUID", ""))))
            continue
        if "Rows" not in ds:                       # не изображение
            continue
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
        nf = int(ds.get("NumberOfFrames", 1) or 1)
        sop = str(ds.get("SOPInstanceUID", ""))
        inst = int(ds.get("InstanceNumber", 0) or 0)
        acq = _to_int(ds.get("AcquisitionNumber"))
        trig = _to_float(ds.get("TriggerTime"))
        atime = str(ds.get("AcquisitionTime", "") or ds.get("ContentTime", "") or "")
        for f in range(nf):
            iop, ipp, temporal = frame_geometry(ds, f, nf)
            pos = _normal_pos(iop, ipp) if (iop and ipp) else None
            raw[uid].append(Item(
                path=str(p), frame=f, sop=sop, pos=pos, ipp=ipp, iop=iop,
                key=pos if pos is not None else float(inst), inst=inst,
                temporal=temporal, acq=acq, trig=trig, atime=atime))
    for uid, s in groups.items():
        s.items = sorted(raw[uid], key=lambda it: (it.key, it.frame, it.path))
        s.phases, s.phase_names = split_phases(raw[uid])
    return sorted(groups.values(), key=lambda s: s.number), objects


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
    """Читает файл → (сырой массив [кадры, h, w(, 3)], info).

    Массив не переводится в float32 целиком (экономия памяти): перевод в
    реальные единицы делается по одному кадру функцией frame_to_float().
    """
    ds = pydicom.dcmread(path)
    try:
        arr = ds.pixel_array
    except Exception:
        ds, arr = decode_with_dcmtk(path)
    arr = np.asarray(arr)
    rgb = int(ds.get("SamplesPerPixel", 1)) == 3
    if rgb:
        if arr.ndim == 3:
            arr = arr[None]
    elif arr.ndim == 2:
        arr = arr[None]
    slope = float(ds.get("RescaleSlope", 1) or 1)
    inter = float(ds.get("RescaleIntercept", 0) or 0)
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
        try:                                        # многокадровые файлы
            shared = ds.SharedFunctionalGroupsSequence[0]
            v = shared.PixelMeasuresSequence[0].PixelSpacing
            ps = (float(v[0]), float(v[1]))
        except Exception:
            pass
    th = ds.get("SpacingBetweenSlices") or ds.get("SliceThickness")
    try:
        th = float(th)
    except Exception:
        th = None
        try:
            pm = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
            th = float(pm.get("SpacingBetweenSlices") or pm.get("SliceThickness"))
        except Exception:
            pass
    info = {"wc": wc, "ww": ww, "ps": ps, "th": th, "slope": slope, "inter": inter,
            "rgb": rgb, "modality": str(ds.get("Modality", "")),
            "mono1": ds.get("PhotometricInterpretation") == "MONOCHROME1"}
    return arr, info


def frame_to_float(arr, f, info):
    """Один кадр → float32 в реальных единицах (HU для КТ)."""
    fr = arr[min(f, arr.shape[0] - 1)]
    if info["rgb"]:                                  # RGB -> яркость
        fr = fr.astype(np.float32).mean(-1)
    return fr.astype(np.float32) * info["slope"] + info["inter"]


# ───────────────────────────── Геометрия и выравнивание ─────────────────────────────
def _axis_dev(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        return 0.0
    return math.degrees(math.acos(min(1.0, float(np.max(np.abs(v / n))))))


def _slope(ipps):
    """Шаг между срезами (вектор, мм на срез) — линейная регрессия IPP по номеру."""
    P = np.asarray(ipps, float)
    k = np.arange(len(P), dtype=float)
    kc = k - k.mean()
    return (kc[:, None] * (P - P.mean(0))).sum(0) / (kc ** 2).sum()


def obliquity(iop, ipps):
    """Наклон серии в градусах: отклонение осей от осей пациента или наклон гентри."""
    if iop is None:
        return 0.0
    fr, fc = np.asarray(iop[:3], float), np.asarray(iop[3:], float)
    n = np.cross(fr, fc)
    dev = max(_axis_dev(fr), _axis_dev(fc), _axis_dev(n))
    tilt = 0.0
    if ipps is not None and len(ipps) > 1:
        s = _slope(ipps)
        sn = float(np.dot(s, n))
        inpl = s - sn * n
        tilt = math.degrees(math.atan2(float(np.linalg.norm(inpl)), abs(sn)))
    return max(dev, tilt)


def build_affine(iop, ipps, sx, sy, dz):
    """Матрица 4×4: индекс (столбец, строка, срез, 1) → координаты пациента (мм, LPS)."""
    fr, fc = np.asarray(iop[:3], float), np.asarray(iop[3:], float)
    n = np.cross(fr, fc)
    v = dz * n
    if ipps is not None and len(ipps) > 1:
        s = _slope(ipps)
        sn = float(np.dot(s, n))
        if abs(sn) > 1e-6:
            v = s * (dz / sn)                       # сохраняем наклон гентри (сдвиг)
    A = np.eye(4)
    A[:3, 0], A[:3, 1], A[:3, 2] = sx * fr, sy * fc, v
    A[:3, 3] = np.asarray(ipps[0], float)
    return A


# ───────────────────────────────── Объём ─────────────────────────────────
class Volume:
    """3D-массив (Z, Y, X) float32 + геометрия. Может лежать в памяти или на диске."""

    def __init__(self, data, spacing, z_pos, has_pos, calibrated, modality,
                 dicom_wl, mono1, warn="", stats=None, affine=None, cache_path=None):
        self.data = data
        self.calibrated = calibrated
        self.sz, self.sy, self.sx = spacing if calibrated else (1.0, 1.0, 1.0)
        self.z_pos = z_pos
        self.has_pos = has_pos
        self.modality = modality
        self.dicom_wl = dicom_wl
        self.mono1 = mono1
        self.warn = warn
        self.mask = None                 # маска «Объём» (bool, Z×Y×X)
        self.seg = None                  # метки DICOM SEG (uint8, Z×Y×X)
        self.seg_lut = None              # палитра (256, 3)
        self.seg_names = {}              # номер сегмента → название
        self.annots = {}                 # z → [Annot] (PR / SR)
        self.series_uid = ""
        self.phase = 0
        self.sop_index = {}              # (SOP UID, кадр) → индекс исходного среза
        self.src_shape = data.shape      # форма до выравнивания
        self.A = self.A_src = affine     # индекс → координаты пациента
        self.aligned = False
        self.obl_deg = 0.0
        self.warp = None                 # функция переноса массива в выровненную сетку
        self.cache_path = cache_path
        if stats is None:
            keys = [0.1, 1, 5, 95, 99, 99.9, 99.99]
            q = np.percentile(data[:, ::4, ::4], keys)
            stats = (dict(zip(keys, map(float, q))), float(data.min()), float(data.max()))
        self.pct, self.vmin, self.vmax = stats

    def count(self, plane):
        return self.data.shape[plane]

    def spacing2d(self, plane):
        return {AX: (self.sy, self.sx), COR: (self.sz, self.sx),
                SAG: (self.sz, self.sy)}[plane]

    def nbytes(self):
        return int(self.data.size) * 4

    def on_disk(self):
        return self.cache_path is not None

    def src_to_cur(self, x, y, k):
        """Индекс исходного среза (непрерывный) → индекс в текущем объёме."""
        if not self.aligned or self.A_src is None or self.A is None:
            return x, y, k
        w = self.A_src @ np.array([x, y, k, 1.0])
        c = np.linalg.solve(self.A, w)
        return float(c[0]), float(c[1]), float(c[2])

    def release(self):
        """Освобождает память и удаляет временный файл."""
        self.data = None
        self.mask = self.seg = None
        _drop_cache_file(self.cache_path)
        self.cache_path = None

    @staticmethod
    def cut(arr, plane, i):
        if plane == AX:
            return arr[i]
        if plane == COR:
            return arr[::-1, i, :]       # верх изображения = голова
        return arr[::-1, :, i]


def align_volume(vol, progress=None, base=0, disk=False):
    """Пересчитывает наклонную серию в сетку, выровненную по осям пациента.

    Сетка: x — влево, y — назад, z — к голове (LPS). Интерполяция линейная.
    """
    if ndimage is None:
        raise RuntimeError("Для выравнивания нужен scipy: pip install scipy")
    Z, Y, X = vol.data.shape
    A = vol.A_src
    corners = np.array([[i, j, k, 1.0] for i in (0, X - 1) for j in (0, Y - 1)
                        for k in (0, Z - 1)]).T
    world = (A @ corners)[:3]
    lo, hi = world.min(1), world.max(1)
    h = np.array([min(vol.sx, vol.sy), min(vol.sx, vol.sy), vol.sz], float)
    ext = hi - lo
    dims = np.ceil(ext / h).astype(np.int64) + 1
    total = int(np.prod(dims))
    if total > MAX_ALIGN_VOX:
        h = h * (total / MAX_ALIGN_VOX) ** (1 / 3)
        dims = np.ceil(ext / h).astype(np.int64) + 1
    nx, ny, nz = (int(d) for d in dims)
    out, path = alloc_array((nz, ny, nx), force_disk=disk)

    R = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float)   # (x,y,z) ↔ (z,y,x)
    M = R @ np.linalg.inv(A[:3, :3]) @ np.diag(h) @ R
    off0 = R @ np.linalg.inv(A[:3, :3]) @ (lo - A[:3, 3])
    chunk = max(1, int(4_000_000 // (ny * nx)))
    nchunks = (nz + chunk - 1) // chunk
    cval = float(vol.vmin)

    def transform(src, order, cval_, dtype, dst, cb=None):
        for c, z0 in enumerate(range(0, nz, chunk)):
            z1 = min(nz, z0 + chunk)
            if cb and not cb(c):
                return False
            dst[z0:z1] = ndimage.affine_transform(
                src, M, offset=off0 + M[:, 0] * z0, output_shape=(z1 - z0, ny, nx),
                output=dtype, order=order, mode="constant", cval=cval_, prefilter=False)
        return True

    def cb(c):
        return progress(base + c, base + nchunks) if progress else True

    if not transform(vol.data, 1, cval, np.float32, out, cb):
        _drop_cache_file(path)
        return None
    if path is not None:
        out.flush()

    An = np.eye(4)
    An[0, 0], An[1, 1], An[2, 2] = h
    An[:3, 3] = lo
    z_pos = lo[2] + np.arange(nz) * h[2]
    new = Volume(out, (float(h[2]), float(h[1]), float(h[0])), z_pos, True, True,
                 vol.modality, vol.dicom_wl, vol.mono1, vol.warn,
                 stats=(vol.pct, vol.vmin, vol.vmax), affine=An, cache_path=path)
    new.A_src, new.aligned, new.src_shape = A, True, vol.src_shape
    new.series_uid, new.phase, new.sop_index = vol.series_uid, vol.phase, vol.sop_index
    new.obl_deg = vol.obl_deg

    def warp(src, order=0):
        dst = np.zeros((nz, ny, nx), src.dtype)
        transform(src, order, 0, src.dtype, dst)
        return dst

    new.warp = warp
    return new


def load_volume(series, progress=None, phase=0, align=True):
    """Собирает 3D-объём серии (или одной её фазы).

    progress(n, всего) -> bool: вернуть False для отмены.
    Крупные серии пишутся во временный memmap-файл, а не держатся в памяти.
    """
    items = series.phases[phase] if series.phases else series.items
    if not items:
        raise RuntimeError("В серии нет срезов")
    n_items = len(items)
    iop0 = next((it.iop for it in items if it.iop), None)
    ipps = [it.ipp for it in items]
    have_ipp = iop0 is not None and all(p is not None for p in ipps)
    obl = obliquity(iop0, ipps) if (have_ipp and n_items > 1) else 0.0

    kept, data, cache, shape, first = [], None, None, None, None
    last_path, raw, info, skipped = None, None, None, 0
    vmin, vmax, samples = math.inf, -math.inf, []
    disk_all = series.nphases > 1            # динамику держим на диске: быстрее листать фазы
    for n, it in enumerate(items):
        if progress and not progress(n, n_items):
            if data is not None:
                del data
            _drop_cache_file(cache)
            return None
        if it.path != last_path:
            raw, info = read_pixels(it.path)
            last_path = it.path
        sl = frame_to_float(raw, it.frame, info)
        if shape is None:
            shape, first = sl.shape, info
            data, cache = alloc_array((n_items, *shape), force_disk=disk_all)
        if sl.shape != shape:
            skipped += 1
            continue
        data[len(kept)] = sl
        kept.append(it)
        vmin, vmax = min(vmin, float(sl.min())), max(vmax, float(sl.max()))
        samples.append(sl[::4, ::4].ravel())
    if not kept:
        raise RuntimeError("В серии нет пригодных срезов")
    if len(kept) < n_items:
        data = data[:len(kept)]
    if cache is not None:
        data.flush()
    n = len(kept)
    keys = [0.1, 1, 5, 95, 99, 99.9, 99.99]
    q = np.percentile(np.concatenate(samples), keys)
    stats = (dict(zip(keys, map(float, q))), vmin, vmax)

    th = first["th"] or 1.0
    pos = [it.pos for it in kept]
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
    sy, sx = ps or (1.0, 1.0)
    affine = None
    if calibrated and have_ipp and (has_pos or n == 1):
        try:
            affine = build_affine(iop0, [it.ipp for it in kept], sx, sy, dz)
        except Exception:
            affine = None
    warn = []
    if skipped:
        warn.append(f"Пропущено срезов другого размера: {skipped}")
    if not calibrated:
        warn.append("Нет PixelSpacing — измерения в пикселях")
    dicom_wl = None
    if first["wc"] is not None and first["ww"]:
        dicom_wl = (first["wc"], first["ww"])
    vol = Volume(data, (dz, sy, sx), z, has_pos, calibrated, first["modality"],
                 dicom_wl, first["mono1"], "", stats=stats, affine=affine, cache_path=cache)
    vol.series_uid, vol.phase, vol.obl_deg = series.uid, phase, obl
    multi = {s_ for s_, c in Counter(it.sop for it in kept).items() if c > 1}
    for k, it in enumerate(kept):                # (SOP UID, номер кадра с 1; 0 — любой кадр)
        if it.sop in multi:
            vol.sop_index[(it.sop, it.frame + 1)] = k
        vol.sop_index.setdefault((it.sop, 0), k)

    can_align = affine is not None and calibrated and n > 1
    if obl > OBLIQUE_DEG:
        if not align:
            warn.append(f"Срезы наклонные ({obl:.0f}°), показаны как есть")
        elif not can_align:
            warn.append(f"Срезы наклонные ({obl:.0f}°), но нет данных для выравнивания")
        elif ndimage is None:
            warn.append(f"Срезы наклонные ({obl:.0f}°); для выравнивания нужен scipy")
        else:
            new = align_volume(vol, progress, base=n_items, disk=disk_all)
            if new is None:                      # отмена
                vol.release()
                return None
            vol.release()
            vol = new
            warn.append(f"Выровнено: наклон {obl:.0f}°")
    vol.warn = "; ".join(warn)
    return vol


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
    values: dict = field(default_factory=dict)


def measure_calc(kind, pts, arr, sp, calibrated, modality):
    """Результат измерения: (строки текста, числа). pts — (столбец, строка)."""
    sy, sx = sp
    unit = "мм" if calibrated else "px"
    ct = modality == "CT"
    h, w = arr.shape
    if kind == "dist":
        (x0, y0), (x1, y1) = pts
        d = math.hypot((x1 - x0) * sx, (y1 - y0) * sy)
        return [f"{d:.1f} {unit}"], {"distance": round(d, 3), "unit": unit}
    if kind == "angle":
        a, b, c = pts
        v1 = np.array([(a[0] - b[0]) * sx, (a[1] - b[1]) * sy])
        v2 = np.array([(c[0] - b[0]) * sx, (c[1] - b[1]) * sy])
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        ang = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1, 1)))) if n > 0 else 0.0
        return [f"{ang:.1f}°"], {"angle_deg": round(ang, 2)}
    if kind == "probe":
        x = min(max(int(pts[0][0]), 0), w - 1)
        y = min(max(int(pts[0][1]), 0), h - 1)
        v = float(arr[y, x])
        return ([f"{v:.0f} HU" if ct else f"{v:.1f}"],
                {"value": round(v, 3), "unit": "HU" if ct else ""})
    if kind == "ellipse":
        (x0, y0), (x1, y1) = pts
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a, b = abs(x1 - x0) / 2, abs(y1 - y0) / 2
        if a < 0.5 or b < 0.5:
            return [], {}
        xa, xb = max(int(cx - a), 0), min(int(cx + a) + 1, w)
        ya, yb = max(int(cy - b), 0), min(int(cy + b) + 1, h)
        yy, xx = np.mgrid[ya:yb, xa:xb]
        m = ((xx + 0.5 - cx) / a) ** 2 + ((yy + 0.5 - cy) / b) ** 2 <= 1
        vals = arr[ya:yb, xa:xb][m]
        if vals.size == 0:
            return [], {}
        area = math.pi * a * sx * b * sy
        u = " HU" if ct else ""
        return ([f"S = {area:.1f} {unit}²",
                 f"μ = {vals.mean():.1f} ± {vals.std():.1f}{u}",
                 f"min {vals.min():.0f} / max {vals.max():.0f}"],
                {"area": round(area, 3), "area_unit": unit + "²",
                 "mean": round(float(vals.mean()), 3), "std": round(float(vals.std()), 3),
                 "min": float(vals.min()), "max": float(vals.max()),
                 "unit": "HU" if ct else ""})
    return [], {}


def measure_text(kind, pts, arr, sp, calibrated, modality):
    """Текст измерения (совместимость): только строки."""
    return measure_calc(kind, pts, arr, sp, calibrated, modality)[0]


def draw_text(p, x, y, s, color="#ffd54a"):
    p.setPen(QColor(0, 0, 0, 200))
    p.drawText(int(x) + 1, int(y) + 1, s)
    p.setPen(QColor(color))
    p.drawText(int(x), int(y), s)


# ───────────────── Аннотации и отчёты DICOM: PR, SR, SEG ─────────────────
@dataclass
class Annot:
    kind: str                  # "point" | "poly" | "text"
    pts: list                  # [(x, y)] — непрерывные пиксельные координаты изображения
    text: str = ""
    units: str = "PIXEL"       # PIXEL | DISPLAY (доли 0..1 от размера изображения)
    sop: str = ""
    frame: int = 0             # 0 — любой кадр
    closed: bool = False
    color: str = "#7CFC00"


SEG_COLORS = [(255, 40, 40), (40, 200, 60), (60, 120, 255), (255, 200, 0),
              (200, 60, 255), (0, 220, 220), (255, 130, 0), (150, 255, 100)]


def _refs_from_seq(seq):
    out = []
    for r in seq or []:
        sop = str(r.get("ReferencedSOPInstanceUID", ""))
        fr = r.get("ReferencedFrameNumber")
        if fr is None:
            out.append((sop, 0))
        else:
            frs = list(fr) if isinstance(fr, (list, MultiValue)) else [fr]
            out += [(sop, int(f)) for f in frs]
    return out


def _circle_poly(c, p, n=48):
    r = math.hypot(p[0] - c[0], p[1] - c[1])
    return [(c[0] + r * math.cos(2 * math.pi * i / n),
             c[1] + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def _ellipse_poly(pts, n=48):
    """Эллипс DICOM: концы большой оси (0,1) и малой оси (2,3)."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts[:4]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a, b = math.hypot(x1 - x0, y1 - y0) / 2, math.hypot(x3 - x2, y3 - y2) / 2
    th = math.atan2(y1 - y0, x1 - x0)
    ct, st = math.cos(th), math.sin(th)
    out = []
    for i in range(n):
        t = 2 * math.pi * i / n
        u, v = a * math.cos(t), b * math.sin(t)
        out.append((cx + u * ct - v * st, cy + u * st + v * ct))
    return out


def _graphic_to_annots(gtype, pts, sop, frame, units="PIXEL", text=""):
    gtype = gtype.upper()
    if gtype in ("POINT", "MULTIPOINT"):
        return [Annot("point", [p], text, units, sop, frame) for p in pts]
    if gtype in ("POLYLINE", "INTERPOLATED") and len(pts) >= 2:
        closed = len(pts) > 2 and pts[0] == pts[-1]
        return [Annot("poly", pts, text, units, sop, frame, closed=closed)]
    if gtype == "CIRCLE" and len(pts) >= 2:
        return [Annot("poly", _circle_poly(pts[0], pts[1]), text, units, sop, frame, closed=True)]
    if gtype == "ELLIPSE" and len(pts) >= 4:
        return [Annot("poly", _ellipse_poly(pts), text, units, sop, frame, closed=True)]
    return []


def _pairs(data):
    d = [float(v) for v in data]
    return list(zip(d[0::2], d[1::2]))


def parse_pr(ds):
    """Presentation State (GSPS): графика и текст на изображениях."""
    refs, annots = set(), []
    for rs in ds.get("ReferencedSeriesSequence") or []:
        for r in rs.get("ReferencedImageSequence") or []:
            refs.add(str(r.get("ReferencedSOPInstanceUID", "")))
    for ga in ds.get("GraphicAnnotationSequence") or []:
        targets = _refs_from_seq(ga.get("ReferencedImageSequence"))
        refs.update(s for s, _ in targets)
        for go in ga.get("GraphicObjectSequence") or []:
            pts = _pairs(go.get("GraphicData", []))
            units = str(go.get("GraphicAnnotationUnits", "PIXEL")).upper()
            for sop, fr in targets:
                annots += _graphic_to_annots(str(go.get("GraphicType", "")), pts, sop, fr, units)
        for to in ga.get("TextObjectSequence") or []:
            txt = str(to.get("UnformattedTextValue", ""))
            if "AnchorPoint" in to:
                pt, units = tuple(float(v) for v in to.AnchorPoint), \
                    str(to.get("AnchorPointAnnotationUnits", "PIXEL")).upper()
            elif "BoundingBoxTopLeftHandCorner" in to:
                pt, units = tuple(float(v) for v in to.BoundingBoxTopLeftHandCorner), \
                    str(to.get("BoundingBoxAnnotationUnits", "PIXEL")).upper()
            else:
                continue
            for sop, fr in targets:
                annots.append(Annot("text", [pt], txt, units, sop, fr))
    return {"refs": refs, "annots": annots, "text": ""}


def _code_meaning(seq):
    try:
        return str(seq[0].get("CodeMeaning", ""))
    except Exception:
        return ""


def _sr_value(it):
    vt = str(it.get("ValueType", "")).upper()
    if vt == "TEXT":
        return str(it.get("TextValue", ""))
    if vt == "CODE":
        return _code_meaning(it.get("ConceptCodeSequence"))
    if vt == "NUM":
        ms = it.get("MeasuredValueSequence")
        if ms:
            u = ms[0].get("MeasurementUnitsCodeSequence")
            unit = (str(u[0].get("CodeValue", "")) if u else "")
            return f"{ms[0].get('NumericValue', '')} {unit}".strip()
        return ""
    if vt in ("DATE", "TIME", "DATETIME"):
        return str(it.get({"DATE": "Date", "TIME": "Time", "DATETIME": "DateTime"}[vt], ""))
    if vt == "PNAME":
        return str(it.get("PersonName", ""))
    if vt == "UIDREF":
        return str(it.get("UID", ""))
    if vt == "SCOORD":
        return f"{it.get('GraphicType', '')}, точек: {len(it.get('GraphicData', [])) // 2}"
    if vt in ("IMAGE", "COMPOSITE"):
        return "ссылка на изображение"
    return ""


def parse_sr(ds):
    """Structured Report: текстовое дерево, ссылки на снимки и графика (SCOORD)."""
    refs, annots, lines = set(), [], []
    title = _code_meaning(ds.get("ConceptNameCodeSequence")) or "Структурированный отчёт"
    lines.append(title)
    for k in ("PatientName", "StudyDate", "CompletionFlag", "VerificationFlag"):
        if ds.get(k):
            lines.append(f"{k}: {ds.get(k)}")
    lines.append("")
    for ev in ds.get("CurrentRequestedProcedureEvidenceSequence") or []:
        for rs in ev.get("ReferencedSeriesSequence") or []:
            for r in rs.get("ReferencedSOPSequence") or []:
                refs.add(str(r.get("ReferencedSOPInstanceUID", "")))

    def walk(items, depth, ctx):
        label_parts = []
        for it in items:
            if str(it.get("ValueType", "")).upper() in ("NUM", "TEXT", "CODE"):
                nm = _code_meaning(it.get("ConceptNameCodeSequence"))
                val = _sr_value(it)
                if val:
                    label_parts.append(f"{nm}: {val}" if nm else val)
        for it in items:
            vt = str(it.get("ValueType", "")).upper()
            nm = _code_meaning(it.get("ConceptNameCodeSequence"))
            val = _sr_value(it)
            pad = "    " * depth
            if vt == "CONTAINER":
                lines.append(f"{pad}■ {nm}")
            elif nm and val:
                lines.append(f"{pad}• {nm}: {val}")
            else:
                lines.append(f"{pad}• {nm or val}")
            children = it.get("ContentSequence") or []
            if vt == "SCOORD":
                targets = []
                for ch in children:
                    if str(ch.get("ValueType", "")).upper() == "IMAGE":
                        targets += _refs_from_seq(ch.get("ReferencedSOPSequence"))
                refs.update(s for s, _ in targets)
                text = "; ".join(label_parts) or ctx
                for sop, fr in targets:
                    annots.extend(_graphic_to_annots(str(it.get("GraphicType", "")),
                                                     _pairs(it.get("GraphicData", [])), sop, fr, "PIXEL", text))
            elif vt == "IMAGE":
                refs.update(s for s, _ in _refs_from_seq(it.get("ReferencedSOPSequence")))
            walk(children, depth + 1, nm or ctx)

    walk(ds.get("ContentSequence") or [], 0, title)
    return {"refs": refs, "annots": annots, "text": "\n".join(lines)}


def cielab_to_rgb(lab):
    """DICOM RecommendedDisplayCIELabValue (16 бит) → (R, G, B)."""
    try:
        L = lab[0] * 100.0 / 65535.0
        a = lab[1] * 255.0 / 65535.0 - 128.0
        b = lab[2] * 255.0 / 65535.0 - 128.0
        fy = (L + 16) / 116.0
        fx, fz = fy + a / 500.0, fy - b / 200.0

        def finv(t):
            return t ** 3 if t ** 3 > 0.008856 else (t - 16 / 116) / 7.787
        X, Y, Z = 0.96422 * finv(fx), finv(fy), 0.82521 * finv(fz)      # D50
        r = 3.1338561 * X - 1.6168667 * Y - 0.4906146 * Z
        g = -0.9787684 * X + 1.9161415 * Y + 0.0334540 * Z
        bl = 0.0719453 * X - 0.2289914 * Y + 1.4052427 * Z

        def gam(c):
            c = max(0.0, min(1.0, c))
            return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return tuple(int(round(255 * gam(c))) for c in (r, g, bl))
    except Exception:
        return None


def parse_seg(ds):
    """Заголовок SEG: сегменты (номер → (название, цвет)) и ссылки на снимки."""
    refs, segs = set(), {}
    for s in ds.get("SegmentSequence") or []:
        num = int(s.get("SegmentNumber", 1))
        lab = s.get("RecommendedDisplayCIELabValue")
        segs[num] = (str(s.get("SegmentLabel", f"Сегмент {num}")),
                     cielab_to_rgb([int(v) for v in lab]) if lab else None)
    for pf in ds.get("PerFrameFunctionalGroupsSequence") or []:
        for d in pf.get("DerivationImageSequence") or []:
            for si in d.get("SourceImageSequence") or []:
                refs.add(str(si.get("ReferencedSOPInstanceUID", "")))
    for rs in ds.get("ReferencedSeriesSequence") or []:
        for r in rs.get("ReferencedInstanceSequence") or []:
            refs.add(str(r.get("ReferencedSOPInstanceUID", "")))
    return {"refs": refs, "segments": segs, "annots": [], "text": ""}


def load_seg_labels(path, vol):
    """Читает пиксели SEG и собирает карту меток (Z×Y×X) в геометрии объёма."""
    ds = pydicom.dcmread(path)
    nf = int(ds.get("NumberOfFrames", 1) or 1)
    rows, cols = int(ds.Rows), int(ds.Columns)
    if int(ds.get("BitsAllocated", 1)) == 1:
        bits = np.unpackbits(np.frombuffer(ds.PixelData, np.uint8), bitorder="little")
        arr = bits[:nf * rows * cols].reshape(nf, rows, cols)
    else:
        arr = np.asarray(ds.pixel_array).reshape(nf, rows, cols)
    if str(ds.get("SegmentationType", "BINARY")).upper() == "FRACTIONAL":
        thr = float(ds.get("MaxFractionalValue", 255)) * 0.5
        arr = arr >= thr
    else:
        arr = arr > 0
    Zs, Ys, Xs = vol.src_shape
    if (rows, cols) != (Ys, Xs):
        raise RuntimeError(f"Размер SEG {cols}×{rows} не совпадает с серией {Xs}×{Ys}")
    labels = np.zeros((Zs, Ys, Xs), np.uint8)
    perframe = ds.get("PerFrameFunctionalGroupsSequence") or []
    used = 0
    for f in range(nf):
        pf = perframe[f] if f < len(perframe) else None
        try:
            segnum = int(pf.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        except Exception:
            segnum = 1
        try:
            sop = str(pf.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)
        except Exception:
            continue
        k = vol.sop_index.get((sop, 0))
        if k is None:
            continue
        labels[k][arr[f]] = segnum
        used += 1
    if not used:
        raise RuntimeError("SEG не содержит кадров, относящихся к этой серии")
    if vol.aligned and vol.warp:
        labels = vol.warp(labels, 0)
    return labels


def parse_object(obj):
    """Разбор объекта (результат кэшируется)."""
    if obj.info is None:
        if obj.kind == "SEG":
            ds = pydicom.dcmread(obj.path, stop_before_pixels=True)
            obj.info = parse_seg(ds)
        else:
            ds = pydicom.dcmread(obj.path)
            obj.info = parse_pr(ds) if obj.kind == "PR" else parse_sr(ds)
    return obj.info


def place_annot(vol, a):
    """Переносит аннотацию на текущий объём → (индекс среза z, Annot) или None."""
    k = vol.sop_index.get((a.sop, a.frame)) if a.frame else None
    if k is None:
        k = vol.sop_index.get((a.sop, 0))
    if k is None:
        return None
    _, h, w = vol.src_shape
    pts = [(x * w, y * h) for x, y in a.pts] if a.units == "DISPLAY" else list(a.pts)
    if not vol.aligned:
        return k, Annot(a.kind, pts, a.text, "PIXEL", a.sop, a.frame, a.closed, a.color)
    out, zs = [], []
    for x, y in pts:
        cx, cy, cz = vol.src_to_cur(x - 0.5, y - 0.5, k)
        out.append((cx + 0.5, cy + 0.5))
        zs.append(cz)
    z = int(round(float(np.mean(zs))))
    if not (0 <= z < vol.data.shape[0]):
        return None
    return z, Annot(a.kind, out, a.text, "PIXEL", a.sop, a.frame, a.closed, a.color)


def attach_object(vol, obj):
    """Применяет объект PR / SR / SEG к объёму. Возвращает число элементов (0 — не относится)."""
    info = parse_object(obj)
    if not (info["refs"] & {s for s, _ in vol.sop_index}):
        return 0
    if obj.kind == "SEG":
        labels = load_seg_labels(obj.path, vol)
        lut = np.zeros((256, 3), np.float32)
        names = {}
        for num in range(1, 256):
            lut[num] = SEG_COLORS[(num - 1) % len(SEG_COLORS)]
        for num, (name, rgb) in info["segments"].items():
            if 0 < num < 256:
                names[num] = name
                if rgb:
                    lut[num] = rgb
        vol.seg, vol.seg_lut, vol.seg_names = labels, lut, names
        return int(len(np.unique(labels)) - (1 if (labels == 0).any() else 0))
    n = 0
    for a in info["annots"]:
        placed = place_annot(vol, a)
        if placed:
            vol.annots.setdefault(placed[0], []).append(placed[1])
            n += 1
    return n


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
        self.force_idx, self.force, self.snapshot = None, False, False   # для отчётов

    def _idx(self):
        return self.panel.pos[self.plane] if self.force_idx is None else self.force_idx

    def reset_view(self):
        self.zoom, self.pan = 1.0, QPoint(0, 0)
        self.update()

    # --- отрисовка среза с окном W/L
    def render(self):
        if self.isHidden() and not self.force:
            return
        p, vol = self.panel, self.panel.vol
        if vol is None or vol.data is None:
            self._img = self._arr = None
            return self.update()
        idx = self._idx()
        arr = Volume.cut(vol.data, self.plane, idx)
        self._arr = arr
        self._sp = vol.spacing2d(self.plane)
        lo = p.wc - p.ww / 2
        img = (np.clip((arr - lo) / max(p.ww, 1e-6), 0, 1) * 255).astype(np.uint8)
        if vol.mono1 != p.invert:
            img = 255 - img
        h, w = img.shape
        mk = sg = None
        if vol.mask is not None and p.show_mask:
            mk = Volume.cut(vol.mask, self.plane, idx)
        if vol.seg is not None and p.show_seg:
            sg = Volume.cut(vol.seg, self.plane, idx)
        has_mk = mk is not None and mk.any()
        has_sg = sg is not None and sg.any()
        if has_mk or has_sg:
            rgb = np.stack([img] * 3, -1).astype(np.float32)
            if has_sg:                                  # сегментация DICOM SEG
                sel = sg > 0
                rgb[sel] = rgb[sel] * 0.55 + vol.seg_lut[sg[sel]] * 0.45
            if has_mk:                                  # маска инструмента «Объём»
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
        idx = self._idx()
        h, w = self._arr.shape
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(QRectF(x0, y0, dw, dh), self._img)

        if pan.multi and not self.snapshot:             # перекрестие MPR
            u, v = pan.cross_uv(self.plane)
            cx, cy = self.to_wid(u, v)
            p.setPen(QPen(QColor(255, 213, 74, 170), 1))
            p.drawLine(QPointF(x0, cy), QPointF(x0 + dw, cy))
            p.drawLine(QPointF(cx, y0), QPointF(cx, y0 + dh))

        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for m in pan.meas.get((self.plane, idx), []):
            self._draw_measure(p, m.kind, m.pts, m.text)
        if pan.show_annots and self.plane == AX:        # аннотации PR / SR
            for a in vol.annots.get(idx, []):
                self._draw_annot(p, a)
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

    def _draw_annot(self, p, a):
        if not a.pts:
            return
        wp = [QPointF(*self.to_wid(u, v)) for u, v in a.pts]
        p.setPen(QPen(QColor(a.color), 1.6))
        if a.kind == "poly" and len(wp) >= 2:
            if a.closed:
                p.drawPolygon(QPolygonF(wp))
            else:
                p.drawPolyline(QPolygonF(wp))
            ax, ay = wp[0].x() + 6, wp[0].y() - 6
        elif a.kind == "text":
            ax, ay = wp[0].x(), wp[0].y()
        else:
            q = wp[0]
            p.drawLine(QPointF(q.x() - 5, q.y()), QPointF(q.x() + 5, q.y()))
            p.drawLine(QPointF(q.x(), q.y() - 5), QPointF(q.x(), q.y() + 5))
            ax, ay = q.x() + 8, q.y() - 8
        if a.text:
            for k, line in enumerate(a.text.split("\n")):
                draw_text(p, ax, ay + 14 * k, line, a.color)

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
        text, values = measure_calc(kind, pts, self._arr, self._sp, vol.calibrated, vol.modality)
        if text:
            self.panel.add_measure(self.plane, Measure(kind, list(pts), text, values))
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
        self.series = None                         # открытая серия
        self.phase = 0
        self.phase_vols = {}                       # фаза → Volume (уже загруженные)
        self.base_title = ""
        self.pos = [0, 0, 0]                       # (z, y, x)
        self.wc = self.ww = 1.0
        self.invert = False
        self.tool = "wl"
        self.meas = {}                             # (plane, idx) -> [Measure]
        self.title = ""
        self.presets = OrderedDict()
        self.show_mask, self.tol_pct, self.multi = True, 15, False
        self.show_annots = self.show_seg = True
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
        # Растягиваем только занятые строки и столбцы: один вид занимает всё окно.
        used = 2 if mode in ("MPR", "MPR + 3D") else 1
        for i in range(2):
            self.grid.setColumnStretch(i, 1 if i < used else 0)
            self.grid.setRowStretch(i, 1 if i < used else 0)
        self.multi = mode in ("MPR", "MPR + 3D")
        self.refresh()
        self.v3d.render()

    # --- объём
    def _compose_title(self):
        t = self.base_title
        if self.series is not None and self.series.nphases > 1:
            name = self.series.phase_names[self.phase] if self.phase < len(self.series.phase_names) else ""
            t += f" · фаза {self.phase + 1}/{self.series.nphases}" + (f" ({name})" if name else "")
        if self.vol is not None and self.vol.aligned:
            t += f" · выровнено ({self.vol.obl_deg:.0f}°)"
        self.title = t

    def release_all(self):
        """Освобождает все объёмы панели (память и временные файлы)."""
        for v in {id(v): v for v in (*self.phase_vols.values(), self.vol) if v is not None}.values():
            v.release()
        self.phase_vols = {}
        self.vol = None

    def set_volume(self, vol, title, series=None, phase=0):
        """Открывает новую серию (прежняя освобождается)."""
        self.release_all()
        self.vol, self.base_title = vol, title
        self.series, self.phase = series, phase
        self.phase_vols = {phase: vol}
        self._compose_title()
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

    def swap_phase(self, vol, phase):
        """Переключение на другую фазу той же серии: положение и яркость сохраняются."""
        old = self.vol
        self.vol, self.phase = vol, phase
        self.phase_vols[phase] = vol
        if old is not None and old.mask is not None and vol.mask is None \
                and old.mask.shape == vol.data.shape:
            vol.mask = old.mask                       # область объёма та же
        if old is not None and old.seg is not None and vol.seg is None \
                and old.seg.shape == vol.data.shape:
            vol.seg, vol.seg_lut, vol.seg_names = old.seg, old.seg_lut, old.seg_names
        if old is not None and old.data.shape != vol.data.shape:
            Z, Y, X = vol.data.shape
            self.pos = [Z // 2, Y // 2, X // 2]
            self.meas.clear()
        else:
            self.recompute_measures()
        self._compose_title()
        self.refresh()
        self.v3d.render()
        self.changed.emit()

    def recompute_measures(self):
        """Пересчитывает числа измерений на данных текущей фазы (позиции остаются)."""
        vol = self.vol
        for (plane, idx), lst in self.meas.items():
            if vol is None or idx >= vol.count(plane):
                continue
            arr = Volume.cut(vol.data, plane, idx)
            sp = vol.spacing2d(plane)
            for m in lst:
                if m.kind == "region":
                    continue
                text, values = measure_calc(m.kind, m.pts, arr, sp, vol.calibrated, vol.modality)
                if text:
                    m.text, m.values = text, values

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
        """Объём структуры: 3D-связная область с интенсивностью seed ± допуск.

        Экономно по памяти: порог считается блоками, разрастание — через
        binary_propagation (без целочисленной карты меток).
        """
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
            lo, hi = ref - tol, ref + tol
            m = np.zeros(vol.data.shape, bool)
            for z0 in range(0, Z, 16):
                blk = vol.data[z0:z0 + 16]
                m[z0:z0 + 16] = (blk >= lo) & (blk <= hi)
            if not m[z, y, x]:
                return self.status.emit("Точка вне допуска — увеличьте допуск")
            seed = np.zeros_like(m)
            seed[z, y, x] = True
            comp = ndimage.binary_propagation(seed, mask=m)
            del m, seed
            cnt = int(comp.sum())
        finally:
            QApplication.restoreOverrideCursor()
        vol.mask = comp
        self.show_mask = True
        values = {"voxels": cnt}
        if vol.calibrated:
            ml = cnt * vol.sz * vol.sy * vol.sx / 1000.0
            text = [f"V = {ml:.2f} мл", f"{cnt} вокс."]
            values["volume_ml"] = round(ml, 4)
        else:
            text = [f"V = {cnt} вокс."]
        self.add_measure(plane, Measure("region", [(u, v)], text, values))
        self.status.emit("Объём: " + ", ".join(text))
        self.refresh()


# ───────────────────── Сохранение результатов: сессия, CSV, отчёт PDF ─────────────────────
SESSION_FORMAT = "mrtviewer-session"


def _series_label(panel):
    return panel.base_title or (panel.series.label() if panel.series else "")


def sorted_measures(panel):
    """[(плоскость, срез, Measure)] в порядке плоскость → срез."""
    out = []
    for (plane, idx), lst in sorted(panel.meas.items()):
        out += [(plane, idx, m) for m in lst]
    return out


def mask_stats(vol):
    if vol is None or vol.mask is None or not vol.mask.any():
        return None
    cnt = int(vol.mask.sum())
    d = {"voxels": cnt}
    if vol.calibrated:
        d["volume_ml"] = round(cnt * vol.sz * vol.sy * vol.sx / 1000.0, 4)
    return d


def save_session(panel, path):
    """Сохраняет измерения (JSON) и маску объёма (рядом, *.mask.npz)."""
    vol = panel.vol
    if vol is None:
        raise RuntimeError("Нет открытой серии")
    path = Path(path)
    data = {
        "format": SESSION_FORMAT, "version": 1,
        "saved": datetime.now().isoformat(timespec="seconds"),
        "series": {"uid": vol.series_uid, "title": _series_label(panel),
                   "phase": panel.phase, "modality": vol.modality,
                   "shape": list(vol.data.shape),
                   "spacing_zyx": [vol.sz, vol.sy, vol.sx],
                   "calibrated": vol.calibrated, "aligned": vol.aligned},
        "measurements": [
            {"plane": plane, "index": idx, "kind": m.kind,
             "points": [[round(float(a), 3), round(float(b), 3)] for a, b in m.pts],
             "text": m.text, "values": m.values}
            for plane, idx, m in sorted_measures(panel)],
        "mask": None,
    }
    ms = mask_stats(vol)
    if ms:
        mpath = path.with_name(path.stem + ".mask.npz")
        np.savez_compressed(mpath, mask=np.packbits(vol.mask.ravel()),
                            shape=np.array(vol.mask.shape))
        data["mask"] = {"file": mpath.name, **ms}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def load_session(panel, path):
    """Загружает измерения и маску в открытую серию. Возвращает (число измерений, маска?)."""
    vol = panel.vol
    if vol is None:
        raise RuntimeError("Сначала откройте серию, к которой относится файл")
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != SESSION_FORMAT:
        raise RuntimeError("Это не файл сохранения MRT Viewer")
    s = data["series"]
    if s.get("uid") and vol.series_uid and s["uid"] != vol.series_uid:
        raise RuntimeError("Файл относится к другой серии")
    if tuple(s.get("shape", ())) != tuple(vol.data.shape):
        raise RuntimeError("Размер серии не совпадает с сохранённым "
                           f"({s.get('shape')} и {list(vol.data.shape)}). "
                           "Проверьте фазу и режим выравнивания")
    panel.meas.clear()
    for m in data.get("measurements", []):
        panel.meas.setdefault((int(m["plane"]), int(m["index"])), []).append(
            Measure(m["kind"], [tuple(p) for p in m["points"]], list(m["text"]),
                    dict(m.get("values", {}))))
    has_mask = False
    if data.get("mask"):
        z = np.load(path.with_name(data["mask"]["file"]))
        shape = tuple(int(v) for v in z["shape"])
        if shape == tuple(vol.data.shape):
            vol.mask = np.unpackbits(z["mask"])[:int(np.prod(shape))].reshape(shape).astype(bool)
            has_mask = True
    panel.recompute_measures()
    panel.refresh()
    return len(data.get("measurements", [])), has_mask


CSV_COLUMNS = ["series", "plane", "slice", "kind", "result", "distance", "angle_deg", "value",
               "area", "mean", "std", "min", "max", "volume_ml", "voxels", "unit"]
KIND_NAMES = {"dist": "Линейка", "angle": "Угол", "probe": "Значение", "ellipse": "ROI",
              "region": "Объём"}


def export_csv(panel, path):
    rows = sorted_measures(panel)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(CSV_COLUMNS)
        for plane, idx, m in rows:
            v = m.values
            w.writerow([_series_label(panel), PLANE_NAMES[plane], idx + 1, KIND_NAMES.get(m.kind, m.kind),
                        " | ".join(m.text)] + [v.get(c, "") for c in CSV_COLUMNS[5:]])
    return len(rows)


def slice_image(panel, plane, idx, width=1000):
    """Снимок среза со всеми наложениями (измерения, маска, аннотации) в QImage."""
    v = SliceView(panel, plane)
    v.snapshot, v.force, v.force_idx = True, True, idx
    v.resize(width, width)
    v.render()
    if v._arr is None:
        return None
    pw, ph = v._arr.shape[1] * v._sp[1], v._arr.shape[0] * v._sp[0]
    v.resize(width, int(max(200, min(width * 1.3, width * ph / pw))))
    img = v.grab().toImage()
    v.deleteLater()
    return img


class _Pager:
    """Постраничная печать в QPdfWriter."""

    def __init__(self, writer):
        self.w = writer
        self.p = QPainter(writer)
        self.W, self.H = writer.width(), writer.height()
        self.y, self.page = 0, 1
        self.base = max(10, int(self.W / 62))

    def font(self, size=1.0, bold=False, color="#000000"):
        f = QFont()
        f.setPixelSize(int(self.base * size))
        f.setBold(bold)
        self.p.setFont(f)
        self.p.setPen(QColor(color))

    def footer(self):
        self.font(0.8, color="#777777")
        self.p.drawText(QRectF(0, self.H - self.base * 1.5, self.W, self.base * 1.5),
                        int(Qt.AlignmentFlag.AlignRight), f"MRT Viewer · стр. {self.page}")

    def new_page(self):
        self.footer()
        self.w.newPage()
        self.page += 1
        self.y = 0

    def need(self, h):
        if self.y + h > self.H - self.base * 2:
            self.new_page()

    def text(self, s, size=1.0, bold=False, color="#000000", indent=0, gap=0.4):
        self.font(size, bold, color)
        rect = QRectF(indent, 0, self.W - indent, self.H)
        flags = int(Qt.TextFlag.TextWordWrap)
        h = self.p.boundingRect(rect, flags, s).height()
        self.need(h)
        self.p.drawText(QRectF(indent, self.y, self.W - indent, h + 4), flags, s)
        self.y += h + self.base * gap

    def image(self, img, width=None, max_h=None):
        if img is None:
            return
        width = width or self.W
        h = width * img.height() / img.width()
        max_h = max_h or (self.H * 0.62)
        if h > max_h:
            width, h = width * max_h / h, max_h
        self.need(h)
        self.p.drawImage(QRectF((self.W - width) / 2, self.y, width, h), img)
        self.y += h + self.base * 0.6

    def close(self):
        self.footer()
        self.p.end()


def build_report(panel, path):
    """Отчёт PDF: сведения о серии, таблица измерений, снимки срезов, сегментация."""
    vol = panel.vol
    if vol is None:
        raise RuntimeError("Нет открытой серии")
    writer = QPdfWriter(str(path))
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setResolution(110)
    writer.setPageMargins(QMarginsF(12, 12, 12, 12), QPageLayout.Unit.Millimeter)
    writer.setTitle("Отчёт MRT Viewer")
    pg = _Pager(writer)
    pg.text("Отчёт MRT Viewer", 2.0, True, "#1f4e8c", gap=0.2)
    pg.text(datetime.now().strftime("%d.%m.%Y %H:%M"), 0.9, color="#666666", gap=0.8)
    pg.text(_series_label(panel), 1.1, True, gap=0.5)
    Z, Y, X = vol.data.shape
    info = [f"Модальность: {vol.modality or '—'}",
            f"Матрица: {X} × {Y} × {Z}",
            (f"Размер вокселя: {vol.sx:.2f} × {vol.sy:.2f} × {vol.sz:.2f} мм" if vol.calibrated
             else "Размеры в пикселях (нет PixelSpacing)"),
            f"Окно: W {panel.ww:.0f}, L {panel.wc:.0f}"]
    if vol.aligned:
        info.append(f"Срезы выровнены по осям пациента (исходный наклон {vol.obl_deg:.0f}°)")
    if panel.series is not None and panel.series.nphases > 1:
        info.append(f"Фаза: {panel.phase + 1} из {panel.series.nphases}")
    for line in info:
        pg.text(line, 0.95, gap=0.15)
    ms = mask_stats(vol)
    if ms:
        s = f"Сегментация (инструмент «Объём»): {ms['voxels']} вокс."
        if "volume_ml" in ms:
            s += f", объём {ms['volume_ml']:.2f} мл"
        pg.text(s, 1.0, True, "#b00020", gap=0.4)
    if vol.seg is not None and vol.seg_names:
        pg.text("Сегменты DICOM SEG: " + ", ".join(vol.seg_names.values()), 0.95, gap=0.4)
    pg.y += pg.base * 0.5

    meas = sorted_measures(panel)
    if meas:
        pg.text("Измерения", 1.3, True, "#1f4e8c", gap=0.4)
        for n, (plane, idx, m) in enumerate(meas, 1):
            pg.text(f"{n}. {KIND_NAMES.get(m.kind, m.kind)} — {PLANE_NAMES[plane].lower()}, срез "
                    f"{idx + 1}/{vol.count(plane)}:  " + " · ".join(m.text), 0.95, gap=0.25)
    else:
        pg.text("Измерений нет.", 1.0, color="#666666")

    groups = {}
    for n, (plane, idx, m) in enumerate(meas, 1):
        groups.setdefault((plane, idx), []).append((n, m))
    for (plane, idx), lst in groups.items():
        pg.new_page()
        pg.text(f"{PLANE_NAMES[plane]} плоскость, срез {idx + 1}/{vol.count(plane)}", 1.3, True, "#1f4e8c")
        pg.image(slice_image(panel, plane, idx))
        for n, m in lst:
            pg.text(f"{n}. {KIND_NAMES.get(m.kind, m.kind)}: " + " · ".join(m.text), 0.95, gap=0.2)

    if ms:                                         # три среза через центр сегментации
        mk = vol.mask
        c = [int(np.argmax(mk.sum(axis=ax))) for ax in ((1, 2), (0, 2), (0, 1))]
        pg.new_page()
        pg.text("Сегментация: срезы через область максимального сечения", 1.3, True, "#1f4e8c")
        for plane, i in zip((AX, COR, SAG), c):
            pg.text(f"{PLANE_NAMES[plane]}, срез {i + 1}", 0.95, gap=0.2)
            pg.image(slice_image(panel, plane, i, 800), width=pg.W * 0.55, max_h=pg.H * 0.27)
    if panel.v3d._img is not None and panel.v3d.isVisible():
        pg.new_page()
        pg.text("3D-вид", 1.3, True, "#1f4e8c")
        pg.image(panel.v3d._img.copy(), width=pg.W * 0.8)
    pg.close()
    return len(meas)


# ─────────────────────────────────── Окно ───────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRT Viewer — DICOM / PACS / MPR / 3D")
        self.series, self.proc, self.find_dir = [], None, None
        self.objects = []                       # объекты SR / PR / SEG
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
        tabs.addTab(self._annot_tab(), "Аннотации")

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
        tb.addSeparator()

        self.act_align = QAction("Выравнивать наклон", self)
        self.act_align.setCheckable(True)
        self.act_align.setChecked(True)
        self.act_align.setToolTip("Наклонные срезы пересчитываются по осям пациента при открытии серии "
                                  "(нужен scipy). Действует на следующую загрузку.")
        tb.addAction(self.act_align)

        # динамические серии: переключатель фаз и воспроизведение
        self.phase_actions = []
        self.phase_actions.append(tb.addWidget(QLabel("  Фаза: ")))
        self.sp_phase = QSpinBox()
        self.sp_phase.setRange(1, 1)
        self.sp_phase.valueChanged.connect(lambda v: self.change_phase(self.active, v - 1))
        self.phase_actions.append(tb.addWidget(self.sp_phase))
        self.act_play = QAction("▶ Играть", self)
        self.act_play.setCheckable(True)
        self.act_play.toggled.connect(self.toggle_play)
        tb.addAction(self.act_play)
        self.phase_actions.append(tb.actions()[-1])
        for a in self.phase_actions:
            a.setVisible(False)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(500)
        self.play_timer.timeout.connect(self._step_phase)

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
        n = p.series.nphases if p.series is not None else 1
        for a in self.phase_actions:
            a.setVisible(n > 1)
        if n <= 1 and self.act_play.isChecked():
            self.act_play.setChecked(False)
        self.sp_phase.blockSignals(True)
        self.sp_phase.setRange(1, max(n, 1))
        self.sp_phase.setSuffix(f" / {n}")
        self.sp_phase.setValue(p.phase + 1)
        self.sp_phase.blockSignals(False)
        self._refresh_annot_info()

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
        lay.addWidget(QLabel("<b>Результаты работы</b> (для активного окна)"))
        for text, fn in (("Сохранить измерения и маску…", self.ui_save_session),
                         ("Загрузить сохранённые измерения…", self.ui_load_session),
                         ("Экспорт измерений в CSV…", self.ui_export_csv),
                         ("Создать отчёт PDF…", self.ui_report)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, f=fn: f())
            lay.addWidget(b)
        lay.addStretch(1)
        return w

    def open_folder(self, append=False):
        d = QFileDialog.getExistingDirectory(self, "Папка с DICOM")
        if d:
            self.load_folder(d, append=bool(append))

    def load_folder(self, folder, append=False):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            found, objs = scan_folder(folder)
        finally:
            QApplication.restoreOverrideCursor()
        if not found and not objs:
            return QMessageBox.information(self, "DICOM", "DICOM-файлы не найдены")
        if append:
            self.series += found
            self.objects += objs
        else:
            self.series = found
            self.objects = objs
        self._fill_objects()
        self.series_list.clear()
        for s in self.series:
            self.series_list.addItem(s.label())
        row = len(self.series) - len(found)
        self.series_list.setCurrentRow(row)
        if not append and found:
            self.load_series_into(self.pA, row)

    def _loader(self, series, phase):
        """Загрузка объёма с индикатором прогресса. None — отмена."""
        total = len(series.phases[phase] if series.phases else series.items)
        dlg = QProgressDialog("Загрузка серии…", "Отмена", 0, total, self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(300)

        def prog(n, tot=None):
            if tot is not None and tot != dlg.maximum():
                dlg.setMaximum(tot)
                dlg.setLabelText("Выравнивание наклонных срезов…")
            dlg.setValue(n)
            QApplication.processEvents()
            return not dlg.wasCanceled()

        try:
            vol = load_volume(series, prog, phase, align=self.act_align.isChecked())
        finally:
            dlg.close()
        return vol

    def load_series_into(self, panel, row):
        if not (0 <= row < len(self.series)):
            return
        s = self.series[row]
        try:
            vol = self._loader(s, 0)
        except Exception as ex:
            return QMessageBox.critical(self, "DICOM", f"Ошибка: {ex}")
        if vol is None:
            return
        self.play_timer.stop()
        self.act_play.setChecked(False)
        panel.set_volume(vol, f"{s.patient} | {s.label()}", series=s, phase=0)
        msgs = [vol.warn] if vol.warn else []
        msgs += self.attach_annotations(panel)
        if s.nphases > 1:
            msgs.append(f"Динамическая серия: {s.nphases} фаз (переключатель «Фаза» на панели)")
        if msgs:
            self.statusBar().showMessage("; ".join(msgs), 12000)
        self._sync_toolbar()

    # --- динамические серии
    def change_phase(self, panel, k):
        s = panel.series
        if s is None or not (0 <= k < s.nphases) or k == panel.phase:
            return
        vol = panel.phase_vols.get(k)
        if vol is None:
            try:
                vol = self._loader(s, k)
            except Exception as ex:
                return QMessageBox.critical(self, "DICOM", f"Ошибка: {ex}")
            if vol is None:
                self.sp_phase.blockSignals(True)
                self.sp_phase.setValue(panel.phase + 1)
                self.sp_phase.blockSignals(False)
                return
            self.attach_annotations(panel, vol)
        panel.swap_phase(vol, k)
        if panel is self.active:
            self.sp_phase.blockSignals(True)
            self.sp_phase.setValue(k + 1)
            self.sp_phase.blockSignals(False)

    def toggle_play(self, on):
        if on:
            self.play_timer.start()
        else:
            self.play_timer.stop()

    def _step_phase(self):
        p = self.active
        if p.series is None or p.series.nphases < 2:
            return self.act_play.setChecked(False)
        self.change_phase(p, (p.phase + 1) % p.series.nphases)

    # --- аннотации DICOM (PR / SR / SEG)
    def attach_annotations(self, panel, vol=None):
        """Подключает к объёму все подходящие объекты. Возвращает сообщения для статуса."""
        vol = vol or panel.vol
        msgs = []
        for obj in self.objects:
            try:
                n = attach_object(vol, obj)
            except Exception as ex:
                msgs.append(f"{obj.kind}: {ex}")
                continue
            if n:
                what = {"SEG": "сегментов", "PR": "элементов аннотаций", "SR": "элементов отчёта"}[obj.kind]
                msgs.append(f"{obj.kind} «{obj.desc or Path(obj.path).name}»: {n} {what}")
        return msgs

    def _annot_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Найденные объекты DICOM (SR, PR, SEG):"))
        self.obj_list = QListWidget()
        lay.addWidget(self.obj_list, 1)
        row = QHBoxLayout()
        b1 = QPushButton("Применить к окну")
        b1.clicked.connect(self.apply_selected_object)
        b2 = QPushButton("Показать отчёт SR…")
        b2.clicked.connect(self.show_sr)
        row.addWidget(b1)
        row.addWidget(b2)
        lay.addLayout(row)
        self.chk_annots = QCheckBox("Показывать аннотации (PR / SR)")
        self.chk_annots.setChecked(True)
        self.chk_annots.toggled.connect(lambda on: self._set_flag("show_annots", on))
        self.chk_seg = QCheckBox("Показывать сегментации (SEG)")
        self.chk_seg.setChecked(True)
        self.chk_seg.toggled.connect(lambda on: self._set_flag("show_seg", on))
        lay.addWidget(self.chk_annots)
        lay.addWidget(self.chk_seg)
        self.lbl_annot = QLabel("")
        self.lbl_annot.setWordWrap(True)
        lay.addWidget(self.lbl_annot)
        return w

    def _set_flag(self, name, on):
        for p in self.panels:
            setattr(p, name, on)
            p.refresh()

    def _fill_objects(self):
        self.obj_list.clear()
        for o in self.objects:
            self.obj_list.addItem(o.label())

    def _refresh_annot_info(self):
        v = self.active.vol
        if v is None:
            self.lbl_annot.setText("")
            return
        parts = []
        n = sum(len(x) for x in v.annots.values())
        if n:
            parts.append(f"Аннотации на срезах: {n}")
        if v.seg is not None and v.seg_names:
            parts.append("Сегменты: " + ", ".join(
                f'<span style="color:rgb{tuple(int(c) for c in v.seg_lut[k])}">■</span> {name}'
                for k, name in v.seg_names.items()))
        self.lbl_annot.setText("<br>".join(parts))

    def apply_selected_object(self):
        row = self.obj_list.currentRow()
        p = self.active
        if not (0 <= row < len(self.objects)) or p.vol is None:
            return QMessageBox.information(self, "Аннотации", "Откройте серию и выберите объект в списке")
        obj = self.objects[row]
        if obj.kind in ("PR", "SR"):                # повторное применение не должно дублировать
            p.vol.annots.clear()
        try:
            n = attach_object(p.vol, obj)
        except Exception as ex:
            return QMessageBox.warning(self, "Аннотации", str(ex))
        if not n:
            return QMessageBox.information(self, "Аннотации", "Объект не относится к открытой серии")
        p.refresh()
        self._refresh_annot_info()
        self.statusBar().showMessage(f"Применено: {obj.label()}", 8000)

    def show_sr(self):
        row = self.obj_list.currentRow()
        if not (0 <= row < len(self.objects)) or self.objects[row].kind != "SR":
            return QMessageBox.information(self, "Отчёт SR", "Выберите в списке объект типа «Отчёт SR»")
        try:
            info = parse_object(self.objects[row])
        except Exception as ex:
            return QMessageBox.warning(self, "Отчёт SR", str(ex))
        dlg = QDialog(self)
        dlg.setWindowTitle(self.objects[row].label())
        dlg.resize(720, 520)
        lay = QVBoxLayout(dlg)
        view = QPlainTextEdit(info["text"])
        view.setReadOnly(True)
        lay.addWidget(view)
        dlg.exec()

    # --- сохранение результатов
    def _need_volume(self):
        if self.active.vol is None:
            QMessageBox.information(self, "MRT Viewer", "Сначала откройте серию")
            return False
        return True

    def _default_name(self, ext):
        p = self.active
        base = "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in (p.series.desc if p.series else "series")) or "series"
        return f"{base}_{datetime.now():%Y%m%d_%H%M}{ext}"

    def ui_save_session(self):
        if not self._need_volume():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить измерения и маску",
                                              self._default_name(".json"), "JSON (*.json)")
        if path:
            try:
                d = save_session(self.active, path)
            except Exception as ex:
                return QMessageBox.critical(self, "Сохранение", f"Ошибка: {ex}")
            self.statusBar().showMessage(
                f"Сохранено: {len(d['measurements'])} измерений" + (", маска" if d["mask"] else ""), 8000)

    def ui_load_session(self):
        if not self._need_volume():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить сохранённые измерения", "", "JSON (*.json)")
        if path:
            try:
                n, mask = load_session(self.active, path)
            except Exception as ex:
                return QMessageBox.warning(self, "Загрузка", str(ex))
            self.statusBar().showMessage(f"Загружено: {n} измерений" + (", маска" if mask else ""), 8000)

    def ui_export_csv(self):
        if not self._need_volume():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт измерений",
                                              self._default_name(".csv"), "CSV (*.csv)")
        if path:
            try:
                n = export_csv(self.active, path)
            except Exception as ex:
                return QMessageBox.critical(self, "Экспорт", f"Ошибка: {ex}")
            self.statusBar().showMessage(f"Экспортировано строк: {n}", 8000)

    def ui_report(self):
        if not self._need_volume():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Создать отчёт",
                                              self._default_name(".pdf"), "PDF (*.pdf)")
        if path:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                n = build_report(self.active, path)
            except Exception as ex:
                QApplication.restoreOverrideCursor()
                return QMessageBox.critical(self, "Отчёт", f"Ошибка: {ex}")
            QApplication.restoreOverrideCursor()
            self.statusBar().showMessage(f"Отчёт создан (измерений: {n}): {path}", 10000)

    def closeEvent(self, e):
        self.play_timer.stop()
        for p in self.panels:
            p.release_all()
        super().closeEvent(e)

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
    _purge_stale_cache()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.resize(1500, 900)
    w.show()
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):   # python mrtViewer.py <папка>
        w.load_folder(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()