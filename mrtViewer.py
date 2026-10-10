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
from PySide6.QtCore import QPoint, QProcess, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget, QAbstractItemView,
)


@dataclass
class Series:
    uid: str
    patient: str
    desc: str
    number: int
    modality: str
    items: list = field(default_factory=list)  # [(path, frame_index)]

    def label(self):
        return f"#{self.number} {self.modality} {self.desc} ({len(self.items)})"


def slice_key(ds):
    """Ключ сортировки среза: проекция позиции на нормаль, иначе InstanceNumber."""
    try:
        ipp = np.array(ds.ImagePositionPatient, dtype=float)
        iop = np.array(ds.ImageOrientationPatient, dtype=float)
        normal = np.cross(iop[:3], iop[3:])
        return float(np.dot(ipp, normal))
    except Exception:
        return float(ds.get("InstanceNumber", 0) or 0)


def scan_folder(folder):
    """Рекурсивно сканирует папку и группирует файлы по SeriesInstanceUID."""
    tags = ["SeriesInstanceUID", "PatientName", "SeriesDescription",
            "SeriesNumber", "Modality", "ImagePositionPatient",
            "ImageOrientationPatient", "InstanceNumber", "NumberOfFrames"]
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


class SliceStore:
    """Ленивая загрузка срезов с LRU-кэшем."""

    def __init__(self, items, cap=200):
        self.items = items
        self.cache = OrderedDict()
        self.cap = cap

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
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
        info = {"wc": wc, "ww": ww,
                "mono1": ds.get("PhotometricInterpretation") == "MONOCHROME1"}
        self.cache[path] = (arr, info)
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return self.cache[path]

    def get(self, i):
        path, frame = self.items[i]
        arr, info = self._load(path)
        return arr[min(frame, arr.shape[0] - 1)], info

class Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 400)
        self.store = None
        self.idx = 0
        self.wc = self.ww = None
        self.zoom = 1.0
        self.pan = QPoint(0, 0)
        self.title = ""
        self.msg = "Откройте папку с DICOM или загрузите исследование из PACS"
        self._img = None
        self._buf = None
        self._last = QPoint()
        self._btn = None

    def set_series(self, s):
        self.store = SliceStore(s.items)
        self.idx = len(s.items) // 2
        self.wc = self.ww = None
        self.zoom, self.pan = 1.0, QPoint(0, 0)
        self.title = f"{s.patient} | {s.label()}"
        self.render()

    def render(self):
        if not self.store or not len(self.store):
            self._img = None
            return self.update()
        try:
            arr, info = self.store.get(self.idx)
            if self.wc is None:
                if info["wc"] is not None and info["ww"]:
                    self.wc, self.ww = info["wc"], info["ww"]
                else:
                    lo, hi = np.percentile(arr, [1, 99])
                    self.wc, self.ww = (lo + hi) / 2, max(hi - lo, 1.0)
            lo = self.wc - self.ww / 2
            img = np.clip((arr - lo) / max(self.ww, 1e-6), 0, 1) * 255
            img = img.astype(np.uint8)
            if info["mono1"]:
                img = 255 - img
            h, w = img.shape
            self._buf = np.ascontiguousarray(img)   # держим ссылку на буфер
            self._img = QImage(self._buf.data, w, h, w,
                               QImage.Format.Format_Grayscale8)
        except Exception as ex:
            self._img = None
            self.msg = f"Ошибка: {ex}"
        self.update()

    def step(self, d):
        if not self.store:
            return
        self.idx = max(0, min(len(self.store) - 1, self.idx + d))
        self.render()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._img is None:
            p.setPen(QColor("#888"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.msg)
            return
        iw, ih = self._img.width(), self._img.height()
        fit = min(self.width() / iw, self.height() / ih) * self.zoom
        w, h = iw * fit, ih * fit
        x = (self.width() - w) / 2 + self.pan.x()
        y = (self.height() - h) / 2 + self.pan.y()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(QRectF(x, y, w, h), self._img)
        p.setPen(QColor("#ffd54a"))
        p.drawText(10, 20, self.title)
        p.drawText(10, self.height() - 30,
                   f"Срез {self.idx + 1}/{len(self.store)}")
        p.drawText(10, self.height() - 12,
                   f"W {self.ww:.0f}  L {self.wc:.0f}  zoom {self.zoom:.2f}")

    def wheelEvent(self, e):
        dy = e.angleDelta().y()
        if not self.store or dy == 0:
            return
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            k = 1.1 if dy > 0 else 1 / 1.1
            self.zoom = min(20.0, max(0.1, self.zoom * k))
            self.update()
        else:
            self.step(-1 if dy > 0 else 1)

    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key.Key_Up, Qt.Key.Key_Left):
            self.step(-1)
        elif k in (Qt.Key.Key_Down, Qt.Key.Key_Right):
            self.step(1)
        elif k == Qt.Key.Key_PageUp:
            self.step(-10)
        elif k == Qt.Key.Key_PageDown:
            self.step(10)

    def mousePressEvent(self, e):
        self.setFocus()
        self._last = e.position().toPoint()
        self._btn = e.button()

    def mouseMoveEvent(self, e):
        p = e.position().toPoint()
        d = p - self._last
        self._last = p
        if self._img is None:
            return
        if self._btn == Qt.MouseButton.LeftButton:
            k = max(self.ww / 400.0, 0.05)
            self.ww = max(1.0, self.ww + d.x() * k)
            self.wc += d.y() * k
            self.render()
        elif self._btn == Qt.MouseButton.RightButton:
            self.pan += d
            self.update()

    def mouseDoubleClickEvent(self, _):
        self.wc = self.ww = None
        self.zoom, self.pan = 1.0, QPoint(0, 0)
        self.render()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI Viewer — DICOM / PACS / DCMTK")
        self.viewer = Viewer()
        self.series = []
        self.proc = None
        self.find_dir = None

        tabs = QTabWidget()
        tabs.addTab(self._local_tab(), "Локально")
        tabs.addTab(self._pacs_tab(), "PACS")

        self.series_list = QListWidget()
        self.series_list.currentRowChanged.connect(self.open_series)

        left = QWidget()
        lay = QVBoxLayout(left)
        lay.addWidget(tabs)
        lay.addWidget(QLabel("Серии:"))
        lay.addWidget(self.series_list, 1)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.viewer)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 900])
        self.setCentralWidget(split)

    def _local_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        btn = QPushButton("Открыть папку с DICOM…")
        btn.clicked.connect(self.open_folder)
        lay.addWidget(btn)
        lay.addStretch(1)
        return w

    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Папка с DICOM")
        if d:
            self.load_folder(d)

    def load_folder(self, folder):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.series = scan_folder(folder)
        finally:
            QApplication.restoreOverrideCursor()
        self.series_list.clear()
        for s in self.series:
            self.series_list.addItem(s.label())
        if self.series:
            self.series_list.setCurrentRow(0)
        else:
            QMessageBox.information(self, "DICOM", "DICOM-файлы не найдены")

    def open_series(self, row):
        if 0 <= row < len(self.series):
            self.viewer.set_series(self.series[row])
            self.viewer.setFocus()

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
            self.load_folder(out)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.resize(1300, 850)
    w.show()
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):   # python mri_viewer.py <папка>
        w.load_folder(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()