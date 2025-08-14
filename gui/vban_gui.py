#!/usr/bin/env python3
import os
import sys
import threading
import subprocess
from PyQt6 import QtWidgets, QtCore, QtGui
import json


class Worker(QtCore.QThread):
    line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(int)

    def __init__(self, cmd: list[str], parent=None):
        super().__init__(parent)
        self.cmd = cmd
        self.proc: subprocess.Popen | None = None

    def run(self):
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.line.emit(line)
        rc = self.proc.wait()
        self.finished.emit(rc)

    def stop(self):
        try:
            if self.proc:
                self.proc.terminate()
        except Exception:
            pass


class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VBAN → BlackHole")
        self.resize(820, 560)

        self.worker: Worker | None = None

        self._build_ui()

    @staticmethod
    def resource_path(rel_path: str) -> str:
        # Resolve path to bundled resources when frozen by PyInstaller
        base = getattr(sys, "_MEIPASS", None)
        if base:
            return os.path.join(base, rel_path)
        return os.path.join(os.path.dirname(__file__), "..", rel_path)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QGridLayout()
        row = 0
        form.addWidget(QtWidgets.QLabel("Listen IP:"), row, 0)
        self.listen_ip = QtWidgets.QLineEdit("0.0.0.0")
        form.addWidget(self.listen_ip, row, 1)
        form.addWidget(QtWidgets.QLabel("Port:"), row, 2)
        self.listen_port = QtWidgets.QLineEdit("6980")
        form.addWidget(self.listen_port, row, 3)

        row += 1
        form.addWidget(QtWidgets.QLabel("Output Device:"), row, 0)
        self.output_device = QtWidgets.QLineEdit("BlackHole 16ch")
        form.addWidget(self.output_device, row, 1, 1, 3)

        row += 1
        form.addWidget(QtWidgets.QLabel("Jitter (ms):"), row, 0)
        self.jitter_ms = QtWidgets.QLineEdit("20")
        form.addWidget(self.jitter_ms, row, 1)
        form.addWidget(QtWidgets.QLabel("Blocksize:"), row, 2)
        self.blocksize = QtWidgets.QLineEdit("1024")
        form.addWidget(self.blocksize, row, 3)

        row += 1
        form.addWidget(QtWidgets.QLabel("Map (comma-separated):"), row, 0)
        self.map_entry = QtWidgets.QLineEdit()
        form.addWidget(self.map_entry, row, 1, 1, 3)

        row += 1
        self.starve_fill = QtWidgets.QCheckBox("Starve fill")
        self.starve_fill.setChecked(True)
        self.show_jitter = QtWidgets.QCheckBox("Show jitter")
        self.show_jitter.setChecked(True)
        self.verbose = QtWidgets.QCheckBox("Verbose")
        form.addWidget(self.starve_fill, row, 0)
        form.addWidget(self.show_jitter, row, 1)
        form.addWidget(self.verbose, row, 2)

        layout.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.list_btn = QtWidgets.QPushButton("List Devices")
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.list_btn.clicked.connect(self.on_list_devices)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.list_btn)
        layout.addLayout(btns)

        # Stats area: jitter labels and per-channel bars
        stats = QtWidgets.QVBoxLayout()
        self.jitter_label = QtWidgets.QLabel("Jitter p95/max: -- / -- ms")
        stats.addWidget(self.jitter_label)
        self.bars = QtWidgets.QGridLayout()
        self.bars.setHorizontalSpacing(10)
        self.bars.setVerticalSpacing(6)
        stats.addLayout(self.bars)
        layout.addLayout(stats, 1)
        self.bar_widgets: list[VUBar] = []
        self.db_labels: list[QtWidgets.QLabel] = []

    # ===== Backend → UI bridging =====
    def update_stats(self, obj: dict):
        if obj.get("type") == "stats":
            p95 = obj.get("jitter_p95_ms", 0.0)
            mx = obj.get("jitter_max_ms", 0.0)
            self.jitter_label.setText(f"Jitter p95/max: {p95:.2f} / {mx:.2f} ms")
        elif obj.get("type") == "vu":
            levels = obj.get("levels", [])
            ch = obj.get("channels", len(levels))
            # Create bars lazily if channel count changes
            if len(self.bar_widgets) != ch:
                # Clear
                for i in reversed(range(self.bars.count())):
                    w = self.bars.itemAt(i).widget()
                    if w:
                        w.setParent(None)
                self.bar_widgets.clear()
                self.db_labels.clear()
                for i in range(ch):
                    lbl = QtWidgets.QLabel(f"Ch {i+1:02d}")
                    bar = VUBar()
                    db_lbl = QtWidgets.QLabel("-60.0 dB  P:-60.0 dB")
                    db_lbl.setMinimumWidth(120)
                    self.bars.addWidget(lbl, i, 0)
                    self.bars.addWidget(bar, i, 1)
                    self.bars.addWidget(db_lbl, i, 2)
                    self.bar_widgets.append(bar)
                    self.db_labels.append(db_lbl)
            # Update values
            import math
            for i, st in enumerate(levels[:len(self.bar_widgets)]):
                lvl = float(st.get("level", 0.0))
                pk = float(st.get("peak", 0.0))
                self.bar_widgets[i].set_values(lvl, pk)
                def to_db(x: float) -> float:
                    if x <= 0.0:
                        return -60.0
                    return 20.0 * math.log10(max(1e-6, x))
                self.db_labels[i].setText(f"{to_db(lvl):6.1f} dB  P:{to_db(pk):6.1f} dB")

    def on_list_devices(self):
        exe = sys.executable
        script = self.resource_path("vban_to_blackhole16.py")
        cmd = [exe, script, "--list-devices"]
        try:
            out = subprocess.check_output(cmd, text=True)
        except subprocess.CalledProcessError as e:
            out = e.output or str(e)
        # Show as dialog to avoid console widget
        QtWidgets.QMessageBox.information(self, "Devices", out or "(no output)")

    def on_start(self):
        if self.worker is not None:
            return
        exe = sys.executable
        script = self.resource_path("vban_to_blackhole16.py")
        cmd = [
            exe, script,
            "--listen-ip", self.listen_ip.text().strip() or "0.0.0.0",
            "--listen-port", self.listen_port.text().strip() or "6980",
            "--output-device", self.output_device.text().strip() or "BlackHole 16ch",
            "--jitter-ms", self.jitter_ms.text().strip() or "20",
            "--device-blocksize", self.blocksize.text().strip() or "1024",
            "--starve-fill",
            "--show-jitter",
            "--json",
        ]
        if not self.starve_fill.isChecked():
            cmd.remove("--starve-fill")
        if not self.show_jitter.isChecked():
            cmd.remove("--show-jitter")
        if self.verbose.isChecked():
            cmd.append("--verbose")
        map_str = self.map_entry.text().strip()
        if map_str:
            cmd += ["--map", map_str]

        self.worker = Worker(cmd)
        self.worker.line.connect(self.on_backend_line)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def on_stop(self):
        if self.worker is None:
            return
        self.worker.stop()
        self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_finished(self, rc: int):
        # Reset UI on stop
        self.jitter_label.setText("Jitter p95/max: -- / -- ms")
        for bar in self.bar_widgets:
            bar.set_values(0.0, 0.0)
        self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @QtCore.pyqtSlot(str)
    def on_backend_line(self, line: str):
        line = line.strip()
        if not line:
            return
        try:
            obj = json.loads(line)
        except Exception:
            return
        self.update_stats(obj)


class VUBar(QtWidgets.QWidget):
    """Custom bar approximating the console VU meter with a peak marker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.level_lin: float = 0.0
        self.peak_lin: float = 0.0
        self.setMinimumHeight(18)

    @staticmethod
    def lin_to_bar(x: float) -> float:
        # Map linear amplitude to bar fraction like console (-60 dB to 0 dB)
        if x <= 0.0:
            return 0.0
        import math
        db = 20.0 * math.log10(max(1e-6, x))
        return max(0.0, min(1.0, (db + 60.0) / 60.0))

    def set_values(self, level_lin: float, peak_lin: float):
        self.level_lin = max(0.0, float(level_lin))
        self.peak_lin = max(0.0, float(peak_lin))
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        rect = self.rect().adjusted(0, 6, 0, -6)
        # Draw track
        track = QtGui.QColor(80, 80, 80)
        p.fillRect(rect, track)
        # Draw filled bar with intensity shading (first 70% solid, then softer)
        frac = self.lin_to_bar(self.level_lin)
        width = int(rect.width() * frac)
        if width > 0:
            solid_w = int(width * 0.7)
            mid_w = int(width * 0.9)
            col_solid = QtGui.QColor(0, 122, 255)
            col_mid = QtGui.QColor(60, 160, 255)
            col_light = QtGui.QColor(140, 200, 255)
            if solid_w > 0:
                p.fillRect(QtCore.QRect(rect.left(), rect.top(), solid_w, rect.height()), col_solid)
            if mid_w - solid_w > 0:
                p.fillRect(QtCore.QRect(rect.left() + solid_w, rect.top(), mid_w - solid_w, rect.height()), col_mid)
            if width - mid_w > 0:
                p.fillRect(QtCore.QRect(rect.left() + mid_w, rect.top(), width - mid_w, rect.height()), col_light)
        # Draw peak marker
        peak_frac = self.lin_to_bar(self.peak_lin)
        peak_x = rect.left() + int(rect.width() * peak_frac)
        pen = QtGui.QPen(QtGui.QColor(255, 165, 0))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(peak_x, rect.top(), peak_x, rect.bottom())
        p.end()

    def update_stats(self, obj: dict):
        if obj.get("type") == "stats":
            p95 = obj.get("jitter_p95_ms", 0.0)
            mx = obj.get("jitter_max_ms", 0.0)
            self.jitter_label.setText(f"Jitter p95/max: {p95:.2f} / {mx:.2f} ms")
        elif obj.get("type") == "vu":
            levels = obj.get("levels", [])
            ch = obj.get("channels", len(levels))
            # Create bars lazily if channel count changes
            if len(self.bar_widgets) != ch:
                # Clear
                for i in reversed(range(self.bars.count())):
                    w = self.bars.itemAt(i).widget()
                    if w:
                        w.setParent(None)
                self.bar_widgets.clear()
                self.db_labels.clear()
                for i in range(ch):
                    lbl = QtWidgets.QLabel(f"Ch {i+1:02d}")
                    bar = VUBar()
                    db_lbl = QtWidgets.QLabel("-60.0 dB  P:-60.0 dB")
                    db_lbl.setMinimumWidth(120)
                    self.bars.addWidget(lbl, i, 0)
                    self.bars.addWidget(bar, i, 1)
                    self.bars.addWidget(db_lbl, i, 2)
                    self.bar_widgets.append(bar)
                    self.db_labels.append(db_lbl)
            # Update values (convert linear [0..1] to dB scale 0..1000)
            import math
            for i, st in enumerate(levels[:len(self.bar_widgets)]):
                lvl = float(st.get("level", 0.0))
                pk = float(st.get("peak", 0.0))
                self.bar_widgets[i].set_values(lvl, pk)
                def to_db(x: float) -> float:
                    if x <= 0.0:
                        return -60.0
                    return 20.0 * math.log10(max(1e-6, x))
                self.db_labels[i].setText(f"{to_db(lvl):6.1f} dB  P:{to_db(pk):6.1f} dB")

    def on_list_devices(self):
        exe = sys.executable
        script = os.path.join(os.path.dirname(__file__), "..", "vban_to_blackhole16.py")
        cmd = [exe, script, "--list-devices"]
        try:
            out = subprocess.check_output(cmd, text=True)
        except subprocess.CalledProcessError as e:
            out = e.output or str(e)
        self.append_output(out)

    def on_start(self):
        if self.worker is not None:
            return
        exe = sys.executable
        script = os.path.join(os.path.dirname(__file__), "..", "vban_to_blackhole16.py")
        cmd = [
            exe, script,
            "--listen-ip", self.listen_ip.text().strip() or "0.0.0.0",
            "--listen-port", self.listen_port.text().strip() or "6980",
            "--output-device", self.output_device.text().strip() or "BlackHole 16ch",
            "--jitter-ms", self.jitter_ms.text().strip() or "20",
            "--device-blocksize", self.blocksize.text().strip() or "1024",
            "--starve-fill",
            "--show-jitter",
        ]
        if not self.starve_fill.isChecked():
            cmd.remove("--starve-fill")
        if not self.show_jitter.isChecked():
            cmd.remove("--show-jitter")
        if self.verbose.isChecked():
            cmd.append("--verbose")
        map_str = self.map_entry.text().strip()
        if map_str:
            cmd += ["--map", map_str]

        # Force JSON output from backend for GUI consumption
        cmd.append("--json")
        self.worker = Worker(cmd)
        self.worker.line.connect(self.on_backend_line)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def on_stop(self):
        if self.worker is None:
            return
        self.worker.stop()
        self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_finished(self, rc: int):
        # Reset UI on stop
        self.jitter_label.setText("Jitter p95/max: -- / -- ms")
        for bar in self.bar_widgets:
            bar.setValue(0)
        for pk in self.peak_widgets:
            pk.setValue(0)
        self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @QtCore.pyqtSlot(str)
    def on_backend_line(self, line: str):
        line = line.strip()
        if not line:
            return
        try:
            obj = json.loads(line)
        except Exception:
            return
        self.update_stats(obj)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


