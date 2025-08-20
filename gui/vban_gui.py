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

    @staticmethod
    def backend_path() -> str:
        # Prefer bundled backend binary if available
        base = getattr(sys, "_MEIPASS", None)
        if base:
            cand = os.path.join(base, "backend", "vban-backend")
            if os.path.exists(cand):
                return cand
        # Fallback to Python script
        return App.resource_path("vban_to_blackhole16.py")

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
        form.addWidget(self.starve_fill, row, 0)

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


        
        # VU meter bars - main section
        vu_section = QtWidgets.QVBoxLayout()
        vu_section.addWidget(QtWidgets.QLabel("VU Meters:"))
        
        self.bars = QtWidgets.QGridLayout()
        self.bars.setHorizontalSpacing(10)
        self.bars.setVerticalSpacing(6)
        vu_section.addLayout(self.bars)
        layout.addLayout(vu_section)
        
        # Status bar below VU meters - truly responsive horizontal layout
        status_bar = QtWidgets.QHBoxLayout()
        status_bar.setSpacing(20)
        
        # Column 1: Jitter (expandable)
        jitter_section = QtWidgets.QHBoxLayout()
        jitter_section.setSpacing(5)
        jitter_section.addWidget(QtWidgets.QLabel("Jitter:"))
        self.jitter_p95_label = QtWidgets.QLabel("--")
        self.jitter_max_label = QtWidgets.QLabel("--")
        self.jitter_p95_label.setMinimumWidth(50)
        self.jitter_max_label.setMinimumWidth(50)
        self.jitter_p95_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.jitter_max_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        jitter_section.addWidget(self.jitter_p95_label)
        jitter_section.addWidget(QtWidgets.QLabel("/"))
        jitter_section.addWidget(self.jitter_max_label)
        jitter_section.addWidget(QtWidgets.QLabel("ms"))
        jitter_section.addStretch()  # Allow this section to expand
        status_bar.addLayout(jitter_section, 1)  # Stretch factor 1
        
        # Column 2: Bitrate (expandable)
        bitrate_section = QtWidgets.QHBoxLayout()
        bitrate_section.setSpacing(5)
        bitrate_section.addWidget(QtWidgets.QLabel("Bitrate:"))
        self.bitrate_value_label = QtWidgets.QLabel("--")
        self.bitrate_value_label.setMinimumWidth(70)
        self.bitrate_value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        bitrate_section.addWidget(self.bitrate_value_label)
        bitrate_section.addWidget(QtWidgets.QLabel("Mbps"))
        bitrate_section.addStretch()  # Allow this section to expand
        status_bar.addLayout(bitrate_section, 1)  # Stretch factor 1
        
        # Column 3: Packets (expandable)
        packets_section = QtWidgets.QHBoxLayout()
        packets_section.setSpacing(5)
        packets_section.addWidget(QtWidgets.QLabel("Packets:"))
        self.packets_received_label = QtWidgets.QLabel("--")
        self.packets_lost_label = QtWidgets.QLabel("--")
        self.packets_dup_label = QtWidgets.QLabel("--")
        self.packet_loss_rate_label = QtWidgets.QLabel("--")
        self.packets_received_label.setMinimumWidth(60)
        self.packets_lost_label.setMinimumWidth(40)
        self.packets_dup_label.setMinimumWidth(40)
        self.packet_loss_rate_label.setMinimumWidth(50)
        self.packets_received_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.packets_lost_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.packets_dup_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.packet_loss_rate_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        packets_section.addWidget(self.packets_received_label)
        packets_section.addWidget(QtWidgets.QLabel("R/"))
        packets_section.addWidget(self.packets_lost_label)
        packets_section.addWidget(QtWidgets.QLabel("L/"))
        packets_section.addWidget(self.packets_dup_label)
        packets_section.addWidget(QtWidgets.QLabel("D/"))
        packets_section.addWidget(self.packet_loss_rate_label)
        packets_section.addWidget(QtWidgets.QLabel("%"))
        packets_section.addStretch()  # Allow this section to expand
        status_bar.addLayout(packets_section, 1)  # Stretch factor 1
        
        # Column 4: Rate (expandable)
        rate_section = QtWidgets.QHBoxLayout()
        rate_section.setSpacing(5)
        rate_section.addWidget(QtWidgets.QLabel("Rate:"))
        self.packet_rate_label = QtWidgets.QLabel("--")
        self.packet_rate_label.setMinimumWidth(60)
        self.packet_rate_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        rate_section.addWidget(self.packet_rate_label)
        rate_section.addWidget(QtWidgets.QLabel("/s"))
        rate_section.addStretch()  # Allow this section to expand
        status_bar.addLayout(rate_section, 1)  # Stretch factor 1
        
        layout.addLayout(status_bar)
        self.bar_widgets: list[VUBar] = []
        self.db_labels: list[tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = []
        
        # Create 16 VU meters by default showing minimal values
        self.create_default_vu_meters()

    def create_default_vu_meters(self):
        """Create 16 VU meters by default showing minimal values."""
        # Create 16 channels by default
        for i in range(16):
            lbl = QtWidgets.QLabel(f"Ch {i+1:02d}")
            bar = VUBar()
            
            # Split VU meter labels into separate fields to prevent text jumping
            db_container = QtWidgets.QWidget()
            db_layout = QtWidgets.QHBoxLayout(db_container)
            db_layout.setContentsMargins(0, 0, 0, 0)
            
            db_layout.addWidget(QtWidgets.QLabel("Level:"))
            db_level_label = QtWidgets.QLabel("-60.0 dB")
            db_level_label.setMinimumWidth(70)
            db_layout.addWidget(db_level_label)
            
            db_layout.addWidget(QtWidgets.QLabel("Peak:"))
            db_peak_label = QtWidgets.QLabel("-60.0 dB")
            db_peak_label.setMinimumWidth(70)
            db_layout.addWidget(db_peak_label)
            
            # Store both labels for updating
            self.db_labels.append((db_level_label, db_peak_label))
            
            self.bars.addWidget(lbl, i, 0)
            self.bars.addWidget(bar, i, 1)
            self.bars.addWidget(db_container, i, 2)
            self.bar_widgets.append(bar)
            
            # Set minimal values (essentially silent)
            bar.set_values(0.001, 0.001)

    # ===== Backend → UI bridging =====
    def update_stats(self, obj: dict):
        if obj.get("type") == "stats":
            p95 = obj.get("jitter_p95_ms", 0.0)
            mx = obj.get("jitter_max_ms", 0.0)
            self.jitter_p95_label.setText(f"{p95:.2f}")
            self.jitter_max_label.setText(f"{mx:.2f}")
        elif obj.get("type") == "vu":
            # Update network stats
            bitrate = obj.get("bitrate_mbps", 0.0)
            packets_received = obj.get("packets_received", 0)
            lost_packets = obj.get("lost_packets", 0)
            duplicate_packets = obj.get("duplicate_packets", 0)
            packet_loss_rate = obj.get("packet_loss_rate", 0.0)
            current_packet_rate = obj.get("current_packet_rate", 0.0)
            
            self.bitrate_value_label.setText(f"{bitrate:.2f}")
            self.packets_received_label.setText(f"{packets_received}")
            self.packets_lost_label.setText(f"{lost_packets}")
            self.packets_dup_label.setText(f"{duplicate_packets}")
            self.packet_loss_rate_label.setText(f"{packet_loss_rate:.2f}")
            self.packet_rate_label.setText(f"{current_packet_rate:.1f}")
            
            levels = obj.get("levels", [])
            ch = obj.get("channels", len(levels))
            # Create bars lazily if channel count changes
            if len(self.bar_widgets) != ch:
                # Clear existing bars (including test ones)
                for i in reversed(range(self.bars.count())):
                    w = self.bars.itemAt(i).widget()
                    if w:
                        w.setParent(None)
                self.bar_widgets.clear()
                self.db_labels.clear()
                for i in range(ch):
                    lbl = QtWidgets.QLabel(f"Ch {i+1:02d}")
                    bar = VUBar()
                    
                    # Split VU meter labels into separate fields to prevent text jumping
                    db_container = QtWidgets.QWidget()
                    db_layout = QtWidgets.QHBoxLayout(db_container)
                    db_layout.setContentsMargins(0, 0, 0, 0)
                    
                    db_layout.addWidget(QtWidgets.QLabel("Level:"))
                    db_level_label = QtWidgets.QLabel("-60.0 dB")
                    db_level_label.setMinimumWidth(70)
                    db_layout.addWidget(db_level_label)
                    
                    db_layout.addWidget(QtWidgets.QLabel("Peak:"))
                    db_peak_label = QtWidgets.QLabel("-60.0 dB")
                    db_peak_label.setMinimumWidth(70)
                    db_layout.addWidget(db_peak_label)
                    
                    # Store both labels for updating
                    self.db_labels.append((db_level_label, db_peak_label))
                    
                    self.bars.addWidget(lbl, i, 0)
                    self.bars.addWidget(bar, i, 1)
                    self.bars.addWidget(db_container, i, 2)
                    self.bar_widgets.append(bar)
            
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
                
                # Update separate level and peak labels
                level_label, peak_label = self.db_labels[i]
                level_label.setText(f"{to_db(lvl):6.1f} dB")
                peak_label.setText(f"{to_db(pk):6.1f} dB")

    def on_list_devices(self):
        backend = self.backend_path()
        if backend.endswith("vban-backend"):
            cmd = [backend, "--list-devices"]
        else:
            cmd = [sys.executable, backend, "--list-devices"]
        try:
            out = subprocess.check_output(cmd, text=True)
        except subprocess.CalledProcessError as e:
            out = e.output or str(e)
        # Show as dialog to avoid console widget
        QtWidgets.QMessageBox.information(self, "Devices", out or "(no output)")

    def on_start(self):
        if self.worker is not None:
            return
        backend = self.backend_path()
        cmd = ([backend] if backend.endswith("vban-backend") else [sys.executable, backend]) + [
            "--listen-ip", self.listen_ip.text().strip() or "0.0.0.0",
            "--listen-port", self.listen_port.text().strip() or "6980",
            "--output-device", self.output_device.text().strip() or "BlackHole 16ch",
            "--jitter-ms", self.jitter_ms.text().strip() or "20",
            "--device-blocksize", self.blocksize.text().strip() or "1024",
            "--starve-fill",
            "--show-jitter",
            "--show-network",
            "--json",
        ]
        if not self.starve_fill.isChecked():
            cmd.remove("--starve-fill")
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
        self.worker.wait()  # Wait for thread to finish
        self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_finished(self, rc: int):
        # Reset UI on stop
        self.jitter_p95_label.setText("--")
        self.jitter_max_label.setText("--")
        self.bitrate_value_label.setText("--")
        self.packets_received_label.setText("--")
        self.packets_lost_label.setText("--")
        self.packets_dup_label.setText("--")
        self.packet_loss_rate_label.setText("--")
        self.packet_rate_label.setText("--")
        
        for bar in self.bar_widgets:
            bar.set_values(0.0, 0.0)
        
        # Reset VU meter labels
        for level_label, peak_label in self.db_labels:
            level_label.setText("-60.0 dB")
            peak_label.setText("-60.0 dB")
        
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




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


