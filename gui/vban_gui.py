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
        self.device_info_cache = {}  # Cache device information for channel detection
        self.current_channel_mapping = ""  # Store current channel mapping configuration

        self._build_ui()

    def debug_print(self, message: str):
        """Print debug message only if verbose mode is enabled"""
        if self.verbose.isChecked():
            print(f"DEBUG: {message}")

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
        self.output_device = QtWidgets.QComboBox()
        self.output_device.setEditable(True)  # Allow custom device names
        self.output_device.setMinimumWidth(300)
        form.addWidget(self.output_device, row, 1, 1, 3)

        row += 1
        form.addWidget(QtWidgets.QLabel("Jitter (ms):"), row, 0)
        self.jitter_ms = QtWidgets.QLineEdit("20")
        form.addWidget(self.jitter_ms, row, 1)
        form.addWidget(QtWidgets.QLabel("Blocksize:"), row, 2)
        self.blocksize = QtWidgets.QLineEdit("1024")
        form.addWidget(self.blocksize, row, 3)

        row += 1
        form.addWidget(QtWidgets.QLabel("Channel Mapping:"), row, 0)
        self.map_btn = QtWidgets.QPushButton("Configure Channel Map")
        self.map_btn.clicked.connect(self.open_channel_mapping)
        form.addWidget(self.map_btn, row, 1, 1, 3)

        row += 1
        self.starve_fill = QtWidgets.QCheckBox("Starve fill")
        self.starve_fill.setChecked(True)
        form.addWidget(self.starve_fill, row, 0)
        
        # Add verbose flag checkbox
        self.verbose = QtWidgets.QCheckBox("Verbose")
        self.verbose.setChecked(False)
        form.addWidget(self.verbose, row, 1)

        layout.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.refresh_btn = QtWidgets.QPushButton("Refresh Sources")
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.refresh_btn.clicked.connect(self.on_refresh_sources)
        

        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.refresh_btn)
        layout.addLayout(btns)


        
        # VU meter bars - main section
        vu_section = QtWidgets.QVBoxLayout()
        vu_section.addWidget(QtWidgets.QLabel("VU Meters:"))
        
        self.bars = QtWidgets.QGridLayout()
        self.bars.setHorizontalSpacing(10)
        self.bars.setVerticalSpacing(6)
        vu_section.addLayout(self.bars)
        layout.addLayout(vu_section)
        
        # Divider between VU meters and stats
        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        divider.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        divider.setStyleSheet("background-color: #bdc3c7; margin: 5px 0px;")
        layout.addWidget(divider)
        
        # Status bar below VU meters - truly responsive horizontal layout
        status_bar_container = QtWidgets.QWidget()
        status_bar_container.setFixedHeight(40)  # Fixed height for stats panel
        
        status_bar = QtWidgets.QHBoxLayout(status_bar_container)
        status_bar.setSpacing(20)
        status_bar.setContentsMargins(10, 5, 10, 5)
        
        # Column 1: Jitter (expandable)
        jitter_section = QtWidgets.QHBoxLayout()
        jitter_section.setSpacing(5)
        jitter_section.addWidget(QtWidgets.QLabel("Jitter:"))
        self.jitter_p95_label = QtWidgets.QLabel("--")
        self.jitter_max_label = QtWidgets.QLabel("--")
        self.jitter_p95_label.setMinimumWidth(50)
        self.jitter_max_label.setMinimumWidth(50)
        self.jitter_p95_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.jitter_max_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
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
        self.bitrate_value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
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
        self.packets_received_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.packets_lost_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.packets_dup_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.packet_loss_rate_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
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
        self.packet_rate_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        rate_section.addWidget(self.packet_rate_label)
        rate_section.addWidget(QtWidgets.QLabel("/s"))
        rate_section.addStretch()  # Allow this section to expand
        status_bar.addLayout(rate_section, 1)  # Stretch factor 1
        
        layout.addWidget(status_bar_container)
        self.bar_widgets: list[VUBar] = []
        self.db_labels: list[tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = []
        
        # Create 16 VU meters by default showing minimal values
        self.create_default_vu_meters()
        
        # Populate device list on startup
        self.populate_device_list()
        
        # Connect text change signal for editable combo box
        self.output_device.currentTextChanged.connect(self.on_device_selection_changed)

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
        
        # Get selected device from userData, but skip "Custom..." option
        selected_device = self.output_device.currentData()
        if selected_device == "Custom...":
            QtWidgets.QMessageBox.warning(self, "Device Selection", "Please select a valid device from the dropdown or type a custom device name.")
            return
        
        # If the device is editable and user typed something custom, use that
        if self.output_device.isEditable() and selected_device:
            # Allow custom device names
            pass
        elif not selected_device or selected_device.strip() == "":
            QtWidgets.QMessageBox.warning(self, "Device Selection", "Please select a valid device from the dropdown.")
            return
        
        self.debug_print(f"Starting with device: '{selected_device}'")
        
        # Get the device's channel count from the current selection
        selected_index = self.output_device.currentIndex()
        device_channels = None
        
        # Get the device data from userData (which contains the actual device info)
        device_data = self.output_device.currentData()
        self.debug_print(f"Current device data: '{device_data}'")
        
        # If we have device data and it's not "Custom...", get channel count from cache
        if device_data and device_data != "Custom...":
            if device_data in self.device_info_cache:
                device_info = self.device_info_cache[device_data]
                device_channels = device_info.get("max_output_channels", 0)
                self.debug_print(f"Found device '{device_data}' with {device_channels} channels in cache")
            else:
                self.debug_print(f"Device '{device_data}' not found in cache")
        else:
            self.debug_print(f"No valid device data found: '{device_data}'")
        
        cmd = ([backend] if backend.endswith("vban-backend") else [sys.executable, backend]) + [
            "--listen-ip", self.listen_ip.text().strip() or "0.0.0.0",
            "--listen-port", self.listen_port.text().strip() or "6980",
            "--output-device", selected_device,
        ]
        
        # Add channels parameter if we detected the device's channel count
        if device_channels and device_channels > 0:
            cmd.extend(["--channels", str(device_channels)])
            self.debug_print(f"Added --channels {device_channels} to command")
        
        cmd.extend([
            "--jitter-ms", self.jitter_ms.text().strip() or "20",
            "--device-blocksize", self.blocksize.text().strip() or "1024",
            "--starve-fill",
            "--show-jitter",
            "--show-network",
            "--json",
        ])
        
        # Add verbose flag if checked
        if self.verbose.isChecked():
            cmd.append("--verbose")
        
        if not self.starve_fill.isChecked():
            cmd.remove("--starve-fill")
        map_str = self.current_channel_mapping
        if map_str:
            cmd += ["--map", map_str]

        self.debug_print(f"Command: {cmd}")

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

    def populate_device_list(self):
        """Populate the device dropdown with available output devices"""
        try:
            backend = self.backend_path()
            if backend.endswith("vban-backend"):
                cmd = [backend, "--list-devices", "--json"]
            else:
                cmd = [sys.executable, backend, "--list-devices", "--json"]
            
            out = subprocess.check_output(cmd, text=True)
            
            # Parse JSON output
            try:
                data = json.loads(out)
                if data.get("type") == "devices" and "devices" in data:
                    devices = data["devices"]
                    self.debug_print(f"Parsed {len(devices)} devices from JSON")
                    
                    # Clear existing items
                    self.output_device.clear()
                    
                    # Add devices to dropdown with channel information
                    for device in devices:
                        device_name = device.get("name", "")
                        channel_count = device.get("max_output_channels", 0)
                        
                        if device_name:
                            if channel_count > 0:
                                display_text = f"{device_name} ({channel_count} ch)"
                            else:
                                display_text = device_name
                            
                            # Store original name as user data for backend communication
                            self.output_device.addItem(display_text, userData=device_name)
                            
                            # Cache device information for channel detection
                            self.device_info_cache[device_name] = {
                                "name": device_name,
                                "max_output_channels": channel_count,
                                "index": device.get("index", -1)
                            }
                            
                            self.debug_print(f"Added device: {display_text}")
                    
                    # Try to set "BlackHole 16ch" as default if it exists
                    blackhole_index = -1
                    for i in range(self.output_device.count()):
                        if "BlackHole 16ch" in self.output_device.itemText(i):
                            blackhole_index = i
                            break
                    
                    if blackhole_index >= 0:
                        self.output_device.setCurrentIndex(blackhole_index)
                        self.debug_print("Set BlackHole 16ch as default")
                    elif self.output_device.count() > 0:
                        # Set first device as default if BlackHole not found
                        self.output_device.setCurrentIndex(0)
                        self.debug_print("Set first device as default")
                    
                    # Add a custom entry option
                    self.output_device.addItem("Custom...", userData="Custom...")
                    
                else:
                    self.debug_print(f"Invalid JSON format: {data}")
                    self._fallback_device_list()
                    
            except json.JSONDecodeError as e:
                self.debug_print(f"JSON decode error: {e}")
                self.debug_print(f"Raw output: {out}")
                self._fallback_device_list()
                
        except Exception as e:
            self.debug_print(f"Error populating device list: {e}")
            self._fallback_device_list()
    
    def _fallback_device_list(self):
        """Fallback device list if JSON parsing fails"""
        self.debug_print("Using fallback device list")
        self.output_device.clear()
        self.output_device.addItem("BlackHole 16ch", userData="BlackHole 16ch")
        self.output_device.addItem("System Default", userData="System Default")
        self.output_device.addItem("Custom...", userData="Custom...")

    def on_device_selection_changed(self, text: str):
        """Handle device selection changes"""
        self.debug_print(f"Device selection changed to: '{text}'")
        if text == "Custom...":
            # Clear the text when Custom... is selected
            self.output_device.setEditText("")

    def on_refresh_sources(self):
        """Refresh the list of available audio output devices"""
        self.populate_device_list()
        QtWidgets.QMessageBox.information(self, "Sources Refreshed", "Audio output device list has been refreshed.")

    def open_channel_mapping(self):
        """Open a dialog to configure the channel mapping."""
        # Get current device channel count for output channels
        output_channels = 16  # Default
        selected_device = self.output_device.currentData()
        if selected_device and selected_device != "Custom...":
            if selected_device in self.device_info_cache:
                device_info = self.device_info_cache[selected_device]
                output_channels = device_info.get("max_output_channels", 16)
        
        # Create and show the channel mapping dialog
        dialog = ChannelMappingMatrix(input_channels=16, output_channels=output_channels, parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            map_string = dialog.get_map_string()
            if map_string:
                self.debug_print(f"Channel mapping applied: {map_string}")
                # Store the mapping for use in the backend command
                self.current_channel_mapping = map_string
            else:
                self.debug_print("No channel mapping configured")
                self.current_channel_mapping = ""
        else:
            self.debug_print("Channel mapping dialog cancelled")


class ChannelMappingMatrix(QtWidgets.QDialog):
    """Channel mapping matrix dialog for configuring input to output channel mappings."""
    
    def __init__(self, input_channels=8, output_channels=16, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Channel Mapping Matrix")
        self.resize(600, 500)
        self.setModal(True)
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.mapping_matrix = {}  # (input, output) -> bool
        self.result_map_string = ""
        
        # Initialize default 1:1 mapping
        for i in range(min(self.input_channels, self.output_channels)):
            self.mapping_matrix[(i, i)] = True
        
        self._build_ui()
    
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Matrix container with scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Matrix widget
        matrix_widget = QtWidgets.QWidget()
        matrix_layout = QtWidgets.QGridLayout(matrix_widget)
        matrix_layout.setSpacing(5)
        
        # Create matrix headers (output channels)
        for out_ch in range(self.output_channels):
            header_label = QtWidgets.QLabel(f"Out{out_ch + 1}")
            header_label.setStyleSheet("background-color: #34495e; color: white; padding: 3px; border-radius: 2px; font-weight: bold;")
            header_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            header_label.setMinimumWidth(35)
            matrix_layout.addWidget(header_label, 0, out_ch + 1)
        
        # Create matrix cells with checkboxes
        self.checkbox_widgets = {}
        for in_ch in range(self.input_channels):
            # Input channel label
            input_label = QtWidgets.QLabel(f"In{in_ch + 1}")
            input_label.setStyleSheet("background-color: #2c3e50; color: white; padding: 3px; border-radius: 2px; font-weight: bold;")
            input_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            input_label.setMinimumWidth(35)
            matrix_layout.addWidget(input_label, in_ch + 1, 0)
            
            # Matrix checkboxes
            for out_ch in range(self.output_channels):
                checkbox = QtWidgets.QCheckBox()
                checkbox.setChecked(self.mapping_matrix.get((in_ch, out_ch), False))
                checkbox.stateChanged.connect(lambda state, i=in_ch, o=out_ch: self.on_checkbox_changed(i, o, state))
                
                # Store reference to checkbox
                self.checkbox_widgets[(in_ch, out_ch)] = checkbox
                
                # Add to layout
                matrix_layout.addWidget(checkbox, in_ch + 1, out_ch + 1)
        
        scroll_area.setWidget(matrix_widget)
        layout.addWidget(scroll_area)
        
        # Legend
        legend_layout = QtWidgets.QHBoxLayout()
        legend_layout.addWidget(QtWidgets.QLabel("Legend:"))
        
        mapped_indicator = QtWidgets.QLabel("☑ = Mapped")
        mapped_indicator.setStyleSheet("color: #27ae60; font-weight: bold;")
        legend_layout.addWidget(mapped_indicator)
        
        unmapped_indicator = QtWidgets.QLabel("☐ = Unmapped")
        unmapped_indicator.setStyleSheet("color: #95a5a6;")
        legend_layout.addWidget(unmapped_indicator)
        
        legend_layout.addStretch()
        layout.addLayout(legend_layout)
        
        # Action buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 8px 16px; border-radius: 4px; font-weight: bold;")
        apply_btn.clicked.connect(self.apply_mapping)
        button_layout.addWidget(apply_btn)
        
        reset_btn = QtWidgets.QPushButton("Reset to Default")
        reset_btn.setStyleSheet("background-color: #e74c3c; color: white; padding: 8px 16px; border-radius: 4px; font-weight: bold;")
        reset_btn.clicked.connect(self.reset_to_default)
        button_layout.addWidget(reset_btn)
        
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: #95a5a6; color: white; padding: 8px 16px; border-radius: 4px; font-weight: bold;")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def on_checkbox_changed(self, input_ch, output_ch, state):
        """Handle checkbox state changes with constraint enforcement"""
        if state == QtCore.Qt.CheckState.Checked.value:
            # Check if this output is already mapped to another input
            conflicting_input = None
            for (i, o), checkbox in self.checkbox_widgets.items():
                if o == output_ch and i != input_ch and checkbox.isChecked():
                    conflicting_input = i
                    break
            
            if conflicting_input is not None:
                # Remove the conflicting mapping first
                self.mapping_matrix.pop((conflicting_input, output_ch), None)
                self.checkbox_widgets[(conflicting_input, output_ch)].setChecked(False)
            
            # Now add the new mapping
            self.mapping_matrix[(input_ch, output_ch)] = True
        else:
            # Remove mapping
            self.mapping_matrix.pop((input_ch, output_ch), None)
        
        # Update status (no longer displayed, but keep for debugging)
        mapped_count = len(self.mapping_matrix)
    
    def reset_to_default(self):
        """Reset to 1:1 mapping"""
        # Reset to 1:1 mapping
        self.mapping_matrix.clear()
        for checkbox in self.checkbox_widgets.values():
            checkbox.setChecked(False)
        
        for i in range(min(self.input_channels, self.output_channels)):
            self.mapping_matrix[(i, i)] = True
            self.checkbox_widgets[(i, i)].setChecked(True)
    
    def apply_mapping(self):
        """Generate the map parameter string and close dialog"""
        if not self.mapping_matrix:
            self.result_map_string = ""
        else:
            # Create a list where index = output_channel, value = input_channel
            # Initialize with None (unmapped)
            output_mapping = [None] * self.output_channels
            
            # Fill in the mappings
            for (in_ch, out_ch) in self.mapping_matrix.keys():
                if self.mapping_matrix[(in_ch, out_ch)]:
                    output_mapping[out_ch] = in_ch
            
            # Convert to the correct format: --map 2,1,3,3 means:
            # out1 gets in2, out2 gets in1, out3 gets in3, out4 gets in3
            mappings = []
            for out_ch in range(self.output_channels):
                if output_mapping[out_ch] is not None:
                    mappings.append(str(output_mapping[out_ch] + 1))  # +1 for 1-based indexing
                else:
                    mappings.append("0")  # 0 means no mapping for this output
            
            self.result_map_string = ",".join(mappings)
        
        self.accept()
    
    def get_map_string(self):
        """Return the generated map string"""
        return self.result_map_string


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


