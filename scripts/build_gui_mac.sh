#!/usr/bin/env bash
set -euo pipefail

# PyInstaller build script for the PyQt6 GUI on macOS.
# It bundles the backend `vban_to_blackhole16.py` as a data file so the GUI can spawn it.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

APP_NAME="VBAN-BlackHole-GUI"
ENTRYPOINT="gui/vban_gui.py"

echo "[1/3] Preparing build virtualenv..."
python3 -m venv .venv_build_gui
source .venv_build_gui/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

echo "[2/3] Building backend console binary..."
# Build standalone backend first so GUI can spawn it without Python
pyinstaller \
  --onefile \
  --name vban-backend \
  --clean \
  --log-level=WARN \
  vban_to_blackhole16.py

echo "[3/3] Building app bundle with PyInstaller..."
# --windowed: no console window; create a .app bundle
# --add-data: bundle backend script; format src:dest (colon works on macOS)
pyinstaller \
  --windowed \
  --name "$APP_NAME" \
  --clean \
  --log-level=WARN \
  --add-binary "dist/vban-backend:backend" \
  "$ENTRYPOINT"

echo "Build complete"
echo "App bundle: dist/$APP_NAME.app"
echo
echo "If Gatekeeper blocks execution, ad-hoc sign the app bundle:"
echo "  codesign --force --deep --sign - dist/$APP_NAME.app"
echo

