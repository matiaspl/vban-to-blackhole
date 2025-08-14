#!/usr/bin/env bash
set -euo pipefail

# PyInstaller build script for macOS (arm64/x86_64)
# Produces a self-contained CLI binary in ./dist/vban-to-blackhole

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BIN_NAME="vban-to-blackhole"
ENTRYPOINT="vban_to_blackhole16.py"

echo "[1/3] Preparing build virtualenv..."
python3 -m venv .venv_build
source .venv_build/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

echo "[2/3] Building with PyInstaller..."
# Notes:
# - --onefile creates a single self-extracting binary
# - --strip reduces size by stripping symbols
# - --clean removes previous build artifacts
# - We build a console app (no macOS bundle) suitable for CLI use
pyinstaller \
  --onefile \
  --name "$BIN_NAME" \
  --clean \
  --strip \
  --log-level=WARN \
  "$ENTRYPOINT"

echo "[3/3] Build complete"
ls -lh "dist/$BIN_NAME" || true

echo
echo "Run the binary (example):"
echo "  ./dist/$BIN_NAME --listen-ip 0.0.0.0 --listen-port 6980 --output-device 'BlackHole 16ch'"
echo
echo "If Gatekeeper blocks execution, you can ad-hoc sign the binary:"
echo "  codesign --force --deep --sign - dist/$BIN_NAME"
echo

