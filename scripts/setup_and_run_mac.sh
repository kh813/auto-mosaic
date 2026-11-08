#!/bin/bash
# setup_and_run_mac.sh
# Automates macOS environment setup and launches auto-mosaic GUI / CLI.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}==========================================================${NC}"
echo -e "${CYAN}        auto-mosaic: macOS Auto-Setup & Launcher          ${NC}"
echo -e "${CYAN}==========================================================${NC}"

# 1. Check Python 3 (>= 3.10)
echo -e "\n${YELLOW}[1/4] Checking Python environment...${NC}"

PYTHON_BIN=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        VER=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ -n "$MAJOR" ] && [ "$MAJOR" -ge 3 ] && [ -n "$MINOR" ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_BIN="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo -e "${RED}[Error] Python 3.10 or higher is required but was not found on your system.${NC}"
    echo -e "Please install Python using Homebrew or from python.org:"
    echo -e "    ${GREEN}brew install python@3.11${NC}"
    echo -e "or download from: https://www.python.org/downloads/macos/"
    read -p "Press Enter to exit..."
    exit 1
fi

echo -e "Found Python: ${GREEN}$($PYTHON_BIN --version)${NC} ($PYTHON_BIN)"

# Setup virtual environment
VENV_DIR="$ROOT_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python"

NEED_INSTALL=0
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    NEED_INSTALL=1
else
    # Check if core modules can be imported
    if ! "$VENV_PYTHON" -c "import cv2, PySide6, onnxruntime" >/dev/null 2>&1; then
        NEED_INSTALL=1
    fi
fi

if [ $NEED_INSTALL -eq 1 ]; then
    echo -e "${YELLOW}Installing project requirements (PySide6, ONNX Runtime, OpenCV, etc.)...${NC}"
    "$VENV_PYTHON" -m pip install --upgrade pip
    "$VENV_PYTHON" -m pip install -r "$ROOT_DIR/requirements/requirements-mac.txt"
    echo -e "${GREEN}Python dependencies installed successfully!${NC}"
else
    echo -e "${GREEN}Python environment is ready.${NC}"
fi

# 2. Check FFmpeg
echo -e "\n${YELLOW}[2/4] Checking FFmpeg / FFprobe...${NC}"
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo -e "${RED}[Warning] ffmpeg or ffprobe was not found in your PATH.${NC}"
    echo -e "FFmpeg is essential for high-fidelity video processing and HDR preservation."
    echo -e "You can install it easily with Homebrew:"
    echo -e "    ${GREEN}brew install ffmpeg${NC}"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " CONTINUE_CHOICE
    if [[ ! "$CONTINUE_CHOICE" =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}FFmpeg is ready.$(ffmpeg -version 2>&1 | head -n 1)${NC}"
fi

# 3. Check or Download SCRFD & License Plate Models
echo -e "\n${YELLOW}[3/4] Checking AI Detection Models (Face & License Plate)...${NC}"
MODEL_FILE="$ROOT_DIR/models/scrfd_10g_bnkps.onnx"
PLATE_MODEL_FILE="$ROOT_DIR/models/license-plate-finetune-v1n.onnx"
if [ ! -f "$MODEL_FILE" ] || [ ! -f "$PLATE_MODEL_FILE" ]; then
    echo -e "${YELLOW}Downloading AI detection models (this only happens on first run)...${NC}"
    "$VENV_PYTHON" "$ROOT_DIR/scripts/download_models.py"
    echo -e "${GREEN}Models successfully downloaded!${NC}"
else
    echo -e "${GREEN}Face & License Plate detection models are ready.${NC}"
fi

# 4. Launch Application
echo -e "\n${CYAN}[4/4] Launching auto-mosaic...${NC}"
echo -e "${CYAN}==========================================================${NC}\n"

if [ $# -gt 0 ]; then
    # CLI mode
    exec "$VENV_PYTHON" -m face_mosaic.cli "$@"
else
    # GUI mode
    echo "Starting GUI application..."
    "$VENV_PYTHON" -m face_mosaic.gui.app
fi
