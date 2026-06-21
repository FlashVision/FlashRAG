#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-flashrag}"
PYTHON_VERSION="${2:-3.11}"

echo "========================================"
echo " FlashRAG Environment Setup"
echo "========================================"
echo "Environment : $ENV_NAME"
echo "Python      : $PYTHON_VERSION"
echo "========================================"

if command -v conda &>/dev/null; then
    echo "[1/4] Creating conda environment..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
else
    echo "[1/4] Creating venv (conda not found)..."
    python"$PYTHON_VERSION" -m venv "$ENV_NAME"
    source "$ENV_NAME/bin/activate"
fi

echo "[2/4] Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "[3/4] Installing flashrag with all extras..."
pip install -e ".[all,dev]"

echo "[4/4] Installing pre-commit hooks..."
pre-commit install

echo ""
echo "========================================"
echo " Setup complete!"
echo " Activate with: conda activate $ENV_NAME"
echo " Or:            source $ENV_NAME/bin/activate"
echo "========================================"
