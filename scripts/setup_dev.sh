#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${ROOT_DIR}/.venv"
VENV_PY="${VENV_DIR}/bin/python"

if [[ ! -f "${ROOT_DIR}/.env" ]] && [[ -f "${ROOT_DIR}/.env.example" ]]; then
  cp "${ROOT_DIR}/.env.example" "${ROOT_DIR}/.env"
fi

if [[ ! -x "${VENV_PY}" ]]; then
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" -m venv ".venv"
  )
fi

echo "Installing Python dependencies (editable) ..."
(
  cd "${ROOT_DIR}"
  "${VENV_PY}" -m pip install -U pip
  "${VENV_PY}" -m pip install -e ".[dev]"
)

echo "Installing frontend dependencies ..."
(
  cd "${ROOT_DIR}/front_end"
  if command -v npm >/dev/null 2>&1; then
    npm install
  else
    echo "npm is required to install frontend dependencies."
    exit 1
  fi
)

echo "Done."
