#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
else
  echo "Missing ${VENV_DIR}. Run ./scripts/setup_dev.sh first."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
HOST="${HOST:-127.0.0.1}"
NO_RELOAD="${NO_RELOAD:-0}"
SAFETY_ENABLED="${SAFETY_ENABLED:-0}"
RELEVANCE_ENABLED="${RELEVANCE_ENABLED:-0}"

SERVICES_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  echo "Stopping dev processes..."

  if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi

  if [[ -n "${SERVICES_PID}" ]] && kill -0 "${SERVICES_PID}" 2>/dev/null; then
    kill "${SERVICES_PID}" 2>/dev/null || true
  fi

  wait || true
}

trap cleanup INT TERM EXIT

SERVICES_ARGS=("services" "up" "--host" "${HOST}")
if [[ "${NO_RELOAD}" == "1" ]]; then
  SERVICES_ARGS+=("--no-reload")
fi

echo "Starting API services..."
( 
  cd "${ROOT_DIR}"
  ORCHESTRATOR_SAFETY_ENABLED="$( [[ "${SAFETY_ENABLED}" == "1" ]] && printf '%s' "true" || printf '%s' "false" )" \
  ORCHESTRATOR_RELEVANCE_ENABLED="$( [[ "${RELEVANCE_ENABLED}" == "1" ]] && printf '%s' "true" || printf '%s' "false" )" \
  "${PYTHON_BIN}" -m services.run_services --host "${HOST}" $( [[ "${NO_RELOAD}" == "1" ]] && printf '%s' "--no-reload" )
 ) &
SERVICES_PID=$!

echo "Starting front end (Vite)..."
(
  cd "${ROOT_DIR}/front_end"
  npm run dev -- --open
) &
FRONTEND_PID=$!

echo ""
echo "Services PID: ${SERVICES_PID}"
echo "Frontend PID: ${FRONTEND_PID}"
echo ""
echo "Press Ctrl+C to stop."

echo ""

# Keep the frontend running even if the services supervisor exits.
# We stop when the frontend exits or when the user hits Ctrl+C.
services_exited_logged="0"
while kill -0 "${FRONTEND_PID}" 2>/dev/null; do
  if [[ "${services_exited_logged}" == "0" ]] && ! kill -0 "${SERVICES_PID}" 2>/dev/null; then
    echo ""
    echo "Services process exited (PID: ${SERVICES_PID}). Frontend will keep running."
    echo "Press Ctrl+C to stop."
    services_exited_logged="1"
  fi
  sleep 1
done

# If one exits, we exit and let the trap clean up the other.
exit 0
