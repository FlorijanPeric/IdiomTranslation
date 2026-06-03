#!/usr/bin/env zsh
set -euo pipefail

PYTHON_BIN=""
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v python3.9 >/dev/null 2>&1; then
  PYTHON_BIN="python3.9"
else
  PYTHON_BIN="python3"
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

python run_server.py --host 0.0.0.0 --port 8000
