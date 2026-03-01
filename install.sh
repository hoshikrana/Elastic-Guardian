#!/usr/bin/env bash
set -euo pipefail
echo "EGX — Install"
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[dev]"
python -c "import egx; print('EGX', egx.__version__, 'OK')"
