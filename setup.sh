#!/usr/bin/env bash
# Creates the Python 3.13 virtual environment and links the Optris SDK bindings into it.
# The SDK bindings (Debian 13 package) are compiled for Python 3.13 and crash with other versions.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3.13}
if ! command -v "$PYTHON" >/dev/null; then
    echo "$PYTHON not found. Install it with: sudo apt install python3.13 python3.13-venv" >&2
    exit 1
fi
if [ -x .venv/bin/python ] && ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 13))'; then
    echo ".venv uses another Python version. Delete it (rm -rf .venv) and run ./setup.sh again." >&2
    exit 1
fi

[ -x .venv/bin/python ] || "$PYTHON" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

SDK=/usr/lib/python3/dist-packages/optris
SITE=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
if [ -d "$SDK" ]; then
    ln -sfn "$SDK" "$SITE/optris"
    .venv/bin/python -c 'import optris.otcsdk' && echo "Optris SDK: OK"
else
    echo "Optris SDK not installed yet. Install it (see README), then run ./setup.sh again:"
    echo "  sudo apt install ~/Downloads/otcsdk-11.4.10-debian-13-amd64-ubuntu26.04.deb"
fi
echo "Start the app with: .venv/bin/python -m thermal_table"
