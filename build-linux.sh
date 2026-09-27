#!/usr/bin/env bash
# Build the ExVR x86_64 Linux package inside a container.
# PyInstaller cannot cross-compile, so on macOS run it through Docker/colima:
#
#   docker run --rm --platform linux/amd64 -v "$PWD":/src -w /src python:3.11 bash build-linux.sh
#
# Artifacts land in ./dist-linux64/ExVR on the host.
set -euo pipefail

echo "== Installing runtime deps for wheels =="
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libgl1 libegl1 libglib2.0-0 libfontconfig1 libxkbcommon0 libdbus-1-3 \
    libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1 libxcb-xinerama0 \
    libxcb-image0 libxcb-render-util0 libxcb-shape0 libxcb-xkb1 \
    libxcb-cursor0 libwayland-client0 libwayland-cursor0 \
    xvfb > /dev/null

echo "== Installing Python dependencies =="
pip install --no-cache-dir --retries 10 --timeout 60 \
    -r requirements-linux.txt pyinstaller==6.10.0

echo "== Numerical compatibility test (ExVR -> upstream VRto3D OpenTrack) =="
xvfb-run -a python3 test_vrto3d_compat.py

echo "== Import smoke test =="
xvfb-run -a python3 -c "import main; print('ExVR imports OK')"

echo "== PyInstaller build =="
pyinstaller --clean -y main-linux.spec

echo "== GUI smoke test (10s under Xvfb) =="
set +e
timeout 10 xvfb-run -a ./dist/ExVR/ExVR
status=$?
set -e
if [ "$status" -ne 0 ] && [ "$status" -ne 124 ]; then
    echo "GUI smoke test FAILED (exit $status)" >&2
    exit 1
fi
echo "GUI smoke test OK (exit $status; 124 = killed by timeout as expected)"

# Copy the result to a stable host-visible name: dist-linux64/ is the
# runnable onedir bundle (ExVR binary + _internal/ side by side).
rm -rf dist-linux64
cp -r dist/ExVR dist-linux64

echo "== Done: $(pwd)/dist-linux64/ExVR =="
