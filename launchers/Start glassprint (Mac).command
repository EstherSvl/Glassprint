#!/bin/bash
# Double-click this file to start glassprint.
#
# It sets itself up the first time (a couple of minutes), then prints two
# addresses: one for this computer, and one to type into an iPad or phone on
# the same Wi-Fi. Leave the window open while you work; close it to stop.

cd "$(dirname "$0")/.." || exit 1

fail() {
  echo
  echo "$1"
  echo
  echo "Press any key to close this window."
  read -r -n 1 -s
  exit 1
}

PYTHON=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done
[ -n "$PYTHON" ] || fail "Python 3 is not installed. Get it from python.org, then try again."

if [ ! -x .venv/bin/python ]; then
  echo "First run — installing glassprint. This takes a minute or two…"
  "$PYTHON" -m venv .venv || fail "Could not create the environment in $(pwd)/.venv"
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -e . || fail "Install failed — see the messages above."
  echo "Done."
  echo
fi

.venv/bin/python -m glassprint.cli serve --lan
echo
echo "glassprint has stopped. Press any key to close this window."
read -r -n 1 -s
