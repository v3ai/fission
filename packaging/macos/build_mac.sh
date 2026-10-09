#!/usr/bin/env bash
# Build Fission.app and a drag-to-Applications .dmg on a Mac (Apple Silicon or Intel - builds for the machine's CPU).
#
#   bash packaging/macos/build_mac.sh            ->  dist/Fission-<version>-macOS-<arm64|x86_64>.dmg
#
# Needs: macOS 11+, Python 3.11-3.13 (python.org installer or Homebrew), internet for the first pip install.
# Optional signing + notarization (otherwise the app is ad-hoc signed and users right-click > Open the first time):
#   MAC_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
#   APPLE_ID=you@example.com  APPLE_TEAM_ID=TEAMID  APPLE_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx   (app-specific password)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-python3}"
SRC="${FISSION_SRC:-$ROOT/fission.py}"
VERSION="${FISSION_VERSION:-$(head -n 20 "$SRC" | grep -oE 'Fission [0-9]+(\.[0-9]+)+' | head -n1 | cut -d' ' -f2)}"
ARCH="$(uname -m)"
OUT="$ROOT/dist/Fission-$VERSION-macOS-$ARCH.dmg"
echo "==> Fission $VERSION for macOS $ARCH"

if [ ! -d .venv-mac ]; then "$PY" -m venv .venv-mac; fi
source .venv-mac/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r packaging/requirements-build.txt

python packaging/make_icons.py "$SRC"
export FISSION_SRC="$SRC" FISSION_VERSION="$VERSION"
pyinstaller --noconfirm --clean --distpath dist --workpath build/pyinstaller packaging/fission.spec
APP="dist/Fission.app"

echo "==> self-test"
"$APP/Contents/MacOS/Fission" --selftest build/selftest.txt || { cat build/selftest.txt; exit 1; }
cat build/selftest.txt

if [ -n "${MAC_SIGN_IDENTITY:-}" ]; then
  echo "==> signing with $MAC_SIGN_IDENTITY"
  codesign --force --deep --timestamp --options runtime --entitlements packaging/macos/entitlements.plist \
           --sign "$MAC_SIGN_IDENTITY" "$APP"
else
  echo "==> ad-hoc signing (no MAC_SIGN_IDENTITY set)"
  codesign --force --deep --sign - "$APP"
fi
codesign --verify --deep --strict "$APP"

echo "==> disk image"
STAGE="build/dmg"; rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"; ln -s /Applications "$STAGE/Applications"
rm -f "$OUT"
hdiutil create -volname "Fission $VERSION" -srcfolder "$STAGE" -fs HFS+ -format UDZO -imagekey zlib-level=9 -ov "$OUT"
if [ -n "${MAC_SIGN_IDENTITY:-}" ]; then codesign --force --timestamp --sign "$MAC_SIGN_IDENTITY" "$OUT"; fi

if [ -n "${MAC_SIGN_IDENTITY:-}" ] && [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ] && [ -n "${APPLE_APP_PASSWORD:-}" ]; then
  echo "==> notarizing (takes a few minutes)"
  xcrun notarytool submit "$OUT" --apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" --password "$APPLE_APP_PASSWORD" --wait
  xcrun stapler staple "$OUT"
fi
echo "==> done: $OUT"
