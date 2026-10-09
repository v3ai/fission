#!/usr/bin/env bash
# Build a self-contained Fission AppImage (Linux x86_64).
#
#   ./build_appimage.sh path/to/fission.py                 -> ./Fission-<version>-x86_64.AppImage
#   ./build_appimage.sh fission.py -v 0.5 -o ~/Fission.AppImage
#   ./build_appimage.sh --help
#
# Needs: bash, curl, tar, and internet access the first time (downloads are cached in ./build/cache).
# The AppImage carries its own Python + PySide6 + OpenCascade, so users don't install anything.
# Everything inside is built for glibc <= 2.28, so the AppImage runs on Ubuntu 20.04+, Debian 10+,
# Fedora 29+, RHEL/Rocky 8+, Mint 20+, and similar - no matter how new the build machine is.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="Fission"
ARCH="x86_64"
PY_TAG="${PY_TAG:-20241016}"          # python-build-standalone release (portable, relocatable CPython)
PY_VER="${PY_VER:-3.12.7}"
PYSIDE="${PYSIDE:-6.9.3}"             # newest PySide6 that still ships glibc-2.28 (manylinux_2_28) wheels
UBUNTU="http://archive.ubuntu.com/ubuntu/pool"
# X11 helper libraries Qt needs that many distros don't install by default (libxcb-cursor0 especially),
# plus libGLU for the 3D view - taken from Ubuntu 20.04 so they work on any newer glibc.
# (libxkbcommon-x11 is deliberately NOT bundled: it must match the system's libxkbcommon, and an old
#  copy crashes Qt. Every X11/Wayland desktop already has it.)
DEBS=(
  "universe/x/xcb-util-cursor/libxcb-cursor0_0.1.1-4ubuntu1_amd64.deb"
  "main/x/xcb-util-wm/libxcb-icccm4_0.4.1-1.1_amd64.deb"
  "main/x/xcb-util-image/libxcb-image0_0.4.0-1build1_amd64.deb"
  "main/x/xcb-util-keysyms/libxcb-keysyms1_0.4.0-1build1_amd64.deb"
  "main/x/xcb-util-renderutil/libxcb-render-util0_0.3.9-1build1_amd64.deb"
  "main/x/xcb-util/libxcb-util1_0.4.0-0ubuntu3_amd64.deb"
  "main/libg/libglu/libglu1-mesa_9.0.1-1build1_amd64.deb"
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] SOURCE.py

Packages a Fission script (e.g. fission.py / fission4.py) into a single-file AppImage.

Options:
  -v, --version VER    version shown in the file name / desktop entry
                       (default: read from the script's "Fission X.Y" header, else 0.0)
  -o, --output FILE    where to write the AppImage (default: ./$APP-<version>-$ARCH.AppImage)
  -b, --build-dir DIR  working + download-cache directory (default: build/ next to this script)
  -h, --help           show this help
EOF
}

SRC="" VERSION="" OUT="" BUILD="$HERE/build"
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -v|--version) [ $# -ge 2 ] || { usage >&2; exit 2; }; VERSION="$2"; shift 2 ;;
    -o|--output) [ $# -ge 2 ] || { usage >&2; exit 2; }; OUT="$2"; shift 2 ;;
    -b|--build-dir) [ $# -ge 2 ] || { usage >&2; exit 2; }; BUILD="$2"; shift 2 ;;
    --version=*) VERSION="${1#*=}"; shift ;;
    --output=*) OUT="${1#*=}"; shift ;;
    --build-dir=*) BUILD="${1#*=}"; shift ;;
    --) shift; [ $# -gt 0 ] && { SRC="$1"; shift; } ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) [ -z "$SRC" ] || { echo "Only one source file, please (got '$SRC' and '$1')" >&2; exit 2; }; SRC="$1"; shift ;;
  esac
done
[ -n "$SRC" ] || { echo "Error: give the Fission script to package, e.g.  $(basename "$0") fission.py" >&2; usage >&2; exit 2; }
[ -f "$SRC" ] || { echo "Error: no such file: $SRC" >&2; exit 1; }
case "$SRC" in *.py) ;; *) echo "Error: $SRC doesn't look like a Python script (.py)" >&2; exit 1 ;; esac
SRC="$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")"
if [ -z "$VERSION" ]; then
  VERSION="$(head -n 20 "$SRC" | grep -oE 'Fission [0-9]+(\.[0-9]+)+' | head -n1 | cut -d' ' -f2 || true)"
  VERSION="${VERSION:-0.0}"
fi
OUT="${OUT:-$PWD/$APP-$VERSION-$ARCH.AppImage}"
mkdir -p "$(dirname "$OUT")" && OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
mkdir -p "$BUILD" && BUILD="$(cd "$BUILD" && pwd)"
APPDIR="$BUILD/$APP.AppDir"
CACHE="$BUILD/cache"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PY_TAG}/cpython-${PY_VER}+${PY_TAG}-${ARCH}-unknown-linux-gnu-install_only.tar.gz"
TOOL_URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${ARCH}.AppImage"

# Keep the build hermetic: nothing from the caller's Python setup (an active venv, PYTHONPATH,
# packages in ~/.local, pip.conf settings like user=true) may leak into or substitute for the bundle.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP VIRTUAL_ENV CONDA_PREFIX PIP_USER PIP_TARGET PIP_PREFIX PIP_REQUIRE_VIRTUALENV
export PYTHONNOUSERSITE=1

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mError: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(uname -m)" = "$ARCH" ] || die "run this on an $ARCH Linux machine"
echo "Packaging $SRC  (version $VERSION)  ->  $OUT"
command -v curl >/dev/null || die "curl is required"

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/app" "$APPDIR/usr/lib" "$CACHE"

# ---------------------------------------------------------------------------------------------
say "Portable Python $PY_VER"
[ -s "$CACHE/python.tgz" ] || curl -fL --retry 3 -o "$CACHE/python.tgz" "$PY_URL"
tar -xzf "$CACHE/python.tgz" -C "$APPDIR/usr"                      # -> usr/python
PYBIN="$APPDIR/usr/python/bin/python3"
PY=("$PYBIN" -I)            # -I: isolated mode - ignores PYTHON* variables and the user's ~/.local site-packages
"${PY[@]}" -c 'import sys; print(sys.version)'

# ---------------------------------------------------------------------------------------------
say "Python packages (OpenCascade, PySide6, PyOpenGL, numpy)"
PIP=("${PY[@]}" -m pip --isolated --disable-pip-version-check --no-input)   # --isolated: ignore pip.conf & PIP_* vars
"${PIP[@]}" install -q --no-warn-script-location --upgrade pip
"${PIP[@]}" install --no-warn-script-location --no-compile --only-binary=:all: \
    "cadquery-ocp" "PySide6-Essentials==$PYSIDE" "PyOpenGL" "numpy"

SP="$("${PY[@]}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for pkg in OCP PySide6 shiboken6 OpenGL numpy; do      # every package must really be inside the bundle
  [ -d "$SP/$pkg" ] || die "$pkg did not get installed into the AppImage (found in $SP: $(ls "$SP" | tr '\n' ' '))"
done

# OpenCascade's module links a few VTK libraries, but the vtk wheel is 640 MB of Python modules and
# libraries Fission never touches. Keep exactly the VTK libraries OpenCascade loads; drop the rest,
# along with matplotlib & co. which only VTK's Python side uses.
say "Keeping only the VTK libraries OpenCascade links to"
keep="$(ldd "$SP"/OCP/*.so | awk '/vtkmodules/{print $3}' | xargs -r -n1 readlink -f | sort -u)"
[ -n "$keep" ] || die "couldn't work out which VTK libraries OpenCascade needs"
before="$(du -sh "$SP/vtkmodules" | cut -f1)"
find "$SP/vtkmodules" -type f | while read -r f; do
  grep -qxF "$(readlink -f "$f")" <<<"$keep" || rm -f "$f"
done
find "$SP/vtkmodules" -type d -empty -delete
echo "  vtkmodules: $before -> $(du -sh "$SP/vtkmodules" | cut -f1) ($(wc -l <<<"$keep") libraries kept)"
rm -rf "$SP"/{matplotlib,mpl_toolkits,fontTools,PIL,pillow.libs,kiwisolver,contourpy,pyparsing,dateutil,cycler,vtk.py} \
       "$SP"/matplotlib*.pth 2>/dev/null || true

# ---------------------------------------------------------------------------------------------
say "Trimming things Fission never uses"
rm -rf "$SP"/PySide6/{Qt/qml,Qt/translations,include,typesystems,glue,scripts,examples} \
       "$SP"/PySide6/{designer,assistant,linguist,lupdate,lrelease,qmlls,qmllint,qmlformat,qmlcachegen,balsam,balsamui,uic,rcc} \
       "$APPDIR"/usr/python/lib/python3*/{test,idlelib,tkinter,turtledemo,ensurepip} \
       "$APPDIR"/usr/python/lib/{tcl*,tk*,itcl*,thread*} \
       "$APPDIR"/usr/python/{include,share} 2>/dev/null || true
find "$APPDIR/usr/python" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "$APPDIR/usr/python" -path '*/site-packages/*' -type d -name 'tests' -prune -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------------------------
say "Application"
cp "$SRC" "$APPDIR/usr/app/fission.py"
"${PY[@]}" -m compileall -q -j 0 "$APPDIR/usr/app" "$APPDIR/usr/python/lib" >/dev/null || true   # AppImages are read-only: precompile

# ---------------------------------------------------------------------------------------------
say "Bundling X11 helper libraries + libGLU (Ubuntu 20.04 builds)"
for deb in "${DEBS[@]}"; do
  f="$CACHE/$(basename "$deb")"
  [ -s "$f" ] || curl -fsSL --retry 3 -o "$f" "$UBUNTU/$deb" || curl -fsSL --retry 3 -o "$f" "http://old-releases.ubuntu.com/ubuntu/pool/$deb" \
    || die "couldn't download $deb"
done
# A .deb is an 'ar' archive holding data.tar.*; unpack the shared libraries with Python (no dpkg needed).
"${PY[@]}" - "$APPDIR/usr/lib" "$CACHE" "${DEBS[@]}" <<'EOF'
import io, os, sys, tarfile
dest, cache, debs = sys.argv[1], sys.argv[2], sys.argv[3:]
for deb in debs:
    data = open(os.path.join(cache, os.path.basename(deb)), "rb").read()
    assert data[:8] == b"!<arch>\n", deb
    pos = 8
    while pos < len(data):
        name, size = data[pos:pos + 16].decode().strip().rstrip("/"), int(data[pos + 48:pos + 58])
        body = data[pos + 60:pos + 60 + size]; pos += 60 + size + (size & 1)
        if not name.startswith("data.tar"): continue
        with tarfile.open(fileobj=io.BytesIO(body)) as t:
            for m in t.getmembers():
                base = os.path.basename(m.name)
                if ".so." not in base or not (m.isfile() or m.issym()): continue
                out = os.path.join(dest, base)
                if os.path.lexists(out): os.remove(out)
                if m.issym(): os.symlink(m.linkname, out)
                else:
                    with open(out, "wb") as fh: fh.write(t.extractfile(m).read())
                print("   ", base)
EOF

# ---------------------------------------------------------------------------------------------
say "Launcher, desktop entry and icon"
cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
export LD_LIBRARY_PATH="$HERE/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
# The 3D view uses classic (compatibility-profile) OpenGL, which X11 / XWayland provides everywhere.
# Set QT_QPA_PLATFORM=wayland yourself to try native Wayland.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
exec "$HERE/usr/python/bin/python3" "$HERE/usr/app/fission.py" "$@"
EOF
chmod +x "$APPDIR/AppRun"

cat > "$APPDIR/fission.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$APP
GenericName=CAD
Comment=Free parametric 3D CAD in the style of Fusion 360
Exec=$APP %f
Icon=fission
Categories=Graphics;Engineering;3DGraphics;
Keywords=CAD;3D;modeling;OpenCascade;
Terminal=false
StartupWMClass=fission
X-AppImage-Version=$VERSION
EOF

QT_QPA_PLATFORM=offscreen "${PY[@]}" - "$APPDIR/usr/app" "$APPDIR/fission.png" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
from PySide6 import QtWidgets as W, QtGui as G, QtCore as C
app = W.QApplication([])
import fission
S = 256
pm = G.QPixmap(S, S); pm.fill(C.Qt.transparent)
p = G.QPainter(pm); p.setRenderHint(G.QPainter.Antialiasing)
g = G.QLinearGradient(0, 0, 0, S); g.setColorAt(0, G.QColor("#f7f9fb")); g.setColorAt(1, G.QColor("#d9e2ea"))
p.setBrush(g); p.setPen(G.QPen(G.QColor("#9fb2c4"), 4)); p.drawRoundedRect(C.QRectF(8, 8, S - 16, S - 16), 44, 44)
p.drawPixmap(28, 28, fission.pix("extrude", 200, 1))
p.end(); pm.save(sys.argv[2])
EOF
ln -sf fission.png "$APPDIR/.DirIcon"

# ---------------------------------------------------------------------------------------------
say "Smoke test (imports only)"
# Run exactly as the AppImage will: isolated, from inside the bundle, with no access to the caller's packages.
( cd / && env -i HOME="$HOME" PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="$APPDIR/usr/lib" \
    "$PYBIN" -I -c '
import sys, OCP, PySide6.QtOpenGLWidgets, OpenGL.GL, numpy
root = sys.argv[1]
for m in (OCP, PySide6, OpenGL, numpy):
    assert m.__file__.startswith(root), f"{m.__name__} was loaded from outside the AppImage: {m.__file__}"
print("  ok - all modules load from inside the AppImage")' "$APPDIR" ) || die "smoke test failed"

# ---------------------------------------------------------------------------------------------
say "Packing the AppImage"
TOOL="$CACHE/appimagetool"
[ -x "$TOOL" ] || { curl -fL --retry 3 -o "$TOOL" "$TOOL_URL"; chmod +x "$TOOL"; }
du -sh "$APPDIR" | sed 's/^/  unpacked size: /'
rm -f "$OUT"
# Extract-and-run means appimagetool itself works even where FUSE isn't available (containers, CI).
APPIMAGE_EXTRACT_AND_RUN=1 ARCH="$ARCH" VERSION="$VERSION" "$TOOL" --no-appstream "$APPDIR" "$OUT"
chmod +x "$OUT"
say "Done: $OUT ($(du -h "$OUT" | cut -f1))"
echo "  Run it with:  $OUT"
