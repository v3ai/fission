# -*- mode: python -*-
# PyInstaller recipe for Fission - the same file builds Windows, macOS and Linux (run it on each OS).
#
#   pyinstaller --noconfirm --clean packaging/fission.spec
#
# Environment knobs:  FISSION_SRC (default: fission.py next to packaging/), FISSION_VERSION (default: read from the source)
import glob, os, re, sys, sysconfig
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
SRC = os.path.abspath(os.environ.get("FISSION_SRC") or os.path.join(ROOT, "fission.py"))
head = open(SRC, encoding="utf-8").read(2000)
VERSION = os.environ.get("FISSION_VERSION") or (re.search(r"Fission (\d+(?:\.\d+)+)", head) or [None, "0.0"])[1]
ASSETS = os.path.join(SPECPATH, "assets")
SITE = sysconfig.get_paths()["purelib"]
IS_WIN, IS_MAC = sys.platform == "win32", sys.platform == "darwin"
STRIP = not IS_WIN and os.environ.get("FISSION_STRIP", "1") == "1"      # drop debug symbols (OCCT / VTK ship with lots)

# ---- OpenCascade (cadquery-ocp) ----
# The OCP extension links against ~100 OCCT libraries (and a few VTK ones) that the wheel keeps in
# cadquery_ocp.libs/ (Windows / Linux) or .dylibs folders (macOS). Ship that folder unchanged, next to OCP,
# so both PyInstaller's loader and OCP's own DLL-directory setup find it.
binaries = []
for libdir in ("cadquery_ocp.libs", os.path.join("cadquery_ocp", ".dylibs"), os.path.join("OCP", ".dylibs")):
    src = os.path.join(SITE, libdir)
    if os.path.isdir(src):
        for f in glob.glob(os.path.join(src, "*")):
            if os.path.isfile(f): binaries.append((f, libdir))
datas = [(f, "cadquery_ocp.libs") for f in glob.glob(os.path.join(SITE, "cadquery_ocp.libs", ".load-order*"))]
if IS_MAC: datas.append((os.path.join(ASSETS, "fission-file.icns"), "."))     # document icon, ends up in Contents/Resources

# On Windows, OCP needs some VTK DLLs from vtk.libs\, and its __init__ adds that exact folder to the DLL path (it must
# exist). PyInstaller's import scanner misses most of them (delvewheel moves the import names where pefile stops
# reading), so walk the real dependencies ourselves and ship the needed ones in vtk.libs\.
if IS_WIN:
    sys.path.insert(0, os.path.join(SPECPATH, "xbuild"))
    from xbuild import windows_closure
    roots = glob.glob(os.path.join(SITE, "OCP", "*.pyd")) + glob.glob(os.path.join(SITE, "cadquery_ocp.libs", "*.dll"))
    need = windows_closure(SITE, roots)
    vtklibs = os.path.realpath(os.path.join(SITE, "vtk.libs"))
    binaries += [(p, "vtk.libs") for p in sorted(need) if os.path.dirname(p) == vtklibs]
    os.environ["PATH"] = os.pathsep.join([vtklibs, os.path.join(SITE, "cadquery_ocp.libs"), os.environ.get("PATH", "")])

hiddenimports = (collect_submodules("OCP")
                 + ["OpenGL.platform.win32", "OpenGL.platform.darwin", "OpenGL.platform.glx", "OpenGL.platform.egl",
                    "OpenGL.arrays.numpymodule", "OpenGL.arrays.ctypesarrays", "OpenGL.arrays.ctypespointers",
                    "OpenGL.arrays.ctypesparameters", "OpenGL.arrays.numbers", "OpenGL.arrays.strings", "OpenGL.arrays.lists",
                    "OpenGL.arrays.nones", "OpenGL.converters", "OpenGL.GL.shaders", "OpenGL.GLU"])

a = Analysis(
    [SRC],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas + collect_data_files("OpenGL"),
    hiddenimports=hiddenimports,
    runtime_hooks=[os.path.join(SPECPATH, "rthook_dlls.py")],
    excludes=["tkinter", "matplotlib", "vtkmodules.all", "vtkmodules.qt", "vtkmodules.web", "vtkmodules.tk", "vtk",
              "IPython", "pytest", "scipy", "pandas",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml",
              "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtPdf"],
    noarchive=False,
)
# Qt pulls in Quick / QML / PDF / virtual-keyboard libraries through plugins Fission never uses - leave them out.
DROP = re.compile(r"Qt6?(Pdf|Qml|Quick|VirtualKeyboard)|qtvirtualkeyboard|qpdf|[/\\]translations[/\\]", re.I)
a.binaries = [x for x in a.binaries if not DROP.search(x[0])]
a.datas = [x for x in a.datas if not DROP.search(x[0])]
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Fission",
    console=False,                       # a normal windowed app (use --selftest FILE to check a build)
    icon=os.path.join(ASSETS, "fission.ico" if IS_WIN else "fission.icns" if IS_MAC else "fission.png"),
    version=os.path.join(SPECPATH, "windows", "version_info.txt") if IS_WIN and os.path.exists(os.path.join(SPECPATH, "windows", "version_info.txt")) else None,
    argv_emulation=False,
    target_arch=os.environ.get("FISSION_ARCH") or None,
    codesign_identity=os.environ.get("MAC_SIGN_IDENTITY") or None,
    entitlements_file=os.path.join(SPECPATH, "macos", "entitlements.plist") if IS_MAC else None,
    upx=False,
    strip=STRIP,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Fission", upx=False, strip=STRIP)

if IS_MAC:
    app = BUNDLE(
        coll,
        name="Fission.app",
        icon=os.path.join(ASSETS, "fission.icns"),
        bundle_identifier="org.fission.cad",
        version=VERSION,
        info_plist={
            "CFBundleName": "Fission",
            "CFBundleDisplayName": "Fission",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": True,          # the UI is designed light
            "LSMinimumSystemVersion": "11.0",
            "LSApplicationCategoryType": "public.app-category.graphics-design",
            "CFBundleDocumentTypes": [{
                "CFBundleTypeName": "Fission design",
                "CFBundleTypeRole": "Editor",
                "LSHandlerRank": "Owner",
                "CFBundleTypeIconFile": "fission-file.icns",
                "LSItemContentTypes": ["org.fission.design"],
            }],
            "UTExportedTypeDeclarations": [{
                "UTTypeIdentifier": "org.fission.design",
                "UTTypeDescription": "Fission design",
                "UTTypeConformsTo": ["public.json", "public.data"],
                "UTTypeIconFile": "fission-file.icns",
                "UTTypeTagSpecification": {"public.filename-extension": ["fission"]},
            }],
        },
    )
