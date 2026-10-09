#!/usr/bin/env python3
"""Build Fission for Windows and macOS on a Linux machine - no Windows PC, Mac or GitHub needed.

    python3 packaging/xbuild/xbuild.py                      # everything:  windows, mac-arm64, mac-x86_64
    python3 packaging/xbuild/xbuild.py windows               # or pick targets
    python3 packaging/xbuild/xbuild.py mac-arm64 --src ~/fission7.py --version 0.7

Output (dist/):
    Fission-<ver>-Windows-x64-Setup.exe      installer (per-user, Start menu, .fission files, uninstaller)
    Fission-<ver>-Windows-x64-portable.zip   unzip-and-run folder
    Fission-<ver>-macOS-arm64.zip            Fission.app for Apple Silicon Macs (macOS 13+)
    Fission-<ver>-macOS-x86_64.zip           Fission.app for Intel Macs (macOS 13+)

How it works: each app is a portable CPython (python-build-standalone) plus the official Windows / macOS wheels
of OpenCascade, Qt, PyOpenGL and numpy (pip downloads them for the other platform). Unused Qt / VTK libraries are
dropped by following the real library dependencies, all Python code is precompiled, and then:
  Windows - a small Fission.exe (cross-compiled with MinGW) runs Python inside its own process; NSIS builds the installer.
  macOS   - the bundled Python binary *is* Contents/MacOS/Fission (sitecustomize starts the app); the bundle is
            thinned to one CPU type and ad-hoc signed with rcodesign so Apple Silicon will run it.

Needs on the Linux machine:  python3 + pip, curl, zip;  makensis + x86_64-w64-mingw32-gcc (Windows);
nothing extra for macOS;  on Ubuntu/Debian:  sudo apt install nsis gcc-mingw-w64-x86-64 zip
rcodesign is downloaded automatically. Optional: wine (+ xvfb) to self-test the Windows build ( --test ).
"""
import argparse, glob, hashlib, json, os, plistlib, re, shutil, stat, subprocess, sys, tarfile, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ASSETS = os.path.join(ROOT, "packaging", "assets")
PY_TAG, PY_VER = "20241016", "3.12.7"
PBS = "https://github.com/astral-sh/python-build-standalone/releases/download/{tag}/cpython-{ver}+{tag}-{triple}-install_only.tar.gz"
RCODESIGN = "https://github.com/indygreg/apple-platform-rs/releases/download/apple-codesign%2F0.29.0/apple-codesign-0.29.0-x86_64-unknown-linux-musl.tar.gz"
PACKAGES = ["cadquery-ocp==8.0.1.1.0", "cadquery-ocp-proxy==8.0.1.1.0", "vtk==9.6.2", "PySide6-Essentials==6.11.2",
            "shiboken6==6.11.2", "PyOpenGL==3.1.10", "numpy==2.2.6"]
TARGETS = {
    "windows":    dict(triple="x86_64-pc-windows-msvc", plat=["win_amd64"]),
    "mac-arm64":  dict(triple="aarch64-apple-darwin", plat=["macosx_13_0_arm64"], arch="arm64"),
    "mac-x86_64": dict(triple="x86_64-apple-darwin", plat=["macosx_13_0_x86_64"], arch="x86_64"),
}
QT_KEEP = ("QtCore", "QtGui", "QtWidgets", "QtOpenGL", "QtOpenGLWidgets")      # the PySide6 modules Fission imports
STDLIB_DROP = ("test", "idlelib", "tkinter", "turtledemo", "ensurepip", "lib2to3", "turtle.py", "pydoc_data", "unittest/test")

def log(*a): print("==>", *a, flush=True)
def run(cmd, **kw):
    print("   $", " ".join(map(str, cmd)), flush=True); subprocess.run(cmd, check=True, **kw)
def untar(tgz, dest):
    with tarfile.open(tgz) as t:
        try: t.extractall(dest, filter="data")
        except TypeError: t.extractall(dest)                        # Python < 3.12 without the backport
def rmtree(p):
    if os.path.islink(p) or os.path.isfile(p): os.remove(p)
    elif os.path.isdir(p): shutil.rmtree(p)
def dir_size(p):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs if not os.path.islink(os.path.join(d, f)))
def mb(n): return f"{n/1e6:.0f} MB"

# ---------------------------------------------------------------- downloads
def fetch(url, dest):
    if not os.path.exists(dest):
        log("download", os.path.basename(dest)); run(["curl", "-fsSL", "-o", dest + ".part", url]); os.replace(dest + ".part", dest)
    return dest

def runtime_tarball(cache, triple):
    return fetch(PBS.format(tag=PY_TAG, ver=PY_VER, triple=triple), os.path.join(cache, f"cpython-{PY_VER}-{triple}.tar.gz"))

def host_python(cache):
    """A Linux CPython of the same version as the bundles - used to precompile bytecode."""
    d = os.path.join(cache, "host-python")
    if not os.path.exists(os.path.join(d, "python", "bin", "python3")):
        untar(runtime_tarball(cache, "x86_64-unknown-linux-gnu"), d)
    return os.path.join(d, "python", "bin", "python3")

def wheels(cache, plat):
    d = os.path.join(cache, "wheels-" + plat[0]); os.makedirs(d, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", "-q", "--no-deps", "--only-binary=:all:", "--python-version", "3.12",
           "--implementation", "cp", "--abi", "cp312", "--abi", "abi3", "--abi", "none", "-d", d]
    for p in plat: cmd += ["--platform", p]
    run(cmd + PACKAGES)
    return d

def install_wheels(cache, plat, site):
    d = wheels(cache, plat); os.makedirs(site, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index", "--find-links", d, "--only-binary=:all:",
           "--python-version", "3.12", "--implementation", "cp", "--abi", "cp312", "--abi", "abi3", "--abi", "none",
           "--target", site, "--no-compile"]
    for p in plat: cmd += ["--platform", p]
    run(cmd + PACKAGES)
    for junk in glob.glob(os.path.join(site, "bin")) + glob.glob(os.path.join(site, "*.dist-info", "RECORD")): rmtree(junk)

def rcodesign(cache):
    exe = os.path.join(cache, "rcodesign")
    if not os.path.exists(exe):
        tgz = fetch(RCODESIGN, os.path.join(cache, "rcodesign.tar.gz"))
        with tarfile.open(tgz) as t:
            m = next(x for x in t.getmembers() if x.name.endswith("/rcodesign"))
            with t.extractfile(m) as src, open(exe, "wb") as dst: shutil.copyfileobj(src, dst)
        os.chmod(exe, 0o755)
    return exe

def tool(*names):
    for n in names:
        p = shutil.which(n)
        if p: return p
    raise SystemExit(f"missing tool: {' or '.join(names)} (see the top of xbuild.py for the apt command)")

# ---------------------------------------------------------------- shared steps
def prune_stdlib(lib):
    for name in STDLIB_DROP: rmtree(os.path.join(lib, name))
    for d, ds, fs in os.walk(lib):
        for x in list(ds):
            if x in ("__pycache__",): rmtree(os.path.join(d, x)); ds.remove(x)

def prune_site(site):
    """Things no Fission user needs: tests, type stubs, Qt tools / QML / translations, the VTK Python layer."""
    for pat in ("numpy/*/tests", "numpy/tests", "numpy/_core/tests", "numpy/f2py", "numpy/*/*.pyi", "numpy/*.pyi",
                "OpenGL/Tk", "PySide6/*.pyi", "PySide6/scripts", "PySide6/include", "PySide6/typesystems", "PySide6/glue",
                "PySide6/translations", "PySide6/qml", "PySide6/metatypes", "PySide6/doc", "PySide6/examples",
                "PySide6/Qt/translations", "PySide6/Qt/qml", "PySide6/Qt/metatypes", "PySide6/Qt/libexec",
                "PySide6/*.app", "PySide6/*.exe", "PySide6/Qt/bin", "vtk.py", "shiboken6_generator"):
        for p in glob.glob(os.path.join(site, pat)): rmtree(p)
    for p in glob.glob(os.path.join(site, "PySide6", "Qt*.pyd")) + glob.glob(os.path.join(site, "PySide6", "Qt*.abi3.so")):
        mod = os.path.basename(p).split(".")[0]
        if mod not in QT_KEEP: os.remove(p)

def compile_all(cache, *dirs):
    py = host_python(cache)
    for d in dirs:
        subprocess.run([py, "-m", "compileall", "-q", "-j", "0", "--invalidation-mode", "unchecked-hash", "-s", d, "-p", "", d],
                       check=False, stdout=subprocess.DEVNULL)

def copy_app(src, version, app_dir):
    os.makedirs(app_dir, exist_ok=True)
    text = open(src, encoding="utf-8").read()
    with open(os.path.join(app_dir, "fission.py"), "w", encoding="utf-8") as f: f.write(text)
    shutil.copy(os.path.join(HERE, "boot.py"), app_dir)

# ---------------------------------------------------------------- dependency walking (keeps only libraries something loads)
def pe_imports(path):
    """DLL names a PE file imports (normal + delay-load). Reads the descriptor tables directly: pefile's own parser
    stops early on DLLs post-processed by delvewheel (their renamed imports live in an added section)."""
    import pefile, struct
    try: pe = pefile.PE(path, fast_load=True)
    except Exception: return []
    names = []
    def read_names(dir_index, size, name_off):
        d = pe.OPTIONAL_HEADER.DATA_DIRECTORY[dir_index]
        if not d.VirtualAddress: return
        rva = d.VirtualAddress
        for i in range(4096):
            try: raw = pe.get_data(rva + i*size, size)
            except Exception: break
            if len(raw) < size or raw == b"\0"*size: break
            name_rva = struct.unpack_from("<I", raw, name_off)[0]
            if not name_rva: break
            try: names.append(pe.get_string_at_rva(name_rva, 512).decode(errors="ignore"))
            except Exception: pass
    read_names(pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"], 20, 12)          # IMAGE_IMPORT_DESCRIPTOR.Name
    read_names(pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"], 32, 4)     # ImgDelayDescr.DllNameRVA
    pe.close(); return [n for n in names if n]

def windows_closure(top, roots):
    index = {}
    for d, _, fs in os.walk(top):
        for f in fs:
            if f.lower().endswith((".dll", ".pyd")): index.setdefault(f.lower(), os.path.join(d, f))
    keep, todo = set(), [os.path.realpath(r) for r in roots]
    while todo:
        p = todo.pop()
        if p in keep: continue
        keep.add(p)
        for name in pe_imports(p):
            q = index.get(name.lower())
            if q and os.path.realpath(q) not in keep: todo.append(os.path.realpath(q))
    return keep

def macho_info(path):
    """(archs, load commands, rpaths) of a Mach-O file, or None."""
    from macholib.MachO import MachO
    from macholib import mach_o
    try: m = MachO(path)
    except Exception: return None
    archs, loads, rpaths = set(), [], []
    for h in m.headers:
        archs.add({0x0100000C: "arm64", 0x01000007: "x86_64"}.get(h.header.cputype, str(h.header.cputype)))
        dylib_cmds = {getattr(mach_o, n, -1) for n in ("LC_LOAD_DYLIB", "LC_LOAD_WEAK_DYLIB", "LC_REEXPORT_DYLIB", "LC_LAZY_LOAD_DYLIB", "LC_LOAD_UPWARD_DYLIB")}
        for lc, cmd, data in h.commands:
            if lc.cmd in dylib_cmds:
                s = data.rstrip(b"\0").decode(errors="ignore"); loads.append((s, lc.cmd == getattr(mach_o, "LC_LOAD_WEAK_DYLIB", -2)))
            elif lc.cmd == mach_o.LC_RPATH:
                rpaths.append(data.rstrip(b"\0").decode(errors="ignore"))
    return archs, loads, rpaths

def thin_macho(path, arch):
    """Keep one CPU slice of a universal ("fat") Mach-O file - what `lipo -thin` does. The slice is copied byte for
    byte, so its code signature stays valid."""
    import struct
    want = {"arm64": 0x0100000C, "x86_64": 0x01000007}[arch]
    with open(path, "rb") as f: data = f.read()
    magic = struct.unpack_from(">I", data, 0)[0]
    if magic not in (0xCAFEBABE, 0xCAFEBABF): return False
    n = struct.unpack_from(">I", data, 4)[0]
    for i in range(n):
        if magic == 0xCAFEBABE: cpu, _, off, size, _ = struct.unpack_from(">iiIII", data, 8 + 20*i)
        else: cpu, _, off, size, _, _ = struct.unpack_from(">iiQQII", data, 8 + 32*i)
        if cpu == want:
            mode = os.stat(path).st_mode
            with open(path + ".thin", "wb") as f: f.write(data[off:off + size])
            os.chmod(path + ".thin", mode); os.replace(path + ".thin", path); return True
    raise SystemExit(f"{path} has no {arch} code")

def is_macho(path):
    try:
        with open(path, "rb") as f: magic = f.read(4)
    except Exception: return False
    return magic in (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca")

def mac_resolve(name, loader, exe_dir, rpaths):
    if name.startswith("/usr/lib/") or name.startswith("/System/"): return "system"
    cands = []
    if name.startswith("@loader_path/"): cands.append(os.path.join(os.path.dirname(loader), name[13:]))
    elif name.startswith("@executable_path/"): cands.append(os.path.join(exe_dir, name[17:]))
    elif name.startswith("@rpath/"):
        for r in rpaths:
            r = r.replace("@loader_path", os.path.dirname(loader)).replace("@executable_path", exe_dir)
            cands.append(os.path.join(r, name[7:]))
    else: cands.append(name)
    for c in cands:
        c = os.path.normpath(c)
        if os.path.exists(c): return os.path.realpath(c)
    return None

def mac_closure(roots, exe_dir):
    keep, missing, todo = set(), [], [(os.path.realpath(r), []) for r in roots]
    while todo:
        p, inherited = todo.pop()
        if p in keep: continue
        keep.add(p); info = macho_info(p)
        if not info: continue
        _, loads, rpaths = info
        # an LC_RPATH's @loader_path means the image that declares it, also when a library it loads uses it
        rp = [r.replace("@loader_path", os.path.dirname(p)).replace("@executable_path", exe_dir) for r in rpaths] + inherited
        for name, weak in loads:
            q = mac_resolve(name, p, exe_dir, rp)
            if q is None:
                if not weak: missing.append((p, name))
            elif q != "system" and q not in keep: todo.append((q, rp))
    return keep, missing

# ---------------------------------------------------------------- Windows
def build_windows(args, cache, out):
    t = TARGETS["windows"]; stage = os.path.join(args.build, "windows", "Fission"); rmtree(os.path.dirname(stage)); os.makedirs(stage)
    rt = os.path.join(stage, "runtime"); site = os.path.join(rt, "Lib", "site-packages")
    log("Windows: Python runtime")
    untar(runtime_tarball(cache, t["triple"]), os.path.join(args.build, "windows"))
    os.rename(os.path.join(args.build, "windows", "python"), rt)
    for junk in ("Scripts", "include", "libs", "tcl", "LICENSE.txt", "pythonw.exe"):
        rmtree(os.path.join(rt, junk))
    for p in glob.glob(os.path.join(rt, "**", "*.pdb"), recursive=True) + glob.glob(os.path.join(rt, "DLLs", "*tk*")) + \
             glob.glob(os.path.join(rt, "DLLs", "tcl*")) + glob.glob(os.path.join(rt, "DLLs", "_tkinter*")): rmtree(p)
    rmtree(site); prune_stdlib(os.path.join(rt, "Lib"))
    log("Windows: packages"); install_wheels(cache, t["plat"], site); prune_site(site)
    log("Windows: dropping libraries nothing loads")
    roots = glob.glob(os.path.join(site, "OCP", "*.pyd")) + glob.glob(os.path.join(site, "cadquery_ocp.libs", "*.dll"))
    roots += [p for m in QT_KEEP for p in glob.glob(os.path.join(site, "PySide6", m + ".pyd"))]
    plugins = os.path.join(site, "PySide6", "plugins")
    for cat in os.listdir(plugins) if os.path.isdir(plugins) else []:
        if cat not in ("platforms", "styles", "imageformats", "iconengines"): rmtree(os.path.join(plugins, cat)); continue
        for p in glob.glob(os.path.join(plugins, cat, "*.dll")):
            if cat == "platforms" and "qwindows" not in os.path.basename(p): os.remove(p)
            elif "pdf" in os.path.basename(p).lower(): os.remove(p)
            else: roots.append(p)
    roots += glob.glob(os.path.join(site, "numpy", "**", "*.pyd"), recursive=True) + glob.glob(os.path.join(site, "numpy.libs", "*.dll"))
    keep = windows_closure(site, roots); dropped = 0
    for d in ("vtk.libs", "vtkmodules", "PySide6"):
        for dd, _, fs in os.walk(os.path.join(site, d)):
            for f in fs:
                p = os.path.join(dd, f)
                if os.path.realpath(p) in keep: continue
                if d == "vtkmodules" or f.lower().endswith((".dll", ".pyd")):     # VTK's Python layer isn't used at all
                    dropped += os.path.getsize(p); os.remove(p)
    os.makedirs(os.path.join(site, "vtk.libs"), exist_ok=True)              # OCP's __init__ adds this folder to the DLL path
    log(f"   removed {mb(dropped)} of unused libraries")
    app = os.path.join(stage, "app"); copy_app(args.src, args.version, app); shutil.copy(os.path.join(ASSETS, "fission-file.ico"), app)
    log("Windows: precompiling"); compile_all(cache, os.path.join(rt, "Lib"), app)
    log("Windows: Fission.exe launcher")
    gcc, windres = tool("x86_64-w64-mingw32-gcc"), tool("x86_64-w64-mingw32-windres")
    wb = os.path.join(args.build, "windows", "launcher"); os.makedirs(wb, exist_ok=True)
    shutil.copy(os.path.join(ASSETS, "fission.ico"), wb); shutil.copy(os.path.join(HERE, "launcher.rc"), wb)
    nums = [int(re.sub(r"\D", "", x) or 0) for x in (args.version.split(".") + ["0"])[:2]]
    run([windres, f"-DVMAJ={nums[0]}", f"-DVMIN={nums[1]}", '-DVSTR=\\"%s\\"' % args.version,
         "launcher.rc", "-O", "coff", "-o", "launcher.res"], cwd=wb)
    run([gcc, "-O2", "-s", "-municode", "-mwindows", os.path.join(HERE, "launcher.c"), os.path.join(wb, "launcher.res"),
         "-o", os.path.join(stage, "Fission.exe"), "-lshell32"])
    log(f"Windows: app folder is {mb(dir_size(stage))}")
    if args.test: test_windows(stage, args)
    portable = os.path.join(out, f"Fission-{args.version}-Windows-x64-portable.zip")
    log("Windows: portable zip"); make_zip(os.path.dirname(stage), "Fission", portable)
    setup = os.path.join(out, f"Fission-{args.version}-Windows-x64-Setup.exe")
    log("Windows: installer")
    run([tool("makensis"), "-V2", f"-DVERSION={args.version}", f"-DSRC={stage}", f"-DOUT={setup}", f"-DASSETS={ASSETS}",
         os.path.join(HERE, "fission.nsi")])
    return [setup, portable]

def test_windows(stage, args):
    """Run the app's --selftest under Wine. Wine 9 lacks two things every Windows 10/11 PC has, so the TEST copy gets
    stand-ins (the shipped files are untouched): an ICU stub in the Wine prefix (Qt's legacy codecs), and numpy 1.26
    instead of 2.x (Wine's C runtime is missing the complex-math functions numpy 2 calls on import)."""
    wine = shutil.which("wine")
    if not wine: log("   (no wine - skipping the Windows self-test)"); return
    log("Windows: self-test under wine")
    env = dict(os.environ, WINEDEBUG="-all", WINEDLLOVERRIDES="winedbg.exe=d")
    subprocess.run([wine, "cmd", "/c", "exit"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)   # create the prefix
    sys32 = os.path.join(env.get("WINEPREFIX", os.path.expanduser("~/.wine")), "drive_c", "windows", "system32")
    if not os.path.exists(os.path.join(sys32, "icuuc.dll")):
        run([tool("x86_64-w64-mingw32-gcc"), "-shared", "-O2", "-s", os.path.join(HERE, "wine", "icustub.c"), "-o", os.path.join(sys32, "icuuc.dll")])
    t = os.path.join(args.build, "windows", "wine-test", "Fission"); rmtree(os.path.dirname(t)); shutil.copytree(stage, t)
    site = os.path.join(t, "runtime", "Lib", "site-packages")
    for p in glob.glob(os.path.join(site, "numpy*")): rmtree(p)
    nd = os.path.join(args.cache, "wine-numpy"); os.makedirs(nd, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--only-binary=:all:", "--platform", "win_amd64", "--python-version", "3.12",
         "--implementation", "cp", "--abi", "cp312", "--target", site, "numpy==1.26.4"])
    out = os.path.join(args.build, "windows", "selftest.txt"); rmtree(out)
    cmd = [wine, os.path.join(t, "Fission.exe"), "--selftest", out]
    if not os.environ.get("DISPLAY") and shutil.which("xvfb-run"): cmd = ["xvfb-run", "-a"] + cmd
    # Wine maps a terminal / file / /dev/null stdout to handles Python rejects; a pipe works (on Windows there's no console at all)
    r = subprocess.run(cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    text = open(out).read() if os.path.exists(out) else ""
    print(text or r.stdout.decode(errors="ignore")[-3000:])
    if r.returncode != 0 or "SELFTEST OK" not in text: raise SystemExit("Windows self-test failed")

# ---------------------------------------------------------------- macOS
def build_mac(name, args, cache, out):
    t = TARGETS[name]; arch = t["arch"]; base = os.path.join(args.build, name); rmtree(base); os.makedirs(base)
    app = os.path.join(base, "Fission.app"); C = os.path.join(app, "Contents")
    for d in ("MacOS", "Resources", "lib"): os.makedirs(os.path.join(C, d))
    log(f"macOS {arch}: Python runtime")
    untar(runtime_tarball(cache, t["triple"]), base)
    py = os.path.join(base, "python")
    shutil.copy2(os.path.join(py, "bin", "python3.12"), os.path.join(C, "MacOS", "Fission"))     # the app's own executable
    shutil.copy2(os.path.join(py, "lib", "libpython3.12.dylib"), os.path.join(C, "lib"))
    shutil.copytree(os.path.join(py, "lib", "python3.12"), os.path.join(C, "lib", "python3.12"), symlinks=False)
    lib = os.path.join(C, "lib", "python3.12"); site = os.path.join(lib, "site-packages"); rmtree(py)
    for junk in glob.glob(os.path.join(lib, "config-3.12-*")) + glob.glob(os.path.join(lib, "lib-dynload", "_tkinter*")): rmtree(junk)
    rmtree(site); prune_stdlib(lib)
    log(f"macOS {arch}: packages"); install_wheels(cache, t["plat"], site); prune_site(site)
    shutil.copy(os.path.join(HERE, "sitecustomize.py"), site)
    log(f"macOS {arch}: thinning universal binaries to {arch}")
    thinned = 0
    for p in glob.glob(os.path.join(site, "PySide6", "*")):                 # Qt's command-line tools (qmllint, lupdate, ...)
        if os.path.isfile(p) and "." not in os.path.basename(p) and is_macho(p): os.remove(p)
    for d, _, fs in os.walk(C):
        for f in fs:
            p = os.path.join(d, f)
            if os.path.islink(p) or not is_macho(p): continue
            info = macho_info(p)
            if info and len(info[0]) > 1 and arch in info[0]:
                before = os.path.getsize(p); thin_macho(p, arch); thinned += before - os.path.getsize(p)
    log(f"   saved {mb(thinned)}")
    log(f"macOS {arch}: dropping libraries nothing loads")
    exe_dir = os.path.join(C, "MacOS")
    roots = [os.path.join(exe_dir, "Fission")] + glob.glob(os.path.join(lib, "lib-dynload", "*.so"))
    roots += glob.glob(os.path.join(site, "OCP", "*.so")) + [p for m in QT_KEEP for p in glob.glob(os.path.join(site, "PySide6", m + ".abi3.so"))]
    roots += glob.glob(os.path.join(site, "numpy", "**", "*.so"), recursive=True)
    plugins = os.path.join(site, "PySide6", "Qt", "plugins")
    for cat in os.listdir(plugins) if os.path.isdir(plugins) else []:
        if cat not in ("platforms", "styles", "imageformats", "iconengines"): rmtree(os.path.join(plugins, cat)); continue
        for p in glob.glob(os.path.join(plugins, cat, "*.dylib")):
            if cat == "platforms" and "cocoa" not in os.path.basename(p): os.remove(p)
            elif "pdf" in os.path.basename(p).lower(): os.remove(p)
            else: roots.append(p)
    keep, missing = mac_closure(roots, exe_dir)
    if missing:
        for p, n in missing[:20]: print("   MISSING", os.path.relpath(p, C), "->", n)
        raise SystemExit("some libraries can't be found inside the app")
    dropped = 0
    qtlib = os.path.join(site, "PySide6", "Qt", "lib")
    for fw in glob.glob(os.path.join(qtlib, "*.framework")):
        binary = os.path.join(fw, "Versions", "A", os.path.basename(fw)[:-10])
        if not any(k.startswith(os.path.realpath(fw)) for k in keep): dropped += dir_size(fw); rmtree(fw)
    for d in ("vtkmodules", "vtk.libs", os.path.join("PySide6", "Qt", "lib")):
        for dd, _, fs in os.walk(os.path.join(site, d)):
            for f in fs:
                p = os.path.join(dd, f)
                if is_macho(p) and os.path.realpath(p) not in keep: dropped += os.path.getsize(p); os.remove(p)
                elif d == "vtkmodules" and f.endswith((".py", ".pyi")): os.remove(p)
    log(f"   removed {mb(dropped)} of unused libraries")
    keep, missing = mac_closure(roots, exe_dir)                           # re-check after pruning
    if missing: raise SystemExit(f"pruning broke a dependency: {missing[:3]}")
    bad = [p for d, _, fs in os.walk(C) for f in fs for p in [os.path.join(d, f)] if is_macho(p) and arch not in (macho_info(p) or [set()])[0]]
    if bad: raise SystemExit(f"binaries without {arch} code: {bad[:5]}")
    res = os.path.join(C, "Resources"); appdir = os.path.join(res, "app"); copy_app(args.src, args.version, appdir)
    shutil.copy(os.path.join(ASSETS, "fission.icns"), res); shutil.copy(os.path.join(ASSETS, "fission-file.icns"), res)
    log(f"macOS {arch}: precompiling"); compile_all(cache, lib, appdir)
    with open(os.path.join(C, "Info.plist"), "wb") as f: plistlib.dump(info_plist(args.version), f)
    with open(os.path.join(C, "PkgInfo"), "w") as f: f.write("APPL????")
    log(f"macOS {arch}: ad-hoc signing"); rcs = rcodesign(cache)
    run_quiet([rcs, "sign", app])
    # every Mach-O must carry a code directory (Apple Silicon refuses to run unsigned code), and the bundle a seal
    unsigned = []
    for d, _, fs in os.walk(C):
        for f in fs:
            p = os.path.join(d, f)
            if os.path.islink(p) or not is_macho(p): continue
            r = subprocess.run([rcs, "print-signature-info", p], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if "code_directory:" not in r.stdout: unsigned.append(os.path.relpath(p, C))
    if unsigned: raise SystemExit(f"unsigned binaries: {unsigned[:5]}")
    if not os.path.exists(os.path.join(C, "_CodeSignature", "CodeResources")): raise SystemExit("the bundle has no signature seal")
    log("   every binary signed, bundle sealed")
    log(f"macOS {arch}: Fission.app is {mb(dir_size(app))}")
    z = os.path.join(out, f"Fission-{args.version}-macOS-{arch}.zip"); make_zip(base, "Fission.app", z)
    return [z]

def info_plist(version):
    return {
        "CFBundleName": "Fission", "CFBundleDisplayName": "Fission", "CFBundleExecutable": "Fission",
        "CFBundleIdentifier": "org.fission.cad", "CFBundlePackageType": "APPL", "CFBundleIconFile": "fission.icns",
        "CFBundleShortVersionString": version, "CFBundleVersion": version, "CFBundleInfoDictionaryVersion": "6.0",
        "LSMinimumSystemVersion": "13.0", "NSHighResolutionCapable": True, "NSRequiresAquaSystemAppearance": True,
        "LSApplicationCategoryType": "public.app-category.graphics-design", "NSPrincipalClass": "NSApplication",
        "CFBundleDocumentTypes": [{"CFBundleTypeName": "Fission design", "CFBundleTypeRole": "Editor", "LSHandlerRank": "Owner",
                                   "CFBundleTypeIconFile": "fission-file.icns", "LSItemContentTypes": ["org.fission.design"]}],
        "UTExportedTypeDeclarations": [{"UTTypeIdentifier": "org.fission.design", "UTTypeDescription": "Fission design",
                                        "UTTypeConformsTo": ["public.json", "public.data"], "UTTypeIconFile": "fission-file.icns",
                                        "UTTypeTagSpecification": {"public.filename-extension": ["fission"]}}],
    }

def run_quiet(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode: print(r.stdout[-3000:]); raise SystemExit(f"failed: {' '.join(cmd[:3])}")

def make_zip(parent, name, dest):
    """Zip keeping Unix permissions (the Mac app's executables must stay executable)."""
    rmtree(dest)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        top = os.path.join(parent, name)
        for d, ds, fs in os.walk(top):
            ds.sort()
            rel_d = os.path.relpath(d, parent)
            zi = zipfile.ZipInfo(rel_d + "/"); zi.external_attr = (0o40755 << 16) | 0x10; z.writestr(zi, "")
            for f in sorted(fs):
                p = os.path.join(d, f); st = os.stat(p)
                zi = zipfile.ZipInfo.from_file(p, os.path.join(rel_d, f)); zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = ((st.st_mode & 0xFFFF) << 16); zi.create_system = 3
                with open(p, "rb") as src: z.writestr(zi, src.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    log(f"   {os.path.basename(dest)}  {mb(os.path.getsize(dest))}")

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="windows, mac-arm64, mac-x86_64, mac, all (default)")
    ap.add_argument("--src", default=os.path.join(ROOT, "fission.py"))
    ap.add_argument("--version")
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--build", default=os.path.join(ROOT, "build", "xbuild"))
    ap.add_argument("--cache", default=os.environ.get("FISSION_CACHE") or os.path.join(ROOT, "build", "cache"))
    ap.add_argument("--test", action="store_true", help="self-test the Windows build under wine")
    args = ap.parse_args()
    want = []
    for x in args.targets or ["all"]:
        want += list(TARGETS) if x == "all" else ["mac-arm64", "mac-x86_64"] if x == "mac" else [x]
    for x in want:
        if x not in TARGETS: ap.error(f"unknown target {x}")
    args.src = os.path.abspath(args.src)
    if not args.version:
        m = re.search(r"Fission (\d+(?:\.\d+)+)", open(args.src, encoding="utf-8").read(3000)); args.version = m.group(1) if m else "0.0"
    for d in (args.out, args.build, args.cache): os.makedirs(d, exist_ok=True)
    for mod in ("pefile", "macholib"):
        try: __import__(mod)
        except ImportError: run([sys.executable, "-m", "pip", "install", "-q", "--user", mod]); __import__(mod)
    made = []
    for x in dict.fromkeys(want):
        made += build_windows(args, args.cache, args.out) if x == "windows" else build_mac(x, args, args.cache, args.out)
    log("done:")
    for p in made: print(f"    {p}  ({mb(os.path.getsize(p))})")

if __name__ == "__main__":
    main()
