# macOS app bundle: Contents/MacOS/Fission is the bundled Python itself. When it starts as the app (from Finder,
# the Dock, "open", or with a .fission file / --selftest), hand over to Fission and exit with its exit code.
# Run any other way (e.g. "Fission -c ..."), it behaves as plain Python.
import os, sys
def _fission_boot():
    exe = os.path.realpath(sys.executable)
    if not exe.endswith(os.path.join("Contents", "MacOS", "Fission")): return
    a = list(sys.argv)
    first = a[0] if a else ""
    if first and not first.startswith("-psn") and not first.startswith("--") and not first.lower().endswith(".fission"): return
    args = a[1:] if first == "" or first.startswith("-psn") else a
    sys.dont_write_bytecode = True        # everything is precompiled; writing into the signed bundle would break its seal
    app = os.path.join(os.path.dirname(os.path.dirname(exe)), "Resources", "app")
    sys.path.insert(0, app)
    code = 0
    try:
        import boot; boot.run(args)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        import traceback; traceback.print_exc(); code = 1
    try: sys.stdout.flush(); sys.stderr.flush()
    except Exception: pass
    os._exit(code)
_fission_boot()
