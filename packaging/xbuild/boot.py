"""Starts Fission inside the bundled Python (Windows: run by Fission.exe; macOS: called from sitecustomize).
Runs fission as a module so its precompiled bytecode is used (much faster start than compiling the 11k-line script)."""
import os, runpy, sys

def run(args):
    app = os.path.dirname(os.path.abspath(__file__))
    if app not in sys.path: sys.path.insert(0, app)
    # PyOpenGL would let a stray XDG_SESSION_TYPE (Wine, MSYS, ...) pick the Linux GLX backend - pin the native one
    os.environ["PYOPENGL_PLATFORM"] = {"win32": "nt", "darwin": "darwin"}.get(sys.platform, os.environ.get("PYOPENGL_PLATFORM", ""))
    if not os.environ["PYOPENGL_PLATFORM"]: del os.environ["PYOPENGL_PLATFORM"]
    sys.argv = [os.path.join(app, "fission.py")] + [a for a in args if not a.startswith("-psn")]
    runpy.run_module("fission", run_name="__main__", alter_sys=True)

if __name__ == "__main__":
    run(sys.argv[1:])
