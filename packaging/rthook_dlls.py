# PyInstaller runtime hook: on Windows, DLLs collected into sub-folders (VTK's vtkmodules\, OpenCascade's
# cadquery_ocp.libs\) aren't on the DLL search path by default - add them before OCP is imported.
import os, sys
if sys.platform == "win32" and hasattr(sys, "_MEIPASS"):
    for sub in ("", "vtkmodules", "cadquery_ocp.libs", "vtk.libs", "numpy.libs", os.path.join("PySide6")):
        d = os.path.join(sys._MEIPASS, sub)
        if os.path.isdir(d):
            try: os.add_dll_directory(d)
            except OSError: pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
