# Fission: Free 3d modeling software


![PISTON](piston.png)


## How to run

Currently only linux versions have been tested but there's windows and mac
options available up now if you'd like to try them out, however linux will be 
the primary focus

### Running appimage

Easist option

1. Download AppImage from releases
2. right click --> properties --> permissions, click "Allow Executing File as Program" (or similar)
3. Double click it


From terminal

1. Download AppImage from releases
2. chmod +x fission.AppImage
3. ./fission.AppImage


### Running on Windows / macOS

Download from releases:

- Windows: `Fission-<version>-Windows-x64-Setup.exe` (installer) or the `-portable.zip`
- macOS: `Fission-<version>-macOS-arm64.zip` (Apple Silicon) or `-macOS-x86_64.zip` (Intel), unzip and
  drag Fission.app to Applications. The first time, right click --> Open (the app isn't notarized)


### Running in python 

1. clone the project 
2. python3 -m venv venv
4. source ./venv/bin/activate
5. pip install -r requirements.txt
6. python3 fission.py

### Building

Releases are built automatically by GitHub Actions (`.github/workflows/release.yml`) whenever the version in
the header of `fission.py` ("Fission X.Y") changes on main, or a `vX.Y` tag is pushed.

To build locally on Linux:

    ./packaging/linux/build_appimage.sh fission.py              # Linux AppImage
    python3 packaging/xbuild/xbuild.py windows mac --src fission.py   # Windows + macOS, cross-built from Linux
                                                                # (needs: sudo apt install nsis gcc-mingw-w64-x86-64 zip)

Any build can be checked with `<app> --selftest report.txt` (exits 0 and writes "SELFTEST OK");
on macOS run `Fission.app/Contents/MacOS/Fission -- --selftest report.txt`.
`packaging/windows/build_windows.ps1` and `packaging/macos/build_mac.sh` build natively with PyInstaller instead.


### Contributing

Contributions are welcome from anyone, submit a PR, there are likely bugs
and things that don't work as they're intended, please let me know

Currently Fission only has basic solid sketch, but surface modeling, and others
are in the works.

Has only been tried on linux mint, if other distros don't work please lmk

fission uses a bunch of open source libraries all basically bundled together
with python.

### Documentation

basically click create sketch, select a plane, draw it, and extrude.
if you want to take away material in the size of a sketch put a negative
number in while extruding.


