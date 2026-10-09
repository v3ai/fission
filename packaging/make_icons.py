"""Render Fission's app icon and write it in every format the installers need.

    python packaging/make_icons.py [fission.py]   ->  packaging/assets/fission.png / .ico / .icns  (+ fission-file.ico)

Uses the same vector drawing code as the toolbar icons, so the app icon always matches the UI.
"""
import importlib.util, io, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "fission.py"))
OUT = os.path.join(HERE, "assets")

from PySide6 import QtWidgets as W, QtGui as G, QtCore as C
from PIL import Image

app = W.QApplication.instance() or W.QApplication([])
spec = importlib.util.spec_from_file_location("fission_src", SRC); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)

def render(size, glyph="extrude", doc=False):
    """Rounded tile (light, like the app's ribbon) with the extrude glyph; doc=True draws a page for .fission files."""
    img = G.QImage(size, size, G.QImage.Format_ARGB32); img.fill(C.Qt.transparent)
    p = G.QPainter(img); p.setRenderHint(G.QPainter.Antialiasing)
    s = size/1024.0
    if doc:
        page = G.QPainterPath(); page.moveTo(200*s, 70*s); page.lineTo(650*s, 70*s); page.lineTo(824*s, 244*s)
        page.lineTo(824*s, 954*s); page.lineTo(200*s, 954*s); page.closeSubpath()
        p.setPen(G.QPen(G.QColor("#9aa4ae"), 14*s)); p.setBrush(G.QColor("#ffffff")); p.drawPath(page)
        fold = G.QPainterPath(); fold.moveTo(650*s, 70*s); fold.lineTo(650*s, 244*s); fold.lineTo(824*s, 244*s); fold.closeSubpath()
        p.setBrush(G.QColor("#e3e8ee")); p.drawPath(fold)
        box = C.QRectF(262*s, 330*s, 500*s, 500*s)
    else:
        r = C.QRectF(64*s, 64*s, 896*s, 896*s)
        grad = G.QLinearGradient(r.topLeft(), r.bottomLeft()); grad.setColorAt(0, G.QColor("#ffffff")); grad.setColorAt(1, G.QColor("#dfe7ef"))
        p.setPen(G.QPen(G.QColor("#b9c6d3"), 10*s)); p.setBrush(grad); p.drawRoundedRect(r, 200*s, 200*s)
        box = C.QRectF(150*s, 150*s, 724*s, 724*s)
    p.save(); p.translate(box.topLeft()); p.scale(box.width()/40.0, box.height()/40.0)
    F.ICONS[glyph](p); p.restore(); p.end()
    buf = C.QBuffer(); buf.open(C.QIODevice.WriteOnly); img.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA")

os.makedirs(OUT, exist_ok=True)
big = render(1024); big.save(os.path.join(OUT, "fission.png"))
big.resize((256, 256), Image.LANCZOS).save(os.path.join(OUT, "fission-256.png"))
sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
big.save(os.path.join(OUT, "fission.ico"), sizes=sizes)
render(1024, doc=True).save(os.path.join(OUT, "fission-file.ico"), sizes=sizes)
big.save(os.path.join(OUT, "fission.icns"))
render(1024, doc=True).save(os.path.join(OUT, "fission-file.icns"))
print("icons written to", OUT)
