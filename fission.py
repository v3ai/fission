#!/usr/bin/env python3
"""Fission 0.7 - a free, Fusion-360-styled CAD app on the OpenCascade kernel (cadquery-ocp).

Sketch (S) on the ground, the front / right origin planes or any flat face. Inside the sketch: Line (L), Rectangle (R),
Circle (C), arcs, polygons, ellipses, slots, splines, conics, text, Dimension (D), Trim (T), Offset (O), Project (P),
Construction (X) and every geometric constraint; Finish Sketch (Ctrl+Enter) to return to the solid tools.
Click the regions you want (e.g. the ring between two circles), then Extrude (E) or Revolve (V).
Modify: Fillet (F), Chamfer (H), Move (M), Circular pattern (P), Gear (G), Thread (T), Drawing sheet (D).
Undo / redo: Ctrl+Z / Ctrl+Y, or click a step in the timeline at the bottom.
Viewport: wheel = zoom, middle-drag = pan, shift+middle-drag (or right-drag) = orbit, Ctrl while drawing = no snap.
Run:  pip install -r requirements.txt  &&  python fission.py
"""
import copy, datetime, json, math, os, re, sys, tempfile, time, traceback, zipfile
import numpy as np
from PySide6 import QtWidgets as W, QtCore as C, QtGui as G
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt, gluUnProject
from OCP.gp import gp_Pnt, gp_Vec, gp_Ax1, gp_Ax2, gp_Ax3, gp_Dir, gp_Circ, gp_Trsf, gp_Pnt2d, gp_Dir2d
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge,
                                BRepBuilderAPI_MakeWire, BRepBuilderAPI_Transform)
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol, BRepPrimAPI_MakeCylinder
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet, BRepFilletAPI_MakeChamfer
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.GCPnts import GCPnts_QuasiUniformDeflection
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Circle
from OCP.BRep import BRep_Tool, BRep_Builder
from OCP.BRepTools import BRepTools
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SOLID, TopAbs_REVERSED
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common
from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.TopoDS import TopoDS, TopoDS_Compound
from OCP.TopLoc import TopLoc_Location
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
from OCP.Geom import Geom_CylindricalSurface
from OCP.Geom2d import Geom2d_Line
from OCP.BRepLib import BRepLib
from OCP.GC import GC_MakeArcOfCircle
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

def st(cls, name):
    """OCP static methods are 'Name_s' in older builds and plain 'Name' in newer ones."""
    return getattr(cls, name + "_s", None) or getattr(cls, name)
to_face, to_edge, to_solid = st(TopoDS, "Face"), st(TopoDS, "Edge"), st(TopoDS, "Solid")
triangulation, brep_write = st(BRep_Tool, "Triangulation"), st(BRepTools, "Write")
is_closed_on, degenerated = st(BRep_Tool, "IsClosed"), st(BRep_Tool, "Degenerated")
volume_props, bnd_add, build_curves3d = st(BRepGProp, "VolumeProperties"), st(BRepBndLib, "Add"), st(BRepLib, "BuildCurves3d")

class V:
    """Minimal 3-vector for display math."""
    __slots__ = ("x", "y", "z")
    def __init__(s, x=0, y=0, z=0): s.x, s.y, s.z = float(x), float(y), float(z)
    def __add__(s, o): return V(s.x+o.x, s.y+o.y, s.z+o.z)
    def __sub__(s, o): return V(s.x-o.x, s.y-o.y, s.z-o.z)
    def __mul__(s, k): return V(s.x*k, s.y*k, s.z*k)
    def __neg__(s): return V(-s.x, -s.y, -s.z)
    @property
    def Length(s): return math.sqrt(s.x**2 + s.y**2 + s.z**2)
    def cross(s, o): return V(s.y*o.z - s.z*o.y, s.z*o.x - s.x*o.z, s.x*o.y - s.y*o.x)
    def dot(s, o): return s.x*o.x + s.y*o.y + s.z*o.z
    def normalize(s): L = s.Length or 1; return V(s.x/L, s.y/L, s.z/L)
    def t(s): return (s.x, s.y, s.z)

# ---- units: the model is always in millimetres; DISPLAY["unit"] only changes what you see and type ----
UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "um": 0.001, "µm": 0.001, "in": 25.4, "inch": 25.4, "inches": 25.4, '"': 25.4,
         "ft": 304.8, "foot": 304.8, "feet": 304.8, "'": 304.8, "yd": 914.4}
DISPLAY = {"unit": "mm"}
_UNIT_RE = r'(mm|cm|µm|um|m|inches|inch|in|ft|feet|foot|yd|"|\')'
_NUM_RE = r'(\d+\.?\d*(?:e[-+]?\d+)?|\.\d+)'

def unit_k(): return UNITS[DISPLAY["unit"]]

def flen(mm, nd=None):
    """A length (stored in mm) as text in the display unit, e.g. '12.5 mm' or '0.492 in'."""
    u = DISPLAY["unit"]; nd = nd if nd is not None else (2 if u == "mm" else 3)
    return f"{fmt(mm/UNITS[u], nd)} {u}"

def parse_len(text):
    """Typed length -> mm. Accepts '12', '12 mm', '3.5cm', '0.2 m', '1/4 in', '2 ft', '3"', '1 in + 2 mm', '10*2'.
    A number with no unit is in the current display unit."""
    t = text.strip().lower().replace(",", ".")
    if not t: raise ValueError("empty")
    units = re.findall(_UNIT_RE, t); k = unit_k()
    m = re.fullmatch(r'(.*?\S)\s*' + _UNIT_RE, t)
    if m and len(units) == 1: expr, k = m.group(1), UNITS[m.group(2)]          # '1/4 in': unit applies to all of it
    else: expr = re.sub(_NUM_RE + r'\s*' + _UNIT_RE, lambda q: f"({q.group(1)}*{UNITS[q.group(2)]/k!r})", t)
    if not re.fullmatch(r'[\d\.\s+\-*/()e]*', expr): raise ValueError("not a length")
    v = eval(expr, {"__builtins__": {}}, {})
    return float(v)*k

def fmt(x, nd=2):
    t = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if t in ("-0", "") else t

# ---- Fusion-like palette ----
BG_TOP, BG_BOT = (0.996, 0.996, 1.0), (0.905, 0.915, 0.93)
BODY, EDGE, SEL, SKETCH = (0.66, 0.65, 0.61), (0.11, 0.11, 0.12), (0.02, 0.59, 0.84), (0.1, 0.3, 0.65)
HILITE = (0.3, 0.64, 1.0)        # translucent blue used for hovered / selected faces
FILL, DRAW = (0.35, 0.6, 0.9), ()
LIGHT = (0.2, 0.45, 1.0)         # key-light direction (world space); also drives the ground shadow
ACCENT = "#0696d7"               # Fusion's UI blue

# ---- procedurally drawn icons (no image assets to ship) ----
INK, BLUE, BLUE_D, GREEN, ORANGE = "#555d66", "#2f86d0", "#1d5c93", "#3fae49", "#f2a33a"

def _pen(c, w=1.2, dash=False):
    return G.QPen(G.QColor(c), w, C.Qt.DashLine if dash else C.Qt.SolidLine, C.Qt.RoundCap, C.Qt.RoundJoin)
def _pg(p, pts, fill=None, line=None, w=1.2):
    p.setBrush(G.QColor(fill) if fill else C.Qt.NoBrush); p.setPen(_pen(line, w) if line else C.Qt.NoPen)
    p.drawPolygon(G.QPolygonF([C.QPointF(*q) for q in pts]))
def _ln(p, x1, y1, x2, y2, col=INK, w=2.4):
    p.setPen(_pen(col, w)); p.drawLine(C.QLineF(x1, y1, x2, y2))
def _dots(p, pts, r=3, fill=BLUE, line=BLUE_D):
    p.setPen(_pen(line, 1)); p.setBrush(G.QColor(fill))
    for x, y in pts: p.drawRect(C.QRectF(x - r, y - r, 2*r, 2*r))
def _box(p, cx, ty, a, h, cols=("#a9d6f5", "#4a9fdc", "#2b79bd"), line=BLUE_D):
    b, m = ty + a, ty + a/2                      # isometric cuboid
    _pg(p, [(cx, ty), (cx+a, m), (cx, b), (cx-a, m)], cols[0], line)
    _pg(p, [(cx-a, m), (cx, b), (cx, b+h), (cx-a, m+h)], cols[1], line)
    _pg(p, [(cx, b), (cx+a, m), (cx+a, m+h), (cx, b+h)], cols[2], line)
def _layers(p, build):                           # block with a darker "depth" copy behind it
    for dx, dy, f in ((5, -4, "#2b79bd"), (0, 0, "#86c1ec")):
        q = G.QPainterPath(); build(q, dx, dy); q.closeSubpath()
        p.setBrush(G.QColor(f)); p.setPen(_pen(BLUE_D, 1.3)); p.drawPath(q)
def _arrowhead(p, x, y, ang, col, size=7):
    a = math.radians(ang)
    pts = [(x + size*math.cos(a), y + size*math.sin(a)), (x + size*0.75*math.cos(a + 2.3), y + size*0.75*math.sin(a + 2.3)),
           (x + size*0.75*math.cos(a - 2.3), y + size*0.75*math.sin(a - 2.3))]
    _pg(p, pts, col, col, 1)

def _sketch(p):
    p.setBrush(G.QColor(255, 255, 255, 200)); p.setPen(_pen("#6b7a8a", 1.6, True)); p.drawRect(C.QRectF(5, 6, 26, 26))
    _dots(p, ((5, 6), (31, 6), (5, 32)), 2.5)
    p.setBrush(G.QColor(GREEN)); p.setPen(_pen("#2b8a35", 1)); p.drawEllipse(C.QRectF(22, 22, 15, 15))
    _ln(p, 29.5, 26, 29.5, 33, "white", 2.2); _ln(p, 26, 29.5, 33, 29.5, "white", 2.2)
def _line(p): _ln(p, 9, 31, 31, 9); _dots(p, ((9, 31), (31, 9)), 3.5)
def _rect(p):
    p.setPen(_pen(BLUE, 2.4)); p.setBrush(G.QColor(143, 201, 242, 120)); p.drawRect(C.QRectF(6, 10, 28, 20))
    _dots(p, ((6, 10), (34, 30)), 3)
def _circle(p):
    p.setPen(_pen(BLUE, 2.4)); p.setBrush(G.QColor(143, 201, 242, 120)); p.drawEllipse(C.QRectF(7, 7, 26, 26))
    _dots(p, ((20, 20),), 2.2)
def _extrude(p):
    _box(p, 15, 15, 11, 11); _ln(p, 31.5, 12, 31.5, 30, ORANGE, 3.2); _pg(p, [(31.5, 2), (38, 13), (25, 13)], ORANGE, "#c77d12")
def _revolve(p):
    _ln(p, 20, 3, 20, 37, "#8a929b", 1.6)
    p.setPen(_pen(BLUE_D, 1.3)); p.setBrush(G.QColor("#86c1ec")); p.drawRect(C.QRectF(23, 9, 9, 18))
    q = G.QPainterPath(); q.moveTo(7, 26); q.cubicTo(5, 36, 35, 36, 34, 29)
    p.setPen(_pen(ORANGE, 2.8)); p.setBrush(C.Qt.NoBrush); p.drawPath(q); _arrowhead(p, 34, 28, -80, ORANGE, 7)
def _fillet(p):
    def f(q, dx, dy):
        q.moveTo(5+dx, 34+dy); q.lineTo(5+dx, 10+dy); q.lineTo(20+dx, 10+dy)
        q.arcTo(C.QRectF(8+dx, 10+dy, 24, 24), 90, -90); q.lineTo(32+dx, 34+dy)
    _layers(p, f)
def _chamfer(p):
    def f(q, dx, dy):
        q.moveTo(5+dx, 34+dy); q.lineTo(5+dx, 10+dy); q.lineTo(21+dx, 10+dy); q.lineTo(32+dx, 21+dy); q.lineTo(32+dx, 34+dy)
    _layers(p, f)
def _move(p):
    _box(p, 20, 12, 8, 8)
    for x1, y1, x2, y2, ang in ((20, 10, 20, 2, -90), (20, 30, 20, 38, 90), (12, 22, 3, 22, 180), (28, 22, 37, 22, 0)):
        _ln(p, x1, y1, x2, y2, ORANGE, 2.2); _arrowhead(p, x2, y2, ang, ORANGE, 5)
def _pattern(p):
    p.setPen(_pen("#8a929b", 1.2, True)); p.setBrush(C.Qt.NoBrush); p.drawEllipse(C.QRectF(7, 7, 26, 26))
    p.setPen(_pen(BLUE_D, 1)); p.setBrush(G.QColor("#4a9fdc"))
    for i in range(6):
        a = i*math.tau/6; p.drawEllipse(C.QPointF(20 + 13*math.cos(a), 20 + 13*math.sin(a)), 4, 4)
    p.setBrush(G.QColor(INK)); p.drawEllipse(C.QPointF(20, 20), 2, 2)
def gear_poly(cx, cy, n, ro, ri, frac=0.5):
    pts = []
    for i in range(n):
        a0 = i*math.tau/n; w = math.tau/n
        for a, r in ((a0, ri), (a0 + w*0.15, ro), (a0 + w*(0.15 + frac*0.7), ro), (a0 + w*(0.3 + frac*0.7), ri)):
            pts.append((cx + r*math.cos(a), cy + r*math.sin(a)))
    return pts
def _gear(p):
    _pg(p, gear_poly(20, 20, 10, 17, 12.5), "#86c1ec", BLUE_D, 1.2)
    p.setBrush(G.QColor("#f4f4f4")); p.setPen(_pen(BLUE_D, 1.2)); p.drawEllipse(C.QPointF(20, 20), 5, 5)
def _thread(p):
    _pg(p, [(13, 4), (27, 4), (27, 36), (13, 36)], "#c9d3dc", "#6f757d", 1.2)
    for y in range(7, 35, 5): _ln(p, 12, y + 3, 28, y, "#4a5159", 1.8)
def _drawing(p):
    _pg(p, [(4, 6), (36, 6), (36, 34), (4, 34)], "#ffffff", "#6f757d", 1.4)
    p.setPen(_pen(BLUE_D, 1.4)); p.setBrush(G.QColor("#cfe6f8"))
    p.drawRect(C.QRectF(8, 19, 10, 9)); p.drawRect(C.QRectF(8, 10, 10, 6)); p.drawRect(C.QRectF(21, 19, 6, 9))
    _pg(p, [(26, 29), (35, 29), (35, 33), (26, 33)], "#e3e6ea", "#6f757d", 1)
def _delete(p):
    _pg(p, [(11, 12), (29, 12), (27, 35), (13, 35)], "#c9d3dc", "#6f757d", 1.3); _ln(p, 7, 10, 33, 10, "#6f757d", 2.4)
    _ln(p, 17, 5, 23, 5, "#6f757d", 2.4)
    for x in (16, 20, 24): _ln(p, x, 16, x, 31, "#6f757d", 1.4)
def _eye(p, col="#6b7280", off=False):
    q = G.QPainterPath(); q.moveTo(4, 20); q.cubicTo(12, 9, 28, 9, 36, 20); q.cubicTo(28, 31, 12, 31, 4, 20)
    p.setPen(_pen(col, 3)); p.setBrush(C.Qt.NoBrush); p.drawPath(q)
    p.setBrush(G.QColor(col)); p.drawEllipse(C.QRectF(15, 15, 10, 10))
    if off: _ln(p, 7, 34, 33, 6, col, 3)
def _folder(p):
    _pg(p, [(4, 9), (16, 9), (19, 13), (36, 13), (36, 33), (4, 33)], "#a9aeb4", "#7d838a", 1.4)
def _body(p): _box(p, 20, 7, 14, 14, ("#e1e4e8", "#b1b7bf", "#8f969f"), "#6f757d")
def _doc(p):
    _pg(p, [(8, 4), (25, 4), (32, 11), (32, 36), (8, 36)], "#f7f7f7", "#7a828b", 1.6); _box(p, 20, 15, 7, 8)
def _file(p):
    _pg(p, [(9, 4), (24, 4), (31, 11), (31, 36), (9, 36)], "#5d6670", "#444c54", 1); _pg(p, [(24, 4), (24, 11), (31, 11)], "#b9c0c7")
def _save(p):
    _pg(p, [(7, 7), (29, 7), (33, 11), (33, 33), (7, 33)], "#5d6670", "#444c54", 1)
    p.setPen(C.Qt.NoPen); p.setBrush(G.QColor("#eef0f2")); p.drawRect(C.QRectF(13, 7, 14, 9)); p.setBrush(G.QColor("#9aa3ab")); p.drawRect(C.QRectF(12, 22, 16, 11))
def _undo(p):
    q = G.QPainterPath(); q.moveTo(31, 32); q.cubicTo(34, 20, 24, 12, 12, 16)
    p.setPen(_pen("#5d6670", 3.2)); p.setBrush(C.Qt.NoBrush); p.drawPath(q); _pg(p, [(5, 17), (16, 7), (17, 23)], "#5d6670", "#5d6670", 1)
def _redo(p): p.translate(40, 0); p.scale(-1, 1); _undo(p)
def _rot_l(p):
    q = G.QPainterPath(); q.moveTo(33, 24); q.cubicTo(30, 8, 12, 6, 8, 20)
    p.setPen(_pen("#4a5159", 3)); p.setBrush(C.Qt.NoBrush); p.drawPath(q); _arrowhead(p, 8, 23, 100, "#4a5159", 9)
def _rot_r(p): p.translate(40, 0); p.scale(-1, 1); _rot_l(p)
def _orbit(p):
    q = G.QPainterPath(); q.arcMoveTo(C.QRectF(8, 8, 24, 24), 60); q.arcTo(C.QRectF(8, 8, 24, 24), 60, 270)
    p.setPen(_pen(INK, 3)); p.setBrush(C.Qt.NoBrush); p.drawPath(q)
    _pg(p, [(25, 27), (37, 26), (31, 17)], INK, INK, 1); p.setBrush(G.QColor(INK)); p.drawEllipse(C.QRectF(17, 17, 6, 6))
def _pan(p):
    _ln(p, 20, 8, 20, 32, INK, 2.6); _ln(p, 8, 20, 32, 20, INK, 2.6)
    for t in ([(20, 2), (14, 10), (26, 10)], [(20, 38), (14, 30), (26, 30)], [(2, 20), (10, 14), (10, 26)], [(38, 20), (30, 14), (30, 26)]):
        _pg(p, t, INK, INK, 1)
def _zoom(p):
    p.setPen(_pen(INK, 3)); p.setBrush(G.QColor(255, 255, 255, 120)); p.drawEllipse(C.QRectF(6, 6, 20, 20)); _ln(p, 23, 23, 34, 34, INK, 4.5)
def _fit(p):
    for sx, sy in ((6, 6), (34, 6), (6, 34), (34, 34)):
        dx, dy = (9 if sx < 20 else -9), (9 if sy < 20 else -9)
        _ln(p, sx, sy, sx + dx, sy, INK, 2.6); _ln(p, sx, sy, sx, sy + dy, INK, 2.6)
    p.setPen(_pen(INK, 2)); p.setBrush(C.Qt.NoBrush); p.drawRect(C.QRectF(15, 15, 10, 10))
def _grid(p):
    p.setPen(_pen(INK, 2)); p.setBrush(C.Qt.NoBrush); p.drawRect(C.QRectF(6, 6, 28, 28))
    for t in (15.3, 24.7): _ln(p, t, 6, t, 34, INK, 1.8); _ln(p, 6, t, 34, t, INK, 1.8)
def _home(p): _pg(p, [(20, 5), (36, 20), (31, 20), (31, 34), (9, 34), (9, 20), (4, 20)], "#8a929b", "#6b727a", 1.2)
def _arrow_r(p): _pg(p, [(14, 10), (14, 30), (28, 20)], "#6b7280")
def _arrow_d(p): _pg(p, [(10, 14), (30, 14), (20, 28)], "#6b7280")
def _arrow_u(p): _pg(p, [(10, 26), (30, 26), (20, 12)], "#6b7280")
def _bar(p, x): p.setPen(C.Qt.NoPen); p.setBrush(G.QColor(INK)); p.drawRect(C.QRectF(x, 10, 4, 20))
def _first(p): _bar(p, 8); _pg(p, [(32, 10), (32, 30), (14, 20)], INK)
def _prev(p): _bar(p, 28); _pg(p, [(25, 10), (25, 30), (8, 20)], INK)
def _play(p): _pg(p, [(12, 9), (12, 31), (31, 20)], INK)
def _stop(p): p.setPen(C.Qt.NoPen); p.setBrush(G.QColor(INK)); p.drawRect(C.QRectF(11, 11, 18, 18))
def _next(p): _bar(p, 8); _pg(p, [(15, 10), (15, 30), (32, 20)], INK)
def _last(p): _bar(p, 28); _pg(p, [(8, 10), (8, 30), (25, 20)], INK)

ICONS = dict(sketch=_sketch, poly=_line, line=_line, rect=_rect, circle=_circle, extrude=_extrude, revolve=_revolve, fillet=_fillet,
             chamfer=_chamfer, move=_move, pattern=_pattern, gear=_gear, thread=_thread, drawing=_drawing, delete=_delete,
             eye=_eye, eyeoff=lambda p: _eye(p, "#b4b8be", True), folder=_folder, body=_body, doc=_doc, file=_file,
             save=_save, undo=_undo, redo=_redo, rot_l=_rot_l, rot_r=_rot_r, orbit=_orbit, pan=_pan, zoom=_zoom, fit=_fit,
             grid=_grid, home=_home, arrow_r=_arrow_r, arrow_d=_arrow_d, arrow_u=_arrow_u, first=_first, prev=_prev,
             play=_play, stop=_stop, next=_next, last=_last)

# ---- icons for the Fission 0.5 solid tools ----
def _circ(p, cx, cy, r, fill=None, line=BLUE_D, w=1.3):
    p.setPen(_pen(line, w) if line else C.Qt.NoPen); p.setBrush(G.QColor(fill) if fill else C.Qt.NoBrush); p.drawEllipse(C.QPointF(cx, cy), r, r)
def _ell(p, cx, cy, rx, ry, fill=None, line=BLUE_D, w=1.3):
    p.setPen(_pen(line, w) if line else C.Qt.NoPen); p.setBrush(G.QColor(fill) if fill else C.Qt.NoBrush); p.drawEllipse(C.QPointF(cx, cy), rx, ry)
def _path(p, pts, col=ORANGE, w=2.6, closed=False):
    q = G.QPainterPath(); q.moveTo(*pts[0])
    for i in range(1, len(pts) - 2, 3): q.cubicTo(*pts[i], *pts[i + 1], *pts[i + 2])
    p.setPen(_pen(col, w)); p.setBrush(C.Qt.NoBrush); p.drawPath(q)
def _cyl(p, cx, top, rx, h, fill="#86c1ec", side="#4a9fdc"):
    p.setPen(_pen(BLUE_D, 1.2)); p.setBrush(G.QColor(side))
    q = G.QPainterPath(); q.moveTo(cx - rx, top); q.lineTo(cx - rx, top + h); q.arcTo(C.QRectF(cx - rx, top + h - rx*0.4, 2*rx, rx*0.8), 180, 180)
    q.lineTo(cx + rx, top); q.closeSubpath(); p.drawPath(q); _ell(p, cx, top, rx, rx*0.4, fill)

def _i_sweep(p):
    _path(p, [(6, 34), (6, 18), (14, 8), (34, 8)], ORANGE, 2.6); _ell(p, 6, 32, 5, 2.4, "#86c1ec")
    p.setPen(_pen(BLUE_D, 1.2)); p.setBrush(G.QColor(74, 159, 220, 140)); q = G.QPainterPath(); q.moveTo(1, 32); q.cubicTo(1, 14, 10, 3, 34, 3)
    q.lineTo(34, 13); q.cubicTo(18, 13, 11, 20, 11, 32); q.closeSubpath(); p.drawPath(q)
def _i_loft(p):
    _pg(p, [(12, 4), (28, 4), (34, 10), (6, 10)], "#86c1ec", BLUE_D); _ell(p, 20, 33, 12, 4, "#86c1ec")
    _ln(p, 6, 10, 8, 33, BLUE_D, 1.4); _ln(p, 34, 10, 32, 33, BLUE_D, 1.4); _ln(p, 20, 10, 20, 37, "#8a929b", 1)
def _i_rib(p):
    _pg(p, [(4, 30), (36, 30), (36, 35), (4, 35)], "#c9d3dc", "#6f757d"); _pg(p, [(4, 8), (9, 8), (9, 30), (4, 30)], "#c9d3dc", "#6f757d")
    _pg(p, [(9, 12), (32, 30), (27, 30), (9, 17)], "#4a9fdc", BLUE_D)
def _i_web(p):
    _pg(p, [(4, 4), (36, 4), (36, 36), (4, 36)], None, "#6f757d", 1.6)
    for x in (13, 22, 31): _pg(p, [(x - 2, 4), (x + 1, 4), (x + 1, 36), (x - 2, 36)], "#4a9fdc", BLUE_D, 1)
    _pg(p, [(4, 18), (36, 18), (36, 21), (4, 21)], "#4a9fdc", BLUE_D, 1)
def _i_emboss(p):
    _pg(p, [(3, 22), (25, 10), (37, 18), (15, 30)], "#c9d3dc", "#6f757d")
    f = p.font(); f.setPixelSize(17); f.setBold(True); p.setFont(f); p.setPen(G.QColor(BLUE_D)); p.drawText(C.QRectF(9, 6, 24, 24), C.Qt.AlignCenter, "A")
def _i_hole(p):
    _box(p, 20, 6, 15, 12, ("#e1e4e8", "#b1b7bf", "#8f969f"), "#6f757d"); _ell(p, 20, 13.5, 6, 3, "#3b4148", "#2a2f35")
    _ell(p, 20, 13.5, 3.4, 1.7, "#111", None)
def _i_box(p): _box(p, 20, 6, 14, 14)
def _i_cylinder(p): _cyl(p, 20, 9, 12, 22)
def _i_sphere(p):
    g = G.QRadialGradient(15, 14, 18); g.setColorAt(0, G.QColor("#d4ebfb")); g.setColorAt(1, G.QColor("#2b79bd"))
    p.setPen(_pen(BLUE_D, 1.2)); p.setBrush(g); p.drawEllipse(C.QPointF(20, 20), 15, 15)
def _i_torus(p):
    p.setPen(_pen(BLUE_D, 1.2)); p.setBrush(G.QColor("#4a9fdc")); q = G.QPainterPath()
    q.addEllipse(C.QPointF(20, 20), 17, 10); q.addEllipse(C.QPointF(20, 19), 7, 3.5); q.setFillRule(C.Qt.OddEvenFill); p.drawPath(q)
def _i_coil(p):
    for k in range(5):
        y = 8 + k*6; _ell(p, 20, y, 13, 3.2, None, BLUE_D if k % 2 else "#4a9fdc", 2.4)
def _i_pipe(p):
    _path(p, [(5, 34), (5, 16), (12, 8), (34, 8)], "#4a9fdc", 8); _path(p, [(5, 34), (5, 16), (12, 8), (34, 8)], "#cfe6f8", 3)
def _i_rpattern(p):
    for i in range(3):
        for j in range(3): _pg(p, [(5 + i*11, 5 + j*11), (12 + i*11, 5 + j*11), (12 + i*11, 12 + j*11), (5 + i*11, 12 + j*11)], "#4a9fdc" if i == j == 0 else "#a9d6f5", BLUE_D, 1)
def _i_ppattern(p):
    _path(p, [(3, 32), (12, 6), (26, 36), (37, 10)], "#8a929b", 1.4)
    for x, y in ((4, 30), (13, 18), (21, 24), (31, 20)): _circ(p, x + 1, y, 3.6, "#4a9fdc")
def _i_mirror(p):
    p.setPen(_pen("#8a929b", 1.4, True)); p.drawLine(C.QLineF(20, 3, 20, 37))
    _pg(p, [(17, 8), (17, 32), (4, 32)], "#4a9fdc", BLUE_D); _pg(p, [(23, 8), (23, 32), (36, 32)], "#a9d6f5", BLUE_D)
def _i_thicken(p):
    _pg(p, [(3, 24), (25, 12), (37, 18), (15, 30)], "#4a9fdc", BLUE_D); _pg(p, [(3, 30), (15, 36), (15, 30), (3, 24)], "#2b79bd", BLUE_D)
    _ln(p, 26, 13, 26, 3, ORANGE, 2.4); _arrowhead(p, 26, 3, -90, ORANGE, 5)
def _i_bfill(p):
    _pg(p, [(4, 10), (36, 10), (36, 34), (4, 34)], None, "#6f757d", 1.4); _pg(p, [(4, 22), (20, 22), (20, 34), (4, 34)], "#4a9fdc", BLUE_D)
    _ln(p, 20, 4, 20, 38, "#8a929b", 1.4); _ln(p, 1, 22, 39, 22, "#8a929b", 1.4)
def _i_presspull(p):
    _box(p, 20, 14, 13, 10, ("#a9d6f5", "#b1b7bf", "#8f969f"), "#6f757d"); _ln(p, 20, 18, 20, 3, ORANGE, 2.6); _arrowhead(p, 20, 2, -90, ORANGE, 6)
def _i_shell(p):
    _box(p, 20, 8, 14, 14, ("#e1e4e8", "#b1b7bf", "#8f969f"), "#6f757d"); _pg(p, [(20, 11), (31, 16.5), (20, 22), (9, 16.5)], "#3b4148", "#2a2f35")
def _i_draft(p):
    _pg(p, [(10, 6), (30, 6), (36, 34), (4, 34)], "#86c1ec", BLUE_D); p.setPen(_pen("#8a929b", 1.2, True)); p.drawLine(C.QLineF(10, 6, 10, 34))
def _i_scale(p):
    _pg(p, [(4, 22), (18, 22), (18, 36), (4, 36)], "#a9d6f5", BLUE_D); _pg(p, [(12, 4), (36, 4), (36, 28), (12, 28)], None, BLUE_D, 1.6)
    _ln(p, 18, 22, 32, 8, ORANGE, 2.2); _arrowhead(p, 33, 7, -45, ORANGE, 5)
def _i_combine(p):
    _circ(p, 15, 20, 11, "#a9d6f5"); p.setBrush(G.QColor(74, 159, 220, 170)); p.drawEllipse(C.QPointF(25, 20), 11, 11)
def _i_offsetface(p):
    _box(p, 18, 12, 13, 12, ("#e1e4e8", "#b1b7bf", "#4a9fdc"), "#6f757d"); _ln(p, 26, 28, 37, 34, ORANGE, 2.4); _arrowhead(p, 37, 34, 30, ORANGE, 5)
def _i_replaceface(p):
    _box(p, 16, 12, 12, 12, ("#4a9fdc", "#b1b7bf", "#8f969f"), "#6f757d"); _pg(p, [(2, 8), (24, 0), (38, 6), (16, 14)], None, ORANGE, 1.6)
def _i_splitface(p):
    _pg(p, [(4, 6), (36, 6), (36, 34), (4, 34)], "#a9d6f5", BLUE_D); _ln(p, 4, 30, 36, 10, ORANGE, 2.4)
def _i_splitbody(p):
    _box(p, 20, 6, 14, 14, ("#e1e4e8", "#b1b7bf", "#8f969f"), "#6f757d"); _pg(p, [(2, 18), (24, 7), (38, 22), (16, 33)], None, ORANGE, 1.8)
def _i_align(p):
    _pg(p, [(4, 22), (18, 22), (18, 36), (4, 36)], "#a9d6f5", BLUE_D); _pg(p, [(22, 4), (36, 4), (36, 18), (22, 18)], "#4a9fdc", BLUE_D)
    _ln(p, 18, 22, 22, 18, ORANGE, 2.4)
def _i_material(p):
    g = G.QRadialGradient(15, 14, 18); g.setColorAt(0, G.QColor("#f2f4f6")); g.setColorAt(1, G.QColor("#6f757d"))
    p.setPen(_pen("#555", 1.2)); p.setBrush(g); p.drawEllipse(C.QPointF(20, 20), 15, 15)
def _i_appearance(p):
    for (x, y), c in zip(((13, 13), (27, 13), (13, 27), (27, 27)), ("#d33f2f", "#2f86d0", "#3fae49", "#f2a33a")): _circ(p, x, y, 7, c, "#555", 1)
def _i_defeature(p):
    _pg(p, [(4, 6), (36, 6), (36, 34), (4, 34)], "#c9d3dc", "#6f757d"); _circ(p, 20, 20, 7, "#3b4148", "#2a2f35")
    _ln(p, 9, 9, 31, 31, "#d33f2f", 2.8); _ln(p, 31, 9, 9, 31, "#d33f2f", 2.8)
def _i_compute(p):
    q = G.QPainterPath(); q.arcMoveTo(C.QRectF(7, 7, 26, 26), 30); q.arcTo(C.QRectF(7, 7, 26, 26), 30, 300)
    p.setPen(_pen(BLUE, 3)); p.setBrush(C.Qt.NoBrush); p.drawPath(q); _arrowhead(p, 31, 13, 70, BLUE, 7)
def _i_cplane(p): _pg(p, [(3, 26), (24, 12), (37, 18), (16, 32)], "#f6c391", "#d67a1e", 1.5)
def _i_caxis(p):
    p.setPen(_pen("#d67a1e", 2.4, True)); p.drawLine(C.QLineF(6, 34, 34, 6))
def _i_cpoint(p):
    _ln(p, 6, 20, 34, 20, "#c7ccd1", 1); _ln(p, 20, 6, 20, 34, "#c7ccd1", 1); _pg(p, [(20, 13), (27, 20), (20, 27), (13, 20)], "#f6c391", "#d67a1e", 1.6)

ICONS.update(sweep=_i_sweep, loft=_i_loft, rib=_i_rib, web=_i_web, emboss=_i_emboss, hole=_i_hole, box=_i_box, cylinder=_i_cylinder,
             sphere=_i_sphere, torus=_i_torus, coil=_i_coil, pipe=_i_pipe, rpattern=_i_rpattern, ppattern=_i_ppattern, mirror=_i_mirror,
             thicken=_i_thicken, bfill=_i_bfill, presspull=_i_presspull, shell=_i_shell, draft=_i_draft, scale=_i_scale, combine=_i_combine,
             offsetface=_i_offsetface, replaceface=_i_replaceface, splitface=_i_splitface, splitbody=_i_splitbody, align=_i_align,
             material=_i_material, appearance=_i_appearance, defeature=_i_defeature, compute=_i_compute, cplane=_i_cplane, caxis=_i_caxis,
             cpoint=_i_cpoint, construct=_i_cplane, thread2=_thread, fillet2=_fillet, chamfer2=_chamfer, prim=_i_box, cpattern=_pattern)


# ---- icons for the Fission 0.6 sketch environment ----
SKB, SKD = "#2f86d0", "#1d5c93"
def _arcp(p, cx, cy, r, a0, span, col=SKB, w=2.4):
    p.setPen(_pen(col, w)); p.setBrush(C.Qt.NoBrush); p.drawArc(C.QRectF(cx - r, cy - r, 2*r, 2*r), int(a0*16), int(span*16))
def _s_rect3(p):
    _pg(p, [(6, 26), (24, 8), (34, 18), (16, 36)], "#c6e2f7", SKB, 2.2); _dots(p, ((6, 26), (24, 8), (34, 18)), 2.8)
def _s_rectc(p):
    p.setPen(_pen(SKB, 2.2)); p.setBrush(G.QColor(143, 201, 242, 110)); p.drawRect(C.QRectF(5, 9, 30, 22))
    p.setPen(_pen("#8a929b", 1, True)); p.drawLine(C.QLineF(5, 9, 35, 31)); _dots(p, ((20, 20), (35, 31)), 2.8)
def _s_circle2p(p):
    _circ(p, 20, 20, 13, "#d8ecfa", SKB, 2.2); p.setPen(_pen("#8a929b", 1, True)); p.drawLine(C.QLineF(7, 20, 33, 20)); _dots(p, ((7, 20), (33, 20)), 2.8)
def _s_circle3p(p): _circ(p, 20, 20, 13, "#d8ecfa", SKB, 2.2); _dots(p, ((7, 20), (29, 11), (25, 32)), 2.8)
def _s_circle2t(p):
    _ln(p, 4, 34, 36, 34, INK, 2); _ln(p, 4, 34, 22, 4, INK, 2); _circ(p, 20, 25, 9, "#d8ecfa", SKB, 2.2)
def _s_circle3t(p):
    _ln(p, 3, 35, 37, 35, INK, 2); _ln(p, 3, 35, 20, 4, INK, 2); _ln(p, 37, 35, 20, 4, INK, 2); _circ(p, 20, 25, 8.5, "#d8ecfa", SKB, 2.2)
def _s_arc3p(p): _arcp(p, 20, 26, 15, 10, 160); _dots(p, ((34.8, 23.4), (5.2, 23.4), (20, 11)), 2.8)
def _s_arcc(p):
    _arcp(p, 14, 26, 18, 0, 90); p.setPen(_pen("#8a929b", 1, True)); p.drawLine(C.QLineF(14, 26, 32, 26)); p.drawLine(C.QLineF(14, 26, 14, 8))
    _dots(p, ((14, 26), (32, 26), (14, 8)), 2.6)
def _s_arct(p): _ln(p, 4, 32, 18, 32, INK, 2.4); _arcp(p, 18, 20, 12, -90, 160); _dots(p, ((18, 32),), 2.6)
def _poly(p, n, r, rot=0.0, fill="#d8ecfa"):
    _pg(p, [(20 + r*math.cos(rot + k*math.tau/n), 20 + r*math.sin(rot + k*math.tau/n)) for k in range(n)], fill, SKB, 2.2)
def _s_polyc(p): _poly(p, 6, 15); _circ(p, 20, 20, 13, None, "#8a929b", 1)
def _s_polyi(p): _circ(p, 20, 20, 15, None, "#8a929b", 1); _poly(p, 6, 15)
def _s_polye(p): _poly(p, 6, 15, math.pi/6); _dots(p, ((33, 27.5), (20, 35)), 2.8)
def _s_ellipse(p):
    _ell(p, 20, 20, 16, 9, "#d8ecfa", SKB, 2.2); p.setPen(_pen("#8a929b", 1, True)); p.drawLine(C.QLineF(4, 20, 36, 20)); _dots(p, ((20, 20), (36, 20)), 2.6)
def _slotpath(p, x0, x1, y, r):
    q = G.QPainterPath(); q.moveTo(x0, y - r); q.lineTo(x1, y - r); q.arcTo(C.QRectF(x1 - r, y - r, 2*r, 2*r), 90, -180)
    q.lineTo(x0, y + r); q.arcTo(C.QRectF(x0 - r, y - r, 2*r, 2*r), 270, -180); p.setPen(_pen(SKB, 2.2)); p.setBrush(G.QColor("#d8ecfa")); p.drawPath(q)
def _s_slot(p): _slotpath(p, 12, 28, 20, 8); _dots(p, ((12, 20), (28, 20)), 2.4)
def _s_slotc(p): _slotpath(p, 12, 28, 20, 8); _dots(p, ((20, 20), (28, 20)), 2.4)
def _s_slota(p):
    p.setPen(_pen(SKB, 7.5)); p.setBrush(C.Qt.NoBrush); p.drawArc(C.QRectF(6, 10, 28, 28), 20*16, 140*16)
    p.setPen(_pen("#d8ecfa", 4.5)); p.drawArc(C.QRectF(6, 10, 28, 28), 20*16, 140*16)
def _s_spline(p): _path(p, [(4, 30), (12, 2), (24, 40), (36, 10)], SKB, 2.4); _dots(p, ((4, 30), (16, 20), (36, 10)), 2.6)
def _s_cspline(p):
    p.setPen(_pen("#8a929b", 1, True)); p.drawPolyline(G.QPolygonF([C.QPointF(4, 32), C.QPointF(12, 6), C.QPointF(28, 34), C.QPointF(36, 8)]))
    _path(p, [(4, 32), (12, 6), (28, 34), (36, 8)], SKB, 2.4); _dots(p, ((4, 32), (12, 6), (28, 34), (36, 8)), 2.4)
def _s_conic(p):
    p.setPen(_pen("#8a929b", 1, True)); p.drawPolyline(G.QPolygonF([C.QPointF(5, 33), C.QPointF(20, 5), C.QPointF(35, 33)]))
    q = G.QPainterPath(); q.moveTo(5, 33); q.quadTo(20, 5, 35, 33); p.setPen(_pen(SKB, 2.4)); p.drawPath(q); _dots(p, ((5, 33), (35, 33), (20, 5)), 2.4)
def _s_point(p):
    _ln(p, 20, 8, 20, 32, "#8a929b", 1); _ln(p, 8, 20, 32, 20, "#8a929b", 1); _dots(p, ((20, 20),), 4)
def _s_text(p):
    f = p.font(); f.setPixelSize(26); f.setBold(True); p.setFont(f); p.setPen(G.QColor(SKB)); p.drawText(C.QRectF(2, 2, 36, 36), C.Qt.AlignCenter, "A")
    _ln(p, 6, 35, 34, 35, "#8a929b", 1.2)
def _s_dim(p):
    _ln(p, 6, 30, 34, 30, INK, 1.6); _ln(p, 6, 10, 6, 34, "#8a929b", 1.2); _ln(p, 34, 10, 34, 34, "#8a929b", 1.2)
    _ln(p, 6, 18, 34, 18, SKB, 1.6); _arrowhead(p, 6, 18, 180, SKB, 5); _arrowhead(p, 34, 18, 0, SKB, 5)
    f = p.font(); f.setPixelSize(10); p.setFont(f); p.setPen(G.QColor(SKD)); p.drawText(C.QRectF(8, 4, 24, 12), C.Qt.AlignCenter, "10")
def _s_trim(p):
    _ln(p, 4, 20, 36, 20, INK, 2.4); _ln(p, 14, 6, 14, 34, SKB, 2.4); _ln(p, 26, 6, 26, 34, SKB, 2.4)
    p.setPen(_pen("#d33f2f", 2.6, True)); p.drawLine(C.QLineF(15, 20, 25, 20))
def _s_extend(p):
    _ln(p, 4, 20, 18, 20, INK, 2.4); p.setPen(_pen(SKB, 2.2, True)); p.drawLine(C.QLineF(18, 20, 32, 20)); _ln(p, 33, 6, 33, 34, INK, 2.4)
def _s_break(p):
    _ln(p, 4, 20, 18, 20, INK, 2.4); _ln(p, 22, 20, 36, 20, SKB, 2.4); _ln(p, 20, 6, 20, 34, "#8a929b", 1.4); _dots(p, ((20, 20),), 2.8)
def _s_fillet(p):
    q = G.QPainterPath(); q.moveTo(8, 36); q.lineTo(8, 18); q.arcTo(C.QRectF(8, 6, 24, 24), 180, -90); q.lineTo(36, 6)
    p.setPen(_pen(SKB, 2.6)); p.setBrush(C.Qt.NoBrush); p.drawPath(q); p.setPen(_pen("#8a929b", 1, True)); p.drawPolyline(G.QPolygonF([C.QPointF(8, 18), C.QPointF(8, 6), C.QPointF(20, 6)]))
def _s_chamfer(p):
    p.setPen(_pen(SKB, 2.6)); p.setBrush(C.Qt.NoBrush); p.drawPolyline(G.QPolygonF([C.QPointF(8, 36), C.QPointF(8, 18), C.QPointF(20, 6), C.QPointF(36, 6)]))
    p.setPen(_pen("#8a929b", 1, True)); p.drawPolyline(G.QPolygonF([C.QPointF(8, 18), C.QPointF(8, 6), C.QPointF(20, 6)]))
def _s_offset(p):
    _pg(p, [(12, 12), (28, 12), (28, 28), (12, 28)], None, INK, 2); _pg(p, [(5, 5), (35, 5), (35, 35), (5, 35)], None, SKB, 2.2)
def _s_mirror(p):
    p.setPen(_pen("#8a929b", 1.2, True)); p.drawLine(C.QLineF(20, 3, 20, 37)); _pg(p, [(16, 10), (16, 30), (5, 30)], None, INK, 2)
    _pg(p, [(24, 10), (24, 30), (35, 30)], None, SKB, 2)
def _s_cpat(p):
    _circ(p, 20, 20, 12, None, "#8a929b", 1)
    for k in range(6): _circ(p, 20 + 12*math.cos(k*math.tau/6), 20 + 12*math.sin(k*math.tau/6), 3.6, "#d8ecfa" if k else SKB, SKB, 1.6)
def _s_rpat(p):
    for i in range(3):
        for j in range(2): _circ(p, 9 + i*11, 13 + j*14, 4, "#d8ecfa" if i or j else SKB, SKB, 1.6)
def _s_move(p):
    _pg(p, [(6, 14), (20, 14), (20, 28), (6, 28)], None, INK, 1.8, ); _ln(p, 20, 21, 34, 21, ORANGE, 2.2); _arrowhead(p, 36, 21, 0, ORANGE, 5)
    _ln(p, 13, 14, 13, 4, ORANGE, 2.2); _arrowhead(p, 13, 3, -90, ORANGE, 5)
def _s_scale(p):
    _pg(p, [(4, 22), (18, 22), (18, 36), (4, 36)], None, INK, 1.8); _pg(p, [(4, 6), (34, 6), (34, 36), (4, 36)], None, SKB, 2)
    _dots(p, ((4, 36),), 2.6)
def _s_project(p):
    _box(p, 20, 3, 12, 9, ("#e1e4e8", "#b1b7bf", "#8f969f"), "#6f757d"); _pg(p, [(4, 30), (20, 24), (36, 30), (20, 36)], None, "#9b4fc9", 2)
    p.setPen(_pen("#9b4fc9", 1, True)); p.drawLine(C.QLineF(8, 12, 8, 30)); p.drawLine(C.QLineF(32, 12, 32, 30))
def _s_intersect(p):
    _box(p, 20, 5, 12, 14, ("#e1e4e8", "#b1b7bf", "#8f969f"), "#6f757d"); _pg(p, [(2, 22), (24, 12), (38, 20), (16, 30)], None, "#9b4fc9", 2)
def _s_dxf(p):
    _pg(p, [(8, 3), (26, 3), (33, 10), (33, 37), (8, 37)], "white", "#6f757d", 1.2)
    f = p.font(); f.setPixelSize(10); f.setBold(True); p.setFont(f); p.setPen(G.QColor(SKD)); p.drawText(C.QRectF(8, 18, 25, 14), C.Qt.AlignCenter, "DXF")
def _s_svg(p):
    _pg(p, [(8, 3), (26, 3), (33, 10), (33, 37), (8, 37)], "white", "#6f757d", 1.2)
    f = p.font(); f.setPixelSize(10); f.setBold(True); p.setFont(f); p.setPen(G.QColor("#d0752f")); p.drawText(C.QRectF(8, 18, 25, 14), C.Qt.AlignCenter, "SVG")
def _s_canvas(p):
    _pg(p, [(4, 8), (36, 8), (36, 32), (4, 32)], "#eef5fb", "#6f757d", 1.4); _pg(p, [(6, 30), (16, 18), (24, 26), (28, 22), (34, 30)], "#7fbf6a", None)
    _circ(p, 28, 14, 3, "#f2c94c", None)
def _s_decal(p):
    _box(p, 20, 6, 14, 14, ("#e1e4e8", "#b1b7bf", "#8f969f"), "#6f757d"); _pg(p, [(23, 22), (31, 18), (31, 26), (23, 30)], "#f2c94c", "#d67a1e", 1)
def _s_params(p):
    f = p.font(); f.setPixelSize(20); f.setItalic(True); f.setBold(True); p.setFont(f); p.setPen(G.QColor(SKD)); p.drawText(C.QRectF(0, 0, 40, 40), C.Qt.AlignCenter, "ƒx")
def _s_finish(p):
    _circ(p, 20, 20, 15, "#3fae49", "#2b8a35", 1.2); p.setPen(_pen("white", 3.4)); p.drawPolyline(G.QPolygonF([C.QPointF(12, 20), C.QPointF(18, 26), C.QPointF(29, 14)]))
def _s_cons(p): p.setPen(_pen("#e08a1e", 2.6, True)); p.drawLine(C.QLineF(6, 34, 34, 6))
def _s_cl(p):
    q = G.QPen(G.QColor("#e08a1e"), 2.4); q.setDashPattern([6, 2, 1.5, 2]); p.setPen(q); p.drawLine(C.QLineF(6, 34, 34, 6))
def _s_look(p):
    _pg(p, [(8, 12), (32, 12), (32, 32), (8, 32)], "#d8ecfa", SKD, 1.4); _ln(p, 20, 2, 20, 22, ORANGE, 2.2); _arrowhead(p, 20, 23, 90, ORANGE, 5)
def _con_icon(sym, extra=None):
    def f(p):
        p.setPen(_pen("#6f757d", 1.2)); p.setBrush(G.QColor("#f4f6f8")); p.drawRoundedRect(C.QRectF(4, 4, 32, 32), 5, 5)
        if extra: extra(p); return
        ft = p.font(); ft.setPixelSize(18); ft.setBold(True); p.setFont(ft); p.setPen(G.QColor(SKD)); p.drawText(C.QRectF(4, 4, 32, 32), C.Qt.AlignCenter, sym)
    return f
def _c_hv(p): _ln(p, 9, 28, 31, 28, SKD, 2.6); _ln(p, 12, 9, 12, 24, SKD, 2.6)
def _c_coin(p): _ln(p, 8, 30, 30, 10, "#8a929b", 1.6); _dots(p, ((19, 20),), 4, "#d33f2f", "#9b2b20")
def _c_tan(p): _ln(p, 6, 30, 34, 30, SKD, 2.4); _arcp(p, 20, 20, 10, 0, 360, SKB, 2.2)
def _c_eq(p): _ln(p, 10, 16, 30, 16, SKD, 2.8); _ln(p, 10, 24, 30, 24, SKD, 2.8)
def _c_par(p): _ln(p, 10, 30, 24, 8, SKD, 2.6); _ln(p, 17, 32, 31, 10, SKD, 2.6)
def _c_perp(p): _ln(p, 9, 30, 31, 30, SKD, 2.6); _ln(p, 20, 30, 20, 9, SKD, 2.6)
def _c_fix(p):
    p.setPen(_pen(SKD, 2.4)); p.setBrush(C.Qt.NoBrush); p.drawArc(C.QRectF(14, 8, 12, 14), 0, 180*16)
    _pg(p, [(11, 17), (29, 17), (29, 31), (11, 31)], "#3fae49", "#2b8a35", 1.2)
def _c_mid(p): _ln(p, 7, 28, 33, 28, SKD, 2.4); _pg(p, [(20, 18), (25, 27), (15, 27)], "#d33f2f", "#9b2b20", 1)
def _c_conc(p): _circ(p, 20, 20, 12, None, SKD, 2.2); _circ(p, 20, 20, 6, None, SKB, 2.2)
def _c_col(p): _ln(p, 6, 30, 16, 22, SKD, 2.6); _ln(p, 23, 17, 34, 9, SKD, 2.6); p.setPen(_pen("#8a929b", 1, True)); p.drawLine(C.QLineF(16, 22, 23, 17))
def _c_sym(p):
    p.setPen(_pen("#8a929b", 1.2, True)); p.drawLine(C.QLineF(20, 6, 20, 34)); _dots(p, ((11, 20), (29, 20)), 3)
def _c_smooth(p):
    _path(p, [(6, 30), (14, 30), (18, 10), (34, 10)], SKD, 2.4)
    f = p.font(); f.setPixelSize(9); f.setBold(True); p.setFont(f); p.setPen(G.QColor("#d0752f")); p.drawText(C.QRectF(20, 20, 16, 12), C.Qt.AlignCenter, "G2")

def _s_include3d(p):
    _box(p, 20, 4, 13, 12, ("#e1e4e8", "#b1b7bf", "#8f969f"), "#6f757d"); _ln(p, 7, 10.5, 20, 17, "#9b4fc9", 2.6); _ln(p, 20, 17, 20, 29, "#9b4fc9", 2.6)
def _s_projsurf(p):
    _cyl(p, 20, 18, 14, 14, "#e1e4e8", "#b1b7bf"); _path(p, [(8, 26), (14, 34), (26, 34), (32, 26)], "#9b4fc9", 2.4)
    p.setPen(_pen("#9b4fc9", 1, True)); p.drawLine(C.QLineF(12, 4, 12, 24)); p.drawLine(C.QLineF(28, 4, 28, 24))
ICONS.update(include3d=_s_include3d, projsurf=_s_projsurf)
ICONS.update(rect3=_s_rect3, rectc=_s_rectc, circle2p=_s_circle2p, circle3p=_s_circle3p, circle2t=_s_circle2t, circle3t=_s_circle3t,
             arc3p=_s_arc3p, arcc=_s_arcc, arct=_s_arct, polyc=_s_polyc, polyi=_s_polyi, polye=_s_polye, ellipse=_s_ellipse, slot=_s_slot,
             slotc=_s_slotc, slota=_s_slota, spline=_s_spline, cspline=_s_cspline, conic=_s_conic, skpoint=_s_point, text=_s_text,
             skdim=_s_dim, trim=_s_trim, extend=_s_extend, **{"break": _s_break}, skfillet=_s_fillet, skchamfer=_s_chamfer,
             skoffset=_s_offset, skmirror=_s_mirror, skcpat=_s_cpat, skrpat=_s_rpat, skmove=_s_move, skscale=_s_scale,
             project=_s_project, intersect=_s_intersect, dxf=_s_dxf, svg=_s_svg, canvas=_s_canvas, decal=_s_decal, params=_s_params,
             finish=_s_finish, skcons=_s_cons, skcl=_s_cl, lookat=_s_look, sk2=_sketch, skgeo=_sketch, skmod=_sketch,
             c_hv=_con_icon("", _c_hv), c_coin=_con_icon("", _c_coin), c_tan=_con_icon("", _c_tan), c_eq=_con_icon("", _c_eq),
             c_par=_con_icon("", _c_par), c_perp=_con_icon("", _c_perp), c_fix=_con_icon("", _c_fix), c_mid=_con_icon("", _c_mid),
             c_conc=_con_icon("", _c_conc), c_col=_con_icon("", _c_col), c_sym=_con_icon("", _c_sym), c_smooth=_con_icon("", _c_smooth))

def _check(p):
    p.setPen(_pen("white", 5.5)); p.setBrush(C.Qt.NoBrush); p.drawPolyline(G.QPolygonF([C.QPointF(8, 21), C.QPointF(16, 29), C.QPointF(32, 11)]))
ICONS["check"] = _check


# ---- icons for the SURFACE workspace (surfaces drawn as thin orange sheets) ----
SF_F, SF_L, SF_B = "#f8c99a", "#d6741c", "#e9a35f"
def _sheet(p, pts, fill=SF_F, line=SF_L, w=1.5): _pg(p, pts, fill, line, w)
def _f_sextrude(p):
    _sheet(p, [(6, 30), (6, 12), (22, 6), (22, 24)]); _sheet(p, [(22, 24), (22, 6), (34, 12), (34, 30)], SF_B)
    _path(p, [(6, 30), (22, 24), (34, 30)], SKB, 2.4); _arrowhead(p, 20, 4, -math.pi/2, "#4a9fdc", 6)
def _f_srevolve(p):
    p.setPen(_pen(SF_L, 1.5)); p.setBrush(G.QColor(SF_F)); p.drawEllipse(C.QRectF(6, 8, 28, 9))
    _pg(p, [(6, 12.5), (6, 28), (34, 28), (34, 12.5)], SF_B, SF_L, 1.5); p.setBrush(G.QColor(SF_F)); p.drawEllipse(C.QRectF(6, 23.5, 28, 9))
    p.setPen(_pen("#8a929b", 1, True)); p.drawLine(C.QLineF(20, 2, 20, 38))
def _f_ssweep(p):
    pa = G.QPainterPath(); pa.moveTo(6, 30); pa.cubicTo(14, 8, 26, 34, 34, 10); pa.lineTo(34, 18); pa.cubicTo(26, 42, 14, 16, 6, 38); pa.closeSubpath()
    p.setPen(_pen(SF_L, 1.4)); p.setBrush(G.QColor(SF_F)); p.drawPath(pa); _path(p, [(6, 34), (14, 20), (26, 30), (34, 14)], "#9b4fc9", 1.6)
def _f_sloft(p):
    pa = G.QPainterPath(); pa.moveTo(5, 32); pa.cubicTo(12, 20, 12, 14, 18, 6); pa.lineTo(35, 10); pa.cubicTo(30, 18, 30, 24, 28, 34); pa.closeSubpath()
    p.setPen(_pen(SF_L, 1.4)); p.setBrush(G.QColor(SF_F)); p.drawPath(pa)
    _path(p, [(5, 32), (28, 34)], SKB, 2.4); _path(p, [(18, 6), (35, 10)], SKB, 2.4)
def _f_patch(p):
    _path(p, [(6, 28), (12, 10), (30, 8), (35, 26), (18, 34)], SKB, 2.2, True)
    pa = G.QPainterPath(); pa.moveTo(6, 28); pa.lineTo(12, 10); pa.lineTo(30, 8); pa.lineTo(35, 26); pa.lineTo(18, 34); pa.closeSubpath()
    p.setPen(C.Qt.NoPen); p.setBrush(G.QColor(248, 201, 154, 200)); p.drawPath(pa); _path(p, [(6, 28), (12, 10), (30, 8), (35, 26), (18, 34)], SKB, 2.2, True)
def _f_ruled(p):
    _sheet(p, [(4, 30), (22, 22), (22, 10), (4, 18)], "#d9dde2", "#8f969f"); _sheet(p, [(22, 22), (36, 30), (36, 18), (22, 10)])
    _path(p, [(22, 22), (22, 10)], SKB, 2.6)
def _f_soffset(p):
    _sheet(p, [(4, 30), (18, 22), (36, 28), (22, 36)], "#d9dde2", "#8f969f"); _sheet(p, [(4, 18), (18, 10), (36, 16), (22, 24)])
    p.setPen(_pen("#4a9fdc", 1.4, True)); p.drawLine(C.QLineF(20, 23, 20, 29))
def _f_trim(p):
    _sheet(p, [(4, 30), (18, 22), (18, 8), (4, 16)]); _pg(p, [(18, 22), (36, 30), (36, 16), (18, 8)], None, SF_L, 1.2)
    p.setPen(_pen("#b3261e", 1.4, True)); p.drawLine(C.QLineF(18, 4, 18, 36)); _ln(p, 25, 14, 31, 24, "#b3261e", 2); _ln(p, 31, 14, 25, 24, "#b3261e", 2)
def _f_untrim(p):
    _sheet(p, [(4, 28), (20, 34), (36, 26), (20, 20)]); p.setPen(_pen(SF_L, 1.2, True)); p.setBrush(C.Qt.NoBrush)
    p.drawPolygon(G.QPolygonF([C.QPointF(4, 28), C.QPointF(4, 10), C.QPointF(20, 4), C.QPointF(36, 8), C.QPointF(36, 26)]))
    _arrowhead(p, 20, 8, -math.pi/2, "#4a9fdc", 6); _ln(p, 20, 20, 20, 10, "#4a9fdc", 1.6)
def _f_sextend(p):
    _sheet(p, [(4, 30), (4, 14), (20, 8), (20, 24)]); _pg(p, [(20, 24), (20, 8), (34, 4), (34, 20)], None, SF_L, 1.2)
    _ln(p, 22, 16, 31, 13, "#4a9fdc", 1.8); _arrowhead(p, 33, 12.5, -0.3, "#4a9fdc", 6)
def _f_stitch(p):
    _sheet(p, [(4, 30), (4, 12), (19, 8), (19, 26)]); _sheet(p, [(21, 26), (21, 8), (36, 12), (36, 30)], SF_B)
    for y in (11, 16, 21): _ln(p, 16, y, 24, y + 2, "#1d5c93", 1.8)
def _f_unstitch(p):
    _sheet(p, [(3, 30), (3, 12), (16, 8), (16, 26)]); _sheet(p, [(24, 26), (24, 8), (37, 12), (37, 30)], SF_B)
    _ln(p, 18, 16, 22, 16, "#b3261e", 1.6); _arrowhead(p, 22, 16, 0, "#b3261e", 4); _arrowhead(p, 18, 16, math.pi, "#b3261e", 4)
def _f_revnormal(p):
    _sheet(p, [(4, 30), (18, 22), (36, 28), (22, 36)]); _ln(p, 20, 29, 20, 9, "#4a9fdc", 2); _arrowhead(p, 20, 7, -math.pi/2, "#4a9fdc", 6)
    p.setPen(_pen("#6b7280", 1.4)); p.setBrush(C.Qt.NoBrush); p.drawArc(C.QRectF(24, 6, 12, 12), 90*16, -270*16); _arrowhead(p, 30, 6, math.pi, "#6b7280", 5)
def _f_sdelete(p):
    _box(p, 20, 9, 13, 13, ("#e1e4e8", "#b1b7bf", "#8f969f"), "#6f757d"); _pg(p, [(7, 12), (20, 5), (33, 12), (20, 19)], "#ffffff", "#b3261e", 1.4)
    _ln(p, 15, 9, 25, 15, "#b3261e", 2); _ln(p, 25, 9, 15, 15, "#b3261e", 2)
def _f_surfbody(p): _sheet(p, [(4, 30), (14, 12), (36, 10), (28, 30)], SF_F, SF_L, 1.6)
def _f_surface_tab(p): _f_sloft(p)
ICONS.update(sextrude=_f_sextrude, srevolve=_f_srevolve, ssweep=_f_ssweep, sloft=_f_sloft, patch=_f_patch, ruled=_f_ruled,
             soffset=_f_soffset, trim3d=_f_trim, untrim=_f_untrim, sextend=_f_sextend, stitch=_f_stitch, unstitch=_f_unstitch,
             revnormal=_f_revnormal, sdelete=_f_sdelete, surfbody=_f_surfbody)


# ---- icons for Inspect / Select / view tools (0.7) ----
def _n_measure(p):
    p.setPen(_pen("#4a5159", 1.6)); p.setBrush(G.QColor("#f2d36b")); p.drawRoundedRect(C.QRectF(4, 14, 32, 12), 2, 2)
    for i, x in enumerate(range(8, 34, 4)): _ln(p, x, 14, x, 18 if i % 2 else 21, "#4a5159", 1.2)
def _n_section(p):
    _box(p, 20, 7, 14, 14); p.setPen(C.Qt.NoPen); p.setBrush(G.QColor("#f0f0f0")); p.drawRect(C.QRectF(20, 4, 18, 34))
    _pg(p, [(20, 14), (30, 9), (30, 24), (20, 30)], "#e38a5b", "#a2471b", 1.2)
    for y in range(13, 30, 4): _ln(p, 21, y + 3, 29, y - 1, "#a2471b", 1)
def _n_interf(p):
    _box(p, 15, 10, 10, 10, ("#a9d6f5", "#4a9fdc", "#2b79bd")); _box(p, 25, 16, 10, 10, ("#e1e4e8", "#b1b7bf", "#8f969f"), "#6f757d")
    _pg(p, [(19, 22), (25, 19), (25, 28), (19, 31)], "#e04a3a", "#a32418", 1.2)
def _n_zebra(p):
    p.setBrush(G.QColor("#ffffff")); p.setPen(_pen("#4a5159", 1.4)); p.drawEllipse(C.QRectF(5, 5, 30, 30))
    p.save(); path = G.QPainterPath(); path.addEllipse(C.QRectF(5, 5, 30, 30)); p.setClipPath(path)
    for y in range(5, 36, 6): p.fillRect(C.QRectF(5, y, 30, 3), G.QColor("#222"))
    p.restore()
def _n_draft(p):
    _pg(p, [(8, 34), (14, 6), (26, 6), (32, 34)], None, None); p.setPen(C.Qt.NoPen)
    p.setBrush(G.QColor("#4caf50")); p.drawPolygon(G.QPolygonF([C.QPointF(8, 34), C.QPointF(14, 6), C.QPointF(20, 6), C.QPointF(20, 34)]))
    p.setBrush(G.QColor("#e04a3a")); p.drawPolygon(G.QPolygonF([C.QPointF(20, 34), C.QPointF(20, 6), C.QPointF(26, 6), C.QPointF(32, 34)]))
    _ln(p, 20, 2, 20, 38, "#333", 1.2)
def _n_curvmap(p):
    g = G.QLinearGradient(4, 0, 36, 0)
    for t, c in ((0, "#2a4ff2"), (0.35, "#1ccbe6"), (0.6, "#3cc84a"), (0.8, "#f6d127"), (1, "#eb3326")): g.setColorAt(t, G.QColor(c))
    path = G.QPainterPath(); path.moveTo(4, 32); path.cubicTo(14, 32, 20, 6, 36, 8); path.lineTo(36, 34); path.lineTo(4, 34); path.closeSubpath()
    p.setPen(_pen("#4a5159", 1.2)); p.setBrush(g); p.drawPath(path)
def _n_comb(p):
    path = G.QPainterPath(); path.moveTo(4, 30); path.cubicTo(14, 30, 24, 10, 36, 10)
    p.setPen(_pen("#2f86d0", 2.2)); p.setBrush(C.Qt.NoBrush); p.drawPath(path)
    for i in range(1, 9):
        t = i/9; pt = path.pointAtPercent(t); a = path.angleAtPercent(t); k = math.sin(t*math.pi)*9
        nx, ny = math.sin(math.radians(a)), math.cos(math.radians(a)); _ln(p, pt.x(), pt.y(), pt.x() - nx*k, pt.y() - ny*k, "#c2347a", 1.3)
def _n_com(p):
    p.setPen(_pen("#333", 1.6)); p.setBrush(G.QColor("white")); p.drawEllipse(C.QRectF(9, 9, 22, 22))
    p.setPen(C.Qt.NoPen); p.setBrush(G.QColor("#333")); p.drawPie(C.QRectF(9, 9, 22, 22), 0, 90*16); p.drawPie(C.QRectF(9, 9, 22, 22), 180*16, 90*16)
def _n_props(p):
    p.setPen(_pen("#4a5159", 1.4)); p.setBrush(G.QColor("white")); p.drawRoundedRect(C.QRectF(8, 4, 24, 32), 2, 2)
    for y in (11, 17, 23, 29): _ln(p, 12, y, 28, y, "#7c8590", 1.6)
def _n_selfilter(p):
    _pg(p, [(6, 8), (34, 8), (23, 21), (23, 32), (17, 35), (17, 21)], "#d6e9f8", "#2b79bd", 1.4)
def _n_selwin(p):
    p.setPen(_pen("#2b79bd", 1.4, True)); p.setBrush(G.QColor(6, 150, 215, 40)); p.drawRect(C.QRectF(5, 7, 30, 24))
    _pg(p, [(24, 22), (24, 37), (28, 33), (31, 39), (33, 38), (30, 32), (35, 32)], "white", "#333", 1.1)
def _n_search(p):
    p.setPen(_pen("#4a5159", 2.4)); p.setBrush(C.Qt.NoBrush); p.drawEllipse(C.QRectF(7, 7, 18, 18)); _ln(p, 23, 23, 33, 33, "#4a5159", 3)
def _n_display(p):
    _box(p, 20, 8, 12, 12, ("#e1e4e8", "#b1b7bf", "#8f969f"), "#6f757d")
    p.setPen(_pen("#2b79bd", 1.2, True)); p.setBrush(C.Qt.NoBrush); p.drawRect(C.QRectF(4, 4, 32, 32))
def _n_view(p): _pg(p, [(6, 14), (26, 14), (34, 8), (34, 32), (26, 26), (6, 26)], "#e1e4e8", "#6f757d", 1.3)
ICONS.update(measure=_n_measure, section=_n_section, interference=_n_interf, zebra=_n_zebra, draftan=_n_draft, curvmap=_n_curvmap,
             comb=_n_comb, com=_n_com, props=_n_props, selfilter=_n_selfilter, selwin=_n_selwin, search=_n_search, display=_n_display,
             view=_n_view)


# ---- icons for the MESH workspace and file exchange (0.7) ----
MS_F, MS_L, MS_B = "#c9b6ea", "#6b4fa8", "#a68bd8"
def _mesh_patch(p, pts, fill=MS_F, line=MS_L):
    """A triangulated quad (4 corners) drawn as two triangles with a centre fan."""
    _pg(p, pts, fill, line, 1.3); a, b, c, d = pts
    cx, cy = sum(q[0] for q in pts)/4, sum(q[1] for q in pts)/4
    for q in pts: _ln(p, cx, cy, q[0], q[1], line, 0.9)
def _mesh_box(p):
    _mesh_patch(p, [(20, 6), (34, 13), (20, 20), (6, 13)], "#e2d8f5"); _mesh_patch(p, [(6, 13), (20, 20), (20, 34), (6, 27)])
    _mesh_patch(p, [(20, 20), (34, 13), (34, 27), (20, 34)], MS_B)
def _m_body(p): _mesh_box(p)
def _m_import(p):
    _mesh_box(p); _pg(p, [(26, 30), (38, 30), (38, 38), (26, 38)], "white", "#4a5159", 1.2); _ln(p, 32, 22, 32, 33, "#2b79bd", 2.2)
    _arrowhead(p, 32, 35, math.pi/2, "#2b79bd", 5)
def _m_cadimport(p):
    _box(p, 18, 6, 12, 14); p.setPen(_pen("#4a5159", 1.2)); p.setBrush(G.QColor("white")); p.drawRect(C.QRectF(24, 24, 14, 14))
    _ln(p, 31, 16, 31, 30, "#2b79bd", 2.2); _arrowhead(p, 31, 32, math.pi/2, "#2b79bd", 5)
def _m_export(p):
    p.setPen(_pen("#4a5159", 1.4)); p.setBrush(G.QColor("white")); p.drawRoundedRect(C.QRectF(8, 4, 22, 30), 2, 2)
    _ln(p, 20, 24, 34, 24, "#3c9a4a", 2.4); _arrowhead(p, 36, 24, 0, "#3c9a4a", 6)
def _m_tess(p):
    _box(p, 14, 8, 9, 12); _ln(p, 24, 20, 28, 20, "#6b7280", 2); _arrowhead(p, 30, 20, 0, "#6b7280", 5)
    _mesh_patch(p, [(33, 10), (39, 14), (39, 30), (33, 26)])
def _m_convert(p):
    _mesh_patch(p, [(2, 14), (12, 10), (12, 28), (2, 32)]); _ln(p, 14, 20, 20, 20, "#6b7280", 2); _arrowhead(p, 22, 20, 0, "#6b7280", 5)
    _box(p, 31, 8, 8, 14)
def _m_rev(p):
    _mesh_patch(p, [(6, 26), (20, 18), (34, 26), (20, 34)])
    _ln(p, 14, 24, 14, 8, "#2b79bd", 2); _arrowhead(p, 14, 6, -math.pi/2, "#2b79bd", 5)
    _ln(p, 26, 12, 26, 26, "#e04a3a", 2); _arrowhead(p, 26, 28, math.pi/2, "#e04a3a", 5)
def _m_repair(p):
    _mesh_patch(p, [(4, 22), (20, 12), (36, 22), (20, 32)]); _pg(p, [(16, 20), (24, 18), (26, 24), (18, 26)], "white", "#e04a3a", 1.2)
    p.setPen(_pen("#4a5159", 2.6)); p.drawLine(C.QLineF(24, 6, 34, 16)); p.setBrush(G.QColor("#9aa3ad")); p.drawEllipse(C.QRectF(30, 2, 8, 8))
def _m_cut(p):
    _mesh_box(p); p.setPen(_pen("#e04a3a", 1.6, True)); p.drawLine(C.QLineF(2, 22, 38, 16))
def _m_combine(p):
    _mesh_patch(p, [(4, 12), (20, 6), (20, 26), (4, 32)]); _mesh_patch(p, [(16, 16), (36, 10), (36, 30), (16, 36)], MS_B)
def _m_sep(p):
    _mesh_patch(p, [(2, 14), (14, 10), (14, 26), (2, 30)]); _mesh_patch(p, [(24, 10), (38, 14), (38, 30), (24, 26)], MS_B)
    p.setPen(_pen("#6b7280", 1.2, True)); p.drawLine(C.QLineF(19, 4, 19, 36))
def _m_smooth(p):
    pa = G.QPainterPath(); pa.moveTo(4, 30); pa.lineTo(10, 18); pa.lineTo(16, 26); pa.lineTo(22, 12); pa.lineTo(28, 24); pa.lineTo(36, 14)
    p.setPen(_pen("#9aa3ad", 1.4, True)); p.setBrush(C.Qt.NoBrush); p.drawPath(pa)
    pb = G.QPainterPath(); pb.moveTo(4, 32); pb.cubicTo(14, 18, 26, 18, 36, 20); p.setPen(_pen(MS_L, 2.6)); p.drawPath(pb)
def _m_reduce(p):
    _mesh_patch(p, [(2, 8), (18, 8), (18, 32), (2, 32)]); _pg(p, [(24, 8), (38, 8), (38, 32), (24, 32)], MS_F, MS_L, 1.3); _ln(p, 24, 8, 38, 32, MS_L, 1)
def _m_section(p):
    _mesh_box(p); _pg(p, [(2, 26), (24, 18), (38, 22), (16, 30)], None, None)
    p.setPen(_pen("#2b79bd", 2.2)); p.setBrush(C.Qt.NoBrush); p.drawPolyline(G.QPolygonF([C.QPointF(6, 20), C.QPointF(20, 27), C.QPointF(34, 20)]))
ICONS.update(meshbody=_m_body, meshimport=_m_import, cadimport=_m_cadimport, export=_m_export, tessellate=_m_tess, meshconvert=_m_convert,
             mrev=_m_rev, mrepair=_m_repair, mcut=_m_cut, mcombine=_m_combine, msep=_m_sep, msmooth=_m_smooth, mreduce=_m_reduce,
             msection=_m_section)

def _tl_folder(p):
    _pg(p, [(4, 10), (16, 10), (19, 14), (36, 14), (36, 32), (4, 32)], "#f6c75c", "#b8860b", 1.4)
ICONS["tlfolder"] = _tl_folder

# ---- icons for assemblies (0.7) ----
def _a_comp(p):
    _box(p, 14, 6, 9, 10, ("#f5e3a8", "#e0b84a", "#c0962a"), "#8a6a12"); _box(p, 27, 16, 9, 10)
def _a_joint(p):
    _box(p, 13, 14, 9, 10, ("#e1e4e8", "#b1b7bf", "#8f969f"), "#6f757d"); _box(p, 28, 4, 8, 9)
    p.setPen(_pen("#0696d7", 2)); p.setBrush(C.Qt.NoBrush); p.drawEllipse(C.QRectF(16, 16, 10, 6)); _ln(p, 21, 6, 21, 34, "#0696d7", 1.6)
def _a_drive(p):
    p.setPen(_pen("#4a5159", 2)); p.setBrush(G.QColor("#e1e4e8")); p.drawEllipse(C.QRectF(8, 8, 24, 24))
    path = G.QPainterPath(); path.arcMoveTo(C.QRectF(4, 4, 32, 32), 30); path.arcTo(C.QRectF(4, 4, 32, 32), 30, 220)
    p.setPen(_pen("#0696d7", 2.4)); p.setBrush(C.Qt.NoBrush); p.drawPath(path); _arrowhead(p, 9, 29, 2.2, "#0696d7", 6)
def _a_link(p):
    for cx, r in ((13, 9), (29, 6)):
        p.setPen(_pen("#4a5159", 1.6)); p.setBrush(G.QColor("#e1e4e8")); p.drawEllipse(C.QPointF(cx, 20), r, r)
        for k in range(8):
            a = k*math.pi/4; _ln(p, cx + r*math.cos(a), 20 + r*math.sin(a), cx + (r + 3)*math.cos(a), 20 + (r + 3)*math.sin(a), "#4a5159", 2)
def _a_motion(p):
    _pg(p, [(6, 30), (16, 30), (16, 20), (6, 20)], "#d6e9f8", "#2b79bd", 1.2); _pg(p, [(22, 22), (32, 22), (32, 12), (22, 12)], "#a9d6f5", "#2b79bd", 1.4)
    p.setPen(_pen("#6b7280", 1.2, True)); p.drawLine(C.QLineF(16, 22, 22, 18)); _pg(p, [(14, 34), (14, 40), (19, 37)], "#3c9a4a", None)
def _a_bom(p):
    p.setPen(_pen("#4a5159", 1.4)); p.setBrush(G.QColor("white")); p.drawRect(C.QRectF(6, 5, 28, 30))
    for y in (13, 20, 27): _ln(p, 6, y, 34, y, "#9aa3ad", 1)
    _ln(p, 14, 5, 14, 35, "#9aa3ad", 1)
ICONS.update(component=_a_comp, joint=_a_joint, jdrive=_a_drive, mlink=_a_link, motion=_a_motion, bom=_a_bom,
             comp=_a_comp, compprops=_a_comp)


# ---- icons for sheet metal (0.7) ----
SMF, SML = "#cfd8e3", "#56657a"
def _s_base(p): _pg(p, [(4, 24), (22, 16), (36, 22), (18, 30)], SMF, SML, 1.4); _pg(p, [(4, 24), (18, 30), (18, 33), (4, 27)], "#9aa8ba", SML, 1)
def _s_flange(p):
    _pg(p, [(4, 26), (20, 18), (30, 23), (14, 31)], SMF, SML, 1.4); _pg(p, [(20, 18), (30, 23), (30, 7), (20, 2)], "#a9d6f5", "#2b79bd", 1.4)
def _s_hem(p):
    _pg(p, [(4, 22), (28, 22), (28, 26), (4, 26)], SMF, SML, 1.3)
    path = G.QPainterPath(); path.moveTo(28, 22); path.cubicTo(38, 22, 38, 14, 28, 14); path.lineTo(12, 14)
    p.setPen(_pen("#2b79bd", 3)); p.setBrush(C.Qt.NoBrush); p.drawPath(path)
def _s_fold(p):
    _pg(p, [(4, 30), (20, 30), (20, 26), (4, 26)], SMF, SML, 1.3); _pg(p, [(20, 26), (32, 8), (35, 10), (22, 30)], "#a9d6f5", "#2b79bd", 1.3)
    p.setPen(_pen("#e04a3a", 1.2, True)); p.drawLine(C.QLineF(20, 34, 20, 20))
def _s_unfold(p):
    _pg(p, [(4, 30), (20, 30), (20, 26), (4, 26)], SMF, SML, 1.3); _pg(p, [(20, 30), (36, 30), (36, 26), (20, 26)], "#a9d6f5", "#2b79bd", 1.3)
    p.setPen(_pen("#6b7280", 1.4, True)); p.setBrush(C.Qt.NoBrush); p.drawArc(C.QRectF(14, 8, 24, 36), 30*16, 60*16)
def _s_refold(p):
    _pg(p, [(4, 30), (20, 30), (20, 26), (4, 26)], SMF, SML, 1.3); _pg(p, [(20, 26), (24, 26), (24, 8), (20, 8)], "#a9d6f5", "#2b79bd", 1.3)
def _s_flat(p):
    _pg(p, [(4, 10), (14, 10), (14, 4), (26, 4), (26, 10), (36, 10), (36, 30), (4, 30)], SMF, SML, 1.3)
    p.setPen(_pen("#e04a3a", 1.2, True)); p.drawLine(C.QLineF(4, 16, 36, 16)); p.drawLine(C.QLineF(14, 10, 14, 30)); p.drawLine(C.QLineF(26, 10, 26, 30))
def _s_rule(p):
    p.setPen(_pen("#4a5159", 1.4)); p.setBrush(G.QColor("white")); p.drawRoundedRect(C.QRectF(8, 4, 24, 32), 2, 2)
    _pg(p, [(12, 22), (28, 22), (28, 25), (12, 25)], SMF, SML, 1); _ln(p, 12, 12, 28, 12, "#7c8590", 1.6); _ln(p, 12, 30, 24, 30, "#7c8590", 1.6)
def _s_body(p):
    _pg(p, [(4, 24), (20, 16), (30, 21), (14, 29)], SMF, SML, 1.3); _pg(p, [(20, 16), (30, 21), (30, 9), (20, 4)], "#e1e7ef", SML, 1.3)
ICONS.update(sbase=_s_base, sflange=_s_flange, shem=_s_hem, sfold=_s_fold, unfold=_s_unfold, refold=_s_refold, flatpattern=_s_flat,
             smrule=_s_rule, sheetbody=_s_body)


# ---- plastic icons (0.7) ----
def _p_boss(p):
    _pg(p, [(4, 30), (20, 22), (36, 30), (20, 38)], "#e1e4e8", "#6f757d", 1.2)
    p.setPen(_pen("#2b79bd", 1.4)); p.setBrush(G.QColor("#a9d6f5")); p.drawRect(C.QRectF(14, 10, 12, 20)); p.drawEllipse(C.QRectF(14, 6, 12, 7))
    p.setBrush(G.QColor("#333")); p.drawEllipse(C.QRectF(18, 8, 4, 3))
def _p_lip(p):
    _pg(p, [(4, 34), (4, 16), (14, 16), (14, 34)], "#e1e4e8", "#6f757d", 1.2); _pg(p, [(10, 16), (10, 10), (14, 10), (14, 16)], "#a9d6f5", "#2b79bd", 1.2)
    _pg(p, [(24, 4), (24, 20), (36, 20), (36, 4)], "#e1e4e8", "#6f757d", 1.2); _pg(p, [(29, 20), (29, 25), (34, 25), (34, 20)], "white", "#e04a3a", 1)
def _p_snap(p):
    _pg(p, [(4, 34), (36, 34), (36, 38), (4, 38)], "#e1e4e8", "#6f757d", 1.2); _pg(p, [(16, 34), (16, 6), (20, 6), (20, 34)], "#a9d6f5", "#2b79bd", 1.3)
    _pg(p, [(20, 6), (20, 14), (27, 12)], "#a9d6f5", "#2b79bd", 1.3)
ICONS.update(boss=_p_boss, lipgroove=_p_lip, snapfit=_p_snap)


# ---- render / animation / manufacture icons (0.7) ----
def _r_render(p):
    p.setPen(_pen("#4a5159", 1.4)); g = G.QRadialGradient(16, 14, 16); g.setColorAt(0, G.QColor("#ffffff")); g.setColorAt(1, G.QColor("#7d8fa6"))
    p.setBrush(g); p.drawEllipse(C.QRectF(6, 6, 26, 26)); p.setPen(C.Qt.NoPen); p.setBrush(G.QColor(0, 0, 0, 50)); p.drawEllipse(C.QRectF(8, 32, 24, 5))
def _r_turn(p):
    _r_render(p); p.setPen(_pen("#0696d7", 2)); p.setBrush(C.Qt.NoBrush); p.drawArc(C.QRectF(2, 22, 36, 14), 200*16, 140*16); _arrowhead(p, 34, 31, 0.3, "#0696d7", 5)
def _r_explode(p):
    _box(p, 12, 4, 7, 8); _box(p, 28, 4, 7, 8, ("#e1e4e8", "#b1b7bf", "#8f969f"), "#6f757d"); _box(p, 20, 22, 7, 8)
    for x1, y1, x2, y2 in ((16, 18, 13, 15), (24, 18, 27, 15)): _ln(p, x1, y1, x2, y2, "#e04a3a", 1.4)
def _c_setup(p):
    _pg(p, [(4, 26), (20, 18), (36, 26), (20, 34)], "#f3e2c0", "#b07a26", 1.3); _pg(p, [(4, 26), (20, 34), (20, 38), (4, 30)], "#d9b56d", "#b07a26", 1)
    _ln(p, 20, 18, 30, 18, "#e04a3a", 2); _ln(p, 20, 18, 20, 6, "#2b79bd", 2); _ln(p, 20, 18, 14, 22, "#3c9a4a", 2)
def _c_tool(p):
    p.setPen(_pen("#4a5159", 1.4)); p.setBrush(G.QColor("#c5ccd4")); p.drawRect(C.QRectF(15, 4, 10, 16))
    p.setBrush(G.QColor("#8f9aa6")); p.drawRect(C.QRectF(16, 20, 8, 14))
    for y in (22, 26, 30): _ln(p, 16, y, 24, y + 3, "#4a5159", 1)
def _c_face(p):
    _pg(p, [(4, 26), (20, 18), (36, 26), (20, 34)], "#f3e2c0", "#b07a26", 1.3)
    for k in range(4): _ln(p, 8 + k*6, 24 - k*3 + 2, 20 + k*6, 30 - k*3 + 2, "#0f5bd8", 1.4)
def _c_contour(p):
    _pg(p, [(8, 14), (30, 14), (30, 30), (8, 30)], "#e1e4e8", "#6f757d", 1.2)
    p.setPen(_pen("#0f5bd8", 2)); p.setBrush(C.Qt.NoBrush); p.drawRoundedRect(C.QRectF(4, 10, 30, 24), 4, 4)
def _c_pocket(p):
    _pg(p, [(4, 6), (36, 6), (36, 34), (4, 34)], "#e1e4e8", "#6f757d", 1.2)
    p.setPen(_pen("#0f5bd8", 1.4)); p.setBrush(C.Qt.NoBrush)
    for k in range(3): p.drawRoundedRect(C.QRectF(9 + k*4, 11 + k*4, 22 - k*8, 18 - k*8), 2, 2)
def _c_drill(p):
    _pg(p, [(4, 26), (36, 26), (36, 34), (4, 34)], "#e1e4e8", "#6f757d", 1.2)
    p.setPen(_pen("#4a5159", 1.3)); p.setBrush(G.QColor("#c5ccd4")); p.drawRect(C.QRectF(17, 4, 6, 18)); _pg(p, [(17, 22), (23, 22), (20, 27)], "#c5ccd4", "#4a5159", 1.2)
    p.setPen(_pen("#0f5bd8", 1.2, True)); p.drawLine(C.QLineF(20, 27, 20, 36))
ICONS.update(render=_r_render, turntable=_r_turn, explode=_r_explode, camsetup=_c_setup, camtools=_c_tool, camface=_c_face, camcontour=_c_contour,
             campocket=_c_pocket, camdrill=_c_drill, cam_setup=_c_setup, cam_face=_c_face, cam_contour=_c_contour, cam_pocket=_c_pocket, cam_drill=_c_drill)

def pix(kind, size=40, dpr=2):
    pm = G.QPixmap(size*dpr, size*dpr); pm.setDevicePixelRatio(dpr); pm.fill(C.Qt.transparent)
    p = G.QPainter(pm); p.setRenderHint(G.QPainter.Antialiasing); p.scale(size/40, size/40); ICONS[kind](p); p.end()
    return pm
def icon(kind, size=40): return G.QIcon(pix(kind, size))

def ui_assets():
    """Qt stylesheets can only reference image files, so write the few glyphs they need to a temp dir."""
    d = os.path.join(tempfile.gettempdir(), "fission_ui"); os.makedirs(d, exist_ok=True)
    for name, kind in (("eye_on", "eye"), ("eye_off", "eyeoff"), ("br_closed", "arrow_r"), ("br_open", "arrow_d"), ("up", "arrow_u"), ("check", "check")):
        pix(kind, 18, 1).save(os.path.join(d, name + ".png")); pix(kind, 18, 2).save(os.path.join(d, name + "@2x.png"))
    return d.replace("\\", "/")

def style(d):
    return f"""
QMainWindow,#root{{background:#f0f0f0}}
QLabel{{color:#3c3c3c}}
QToolTip{{background:#fff;color:#333;border:1px solid #a8a8a8;padding:3px 6px}}
QMenu{{background:#fafafa;border:1px solid #bdbdbd;padding:4px 0}}
QMenu::item{{padding:5px 30px 5px 22px;color:#333}} QMenu::item:selected{{background:#d6e9f8}} QMenu::item:disabled{{color:#aaa}}
QMenu::separator{{height:1px;background:#ddd;margin:4px 8px}}
QStatusBar{{background:#f0f0f0;color:#555;border-top:1px solid #d0d0d0;font-size:12px}} QStatusBar::item{{border:none}}
QSplitter::handle{{background:#d0d0d0}}
QDialog{{background:#f4f4f4}}
QPushButton{{background:#fafafa;color:#333;border:1px solid #b4b4b4;border-radius:3px;padding:5px 16px;min-width:58px}}
QPushButton:hover{{background:#e8f3fb;border-color:{ACCENT}}} QPushButton:pressed{{background:#d3e8f6}}
QPushButton#primary,QPushButton:default{{background:{ACCENT};color:white;border-color:#0580b8}}
QPushButton#primary:hover,QPushButton:default:hover{{background:#0a86c0}}
QDoubleSpinBox,QSpinBox,QComboBox,QLineEdit{{background:white;color:#333;border:1px solid #b4b4b4;border-radius:2px;padding:3px 6px;min-height:18px}}
QDoubleSpinBox:focus,QSpinBox:focus,QComboBox:focus,QLineEdit:focus{{border-color:{ACCENT}}}
QDoubleSpinBox::up-button,QDoubleSpinBox::down-button,QSpinBox::up-button,QSpinBox::down-button{{width:16px;border:none;background:transparent}}
QDoubleSpinBox::up-arrow,QSpinBox::up-arrow{{image:url({d}/up.png);width:9px;height:9px}}
QDoubleSpinBox::down-arrow,QSpinBox::down-arrow{{image:url({d}/br_open.png);width:9px;height:9px}}
QComboBox::drop-down{{border:none;width:20px}} QComboBox::down-arrow{{image:url({d}/br_open.png);width:9px;height:9px}}
QComboBox QAbstractItemView{{background:white;color:#333;border:1px solid #bdbdbd;selection-background-color:#d6e9f8;selection-color:#222}}
QDialog,QMessageBox,QInputDialog,QFileDialog{{background:#f3f3f3;color:#333}}
QCheckBox,QRadioButton{{color:#333;background:transparent}}
QCheckBox::indicator{{width:14px;height:14px;background:white;border:1px solid #9a9a9a;border-radius:2px}}
QCheckBox::indicator:checked{{background:{ACCENT};border-color:{ACCENT};image:url({d}/check.png)}}
QScrollArea,QScrollArea > QWidget > QWidget{{background:transparent}}
#cmd QScrollArea, #cmd QScrollArea > QWidget, #cmd QScrollArea > QWidget > QWidget{{background:white}}
QTableWidget,QTableView,QTreeView,QListView{{background:white;color:#222;alternate-background-color:#f5f7f9;selection-background-color:#d6e9f8;selection-color:#1d2a35;gridline-color:#e0e0e0}}
QHeaderView::section{{background:#eeeeee;color:#333;border:none;border-right:1px solid #d8d8d8;border-bottom:1px solid #d8d8d8;padding:3px 6px}}
QTabWidget::pane{{background:white}} QTabBar::tab{{background:#eeeeee;color:#333;padding:4px 10px}} QTabBar::tab:selected{{background:white}}
QCheckBox{{color:#333}}

#topbar{{background:#e3e3e3;border-bottom:1px solid #cfcfcf}}
#topbtn{{border:none;border-radius:3px;padding:3px 4px;background:transparent}} #topbtn:hover{{background:#d1dce6}}
#doctab{{background:#f4f4f4;border:1px solid #cfcfcf;border-bottom:none;border-top-left-radius:5px;border-top-right-radius:5px}}
#doctab QLabel{{font-size:13px;padding:6px 2px}}
#ribbon{{background:#f4f4f4;border-bottom:1px solid #cfcfcf}}
#mode{{background:#fafafa;border:1px solid #c4c4c4;border-radius:4px;font-weight:700;font-size:13px;color:#333;padding:0 14px;min-height:62px;min-width:84px}}
#mode:hover{{border-color:{ACCENT}}} #mode::menu-indicator{{image:none;width:0}}
#tab{{background:transparent;border:none;border-bottom:3px solid transparent;color:#444;font-size:12px;padding:7px 15px 4px 15px}}
#tab:hover{{color:{ACCENT}}} #tab:checked{{color:{ACCENT};border-bottom:3px solid {ACCENT}}}
#rb{{background:transparent;border:1px solid transparent;border-radius:4px}}
#rb:hover{{background:#dfeaf3;border-color:#bcd3e6}} #rb:pressed{{background:#c4dff3}} #rb:checked{{background:#cbe5f7;border-color:{ACCENT}}}
#grp{{background:transparent;border:none;color:#505050;font-size:11px;font-weight:600;padding:1px 6px 3px 6px}}
#grp:hover{{color:{ACCENT}}} #grp::menu-indicator{{image:none;width:0}}
#sep{{background:#d6d6d6;border:none}}

#browser{{background:#f6f6f6;border-right:1px solid #cfcfcf}}
#panelhdr{{background:#ececec;border-bottom:1px solid #d2d2d2}} #panelhdr QLabel{{font-weight:600;font-size:12px;color:#444}}
#bcol{{background:transparent;border:none;font-size:14px;font-weight:700;color:#666;padding:0 4px}} #bcol:hover{{color:{ACCENT}}}
QTreeWidget{{background:#f6f6f6;color:#333;border:none;outline:0;font-size:13px}}
QTreeWidget::item{{padding:3px 0}} QTreeWidget::item:hover{{background:#e7f0f7}}
QTreeWidget::item:selected{{background:#d3e7f6;color:#222}}
QTreeWidget::indicator{{width:18px;height:18px}}
QTreeWidget::indicator:checked{{image:url({d}/eye_on.png)}} QTreeWidget::indicator:unchecked{{image:url({d}/eye_off.png)}}
QTreeView::branch{{background:transparent}}
QTreeView::branch:has-children:closed{{image:url({d}/br_closed.png)}} QTreeView::branch:has-children:open{{image:url({d}/br_open.png)}}

#navbar{{background:rgba(250,250,250,235);border:1px solid #c4c4c4;border-radius:5px}}
#nb{{background:transparent;border:1px solid transparent;border-radius:3px}}
#nb:hover{{background:#dfeaf3}} #nb:checked{{background:#c9e3f6;border-color:{ACCENT}}}
#timeline{{background:#f0f0f0;border-top:1px solid #cfcfcf}}
#tlbtn{{background:transparent;border:none;border-radius:3px}} #tlbtn:hover{{background:#dfeaf3}}
#tlstep{{background:transparent;border:1px solid transparent;border-radius:3px}} #tlstep:hover{{background:#dfeaf3;border-color:#bcd3e6}}
#tlmarker{{background:{ACCENT};border:none;border-radius:1px}}

#cmd{{background:#f6f6f6;border:1px solid #a9a9a9;border-radius:4px}}
#cmdhdr{{background:#e6e6e6;border:none;border-bottom:1px solid #cbcbcb;border-top-left-radius:4px;border-top-right-radius:4px;
        padding:8px 12px;font-weight:700;font-size:12px;color:#333}}
#hint{{color:#777;font-size:11px}}
#selbtn{{background:white;color:#333;border:1px solid #b4b4b4;border-radius:3px;padding:4px 8px;text-align:left;min-width:0}}
#selbtn:checked{{border:2px solid {ACCENT};background:#e8f3fb;color:#0b4f7a}}
#selx{{border:none;background:transparent;color:#888;padding:2px 4px}} #selx:hover{{color:#b3261e}}
QListWidget{{background:white;border:1px solid #b4b4b4}}
"""

# ---------------------------------------------------------------------------------------------------------------
#  2D sketch helpers
# ---------------------------------------------------------------------------------------------------------------
def loop(kind, pts):
    a = pts[0]
    if kind in ("poly", "line"): return pts
    b = pts[1]
    if kind == "rect": return [a, V(b.x, a.y, 0), b, V(a.x, b.y, 0)]
    r = (b - a).Length
    return [a + V(r*math.cos(i*math.tau/64), r*math.sin(i*math.tau/64), 0) for i in range(64)]

class Plane:
    """Sketch plane: origin o, in-plane axes u,v, normal n = u x v. Local (x,y,0) <-> world."""
    def __init__(s, o, u, v): s.o, s.u, s.v, s.n = o, u, v, u.cross(v)
    def w(s, p): return s.o + s.u*p.x + s.v*p.y + s.n*p.z
    def l(s, p): d = p - s.o; return V(d.dot(s.u), d.dot(s.v), d.dot(s.n))
    def gl(s):
        return [s.u.x, s.u.y, s.u.z, 0, s.v.x, s.v.y, s.v.z, 0, s.n.x, s.n.y, s.n.z, 0, s.o.x, s.o.y, s.o.z, 1]
    def is_ground(s): return s.n.z > 0.9999 and abs(s.o.z) < 1e-9

XY = Plane(V(0, 0, 0), V(1, 0, 0), V(0, 1, 0))
XZ = Plane(V(0, 0, 0), V(1, 0, 0), V(0, 0, 1))      # front plane (normal -Y, toward the FRONT view)
YZ = Plane(V(0, 0, 0), V(0, 1, 0), V(0, 0, 1))      # right plane (normal +X, toward the RIGHT view)
ORIGIN_PLANES = (("XY", XY, (0.30, 0.55, 0.95)), ("XZ", XZ, (0.35, 0.75, 0.40)), ("YZ", YZ, (0.95, 0.45, 0.35)))

def plane_for(n, a):
    """Plane lying on a face with outward normal n through point a (grid origin = world origin projected)."""
    o = n * n.dot(a)
    u = V(1, 0, 0) if abs(n.z) > 0.99 else V(0, 0, 1).cross(n).normalize()
    return Plane(o, u, n.cross(u))

def triangulate(P):
    """Ear-clipping for simple (possibly concave) polygons in XY (used only for live previews)."""
    n = len(P); idx = list(range(n))
    if n < 3: return []
    if sum(P[i].x*P[(i+1) % n].y - P[(i+1) % n].x*P[i].y for i in range(n)) < 0: idx.reverse()
    cr = lambda a, b, c: (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x)
    out = []
    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k-1], idx[k], idx[(k+1) % len(idx)]
            a, b, c = P[i0], P[i1], P[i2]
            if cr(a, b, c) <= 0: continue
            if any(cr(a, b, P[j]) >= 0 and cr(b, c, P[j]) >= 0 and cr(c, a, P[j]) >= 0
                   for j in idx if j not in (i0, i1, i2)): continue
            out.append((a, b, c)); idx.pop(k); break
        else: break
    if len(idx) == 3: out.append(tuple(P[i] for i in idx))
    return out

def tri_fill(kind, L):
    return triangulate(L) if kind == "poly" else [(L[0], L[i], L[i+1]) for i in range(1, len(L) - 1)]

def inside(p, L):
    c = False
    for i in range(len(L)):
        a, b = L[i-1], L[i]
        if (a.y > p.y) != (b.y > p.y) and p.x < (b.x-a.x)*(p.y-a.y)/(b.y-a.y) + a.x: c = not c
    return c

def seg_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2-x1, y2-y1
    t = 0 if dx == dy == 0 else max(0, min(1, ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)))
    return math.hypot(px - (x1 + t*dx), py - (y1 + t*dy))

def line_param(o, d, P, A):
    """Parameter t of the point P + A*t closest to the ray o + d*s (A is a unit vector)."""
    a, b, r = d.dot(d), d.dot(A), o - P
    den = a - b*b
    if abs(den) < 1e-12: return None
    return (a*A.dot(r) - b*d.dot(r)) / den

def ray_hits(o, d, A, B, Cc):
    """Vectorised Moller-Trumbore: ray parameter per triangle (inf where missed)."""
    e1, e2 = B - A, Cc - A; h = np.cross(d, e2); det = (e1*h).sum(1)
    ok = np.abs(det) > 1e-12; inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    sv = o - A; u = inv*(sv*h).sum(1); q = np.cross(sv, e1); v = inv*(q @ d); t = inv*(e2*q).sum(1)
    m = ok & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
    return np.where(m, t, np.inf)

def cross2(a, b): return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]

def rot2(p, c, ang):
    ca, sa = math.cos(ang), math.sin(ang); d = p - c
    return V(c.x + d.x*ca - d.y*sa, c.y + d.x*sa + d.y*ca, 0)

# ---------------------------------------------------------------------------------------------------------------
#  OpenCascade helpers
# ---------------------------------------------------------------------------------------------------------------
def pnt(p): return gp_Pnt(p.x, p.y, p.z)
def gdir(p): return gp_Dir(p.x, p.y, p.z)

def _shape_key(sh):
    try: return sh.HashCode(2147483647)          # OCCT <= 7.7
    except Exception:
        try: return hash(sh)                      # OCCT >= 7.8 hashes by IsSame
        except Exception: return 0

class ShapeIndex:
    """Unique sub-shapes (IsSame semantics) without TopTools maps, whose Python names vary between OCP releases."""
    def __init__(s): s.d, s.items = {}, []
    def find(s, sh):
        for i in s.d.get(_shape_key(sh), ()):
            if s.items[i].IsSame(sh): return i
        return -1
    def add(s, sh):
        i = s.find(sh)
        if i < 0: i = len(s.items); s.items.append(sh); s.d.setdefault(_shape_key(sh), []).append(i)
        return i

def subshapes(shape, kind, unique=True):
    from OCP.TopAbs import TopAbs_WIRE, TopAbs_VERTEX, TopAbs_SHELL
    cast = {TopAbs_FACE: to_face, TopAbs_EDGE: to_edge, TopAbs_SOLID: to_solid, TopAbs_WIRE: st(TopoDS, "Wire"),
            TopAbs_VERTEX: st(TopoDS, "Vertex"), TopAbs_SHELL: st(TopoDS, "Shell")}[kind]
    ex, idx, out = TopExp_Explorer(shape, kind), ShapeIndex(), []
    while ex.More():
        x = cast(ex.Current()); ex.Next()
        if not unique: out.append(x)
        elif idx.find(x) < 0: idx.add(x); out.append(x)
    return out

def solids(shape):
    return subshapes(shape, TopAbs_SOLID) or [shape]

def bbox(shape):
    b = Bnd_Box(); bnd_add(shape, b)
    if b.IsVoid(): return V(), V()
    lo, hi = b.CornerMin(), b.CornerMax()
    return V(lo.X(), lo.Y(), lo.Z()), V(hi.X(), hi.Y(), hi.Z())

def volume(shape):
    p = GProp_GProps(); volume_props(shape, p); return p.Mass()

def compound(shapes):
    c, b = TopoDS_Compound(), BRep_Builder(); b.MakeCompound(c)
    for s in shapes: b.Add(c, s)
    return c

def make_face(kind, pts, plane):
    if kind == "circle":
        c, n, u = plane.w(pts[0]), plane.n, plane.u
        circ = gp_Circ(gp_Ax2(pnt(c), gdir(n), gdir(u)), (pts[1] - pts[0]).Length)
        wire = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(circ).Edge()).Wire()
    else:
        mk = BRepBuilderAPI_MakePolygon()
        for p in loop(kind, pts): mk.Add(pnt(plane.w(p)))
        mk.Close(); wire = mk.Wire()
    return BRepBuilderAPI_MakeFace(wire, True).Face()

def outward(shape):
    return shape.Reversed() if volume(shape) < 0 else shape   # keep solids outward-facing

def prism(face, n, d):
    return outward(BRepPrimAPI_MakePrism(face, gp_Vec(n.x*d, n.y*d, n.z*d)).Shape())

def revolve(face, origin, axis, deg):
    return outward(BRepPrimAPI_MakeRevol(face, gp_Ax1(pnt(origin), gdir(axis)), math.radians(deg)).Shape())

def clean(shape):
    """Merge coplanar / co-cylindrical faces left behind by booleans (like Fusion does)."""
    try:
        u = ShapeUpgrade_UnifySameDomain(shape, True, True, True); u.Build(); return u.Shape()
    except Exception:
        return shape

def boolean(a, b, kind):
    mk = {"fuse": BRepAlgoAPI_Fuse, "cut": BRepAlgoAPI_Cut, "common": BRepAlgoAPI_Common}[kind](a, b)
    if not mk.IsDone(): raise RuntimeError("Boolean operation failed")
    return mk.Shape()

def fuse_all(shapes):
    out = shapes[0]
    for sh in shapes[1:]: out = boolean(out, sh, "fuse")
    return clean(out) if len(shapes) > 1 else out

def transformed(shape, m):
    """m = 3x4 row-major affine matrix (rotation + translation)."""
    t = gp_Trsf(); t.SetValues(*m[0], *m[1], *m[2])
    return BRepBuilderAPI_Transform(shape, t, True).Shape()

def rot_matrix(axis_pt, axis, ang):
    """3x4 matrix rotating by ang (radians) about the line axis_pt + axis*t."""
    k = axis.normalize(); c, s_ = math.cos(ang), math.sin(ang); x, y, z = k.x, k.y, k.z; C1 = 1 - c
    R = [[c + x*x*C1, x*y*C1 - z*s_, x*z*C1 + y*s_], [y*x*C1 + z*s_, c + y*y*C1, y*z*C1 - x*s_], [z*x*C1 - y*s_, z*y*C1 + x*s_, c + z*z*C1]]
    p = axis_pt.t()
    return [R[i] + [p[i] - sum(R[i][j]*p[j] for j in range(3))] for i in range(3)]

def move_matrix(c, d, rot):
    """3x4 row-major matrix: rotate by rot = (x, y, z) degrees about centre c, then translate by d."""
    rx, ry, rz = (math.radians(a) for a in rot)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]); R = Rz @ Ry @ Rx
    cc = np.array(c.t()); tr = cc + np.array(d.t()) - R @ cc
    return [list(R[i]) + [tr[i]] for i in range(3)]

def fillet(shape, edges, r, r2=None):
    mk = BRepFilletAPI_MakeFillet(shape)
    for e in edges:
        if r2 is None or abs(r2 - r) < 1e-9: mk.Add(r, e)
        else: mk.Add(r, r2, e)
    mk.Build()
    if not mk.IsDone(): raise RuntimeError("OpenCascade could not build this fillet")
    return mk.Shape()

def chamfer(shape, edges, d, mode="eq", angle=45.0, d2=None, faces=None):
    mk = BRepFilletAPI_MakeChamfer(shape)
    for i, e in enumerate(edges):
        if mode == "da": mk.AddDA(d, math.radians(angle), e, faces[i])
        elif mode == "dd": mk.Add(d, d2, e, faces[i])
        else: mk.Add(d, e)
    mk.Build()
    if not mk.IsDone(): raise RuntimeError("OpenCascade could not build this chamfer")
    return mk.Shape()

def node(tri, i):
    try: return tri.Node(i)
    except AttributeError: return tri.Nodes().Value(i)

def face_mesh(f):
    """(points Nx3, triangles Mx3) of a meshed face, oriented by the face orientation."""
    loc = TopLoc_Location(); tri = triangulation(f, loc)
    if tri is None or tri.NbTriangles() == 0: return None
    tr = None if loc.IsIdentity() else loc.Transformation()
    P = np.empty((tri.NbNodes(), 3))
    for i in range(1, tri.NbNodes() + 1):
        q = node(tri, i)
        if tr is not None: q = q.Transformed(tr)
        P[i-1] = (q.X(), q.Y(), q.Z())
    T = np.empty((tri.NbTriangles(), 3), dtype=np.int64)
    for i in range(1, tri.NbTriangles() + 1):
        t = tri.Triangle(i); T[i-1] = (t.Value(1), t.Value(2), t.Value(3))
    T -= 1
    if f.Orientation() == TopAbs_REVERSED: T = T[:, [0, 2, 1]]
    return P, T

def edge_pts(e, defl):
    try:
        c = BRepAdaptor_Curve(e); d = GCPnts_QuasiUniformDeflection(c, defl)
        if d.IsDone() and d.NbPoints() > 1:
            return np.array([(q.X(), q.Y(), q.Z()) for q in (d.Value(i) for i in range(1, d.NbPoints() + 1))])
        a, b = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
        return np.array([(a.X(), a.Y(), a.Z()), (b.X(), b.Y(), b.Z())])
    except Exception:
        return np.zeros((0, 3))

# ---- gears & threads ----
ISO_COARSE = [(1, .25), (1.2, .25), (1.6, .35), (2, .4), (2.5, .45), (3, .5), (4, .7), (5, .8), (6, 1.0), (8, 1.25), (10, 1.5),
              (12, 1.75), (14, 2.0), (16, 2.0), (18, 2.5), (20, 2.5), (22, 2.5), (24, 3.0), (27, 3.0), (30, 3.5), (36, 4.0),
              (42, 4.5), (48, 5.0), (56, 5.5), (64, 6.0)]

def iso_suggest(dia, external=True):
    """Nearest ISO metric coarse thread for a rod (external) or a tap-drill hole (internal)."""
    key = (lambda t: abs(t[0] - dia)) if external else (lambda t: abs(t[0] - 1.0825*t[1] - dia))
    return min(ISO_COARSE, key=key)

def spur_gear(m, z, alpha=20.0, thick=8.0, bore=0.0):
    """Involute spur gear centred on the origin, extruded +Z. Flanks are approximated by two circular arcs each."""
    z = int(z); a = math.radians(alpha); r = m*z/2; rb = r*math.cos(a); ra = r + m; rf = max(r - 1.25*m, 0.25*r)
    rs = max(rb, rf)
    inv_pt = lambda t: (rb*(math.cos(t) + t*math.sin(t)), rb*(math.sin(t) - t*math.cos(t)))
    theta = lambda t: t - math.atan(t)
    phi0 = -math.pi/(2*z) - (math.tan(a) - a)
    t0, t1 = math.sqrt(max((rs/rb)**2 - 1, 0)), math.sqrt((ra/rb)**2 - 1)
    while phi0 + theta(t1) > -0.02*math.pi/z and t1 > t0: t1 *= 0.98          # pointed teeth: trim the tip
    def rot(p, g): return (p[0]*math.cos(g) - p[1]*math.sin(g), p[0]*math.sin(g) + p[1]*math.cos(g))
    lower = [rot(inv_pt(t0 + (t1 - t0)*i/4), phi0) for i in range(5)]
    upper = [(x, -y) for x, y in reversed(lower)]
    base_ang = math.atan2(lower[0][1], lower[0][0])
    teeth = []
    for k in range(z):
        g = k*math.tau/z
        lo = [gp_Pnt(*rot(p, g), 0) for p in lower]; up = [gp_Pnt(*rot(p, g), 0) for p in upper]
        rl = gp_Pnt(rf*math.cos(g + base_ang), rf*math.sin(g + base_ang), 0)
        ru = gp_Pnt(rf*math.cos(g - base_ang), rf*math.sin(g - base_ang), 0)
        teeth.append((lo, up, rl, ru, g))
    mkw = BRepBuilderAPI_MakeWire()
    def arc(p, q_, r_): mkw.Add(BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(p, q_, r_).Value()).Edge())
    def seg(p, q_): mkw.Add(BRepBuilderAPI_MakeEdge(p, q_).Edge())
    radial = rf < rb - 1e-6
    for k, (lo, up, rl, ru, g) in enumerate(teeth):
        if radial: seg(rl, lo[0])
        arc(lo[0], lo[1], lo[2]); arc(lo[2], lo[3], lo[4])
        mid = gp_Pnt(ra, 0, 0).Rotated(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), g)
        arc(lo[4], mid, up[0])
        arc(up[0], up[1], up[2]); arc(up[2], up[3], up[4])
        nxt = teeth[(k + 1) % z]
        start = ru if radial else up[4]
        if radial: seg(up[4], ru)
        end = nxt[2] if radial else nxt[0][0]
        gm = g + math.pi/z
        arc(start, gp_Pnt(rf*math.cos(gm), rf*math.sin(gm), 0), end)
    face = BRepBuilderAPI_MakeFace(mkw.Wire(), True).Face()
    shape = outward(BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, thick)).Shape())
    if bore > 0:
        cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), bore/2, thick + 2).Shape()
        shape = boolean(shape, cyl, "cut")
    return shape

def thread_tool(loc, axis, xdir, R, pitch, length, external):
    """Helical 60-degree V groove (ISO profile depth) swept around a cylinder of radius R."""
    surf = Geom_CylindricalSurface(gp_Ax3(pnt(loc), gdir(axis), gdir(xdir)), R)
    line = Geom2d_Line(gp_Pnt2d(0, 0), gp_Dir2d(math.tau, pitch))
    edge = BRepBuilderAPI_MakeEdge(line, surf, 0, length/pitch*math.hypot(math.tau, pitch)).Edge(); build_curves3d(edge)
    spine = BRepBuilderAPI_MakeWire(edge).Wire()
    depth = (0.6134 if external else 0.5413)*pitch; e = 0.15*pitch
    r_out, r_in = (R + e, R - depth) if external else (R - e, R + depth)
    half = (depth + e)*math.tan(math.radians(30))
    P = lambda r, a: pnt(loc + xdir*r + axis*a)
    prof = BRepBuilderAPI_MakePolygon(P(r_out, -half), P(r_out, half), P(r_in, 0), True).Wire()
    ps = BRepOffsetAPI_MakePipeShell(spine); ps.SetMode(gdir(axis)); ps.Add(prof); ps.Build()
    if not ps.IsDone(): raise RuntimeError("Could not sweep the thread profile")
    ps.MakeSolid()
    return outward(ps.Shape())

# ---- native .fission files: the timeline steps (recipes) as JSON, rebuilt on open ----
def enc(x):
    if isinstance(x, V): return {"V": [x.x, x.y, x.z]}
    if isinstance(x, Plane): return {"P": [enc(x.o), enc(x.u), enc(x.v)]}
    if isinstance(x, tuple): return {"T": [enc(i) for i in x]}
    if isinstance(x, list): return [enc(i) for i in x]
    if isinstance(x, dict): return {str(k): enc(v) for k, v in x.items()}
    return x
def dec(x):
    if isinstance(x, list): return [dec(i) for i in x]
    if isinstance(x, dict):
        if len(x) == 1 and "V" in x: return V(*x["V"])
        if len(x) == 1 and "P" in x: return Plane(*(dec(i) for i in x["P"]))
        if len(x) == 1 and "T" in x: return tuple(dec(i) for i in x["T"])
        return {k: dec(v) for k, v in x.items()}
    return x

# ---- file export ----
def write_fcstd(path, shapes):
    """FreeCAD document = zip of Document.xml + one BREP file per Part::Feature."""
    n, objs, data = len(shapes), "", ""
    for i in range(n):
        objs += f'<Object type="Part::Feature" name="Body{i+1}" id="{i+1}"/>'
        data += (f'<Object name="Body{i+1}"><Properties Count="2" TransientCount="0">'
                 f'<Property name="Label" type="App::PropertyString" status="134217728"><String value="Body{i+1}"/></Property>'
                 f'<Property name="Shape" type="Part::PropertyPartShape" status="0"><Part file="PartShape{i}.brp"/></Property>'
                 f'</Properties></Object>')
    xml = ('<?xml version="1.0" encoding="utf-8" standalone="yes"?>'
           '<Document SchemaVersion="4" ProgramVersion="0.21.0" FileVersion="1">'
           f'<Properties Count="0" TransientCount="0"/><Objects Count="{n}">{objs}</Objects>'
           f'<ObjectData Count="{n}">{data}</ObjectData></Document>')
    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Document.xml", xml)
        for i, sh in enumerate(shapes):
            f = os.path.join(td, "s.brp"); brep_write(sh, f); z.write(f, f"PartShape{i}.brp")

def welded(b):
    """Indexed mesh of a body: (unique vertices, triangle index triples) with seam duplicates merged."""
    if not len(b.tv): return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    keys = np.round(b.tv.reshape(-1, 3), 5)
    verts, inv = np.unique(keys, axis=0, return_inverse=True)
    tris = inv.reshape(-1, 3)
    ok = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    return verts, tris[ok]

def write_3mf(path, bodies):
    """3MF = zip with a content-types part, a rels part and 3D/3dmodel.model (mm, indexed meshes)."""
    objs = []
    for oid, b in enumerate(bodies, 1):
        verts, tris = welded(b)
        vx = "".join(f'<vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>' for x, y, z in verts)
        tx = "".join(f'<triangle v1="{i}" v2="{j}" v3="{k}"/>' for i, j, k in tris)
        objs.append(f'<object id="{oid}" name="Body{oid}" type="model"><mesh><vertices>{vx}</vertices>'
                    f'<triangles>{tx}</triangles></mesh></object>')
    items = "".join(f'<item objectid="{i}"/>' for i in range(1, len(bodies) + 1))
    model = ('<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             f'<resources>{"".join(objs)}</resources><build>{items}</build></model>')
    types = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types); z.writestr("_rels/.rels", rels); z.writestr("3D/3dmodel.model", model)

def write_obj(path, bodies):
    with open(path, "w") as f:
        base = 1
        for i, b in enumerate(bodies):
            verts, tris = welded(b)
            f.write(f"o Body{i+1}\n")
            f.writelines(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in verts)
            f.writelines(f"f {a+base} {c+base} {d+base}\n" for a, c, d in tris)
            base += len(verts)

# ---------------------------------------------------------------------------------------------------------------
#  Solid-modeling kernel (Fission 0.5): extrude options, sweep, loft, rib/web, emboss, hole, primitives, patterns,
#  mirror, thicken, boundary fill, shell, draft, scale, combine, offset/replace/split faces, split body, defeature
# ---------------------------------------------------------------------------------------------------------------
from OCP.BRepOffsetAPI import (BRepOffsetAPI_ThruSections, BRepOffsetAPI_MakeOffset, BRepOffsetAPI_MakeThickSolid,
                               BRepOffsetAPI_DraftAngle, BRepOffsetAPI_MakeOffsetShape)
from OCP.BRepOffset import BRepOffset_MakeOffset, BRepOffset_Skin
from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing, BRepAlgoAPI_Section
from OCP.BRepFeat import BRepFeat_SplitShape
from OCP.BOPAlgo import BOPAlgo_Splitter, BOPAlgo_MakerVolume
from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform, BRepBuilderAPI_RightCorner, BRepBuilderAPI_MakeVertex
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf, GeomAPI_PointsToBSpline
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepAdaptor import BRepAdaptor_CompCurve
from OCP.BRepFill import BRepFill_Contact
from OCP.ChFi3d import ChFi3d_Polynomial, ChFi3d_Rational
from OCP.GeomAbs import GeomAbs_Arc, GeomAbs_Intersection, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_Line
from OCP.TopAbs import TopAbs_VERTEX, TopAbs_WIRE, TopAbs_IN, TopAbs_ON
from OCP.gp import gp_GTrsf, gp_Mat, gp_XYZ, gp_Pln
from OCP.collections import List_TopoDS_Shape
from OCP.Geom import Geom_ConicalSurface

to_wire, to_vertex = st(TopoDS, "Wire"), st(TopoDS, "Vertex")
outer_wire = st(BRepTools, "OuterWire")
surface_props = st(BRepGProp, "SurfaceProperties")
linear_props = st(BRepGProp, "LinearProperties")

def vec(p): return V(p.X(), p.Y(), p.Z())

def as_face(sh):
    """A TopoDS_Face from a face / compound holding one face (booleans of faces return compounds)."""
    if sh.ShapeType() == TopAbs_FACE: return to_face(sh)
    fs = subshapes(sh, TopAbs_FACE)
    if not fs: raise RuntimeError("expected a face")
    return fs[0]

def as_solid(sh):
    """Closed shells (what BRepOffset returns) are wrapped into solids."""
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    if subshapes(sh, TopAbs_SOLID): return sh
    shells = subshapes(sh, TopAbs_SHELL)
    if not shells: raise RuntimeError("the result isn't a closed solid")
    mk = BRepBuilderAPI_MakeSolid()
    for x in shells: mk.Add(x)
    return outward(mk.Solid())

def shape_list(shapes):
    l = List_TopoDS_Shape()
    for x in shapes: l.Append(x)
    return l

def centroid(shape):
    p = GProp_GProps()
    try:
        if shape.ShapeType() == TopAbs_SOLID or subshapes(shape, TopAbs_SOLID): volume_props(shape, p)
        elif subshapes(shape, TopAbs_FACE): surface_props(shape, p)
        else: linear_props(shape, p)
        return vec(p.CentreOfMass())
    except Exception:
        lo, hi = bbox(shape); return (lo + hi)*0.5

def face_frame(face):
    """(centroid, unit outward normal) of a planar face."""
    srf = BRepAdaptor_Surface(face)
    if srf.GetType() != GeomAbs_Plane: raise RuntimeError("that face isn't flat")
    pl = srf.Plane(); n = vec(pl.Axis().Direction())
    if face.Orientation() == TopAbs_REVERSED: n = -n
    return centroid(face), n.normalize()

def normal_at(face, p):
    """Outward unit normal of any face at (the projection of) world point p, plus the projected point."""
    srf = BRepAdaptor_Surface(face); pr = GeomAPI_ProjectPointOnSurf(pnt(p), st(BRep_Tool, "Surface")(face))
    u, v = pr.LowerDistanceParameters() if pr.NbPoints() else (srf.FirstUParameter(), srf.FirstVParameter())
    q, du, dv = gp_Pnt(), gp_Vec(), gp_Vec(); srf.D1(u, v, q, du, dv)
    n = vec(du.Crossed(dv))
    if face.Orientation() == TopAbs_REVERSED: n = -n
    return vec(q), n.normalize()

def translated(shape, d):
    t = gp_Trsf(); t.SetTranslation(gp_Vec(d.x, d.y, d.z)); return BRepBuilderAPI_Transform(shape, t, True).Shape()

def mirrored(shape, o, n):
    t = gp_Trsf(); t.SetMirror(gp_Ax2(pnt(o), gdir(n))); return BRepBuilderAPI_Transform(shape, t, True).Shape()

def rotated(shape, o, axis, ang):
    t = gp_Trsf(); t.SetRotation(gp_Ax1(pnt(o), gdir(axis)), ang); return BRepBuilderAPI_Transform(shape, t, True).Shape()

def scaled(shape, c, sx, sy=None, sz=None):
    if sy is None or (abs(sx - sy) < 1e-12 and abs(sx - sz) < 1e-12):
        t = gp_Trsf(); t.SetScale(pnt(c), sx); return BRepBuilderAPI_Transform(shape, t, True).Shape()
    g = gp_GTrsf(); g.SetVectorialPart(gp_Mat(sx, 0, 0, 0, sy, 0, 0, 0, sz))
    g.SetTranslationPart(gp_XYZ(c.x*(1 - sx), c.y*(1 - sy), c.z*(1 - sz)))
    return BRepBuilderAPI_GTransform(shape, g, True).Shape()

def big_plane_face(o, n, size):
    pl = gp_Pln(pnt(o), gdir(n)); return BRepBuilderAPI_MakeFace(pl, -size, size, -size, size).Face()

def big_surface_face(face, size):
    """The face's underlying surface, untrimmed (planes are clipped to a big square)."""
    srf = BRepAdaptor_Surface(face)
    if srf.GetType() == GeomAbs_Plane:
        c, n = face_frame(face); return big_plane_face(c, n, size)
    s = st(BRep_Tool, "Surface")(face)
    u0, u1, v0, v1 = srf.FirstUParameter(), srf.LastUParameter(), srf.FirstVParameter(), srf.LastVParameter()
    if srf.GetType() in (GeomAbs_Cylinder, GeomAbs_Cone): u0, u1, v0, v1 = 0, math.tau, v0 - size, v1 + size
    return BRepBuilderAPI_MakeFace(s, u0, u1, v0, v1, 1e-6).Face()

def split_keep(shape, tool, keep_pt):
    """Split shape by tool and keep the solid piece that contains (or touches) keep_pt."""
    sp = BOPAlgo_Splitter(); sp.AddArgument(shape); sp.AddTool(tool); sp.Perform()
    pieces = subshapes(sp.Shape(), TopAbs_SOLID)
    if not pieces: raise RuntimeError("couldn't split against that object")
    def dist(p):
        d = BRepExtrema_DistShapeShape(p, BRepBuilderAPI_MakeVertex(pnt(keep_pt)).Vertex()); d.Perform(); return d.Value()
    return min(pieces, key=dist)

def classify(shape, p):
    c = BRepClass3d_SolidClassifier(shape, pnt(p), 1e-6); return c.State()

def shape_dist(a, b):
    d = BRepExtrema_DistShapeShape(a, b); d.Perform(); return d.Value() if d.IsDone() else 1e9

def model_span(shapes):
    if not shapes: return 1000.0
    lo, hi = bbox(compound(shapes)); return max((hi - lo).Length, 10.0)

# ---- extrude ----
def offset_face(face, d):
    """2D offset of a planar face (outer grows, holes shrink for d > 0). Returns a face."""
    face = as_face(face); mo = BRepOffsetAPI_MakeOffset(face, GeomAbs_Arc); mo.Perform(d)
    if not mo.IsDone(): raise RuntimeError("couldn't offset the profile")
    wires = subshapes(mo.Shape(), TopAbs_WIRE)
    if not wires: raise RuntimeError("the taper closes the profile")
    def area(w):
        try: return abs(face_area(BRepBuilderAPI_MakeFace(w, True).Face()))
        except Exception: return 0
    wires.sort(key=area, reverse=True); mk = BRepBuilderAPI_MakeFace(wires[0], True)
    for w in wires[1:]: mk.Add(to_wire(w.Reversed()))
    return mk.Face()

def face_area(f):
    p = GProp_GProps(); surface_props(f, p); return p.Mass()

def wires_of(face):
    face = as_face(face); ow = outer_wire(face); inner = [w for w in subshapes(face, TopAbs_WIRE) if not w.IsSame(ow)]
    return ow, inner

def loft_wires(w1, w2, ruled=True):
    ts = BRepOffsetAPI_ThruSections(True, ruled, 1e-6); ts.AddWire(w1); ts.AddWire(w2); ts.CheckCompatibility(True); ts.Build()
    if not ts.IsDone(): raise RuntimeError("loft failed")
    return outward(ts.Shape())

def tapered_prism(face, n, L, taper_deg):
    """Face swept L along n while its outline shrinks (positive angle) or grows (negative)."""
    face = as_face(face)
    if abs(taper_deg) < 1e-9: return prism(face, n, L)
    top = translated(offset_face(face, -L*math.tan(math.radians(taper_deg))), n*L)
    o1, i1 = wires_of(face); o2, i2 = wires_of(top)
    solid = loft_wires(o1, o2)
    for w in i1:                                            # matching hole in the top by nearest centre
        c = centroid(w); w2 = min(i2, key=lambda x: (centroid(x) - n*L - c).Length) if i2 else None
        if w2 is not None: solid = boolean(solid, loft_wires(w, w2), "cut")
    return solid

def thin_face(face_or_wire, thick, side):
    """Ring-shaped face around a profile outline (side: 'in' | 'out' | 'center')."""
    if face_or_wire.ShapeType() == TopAbs_FACE: f = face_or_wire
    else: f = BRepBuilderAPI_MakeFace(to_wire(face_or_wire), True).Face()
    a, b = {"in": (-thick, 0), "out": (0, thick), "center": (-thick/2, thick/2)}[side]
    outer = offset_face(f, b) if b else f
    inner = offset_face(f, a) if a else f
    return boolean(outer, inner, "cut")

def open_wire_thick_face(pts, thick, plane_n, side="center"):
    """Flat-ended strip of width `thick` along an open polyline (world points) lying in the plane with normal plane_n.
    side: 'center' | 'left' | 'right' (relative to the line direction seen from plane_n)."""
    if len(pts) < 2: raise RuntimeError("that line is too short")
    sh = {"center": (-0.5, 0.5), "left": (0.0, 1.0), "right": (-1.0, 0.0)}[side]
    pieces = []
    for a, b in zip(pts, pts[1:]):
        d = b - a
        if d.Length < 1e-9: continue
        m = plane_n.cross(d).normalize(); lo, hi = m*(sh[0]*thick), m*(sh[1]*thick)
        mk = BRepBuilderAPI_MakePolygon(pnt(a + lo), pnt(b + lo), pnt(b + hi), pnt(a + hi), True)
        pieces.append(BRepBuilderAPI_MakeFace(mk.Wire(), True).Face())
    for p in pts[1:-1]:                                           # round the inside corners
        if side == "center":
            pieces.append(BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(pnt(p), gdir(plane_n)), thick/2)).Edge()).Wire(), True).Face())
    out = pieces[0]
    for p in pieces[1:]: out = boolean(out, p, "fuse")
    u = ShapeUpgrade_UnifySameDomain(out, True, True, True); u.Build()
    return as_face(u.Shape())

# ---- paths & curves ----
def edge_ends(e):
    c = BRepAdaptor_Curve(e); return vec(c.Value(c.FirstParameter())), vec(c.Value(c.LastParameter()))

def chain_wire(edges):
    """Order edges end-to-end and make one wire (raises if they don't connect)."""
    edges = list(edges)
    if not edges: raise RuntimeError("no path selected")
    if len(edges) == 1: return BRepBuilderAPI_MakeWire(edges[0]).Wire()
    order = [edges.pop(0)]; a, b = edge_ends(order[0]); tol = 1e-4
    while edges:
        for i, e in enumerate(edges):
            p, q = edge_ends(e)
            if min((p - b).Length, (q - b).Length) < tol:
                order.append(e); b = q if (p - b).Length < tol else p; edges.pop(i); break
            if min((p - a).Length, (q - a).Length) < tol:
                order.insert(0, e); a = q if (q - a).Length >= tol and (p - a).Length < tol else p
                a = p if (q - a).Length < tol else q if (p - a).Length < tol else a; edges.pop(i); break
        else: raise RuntimeError("the path pieces don't join up end to end")
    mk = BRepBuilderAPI_MakeWire()
    for e in order: mk.Add(e)
    if not mk.IsDone(): raise RuntimeError("the path pieces don't join up end to end")
    return mk.Wire()

def wire_length(w):
    p = GProp_GProps(); linear_props(w, p); return p.Mass()

def wire_eval(w, frac):
    """Point and unit tangent at a fraction of the wire's length."""
    c = BRepAdaptor_CompCurve(w); L = wire_length(w); f0, f1 = c.FirstParameter(), c.LastParameter()
    from OCP.GCPnts import GCPnts_AbscissaPoint
    try: u = GCPnts_AbscissaPoint(c, frac*L, f0).Parameter()
    except Exception: u = f0 + (f1 - f0)*frac
    p, d = gp_Pnt(), gp_Vec(); c.D1(u, p, d)
    return vec(p), vec(d).normalize()

def trim_wire(w, frac):
    """The first `frac` of a wire (by length)."""
    if frac >= 0.9999: return w
    L = wire_length(w)*max(frac, 1e-3); acc = 0.0; mk = BRepBuilderAPI_MakeWire()
    from OCP.GCPnts import GCPnts_AbscissaPoint
    from OCP.BRepTools import BRepTools_WireExplorer
    ex = BRepTools_WireExplorer(w)
    while ex.More():
        e = ex.Current(); ex.Next(); c = BRepAdaptor_Curve(e); el = GCPnts_AbscissaPoint.Length_s(c) if hasattr(GCPnts_AbscissaPoint, "Length_s") else GCPnts_AbscissaPoint.Length(c)
        if acc + el <= L + 1e-9: mk.Add(e); acc += el; continue
        if L - acc < 1e-7: break                                   # cut falls exactly on a corner
        u0, u1 = c.FirstParameter(), c.LastParameter(); rev = e.Orientation() == TopAbs_REVERSED
        start = u1 if rev else u0; um = GCPnts_AbscissaPoint(c, (-1 if rev else 1)*(L - acc), start).Parameter()
        crv = st(BRep_Tool, "Curve")(e, 0.0, 1.0)
        ne = BRepBuilderAPI_MakeEdge(crv, min(start, um), max(start, um)).Edge()
        mk.Add(to_edge(ne.Reversed()) if rev else ne); break
    return mk.Wire()

# ---- sweep ----
def rmf_frames(w, n):
    """n+1 rotation-minimising frames (point, tangent, normal) along a wire."""
    out = []; prev = None
    for k in range(n + 1):
        p, t = wire_eval(w, k/n)
        if prev is None:
            a = V(0, 0, 1) if abs(t.z) < 0.9 else V(1, 0, 0); r = t.cross(a).normalize()
        else:                                               # double reflection
            p0, t0, r0 = prev; v1 = p - p0; c1 = v1.dot(v1)
            if c1 < 1e-18: r = r0
            else:
                rl = r0 - v1*(2/c1*v1.dot(r0)); tl = t0 - v1*(2/c1*v1.dot(t0)); v2 = t - tl; c2 = v2.dot(v2)
                r = rl - v2*(2/c2*v2.dot(rl)) if c2 > 1e-18 else rl
        r = (r - t*r.dot(t)).normalize(); prev = (p, t, r); out.append((p, t, r))
    return out

def frame_trsf(p0, t0, r0, p1, t1, r1, scale=1.0, twist=0.0):
    """Transform taking frame 0 to frame 1 (with scale about p1 and twist about t1)."""
    b0, b1 = t0.cross(r0), t1.cross(r1)
    M0 = np.array([r0.t(), b0.t(), t0.t()]).T; M1 = np.array([r1.t(), b1.t(), t1.t()]).T
    ca, sa = math.cos(twist), math.sin(twist); Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    R = M1 @ Rz @ M0.T * scale; tr = np.array(p1.t()) - R @ np.array(p0.t())
    return [list(R[i]) + [tr[i]] for i in range(3)]

def gtrsf_apply(shape, m):
    g = gp_GTrsf(); g.SetVectorialPart(gp_Mat(*m[0][:3], *m[1][:3], *m[2][:3])); g.SetTranslationPart(gp_XYZ(m[0][3], m[1][3], m[2][3]))
    return BRepBuilderAPI_GTransform(shape, g, True).Shape()

def sweep_face(face, path, guide=None, frac=1.0, taper=0.0, twist=0.0, orient="perp"):
    """Sweep a planar profile along a path wire. Plain sweeps keep sharp corners; taper / twist use lofted sections."""
    path = trim_wire(path, frac); ow, inner = wires_of(face)
    if abs(taper) < 1e-9 and abs(twist) < 1e-9:
        def one(w):
            ps = BRepOffsetAPI_MakePipeShell(path)
            if guide is not None: ps.SetMode(guide, True, BRepFill_Contact)
            elif orient == "parallel":
                c, n = face_frame(face); ps.SetMode(gp_Ax2(pnt(c), gdir(n)))
            ps.SetTransitionMode(BRepBuilderAPI_RightCorner); ps.Add(w, False, False); ps.Build()
            if not ps.IsDone(): raise RuntimeError("the sweep failed - is the profile at the start of the path?")
            ps.MakeSolid(); return outward(ps.Shape())
        solid = one(ow)
        for w in inner: solid = boolean(solid, one(w), "cut")
        return solid
    N = 24; fr = rmf_frames(path, N); p0, t0, r0 = fr[0]; L = wire_length(path)
    lo, hi = bbox(face); R = max((hi - lo).Length/2, 1e-6); s_end = max(0.02, 1 - L*math.tan(math.radians(taper))/R)
    def loft(w):
        ts = BRepOffsetAPI_ThruSections(True, False, 1e-5)
        for k, (p, t, r) in enumerate(fr):
            u = k/N; ts.AddWire(to_wire(gtrsf_apply(w, frame_trsf(p0, t0, r0, p, t, r, 1 + (s_end - 1)*u, math.radians(twist)*u))))
        ts.Build()
        if not ts.IsDone(): raise RuntimeError("the sweep failed")
        return outward(ts.Shape())
    solid = loft(ow)
    for w in inner: solid = boolean(solid, loft(w), "cut")
    return solid

# ---- loft ----
def loft_sections(sections, rails=None, centerline=None, closed=False, ruled=False):
    """sections: wires or vertices (end points). Rails / centreline steer the shape."""
    if len(sections) < 2: raise RuntimeError("pick at least two profiles")
    if centerline is not None or rails:
        spine = centerline
        if spine is None:                                      # rails without a centreline: spine through section centres
            cs = [centroid(s_) for s_ in sections]; mk = BRepBuilderAPI_MakePolygon()
            for c in cs: mk.Add(pnt(c))
            spine = mk.Wire()
        from OCP.BRepFill import BRepFill_ContactOnBorder, BRepFill_NoContact
        for kc in ((BRepFill_ContactOnBorder, BRepFill_NoContact) if rails else (None,)):
            ps = BRepOffsetAPI_MakePipeShell(spine)
            if rails: ps.SetMode(rails[0], True, kc)
            for s_ in sections: ps.Add(s_, False, True)
            ps.Build()
            if ps.IsDone():
                ps.MakeSolid(); return outward(ps.Shape())
        raise RuntimeError("the guided loft failed - check the profiles lie along the rail / centreline")
    ts = BRepOffsetAPI_ThruSections(True, ruled, 1e-6); ts.CheckCompatibility(True)
    for s_ in sections + ([sections[0]] if closed else []):
        if s_.ShapeType() == TopAbs_VERTEX: ts.AddVertex(to_vertex(s_))
        else: ts.AddWire(to_wire(s_))
    ts.Build()
    if not ts.IsDone(): raise RuntimeError("the loft failed")
    return outward(ts.Shape())

# ---- primitives ----
def frame_on(pl, cx, cy):
    return pl.w(V(cx, cy, 0)), pl.n, pl.u

def make_box(pl, cx, cy, L, W_, H):
    o = pl.w(V(cx - L/2, cy - W_/2, 0)); ax = gp_Ax2(pnt(o), gdir(pl.n if H >= 0 else -pl.n), gdir(pl.u))
    if H < 0: ax = gp_Ax2(pnt(pl.w(V(cx - L/2, cy + W_/2, 0))), gdir(-pl.n), gdir(pl.u))
    return outward(BRepPrimAPI_MakeBox(ax, L, W_, abs(H)).Shape())

def make_cylinder(pl, cx, cy, D, H):
    c = pl.w(V(cx, cy, 0)); n = pl.n if H >= 0 else -pl.n
    return BRepPrimAPI_MakeCylinder(gp_Ax2(pnt(c), gdir(n)), D/2, abs(H)).Shape()

def make_sphere(pl, cx, cy, D):
    return BRepPrimAPI_MakeSphere(pnt(pl.w(V(cx, cy, D/2))), D/2).Shape()

def make_torus(pl, cx, cy, Dmaj, Dmin):
    return BRepPrimAPI_MakeTorus(gp_Ax2(pnt(pl.w(V(cx, cy, Dmin/2))), gdir(pl.n)), Dmaj/2, Dmin/2).Shape()

def helix_wire(c, n, x, R, pitch, height, angle_deg=0.0):
    """Helix around axis (c, n) starting along x: cylindrical or conical (taper angle)."""
    if abs(angle_deg) < 1e-6: srf = Geom_CylindricalSurface(gp_Ax3(pnt(c), gdir(n), gdir(x)), R)
    else: srf = Geom_ConicalSurface(gp_Ax3(pnt(c), gdir(n), gdir(x)), math.radians(angle_deg), R)
    if abs(angle_deg) < 1e-6:
        ln = Geom2d_Line(gp_Pnt2d(0, 0), gp_Dir2d(math.tau, pitch)); L = height/pitch*math.hypot(math.tau, pitch)
    else:
        sl = pitch/math.cos(math.radians(angle_deg)); ln = Geom2d_Line(gp_Pnt2d(0, 0), gp_Dir2d(math.tau, sl))
        L = height/math.cos(math.radians(angle_deg))/sl*math.hypot(math.tau, sl)
    e = BRepBuilderAPI_MakeEdge(ln, srf, 0, L).Edge(); build_curves3d(e)
    return BRepBuilderAPI_MakeWire(e).Wire()

def section_face(shape, size, at, radial, axial):
    """Coil / pipe cross-section centred at `at` in the plane spanned by radial & axial."""
    if shape == "circle":
        nrm = radial.cross(axial).normalize()
        return BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(pnt(at), gdir(nrm), gdir(radial)), size/2)).Edge()).Wire(), True).Face()
    h = size/2
    if shape == "square": pts = [(-h, -h), (h, -h), (h, h), (-h, h)]
    elif shape == "tri_out": pts = [(-h*0.577, -h), (h*1.155 - h*0.577, 0), (-h*0.577, h)]
    else: pts = [(h*0.577, -h), (-(h*1.155 - h*0.577), 0), (h*0.577, h)]
    mk = BRepBuilderAPI_MakePolygon()
    for a, b in pts: mk.Add(pnt(at + radial*a + axial*b))
    mk.Close(); return BRepBuilderAPI_MakeFace(mk.Wire(), True).Face()

def make_coil(pl, cx, cy, D, revs, height, pitch, mode, angle, section, sec_pos, size):
    """Spring / coil like Fusion's Coil command."""
    if mode == "rev_height": pitch = height/max(revs, 1e-6)
    elif mode == "rev_pitch": height = revs*pitch
    elif mode == "height_pitch": revs = height/max(pitch, 1e-6)
    R = D/2 + {"inside": -size/2, "center": 0.0, "outside": size/2}[sec_pos]
    c = pl.w(V(cx, cy, 0)); n, x = pl.n, pl.u
    if mode == "spiral": return make_spiral(pl, cx, cy, D, revs, pitch, section, size)
    w = helix_wire(c, n, x, R, pitch, height, angle)
    prof = section_face(section, size, c + x*R, x, n)
    ps = BRepOffsetAPI_MakePipeShell(w); ps.SetMode(gdir(n)); ps.Add(outer_wire(prof), False, False); ps.Build()
    if not ps.IsDone(): raise RuntimeError("the coil couldn't be built - try a smaller section or larger pitch")
    ps.MakeSolid(); return outward(ps.Shape())

def spiral_wire(c, n, x, R, pitch, revs):
    from OCP.collections import Array1_gp_Pnt
    N = max(int(revs*36), 8); arr = Array1_gp_Pnt(1, N + 1); y = n.cross(x)
    for k in range(N + 1):
        a = k/36*math.tau; r = R + pitch*a/math.tau; arr.SetValue(k + 1, pnt(c + x*(r*math.cos(a)) + y*(r*math.sin(a))))
    bs = GeomAPI_PointsToBSpline(arr, 3, 8).Curve()
    return BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(bs).Edge()).Wire()

def make_spiral(pl, cx, cy, D, revs, pitch, section, size):
    c = pl.w(V(cx, cy, size/2)); n, x = pl.n, pl.u; R = D/2
    w = spiral_wire(c, n, x, R, pitch, revs)
    prof = section_face(section, size, c + x*R, x, n)
    ps = BRepOffsetAPI_MakePipeShell(w); ps.SetMode(gdir(n)); ps.Add(outer_wire(prof), False, False); ps.Build()
    if not ps.IsDone(): raise RuntimeError("the spiral couldn't be built - try a larger pitch")
    ps.MakeSolid(); return outward(ps.Shape())

def make_pipe(path, section, size, hollow, thick, frac=1.0):
    path = trim_wire(path, frac); p, t = wire_eval(path, 0.0)
    a = V(0, 0, 1) if abs(t.z) < 0.9 else V(1, 0, 0); r = t.cross(a).normalize(); b = t.cross(r)
    def sw(sz):
        f = section_face(section, sz, p, r, b)
        ps = BRepOffsetAPI_MakePipeShell(path); ps.SetTransitionMode(BRepBuilderAPI_RightCorner)
        ps.Add(outer_wire(f), False, False); ps.Build()
        if not ps.IsDone(): raise RuntimeError("the pipe couldn't follow that path")
        ps.MakeSolid(); return outward(ps.Shape())
    solid = sw(size)
    if hollow:
        if thick*2 >= size: raise RuntimeError("the wall is thicker than half the pipe size")
        solid = boolean(solid, sw(size - 2*thick), "cut")
    return solid

# ---- hole ----
def hole_tool(c, d, spec, big):
    """Revolved cutter for a hole at point c drilling along unit vector d (into the material)."""
    R = spec["dia"]/2; eps = max(spec["dia"]*0.02, 0.01)
    depth = big if spec.get("extent") == "all" else spec["depth"]
    pts = [(0, -eps)]
    if spec["type"] == "cbore":
        rc = max(spec["cb_dia"]/2, R); pts += [(rc, -eps), (rc, spec["cb_depth"]), (R, spec["cb_depth"])]
    elif spec["type"] == "csink":
        rs = max(spec["cs_dia"]/2, R); h = (rs - R)/math.tan(math.radians(spec["cs_angle"])/2)
        pts += [(rs + eps*math.tan(math.radians(spec["cs_angle"])/2), -eps), (R, h)]
    else: pts += [(R, -eps)]
    pts += [(R, depth)]
    if spec.get("tip") == "angle" and spec.get("extent") != "all":
        pts += [(0, depth + R/math.tan(math.radians(spec.get("tip_angle", 118))/2))]
    else: pts += [(0, depth)]
    x = V(1, 0, 0) if abs(d.x) < 0.9 else V(0, 1, 0); x = (x - d*x.dot(d)).normalize()
    mk = BRepBuilderAPI_MakePolygon()
    for r, z in pts: mk.Add(pnt(c + x*r + d*z))
    mk.Close(); f = BRepBuilderAPI_MakeFace(mk.Wire(), True).Face()
    return outward(BRepPrimAPI_MakeRevol(f, gp_Ax1(pnt(c), gdir(d)), math.tau).Shape())

# ---- thread tables: (label, major diameter mm, pitch mm, profile) ----
def _inch(name, d_in, tpi): return (name, d_in*25.4, 25.4/tpi)
THREADS = {
    "ISO Metric coarse": [(f"M{fmt(d)}x{fmt(p)}", d, p) for d, p in ISO_COARSE],
    "ISO Metric fine": [(f"M{fmt(d)}x{fmt(p)}", d, p) for d, p in ((6, .75), (8, 1.0), (10, 1.25), (10, 1.0), (12, 1.5), (12, 1.25), (14, 1.5),
                                                                    (16, 1.5), (18, 1.5), (20, 1.5), (22, 1.5), (24, 2.0), (27, 2.0), (30, 2.0), (36, 3.0))],
    "ANSI Unified Coarse (UNC)": [_inch(n, d, t) for n, d, t in (("#2-56", .086, 56), ("#4-40", .112, 40), ("#6-32", .138, 32), ("#8-32", .164, 32),
                                  ("#10-24", .19, 24), ("1/4-20", .25, 20), ("5/16-18", .3125, 18), ("3/8-16", .375, 16), ("7/16-14", .4375, 14),
                                  ("1/2-13", .5, 13), ("9/16-12", .5625, 12), ("5/8-11", .625, 11), ("3/4-10", .75, 10), ("7/8-9", .875, 9), ("1-8", 1.0, 8))],
    "ANSI Unified Fine (UNF)": [_inch(n, d, t) for n, d, t in (("#2-64", .086, 64), ("#4-48", .112, 48), ("#6-40", .138, 40), ("#8-36", .164, 36),
                                 ("#10-32", .19, 32), ("1/4-28", .25, 28), ("5/16-24", .3125, 24), ("3/8-24", .375, 24), ("7/16-20", .4375, 20),
                                 ("1/2-20", .5, 20), ("9/16-18", .5625, 18), ("5/8-18", .625, 18), ("3/4-16", .75, 16), ("7/8-14", .875, 14), ("1-12", 1.0, 12))],
    "BSP parallel (G)": [_inch(n, d, t) for n, d, t in (("G1/8", .383, 28), ("G1/4", .518, 19), ("G3/8", .656, 19), ("G1/2", .825, 14),
                         ("G3/4", 1.041, 14), ("G1", 1.309, 11))],
    "ACME": [_inch(n, d, t) for n, d, t in (("1/4-16", .25, 16), ("3/8-12", .375, 12), ("1/2-10", .5, 10), ("5/8-8", .625, 8), ("3/4-6", .75, 6), ("1-5", 1.0, 5))],
}
THREAD_PROFILE = {"BSP parallel (G)": 55.0, "ACME": 29.0}

def thread_tool2(loc, axis, xdir, R, pitch, length, external, angle=60.0, lefthand=False):
    """Helical groove with a V (60 / 55 deg) or trapezoidal (ACME 29 deg) profile; left- or right-handed."""
    surf = Geom_CylindricalSurface(gp_Ax3(pnt(loc), gdir(axis), gdir(xdir)), R)
    line = Geom2d_Line(gp_Pnt2d(0, 0), gp_Dir2d(-math.tau if lefthand else math.tau, pitch))
    edge = BRepBuilderAPI_MakeEdge(line, surf, 0, length/pitch*math.hypot(math.tau, pitch)).Edge(); build_curves3d(edge)
    spine = BRepBuilderAPI_MakeWire(edge).Wire(); e = 0.15*pitch
    half_ang = math.radians(angle)/2
    if angle < 40:                                        # ACME: flat crest and root, depth 0.5 p
        depth = 0.5*pitch; flat = 0.3707*pitch/2
        r_out, r_in = (R + e, R - depth) if external else (R - e, R + depth)
        half_out = flat + (depth + e)*math.tan(half_ang)
        P = lambda r, a: pnt(loc + xdir*r + axis*a)
        prof = BRepBuilderAPI_MakePolygon(P(r_out, -half_out), P(r_out, half_out), P(r_in, flat), P(r_in, -flat), True).Wire()
    else:
        depth = (0.6134 if external else 0.5413)*pitch*(1.0 if angle >= 59 else 1.05)
        r_out, r_in = (R + e, R - depth) if external else (R - e, R + depth)
        half = (depth + e)*math.tan(half_ang)
        P = lambda r, a: pnt(loc + xdir*r + axis*a)
        prof = BRepBuilderAPI_MakePolygon(P(r_out, -half), P(r_out, half), P(r_in, 0), True).Wire()
    ps = BRepOffsetAPI_MakePipeShell(spine); ps.SetMode(gdir(axis)); ps.Add(prof); ps.Build()
    if not ps.IsDone(): raise RuntimeError("Could not sweep the thread profile")
    ps.MakeSolid(); return outward(ps.Shape())

# ---- modify ----
def shell_solid(shape, faces, thick, direction):
    off = {"in": -thick, "out": thick, "both": -thick/2}[direction]
    mk = BRepOffsetAPI_MakeThickSolid(); mk.MakeThickSolidByJoin(shape, shape_list(faces), off, 1e-4, BRepOffset_Skin, False, False, GeomAbs_Intersection)
    mk.Build()
    if not mk.IsDone(): raise RuntimeError("the shell failed - try a thinner wall")
    out = mk.Shape()
    if direction == "both":                                  # wall centred on the old surface: inner half + outer half
        mk2 = BRepOffsetAPI_MakeThickSolid(); mk2.MakeThickSolidByJoin(shape, shape_list(faces), thick/2, 1e-4, BRepOffset_Skin, False, False, GeomAbs_Intersection); mk2.Build()
        if not mk2.IsDone(): raise RuntimeError("the shell failed - try a thinner wall")
        out = clean(boolean(out, mk2.Shape(), "fuse"))
    return out

def draft_faces(shape, faces, pull_dir, neutral_o, neutral_n, angle_deg):
    mk = BRepOffsetAPI_DraftAngle(shape); pln = gp_Pln(pnt(neutral_o), gdir(neutral_n))
    for f in faces: mk.Add(f, gdir(pull_dir), math.radians(angle_deg), pln)
    mk.Build()
    if not mk.IsDone(): raise RuntimeError("the draft failed for one of the faces (only faces that run along the pull direction can be drafted)")
    return mk.Shape()

def offset_faces(shape, faces, d):
    """Push / pull faces of a solid by d along their normals (like Fusion's Offset Face / Press Pull)."""
    try:
        mk = BRepOffset_MakeOffset(); mk.Initialize(shape, 0.0, 1e-4, BRepOffset_Skin, False, False, GeomAbs_Intersection, False)
        for f in faces: mk.SetOffsetOnFace(f, d)
        mk.MakeOffsetShape()
        if mk.IsDone():
            out = as_solid(mk.Shape())
            if abs(volume(out)) > 1e-9: return clean(out)
    except Exception: pass
    res = shape                                              # fallback: planar faces only
    for f in faces:
        c, n = face_frame(f)
        res = boolean(res, prism(f, n, d), "fuse" if d > 0 else "cut")
    return clean(res)

def replace_with_plane(shape, faces, o, n, big):
    res = shape
    for f in faces:
        c, fn = face_frame(f); den = fn.dot(n)
        if abs(den) < 1e-9: raise RuntimeError("the new plane is parallel to the face's normal")
        L = (o - c).dot(n)/den
        if abs(L) < 1e-9: continue
        res = boolean(res, prism(f, fn, L), "fuse" if L > 0 else "cut")
    return clean(res)

def split_body(shape, tool):
    sp = BOPAlgo_Splitter(); sp.AddArgument(shape); sp.AddTool(tool); sp.Perform()
    out = subshapes(sp.Shape(), TopAbs_SOLID)
    if len(out) < 2: raise RuntimeError("the splitting tool doesn't cut through the body")
    return out

def split_faces(shape, faces, tool):
    sp = BRepFeat_SplitShape(shape); n = 0
    for f in faces:
        sec = BRepAlgoAPI_Section(f, tool); sec.Build()
        for e in subshapes(sec.Shape(), TopAbs_EDGE): sp.Add(e, f); n += 1
    if not n: raise RuntimeError("the tool doesn't cross those faces")
    sp.Build()
    if not sp.IsDone(): raise RuntimeError("the face split failed")
    return sp.Shape()

def defeature(shape, faces):
    mk = BRepAlgoAPI_Defeaturing(); mk.SetShape(shape)
    for f in faces: mk.AddFaceToRemove(f)
    mk.SetRunParallel(True); mk.Build()
    if not mk.IsDone(): raise RuntimeError("those faces can't be removed and healed")
    return mk.Shape()

def thicken_faces(faces, thick, sym):
    out = []
    for f in faces:
        src = f
        if sym:
            c, n = normal_at(f, centroid(f)); src = translated(f, n*(-thick/2))
        mk = BRepOffsetAPI_MakeThickSolid(); mk.MakeThickSolidBySimple(src, thick); mk.Build()
        if not mk.IsDone(): raise RuntimeError("couldn't thicken that face")
        out.append(outward(mk.Shape()))
    return fuse_all(out) if out else None

def boundary_cells(tools):
    mv = BOPAlgo_MakerVolume()
    for t in tools: mv.AddArgument(t)
    mv.SetIntersect(True); mv.Perform()
    cells = subshapes(mv.Shape(), TopAbs_SOLID)
    if not cells: raise RuntimeError("those tools don't enclose any volume")
    return cells

def tangent_chain(body, ei, ang_tol=1.0):
    """Indices of edges tangent-continuous with edge ei (Fusion's 'tangent chain')."""
    def ends(e):
        c = BRepAdaptor_Curve(e); out = []
        for u, sgn in ((c.FirstParameter(), -1), (c.LastParameter(), 1)):
            p, d = gp_Pnt(), gp_Vec(); c.D1(u, p, d); out.append((vec(p), vec(d).normalize()*sgn))
        return out
    data = [ends(e) for e in body.eds]; chain = {ei}; todo = [ei]; ct = math.cos(math.radians(ang_tol))
    while todo:
        i = todo.pop()
        for (p, t) in data[i]:
            for j, other in enumerate(data):
                if j in chain: continue
                for (q, u) in other:
                    if (p - q).Length < 1e-5 and t.dot(u) < -ct: chain.add(j); todo.append(j)
    return sorted(chain)

# ---------------------------------------------------------------------------------------------------------------
#  Parametric sketches: shared points + curves that index them + constraints / dimensions, solved numerically
# ---------------------------------------------------------------------------------------------------------------
from OCP.gp import gp_Pln, gp_Ax2d, gp_Ax22d, gp_Elips2d, gp_Vec2d
from OCP.Geom2d import Geom2d_Circle, Geom2d_Ellipse, Geom2d_TrimmedCurve, Geom2d_BSplineCurve, Geom2d_BezierCurve
from OCP.Geom2dAPI import Geom2dAPI_InterCurveCurve, Geom2dAPI_Interpolate, Geom2dAPI_ProjectPointOnCurve
from OCP.Geom2dConvert import Geom2dConvert
from OCP.Geom2dAdaptor import Geom2dAdaptor_Curve
from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
from OCP.GeomAPI import GeomAPI
from OCP.GeomAbs import GeomAbs_Ellipse, GeomAbs_BSplineCurve, GeomAbs_BezierCurve
from OCP.collections import HArray1_gp_Pnt2d, Array1_gp_Vec2d, HArray1_bool, Array1_gp_Pnt2d, Array1_double, Array1_int

# ---- expressions: "width/2 + 3 mm", "1.5 in", "30 deg" ----
UNIT_LEN = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "inch": 25.4, "ft": 304.8, "um": 0.001}
UNIT_ANG = {"deg": 1.0, "rad": 180/math.pi}
_EXPR_UNIT_RE = re.compile(r'(\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+)\s*(mm|cm|um|m|inch|in|ft|deg|rad|°|")(?![A-Za-z_0-9])')
_FUNCS = {"sin": lambda a: math.sin(math.radians(a)), "cos": lambda a: math.cos(math.radians(a)), "tan": lambda a: math.tan(math.radians(a)),
          "asin": lambda v: math.degrees(math.asin(v)), "acos": lambda v: math.degrees(math.acos(v)), "atan": lambda v: math.degrees(math.atan(v)),
          "atan2": lambda y, x: math.degrees(math.atan2(y, x)), "sqrt": math.sqrt, "abs": abs, "min": min, "max": max,
          "round": round, "floor": math.floor, "ceil": math.ceil, "hypot": math.hypot}

class ExprError(Exception): pass

def expr_names(text):
    try:
        import ast
        t = _EXPR_UNIT_RE.sub(lambda m: "0", str(text))
        return {n.id for n in ast.walk(ast.parse(t, mode="eval")) if isinstance(n, ast.Name)} - set(_FUNCS) - {"pi", "e"}
    except Exception: return set()

def eval_expr(text, env=None, kind="len"):
    """Evaluate a dimension / parameter expression. Lengths come back in mm, angles in degrees.
    A bare number is in the display unit (mm or inches); names refer to parameters and other dimensions."""
    import ast
    env = env or {}; text = str(text).strip()
    if not text: raise ExprError("empty value")
    bare = re.fullmatch(r'[-+]?(\d+(?:\.\d*)?|\.\d+)([eE][-+]?\d+)?', text)
    if bare:
        v = float(text); return v*unit_k() if kind == "len" else v
    def unit(m):
        u = m.group(2); u = {"°": "deg", '"': "in"}.get(u, u)
        f = UNIT_LEN.get(u) if u in UNIT_LEN else UNIT_ANG.get(u)
        return f"({m.group(1)}*{f!r})"
    src = _EXPR_UNIT_RE.sub(unit, text)
    try: tree = ast.parse(src, mode="eval")
    except SyntaxError: raise ExprError(f"can't read '{text}'")
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return float(n.value)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right); op = type(n.op)
            if op is ast.Add: return a + b
            if op is ast.Sub: return a - b
            if op is ast.Mult: return a*b
            if op is ast.Div:
                if b == 0: raise ExprError("division by zero")
                return a/b
            if op is ast.Pow: return a**b
            if op is ast.Mod: return a % b
            raise ExprError("unsupported operator")
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand); return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Name):
            if n.id in env: return float(env[n.id])
            if n.id == "pi": return math.pi
            if n.id == "e": return math.e
            raise ExprError(f"unknown name '{n.id}'")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return float(_FUNCS[n.func.id](*[ev(a) for a in n.args]))
        raise ExprError(f"can't evaluate '{text}'")
    try: return ev(tree)
    except ExprError: raise
    except Exception as ex: raise ExprError(f"{text}: {ex}")

# ---- the geometry container ----
class Geo:
    """Points [[x, y]], curves {k: kind, p: [point indices], ...}, constraints / dimensions {t: type, e: [refs], ...}.
    A ref is ["p", point index] or ["c", curve index]. Copy-on-write: edits work on .copy() so history snapshots stay valid."""
    def __init__(s, P=None, Cs=None, K=None):
        s.P = P if P is not None else []; s.C = Cs if Cs is not None else []; s.K = K if K is not None else []
        s.ver = 0; s._c = {}; s.status = None
    def copy(s): return Geo([list(p) for p in s.P], copy.deepcopy(s.C), copy.deepcopy(s.K))
    def touch(s): s.ver += 1; s._c = {}; s.status = None
    def to_json(s): return {"pts": [list(p) for p in s.P], "crv": copy.deepcopy(s.C), "con": copy.deepcopy(s.K)}
    @staticmethod
    def from_json(d):
        if isinstance(d, Geo): return d.copy()
        g = Geo([list(map(float, p)) for p in d.get("pts", [])], copy.deepcopy(d.get("crv", [])), copy.deepcopy(d.get("con", [])))
        g.normalize(); return g
    def normalize(s):
        """Saved files turn tuples into lists and back: keep every reference a [kind, index] list."""
        for k in s.K:
            k["e"] = [[r[0], int(r[1])] for r in k["e"]]
            if k.get("j"): k["j"] = [int(i) for i in k["j"]]
            if k.get("pos") is not None: k["pos"] = [float(x) for x in k["pos"]]
        for c in s.C:
            c["p"] = [int(i) for i in c["p"]]
            for key in ("q",):
                if key in c: c[key] = [list(map(float, q)) for q in c[key]]
            if "faces" in c: c["faces"] = [tuple(f) for f in c["faces"]]
        return s
    def add_pt(s, x, y=None):
        if y is None: x, y = (x.x, x.y) if isinstance(x, V) else x
        s.P.append([float(x), float(y)]); return len(s.P) - 1
    def add(s, kind, pts, **kw):
        c = dict(k=kind, p=list(pts)); c.update(kw); s.C.append(c); return len(s.C) - 1
    def con(s, t, *refs, **kw):
        k = dict(t=t, e=[list(r) for r in refs]); k.update(kw); s.K.append(k); return len(s.K) - 1
    def PT(s, i): return s.P[i]
    def v(s, i): return V(s.P[i][0], s.P[i][1], 0)
    def empty(s): return not s.C and not s.P

def P_(i): return ["p", i]
def C_(i): return ["c", i]

# ---- curve geometry (sketch-local 2D) ----
def crv_ends(c):
    """Start / end point indices of an open curve (None for closed ones)."""
    k, p = c["k"], c["p"]
    if k == "line": return p[0], p[1]
    if k == "arc": return p[1], p[2]
    if k == "earc": return p[2], p[3]
    if k in ("spline", "cspline"): return (None, None) if c.get("closed") else (p[0], p[-1])
    if k == "conic": return p[0], p[2]
    return None, None

def crv_center(c):
    return c["p"][0] if c["k"] in ("circle", "arc", "ellipse", "earc") else None

def crv_points(c):
    """Every point index a curve uses (including spline tangent handles)."""
    out = list(c["p"])
    if c["k"] == "spline": out += [h for h in (c.get("h") or []) if h is not None]
    return out

def is_round(c): return c["k"] in ("circle", "arc")
def is_line(c): return c["k"] == "line"

def ell_frame(c, PT, SC):
    (cx, cy), (mx, my) = PT(c["p"][0]), PT(c["p"][1])
    r1 = math.hypot(mx - cx, my - cy); r2 = abs(SC("r2"))
    if r1 < 1e-12: return None
    return cx, cy, (mx - cx)/r1, (my - cy)/r1, r1, max(r2, 1e-9)

def ell_param(fr, x, y):
    cx, cy, ux, uy, r1, r2 = fr; dx, dy = x - cx, y - cy
    return math.atan2((-uy*dx + ux*dy)/r2, (ux*dx + uy*dy)/r1)

def ell_point(fr, t):
    cx, cy, ux, uy, r1, r2 = fr; a, b = r1*math.cos(t), r2*math.sin(t)
    return cx + ux*a - uy*b, cy + uy*a + ux*b

def ell_geom(fr):
    cx, cy, ux, uy, r1, r2 = fr
    if r1 >= r2: return Geom2d_Ellipse(gp_Elips2d(gp_Ax22d(gp_Pnt2d(cx, cy), gp_Dir2d(ux, uy), gp_Dir2d(-uy, ux)), r1, r2)), 0.0
    return Geom2d_Ellipse(gp_Elips2d(gp_Ax22d(gp_Pnt2d(cx, cy), gp_Dir2d(-uy, ux), gp_Dir2d(-ux, -uy)), r2, r1)), -math.pi/2

def arc_angles(c, PT):
    (cx, cy), s_, e_ = PT(c["p"][0]), PT(c["p"][1]), PT(c["p"][2])
    a0 = math.atan2(s_[1] - cy, s_[0] - cx); a1 = math.atan2(e_[1] - cy, e_[0] - cx)
    while a1 <= a0 + 1e-9: a1 += math.tau
    return a0, a1, math.hypot(s_[0] - cx, s_[1] - cy)

def _arr_pnt2d(P, h=False):
    a = HArray1_gp_Pnt2d(1, len(P)) if h else Array1_gp_Pnt2d(1, len(P))
    for i, (x, y) in enumerate(P): a.SetValue(i + 1, gp_Pnt2d(float(x), float(y)))
    return a

def _arr(vals, cls):
    a = cls(1, len(vals))
    for i, v in enumerate(vals): a.SetValue(i + 1, v)
    return a

def bspline2d(P, deg=3, knots=None, mults=None, weights=None, periodic=False):
    n = len(P); deg = max(1, min(deg, n - 1))
    if not knots:
        nk = n - deg + 1; knots = [float(i) for i in range(nk)]; mults = [deg + 1] + [1]*(nk - 2) + [deg + 1]
    poles = _arr_pnt2d(P); kn = _arr([float(k) for k in knots], Array1_double); mu = _arr([int(m) for m in mults], Array1_int)
    if weights: return Geom2d_BSplineCurve(poles, _arr([float(w) for w in weights], Array1_double), kn, mu, deg, periodic)
    return Geom2d_BSplineCurve(poles, kn, mu, deg, periodic)

def crv_g2d(c, PT, SC=None):
    """The Geom2d curve(s) of a sketch curve.  PT(i) -> (x, y) of point i;  SC(key) -> a scalar of this curve."""
    SC = SC or (lambda key: c[key])
    k, p = c["k"], c["p"]
    try:
        if k == "line":
            a, b = PT(p[0]), PT(p[1]); L = math.hypot(b[0] - a[0], b[1] - a[1])
            return [Geom2d_TrimmedCurve(Geom2d_Line(gp_Pnt2d(*a), gp_Dir2d(b[0] - a[0], b[1] - a[1])), 0, L)] if L > 1e-12 else []
        if k == "circle":
            (cx, cy), r = PT(p[0]), abs(SC("r"))
            return [Geom2d_Circle(gp_Ax2d(gp_Pnt2d(cx, cy), gp_Dir2d(1, 0)), r)] if r > 1e-12 else []
        if k == "arc":
            a0, a1, r = arc_angles(c, PT); cx, cy = PT(p[0])
            return [Geom2d_TrimmedCurve(Geom2d_Circle(gp_Ax2d(gp_Pnt2d(cx, cy), gp_Dir2d(1, 0)), r), a0, a1)] if r > 1e-12 else []
        if k in ("ellipse", "earc"):
            fr = ell_frame(c, PT, SC)
            if not fr: return []
            g, sh = ell_geom(fr)
            if k == "ellipse": return [g]
            t0 = ell_param(fr, *PT(p[2])); t1 = ell_param(fr, *PT(p[3]))
            while t1 <= t0 + 1e-9: t1 += math.tau
            return [Geom2d_TrimmedCurve(g, t0 + sh, t1 + sh)]
        if k == "spline":
            P = [PT(i) for i in p]
            if len(P) < 2: return []
            it = Geom2dAPI_Interpolate(_arr_pnt2d(P, True), bool(c.get("closed")), 1e-7)
            hs = c.get("h") or []
            if any(h is not None for h in hs):
                hl = c.get("hl", 1.0) or 1.0; tv = Array1_gp_Vec2d(1, len(P)); fl = HArray1_bool(1, len(P))
                for i in range(len(P)):
                    h = hs[i] if i < len(hs) else None
                    if h is None: tv.SetValue(i + 1, gp_Vec2d(1, 0)); fl.SetValue(i + 1, False)
                    else:
                        hx, hy = PT(h); tv.SetValue(i + 1, gp_Vec2d((hx - P[i][0])/hl, (hy - P[i][1])/hl)); fl.SetValue(i + 1, True)
                it.Load(tv, fl, False)
            it.Perform()
            return [it.Curve()] if it.IsDone() else []
        if k == "cspline":
            P = [PT(i) for i in p]
            if len(P) < 2: return []
            return [bspline2d(P, c.get("deg", 3), c.get("kn"), c.get("mu"), c.get("w"))]
        if k == "conic":
            rho = min(max(float(c.get("rho", 0.5)), 0.01), 0.99); w = rho/(1 - rho)
            return [Geom2d_BezierCurve(_arr_pnt2d([PT(i) for i in p]), _arr([1.0, w, 1.0], Array1_double))]
        if k == "text": return text_curves(c, PT)
    except Exception:
        return []
    return []

def g2d_sample(g, fine=1.0):
    a = Geom2dAdaptor_Curve(g); u0, u1 = a.FirstParameter(), a.LastParameter(); t = a.GetType()
    if t == GeomAbs_Line: us = (u0, u1)
    elif t in (GeomAbs_Circle, GeomAbs_Ellipse): us = np.linspace(u0, u1, max(6, int(abs(u1 - u0)/math.tau*96*fine) + 2))
    elif t == GeomAbs_BSplineCurve: us = np.linspace(u0, u1, int(min(600, 12*max(a.NbKnots(), 2)*a.Degree()*fine)) + 2)
    else: us = np.linspace(u0, u1, int(48*fine))
    out = np.empty((len(us), 2))
    for i, u in enumerate(us): q = a.Value(float(u)); out[i] = q.X(), q.Y()
    return out

def geo_curve_pts(geo, ci):
    """Display polylines (list of Nx2 arrays) of curve ci, cached per geometry version."""
    key = ("pl", ci)
    if key not in geo._c: geo._c[key] = [g2d_sample(g) for g in geo_g2d(geo, ci)]
    return geo._c[key]

def geo_g2d(geo, ci):
    key = ("g", ci)
    if key not in geo._c:
        c = geo.C[ci]
        if c["k"] == "text" and c.get("path") is not None and 0 <= c["path"] < len(geo.C) and c["path"] != ci:
            pp = geo_curve_pts(geo, c["path"]); geo._c[key] = text_curves(c, geo.PT, pp[0] if pp else None)
        else: geo._c[key] = crv_g2d(c, geo.PT)
    return geo._c[key]

def geo_polys(geo, ci):
    """Display polylines of any curve: (N, 2) for plane curves, (N, 3) for 3D sketch curves."""
    return geo_curve_pts3(geo, ci) if geo.C[ci]["k"] in ("l3", "s3", "inc3", "proj3") else geo_curve_pts(geo, ci)

def sk_pln(plane): return gp_Pln(gp_Ax3(pnt(plane.o), gdir(plane.n), gdir(plane.u)))
def g2d_edge(g, pln): return BRepBuilderAPI_MakeEdge(GeomAPI.To3d_s(g, pln)).Edge()

def crv_edges(geo, ci, plane):
    if geo.C[ci]["k"] in ("l3", "s3", "inc3", "proj3"): return crv3_edges(geo, ci, plane)
    pln = sk_pln(plane); out = []
    for g in geo_g2d(geo, ci):
        try: out.append(g2d_edge(g, pln))
        except Exception: pass
    return out

# ---- text (glyph outlines from Qt as lines + cubic Beziers) ----
_TEXT_CACHE = {}
def text_paths(c):
    """Glyph outlines as subpaths of ('L', (x0, y0, x1, y1)) / ('C', (4 points)) in text-local units (baseline at y = 0)."""
    key = (c.get("text", ""), c.get("font", "Sans Serif"), round(float(c.get("size", 10)), 6), bool(c.get("bold")), bool(c.get("italic")))
    if key in _TEXT_CACHE: return _TEXT_CACHE[key]
    f = G.QFont(key[1]); f.setPixelSize(1000); f.setBold(key[3]); f.setItalic(key[4])
    fm = G.QFontMetricsF(f); k = key[2]/max(fm.capHeight(), 1e-6)       # size = height of capital letters
    glyphs = []; x = 0.0
    for ch in key[0]:
        path = G.QPainterPath(); path.addText(C.QPointF(0, 0), f, ch)
        segs, cur, start, i, n = [], None, None, 0, path.elementCount()
        while i < n:
            e = path.elementAt(i)
            if e.type == G.QPainterPath.MoveToElement: cur = start = (e.x*k, -e.y*k); i += 1
            elif e.type == G.QPainterPath.LineToElement:
                q = (e.x*k, -e.y*k)
                if math.hypot(q[0] - cur[0], q[1] - cur[1]) > 1e-9: segs.append(("L", (cur, q)))
                cur = q; i += 1
            else:
                c1, c2, q = path.elementAt(i), path.elementAt(i + 1), path.elementAt(i + 2)
                pts = (cur, (c1.x*k, -c1.y*k), (c2.x*k, -c2.y*k), (q.x*k, -q.y*k)); segs.append(("C", pts)); cur = pts[3]; i += 3
        adv = fm.horizontalAdvance(ch)*k
        glyphs.append((x, adv, segs)); x += adv
    _TEXT_CACHE[key] = (glyphs, x); return _TEXT_CACHE[key]

def text_layout(c, PT, path_pts=None):
    """Placement of each glyph: list of (segments, transform(x, y) -> (X, Y))."""
    glyphs, width = text_paths(c)
    ax, ay = PT(c["p"][0]); ang = math.radians(float(c.get("ang", 0.0)))
    out = []
    if path_pts is not None and len(path_pts) > 1:                      # text along a path
        P = np.asarray(path_pts, float)
        if (P[-1, 0] < P[0, 0]) != bool(c.get("flip")): P = P[::-1]     # read left to right unless flipped
        d = np.hypot(*np.diff(P, axis=0).T); s = np.concatenate([[0], np.cumsum(d)]); L = s[-1]
        start = float(c.get("pos", 0.0))*max(L - width, 0) if c.get("align", "start") == "start" else max((L - width)/2, 0)
        for gx, adv, segs in glyphs:
            m = start + gx + adv/2; j = int(np.clip(np.searchsorted(s, m) - 1, 0, len(P) - 2)); u = (m - s[j])/max(d[j], 1e-12)
            px, py = P[j] + (P[j + 1] - P[j])*u; tx, ty = (P[j + 1] - P[j])/max(d[j], 1e-12)
            cx = adv/2
            out.append((segs, lambda x, y, px=px, py=py, tx=tx, ty=ty, cx=cx: (px + tx*(x - cx) - ty*y, py + ty*(x - cx) + tx*y)))
        return out
    ca, sa = math.cos(ang), math.sin(ang)
    for gx, adv, segs in glyphs: out.append((segs, lambda x, y, gx=gx: (ax + ca*(x + gx) - sa*y, ay + sa*(x + gx) + ca*y)))
    return out

def text_curves(c, PT, path_pts=None):
    if path_pts is None: path_pts = c.get("_path")
    out = []
    for segs, T in text_layout(c, PT, path_pts):
        for kind, pts in segs:
            Q = [T(*q) for q in pts]
            try:
                if kind == "L":
                    (x0, y0), (x1, y1) = Q; L = math.hypot(x1 - x0, y1 - y0)
                    if L > 1e-9: out.append(Geom2d_TrimmedCurve(Geom2d_Line(gp_Pnt2d(x0, y0), gp_Dir2d(x1 - x0, y1 - y0)), 0, L))
                else: out.append(Geom2d_BezierCurve(_arr_pnt2d(Q)))
            except Exception: pass
    return out

def text_width(c): return text_paths(c)[1]

# ---- legacy (v0.5) shapes ----
def geo_add_legacy(geo, kind, pts, constrain=True):
    """Add an old-style (kind, points) entity: rect / poly / line / circle."""
    if kind == "circle":
        a, b = pts[0], pts[1]; c = geo.add_pt(a.x, a.y); return [geo.add("circle", [c], r=(b - a).Length)]
    if kind == "rect":
        a, b = pts[0], pts[1]; L = [a, V(b.x, a.y, 0), b, V(a.x, b.y, 0)]
    else: L = list(pts)
    ids = [geo.add_pt(q.x, q.y) for q in L]; out = []
    closed = kind in ("rect", "poly") and len(ids) > 2
    for j in range(len(ids) - (0 if closed else 1)):
        out.append(geo.add("line", [ids[j], ids[(j + 1) % len(ids)]]))
    if kind == "rect" and constrain:
        geo.con("hor", C_(out[0])); geo.con("hor", C_(out[2])); geo.con("ver", C_(out[1])); geo.con("ver", C_(out[3]))
    return out

def legacy_geo(entities):
    g = Geo()
    for kind, pts in entities: geo_add_legacy(g, kind, pts)
    return g
# ---- constraint solver: residual equations + damped minimum-norm Newton steps ----
CON_NAMES = {"coin": "Coincident", "mid": "Midpoint", "hor": "Horizontal", "ver": "Vertical", "par": "Parallel", "perp": "Perpendicular",
             "col": "Collinear", "conc": "Concentric", "tan": "Tangent", "smooth": "Curvature (G2)", "eq": "Equal", "sym": "Symmetry",
             "fix": "Fix", "dist": "Distance", "hdist": "Horizontal distance", "vdist": "Vertical distance", "pldist": "Point-line distance",
             "lldist": "Line-line distance", "ang": "Angle", "rad": "Radius", "dia": "Diameter", "erad": "Ellipse radius"}
DIM_TYPES = {"dist", "hdist", "vdist", "pldist", "lldist", "ang", "rad", "dia", "erad"}

class SkSys:
    """Unknowns of a sketch: x, y of every point, then one radius per circle and minor radius per ellipse."""
    def __init__(s, geo):
        s.geo = geo; nP = len(geo.P); s.scal = {}; n = 2*nP
        for ci, c in enumerate(geo.C):
            if c["k"] == "circle": s.scal[(ci, "r")] = n; n += 1
            elif c["k"] in ("ellipse", "earc"): s.scal[(ci, "r2")] = n; n += 1
        s.n = n; x = np.zeros(n)
        if nP: x[:2*nP] = np.asarray(geo.P, float).reshape(-1)
        for (ci, key), j in s.scal.items(): x[j] = float(geo.C[ci].get(key, 1.0))
        s.x = x; s.fixed = set(); s.rows = []          # rows: (var list, f(x) -> list, constraint index)
        s.errors = {}
        s._scale = float(np.ptp(x[:2*nP:2]) + np.ptp(x[1:2*nP:2]) + 1.0) if nP else 1.0

    # --- accessors over an unknown vector ---
    def pv(s, i): return [2*i, 2*i + 1]
    def cvars(s, ci):
        c = s.geo.C[ci]; out = []
        for i in crv_points(c): out += s.pv(i)
        for key in ("r", "r2"):
            if (ci, key) in s.scal: out.append(s.scal[(ci, key)])
        return out
    def PTx(s, x): return lambda i: (x[2*i], x[2*i + 1])
    def SCx(s, x, ci): return lambda key: x[s.scal[(ci, key)]] if (ci, key) in s.scal else s.geo.C[ci][key]

    def center(s, x, ci):
        c = s.geo.C[ci]; i = c["p"][0]; return x[2*i], x[2*i + 1]
    def radius(s, x, ci):
        c = s.geo.C[ci]
        if c["k"] == "circle": return x[s.scal[(ci, "r")]]
        cx, cy = s.center(x, ci); i = c["p"][1]; return math.hypot(x[2*i] - cx, x[2*i + 1] - cy)
    def line(s, x, ci):
        a, b = s.geo.C[ci]["p"]; return x[2*a], x[2*a + 1], x[2*b], x[2*b + 1]
    def ell(s, x, ci): return ell_frame(s.geo.C[ci], s.PTx(x), s.SCx(x, ci))

    def sdist_line(s, x, ci, px, py):
        ax, ay, bx, by = s.line(x, ci); dx, dy = bx - ax, by - ay; L = math.hypot(dx, dy) or 1e-12
        return (dx*(py - ay) - dy*(px - ax))/L
    def on_curve(s, x, ci, px, py):
        """Signed distance-like residual of a point on (the extension of) a curve."""
        c = s.geo.C[ci]; k = c["k"]
        if k == "line": return s.sdist_line(x, ci, px, py)
        if k in ("circle", "arc"):
            cx, cy = s.center(x, ci); return math.hypot(px - cx, py - cy) - s.radius(x, ci)
        if k in ("ellipse", "earc"):
            fr = s.ell(x, ci)
            if not fr: return 0.0
            cx, cy, ux, uy, r1, r2 = fr; dx, dy = px - cx, py - cy
            X, Y = ux*dx + uy*dy, -uy*dx + ux*dy; return (math.hypot(X/r1, Y/r2) - 1)*math.sqrt(r1*r2)
        gs = crv_g2d(c, s.PTx(x), s.SCx(x, ci))
        if not gs: return 0.0
        g = gs[0]; pr = Geom2dAPI_ProjectPointOnCurve(gp_Pnt2d(px, py), g)
        if pr.NbPoints() == 0:
            a = g.Value(g.FirstParameter()); b = g.Value(g.LastParameter())
            return min(math.hypot(px - a.X(), py - a.Y()), math.hypot(px - b.X(), py - b.Y()))
        u = pr.LowerDistanceParameter(); P = gp_Pnt2d(); D = gp_Vec2d(); g.D1(u, P, D)
        sg = 1.0 if D.X()*(py - P.Y()) - D.Y()*(px - P.X()) >= 0 else -1.0
        return sg*pr.LowerDistance()
    def end_d(s, x, ci, at_start):
        """First / second derivative of curve ci at its start or end."""
        c = s.geo.C[ci]; gs = crv_g2d(c, s.PTx(x), s.SCx(x, ci))
        if not gs: return (1.0, 0.0), (0.0, 0.0)
        g = gs[0]; u = g.FirstParameter() if at_start else g.LastParameter()
        P = gp_Pnt2d(); d1 = gp_Vec2d(); d2 = gp_Vec2d(); g.D2(u, P, d1, d2)
        return (d1.X(), d1.Y()), (d2.X(), d2.Y())
    def meas(s, x, k):
        """Current value of a dimension (mm or degrees)."""
        t, e = k["t"], k["e"]
        P = lambda r: (x[2*r[1]], x[2*r[1] + 1]) if r[0] == "p" else s.center(x, r[1])
        if t == "dist":
            (ax, ay), (bx, by) = P(e[0]), P(e[1]); return math.hypot(bx - ax, by - ay)
        if t in ("hdist", "vdist"):
            (ax, ay), (bx, by) = P(e[0]), P(e[1]); return ((bx - ax) if t == "hdist" else (by - ay))*k.get("sg", 1)
        if t == "pldist":
            px, py = P(e[0]); return s.sdist_line(x, e[1][1], px, py)*k.get("sg", 1)
        if t == "lldist":
            ax, ay, bx, by = s.line(x, e[1][1]); return s.sdist_line(x, e[0][1], (ax + bx)/2, (ay + by)/2)*k.get("sg", 1)
        if t == "ang":
            a1x, a1y, b1x, b1y = s.line(x, e[0][1]); a2x, a2y, b2x, b2y = s.line(x, e[1][1])
            f1, f2 = k.get("f1", 1), k.get("f2", 1)
            u = ((b1x - a1x)*f1, (b1y - a1y)*f1); v = ((b2x - a2x)*f2, (b2y - a2y)*f2)
            return math.degrees(math.atan2(u[0]*v[1] - u[1]*v[0], u[0]*v[0] + u[1]*v[1]))*k.get("sg", 1)
        if t in ("rad", "dia"):
            r = s.radius(x, e[0][1]); return r if t == "rad" else 2*r
        if t == "erad":
            fr = s.ell(x, e[0][1]); return 0.0 if not fr else (fr[4] if k.get("ax", 1) == 1 else fr[5])
        raise ValueError(t)

    # --- equations ---
    def add_rows(s, vars_, f, ki): s.rows.append((vars_, f, ki))

    def build(s, skip=(), only=None):
        geo = s.geo; C_ = geo.C
        for ci, c in enumerate(C_):                                   # curve-internal relations
            if c["k"] == "arc":
                cc, a, b = c["p"]
                s.add_rows(s.pv(cc) + s.pv(a) + s.pv(b), lambda x, cc=cc, a=a, b=b: [math.hypot(x[2*b] - x[2*cc], x[2*b + 1] - x[2*cc + 1])
                                                                                       - math.hypot(x[2*a] - x[2*cc], x[2*a + 1] - x[2*cc + 1])], None)
            elif c["k"] == "earc":
                vs = s.cvars(ci)
                s.add_rows(vs, lambda x, ci=ci, c=c: [s.on_curve(x, ci, x[2*c["p"][2]], x[2*c["p"][2] + 1]),
                                                      s.on_curve(x, ci, x[2*c["p"][3]], x[2*c["p"][3] + 1])], None)
            if c.get("src") is not None or c.get("lock"):                  # projected / fixed geometry
                s.fixed |= set(s.cvars(ci))
        for ki, k in enumerate(geo.K):
            if ki in skip or k.get("off"): continue
            if only is not None and ki not in only: continue
            try: err = s.add_con(ki, k)
            except Exception as ex: err = str(ex)
            if err: s.errors[ki] = err

    def add_con(s, ki, k):
        t, e = k["t"], k["e"]; geo = s.geo
        kinds = [r[0] for r in e]
        cv = lambda r: geo.C[r[1]]
        R = lambda f, vs: s.add_rows(vs, f, ki)
        if t == "fix":
            for r in e: s.fixed |= set(s.pv(r[1]) if r[0] == "p" else s.cvars(r[1]))
            return None
        if t in DIM_TYPES:
            if k.get("drv"): return None
            vs = []
            for r in e: vs += s.pv(r[1]) if r[0] == "p" else (s.pv(crv_center(cv(r))) if t in ("dist", "hdist", "vdist") else s.cvars(r[1]))
            val = float(k.get("v", 0.0))
            if t == "ang":
                R(lambda x: [math.radians(((s.meas(x, k) - val) + 180) % 360 - 180)], vs)
            else: R(lambda x: [s.meas(x, k) - val], vs)
            return None
        if t == "coin":
            if kinds == ["p", "p"]:
                i, j = e[0][1], e[1][1]; R(lambda x: [x[2*i] - x[2*j], x[2*i + 1] - x[2*j + 1]], s.pv(i) + s.pv(j)); return None
            if kinds == ["p", "c"]:
                i, ci = e[0][1], e[1][1]
                if cv(e[1])["k"] in ("point", "text"): return "can't use that curve"
                R(lambda x: [s.on_curve(x, ci, x[2*i], x[2*i + 1])], s.pv(i) + s.cvars(ci)); return None
            return "pick a point and a point or curve"
        if t == "mid":
            if kinds != ["p", "c"]: return "pick a point and a line or arc"
            i, ci = e[0][1], e[1][1]; c = cv(e[1])
            if c["k"] == "line":
                a, b = c["p"]; R(lambda x: [x[2*i] - (x[2*a] + x[2*b])/2, x[2*i + 1] - (x[2*a + 1] + x[2*b + 1])/2], s.pv(i) + s.cvars(ci)); return None
            if c["k"] == "arc":
                def f(x):
                    a0, a1, r = arc_angles(c, s.PTx(x)); cx, cy = s.center(x, ci); m = (a0 + a1)/2
                    return [x[2*i] - (cx + r*math.cos(m)), x[2*i + 1] - (cy + r*math.sin(m))]
                R(f, s.pv(i) + s.cvars(ci)); return None
            return "midpoint needs a line or an arc"
        if t in ("hor", "ver"):
            j = 1 if t == "hor" else 0
            if kinds == ["c"] and cv(e[0])["k"] == "line": a, b = cv(e[0])["p"]
            elif kinds == ["p", "p"]: a, b = e[0][1], e[1][1]
            elif kinds == ["c"] and cv(e[0])["k"] in ("ellipse", "earc"): a, b = cv(e[0])["p"][:2]
            else: return "pick a line or two points"
            R(lambda x: [x[2*b + j] - x[2*a + j]], s.pv(a) + s.pv(b)); return None
        if t in ("par", "perp", "col"):
            if kinds != ["c", "c"] or not all(cv(r)["k"] == "line" for r in e):
                if t != "col" and kinds == ["c", "c"] and all(cv(r)["k"] in ("line", "ellipse", "earc") for r in e): pass
                else: return "pick two lines"
            c1, c2 = e[0][1], e[1][1]
            def dirs(x):
                out = []
                for ci in (c1, c2):
                    c = geo.C[ci]; a, b = c["p"][:2] if c["k"] != "line" else c["p"]
                    if c["k"] == "line": out.append((x[2*b] - x[2*a], x[2*b + 1] - x[2*a + 1]))
                    else: cx, cy = s.center(x, ci); out.append((x[2*c["p"][1]] - cx, x[2*c["p"][1] + 1] - cy))
                return out
            vs = s.cvars(c1) + s.cvars(c2)
            if t == "par":
                def f(x):
                    (ux, uy), (vx, vy) = dirs(x); return [(ux*vy - uy*vx)/((math.hypot(ux, uy)*math.hypot(vx, vy)) or 1e-12)]
            elif t == "perp":
                def f(x):
                    (ux, uy), (vx, vy) = dirs(x); return [(ux*vx + uy*vy)/((math.hypot(ux, uy)*math.hypot(vx, vy)) or 1e-12)]
            else:
                a2, b2 = geo.C[c2]["p"]
                def f(x): return [s.sdist_line(x, c1, x[2*a2], x[2*a2 + 1]), s.sdist_line(x, c1, x[2*b2], x[2*b2 + 1])]
            R(f, vs); return None
        if t == "conc":
            pts = []
            for r in e:
                if r[0] == "p": pts.append(r[1])
                elif crv_center(cv(r)) is not None: pts.append(crv_center(cv(r)))
                else: return "pick circles, arcs or ellipses"
            i, j = pts[0], pts[1]; R(lambda x: [x[2*i] - x[2*j], x[2*i + 1] - x[2*j + 1]], s.pv(i) + s.pv(j)); return None
        if t == "eq":
            if kinds != ["c", "c"]: return "pick two curves"
            c1, c2 = cv(e[0]), cv(e[1]); i1, i2 = e[0][1], e[1][1]; vs = s.cvars(i1) + s.cvars(i2)
            if c1["k"] == "line" and c2["k"] == "line":
                def L(x, ci): ax, ay, bx, by = s.line(x, ci); return math.hypot(bx - ax, by - ay)
                R(lambda x: [L(x, i2) - L(x, i1)], vs); return None
            if is_round(c1) and is_round(c2): R(lambda x: [s.radius(x, i2) - s.radius(x, i1)], vs); return None
            if c1["k"] in ("ellipse", "earc") and c2["k"] in ("ellipse", "earc"):
                def f(x):
                    a, b = s.ell(x, i1), s.ell(x, i2)
                    return [b[4] - a[4], b[5] - a[5]] if a and b else [0, 0]
                R(f, vs); return None
            return "pick two lines, two circles / arcs or two ellipses"
        if t in ("tan", "smooth"):
            if kinds != ["c", "c"]: return "pick two curves"
            i1, i2 = e[0][1], e[1][1]; c1, c2 = cv(e[0]), cv(e[1]); vs = s.cvars(i1) + s.cvars(i2)
            j = k.get("j")
            if j:                                                     # curves meeting at an end point: tangent directions agree
                st1, st2 = j[0] == crv_ends(c1)[0], j[1] == crv_ends(c2)[0]
                def f(x):
                    (d1, k1), (d2, k2) = s.end_d(x, i1, st1), s.end_d(x, i2, st2)
                    n1, n2 = math.hypot(*d1) or 1e-12, math.hypot(*d2) or 1e-12
                    out = [(d1[0]*d2[1] - d1[1]*d2[0])/(n1*n2)]
                    if t == "smooth":
                        kk1 = (d1[0]*k1[1] - d1[1]*k1[0])/n1**3; kk2 = (d2[0]*k2[1] - d2[1]*k2[0])/n2**3
                        kin = kk1*(1 if not st1 else -1); kout = kk2*(1 if st2 else -1)
                        L = max(s.scale, 1e-6); out.append((kin - kout)*L)
                    return out
                R(f, vs); return None
            if t == "smooth": return "curvature continuity needs two curves that share an end point"
            sg = k.get("sg", 1)
            if c1["k"] == "line" and is_round(c2) or c2["k"] == "line" and is_round(c1):
                li, ri = (i1, i2) if c1["k"] == "line" else (i2, i1)
                R(lambda x: [s.sdist_line(x, li, *s.center(x, ri))*sg - s.radius(x, ri)], vs); return None
            if is_round(c1) and is_round(c2):
                io = k.get("io", 0)                                   # 0: outside each other, +1 / -1: one inside the other
                def f(x):
                    (ax, ay), (bx, by) = s.center(x, i1), s.center(x, i2); d = math.hypot(bx - ax, by - ay)
                    r1, r2 = s.radius(x, i1), s.radius(x, i2)
                    return [d - (r1 + r2)] if io == 0 else [d - (r1 - r2)*io]
                R(f, vs); return None
            if c1["k"] == "line" and c2["k"] in ("ellipse", "earc") or c2["k"] == "line" and c1["k"] in ("ellipse", "earc"):
                li, ei = (i1, i2) if c1["k"] == "line" else (i2, i1)
                def f(x):
                    fr = s.ell(x, ei)
                    if not fr: return [0.0]
                    cx, cy, ux, uy, r1, r2 = fr; ax, ay, bx, by = s.line(x, li); dx, dy = bx - ax, by - ay; L = math.hypot(dx, dy) or 1e-12
                    nx, ny = -dy/L, dx/L; nX, nY = ux*nx + uy*ny, -uy*nx + ux*ny
                    return [s.sdist_line(x, li, cx, cy)*sg - math.hypot(r1*nX, r2*nY)]
                R(f, vs); return None
            return "these curves need to share an end point to be made tangent"
        if t == "sym":
            if len(e) != 3 or e[2][0] != "c" or cv(e[2])["k"] != "line": return "pick two points or curves, then the symmetry line"
            li = e[2][1]
            def pair(i, j):
                def f(x):
                    ax, ay, bx, by = s.line(x, li); dx, dy = bx - ax, by - ay
                    mx, my = (x[2*i] + x[2*j])/2, (x[2*i + 1] + x[2*j + 1])/2
                    L = math.hypot(dx, dy) or 1e-12
                    return [s.sdist_line(x, li, mx, my), ((x[2*j] - x[2*i])*dx + (x[2*j + 1] - x[2*i + 1])*dy)/L]
                R(f, s.pv(i) + s.pv(j) + s.cvars(li))
            if kinds[:2] == ["p", "p"]: pair(e[0][1], e[1][1]); return None
            if kinds[:2] == ["c", "c"]:
                c1, c2 = cv(e[0]), cv(e[1])
                if c1["k"] == "line" and c2["k"] == "line":
                    a1, b1 = c1["p"]; a2, b2 = c2["p"]
                    if k.get("sw"): a2, b2 = b2, a2
                    pair(a1, a2); pair(b1, b2); return None
                if is_round(c1) and is_round(c2):
                    pair(crv_center(c1), crv_center(c2)); i1, i2 = e[0][1], e[1][1]
                    R(lambda x: [s.radius(x, i2) - s.radius(x, i1)], s.cvars(i1) + s.cvars(i2)); return None
                if c1["k"] == c2["k"] and len(c1["p"]) == len(c2["p"]):
                    for a, b in zip(c1["p"], c2["p"]): pair(a, b)
                    return None
            return "symmetry needs two points, two lines or two circles / arcs"
        return f"unknown constraint {t}"

    # --- numerics ---
    @property
    def scale(s): return s._scale

    def F(s, x, rows):
        out = []
        for vs, f, _ in rows: out += list(f(x))
        return np.array(out, float)

    def J(s, x, rows, col, m):
        J = np.zeros((m, len(col))); r = 0
        for vs, f, _ in rows:
            n = len(f(x))
            for v in set(vs):
                j = col.get(v)
                if j is None: continue
                h = 1e-7*(1.0 + abs(x[v])); x0 = x[v]
                x[v] = x0 + h; f1 = np.asarray(f(x), float); x[v] = x0 - h; f2 = np.asarray(f(x), float); x[v] = x0
                J[r:r + n, j] = (f1 - f2)/(2*h)
            r += n
        return J

    def components(s):
        """Groups of rows that share unknowns (solved separately: smaller and more robust)."""
        par = {}
        def find(a):
            while par.setdefault(a, a) != a: par[a] = par[par[a]]; a = par[a]
            return a
        for vs, f, _ in s.rows:
            fv = [v for v in vs if v not in s.fixed]
            for v in fv[1:]: par[find(v)] = find(fv[0])
            for v in fv[:1]: find(v)
        groups = {}
        for row in s.rows:
            fv = [v for v in row[0] if v not in s.fixed]
            key = find(fv[0]) if fv else ("fixed", id(row))
            groups.setdefault(key, []).append(row)
        out = []
        for key, rows in groups.items():
            free = sorted({v for vs, f, _ in rows for v in vs if v not in s.fixed})
            out.append((rows, free))
        return out

    def newton(s, x, rows, free, w=None, maxit=60, tol=1e-9):
        if not rows: return True, 0.0
        col = {v: i for i, v in enumerate(free)}; idx = np.array(free, int)
        F = s.F(x, rows); nf = np.linalg.norm(F)
        for it in range(maxit):
            if np.abs(F).max() < tol: return True, float(np.abs(F).max())
            if not len(idx): break
            J = s.J(x, rows, col, len(F))
            if w is not None: J = J/w[idx]
            dx = np.linalg.lstsq(J, -F, rcond=1e-12)[0]
            if w is not None: dx = dx/w[idx]
            x0 = x[idx].copy(); t = 1.0
            while True:
                x[idx] = x0 + t*dx; F2 = s.F(x, rows); n2 = np.linalg.norm(F2)
                if n2 < nf or t < 1/64: break
                t *= 0.5
            if n2 >= nf and t < 1/64:
                x[idx] = x0 + dx; F2 = s.F(x, rows); n2 = np.linalg.norm(F2)
                if n2 >= nf*4: x[idx] = x0; break
            F, nf = F2, n2
        m = float(np.abs(F).max()) if len(F) else 0.0
        return m < 1e-6*max(1.0, s.scale/100), m

    def write(s, x):
        geo = s.geo; nP = len(geo.P)
        for i in range(nP): geo.P[i][0], geo.P[i][1] = float(x[2*i]), float(x[2*i + 1])
        for (ci, key), j in s.scal.items(): geo.C[ci][key] = float(abs(x[j]))
        geo.touch()

def geo_solve(geo, drag=None, skip=(), maxit=60):
    """Solve the sketch in place.  drag = {point index: (x, y)} pulls points toward the mouse (others move as little as possible).
    Returns (ok, worst residual, {constraint index: error})."""
    sy = SkSys(geo); sy.build(skip)
    if drag:                                                    # first try: dragged points exactly at the mouse
        x = sy.x.copy(); held = set()
        for i, (tx, ty) in drag.items():
            for v, tv in ((2*i, tx), (2*i + 1, ty)):
                if v not in sy.fixed: x[v] = tv; held.add(v)
        fixed0 = set(sy.fixed); sy.fixed |= held; ok = True
        for rows, free in sy.components():
            if np.abs(sy.F(x, rows)).max() > 1e-9:
                good, m = sy.newton(x, rows, free, None, 30); ok &= good
                if not ok: break
        sy.fixed = fixed0
        if ok: sy.write(x); return True, 0.0, sy.errors
        x = sy.x.copy(); w = np.ones(sy.n)                      # otherwise: as close to the mouse as the constraints allow
        for i, (tx, ty) in drag.items():
            for v, tv in ((2*i, tx), (2*i + 1, ty)):
                if v not in sy.fixed: x[v] = tv; w[v] = 30.0
    else: x = sy.x.copy(); w = None
    worst, ok = 0.0, True
    dvars = {v for i in (drag or {}) for v in (2*i, 2*i + 1)}
    for rows, free in sy.components():
        if drag is None or dvars & set(free) or np.abs(sy.F(x, rows)).max() > 1e-9:
            good, m = sy.newton(x, rows, free, w, maxit)
        else: good, m = True, 0.0
        ok &= good; worst = max(worst, m)
    if drag and not ok: return False, worst, sy.errors
    sy.write(x)
    return ok, worst, sy.errors

def geo_drag_scalar(geo, ci, key, val):
    """Drag a radius: set it and re-solve."""
    g = geo; g.C[ci][key] = float(val); return geo_solve(g)

def geo_status(geo):
    """Which points / curves are fully defined, and the remaining degrees of freedom."""
    sy = SkSys(geo); sy.build()
    used = {v for vs, f, _ in sy.rows for v in vs}
    det = set(sy.fixed); dof = 0
    for rows, free in sy.components():
        if not free: continue
        col = {v: i for i, v in enumerate(free)}; x = sy.x.copy(); m = len(sy.F(x, rows))
        J = sy.J(x, rows, col, m)
        try: U, S, Vt = np.linalg.svd(J)
        except np.linalg.LinAlgError: continue
        tol = max(J.shape)*(S[0] if len(S) else 0)*1e-9 + 1e-12; rank = int((S > tol).sum())
        N = Vt[rank:]; dof += len(free) - rank
        nn = np.linalg.norm(N, axis=0) if len(N) else np.zeros(len(free))
        for v, i in col.items():
            if nn[i] < 1e-6: det.add(v)
    allv = set()
    for ci, c in enumerate(geo.C):
        if c["k"] == "text": allv |= set(sy.pv(c["p"][0]))
        else: allv |= set(sy.cvars(ci))
    dof += len([v for v in allv if v not in used and v not in sy.fixed])
    pts_ok = {i for i in range(len(geo.P)) if 2*i in det and 2*i + 1 in det}
    crv_ok = {ci for ci, c in enumerate(geo.C) if all(v in det for v in (sy.cvars(ci) if c["k"] != "text" else sy.pv(c["p"][0])))}
    return dict(points=pts_ok, curves=crv_ok, dof=dof, errors=sy.errors)

def geo_rank(geo, skip=()):
    sy = SkSys(geo); sy.build(skip); r = 0
    for rows, free in sy.components():
        if not free: continue
        col = {v: i for i, v in enumerate(free)}; x = sy.x.copy(); m = len(sy.F(x, rows))
        S = np.linalg.svd(sy.J(x, rows, col, m), compute_uv=False)
        if len(S): r += int((S > max(m, len(free))*S[0]*1e-9 + 1e-12).sum())
    return r

def hold_points(geo, k):
    """Points that should stay put when constraint / dimension k is applied (Fusion moves the second pick)."""
    e = k["e"]
    if not e: return set()
    r = e[0]; t = k["t"]
    try:
        if t in ("rad", "dia", "erad"):
            cc = crv_center(geo.C[r[1]]); return {cc} if cc is not None else set()
        if t in ("coin", "mid") and len(e) > 1 and e[1][0] == "c": return set(crv_points(geo.C[e[1][1]]))
        if t == "sym" and len(e) > 2: return (set(crv_points(geo.C[r[1]])) if r[0] == "c" else {r[1]}) | set(crv_points(geo.C[e[2][1]]))
        if r[0] == "p": return {r[1]}
        if t in ("hor", "ver") and len(e) == 1: return {crv_points(geo.C[r[1]])[0]}
        return set(crv_points(geo.C[r[1]]))
    except Exception: return set()

def geo_try_add(geo, k, check_redundant=True, hold=None):
    """Add constraint / dimension k to a copy; solve.  Returns (new geo, None) or (None, reason)."""
    g = geo.copy(); g.K.append(k); ki = len(g.K) - 1
    hold = hold_points(geo, k) if hold is None else hold
    ok, worst, errs = geo_solve(g, drag={i: tuple(g.P[i]) for i in hold} if hold else None)
    if ki in errs: return None, errs[ki]
    if not ok:
        g = geo.copy(); g.K.append(k); ok, worst, errs = geo_solve(g)
    if not ok:
        return None, "This would over-constrain the sketch (it conflicts with existing constraints or dimensions)."
    if check_redundant and not k.get("drv") and k["t"] != "fix":
        if geo_rank(g) <= geo_rank(geo):
            return None, "This would over-constrain the sketch: it is already fully determined by other constraints."
    return g, None

def dim_env(params, sketches=()):
    """Names usable in expressions: user parameters + every dimension."""
    env = {}
    for nm, p in (params or {}).items():
        if "v" in p: env[nm] = p["v"]
    for g in sketches:
        for k in g.K:
            if k.get("n") and k["t"] in DIM_TYPES: env[k["n"]] = k.get("v", 0.0)
    return env

def geo_eval_dims(geo, env):
    """Re-evaluate dimension expressions (several passes so dims may refer to each other).  Returns list of errors."""
    errs = []; env = dict(env)
    for k in geo.K:
        if k.get("n") and k["t"] in DIM_TYPES: env[k["n"]] = k.get("v", 0.0)
    for _ in range(4):
        changed = False
        for k in geo.K:
            if k["t"] not in DIM_TYPES or k.get("drv") or not k.get("x"): continue
            try: v = eval_expr(k["x"], env, "ang" if k["t"] == "ang" else "len")
            except ExprError as ex: errs.append(f"{k.get('n', '?')}: {ex}"); continue
            if abs(v - k.get("v", 0.0)) > 1e-12: k["v"] = v; changed = True
            env[k["n"]] = v
        if not changed: break
    return errs

def geo_measure(geo, k):
    sy = SkSys(geo); return sy.meas(sy.x, k)
# ---- sketch regions (profiles): split a big face by every curve, keep the bounded cells ----
def geo_regions(geo, plane):
    pln = sk_pln(plane); edges, src = [], []
    for ci, c in enumerate(geo.C):
        if c.get("cons") or c.get("cl") or c["k"] == "point": continue
        for g in geo_g2d(geo, ci):
            try: edges.append(g2d_edge(g, pln)); src.append(ci)
            except Exception: pass
    if not edges: return []
    P = np.vstack([q for ci in set(src) for q in geo_curve_pts(geo, ci)])
    lo, hi = P.min(0), P.max(0); cx, cy = (lo + hi)/2; R = float(max(hi - lo))*2 + 100
    big = BRepBuilderAPI_MakeFace(pln, cx - R, cx + R, cy - R, cy + R).Face()
    sp = BRepAlgoAPI_Splitter(); a = List_TopoDS_Shape(); a.Append(big); sp.SetArguments(a)
    t = List_TopoDS_Shape()
    for e in edges: t.Append(e)
    sp.SetTools(t); sp.Build()
    if not sp.IsDone(): return []
    idx, srcmap = ShapeIndex(), {}
    for e, ci in zip(edges, src):
        mods = list(sp.Modified(e))
        for m in (mods or [e]): srcmap[idx.add(m)] = ci
    out = []
    for f in subshapes(sp.Shape(), TopAbs_FACE):
        keys = set(); outer = False
        for e in subshapes(f, TopAbs_EDGE):
            j = idx.find(e)
            if j < 0 or j not in srcmap: outer = True; break
            keys.add(srcmap[j])
        if outer: continue
        c = centroid(f); loc = plane.l(c)
        out.append((tuple(sorted(keys)), round(loc.x, 6), round(loc.y, 6), f))
    out.sort(key=lambda r: r[:3])
    return [r[3] for r in out]

# ---- helpers ----
def geo_compact(geo):
    """Drop points no curve uses, and constraints that refer to missing points / curves.  Re-indexes everything."""
    used = set()
    for c in geo.C: used |= set(crv_points(c))
    pmap = {}; P2 = []
    for i, p in enumerate(geo.P):
        if i in used: pmap[i] = len(P2); P2.append(p)
    for c in geo.C:
        c["p"] = [pmap[i] for i in c["p"]]
        if c["k"] == "spline" and c.get("h"): c["h"] = [pmap.get(h) if h is not None else None for h in c["h"]]
    K2 = []
    for k in geo.K:
        ok = True; e2 = []
        for r in k["e"]:
            if r[0] == "p":
                if r[1] not in pmap: ok = False; break
                e2.append(["p", pmap[r[1]]])
            else:
                if r[1] >= len(geo.C) or r[1] < 0: ok = False; break
                e2.append(list(r))
        if ok:
            if k.get("j"): k["j"] = [pmap.get(j) for j in k["j"]] if all(j in pmap for j in k["j"]) else None
            k["e"] = e2; K2.append(k)
    geo.P, geo.K = P2, K2; geo.touch(); return pmap

def geo_delete(geo, curves=(), points=(), cons=()):
    """Delete curves (and their unused points), standalone points and constraints; re-indexes."""
    curves = set(curves); cons = set(cons)
    for i in points:                                                     # a deleted point takes its curves with it
        for ci, c in enumerate(geo.C):
            if i in crv_points(c): curves.add(ci)
    cmap = {}; C2 = []
    for ci, c in enumerate(geo.C):
        if ci not in curves: cmap[ci] = len(C2); C2.append(c)
    K2 = []
    for ki, k in enumerate(geo.K):
        if ki in cons: continue
        if any(r[0] == "c" and r[1] not in cmap for r in k["e"]): continue
        k["e"] = [["c", cmap[r[1]]] if r[0] == "c" else r for r in k["e"]]; K2.append(k)
    for c in C2:
        if c["k"] == "text" and c.get("path") is not None: c["path"] = cmap.get(c["path"])
        if c.get("srcc") is not None: c["srcc"] = cmap.get(c["srcc"])
    geo.C, geo.K = C2, K2
    geo_compact(geo); return cmap

def near_point(geo, x, y, tol=1e-6, skip=()):
    best = None
    for i, (px, py) in enumerate(geo.P):
        if i in skip: continue
        d = math.hypot(px - x, py - y)
        if d < tol and (best is None or d < best[0]): best = (d, i)
    return best[1] if best else None

def curves_at_point(geo, i):
    return [ci for ci, c in enumerate(geo.C) if i in c["p"]]

def proj_param(g, x, y):
    pr = Geom2dAPI_ProjectPointOnCurve(gp_Pnt2d(x, y), g)
    if pr.NbPoints() == 0:
        u0, u1 = g.FirstParameter(), g.LastParameter(); a, b = g.Value(u0), g.Value(u1)
        return u0 if math.hypot(a.X() - x, a.Y() - y) < math.hypot(b.X() - x, b.Y() - y) else u1
    return pr.LowerDistanceParameter()

def g_val(g, u): q = g.Value(u); return q.X(), q.Y()

def crv_closed(c):
    return c["k"] in ("circle", "ellipse") or (c["k"] in ("spline", "cspline") and c.get("closed"))

def crv_cuts(geo, ci, extra=None):
    """Points where other curves cross curve ci: list of (param on ci, other curve, (x, y))."""
    gs = geo_g2d(geo, ci)
    if not gs: return []
    g = gs[0]; out = []
    for cj, c in enumerate(geo.C):
        if cj == ci or c["k"] in ("point",): continue
        for h in geo_g2d(geo, cj):
            try: it = Geom2dAPI_InterCurveCurve(g, h, 1e-7)
            except Exception: continue
            for k in range(1, it.NbPoints() + 1):
                q = it.Point(k); out.append((proj_param(g, q.X(), q.Y()), cj, (q.X(), q.Y())))
    for c in [geo.C[ci]]:                                                # self crossings of splines
        if c["k"] in ("spline", "cspline"):
            try:
                it = Geom2dAPI_InterCurveCurve(g, 1e-7)
                for k in range(1, it.NbPoints() + 1): q = it.Point(k); out.append((proj_param(g, q.X(), q.Y()), ci, (q.X(), q.Y())))
            except Exception: pass
    for x, y in (extra or []): out.append((proj_param(g, x, y), None, (x, y)))
    return out

def g2d_bspline_seg(g, u0, u1):
    bs = Geom2dConvert.CurveToBSplineCurve_s(g).Copy()
    if bs.IsPeriodic() and u1 < u0: u1 += bs.Period()
    bs.Segment(u0, u1)
    P = [(bs.Pole(i).X(), bs.Pole(i).Y()) for i in range(1, bs.NbPoles() + 1)]
    kn = [bs.Knot(i) for i in range(1, bs.NbKnots() + 1)]; mu = [bs.Multiplicity(i) for i in range(1, bs.NbKnots() + 1)]
    w = [bs.Weight(i) for i in range(1, bs.NbPoles() + 1)] if bs.IsRational() else None
    return P, kn, mu, w, bs.Degree()

def make_piece(geo, ci, u0, u1, pa, pb):
    """New curve dict: the part of curve ci between parameters u0..u1, ending on point indices pa / pb."""
    c = geo.C[ci]; k = c["k"]; g = geo_g2d(geo, ci)[0]
    base = {x: c[x] for x in ("cons", "cl") if c.get(x)}
    if k == "line": return dict(k="line", p=[pa, pb], **base)
    if k in ("circle", "arc"): return dict(k="arc", p=[c["p"][0], pa, pb], **base)
    if k in ("ellipse", "earc"): return dict(k="earc", p=[c["p"][0], c["p"][1], pa, pb], r2=c["r2"], **base)
    P, kn, mu, w, deg = g2d_bspline_seg(g, u0, u1)
    ids = [pa] + [geo.add_pt(*q) for q in P[1:-1]] + [pb]
    return dict(k="cspline", p=ids, deg=deg, kn=kn, mu=mu, w=w, **base)

def _cut_point(geo, cut, made):
    """Point index for a cut location: an existing end point of the other curve, or a new point on it."""
    u, cj, (x, y) = cut; key = (round(x, 7), round(y, 7))
    if key in made: return made[key], None
    i = near_point(geo, x, y, 1e-6)
    if i is not None: made[key] = i; return i, None
    i = geo.add_pt(x, y); made[key] = i
    return i, cj

def replace_curve(geo, ci, pieces, coin=()):
    """Swap curve ci for pieces (the first keeps index ci so constraints on it survive); coin = (point, curve) pairs to tie."""
    if not pieces: geo_delete(geo, curves=[ci]); return
    geo.C[ci] = pieces[0]
    for p in pieces[1:]: geo.C.append(p)
    for i, cj in coin:
        if cj is not None and geo.C[cj]["k"] not in ("text", "point"): geo.con("coin", P_(i), C_(cj))
    if geo.C[ci]["k"] in ("earc",) and len(pieces) > 1:
        for j in range(len(geo.C) - len(pieces) + 1, len(geo.C)): geo.con("eq", C_(ci), C_(j))
    for k in geo.K:                                                     # constraints that no longer make sense on a piece
        if k["t"] in ("mid",) and any(r == ["c", ci] for r in k["e"]): k["_drop"] = True
    geo.K = [k for k in geo.K if not k.pop("_drop", False)]
    geo_compact(geo)

def geo_trim(geo, ci, x, y):
    """Remove the part of curve ci between the crossings either side of (x, y)."""
    c = geo.C[ci]
    if c["k"] in ("point", "text"): raise RuntimeError("text and points can't be trimmed")
    g = geo_g2d(geo, ci)[0]; u0, u1 = g.FirstParameter(), g.LastParameter(); uc = proj_param(g, x, y)
    closed = crv_closed(c); per = (u1 - u0) if closed else None
    cuts = sorted(crv_cuts(geo, ci), key=lambda t: t[0]); eps = 1e-7*max(1.0, abs(u1 - u0))
    if not closed: cuts = [t for t in cuts if u0 + eps < t[0] < u1 - eps]
    uniq = []
    for t in cuts:
        if not uniq or abs(t[0] - uniq[-1][0]) > eps: uniq.append(t)
    cuts = uniq
    if closed and len(cuts) < 2 or not cuts:
        geo_delete(geo, curves=[ci]); return
    made = {}
    if not closed:
        a = max([t for t in cuts if t[0] < uc], default=None, key=lambda t: t[0])
        b = min([t for t in cuts if t[0] > uc], default=None, key=lambda t: t[0])
        s0, s1 = crv_ends(c); pieces, coin = [], []
        if a is not None:
            pa, cj = _cut_point(geo, a, made); pieces.append(make_piece(geo, ci, u0, a[0], s0, pa)); coin.append((pa, cj))
        if b is not None:
            pb, cj = _cut_point(geo, b, made); pieces.append(make_piece(geo, ci, b[0], u1, pb, s1)); coin.append((pb, cj))
        replace_curve(geo, ci, pieces, coin); return
    us = [t[0] for t in cuts]
    a = max([t for t in cuts if t[0] <= uc], default=cuts[-1], key=lambda t: t[0])
    b = min([t for t in cuts if t[0] > uc], default=cuts[0], key=lambda t: t[0])
    pa, ca = _cut_point(geo, a, made); pb, cb = _cut_point(geo, b, made)
    ub, ua = b[0], a[0]
    if ua <= ub: ua += per
    if c["k"] in ("circle", "ellipse"):
        piece = make_piece(geo, ci, ub, ua, pb, pa)
    else:
        piece = make_piece(geo, ci, ub, ua, pb, pa)
    replace_curve(geo, ci, [piece], [(pa, ca), (pb, cb)])

def geo_break(geo, ci, x, y):
    """Split curve ci at the crossings nearest (x, y) on each side (nothing is removed)."""
    c = geo.C[ci]
    if c["k"] in ("point", "text"): raise RuntimeError("text and points can't be broken")
    g = geo_g2d(geo, ci)[0]; u0, u1 = g.FirstParameter(), g.LastParameter(); uc = proj_param(g, x, y)
    closed = crv_closed(c); eps = 1e-7*max(1.0, abs(u1 - u0))
    cuts = sorted([t for t in crv_cuts(geo, ci) if closed or u0 + eps < t[0] < u1 - eps], key=lambda t: t[0])
    if not cuts: raise RuntimeError("nothing crosses this curve, so there is nowhere to break it")
    made = {}
    if not closed:
        a = max([t for t in cuts if t[0] < uc], default=None, key=lambda t: t[0])
        b = min([t for t in cuts if t[0] > uc], default=None, key=lambda t: t[0])
        s0, s1 = crv_ends(c); marks = [t for t in (a, b) if t is not None]
        ids = [s0]; coin = []
        for t in marks: i, cj = _cut_point(geo, t, made); ids.append(i); coin.append((i, cj))
        ids.append(s1); us = [u0] + [t[0] for t in marks] + [u1]
        pieces = [make_piece(geo, ci, us[j], us[j + 1], ids[j], ids[j + 1]) for j in range(len(us) - 1)]
        replace_curve(geo, ci, pieces, coin); return
    if len(cuts) < 2: raise RuntimeError("a closed curve needs two crossings to be broken")
    a = max([t for t in cuts if t[0] <= uc], default=cuts[-1], key=lambda t: t[0])
    b = min([t for t in cuts if t[0] > uc], default=cuts[0], key=lambda t: t[0])
    pa, ca = _cut_point(geo, a, made); pb, cb = _cut_point(geo, b, made)
    per = u1 - u0; ua, ub = a[0], b[0]
    ub2 = ub if ub > ua else ub + per
    p1 = make_piece(geo, ci, ua, ub2, pa, pb); ua2 = ua if ua > ub else ua + per
    p2 = make_piece(geo, ci, ub, ua2, pb, pa)
    replace_curve(geo, ci, [p1, p2], [(pa, ca), (pb, cb)])

def geo_extend(geo, ci, x, y):
    """Lengthen the end of a line / arc nearest (x, y) up to the next curve it would meet."""
    c = geo.C[ci]; k = c["k"]
    if k not in ("line", "arc", "earc"): raise RuntimeError("extend works on lines and arcs")
    s0, s1 = crv_ends(c); pa, pb = geo.P[s0], geo.P[s1]
    at_end = math.hypot(pb[0] - x, pb[1] - y) < math.hypot(pa[0] - x, pa[1] - y)
    if k == "line":
        (ax, ay), (bx, by) = pa, pb; L = math.hypot(bx - ax, by - ay); dx, dy = (bx - ax)/L, (by - ay)/L
        full = Geom2d_Line(gp_Pnt2d(ax, ay), gp_Dir2d(dx, dy)); ok = lambda u: u > L + 1e-7 if at_end else u < -1e-7
        dist = lambda u: (u - L) if at_end else -u
    else:
        gs = geo_g2d(geo, ci)[0]; base = gs.BasisCurve(); full = base
        u0, u1 = gs.FirstParameter(), gs.LastParameter(); per = math.tau
        ok = lambda u: True
        dist = lambda u: ((u - u1) % per) if at_end else ((u0 - u) % per)
    best = None
    for cj, cc in enumerate(geo.C):
        if cj == ci or cc["k"] == "point": continue
        for h in geo_g2d(geo, cj):
            try: it = Geom2dAPI_InterCurveCurve(full, h, 1e-7)
            except Exception: continue
            for j in range(1, it.NbPoints() + 1):
                q = it.Point(j); u = proj_param(full, q.X(), q.Y())
                if not ok(u): continue
                d = dist(u)
                if d > 1e-7 and (best is None or d < best[0]): best = (d, cj, (q.X(), q.Y()))
    if not best: raise RuntimeError("there is nothing for this end to extend to")
    end = s1 if at_end else s0
    if len(curves_at_point(geo, end)) > 1 or any(r == ["p", end] for kk in geo.K for r in kk["e"] if kk["t"] in ("fix",)):
        new = geo.add_pt(*best[2]); c["p"][c["p"].index(end)] = new; end = new
    else: geo.P[end] = [best[2][0], best[2][1]]
    for kk in geo.K:                                                    # length dimensions on this curve would undo the extension
        if kk["t"] in ("dist", "hdist", "vdist") and ["p", end] in kk["e"]: kk["_drop"] = True
    geo.K = [kk for kk in geo.K if not kk.pop("_drop", False)]
    i2 = near_point(geo, *best[2], tol=1e-6, skip={end})
    if i2 is not None and geo.C[best[1]]["k"] != "text":
        geo.con("coin", P_(end), P_(i2))
    elif geo.C[best[1]]["k"] not in ("text", "point"): geo.con("coin", P_(end), C_(best[1]))
    geo_compact(geo)

def corner_of(geo, c1, c2):
    """Shared end point of two curves (or None)."""
    e1, e2 = set(crv_ends(geo.C[c1])) - {None}, set(crv_ends(geo.C[c2])) - {None}
    s = e1 & e2
    if s: return s.pop()
    for i in e1:
        for j in e2:
            if any(k["t"] == "coin" and sorted([r[1] for r in k["e"] if r[0] == "p"]) == sorted([i, j]) for k in geo.K): return (i, j)
    return None

def _line_corner(geo, l1, l2):
    """Corner X of two lines and the unit directions from X toward each line's far end."""
    (a1, b1), (a2, b2) = geo.C[l1]["p"], geo.C[l2]["p"]
    A1, B1, A2, B2 = (np.array(geo.P[i]) for i in (a1, b1, a2, b2))
    d1, d2 = B1 - A1, B2 - A2; den = d1[0]*d2[1] - d1[1]*d2[0]
    if abs(den) < 1e-12: raise RuntimeError("the lines are parallel")
    t = ((A2 - A1)[0]*d2[1] - (A2 - A1)[1]*d2[0])/den; X = A1 + d1*t
    near1, far1 = (a1, b1) if np.linalg.norm(A1 - X) < np.linalg.norm(B1 - X) else (b1, a1)
    near2, far2 = (a2, b2) if np.linalg.norm(A2 - X) < np.linalg.norm(B2 - X) else (b2, a2)
    u1 = np.array(geo.P[far1]) - X; u2 = np.array(geo.P[far2]) - X
    return X, u1/np.linalg.norm(u1), u2/np.linalg.norm(u2), near1, near2, np.linalg.norm(u1), np.linalg.norm(u2)

def _detach_end(geo, ci, old, xy):
    """Give curve ci its own new end point at xy instead of `old`."""
    c = geo.C[ci]; i = geo.add_pt(*xy); c["p"][c["p"].index(old)] = i; return i

def geo_fillet(geo, l1, l2, r, expr=None):
    if geo.C[l1]["k"] != "line" or geo.C[l2]["k"] != "line": raise RuntimeError("sketch fillet works on two lines that meet at a corner")
    X, u1, u2, n1, n2, L1, L2 = _line_corner(geo, l1, l2)
    th = math.acos(max(-1, min(1, float(u1 @ u2))))
    if th < 1e-6 or th > math.pi - 1e-6: raise RuntimeError("the lines don't form a corner")
    t = r/math.tan(th/2)
    if t >= L1 - 1e-9 or t >= L2 - 1e-9: raise RuntimeError("the fillet radius is too big for these lines")
    T1, T2 = X + u1*t, X + u2*t; bis = (u1 + u2)/np.linalg.norm(u1 + u2); Cc = X + bis*(r/math.sin(th/2))
    i1 = _detach_end(geo, l1, n1, T1); i2 = _detach_end(geo, l2, n2, T2); ic = geo.add_pt(*Cc)
    cr = (T1 - Cc)[0]*(T2 - Cc)[1] - (T1 - Cc)[1]*(T2 - Cc)[0]
    s_, e_ = (i1, i2) if cr > 0 else (i2, i1)
    ai = geo.add("arc", [ic, s_, e_])
    geo.con("tan", C_(l1), C_(ai), j=[i1, i1]); geo.con("tan", C_(l2), C_(ai), j=[i2, i2])
    _virtual_sharp(geo, l1, l2, n1, n2)
    geo_compact(geo); return ai

def _virtual_sharp(geo, l1, l2, n1, n2):
    """Keep the removed corner as a construction point on both lines when dimensions / constraints use it."""
    used = n1 == n2 and any(["p", n1] in k["e"] for k in geo.K)
    if used:
        geo.add("point", [n1], cons=True); geo.con("coin", P_(n1), C_(l1)); geo.con("coin", P_(n1), C_(l2))
    else:
        geo.K = [k for k in geo.K if not (k["t"] == "coin" and any(r_ in (["p", n1], ["p", n2]) for r_ in k["e"]))]

def geo_chamfer(geo, l1, l2, d1, d2=None, ang=None):
    if geo.C[l1]["k"] != "line" or geo.C[l2]["k"] != "line": raise RuntimeError("sketch chamfer works on two lines that meet at a corner")
    X, u1, u2, n1, n2, L1, L2 = _line_corner(geo, l1, l2)
    if ang is not None:
        th = math.acos(max(-1, min(1, float(u1 @ u2)))); a = math.radians(ang)
        d2 = d1*math.sin(a)/math.sin(math.pi - th - a)
    d2 = d1 if d2 is None else d2
    if d1 >= L1 - 1e-9 or d2 >= L2 - 1e-9: raise RuntimeError("the chamfer is too big for these lines")
    T1, T2 = X + u1*d1, X + u2*d2
    i1 = _detach_end(geo, l1, n1, T1); i2 = _detach_end(geo, l2, n2, T2)
    li = geo.add("line", [i1, i2])
    _virtual_sharp(geo, l1, l2, n1, n2)
    geo_compact(geo); return li

# ---- conversion of OpenCascade curves into sketch curves ----
def add_g2d(geo, g, pmap, **flags):
    """Add a Geom2d curve as sketch curve(s), sharing end points through pmap {(x, y): index}."""
    def pt(x, y):
        key = (round(x, 6), round(y, 6))
        if key not in pmap: pmap[key] = geo.add_pt(x, y)
        return pmap[key]
    a = Geom2dAdaptor_Curve(g); u0, u1 = a.FirstParameter(), a.LastParameter(); t = a.GetType()
    p0, p1 = g_val(g, u0), g_val(g, u1); closed = math.hypot(p0[0] - p1[0], p0[1] - p1[1]) < 1e-7
    if t == GeomAbs_Line:
        if closed: return None
        return geo.add("line", [pt(*p0), pt(*p1)], **flags)
    if t == GeomAbs_Circle:
        ci = a.Circle(); c = ci.Location(); r = ci.Radius(); cen = geo.add_pt(c.X(), c.Y())
        if closed and abs(u1 - u0 - math.tau) < 1e-6: return geo.add("circle", [cen], r=r, **flags)
        direct = ci.IsDirect() if hasattr(ci, "IsDirect") else True
        s_, e_ = (pt(*p0), pt(*p1)) if direct else (pt(*p1), pt(*p0))
        return geo.add("arc", [cen, s_, e_], **flags)
    if t == GeomAbs_Ellipse:
        el = a.Ellipse(); c = el.Location(); X = el.XAxis().Direction(); R1, R2 = el.MajorRadius(), el.MinorRadius()
        cen = geo.add_pt(c.X(), c.Y()); m = geo.add_pt(c.X() + X.X()*R1, c.Y() + X.Y()*R1)
        direct = el.Axis().Sense() if hasattr(el.Axis(), "Sense") else True
        if closed and abs(u1 - u0 - math.tau) < 1e-6: return geo.add("ellipse", [cen, m], r2=R2, **flags)
        s_, e_ = (pt(*p0), pt(*p1)) if direct else (pt(*p1), pt(*p0))
        return geo.add("earc", [cen, m, s_, e_], r2=R2, **flags)
    try:
        bs = Geom2dConvert.CurveToBSplineCurve_s(Geom2d_TrimmedCurve(g, u0, u1) if not a.IsPeriodic() else g)
        P = [(bs.Pole(i).X(), bs.Pole(i).Y()) for i in range(1, bs.NbPoles() + 1)]
        kn = [bs.Knot(i) for i in range(1, bs.NbKnots() + 1)]; mu = [bs.Multiplicity(i) for i in range(1, bs.NbKnots() + 1)]
        w = [bs.Weight(i) for i in range(1, bs.NbPoles() + 1)] if bs.IsRational() else None
        if bs.IsPeriodic():
            bs2 = bs.Copy(); bs2.SetNotPeriodic()
            P = [(bs2.Pole(i).X(), bs2.Pole(i).Y()) for i in range(1, bs2.NbPoles() + 1)]
            kn = [bs2.Knot(i) for i in range(1, bs2.NbKnots() + 1)]; mu = [bs2.Multiplicity(i) for i in range(1, bs2.NbKnots() + 1)]
            w = [bs2.Weight(i) for i in range(1, bs2.NbPoles() + 1)] if bs2.IsRational() else None
        ids = [pt(*P[0])] + [geo.add_pt(*q) for q in P[1:-1]] + [pt(*P[-1])]
        return geo.add("cspline", ids, deg=bs.Degree(), kn=kn, mu=mu, w=w, **flags)
    except Exception:
        return None

def edge_to_g2d(edge, pln, project=True):
    """A 3D edge as 2D curve(s) in the plane (projected along the plane normal)."""
    from OCP.GeomProjLib import GeomProjLib
    from OCP.Geom import Geom_Plane, Geom_TrimmedCurve
    ad = BRepAdaptor_Curve(edge); u0, u1 = ad.FirstParameter(), ad.LastParameter()
    c3 = BRep_Tool.Curve_s(edge, u0, u1)
    if c3 is None: return None
    tc = Geom_TrimmedCurve(c3, u0, u1)
    if project:
        tc = GeomProjLib.ProjectOnPlane_s(tc, Geom_Plane(pln), pln.Axis().Direction(), True)
    g = GeomAPI.To2d_s(tc, pln)
    a = Geom2dAdaptor_Curve(g); p0, p1 = a.Value(a.FirstParameter()), a.Value(a.LastParameter())
    L = 0.0; prev = None
    for u in np.linspace(a.FirstParameter(), a.LastParameter(), 9):
        q = a.Value(float(u)); L += 0 if prev is None else q.Distance(prev); prev = q
    if L < 1e-7: return None
    return g

def geo_offset(geo, chain, d, plane):
    """Offset a connected chain of curves by d (positive = left of the first curve's direction).  Returns new curve ids."""
    pln = sk_pln(plane)
    edges = [e for ci in chain for e in crv_edges(geo, ci, plane)]
    ordered = chain_wire(edges) if len(edges) > 1 else BRepBuilderAPI_MakeWire(edges[0]).Wire()
    closed = BRep_Tool.IsClosed_s(ordered)
    from OCP.GeomAbs import GeomAbs_Intersection
    mo = BRepOffsetAPI_MakeOffset(ordered, GeomAbs_Intersection, not closed)
    mo.Perform(d)
    if not mo.IsDone(): raise RuntimeError("the offset failed for this distance")
    pmap, out = {}, []
    for e in subshapes(mo.Shape(), TopAbs_EDGE):
        g = edge_to_g2d(e, pln, False)
        if g is None: continue
        ci = add_g2d(geo, g, pmap)
        if ci is not None: out.append(ci)
    if not out: raise RuntimeError("the offset produced nothing (distance too large?)")
    return out

def chain_from(geo, ci):
    """The curves connected end-to-end with ci (an open chain or a loop)."""
    seen = [ci]; frontier = [ci]
    while frontier:
        c = frontier.pop()
        for p in [q for q in crv_ends(geo.C[c]) if q is not None]:
            nb = [cj for cj in curves_at_point(geo, p) if cj not in seen and geo.C[cj]["k"] not in ("point", "text") and p in crv_ends(geo.C[cj])]
            if len(nb) == 1: seen.append(nb[0]); frontier.append(nb[0])
    return seen

# ---- copying / transforming selections ----
def geo_copy_curves(geo, curves, fn, link=None):
    """Duplicate curves with every point mapped through fn(x, y) -> (x, y).  Returns ({old point: new point}, [new curve ids])."""
    pm = {}; new = []
    def P(i):
        if i not in pm:
            x, y = fn(*geo.P[i]); pm[i] = geo.add_pt(x, y)
        return pm[i]
    for ci in curves:
        c = copy.deepcopy(geo.C[ci]); c["p"] = [P(i) for i in c["p"]]; c.pop("src", None); c.pop("lock", None)
        if c["k"] == "spline" and c.get("h"): c["h"] = [P(h) if h is not None else None for h in c["h"]]
        geo.C.append(c); new.append(len(geo.C) - 1)
    return pm, new

def _affine(ci_list, geo, M):
    """Apply 2x3 matrix to radii (uniform scale) of copied circles / ellipses."""
    k = math.sqrt(abs(M[0][0]*M[1][1] - M[0][1]*M[1][0]))
    for ci in ci_list:
        c = geo.C[ci]
        if "r" in c: c["r"] = c["r"]*k
        if "r2" in c: c["r2"] = c["r2"]*k
        if c["k"] == "text": c["size"] = c.get("size", 10)*k; c["ang"] = c.get("ang", 0) + math.degrees(math.atan2(M[1][0], M[0][0]))

def mat_fn(M): return lambda x, y: (M[0][0]*x + M[0][1]*y + M[0][2], M[1][0]*x + M[1][1]*y + M[1][2])
def mat_rot(cx, cy, deg):
    a = math.radians(deg); ca, sa = math.cos(a), math.sin(a)
    return [[ca, -sa, cx - ca*cx + sa*cy], [sa, ca, cy - sa*cx - ca*cy]]
def mat_move(dx, dy): return [[1, 0, dx], [0, 1, dy]]
def mat_scale(cx, cy, k): return [[k, 0, cx - k*cx], [0, k, cy - k*cy]]
def mat_mirror(ax, ay, bx, by):
    dx, dy = bx - ax, by - ay; L = math.hypot(dx, dy); dx, dy = dx/L, dy/L
    a, b, c = dx*dx - dy*dy, 2*dx*dy, dy*dy - dx*dx
    return [[a, b, ax - a*ax - b*ay], [b, c, ay - b*ax - c*ay]]

def geo_transform(geo, curves, M, copy_=False, points=()):
    """Move / rotate / scale selected curves (and loose points); with copy_ the originals stay."""
    fn = mat_fn(M)
    if copy_:
        pm, new = geo_copy_curves(geo, curves, fn); _affine(new, geo, M)
        if abs(M[0][0]*M[1][1] - M[0][1]*M[1][0] + 1) < 1e-9:           # mirrored copies run the other way round
            for ci in new: _reverse(geo.C[ci])
        return new
    pts = set(points)
    for ci in curves: pts |= set(crv_points(geo.C[ci]))
    for i in pts: geo.P[i] = list(fn(*geo.P[i]))
    _affine(list(curves), geo, M)
    k = math.sqrt(abs(M[0][0]*M[1][1] - M[0][1]*M[1][0]))
    if abs(k - 1) > 1e-12:                                              # scale dimensions that lie inside the selection
        for kk in geo.K:
            if kk["t"] in DIM_TYPES - {"ang"}:
                refs = [r for r in kk["e"]]
                inside = all((r[0] == "p" and r[1] in pts) or (r[0] == "c" and r[1] in curves) for r in refs)
                if inside: kk["v"] = kk.get("v", 0)*k; kk["x"] = fmt_expr(kk["v"])
    geo.touch(); return list(curves)

def fmt_expr(v): return f"{v/unit_k():.6g} {DISPLAY['unit']}"

def _reverse(c):
    k = c["k"]
    if k == "arc": c["p"] = [c["p"][0], c["p"][2], c["p"][1]]
    elif k == "earc": c["p"] = [c["p"][0], c["p"][1], c["p"][3], c["p"][2]]

def geo_mirror(geo, curves, line_ci, points=()):
    a, b = geo.C[line_ci]["p"]; M = mat_mirror(*geo.P[a], *geo.P[b])
    on_line = lambda i: abs(SkSys(geo).sdist_line(SkSys(geo).x, line_ci, *geo.P[i])) < 1e-7
    pm = {}; new = []
    sy = SkSys(geo); fn = mat_fn(M)
    def P(i):
        if i not in pm:
            if abs(sy.sdist_line(sy.x, line_ci, *geo.P[i])) < 1e-7: pm[i] = i
            else: pm[i] = geo.add_pt(*fn(*geo.P[i]))
        return pm[i]
    for ci in curves:
        c = copy.deepcopy(geo.C[ci]); c["p"] = [P(i) for i in c["p"]]; c.pop("src", None); c.pop("lock", None)
        if c["k"] == "spline" and c.get("h"): c["h"] = [P(h) if h is not None else None for h in c["h"]]
        _reverse(c); geo.C.append(c); new.append(len(geo.C) - 1)
    for i in points: P(i)
    for i, j in pm.items():
        if i != j: geo.con("sym", P_(i), P_(j), C_(line_ci))
    for ci, nj in zip(curves, new):
        if geo.C[ci]["k"] == "circle": geo.con("eq", C_(ci), C_(nj))
        if geo.C[ci]["k"] in ("ellipse", "earc"): geo.con("eq", C_(ci), C_(nj))
    geo.touch(); return new

def geo_rect_pattern(geo, curves, d1, n1, s1, d2=None, n2=1, s2=0.0, sym1=False, sym2=False):
    new = []
    o1 = -(n1 - 1)*s1/2 if sym1 else 0.0; o2 = -(n2 - 1)*s2/2 if sym2 else 0.0
    for i in range(n1):
        for j in range(max(n2, 1)):
            if i == 0 and j == 0 and not sym1 and not sym2: continue
            dx = d1[0]*(o1 + i*s1) + (d2[0]*(o2 + j*s2) if d2 else 0); dy = d1[1]*(o1 + i*s1) + (d2[1]*(o2 + j*s2) if d2 else 0)
            if abs(dx) < 1e-12 and abs(dy) < 1e-12: continue
            new += geo_copy_curves(geo, curves, mat_fn(mat_move(dx, dy)))[1]
    return new

def geo_circ_pattern(geo, curves, cx, cy, n, total=360.0, sym=False):
    new = []; full = abs(abs(total) - 360) < 1e-9
    step = total/n if full else total/max(n - 1, 1); start = -total/2 if sym and not full else 0.0
    for i in range(n):
        a = start + i*step
        if abs(a) < 1e-12: continue
        M = mat_rot(cx, cy, a); _, nn = geo_copy_curves(geo, curves, mat_fn(M)); _affine(nn, geo, M); new += nn
    return new

def geo_explode_text(geo, ci):
    """Turn a text into ordinary lines and Bezier splines."""
    pmap = {}
    for g in list(geo_g2d(geo, ci)): add_g2d(geo, g, pmap)
    geo_delete(geo, curves=[ci])

# ---- importing DXF / SVG ----
def dxf_read(path):
    """Minimal DXF reader: LINE, CIRCLE, ARC, LWPOLYLINE, POLYLINE, SPLINE, ELLIPSE, POINT.  Returns (geo, unit scale)."""
    with open(path, errors="replace") as f: raw = f.read().splitlines()
    pairs = [(raw[i].strip(), raw[i + 1].rstrip("\r")) for i in range(0, len(raw) - 1, 2)]
    scale = 1.0
    for i, (c, v) in enumerate(pairs):
        if c == "9" and v.strip() == "$INSUNITS" and i + 1 < len(pairs):
            scale = {1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0, 0: 1.0}.get(int(float(pairs[i + 1][1])), 1.0)
    ents = []; i = 0; in_ent = False; cur = None
    while i < len(pairs):
        c, v = pairs[i]; v = v.strip()
        if c == "0":
            if cur: ents.append(cur)
            cur = [v, []] if v in ("LINE", "CIRCLE", "ARC", "LWPOLYLINE", "POLYLINE", "VERTEX", "SEQEND", "SPLINE", "ELLIPSE", "POINT") else None
        elif cur is not None: cur[1].append((int(c), v))
        i += 1
    if cur: ents.append(cur)
    geo = Geo(); pm = {}
    def pt(x, y):
        key = (round(x, 6), round(y, 6))
        if key not in pm: pm[key] = geo.add_pt(x, y)
        return pm[key]
    def get(fields, code, d=0.0):
        for c, v in fields:
            if c == code: return float(v)
        return d
    def bulge_seg(a, b, bul):
        if abs(bul) < 1e-12: geo.add("line", [pt(*a), pt(*b)]); return
        th = 4*math.atan(bul); dx, dy = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dy)
        r = L/(2*math.sin(th/2)); mx, my = (a[0] + b[0])/2, (a[1] + b[1])/2
        h = math.sqrt(max(r*r - (L/2)**2, 0)); nx, ny = -dy/L, dx/L
        sgn = 1 if (bul > 0) == (abs(th) < math.pi) else -1
        cx, cy = mx + nx*h*sgn*(1 if r > 0 else -1), my + ny*h*sgn*(1 if r > 0 else -1)
        cen = geo.add_pt(cx, cy)
        geo.add("arc", [cen, pt(*a), pt(*b)] if bul > 0 else [cen, pt(*b), pt(*a)])
    k = 0
    while k < len(ents):
        name, F = ents[k]; k += 1
        try:
            if name == "LINE": geo.add("line", [pt(get(F, 10), get(F, 20)), pt(get(F, 11), get(F, 21))])
            elif name == "CIRCLE": geo.add("circle", [geo.add_pt(get(F, 10), get(F, 20))], r=get(F, 40))
            elif name == "ARC":
                cx, cy, r = get(F, 10), get(F, 20), get(F, 40); a0, a1 = math.radians(get(F, 50)), math.radians(get(F, 51))
                geo.add("arc", [geo.add_pt(cx, cy), pt(cx + r*math.cos(a0), cy + r*math.sin(a0)), pt(cx + r*math.cos(a1), cy + r*math.sin(a1))])
            elif name == "POINT": geo.add("point", [geo.add_pt(get(F, 10), get(F, 20))])
            elif name == "LWPOLYLINE":
                closed = int(get(F, 70)) & 1; verts = []
                for c, v in F:
                    if c == 10: verts.append([float(v), 0.0, 0.0])
                    elif c == 20 and verts: verts[-1][1] = float(v)
                    elif c == 42 and verts: verts[-1][2] = float(v)
                n = len(verts)
                for j in range(n if closed else n - 1):
                    a, b = verts[j], verts[(j + 1) % n]; bulge_seg(a[:2], b[:2], a[2])
            elif name == "POLYLINE":
                closed = int(get(F, 70)) & 1; verts = []
                while k < len(ents) and ents[k][0] == "VERTEX":
                    V_ = ents[k][1]; verts.append((get(V_, 10), get(V_, 20), get(V_, 42))); k += 1
                if k < len(ents) and ents[k][0] == "SEQEND": k += 1
                n = len(verts)
                for j in range(n if closed else n - 1):
                    a, b = verts[j], verts[(j + 1) % n]; bulge_seg(a[:2], b[:2], a[2])
            elif name == "ELLIPSE":
                cx, cy, mx, my, ratio = get(F, 10), get(F, 20), get(F, 11), get(F, 21), get(F, 40, 1.0)
                t0, t1 = get(F, 41, 0.0), get(F, 42, math.tau); r1 = math.hypot(mx, my)
                cen = geo.add_pt(cx, cy); m = geo.add_pt(cx + mx, cy + my)
                if abs((t1 - t0) - math.tau) < 1e-6: geo.add("ellipse", [cen, m], r2=r1*ratio)
                else:
                    fr = (cx, cy, mx/r1, my/r1, r1, r1*ratio)
                    geo.add("earc", [cen, m, pt(*ell_point(fr, t0)), pt(*ell_point(fr, t1))], r2=r1*ratio)
            elif name == "SPLINE":
                deg = int(get(F, 71, 3)); knots = [float(v) for c, v in F if c == 40]; wts = [float(v) for c, v in F if c == 41]
                cps = []; fits = []
                for c, v in F:
                    if c == 10: cps.append([float(v), 0.0])
                    elif c == 20 and cps: cps[-1][1] = float(v)
                    elif c == 11: fits.append([float(v), 0.0])
                    elif c == 21 and fits: fits[-1][1] = float(v)
                if cps and knots:
                    kn, mu = [], []
                    for kv in knots:
                        if kn and abs(kv - kn[-1]) < 1e-12: mu[-1] += 1
                        else: kn.append(kv); mu.append(1)
                    ids = [pt(*cps[0])] + [geo.add_pt(*q) for q in cps[1:-1]] + [pt(*cps[-1])]
                    geo.add("cspline", ids, deg=deg, kn=kn, mu=mu, w=wts if wts and any(abs(w - 1) > 1e-12 for w in wts) else None)
                elif fits: geo.add("spline", [pt(*fits[0])] + [geo.add_pt(*q) for q in fits[1:-1]] + [pt(*fits[-1])])
        except Exception: traceback.print_exc()
    if scale != 1.0:
        for p in geo.P: p[0] *= scale; p[1] *= scale
        for c in geo.C:
            for key in ("r", "r2"):
                if key in c: c[key] *= scale
    geo.touch(); return geo

def _svg_num(s): return [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', s)]

def _svg_mat(t):
    M = np.eye(3)
    for name, args in re.findall(r'(\w+)\s*\(([^)]*)\)', t or ""):
        a = _svg_num(args); m = np.eye(3)
        if name == "matrix" and len(a) == 6: m = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1]])
        elif name == "translate": m[0, 2] = a[0]; m[1, 2] = a[1] if len(a) > 1 else 0
        elif name == "scale": m[0, 0] = a[0]; m[1, 1] = a[1] if len(a) > 1 else a[0]
        elif name == "rotate":
            r = math.radians(a[0]); c, s_ = math.cos(r), math.sin(r); R_ = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            if len(a) == 3: T = np.eye(3); T[:2, 2] = a[1:]; Ti = np.eye(3); Ti[:2, 2] = [-a[1], -a[2]]; m = T @ R_ @ Ti
            else: m = R_
        elif name == "skewX": m[0, 1] = math.tan(math.radians(a[0]))
        elif name == "skewY": m[1, 0] = math.tan(math.radians(a[0]))
        M = M @ m
    return M

def _svg_len(s, default=None):
    if s is None: return default
    m = re.match(r'\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(mm|cm|in|pt|pc|px)?', s)
    if not m: return default
    v = float(m.group(1)); u = m.group(2) or "px"
    return v*{"mm": 1, "cm": 10, "in": 25.4, "pt": 25.4/72, "pc": 25.4/6, "px": 25.4/96}[u]

def svg_read(path):
    """SVG paths / shapes as sketch curves (in mm, y up).  Arcs become Bezier splines."""
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    tag = lambda e: e.tag.split("}")[-1]
    vb = _svg_num(root.get("viewBox", "")); W_ = _svg_len(root.get("width")); H_ = _svg_len(root.get("height"))
    if len(vb) == 4 and W_ and vb[2]: sx = W_/vb[2]; sy = (H_/vb[3]) if H_ and vb[3] else sx
    else: sx = sy = 25.4/96
    base = np.array([[sx, 0, -(vb[0] if len(vb) == 4 else 0)*sx], [0, -sy, (vb[1] if len(vb) == 4 else 0)*sy], [0, 0, 1]])
    geo = Geo(); pm = {}
    def pt(M, x, y):
        X, Y, _ = M @ [x, y, 1]; key = (round(X, 6), round(Y, 6))
        if key not in pm: pm[key] = geo.add_pt(X, Y)
        return pm[key]
    def raw(M, x, y): X, Y, _ = M @ [x, y, 1]; return geo.add_pt(X, Y)
    def line(M, a, b):
        if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-12: geo.add("line", [pt(M, *a), pt(M, *b)])
    def cubic(M, a, c1, c2, b): geo.add("cspline", [pt(M, *a), raw(M, *c1), raw(M, *c2), pt(M, *b)], deg=3, kn=[0.0, 1.0], mu=[4, 4])
    def arc_to(M, p0, rx, ry, phi, large, sweep, p1):
        if rx == 0 or ry == 0: line(M, p0, p1); return
        ph = math.radians(phi); cp, sp_ = math.cos(ph), math.sin(ph)
        dx, dy = (p0[0] - p1[0])/2, (p0[1] - p1[1])/2; x1, y1 = cp*dx + sp_*dy, -sp_*dx + cp*dy
        rx, ry = abs(rx), abs(ry); lam = x1*x1/(rx*rx) + y1*y1/(ry*ry)
        if lam > 1: rx *= math.sqrt(lam); ry *= math.sqrt(lam)
        num = rx*rx*ry*ry - rx*rx*y1*y1 - ry*ry*x1*x1; den = rx*rx*y1*y1 + ry*ry*x1*x1
        co = math.sqrt(max(0, num/den)) if den else 0
        if large == sweep: co = -co
        cxp, cyp = co*rx*y1/ry, -co*ry*x1/rx
        cx, cy = cp*cxp - sp_*cyp + (p0[0] + p1[0])/2, sp_*cxp + cp*cyp + (p0[1] + p1[1])/2
        ang = lambda ux, uy, vx, vy: math.atan2(ux*vy - uy*vx, ux*vx + uy*vy)
        t1 = ang(1, 0, (x1 - cxp)/rx, (y1 - cyp)/ry); dt = ang((x1 - cxp)/rx, (y1 - cyp)/ry, (-x1 - cxp)/rx, (-y1 - cyp)/ry)
        if not sweep and dt > 0: dt -= math.tau
        elif sweep and dt < 0: dt += math.tau
        n = max(1, int(math.ceil(abs(dt)/(math.pi/2)))); d = dt/n; k = 4/3*math.tan(d/4)
        E = lambda t: (cx + rx*math.cos(t)*cp - ry*math.sin(t)*sp_, cy + rx*math.cos(t)*sp_ + ry*math.sin(t)*cp)
        Ed = lambda t: (-rx*math.sin(t)*cp - ry*math.cos(t)*sp_, -rx*math.sin(t)*sp_ + ry*math.cos(t)*cp)
        for i in range(n):
            ta, tb = t1 + i*d, t1 + (i + 1)*d; a, b = E(ta), E(tb); da, db = Ed(ta), Ed(tb)
            a_ = p0 if i == 0 else a; b_ = p1 if i == n - 1 else b
            cubic(M, a_, (a[0] + k*da[0], a[1] + k*da[1]), (b[0] - k*db[0], b[1] - k*db[1]), b_)
    def path_d(M, d):
        toks = re.findall(r'[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', d)
        i, cur, start, cmd, lastc = 0, (0.0, 0.0), (0.0, 0.0), None, None
        def nums(n):
            nonlocal i
            out = [float(t) for t in toks[i:i + n]]; i += n; return out
        while i < len(toks):
            if re.match(r'[A-Za-z]', toks[i]): cmd = toks[i]; i += 1
            if cmd is None: break
            rel = cmd.islower(); C_ = cmd.upper(); ox, oy = cur if rel else (0.0, 0.0)
            if C_ == "Z": line(M, cur, start); cur = start; lastc = None; cmd = None; continue
            if i >= len(toks) or re.match(r'[A-Za-z]', toks[i]): continue
            if C_ == "M":
                x, y = nums(2); cur = start = (ox + x, oy + y); cmd = "l" if rel else "L"; lastc = None
            elif C_ == "L": x, y = nums(2); p = (ox + x, oy + y); line(M, cur, p); cur = p; lastc = None
            elif C_ == "H": (x,) = nums(1); p = ((ox if rel else 0) + x, cur[1]); line(M, cur, p); cur = p; lastc = None
            elif C_ == "V": (y,) = nums(1); p = (cur[0], (oy if rel else 0) + y); line(M, cur, p); cur = p; lastc = None
            elif C_ in ("C", "S"):
                if C_ == "C": x1, y1, x2, y2, x, y = nums(6); c1 = (ox + x1, oy + y1)
                else:
                    x2, y2, x, y = nums(4); c1 = (2*cur[0] - lastc[0], 2*cur[1] - lastc[1]) if lastc else cur
                c2 = (ox + x2, oy + y2); p = (ox + x, oy + y); cubic(M, cur, c1, c2, p); lastc = c2; cur = p
            elif C_ in ("Q", "T"):
                if C_ == "Q": x1, y1, x, y = nums(4); q = (ox + x1, oy + y1)
                else: x, y = nums(2); q = (2*cur[0] - lastc[0], 2*cur[1] - lastc[1]) if lastc else cur
                p = (ox + x, oy + y)
                cubic(M, cur, (cur[0] + 2/3*(q[0] - cur[0]), cur[1] + 2/3*(q[1] - cur[1])), (p[0] + 2/3*(q[0] - p[0]), p[1] + 2/3*(q[1] - p[1])), p)
                lastc = q; cur = p
            elif C_ == "A":
                rx, ry, phi, la, sw, x, y = nums(7); p = (ox + x, oy + y); arc_to(M, cur, rx, ry, phi, int(la), int(sw), p); cur = p; lastc = None
    def walk(e, M):
        M = M @ _svg_mat(e.get("transform")); t = tag(e)
        f = lambda k, d=0.0: float(_svg_num(e.get(k, str(d)))[0]) if _svg_num(e.get(k, str(d))) else d
        if t == "path": path_d(M, e.get("d", ""))
        elif t == "rect":
            x, y, w, h = f("x"), f("y"), f("width"), f("height")
            P = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            for a, b in zip(P, P[1:] + P[:1]): line(M, a, b)
        elif t in ("circle", "ellipse"):
            cx, cy = f("cx"), f("cy"); rx = f("r") if t == "circle" else f("rx"); ry = f("r") if t == "circle" else f("ry")
            lin = M[:2, :2]
            if abs(lin[0, 1]) < 1e-12 and abs(lin[1, 0]) < 1e-12 and abs(abs(lin[0, 0]*rx) - abs(lin[1, 1]*ry)) < 1e-9:
                X, Y, _ = M @ [cx, cy, 1]; geo.add("circle", [geo.add_pt(X, Y)], r=abs(lin[0, 0]*rx))
            else:
                arc_to(M, (cx + rx, cy), rx, ry, 0, 0, 1, (cx - rx, cy)); arc_to(M, (cx - rx, cy), rx, ry, 0, 0, 1, (cx + rx, cy))
        elif t == "line": line(M, (f("x1"), f("y1")), (f("x2"), f("y2")))
        elif t in ("polyline", "polygon"):
            v = _svg_num(e.get("points", "")); P = list(zip(v[0::2], v[1::2]))
            for a, b in zip(P, P[1:] + (P[:1] if t == "polygon" else [])): line(M, a, b)
        for ch in e:
            if tag(ch) not in ("defs", "clipPath", "mask", "symbol", "title", "desc", "metadata", "style"): walk(ch, M)
    walk(root, base); geo.touch(); return geo

def geo_merge(geo, other, dx=0.0, dy=0.0):
    """Append another geometry (re-indexed) - used for imports."""
    n = len(geo.P); nc = len(geo.C)
    geo.P += [[p[0] + dx, p[1] + dy] for p in other.P]
    for c in other.C:
        c = copy.deepcopy(c); c["p"] = [i + n for i in c["p"]]
        if c["k"] == "spline" and c.get("h"): c["h"] = [h + n if h is not None else None for h in c["h"]]
        geo.C.append(c)
    for k in other.K:
        k = copy.deepcopy(k); k["e"] = [[r[0], r[1] + (n if r[0] == "p" else nc)] for r in k["e"]]; geo.K.append(k)
    geo.touch(); return list(range(nc, len(geo.C)))
# ---- building sketch shapes (every drawing tool ends in one of these) ----
def use_pt(geo, spec):
    """Point for a clicked position: spec = (x, y, snap) where snap is None, ("pt", i), ("mid", curve) or ("on", curve)."""
    x, y, snap = spec if len(spec) == 3 else (spec[0], spec[1], None)
    if snap and snap[0] == "pt": return snap[1]
    i = geo.add_pt(x, y)
    if snap and snap[0] == "mid": geo.con("mid", P_(i), C_(snap[1]))
    elif snap and snap[0] == "on" and geo.C[snap[1]]["k"] not in ("text", "point"): geo.con("coin", P_(i), C_(snap[1]))
    return i

def xy(spec): return float(spec[0]), float(spec[1])

def tie_on(geo, spec, ci):
    """A snapped click that lies on a circle (2 / 3-point circles) becomes a point-on-curve constraint."""
    snap = spec[2] if len(spec) == 3 else None
    if snap and snap[0] == "pt": geo.con("coin", P_(snap[1]), C_(ci))

def auto_hv(geo, ci, tol_deg=0.0):
    a, b = geo.C[ci]["p"]; (ax, ay), (bx, by) = geo.P[a], geo.P[b]; L = math.hypot(bx - ax, by - ay)
    if L < 1e-12: return
    if abs(by - ay) <= max(1e-9, L*math.sin(math.radians(tol_deg))): geo.con("hor", C_(ci))
    elif abs(bx - ax) <= max(1e-9, L*math.sin(math.radians(tol_deg))): geo.con("ver", C_(ci))

def build_line(geo, a, b, hv=True):
    ia, ib = use_pt(geo, a), use_pt(geo, b)
    if ia == ib: return None
    ci = geo.add("line", [ia, ib])
    if hv: auto_hv(geo, ci)
    return ci

def build_rect2(geo, a, b):
    (ax, ay), (bx, by) = xy(a), xy(b)
    if abs(ax - bx) < 1e-9 or abs(ay - by) < 1e-9: return []
    ids = [use_pt(geo, a), geo.add_pt(bx, ay), use_pt(geo, b), geo.add_pt(ax, by)]
    L = [geo.add("line", [ids[j], ids[(j + 1) % 4]]) for j in range(4)]
    geo.con("hor", C_(L[0])); geo.con("hor", C_(L[2])); geo.con("ver", C_(L[1])); geo.con("ver", C_(L[3]))
    return L

def build_rect3(geo, a, b, c):
    (ax, ay), (bx, by), (cx, cy) = xy(a), xy(b), xy(c)
    dx, dy = bx - ax, by - ay; L = math.hypot(dx, dy)
    if L < 1e-9: return []
    nx, ny = -dy/L, dx/L; h = (cx - bx)*nx + (cy - by)*ny
    if abs(h) < 1e-9: return []
    ids = [use_pt(geo, a), use_pt(geo, b), geo.add_pt(bx + nx*h, by + ny*h), geo.add_pt(ax + nx*h, ay + ny*h)]
    Ls = [geo.add("line", [ids[j], ids[(j + 1) % 4]]) for j in range(4)]
    if abs(dy) < 1e-9 or abs(dx) < 1e-9:
        geo.con("hor" if abs(dy) < 1e-9 else "ver", C_(Ls[0])); geo.con("hor" if abs(dy) < 1e-9 else "ver", C_(Ls[2]))
        geo.con("ver" if abs(dy) < 1e-9 else "hor", C_(Ls[1])); geo.con("ver" if abs(dy) < 1e-9 else "hor", C_(Ls[3]))
    else:
        geo.con("perp", C_(Ls[0]), C_(Ls[1])); geo.con("par", C_(Ls[0]), C_(Ls[2])); geo.con("par", C_(Ls[1]), C_(Ls[3]))
    return Ls

def build_rectc(geo, c, b):
    (cx, cy), (bx, by) = xy(c), xy(b); w, h = abs(bx - cx), abs(by - cy)
    if w < 1e-9 or h < 1e-9: return []
    ic = use_pt(geo, c)
    ids = [geo.add_pt(cx - w, cy - h), geo.add_pt(cx + w, cy - h), geo.add_pt(cx + w, cy + h), geo.add_pt(cx - w, cy + h)]
    if len(b) == 3 and b[2] and b[2][0] == "pt":                         # the corner landed on an existing point
        q = min(range(4), key=lambda j: math.hypot(geo.P[ids[j]][0] - bx, geo.P[ids[j]][1] - by)); geo.con("coin", P_(ids[q]), P_(b[2][1]))
    L = [geo.add("line", [ids[j], ids[(j + 1) % 4]]) for j in range(4)]
    geo.con("hor", C_(L[0])); geo.con("hor", C_(L[2])); geo.con("ver", C_(L[1])); geo.con("ver", C_(L[3]))
    d = geo.add("line", [ids[0], ids[2]], cons=True); geo.con("mid", P_(ic), C_(d))
    return L + [d]

def build_circle(geo, c, r):
    if r < 1e-9: return None
    return geo.add("circle", [use_pt(geo, c)], r=float(r))

def build_circle_2p(geo, a, b):
    (ax, ay), (bx, by) = xy(a), xy(b); r = math.hypot(bx - ax, by - ay)/2
    if r < 1e-9: return None
    ci = geo.add("circle", [geo.add_pt((ax + bx)/2, (ay + by)/2)], r=r); tie_on(geo, a, ci); tie_on(geo, b, ci); return ci

def circumcircle(a, b, c):
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    d = 2*(ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
    if abs(d) < 1e-12: return None
    ux = ((ax*ax + ay*ay)*(by - cy) + (bx*bx + by*by)*(cy - ay) + (cx*cx + cy*cy)*(ay - by))/d
    uy = ((ax*ax + ay*ay)*(cx - bx) + (bx*bx + by*by)*(ax - cx) + (cx*cx + cy*cy)*(bx - ax))/d
    return ux, uy, math.hypot(ax - ux, ay - uy)

def build_circle_3p(geo, a, b, c):
    cc = circumcircle(xy(a), xy(b), xy(c))
    if not cc: return None
    ci = geo.add("circle", [geo.add_pt(cc[0], cc[1])], r=cc[2])
    for s in (a, b, c): tie_on(geo, s, ci)
    return ci

def _side(geo, li, x, y):
    sy = SkSys(geo); return 1 if sy.sdist_line(sy.x, li, x, y) >= 0 else -1

def tangent_con(geo, c1, c2, at=None):
    """Tangent constraint dict with the right mode for the current positions."""
    k = dict(t="tan", e=[C_(c1), C_(c2)]); A, B = geo.C[c1], geo.C[c2]
    j = corner_of(geo, c1, c2)
    if j is not None and not isinstance(j, tuple): k["j"] = [j, j]; return k
    if isinstance(j, tuple): k["j"] = list(j); return k
    sy = SkSys(geo)
    if A["k"] == "line" and B["k"] != "line" or B["k"] == "line" and A["k"] != "line":
        li, ri = (c1, c2) if A["k"] == "line" else (c2, c1)
        if crv_center(geo.C[ri]) is not None:
            k["sg"] = 1 if sy.sdist_line(sy.x, li, *sy.center(sy.x, ri)) >= 0 else -1
    elif is_round(A) and is_round(B):
        (ax, ay), (bx, by) = sy.center(sy.x, c1), sy.center(sy.x, c2); d = math.hypot(bx - ax, by - ay)
        r1, r2 = sy.radius(sy.x, c1), sy.radius(sy.x, c2)
        if abs(d - (r1 + r2)) <= abs(d - abs(r1 - r2)): k["io"] = 0
        else: k["io"] = 1 if r1 >= r2 else -1
    return k

def build_circle_2t(geo, l1, l2, x, y):
    """Circle tangent to two lines, centred near (x, y) on their bisector."""
    sy = SkSys(geo); d1, d2 = sy.sdist_line(sy.x, l1, x, y), sy.sdist_line(sy.x, l2, x, y)
    r = max((abs(d1) + abs(d2))/2, 1e-3)
    ci = geo.add("circle", [geo.add_pt(x, y)], r=r)
    for li in (l1, l2):
        k = tangent_con(geo, li, ci); geo.K.append(k)
    ok, _, _ = geo_solve(geo)
    return ci if ok else None

def build_circle_3t(geo, cs):
    """Circle tangent to three lines / circles (starting from their incircle)."""
    pts = []
    for ci in cs:
        c = geo.C[ci]; pts += [geo.P[i] for i in c["p"]]
    P = np.array(pts); m = P.mean(0)
    lines = [ci for ci in cs if geo.C[ci]["k"] == "line"]
    if len(lines) == 3:                                                  # incircle of the triangle the lines make
        try:
            def X(i, j):
                (a1, b1), (a2, b2) = geo.C[i]["p"], geo.C[j]["p"]
                A1, B1, A2, B2 = (np.array(geo.P[q]) for q in (a1, b1, a2, b2)); d1, d2 = B1 - A1, B2 - A2
                den = d1[0]*d2[1] - d1[1]*d2[0]; t = ((A2 - A1)[0]*d2[1] - (A2 - A1)[1]*d2[0])/den; return A1 + d1*t
            T = [X(lines[0], lines[1]), X(lines[1], lines[2]), X(lines[2], lines[0])]
            a, b, c = (np.linalg.norm(T[1] - T[2]), np.linalg.norm(T[2] - T[0]), np.linalg.norm(T[0] - T[1]))
            m = (a*T[0] + b*T[1] + c*T[2])/(a + b + c)
        except Exception: pass
    sy = SkSys(geo); ds = []
    for ci in cs:
        c = geo.C[ci]
        if c["k"] == "line": ds.append(abs(sy.sdist_line(sy.x, ci, *m)))
        elif is_round(c): ds.append(abs(math.hypot(m[0] - sy.center(sy.x, ci)[0], m[1] - sy.center(sy.x, ci)[1]) - sy.radius(sy.x, ci)))
    r = max(float(np.mean(ds)) if ds else 1.0, 1e-3)
    new = geo.add("circle", [geo.add_pt(*m)], r=r)
    for ci in cs: geo.K.append(tangent_con(geo, ci, new))
    ok, _, _ = geo_solve(geo)
    if not ok: raise RuntimeError("no circle touches all three of those curves")
    return new

def build_arc_3p(geo, a, b, m):
    cc = circumcircle(xy(a), xy(b), xy(m))
    if not cc: return None
    (ax, ay), (bx, by), (mx, my) = xy(a), xy(b), xy(m); ux, uy, r = cc
    t = lambda x, y: math.atan2(y - uy, x - ux)
    ta, tb, tm = t(ax, ay), t(bx, by), t(mx, my)
    ccw = ((tm - ta) % math.tau) < ((tb - ta) % math.tau)
    ia, ib = use_pt(geo, a), use_pt(geo, b); ic = geo.add_pt(ux, uy)
    return geo.add("arc", [ic, ia, ib] if ccw else [ic, ib, ia])

def build_arc_center(geo, c, a, b, ccw=True):
    (cx, cy), (ax, ay), (bx, by) = xy(c), xy(a), xy(b); r = math.hypot(ax - cx, ay - cy)
    if r < 1e-9: return None
    t = math.atan2(by - cy, bx - cx); ex, ey = cx + r*math.cos(t), cy + r*math.sin(t)
    ic, ia = use_pt(geo, c), use_pt(geo, a)
    ib = use_pt(geo, (ex, ey, b[2] if len(b) == 3 and b[2] and b[2][0] != "pt" else None))
    if len(b) == 3 and b[2] and b[2][0] == "pt": geo.con("coin", P_(ib), P_(b[2][1]))
    return geo.add("arc", [ic, ia, ib] if ccw else [ic, ib, ia])

def tangent_arc_geom(px, py, tx, ty, bx, by):
    """Centre and turn direction of the arc leaving (px, py) along (tx, ty) and ending at (bx, by)."""
    L = math.hypot(tx, ty); tx, ty = tx/L, ty/L; nx, ny = -ty, tx
    den = 2*(nx*(bx - px) + ny*(by - py))
    if abs(den) < 1e-12: return None
    s = ((bx - px)**2 + (by - py)**2)/den
    return px + nx*s, py + ny*s, s > 0

def end_tangent(geo, ci, pi):
    """Unit direction leaving curve ci at its end point pi (pointing away from the curve)."""
    c = geo.C[ci]; s0, s1 = crv_ends(c)
    sy = SkSys(geo); d, _ = sy.end_d(sy.x, ci, pi == s0)
    d = (-d[0], -d[1]) if pi == s0 else d
    L = math.hypot(*d) or 1.0; return d[0]/L, d[1]/L

def build_arc_tangent(geo, ci, pi, b):
    tx, ty = end_tangent(geo, ci, pi); px, py = geo.P[pi]; bx, by = xy(b)
    g = tangent_arc_geom(px, py, tx, ty, bx, by)
    if not g: return None
    cx, cy, left = g; ib = use_pt(geo, b); ic = geo.add_pt(cx, cy)
    ai = geo.add("arc", [ic, pi, ib] if left else [ic, ib, pi])
    geo.con("tan", C_(ci), C_(ai), j=[pi, pi]); return ai

def polygon_pts(cx, cy, vx, vy, n, mode):
    """Vertices of a regular polygon.  mode: 'insc' (vertex on the circle), 'circ' (edge midpoint on the circle)."""
    r = math.hypot(vx - cx, vy - cy); a0 = math.atan2(vy - cy, vx - cx)
    if mode == "circ": R = r/math.cos(math.pi/n); a0 += math.pi/n
    else: R = r
    return [(cx + R*math.cos(a0 + k*math.tau/n), cy + R*math.sin(a0 + k*math.tau/n)) for k in range(n)], r

def build_polygon(geo, c, v, n, mode="insc"):
    (cx, cy), (vx, vy) = xy(c), xy(v); n = max(3, int(n))
    if math.hypot(vx - cx, vy - cy) < 1e-9: return []
    P, r = polygon_pts(cx, cy, vx, vy, n, mode)
    ic = use_pt(geo, c); circ = geo.add("circle", [ic], r=r, cons=True)
    ids = [geo.add_pt(*q) for q in P]; L = [geo.add("line", [ids[k], ids[(k + 1) % n]]) for k in range(n)]
    for k in range(1, n): geo.con("eq", C_(L[k - 1]), C_(L[k]))
    if mode == "circ":
        for li in L: geo.K.append(tangent_con(geo, li, circ))
    else:
        for i in ids: geo.con("coin", P_(i), C_(circ))
    return L + [circ]

def build_polygon_edge(geo, a, b, n, side_pt):
    (ax, ay), (bx, by) = xy(a), xy(b); n = max(3, int(n)); L = math.hypot(bx - ax, by - ay)
    if L < 1e-9: return []
    mx, my = (ax + bx)/2, (ay + by)/2; nx, ny = -(by - ay)/L, (bx - ax)/L
    if (side_pt[0] - mx)*nx + (side_pt[1] - my)*ny < 0: nx, ny = -nx, -ny
    apo = L/(2*math.tan(math.pi/n)); cx, cy = mx + nx*apo, my + ny*apo
    a0 = math.atan2(ay - cy, ax - cx); R = math.hypot(ax - cx, ay - cy)
    turn = 1 if ((bx - cx)*(ay - cy) - (by - cy)*(ax - cx)) < 0 else -1
    ia, ib = use_pt(geo, a), use_pt(geo, b)
    ids = [ia, ib] + [geo.add_pt(cx + R*math.cos(a0 + turn*k*math.tau/n), cy + R*math.sin(a0 + turn*k*math.tau/n)) for k in range(2, n)]
    ic = geo.add_pt(cx, cy); circ = geo.add("circle", [ic], r=R, cons=True)
    Ls = [geo.add("line", [ids[k], ids[(k + 1) % n]]) for k in range(n)]
    for k in range(1, n): geo.con("eq", C_(Ls[k - 1]), C_(Ls[k]))
    for i in ids: geo.con("coin", P_(i), C_(circ))
    return Ls + [circ]

def build_ellipse(geo, c, m, p):
    (cx, cy), (mx, my), (px, py) = xy(c), xy(m), xy(p); r1 = math.hypot(mx - cx, my - cy)
    if r1 < 1e-9: return None
    ux, uy = (mx - cx)/r1, (my - cy)/r1; r2 = abs(-uy*(px - cx) + ux*(py - cy))
    if r2 < 1e-9: return None
    ci = geo.add("ellipse", [use_pt(geo, c), use_pt(geo, m)], r2=r2)
    return ci

def _slot(geo, ca, cb, r, cl=True):
    """Straight slot around centre points ca, cb (indices) of half-width r."""
    (ax, ay), (bx, by) = geo.P[ca], geo.P[cb]; L = math.hypot(bx - ax, by - ay)
    if L < 1e-9 or r < 1e-9: return []
    dx, dy = (bx - ax)/L, (by - ay)/L; nx, ny = -dy, dx
    pB1 = geo.add_pt(bx - nx*r, by - ny*r); pB2 = geo.add_pt(bx + nx*r, by + ny*r)
    pA1 = geo.add_pt(ax + nx*r, ay + ny*r); pA2 = geo.add_pt(ax - nx*r, ay - ny*r)
    aB = geo.add("arc", [cb, pB1, pB2]); aA = geo.add("arc", [ca, pA1, pA2])
    top = geo.add("line", [pB2, pA1]); bot = geo.add("line", [pA2, pB1])
    geo.con("tan", C_(top), C_(aB), j=[pB2, pB2]); geo.con("tan", C_(top), C_(aA), j=[pA1, pA1])
    geo.con("tan", C_(bot), C_(aA), j=[pA2, pA2]); geo.con("tan", C_(bot), C_(aB), j=[pB1, pB1]); geo.con("eq", C_(aA), C_(aB))
    out = [aB, aA, top, bot]
    if cl: out.append(geo.add("line", [ca, cb], cons=True))
    return out

def build_slot_cc(geo, a, b, w):
    ca, cb = use_pt(geo, a), use_pt(geo, b); out = _slot(geo, ca, cb, w/2)
    if out and (abs(geo.P[ca][1] - geo.P[cb][1]) < 1e-9 or abs(geo.P[ca][0] - geo.P[cb][0]) < 1e-9):
        geo.con("hor" if abs(geo.P[ca][1] - geo.P[cb][1]) < 1e-9 else "ver", C_(out[-1]))
    return out

def build_slot_overall(geo, a, b, w):
    (ax, ay), (bx, by) = xy(a), xy(b); L = math.hypot(bx - ax, by - ay); r = w/2
    if L <= w + 1e-9: return []
    dx, dy = (bx - ax)/L, (by - ay)/L
    ca = geo.add_pt(ax + dx*r, ay + dy*r); cb = geo.add_pt(bx - dx*r, by - dy*r); out = _slot(geo, ca, cb, r)
    if out and (abs(dy) < 1e-9 or abs(dx) < 1e-9): geo.con("hor" if abs(dy) < 1e-9 else "ver", C_(out[-1]))
    return out

def build_slot_center(geo, c, b, w):
    (cx, cy), (bx, by) = xy(c), xy(b); ic = use_pt(geo, c)
    cb = use_pt(geo, b); ca = geo.add_pt(2*cx - bx, 2*cy - by); out = _slot(geo, ca, cb, w/2)
    if out:
        geo.con("mid", P_(ic), C_(out[-1]))
        if abs(by - cy) < 1e-9 or abs(bx - cx) < 1e-9: geo.con("hor" if abs(by - cy) < 1e-9 else "ver", C_(out[-1]))
    return out

def _arc_slot(geo, O, A, B, w):
    """Curved slot along the (construction) arc O: A -> B (ccw) of width w."""
    r = w/2; (ox, oy) = geo.P[O]; R = math.hypot(geo.P[A][0] - ox, geo.P[A][1] - oy)
    if R - r <= 1e-9: raise RuntimeError("the slot is wider than its arc's radius allows")
    a0 = math.atan2(geo.P[A][1] - oy, geo.P[A][0] - ox); a1 = math.atan2(geo.P[B][1] - oy, geo.P[B][0] - ox)
    P = lambda rr, a: geo.add_pt(ox + rr*math.cos(a), oy + rr*math.sin(a))
    Ao, Ai, Bo, Bi = P(R + r, a0), P(R - r, a0), P(R + r, a1), P(R - r, a1)
    cen = geo.add("arc", [O, A, B], cons=True)
    outer = geo.add("arc", [O, Ao, Bo]); inner = geo.add("arc", [O, Ai, Bi])
    capA = geo.add("arc", [A, Ai, Ao]); capB = geo.add("arc", [B, Bo, Bi])
    geo.con("tan", C_(capA), C_(outer), j=[Ao, Ao]); geo.con("tan", C_(capA), C_(inner), j=[Ai, Ai])
    geo.con("tan", C_(capB), C_(outer), j=[Bo, Bo]); geo.con("tan", C_(capB), C_(inner), j=[Bi, Bi]); geo.con("eq", C_(capA), C_(capB))
    return [outer, inner, capA, capB, cen]

def build_slot_arc3(geo, a, b, m, w):
    cc = circumcircle(xy(a), xy(b), xy(m))
    if not cc: return []
    (ax, ay), (bx, by), (mx, my) = xy(a), xy(b), xy(m); ux, uy, r = cc
    t = lambda x, y: math.atan2(y - uy, x - ux)
    ccw = ((t(mx, my) - t(ax, ay)) % math.tau) < ((t(bx, by) - t(ax, ay)) % math.tau)
    ia, ib = use_pt(geo, a), use_pt(geo, b); O = geo.add_pt(ux, uy)
    return _arc_slot(geo, O, ia, ib, w) if ccw else _arc_slot(geo, O, ib, ia, w)

def build_slot_arcc(geo, c, a, b, w, ccw=True):
    (cx, cy), (ax, ay), (bx, by) = xy(c), xy(a), xy(b); r = math.hypot(ax - cx, ay - cy)
    t = math.atan2(by - cy, bx - cx); O = use_pt(geo, c); ia = use_pt(geo, a); ib = geo.add_pt(cx + r*math.cos(t), cy + r*math.sin(t))
    return _arc_slot(geo, O, ia, ib, w) if ccw else _arc_slot(geo, O, ib, ia, w)

def build_spline(geo, specs, closed=False):
    ids = []
    for s in specs:
        i = use_pt(geo, s)
        if not ids or i != ids[-1]: ids.append(i)
    if closed and len(ids) > 2 and ids[-1] == ids[0]: ids.pop()
    if len(ids) < 2: return None
    return geo.add("spline", ids, closed=bool(closed) and len(ids) > 2, h=[None]*len(ids), hl=0.0)

def build_cspline(geo, specs, deg=3):
    ids = [use_pt(geo, s) for s in specs]
    if len(ids) < 2: return None
    return geo.add("cspline", ids, deg=min(deg, len(ids) - 1))

def build_conic(geo, a, b, apex, rho=0.5):
    return geo.add("conic", [use_pt(geo, a), use_pt(geo, apex), use_pt(geo, b)], rho=float(rho))

def build_point(geo, a): return geo.add("point", [use_pt(geo, a)])

def build_text(geo, a, text, size, font="Sans Serif", bold=False, italic=False, ang=0.0, path=None, flip=False, pos=0.0):
    c = dict(text=text, size=float(size), font=font, bold=bool(bold), italic=bool(italic), ang=float(ang))
    if path is not None: c.update(path=path, flip=bool(flip), pos=float(pos))
    return geo.add("text", [use_pt(geo, a)], **c)

def spline_handle(geo, ci, j):
    """Switch on the tangent handle of fit point j of spline ci."""
    c = geo.C[ci]; hs = list(c.get("h") or [None]*len(c["p"]))
    while len(hs) < len(c["p"]): hs.append(None)
    if hs[j] is not None: return hs[j]
    g = geo_g2d(geo, ci)
    if not g: return None
    g = g[0]; x, y = geo.P[c["p"][j]]; u = proj_param(g, x, y); P = gp_Pnt2d(); D = gp_Vec2d(); g.D1(u, P, D)
    L = math.hypot(D.X(), D.Y()) or 1.0; P_all = np.array([geo.P[i] for i in c["p"]])
    span = float(np.hypot(*np.diff(P_all, axis=0).T).mean()) if len(P_all) > 1 else 1.0
    hl = c.get("hl") or span*0.4; c["hl"] = hl
    i = geo.add_pt(x + D.X()/L*hl, y + D.Y()/L*hl); hs[j] = i; c["h"] = hs; geo.touch(); return i

# ---- Fission 0.5 designs: old sketches keep their old profiles and curve order ----
def legacy_faces(ents, plane):
    """Profiles exactly as Fission 0.5 built them (keeps old designs' face numbering)."""
    faces = []
    for k, p in ents:
        if k == "line": continue
        try: faces.append(make_face(k, p, plane))
        except Exception: pass
    if len(faces) > 1:
        try:
            bld = BOPAlgo_Builder()
            for f in faces: bld.AddArgument(f)
            bld.Perform(); split = subshapes(bld.Shape(), TopAbs_FACE)
            if split: faces = split
        except Exception: pass
    return faces

def legacy_sketch_geo(ents):
    """v0.5 entities -> geometry with the old curve order (straight segments first, then circles).
    Returns (geo, [(kind, pts, curve ids)] in entity order)."""
    g = Geo(); made = {}
    for j, (kind, pts) in enumerate(ents):
        if kind != "circle": made[j] = geo_add_legacy(g, kind, pts)
    for j, (kind, pts) in enumerate(ents):
        if kind == "circle": made[j] = geo_add_legacy(g, kind, pts)
    return g, [(kind, pts, made[j]) for j, (kind, pts) in enumerate(ents)]

# ---- 3D sketch geometry: lines / splines off the plane, included model edges, curves projected onto faces ----
K3D = ("l3", "s3", "inc3", "proj3")
def is3d(c): return c["k"] in K3D

def interp3(P):
    """3D curve through points (local or world coordinates)."""
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.collections import HArray1_gp_Pnt
    arr = HArray1_gp_Pnt(1, len(P))
    for i, q in enumerate(P): arr.SetValue(i + 1, gp_Pnt(*map(float, q)))
    it = GeomAPI_Interpolate(arr, False, 1e-7); it.Perform(); return it.Curve()

def geo_curve_pts3(geo, ci):
    """Polylines (N x 3, sketch-local x, y, z) of a 3D sketch curve."""
    key = ("p3", ci)
    if key in geo._c: return geo._c[key]
    c = geo.C[ci]; out = []
    try:
        if c["k"] == "l3": out = [np.asarray(c["q"], float)]
        elif c["k"] == "s3":
            Q = np.asarray(c["q"], float)
            if len(Q) == 2: out = [Q]
            else:
                cv = interp3(Q); u0, u1 = cv.FirstParameter(), cv.LastParameter()
                out = [np.array([[cv.Value(float(u)).X(), cv.Value(float(u)).Y(), cv.Value(float(u)).Z()] for u in np.linspace(u0, u1, 24*len(Q))])]
        else: out = [np.asarray(q, float) for q in c.get("qs") or []]
    except Exception: out = []
    geo._c[key] = out; return out

def crv3_edges(geo, ci, plane):
    c = geo.C[ci]; W_ = lambda q: plane.w(V(*q))
    if c["k"] == "l3":
        a, b = c["q"]; return [BRepBuilderAPI_MakeEdge(pnt(W_(a)), pnt(W_(b))).Edge()]
    if c["k"] == "s3":
        Q = [W_(q).t() for q in c["q"]]
        if len(Q) == 2: return [BRepBuilderAPI_MakeEdge(gp_Pnt(*Q[0]), gp_Pnt(*Q[1])).Edge()]
        return [BRepBuilderAPI_MakeEdge(interp3(Q)).Edge()]
    out = []
    for A in geo_curve_pts3(geo, ci):
        if len(A) < 2: continue
        Wp = [W_(q).t() for q in A]
        if len(Wp) == 2: out.append(BRepBuilderAPI_MakeEdge(gp_Pnt(*Wp[0]), gp_Pnt(*Wp[1])).Edge()); continue
        from OCP.GeomAPI import GeomAPI_PointsToBSpline
        from OCP.collections import Array1_gp_Pnt
        from OCP.GeomAbs import GeomAbs_C2
        arr = Array1_gp_Pnt(1, len(Wp))
        for i, q in enumerate(Wp): arr.SetValue(i + 1, gp_Pnt(*q))
        out.append(BRepBuilderAPI_MakeEdge(GeomAPI_PointsToBSpline(arr, 3, 8, GeomAbs_C2, 1e-4).Curve()).Edge())
    return out

def edge_local_pts(edge, plane, n=None):
    """Sample a model edge into sketch-local (x, y, z) points."""
    a = BRepAdaptor_Curve(edge); u0, u1 = a.FirstParameter(), a.LastParameter()
    k = 2 if a.GetType() == GeomAbs_Line else (n or 64)
    out = []
    for u in np.linspace(u0, u1, k):
        p = a.Value(float(u)); q = plane.l(V(p.X(), p.Y(), p.Z())); out.append([q.x, q.y, q.z])
    return out

def project_to_faces(edges, faces, plane):
    """Curves on faces: project world edges along the sketch normal and keep only what the sketch 'sees' first."""
    from OCP.BRepProj import BRepProj_Projection
    from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
    from OCP.gp import gp_Lin
    ints = []
    for f in faces:
        it = IntCurvesFace_ShapeIntersector(); it.Load(f, 1e-6); ints.append(it)
    out = []
    for e in edges:
        for f in faces:
            try:
                pr = BRepProj_Projection(e, f, gdir(plane.n))
                if not pr.IsDone(): continue
                pieces = [edge_local_pts(x, plane, 96) for x in subshapes(pr.Shape(), TopAbs_EDGE)]
            except Exception: continue
            for P in pieces:
                run = []
                for q in P:
                    sg = 1.0 if q[2] >= 0 else -1.0; best = math.inf
                    lin = gp_Lin(pnt(plane.w(V(q[0], q[1], 0))), gdir(plane.n*sg))
                    for it in ints:
                        it.Perform(lin, -1e-7, 1e12)
                        for i in range(1, it.NbPnt() + 1): best = min(best, it.WParameter(i))
                    if abs(abs(q[2]) - best) <= 1e-5*(1 + best): run.append(q)
                    else:
                        if len(run) > 1: out.append(run)
                        run = []
                if len(run) > 1: out.append(run)
    return out

# ---------------------------------------------------------------------------------------------------------------
#  Surface modelling: open (zero-thickness) bodies - extrude / revolve / sweep / loft / patch / ruled / offset,
#  trim, untrim, extend, stitch, unstitch, reverse normal, delete face
# ---------------------------------------------------------------------------------------------------------------
from OCP.BRepFill import BRepFill_Filling, BRepFill
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.collections import HSequence_TopoDS_Shape as TopTools_HSequenceOfShape
from OCP.collections import IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as TopTools_IndexedDataMapOfShapeListOfShape
from OCP.TopExp import TopExp
from OCP.TopAbs import TopAbs_SHELL, TopAbs_IN, TopAbs_ON
from OCP.BRepLib import BRepLib
from OCP.BRepTools import BRepTools
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.GeomAbs import GeomAbs_C0, GeomAbs_G1, GeomAbs_G2
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
from OCP.BRepOffset import BRepOffset_Skin
from OCP.GeomAbs import GeomAbs_Intersection
from OCP.TopoDS import TopoDS_Face
from OCP.TopAbs import TopAbs_FORWARD
from OCP.ShapeFix import ShapeFix_Face
from OCP.BRepTools import BRepTools_ReShape

def is_surface_shape(sh):
    """True when a shape has faces but no closed solid (a surface body)."""
    return not subshapes(sh, TopAbs_SOLID) and bool(subshapes(sh, TopAbs_FACE))

def surface_area(sh):
    p = GProp_GProps(); surface_props(sh, p); return p.Mass()

def sew(shapes, tol=1e-3, solid_if_closed=True):
    """Sew faces / shells together. Closed results become solids (like Fusion's Stitch)."""
    sw = BRepBuilderAPI_Sewing(tol)
    for x in shapes:
        for f in subshapes(x, TopAbs_FACE): sw.Add(f)
    sw.Perform(); out = sw.SewedShape()
    if solid_if_closed:
        shells = subshapes(out, TopAbs_SHELL)
        if shells and all(st(BRep_Tool, "IsClosed")(sh) for sh in shells):
            try:
                sol = as_solid(out)
                if abs(volume(sol)) > 1e-9: return sol
            except Exception: pass
    return out

def free_edges(sh):
    """Edges used by only one face (the open boundary of a surface)."""
    m = TopTools_IndexedDataMapOfShapeListOfShape(); TopExp.MapShapesAndAncestors_s(sh, TopAbs_EDGE, TopAbs_FACE, m)
    out = []
    for i in range(1, m.Extent() + 1):
        e = m.FindKey(i)
        if m.FindFromIndex(i).Extent() == 1 and not degenerated(to_edge(e)): out.append(to_edge(e))
    return out

def faces_of_edge_in(sh, e):
    m = TopTools_IndexedDataMapOfShapeListOfShape(); TopExp.MapShapesAndAncestors_s(sh, TopAbs_EDGE, TopAbs_FACE, m)
    i = m.FindIndex(e)
    if i == 0: return []
    return [to_face(x) for x in m.FindFromIndex(i)]

def edges_to_wires(edges, tol=1e-4):
    """Join loose edges into as few wires as possible."""
    if not edges: return []
    seq = TopTools_HSequenceOfShape()
    for e in edges: seq.Append(e)
    out = TopTools_HSequenceOfShape(); ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(seq, tol, False, out)
    return [to_wire(out.Value(i)) for i in range(1, out.Length() + 1)]

def wire_closed(w):
    a, b = None, None
    try: return bool(st(BRep_Tool, "IsClosed")(w))
    except Exception: pass
    es = subshapes(w, TopAbs_EDGE); p0, _ = edge_ends(es[0]); _, p1 = edge_ends(es[-1])
    return (p0 - p1).Length < 1e-4

def wire_plane_normal(w):
    """Normal of the plane a wire lies in (None when it isn't flat or is a straight line)."""
    from OCP.BRepLib import BRepLib_FindSurface
    fs = BRepLib_FindSurface(w, 1e-4, True)
    if not fs.Found(): return None
    srf = BRepAdaptor_Surface(BRepBuilderAPI_MakeFace(fs.Surface(), 1e-6).Face())
    if srf.GetType() != GeomAbs_Plane: return None
    return vec(srf.Plane().Axis().Direction())

def _shell_of(faces_shape):
    """Faces from a sweep / prism / loft as one connected shell where possible."""
    fs = subshapes(faces_shape, TopAbs_FACE)
    if not fs: raise RuntimeError("that made no surface")
    return sew([faces_shape], solid_if_closed=False) if len(fs) > 1 else fs[0]

def surf_extrude(edges, n, d1, d2=0.0, taper=0.0):
    """Extrude curves into an open surface: d1 along n, d2 the other way (two-sided / symmetric)."""
    if abs(d1) < 1e-9 and abs(d2) < 1e-9: raise RuntimeError("the distance is zero")
    out = []
    for w in edges_to_wires(edges):
        base = translated(w, n*(-d2)) if abs(d2) > 1e-12 else w; L = d1 + d2
        if abs(taper) > 1e-9 and wire_closed(w) and wire_plane_normal(w) is not None:
            f = BRepBuilderAPI_MakeFace(to_wire(base), True).Face()
            top = translated(offset_face(f, -L*math.tan(math.radians(taper))), n*L)
            ts = BRepOffsetAPI_ThruSections(False, True, 1e-6); ts.AddWire(to_wire(base)); ts.AddWire(wires_of(top)[0]); ts.Build()
            out.append(ts.Shape())
        else:
            out.append(BRepPrimAPI_MakePrism(base, gp_Vec(n.x*L, n.y*L, n.z*L)).Shape())
    return _shell_of(compound(out))

def surf_revolve(edges, o, axis, ang, start=0.0):
    out = []
    for w in edges_to_wires(edges):
        if abs(start) > 1e-9: w = rotated(w, o, axis, math.radians(start))
        out.append(BRepPrimAPI_MakeRevol(w, gp_Ax1(pnt(o), gdir(axis)), math.radians(ang)).Shape())
    return _shell_of(compound(out))

def surf_sweep(profile_edges, path, orient="perp", frac=1.0):
    path = trim_wire(path, frac) if frac < 1 - 1e-9 else path; out = []
    for w in edges_to_wires(profile_edges):
        ps = BRepOffsetAPI_MakePipeShell(path)
        if orient == "parallel":
            nn = wire_plane_normal(w)
            if nn is not None: ps.SetMode(gp_Ax2(pnt(centroid(w)), gdir(nn)))
        ps.SetTransitionMode(BRepBuilderAPI_RightCorner); ps.Add(w, False, False); ps.Build()
        if not ps.IsDone(): raise RuntimeError("the sweep failed - is the profile at the start of the path?")
        out.append(ps.Shape())
    return _shell_of(compound(out))

def surf_loft(sections, closed=False, ruled=False):
    """Open loft through wires (open or closed) or end points."""
    if len(sections) < 2: raise RuntimeError("pick at least two sections")
    ts = BRepOffsetAPI_ThruSections(False, ruled, 1e-6); ts.CheckCompatibility(True)
    for s_ in sections + ([sections[0]] if closed else []):
        if s_.ShapeType() == TopAbs_VERTEX: ts.AddVertex(to_vertex(s_))
        else: ts.AddWire(to_wire(s_))
    ts.Build()
    if not ts.IsDone(): raise RuntimeError("the loft failed - do the sections all run the same way?")
    return _shell_of(ts.Shape())

def surf_patch(edges, support=None, cont="G0", interior=()):
    """Fill a closed boundary. support: [(edge, adjacent face)] for tangent (G1) / curvature (G2) blending."""
    ws = edges_to_wires(edges)
    if len(ws) != 1 or not wire_closed(ws[0]): raise RuntimeError("the boundary must be one closed loop")
    w = ws[0]
    if cont == "G0" and not interior and wire_plane_normal(w) is not None:
        mk = BRepBuilderAPI_MakeFace(w, True)
        if mk.IsDone(): return mk.Face()
    fl = BRepFill_Filling(); gc = {"G0": GeomAbs_C0, "G1": GeomAbs_G1, "G2": GeomAbs_G2}[cont]
    def sup(e):
        mid = lambda x: vec(BRepAdaptor_Curve(x).Value((BRepAdaptor_Curve(x).FirstParameter() + BRepAdaptor_Curve(x).LastParameter())/2))
        for se, sf in support or []:
            if se.IsSame(e) or (mid(se) - mid(e)).Length < 1e-6: return se, sf
        return e, None
    for e in subshapes(w, TopAbs_EDGE):
        e, f = sup(e)
        if f is not None and cont != "G0": fl.Add(e, f, gc, True)
        else: fl.Add(e, GeomAbs_C0, True)
    for q in interior: fl.Add(pnt(q))
    fl.Build()
    if not fl.IsDone(): raise RuntimeError("couldn't fill that boundary")
    return fl.Face()

def _outward_dir(face, e, p):
    """Unit vector at p (on edge e of face) lying in the surface, across the edge, pointing away from the face."""
    q, n = normal_at(face, p); c = BRepAdaptor_Curve(e)
    u = _edge_param(e, p); pp, T = gp_Pnt(), gp_Vec(); c.D1(u, pp, T); t = vec(T).normalize()
    out = t.cross(n).normalize(); eps = max(1e-3, (bbox(face)[1] - bbox(face)[0]).Length*1e-3)
    cl = BRepClass_FaceClassifier(face, pnt(p + out*eps), 1e-6)
    if cl.State() == TopAbs_IN: out = -out
    return out, n

def _edge_param(e, p):
    c = BRepAdaptor_Curve(e); u0, u1 = c.FirstParameter(), c.LastParameter(); best, bu = 1e18, u0
    for k in range(33):
        u = u0 + (u1 - u0)*k/32; d = (vec(c.Value(u)) - p).Length
        if d < best: best, bu = d, u
    return bu

def ruled_strip(e, face, d, kind="tangent", ang=0.0, direction=None, flip=False):
    """A ruled face hanging off edge e: along the surface (tangent), its normal, or a fixed direction, tilted by ang."""
    c = BRepAdaptor_Curve(e); u0, u1 = c.FirstParameter(), c.LastParameter()
    straight = c.GetType() == GeomAbs_Line and (face is None or BRepAdaptor_Surface(face).GetType() == GeomAbs_Plane or kind == "direction")
    N = 2 if straight else 33; P, Q = [], []
    ref_out = None
    for k in range(N):
        u = u0 + (u1 - u0)*k/(N - 1); p = vec(c.Value(u))
        if kind == "direction":
            dv = direction.normalize()
        else:
            if face is None: raise RuntimeError("pick edges of a surface or body")
            out, n = _outward_dir(face, e, p)
            if ref_out is not None and out.dot(ref_out) < 0: out = -out
            ref_out = out
            a = math.radians(ang); base = out if kind == "tangent" else n
            other = n if kind == "tangent" else out
            dv = (base*math.cos(a) + other*math.sin(a)).normalize()
        if flip: dv = -dv
        P.append(p); Q.append(p + dv*d)
    if N == 2: e2 = BRepBuilderAPI_MakeEdge(pnt(Q[0]), pnt(Q[1])).Edge()
    else: e2 = BRepBuilderAPI_MakeEdge(interp3([q.t() for q in Q])).Edge()
    return BRepFill.Face_s(to_edge(e.Oriented(TopAbs_FORWARD)), e2)

def surf_offset(faces, d):
    """Offset copy of faces (a new surface)."""
    sh = sew(faces, solid_if_closed=False) if len(faces) > 1 else faces[0]
    mk = BRepOffsetAPI_MakeOffsetShape(); mk.PerformByJoin(sh, d, 1e-5, BRepOffset_Skin, False, False, GeomAbs_Intersection)
    if not mk.IsDone():
        mk = BRepOffsetAPI_MakeOffsetShape(); mk.PerformBySimple(sh, d)
        if not mk.IsDone(): raise RuntimeError("couldn't offset those faces")
    out = mk.Shape()
    if not subshapes(out, TopAbs_FACE): raise RuntimeError("the offset vanished - try a smaller distance")
    return out

def face_groups(faces, skip=None):
    """Connected groups (by shared edges) of a list of faces; edges where skip(edge) is true don't connect."""
    if not faces: return []
    m = TopTools_IndexedDataMapOfShapeListOfShape(); TopExp.MapShapesAndAncestors_s(compound(faces), TopAbs_EDGE, TopAbs_FACE, m)
    idx = ShapeIndex()
    for f in faces: idx.add(f)
    par = list(range(len(faces)))
    def root(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(1, m.Extent() + 1):
        fs = [idx.find(x) for x in m.FindFromIndex(i)]; fs = [j for j in fs if j >= 0]
        if len(fs) > 1 and skip is not None and skip(to_edge(m.FindKey(i))): continue
        for j in fs[1:]: par[root(j)] = root(fs[0])
    groups = {}
    for i in range(len(faces)): groups.setdefault(root(i), []).append(faces[i])
    return list(groups.values())

def trim_tool_from_curve(edge, n, size):
    """A sketch curve turned into a cutting wall along its sketch normal."""
    return BRepPrimAPI_MakePrism(translated(edge, n*(-size)), gp_Vec(n.x*2*size, n.y*2*size, n.z*2*size)).Shape()

def split_pieces(target, tools):
    """Split a surface by tools; returns connected pieces (each a face or shell), in a stable order."""
    sp = BOPAlgo_Splitter(); sp.AddArgument(target)
    for t in tools: sp.AddTool(t)
    sp.Perform()
    if sp.HasErrors(): raise RuntimeError("couldn't trim with that tool")
    faces = subshapes(sp.Shape(), TopAbs_FACE); tc = compound(list(tools))
    def on_tool(e):
        c = BRepAdaptor_Curve(e); q = c.Value((c.FirstParameter() + c.LastParameter())/2)
        return shape_dist(tc, BRepBuilderAPI_MakeVertex(q).Vertex()) < 1e-5
    pieces = [compound(g) if len(g) > 1 else g[0] for g in face_groups(faces, on_tool)]
    pieces = [sew([p], solid_if_closed=False) if p.ShapeType() != TopAbs_FACE else p for p in pieces]
    pieces.sort(key=lambda p: tuple(round(x, 4) for x in centroid(p).t()))
    return pieces

def piece_at(pieces, q):
    v = BRepBuilderAPI_MakeVertex(pnt(q)).Vertex()
    return min(range(len(pieces)), key=lambda i: shape_dist(pieces[i], v))

def untrim_face(face, mode="loops", ext=0.0):
    """mode 'loops': fill the holes; 'all': back to the surface's natural boundary (planes / infinite surfaces grow by ext)."""
    face = to_face(face)
    if mode == "loops":
        mk = BRepBuilderAPI_MakeFace(st(BRep_Tool, "Surface")(face), outer_wire(face), True)
        f = mk.Face()
        if face.Orientation() == TopAbs_REVERSED: f = to_face(f.Reversed())
        fix = ShapeFix_Face(f); fix.Perform(); return fix.Face()
    srf = BRepAdaptor_Surface(face); u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    tp = srf.GetType()
    if tp == GeomAbs_Plane:
        lo, hi = bbox(face); m = max(ext, 0.0)
        c, n = face_frame(face); pl = srf.Plane()
        f = BRepBuilderAPI_MakeFace(pl, u0 - m, u1 + m, v0 - m, v1 + m).Face()
    else:
        U0, U1, V0, V1 = srf.FirstUParameter(), srf.LastUParameter(), srf.FirstVParameter(), srf.LastVParameter()
        big = 1e6
        if abs(U0) > big or abs(U1) > big: U0, U1 = u0 - ext, u1 + ext
        if abs(V0) > big or abs(V1) > big: V0, V1 = v0 - ext, v1 + ext
        f = BRepBuilderAPI_MakeFace(st(BRep_Tool, "Surface")(face), U0, U1, V0, V1, 1e-6).Face()
    if face.Orientation() == TopAbs_REVERSED: f = to_face(f.Reversed())
    return f

def replace_faces(shape, mapping, sew_after=True):
    """New shape with some faces swapped (mapping: list of (old face, new face or None to delete))."""
    rs = BRepTools_ReShape()
    for old, new in mapping:
        if new is None: rs.Remove(old)
        else: rs.Replace(old, new)
    out = rs.Apply(shape)
    if sew_after and len(subshapes(out, TopAbs_FACE)) > 1: out = sew([out], solid_if_closed=False)
    return out

def extend_edges(shape, edge_list, d, kind="natural"):
    """Extend a surface past some of its open edges by d."""
    strips, mapping, done = [], [], ShapeIndex()
    for e in edge_list:
        fs = faces_of_edge_in(shape, e)
        if len(fs) != 1: raise RuntimeError("only open (boundary) edges of a surface can be extended")
        f = fs[0]
        if kind == "natural":
            g = _natural_extend(f, e, d)
            if g is not None and done.find(f) < 0:
                done.add(f); mapping.append((f, g)); continue
        strips.append(ruled_strip(e, f, d, "tangent" if kind != "perpendicular" else "normal"))
    out = replace_faces(shape, mapping, sew_after=False) if mapping else shape
    out = sew([out] + strips, solid_if_closed=False) if strips or mapping else out
    return clean(out)

def _natural_extend(f, e, d):
    """Grow the face along the surface past edge e when e lies on the face's natural UV boundary; else None."""
    srf = BRepAdaptor_Surface(f); u0, u1, v0, v1 = BRepTools.UVBounds_s(f)
    natural = BRepBuilderAPI_MakeFace(st(BRep_Tool, "Surface")(f), u0, u1, v0, v1, 1e-6).Face()
    if abs(surface_area(natural) - surface_area(f)) > 1e-4*max(surface_area(f), 1e-9): return None
    c = BRepAdaptor_Curve(e); pm = vec(c.Value((c.FirstParameter() + c.LastParameter())/2))
    pr = GeomAPI_ProjectPointOnSurf(pnt(pm), st(BRep_Tool, "Surface")(f))
    if not pr.NbPoints(): return None
    u, v = pr.LowerDistanceParameters(); tu, tv = (u1 - u0)*1e-4 + 1e-9, (v1 - v0)*1e-4 + 1e-9
    side = (abs(u - u0) < tu, abs(u - u1) < tu, abs(v - v0) < tv, abs(v - v1) < tv)
    if sum(side) != 1: return None
    g = TopoDS_Face(); BRepLib.ExtendFace_s(natural, d, *side, g)
    if g.IsNull(): return None
    if f.Orientation() == TopAbs_REVERSED: g = to_face(g.Reversed())
    return g

def reversed_faces(shape, faces=None):
    """Flip the normals of some (or all) faces."""
    if faces is None: return shape.Reversed()
    from OCP.TopoDS import TopoDS_Shell
    sh, bb = TopoDS_Shell(), BRep_Builder(); bb.MakeShell(sh)
    for f in subshapes(shape, TopAbs_FACE): bb.Add(sh, f.Reversed() if any(f.IsSame(x) for x in faces) else f)
    return sh

def delete_faces_open(shape, faces):
    """Remove faces without healing - solids become open surfaces."""
    keep = [f for f in subshapes(shape, TopAbs_FACE) if all(not f.IsSame(x) for x in faces)]
    if not keep: return []
    return [sew(g, solid_if_closed=False) if len(g) > 1 else g[0] for g in face_groups(keep)]

def _emid(e):
    c = BRepAdaptor_Curve(e); return vec(c.Value((c.FirstParameter() + c.LastParameter())/2))

def edge_same_geom(a, b, tol=1e-6):
    pa, qa = edge_ends(a); pb, qb = edge_ends(b)
    return (_emid(a) - _emid(b)).Length < tol and min((pa - pb).Length + (qa - qb).Length, (pa - qb).Length + (qa - pb).Length) < 2*tol

def chain_from_pool(seed, pool, tol=1e-5):
    """Seed edges plus every pool edge connected to them end to end."""
    out = list(seed); ends = [edge_ends(e) for e in out]; rest = [e for e in pool if not any(e.IsSame(x) or edge_same_geom(e, x) for x in out)]
    grew = True
    while grew:
        grew = False
        for e in list(rest):
            p, q = edge_ends(e)
            if any(min((p - a).Length, (p - b).Length, (q - a).Length, (q - b).Length) < tol for a, b in ends):
                out.append(e); ends.append((p, q)); rest.remove(e); grew = True
    return out

# ---------------------------------------------------------------------------------------------------------------
#  Model objects
# ---------------------------------------------------------------------------------------------------------------
class Body:
    """A solid plus cached display geometry (numpy triangle arrays, edge polylines) and pick data."""
    def __init__(s, shape):
        s.shape, s.visible, s.dl = shape, True, None
        lo, hi = bbox(shape); s.lo, s.hi = lo, hi
        s.defl = min(0.05, max(0.003, (hi - lo).Length*0.0012))
        BRepMesh_IncrementalMesh(shape, s.defl, False, 0.25, True)
        s.faces = subshapes(shape, TopAbs_FACE)
        tv, vn, fid = [], [], []
        for fi, f in enumerate(s.faces):
            m = face_mesh(f)
            if m is None: continue
            P, T = m; tri = P[T]
            nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            keys, inv = np.unique(np.round(P, 4), axis=0, return_inverse=True); inv = inv.reshape(-1)
            acc = np.zeros((len(keys), 3))
            for c in range(3): np.add.at(acc, inv[T[:, c]], nn)            # area-weighted smooth normals within a face
            acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
            tv.append(tri); vn.append(acc[inv[T]]); fid.append(np.full(len(T), fi))
        s.tv = np.concatenate(tv) if tv else np.zeros((0, 3, 3))
        s.vn = np.concatenate(vn) if vn else np.zeros((0, 3, 3))
        s.tri_face = np.concatenate(fid) if fid else np.zeros(0, dtype=int)
        n = np.cross(s.tv[:, 1] - s.tv[:, 0], s.tv[:, 2] - s.tv[:, 0])
        s.tn = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        s.zmin = float(s.tv[:, :, 2].min()) if len(s.tv) else 0.0
        seams = ShapeIndex()                                              # seam = cylinder / sphere wrap line
        for f in s.faces:
            for e in subshapes(f, TopAbs_EDGE):
                if is_closed_on(e, f): seams.add(e)
        s.eds = [e for e in subshapes(shape, TopAbs_EDGE) if not degenerated(e) and seams.find(e) < 0]
        s.edges = [edge_pts(e, s.defl) for e in s.eds]
        segs = [(pl[:-1], pl[1:], np.full(len(pl) - 1, i)) for i, pl in enumerate(s.edges) if len(pl) > 1]
        s.sa = np.concatenate([a for a, _, _ in segs]) if segs else np.zeros((0, 3))
        s.sb = np.concatenate([b for _, b, _ in segs]) if segs else np.zeros((0, 3))
        s.se = np.concatenate([e for _, _, e in segs]) if segs else np.zeros(0, dtype=int)
        s.verts = np.array([st(BRep_Tool, "Pnt")(v).Coord() for v in subshapes(shape, TopAbs_VERTEX)]) if subshapes(shape, TopAbs_VERTEX) else np.zeros((0, 3))
        s.color, s.material, s.face_colors = None, None, {}
        s.surface = is_surface_shape(shape)                                # open (zero-thickness) body

    def copy_look(s, o):
        s.comp = getattr(o, "comp", None)
        if getattr(o, "sheet", None): s.sheet = o.sheet
        s.color, s.material, s.face_colors = o.color, o.material, {}
        s.name = getattr(o, "name", None)
        return s

    def restyled(s):
        nb = copy.copy(s); nb.dl = None; nb.face_colors = dict(s.face_colors); return nb

    def mass(s):
        dens = dict(MATERIALS).get(s.material)
        return abs(volume(s.shape))/1000.0*dens if dens else None

    def face_normal(s, fid):
        """Outward normal if the face is planar, else None."""
        try:
            if BRepAdaptor_Surface(s.faces[fid]).GetType() != GeomAbs_Plane: return None
        except Exception: return None
        ns = s.tn[s.tri_face == fid]
        if not len(ns): return None
        m = ns.mean(0); return V(*m).normalize()

    def face_point(s, fid):
        t = s.tv[s.tri_face == fid]
        return V(*t[0, 0]) if len(t) else V()

    def face_cylinder(s, fid):
        """(axis origin, axis dir, x dir, radius, v0, v1, external) for a cylindrical face, else None."""
        try:
            srf = BRepAdaptor_Surface(s.faces[fid])
            if srf.GetType() != GeomAbs_Cylinder: return None
            cy = srf.Cylinder(); ax = cy.Position()
            loc, d, x = ax.Location(), ax.Direction(), ax.XDirection()
            o, a, xd = V(loc.X(), loc.Y(), loc.Z()), V(d.X(), d.Y(), d.Z()), V(x.X(), x.Y(), x.Z())
            v0, v1 = srf.FirstVParameter(), srf.LastVParameter()
        except Exception:
            return None
        idx = np.nonzero(s.tri_face == fid)[0]
        if not len(idx): return None
        p, n = V(*s.tv[idx[0]].mean(0)), V(*s.tn[idx[0]])
        rad = (p - o) - a*((p - o).dot(a))
        return o, a, xd, cy.Radius(), min(v0, v1), max(v0, v1), n.dot(rad) > 0

    def face_edges(s, fid):
        idx = ShapeIndex()
        for e in s.eds: idx.add(e)
        return [i for i in (idx.find(e) for e in subshapes(s.faces[fid], TopAbs_EDGE)) if i >= 0]

    def faces_of_edge(s, e):
        return [f for f in s.faces if any(e.IsSame(x) for x in subshapes(f, TopAbs_EDGE))]

def sketch_segments(sk):
    """Every straight sketch line as (name, a, b) in sketch coordinates - candidate revolve axes."""
    out, n, g = [], 0, sk.geo
    for c in g.C:
        if c["k"] != "line": continue
        n += 1; a, b = g.v(c["p"][0]), g.v(c["p"][1])
        if (b - a).Length < 1e-9: continue
        out.append((f"{'Centerline' if c.get('cl') else 'Construction line' if c.get('cons') else 'Line'} {n}", a, b))
    return out

class Region:
    """One selectable closed area of a sketch (after splitting overlapping / nested profiles)."""
    def __init__(s, face, plane):
        s.face = face
        BRepMesh_IncrementalMesh(face, 0.02, False, 0.3, True)
        m = face_mesh(face); P, T = m if m else (np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
        loc = np.array([plane.l(V(*p)).t() for p in P]) if len(P) else np.zeros((0, 3))
        if len(loc): loc[:, 2] = 0
        s.tris = loc[T].astype(np.float32) if len(T) else np.zeros((0, 3, 3), np.float32)
        s.outline = []
        for e in subshapes(face, TopAbs_EDGE):
            pts = edge_pts(e, 0.02)
            if len(pts) > 1:
                q = np.array([plane.l(V(*p)).t() for p in pts]); q[:, 2] = 0; s.outline.append(q)
        a, b, c = s.tris[:, 0, :2], s.tris[:, 1, :2], s.tris[:, 2, :2]
        s.area = float(np.abs(cross2(b - a, c - a)).sum() / 2) if len(s.tris) else 0.0

    def contains(s, x, y):
        if not len(s.tris): return False
        a, b, c = s.tris[:, 0, :2], s.tris[:, 1, :2], s.tris[:, 2, :2]; p = np.array([x, y])
        d1, d2, d3 = cross2(b - a, p - a), cross2(c - b, p - b), cross2(a - c, p - c)
        neg, pos = (d1 < 0) | (d2 < 0) | (d3 < 0), (d1 > 0) | (d2 > 0) | (d3 > 0)
        return bool(np.any(~(neg & pos)))

class Sketch:
    """A sketch: plane + parametric geometry (points, curves, constraints, dimensions)."""
    def __init__(s, plane, parent=None, geo=None):
        s.plane, s.parent, s.geo, s.visible, s.sel = plane, parent, geo if geo is not None else Geo(), True, set()
        s._reg, s._key, s.pending, s.pending_op, s.name = None, None, False, None, None
        s.legacy, s.lgeo = [], None                           # v0.5 entities [(kind, pts, curve ids)] and the geometry they made

    @property
    def entities(s): return s.geo.C

    def regions(s):
        """Closed areas of the sketch (cached per geometry version)."""
        key = (id(s.geo), s.geo.ver)
        if s._key == key: return s._reg
        regs = []
        try: faces = legacy_faces([L[:2] for L in s.legacy], s.plane) if s.legacy and s.lgeo is s.geo else geo_regions(s.geo, s.plane)
        except Exception: traceback.print_exc(); faces = []
        for f in faces:
            try:
                r = Region(f, s.plane)
                if r.area > 1e-9: regs.append(r)
            except Exception: traceback.print_exc()
        if s._reg is None or len(regs) != len(s._reg): s.sel = set()
        s._reg, s._key = regs, key
        return regs

    def chosen(s):
        regs = s.regions()
        return [regs[i] for i in sorted(s.sel) if i < len(regs)] or regs

    def region_at(s, g):
        for i, r in enumerate(s.regions()):
            if r.contains(g.x, g.y): return i
        return None

    def snap(s): return (s.plane, s.parent, s.geo, s.visible, s.name, list(s.legacy), s.lgeo)
    @staticmethod
    def restore(t):
        sk = Sketch(t[0], t[1], t[2]); sk.visible = t[3]
        if len(t) > 4: sk.name = t[4]
        if len(t) > 6: sk.legacy, sk.lgeo = list(t[5]), t[6]
        return sk

# ---------------------------------------------------------------------------------------------------------------
#  Small form dialog used by every command that needs numbers
# ---------------------------------------------------------------------------------------------------------------
class LengthSpin(W.QDoubleSpinBox):
    """Number box for lengths: shows the display unit and accepts any unit typed in (cm, m, in, ft, ...).
    value() / setValue() are always millimetres."""
    def __init__(s, lo=-100000.0, hi=100000.0):
        super().__init__(); s.setDecimals(6); s.setRange(lo, hi); s.setKeyboardTracking(False); s.refresh_units()
        s.setToolTip("Type a length in any unit: 12, 12 mm, 3.5 cm, 0.25 m, 1/2 in, 2 ft, 1 in + 3 mm")
    def refresh_units(s):
        s.setSingleStep(1.0 if DISPLAY["unit"] == "mm" else 2.54); s.lineEdit().setText(s.textFromValue(s.value()))
    def textFromValue(s, v): return flen(v, 4 if DISPLAY["unit"] == "mm" else 4)
    def valueFromText(s, text):
        try: return parse_len(text)
        except Exception: return s.value()
    def validate(s, text, pos):
        try: parse_len(text); return G.QValidator.State.Acceptable
        except Exception: return G.QValidator.State.Intermediate
    def fixup(s, text): return s.textFromValue(s.value())

def form_dialog(parent, title, fields, note=None):
    """fields: (key, label, kind, default[, options]); kind in mm/pos/deg/int/combo/check/text. Returns dict or None."""
    dlg = W.QDialog(parent); dlg.setWindowTitle(title); dlg.setMinimumWidth(330)
    form = W.QFormLayout(dlg); form.setHorizontalSpacing(14); widgets = {}
    if note:
        lb = W.QLabel(note); lb.setWordWrap(True); lb.setObjectName("hint"); form.addRow(lb)
    for f in fields:
        key, label, kind, val = f[:4]
        if kind in ("mm", "pos"):
            b = LengthSpin(0.0 if kind == "pos" else -100000.0); b.setValue(val)
        elif kind == "deg":
            b = W.QDoubleSpinBox(); b.setDecimals(2); b.setRange(-360, 360); b.setSuffix(" °"); b.setValue(val)
        elif kind == "int":
            b = W.QSpinBox(); b.setRange(*(f[4] if len(f) > 4 else (1, 1000))); b.setValue(int(val))
        elif kind == "combo":
            b = W.QComboBox()
            for lab, data in f[4]: b.addItem(lab, data)
            i = b.findData(val); b.setCurrentIndex(max(i, 0))
        elif kind == "check":
            b = W.QCheckBox(); b.setChecked(bool(val))
        else:
            b = W.QLineEdit(str(val))
        form.addRow(label, b); widgets[key] = (kind, b)
    bb = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel)
    bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); form.addRow(bb)
    first = next((b for k, b in widgets.values() if k in ("mm", "pos", "deg", "int")), None)
    if first: first.setFocus(); first.selectAll()
    if not dlg.exec(): return None
    out = {}
    for key, (kind, b) in widgets.items():
        out[key] = (b.value() if kind in ("mm", "pos", "deg", "int") else b.currentData() if kind == "combo"
                    else b.isChecked() if kind == "check" else b.text())
    return out

# ---------------------------------------------------------------------------------------------------------------
#  ViewCube
# ---------------------------------------------------------------------------------------------------------------
class ViewCube(W.QWidget):
    """Fusion-style ViewCube: shows the orientation; click a face, an edge or a corner to look from that direction,
    house = home view, curved arrows turn the model 90 degrees left / right."""
    NAMES = {(0, 0, 1): "TOP", (0, 0, -1): "BOTTOM", (0, -1, 0): "FRONT", (0, 1, 0): "BACK", (1, 0, 0): "RIGHT", (-1, 0, 0): "LEFT"}
    CX, CY, K, EDGE = 75, 64, 30, 0.62               # edge / corner strips are the outer 38% of each face
    ROT_L, ROT_R, HOME = C.QRectF(4, 98, 30, 30), C.QRectF(116, 98, 30, 30), C.QRectF(2, 2, 22, 22)
    def __init__(s, vp):
        super().__init__(vp); s.vp, s.hov = vp, None
        s.setFixedSize(150, 130); s.setMouseTracking(True); s.setAttribute(C.Qt.WA_TranslucentBackground)

    def faces(s):
        """Visible faces: (normal, name, face polygon, visibility, centre, [(direction, cell polygon), ...])."""
        y, t = s.vp.yaw, s.vp.pitch
        d = V(math.cos(t)*math.cos(y), math.cos(t)*math.sin(y), math.sin(t))        # unit vector toward the camera
        r = V(-math.sin(y), math.cos(y), 0); u = r.cross(-d)                         # screen right / up
        E, out, k = (V(1, 0, 0), V(0, 1, 0), V(0, 0, 1)), [], s.K
        pr = lambda q: C.QPointF(s.CX + q.dot(r)*k, s.CY - q.dot(u)*k)
        cuts = ((-1, -s.EDGE, -1), (-s.EDGE, s.EDGE, 0), (s.EDGE, 1, 1))
        for n, name in s.NAMES.items():
            nv = V(*n); vis = nv.dot(d)
            if vis <= 1e-3: continue
            ia, ib = [i for i in range(3) if n[i] == 0]; a, b = E[ia], E[ib]
            poly = G.QPolygonF([pr(nv + a*i + b*j) for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            cells = []
            for a0, a1, si in cuts:
                for b0, b1, sj in cuts:
                    dirn = [0, 0, 0]; dirn[ia], dirn[ib] = si, sj
                    for c in range(3): dirn[c] += n[c]
                    cells.append((tuple(dirn), G.QPolygonF([pr(nv + a*i + b*j) for i, j in ((a0, b0), (a1, b0), (a1, b1), (a0, b1))])))
            out.append((n, name, poly, vis, pr(nv), cells))
        return out

    @staticmethod
    def describe(dirn):
        words = [w for c, (neg, pos) in zip(dirn, (("Left", "Right"), ("Front", "Back"), ("Bottom", "Top"))) if c for w in [pos if c > 0 else neg]]
        order = {"Top": 0, "Bottom": 0, "Front": 1, "Back": 1, "Left": 2, "Right": 2}
        words.sort(key=order.get)
        return " ".join(words) + (" view" if len(words) == 1 else " edge" if len(words) == 2 else " corner")

    def paintEvent(s, _):
        p = G.QPainter(s); p.setRenderHint(G.QPainter.Antialiasing)
        for n, name, poly, vis, c, cells in s.faces():
            p.setPen(_pen("#8d949b", 1.1)); p.setBrush(G.QColor(244, 245, 246, 238)); p.drawPolygon(poly)
            if isinstance(s.hov, tuple):
                p.setPen(C.Qt.NoPen); p.setBrush(G.QColor(6, 150, 215, 170))
                for dirn, cp in cells:
                    if dirn == s.hov: p.drawPolygon(cp)
            p.setPen(_pen("#8d949b", 1.1)); p.setBrush(C.Qt.NoBrush); p.drawPolygon(poly)
            if vis > 0.3:
                hot = s.hov == n
                f = p.font(); f.setPixelSize(int(7 + 4*vis)); f.setBold(True); p.setFont(f)
                p.setPen(G.QColor("white" if hot else "#4a5159")); p.drawText(C.QRectF(c.x() - 34, c.y() - 9, 68, 18), C.Qt.AlignCenter, name)
        for key, rect, kind in (("HOME", s.HOME, "home"), ("ROTL", s.ROT_L, "rot_l"), ("ROTR", s.ROT_R, "rot_r")):
            if key == s.hov:
                p.setPen(C.Qt.NoPen); p.setBrush(G.QColor(6, 150, 215, 60)); p.drawRoundedRect(rect, 4, 4)
            sz = int(rect.width()) - 6 if key != "HOME" else 18
            p.drawPixmap(int(rect.x() + (rect.width() - sz)/2), int(rect.y() + (rect.height() - sz)/2), pix(kind, sz))

    def hit(s, pos):
        for key, rect in (("HOME", s.HOME), ("ROTL", s.ROT_L), ("ROTR", s.ROT_R)):
            if rect.contains(pos): return key
        for n, name, poly, vis, c, cells in s.faces():
            for dirn, cp in cells:
                if cp.containsPoint(pos, C.Qt.OddEvenFill): return dirn
        return None
    def mouseMoveEvent(s, e):
        h = s.hit(e.position())
        if h != s.hov:
            s.hov = h; s.update()
            s.setToolTip({"HOME": "Home view", "ROTL": "Turn the view 90° left", "ROTR": "Turn the view 90° right"}.get(h, "")
                         if not isinstance(h, tuple) else s.describe(h))
    def leaveEvent(s, _): s.hov = None; s.update()
    def contextMenuEvent(s, e):
        w = s.vp.window()
        if hasattr(w, "cube_menu"): w.cube_menu(e.globalPos())
    def mousePressEvent(s, e):
        h = s.hit(e.position())
        if h == "HOME": s.vp.home()
        elif h == "ROTL": s.vp.turn(+1)
        elif h == "ROTR": s.vp.turn(-1)
        elif isinstance(h, tuple):
            x, y, z = h; horiz = math.hypot(x, y)
            if horiz < 1e-9: s.vp.animate_to(s.vp.snapped_yaw(), 1.5*z)                  # straight top / bottom
            else: s.vp.animate_to(math.atan2(y, x), math.atan2(z, horiz))               # faces, edges and corners

# ---------------------------------------------------------------------------------------------------------------
#  3D viewport
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
#  References (what a command's selection boxes hold), construction geometry, picking, previews and the new
#  solid-modeling operations. Mixed into Viewport.
#
#  A reference is a plain tuple so it can be saved and replayed:
#     ("origin",) ("vertex", bi, vi) ("spoint", si, pi) ("cpoint", ci)                      points
#     ("edge", bi, ei) ("curve", si, ci) ("caxis", ci) ("oaxis", "X") ("skaxis", si, "X")     curves / axes
#     ("oplane", "XY") ("cplane", ci) ("face", bi, fid, (x, y, z)) ("profile", si, ri) ("body", bi)
# ---------------------------------------------------------------------------------------------------------------
CONS_COL = (0.93, 0.55, 0.18)
PREVIEW_COL = (0.56, 0.70, 0.88)
APPEARANCES = [("Default grey", None), ("Aluminium", (0.78, 0.79, 0.81)), ("Steel", (0.55, 0.57, 0.60)), ("Brass", (0.80, 0.66, 0.32)),
               ("Copper", (0.78, 0.45, 0.30)), ("Black plastic", (0.16, 0.16, 0.18)), ("White plastic", (0.92, 0.92, 0.90)),
               ("Red plastic", (0.80, 0.18, 0.16)), ("Blue plastic", (0.20, 0.38, 0.78)), ("Green plastic", (0.25, 0.60, 0.30)),
               ("Yellow plastic", (0.93, 0.78, 0.18)), ("Orange plastic", (0.95, 0.50, 0.15)), ("Wood (oak)", (0.66, 0.48, 0.30)),
               ("Glass", (0.70, 0.85, 0.92))]
MATERIALS = [("Steel", 7.85), ("Stainless steel", 8.00), ("Aluminium 6061", 2.70), ("Brass", 8.50), ("Copper", 8.96), ("Titanium", 4.43),
             ("PLA", 1.24), ("PETG", 1.27), ("ABS", 1.04), ("Nylon (PA12)", 1.01), ("Polycarbonate", 1.20), ("Oak", 0.75),
             ("Acrylic", 1.18), ("Rubber", 1.10)]                                         # density g/cm^3
MAT_COLOR = {"Steel": (0.55, 0.57, 0.60), "Stainless steel": (0.70, 0.71, 0.73), "Aluminium 6061": (0.78, 0.79, 0.81), "Brass": (0.80, 0.66, 0.32),
             "Copper": (0.78, 0.45, 0.30), "Titanium": (0.60, 0.60, 0.62), "Oak": (0.66, 0.48, 0.30)}

def ref_key(r):
    return tuple(r[:3]) if r and r[0] == "face" else tuple(r) if r else None

def sketch_points(sk):
    return [sk.geo.v(i) for i in range(len(sk.geo.P))]

def sketch_curves(sk):
    """One item per sketch curve (index = curve index): (name, kind, data, index).
    kind 'seg' -> (a, b); 'circle' -> (centre, radius); 'crv' -> curve index (arcs, splines, ...)."""
    out, g = [], sk.geo
    for ci, c in enumerate(g.C):
        k = c["k"]
        if k == "line": out.append((f"Line {ci + 1}", "seg", (g.v(c["p"][0]), g.v(c["p"][1])), ci))
        elif k == "circle": out.append((f"Circle {ci + 1}", "circle", (g.v(c["p"][0]), c["r"]), ci))
        else: out.append((f"{k.title()} {ci + 1}", "crv", ci, ci))
    return out

def sketch_curve_edge(sk, ci):
    es = crv_edges(sk.geo, ci, sk.plane)
    if not es: raise RuntimeError("that sketch curve has no geometry")
    return es[0]

class SolidMixin:
    # ---------------- reference resolution ----------------
    def span(s):
        if not s.bodies: return 200.0
        lo = np.min([b.lo.t() for b in s.bodies], 0); hi = np.max([b.hi.t() for b in s.bodies], 0)
        return max(float(np.linalg.norm(hi - lo)), 10.0)

    def ref_point(s, r):
        k = r[0]
        if k == "origin": return V()
        if k == "spot": return V(*r[3])
        if k == "vertex": return V(*s.bodies[r[1]].verts[r[2]])
        if k == "spoint": sk = s.sketches[r[1]]; return sk.plane.w(sketch_points(sk)[r[2]])
        if k == "cpoint": return s.cons[r[1]]["p"]
        if k == "face": return V(*r[3]) if len(r) > 3 else s.bodies[r[1]].face_point(r[2])
        if k in ("edge", "curve"):
            e = s.ref_edge(r); c = BRepAdaptor_Curve(e)
            if c.GetType() == GeomAbs_Circle: return vec(c.Circle().Location())
            return vec(c.Value((c.FirstParameter() + c.LastParameter())/2))
        raise RuntimeError("that selection isn't a point")

    def ref_plane(s, r):
        k = r[0]
        if k == "oplane": return {"XY": XY, "XZ": XZ, "YZ": YZ}[r[1]]
        if k == "cplane": return s.cons[r[1]]["plane"]
        if k == "face":
            c, n = face_frame(s.bodies[r[1]].faces[r[2]])
            u = V(1, 0, 0) if abs(n.z) > 0.99 else V(0, 0, 1).cross(n).normalize()
            return Plane(c, u, n.cross(u))
        if k == "profile": return s.sketches[r[1]].plane
        raise RuntimeError("that selection isn't flat")

    def ref_axis(s, r):
        k = r[0]
        if k == "oaxis": return V(), {"X": V(1, 0, 0), "Y": V(0, 1, 0), "Z": V(0, 0, 1)}[r[1]]
        if k == "caxis": return s.cons[r[1]]["p"], s.cons[r[1]]["d"]
        if k == "skaxis":
            pl = s.sketches[r[1]].plane; return pl.o, (pl.u if r[2] == "X" else pl.v)
        if k == "curve":
            sk = s.sketches[r[1]]; name, kind, data, _ = sketch_curves(sk)[r[2]]
            if kind == "circle": return sk.plane.w(data[0]), sk.plane.n
            if kind == "crv":
                c = sk.geo.C[r[2]]
                if c["k"] in ("arc", "ellipse", "earc"): return sk.plane.w(sk.geo.v(c["p"][0])), sk.plane.n
                if c["k"] == "l3":
                    a_, b_ = (sk.plane.w(V(*q)) for q in c["q"]); return a_, (b_ - a_).normalize()
                raise RuntimeError("pick a straight line, circle or arc for an axis")
            a, b = data; return sk.plane.w(a), (sk.plane.w(b) - sk.plane.w(a)).normalize()
        if k == "edge":
            c = BRepAdaptor_Curve(s.bodies[r[1]].eds[r[2]])
            if c.GetType() == GeomAbs_Line:
                a, b = vec(c.Value(c.FirstParameter())), vec(c.Value(c.LastParameter())); return a, (b - a).normalize()
            if c.GetType() == GeomAbs_Circle:
                ci = c.Circle(); return vec(ci.Location()), vec(ci.Axis().Direction())
            raise RuntimeError("that edge is curved - pick a straight or circular edge")
        if k == "face":
            cyl = s.bodies[r[1]].face_cylinder(r[2])
            if cyl: return cyl[0], cyl[1]
            srf = BRepAdaptor_Surface(s.bodies[r[1]].faces[r[2]])
            if srf.GetType() == GeomAbs_Cone: ax = srf.Cone().Axis(); return vec(ax.Location()), vec(ax.Direction())
            if srf.GetType() == GeomAbs_Torus: ax = srf.Torus().Axis(); return vec(ax.Location()), vec(ax.Direction())
            if srf.GetType() == GeomAbs_Plane:
                p = s.ref_point(r); return p, face_frame(s.bodies[r[1]].faces[r[2]])[1]
        raise RuntimeError("that selection doesn't define an axis")

    def ref_face(s, r):
        if r[0] == "face": return s.bodies[r[1]].faces[r[2]]
        if r[0] == "profile":
            regs = s.sketches[r[1]].regions()
            if r[2] >= len(regs): raise RuntimeError("a selected profile no longer exists")
            return regs[r[2]].face
        if r[0] in ("oplane", "cplane"):
            pl = s.ref_plane(r); return big_plane_face(pl.o, pl.n, s.span()*4)
        raise RuntimeError("that selection isn't a face")

    def ref_edge(s, r):
        if r[0] == "edge": return s.bodies[r[1]].eds[r[2]]
        if r[0] == "curve":
            sk = s.sketches[r[1]]; name, kind, data, _ = sketch_curves(sk)[r[2]]
            if kind == "circle":
                c, R = data; return BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(pnt(sk.plane.w(c)), gdir(sk.plane.n), gdir(sk.plane.u)), R)).Edge()
            if kind == "crv": return sketch_curve_edge(sk, r[2])
            a, b = data; return BRepBuilderAPI_MakeEdge(pnt(sk.plane.w(a)), pnt(sk.plane.w(b))).Edge()
        if r[0] in ("caxis", "oaxis", "skaxis"):
            p, d = s.ref_axis(r); L = s.span()*2; return BRepBuilderAPI_MakeEdge(pnt(p - d*L), pnt(p + d*L)).Edge()
        raise RuntimeError("that selection isn't an edge or curve")

    def ref_wire(s, refs):
        return chain_wire([s.ref_edge(r) for r in refs])

    def ref_tool_shape(s, r, extend=True):
        """Shape usable as a cutting / splitting tool."""
        if r[0] == "body": return s.bodies[r[1]].shape
        if r[0] in ("oplane", "cplane"): pl = s.ref_plane(r); return big_plane_face(pl.o, pl.n, s.span()*4)
        if r[0] == "face": f = s.ref_face(r); return big_surface_face(f, s.span()*4) if extend else f
        raise RuntimeError("that can't be used as a tool")

    def ref_profile_parent(s, r):
        if r[0] == "profile": return s.sketches[r[1]].parent
        if r[0] == "face": return r[1]
        return None

    # ---------------- construction geometry ----------------
    def exec_construct(s, op):
        R, Vv, how = op["refs"], op["vals"], op["how"]
        P = lambda k: s.ref_plane(R[k]); A = lambda k: s.ref_axis(R[k]); Pt = lambda k: s.ref_point(R[k])
        def plane_from(o, n, hint=None):
            n = n.normalize(); u = hint if hint is not None else (V(1, 0, 0) if abs(n.x) < 0.9 else V(0, 1, 0))
            u = (u - n*u.dot(n)).normalize(); return Plane(o, u, n.cross(u))
        kind = how.split("_")[0]; res = {}
        if how == "plane_offset":
            pl = P("plane"); res["plane"] = Plane(pl.o + pl.n*Vv["d"], pl.u, pl.v)
        elif how == "plane_angle":
            p, d = A("line"); base = None
            r0 = R["line"]
            if r0[0] == "curve": base = s.sketches[r0[1]].plane.n
            elif r0[0] == "edge":
                for f in s.bodies[r0[1]].faces_of_edge(s.bodies[r0[1]].eds[r0[2]]):
                    try: base = face_frame(f)[1]; break
                    except Exception: pass
            if base is None: base = V(0, 0, 1) if abs(d.z) < 0.9 else V(1, 0, 0)
            n0 = (base - d*base.dot(d)).normalize(); a = math.radians(Vv["angle"])
            n = n0*math.cos(a) + d.cross(n0)*math.sin(a); res["plane"] = plane_from(p, n, d)
        elif how == "plane_tangent":
            cyl = s.bodies[R["face"][1]].face_cylinder(R["face"][2])
            if not cyl: raise RuntimeError("pick a cylindrical face")
            o, ax, xd, Rr, *_ = cyl; a = math.radians(Vv["angle"]); y = ax.cross(xd)
            if len(R["face"]) > 3:                                  # start from where the face was clicked
                hp = V(*R["face"][3]) - o; hp = hp - ax*hp.dot(ax); a += math.atan2(hp.dot(y), hp.dot(xd))
            rd = xd*math.cos(a) + y*math.sin(a); res["plane"] = plane_from(o + rd*Rr, rd, ax)
        elif how == "plane_mid":
            a, b = P("a"), P("b")
            if abs(abs(a.n.dot(b.n)) - 1) < 1e-6:
                bo = a.o + a.n*((b.o - a.o).dot(a.n)/2); res["plane"] = Plane(bo, a.u, a.v)
            else:
                nb = b.n if a.n.dot(b.n) > 0 else -b.n; n = (a.n - nb).normalize()
                d = a.n.cross(nb).normalize(); M = np.array([a.n.t(), nb.t(), d.t()])
                p0 = V(*np.linalg.solve(M, [a.n.dot(a.o), nb.dot(b.o), 0])); res["plane"] = plane_from(p0, n, d)
        elif how == "plane_2edges":
            (p1, d1), (p2, d2) = A("a"), A("b"); n = d1.cross(p2 - p1)
            if n.Length < 1e-9: n = d1.cross(d2)
            if n.Length < 1e-9: raise RuntimeError("those lines are the same line")
            res["plane"] = plane_from(p1, n, d1)
        elif how == "plane_3pts":
            a, b, c = Pt("a"), Pt("b"), Pt("c"); n = (b - a).cross(c - a)
            if n.Length < 1e-9: raise RuntimeError("the three points are in a line")
            res["plane"] = plane_from(a, n, (b - a).normalize())
        elif how == "plane_tanpt":
            q, n = normal_at(s.ref_face(R["face"]), Pt("point")); res["plane"] = plane_from(q, n)
        elif how == "plane_path":
            p, t = wire_eval(s.ref_wire(R["path"]), Vv["t"]); res["plane"] = plane_from(p, t)
        elif how == "axis_cyl":
            p, d = s.ref_axis(R["face"]); res.update(p=p, d=d)
        elif how in ("axis_perp_pt", "axis_perp_face"):
            q, n = normal_at(s.ref_face(R["face"]), Pt("point")); res.update(p=q, d=n)
        elif how == "axis_2planes":
            a, b = P("a"), P("b"); d = a.n.cross(b.n)
            if d.Length < 1e-9: raise RuntimeError("those planes are parallel")
            d = d.normalize(); M = np.array([a.n.t(), b.n.t(), d.t()])
            res.update(p=V(*np.linalg.solve(M, [a.n.dot(a.o), b.n.dot(b.o), 0])), d=d)
        elif how == "axis_2pts":
            a, b = Pt("a"), Pt("b")
            if (b - a).Length < 1e-9: raise RuntimeError("the points are the same")
            res.update(p=a, d=(b - a).normalize())
        elif how == "axis_edge":
            p, d = A("edge"); res.update(p=p, d=d)
        elif how == "point_vertex": res["p"] = Pt("point")
        elif how == "point_2edges":
            (p1, d1), (p2, d2) = A("a"), A("b"); w0 = p1 - p2; b_ = d1.dot(d2); den = 1 - b_*b_
            if den < 1e-12: raise RuntimeError("those lines are parallel")
            t1 = (b_*d2.dot(w0) - d1.dot(w0))/den; t2 = (d2.dot(w0) - b_*d1.dot(w0))/den
            res["p"] = (p1 + d1*t1 + p2 + d2*t2)*0.5
        elif how == "point_3planes":
            ps = [P("a"), P("b"), P("c")]; M = np.array([p.n.t() for p in ps])
            if abs(np.linalg.det(M)) < 1e-9: raise RuntimeError("those planes don't meet in a single point")
            res["p"] = V(*np.linalg.solve(M, [p.n.dot(p.o) for p in ps]))
        elif how == "point_center":
            r = R["obj"]
            if r[0] == "face":
                srf = BRepAdaptor_Surface(s.bodies[r[1]].faces[r[2]]); tp = srf.GetType()
                if tp == GeomAbs_Sphere: res["p"] = vec(srf.Sphere().Location())
                elif tp == GeomAbs_Torus: res["p"] = vec(srf.Torus().Location())
                else: res["p"] = s.ref_axis(r)[0]
            else:
                e = s.ref_edge(r); c = BRepAdaptor_Curve(e)
                if c.GetType() != GeomAbs_Circle: raise RuntimeError("pick a circle, arc, sphere or torus")
                res["p"] = vec(c.Circle().Location())
        elif how == "point_edge_plane":
            (p, d), pl = A("edge"), P("plane"); den = d.dot(pl.n)
            if abs(den) < 1e-12: raise RuntimeError("the line is parallel to the plane")
            res["p"] = p + d*((pl.o - p).dot(pl.n)/den)
        elif how == "point_path":
            res["p"] = wire_eval(s.ref_wire(R["path"]), Vv["t"])[0]
        else: raise RuntimeError(f"unknown construction {how}")
        n = sum(1 for c in s.cons if c["kind"] == kind) + 1
        s.cons.append(dict(kind=kind, name=f"{kind.title()}{n}", visible=True, **res))

    # ---------------- picking ----------------
    def point_candidates(s):
        out = [(("origin",), V())]
        for bi, b in enumerate(s.bodies):
            if b.visible: out += [(("vertex", bi, vi), V(*p)) for vi, p in enumerate(b.verts)]
        for si, sk in enumerate(s.sketches):
            if sk.visible: out += [(("spoint", si, pi), sk.plane.w(p)) for pi, p in enumerate(sketch_points(sk))]
        for ci, c in enumerate(s.cons):
            if c["kind"] == "point" and c["visible"]: out.append((("cpoint", ci), c["p"]))
        return out

    def pick_ref(s, p, kinds):
        """Best reference under the mouse for a selection box accepting `kinds`."""
        if s.mvp is None or not kinds: return None
        k = s.devicePixelRatioF(); px, py = s.mouse_dev(p)
        if "point" in kinds:
            cands = s.point_candidates()
            if cands:
                xy, ok = s.project([c[1].t() for c in cands]); d = np.hypot(xy[:, 0] - px, xy[:, 1] - py); d[~ok] = np.inf
                i = int(np.argmin(d))
                if d[i] < 10*k: return cands[i][0]
        best = None
        if kinds & {"curve", "axis", "edge", "path"}:
            kk = s.devicePixelRatioF(); px, py = s.mouse_dev(p)
            for si, sk in enumerate(s.sketches):
                if not sk.visible and sk is not s.esk: continue
                if s.skedit and s.cmd and s.cmd.cmd.group == "SKETCH" and "curve" in kinds and "edge" not in kinds and sk is not s.esk \
                        and not (s.cmd.active == "srefs"): continue
                for ci, c in enumerate(sk.geo.C):
                    if c["k"] in ("point", "text"): continue
                    if not (kinds & {"curve", "path", "axis"}) and c["k"] != "line": continue
                    dd = 1e9
                    for A in geo_polys(sk.geo, ci):
                        xy_, ok = s.sk_proj(sk, A)
                        if ok.all(): dd = min(dd, poly_dist(xy_, px, py)/kk)
                    if dd < 8 and (best is None or dd < best[0]): best = (dd, ("curve", si, ci))
            if kinds & {"axis"}:
                for ci, c in enumerate(s.cons):
                    if c["kind"] == "axis" and c["visible"]:
                        L = s.span()*0.75; dd = s.screen_seg_dist(p, c["p"] - c["d"]*L, c["p"] + c["d"]*L)
                        if dd < 8 and (best is None or dd < best[0]): best = (dd, ("caxis", ci))
                if s.show_grid or s.cmd:
                    for nm, d in (("X", V(1, 0, 0)), ("Y", V(0, 1, 0)), ("Z", V(0, 0, 1))):
                        L = s.dist*1.2; dd = s.screen_seg_dist(p, d*(-L if nm != "Z" else 0), d*L)
                        if dd < 6 and (best is None or dd < best[0]): best = (dd, ("oaxis", nm))
            if kinds & {"edge", "axis", "path"}:
                e = s.edge_near(p)
                if e:
                    if "axis" in kinds and not (kinds & {"edge", "path"}):
                        tp = BRepAdaptor_Curve(s.bodies[e[0]].eds[e[1]]).GetType()
                        if tp not in (GeomAbs_Line, GeomAbs_Circle): e = None
                    if e and best is None: best = (0, ("edge",) + tuple(e))
        if best: return best[1]
        hit = s.pick_face(p)
        if "plane" in kinds:
            o, d = s.ray(p); S = s.cons_size(); cand = []
            if s.cmd:
                for name, pl, _ in ORIGIN_PLANES:
                    dn = d.dot(pl.n)
                    if abs(dn) < 1e-12: continue
                    t = (pl.o - o).dot(pl.n)/dn; q = pl.l(o + d*t)
                    if t > 0 and 0 <= q.x <= s.plane_size() and 0 <= q.y <= s.plane_size(): cand.append((t, ("oplane", name)))
            for ci, c in enumerate(s.cons):
                if c["kind"] != "plane" or not c["visible"]: continue
                pl = c["plane"]; dn = d.dot(pl.n)
                if abs(dn) < 1e-12: continue
                t = (pl.o - o).dot(pl.n)/dn; q = pl.l(o + d*t)
                if t > 0 and abs(q.x) <= S/2 and abs(q.y) <= S/2: cand.append((t, ("cplane", ci)))
            if cand:
                t, r = min(cand)
                if hit is None or t < hit[3]: return r
        if "profile" in kinds:
            r = s.region_under(p)
            if r: return ("profile", s.sketches.index(r[0]), r[1])
        if hit:
            bi, fid = hit[0], hit[1]; o, d = s.ray(p); q = o + d*hit[3]; fr = ("face", bi, fid, (q.x, q.y, q.z))
            b = s.bodies[bi]
            if getattr(b, "mesh", False):                              # a mesh is picked as a whole
                return ("spot", bi, fid, (q.x, q.y, q.z)) if "spot" in kinds else ("body", bi) if "body" in kinds else None
            if "spot" in kinds: return ("spot", bi, fid, (q.x, q.y, q.z))
            if "face" in kinds: return fr
            if "pface" in kinds and b.face_normal(fid) is not None: return fr
            if "plane" in kinds and b.face_normal(fid) is not None: return fr
            if "axis" in kinds and b.face_cylinder(fid): return fr
            if "body" in kinds: return ("body", bi)
        return None

    def cons_size(s): return max(s.span()*0.22, s.dist*0.12, 10.0)

    # ---------------- drawing ----------------
    def draw_ref(s, r, rgb, alpha=1.0, width=3.4):
        try:
            k = r[0]
            if k in ("face",):
                b = s.bodies[r[1]]; glColor4f(*rgb, 0.35*alpha); s.tris_array(b.tv[b.tri_face == r[2]])
            elif k == "body":
                b = s.bodies[r[1]]; glColor4f(*rgb, 0.18*alpha); s.tris_array(b.tv)
            elif k == "profile":
                sk = s.sketches[r[1]]; glPushMatrix(); glMultMatrixd(sk.plane.gl()); glColor4f(*rgb, 0.45*alpha)
                s.tris_array(sk.regions()[r[2]].tris); glPopMatrix()
            elif k == "edge":
                glColor4f(*rgb, alpha); glLineWidth(width); s.strip(s.bodies[r[1]].edges[r[2]])
            elif k == "curve":
                sk = s.sketches[r[1]]; glColor4f(*rgb, alpha); glLineWidth(width)
                for A in geo_polys(sk.geo, r[2]): s.strip([sk.plane.w(V(q[0], q[1], q[2] if len(q) > 2 else 0)).t() for q in A])
            elif k in ("caxis", "oaxis", "skaxis"):
                p, d = s.ref_axis(r); L = s.span()*0.9; glColor4f(*rgb, alpha); glLineWidth(width)
                s.strip([(p - d*L).t(), (p + d*L).t()])
            elif k in ("oplane", "cplane"):
                pl = s.ref_plane(r); S = s.plane_size() if k == "oplane" else s.cons_size()
                lo, hi = (0, S) if k == "oplane" else (-S/2, S/2)
                glPushMatrix(); glMultMatrixd(pl.gl()); glColor4f(*rgb, 0.30*alpha); glBegin(GL_QUADS)
                for x, y in ((lo, lo), (hi, lo), (hi, hi), (lo, hi)): glVertex3f(x, y, 0)
                glEnd(); glPopMatrix()
            else:
                q = s.ref_point(r); glPointSize(11); glColor4f(*rgb, alpha); glBegin(GL_POINTS); glVertex3f(*q.t()); glEnd()
        except Exception:
            pass

    def draw_cons(s, items=None, hot=False):
        S = s.cons_size(); glDepthMask(GL_FALSE)
        for c in (s.cons if items is None else items):
            if not c["visible"]: continue
            if hot and c["kind"] == "plane":                               # the plane a command is about to make
                pl = c["plane"]; glPushMatrix(); glMultMatrixd(pl.gl())
                glColor4f(*SEL, 0.22); glBegin(GL_QUADS)
                for x, y in ((-S/2, -S/2), (S/2, -S/2), (S/2, S/2), (-S/2, S/2)): glVertex3f(x, y, 0)
                glEnd(); glColor4f(*SEL, 0.95); glLineWidth(2.0); glBegin(GL_LINE_LOOP)
                for x, y in ((-S/2, -S/2), (S/2, -S/2), (S/2, S/2), (-S/2, S/2)): glVertex3f(x, y, 0)
                glEnd(); glPopMatrix(); continue
            if c["kind"] == "plane":
                pl = c["plane"]; glPushMatrix(); glMultMatrixd(pl.gl())
                glColor4f(*CONS_COL, 0.14); glBegin(GL_QUADS)
                for x, y in ((-S/2, -S/2), (S/2, -S/2), (S/2, S/2), (-S/2, S/2)): glVertex3f(x, y, 0)
                glEnd(); glColor4f(*CONS_COL, 0.8); glLineWidth(1.4); glBegin(GL_LINE_LOOP)
                for x, y in ((-S/2, -S/2), (S/2, -S/2), (S/2, S/2), (-S/2, S/2)): glVertex3f(x, y, 0)
                glEnd(); glPopMatrix()
            elif c["kind"] == "axis":
                L = s.span()*0.75; glColor4f(*CONS_COL, 0.9); glLineWidth(1.6); glEnable(GL_LINE_STIPPLE); glLineStipple(2, 0xF18F)
                s.strip([(c["p"] - c["d"]*L).t(), (c["p"] + c["d"]*L).t()]); glDisable(GL_LINE_STIPPLE)
            else:
                glPointSize(9); glColor4f(*CONS_COL, 1); glBegin(GL_POINTS); glVertex3f(*c["p"].t()); glEnd()
        for f in s.features:                                             # cosmetic threads: rings one pitch apart
            cos = f.get("cosmetic")
            if not cos: continue
            o, a, R, L, p = cos; x = V(1, 0, 0) if abs(a.x) < 0.9 else V(0, 1, 0); x = (x - a*x.dot(a)).normalize(); y = a.cross(x)
            glColor4f(0.2, 0.25, 0.3, 0.7); glLineWidth(1)
            for k in range(int(L/p) + 1):
                c0 = o + a*(k*p); s.strip([(c0 + x*(R*math.cos(t)) + y*(R*math.sin(t))).t() for t in np.linspace(0, math.tau, 41)])
        glDepthMask(GL_TRUE)

    def draw_cmd_overlay(s):
        if not s.cmd: return
        glDisable(GL_DEPTH_TEST)
        if s.cmd.wants({"plane"}): s.draw_origin_planes()
        for r in s.cmd.all_refs(): s.draw_ref(r, SEL, 1.0)
        if s.cmd_hover and ref_key(s.cmd_hover) not in {ref_key(r) for r in s.cmd.all_refs()}: s.draw_ref(s.cmd_hover, HILITE, 0.6)
        h = s.handle
        if h:
            o, d, v = h["o"], h["d"], h["val"]; a = o + d*v; b = a + d*(s.dist*0.08*(1 if v >= 0 else -1))
            glColor3f(*SEL); glLineWidth(4); s.strip([o.t(), a.t()]); u = (V(0, 0, 1) if abs(d.z) < 0.9 else V(1, 0, 0)).cross(d).normalize(); w = d.cross(u)
            rr = s.dist*0.012; glBegin(GL_TRIANGLE_FAN); glVertex3f(*b.t())
            for k in range(17):
                t = k*math.tau/16; glVertex3f(*(a + u*(rr*math.cos(t)) + w*(rr*math.sin(t))).t())
            glEnd()
        glEnable(GL_DEPTH_TEST)

    def handle_hit(s, p):
        h = s.handle
        if not h: return False
        a = h["o"] + h["d"]*h["val"]; b = a + h["d"]*(s.dist*0.08*(1 if h["val"] >= 0 else -1))
        return s.screen_seg_dist(p, a, b) < 14

    # ---------------- dry runs (previews) ----------------
    def dry_run(s, op):
        asm0 = s.asm_state(); ac0 = s.active_comp
        try: return s.dry_run_core(op)
        finally: s.asm_apply(asm0); s.active_comp = ac0
    def dry_run_core(s, op):
        saved = (list(s.bodies), list(s.features), list(s.cons), list(s.sketches), s.active,
                 [(k, k.geo, k.visible) for k in s.sketches], list(s.canv))
        try:
            s.exec_op(op)
            return dict(bodies=list(s.bodies), cons=list(s.cons),
                        geos=[(k, k.geo, g0) for k, g0, _ in saved[5] if k.geo is not g0])
        finally:
            s.bodies, s.features, s.cons, s.sketches, s.active, sks, s.canv = saved
            for k, g0, vis in sks: k.geo = g0; k.visible = vis; k._key = None

    # ---------------- operations ----------------
    def add_feature(s, name, tool, op, plane=None):
        s.features.append(dict(name=name, tool=tool, op=op, plane=plane or XY))

    def apply_op(s, tool, op, positive=True, parent=None, bodies=None):
        """New / Join / Cut / Intersect a tool into the model (bodies = only these bodies take part)."""
        bodies = [b for b in (bodies or []) if b < len(s.bodies)]
        if op == "Intersect":
            targets = bodies or [i for i, b in enumerate(s.bodies) if b.visible and s.overlaps(i, tool)]
            if not targets: raise RuntimeError("the feature doesn't overlap any body")
            for i in reversed(sorted(targets)):
                common = boolean(s.bodies[i].shape, tool, "common")
                if abs(volume(common)) < 1e-9: del s.bodies[i]
                else: s.replace_body(i, common)
            return "Intersect"
        if bodies and op in ("Cut", "Join"):
            if op == "Cut":
                for i in reversed(sorted(bodies)): s.replace_body(i, boolean(s.bodies[i].shape, tool, "cut"))
                return "Cut"
            s.join(bodies, tool); return "Join"
        return s.apply(tool, op, positive, parent)

    def profile_faces(s, refs):
        out = []
        for r in refs:
            if r[0] in ("profile", "face"):
                f = s.ref_face(r)
                if r[0] == "face": c, n = face_frame(f)
                else: n = s.sketches[r[1]].plane.n
                out.append((f, n))
        return out

    def extrude_side(s, face, n, extent, d, to, taper):
        """Solid for one side of an extrude."""
        c = centroid(face)
        if extent == "all":
            L = s.span()*3 + abs((c - V()).Length)
            return prism(face, n, L) if abs(taper) < 1e-9 else tapered_prism(face, n, L, taper)
        if extent == "to":
            if not to: raise RuntimeError("pick the object to extrude to")
            k = to[0]
            if k in ("oplane", "cplane") or (k == "face" and s.bodies[to[1]].face_normal(to[2]) is not None):
                pl = s.ref_plane(to); den = n.dot(pl.n)
                if abs(den) < 1e-9: raise RuntimeError("that plane is parallel to the extrude direction")
                L = (pl.o - c).dot(pl.n)/den
            elif k in ("vertex", "spoint", "cpoint", "origin"):
                L = (s.ref_point(to) - c).dot(n)
            else:
                big = s.span()*3; tool = s.ref_tool_shape(to)
                for sgn in (1, -1):
                    try:
                        piece = split_keep(prism(face, n*sgn, big), tool, c + n*sgn*1e-3)
                        if abs(volume(piece)) < abs(volume(prism(face, n*sgn, big)))*0.999: return piece
                    except Exception: pass
                raise RuntimeError("the extrude never reaches that object")
            if abs(L) < 1e-9: raise RuntimeError("the profile already touches that object")
            return prism(face, n, L) if abs(taper) < 1e-9 else tapered_prism(face, n*(1 if L > 0 else -1), abs(L), taper)
        if abs(d) < 1e-9: raise RuntimeError("the distance is zero")
        if abs(taper) < 1e-9: return prism(face, n, d)
        return tapered_prism(face, n*(1 if d > 0 else -1), abs(d), taper)

    def exec_extrude2(s, op):
        prof = s.profile_faces(op["profiles"]); curves = [r for r in op["profiles"] if r[0] == "curve"]
        if not prof and not curves: raise RuntimeError("pick at least one profile")
        tools = []; plane = None
        thin = op.get("thin")
        groups = []
        if curves:
            if not thin: raise RuntimeError("open lines can only be extruded with Thin turned on")
            by_sk = {}
            for r in curves: by_sk.setdefault(r[1], []).append(r)
            for si, rs in by_sk.items():
                sk = s.sketches[si]
                for r in rs:
                    name, kind, data, _ = sketch_curves(sk)[r[2]]
                    c_ = sk.geo.C[r[2]]
                    if kind == "circle" or crv_closed(c_):                # thin ring from a closed curve
                        f = BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(s.ref_edge(r)).Wire(), True).Face()
                        groups.append((as_face(thin_face(f, op["thick"], op.get("tside", "center"))), sk.plane.n)); continue
                    pts = [sk.plane.w(data[0]), sk.plane.w(data[1])] if kind == "seg" else \
                          [sk.plane.w(V(q[0], q[1], 0)) for q in geo_curve_pts(sk.geo, r[2])[0]]
                    side = {"center": "center", "in": "right", "out": "left"}[op.get("tside", "center")]
                    groups.append((open_wire_thick_face(pts, op["thick"], sk.plane.n, side), sk.plane.n))
        for f, n in prof:
            if thin: f = as_face(thin_face(f, op["thick"], op.get("tside", "out")))
            groups.append((f, n))
        for f, n in groups:
            st_ = op.get("start", "profile")
            if st_ == "offset" and abs(op.get("soff", 0)) > 1e-12: f = as_face(translated(f, n*op["soff"]))
            elif st_ == "object" and op.get("sobj"):
                pl = s.ref_plane(op["sobj"]); c = centroid(f); den = n.dot(pl.n)
                if abs(den) > 1e-9: f = as_face(translated(f, n*((pl.o - c).dot(pl.n)/den)))
            dirn = op.get("dir", "one"); parts = []
            if dirn == "sym":
                half = op["d"]/2 if op.get("whole") else op["d"]
                parts.append(s.extrude_side(f, n, op.get("e1", "dist"), half, op.get("to"), op.get("taper", 0)))
                parts.append(s.extrude_side(f, -n, op.get("e1", "dist"), half, op.get("to"), op.get("taper", 0)))
            else:
                parts.append(s.extrude_side(f, n, op.get("e1", "dist"), op["d"], op.get("to"), op.get("taper", 0)))
                if dirn == "two":
                    parts.append(s.extrude_side(f, -n, op.get("e2", "dist"), op.get("d2", 0), op.get("to2"), op.get("taper2", 0)))
            tools.append(fuse_all(parts) if len(parts) > 1 else parts[0]); plane = plane or Plane(centroid(f), V(1, 0, 0), V(0, 1, 0))
        tool = fuse_all(tools)
        parents = [s.ref_profile_parent(r) for r in op["profiles"]]; parent = next((p for p in parents if p is not None), None)
        res = s.apply_op(tool, op["op"], op["d"] >= 0, parent, op.get("bodies"))
        sk_plane = None
        for r in op["profiles"]:
            if r[0] in ("profile", "curve"):
                sk = s.sketches[r[1]]; sk.visible = False; sk_plane = sk.plane
                if s.active is sk: s.active = None
        s.add_feature(f"Extrude ({res.lower()})", tool, res, sk_plane)

    def exec_revolve2(s, op):
        prof = s.profile_faces(op["profiles"])
        if not prof: raise RuntimeError("pick at least one profile")
        p, d = s.ref_axis(op["axis"]); dirn = op.get("dir", "one"); ang = op.get("angle", 360.0)
        if op.get("extent") == "full": ang, dirn = 360.0, "one"
        parts = []
        for f, n in prof:
            if dirn == "sym": parts.append(revolve(as_face(rotated(f, p, d, -math.radians(ang/2))), p, d, ang))
            else:
                parts.append(revolve(f, p, d, ang))
                if dirn == "two": parts.append(revolve(f, p, d, -op.get("angle2", 90.0)))
        tool = fuse_all(parts)
        if abs(volume(tool)) < 1e-9: raise RuntimeError("nothing to revolve - the profile probably touches the axis along an edge")
        parents = [s.ref_profile_parent(r) for r in op["profiles"]]; parent = next((x for x in parents if x is not None), None)
        o = op["op"] if op["op"] != "Auto" else ("Join" if parent is not None else "New")
        res = s.apply_op(tool, o, True, parent, op.get("bodies"))
        for r in op["profiles"]:
            if r[0] == "profile":
                sk = s.sketches[r[1]]; sk.visible = False
                if s.active is sk: s.active = None
        s.add_feature(f"Revolve ({res.lower()})", tool, res)

    def hide_sketches_of(s, refs):
        for r in refs:
            if r and r[0] in ("profile", "curve", "spoint") and r[1] < len(s.sketches):
                sk = s.sketches[r[1]]; sk.visible = False
                if s.active is sk: s.active = None

    def pattern_objects(s, op):
        """[(kind, payload)]: ('body', bi) ('feat', fi) ('ent', (si, ei))."""
        t = op.get("objtype", "bodies"); out = []
        if t == "bodies": out = [("body", r[1]) for r in op.get("objs", []) if r[0] == "body"]
        elif t == "features": out = [("feat", i) for i in op.get("feats", [])]
        else:
            seen = set()
            for r in op.get("objs", []):
                if r[0] == "curve":
                    ei = r[2]
                    if (r[1], ei) not in seen: seen.add((r[1], ei)); out.append(("ent", (r[1], ei)))
        if not out: raise RuntimeError("pick what to pattern")
        return out

    def place_copies(s, objs, mats, op):
        """Apply each 3x4 transform in mats to every object."""
        new_bodies = []; new_ents = {}
        for kind, x in objs:
            for m in mats:
                if kind == "body":
                    new_bodies.append(transformed(s.bodies[x].shape, m))
                elif kind == "feat":
                    f = s.features[x]; tool = transformed(f["tool"], m)
                    if f["op"] == "Cut": s.cut_all(tool)
                    elif f["op"] == "Join": s.apply(tool, "Join", True, None)
                    elif f["op"] == "Intersect": s.apply_op(tool, "Intersect")
                    else: s.bodies += [Body(y) for y in solids(tool)]
                else:
                    si, ei = x; new_ents.setdefault(si, []).append((ei, m))
        for si, items in new_ents.items():
            sk = s.sketches[si]; g = sk.geo.copy(); pl = sk.plane
            for ei, m in items:
                M = np.array(m)
                def tf(xx, yy, M=M):
                    w = np.array(pl.w(V(xx, yy, 0)).t()); r = M[:, :3] @ w + M[:, 3]; q = pl.l(V(*r)); return q.x, q.y
                geo_copy_curves(g, [ei], tf)
            g.touch(); sk.geo = g; s.active = sk
        if new_bodies:
            if op == "Join":
                srcs = sorted({x for k, x in objs if k == "body"}); shape = fuse_all([s.bodies[i].shape for i in srcs] + new_bodies)
                for i in reversed(srcs[1:]): del s.bodies[i]
                s.replace_body(srcs[0], shape)
            else: s.bodies += [Body(y) for b in new_bodies for y in solids(b)]

    def skip_set(s, text):
        out = set()
        for part in str(text or "").replace(";", ",").split(","):
            part = part.strip()
            if part.isdigit(): out.add(int(part))
        return out

    # ---------------- surface modelling ----------------
    def ref_edges(s, refs, chain=True, loop=False):
        """Model edges for curve / edge / profile picks, optionally grown into connected chains.
        Returns [(edge, body index or None)]. loop: chain body edges round an open boundary (for patches)."""
        out, seen = [], []
        def add(e, src):
            if any(e.IsSame(x) for x, _ in out): return
            out.append((e, src))
        for r in refs:
            k = r[0]
            if k == "profile":
                for e in subshapes(s.ref_face(r), TopAbs_EDGE): add(e, None)
                continue
            if k == "curve":
                sk = s.sketches[r[1]]; g = sk.geo
                pool = [(j, e) for j, c in enumerate(g.C) if not c.get("cons") and c["k"] not in ("point", "text") for e in crv_edges(g, j, sk.plane)] \
                    if chain else []
                seed = [e for j, e in pool if j == r[2]] or [s.ref_edge(r)]
                for e in (chain_from_pool(seed, [e for _, e in pool]) if chain else seed): add(e, None)
                continue
            if k == "edge":
                b = s.bodies[r[1]]; e0 = b.eds[r[2]]
                if not chain: add(e0, r[1]); continue
                if loop:
                    fe = free_edges(b.shape)
                    if any(e0.IsSame(x) for x in fe):
                        for e in chain_from_pool([e0], fe): add(e, r[1])
                        continue
                for i in tangent_chain(b, r[2]): add(b.eds[i], r[1])
                continue
            add(s.ref_edge(r), None)
        if not out: raise RuntimeError("pick some curves or edges")
        return out

    def ref_sections(s, refs, chain=True):
        """Loft sections: one wire per picked curve chain / profile, or a point."""
        secs, used = [], []
        for r in refs:
            if r[0] in ("spoint", "vertex", "cpoint", "origin"):
                secs.append(BRepBuilderAPI_MakeVertex(pnt(s.ref_point(r))).Vertex()); continue
            if r[0] == "profile": secs.append(wires_of(s.ref_face(r))[0]); continue
            es = [e for e, _ in s.ref_edges([r], chain)]
            if any(any(e.IsSame(u) or edge_same_geom(e, u) for u in used) for e in es): continue
            used += es; ws = edges_to_wires(es)
            if len(ws) != 1: raise RuntimeError("each loft section must be one connected curve")
            secs.append(ws[0])
        return secs

    def surf_dir(s, refs, dirref=None, chain=True):
        """Extrude direction: the picked plane / axis, else the sketch's normal, else the plane the edges lie in."""
        if dirref:
            r = dirref
            if r[0] in ("oplane", "cplane", "profile") or (r[0] == "face" and s.bodies[r[1]].face_normal(r[2]) is not None): return s.ref_plane(r).n
            return s.ref_axis(r)[1]
        r = refs[0]
        if r[0] in ("profile", "curve"): return s.sketches[r[1]].plane.n
        if r[0] == "face": return face_frame(s.ref_face(r))[1]
        ws = edges_to_wires([e for e, _ in s.ref_edges(refs, chain)])
        n = wire_plane_normal(ws[0]) if ws else None
        if n is None: raise RuntimeError("those edges aren't flat - pick a direction (a plane or axis)")
        return n

    def add_surface(s, shape, name, refs=()):
        s.bodies.append(Body(shape)); s.add_feature(name, shape, "New")
        if refs: s.hide_sketches_of(list(refs))

    def set_body_shapes(s, bi, shapes):
        old = s.bodies[bi]; s.bodies[bi:bi + 1] = [Body(x).copy_look(old) for x in shapes]

    def group_by_body(s, refs, kinds=("face",)):
        out = {}
        for r in refs:
            if r[0] == "body": out.setdefault(r[1], None)
            elif r[0] in kinds:
                cur = out.setdefault(r[1], [])
                if cur is not None: cur.append(r)
        return out

    def exec_surface(s, op):
        t = op["t"]; chain = op.get("chain", True)
        if t == "s_extrude":
            edges = [e for e, _ in s.ref_edges(op["profiles"], chain)]; n = s.surf_dir(op["profiles"], op.get("dirref"), chain)
            d = op["d"]; d2 = {"one": 0.0, "two": op.get("d2", 0.0), "sym": d}[op.get("dir", "one")]
            sh = surf_extrude(edges, n, d, d2, op.get("taper", 0.0)); s.add_surface(sh, "Surface extrude", op["profiles"]); return
        if t == "s_revolve":
            o, ax = s.ref_axis(op["axis"]); ang = 360.0 if op.get("extent") == "full" else op["angle"]
            sh = surf_revolve([e for e, _ in s.ref_edges(op["profiles"], chain)], o, ax, ang)
            s.add_surface(sh, "Surface revolve", op["profiles"] + [op["axis"]]); return
        if t == "s_sweep":
            path = chain_wire([e for e, _ in s.ref_edges(op["path"], chain)])
            sh = surf_sweep([e for e, _ in s.ref_edges(op["profiles"], chain)], path, op.get("orient", "perp"))
            s.add_surface(sh, "Surface sweep", op["profiles"] + op["path"]); return
        if t == "s_loft":
            sh = surf_loft(s.ref_sections(op["sections"], chain), op.get("closed", False), op.get("ruled", False))
            s.add_surface(sh, "Surface loft", op["sections"]); return
        if t == "patch":
            pairs = s.ref_edges(op["boundary"], chain, loop=True); sup = []
            for e, bi in pairs:
                if bi is not None:
                    fs = faces_of_edge_in(s.bodies[bi].shape, e)
                    if len(fs) == 1: sup.append((e, fs[0]))
            sh = surf_patch([e for e, _ in pairs], sup, op.get("cont", "G0"), [s.ref_point(r).t() for r in op.get("points", [])])
            s.add_surface(sh, "Patch", op["boundary"]); return
        if t == "ruled":
            kind = op.get("kind", "tangent"); dv = s.surf_dir([], op["dirref"]) if kind == "direction" else None
            if kind == "direction" and not op.get("dirref"): raise RuntimeError("pick the direction (a plane or axis)")
            strips = []
            for e, bi in s.ref_edges(op["edges"], chain):
                f = None
                if bi is not None:
                    fs = faces_of_edge_in(s.bodies[bi].shape, e); f = fs[0] if fs else None
                strips.append(ruled_strip(e, f, op["d"], kind, op.get("ang", 0.0), dv, op.get("flip", False)))
            sh = sew(strips, solid_if_closed=False) if len(strips) > 1 else strips[0]
            s.add_surface(sh, "Ruled surface"); return
        if t == "s_offset":
            faces = []
            for r in op["faces"]:
                faces += list(s.bodies[r[1]].faces) if r[0] == "body" else [s.ref_face(r)]
            s.add_surface(surf_offset(faces, op["d"]), "Offset surface"); return
        if t == "trim":
            spots = {}
            for r in op["remove"]: spots.setdefault(r[1], []).append(V(*r[3]))
            if not spots: raise RuntimeError("click the parts of the surfaces to remove")
            tr = op["tool"]; tool_bi = tr[1] if tr[0] in ("body", "face") else None
            if tr[0] == "curve":
                sk = s.sketches[tr[1]]; base = [trim_tool_from_curve(e, sk.plane.n, s.span()*2) for e in crv_edges(sk.geo, tr[2], sk.plane)]
                s.hide_sketches_of([tr])
            elif tr[0] == "edge": raise RuntimeError("use a surface, plane, face or sketch curve as the trimming tool")
            else: base = [s.ref_tool_shape(tr)]
            for bi in sorted(spots, reverse=True):
                b = s.bodies[bi]
                if not b.surface: raise RuntimeError("Trim works on surface bodies - use Split Body for solids")
                tools = [x for x in base] if bi != tool_bi else []
                tools += [s.bodies[j].shape for j in spots if j != bi and j != tool_bi]
                if bi == tool_bi and not tools: raise RuntimeError("also click the part of the other surface to trim the tool by it")
                pcs = split_pieces(b.shape, tools)
                if len(pcs) < 2: raise RuntimeError("the tool doesn't cut that surface into pieces")
                drop = {piece_at(pcs, q) for q in spots[bi]}; keep = [p for i, p in enumerate(pcs) if i not in drop]
                if not keep: del s.bodies[bi]; continue
                s.set_body_shapes(bi, [sew(keep, solid_if_closed=False) if len(keep) > 1 else keep[0]])
            return
        if t == "untrim":
            for bi, refs in sorted(s.group_by_body(op["faces"]).items(), reverse=True):
                b = s.bodies[bi]; faces = list(b.faces) if refs is None else [b.faces[r[2]] for r in refs]
                if not b.surface: raise RuntimeError("Untrim works on surface bodies")
                m = [(f, untrim_face(f, op.get("mode", "loops"), op.get("ext", 0.0))) for f in faces]
                s.set_body_shapes(bi, [replace_faces(b.shape, m)])
            return
        if t == "s_extend":
            groups = {}
            for e, bi in s.ref_edges(op["edges"], chain):
                if bi is None: raise RuntimeError("pick edges of a surface body")
                groups.setdefault(bi, []).append(e)
            for bi, es in sorted(groups.items(), reverse=True):
                b = s.bodies[bi]
                if not b.surface: raise RuntimeError("Extend works on surface bodies")
                s.set_body_shapes(bi, [extend_edges(b.shape, es, op["d"], op.get("kind", "natural"))])
            return
        if t == "stitch":
            idx = sorted({r[1] for r in op["sbodies"]})
            if not idx: raise RuntimeError("pick the surfaces to stitch")
            sh = sew([s.bodies[i].shape for i in idx], op.get("tol", 0.01)); look = s.bodies[idx[0]]
            for i in reversed(idx): del s.bodies[i]
            s.bodies.append(Body(sh).copy_look(look)); return
        if t == "unstitch":
            for bi, refs in sorted(s.group_by_body(op["faces"]).items(), reverse=True):
                b = s.bodies[bi]; pick = list(b.faces) if refs is None else [b.faces[r[2]] for r in refs]
                rest = [f for f in b.faces if all(not f.IsSame(x) for x in pick)]
                shapes = [sew(g, solid_if_closed=False) if len(g) > 1 else g[0] for g in face_groups(rest)] + list(pick)
                s.set_body_shapes(bi, shapes)
            return
        if t == "revnormal":
            for bi, refs in sorted(s.group_by_body(op["targets"]).items(), reverse=True):
                b = s.bodies[bi]
                if not b.surface: raise RuntimeError("Reverse Normal works on surface bodies")
                s.set_body_shapes(bi, [reversed_faces(b.shape, None if refs is None else [b.faces[r[2]] for r in refs])])
            return
        if t == "s_delface":
            for bi, refs in sorted(s.group_by_body(op["faces"]).items(), reverse=True):
                b = s.bodies[bi]
                if refs is None: del s.bodies[bi]; continue
                pcs = delete_faces_open(b.shape, [b.faces[r[2]] for r in refs])
                if not pcs: del s.bodies[bi]
                else: s.set_body_shapes(bi, [compound(pcs) if len(pcs) > 1 else pcs[0]])
            return
        raise RuntimeError(f"unknown surface operation {t}")

    # ---------------- mesh workspace + file import (0.7) ----------------
    def mesh_at(s, bi, need=True):
        b = s.bodies[bi]
        if need and not getattr(b, "mesh", False): raise RuntimeError("pick mesh bodies (Insert Mesh, or Tessellate a solid first)")
        return b

    def mesh_targets(s, refs, need=True):
        idx = sorted({r[1] for r in refs if r and r[0] in ("body", "face")})
        if not idx: raise RuntimeError("pick a body")
        for i in idx: s.mesh_at(i, need)
        return idx

    def exec_mesh(s, op):
        t = op["t"]
        if t == "meshimport":
            T = unpack_arr(op["data"]).astype(float)*op.get("scale", 1.0)
            if op.get("center"):
                P = T.reshape(-1, 3); c = (P.min(0) + P.max(0))/2; c[2] = P[:, 2].min(); T = T - c
            if op.get("yup"): T = T[:, :, [0, 2, 1]]*np.array([1.0, -1.0, 1.0])            # Y-up file -> Z-up (a rotation)
            s.bodies.append(MeshBody(T, op.get("name"))); return
        if t == "cadimport":
            sh = brep_text_to_shape(unpack_text(op["brep"]))
            if op.get("scale", 1.0) != 1.0:
                k = op["scale"]; sh = transformed(sh, [[k, 0, 0, 0], [0, k, 0, 0], [0, 0, k, 0]])
            parts = split_imported(sh)
            if not parts: raise RuntimeError("that file has no solids or surfaces in it")
            base = op.get("name") or "Import"
            for i, (x, kind) in enumerate(parts):
                b = Body(x); b.name = base if len(parts) == 1 else f"{base} {i + 1}"; s.bodies.append(b)
            s.add_feature(f"Import {base}", compound([x for x, _ in parts]), "New"); return
        if t == "tessellate":
            for bi in reversed(s.mesh_targets(op["targets"], need=False)):
                b = s.bodies[bi]
                if getattr(b, "mesh", False): continue
                defl, ang = mesh_quality(b.shape, op.get("q", "med"), op.get("defl", 0.05), op.get("ang", 15.0))
                mb = MeshBody(tessellate(b.shape, defl, math.radians(ang)), getattr(b, "name", None)); mb.color, mb.material = b.color, b.material
                if op.get("keep"): s.bodies.append(mb); b2 = b.restyled(); b2.visible = False; s.bodies[bi] = b2
                else: s.bodies[bi] = mb
            return
        if t == "meshconvert":
            for bi in reversed(s.mesh_targets(op["targets"])):
                b = s.bodies[bi]; T = b.tv
                if len(T) > MESH_CONVERT_MAX: raise RuntimeError(f"that mesh has {len(T):,} triangles - Reduce it below {MESH_CONVERT_MAX:,} first")
                sh = mesh_to_brep(T, merge_planar=op.get("method", "prismatic") == "prismatic")
                if sh is None or not subshapes(sh, TopAbs_FACE): raise RuntimeError("couldn't convert that mesh")
                nb = Body(sh); nb.color, nb.material, nb.name = b.color, b.material, getattr(b, "name", None); s.bodies[bi] = nb
            return
        if t in ("mrev", "mrepair", "msmooth", "mreduce"):
            for bi in s.mesh_targets(op["targets"]):
                b = s.bodies[bi]; T = b.tv
                if t == "mrev": T2 = T[:, [0, 2, 1]]
                elif t == "mrepair": T2 = repair_mesh(T, op.get("holes", True))
                elif t == "msmooth": T2 = taubin_smooth(T, op.get("iters", 10), 0.33 + 0.3*op.get("strength", 0.5))
                else: T2 = cluster_reduce(T, max(0.01, min(1.0, op.get("pct", 50.0)/100.0)))
                if not len(T2): raise RuntimeError("nothing would be left of that mesh")
                s.bodies[bi] = s.mesh_like(b, T2)
            return
        if t == "msep":
            for bi in reversed(s.mesh_targets(op["targets"])):
                b = s.bodies[bi]; Vv, F = weld(b.tv); comps = components(F, len(Vv))
                if len(comps) < 2: continue
                comps.sort(key=lambda c: -len(c))
                s.bodies[bi:bi + 1] = [s.mesh_like(b, Vv[F[c]], f"{getattr(b, 'name', None) or 'Mesh'} {k + 1}") for k, c in enumerate(comps)]
            return
        if t == "mcut":
            pl = s.ref_plane(op["plane"])
            n = pl.n*(-1 if op.get("flip") else 1); keep = op.get("keep", "below")
            for bi in reversed(s.mesh_targets(op["targets"])):
                b = s.bodies[bi]; out = []
                for below in ((True, False) if keep == "both" else ((keep == "below"),)):
                    T2, segs = clip_by_plane(b.tv, pl.o, n, below)
                    if op.get("cap", True) and segs:
                        loops = [L for L in chain_segments(segs) if len(L) > 3 and np.linalg.norm(L[0] - L[-1]) < 1e-6*(1 + np.ptp(L, 0).max())]
                        cap = cap_loops(loops, n if below else -n)
                        if len(cap): T2 = np.concatenate([T2, cap])
                    if len(T2): out.append(T2)
                if not out: raise RuntimeError("the plane doesn't leave anything of that mesh")
                s.bodies[bi:bi + 1] = [s.mesh_like(b, T2, (getattr(b, "name", None) if k == 0 else None)) for k, T2 in enumerate(out)]
            return
        if t == "mcombine":
            ti = op["target"][1]; tools = sorted({r[1] for r in op["tools"] if r[1] != ti}, reverse=True)
            if not tools: raise RuntimeError("pick the tool bodies")
            tb = s.bodies[ti]; how = op.get("op", "merge")
            if how == "merge":
                T = np.concatenate([tb.tv] + [s.bodies[i].tv for i in tools])
            else:
                for i in [ti] + tools:
                    if len(s.bodies[i].tv) > MESH_CONVERT_MAX: raise RuntimeError(f"Join / Cut / Intersect need meshes under {MESH_CONVERT_MAX:,} triangles - Reduce them first")
                sh = s.bodies[ti].shape
                for i in tools: sh = boolean(sh, s.bodies[i].shape, {"join": "fuse", "cut": "cut", "intersect": "common"}[how])
                T = tessellate(sh, 0.01, 0.5)
                if not len(T): raise RuntimeError("nothing is left after that combine")
            s.bodies[ti] = s.mesh_like(tb, T)
            if not op.get("keep"):
                for i in tools: del s.bodies[i]
            return
        if t == "msection":
            pl = s.ref_plane(op["plane"]); bi = s.mesh_targets([op["target"]], need=False)[0]; b = s.bodies[bi]
            T = b.tv if getattr(b, "mesh", False) else tessellate(b.shape, 0.02, 0.3)
            lines = chain_segments(section_segments(T, pl.o, pl.n))
            if not lines: raise RuntimeError("the plane doesn't cross that body")
            g = Geo(); tol = op.get("tol", 0.05)
            for L in lines:
                closed = len(L) > 3 and np.linalg.norm(L[0] - L[-1]) < 1e-6*(1 + np.ptp(L, 0).max())
                P = simplify_polyline(L, tol) if tol > 0 else L
                loc = [pl.l(V(*map(float, q))) for q in P]
                if closed and len(loc) > 2 and (loc[0] - loc[-1]).Length < 1e-9: loc = loc[:-1]
                if closed:                                                   # the loop's seam point may sit mid-edge
                    k = 0
                    while len(loc) > 3 and k < len(loc):
                        a_, b_, c_ = loc[k - 1], loc[k], loc[(k + 1) % len(loc)]; ac = c_ - a_
                        if ac.Length > 1e-12 and (b_ - a_).cross(ac).Length/ac.Length <= max(tol, 1e-9): del loc[k]
                        else: k += 1
                if len(loc) < 2: continue
                ids = [g.add_pt(q.x, q.y) for q in loc]
                for a, c in zip(ids, ids[1:] + ([ids[0]] if closed and len(ids) > 2 else [])): g.add("line", [a, c])
            if not g.C: raise RuntimeError("the section is too small at that tolerance")
            g.touch(); sk = Sketch(pl, None, g); sk.name = f"Section of {getattr(b, 'name', None) or 'Body' + str(bi + 1)}"
            s.sketches.append(sk); s.active = sk; return
        raise RuntimeError(f"unknown mesh operation {t}")

    def mesh_like(s, b, T, name=None):
        mb = MeshBody(T, name if name is not None else getattr(b, "name", None)); mb.color, mb.material = b.color, b.material
        mb.visible = b.visible; return mb

    def exec_solid(s, op):
        """New-style operations. Returns True when op was handled here."""
        t = op["t"]
        if t in SURF_OPS: s.exec_surface(op); return True
        if t in MESH_OPS: s.exec_mesh(op); return True
        if t in ASM_OPS: s.exec_asm(op); return True
        if t in SM_OPS: s.exec_sheet(op); return True
        if t in PL_OPS: s.exec_plastic(op); return True
        if t == "bodyname": nb = s.bodies[op["bi"]].restyled(); nb.name = op["name"]; s.bodies[op["bi"]] = nb; return True
        if t == "extrude" and "profiles" in op: s.exec_extrude2(op); return True
        if t == "revolve" and "profiles" in op: s.exec_revolve2(op); return True
        if t == "construct": s.exec_construct(op); return True
        if t == "sweep":
            prof = s.profile_faces(op["profiles"]); path = s.ref_wire(op["path"])
            guide = s.ref_wire(op["guide"]) if op.get("type") == "guide" and op.get("guide") else None
            tool = fuse_all([sweep_face(f, path, guide, op.get("dist", 1.0), op.get("taper", 0), op.get("twist", 0), op.get("orient", "perp"))
                             for f, n in prof])
            res = s.apply_op(tool, op["op"], True, s.ref_profile_parent(op["profiles"][0]), op.get("bodies"))
            s.hide_sketches_of(op["profiles"] + op["path"]); s.add_feature(f"Sweep ({res.lower()})", tool, res); return True
        if t == "loft":
            secs = []
            for r in op["sections"]:
                if r[0] in ("profile", "face"): secs.append(wires_of(s.ref_face(r))[0])
                else: secs.append(BRepBuilderAPI_MakeVertex(pnt(s.ref_point(r))).Vertex())
            rails = [s.ref_wire([r]) for r in op.get("rails", [])] if op.get("guide") == "rails" else None
            cl = s.ref_wire(op["centerline"]) if op.get("guide") == "centerline" and op.get("centerline") else None
            tool = loft_sections(secs, rails, cl, op.get("closed", False), op.get("ruled", False))
            res = s.apply_op(tool, op["op"], True, None, op.get("bodies"))
            s.hide_sketches_of(op["sections"]); s.add_feature(f"Loft ({res.lower()})", tool, res); return True
        if t in ("rib", "web"):
            pieces = []
            for r in op["curves"]:
                sk = s.sketches[r[1]]; name, kind, data, _ = sketch_curves(sk)[r[2]]
                if kind != "seg": raise RuntimeError("ribs and webs need straight sketch lines")
                a, b = sk.plane.w(data[0]), sk.plane.w(data[1]); n = sk.plane.n; dl = (b - a).normalize()
                m = n.cross(dl) * (-1 if op.get("flip") else 1); T = op["thick"]; big = s.span()*3
                side = {"sym": (-0.5, 0.5), "one": (0, 1), "other": (-1, 0)}[op.get("side", "sym")]
                if t == "rib":                                         # thickness across the sketch plane, grows in-plane
                    depth = op["depth"] if op.get("extent") == "dist" else big
                    q = [a + n*(side[0]*T), b + n*(side[0]*T)]
                    f = BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakePolygon(pnt(q[0]), pnt(q[1]), pnt(q[1] + n*T), pnt(q[0] + n*T), True).Wire(), True).Face()
                    slab = prism(f, m, depth)
                else:                                                  # web: thickness in-plane, grows along the normal
                    ext = big if op.get("extend", True) else 0
                    aa, bb = a - dl*ext, b + dl*ext
                    f = open_wire_thick_face([aa, bb], T, n, {"sym": "center", "one": "left", "other": "right"}[op.get("side", "sym")])
                    depth = op["depth"] if op.get("extent") == "dist" else big
                    slab = prism(f, -n if op.get("flip") else n, depth)
                if op.get("extent") != "dist" or t == "web":            # trim to the empty space between the body walls
                    slo, shi = bbox(slab)
                    def near(b_):
                        return all(b_.lo.t()[i] <= shi.t()[i] + 1e-6 and b_.hi.t()[i] >= slo.t()[i] - 1e-6 for i in range(3))
                    bodies = [x.shape for x in s.bodies if x.visible and near(x)]
                    if not bodies: raise RuntimeError(f"the {t} doesn't reach any body")
                    from OCP.BOPAlgo import BOPAlgo_Splitter as _Sp
                    sp = _Sp(); sp.AddArgument(slab)
                    for bs in bodies: sp.AddTool(bs)
                    sp.Perform(); cells = [c_ for c_ in subshapes(sp.Shape(), TopAbs_SOLID) if not any(classify(bs, centroid(c_)) == TopAbs_IN for bs in bodies)]
                    grow = m if t == "rib" else (-n if op.get("flip") else n)
                    seed, step, span_ = (a + b)*0.5 + grow*1e-3, max(T/4, 0.05), s.span()
                    for _k in range(int(span_/step) + 2):                # walk out of any material the line sits in
                        if not any(classify(bs, seed) == TopAbs_IN for bs in bodies): break
                        seed = seed + grow*step
                    hit = [c_ for c_ in cells if classify(c_, seed) in (TopAbs_IN, TopAbs_ON)]
                    if not hit: raise RuntimeError(f"couldn't find the space for the {t} - check its direction (Flip)")
                    slab = hit[0]; lo_, hi_ = bbox(slab)
                    if (hi_ - lo_).Length > span_*1.5: raise RuntimeError(f"the {t} doesn't meet the body on every side - it would run off forever")
                pieces.append(slab)
            tool = fuse_all(pieces)
            tlo, thi = bbox(tool)
            def nearb(b_): return all(b_.lo.t()[i] <= thi.t()[i] + 1e-4 and b_.hi.t()[i] >= tlo.t()[i] - 1e-4 for i in range(3))
            touching = [i for i, b_ in enumerate(s.bodies) if b_.visible and nearb(b_) and shape_dist(b_.shape, tool) < 1e-5]   # ribs touch, not overlap
            if op.get("op", "Join") == "Join" and touching: s.join(touching, tool); res = "Join"
            else: res = s.apply_op(tool, op.get("op", "Join"), True, None, op.get("bodies"))
            s.hide_sketches_of(op["curves"]); s.add_feature(f"{t.title()} ({res.lower()})", tool, res); return True
        if t == "emboss":
            face = s.ref_face(op["face"]); bi = op["face"][1]; B = s.bodies[bi].shape; D = op["depth"]; big = s.span()*3
            prof = s.profile_faces(op["profiles"]); P = fuse_all([prism(as_face(translated(f, -n*big)), n, 2*big) for f, n in prof])
            off = BRepOffsetAPI_MakeOffsetShape(); off.PerformByJoin(B, D if op.get("mode") == "emboss" else -D, 1e-4, BRepOffset_Skin, False, False, GeomAbs_Intersection)
            grown = as_solid(off.Shape())
            shell = boolean(grown, B, "cut") if op.get("mode") == "emboss" else boolean(B, grown, "cut")
            tool = boolean(P, shell, "common"); cells = [c for c in subshapes(tool, TopAbs_SOLID) if shape_dist(c, face) < 1e-4]
            if not cells: raise RuntimeError("the profile doesn't land on that face")
            r0 = op["profiles"][0]; skn = s.sketches[r0[1]].plane
            dots = [abs((centroid(c) - skn.o).dot(skn.n)) for c in cells]     # only the wall nearest the sketch (not the far side)
            cells = [c for c, d_ in zip(cells, dots) if d_ <= min(dots) + 3*D + 1e-3]
            tool = fuse_all(cells)
            s.replace_body(bi, boolean(B, tool, "fuse" if op.get("mode") == "emboss" else "cut"))
            s.hide_sketches_of(op["profiles"]); s.add_feature(f"{op.get('mode', 'emboss').title()}", tool, "Join" if op.get("mode") == "emboss" else "Cut"); return True
        if t == "hole": s.exec_hole(op); return True
        if t == "thread2": s.exec_thread2(op); return True
        if t == "prim":
            pl = s.ref_plane(op.get("plane") or ("oplane", "XY")); sh = op["shape"]; cx, cy = op.get("cx", 0), op.get("cy", 0)
            if sh == "box": tool = make_box(pl, cx, cy, op["L"], op["W"], op["H"])
            elif sh == "cyl": tool = make_cylinder(pl, cx, cy, op["D"], op["H"])
            elif sh == "sphere": tool = make_sphere(pl, cx, cy, op["D"])
            elif sh == "torus": tool = make_torus(pl, cx, cy, op["D"], op["D2"])
            elif sh == "coil": tool = make_coil(pl, cx, cy, op["D"], op["revs"], op["H"], op["pitch"], op["mode"], op["angle"], op["section"], op["secpos"], op["size"])
            elif sh == "pipe": tool = make_pipe(s.ref_wire(op["path"]), op["section"], op["size"], op.get("hollow"), op.get("wall", 1.0), op.get("dist", 1.0))
            else: raise RuntimeError(f"unknown shape {sh}")
            res = s.apply_op(tool, op.get("op", "New"), True, None, op.get("bodies"))
            if sh == "pipe": s.hide_sketches_of(op["path"])
            s.add_feature(f"{dict(box='Box', cyl='Cylinder', sphere='Sphere', torus='Torus', coil='Coil', pipe='Pipe')[sh]} ({res.lower()})", tool, res, pl); return True
        if t in ("rpattern", "cpattern", "ppattern"):
            objs = s.pattern_objects(op); skip = s.skip_set(op.get("skip")); mats = []
            sk_plane = s.sketches[objs[0][1][0]].plane if objs[0][0] == "ent" else None
            def dir_of(r):
                if r is None: return None
                p, d = s.ref_axis(r)
                if sk_plane is not None:
                    d = d - sk_plane.n*d.dot(sk_plane.n)
                    if d.Length < 1e-9: raise RuntimeError("that direction is normal to the sketch")
                    d = d.normalize()
                return d
            if t == "rpattern":
                d1 = dir_of(op.get("d1")); d2 = dir_of(op.get("d2"))
                if d1 is None: raise RuntimeError("pick the first direction")
                def offs(q, dist, mode, sym):
                    step = dist/max(q - 1, 1) if mode == "extent" else dist
                    return [k*step for k in (range(-(q - 1), q) if sym else range(q))]
                o1 = offs(op["q1"], op["s1"], op.get("m1", "spacing"), op.get("sym1")); o2 = offs(op.get("q2", 1), op.get("s2", 0), op.get("m2", "spacing"), op.get("sym2")) if d2 is not None else [0.0]
                n = 0
                for b_ in o2:
                    for a_ in o1:
                        n += 1
                        if (abs(a_) < 1e-12 and abs(b_) < 1e-12) or n in skip: continue
                        v = d1*a_ + (d2*b_ if d2 is not None else V())
                        mats.append([[1, 0, 0, v.x], [0, 1, 0, v.y], [0, 0, 1, v.z]])
            elif t == "cpattern":
                p, d = s.ref_axis(op["axis"]); q = op["q"]; ang = op.get("angle", 360.0)
                full = abs(abs(ang) - 360) < 1e-6; step = math.radians(ang/q if full else ang/max(q - 1, 1))
                ks = range(-(q - 1)//2, q//2 + 1) if op.get("sym") and not full else range(q)
                if op.get("sym") and not full: step = math.radians(ang/max(q - 1, 1))
                for idx, k in enumerate(ks, 1):
                    if k == 0 or idx in skip: continue
                    mats.append(rot_matrix(p, d, step*k))
            else:
                w = s.ref_wire(op["path"]); L = wire_length(w); q = op["q"]
                step = (op["dist"]/max(q - 1, 1) if op.get("mode") == "extent" else op["dist"])/L
                p0, t0 = wire_eval(w, 0.0)
                for k in range(1, q):
                    if (k + 1) in skip: continue
                    f = step*k*(-1 if op.get("flip") else 1)
                    if not 0 <= f <= 1.0000001: continue
                    p1, t1 = wire_eval(w, min(max(f, 0), 1)); M = [[1, 0, 0, p1.x - p0.x], [0, 1, 0, p1.y - p0.y], [0, 0, 1, p1.z - p0.z]]
                    if op.get("orient") == "path":
                        ax = t0.cross(t1)
                        if ax.Length > 1e-9:
                            R = rot_matrix(p0, ax, math.atan2(ax.Length, t0.dot(t1)))
                            Rn = np.array(R); Mn = np.array(M); comb = Rn[:, :3]; tr = Rn[:, 3] + Mn[:, 3]
                            M = [list(comb[i]) + [tr[i]] for i in range(3)]
                    mats.append(M)
            if not mats: raise RuntimeError("no copies to make - check the quantity")
            s.place_copies(objs, mats, op.get("op", "New")); return True
        if t == "mirror":
            pl = s.ref_plane(op["plane"]); objs = s.pattern_objects(op)
            T_ = gp_Trsf(); T_.SetMirror(gp_Ax2(pnt(pl.o), gdir(pl.n)))
            if objs[0][0] == "ent":
                n = pl.n; R = np.eye(3) - 2*np.outer(n.t(), n.t()); tr = 2*n.dot(pl.o)*np.array(n.t())
                mats = [[list(R[i]) + [tr[i]] for i in range(3)]]
                s.place_copies(objs, mats, "New"); return True
            new = []
            for kind, x in objs:
                if kind == "body": new.append(mirrored(s.bodies[x].shape, pl.o, pl.n))
                else:
                    f = s.features[x]; tool = mirrored(f["tool"], pl.o, pl.n)
                    if f["op"] == "Cut": s.cut_all(tool)
                    elif f["op"] == "Join": s.apply(tool, "Join", True, None)
                    else: s.bodies += [Body(y) for y in solids(tool)]
            if new:
                if op.get("op") == "Join":
                    srcs = sorted({x for k, x in objs if k == "body"}); shape = fuse_all([s.bodies[i].shape for i in srcs] + new)
                    for i in reversed(srcs[1:]): del s.bodies[i]
                    s.replace_body(srcs[0], shape)
                else: s.bodies += [Body(y) for b in new for y in solids(b)]
            return True
        if t == "thicken":
            faces = [s.ref_face(r) for r in op["faces"]]; th = op["thk"]*(-1 if op.get("flip") else 1)
            tool = thicken_faces(faces, abs(th), op.get("dir") == "sym") if th > 0 or op.get("dir") == "sym" else thicken_faces([to_face(f.Reversed()) for f in faces], abs(th), False)
            res = s.apply_op(tool, op.get("op", "New"), True, op["faces"][0][1] if op["faces"][0][0] == "face" else None)
            s.add_feature(f"Thicken ({res.lower()})", tool, res); return True
        if t == "bfill":
            tools = [s.ref_tool_shape(r) for r in op["tools"]]; cells = boundary_cells(tools)
            sel = s.skip_set(op.get("cells")) if str(op.get("cells", "all")).strip().lower() != "all" else set(range(1, len(cells) + 1))
            keep = [c for i, c in enumerate(cells, 1) if i in sel]
            if not keep: raise RuntimeError(f"pick cells between 1 and {len(cells)}")
            tool = fuse_all(keep)
            if op.get("remove_tools"):
                for r in sorted([r for r in op["tools"] if r[0] == "body"], key=lambda r: -r[1]): del s.bodies[r[1]]
            res = s.apply_op(tool, op.get("op", "New"), True, None)
            s.add_feature(f"Boundary fill ({res.lower()})", tool, res); return True
        if t == "presspull":
            r = op["ref"]
            if r[0] == "edge": return s.exec_solid(dict(t="fillet2", edges=[r], mode="constant", r=abs(op["d"]), chain=True))
            return s.exec_solid(dict(t="offsetface", faces=[r], d=op["d"]))
        if t in ("fillet2", "chamfer2"): s.exec_edge2(op); return True
        if t == "shell":
            groups = {}
            for r in op.get("faces", []): groups.setdefault(r[1], []).append(s.ref_face(r))
            for r in op.get("bodies", []): groups.setdefault(r[1], [])
            if not groups: raise RuntimeError("pick the faces to remove (or a body to hollow out)")
            for bi, faces in groups.items(): s.replace_body(bi, shell_solid(s.bodies[bi].shape, faces, op["thk"], op.get("dir", "in")))
            return True
        if t == "draft":
            pl = s.ref_plane(op["pull"]); pull = -pl.n if op.get("flip") else pl.n; groups = {}
            for r in op["faces"]: groups.setdefault(r[1], []).append(s.ref_face(r))
            for bi, faces in groups.items(): s.replace_body(bi, draft_faces(s.bodies[bi].shape, faces, pull, pl.o, pl.n, op["angle"]))
            return True
        if t == "scale":
            for r in op["bodies"]:
                bi = r[1]; b = s.bodies[bi]; c = s.ref_point(op["point"]) if op.get("point") else (b.lo + b.hi)*0.5
                if getattr(b, "mesh", False):
                    k = np.array([op["sx"], op["sy"], op["sz"]] if op.get("type") == "nonuniform" else [op["s"]]*3, float); cc = np.array(c.t())
                    s.bodies[bi] = s.mesh_like(b, (b.tv - cc)*k + cc); continue
                if op.get("type") == "nonuniform": sh = scaled(b.shape, c, op["sx"], op["sy"], op["sz"])
                else: sh = scaled(b.shape, c, op["s"])
                s.bodies[bi] = Body(sh)
            return True
        if t == "combine":
            ti = op["target"][1]; tools = [r[1] for r in op["tools"] if r[1] != ti]
            if not tools: raise RuntimeError("pick at least one tool body")
            shape = s.bodies[ti].shape
            for i in tools: shape = boolean(shape, s.bodies[i].shape, {"Join": "fuse", "Cut": "cut", "Intersect": "common"}[op["op"]])
            keep = [Body(y) for y in solids(clean(shape))]
            drop = set() if op.get("keep") else set(tools)
            if op.get("new"): rest = [b for i, b in enumerate(s.bodies) if i not in drop]; s.bodies = rest + keep
            else:
                drop.add(ti); rest = [b for i, b in enumerate(s.bodies) if i not in drop]; ins = min(ti, len(rest))
                s.bodies = rest[:ins] + keep + rest[ins:]
            return True
        if t == "offsetface":
            groups = {}
            for r in op["faces"]: groups.setdefault(r[1], []).append(s.ref_face(r))
            for bi, faces in groups.items(): s.replace_body(bi, offset_faces(s.bodies[bi].shape, faces, op["d"]))
            return True
        if t == "replaceface":
            groups = {}
            for r in op["faces"]: groups.setdefault(r[1], []).append(s.ref_face(r))
            tgt = op["target"]
            for bi, faces in groups.items():
                if tgt[0] in ("oplane", "cplane") or (tgt[0] == "face" and s.bodies[tgt[1]].face_normal(tgt[2]) is not None):
                    pl = s.ref_plane(tgt); s.replace_body(bi, replace_with_plane(s.bodies[bi].shape, faces, pl.o, pl.n, s.span()*3))
                else:
                    srf = s.ref_tool_shape(tgt); sh = s.bodies[bi].shape
                    for f in faces:
                        c, fn = face_frame(f); big = s.span()*3
                        grow = split_keep(prism(f, fn, big), srf, c + fn*1e-3)
                        sh = boolean(sh, grow, "fuse")
                    s.replace_body(bi, sh)
            return True
        if t == "splitface":
            tool = s.ref_tool_shape(op["tool"], op.get("extend", True)); groups = {}
            for r in op["faces"]: groups.setdefault(r[1], []).append(s.ref_face(r))
            for bi, faces in groups.items(): s.bodies[bi] = Body(split_faces(s.bodies[bi].shape, faces, tool))
            return True
        if t == "splitbody":
            bi = op["body"][1]; parts = split_body(s.bodies[bi].shape, s.ref_tool_shape(op["tool"], op.get("extend", True)))
            s.bodies[bi:bi + 1] = [Body(p) for p in parts]; return True
        if t == "defeature":
            groups = {}
            for r in op["faces"]: groups.setdefault(r[1], []).append(s.ref_face(r))
            for bi, faces in groups.items(): s.replace_body(bi, defeature(s.bodies[bi].shape, faces))
            return True
        if t == "move2":
            bis = sorted({r[1] for r in op["bodies"]}); mode = op.get("mode", "p2p")
            if mode == "p2p":
                v = s.ref_point(op["to"]) - s.ref_point(op["from"]); M = [[1, 0, 0, v.x], [0, 1, 0, v.y], [0, 0, 1, v.z]]
            elif mode == "rotate":
                p, d = s.ref_axis(op["axis"]); M = rot_matrix(p, d, math.radians(op["angle"]))
            else:
                p, d = s.ref_axis(op["axis"]); v = d*op["dist"]; M = [[1, 0, 0, v.x], [0, 1, 0, v.y], [0, 0, 1, v.z]]
            for bi in bis:
                nb = xform_body(s.bodies[bi], M)
                if op.get("copy"): s.bodies.append(nb)
                else: s.bodies[bi] = nb
            return True
        if t == "align":
            src, dst = op["from"], op["to"]; bi = src[1] if src[0] in ("face", "edge", "vertex", "body") else None
            if bi is None: raise RuntimeError("the first selection must be on the body you want to move")
            M = s.align_matrix(src, dst, op.get("flip"))
            nb = xform_body(s.bodies[bi], M)
            if op.get("copy"): s.bodies.append(nb)
            else: s.bodies[bi] = nb
            return True
        if t in ("material", "appearance"):
            for r in op["targets"]:
                bi = r[1]; b = s.bodies[bi]; nb = b.restyled()
                if t == "material":
                    nb.material = op["mat"]
                    if op.get("recolor", True) and op["mat"] in MAT_COLOR and not b.color: nb.color = MAT_COLOR[op["mat"]]
                elif r[0] == "face": nb.face_colors = dict(b.face_colors); nb.face_colors[r[2]] = op["color"]
                else: nb.color = op["color"]; nb.face_colors = {}
                s.bodies[bi] = nb
            return True
        return False

    def align_matrix(s, src, dst, flip):
        def geom(r):
            if r[0] == "face" and s.bodies[r[1]].face_normal(r[2]) is not None:
                c, n = face_frame(s.bodies[r[1]].faces[r[2]]); return "plane", c, n
            if r[0] in ("oplane", "cplane"): pl = s.ref_plane(r); return "plane", pl.o, pl.n
            if r[0] in ("edge", "curve", "caxis", "oaxis") or (r[0] == "face" and s.bodies[r[1]].face_cylinder(r[2])):
                p, d = s.ref_axis(r); return "axis", p, d
            return "point", s.ref_point(r), None
        k1, p1, d1 = geom(src); k2, p2, d2 = geom(dst)
        R = np.eye(3)
        if d1 is not None and d2 is not None:
            target = -d2 if (k1 == "plane" and k2 == "plane") != bool(flip) else d2
            a, b = np.array(d1.t()), np.array(target.t()); v = np.cross(a, b); c = float(a @ b)
            if np.linalg.norm(v) < 1e-12:
                if c > 0: R = np.eye(3)
                else:                                                  # opposite: half turn about any perpendicular axis
                    u = np.array((V(1, 0, 0) if abs(d1.x) < 0.9 else V(0, 1, 0)).cross(d1).normalize().t()); R = 2*np.outer(u, u) - np.eye(3)
            else:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); R = np.eye(3) + vx + vx @ vx*(1/(1 + c))
        q1 = R @ np.array(p1.t()); tr = np.array(p2.t()) - q1
        if k2 == "plane" and k1 == "plane": tr = np.array(d2.t())*((np.array(p2.t()) - q1) @ np.array(d2.t()))
        return [list(R[i]) + [tr[i]] for i in range(3)]

    def exec_edge2(s, op):
        """Fillet / chamfer with tangent chains, faces (all their edges), rule and full-round fillets."""
        groups = {}
        def add(bi, ei):
            g = groups.setdefault(bi, [])
            if ei not in g: g.append(ei)
        mode = op.get("mode", "constant")
        if op["t"] == "fillet2" and mode == "rule":
            setA = [r for r in op["edges"] if r[0] == "face"]; setB = [r for r in op.get("faces2", []) if r[0] == "face"]
            for ra in setA:
                b = s.bodies[ra[1]]; ea = set(b.face_edges(ra[2]))
                eb = set(i for rb in setB if rb[1] == ra[1] for i in b.face_edges(rb[2])) if setB else set(range(len(b.eds)))
                for ei in ea & eb if setB else ea: add(ra[1], ei)
        elif op["t"] == "fillet2" and mode == "fullround":
            for r in op["edges"]:
                if r[0] != "face": continue
                b = s.bodies[r[1]]; f = b.faces[r[2]]; c, n = face_frame(f)
                eis = b.face_edges(r[2]); lens = []
                for ei in eis:
                    pl = b.edges[ei]; lens.append((float(np.linalg.norm(np.diff(pl, axis=0), axis=1).sum()), ei))
                lens.sort(reverse=True); e1, e2 = lens[0][1], lens[1][1]
                d = shape_dist(b.eds[e1], b.eds[e2]); op = dict(op, r=d/2*0.995, r2=0)
                add(r[1], e1); add(r[1], e2)
        else:
            for r in op["edges"]:
                b = s.bodies[r[1]]
                eis = b.face_edges(r[2]) if r[0] == "face" else [r[2]]
                for ei in eis:
                    for ej in (tangent_chain(b, ei) if op.get("chain", True) else [ei]): add(r[1], ej)
        if not groups: raise RuntimeError("pick edges or faces")
        new = list(s.bodies)
        for bi, eis in groups.items():
            body = s.bodies[bi]; edges = [body.eds[e] for e in eis]
            if op["t"] == "fillet2":
                mk = BRepFilletAPI_MakeFillet(body.shape)
                if op.get("g2"): mk.SetFilletShape(ChFi3d_Polynomial)
                for e in edges:
                    if mode == "variable" and op.get("r2"): mk.Add(op["r"], op["r2"], e)
                    else: mk.Add(op["r"], e)
                mk.Build()
                if not mk.IsDone(): raise RuntimeError("OpenCascade could not build this fillet - try a smaller radius")
                sh = mk.Shape()
            else:
                faces = None
                if op.get("mode", "eq") != "eq":
                    faces = []
                    for e in edges:
                        fs = body.faces_of_edge(e); faces.append(fs[1] if op.get("flip") and len(fs) > 1 else fs[0])
                sh = chamfer(body.shape, edges, op["d"], op.get("mode", "eq"), op.get("angle", 45.0), op.get("d2", op["d"]), faces)
            nb = Body(sh); nb.copy_look(body); new[bi] = nb
        s.bodies = new

    def exec_hole(s, op):
        pts = [s.ref_point(r) for r in op.get("points", [])]
        face_r = op.get("face")
        if not pts and face_r and len(face_r) > 3: pts = [V(*face_r[3])]
        if not pts: raise RuntimeError("pick a face (where you click places the hole) or sketch points")
        if face_r: c0, n = normal_at(s.ref_face(face_r), pts[0])
        elif op["points"][0][0] == "spoint": n = s.sketches[op["points"][0][1]].plane.n
        else: raise RuntimeError("pick the face the holes go into")
        d = -n; big = s.span()*3
        size = op.get("size") or ""; dia = op["dia"]; major = pitch = None
        if op.get("tap") in ("tapped", "clearance") and size:
            std, lab = size.split("|"); row = next((x for x in THREADS[std] if x[0] == lab), None)
            if row: major, pitch = row[1], row[2]
            if op["tap"] == "clearance" and major: dia = major*1.1
            elif op["tap"] == "tapped" and major: dia = major - 1.0825*pitch
        spec = dict(op, dia=dia)
        if op.get("extent") == "to" and op.get("to"):
            pl = s.ref_plane(op["to"]); den = d.dot(pl.n)
            if abs(den) < 1e-9: raise RuntimeError("that plane is parallel to the hole")
            spec["depth"] = abs((pl.o - pts[0]).dot(pl.n)/den); spec["extent"] = "dist"
        tools = []
        for p in pts:
            if face_r: p, nn = normal_at(s.ref_face(face_r), p); d = -nn
            tools.append(hole_tool(p, d, spec, big))
        tool = fuse_all(tools)
        res = s.apply_op(tool, "Cut", False, None, op.get("bodies"))
        if op.get("tap") == "tapped" and major:
            length = spec["depth"]
            if spec.get("extent") == "all":                              # through all: thread the material actually drilled
                lo_, hi_ = bbox(compound([b_.shape for b_ in s.bodies if b_.visible]))
                cs = [V(x, y, z) for x in (lo_.x, hi_.x) for y in (lo_.y, hi_.y) for z in (lo_.z, hi_.z)]
                length = max(1e-3, max((c_ - pts[0]).dot(d) for c_ in cs))
            for p in pts:
                if face_r: p, nn = normal_at(s.ref_face(face_r), p); d = -nn
                x = V(1, 0, 0) if abs(d.x) < 0.9 else V(0, 1, 0); x = (x - d*x.dot(d)).normalize()
                if op.get("modeled"):
                    thr_len = min(length, op.get("tlen") or length) if not op.get("full", True) else length
                    tt = thread_tool2(p, d, x, dia/2, pitch, thr_len, False, THREAD_PROFILE.get(size.split("|")[0], 60.0))
                    s.cut_all(tt)
                else:
                    s.features.append(dict(name="Thread (cosmetic)", tool=compound([]), op="None", plane=XY,
                                           cosmetic=(p, d, major/2, min(length, s.span()), pitch)))
        s.hide_sketches_of(op.get("points", []))
        s.add_feature("Hole", tool, "Cut")

    def exec_thread2(s, op):
        for r in op["faces"]:
            bi, fid = r[1], r[2]; cyl = s.bodies[bi].face_cylinder(fid)
            if not cyl: raise RuntimeError("threads go on cylindrical faces")
            o, a, xd, R, v0, v1, ext = cyl
            size = op.get("size") or "auto"
            if size == "auto":
                d_, p_ = iso_suggest(2*R, ext); std = "ISO Metric coarse"
            else:
                std, lab = size.split("|"); row = next(x for x in THREADS[std] if x[0] == lab); p_ = row[2]
            L = (v1 - v0) if op.get("full", True) else min(op.get("len", v1 - v0), v1 - v0 - op.get("off", 0))
            start = o + a*(v0 + (0 if op.get("full", True) else op.get("off", 0)))
            if op.get("modeled", True):
                tool = thread_tool2(start, a, xd, R, p_, L, ext, THREAD_PROFILE.get(std, 60.0), op.get("lh", False))
                s.replace_body(bi, boolean(s.bodies[bi].shape, tool, "cut"))
            else:
                s.features.append(dict(name="Thread (cosmetic)", tool=compound([]), op="None", plane=XY, cosmetic=(start, a, R, L, p_)))

# ---------------------------------------------------------------------------------------------------------------
#  Sketch environment in the viewport: drawing tools, snapping, constraints, dimensions, dragging
# ---------------------------------------------------------------------------------------------------------------
SKC = dict(free=(0.10, 0.38, 0.86), full=(0.07, 0.07, 0.09), cons=(0.92, 0.50, 0.10), proj=(0.58, 0.26, 0.78), fix=(0.10, 0.56, 0.22),
           sel=(0.05, 0.62, 1.0), hot=(0.98, 0.60, 0.10), pre=(0.10, 0.55, 1.0), other=(0.40, 0.50, 0.66), bad=(0.86, 0.16, 0.16))

# tool id: (title, icon, clicks (None = open ended / special), hint)
SK_TOOLS = {
    "line": ("Line", "poly", None, "Click points to draw connected lines. Drag from the last point to draw a tangent arc. "
             "Click the first point to close; double-click, Enter or Esc to stop. Just type a length (e.g. 30 mm) and press Enter."),
    "rect2": ("2-Point Rectangle", "rect", 2, "Click two opposite corners (or type 'width, height' and press Enter)."),
    "rect3": ("3-Point Rectangle", "rect3", 3, "Click the two ends of one side, then the width."),
    "rectc": ("Center Rectangle", "rectc", 2, "Click the centre, then a corner (or type 'width, height' and press Enter)."),
    "circle": ("Center Diameter Circle", "circle", 2, "Click the centre, then a point on the circle (or type the radius and press Enter)."),
    "circle2p": ("2-Point Circle", "circle2p", 2, "Click the two ends of a diameter (or type the diameter after the first click)."),
    "circle3p": ("3-Point Circle", "circle3p", 3, "Click three points on the circle."),
    "circle2t": ("2-Tangent Circle", "circle2t", None, "Click two lines, then where the circle goes (that sets its size)."),
    "circle3t": ("3-Tangent Circle", "circle3t", None, "Click three lines, circles or arcs to touch."),
    "arc3p": ("3-Point Arc", "arc3p", 3, "Click the start, the end, then a point the arc passes through."),
    "arcc": ("Center Point Arc", "arcc", 3, "Click the centre, the start, then sweep round to the end. Type the radius / sweep angle to place points exactly."),
    "arct": ("Tangent Arc", "arct", None, "Click the end of a line or curve, then where the arc should end."),
    "poly_circ": ("Circumscribed Polygon", "polyc", 2, "Click the centre, then the middle of an edge. Sides: set in the Sketch Palette."),
    "poly_insc": ("Inscribed Polygon", "polyi", 2, "Click the centre, then a corner. Sides: set in the Sketch Palette."),
    "poly_edge": ("Edge Polygon", "polye", 3, "Click the two ends of an edge, then the side the polygon is on."),
    "ellipse": ("Ellipse", "ellipse", 3, "Click the centre, the end of the major axis, then the minor radius."),
    "slot_cc": ("Center to Center Slot", "slot", 3, "Click the two arc centres, then the width."),
    "slot_overall": ("Overall Slot", "slot", 3, "Click the two ends of the slot, then the width."),
    "slot_center": ("Center Point Slot", "slotc", 3, "Click the middle, one arc centre, then the width."),
    "slot_arc3": ("Three Point Arc Slot", "slota", 4, "Click the start, the end and a point on the slot's centre arc, then the width."),
    "slot_arcc": ("Center Point Arc Slot", "slota", 4, "Click the arc centre, the start, sweep to the end, then the width."),
    "spline": ("Fit Point Spline", "spline", None, "Click points the curve passes through; click the first point to close it, "
               "double-click / Enter to finish. Right-click a fit point to show its tangent handle."),
    "cspline": ("Control Point Spline", "cspline", None, "Click control points; double-click / Enter to finish."),
    "conic": ("Conic Curve", "conic", 3, "Click the start, the end, then the apex. Rho (0-1, 0.5 = parabola) is set in the Sketch Palette."),
    "point": ("Point", "skpoint", 1, "Click to place points."),
    "text": ("Text", "text", None, "Click where the text starts (select a curve first to run the text along it)."),
    "dim": ("Sketch Dimension", "skdim", None, "Click a line, circle or arc (or two points / lines), then click where the dimension goes - or type its value and press Enter."),
    "trim": ("Trim", "trim", None, "Click the part of a curve to remove (up to where other curves cross it)."),
    "extend": ("Extend", "extend", None, "Click near the end of a line or arc to lengthen it to the next curve."),
    "break": ("Break", "break", None, "Click a curve to split it where other curves cross it."),
    "fillet": ("Sketch Fillet", "skfillet", None, "Click two lines that meet (or their corner point)."),
    "chamfer": ("Sketch Chamfer", "skchamfer", None, "Click two lines that meet (or their corner point)."),
}
CON_TOOLS = (("hor", "Horizontal / Vertical", "c_hv"), ("coin", "Coincident", "c_coin"), ("tan", "Tangent", "c_tan"), ("eq", "Equal", "c_eq"),
             ("par", "Parallel", "c_par"), ("perp", "Perpendicular", "c_perp"), ("fix", "Fix / Unfix", "c_fix"), ("mid", "Midpoint", "c_mid"),
             ("conc", "Concentric", "c_conc"), ("col", "Collinear", "c_col"), ("sym", "Symmetry", "c_sym"), ("smooth", "Curvature (G2)", "c_smooth"))
CON_HINT = {"hor": "Click a line (or two points) to make it horizontal or vertical (whichever it is closer to).",
            "coin": "Click a point, then another point or a curve.", "tan": "Click two curves.", "eq": "Click two lines, circles or arcs.",
            "par": "Click two lines.", "perp": "Click two lines.", "fix": "Click points or curves to fix them in place (click again to unfix).",
            "mid": "Click a point, then a line or arc.", "conc": "Click two circles, arcs or ellipses.", "col": "Click two lines.",
            "sym": "Click two points / lines / circles, then the symmetry line.", "smooth": "Click two curves that share an end point (e.g. a spline and an arc)."}
GLYPH = {"hor": "—", "ver": "|", "par": "∥", "perp": "⊥", "col": "⋯", "conc": "◎", "coin": "•", "mid": "△", "tan": "◡", "smooth": "G2",
         "eq": "=", "sym": "[ ]", "fix": "🔒"}

def poly_dist(A, px, py):
    if len(A) < 2: return float(np.hypot(A[0, 0] - px, A[0, 1] - py)) if len(A) else 1e18
    a, b = A[:-1], A[1:]; d = b - a; L2 = (d*d).sum(1); L2 = np.where(L2 < 1e-12, 1e-12, L2)
    t = np.clip(((px - a[:, 0])*d[:, 0] + (py - a[:, 1])*d[:, 1])/L2, 0, 1)
    return float(np.hypot(a[:, 0] + d[:, 0]*t - px, a[:, 1] + d[:, 1]*t - py).min())

class SketchMixin:
    # ---------------- state ----------------
    def sk_init(s):
        s.skedit = None; s.sk_sel = set(); s.sk_hover = None; s.tp = []; s.tsel = []; s.sk_drag = None; s.sk_down = None
        s.sk_box = None; s.sk_cursor = None; s.sk_prev = None; s.chain = None; s.line_arc = None; s.arc_acc = 0.0; s.arc_last = None
        s.dim_items = []; s.dim_prev = None; s.glyphs = []; s.sk_hint = ""; s.params = {}; s.canv = []; s._sk_st = (None, None)
        s.sk_opts = dict(profiles=True, points=True, dims=True, cons=True, proj=True, slice=False, snap=True, grid=True,
                         construction=False, centerline=False, sides=6, rho=0.5, sk3d=False)
        s.tp3 = []; s.sk_cursor3 = None; s.type_buf = None; s.type_values = None; s.tp_typed = {}
        s.pending_tool = None; s.sk_trim_prev = None

    @property
    def esk(s):
        e = s.skedit
        return s.sketches[e["si"]] if e and e["si"] < len(s.sketches) else None

    def dim_env(s, upto=None):
        env = {}
        for nm, p in s.params.items():
            if "v" in p: env[nm] = p["v"]
        for sk in (s.sketches if upto is None else s.sketches[:upto]):
            for k in sk.geo.K:
                if k.get("n"): env[k["n"]] = k.get("v", 0.0)
        return env

    def next_dim_name(s, extra=None):
        used = {k["n"] for k in extra.K if k.get("n")} if extra is not None else set()
        for st_ in s.states:
            op = st_.get("op") or {}
            g = op.get("geo")
            if isinstance(g, dict):
                for k in g.get("con", []):
                    if k.get("n"): used.add(k["n"])
        for sk in s.sketches:
            for k in sk.geo.K:
                if k.get("n"): used.add(k["n"])
        used |= set(s.params)
        i = 1
        while f"d{i}" in used: i += 1
        return f"d{i}"

    # ---------------- the sketch timeline step ----------------
    def op_plane(s, op):
        if op.get("pref"): return s.ref_plane(op["pref"]), None
        if op.get("face"):
            bi, fid = op["face"]; b = s.bodies[bi]; n = b.face_normal(fid)
            if n is None: raise RuntimeError("the face this sketch sits on is no longer flat")
            return plane_for(n, b.face_point(fid)), bi
        return op["plane"], None

    def sk_refresh(s, geo, plane):
        """Bring projected geometry up to date, evaluate dimension expressions and solve."""
        s.sk_update_links(geo, plane); geo_eval_dims(geo, s.dim_env()); geo_solve(geo)

    def exec_sketch_op(s, op):
        t = op["t"]
        if t == "sk2":
            plane, parent = s.op_plane(op); geo = Geo.from_json(op.get("geo") or {})
            s.sk_refresh(geo, plane); sk = Sketch(plane, parent, geo); sk.name = op.get("name")
            s.sketches.append(sk); s.active = sk; return True
        if t == "skgeo":
            sk = s.sketches[op["sk"]]; geo = Geo.from_json(op["geo"]); s.sk_refresh(geo, sk.plane); sk.geo = geo; s.active = sk; return True
        if t == "skmod": s.exec_skmod(op); return True
        if t == "canvas":
            c = dict(op); c.pop("t", None); c.pop("icon", None)
            if c.get("face"): pl, _ = s.op_plane({"face": c["face"]}); c["plane_obj"] = pl
            elif c.get("pref"): c["plane_obj"] = s.ref_plane(c["pref"])
            else: c["plane_obj"] = c["plane"]
            s.canv = list(s.canv) + [c]; return True
        return False

    def sk_update_links(s, geo, plane):
        if not any(c.get("src") or c["k"] == "proj3" for c in geo.C): return
        pln = sk_pln(plane)
        for ci, c in enumerate(geo.C):
            if c["k"] == "proj3":
                try:
                    if c.get("srcc") is None: continue
                    qs = project_to_faces(crv_edges(geo, c["srcc"], plane), [s.ref_face(f) for f in c["faces"]], plane)
                    if qs: c["qs"] = qs; c.pop("broken", None)
                except Exception: c["broken"] = True
                continue
            src = c.get("src")
            if not src: continue
            try:
                if c["k"] == "inc3":
                    c["qs"] = [edge_local_pts(s.bodies[src[1]].eds[src[2]], plane)]
                elif src[0] == "vertex":
                    q = plane.l(V(*s.bodies[src[1]].verts[src[2]])); geo.P[c["p"][0]] = [q.x, q.y]
                elif src[0] == "edge":
                    g = edge_to_g2d(s.bodies[src[1]].eds[src[2]], pln)
                    if g is None: continue
                    tmp = Geo(); j = add_g2d(tmp, g, {})
                    if j is None: continue
                    nc = tmp.C[j]
                    if nc["k"] == c["k"] and len(nc["p"]) == len(c["p"]):
                        for a, b in zip(c["p"], nc["p"]): geo.P[a] = list(tmp.P[b])
                        for key in ("r", "r2", "kn", "mu", "w", "deg"):
                            if key in nc: c[key] = nc[key]
                    else:
                        ids = [geo.add_pt(*tmp.P[b]) for b in nc["p"]]; nc = dict(nc, p=ids, src=src, lock=True)
                        for key in ("cons", "cl"):
                            if c.get(key): nc[key] = c[key]
                        geo.C[ci] = nc
                c.pop("broken", None)
            except Exception:
                c["broken"] = True
        geo_compact(geo)

    # ---------------- entering / leaving the sketch environment ----------------
    def sk_begin_new(s, how, tool=None):
        """how: {"plane": Plane} / {"face": (bi, fid)} / {"pref": ref}"""
        if s.skedit: s.sk_finish()
        s.drop_pending(); s.tool = None
        op = {"t": "sk2", "geo": Geo().to_json(), "icon": "sketch"}; op.update(how)
        s.do(op); si = len(s.sketches) - 1
        s.skedit = dict(si=si, i=s.pos, new=True, back=None, hist=[s.sketches[si].geo], hpos=0)
        s.sk_enter(tool)

    def sk_edit(s, si):
        """Edit an existing sketch: roll the timeline back to it (later steps are recomputed on Finish)."""
        if s.skedit: s.sk_finish()
        n = -1; idx = None
        for i, stt in enumerate(s.states):
            op = stt.get("op") or {}
            if op.get("t") in ("sk2", "sketch"):
                n += 1
                if n == si: idx = i; break
        if idx is None: raise RuntimeError("can't find the step that made this sketch")
        op = s.states[idx]["op"]
        if op["t"] == "sk2":
            back = s.pos if s.pos > idx else None
            s.restore(idx); s.skedit = dict(si=si, i=idx, new=False, back=back, hist=[s.sketches[si].geo], hpos=0)
        else:                                                   # an old-style sketch: its edits become one new step at the end
            s.restore(len(s.states) - 1)
            s.do({"t": "skgeo", "sk": si, "geo": s.sketches[si].geo.to_json(), "icon": "sketch"})
            s.skedit = dict(si=si, i=s.pos, new=False, back=None, hist=[s.sketches[si].geo], hpos=0, legacy=True)
        s.sk_enter(None)

    def sk_enter(s, tool):
        sk = s.esk; sk.visible = True; s.active = sk
        s.sk_sel, s.tp, s.tsel, s.dim_items = set(), [], [], []; s.sk_set_tool(tool)
        s.look_at_plane(sk.plane.n); s.sk_mode.emit(True); s.changed.emit(); s.update()

    def sk_finish(s):
        e = s.skedit
        if not e: return
        s.skedit = None; s.tool = None; s.tp, s.tsel, s.dim_items, s.sk_sel = [], [], [], set(); s.sk_drag = None; s.sk_prev = None
        sk = s.sketches[e["si"]] if e["si"] < len(s.sketches) else None
        try:
            if sk is not None and e["new"] and sk.geo.empty() and e["i"] == len(s.states) - 1:
                del s.states[e["i"]:]; s.restore(e["i"] - 1)
            elif e.get("back") is not None and e["back"] > e["i"] and len(s.states) > e["i"] + 1:
                try: s.replay_edit(e["i"], s.states[e["i"]]["op"]); s.restore(e["back"])
                except Exception as ex:
                    s.restore(e["back"]); s.msg.emit(str(ex).split("\n")[0])
                    W.QMessageBox.warning(s, "Finish Sketch", f"Some later steps couldn't be rebuilt from the changed sketch:\n\n{ex}")
        finally:
            s.msg.emit("Sketch finished. Extrude (E) or Revolve (V) its profiles, or double-click it to edit it again.")
            s.sk_mode.emit(False); s.changed.emit(); s.update()

    # ---------------- edits and history inside the sketch ----------------
    def sk_commit(s, geo, solve=True, keep_sel=False, msg=None):
        sk = s.esk; e = s.skedit
        if sk is None: return False
        ok = True
        if solve:
            errs = geo_eval_dims(geo, s.dim_env()); ok, worst, _ = geo_solve(geo)
            if errs: s.msg.emit("; ".join(errs))
        geo.touch(); sk.geo = geo; sk._key = None
        del e["hist"][e["hpos"] + 1:]; e["hist"].append(geo); e["hpos"] = len(e["hist"]) - 1
        s.sk_store()
        if not keep_sel: s.sk_sel = set()
        if not ok: s.msg.emit("The sketch couldn't satisfy every constraint - check for conflicting dimensions.")
        elif msg: s.msg.emit(msg)
        s.sk_prev = None; s.changed.emit(); s.update(); return True

    def sk_store(s):
        e = s.skedit; i = e["i"]; sk = s.esk
        op = dict(s.states[i]["op"]); op["geo"] = sk.geo.to_json()
        if e["i"] != s.pos: s.pos = e["i"]
        s.states[i] = s.snapshot(s.states[i]["kind"], op)

    def sk_apply(s, fn, msg=None, solve=True):
        g = s.esk.geo.copy()
        try: res = fn(g)
        except Exception as ex:
            traceback.print_exc(); s.msg.emit(str(ex)); s.sk_flash(str(ex)); return None
        s.sk_commit(g, solve, msg=msg); return res

    def sk_undo(s):
        e = s.skedit
        if s.tp: s.tp.pop(); s.sk_prev = None; s.update(); return
        if s.dim_items: s.dim_items.pop(); s.update(); return
        if e["hpos"] > 0:
            e["hpos"] -= 1; s.esk.geo = e["hist"][e["hpos"]]; s.esk._key = None; s.sk_store(); s.sk_sel = set(); s.changed.emit(); s.update()
    def sk_redo(s):
        e = s.skedit
        if e["hpos"] < len(e["hist"]) - 1:
            e["hpos"] += 1; s.esk.geo = e["hist"][e["hpos"]]; s.esk._key = None; s.sk_store(); s.sk_sel = set(); s.changed.emit(); s.update()

    def sk_flash(s, text):
        s.sk_hint = text; s.msg.emit(text); C.QTimer.singleShot(3500, lambda: (setattr(s, "sk_hint", ""), s.update()))

    def sk_status(s):
        sk = s.esk
        if sk is None: return None
        key = (id(sk.geo), sk.geo.ver)
        if s._sk_st[0] != key:
            try: st_ = geo_status(sk.geo)
            except Exception: traceback.print_exc(); st_ = dict(points=set(), curves=set(), dof=-1, errors={})
            s._sk_st = (key, st_)
        return s._sk_st[1]

    # ---------------- tools ----------------
    def sk_set_tool(s, tool):
        s.tool = tool; s.tp, s.tsel, s.dim_items = [], [], []; s.sk_prev = s.dim_prev = None; s.chain = None; s.line_arc = None
        s.arc_acc = 0.0; s.arc_last = None; s.sk_trim_prev = None; s.tp3 = []; s.sk_cursor3 = None; s.type_buf = None; s.tp_typed = {}
        if tool is None: s.msg.emit("Select, drag or delete sketch geometry. Click a dimension to change it.")
        elif s.sk_opts["sk3d"] and tool in ("line", "spline"):
            s.msg.emit("3D Sketch: click model corners, edges or sketch points (or the sketch plane) - Enter / double-click to finish.")
        elif tool.startswith("con:"):
            t = tool[4:]
            if s.sk_sel and s.sk_con_from_sel(t): pass
            s.msg.emit(CON_HINT.get(t, ""))
        elif tool == "text" and any(i[0] == "c" for i in s.sk_sel):
            ci = next(i[1] for i in s.sk_sel if i[0] == "c"); s.text_dialog.emit(dict(path=ci)); s.tool = None
        elif tool in ("fillet", "chamfer"):
            sel = [i[1] for i in s.sk_sel if i[0] == "c"]
            s.msg.emit(SK_TOOLS[tool][3])
            if len(sel) == 2: s.tsel = sel; s.corner_dialog.emit(tool, sel)
        else: s.msg.emit(SK_TOOLS.get(tool, ("", "", 0, ""))[3])
        s.setCursor(C.Qt.CrossCursor if tool else C.Qt.ArrowCursor); s.update()

    def sk_proj(s, sk, P):
        P = np.asarray(P, float); P = P.reshape(-1, P.shape[-1] if P.ndim > 1 else 2); pl = sk.plane
        Wd = np.array(pl.o.t()) + P[:, :1]*np.array(pl.u.t()) + P[:, 1:2]*np.array(pl.v.t())
        if P.shape[1] > 2: Wd = Wd + P[:, 2:3]*np.array(pl.n.t())
        return s.project(Wd)

    def sk_px(s, sk, x, y):
        xy, ok = s.sk_proj(sk, [[x, y]]); return xy[0], bool(ok[0])

    def sk_world_len(s, px):
        """Sketch-plane length of about `px` screen pixels near the camera target."""
        vp = s.vp; h = max(vp[3], 1)
        return s.dist*2*math.tan(math.radians(22.5))*px*s.devicePixelRatioF()/h

    def sk_snap(s, p, ref=None, allow_on=True, skip=()):
        """Where a click lands and how it attaches: (x, y, snap) + s.sk_cursor for drawing hints."""
        sk = s.esk; geo = sk.geo; g = s.ground(p, False, sk.plane)
        if g is None: return None
        free = bool(W.QApplication.keyboardModifiers() & C.Qt.ControlModifier)
        k = s.devicePixelRatioF(); px, py = s.mouse_dev(p); hint = None
        if not free:
            if geo.P:
                P = np.array(geo.P, float); xy, ok = s.sk_proj(sk, P); d = np.hypot(xy[:, 0] - px, xy[:, 1] - py)/k; d[~ok] = np.inf
                for i in skip:
                    if i < len(d): d[i] = np.inf
                if not s.sk_opts["proj"]:
                    for c in geo.C:
                        if c.get("src"):
                            for i in c["p"]: d[i] = np.inf
                i = int(np.argmin(d))
                if d[i] < 9: s.sk_cursor = (geo.P[i][0], geo.P[i][1], "pt"); return (geo.P[i][0], geo.P[i][1], ("pt", i))
            for ci, c in enumerate(geo.C):
                if c["k"] != "line": continue
                a, b = c["p"]; m = ((geo.P[a][0] + geo.P[b][0])/2, (geo.P[a][1] + geo.P[b][1])/2)
                q, ok = s.sk_px(sk, *m)
                if ok and math.hypot(q[0] - px, q[1] - py)/k < 8: s.sk_cursor = (m[0], m[1], "mid"); return (m[0], m[1], ("mid", ci))
            for bi, b in enumerate(s.bodies):                                   # sketch on the fly: body corners
                if not b.visible or not len(b.verts) or len(b.verts) > 4000: continue
                xy, ok = s.project(np.asarray(b.verts, float)); d = np.hypot(xy[:, 0] - px, xy[:, 1] - py)/k; d[~ok] = np.inf
                j = int(np.argmin(d))
                if d[j] < 8:
                    q = sk.plane.l(V(*b.verts[j])); s.sk_cursor = (q.x, q.y, "vtx"); return (q.x, q.y, ("vtx", bi, j))
            e = s.edge_near(p) if s.bodies else None
            if e:
                bi, ei = e; P3 = np.asarray(s.bodies[bi].edges[ei], float); xy_, ok = s.project(P3)
                if ok.all() and len(P3) > 1:
                    a, b_ = xy_[:-1], xy_[1:]; d_ = b_ - a; L2 = np.maximum((d_*d_).sum(1), 1e-12)
                    t = np.clip(((px - a[:, 0])*d_[:, 0] + (py - a[:, 1])*d_[:, 1])/L2, 0, 1)
                    dd = np.hypot(a[:, 0] + d_[:, 0]*t - px, a[:, 1] + d_[:, 1]*t - py); j = int(np.argmin(dd))
                    wq = P3[j] + (P3[j + 1] - P3[j])*t[j]; q = sk.plane.l(V(*wq))
                    s.sk_cursor = (q.x, q.y, "edge"); return (q.x, q.y, ("bedge", bi, ei))
            if allow_on:
                best = None
                for ci, c in enumerate(geo.C):
                    if c["k"] in ("point", "text") or is3d(c): continue
                    for A in geo_curve_pts(geo, ci):
                        xy, ok = s.sk_proj(sk, A)
                        if not ok.all(): continue
                        dd = poly_dist(xy, px, py)/k
                        if dd < 6 and (best is None or dd < best[0]): best = (dd, ci)
                if best:
                    gs = geo_g2d(geo, best[1])
                    if gs:
                        u = proj_param(gs[0], g.x, g.y); x, y = g_val(gs[0], u)
                        s.sk_cursor = (x, y, "on"); return (x, y, ("on", best[1]))
        x, y = g.x, g.y
        if s.sk_opts["snap"] and not free:
            st_ = s.snap_step()/5; x, y = round(x/st_)*st_, round(y/st_)*st_
        if ref is not None and not free:
            q0, ok0 = s.sk_px(sk, ref[0], ref[1]); q1, ok1 = s.sk_px(sk, x, y)
            if ok0 and ok1:
                if abs(q1[1] - q0[1])/k < 7 and abs(q1[0] - q0[0]) > abs(q1[1] - q0[1]): y = ref[1]; hint = "hor"
                elif abs(q1[0] - q0[0])/k < 7: x = ref[0]; hint = "ver"
        s.sk_cursor = (x, y, hint); return (x, y, None)

    def sk_prep(s, g, sp):
        """Turn a body-corner / body-edge snap into projected (linked) sketch geometry."""
        if sp is None or len(sp) < 3 or not sp[2] or sp[2][0] not in ("vtx", "bedge"): return sp
        if sp[2][0] == "vtx":
            i = g.add_pt(sp[0], sp[1]); g.add("point", [i], src=["vertex", sp[2][1], sp[2][2]], lock=True)
            return (sp[0], sp[1], ("pt", i))
        _, bi, ei = sp[2]; src = ["edge", bi, ei]
        ci = next((j for j, c in enumerate(g.C) if c.get("src") == src and not is3d(c)), None)
        if ci is None:
            try:
                gg = edge_to_g2d(s.bodies[bi].eds[ei], sk_pln(s.esk.plane))
                ci = add_g2d(g, gg, {}, src=src, lock=True) if gg is not None else None
            except Exception: ci = None
        if ci is None: return (sp[0], sp[1], None)
        gs = crv_g2d(g.C[ci], g.PT)
        x, y = g_val(gs[0], proj_param(gs[0], sp[0], sp[1])) if gs else (sp[0], sp[1])
        i = g.add_pt(x, y); g.con("coin", P_(i), C_(ci)); return (x, y, ("pt", i))

    def sk_mark_new(s, g, n0):
        o = s.sk_opts
        for c in g.C[n0:]:
            if o["construction"]: c["cons"] = True
            elif o["centerline"] and c["k"] == "line": c["cl"] = True

    def sk_build(s, T, g, sp):
        o = s.sk_opts; n = len(sp)
        if T == "rect2": return build_rect2(g, sp[0], sp[1])
        if T == "rect3": return build_rect3(g, sp[0], sp[1], sp[2])
        if T == "rectc": return build_rectc(g, sp[0], sp[1])
        if T == "circle":
            (cx, cy), (x, y) = xy(sp[0]), xy(sp[1]); ci = build_circle(g, sp[0], math.hypot(x - cx, y - cy))
            if ci is not None: tie_on(g, sp[1], ci)
            return ci
        if T == "circle2p": return build_circle_2p(g, sp[0], sp[1])
        if T == "circle3p": return build_circle_3p(g, sp[0], sp[1], sp[2])
        if T == "arc3p": return build_arc_3p(g, sp[0], sp[1], sp[2])
        if T == "arcc": return build_arc_center(g, sp[0], sp[1], sp[2], s.arc_acc >= 0)
        if T in ("poly_insc", "poly_circ"): return build_polygon(g, sp[0], sp[1], o["sides"], "insc" if T == "poly_insc" else "circ")
        if T == "poly_edge": return build_polygon_edge(g, sp[0], sp[1], o["sides"], xy(sp[2]))
        if T == "ellipse": return build_ellipse(g, sp[0], sp[1], sp[2])
        if T in ("slot_cc", "slot_overall", "slot_center"):
            (ax, ay), (bx, by), (wx, wy) = xy(sp[0]), xy(sp[1]), xy(sp[2]); L = math.hypot(bx - ax, by - ay) or 1e-12
            w = 2*abs(((bx - ax)*(wy - ay) - (by - ay)*(wx - ax))/L)
            return {"slot_cc": build_slot_cc, "slot_overall": build_slot_overall, "slot_center": build_slot_center}[T](g, sp[0], sp[1], w)
        if T == "slot_arc3":
            cc = circumcircle(xy(sp[0]), xy(sp[1]), xy(sp[2]))
            if not cc: return None
            w = 2*abs(math.hypot(sp[3][0] - cc[0], sp[3][1] - cc[1]) - cc[2]); return build_slot_arc3(g, sp[0], sp[1], sp[2], w)
        if T == "slot_arcc":
            (cx, cy), (ax, ay) = xy(sp[0]), xy(sp[1]); R = math.hypot(ax - cx, ay - cy)
            w = 2*abs(math.hypot(sp[3][0] - cx, sp[3][1] - cy) - R); return build_slot_arcc(g, sp[0], sp[1], sp[2], w, s.arc_acc >= 0)
        if T == "conic": return build_conic(g, sp[0], sp[1], sp[2], o["rho"])
        if T == "point": return build_point(g, sp[0])
        return None

    def sk_preview(s):
        """Rubber-band preview of the active tool: a copy of the geometry with the shape built up to the cursor."""
        sk = s.esk; T = s.tool; cur = s.sk_cursor
        if sk is None or T is None or cur is None: s.sk_prev = None; return
        spec = (cur[0], cur[1], None); n = SK_TOOLS.get(T, (0, 0, None))[2]
        g = sk.geo.copy(); n0 = len(g.C); lines = []
        try:
            if T == "line" and s.tp:
                if s.line_arc and s.chain and s.chain.get("prev") is not None:
                    build_arc_tangent(g, s.chain["prev"], s.chain["end"], spec)
                else: lines = [xy(s.tp[-1]), (cur[0], cur[1])]
            elif T == "arct" and s.tsel:
                build_arc_tangent(g, s.tsel[0], s.tsel[1], spec)
            elif T in ("spline", "cspline") and s.tp:
                sp = [(q[0], q[1], None) for q in s.tp] + [spec]
                if T == "spline": build_spline(g, sp)
                else: build_cspline(g, sp); lines = [xy(q) for q in sp]
            elif T == "circle2t" and len(s.tsel) == 2:
                build_circle_2t(g, s.tsel[0], s.tsel[1], cur[0], cur[1])
            elif n and s.tp:
                sp = [(q[0], q[1], None) for q in s.tp] + [spec]
                if len(sp) >= n or (T in ("rect3", "poly_edge", "ellipse", "slot_cc", "slot_overall", "slot_center", "arc3p", "arcc",
                                         "slot_arc3", "slot_arcc", "conic") and len(sp) == n - 1 and False):
                    s.sk_build(T, g, sp[:n])
                elif T in ("arc3p",) and len(sp) == 2: lines = [xy(sp[0]), xy(sp[1])]
                elif T in ("slot_cc", "slot_overall", "slot_center") and len(sp) == 2: s.sk_build(T, g, sp + [(sp[1][0], sp[1][1], None)]) ; lines = [xy(sp[0]), xy(sp[1])]
                elif T in ("slot_arc3",) and len(sp) == 3: build_arc_3p(g, sp[0], sp[1], sp[2])
                elif T in ("slot_arcc", "arcc") and len(sp) == 3: build_arc_center(g, sp[0], sp[1], sp[2], s.arc_acc >= 0)
                elif T == "ellipse" and len(sp) == 2:
                    (cx, cy), (mx, my) = xy(sp[0]), xy(sp[1]); r = math.hypot(mx - cx, my - cy)
                    if r > 1e-9: build_ellipse(g, sp[0], sp[1], (cx - (my - cy)/2, cy + (mx - cx)/2, None))
                else: lines = [xy(q) for q in sp]
        except Exception:
            pass
        s.sk_prev = dict(geo=g, n0=n0, lines=lines)

    # ---------------- picking ----------------
    def sk_pick(s, p, kinds=("d", "k", "p", "c")):
        sk = s.esk
        if sk is None: return None
        geo = sk.geo; k = s.devicePixelRatioF(); px, py = s.mouse_dev(p)
        if "d" in kinds:
            for lab in reversed(s.labels):
                if lab["ref"][0] == "dim" and lab["rect"].adjusted(-2, -2, 2, 2).contains(p): return ("k", lab["ref"][1])
        if "k" in kinds:
            for gl in reversed(s.glyphs):
                if gl["rect"].adjusted(-1, -1, 1, 1).contains(p): return ("k", gl["ki"])
        if "p" in kinds and geo.P:
            P = np.array(geo.P, float); xy_, ok = s.sk_proj(sk, P); d = np.hypot(xy_[:, 0] - px, xy_[:, 1] - py)/k; d[~ok] = np.inf
            if not s.sk_opts["proj"]:
                for c in geo.C:
                    if c.get("src"):
                        for i in c["p"]: d[i] = np.inf
            used = set()
            for c in geo.C: used |= set(crv_points(c))
            for i in range(len(d)):
                if i not in used: d[i] = np.inf
            i = int(np.argmin(d))
            if d[i] < 7: return ("p", i)
        if "c" in kinds:
            best = None
            for ci, c in enumerate(geo.C):
                if c["k"] == "point" or (c.get("src") and not s.sk_opts["proj"]): continue
                for A in geo_polys(geo, ci):
                    xy_, ok = s.sk_proj(sk, A)
                    if not ok.all(): continue
                    dd = poly_dist(xy_, px, py)/k
                    if c["k"] == "text" and dd > 6:
                        lo, hi = xy_.min(0), xy_.max(0)
                        if lo[0] <= px <= hi[0] and lo[1] <= py <= hi[1]: dd = 5.9
                    if dd < 6 and (best is None or dd < best[0]): best = (dd, ci)
            if best: return ("c", best[1])
        return None

    def sk_local_mouse(s, p):
        g = s.ground(p, False, s.esk.plane); return (g.x, g.y) if g is not None else None

    # ---------------- mouse ----------------
    def sk_press(s, e):
        p = e.position(); s.sk_down = dict(pos=p, item=None, started=False)
        if s.tool is None:
            it = s.sk_pick(p); s.sk_down["item"] = it
        elif s.tool == "line" and s.tp and s.chain and s.chain.get("prev") is not None:
            sn = s.sk_snap(p)
            if sn and sn[2] == ("pt", s.chain["end"]): s.line_arc = True

    def sk_mouse_move(s, e):
        p = e.position(); b = e.buttons(); sk = s.esk
        if b & C.Qt.LeftButton and s.sk_down:
            d0 = s.sk_down["pos"]
            if not s.sk_down["started"] and abs(p.x() - d0.x()) + abs(p.y() - d0.y()) > 4:
                s.sk_down["started"] = True
                if s.tool is None: s.sk_drag_start(s.sk_down["item"], d0)
            if s.sk_drag: s.sk_drag_move(p); return True
            if s.sk_box is not None: s.sk_box[1] = p; s.update(); return True
            if s.line_arc: s.sk_snap(p); s.sk_preview(); s.update(); return True
        return False

    def sk_hover_update(s, p):
        sk = s.esk; T = s.tool
        if T is None:
            s.sk_hover = s.sk_pick(p); s.sk_cursor = None
            s.setCursor(C.Qt.PointingHandCursor if s.sk_hover else C.Qt.ArrowCursor)
            if s.sk_hover and s.sk_hover[0] == "k" and s.sk_hover[1] < len(sk.geo.K):
                k = sk.geo.K[s.sk_hover[1]]
                if k["t"] in DIM_TYPES: s.setToolTip(f"{k.get('n', '')} = {k.get('x', '')}" + ("  (driven)" if k.get("drv") else "") + "\nClick to edit · drag to move")
                else: s.setToolTip(CON_NAMES.get(k["t"], k["t"]) + (" (suppressed)" if k.get("off") else "") + "\nClick to select · Delete removes · right-click for more")
            else: s.setToolTip("")
            s.update(); return
        s.setToolTip("")
        if T.startswith("con:") or T in ("trim", "extend", "break", "fillet", "chamfer", "circle3t") or (T == "circle2t" and len(s.tsel) < 2) \
                or (T == "arct" and not s.tsel) or (T == "dim"):
            kinds = ("p", "c") if T in ("dim", "fillet", "chamfer", "arct") or T.startswith("con:") else ("c",)
            s.sk_hover = s.sk_pick(p, kinds)
            if T == "dim": s.sk_dim_preview(p)
            if T == "trim" and s.sk_hover and s.sk_hover[0] == "c":
                g = s.sk_local_mouse(p); s.sk_trim_prev = (s.sk_hover[1], g) if g else None
            else: s.sk_trim_prev = None
            s.sk_cursor = None; s.update(); return
        if s.sk_opts["sk3d"] and T in ("line", "spline"):
            s.sk_snap3(p); s.sk_cursor = None; s.update(); return
        s.sk_cursor3 = None
        ref = None
        if T == "line" and s.tp: ref = xy(s.tp[-1])
        elif s.tp and T in ("rect3", "poly_edge", "slot_cc", "slot_overall", "slot_center", "ellipse", "spline", "cspline", "arc3p") and len(s.tp) < 3: ref = xy(s.tp[-1])
        s.sk_snap(p, ref=ref)
        if T == "line" and s.tp and s.chain and s.chain.get("prev") is not None and s.sk_cursor and s.sk_cursor[2] is None:
            s.sk_infer_perp()
        if T in ("arcc", "slot_arcc") and len(s.tp) == 2 and s.sk_cursor:
            (cx, cy) = xy(s.tp[0]); a = math.atan2(s.sk_cursor[1] - cy, s.sk_cursor[0] - cx)
            if s.arc_last is None: s.arc_last = math.atan2(s.tp[1][1] - cy, s.tp[1][0] - cx)
            s.arc_acc += (a - s.arc_last + math.pi) % math.tau - math.pi; s.arc_last = a
        s.sk_preview(); s.update()

    def sk_snap3(s, p):
        """3D sketch snapping: sketch / model points and edges in space, otherwise the sketch plane."""
        sk = s.esk; geo = sk.geo; k = s.devicePixelRatioF(); px, py = s.mouse_dev(p); cand = []
        for c in geo.C:
            if c["k"] in ("l3", "s3"): cand += [list(q) for q in c["q"]]
            elif c["k"] in ("inc3", "proj3"): cand += [list(A[0]) for A in c.get("qs") or [] if len(A)] + [list(A[-1]) for A in c.get("qs") or [] if len(A)]
        cand += [[x, y, 0.0] for x, y in geo.P]
        for b in s.bodies:
            if b.visible and len(b.verts) < 4000:
                for v_ in b.verts: q = sk.plane.l(V(*v_)); cand.append([q.x, q.y, q.z])
        if cand and not (W.QApplication.keyboardModifiers() & C.Qt.ControlModifier):
            Q = np.array(cand, float); xy_, ok = s.sk_proj(sk, Q); d = np.hypot(xy_[:, 0] - px, xy_[:, 1] - py)/k; d[~ok] = np.inf
            i = int(np.argmin(d))
            if d[i] < 9: s.sk_cursor3 = (*Q[i], "pt"); return tuple(Q[i])
            e = s.edge_near(p) if s.bodies else None
            if e:
                P3 = np.asarray(s.bodies[e[0]].edges[e[1]], float); xy_, ok = s.project(P3)
                if ok.all() and len(P3) > 1:
                    a, b_ = xy_[:-1], xy_[1:]; d_ = b_ - a; L2 = np.maximum((d_*d_).sum(1), 1e-12)
                    t = np.clip(((px - a[:, 0])*d_[:, 0] + (py - a[:, 1])*d_[:, 1])/L2, 0, 1)
                    dd = np.hypot(a[:, 0] + d_[:, 0]*t - px, a[:, 1] + d_[:, 1]*t - py); j = int(np.argmin(dd))
                    q = sk.plane.l(V(*(P3[j] + (P3[j + 1] - P3[j])*t[j]))); s.sk_cursor3 = (q.x, q.y, q.z, "edge"); return (q.x, q.y, q.z)
        g = s.ground(p, s.sk_opts["snap"], sk.plane)
        if g is None: return None
        s.sk_cursor3 = (g.x, g.y, 0.0, None); return (g.x, g.y, 0.0)

    def sk_click3(s, p):
        q = s.sk_snap3(p)
        if q is None: return
        if s.tool == "line":
            if s.tp3 and math.dist(s.tp3[-1], q) > 1e-9:
                a = list(s.tp3[-1]); s.sk_apply(lambda g: g.add("l3", [], q=[a, list(q)], **({"cons": True} if s.sk_opts["construction"] else {})), solve=False)
            s.tp3 = [list(q)]
        else: s.tp3.append(list(q))
        s.update()

    def sk_finish3(s):
        pts = s.tp3; s.tp3 = []
        if s.tool == "spline" and len(pts) >= 2:
            s.sk_apply(lambda g: g.add("s3", [], q=[list(x) for x in pts]), solve=False)
        s.update()

    def sk_infer_perp(s):
        """While drawing lines: lock to perpendicular with the previous line when close."""
        g = s.esk.geo; c = g.C[s.chain["prev"]]
        if c["k"] != "line": return
        a, b = c["p"]; dx, dy = g.P[b][0] - g.P[a][0], g.P[b][1] - g.P[a][1]; L = math.hypot(dx, dy)
        x0, y0 = xy(s.tp[-1]); cx, cy = s.sk_cursor[:2]; vx, vy = cx - x0, cy - y0; M = math.hypot(vx, vy)
        if L < 1e-9 or M < 1e-9: return
        if abs((dx*vx + dy*vy)/(L*M)) < math.sin(math.radians(2.0)):
            nx, ny = -dy/L, dx/L; t = vx*nx + vy*ny; s.sk_cursor = (x0 + nx*t, y0 + ny*t, "perp")

    def sk_release(s, e):
        p = e.position(); down = s.sk_down; s.sk_down = None
        if s.sk_drag: s.sk_drag_end(); return
        if s.sk_box is not None: s.sk_box_select(); return
        if s.line_arc:
            s.line_arc = None
            sn = s.sk_cursor; spec = (sn[0], sn[1], None) if sn else None
            sp = s.sk_snap(p)
            if sp and s.chain and s.chain.get("prev") is not None:
                prev, end = s.chain["prev"], s.chain["end"]
                def f(g):
                    ai = build_arc_tangent(g, prev, end, s.sk_prep(g, sp)); n0 = len(g.C) - 1; s.sk_mark_new(g, n0); return ai
                ai = s.sk_apply(f)
                if ai is not None:
                    g = s.esk.geo; c = g.C[ai]; e_ = c["p"][2] if c["p"][1] == end else c["p"][1]
                    s.chain = dict(start=s.chain["start"], end=e_, prev=ai); s.tp = [(g.P[e_][0], g.P[e_][1], ("pt", e_))]
            return
        if down and down["started"]: return
        if s.tool is None: s.sk_click_select(p, e.modifiers()); return
        s.sk_tool_click(p)

    def sk_double(s, e):
        p = e.position(); T = s.tool
        if s.sk_opts["sk3d"] and T in ("line", "spline"): s.sk_finish3(); return
        if T in ("line",): s.tp = []; s.chain = None; s.sk_prev = None; s.update(); return
        if T in ("spline", "cspline"):
            if len(s.tp) >= 2:
                pts = list(s.tp)
                while len(pts) >= 2 and abs(pts[-1][0] - pts[-2][0]) < 1e-9 and abs(pts[-1][1] - pts[-2][1]) < 1e-9: pts.pop()
                s.sk_finish_spline(pts)
            return
        if T is not None: return
        it = s.sk_pick(p)
        if not it: return
        g = s.esk.geo
        if it[0] == "k" and g.K[it[1]]["t"] in DIM_TYPES: s.dim_edit.emit(it[1]); return
        if it[0] == "c":
            c = g.C[it[1]]
            if c["k"] == "text": s.text_dialog.emit(dict(edit=it[1])); return
            if c["k"] == "conic":
                v, ok = W.QInputDialog.getDouble(s, "Conic", "Rho (0-1, 0.5 = parabola)", c.get("rho", 0.5), 0.01, 0.99, 3)
                if ok:
                    def f(g2): g2.C[it[1]]["rho"] = v
                    s.sk_apply(f)
                return
            s.sk_sel = {("c", ci) for ci in chain_from(g, it[1])}; s.update()

    def sk_click_select(s, p, mods):
        si = s.skedit["si"]
        for lab in reversed(s.labels):
            r = lab["ref"]
            if r[0] == "auto" and r[1] == si and lab["rect"].adjusted(-2, -2, 2, 2).contains(p):
                s.auto_dim_place(r[2], r[3]); return
        it = s.sk_pick(p); add = bool(mods & (C.Qt.ControlModifier | C.Qt.ShiftModifier))
        if it and it[0] == "k" and it[1] < len(s.esk.geo.K) and s.esk.geo.K[it[1]]["t"] in DIM_TYPES and not add:
            s.dim_edit.emit(it[1]); return
        if not it:
            if not add: s.sk_sel = set()
            r = s.region_under(p, s.esk)
            if r and s.sk_opts["profiles"] and not add:
                sk, i = r; sk.sel ^= {i}
        elif add: s.sk_sel ^= {it}
        else: s.sk_sel = {it}
        n = len(s.sk_sel)
        if n: s.msg.emit(f"{n} selected - Delete removes, X toggles construction, drag to move. Constraint tools apply to the selection.")
        s.update()

    def sk_box_select(s):
        a, b = s.sk_box; s.sk_box = None; sk = s.esk; geo = sk.geo; k = s.devicePixelRatioF()
        x0, x1 = sorted((a.x(), b.x())); y0, y1 = sorted((a.y(), b.y())); crossing = b.x() < a.x()
        def inside(xy_):
            lx = xy_[:, 0]/k; ly = (s.vp[3] - xy_[:, 1])/k
            m = (lx >= x0) & (lx <= x1) & (ly >= y0) & (ly <= y1); return m
        sel = set() if not (W.QApplication.keyboardModifiers() & (C.Qt.ControlModifier | C.Qt.ShiftModifier)) else set(s.sk_sel)
        for ci, c in enumerate(geo.C):
            pl = geo_curve_pts(geo, ci) if c["k"] != "point" else [np.array([geo.P[c["p"][0]]])]
            ms = [inside(s.sk_proj(sk, A)[0]) for A in pl if len(A)]
            if not ms: continue
            if (crossing and any(m.any() for m in ms)) or (not crossing and all(m.all() for m in ms)): sel.add(("c", ci))
        if geo.P:
            m = inside(s.sk_proj(sk, np.array(geo.P))[0])
            for i in np.nonzero(m)[0]:
                if not any(int(i) in c["p"] for c in geo.C if ("c", geo.C.index(c)) in sel): sel.add(("p", int(i)))
        s.sk_sel = sel; s.msg.emit(f"{len(sel)} selected"); s.update()

    # ---------------- dragging ----------------
    def sk_drag_start(s, it, p0):
        sk = s.esk; geo = sk.geo; g0 = s.sk_local_mouse(p0)
        if g0 is None: return
        if it is None: s.sk_box = [p0, p0]; return
        work = geo.copy(); pts = set(); mode = "pts"
        if it[0] == "k":
            k = geo.K[it[1]]
            if k["t"] in DIM_TYPES:
                pos = k.get("pos") or s.dim_default_pos(geo, k)
                s.sk_drag = dict(mode="label", ki=it[1], work=work, g0=g0, pos0=list(pos)); sk.geo = work; return
            return
        items = s.sk_sel if it in s.sk_sel else {it}
        for i in items:
            if i[0] == "p": pts.add(i[1])
            elif i[0] == "c": pts |= set(crv_points(geo.C[i[1]]))
        if it[0] == "c" and len(items) == 1 and geo.C[it[1]]["k"] == "circle":
            mode = "radius"
        locked = set()
        for c in geo.C:
            if c.get("src") or c.get("lock"): locked |= set(c["p"])
        pts -= locked
        if not pts and mode != "radius": return
        s.sk_drag = dict(mode=mode, ci=it[1] if it[0] == "c" else None, work=work, g0=g0, pts={i: tuple(geo.P[i]) for i in pts})
        sk.geo = work; sk._key = None

    def sk_drag_move(s, p):
        d = s.sk_drag; sk = s.esk; g = s.sk_local_mouse(p)
        if g is None: return
        dx, dy = g[0] - d["g0"][0], g[1] - d["g0"][1]
        if not (W.QApplication.keyboardModifiers() & C.Qt.ControlModifier) and s.sk_opts["snap"] and d["mode"] != "label":
            st_ = s.snap_step()/5; dx, dy = round(dx/st_)*st_, round(dy/st_)*st_
        w = d["work"]
        if d["mode"] == "label":
            k = w.K[d["ki"]]; k["pos"] = [d["pos0"][0] + dx, d["pos0"][1] + dy]; w.touch()
        elif d["mode"] == "radius":
            c = w.C[d["ci"]]; cx, cy = w.P[c["p"][0]]; old = c["r"]; c["r"] = max(math.hypot(g[0] - cx, g[1] - cy), 1e-6)
            ok, _, _ = geo_solve(w)
            if not ok: c["r"] = old; geo_solve(w)
        else:
            tg = {i: (x + dx, y + dy) for i, (x, y) in d["pts"].items()}
            if len(tg) == 1 and not (W.QApplication.keyboardModifiers() & C.Qt.ControlModifier):
                i = next(iter(tg)); sn = s.sk_snap(p, allow_on=False, skip={i})
                if sn and sn[2] is None: tg[i] = (sn[0], sn[1])
            geo_solve(w, drag=tg)
        sk.geo = w; sk._key = None; s.update()

    def sk_drag_end(s):
        d = s.sk_drag; s.sk_drag = None; sk = s.esk
        geo0 = s.skedit["hist"][s.skedit["hpos"]]; sk.geo = geo0; sk._key = None; w_ = d["work"]
        if all(abs(a[0] - b[0]) < 1e-12 and abs(a[1] - b[1]) < 1e-12 for a, b in zip(w_.P, geo0.P)) and w_.C == geo0.C and w_.K == geo0.K:
            s.update(); return
        s.sk_commit(w_, solve=d["mode"] != "label", keep_sel=True)

    # ---------------- tool clicks ----------------
    def sk_tool_click(s, p):
        T = s.tool; sk = s.esk; geo = sk.geo; s.type_buf = None
        if T.startswith("con:"):
            t = T[4:]
            it = s.sk_pick(p, ("c",) if t in ("par", "perp", "col", "eq", "tan", "smooth") or (t == "sym" and len(s.tsel) == 2) else ("p", "c"))
            if it: s.sk_con_pick(t, it)
            return
        if T == "dim": s.sk_dim_click(p); return
        if T in ("trim", "extend", "break"):
            it = s.sk_pick(p, ("c",)); g = s.sk_local_mouse(p)
            if not it or not g: return
            fn = {"trim": geo_trim, "extend": geo_extend, "break": geo_break}[T]
            s.sk_apply(lambda g2: fn(g2, it[1], g[0], g[1])); s.sk_trim_prev = None; return
        if T in ("fillet", "chamfer"):
            it = s.sk_pick(p, ("p", "c"))
            if not it: return
            if it[0] == "p":
                ls = [ci for ci in curves_at_point(geo, it[1]) if geo.C[ci]["k"] == "line"]
                if len(ls) == 2: s.tsel = ls
                else: s.sk_flash("Pick a corner where exactly two lines meet."); return
            else:
                if geo.C[it[1]]["k"] != "line": s.sk_flash("Sketch fillet / chamfer works on lines."); return
                if it[1] not in s.tsel: s.tsel.append(it[1])
            if len(s.tsel) == 2: s.corner_dialog.emit(T, list(s.tsel)); s.tsel = []
            s.update(); return
        if T == "circle3t":
            it = s.sk_pick(p, ("c",))
            if it and it[1] not in s.tsel and (geo.C[it[1]]["k"] == "line" or is_round(geo.C[it[1]])): s.tsel.append(it[1])
            if len(s.tsel) == 3:
                cs = list(s.tsel); s.tsel = []; s.sk_apply(lambda g: build_circle_3t(g, cs))
            s.update(); return
        if T == "circle2t":
            if len(s.tsel) < 2:
                it = s.sk_pick(p, ("c",))
                if it and geo.C[it[1]]["k"] == "line" and it[1] not in s.tsel: s.tsel.append(it[1])
                s.update(); return
            g = s.sk_local_mouse(p); ls = list(s.tsel); s.tsel = []
            if g: s.sk_apply(lambda g2: build_circle_2t(g2, ls[0], ls[1], g[0], g[1]))
            return
        if T == "arct":
            if not s.tsel:
                it = s.sk_pick(p, ("p",))
                if it:
                    cs = [ci for ci in curves_at_point(geo, it[1]) if it[1] in crv_ends(geo.C[ci])]
                    if cs: s.tsel = [cs[-1], it[1]]
                    else: s.sk_flash("Click the end of a line, arc or spline.")
                return
            sp = s.sk_snap(p); ci, pi = s.tsel; s.tsel = []
            if sp:
                def f(g):
                    n0 = len(g.C); r = build_arc_tangent(g, ci, pi, s.sk_prep(g, sp)); s.sk_mark_new(g, n0); return r
                s.sk_apply(f)
            return
        if T == "text":
            sp = s.sk_snap(p)
            if sp: s.text_dialog.emit(dict(at=sp))
            return
        if s.sk_opts["sk3d"] and T in ("line", "spline"): s.sk_click3(p); return
        ref = xy(s.tp[-1]) if s.tp else None
        sp = s.sk_snap(p, ref=ref)
        if sp is None: return
        if T == "line" and s.tp and s.chain and s.chain.get("prev") is not None and sp[2] is None and s.sk_cursor and s.sk_cursor[2] is None:
            s.sk_infer_perp()
        if s.sk_cursor and s.sk_cursor[2] in ("perp", "hor", "ver") and sp[2] is None: sp = (s.sk_cursor[0], s.sk_cursor[1], None)
        if T == "line":
            if not s.tp:
                s.tp = [sp]; s.chain = dict(start=None, end=None, prev=None); return
            a = s.tp[-1]; perp = s.sk_cursor and s.sk_cursor[2] == "perp"; prev = s.chain.get("prev")
            if math.hypot(sp[0] - a[0], sp[1] - a[1]) < 1e-9: return
            def f(g):
                n0 = len(g.C); ci = build_line(g, s.sk_prep(g, a), s.sk_prep(g, sp))
                if ci is not None and perp and prev is not None and not any(k["t"] in ("hor", "ver") and k["e"] == [["c", ci]] for k in g.K):
                    g.con("perp", C_(prev), C_(ci))
                s.sk_mark_new(g, n0); return ci
            ci = s.sk_apply(f)
            if ci is None: return
            g = s.esk.geo; a_i, b_i = g.C[ci]["p"]
            start = s.chain["start"] if s.chain["start"] is not None else a_i
            if sp[2] == ("pt", start) and s.chain["start"] is not None:
                s.tp = []; s.chain = None; s.sk_prev = None; s.update(); return
            s.chain = dict(start=start, end=b_i, prev=ci); s.tp = [(g.P[b_i][0], g.P[b_i][1], ("pt", b_i))]
            return
        if T in ("spline", "cspline"):
            if len(s.tp) > 2 and T == "spline":
                q0, ok0 = s.sk_px(s.esk, s.tp[0][0], s.tp[0][1]); px, py = s.mouse_dev(p)
                if ok0 and math.hypot(q0[0] - px, q0[1] - py)/s.devicePixelRatioF() < 9: s.sk_finish_spline(s.tp, closed=True); return
            s.tp.append(sp); return
        s.sk_add_tp(sp)

    def sk_add_tp(s, sp):
        """Add the next point of a fixed-point-count tool (clicked or placed by a typed size); build when complete."""
        T = s.tool; n = SK_TOOLS[T][2]
        s.tp.append(sp)
        if len(s.tp) >= n:
            specs = list(s.tp[:n]); s.tp = []; typed = dict(s.tp_typed); s.tp_typed = {}
            typed = {i: tv for i, tv in typed.items() if i < len(specs) and xy(specs[i]) == tv[1]}
            def f(g):
                n0 = len(g.C); sp2 = [s.sk_prep(g, q) for q in specs]; r = s.sk_build(T, g, sp2)
                if typed and r is not None and r != []:
                    try: s.sk_typed_dims(T, g, r, typed)
                    except Exception: traceback.print_exc()
                s.sk_mark_new(g, n0); return r
            s.sk_apply(f); s.arc_acc = 0.0; s.arc_last = None
        s.update()

    def sk_finish_spline(s, pts, closed=False):
        T = s.tool; s.tp = []
        def f(g):
            n0 = len(g.C); sp = [s.sk_prep(g, q) for q in pts]
            r = build_spline(g, sp, closed) if T == "spline" else build_cspline(g, sp); s.sk_mark_new(g, n0); return r
        if len(pts) >= 2: s.sk_apply(f)
        s.sk_prev = None; s.update()

    # ---------------- keys ----------------
    # Whenever exactly one size is being asked for (drawing, placing a dimension, a command's distance / radius),
    # just type it - '30 mm' Enter - no need to click into a box first.
    TYPED_TOOLS = ("line", "circle", "rect2", "rectc", "poly_insc", "poly_circ", "slot_cc", "slot_overall", "slot_center", "ellipse")
    TOOL_TYPE_LABEL = {"line": "Length", "circle": "Radius", "rect2": "Width, height", "rectc": "Width, height", "poly_insc": "Radius",
                       "poly_circ": "Radius", "slot_cc": "Length, width", "slot_overall": "Length, width", "slot_center": "Length, width",
                       "ellipse": "Radius, minor radius"}
    # (tool, points placed so far) -> what the next point's typed size is
    STAGE_TYPED = {("circle2p", 1): "Diameter", ("rect3", 1): "Width", ("rect3", 2): "Height", ("poly_edge", 1): "Edge length",
                   ("arc3p", 1): "Chord", ("arcc", 1): "Radius", ("arcc", 2): "Sweep angle", ("ellipse", 2): "Minor radius",
                   ("slot_cc", 2): "Width", ("slot_overall", 2): "Width", ("slot_center", 2): "Width",
                   ("slot_arc3", 3): "Width", ("slot_arcc", 1): "Radius", ("slot_arcc", 2): "Sweep angle", ("slot_arcc", 3): "Width"}
    TYPE_START = "0123456789.-+(="

    def cmd_type_field(s):
        """The one number field of the open command that typing goes to (the dragged handle's, or the only one shown)."""
        f = s.cmd
        if not f: return None
        fields = [x for x in f.cmd.fields if x["type"] in ("len", "ang", "num", "int") and f.visible(x)]
        if s.handle: return next((x for x in fields if x["key"] == s.handle["key"]), None)
        return fields[0] if len(fields) == 1 else None

    def type_target(s):
        """Where typed numbers go now: ('tool'|'stage'|'dim'|'cmd', label, extra) or None."""
        if s.cmd:
            fd = s.cmd_type_field()
            return ("cmd", fd["label"].rstrip(":"), fd) if fd else None
        if not s.skedit or s.sk_opts["sk3d"]: return None
        T, n = s.tool, len(s.tp)
        if T == "dim" and s.dim_items and s.dim_prev: return ("dim", "Dimension", None)
        if (T, n) in s.STAGE_TYPED: return ("stage", s.STAGE_TYPED[(T, n)], None)
        if T in s.TYPED_TOOLS and n: return ("tool", s.TOOL_TYPE_LABEL.get(T, "Size"), None)
        return None

    def typing_active(s): return s.type_target() is not None

    def type_anchor(s):
        """Where the typing box goes (viewport logical coordinates)."""
        tg = s.type_target()
        if tg and tg[0] in ("tool", "stage") and s.sk_cursor:
            xy_, ok = s.sk_proj(s.esk, [[s.sk_cursor[0], s.sk_cursor[1]]])
            if ok[0]: return s.to_logical(xy_[0])
        if tg and tg[0] == "dim" and s.dim_prev and s.dim_prev.get("pos"):               # next to the dimension being placed
            q = s.dim_prev["pos"]; xy_, ok = s.sk_proj(s.esk, [[q[0], q[1]]])
            if ok[0]: return s.to_logical(xy_[0]) + C.QPointF(0, 6)
        if tg and tg[0] == "cmd" and s.handle and s.handle["key"] == tg[2]["key"] and s.mvp is not None:   # by the drag arrow's label
            h = s.handle; W_ = h["o"] + h["d"]*(h["val"] + s.dist*0.1*(1 if h["val"] >= 0 else -1))
            xy_, ok = s.project([W_.t()])
            if ok[0]: return s.to_logical(xy_[0]) + C.QPointF(-10, -4)
        q = s.mapFromGlobal(G.QCursor.pos())
        if not s.rect().contains(q): q = C.QPoint(s.width()//2, s.height()//2)
        return C.QPointF(q)

    def event(s, e):
        """While a size is being typed, keep letters (units like 'mm') away from the window's shortcuts."""
        if e.type() == C.QEvent.ShortcutOverride and s.typing_active():
            txt = e.text()
            if s.type_buf is not None or (txt and txt in s.TYPE_START):
                if txt and (txt.isprintable() or e.key() in (C.Qt.Key_Backspace,)): e.accept(); return True
        return super().event(e)

    def sk_type_key(s, e):
        """Type a size straight away: '30 mm' Enter.  Two values ('30, 20') for rectangles, slots, ellipses."""
        k = e.key(); txt = e.text()
        if s.type_buf is None:
            if txt and txt in s.TYPE_START: s.type_buf = "" if txt == "=" else txt; s.update(); return True
            return False
        if k == C.Qt.Key_Escape: s.type_buf = None; s.update(); return True
        if k == C.Qt.Key_Backspace:
            s.type_buf = s.type_buf[:-1] or None; s.update(); return True
        if k in (C.Qt.Key_Return, C.Qt.Key_Enter):
            if s.type_buf.strip(): s.apply_typed()
            else: s.type_buf = None; s.update()
            return True
        if txt and txt.isprintable(): s.type_buf += txt; s.update(); return True
        return True

    def type_error(s, text):
        if s.cmd: s.cmd.status.setText(text)
        if s.skedit: s.sk_flash(text)
        s.msg.emit(text)

    def read_typed(s, text, kind):
        """A typed value: lengths in mm (any unit, or parameters), angles in degrees, plain numbers."""
        text = text.strip()
        if kind == "len":
            try: return parse_len(text)
            except Exception: pass
        env = s.dim_env() if s.skedit else {}
        env = {**{k: p["v"] for k, p in (getattr(s, "params", None) or {}).items() if isinstance(p, dict) and "v" in p}, **env}
        return eval_expr(text.rstrip("°") if kind != "len" else text, env, "len" if kind == "len" else "ang" if kind == "ang" else "num")

    def apply_typed_cmd(s, fd):
        f = s.cmd; t = fd["type"]; buf = s.type_buf
        try: val = s.read_typed(buf, "len" if t == "len" else "ang" if t == "ang" else "num")
        except Exception: return s.type_error(f"Can't read '{buf}' - type a value like 30, 30 mm or 1.5 in")
        if t == "int": val = int(round(val))
        lo, hi = fd.get("lo"), fd.get("hi")
        if lo is not None and hi is not None and not (lo - 1e-9 <= val <= hi + 1e-9):
            return s.type_error(f"{fd['label'].rstrip(':')} must be between {fmt(lo)} and {fmt(hi)}.")
        s.type_buf = None; f.set_value(fd["key"], val)
        if s.handle and s.handle["key"] == fd["key"]: s.handle["val"] = val
        s.update()

    def apply_typed_dim(s):
        k = dict(s.dim_prev); buf = s.type_buf.strip()
        for x in ("drv", "v"): k.pop(x, None)
        s.type_buf = None; s.dim_items = []; s.dim_prev = None
        s.sk_add_dim(k, buf); s.update()

    def apply_typed_stage(s, label):
        T, tp = s.tool, list(s.tp); n = len(tp); buf = s.type_buf
        ang = "angle" in label
        try: val = s.read_typed(buf, "ang" if ang else "len")
        except Exception: return s.sk_flash(f"Can't read '{buf}' - type a size like 30, 30 mm or 1.5 in")
        if val <= 0 or (ang and val >= 360): return s.sk_flash("The angle must be between 0 and 360°." if ang else "The size must be bigger than zero.")
        cur = s.sk_cursor; last = xy(tp[-1]); cx, cy = (cur[0], cur[1]) if cur else (last[0] + 10, last[1] + 1)
        def unit(dx, dy):
            L = math.hypot(dx, dy); return (dx/L, dy/L) if L > 1e-12 else (1.0, 0.0)
        if ang:                                                  # sweep of a centre-point arc
            (ox, oy), (ax_, ay_) = xy(tp[0]), xy(tp[1]); R = math.hypot(ax_ - ox, ay_ - oy)
            sg = 1 if s.arc_acc >= 0 else -1; a0 = math.atan2(ay_ - oy, ax_ - ox) + sg*math.radians(val)
            s.arc_acc = sg*math.radians(val); pt = (ox + R*math.cos(a0), oy + R*math.sin(a0))
        elif label == "Width" and T.startswith("slot_arc"):
            if T == "slot_arc3": cc = circumcircle(xy(tp[0]), xy(tp[1]), xy(tp[2]))
            else: (ox, oy), (ax_, ay_) = xy(tp[0]), xy(tp[1]); cc = (ox, oy, math.hypot(ax_ - ox, ay_ - oy))
            if not cc: return s.sk_flash("Those points don't make an arc.")
            ux, uy = unit(cx - cc[0], cy - cc[1]); rr = cc[2] + val/2; pt = (cc[0] + ux*rr, cc[1] + uy*rr)
        elif n == 2 and label in ("Height", "Width", "Minor radius"):          # off the first side / axis
            (ax_, ay_), (bx, by) = xy(tp[0]), xy(tp[1]); dx, dy = unit(bx - ax_, by - ay_); nx, ny = -dy, dx
            base = (bx, by) if T == "rect3" else (ax_, ay_) if T == "ellipse" else (bx, by)
            side = 1 if (cx - base[0])*nx + (cy - base[1])*ny >= 0 else -1
            d = val/2 if T.startswith("slot") else val; pt = (base[0] + side*nx*d, base[1] + side*ny*d)
        else:                                                    # a distance from the last point towards the cursor
            ux, uy = unit(cx - last[0], cy - last[1]); pt = (last[0] + ux*val, last[1] + uy*val)
        s.type_buf = None; s.tp_typed[n] = (val, pt); s.sk_prev = None
        s.sk_add_tp((pt[0], pt[1], None))
        if s.tp: s.sk_preview()
        s.update()

    def sk_typed_dims(s, T, g, r, typed):
        """Dimensions for sizes that were typed while drawing (keyed by the point index they placed)."""
        def dim(t, e, val, **kw):
            k = dict(t=t, e=e, v=val, x=fmt_expr(val) if t != "ang" else fmt(val, 4), n=s.next_dim_name(g)); k.update(kw)
            k["pos"] = s.dim_default_pos(g, k); g.K.append(k)
        def seg(ci, val): a, b = g.C[ci]["p"][:2]; dim("dist", [list(P_(a)), list(P_(b))], val)
        for i, (val, _) in sorted(typed.items()):
            if T == "circle2p" and i == 1: dim("dia", [list(C_(r))], val)
            elif T == "rect3": seg(r[i - 1], val)
            elif T == "poly_edge" and i == 1: seg(r[0], val)
            elif T == "arc3p" and i == 1: dim("dist", [list(P_(g.C[r]["p"][1])), list(P_(g.C[r]["p"][2]))], val)
            elif T == "arcc" and i == 1: dim("rad", [list(C_(r))], val)
            elif T == "slot_arcc" and i == 1: dim("rad", [list(C_(r[4]))], val)
            elif T == "ellipse" and i == 2: dim("erad", [list(C_(r))], val, ax=2)
            elif T in ("slot_cc", "slot_overall", "slot_center") and i == 2: dim("dia", [list(C_(r[0]))], val)
            elif T in ("slot_arc3", "slot_arcc") and i == 3: dim("dia", [list(C_(r[2]))], val)

    def apply_typed(s):
        tg = s.type_target()
        if not tg: s.type_buf = None; s.update(); return
        if tg[0] == "cmd": return s.apply_typed_cmd(tg[2])
        if tg[0] == "dim": return s.apply_typed_dim()
        if tg[0] == "stage": return s.apply_typed_stage(tg[1])
        T = s.tool; parts = [p for p in re.split(r"[,;×]|\s+x\s+", s.type_buf) if p.strip()]
        try:
            vals = [parse_len(p) for p in parts[:1]]
            if len(parts) > 1: vals.append(float(parts[1].strip().rstrip("°").replace("deg", "")) if T == "line" else parse_len(parts[1]))
        except Exception:
            s.sk_flash(f"Can't read '{s.type_buf}' - type a size like 30, 30 mm or 1.5 in"); return
        if not vals or (T != "line" and vals[0] <= 0) or (T == "line" and vals[0] == 0): s.sk_flash("The size must be bigger than zero."); return
        a = xy(s.tp[-1] if T == "line" else s.tp[0]); cur = s.sk_cursor or (a[0] + 10, a[1] + 10, None)
        dx, dy = cur[0] - a[0], cur[1] - a[1]; v0 = vals[0]; v1 = vals[1] if len(vals) > 1 else None
        if T == "line":
            ang = v1 if v1 is not None else (math.degrees(math.atan2(dy, dx)) if math.hypot(dx, dy) > 1e-9 else 0.0)
            if v0 < 0: v0, ang = -v0, ang + 180
            tv = dict(l=v0, a=ang)
        elif T == "circle": tv = dict(r=v0)
        elif T in ("rect2", "rectc"):
            k_ = 2 if T == "rectc" else 1; tv = dict(w=v0, h=v1 if v1 is not None else (abs(dy)*k_ or v0))
        elif T in ("poly_insc", "poly_circ"): tv = dict(d=2*v0, n=s.sk_opts["sides"])
        elif T.startswith("slot"): tv = dict(l=v0, w=v1 if v1 is not None else 6.0)
        else: tv = dict(a=v0, b=v1 if v1 is not None else v0/2, t=math.degrees(math.atan2(dy, dx)) if math.hypot(dx, dy) > 1e-9 else 0.0)
        s.type_buf = None; s.type_values = tv; s.type_dialog.emit(T); s.type_values = None; s.sk_prev = None; s.update()

    def sk_key(s, e):
        k = e.key(); ret = k in (C.Qt.Key_Return, C.Qt.Key_Enter)
        if s.typing_active() and s.sk_type_key(e): return True
        if s.type_buf is not None and not s.typing_active(): s.type_buf = None
        if k == C.Qt.Key_Escape:
            if s.sk_drag: s.esk.geo = s.skedit["hist"][s.skedit["hpos"]]; s.sk_drag = None
            elif s.tool and (s.tp or s.tsel or s.dim_items): s.tp, s.tsel, s.dim_items = [], [], []; s.chain = None; s.sk_prev = None
            elif s.tool: s.sk_set_tool(None)
            else: s.sk_sel = set()
            s.changed.emit(); s.update(); return True
        if ret and s.sk_opts["sk3d"] and s.tool in ("line", "spline") and s.tp3: s.sk_finish3(); return True
        if k == C.Qt.Key_Escape and s.tp3: s.tp3 = []; s.update(); return True
        if ret:
            if s.tool in ("spline", "cspline") and len(s.tp) >= 2: s.sk_finish_spline(list(s.tp)); return True
            if s.tool == "line": s.tp = []; s.chain = None; s.sk_prev = None; s.update(); return True
            if s.tool and not s.tp: s.sk_set_tool(None); s.changed.emit(); return True
            return True
        if k == C.Qt.Key_Tab and s.tool and s.tp: s.type_dialog.emit(s.tool); return True
        if k in (C.Qt.Key_Delete, C.Qt.Key_Backspace) and s.sk_sel: s.sk_delete_sel(); return True
        if k == C.Qt.Key_X and not e.modifiers(): s.sk_toggle_flag("cons"); s.changed.emit(); return True
        return False

    # ---------------- selection actions ----------------
    def sk_delete_sel(s):
        sel = set(s.sk_sel); s.sk_sel = set()
        cs = [i[1] for i in sel if i[0] == "c"]; ps = [i[1] for i in sel if i[0] == "p"]; ks = {i[1] for i in sel if i[0] == "k"}
        def f(g):
            g.K = [k for ki, k in enumerate(g.K) if ki not in ks]
            geo_delete(g, curves=cs, points=ps)
        s.sk_apply(f, msg=f"Deleted {len(sel)} item(s).")

    def sk_toggle_flag(s, flag):
        cs = [i[1] for i in s.sk_sel if i[0] == "c"]
        if not cs: s.sk_opts["construction" if flag == "cons" else "centerline"] ^= True; s.changed.emit(); return
        def f(g):
            on = not all(g.C[ci].get(flag) for ci in cs)
            for ci in cs:
                if on: g.C[ci][flag] = True; g.C[ci].pop("cl" if flag == "cons" else "cons", None)
                else: g.C[ci].pop(flag, None)
        s.sk_apply(f, solve=False)

    # ---------------- constraints ----------------
    def sk_con_from_sel(s, t):
        items = sorted(s.sk_sel, key=lambda i: (i[0] != "p", i[1]))
        items = [i for i in items if i[0] in ("p", "c")]
        if not items: return False
        if t == "fix":
            for it in items: s.sk_con_pick("fix", it)
            return True
        s.tsel = []
        for it in items: s.sk_con_pick(t, it, quiet=True)
        s.sk_sel = set(); return True

    def sk_con_pick(s, t, it, quiet=False):
        geo = s.esk.geo
        if t == "fix":
            ex = next((ki for ki, k in enumerate(geo.K) if k["t"] == "fix" and k["e"] == [list(it)]), None)
            if ex is not None:
                def f(g): del g.K[ex]
                s.sk_apply(f, msg="Unfixed."); return
            s.sk_add_con(dict(t="fix", e=[list(it)])); return
        s.tsel.append(it); items = s.tsel; kinds = [i[0] for i in items]
        need = {"hor": None, "coin": 2, "mid": 2, "par": 2, "perp": 2, "col": 2, "conc": 2, "tan": 2, "smooth": 2, "eq": 2, "sym": 3}[t]
        if t == "hor":
            if kinds == ["c"] and geo.C[it[1]]["k"] in ("line", "ellipse", "earc"):
                c = geo.C[it[1]]; a, b = c["p"][:2]; dx, dy = geo.P[b][0] - geo.P[a][0], geo.P[b][1] - geo.P[a][1]
                s.tsel = []; s.sk_add_con(dict(t="hor" if abs(dx) >= abs(dy) else "ver", e=[list(it)])); return
            if kinds == ["p", "p"]:
                (ax, ay), (bx, by) = geo.P[items[0][1]], geo.P[items[1][1]]; s.tsel = []
                s.sk_add_con(dict(t="hor" if abs(bx - ax) >= abs(by - ay) else "ver", e=[list(i) for i in items])); return
            if kinds != ["p"]: s.tsel = []; s.sk_flash("Horizontal / vertical: pick a line or two points.")
            s.update(); return
        if len(items) < need: s.update(); return
        s.tsel = []
        if t == "mid" and kinds == ["c", "p"]: items = items[::-1]
        if t == "coin" and kinds == ["c", "p"]: items = items[::-1]
        if t == "conc":
            items = [i if i[0] == "c" else i for i in items]
        k = dict(t=t, e=[list(i) for i in items])
        if t in ("tan", "smooth"):
            if any(i[0] != "c" for i in items): s.sk_flash("Pick two curves."); return
            k = tangent_con(geo, items[0][1], items[1][1]); k["t"] = t
            if t == "smooth" and not k.get("j"): s.sk_flash(CON_HINT["smooth"]); return
        if t == "sym":
            if items[2][0] != "c" or geo.C[items[2][1]]["k"] != "line": s.sk_flash("The third pick must be a line."); return
            if kinds[:2] == ["c", "c"] and geo.C[items[0][1]]["k"] == "line" and geo.C[items[1][1]]["k"] == "line":
                a1, b1 = geo.C[items[0][1]]["p"]; a2, b2 = geo.C[items[1][1]]["p"]
                M = mat_mirror(*geo.P[geo.C[items[2][1]]["p"][0]], *geo.P[geo.C[items[2][1]]["p"][1]]); fn = mat_fn(M)
                ma = fn(*geo.P[a1]); d_same = math.hypot(ma[0] - geo.P[a2][0], ma[1] - geo.P[a2][1]); d_sw = math.hypot(ma[0] - geo.P[b2][0], ma[1] - geo.P[b2][1])
                if d_sw < d_same: k["sw"] = True
        if t == "coin" and kinds == ["p", "p"]:
            i, j = items[0][1], items[1][1]
            if i == j: return
            def f(g):                                            # merge the two points (Fusion keeps them coincident)
                g.con("coin", P_(i), P_(j))
            g2, err = geo_try_add(geo, dict(t="coin", e=[P_(i), P_(j)]))
            if err: s.sk_flash(err); return
            s.sk_commit(g2, solve=False); return
        s.sk_add_con(k)

    def sk_add_con(s, k):
        k["e"] = [[r[0], int(r[1])] for r in k["e"]]
        geo = s.esk.geo; g2, err = geo_try_add(geo, k)
        if err: s.sk_flash(err); s.update(); return False
        s.sk_commit(g2, solve=False, msg=f"{CON_NAMES.get(k['t'], k['t'])} added."); return True

    def sk_con_suppress(s, ki):
        def f(g):
            k = g.K[ki]
            if k.get("off"): k.pop("off")
            else: k["off"] = True
        s.sk_apply(f)

    # ---------------- dimensions ----------------
    def dim_default_pos(s, geo, k):
        sy = SkSys(geo); x = sy.x; e = k["e"]; off = s.sk_world_len(26)
        P = lambda r: (x[2*r[1]], x[2*r[1] + 1]) if r[0] == "p" else sy.center(x, r[1])
        try:
            if k["t"] in ("rad", "dia", "erad"):
                c = sy.center(x, e[0][1]); r = sy.radius(x, e[0][1]) if k["t"] != "erad" else (sy.ell(x, e[0][1])[4])
                return [c[0] + r*0.7 + off, c[1] + r*0.7 + off]
            if k["t"] == "ang":
                ax, ay, bx, by = sy.line(x, e[0][1]); return [(ax + bx)/2 + off, (ay + by)/2 + off]
            if k["t"] in ("pldist", "lldist"):
                a = P(e[0]) if e[0][0] == "p" else ((sy.line(x, e[0][1])[0] + sy.line(x, e[0][1])[2])/2, (sy.line(x, e[0][1])[1] + sy.line(x, e[0][1])[3])/2)
                return [a[0] + off, a[1] + off]
            a, b = P(e[0]), P(e[1])
            if k["t"] == "hdist": return [(a[0] + b[0])/2, max(a[1], b[1]) + off]
            if k["t"] == "vdist": return [max(a[0], b[0]) + off, (a[1] + b[1])/2]
            dx, dy = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dy) or 1
            return [(a[0] + b[0])/2 - dy/L*off, (a[1] + b[1])/2 + dx/L*off]
        except Exception: return [0.0, 0.0]

    def sk_dim_build(s, items, m):
        """The dimension the picked items + mouse position m (sketch coords) describe, or None."""
        geo = s.esk.geo; sy = SkSys(geo); x = sy.x
        if not items: return None
        def as_pt(i):
            if i[0] == "p": return i
            c = geo.C[i[1]]; cc = crv_center(c)
            return ["p", cc] if cc is not None else None
        def hv(a, b):
            (ax, ay), (bx, by) = geo.P[a[1]], geo.P[b[1]]
            lo_x, hi_x, lo_y, hi_y = min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)
            if lo_x <= m[0] <= hi_x and not (lo_y <= m[1] <= hi_y) and abs(bx - ax) > 1e-9: t = "hdist"
            elif lo_y <= m[1] <= hi_y and not (lo_x <= m[0] <= hi_x) and abs(by - ay) > 1e-9: t = "vdist"
            else: t = "dist"
            if abs(by - ay) < 1e-9 and t == "dist": t = "hdist"
            if abs(bx - ax) < 1e-9 and t == "dist": t = "vdist"
            k = dict(t=t, e=[list(a), list(b)])
            if t == "hdist": k["sg"] = 1 if bx >= ax else -1
            if t == "vdist": k["sg"] = 1 if by >= ay else -1
            return k
        if len(items) == 1:
            i = items[0]
            if i[0] == "p": return None
            c = geo.C[i[1]]
            if c["k"] == "line": a, b = c["p"]; return hv(["p", a], ["p", b])
            if c["k"] == "circle": return dict(t="dia", e=[list(i)])
            if c["k"] == "arc": return dict(t="rad", e=[list(i)])
            if c["k"] in ("ellipse", "earc"):
                fr = sy.ell(x, i[1]); dx, dy = m[0] - fr[0], m[1] - fr[1]
                major = abs(dx*fr[2] + dy*fr[3])/fr[4] >= abs(-dx*fr[3] + dy*fr[2])/fr[5]
                return dict(t="erad", e=[list(i)], ax=1 if major else 2)
            return None
        A, B = items[0], items[1]
        cA = geo.C[A[1]] if A[0] == "c" else None; cB = geo.C[B[1]] if B[0] == "c" else None
        if cA and cB and cA["k"] == "line" and cB["k"] == "line":
            ax, ay, bx, by = sy.line(x, A[1]); cx, cy, dx, dy = sy.line(x, B[1])
            u = (bx - ax, by - ay); v = (dx - cx, dy - cy); cr = u[0]*v[1] - u[1]*v[0]
            if abs(cr) < 1e-9*math.hypot(*u)*math.hypot(*v):
                k = dict(t="lldist", e=[list(A), list(B)]); k["sg"] = 1 if sy.meas(x, dict(k, sg=1)) >= 0 else -1; return k
            den = cr; t_ = ((cx - ax)*v[1] - (cy - ay)*v[0])/den; X = (ax + u[0]*t_, ay + u[1]*t_)
            mx, my = m[0] - X[0], m[1] - X[1]; best = None
            for f1 in (1, -1):
                for f2 in (1, -1):
                    p1 = (u[0]*f1, u[1]*f1); p2 = (v[0]*f2, v[1]*f2)
                    a1, a2, am = math.atan2(p1[1], p1[0]), math.atan2(p2[1], p2[0]), math.atan2(my, mx)
                    span = (a2 - a1) % math.tau; rel = (am - a1) % math.tau
                    if span > math.pi: span2 = math.tau - span; rel2 = (a1 - am) % math.tau; inside_ = rel2 <= span2
                    else: inside_ = rel <= span
                    if inside_: best = (f1, f2); break
                if best: break
            f1, f2 = best or (1, 1)
            k = dict(t="ang", e=[list(A), list(B)], f1=f1, f2=f2); k["sg"] = 1 if sy.meas(x, dict(k, sg=1)) >= 0 else -1; return k
        if (cA and cA["k"] == "line") != (cB and cB["k"] == "line"):
            L_, P0 = (A, B) if cA and cA["k"] == "line" else (B, A)
            pp = as_pt(P0)
            if pp is None: return None
            k = dict(t="pldist", e=[pp, list(L_)]); k["sg"] = 1 if sy.meas(x, dict(k, sg=1)) >= 0 else -1; return k
        pa, pb = as_pt(A), as_pt(B)
        if pa is None or pb is None or pa == pb: return None
        return hv(pa, pb)

    def sk_dim_preview(s, p):
        m = s.sk_local_mouse(p)
        if m is None or not s.dim_items: s.dim_prev = None; return
        items = list(s.dim_items)
        if len(items) == 1 and s.sk_hover and s.sk_hover not in items and s.sk_hover[0] in ("p", "c"): items.append(s.sk_hover)
        k = s.sk_dim_build(items if len(items) <= 2 else items[:2], m)
        if k: k["pos"] = list(m); k["v"] = geo_measure(s.esk.geo, k); k["drv"] = True
        s.dim_prev = k

    def sk_dim_click(s, p):
        it = s.sk_pick(p, ("p", "c")); m = s.sk_local_mouse(p)
        if it and it not in s.dim_items and len(s.dim_items) < 2:
            if it[0] == "c" and s.esk.geo.C[it[1]]["k"] in ("point", "text"): return
            s.dim_items.append(it); s.sk_dim_preview(p); s.update(); return
        if not s.dim_items or m is None: return
        k = s.sk_dim_build(s.dim_items, m); s.dim_items = []; s.dim_prev = None
        if not k: s.sk_flash("Those items can't be dimensioned together."); return
        k["pos"] = list(m); s.dim_place.emit(k); s.update()

    def sk_add_dim(s, k, expr):
        """Called by the window after the value was typed."""
        geo = s.esk.geo; k = dict(k); k["n"] = s.next_dim_name(); k["x"] = expr; k["e"] = [[r[0], int(r[1])] for r in k["e"]]
        try: k["v"] = eval_expr(expr, s.dim_env(), "ang" if k["t"] == "ang" else "len")
        except ExprError as ex: s.sk_flash(str(ex)); return False
        if k["t"] == "ang": k["v"] = abs(k["v"])
        k.pop("drv", None)
        g2, err = geo_try_add(geo, k)
        if err:
            r = W.QMessageBox.question(s, "Sketch Dimension", f"{err}\n\nCreate a driven (reference) dimension instead?")
            if r != W.QMessageBox.Yes: return False
            k["drv"] = True; k["x"] = ""; g2 = geo.copy(); g2.K.append(k)
        s.sk_commit(g2, solve=False, msg=f"Dimension {k['n']} added."); return True

    def sk_set_dim(s, ki, expr, driven=None):
        def f(g):
            k = g.K[ki]
            if driven is not None:
                if driven: k["drv"] = True
                else: k.pop("drv", None)
            if expr is not None and not k.get("drv"): k["x"] = expr
        g = s.esk.geo.copy(); f(g)
        errs = geo_eval_dims(g, s.dim_env())
        if errs: s.sk_flash(errs[0]); return False
        ok, worst, _ = geo_solve(g)
        if not ok: s.sk_flash("That value can't be reached with the other constraints."); return False
        s.sk_commit(g, solve=False); return True

    def dim_text(s, geo, k):
        try: v = geo_measure(geo, k) if k.get("drv") else k.get("v", 0.0)
        except Exception: v = k.get("v", 0.0)
        t = k["t"]
        if t == "ang": txt = f"{fmt(abs(v), 2)}°"
        elif t == "rad": txt = f"R {flen(abs(v))}"
        elif t == "dia": txt = f"⌀ {flen(abs(v))}"
        elif t == "erad": txt = f"R{'a' if k.get('ax', 1) == 1 else 'b'} {flen(abs(v))}"
        else: txt = flen(abs(v))
        return f"({txt})" if k.get("drv") else txt

    def dim_geom(s, geo, k):
        """Dimension graphics in sketch coordinates: (segments, arrows (tip, direction)), label position."""
        sy = SkSys(geo); x = sy.x; e = k["e"]; t = k["t"]; L = k.get("pos") or s.dim_default_pos(geo, k)
        P = lambda r: (x[2*r[1]], x[2*r[1] + 1]) if r[0] == "p" else sy.center(x, r[1])
        segs, arrows = [], []; ext = s.sk_world_len(6)
        def dimline(a, b, inside=True):
            segs.append((a, b)); d = (b[0] - a[0], b[1] - a[1]); n = math.hypot(*d) or 1
            arrows.append((a, (-d[0]/n, -d[1]/n))); arrows.append((b, (d[0]/n, d[1]/n)))
        if t in ("dist", "hdist", "vdist"):
            a, b = P(e[0]), P(e[1])
            if t == "hdist":
                y = L[1]; A2, B2 = (a[0], y), (b[0], y); sg = 1 if y >= a[1] else -1; sgb = 1 if y >= b[1] else -1
                segs += [(a, (a[0], y + sg*ext)), (b, (b[0], y + sgb*ext))]
            elif t == "vdist":
                xx = L[0]; A2, B2 = (xx, a[1]), (xx, b[1]); sg = 1 if xx >= a[0] else -1; sgb = 1 if xx >= b[0] else -1
                segs += [(a, (xx + sg*ext, a[1])), (b, (xx + sgb*ext, b[1]))]
            else:
                dx, dy = b[0] - a[0], b[1] - a[1]; Ln = math.hypot(dx, dy) or 1; nx, ny = -dy/Ln, dx/Ln
                o = (L[0] - a[0])*nx + (L[1] - a[1])*ny; sg = 1 if o >= 0 else -1
                A2, B2 = (a[0] + nx*o, a[1] + ny*o), (b[0] + nx*o, b[1] + ny*o)
                segs += [(a, (A2[0] + nx*ext*sg, A2[1] + ny*ext*sg)), (b, (B2[0] + nx*ext*sg, B2[1] + ny*ext*sg))]
            dimline(A2, B2)
        elif t in ("pldist", "lldist"):
            if t == "pldist": p = P(e[0]); li = e[1][1]
            else:
                ax, ay, bx, by = sy.line(x, e[1][1]); p = ((ax + bx)/2, (ay + by)/2); li = e[0][1]
            ax, ay, bx, by = sy.line(x, li); dx, dy = bx - ax, by - ay; Ln = math.hypot(dx, dy) or 1; ux, uy = dx/Ln, dy/Ln
            tt = (p[0] - ax)*ux + (p[1] - ay)*uy; F = (ax + ux*tt, ay + uy*tt)
            sh = (L[0] - p[0])*ux + (L[1] - p[1])*uy
            A2, B2 = (p[0] + ux*sh, p[1] + uy*sh), (F[0] + ux*sh, F[1] + uy*sh)
            segs += [(p, A2), (F, B2)]; dimline(A2, B2)
        elif t == "ang":
            ax, ay, bx, by = sy.line(x, e[0][1]); cx, cy, dx, dy = sy.line(x, e[1][1])
            u = ((bx - ax)*k.get("f1", 1), (by - ay)*k.get("f1", 1)); v = ((dx - cx)*k.get("f2", 1), (dy - cy)*k.get("f2", 1))
            cr = u[0]*v[1] - u[1]*v[0]
            if abs(cr) < 1e-12: return segs, arrows, L
            den = (bx - ax)*(dy - cy) - (by - ay)*(dx - cx); tt = ((cx - ax)*(dy - cy) - (cy - ay)*(dx - cx))/den
            X = (ax + (bx - ax)*tt, ay + (by - ay)*tt)
            R = math.hypot(L[0] - X[0], L[1] - X[1]); a1 = math.atan2(u[1], u[0]); a2 = math.atan2(v[1], v[0])
            span = (a2 - a1 + math.pi) % math.tau - math.pi
            pts = [(X[0] + R*math.cos(a1 + span*q), X[1] + R*math.sin(a1 + span*q)) for q in np.linspace(0, 1, 25)]
            segs += list(zip(pts, pts[1:]))
            for (lx0, ly0, lx1, ly1), aa, P_end in (((ax, ay, bx, by), a1, pts[0]), ((cx, cy, dx, dy), a1 + span, pts[-1])):
                near = min(((lx0, ly0), (lx1, ly1)), key=lambda q: math.hypot(q[0] - P_end[0], q[1] - P_end[1]))
                if math.hypot(near[0] - X[0], near[1] - X[1]) < R: segs.append((near, (X[0] + (R + ext)*math.cos(aa), X[1] + (R + ext)*math.sin(aa))))
            sgn = 1 if span >= 0 else -1
            arrows.append((pts[0], (math.sin(a1)*sgn, -math.cos(a1)*sgn)))
            arrows.append((pts[-1], (-math.sin(a1 + span)*sgn, math.cos(a1 + span)*sgn)))
        elif t in ("rad", "dia", "erad"):
            ci = e[0][1]; c = sy.center(x, ci)
            if t == "erad":
                fr = sy.ell(x, ci)
                if not fr: return segs, arrows, L
                ux, uy = (fr[2], fr[3]) if k.get("ax", 1) == 1 else (-fr[3], fr[2]); r = fr[4] if k.get("ax", 1) == 1 else fr[5]
                sgn = 1 if (L[0] - c[0])*ux + (L[1] - c[1])*uy >= 0 else -1; ux, uy = ux*sgn, uy*sgn
            else:
                r = sy.radius(x, ci); dx, dy = L[0] - c[0], L[1] - c[1]; n = math.hypot(dx, dy) or 1; ux, uy = dx/n, dy/n
            Q = (c[0] + ux*r, c[1] + uy*r)
            if t == "dia": segs.append(((c[0] - ux*r, c[1] - uy*r), Q)); arrows.append(((c[0] - ux*r, c[1] - uy*r), (-ux, -uy)))
            else: segs.append((c, Q))
            arrows.append((Q, (ux, uy))); segs.append((Q, tuple(L)))
        return segs, arrows, L

    # ---------------- drawing ----------------
    def sk_glline(s, pts, joints=0.0):
        glBegin(GL_LINE_STRIP)
        for q in pts: glVertex3f(float(q[0]), float(q[1]), float(q[2]) if len(q) > 2 else 0.0)
        glEnd()
        if joints and len(pts) > 2:                                    # round off the joints of thick polylines
            glPointSize(joints); glBegin(GL_POINTS)
            for q in pts[1:-1]: glVertex3f(float(q[0]), float(q[1]), float(q[2]) if len(q) > 2 else 0.0)
            glEnd()

    def sk_draw(s, sk, editing):
        geo = sk.geo; st_ = s.sk_status() if editing else None
        glPushMatrix(); glMultMatrixd(sk.plane.gl())
        if (not editing or s.sk_opts["profiles"]) and not s.sk_drag:          # profiles (closed regions)
            try: regs = sk.regions()
            except Exception: regs = []
            for i, r in enumerate(regs):
                sel = i in sk.sel; hov = s.hover_reg == (sk, i)
                al = 0.5 if sel else 0.34 if hov else (0.16 if editing else 0.12)
                if s.ext and sk is s.active and not sk.sel: al = 0.32
                glColor4f(*((0.2, 0.5, 0.95) if sel else FILL), al); s.tris_array(r.tris)
        hov = s.sk_hover if editing else None; sel = s.sk_sel if editing else set()
        tsel = set(s.tsel) if editing else set(); dsel = set(i[1] for i in s.dim_items if i[0] == "c") if editing else set()
        fixed_pts = set()
        if editing:
            for k in geo.K:
                if k["t"] == "fix" and not k.get("off"):
                    for r in k["e"]:
                        if r[0] == "p": fixed_pts.add(r[1])
                        else: fixed_pts |= set(crv_points(geo.C[r[1]]))
        trim_hl = None
        if editing and s.sk_trim_prev:
            try:
                g2 = geo.copy(); geo_trim(g2, s.sk_trim_prev[0], *s.sk_trim_prev[1]); trim_hl = s.sk_trim_prev[0]
            except Exception: trim_hl = None
        for ci, c in enumerate(geo.C):
            k_ = c["k"]
            if k_ == "point": continue
            if c.get("src") and editing and not s.sk_opts["proj"]: continue
            if not editing: col, w = SKC["other"], 1.5
            elif ("c", ci) in sel: col, w = SKC["sel"], 3.0
            elif hov == ("c", ci) or ci in tsel or ci in dsel: col, w = SKC["hot"], 2.6
            elif c.get("src") or c.get("lock") or k_ == "proj3": col, w = SKC["proj"], 1.8
            elif c.get("cons") or c.get("cl"): col, w = SKC["cons"], 1.4
            elif k_ in ("l3", "s3"): col, w = SKC["free"], 2.0
            elif crv_points(c) and all(i in fixed_pts for i in crv_points(c)): col, w = SKC["fix"], 2.0
            elif st_ and ci in st_["curves"]: col, w = SKC["full"], 2.0
            else: col, w = SKC["free"], 2.0
            if trim_hl == ci: col, w = SKC["bad"], 2.6
            if c.get("broken"): col = SKC["bad"]
            glColor3f(*col); glLineWidth(w)
            dashed = c.get("cons") or c.get("cl")
            if dashed: glEnable(GL_LINE_STIPPLE); glLineStipple(2, 0x18FF if c.get("cl") else 0x0F0F)
            for A in geo_polys(geo, ci): s.sk_glline(A, 0 if dashed or w < 1.9 else w*0.85)
            if dashed: glDisable(GL_LINE_STIPPLE)
            if editing and k_ == "cspline":                                   # control polygon
                glColor4f(*SKC["free"], 0.45); glLineWidth(1); glEnable(GL_LINE_STIPPLE); glLineStipple(1, 0x3333)
                s.sk_glline([geo.P[i] for i in c["p"]]); glDisable(GL_LINE_STIPPLE)
            if editing and k_ == "spline":                                    # tangent handles
                for j, h in enumerate(c.get("h") or []):
                    if h is None: continue
                    glColor4f(*SKC["free"], 0.8); glLineWidth(1.2); s.sk_glline([geo.P[c["p"][j]], geo.P[h]])
        if s.sk_opts["dims"] and (editing or sk.visible):                          # dimensions (every visible sketch)
            for ki, k in enumerate(geo.K):
                if k["t"] not in DIM_TYPES: continue
                try: segs, arrows, L = s.dim_geom(geo, k)
                except Exception: continue
                hot = hov == ("k", ki) or ("k", ki) in sel
                glColor4f(*(SKC["sel"] if hot else (0.25, 0.28, 0.33) if editing else (0.42, 0.47, 0.55)), 0.95 if editing else 0.8)
                glLineWidth(1.6 if hot else 1.0); s.sk_draw_dim(segs, arrows)
        if editing and s.dim_prev:
            try:
                segs, arrows, L = s.dim_geom(geo, s.dim_prev); glColor4f(*SKC["pre"], 0.9); glLineWidth(1.2); s.sk_draw_dim(segs, arrows)
            except Exception: pass
        if editing and s.sk_opts["points"] or editing and s.tool:                # points
            used = []
            for c in geo.C:
                if c.get("src") and not s.sk_opts["proj"]: continue
                for i in crv_points(c):
                    used.append(i)
            used = sorted(set(used))
            if used:
                glPointSize(7); glBegin(GL_POINTS)
                for i in used:
                    if ("p", i) in sel or hov == ("p", i): glColor3f(*SKC["sel"])
                    elif i in fixed_pts: glColor3f(*SKC["fix"])
                    elif st_ and i in st_["points"]: glColor3f(*SKC["full"])
                    else: glColor3f(*SKC["free"])
                    glVertex3f(geo.P[i][0], geo.P[i][1], 0)
                glEnd()
                glPointSize(4); glColor3f(1, 1, 1); glBegin(GL_POINTS)
                for i in used:
                    if not (st_ and i in st_["points"]) and i not in fixed_pts and ("p", i) not in sel and hov != ("p", i):
                        glVertex3f(geo.P[i][0], geo.P[i][1], 0)
                glEnd()
        else:
            pts = [geo.P[c["p"][0]] for c in geo.C if c["k"] == "point" and not c.get("src")]
            if pts:
                glPointSize(5); glColor3f(*SKC["other"]); glBegin(GL_POINTS)
                for q in pts: glVertex3f(q[0], q[1], 0)
                glEnd()
        if editing and s.sk_prev:                                                # tool preview
            pv = s.sk_prev; g2 = pv["geo"]; glColor4f(*SKC["pre"], 0.95); glLineWidth(1.8)
            for ci in range(pv["n0"], len(g2.C)):
                c = g2.C[ci]
                if c["k"] == "point": continue
                if c.get("cons"): glEnable(GL_LINE_STIPPLE); glLineStipple(2, 0x0F0F)
                for A in geo_curve_pts(g2, ci): s.sk_glline(A)
                if c.get("cons"): glDisable(GL_LINE_STIPPLE)
            if pv["lines"]: glEnable(GL_LINE_STIPPLE); glLineStipple(2, 0x3333); s.sk_glline(pv["lines"]); glDisable(GL_LINE_STIPPLE)
        if editing:                                                              # 3D sketch: end points, rubber band, cursor
            P3 = [q for c in geo.C if c["k"] in ("l3", "s3") for q in (c["q"][0], c["q"][-1])]
            if P3:
                glPointSize(6); glColor3f(*SKC["free"]); glBegin(GL_POINTS)
                for q in P3: glVertex3f(*q)
                glEnd()
            if s.tp3:
                pts = s.tp3 + ([s.sk_cursor3[:3]] if s.sk_cursor3 else [])
                glColor4f(*SKC["pre"], 0.95); glLineWidth(1.8)
                if s.tool == "spline" and len(pts) > 2:
                    try:
                        cv = interp3(pts); us = np.linspace(cv.FirstParameter(), cv.LastParameter(), 20*len(pts))
                        s.sk_glline([(cv.Value(float(u)).X(), cv.Value(float(u)).Y(), cv.Value(float(u)).Z()) for u in us])
                    except Exception: s.sk_glline(pts)
                else: s.sk_glline(pts)
                glPointSize(7); glBegin(GL_POINTS)
                for q in s.tp3: glVertex3f(*q)
                glEnd()
            if s.sk_cursor3 and s.tool:
                glPointSize(9); glColor3f(*SEL); glBegin(GL_POINTS); glVertex3f(*s.sk_cursor3[:3]); glEnd()
        if editing and s.tool and s.sk_cursor and not (s.sk_opts["sk3d"] and s.tool in ("line", "spline")):
            glPointSize(9); glColor3f(*SEL); glBegin(GL_POINTS); glVertex3f(s.sk_cursor[0], s.sk_cursor[1], 0); glEnd()
        if editing and s.tp:
            glPointSize(7); glColor3f(*SKC["pre"]); glBegin(GL_POINTS)
            for q in s.tp: glVertex3f(q[0], q[1], 0)
            glEnd()
        glPopMatrix()

    def sk_draw_dim(s, segs, arrows):
        glBegin(GL_LINES)
        for a, b in segs: glVertex3f(a[0], a[1], 0); glVertex3f(b[0], b[1], 0)
        glEnd()
        L = s.sk_world_len(9); Wd = L*0.33
        glBegin(GL_TRIANGLES)
        for tip, d in arrows:
            n = math.hypot(*d) or 1; ux, uy = d[0]/n, d[1]/n; bx, by = tip[0] - ux*L, tip[1] - uy*L
            glVertex3f(tip[0], tip[1], 0); glVertex3f(bx - uy*Wd, by + ux*Wd, 0); glVertex3f(bx + uy*Wd, by - ux*Wd, 0)
        glEnd()

    def draw_canvases(s):
        if not s.canv: return
        for c in s.canv:
            if not c.get("visible", True): continue
            tex = s.canvas_tex(c)
            if not tex: continue
            pl = c["plane_obj"]; w = float(c.get("w", 100)); img = c["_img"]; h = w*img.height()/max(img.width(), 1)
            x0, y0 = float(c.get("x", 0)), float(c.get("y", 0)); rot = math.radians(float(c.get("rot", 0)))
            ca, sa = math.cos(rot), math.sin(rot); fx, fy = (-1 if c.get("flipx") else 1), (-1 if c.get("flipy") else 1)
            off = float(c.get("lift", 0.0)) + (s.dist*0.0004 if c.get("decal") else 0.0)
            glPushMatrix(); glMultMatrixd(pl.gl()); glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, tex)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            if not c.get("decal"): glDepthMask(GL_FALSE)
            glColor4f(1, 1, 1, float(c.get("op", 0.6))); glBegin(GL_QUADS)
            for tu, tv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                lx, ly = (tu - 0.5)*w*fx, (tv - 0.5)*h*fy
                glTexCoord2f(tu, 1 - tv); glVertex3f(x0 + ca*lx - sa*ly, y0 + sa*lx + ca*ly, off)
            glEnd(); glDisable(GL_TEXTURE_2D); glDepthMask(GL_TRUE); glPopMatrix()

    def canvas_tex(s, c):
        key = c.get("img", "")[:64] + str(len(c.get("img", "")))
        if not hasattr(s, "_tex"): s._tex = {}
        if key in s._tex: c["_img"] = s._tex[key][1]; return s._tex[key][0]
        import base64
        img = G.QImage(); img.loadFromData(base64.b64decode(c["img"]))
        if img.isNull(): return None
        img = img.convertToFormat(G.QImage.Format_RGBA8888); c["_img"] = img
        t = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, t)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        ptr = img.constBits(); data = bytes(ptr)[:img.sizeInBytes()]
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width(), img.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        s._tex[key] = (t, img); return t

    # ---------------- labels & glyphs (overlay) ----------------
    def auto_dims(s, geo):
        """Measured sizes of curves that have no dimension yet: (sketch pos, text, kind, curve)."""
        out = []; off = s.sk_world_len(14); have = set()
        for k in geo.K:
            if k["t"] in ("dist", "hdist", "vdist") and all(r[0] == "p" for r in k["e"]): have.add(frozenset(r[1] for r in k["e"]))
            elif k["t"] in ("rad", "dia"): have.add(("r", k["e"][0][1]))
        for ci, c in enumerate(geo.C):
            if c.get("src") or c.get("lock") or c["k"] not in ("line", "circle", "arc"): continue
            try:
                if c["k"] == "line":
                    a, b = c["p"]
                    if frozenset((a, b)) in have: continue
                    (ax, ay), (bx, by) = geo.P[a], geo.P[b]; L = math.hypot(bx - ax, by - ay)
                    if L < 1e-9: continue
                    nx, ny = -(by - ay)/L, (bx - ax)/L
                    try:                                                      # put the size outside its shape
                        ids = {i for cj in chain_from(geo, ci) for i in geo.C[cj]["p"]}
                        if len(ids) > 2:
                            mx_, my_ = np.mean([geo.P[i] for i in ids], axis=0)
                            if (mx_ - (ax + bx)/2)*nx + (my_ - (ay + by)/2)*ny > 0: nx, ny = -nx, -ny
                    except Exception: pass
                    out.append(([(ax + bx)/2 + nx*off, (ay + by)/2 + ny*off], flen(L), "len", ci))
                elif ("r", ci) not in have:
                    cx, cy = geo.P[c["p"][0]]
                    if c["k"] == "circle":
                        r = c["r"]; out.append(([cx + r*0.71 + off*0.5, cy + r*0.71 + off*0.5], f"⌀ {flen(2*r)}", "dia", ci))
                    else:
                        a0, a1, r = arc_angles(c, geo.PT); m = (a0 + a1)/2
                        out.append(([cx + (r + off)*math.cos(m), cy + (r + off)*math.sin(m)], f"R {flen(r)}", "rad", ci))
            except Exception: continue
        return out

    def auto_dim_place(s, ci, pos):
        """Click on a measured size: turn it into a real (driving) dimension."""
        k = s.sk_dim_build([("c", ci)], pos)
        if k: k["pos"] = list(pos); s.dim_place.emit(k)

    def sk_labels(s, labels):
        if s.sk_opts["dims"]:                                                   # every visible sketch shows its sizes
            for si, osk in enumerate(s.sketches):
                if not (osk.visible or osk is s.esk): continue
                editing = osk is s.esk and not s.sk_drag
                for pos, text, kind, ci in s.auto_dims(osk.geo):
                    labels.append((osk.plane.w(V(pos[0], pos[1], 0)), text, False, (0, 0), ("auto", si, ci, tuple(pos))))
                if osk is s.esk and s.sk_prev:                                  # live sizes of the shape being drawn
                    pv = s.sk_prev; g2 = pv["geo"]
                    tmp = Geo(g2.P, g2.C[pv["n0"]:], [])
                    for pos, text, kind, ci in s.auto_dims(tmp):
                        if s.tool == "circle" and kind == "dia": text = f"R {flen(tmp.C[ci]['r'])}"
                        labels.append((osk.plane.w(V(pos[0], pos[1], 0)), text, True, (0, 0), ("live",)))
                    if len(pv["lines"]) == 2:
                        (ax, ay), (bx, by) = pv["lines"]; L = math.hypot(bx - ax, by - ay)
                        if L > 1e-9: labels.append((osk.plane.w(V((ax + bx)/2, (ay + by)/2, 0)), flen(L), True, (0, -14), ("live",)))
                if osk is s.esk and s.tp3 and s.sk_cursor3:
                    a = s.tp3[-1]; b = s.sk_cursor3[:3]; L = math.dist(a, b)
                    if L > 1e-9: labels.append((osk.plane.w(V(*[(x + y)/2 for x, y in zip(a, b)])), flen(L), True, (0, -14), ("live",)))
            for si, osk in enumerate(s.sketches):
                if osk is s.esk or not osk.visible: continue
                for ki, k in enumerate(osk.geo.K):
                    if k["t"] not in DIM_TYPES: continue
                    try: L = k.get("pos") or s.dim_default_pos(osk.geo, k)
                    except Exception: continue
                    labels.append((osk.plane.w(V(L[0], L[1], 0)), s.dim_text(osk.geo, k), False, (0, 0), ("odim", si, ki)))
        sk = s.esk
        if sk is None or s.cmd: s.glyphs = []; return
        geo = sk.geo; pl = sk.plane
        if s.sk_opts["dims"]:
            for ki, k in enumerate(geo.K):
                if k["t"] not in DIM_TYPES: continue
                try: L = k.get("pos") or s.dim_default_pos(geo, k)
                except Exception: continue
                labels.append((pl.w(V(L[0], L[1], 0)), s.dim_text(geo, k), not k.get("drv"), (0, 0), ("dim", ki)))
            if s.dim_prev:
                L = s.dim_prev["pos"]; labels.append((pl.w(V(L[0], L[1], 0)), s.dim_text(geo, s.dim_prev).strip("()"), True, (0, 0), ("dimpre",)))
        s.glyphs = []
        if not s.sk_opts["cons"] or s.mvp is None: return
        anchors = []
        sy = SkSys(geo); x = sy.x
        def mid_of(ci):
            A = geo_curve_pts(geo, ci)
            if not A or not len(A[0]): return None
            A = A[0]; return A[len(A)//2]
        for ki, k in enumerate(geo.K):
            t = k["t"]
            if t in DIM_TYPES: continue
            spots = []
            try:
                if t in ("hor", "ver", "fix", "par", "perp", "col", "eq", "tan", "smooth", "sym"):
                    if t in ("tan", "smooth") and k.get("j"): spots = [geo.P[k["j"][0]]]
                    else:
                        for r in k["e"][:2] if t == "sym" else k["e"]:
                            spots.append(geo.P[r[1]] if r[0] == "p" else mid_of(r[1]))
                elif t in ("coin", "mid"): spots = [geo.P[k["e"][0][1]]]
                elif t == "conc": spots = [geo.P[crv_center(geo.C[k["e"][0][1]])] if k["e"][0][0] == "c" else geo.P[k["e"][0][1]]]
            except Exception: continue
            for sp in spots:
                if sp is None: continue
                anchors.append((sp, GLYPH.get(t, "?"), ki, k.get("off")))
        if not anchors: return
        xy_, ok = s.sk_proj(sk, np.array([a[0] for a in anchors], float)); used = {}; seen = set()
        for (sp, sym, ki, off), c, good in zip(anchors, xy_, ok):
            if not good: continue
            pt = s.to_logical(c); key = (round(pt.x()/14), round(pt.y()/14))
            if (key, sym) in seen: continue
            seen.add((key, sym)); n = used.get(key, 0); used[key] = n + 1
            r = C.QRectF(pt.x() + 9 + n*17, pt.y() + 6, 16, 15)
            s.glyphs.append(dict(rect=r, sym=sym, ki=ki, off=bool(off)))

    def sk_context(s, p):
        """Right-click menu in the sketch."""
        it = s.sk_pick(p); geo = s.esk.geo; m = W.QMenu(s)
        if it and it not in s.sk_sel: s.sk_sel = {it}
        sel = set(s.sk_sel)
        if sel:
            m.addAction("Delete", s.sk_delete_sel)
            cs = [i[1] for i in sel if i[0] == "c"]
            if cs:
                m.addAction("Normal / Construction  (X)", lambda: s.sk_toggle_flag("cons"))
                m.addAction("Centerline", lambda: s.sk_toggle_flag("cl"))
                m.addAction("Fix / Unfix", lambda: s.sk_con_from_sel("fix"))
                if len(cs) == 1 and geo.C[cs[0]]["k"] == "text":
                    m.addAction("Edit Text...", lambda: s.text_dialog.emit(dict(edit=cs[0])))
                    m.addAction("Explode Text", lambda: s.sk_apply(lambda g: geo_explode_text(g, cs[0])))
                m.addAction("Select chain", lambda: (setattr(s, "sk_sel", {("c", c) for c in chain_from(geo, cs[0])}), s.update()))
            ks = [i[1] for i in sel if i[0] == "k"]
            if len(ks) == 1 and ks[0] < len(geo.K):
                k = geo.K[ks[0]]
                if k["t"] in DIM_TYPES:
                    m.addAction("Edit dimension...", lambda: s.dim_edit.emit(ks[0]))
                    m.addAction("Make driving" if k.get("drv") else "Make driven (reference)", lambda: s.sk_set_dim(ks[0], None, not k.get("drv")))
                else: m.addAction("Unsuppress" if k.get("off") else "Suppress", lambda: s.sk_con_suppress(ks[0]))
            ps = [i[1] for i in sel if i[0] == "p"]
            if len(ps) == 1:
                for ci, c in enumerate(geo.C):
                    if c["k"] == "spline" and ps[0] in c["p"]:
                        j = c["p"].index(ps[0])
                        m.addAction("Show tangent handle", lambda ci=ci, j=j: s.sk_apply(lambda g: spline_handle(g, ci, j), solve=False))
                        break
            m.addSeparator()
        m.addAction("Look At", lambda: s.look_at_plane(s.esk.plane.n))
        m.addAction("Finish Sketch", s.sk_finish)
        m.exec(s.mapToGlobal(p.toPoint())); s.update()

    # ---------------- sketch commands (offset, mirror, patterns, move, scale, project, intersect) ----------------
    def sk_local_pt(s, sk, r):
        q = sk.plane.l(s.ref_point(r)); return q.x, q.y

    def sk_local_dir(s, sk, r):
        if r[0] == "curve" and r[1] == s.sketches.index(sk):
            c = sk.geo.C[r[2]]
            if c["k"] != "line": raise RuntimeError("pick a line for the direction")
            a, b = c["p"]; dx, dy = sk.geo.P[b][0] - sk.geo.P[a][0], sk.geo.P[b][1] - sk.geo.P[a][1]
        else:
            p, d = s.ref_axis(r); dx, dy = d.dot(sk.plane.u), d.dot(sk.plane.v)
        L = math.hypot(dx, dy)
        if L < 1e-9: raise RuntimeError("that direction is perpendicular to the sketch")
        return dx/L, dy/L

    def exec_skmod(s, op):
        si = op["sk"]; sk = s.sketches[si]; g = sk.geo.copy(); fn = op["fn"]
        def mine(key, kind="curve"):
            x = op.get(key) or []
            if isinstance(x, tuple): x = [x]
            return [r[2] for r in x if r[0] == kind and r[1] == si]
        if fn == "offset":
            cs = mine("curves")
            if not cs: raise RuntimeError("select the curves to offset")
            chain = []
            for ci in cs:
                for c in (chain_from(g, ci) if op.get("chain", True) else [ci]):
                    if c not in chain: chain.append(c)
            geo_offset(g, chain, op["d"]*(-1 if op.get("flip") else 1), sk.plane)
        elif fn == "mirror":
            cs, ps = mine("objs"), mine("objs", "spoint"); ln = mine("line")
            if not ln or g.C[ln[0]]["k"] != "line": raise RuntimeError("pick a line of this sketch as the mirror line")
            geo_mirror(g, [c for c in cs if c != ln[0]], ln[0], ps)
        elif fn == "cpattern":
            cs = mine("objs")
            if not cs: raise RuntimeError("select the curves to pattern")
            cx, cy = s.sk_local_pt(sk, op["center"]); geo_circ_pattern(g, cs, cx, cy, int(op["n"]), op["ang"], op.get("sym", False))
        elif fn == "rpattern":
            cs = mine("objs")
            if not cs: raise RuntimeError("select the curves to pattern")
            d1 = s.sk_local_dir(sk, op["d1"]); d2 = s.sk_local_dir(sk, op["d2"]) if op.get("d2") else (-d1[1], d1[0])
            s1 = op["s1"] if op.get("mode1", "spacing") == "spacing" else op["s1"]/max(op["n1"] - 1, 1)
            s2 = op["s2"] if op.get("mode2", "spacing") == "spacing" else op["s2"]/max(op["n2"] - 1, 1)
            geo_rect_pattern(g, cs, d1, int(op["n1"]), s1, d2, int(op["n2"]), s2, op.get("sym1", False), op.get("sym2", False))
        elif fn in ("move", "scale"):
            cs, ps = mine("objs"), mine("objs", "spoint")
            if not cs and not ps: raise RuntimeError("select what to move")
            if fn == "scale":
                bx, by = s.sk_local_pt(sk, op["base"]); M = mat_scale(bx, by, op["k"])
            elif op.get("mode") == "ptp":
                ax, ay = s.sk_local_pt(sk, op["p0"]); bx, by = s.sk_local_pt(sk, op["p1"]); M = mat_move(bx - ax, by - ay)
            else:
                cx, cy = s.sk_local_pt(sk, op["pivot"]) if op.get("pivot") else (0.0, 0.0)
                R = mat_rot(cx, cy, op.get("ang", 0.0)); M = [[R[0][0], R[0][1], R[0][2] + op.get("dx", 0.0)], [R[1][0], R[1][1], R[1][2] + op.get("dy", 0.0)]]
            geo_transform(g, cs, M, op.get("copy", False), ps)
        elif fn in ("project", "intersect"):
            pln = sk_pln(sk.plane); pm = {}
            for r in op.get("refs") or []:
                k = r[0]
                if fn == "intersect":
                    shp = s.bodies[r[1]].shape if k == "body" else s.bodies[r[1]].faces[r[2]] if k == "face" else None
                    if shp is None: continue
                    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
                    sec = BRepAlgoAPI_Section(shp, pln); sec.Build()
                    for e in subshapes(sec.Shape(), TopAbs_EDGE):
                        gg = edge_to_g2d(e, pln, False)
                        if gg is not None: add_g2d(g, gg, pm, lock=True, cons=bool(op.get("cons")))
                    continue
                if k == "edge":
                    gg = edge_to_g2d(s.bodies[r[1]].eds[r[2]], pln)
                    if gg is not None: add_g2d(g, gg, pm, src=["edge", r[1], r[2]], lock=True, cons=bool(op.get("cons")))
                elif k == "face":
                    b = s.bodies[r[1]]
                    for ei in b.face_edges(r[2]):
                        gg = edge_to_g2d(b.eds[ei], pln)
                        if gg is not None: add_g2d(g, gg, pm, src=["edge", r[1], ei], lock=True, cons=bool(op.get("cons")))
                elif k == "vertex":
                    q = sk.plane.l(V(*s.bodies[r[1]].verts[r[2]])); i = g.add_pt(q.x, q.y); g.add("point", [i], src=["vertex", r[1], r[2]], lock=True)
                elif k == "body":
                    for gg in hlr_outline_2d(s.bodies[r[1]].shape, sk.plane, op.get("sil", True)):
                        add_g2d(g, gg, pm, lock=True, cons=bool(op.get("cons")))
                elif k == "curve" and r[1] != si:
                    for e in crv_edges(s.sketches[r[1]].geo, r[2], s.sketches[r[1]].plane):
                        gg = edge_to_g2d(e, pln)
                        if gg is not None: add_g2d(g, gg, pm, lock=True, cons=bool(op.get("cons")))
        elif fn == "include":
            for r in op.get("refs") or []:
                if r[0] != "edge": continue
                g.add("inc3", [], src=["edge", r[1], r[2]], lock=True, qs=[edge_local_pts(s.bodies[r[1]].eds[r[2]], sk.plane)])
        elif fn == "projsurf":
            cs = mine("curves"); faces = [r for r in op.get("faces") or [] if r[0] == "face"]
            if not cs or not faces: raise RuntimeError("pick sketch curves and the faces to project onto")
            for ci in cs:
                qs = project_to_faces(crv_edges(g, ci, sk.plane), [s.ref_face(f) for f in faces], sk.plane)
                if qs: g.add("proj3", [], srcc=ci, faces=[list(f) for f in faces], qs=qs)
            if not any(c["k"] == "proj3" for c in g.C[len(sk.geo.C):]): raise RuntimeError("the curves don't land on those faces")
        else: raise RuntimeError(f"unknown sketch command {fn}")
        g.touch(); geo_solve(g); sk.geo = g

def hlr_outline_2d(shape, plane, sil_only=True):
    """Visible outline of a body seen along the sketch normal, as 2D curves in sketch coordinates."""
    algo = HLRBRep_Algo(); algo.Add(shape)
    ax = gp_Ax2(pnt(plane.o), gdir(plane.n), gdir(plane.u)); algo.Projector(HLRAlgo_Projector(ax)); algo.Update(); algo.Hide()
    hl = HLRBRep_HLRToShape(algo); out = []
    xy = gp_Pln(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)))
    comps = [hl.OutLineVCompound(), hl.VCompound()] if not sil_only else [hl.OutLineVCompound(), hl.VCompound()]
    for comp in comps:
        if comp is None or comp.IsNull(): continue
        try: BRepLib.BuildCurves3d_s(comp)
        except Exception: pass
        for e in subshapes(comp, TopAbs_EDGE):
            try:
                g = edge_to_g2d(e, xy, False)
                if g is not None: out.append(g)
            except Exception: pass
    return out


# ---------------------------------------------------------------------------------------------------------------
#  Inspect & view (0.7): measure, section analysis, zebra / draft / curvature, combs, centre of mass, interference
#  display, visual styles, orthographic camera, named views, selection filters + window selection, body display
# ---------------------------------------------------------------------------------------------------------------
from OCP.BRepLProp import BRepLProp_SLProps, BRepLProp_CLProps
VSTYLES = [("Shaded with visible edges", "edges"), ("Shaded", "shaded"), ("Shaded with hidden edges", "hidden"),
           ("Wireframe", "wire"), ("Wireframe with hidden edges", "wire_hidden")]
ANALYSIS_OPS = ("section", "zebra", "draftan", "curvmap", "comb", "interference")
ANA_NAMES = {"section": "Section", "zebra": "Zebra", "draftan": "Draft", "curvmap": "Curvature map", "comb": "Curvature comb"}
MEAS_COL = (0.92, 0.45, 0.10)

def farea(mm2, nd=None):
    u = DISPLAY["unit"]; nd = nd if nd is not None else (2 if u == "mm" else 4); return f"{fmt(mm2/UNITS[u]**2, nd)} {u}²"
def fvol(mm3, nd=None):
    u = DISPLAY["unit"]; nd = nd if nd is not None else (2 if u == "mm" else 4); return f"{fmt(mm3/UNITS[u]**3, nd)} {u}³"

def _hatch_stipple():
    """32x32 polygon-stipple bitmap of 45-degree hatch lines (classic section hatching)."""
    rows = []
    for y in range(32):
        bits = 0
        for x in range(32):
            if (x + y) % 8 in (0, 1): bits |= 1 << (31 - x)
        rows.append(bits.to_bytes(4, "big"))
    return b"".join(rows)
HATCH = _hatch_stipple()

def heat(t):
    """0..1 -> blue, cyan, green, yellow, red."""
    t = min(1.0, max(0.0, t)); stops = [(0.15, 0.3, 0.95), (0.1, 0.8, 0.9), (0.2, 0.8, 0.25), (0.98, 0.85, 0.15), (0.92, 0.2, 0.15)]
    x = t*(len(stops) - 1); i = min(int(x), len(stops) - 2); f = x - i
    a, b = stops[i], stops[i + 1]; return tuple(a[k] + (b[k] - a[k])*f for k in range(3))

def body_curvatures(b):
    """Per-triangle-vertex curvature magnitude (1/mm) of a body, in the same order as Body.tv."""
    out = []
    if getattr(b, "mesh", False): return mesh_vertex_curvature(b.tv)
    for f in b.faces:
        m = face_mesh(f)
        if m is None: continue
        P, T = m; vals = np.zeros(len(P))
        try:
            loc = TopLoc_Location(); tri = triangulation(f, loc); srf = BRepAdaptor_Surface(f)
            if srf.GetType() != GeomAbs_Plane and tri.HasUVNodes():
                for i in range(1, tri.NbNodes() + 1):
                    uv = tri.UVNode(i); pr = BRepLProp_SLProps(srf, uv.X(), uv.Y(), 2, 1e-6)
                    if pr.IsCurvatureDefined(): vals[i - 1] = max(abs(pr.MaxCurvature()), abs(pr.MinCurvature()))
        except Exception: pass
        out.append(vals[T])
    return np.concatenate(out) if out else np.zeros((0, 3))

class InspectMixin:
    def insp_init(s):
        s.ortho, s.vstyle, s.views, s.home_cam = False, "edges", {}, None
        s.analyses, s.ana_prev, s._anacache, s._zebra_tex = [], None, {}, None
        s.tl_groups, s.tl_ver, s.last_fails = [], 0, []
        s.drawing_doc = None; s.disp_off = {}; s.storyboard = []; s.cam_init()
        s.meas, s.meas_cb, s.combs, s.com_marks, s.interf = None, None, [], False, []
        s.sel_filter, s.box, s.box_start, s.msel = {"bodies", "faces", "edges", "profiles"}, None, None, set()
        s.origin_vis = {"XY": False, "XZ": False, "YZ": False, "axes": False}
        s.nav_preset, s.zoom_invert = "fission", False

    def insp_reset(s):
        s.analyses, s.ana_prev, s._anacache, s.views, s.home_cam, s.interf, s.msel = [], None, {}, {}, None, [], set()
        s.tl_groups = []; s.drawing_doc = None; s.disp_off = {}; s.storyboard = []; s.cam_init()
        if s.meas is not None: s.meas["picks"] = []

    # ---------------- camera ----------------
    def apply_projection(s, vp):
        asp = vp[2]/max(vp[3], 1)
        if s.ortho:
            h = s.dist*math.tan(math.radians(22.5)); glOrtho(-h*asp, h*asp, -h, h, -s.dist*200, s.dist*200)
        else: gluPerspective(45, asp, s.dist/50, s.dist*2000)

    def cam_state(s): return [s.yaw, s.pitch, s.dist, list(s.target.t()), s.ortho]
    def go_cam(s, c):
        s.target, s.dist = V(*c[3]), c[2]
        if len(c) > 4: s.ortho = bool(c[4])
        s.animate_to(c[0], c[1])

    def look_at_selection(s):
        """Look straight at the selected face (or the active sketch's plane)."""
        if s.sel_face:
            bi, fid = s.sel_face[:2]; b = s.bodies[bi]; n = b.face_normal(fid)
            if n is None:
                c = V(*b.tv[b.tri_face == fid].reshape(-1, 3).mean(0)); n = normal_at(b.faces[fid], c)[1]
            s.target = V(*b.tv[b.tri_face == fid].reshape(-1, 3).mean(0)); s.look_at_plane(n); return True
        sk = s.esk if s.skedit else s.active
        if sk: s.look_at_plane(sk.plane.n); return True
        return False

    # ---------------- bodies: styles, opacity, analyses, sections ----------------
    def active_analyses(s):
        ed = s.ana_prev.get("_edit") if s.ana_prev else None
        out = [a for i, a in enumerate(s.analyses) if a.get("visible", True) and i != ed]
        return out + ([s.ana_prev] if s.ana_prev else [])

    def analysis_preview(s, op):
        """Live preview while an Inspect command's dialog is open (None = stop)."""
        if op is None or op.get("t") == "interference": s.ana_prev = None; s.update(); return
        a = {k: x for k, x in op.items() if k != "icon"}; a["_edit"] = getattr(s.window(), "_ana_edit", None)
        s.ana_prev = a; s.update()

    def section_planes(s):
        out = []
        for a in s.active_analyses():
            if a["t"] != "section" or not a.get("plane"): continue
            try:
                pl = s.ref_plane(a["plane"]); n = pl.n*(-1 if a.get("flip") else 1); o = pl.o + pl.n*a.get("d", 0.0)
                out.append((o, n.normalize(), a))
            except Exception: pass
        return out[:4]

    def enable_section_clips(s, skip=None):
        for k, (o, n, a) in enumerate(s.section_planes()):
            if k == skip: continue
            glClipPlane(GL_CLIP_PLANE1 + k, (-n.x, -n.y, -n.z, n.dot(o))); glEnable(GL_CLIP_PLANE1 + k)
    def disable_section_clips(s):
        for k in range(4): glDisable(GL_CLIP_PLANE1 + k)

    def ana_targets(s, a, bi):
        ids = a.get("bodies") or []
        return not ids or bi in ids

    def body_colors(s, bi, b):
        """Per-vertex colours from the topmost colour analysis that covers this body (or None)."""
        for a in reversed(s.active_analyses()):
            if a["t"] not in ("draftan", "curvmap") or not s.ana_targets(a, bi): continue
            if a["t"] == "draftan":
                try: d = s.ref_axis(a["dir"][0])[1] if a.get("dir") else V(0, 0, 1)
                except Exception:
                    try: d = s.ref_plane(a["dir"][0]).n
                    except Exception: d = V(0, 0, 1)
                d = d.normalize()*(-1 if a.get("flip") else 1); ang = a.get("angle", 1.0)
                key = ("draft", id(b), d.t(), ang)
                if key not in s._anacache:
                    c = np.clip(b.vn.reshape(-1, 3) @ np.array(d.t()), -1, 1); dr = 90 - np.degrees(np.arccos(c))   # draft angle
                    col = np.empty((len(dr), 3), np.float32)
                    col[:] = (0.98, 0.84, 0.18)                                                            # not enough draft
                    col[dr >= ang] = (0.25, 0.75, 0.30); col[dr <= -ang] = (0.88, 0.25, 0.20)
                    s._anacache[key] = col
                return s._anacache[key]
            key = ("curv", id(b), a.get("scale", "auto"))
            if key not in s._anacache:
                k = body_curvatures(b).reshape(-1)
                if len(k) != len(b.tv)*3: k = np.zeros(len(b.tv)*3)
                hi = float(a.get("max") or 0) or (np.percentile(k[k > 1e-9], 95) if (k > 1e-9).any() else 1.0)
                t = np.log1p(k/max(hi, 1e-9)*9)/math.log(10)
                s._anacache[key] = np.array([heat(x) for x in t], np.float32)
            return s._anacache[key]
        return None

    def zebra_on(s, bi):
        return any(a["t"] == "zebra" and s.ana_targets(a, bi) for a in s.active_analyses())

    def zebra_texture(s):
        if s._zebra_tex is None:
            n = 64; img = np.zeros((n, n, 3), np.uint8)
            yy, xx = np.mgrid[0:n, 0:n]; img[:] = np.where((((xx + yy)//3) % 2)[..., None] == 1, 25, 245).astype(np.uint8)
            t = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, t)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, n, n, 0, GL_RGB, GL_UNSIGNED_BYTE, img.tobytes()); s._zebra_tex = t
        return s._zebra_tex

    def draw_vertex_colored(s, b, col):
        vtx = np.ascontiguousarray(b.tv.reshape(-1, 3), dtype=np.float32); nrm = np.ascontiguousarray(b.vn.reshape(-1, 3), dtype=np.float32)
        glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY); glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vtx); glNormalPointer(GL_FLOAT, 0, nrm); glColorPointer(3, GL_FLOAT, 0, np.ascontiguousarray(col))
        glDrawArrays(GL_TRIANGLES, 0, len(vtx))
        glDisableClientState(GL_COLOR_ARRAY); glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_VERTEX_ARRAY)

    def draw_body_fills(s, shown, moving, olds):
        wire = s.vstyle in ("wire", "wire_hidden")
        s.enable_section_clips()
        if wire:
            if s.vstyle == "wire_hidden":                                    # depth only, so hidden edges can be found
                glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
                for i, b in enumerate(shown):
                    if b.visible and i != moving: glCallList(s.lists(b)[0])
                glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
            s.disable_section_clips(); return
        trans = []
        for i, b in enumerate(shown):
            if not b.visible or i == moving: continue
            op = getattr(b, "opacity", 1.0)
            if op < 0.999: trans.append((i, b, op)); continue
            pushed = s.push_off(i); s.draw_one_body(i, b, olds, 1.0)
            if pushed: glPopMatrix()
        if trans:
            glDepthMask(GL_FALSE)
            for i, b, op in trans:
                pushed = s.push_off(i); s.draw_one_body(i, b, olds, op)
                if pushed: glPopMatrix()
            glDepthMask(GL_TRUE)
        s.disable_section_clips()
        s.draw_section_caps(shown, moving)

    def push_off(s, i):
        d = (getattr(s, "disp_off", None) or {}).get(i)
        if not d or s.preview: return False
        glPushMatrix(); glTranslated(*d); return True

    def draw_one_body(s, i, b, olds, alpha):
        sf = getattr(b, "surface", False)
        if sf: glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE); glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, (0.93, 0.66, 0.38, alpha))
        base = PREVIEW_COL if s.preview and id(b) not in olds else (b.color or BODY)
        col = None if (s.preview and id(b) not in olds) else s.body_colors(i, b)
        if col is not None: s.draw_vertex_colored(b, col)
        elif not (s.preview and id(b) not in olds) and s.zebra_on(i):
            glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, s.zebra_texture()); glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP); glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)
            glEnable(GL_TEXTURE_GEN_S); glEnable(GL_TEXTURE_GEN_T); glColor4f(1, 1, 1, alpha); glCallList(s.lists(b)[0])
            glDisable(GL_TEXTURE_GEN_S); glDisable(GL_TEXTURE_GEN_T); glDisable(GL_TEXTURE_2D)
        else:
            glColor4f(*base, alpha); glCallList(s.lists(b)[0])
        if sf: glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    def draw_section_caps(s, shown, moving):
        planes = s.section_planes()
        if not planes: return
        for k, (o, n, a) in enumerate(planes):
            s.enable_section_clips(skip=k)
            for i, b in enumerate(shown):
                if not b.visible or i == moving or getattr(b, "surface", False): continue
                key = ("cap", id(b), o.t(), n.t())
                if getattr(b, "mesh", False) and key not in s._anacache:
                    try:
                        loops = [L for L in chain_segments(section_segments(b.tv, o, n)) if len(L) > 3]
                        cap = cap_loops(loops, -n) if loops else np.zeros((0, 3, 3))
                        s._anacache[key] = (cap - np.array(n.t())*s.dist*2e-5) if len(cap) else None
                    except Exception: s._anacache[key] = None
                if key not in s._anacache:
                    try:
                        cut = boolean(b.shape, big_plane_face(o, n, s.span()*4), "common")
                        tris = []
                        BRepMesh_IncrementalMesh(cut, b.defl, False, 0.3, True)
                        for f in subshapes(cut, TopAbs_FACE):
                            m = face_mesh(f)
                            if m: tris.append(m[0][m[1]])
                        s._anacache[key] = (np.concatenate(tris) - np.array(n.t())*s.dist*2e-5) if tris else None
                    except Exception: s._anacache[key] = None
                tri = s._anacache[key]
                if tri is None: continue
                col = b.color or BODY
                glEnable(GL_LIGHTING); glNormal3f(*n.t()); glColor3f(*(c*0.85 for c in col)); s.tris_array(tri)
                if a.get("hatch", True):
                    glDisable(GL_LIGHTING); glEnable(GL_POLYGON_STIPPLE); glPolygonStipple(HATCH)
                    glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-1, -1); glColor3f(*(c*0.45 for c in col)); s.tris_array(tri)
                    glDisable(GL_POLYGON_STIPPLE); glPolygonOffset(1, 1)
            s.disable_section_clips()
        glDisable(GL_LIGHTING)

    def draw_body_edges(s, shown, moving):
        if s.vstyle == "shaded": return
        s.enable_section_clips()
        wire = s.vstyle == "wire"
        if wire: glDisable(GL_DEPTH_TEST)
        glColor3f(*EDGE); glLineWidth(1.3)
        for i, b in enumerate(shown):
            if b.visible and i != moving:
                pushed = s.push_off(i); glCallList(s.lists(b)[1])
                if pushed: glPopMatrix()
        if wire: glEnable(GL_DEPTH_TEST)
        if s.vstyle in ("hidden", "wire_hidden"):                            # hidden edges: dashed, behind the surfaces
            glDepthFunc(GL_GREATER); glDepthMask(GL_FALSE); glEnable(GL_LINE_STIPPLE); glLineStipple(2, 0x3333)
            glColor4f(0.45, 0.48, 0.52, 0.8); glLineWidth(1.0)
            for i, b in enumerate(shown):
                if b.visible and i != moving: glCallList(s.lists(b)[1])
            glDisable(GL_LINE_STIPPLE); glDepthMask(GL_TRUE); glDepthFunc(GL_LEQUAL)
        s.disable_section_clips()

    # ---------------- overlays drawn in 3D after the model ----------------
    def draw_inspect_3d(s):
        glDisable(GL_LIGHTING)
        for k, (o, n, a) in enumerate(s.section_planes()):                  # the section plane's outline
            u = V(1, 0, 0) if abs(n.x) < 0.9 else V(0, 1, 0); u = (u - n*u.dot(n)).normalize(); w = n.cross(u); S = s.span()*0.6
            glColor4f(*SEL, 0.9); glLineWidth(1.4); glEnable(GL_LINE_STIPPLE); glLineStipple(3, 0xAAAA); glBegin(GL_LINE_LOOP)
            for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)): glVertex3f(*(o + u*(x*S) + w*(y*S)).t())
            glEnd(); glDisable(GL_LINE_STIPPLE)
        for a in s.active_analyses():                                        # curvature combs
            if a["t"] != "comb": continue
            for r in a.get("edges", []):
                try: e = s.ref_edge(r)
                except Exception: continue
                key = ("comb", r[0], r[1], r[2] if len(r) > 2 else 0, id(s.bodies[r[1]]) if r[0] == "edge" else id(s.sketches[r[1]].geo), a.get("scale", 1.0))
                if key not in s._anacache:
                    c = BRepAdaptor_Curve(e); u0, u1 = c.FirstParameter(), c.LastParameter(); pts, ks = [], []
                    for j in range(81):
                        u = u0 + (u1 - u0)*j/80; pr = BRepLProp_CLProps(c, u, 2, 1e-6); p = vec(c.Value(u))
                        kk = pr.Curvature() if pr.IsTangentDefined() else 0.0; nn = V()
                        if kk > 1e-9:
                            g = gp_Dir(); pr.Normal(g); nn = vec(g)
                        pts.append(p); ks.append((kk, nn))
                    kmax = max([k for k, _ in ks] + [1e-9]); sc = s.span()*0.08*a.get("scale", 1.0)/kmax
                    tips = [p - nn*(k*sc) for p, (k, nn) in zip(pts, ks)]
                    s._anacache[key] = (pts, tips)
                pts, tips = s._anacache[key]
                glColor4f(0.85, 0.25, 0.55, 0.9); glLineWidth(1.0); glBegin(GL_LINES)
                for p, t in zip(pts, tips): glVertex3f(*p.t()); glVertex3f(*t.t())
                glEnd(); glColor4f(0.6, 0.1, 0.4, 0.95); glLineWidth(1.6); s.strip([t.t() for t in tips])
        if s.interf:                                                         # interference volumes
            glDisable(GL_DEPTH_TEST); glColor4f(0.9, 0.12, 0.12, 0.55)
            for b in s.interf: s.tris_array(b.tv)
            glEnable(GL_DEPTH_TEST)
        if s.com_marks:
            glDisable(GL_DEPTH_TEST); r = s.dist*0.012
            for b in s.bodies:
                if not b.visible or getattr(b, "surface", False): continue
                c = s.body_props(b)["com"]
                glColor4f(0.1, 0.1, 0.12, 1); glPointSize(10); glBegin(GL_POINTS); glVertex3f(*c.t()); glEnd()
                glColor4f(1, 1, 1, 1); glPointSize(5); glBegin(GL_POINTS); glVertex3f(*c.t()); glEnd()
                glLineWidth(1.5); glBegin(GL_LINES)
                for ax, col in ((V(1, 0, 0), (0.85, 0.2, 0.2)), (V(0, 1, 0), (0.2, 0.65, 0.25)), (V(0, 0, 1), (0.2, 0.4, 0.9))):
                    glColor3f(*col); glVertex3f(*(c - ax*r).t()); glVertex3f(*(c + ax*r).t())
                glEnd()
            glEnable(GL_DEPTH_TEST)
        S = s.plane_size()*1.5
        for name, pl, col in ORIGIN_PLANES:                                  # origin planes switched on in the browser
            if not s.origin_vis.get(name): continue
            glPushMatrix(); glMultMatrixd(pl.gl()); glDepthMask(GL_FALSE); glColor4f(*col, 0.12); glBegin(GL_QUADS)
            for x, y in ((0, 0), (S, 0), (S, S), (0, S)): glVertex3f(x, y, 0)
            glEnd(); glColor4f(*col, 0.7); glLineWidth(1.2); glBegin(GL_LINE_LOOP)
            for x, y in ((0, 0), (S, 0), (S, S), (0, S)): glVertex3f(x, y, 0)
            glEnd(); glDepthMask(GL_TRUE); glPopMatrix()
        if s.origin_vis.get("axes"):
            L = s.plane_size()*2; glLineWidth(1.6); glBegin(GL_LINES)
            for ax, col in ((V(1, 0, 0), (0.85, 0.2, 0.2)), (V(0, 1, 0), (0.2, 0.65, 0.25)), (V(0, 0, 1), (0.2, 0.4, 0.9))):
                glColor3f(*col); glVertex3f(0, 0, 0); glVertex3f(*(ax*L).t())
            glEnd()
        for bi in sorted(s.msel):                                            # bodies picked by window / ctrl-click
            if bi < len(s.bodies) and s.bodies[bi].visible:
                glDepthMask(GL_FALSE); glColor4f(*HILITE, 0.16); s.tris_array(s.bodies[bi].tv); glDepthMask(GL_TRUE)
        if s.meas is not None: s.draw_measure()

    # ---------------- measure ----------------
    def meas_shape(s, r):
        k = r[0]
        if k == "vertex": return BRepBuilderAPI_MakeVertex(pnt(V(*s.bodies[r[1]].verts[r[2]]))).Vertex()
        if k == "edge": return s.bodies[r[1]].eds[r[2]]
        if k == "face" and getattr(s.bodies[r[1]], "mesh", False): return s.bodies[r[1]].shape
        if k == "face": return s.bodies[r[1]].faces[r[2]]
        if k == "body": return s.bodies[r[1]].shape
        if k in ("spoint", "origin", "cpoint"): return BRepBuilderAPI_MakeVertex(pnt(s.ref_point(r))).Vertex()
        if k == "curve": return s.ref_edge(r)
        raise RuntimeError("can't measure that")

    def meas_pick(s, p):
        kind = s.meas.get("kind", "geom")
        if kind == "body":
            hit = s.pick_face(p); return ("body", hit[0]) if hit else None
        k = s.devicePixelRatioF(); px, py = s.mouse_dev(p); hit = s.pick_face(p); best = None
        for bi, b in enumerate(s.bodies):                                    # vertices first (within 9 px, not hidden)
            if not b.visible or not len(b.verts) or not getattr(b, "selectable", True): continue
            xy, ok = s.project(b.verts); d = np.hypot(xy[:, 0] - px, xy[:, 1] - py); d[~ok] = np.inf; j = int(np.argmin(d))
            if d[j] < 9*k and (best is None or d[j] < best[0]):
                if hit is None or (V(*b.verts[j]) - s.eye()).Length <= (s.ray(p)[1]*hit[3] + s.ray(p)[0] - s.eye()).Length + s.dist*0.01 or s.ortho:
                    best = (d[j], ("vertex", bi, j))
        if best: return best[1]
        e = s.edge_near(p)
        if e: return ("edge",) + tuple(e)
        r = s.pick_ref(p, {"point", "curve"})
        if r: return r
        return ("face", hit[0], hit[1]) if hit else None

    def meas_point(s, r):
        """A representative point for labels / centre-to-centre distance."""
        if r[0] == "vertex": return V(*s.bodies[r[1]].verts[r[2]])
        sh = s.meas_shape(r)
        if r[0] == "edge" or r[0] == "curve":
            c = BRepAdaptor_Curve(sh)
            if c.GetType() == GeomAbs_Circle: return vec(c.Circle().Location())
        return centroid(sh)

    def meas_dir(s, r):
        """A direction for angle measurements: straight edges, flat faces (normal), cylinders (axis)."""
        try:
            if r[0] in ("edge", "curve"):
                c = BRepAdaptor_Curve(s.meas_shape(r))
                if c.GetType() == GeomAbs_Line: return vec(c.Line().Direction()), "line"
                if c.GetType() == GeomAbs_Circle: return vec(c.Circle().Axis().Direction()), "axis"
            if r[0] == "face":
                b = s.bodies[r[1]]; n = b.face_normal(r[2])
                if n is not None: return n, "plane"
                cyl = b.face_cylinder(r[2])
                if cyl: return cyl[1], "axis"
        except Exception: pass
        return None, None

    def meas_compute(s):
        m = s.meas; P = m["picks"]; out = []; m["line"] = None
        try:
            if len(P) >= 1:
                r = P[0]; sh = s.meas_shape(r)
                if r[0] == "vertex" or r[0] in ("spoint", "origin", "cpoint"):
                    q = s.meas_point(r); out.append(("Point", f"X {flen(q.x)}   Y {flen(q.y)}   Z {flen(q.z)}"))
                elif r[0] in ("edge", "curve"):
                    c = BRepAdaptor_Curve(sh); p = GProp_GProps(); linear_props(sh, p); out.append(("Length", flen(p.Mass())))
                    if c.GetType() == GeomAbs_Circle: out.append(("Radius", flen(c.Circle().Radius())))
                elif r[0] == "face":
                    out.append(("Area", farea(face_area(sh))))
                    cyl = s.bodies[r[1]].face_cylinder(r[2])
                    if cyl: out.append(("Radius", flen(cyl[3])))
                elif r[0] == "body":
                    pr = s.body_props(s.bodies[r[1]]); out.append(("Volume", fvol(pr["volume"]))); out.append(("Area", farea(pr["area"])))
            if len(P) >= 2:
                a, b = P[0], P[1]; A, B = s.meas_shape(a), s.meas_shape(b)
                if len(P) == 2: out = []
                mode = m.get("dist", "min")
                if mode == "centre": p1, p2 = s.meas_point(a), s.meas_point(b)
                else:
                    d = BRepExtrema_DistShapeShape(A, B); d.Perform()
                    if not d.IsDone() or not d.NbSolution(): raise RuntimeError("no distance found")
                    best = 0 if mode == "min" else 0
                    p1, p2 = vec(d.PointOnShape1(1)), vec(d.PointOnShape2(1))
                    if mode == "max":                                        # farthest pair of sample points
                        sa = s.meas_samples(A); sb = s.meas_samples(B)
                        D = np.linalg.norm(sa[:, None, :] - sb[None, :, :], axis=2); i, j = np.unravel_index(int(np.argmax(D)), D.shape)
                        p1, p2 = V(*sa[i]), V(*sb[j])
                dv = p2 - p1; m["line"] = (p1, p2)
                out.append(("Distance", flen(dv.Length)))
                out.append(("ΔX ΔY ΔZ", f"{flen(abs(dv.x))}   {flen(abs(dv.y))}   {flen(abs(dv.z))}"))
                (da, ka), (db, kb) = s.meas_dir(a), s.meas_dir(b)
                if da is not None and db is not None:
                    c = max(-1.0, min(1.0, abs(da.dot(db)))); ang = math.degrees(math.acos(c))
                    if ka == "plane" and kb == "plane": out.append(("Angle", f"{fmt(ang, m.get('prec', 2))}°  (between faces)"))
                    elif "plane" in (ka, kb) and ("line" in (ka, kb) or "axis" in (ka, kb)): out.append(("Angle", f"{fmt(90 - ang, m.get('prec', 2))}°"))
                    else: out.append(("Angle", f"{fmt(ang, m.get('prec', 2))}°"))
        except Exception as ex: out.append(("", str(ex)))
        m["out"] = out
        if s.meas_cb: s.meas_cb()
        s.update()

    def meas_samples(s, sh):
        BRepMesh_IncrementalMesh(sh, 0.5, False, 0.3, True); pts = []
        for f in subshapes(sh, TopAbs_FACE) or []:
            mm = face_mesh(f)
            if mm: pts.append(mm[0])
        if not pts:
            for e in subshapes(sh, TopAbs_EDGE) or [sh] if sh.ShapeType() == TopAbs_EDGE else subshapes(sh, TopAbs_EDGE):
                pts.append(edge_pts(e, 0.2))
        if not pts: return np.array([centroid(sh).t()])
        a = np.concatenate(pts)
        return a[:: max(1, len(a)//800)]

    def draw_measure(s):
        m = s.meas
        if s.cmd_hover is not None and not s.cmd: s.draw_ref(s.cmd_hover, HILITE, 0.8, 3.0)
        for r in m["picks"]: s.draw_ref(r, MEAS_COL, 1.0, 4.0)
        if m.get("line"):
            p1, p2 = m["line"]; glDisable(GL_DEPTH_TEST); glColor4f(*MEAS_COL, 1); glLineWidth(2.2)
            glBegin(GL_LINES); glVertex3f(*p1.t()); glVertex3f(*p2.t()); glEnd()
            glPointSize(8); glBegin(GL_POINTS); glVertex3f(*p1.t()); glVertex3f(*p2.t()); glEnd(); glEnable(GL_DEPTH_TEST)

    # ---------------- inspection numbers ----------------
    def body_props(s, b):
        key = ("props", id(b))
        if key in s._anacache: return s._anacache[key]
        if getattr(b, "mesh", False):
            T = b.tv; vol = abs(mesh_volume(T)); P = T.reshape(-1, 3); dens = dict(MATERIALS).get(b.material)
            out = {"area": mesh_area(T), "volume": vol, "com": mesh_centroid(T), "lo": V(*P.min(0)), "hi": V(*P.max(0)), "density": dens,
                   "mass": vol/1000.0*dens if dens else None, "inertia": None, "triangles": len(T)}
            s._anacache[key] = out; return out
        sh = b.shape; out = {}
        p = GProp_GProps(); surface_props(sh, p); out["area"] = p.Mass()
        if getattr(b, "surface", False):
            out["volume"] = 0.0; out["com"] = vec(p.CentreOfMass()); g = p
        else:
            g = GProp_GProps(); volume_props(sh, g); out["volume"] = abs(g.Mass()); out["com"] = vec(g.CentreOfMass())
        lo, hi = bbox(sh); out["lo"], out["hi"] = lo, hi
        dens = dict(MATERIALS).get(b.material)
        out["density"] = dens; out["mass"] = out["volume"]/1000.0*dens if dens else None
        try:
            pp = g.PrincipalProperties(); I1, I2, I3 = pp.Moments()
            k = (dens or 1.0)/1000.0                                      # g/mm^3 -> I in g*mm^2
            out["inertia"] = (I1*k, I2*k, I3*k)
            M = g.MatrixOfInertia(); out["inertia_matrix"] = [[M.Value(i, j)*k for j in (1, 2, 3)] for i in (1, 2, 3)]
        except Exception: out["inertia"] = None
        s._anacache[key] = out; return out

    def interference(s, ids):
        """[(i, j, overlap shape, volume)] for every pair of the given solid bodies that overlap."""
        out = []
        ids = [i for i in ids if i < len(s.bodies) and not getattr(s.bodies[i], "surface", False)
               and not (getattr(s.bodies[i], "mesh", False) and len(s.bodies[i].tv) > MESH_CONVERT_MAX)]
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                i, j = ids[x], ids[y]; a, b = s.bodies[i], s.bodies[j]
                if any(a.hi.t()[k] < b.lo.t()[k] - 1e-6 or b.hi.t()[k] < a.lo.t()[k] - 1e-6 for k in range(3)): continue
                try:
                    c = boolean(a.shape, b.shape, "common"); v = abs(volume(c))
                    if v > 1e-6: out.append((i, j, c, v))
                except Exception: pass
        return out

    # ---------------- selection: filters, window, multi-select ----------------
    def idle(s):
        return not (s.skedit or s.cmd or s.tool or s.ext or s.mv or s.rev or s.calib or s.nav)

    def insp_press(s, e):
        s.box_start = None
        if e.button() != C.Qt.LeftButton: return False
        if s.meas is not None: return True
        if s.idle() and not s.label_at(e.position()) and not s.edge_near(e.position()) and s.pick_face(e.position()) is None \
                and not s.region_under(e.position()):
            s.box_start = e.position()
        return False

    def insp_move(s, e):
        if s.box_start is not None and e.buttons() & C.Qt.LeftButton:
            p = e.position()
            if s.box is not None or (p - s.box_start).manhattanLength() > 5:
                s.box = (s.box_start, p); s.moved = True; s.update(); return True
        if s.meas is not None and not e.buttons():
            r = s.meas_pick(e.position()); s.cmd_hover = r; s.update()
        return False

    def insp_release(s, e):
        if s.box is not None:
            a, b = s.box; s.box = s.box_start = None; s.window_select(a, b, e.modifiers()); s.changed.emit(); s.update(); return True
        s.box_start = None
        if s.meas is not None:
            if e.button() == C.Qt.LeftButton and not s.moved:
                r = s.meas_pick(e.position())
                if r is not None:
                    P = s.meas["picks"]
                    if len(P) >= 2: P.clear()
                    P.append(r); s.meas_compute()
                return True
            return e.button() == C.Qt.LeftButton
        if e.button() == C.Qt.LeftButton and not s.moved and s.idle():
            if e.modifiers() & (C.Qt.ControlModifier | C.Qt.ShiftModifier) and "bodies" in s.sel_filter:
                hit = s.pick_face(e.position())
                if hit and not s.edge_near(e.position()):
                    bi = hit[0]; cur = s.msel | ({s.sel_body} if s.sel_body is not None else set())
                    cur ^= {bi}; s.msel = cur; s.sel_body = min(cur) if cur else None; s.sel_face = None; s.sel.clear()
                    s.msg.emit(f"{len(cur)} bodies selected."); s.changed.emit(); s.update(); return True
            s.msel = set()
            if "faces" not in s.sel_filter or "edges" not in s.sel_filter:
                e2 = s.edge_near(e.position()) if "edges" in s.sel_filter else None
                if e2 is None:
                    hit = s.pick_face(e.position())
                    if hit and "faces" not in s.sel_filter:
                        s.sel.clear(); s.sel_face = None
                        s.sel_body = hit[0] if "bodies" in s.sel_filter else None
                        if s.sel_body is not None: s.msg.emit(f"{s.body_name(hit[0])} selected.")
                        s.changed.emit(); s.update(); return True
                    if not hit and "profiles" not in s.sel_filter:
                        s.sel.clear(); s.sel_face = s.sel_body = None; s.changed.emit(); s.update(); return True
        return False

    def body_name(s, bi):
        b = s.bodies[bi]; return getattr(b, "name", None) or f"Body{bi + 1}"

    def window_select(s, a, b, mods):
        """Drag left-to-right: whatever is fully inside; right-to-left: whatever the box touches."""
        k = s.devicePixelRatioF(); x0, x1 = sorted((a.x()*k, b.x()*k)); cross = b.x() < a.x()
        y0, y1 = sorted(((s.vp[3] - a.y()*k), (s.vp[3] - b.y()*k)))
        def inside(xy, ok):
            m = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1) & ok
            return m.any() if cross else m.all()
        add = bool(mods & (C.Qt.ControlModifier | C.Qt.ShiftModifier))
        if not add: s.sel.clear(); s.msel = set(); s.sel_face = None; s.sel_body = None
        if "bodies" in s.sel_filter:
            for bi, bd in enumerate(s.bodies):
                if not bd.visible or not getattr(bd, "selectable", True) or not len(bd.tv): continue
                pts = bd.tv.reshape(-1, 3)[:: max(1, len(bd.tv)*3//2000)]
                xy, ok = s.project(pts)
                if inside(xy, ok): s.msel.add(bi)
            s.sel_body = min(s.msel) if s.msel else None
            s.msg.emit(f"{len(s.msel)} bodies selected - Delete removes them; commands that take bodies start with them."); return
        if "edges" in s.sel_filter:
            for bi, bd in enumerate(s.bodies):
                if not bd.visible or not getattr(bd, "selectable", True): continue
                for ei, pl in enumerate(bd.edges):
                    if len(pl) and inside(*s.project(pl)): s.sel.add((bi, ei))
            s.msg.emit(f"{len(s.sel)} edges selected.")

    def paint_meas_label(s, p):
        m = s.meas
        if m is None or not m.get("line") or s.mvp is None: return
        p1, p2 = m["line"]; xy, ok = s.project([((p1 + p2)*0.5).t()])
        if not ok[0]: return
        q = s.to_logical(xy[0]); txt = flen((p2 - p1).Length, m.get("prec", 2))
        f = p.font(); f.setPixelSize(12); f.setBold(True); p.setFont(f); fm = G.QFontMetricsF(f)
        r = C.QRectF(q.x() + 8, q.y() - 22, fm.horizontalAdvance(txt) + 14, 20)
        p.setPen(G.QPen(G.QColor(235, 115, 25), 1.5)); p.setBrush(G.QColor(255, 255, 255, 240)); p.drawRoundedRect(r, 3, 3)
        p.setPen(G.QColor("#7a3a08")); p.drawText(r, C.Qt.AlignCenter, txt); f.setBold(False); p.setFont(f)

    def draw_box(s, p):
        if s.box is None: return
        a, b = s.box; r = C.QRectF(a, b).normalized(); cross = b.x() < a.x()
        p.setPen(G.QPen(G.QColor(ACCENT), 1, C.Qt.DashLine if cross else C.Qt.SolidLine)); p.setBrush(G.QColor(6, 150, 215, 30 if cross else 45)); p.drawRect(r)

    def nav_kind(s, b, mods):
        pre = s.nav_preset
        if pre == "solidworks":
            if b & C.Qt.MiddleButton: return "pan" if mods & C.Qt.ControlModifier else "orbit"
            return None
        if pre == "fusion":
            if b & C.Qt.MiddleButton: return "orbit" if mods & C.Qt.ShiftModifier else "pan"
            return None
        if b & C.Qt.RightButton or (b & C.Qt.MiddleButton and mods & C.Qt.ShiftModifier): return "orbit"
        if b & C.Qt.MiddleButton: return "pan"
        return None

    # ---------------- design file ----------------
    def insp_save(s):
        clean = [{k: v for k, v in a.items() if not k.startswith("_")} for a in s.analyses]
        return {"views": s.views, "home": s.home_cam, "analyses": enc(clean), "vstyle": s.vstyle, "ortho": s.ortho,
                "tl_groups": [dict(g, open=False) for g in getattr(s, "tl_groups", [])], "drawing": getattr(s, "drawing_doc", None),
                "story": getattr(s, "storyboard", []), "cam": s.cam if s.cam["ops"] or s.cam["tools"] != DEFAULT_TOOLS else None}
    def insp_load(s, data):
        s.views = data.get("views") or {}; s.home_cam = data.get("home")
        s.tl_groups = [dict(g) for g in data.get("tl_groups") or []]; s.drawing_doc = data.get("drawing"); s.storyboard = data.get("story") or []
        if data.get("cam"):
            s.cam = data["cam"]; s.cam_paths = {}
            try: s.cam_regen()
            except Exception: traceback.print_exc()
        s.analyses = dec(data.get("analyses") or []); s._anacache = {}
        s.vstyle = data.get("vstyle", s.vstyle); s.ortho = bool(data.get("ortho", s.ortho))


# ---------------------------------------------------------------------------------------------------------------
#  Mesh bodies (0.7): triangle meshes from STL / OBJ / 3MF, mesh tools (repair, cut, combine, separate, smooth,
#  reduce, sections), conversion to and from B-rep; plus STEP / IGES / BREP exchange
# ---------------------------------------------------------------------------------------------------------------
import base64, io as _io, struct as _struct, zlib as _zlib
from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCP.IGESControl import IGESControl_Reader, IGESControl_Writer
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.TopoDS import TopoDS_Shape, TopoDS_Shell

def pack_arr(a):
    buf = _io.BytesIO(); np.save(buf, np.asarray(a, np.float32)); return base64.b64encode(_zlib.compress(buf.getvalue(), 6)).decode()
def unpack_arr(s): return np.load(_io.BytesIO(_zlib.decompress(base64.b64decode(s))))
def pack_text(t): return base64.b64encode(_zlib.compress(t.encode("utf-8"), 6)).decode()
def unpack_text(s): return _zlib.decompress(base64.b64decode(s)).decode("utf-8")

def shape_to_brep_text(sh):
    import tempfile as _tf
    fd, p = _tf.mkstemp(suffix=".brep"); os.close(fd)
    try: brep_write(sh, p); return open(p, encoding="utf-8", errors="replace").read()
    finally: os.remove(p)
def brep_text_to_shape(t):
    import tempfile as _tf
    from OCP.BRepTools import BRepTools as _BT
    fd, p = _tf.mkstemp(suffix=".brep"); os.close(fd)
    try:
        open(p, "w", encoding="utf-8").write(t); sh = TopoDS_Shape(); st(_BT, "Read")(sh, p, BRep_Builder()); return sh
    finally: os.remove(p)

# ---------------- mesh algorithms (numpy) ----------------
def weld(tris, tol=None):
    """Shared vertices: (V (M,3), F (N,3) int)."""
    P = np.asarray(tris, float).reshape(-1, 3)
    if not len(P): return np.zeros((0, 3)), np.zeros((0, 3), int)
    span = float(np.ptp(P, 0).max()) or 1.0; tol = tol or span*1e-7
    keys = np.round(P/tol).astype(np.int64)
    _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return P[first], inv.reshape(-1, 3)

def tri_normals(T):
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); L = np.linalg.norm(n, axis=1, keepdims=True)
    return n/np.maximum(L, 1e-300), L[:, 0]/2

def edge_map(F):
    """Undirected edges: (E (k,2) sorted vertex ids, inverse index per face-edge (N,3))."""
    e = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2)
    E, inv = np.unique(np.sort(e, 1), axis=0, return_inverse=True)
    return E, inv.reshape(-1, 3)

def feature_segments(T, angle=35.0):
    """Edges worth drawing on a mesh: open boundaries and creases sharper than angle."""
    if not len(T): return np.zeros((0, 3)), np.zeros((0, 3))
    Vv, F = weld(T); n, _ = tri_normals(T); E, inv = edge_map(F)
    cnt = np.bincount(inv.reshape(-1), minlength=len(E))
    f_of = np.repeat(np.arange(len(F)), 3); order = np.argsort(inv.reshape(-1), kind="stable")
    ei = inv.reshape(-1)[order]; fi = f_of[order]
    start = np.searchsorted(ei, np.arange(len(E)))
    keep = cnt == 1
    two = np.where(cnt == 2)[0]
    if len(two):
        a, b = fi[start[two]], fi[start[two] + 1]
        keep[two] = (n[a]*n[b]).sum(1) < math.cos(math.radians(angle))
    sel = E[keep]
    if len(sel) > 200000: sel = sel[:200000]
    return Vv[sel[:, 0]], Vv[sel[:, 1]]

def mesh_volume(T):
    T = np.asarray(T, float)
    return float(np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2])).sum()/6.0) if len(T) else 0.0
def mesh_area(T): return float(tri_normals(np.asarray(T, float))[1].sum()) if len(T) else 0.0
def mesh_centroid(T):
    T = np.asarray(T, float); v6 = np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2]))
    if abs(v6.sum()) < 1e-12: return V(*T.reshape(-1, 3).mean(0))
    return V(*((T.sum(1)/4.0)*v6[:, None]).sum(0)/v6.sum())

def components(F, nv):
    par = np.arange(nv)
    def find(x):
        r = x
        while par[r] != r: r = par[r]
        while par[x] != r: par[x], x = r, par[x]
        return r
    for a, b, c in F:
        ra, rb, rc = find(a), find(b), find(c)
        if rb != ra: par[rb] = ra
        if rc != ra and find(rc) != ra: par[find(rc)] = ra
    roots = np.array([find(f[0]) for f in F])
    return [np.where(roots == r)[0] for r in np.unique(roots)]

def orient_consistently(F):
    """Flip faces so neighbours agree (BFS over shared edges)."""
    F = F.copy(); E, inv = edge_map(F); nbr = {}
    for fi in range(len(F)):
        for k in range(3): nbr.setdefault(inv[fi, k], []).append(fi)
    seen = np.zeros(len(F), bool)
    def dirs(f): return {(F[f][0], F[f][1]), (F[f][1], F[f][2]), (F[f][2], F[f][0])}
    for s0 in range(len(F)):
        if seen[s0]: continue
        seen[s0] = True; stack = [s0]
        while stack:
            f = stack.pop(); df = dirs(f)
            for k in range(3):
                for g in nbr[inv[f, k]]:
                    if g == f or seen[g]: continue
                    if dirs(g) & df: F[g] = F[g][[0, 2, 1]]                 # same directed edge = opposite winding
                    seen[g] = True; stack.append(g)
    return F

def boundary_loops(F):
    E, inv = edge_map(F); cnt = np.bincount(inv.reshape(-1), minlength=len(E))
    nxt = {}
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (min(a, b), max(a, b))
            idx = np.searchsorted(E[:, 0]*(1 << 31) + E[:, 1], key[0]*(1 << 31) + key[1])
            if idx < len(E) and cnt[idx] == 1: nxt[b] = a                     # walk boundary opposite to face winding
    loops, used = [], set()
    for s0 in list(nxt):
        if s0 in used: continue
        loop = [s0]; used.add(s0); c = nxt.get(s0)
        while c is not None and c != s0 and c not in used and len(loop) < 100000:
            loop.append(c); used.add(c); c = nxt.get(c)
        if c == s0 and len(loop) >= 3: loops.append(loop)
    return loops

def repair_mesh(T, fill_holes=True):
    Vv, F = weld(T); n, area = tri_normals(np.asarray(T, float))
    keep = (area > 1e-14) & (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[keep]
    _, u = np.unique(np.sort(F, 1), axis=0, return_index=True); F = F[np.sort(u)]
    F = orient_consistently(F)
    if fill_holes:
        add = []
        for loop in boundary_loops(F):
            if len(loop) > 400: continue
            c = Vv[loop].mean(0); ci = len(Vv) + len(add)
            add.append(c)
            F = np.vstack([F] + [np.array([[loop[i], loop[(i + 1) % len(loop)], ci]]) for i in range(len(loop))])
        if add: Vv = np.vstack([Vv, np.array(add)])
    T2 = Vv[F]
    if mesh_volume(T2) < 0: T2 = T2[:, [0, 2, 1]]
    return T2

def taubin_smooth(T, iters=10, lam=0.5, mu=-0.53):
    Vv, F = weld(T); E, _ = edge_map(F); P = Vv.copy()
    deg = np.bincount(E.reshape(-1), minlength=len(P)).astype(float)
    fixed = np.zeros(len(P), bool)
    for loop in boundary_loops(F): fixed[loop] = True                           # keep open edges in place
    for _ in range(int(iters)):
        for k in (lam, mu):
            acc = np.zeros_like(P); np.add.at(acc, E[:, 0], P[E[:, 1]]); np.add.at(acc, E[:, 1], P[E[:, 0]])
            lap = acc/np.maximum(deg, 1)[:, None] - P; lap[fixed] = 0; P = P + k*lap
    return P[F]

def cluster_reduce(T, ratio):
    """Vertex-clustering decimation to roughly `ratio` of the triangles."""
    Vv, F = weld(T); target = max(4, int(len(F)*ratio)); span = float(np.ptp(Vv, 0).max()) or 1.0
    lo, hi = span*1e-5, span*0.5; best = Vv[F]
    for _ in range(18):
        cell = math.sqrt(lo*hi)
        key = np.floor((Vv - Vv.min(0))/cell).astype(np.int64)
        _, rep, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.reshape(-1)
        cnt = np.bincount(inv); P = np.zeros((len(cnt), 3)); np.add.at(P, inv, Vv); P /= cnt[:, None]
        G = inv[F]; ok = (G[:, 0] != G[:, 1]) & (G[:, 1] != G[:, 2]) & (G[:, 0] != G[:, 2]); G = G[ok]
        _, u = np.unique(np.sort(G, 1), axis=0, return_index=True); G = G[np.sort(u)]
        if len(G) > target: lo = cell
        else: hi = cell; best = P[G]
        if abs(len(G) - target) < max(10, target*0.02): best = P[G]; break
    return best

def clip_by_plane(T, o, n, keep_below=True):
    """Keep the part of the mesh on one side of a plane. Returns (triangles, cut segments)."""
    T = np.asarray(T, float); nn = np.array(n.t()); d = (T - np.array(o.t())) @ nn
    if not keep_below: d = -d
    out, segs = [], []
    allin = (d <= 0).all(1); out.append(T[allin])
    for t, dd in zip(T[~allin & (d < 0).any(1)], d[~allin & (d < 0).any(1)]):
        poly, cut = [], []
        for i in range(3):
            a, b, da, db = t[i], t[(i + 1) % 3], dd[i], dd[(i + 1) % 3]
            if da <= 0: poly.append(a)
            if (da < 0) != (db < 0) and abs(da - db) > 1e-15:
                q = a + (b - a)*(da/(da - db)); poly.append(q); cut.append(q)
        for k in range(1, len(poly) - 1): out.append(np.array([[poly[0], poly[k], poly[k + 1]]]))
        if len(cut) == 2: segs.append(cut)
    return np.concatenate([x for x in out if len(x)]) if any(len(x) for x in out) else np.zeros((0, 3, 3)), segs

def section_segments(T, o, n):
    T = np.asarray(T, float); nn = np.array(n.t()); d = (T - np.array(o.t())) @ nn; segs = []
    m = (d.min(1) < 0) & (d.max(1) > 0)
    for t, dd in zip(T[m], d[m]):
        pts = []
        for i in range(3):
            a, b, da, db = t[i], t[(i + 1) % 3], dd[i], dd[(i + 1) % 3]
            if (da < 0) != (db < 0): pts.append(a + (b - a)*(da/(da - db)))
        if len(pts) == 2: segs.append(pts)
    return segs

def chain_segments(segs, tol=None):
    """Join loose segments into polylines (closed ones repeat their first point at the end)."""
    if not segs: return []
    P = np.array(segs).reshape(-1, 3); span = float(np.ptp(P, 0).max()) or 1.0; tol = tol or span*1e-6
    key = lambda p: tuple(np.round(p/tol).astype(np.int64))
    adj = {}
    for i, (a, b) in enumerate(segs):
        adj.setdefault(key(a), []).append((i, 1)); adj.setdefault(key(b), []).append((i, 0))
    used = np.zeros(len(segs), bool); out = []
    for s0 in range(len(segs)):
        if used[s0]: continue
        used[s0] = True; line = [segs[s0][0], segs[s0][1]]
        for end in (1, 0):
            while True:
                tip = line[-1] if end else line[0]; nxt = None
                for j, other in adj.get(key(tip), []):
                    if not used[j]: nxt = (j, other); break
                if nxt is None: break
                j, other = nxt; used[j] = True; q = segs[j][other]
                if end: line.append(q)
                else: line.insert(0, q)
        out.append(np.array(line))
    return out

def simplify_polyline(P, tol):
    """Douglas-Peucker."""
    if len(P) < 3: return P
    a, b = P[0], P[-1]; ab = b - a; L = np.linalg.norm(ab)
    d = np.linalg.norm(np.cross(P - a, ab), axis=1)/L if L > 1e-12 else np.linalg.norm(P - a, axis=1)
    i = int(np.argmax(d))
    if d[i] <= tol: return np.array([a, b])
    return np.vstack([simplify_polyline(P[:i + 1], tol)[:-1], simplify_polyline(P[i:], tol)])

def cap_loops(loops, n):
    """Triangles filling closed planar loops (holes handled by nesting) via an OCC face."""
    tris = []
    ws = []
    for L in loops:
        if len(L) < 4: continue
        mk = BRepBuilderAPI_MakePolygon()
        for q in L[:-1]: mk.Add(gp_Pnt(*map(float, q)))
        mk.Close()
        try: ws.append(mk.Wire())
        except Exception: pass
    if not ws: return np.zeros((0, 3, 3))
    faces = [BRepBuilderAPI_MakeFace(w, True).Face() for w in ws]
    areas = [face_area(f) for f in faces]; order = sorted(range(len(ws)), key=lambda i: -areas[i]); placed = []
    for i in order:
        host = None
        for j in placed:
            try:
                if classify_in_face(faces[j], centroid(ws[i])): host = j
            except Exception: pass
        placed.append(i)
        if host is None: continue
    try:
        cut = fuse_all(faces) if len(faces) > 1 else faces[0]
        res = faces[order[0]]
        for i in order[1:]:
            inside = classify_in_face(faces[order[0]], centroid(ws[i]))
            res = boolean(res, faces[i], "cut" if inside else "fuse")
        BRepMesh_IncrementalMesh(res, 0.05, False, 0.3, True)
        for f in subshapes(res, TopAbs_FACE):
            m = face_mesh(f)
            if m:
                tri = m[0][m[1]]
                if (tri_normals(tri)[0].mean(0) @ np.array(n.t())) < 0: tri = tri[:, [0, 2, 1]]
                tris.append(tri)
    except Exception: pass
    return np.concatenate(tris) if tris else np.zeros((0, 3, 3))

def classify_in_face(face, p):
    from OCP.BRepClass import BRepClass_FaceClassifier as _FC
    return _FC(face, pnt(p), 1e-6).State() == TopAbs_IN

def tessellate(shape, defl=0.05, ang=0.3):
    sh = BRepBuilderAPI_Copy(shape).Shape(); BRepMesh_IncrementalMesh(sh, defl, False, ang, True); out = []
    for f in subshapes(sh, TopAbs_FACE):
        m = face_mesh(f)
        if m: out.append(m[0][m[1]])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))

def mesh_to_brep(T, merge_planar=True):
    """Faceted B-rep (one planar face per triangle, shared edges), optionally merging coplanar facets."""
    Vv, F = weld(T)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    if len(F): F = F[tri_normals(Vv[F])[1] > 1e-14]
    verts = [BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, p))).Vertex() for p in Vv]
    E, inv = edge_map(F); edges = []
    for a, b in E:
        try: edges.append(BRepBuilderAPI_MakeEdge(verts[a], verts[b]).Edge())
        except Exception: edges.append(None)
    sh = TopoDS_Shell(); bb = BRep_Builder(); bb.MakeShell(sh)
    for fi, f in enumerate(F):
        if any(edges[inv[fi, k]] is None for k in range(3)): continue
        mw = BRepBuilderAPI_MakeWire()
        for k in range(3):
            e = edges[inv[fi, k]]; a, b = f[k], f[(k + 1) % 3]
            mw.Add(e if E[inv[fi, k]][0] == a else to_edge(e.Reversed()))
        try: bb.Add(sh, BRepBuilderAPI_MakeFace(mw.Wire(), True).Face())
        except Exception: pass
    from OCP.ShapeFix import ShapeFix_Shell
    fx = ShapeFix_Shell(sh); fx.Perform(); res = fx.Shape()
    closed = st(BRep_Tool, "IsClosed")(res)
    if closed:
        try: res = as_solid(res)
        except Exception: pass
    if merge_planar: res = clean(res)
    return res

# ---------------- files ----------------
def read_stl(path):
    data = open(path, "rb").read()
    if len(data) >= 84:
        n = _struct.unpack_from("<I", data, 80)[0]
        if 84 + 50*n == len(data):
            a = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("c", "<u2")]), count=n, offset=84)
            return a["v"].astype(float)
    txt = data.decode("latin-1"); nums = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", txt)
    return np.array(nums, float).reshape(-1, 3, 3)

def read_obj(path):
    Vv, out = [], []
    for line in open(path, encoding="latin-1"):
        if line.startswith("v "): Vv.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            ids = [int(t.split("/")[0]) for t in line.split()[1:]]
            ids = [i - 1 if i > 0 else len(Vv) + i for i in ids]
            for k in range(1, len(ids) - 1): out.append((ids[0], ids[k], ids[k + 1]))
    Vv = np.array(Vv, float); return Vv[np.array(out, int)] if out else np.zeros((0, 3, 3))

def read_3mf(path):
    import xml.etree.ElementTree as ET
    out, scale = [], 1.0
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".model"): continue
            root = ET.fromstring(z.read(name)); scale = {"micron": 0.001, "millimeter": 1, "centimeter": 10, "meter": 1000, "inch": 25.4, "foot": 304.8}.get(root.get("unit", "millimeter"), 1)
            for mesh in root.iter():
                if not mesh.tag.endswith("}mesh") and mesh.tag != "mesh": continue
                vs = [[float(v.get("x")), float(v.get("y")), float(v.get("z"))] for v in mesh.iter() if v.tag.endswith("vertex")]
                ts = [[int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))] for t in mesh.iter() if t.tag.endswith("triangle")]
                if vs and ts: out.append(np.array(vs)[np.array(ts)])
    return (np.concatenate(out) if out else np.zeros((0, 3, 3)))*scale

def read_mesh_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl": return read_stl(path)
    if ext == ".obj": return read_obj(path)
    if ext == ".3mf": return read_3mf(path)
    raise RuntimeError("Fission reads STL, OBJ and 3MF meshes")

def write_stl(path, tris, ascii=False, name="fission"):
    T = np.asarray(tris, float); n, _ = tri_normals(T)
    if ascii:
        with open(path, "w") as f:
            f.write(f"solid {name}\n")
            for t, nn in zip(T, n):
                f.write(f" facet normal {nn[0]:.6e} {nn[1]:.6e} {nn[2]:.6e}\n  outer loop\n")
                for p in t: f.write(f"   vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")
                f.write("  endloop\n endfacet\n")
            f.write(f"endsolid {name}\n")
        return
    rec = np.zeros(len(T), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("c", "<u2")])); rec["n"] = n; rec["v"] = T
    with open(path, "wb") as f: f.write(b"Fission STL".ljust(80, b" ")); f.write(_struct.pack("<I", len(T))); f.write(rec.tobytes())

def read_cad_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".step", ".stp"):
        r = STEPControl_Reader()
        if r.ReadFile(path) != IFSelect_RetDone: raise RuntimeError("couldn't read that STEP file")
        r.TransferRoots(); return r.OneShape()
    if ext in (".iges", ".igs"):
        r = IGESControl_Reader()
        if r.ReadFile(path) != IFSelect_RetDone: raise RuntimeError("couldn't read that IGES file")
        r.TransferRoots(); return r.OneShape()
    if ext == ".brep": return brep_text_to_shape(open(path, encoding="utf-8", errors="replace").read())
    raise RuntimeError("Fission reads STEP, IGES and BREP files")

def write_cad_file(path, shapes):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".step", ".stp"):
        w = STEPControl_Writer()
        for sh in shapes: w.Transfer(sh, STEPControl_AsIs)
        if w.Write(path) != IFSelect_RetDone: raise RuntimeError("couldn't write the STEP file")
    elif ext in (".iges", ".igs"):
        w = IGESControl_Writer("MM", 1)
        for sh in shapes: w.AddShape(sh)
        w.ComputeModel()
        if not w.Write(path): raise RuntimeError("couldn't write the IGES file")
    elif ext == ".brep": brep_write(compound(shapes) if len(shapes) > 1 else shapes[0], path)
    else: raise RuntimeError("unknown CAD format")

def split_imported(sh):
    """Imported shape -> list of (shape, kind): solids, then free shells / faces as surfaces."""
    out = [(x, "solid") for x in subshapes(sh, TopAbs_SOLID)]
    used = ShapeIndex()
    for so, _ in out:
        for f in subshapes(so, TopAbs_FACE): used.add(f)
    free = [f for f in subshapes(sh, TopAbs_FACE) if used.find(f) < 0]
    if free:
        for g in face_groups(free): out.append((sew(g, solid_if_closed=False) if len(g) > 1 else g[0], "surface"))
    return out

class MeshBody(Body):
    """A triangle-mesh body (reference geometry from a scan or STL). Drawn with its facets' crease lines."""
    def __init__(s, tris, name=None):
        T = np.ascontiguousarray(np.asarray(tris, float).reshape(-1, 3, 3))
        s.mesh, s.surface, s.visible, s.dl, s._shape = True, False, True, None, None
        s.tv = T; n, _ = tri_normals(T) if len(T) else (np.zeros((0, 3)), None); s.tn = n
        s.vn = np.repeat(n[:, None, :], 3, 1) if len(T) else np.zeros((0, 3, 3))
        s.tri_face = np.zeros(len(T), dtype=int); s.faces = []
        P = T.reshape(-1, 3); s.lo, s.hi = (V(*P.min(0)), V(*P.max(0))) if len(P) else (V(), V())
        s.zmin = float(P[:, 2].min()) if len(P) else 0.0; s.defl = 0.05
        s.sa, s.sb = feature_segments(T); s.se = np.full(len(s.sa), -1, dtype=int)
        s.eds, s.edges, s.verts = [], [], np.zeros((0, 3))
        s.color, s.material, s.face_colors, s.name = None, None, {}, name
    @property
    def shape(s):
        if s._shape is None: s._shape = mesh_to_brep(s.tv, merge_planar=False)
        return s._shape
    @shape.setter
    def shape(s, v): s._shape = v
    def mass(s):
        dens = dict(MATERIALS).get(s.material)
        return abs(mesh_volume(s.tv))/1000.0*dens if dens else None
    def face_normal(s, fid): return None
    def face_cylinder(s, fid): return None
    def faces_of_edge(s, e): return []

MESH_CONVERT_MAX = 20000
def mesh_quality(shape, q, defl=0.05, ang=15.0):
    """(chordal deviation mm, angle deg) for a refinement preset, scaled to the part's size."""
    if q == "custom": return max(1e-4, defl), max(1.0, ang)
    lo, hi = bbox(shape); d = max((hi - lo).Length, 1.0)
    return {"low": (d*0.002, 30.0), "med": (d*0.0005, 15.0), "high": (d*0.0001, 6.0)}[q]

def mesh_vertex_curvature(T):
    """Rough curvature (1/mm) at each triangle corner: normal change across neighbouring triangles per unit length."""
    T = np.asarray(T, float)
    if not len(T): return np.zeros((0, 3))
    Vv, F = weld(T); n, area = tri_normals(T)
    acc = np.zeros((len(Vv), 3)); np.add.at(acc, F.reshape(-1), np.repeat(n*area[:, None], 3, 0))
    vn = acc/np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    E, _ = edge_map(F); L = np.linalg.norm(Vv[E[:, 0]] - Vv[E[:, 1]], axis=1)
    k = np.linalg.norm(vn[E[:, 0]] - vn[E[:, 1]], axis=1)/np.maximum(L, 1e-9)
    kv = np.zeros(len(Vv)); cnt = np.zeros(len(Vv))
    np.add.at(kv, E[:, 0], k); np.add.at(kv, E[:, 1], k); np.add.at(cnt, E[:, 0], 1); np.add.at(cnt, E[:, 1], 1)
    return (kv/np.maximum(cnt, 1))[F]

def xform_body(b, M):
    """Move / rotate a body by a 3x4 matrix (meshes stay meshes)."""
    if getattr(b, "mesh", False):
        A = np.array(M, float); T = b.tv @ A[:, :3].T + A[:, 3]
        if np.linalg.det(A[:, :3]) < 0: T = T[:, [0, 2, 1]]
        mb = MeshBody(T).copy_look(b); mb.visible = b.visible; return mb
    return Body(transformed(b.shape, M)).copy_look(b)


# ---------------------------------------------------------------------------------------------------------------
#  Assemblies (0.7): components (named groups of bodies), joints with degrees of freedom, drive, motion links
# ---------------------------------------------------------------------------------------------------------------
JOINT_KINDS = [("Rigid", "rigid"), ("Revolute", "revolute"), ("Slider", "slider"), ("Cylindrical", "cylindrical"),
               ("Pin-slot", "pinslot"), ("Planar", "planar"), ("Ball", "ball")]
# each joint's free values: (label, "ang" | "len", motion in the joint frame)
JOINT_DOF = {"rigid": [], "revolute": [("Rotation", "ang", "rz")], "slider": [("Slide", "len", "tz")],
             "cylindrical": [("Rotation", "ang", "rz"), ("Slide", "len", "tz")],
             "pinslot": [("Rotation", "ang", "rz"), ("Slide", "len", "tx")],
             "planar": [("Slide X", "len", "tx"), ("Slide Y", "len", "ty"), ("Rotation", "ang", "rz")],
             "ball": [("Pitch", "ang", "rx"), ("Yaw", "ang", "rz")]}

def frame_mat(o, z, x):
    """4x4 matrix whose columns are the frame's x, y, z axes and origin."""
    z = z.normalize(); x = (x - z*x.dot(z)).normalize(); y = z.cross(x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x.t(), y.t(), z.t(), o.t(); return M

def perp_dir(z):
    z = z.normalize(); a = V(1, 0, 0) if abs(z.x) < 0.9 else V(0, 1, 0)
    return (a - z*a.dot(z)).normalize()

def dof_mat(kind, vals):
    """Motion (4x4, in the joint frame) for a joint's values."""
    M = np.eye(4)
    for (lab, unit, m), val in zip(JOINT_DOF[kind], vals):
        T = np.eye(4)
        if m[0] == "t": T[{"x": 0, "y": 1, "z": 2}[m[1]], 3] = val
        else:
            a = math.radians(val); c, s_ = math.cos(a), math.sin(a)
            if m[1] == "z": T[:2, :2] = [[c, -s_], [s_, c]]
            elif m[1] == "x": T[1:3, 1:3] = [[c, -s_], [s_, c]]
            else: T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s_, -s_, c
        M = M @ T
    return M

def mat34(M): return [list(map(float, M[i])) for i in range(3)]
def xf_pt(M, p): return V(*(M[:3, :3] @ np.array(p.t()) + M[:3, 3]))
def xf_dir(M, d): return V(*(M[:3, :3] @ np.array(d.t())))

class AssemblyMixin:
    def asm_init(s):
        s.comps, s.joints, s.mlinks, s.active_comp = {}, [], [], None
        s.sm = dict(SM_DEFAULT)

    def asm_state(s):
        if not hasattr(s, "comps"): s.asm_init()
        return {"comps": copy.deepcopy(s.comps), "joints": copy.deepcopy(s.joints), "mlinks": copy.deepcopy(s.mlinks), "sm": dict(s.sm)}
    def asm_apply(s, d):
        s.comps = copy.deepcopy(d.get("comps", {})); s.joints = copy.deepcopy(d.get("joints", [])); s.mlinks = copy.deepcopy(d.get("mlinks", []))
        s.sm = dict(d.get("sm") or SM_DEFAULT)
        if getattr(s, "active_comp", None) not in s.comps: s.active_comp = None

    def comp_of(s, bi): return getattr(s.bodies[bi], "comp", None)
    def comp_bodies(s, name): return [i for i, b in enumerate(s.bodies) if getattr(b, "comp", None) == name]
    def new_comp_name(s):
        i = 1
        while f"Component{i}" in s.comps: i += 1
        return f"Component{i}"

    def after_op_comps(s, op, before):
        """Bodies an operation re-created lose their component tag: give it back (same slot) or put new bodies in the
        component that was active when the step was made."""
        same = len(before) == len(s.bodies)
        for i, b in enumerate(s.bodies):
            if getattr(b, "comp", None) is not None: continue
            c = before[i] if same and i < len(before) else op.get("in_comp")
            if c and c in s.comps: nb = b.restyled(); nb.comp = c; s.bodies[i] = nb

    # ---- joint origins ----
    def ref_frame(s, r):
        """(origin, z axis, x axis) a joint snaps to: face centre + normal, circle centre + axis, cylinder end + axis,
        straight edge middle + direction, or a point with Z up."""
        k = r[0]
        if k == "face":
            b = s.bodies[r[1]]; cyl = b.face_cylinder(r[2])
            if cyl:
                o, a, xd, R, v0, v1, ext = cyl; p = V(*r[3]) if len(r) > 3 else o
                t = (p - o).dot(a); end = v0 if abs(t - v0) <= abs(t - v1) else v1
                return o + a*end, a, xd if xd.Length > 1e-9 else perp_dir(a)
            f = b.faces[r[2]]; c, n = face_frame(f); return centroid(f), n, perp_dir(n)
        if k in ("edge", "curve"):
            e = s.ref_edge(r); c = BRepAdaptor_Curve(e)
            if c.GetType() == GeomAbs_Circle:
                ci = c.Circle(); z = vec(ci.Axis().Direction()); o = vec(ci.Location())
                if k == "edge":                                          # point the axis out of the flat face the circle lies on
                    for f in faces_of_edge_in(s.bodies[r[1]].shape, e):
                        try:
                            if BRepAdaptor_Surface(f).GetType() == GeomAbs_Plane:
                                n = face_frame(f)[1]
                                if n.dot(z) < 0: z = -z
                                break
                        except Exception: pass
                return o, z, vec(ci.XAxis().Direction())
            if c.GetType() == GeomAbs_Line:
                a, b = vec(c.Value(c.FirstParameter())), vec(c.Value(c.LastParameter())); d = (b - a).normalize(); return (a + b)*0.5, d, perp_dir(d)
            p = s.ref_point(r); return p, V(0, 0, 1), V(1, 0, 0)
        if k in ("oplane", "cplane"):
            pl = s.ref_plane(r); return pl.o, pl.n, pl.u
        return s.ref_point(r), V(0, 0, 1), V(1, 0, 0)

    def ref_comp(s, r):
        if r[0] in ("face", "edge", "vertex", "body", "spot"): return s.comp_of(r[1])
        return None

    # ---- moving components ----
    def moving_set(s, comp, skip=None):
        """A component plus everything jointed onto it (directly or through other joints)."""
        out = {comp}; changed = True
        while changed:
            changed = False
            for j, jt in enumerate(s.joints):
                if j == skip: continue
                if jt["b"] in out and jt["a"] not in out and not s.comps.get(jt["a"], {}).get("ground"): out.add(jt["a"]); changed = True
        return out

    def move_comps(s, comps, M, skip=None):
        """Transform every body of the components (and the joint frames carried by them) by the 4x4 matrix M."""
        m34 = mat34(M)
        for i, b in enumerate(s.bodies):
            if getattr(b, "comp", None) in comps:
                nb = xform_body(b, m34); nb.comp = b.comp; nb.visible = b.visible; nb.name = getattr(b, "name", None)
                for k in ("opacity", "selectable"):
                    if hasattr(b, k): setattr(nb, k, getattr(b, k))
                s.bodies[i] = nb
        for j, jt in enumerate(s.joints):
            if j != skip and jt["b"] in comps:
                jt["o"], jt["z"], jt["x"] = xf_pt(M, jt["o"]), xf_dir(M, jt["z"]), xf_dir(M, jt["x"])

    def joint_frame(s, jt): return frame_mat(jt["o"], jt["z"], jt["x"])

    def drive_joint(s, j, vals, follow=True, _seen=None):
        jt = s.joints[j]; kind = jt["kind"]; dof = JOINT_DOF[kind]
        if not dof: raise RuntimeError("a rigid joint doesn't move")
        vals = [float(x) for x in vals][:len(dof)] + jt["vals"][len(vals):]
        for k, lim in (jt.get("lim") or {}).items():
            k = int(k)
            if k < len(vals) and lim: vals[k] = min(max(vals[k], lim[0]), lim[1])
        old = list(jt["vals"])
        if all(abs(a - b) < 1e-12 for a, b in zip(old, vals)): return
        J = s.joint_frame(jt); M = J @ dof_mat(kind, vals) @ np.linalg.inv(dof_mat(kind, old)) @ np.linalg.inv(J)
        if s.comps.get(jt["a"], {}).get("ground"): raise RuntimeError(f"{jt['a']} is grounded - unground it to drive this joint")
        s.move_comps(s.moving_set(jt["a"], skip=j), M, skip=j); jt["vals"] = vals
        if follow:
            _seen = (_seen or set()) | {j}
            for ln in s.mlinks:
                for src, dst, sign in ((ln["j1"], ln["j2"], 1), (ln["j2"], ln["j1"], -1)):
                    if src == j and dst not in _seen and dst < len(s.joints):
                        k1, k2 = (ln.get("k1", 0), ln.get("k2", 0)) if sign > 0 else (ln.get("k2", 0), ln.get("k1", 0))
                        r = ln.get("ratio", 1.0) if sign > 0 else 1.0/(ln.get("ratio", 1.0) or 1.0)
                        d = (vals[k1] - old[k1])*r; nv = list(s.joints[dst]["vals"])
                        if k2 < len(nv): nv[k2] += d; s.drive_joint(dst, nv, True, _seen)

    # ---- the timeline operations ----
    def exec_asm(s, op):
        t = op["t"]
        if t == "comp":
            name = op.get("name") or s.new_comp_name()
            s.comps.setdefault(name, {"ground": bool(op.get("ground")), "pn": "", "desc": ""})
            if op.get("ground") is not None: s.comps[name]["ground"] = bool(op.get("ground"))
            for bi in sorted({r[1] for r in op.get("cbodies", []) if r[0] in ("body", "face")}):
                if bi >= len(s.bodies): raise RuntimeError("a body for this component no longer exists")
                nb = s.bodies[bi].restyled(); nb.comp = name; s.bodies[bi] = nb
            if not op.get("cbodies"): s.active_comp = name
            return
        if t == "compprops":
            c = s.comps.get(op["comp"])
            if c is None: raise RuntimeError(f"no component {op['comp']}")
            new = op.get("name")
            for k in ("ground", "pn", "desc"):
                if k in op: c[k] = op[k]
            if new and new != op["comp"]:
                if new in s.comps: raise RuntimeError(f"there's already a component called {new}")
                s.comps = {(new if k == op["comp"] else k): v for k, v in s.comps.items()}            # keep the order
                for i, b in enumerate(s.bodies):
                    if getattr(b, "comp", None) == op["comp"]: nb = b.restyled(); nb.comp = new; s.bodies[i] = nb
                for jt in s.joints:
                    for k in ("a", "b"):
                        if jt[k] == op["comp"]: jt[k] = new
            return
        if t == "joint":
            ca, cb = s.ref_comp(op["a"]), s.ref_comp(op["b"]) if op.get("b") else None
            if ca is None: raise RuntimeError("the first pick must be on a component (Assemble > New Component makes one from bodies)")
            if ca == cb: raise RuntimeError("both picks are on the same component")
            if s.comps.get(ca, {}).get("ground"):
                if cb is None or s.comps.get(cb, {}).get("ground"): raise RuntimeError("both components are grounded")
                ca, cb = cb, ca; ra, rb = op["b"], op["a"]
            else: ra, rb = op["a"], op.get("b")
            oa, za, xa = s.ref_frame(ra)
            ob, zb, xb = s.ref_frame(rb) if rb else (V(), V(0, 0, 1), V(1, 0, 0))
            Fb = frame_mat(ob, zb, xb)
            if not op.get("asbuilt"):
                flip = np.eye(4) if op.get("flip") else np.diag([1.0, -1.0, -1.0, 1.0])       # default: the two origins face each other
                a = math.radians(op.get("angle", 0.0)); R = np.eye(4); R[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
                T = np.eye(4); T[2, 3] = op.get("offset", 0.0)
                M = Fb @ T @ R @ flip @ np.linalg.inv(frame_mat(oa, za, xa))
                s.move_comps(s.moving_set(ca), M)
            lim = {}
            for k in range(len(JOINT_DOF[op["kind"]])):
                if op.get(f"lim{k}"): lim[k] = (float(op[f"lo{k}"]), float(op[f"hi{k}"]))
            s.joints.append({"name": op.get("name") or f"Joint{len(s.joints) + 1}", "kind": op["kind"], "a": ca, "b": cb,
                             "o": ob, "z": zb, "x": xb, "vals": [0.0]*len(JOINT_DOF[op["kind"]]), "lim": lim})
            return
        if t == "jdrive":
            if op["j"] >= len(s.joints): raise RuntimeError("that joint no longer exists")
            s.drive_joint(op["j"], op["vals"]); return
        if t == "mlink":
            if max(op["j1"], op["j2"]) >= len(s.joints): raise RuntimeError("that joint no longer exists")
            s.mlinks.append({"j1": op["j1"], "k1": op.get("k1", 0), "j2": op["j2"], "k2": op.get("k2", 0), "ratio": op.get("ratio", 1.0)}); return
        raise RuntimeError(f"unknown assembly operation {t}")

    # ---- motion study: animate a joint without touching the timeline ----
    def motion_frames(s, j, k, a, b, n):
        base_bodies, base_joints = list(s.bodies), copy.deepcopy(s.joints)
        out = []
        for i in range(n + 1):
            s.bodies, s.joints = list(base_bodies), copy.deepcopy(base_joints)
            vals = list(s.joints[j]["vals"]); vals[k] = a + (b - a)*i/max(n, 1); s.drive_joint(j, vals)
            out.append(list(s.bodies))
        s.bodies, s.joints = base_bodies, base_joints
        return out

    # ---- drawing ----
    def draw_joints(s):
        if not s.joints or s.skedit: return
        L = max(s.span()*0.03, 2.0)
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST); glLineWidth(2.0)
        for jt in s.joints:
            if not s.comps.get(jt["a"]) and jt["a"] is not None: continue
            o, z, x = jt["o"], jt["z"].normalize(), jt["x"].normalize(); y = z.cross(x)
            glColor3f(0.0, 0.55, 0.85); glBegin(GL_LINE_LOOP)
            for i in range(24):
                a = 2*math.pi*i/24; p = o + x*(L*math.cos(a)) + y*(L*math.sin(a)); glVertex3f(*p.t())
            glEnd()
            glBegin(GL_LINES); glColor3f(0.0, 0.55, 0.85); glVertex3f(*o.t()); glVertex3f(*(o + z*L*1.8).t())
            glColor3f(0.9, 0.3, 0.2); glVertex3f(*o.t()); glVertex3f(*(o + x*L*1.2).t()); glEnd()
        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)

    # ---- bill of materials ----
    def bom_rows(s):
        rows = []
        names = list(s.comps) + ([None] if any(getattr(b, "comp", None) is None for b in s.bodies) else [])
        for name in names:
            ids = s.comp_bodies(name) if name else [i for i, b in enumerate(s.bodies) if getattr(b, "comp", None) is None]
            if not ids: continue
            vol_, mass, mats = 0.0, 0.0, set(); lo = hi = None
            for i in ids:
                b = s.bodies[i]; pr = s.body_props(b); vol_ += pr["volume"]; mass += pr["mass"] or 0.0
                if b.material: mats.add(b.material)
                lo = pr["lo"] if lo is None else V(min(lo.x, pr["lo"].x), min(lo.y, pr["lo"].y), min(lo.z, pr["lo"].z))
                hi = pr["hi"] if hi is None else V(max(hi.x, pr["hi"].x), max(hi.y, pr["hi"].y), max(hi.z, pr["hi"].z))
            c = s.comps.get(name, {})
            rows.append({"name": name or "(loose bodies)", "pn": c.get("pn", ""), "desc": c.get("desc", ""), "qty": 1, "bodies": len(ids),
                         "material": ", ".join(sorted(mats)) or "-", "mass": mass if mats else None, "volume": vol_, "size": hi - lo})
        # identical components (same bodies, size and material) count as one line with a quantity
        merged = []
        for r in rows:
            twin = next((m for m in merged if m["pn"] and m["pn"] == r["pn"]), None)
            if twin: twin["qty"] += 1
            else: merged.append(r)
        return merged


# ---------------------------------------------------------------------------------------------------------------
#  Sheet metal (0.7): flanges, hems and folds that keep a constant thickness, and an unfolder for flat patterns
# ---------------------------------------------------------------------------------------------------------------
SM_DEFAULT = {"t": 1.5, "r": 1.5, "k": 0.44, "gap": 0.5}

def poly_face(pts):
    mk = BRepBuilderAPI_MakePolygon()
    for q in pts: mk.Add(pnt(q))
    mk.Close(); return BRepBuilderAPI_MakeFace(mk.Wire(), True).Face()

def rot4(o, d, ang):
    """4x4 rotation by ang (radians) about the line o + d*t."""
    M = np.eye(4); M[:3] = np.array(rot_matrix(o, d, ang)); return M
def trans4(v): M = np.eye(4); M[:3, 3] = v.t(); return M
def apply4(M, p): return V(*(M[:3, :3] @ np.array(p.t()) + M[:3, 3]))
def xform_shape(sh, M): return transformed(sh, mat34(M))

def edge_line(e):
    c = BRepAdaptor_Curve(e)
    if c.GetType() != GeomAbs_Line: raise RuntimeError("pick straight edges")
    a, b = vec(c.Value(c.FirstParameter())), vec(c.Value(c.LastParameter())); return a, b

def planar_extent(f, w):
    vs = [vec(st(BRep_Tool, "Pnt")(x)) for x in subshapes(f, TopAbs_VERTEX)]
    ds = [p.dot(w) for p in vs]; return (max(ds) - min(ds)) if ds else 0.0

def sheet_edge_info(shape, e):
    """For a straight edge on a sheet: (a, b, nP, d_out, t) - ends, the outward normal of the big face it lies on,
    the outward direction of the thin side face and the sheet thickness there."""
    a, b = edge_line(e); ed = (b - a).normalize(); fs = faces_of_edge_in(shape, e)
    flat = [(f, face_frame(f)[1]) for f in fs if BRepAdaptor_Surface(f).GetType() == GeomAbs_Plane]
    if len(flat) != 2: raise RuntimeError("pick an edge where a flat face meets the sheet's thin side")
    ext = [(planar_extent(f, n.cross(ed).normalize()), f, n) for f, n in flat]
    ext.sort(key=lambda x: x[0]); t, side, d_out = ext[0]; big_n = ext[1][2]
    return a, b, big_n, d_out, t

def flange_tool(a, b, nP, d_out, t, r, ang, h, up=True):
    """Bend + straight flange added along the edge a-b (bend starts at the edge, material stays outside the part)."""
    ed = (b - a).normalize()
    if up: ax_o = a + nP*r; rect = [a, b, b - nP*t, a - nP*t]
    else: ax_o = a - nP*(t + r); rect = [a, b, b - nP*t, a - nP*t]
    S = poly_face(rect); c = (a + b)*0.5 - nP*(t/2)
    ax = ed if ed.cross(c - ax_o).dot(d_out) > 0 else -ed                         # rotating S this way moves it outward
    parts = []
    if ang > 1e-9: parts.append(revolve(S, ax_o, ax, ang))
    R = rot4(ax_o, ax, math.radians(ang)); S2 = xform_shape(S, R); dirn = V(*(R[:3, :3] @ np.array(d_out.t())))
    if h > 1e-9: parts.append(prism(S2, dirn, h))
    return fuse_all(parts) if len(parts) > 1 else parts[0]

def inset_ends(a, b, others, d):
    """Pull an edge's ends in by d where they meet another picked edge (corner relief between neighbouring flanges)."""
    ed = (b - a).normalize(); a2, b2 = a, b
    for oa, ob in others:
        for q in (oa, ob):
            if (q - a).Length < 1e-6: a2 = a + ed*d
            if (q - b).Length < 1e-6: b2 = b - ed*d
    return a2, b2

# ---------------- unfolding ----------------
def _tangent_faces(shape):
    """{face index: [(neighbour index, shared edge)]} for faces that continue smoothly into each other on the same side."""
    faces = subshapes(shape, TopAbs_FACE); idx = ShapeIndex()
    for f in faces: idx.add(f)
    m = TopTools_IndexedDataMapOfShapeListOfShape(); TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    nb = {i: [] for i in range(len(faces))}
    for k in range(1, m.Extent() + 1):
        e = to_edge(m.FindKey(k)); lst = [to_face(x) for x in m.FindFromIndex(k)]
        if len(lst) != 2 or degenerated(e): continue
        i, j = idx.find(lst[0]), idx.find(lst[1])
        if i < 0 or j < 0 or i == j: continue
        p = _emid(e)
        try: n1, n2 = normal_at(faces[i], p)[1], normal_at(faces[j], p)[1]
        except Exception: continue
        if n1.dot(n2) > math.cos(math.radians(1.0)): nb[i].append((j, e)); nb[j].append((i, e))
    return faces, nb

def face_loops_pts(f, defl=0.05):
    """Boundary wires of a face as point lists (outer first)."""
    from OCP.BRepTools import BRepTools as _BT
    outer = st(_BT, "OuterWire")(f); out = []
    for w in [outer] + [x for x in subshapes(f, TopAbs_WIRE) if not x.IsSame(outer)]:
        from OCP.BRepTools import BRepTools_WireExplorer
        ex = BRepTools_WireExplorer(w, f); pts = []
        while ex.More():
            e = ex.Current(); P = edge_pts(e, defl)
            if len(P):
                if e.Orientation() == TopAbs_REVERSED: P = P[::-1]
                pts += [V(*q) for q in (P if not pts else P[1:])]
            ex.Next()
        if len(pts) > 2: out.append(pts)
    return out

def unfold_sheet(shape, t, k, base_face=None):
    """Flatten a constant-thickness sheet made of flat faces and cylindrical bends.
    Returns (flat face in the base face's plane, bend lines [(p, q)], base frame (o, n, u))."""
    faces, nb = _tangent_faces(shape)
    planar = [i for i, f in enumerate(faces) if BRepAdaptor_Surface(f).GetType() == GeomAbs_Plane]
    if not planar: raise RuntimeError("that body has no flat faces")
    f0 = base_face if base_face is not None else max(planar, key=lambda i: face_area(faces[i]))
    c0, n0 = face_frame(faces[f0])
    M = {f0: np.eye(4)}; bends = []; order = [f0]; info = {}
    while order:
        i = order.pop(0)
        for j, e in nb[i]:
            if j in M: continue
            srf_i = BRepAdaptor_Surface(faces[i]).GetType(); srf_j = BRepAdaptor_Surface(faces[j])
            if srf_j.GetType() == GeomAbs_Plane:
                if srf_i == GeomAbs_Plane: M[j] = M[i]; order.append(j); continue
                # leaving a bend: handled when the bend was entered
                continue
            if srf_j.GetType() != GeomAbs_Cylinder or srf_i != GeomAbs_Plane: continue
            cyl = srf_j.Cylinder(); ax = cyl.Axis(); ao, ad = vec(ax.Location()), vec(ax.Direction()).normalize(); Rs = cyl.Radius()
            ea, eb = edge_line(e); foot = lambda p: ao + ad*((p - ao).dot(ad))
            r0 = ea - foot(ea)
            fm = centroid(faces[j]); rm = fm - foot(fm)
            rot_ax = ad if ad.dot(r0.cross(rm)) > 0 else -ad
            # span of the bend: the largest angle of the bend face's vertices from the start line
            ang = 0.0
            for v_ in subshapes(faces[j], TopAbs_VERTEX):
                p = vec(st(BRep_Tool, "Pnt")(v_)); rv = p - foot(p)
                if rv.Length < 1e-9: continue
                a_ = math.atan2(rot_ax.dot(r0.cross(rv)), r0.dot(rv))
                if a_ < -1e-6: a_ += 2*math.pi
                ang = max(ang, a_)
            if ang < 1e-6: continue
            p_in = fm; n_in = normal_at(faces[j], fm)[1]
            inner = n_in.dot(p_in - foot(p_in)) < 0
            Rn = Rs + k*t if inner else Rs - (1 - k)*t
            d_out = rot_ax.cross(r0).normalize()
            L = ang*Rn
            info[j] = dict(ao=ao, rot_ax=rot_ax, r0=r0, Rn=Rn, d_out=d_out, ang=ang, start=ea, Mbase=M[i])
            M[j] = M[i]
            bends.append((M[i], ea, eb, d_out, L))
            Mq = M[i] @ trans4(d_out*L) @ rot4(ao, rot_ax, -ang)
            for q, e2 in nb[j]:
                if q in M or q == i: continue
                if BRepAdaptor_Surface(faces[q]).GetType() == GeomAbs_Plane: M[q] = Mq; order.append(q)
    # flat outline: every reached face mapped into the base plane
    pieces = []
    for i, Mi in M.items():
        f = faces[i]
        if i in info:
            d = info[i]; ao, ra, r0, Rn = d["ao"], d["rot_ax"], d["r0"], d["Rn"]; base = d["start"]
            ad = ra.normalize()
            def mp(p, d=d, ao=ao, ra=ra, r0=r0, Rn=Rn, base=base, ad=ad):
                ft = ao + ad*((p - ao).dot(ad)); rv = p - ft
                a_ = math.atan2(ra.dot(r0.cross(rv)), r0.dot(rv))
                if a_ < -1e-6: a_ += 2*math.pi
                along = ad*((p - base).dot(ad))
                return apply4(d["Mbase"], base + along + d["d_out"]*(a_*Rn))
        else:
            mp = lambda p, Mi=Mi: apply4(Mi, p)
        loops = face_loops_pts(f)
        if not loops: continue
        try:
            fc = poly_face([mp(p) for p in loops[0]])
            for h in loops[1:]: fc = as_face(boolean(fc, poly_face([mp(p) for p in h]), "cut"))
            pieces.append(fc)
        except Exception: pass
    if not pieces: raise RuntimeError("couldn't flatten that body")
    flat = clean(fuse_all(pieces)) if len(pieces) > 1 else pieces[0]
    lines = []
    for Mi, ea, eb, d_out, L in bends:
        lines.append((apply4(Mi, ea + d_out*(L/2)), apply4(Mi, eb + d_out*(L/2))))
    u = perp_dir(n0)
    return flat, lines, (c0, n0, u)

def flat_solid(flat, n0, t):
    return prism(flat, -n0, t)

def flat_dxf(path, flat, lines, frame):
    """Outline (layer OUTLINE) and bend centre lines (layer BEND) in the base face's 2D frame."""
    o, n, u = frame; v = n.cross(u)
    to2 = lambda p: ((p - o).dot(u), (p - o).dot(v))
    L = ["0", "SECTION", "2", "ENTITIES"]
    for e in subshapes(flat, TopAbs_EDGE):
        P = edge_pts(e, 0.01)
        for p1, p2 in zip(P, P[1:]):
            (x1, y1), (x2, y2) = to2(V(*p1)), to2(V(*p2))
            L += ["0", "LINE", "8", "OUTLINE", "10", f"{x1:.4f}", "20", f"{y1:.4f}", "11", f"{x2:.4f}", "21", f"{y2:.4f}"]
    for a, b in lines:
        (x1, y1), (x2, y2) = to2(a), to2(b)
        L += ["0", "LINE", "8", "BEND", "62", "1", "10", f"{x1:.4f}", "20", f"{y1:.4f}", "11", f"{x2:.4f}", "21", f"{y2:.4f}"]
    L += ["0", "ENDSEC", "0", "EOF"]
    with open(path, "w", encoding="utf-8") as fh: fh.write("\n".join(L) + "\n")

SM_OPS = ("smrule", "sbase", "sflange", "shem", "sfold", "unfold", "refold", "flatpattern")

class SheetMixin:
    def sheet_body(s, shape, look=None, name=None):
        b = Body(shape)
        if look is not None: b.copy_look(look)
        b.sheet = dict(s.sm)
        if name: b.name = name
        return b

    def sheet_target(s, refs):
        bis = sorted({r[1] for r in refs if r[0] in ("body", "face", "edge")})
        if not bis: raise RuntimeError("pick a sheet metal body")
        return bis

    def exec_sheet(s, op):
        t = op["t"]; sm = s.sm
        if t == "smrule":
            s.sm = {"t": op["thick"], "r": op["radius"], "k": op["kfactor"], "gap": op.get("gap", sm.get("gap", 0.5))}; return
        if t == "sbase":
            regs = op["profiles"]
            if not regs: raise RuntimeError("pick sketch profiles")
            faces = [s.ref_face(r) for r in regs]; pl = s.ref_plane(regs[0]); n = pl.n*(-1 if op.get("flip") else 1)
            sh = fuse_all([prism(f, n, sm["t"]) for f in faces]) if len(faces) > 1 else prism(faces[0], n, sm["t"])
            for x in solids(clean(sh)): s.bodies.append(s.sheet_body(x))
            s.hide_sketches_of(regs); return
        if t in ("sflange", "shem"):
            groups = {}
            for r in op["edges"]: groups.setdefault(r[1], []).append(r[2])
            for bi, eis in sorted(groups.items(), reverse=True):
                b = s.bodies[bi]; infos = [sheet_edge_info(b.shape, b.eds[ei]) for ei in eis]
                th = infos[0][4]; rr = op.get("r") if op.get("custom_r") else sm["r"]
                ends = [(a, bb) for a, bb, *_ in infos]; tools = []
                for k_, (a, bb, nP, d_out, tt) in enumerate(infos):
                    others = ends[:k_] + ends[k_ + 1:]
                    if t == "shem":
                        gap = op.get("gap", sm["gap"]); a2, b2 = inset_ends(a, bb, others, gap/2 + tt)
                        tools.append(flange_tool(a2, b2, nP, d_out, tt, gap/2, 180.0, op["h"], not op.get("flip")))
                    else:
                        a2, b2 = inset_ends(a, bb, others, rr + tt + sm.get("gap", 0.5)/2)
                        tools.append(flange_tool(a2, b2, nP, d_out, tt, rr, op.get("angle", 90.0), op["h"], not op.get("flip")))
                sh = b.shape
                for x in tools: sh = boolean(sh, x, "fuse")
                parts = solids(clean(sh)); vmax = abs(volume(b.shape)) + sum(abs(volume(x)) for x in tools)
                if not parts or sum(abs(volume(x)) for x in parts) > vmax*(1 + 1e-6) + 1e-6:
                    raise RuntimeError("the new wall runs into the existing part - pick other edges or flip it")
                s.bodies[bi:bi + 1] = [s.sheet_body(x, b) for x in parts]
            return
        if t == "sfold":
            r = op["line"]
            if r[0] != "curve": raise RuntimeError("pick a sketch line on the sheet")
            sk = s.sketches[r[1]]; c = sk.geo.C[r[2]]
            if c["k"] != "line": raise RuntimeError("the fold line must be a straight sketch line")
            a, bb = sk.plane.w(sk.geo.v(c["p"][0])), sk.plane.w(sk.geo.v(c["p"][1])); mid = (a + bb)*0.5; ed = (bb - a).normalize()
            bi = None
            for i, b in enumerate(s.bodies):
                for fid, f in enumerate(b.faces):
                    nn = b.face_normal(fid)
                    if nn is None: continue
                    if abs(abs(nn.dot(sk.plane.n)) - 1) < 1e-6 and abs((mid - centroid(f)).dot(nn)) < 1e-6 and classify_in_face(f, mid):
                        bi, nP = i, nn; break
                if bi is not None: break
            if bi is None: raise RuntimeError("the fold line has to lie on a flat face of a sheet body")
            b = s.bodies[bi]; th = b.sheet["t"] if getattr(b, "sheet", None) else sm["t"]; rr = sm["r"]; kk = sm["k"]
            ang = op.get("angle", 90.0); up = not op.get("flip"); d = nP.cross(ed).normalize()*(-1 if op.get("side") else 1)
            w = math.radians(ang)*(rr + kk*th); big = s.span()*4 + 100
            oA, oB = mid - d*(w/2), mid + d*(w/2)
            A = boolean(b.shape, prism(big_plane_face(oA, d, big), -d, big), "common")
            B = boolean(b.shape, prism(big_plane_face(oB, d, big), d, big), "common")
            S = as_face(boolean(b.shape, big_plane_face(oA, d, big), "common"))
            ax_o = oA + nP*rr if up else oA - nP*(th + rr)
            cS = centroid(S); ax = ed if ed.cross(cS - ax_o).dot(d) > 0 else -ed
            bend = revolve(S, ax_o, ax, ang)
            Mb = rot4(ax_o, ax, math.radians(ang)) @ trans4(-d*w)
            sh = fuse_all([A, bend, xform_shape(B, Mb)])
            s.bodies[bi:bi + 1] = [s.sheet_body(x, b) for x in solids(clean(sh))]
            s.hide_sketches_of([r]); return
        if t in ("unfold", "flatpattern"):
            for bi in reversed(s.sheet_target(op["targets"])):
                b = s.bodies[bi]; th = b.sheet["t"] if getattr(b, "sheet", None) else sm["t"]; kk = b.sheet["k"] if getattr(b, "sheet", None) else sm["k"]
                flat, lines, (o, n, u) = unfold_sheet(b.shape, th, kk)
                fs = flat_solid(flat, n, th)
                if t == "unfold":
                    nb = s.sheet_body(fs, b); nb.folded = b.shape; nb.bend_lines = lines; s.bodies[bi] = nb
                else:                                                    # lay the blank on the ground, beside the part
                    M = np.linalg.inv(frame_mat(o, n, u)); M[2, 3] += th
                    lo, hi = bbox(xform_shape(fs, M)); M = trans4(V(b.hi.x + 20.0 - lo.x, b.lo.y - lo.y, 0)) @ M
                    nb = s.sheet_body(xform_shape(fs, M), b, f"{getattr(b, 'name', None) or 'Body' + str(bi + 1)} flat pattern")
                    nb.bend_lines = [(apply4(M, p), apply4(M, q)) for p, q in lines]; nb.flat_of = bi; s.bodies.append(nb)
            return
        if t == "refold":
            for bi in s.sheet_target(op["targets"]):
                b = s.bodies[bi]
                if getattr(b, "folded", None) is None: raise RuntimeError("that body isn't unfolded")
                s.bodies[bi] = s.sheet_body(b.folded, b)
            return
        raise RuntimeError(f"unknown sheet metal operation {t}")

    def draw_bend_lines(s):
        lines = [(p, q) for b in s.bodies if b.visible for p, q in (getattr(b, "bend_lines", None) or [])]
        if not lines or s.skedit: return
        glDisable(GL_LIGHTING); glEnable(GL_LINE_STIPPLE); glLineStipple(2, 0x33FF); glLineWidth(1.6); glColor3f(0.85, 0.2, 0.2)
        glBegin(GL_LINES)
        for p, q in lines: glVertex3f(*p.t()); glVertex3f(*q.t())
        glEnd(); glDisable(GL_LINE_STIPPLE); glEnable(GL_LIGHTING)


# ---------------------------------------------------------------------------------------------------------------
#  Plastic part features (0.7): boss, lip / groove, snap-fit hook
# ---------------------------------------------------------------------------------------------------------------
PL_OPS = ("boss", "lipgroove", "snapfit")
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone

def cyl_solid(p, n, r, h): return BRepPrimAPI_MakeCylinder(gp_Ax2(pnt(p), gdir(n)), r, h).Shape()
def cone_solid(p, n, r1, r2, h):
    if abs(r1 - r2) < 1e-9: return cyl_solid(p, n, r1, h)
    return BRepPrimAPI_MakeCone(gp_Ax2(pnt(p), gdir(n)), r1, r2, h).Shape()

def box_frame(o, u, v, n, a, b, c):
    """Box with one corner at o and edges a·u, b·v, c·n."""
    pts = [o, o + u*a, o + u*a + v*b, o + v*b]; return prism(poly_face(pts), n, c)

def inner_face(ring):
    """The ring face's hole, filled (the opening a lip / groove runs around)."""
    from OCP.BRepTools import BRepTools as _BT
    outer = st(_BT, "OuterWire")(ring); holes = [w for w in subshapes(ring, TopAbs_WIRE) if not w.IsSame(outer)]
    if not holes: raise RuntimeError("pick the flat rim of a shelled part (a face with an opening in it)")
    holes.sort(key=lambda w: -abs(face_area(BRepBuilderAPI_MakeFace(w, True).Face())))
    return BRepBuilderAPI_MakeFace(holes[0], True).Face()

class PlasticMixin:
    def exec_plastic(s, op):
        t = op["t"]
        if t == "boss":
            r = op["face"]; b = s.bodies[r[1]]; n = b.face_normal(r[2])
            if n is None: raise RuntimeError("bosses go on flat faces")
            n = -n if op.get("flip") else n; p = s.ref_point(r); h = op["h"]
            dr = math.tan(math.radians(op.get("draft", 0.0)))*h; R = op["od"]/2
            tool = cone_solid(p, n, R + dr, R, h)
            ribs = int(op.get("ribs", 0))
            if ribs:
                u = perp_dir(n); v_ = n.cross(u); tw, rl, rh = op.get("rib_t", 1.2), op.get("rib_l", R*1.2), h*op.get("rib_h", 0.7)
                for k in range(ribs):
                    a = 2*math.pi*k/ribs; d = u*math.cos(a) + v_*math.sin(a); w_ = n.cross(d)
                    o = p + d*(R*0.5) - w_*(tw/2)
                    rib = prism(poly_face([o, o + d*(R*0.5 + rl), o + d*(R*0.5) + n*rh]), w_, tw)
                    tool = boolean(tool, rib, "fuse")
            sh = boolean(b.shape, tool, "fuse")
            if op.get("id", 0) > 0:
                depth = op.get("hole_d") or h
                sh = boolean(sh, cyl_solid(p + n*(h + 0.01), -n, op["id"]/2, depth + 0.01), "cut")
            s.replace_body(r[1], sh); return
        if t == "lipgroove":
            r = op["face"]; b = s.bodies[r[1]]; F = b.faces[r[2]]; n = b.face_normal(r[2])
            if n is None: raise RuntimeError("pick the flat rim of the part")
            hole = inner_face(F); w, h, c = op["w"], op["h"], op.get("clear", 0.1)
            if op.get("kind", "lip") == "lip":
                band = as_face(boolean(offset_face(hole, w), hole, "cut")); tool = prism(band, n, h)
                s.replace_body(r[1], boolean(b.shape, tool, "fuse"))
            else:
                band = as_face(boolean(offset_face(hole, w + c), offset_face(hole, -c), "cut"))
                tool = prism(band, -n, h + c)
                s.replace_body(r[1], boolean(b.shape, tool, "cut"))
            return
        if t == "snapfit":
            r = op["face"]; b = s.bodies[r[1]]; n = b.face_normal(r[2])
            if n is None: raise RuntimeError("snap fits go on flat faces")
            n = -n if op.get("flip") else n; p = s.ref_point(r)
            u = perp_dir(n); a = math.radians(op.get("rot", 0.0)); v0 = n.cross(u); d = u*math.cos(a) + v0*math.sin(a); wv = n.cross(d)
            L, bw, th, y, ang = op["L"], op["b"], op["th"], op["y"], op.get("ang", 30.0)
            o = p - d*(th/2) - wv*(bw/2)
            beam = box_frame(o, d, wv, n, th, bw, L)
            hl = y/math.tan(math.radians(max(5.0, min(85.0, ang))))                  # hook ramp length
            top = o + n*L; q0 = top + d*th
            A = q0 - n*(y*0.5 + hl); B = A + d*y; B2 = B + n*(y*0.5)              # flat catch, short land, lead-in ramp
            hook = prism(poly_face([A, B, B2, q0]), wv, bw)
            tool = boolean(beam, hook, "fuse")
            s.replace_body(r[1], boolean(b.shape, tool, "fuse")); return
        raise RuntimeError(f"unknown plastic operation {t}")

def make_plastic_commands():
    out = {}
    def add(c): c.group = "PLASTIC"; out[c.name] = c
    add(Cmd("boss", "BOSS", "boss", [
        Fsel("face", "Position (click a flat face)", {"face"}), Flen("od", "Outer Ø", 8.0, 0.1), Flen("id", "Hole Ø (0 = none)", 3.0, 0.0),
        Flen("h", "Height", 10.0, 0.1), Flen("hole_d", "Hole depth (0 = full)", 0.0, 0.0), Fang("draft", "Draft", 0.5, 0.0, 15.0),
        Fint("ribs", "Ribs", 0, 0, 8), Flen("rib_t", "Rib thickness", 1.2, 0.1, show=lambda v: v["ribs"] > 0),
        Flen("rib_l", "Rib length", 4.0, 0.1, show=lambda v: v["ribs"] > 0), Fcheck("flip", "Flip direction")],
        hint="A screw boss where you click: a drafted post with a pilot hole, optionally braced with ribs."))
    add(Cmd("lipgroove", "LIP / GROOVE", "lipgroove", [
        Fsel("face", "Rim face", {"face"}), Fcombo("kind", "Type", "lip", [("Lip (raised)", "lip"), ("Groove (cut)", "groove")]),
        Flen("w", "Width", 1.0, 0.05), Flen("h", "Height / depth", 1.5, 0.05), Flen("clear", "Clearance (groove)", 0.1, 0.0, show=lambda v: v["kind"] == "groove")],
        hint="Pick the flat rim of a shelled half. A lip stands up along the inside of the wall; a groove is cut to take the "
             "other half's lip (with clearance)."))
    add(Cmd("snapfit", "SNAP FIT", "snapfit", [
        Fsel("face", "Position (click a flat face)", {"face"}), Flen("L", "Beam length", 10.0, 0.5), Flen("b", "Beam width", 4.0, 0.2),
        Flen("th", "Beam thickness", 1.2, 0.2), Flen("y", "Hook depth", 0.8, 0.05), Fang("ang", "Lead-in angle", 30.0, 5.0, 85.0),
        Fang("rot", "Rotation", 0.0), Fcheck("flip", "Flip direction")],
        hint="A cantilever snap hook standing on the face where you click. Rotation turns which way the hook faces."))
    return out


# ---------------------------------------------------------------------------------------------------------------
#  Local rendering (0.7): environment light, soft ground shadow and reflection, ambient occlusion, high-res PNG,
#  turntables; explode storyboards
# ---------------------------------------------------------------------------------------------------------------
ENVS = {"studio": dict(sky=(0.92, 0.94, 0.97), ground=(0.42, 0.40, 0.38), bg_top=(0.97, 0.97, 0.98), bg_bot=(0.80, 0.82, 0.85),
                       key=(1.0, 0.98, 0.95), keydir=(-0.45, -0.6, 0.66), fill=(0.35, 0.38, 0.45), filldir=(0.7, 0.2, 0.3), floor=(0.88, 0.88, 0.89)),
        "warm": dict(sky=(0.98, 0.93, 0.85), ground=(0.40, 0.33, 0.27), bg_top=(0.99, 0.95, 0.88), bg_bot=(0.86, 0.80, 0.72),
                     key=(1.0, 0.92, 0.80), keydir=(-0.6, -0.4, 0.7), fill=(0.30, 0.34, 0.45), filldir=(0.6, 0.5, 0.25), floor=(0.90, 0.86, 0.80)),
        "dark": dict(sky=(0.45, 0.50, 0.58), ground=(0.10, 0.10, 0.12), bg_top=(0.20, 0.21, 0.24), bg_bot=(0.05, 0.05, 0.06),
                     key=(1.0, 1.0, 1.0), keydir=(-0.3, -0.7, 0.65), fill=(0.25, 0.30, 0.42), filldir=(0.8, 0.3, 0.2), floor=(0.13, 0.13, 0.15))}
METALS = {"Steel", "Stainless steel", "Aluminium 6061", "Brass", "Copper", "Titanium"}
MAT_ALBEDO = {"Steel": (0.62, 0.63, 0.65), "Stainless steel": (0.72, 0.72, 0.74), "Aluminium 6061": (0.80, 0.81, 0.83), "Brass": (0.80, 0.65, 0.33),
              "Copper": (0.80, 0.48, 0.32), "Titanium": (0.62, 0.60, 0.58)}

def shade_vertices(P, N, eye, albedo, metal, env):
    """Per-vertex colour: hemisphere ambient + key/fill diffuse + Blinn specular (+ environment reflection for metals)."""
    N = N/np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    Vd = np.array(eye.t()) - P; Vd /= np.maximum(np.linalg.norm(Vd, axis=1, keepdims=True), 1e-12)
    flip = (N*Vd).sum(1) < 0; N[flip] = -N[flip]                           # two-sided
    sky, gnd = np.array(env["sky"]), np.array(env["ground"]); a = np.array(albedo)
    hemi = gnd + (sky - gnd)*((N[:, 2:3] + 1)/2)
    col = a*hemi*0.55
    for L, c, sp in ((env["keydir"], env["key"], 1.0), (env["filldir"], env["fill"], 0.25)):
        L = np.array(L)/np.linalg.norm(L); d = np.clip(N @ L, 0, 1)[:, None]
        col += a*np.array(c)*d*0.75
        H = L + Vd; H /= np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
        spec = np.clip((N*H).sum(1), 0, 1)**(60 if metal else 30)
        col += np.array(c)*spec[:, None]*(0.6 if metal else 0.25)*sp
    if metal:
        R = 2*(N*Vd).sum(1, keepdims=True)*N - Vd
        envc = gnd + (sky - gnd)*np.clip((R[:, 2:3] + 0.25)*1.4, 0, 1)
        col = col*0.55 + a*envc*0.55
    return np.clip(col, 0, 1).astype(np.float32)

def ssao(depth, near, far, ortho, strength=0.6, radius=None):
    """Screen-space ambient occlusion from a depth buffer (numpy). Returns an occlusion factor image (1 = open)."""
    d = depth.astype(np.float64); bg = d >= 0.99999
    z = (near + d*(far - near)) if ortho else (2*near*far/(far + near - (2*d - 1)*(far - near)))
    h, w = z.shape; r0 = radius or max(2, int(min(h, w)/120)); occ = np.zeros_like(z); cnt = 0
    span = np.nanmedian(z[~bg]) if (~bg).any() else 1.0; rng = span*0.01
    for r in (r0, r0*2, r0*4):
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):                 # opposite pairs: flat or slanted planes cancel out
            za = np.roll(np.roll(z, dy*r, 0), dx*r, 1); zb = np.roll(np.roll(z, -dy*r, 0), -dx*r, 1)
            diff = z - (za + zb)/2; ok = (np.abs(z - za) < rng*20) & (np.abs(z - zb) < rng*20)
            occ += np.clip(diff/rng, 0, 1)*ok; cnt += 1
    occ /= cnt; occ[bg] = 0
    k = np.ones(5)/5
    for ax in (0, 1): occ = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), ax, occ)
    return np.clip(1 - strength*occ*2.2, 0, 1)

def turntable_cam(yaw0, i, n): return yaw0 + 2*math.pi*i/n

class RenderMixin:
    def render_image(s, W_, H_, o):
        """Offscreen render of the visible bodies. o: env, reflect, ao, shadow, ss (supersampling). Returns a QImage."""
        from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat
        env = ENVS[o.get("env", "studio")]; ss = int(o.get("ss", 2))
        mx = int(glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE)) if True else 8192
        ss = max(1, min(ss, mx//max(W_, H_, 1)))
        w, h = W_*ss, H_*ss
        s.makeCurrent()
        fmt = QOpenGLFramebufferObjectFormat(); fmt.setAttachment(QOpenGLFramebufferObject.Depth); fmt.setInternalTextureFormat(GL_RGBA8)
        fbo = QOpenGLFramebufferObject(w, h, fmt); fbo.bind()
        try:
            glViewport(0, 0, w, h); glUseProgram(0)
            for cap in (GL_LIGHTING, GL_TEXTURE_2D, GL_CULL_FACE, GL_CLIP_PLANE0, GL_CLIP_PLANE1, GL_CLIP_PLANE2, GL_CLIP_PLANE3, GL_CLIP_PLANE4): glDisable(cap)
            glClearColor(*env["bg_bot"], 1); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glDisable(GL_DEPTH_TEST); glMatrixMode(GL_PROJECTION); glLoadIdentity(); glMatrixMode(GL_MODELVIEW); glLoadIdentity()
            glBegin(GL_QUADS); glColor3f(*env["bg_top"]); glVertex2f(-1, 1); glVertex2f(1, 1); glColor3f(*env["bg_bot"]); glVertex2f(1, -1); glVertex2f(-1, -1); glEnd()
            glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL); glDepthMask(GL_TRUE)
            glMatrixMode(GL_PROJECTION); glLoadIdentity(); near, far = s.dist/50, s.dist*2000
            if s.ortho: hh = s.dist*math.tan(math.radians(22.5)); glOrtho(-hh*w/h, hh*w/h, -hh, hh, -s.dist*200, s.dist*200); near, far = -s.dist*200, s.dist*200
            else: gluPerspective(45, w/h, near, far)
            glMatrixMode(GL_MODELVIEW); glLoadIdentity(); e, t = s.eye(), s.target; gluLookAt(e.x, e.y, e.z, t.x, t.y, t.z, 0, 0, 1)
            glEnable(GL_BLEND); glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)   # keep it opaque
            bodies = [(i, b) for i, b in enumerate(s.bodies) if b.visible and len(b.tv)]
            if not bodies: raise RuntimeError("nothing to render")
            off = lambda i: np.array((s.disp_off or {}).get(i, (0, 0, 0)), float)
            P_all = np.concatenate([b.tv.reshape(-1, 3) + off(i) for i, b in bodies]); zmin = float(P_all[:, 2].min())
            span = float(np.ptp(P_all, 0).max()) or 1.0; c = P_all.mean(0)
            arrays = []
            for i, b in bodies:
                P = (b.tv.reshape(-1, 3) + off(i)).astype(np.float32); N = b.vn.reshape(-1, 3).astype(np.float64)
                alb = MAT_ALBEDO.get(b.material) or b.color or BODY
                arrays.append((P, N, alb, b.material in METALS))
            def draw(mirror=False, dim=1.0):
                for P, N, alb, metal in arrays:
                    PP, NN = P.astype(np.float64), N.copy()
                    if mirror: PP = PP.copy(); PP[:, 2] = 2*zmin - PP[:, 2]; NN[:, 2] = -NN[:, 2]
                    col = shade_vertices(PP, NN, e, alb, metal, env)*dim
                    glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_COLOR_ARRAY)
                    glVertexPointer(3, GL_FLOAT, 0, np.ascontiguousarray(PP, np.float32)); glColorPointer(3, GL_FLOAT, 0, np.ascontiguousarray(col))
                    glDrawArrays(GL_TRIANGLES, 0, len(PP)); glDisableClientState(GL_COLOR_ARRAY); glDisableClientState(GL_VERTEX_ARRAY)
            R = span*6; fl = np.array(env["floor"])
            def floor(alpha_c, alpha_e):
                glBegin(GL_TRIANGLE_FAN); glColor4f(*fl, alpha_c); glVertex3f(c[0], c[1], zmin)
                for k in range(65):
                    a = 2*math.pi*k/64; glColor4f(*env["bg_bot"], alpha_e); glVertex3f(c[0] + R*math.cos(a), c[1] + R*math.sin(a), zmin)
                glEnd()
            if o.get("reflect", True):
                draw(mirror=True, dim=0.9); glClear(GL_DEPTH_BUFFER_BIT); glDepthMask(GL_FALSE); floor(0.78, 1.0); glDepthMask(GL_TRUE)
            else:
                glDepthMask(GL_FALSE); floor(1.0, 1.0); glDepthMask(GL_TRUE)
            if o.get("shadow", True):                                   # soft contact shadow: the parts squashed onto the floor, jittered
                glDepthMask(GL_FALSE); glEnableClientState(GL_VERTEX_ARRAY)
                L = np.array(env["keydir"]); L = L/np.linalg.norm(L)
                for j in range(16):
                    jit = np.array([math.cos(j*2.4), math.sin(j*2.4), 0])*0.3*math.sqrt(j/16)
                    Ld = L + jit; Ld = Ld/np.linalg.norm(Ld)
                    for P, _, _, _ in arrays:
                        Q = P.astype(np.float64).copy(); t_ = (Q[:, 2] - zmin)/max(Ld[2], 0.2); Q -= np.outer(t_, Ld); Q[:, 2] = zmin + span*1e-4
                        glColor4f(0, 0, 0, 0.032); glVertexPointer(3, GL_FLOAT, 0, np.ascontiguousarray(Q, np.float32)); glDrawArrays(GL_TRIANGLES, 0, len(Q))
                for j in range(8):                                          # contact halo round the footprint
                    for P, _, _, _ in arrays:
                        Q = P.astype(np.float64).copy(); cc = Q.mean(0); Q[:, :2] = cc[:2] + (Q[:, :2] - cc[:2])*(1.0 + 0.025*(j + 1)); Q[:, 2] = zmin + span*1e-4
                        glColor4f(0, 0, 0, 0.035); glVertexPointer(3, GL_FLOAT, 0, np.ascontiguousarray(Q, np.float32)); glDrawArrays(GL_TRIANGLES, 0, len(Q))
                glDisableClientState(GL_VERTEX_ARRAY); glDepthMask(GL_TRUE)
            draw()
            img = fbo.toImage().convertToFormat(G.QImage.Format_RGB888)
            if o.get("ao", True):
                dep = glReadPixels(0, 0, w, h, GL_DEPTH_COMPONENT, GL_FLOAT)
                dep = np.frombuffer(dep, np.float32).reshape(h, w)[::-1]
                occ = ssao(dep, near, far, s.ortho)
                ptr = img.bits(); arr = np.frombuffer(ptr, np.uint8, count=img.bytesPerLine()*h).reshape(h, img.bytesPerLine())[:, :w*3].reshape(h, w, 3)
                out = (arr.astype(np.float32)*occ[:, :, None]).clip(0, 255).astype(np.uint8)
                img = G.QImage(out.tobytes(), w, h, w*3, G.QImage.Format_RGB888).copy()
        finally:
            fbo.release(); s.doneCurrent()
        if ss > 1: img = img.scaled(W_, H_, C.Qt.IgnoreAspectRatio, C.Qt.SmoothTransformation)
        s.update(); return img

    # ---- explode storyboard ----
    def explode_auto(s, factor=1.0):
        """One step per body (or component), pushed out from the assembly centre."""
        groups = {}
        for i, b in enumerate(s.bodies):
            if b.visible: groups.setdefault(getattr(b, "comp", None) or f"#{i}", []).append(i)
        if len(groups) < 2: raise RuntimeError("explode needs at least two visible bodies or components")
        cen = lambda ids: sum(((s.bodies[i].lo + s.bodies[i].hi)*0.5 for i in ids), V())*(1.0/len(ids))
        allc = cen([i for ids in groups.values() for i in ids]); span = s.span(); steps = []
        for name, ids in groups.items():
            d = cen(ids) - allc
            if d.Length < 1e-6: d = V(0, 0, 1)
            steps.append({"bodies": ids, "d": (d.normalize()*(span*0.35*factor) + d*(0.4*factor)).t(), "dur": 1.0})
        steps.sort(key=lambda x: -V(*x["d"]).z)
        return steps

    def explode_offsets(s, steps, t):
        """Body offsets at time t (seconds) through a storyboard of sequential steps."""
        off = {}; t0 = 0.0
        for st_ in steps:
            p = 1.0 if st_["dur"] <= 0 else min(1.0, max(0.0, (t - t0)/st_["dur"])); p = p*p*(3 - 2*p)
            for i in st_["bodies"]:
                cur = off.get(i, (0, 0, 0)); off[i] = tuple(cur[k] + st_["d"][k]*p for k in range(3))
            t0 += st_["dur"]
        return off

# ---------------------------------------------------------------------------------------------------------------
#  Manufacture (0.7): setup / stock, tool library, 2D face, contour, pocket and drill toolpaths, preview, G-code
# ---------------------------------------------------------------------------------------------------------------
CAM_OPS = ("cam_setup", "cam_face", "cam_contour", "cam_pocket", "cam_drill")
DEFAULT_TOOLS = [
    {"n": 1, "name": "6 mm flat end mill", "type": "flat", "dia": 6.0, "flutes": 3, "rpm": 12000, "feed": 900.0, "plunge": 300.0, "stepdown": 1.5},
    {"n": 2, "name": "3 mm flat end mill", "type": "flat", "dia": 3.0, "flutes": 2, "rpm": 16000, "feed": 600.0, "plunge": 200.0, "stepdown": 0.8},
    {"n": 3, "name": "5 mm drill", "type": "drill", "dia": 5.0, "flutes": 2, "rpm": 3000, "feed": 150.0, "plunge": 150.0, "stepdown": 2.0},
    {"n": 4, "name": "6 mm ball end mill", "type": "ball", "dia": 6.0, "flutes": 2, "rpm": 12000, "feed": 800.0, "plunge": 250.0, "stepdown": 1.0},
    {"n": 5, "name": "25 mm face mill", "type": "flat", "dia": 25.0, "flutes": 4, "rpm": 6000, "feed": 1200.0, "plunge": 300.0, "stepdown": 1.0}]

def new_cam():
    return {"setup": {"origin": "corner", "side": 2.0, "top": 1.0, "bottom": 0.0, "safe": 5.0, "retract": 2.0},
            "tools": copy.deepcopy(DEFAULT_TOOLS), "ops": []}

def wire_xy_pts(w, z=None, defl=0.02):
    """A wire's points (closed polyline) in XY, optionally at a fixed z."""
    pts = []
    from OCP.BRepTools import BRepTools_WireExplorer
    ex = BRepTools_WireExplorer(w)
    while ex.More():
        e = ex.Current(); P = edge_pts(e, defl)
        if len(P):
            if e.Orientation() == TopAbs_REVERSED: P = P[::-1]
            pts += [tuple(q) for q in (P if not pts else P[1:])]
        ex.Next()
    if pts and (np.linalg.norm(np.array(pts[0]) - np.array(pts[-1])) > 1e-6): pts.append(pts[0])
    return [(x, y, z if z is not None else zz) for x, y, zz in pts]

def flat_face_xy(f):
    """A horizontal planar face moved to z = 0 (for 2D offsets)."""
    c, n = face_frame(f)
    if abs(abs(n.z) - 1) > 1e-6: raise RuntimeError("pick horizontal (flat, upward or downward) faces")
    return to_face(transformed(f, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -c.z]])), c.z

def offset_loops(face0, d):
    """Wires of a 2D offset (d > 0 grows); [] once the offset closes up."""
    try: f = offset_face(face0, d)
    except Exception: return []
    if face_area(f) < 1e-6: return []
    return subshapes(f, TopAbs_WIRE)

def z_levels(top, bottom, step):
    if top <= bottom + 1e-9: return [bottom]
    n = max(1, int(math.ceil((top - bottom)/max(step, 1e-3) - 1e-9)))
    return [top - (top - bottom)*(k + 1)/n for k in range(n)]

class PathBuilder:
    def __init__(s, safe): s.moves, s.safe, s.cur = [], safe, None
    def rapid(s, p): s.moves.append(("rapid", tuple(map(float, p)))); s.cur = p
    def cut(s, p, f): s.moves.append(("cut", tuple(map(float, p)), f)); s.cur = p
    def loop(s, pts, z, feed, plunge):
        x0, y0 = pts[0][0], pts[0][1]
        s.rapid((x0, y0, s.safe)); s.cut((x0, y0, z), plunge)
        for x, y, _ in pts[1:]: s.cut((x, y, z), feed)
        s.rapid((pts[-1][0], pts[-1][1], s.safe))

def path_stats(moves, rapid_feed=5000.0):
    """(cut length, rapid length, estimated minutes)."""
    cl = rl = mins = 0.0; prev = None
    for m in moves:
        p = np.array(m[1])
        if prev is not None:
            L = float(np.linalg.norm(p - prev))
            if m[0] == "rapid": rl += L; mins += L/rapid_feed
            else: cl += L; mins += L/max(m[2], 1e-6)
        prev = p
    return cl, rl, mins

class CamMixin:
    def cam_init(s):
        s.cam = new_cam(); s.cam_paths = {}; s.cam_sim = None; s.show_paths = True

    def cam_stock(s):
        """(lo, hi) of the stock box around the visible solid bodies."""
        bs = [b for b in s.bodies if b.visible and not getattr(b, "mesh", False)]
        if not bs: raise RuntimeError("there's nothing to machine")
        lo, hi = exact_bbox(compound([b.shape for b in bs])); st_ = s.cam["setup"]
        return lo - V(st_["side"], st_["side"], st_["bottom"]), hi + V(st_["side"], st_["side"], st_["top"]), lo, hi

    def cam_wcs(s):
        lo, hi, _, _ = s.cam_stock(); o = s.cam["setup"]["origin"]
        return V(lo.x, lo.y, hi.z) if o == "corner" else V((lo.x + hi.x)/2, (lo.y + hi.y)/2, hi.z)

    def cam_tool(s, n):
        t = next((x for x in s.cam["tools"] if x["n"] == int(n)), None)
        if t is None: raise RuntimeError(f"there's no tool T{n} in the tool library")
        return t

    def cam_generate(s, op):
        """Toolpath moves (world coordinates) for one machining operation."""
        t = op["t"]; tool = s.cam_tool(op.get("tool", 1)); R = tool["dia"]/2
        slo, shi, mlo, mhi = s.cam_stock(); safe = shi.z + s.cam["setup"]["safe"]; pb = PathBuilder(safe)
        step = op.get("stepdown") or tool["stepdown"]; feed, plunge = tool["feed"], tool["plunge"]
        if t == "cam_face":
            if shi.z - mhi.z < 1e-6: return pb.moves
            so = tool["dia"]*op.get("stepover", 60.0)/100.0; x0, x1 = slo.x - R, shi.x + R
            for z in z_levels(shi.z, mhi.z, step):
                y = slo.y + R*0.5; rev = False; pb.rapid((x0, y, safe)); pb.cut((x0, y, z), plunge)
                while True:
                    xa, xb = (x1, x0) if rev else (x0, x1); pb.cut((xb, y, z), feed)
                    if y >= shi.y - R*0.5 - 1e-9: break
                    y = min(y + so, shi.y - R*0.5); pb.cut((xb, y, z), feed); rev = not rev
                pb.rapid((pb.cur[0], pb.cur[1], safe))
            return pb.moves
        if t == "cam_contour":
            loops = []
            for r in op["geo"]:
                if r[0] == "face":
                    f0, fz = flat_face_xy(s.bodies[r[1]].faces[r[2]]); loops.append((f0, fz))
                elif r[0] == "edge":
                    e = s.bodies[r[1]].eds[r[2]]; ws = edges_to_wires([e])
                    for w in ws:
                        try: f0 = BRepBuilderAPI_MakeFace(w, True).Face(); c_ = centroid(f0); loops.append((to_face(transformed(f0, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -c_.z]])), c_.z))
                        except Exception: raise RuntimeError("contour edges must form a closed loop")
            if not loops: raise RuntimeError("pick a flat face (its outline is cut) or a closed edge loop")
            side = op.get("side", "outside"); d = R if side == "outside" else -R if side == "inside" else 0.0
            bottom = (mlo.z if op.get("to", "model") == "model" else min(z for _, z in loops)) - op.get("extra", 0.0)
            top = shi.z
            for f0, fz in loops:
                from OCP.BRepTools import BRepTools as _BT
                outer = st(_BT, "OuterWire")(f0)
                base = BRepBuilderAPI_MakeFace(outer, True).Face()
                ws = offset_loops(base, d) if abs(d) > 1e-9 else [outer]
                if not ws: raise RuntimeError("the tool is too big for that contour")
                for z in z_levels(top, bottom, step):
                    for w in ws: pb.loop(wire_xy_pts(w, z), z, feed, plunge)
            return pb.moves
        if t == "cam_pocket":
            so = tool["dia"]*op.get("stepover", 50.0)/100.0
            for r in op["faces"]:
                f0, fz = flat_face_xy(s.bodies[r[1]].faces[r[2]])
                rings = []; d = -R
                while len(rings) < 500:
                    ws = offset_loops(f0, d)
                    if not ws: break
                    rings.append(ws); d -= so
                if not rings: raise RuntimeError("the tool doesn't fit in that pocket")
                top = mhi.z if fz < mhi.z - 1e-6 else shi.z
                for z in z_levels(top, fz, step):
                    for ws in reversed(rings):                         # inside out
                        for w in ws: pb.loop(wire_xy_pts(w, z), z, feed, plunge)
            return pb.moves
        if t == "cam_drill":
            pts = []
            for r in op["holes"]:
                if r[0] == "face":
                    cyl = s.bodies[r[1]].face_cylinder(r[2])
                    if not cyl: raise RuntimeError("pick the round walls of holes")
                    o_, a_, xd, Rh, v0, v1, ext = cyl
                    if abs(abs(a_.z) - 1) > 1e-6: raise RuntimeError("only vertical holes can be drilled")
                    zs = [(o_ + a_*v0).z, (o_ + a_*v1).z]; pts.append((o_.x, o_.y, min(zs), max(zs)))
                elif r[0] == "edge":
                    c = BRepAdaptor_Curve(s.bodies[r[1]].eds[r[2]])
                    if c.GetType() != GeomAbs_Circle: raise RuntimeError("pick circular hole edges")
                    p = vec(c.Circle().Location()); pts.append((p.x, p.y, mlo.z if op.get("through", True) else p.z - op.get("depth", 5.0), p.z))
            seen = set(); peck = op.get("peck", 0.0); out = []
            for x, y, zb, zt in pts:
                key = (round(x, 4), round(y, 4))
                if key in seen: continue
                seen.add(key); out.append((x, y, zb - op.get("extra", 0.0) - (tool["dia"]*0.3 if op.get("tip", True) else 0.0), zt))
            out.sort(key=lambda q: (round(q[1], 3), q[0]))
            for x, y, zb, zt in out:
                pb.rapid((x, y, safe)); pb.rapid((x, y, zt + s.cam["setup"]["retract"]))
                if peck > 0:
                    z = zt
                    while z > zb + 1e-9:
                        z = max(zb, z - peck); pb.cut((x, y, z), plunge); pb.rapid((x, y, zt + s.cam["setup"]["retract"]))
                else: pb.cut((x, y, zb), plunge); pb.rapid((x, y, zt + s.cam["setup"]["retract"]))
                pb.rapid((x, y, safe))
            return pb.moves
        raise RuntimeError(f"unknown machining operation {t}")

    def cam_regen(s, idx=None):
        """Recompute toolpaths. Returns {op index: error} for operations that failed."""
        errs = {}
        for i, op in enumerate(s.cam["ops"]):
            if idx is not None and i != idx: continue
            try: s.cam_paths[i] = s.cam_generate(op)
            except Exception as ex: s.cam_paths[i] = []; errs[i] = str(ex)
        return errs

    def draw_cam(s):
        if not getattr(s, "show_paths", True) or not s.cam["ops"] or s.skedit: return
        glDisable(GL_LIGHTING); glLineWidth(1.2)
        try:
            slo, shi, _, _ = s.cam_stock(); glColor4f(0.9, 0.55, 0.1, 0.8); glEnable(GL_LINE_STIPPLE); glLineStipple(2, 0x0F0F)
            xs, ys, zs = (slo.x, shi.x), (slo.y, shi.y), (slo.z, shi.z)
            glBegin(GL_LINES)
            for a in range(2):
                for b in range(2):
                    glVertex3f(xs[0], ys[a], zs[b]); glVertex3f(xs[1], ys[a], zs[b]); glVertex3f(xs[a], ys[0], zs[b]); glVertex3f(xs[a], ys[1], zs[b])
                    glVertex3f(xs[a], ys[b], zs[0]); glVertex3f(xs[a], ys[b], zs[1])
            glEnd(); glDisable(GL_LINE_STIPPLE)
            w = s.cam_wcs(); L = s.span()*0.06
            glLineWidth(2.5); glBegin(GL_LINES)
            for d, c in ((V(1, 0, 0), (0.9, 0.2, 0.2)), (V(0, 1, 0), (0.2, 0.7, 0.2)), (V(0, 0, 1), (0.2, 0.4, 0.9))):
                glColor3f(*c); glVertex3f(*w.t()); glVertex3f(*(w + d*L).t())
            glEnd()
        except Exception: pass
        glLineWidth(1.2)
        for i, moves in s.cam_paths.items():
            if i >= len(s.cam["ops"]) or not s.cam["ops"][i].get("visible", True): continue
            prev = None; glBegin(GL_LINES)
            for m in moves:
                if prev is not None:
                    if m[0] == "rapid": glColor4f(0.85, 0.15, 0.15, 0.55)
                    else: glColor4f(0.05, 0.35, 0.85, 0.95)
                    glVertex3f(*prev); glVertex3f(*m[1])
                prev = m[1]
            glEnd()
        if s.cam_sim:                                               # tool marker for the simulation
            p, R = s.cam_sim["p"], s.cam_sim["r"]; glColor4f(0.95, 0.75, 0.2, 0.9); glBegin(GL_LINES)
            for k in range(16):
                a, b = 2*math.pi*k/16, 2*math.pi*(k + 1)/16
                for z in (p[2], p[2] + R*6):
                    glVertex3f(p[0] + R*math.cos(a), p[1] + R*math.sin(a), z); glVertex3f(p[0] + R*math.cos(b), p[1] + R*math.sin(b), z)
                glVertex3f(p[0] + R*math.cos(a), p[1] + R*math.sin(a), p[2]); glVertex3f(p[0] + R*math.cos(a), p[1] + R*math.sin(a), p[2] + R*6)
            glEnd()
        glEnable(GL_LIGHTING)

def post_gcode(cam, paths, wcs, flavor="grbl", name="fission"):
    """G-code for every visible operation, in order. flavor: "grbl" (no tool changes / canned cycles) or "generic"."""
    f3 = lambda v: f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) > 1e-9 else "0"
    L = [f"({name} - Fission CAM)", "G21 G90 G94 G17" + (" G40 G49 G80" if flavor == "generic" else "")]
    cur_tool = None; ox, oy, oz = wcs.t()
    for i, op in enumerate(cam["ops"]):
        moves = paths.get(i) or []
        if not moves: continue
        tool = next((x for x in cam["tools"] if x["n"] == int(op.get("tool", 1))), None)
        L.append(f"({op.get('name') or op['t']} - T{tool['n']} {tool['name']})")
        if tool["n"] != cur_tool:
            if flavor == "generic": L += ["M5", f"T{tool['n']} M6", f"G43 H{tool['n']}"]
            else:
                L.append(f"(tool change: T{tool['n']} {tool['name']})")
                if cur_tool is not None: L += ["M5", "M0 (change tool, then resume)"]
            cur_tool = tool["n"]
        L.append(f"S{int(tool['rpm'])} M3")
        lastf = None; lastp = None
        for m in moves:
            x, y, z = m[1][0] - ox, m[1][1] - oy, m[1][2] - oz
            if lastp is not None and max(abs(x - lastp[0]), abs(y - lastp[1]), abs(z - lastp[2])) < 1e-6: continue
            if m[0] == "rapid": L.append(f"G0 X{f3(x)} Y{f3(y)} Z{f3(z)}")
            else:
                fs = "" if m[2] == lastf else f" F{f3(m[2])}"; lastf = m[2]; L.append(f"G1 X{f3(x)} Y{f3(y)} Z{f3(z)}{fs}")
            lastp = (x, y, z)
    L += ["M5", "G0 Z" + f3(cam["setup"]["safe"] + 0.0), "M30" if flavor == "generic" else "M2", ""]
    return "\n".join(L)

class Viewport(CamMixin, RenderMixin, PlasticMixin, SheetMixin, AssemblyMixin, InspectMixin, SketchMixin, SolidMixin, QOpenGLWidget):
    changed = C.Signal()
    sk_mode = C.Signal(bool)
    dim_place = C.Signal(object)
    dim_edit = C.Signal(int)
    text_dialog = C.Signal(object)
    corner_dialog = C.Signal(str, object)
    type_dialog = C.Signal(str)
    calib_done = C.Signal(object)
    ext_moved = C.Signal(float)
    move_moved = C.Signal()
    cmd_done = C.Signal(bool)
    cmd_key = C.Signal(bool)
    axis_picked = C.Signal(int)
    msg = C.Signal(str)

    def __init__(s):
        super().__init__()
        s.yaw, s.pitch, s.dist, s.target = -0.8, 0.5, 120.0, V(0, 0, 0)
        s.bodies, s.sketches, s.active, s.features = [], [], None, []
        s.pts, s.sel, s.sel_face, s.sel_body = [], set(), None, None
        s.hover, s.hover_edge, s.hover_reg, s.hover_plane, s.hover_label = None, None, None, None, None
        s.tool, s.cur, s.moved, s.ext, s.rev, s.mv = None, V(0, 0, 0), False, None, None, None
        s.nav, s.show_grid, s.navbar, s._dls, s._edge_cache = None, True, None, {}, None
        s.mvp, s.last = None, C.QPointF()
        s.cons, s.cmd, s.cmd_hover, s.handle, s.preview = [], None, None, None, None
        s.sk_init(); s.palette = None; s.calib = None
        s.states, s.pos = [], 0
        s.states.append(s.snapshot("start"))
        s._anim, s._timer = None, C.QTimer(s); s._timer.timeout.connect(s._tick)
        s.setMouseTracking(True); s.setFocusPolicy(C.Qt.StrongFocus); s.cube_key = None; s.labels = []
        s.overlay = LabelOverlay(s); s.cube = ViewCube(s); s.insp_init(); s.asm_init()

    # ---- undo / redo history ----
    def snapshot(s, kind, op=None):
        sks = [k for k in s.sketches if not (k.pending and not k.entities)]     # an untouched face-click sketch isn't history
        return dict(kind=kind, op=op, bodies=list(s.bodies), sk=[k.snap() for k in sks], cons=list(s.cons),
                    act=sks.index(s.active) if s.active in sks else None, feats=list(s.features), canv=list(s.canv), asm=s.asm_state())
    def checkpoint(s, kind, op=None):
        del s.states[s.pos + 1:]; s.states.append(s.snapshot(kind, op)); s.pos = len(s.states) - 1; s.changed.emit()

    # ---- parametric operations: every timeline step is a re-runnable recipe, so old steps can be edited ----
    def do(s, op):
        """Run an operation on the model and record it as a timeline step."""
        if getattr(s, "active_comp", None) and "in_comp" not in op and op.get("t") not in ASM_OPS: op["in_comp"] = s.active_comp
        s.exec_op(op); s.checkpoint(op.get("icon", op["t"]), op)

    def chosen_regions(s, sk, sel):
        regs = sk.regions(); return [regs[i] for i in sel if i < len(regs)] or regs

    def revolve_shape(s, sk, sel, axis, angle):
        if axis in ("X", "Y"): o, d = V(), (V(1, 0, 0) if axis == "X" else V(0, 1, 0))
        else:
            segs = sketch_segments(sk)
            if axis >= len(segs): raise RuntimeError("the revolve axis line no longer exists")
            _, a, b = segs[axis]; o, d = a, (b - a).normalize()
        pl = sk.plane; ax = pl.u*d.x + pl.v*d.y
        return fuse_all([revolve(r.face, pl.w(o), ax, angle) for r in s.chosen_regions(sk, sel)])

    def exec_op(s, op):
        if op.get("suppressed"): return
        if op.get("_x"): op = s.eval_op_exprs(op)
        before = [getattr(b, "comp", None) for b in s.bodies]
        s.exec_op_core(op)
        if s.comps: s.after_op_comps(op, before)
    def exec_op_core(s, op):
        t = op["t"]
        if s.exec_sketch_op(op): return
        if s.exec_solid(op): return
        if t == "sketch":                                                # v0.5 sketch (geometry came as separate steps)
            plane, parent = s.op_plane(op)
            s.active = Sketch(plane, parent); s.sketches.append(s.active)
        elif t in ("ent", "entset"):
            sk = s.sketches[op["sk"]]
            if sk.geo.C and sk.lgeo is not sk.geo:                       # sketch already edited the new way: just add the shapes
                g = sk.geo.copy()
                for kind, pts in ([op["ent"]] if t == "ent" else op["ents"]): geo_add_legacy(g, kind, pts)
                geo_solve(g); sk.geo = g; s.active = sk
            else:
                ents = [L[:2] for L in sk.legacy] + [tuple(op["ent"])] if t == "ent" else [tuple(e) for e in op["ents"]]
                g, legacy = legacy_sketch_geo(ents); geo_solve(g); sk.geo = g; sk.legacy = legacy; sk.lgeo = g; s.active = sk
        elif t in ("extrude", "revolve"):
            sk = s.sketches[op["sk"]]
            if t == "extrude":
                tool = fuse_all([prism(r.face, sk.plane.n, op["d"]) for r in s.chosen_regions(sk, op["sel"])])
                res = s.apply(tool, op["op"], op["d"] > 0, sk.parent)
            else:
                tool = s.revolve_shape(sk, op["sel"], op["axis"], op["angle"])
                o = op["op"] if op["op"] != "Auto" else ("Join" if sk.parent is not None else "New")
                res = s.apply(tool, o, True, sk.parent)
            s.features.append(dict(name=f"{t.title()} ({res.lower()})", tool=tool, op=res, plane=sk.plane))
            sk.visible, s.active = False, None
        elif t == "move":
            b = s.bodies[op["bi"]]
            nb = xform_body(b, move_matrix((b.lo + b.hi)*0.5, V(*op["d"]), op["rot"]))
            if op["copy"]: s.bodies.append(nb)
            else: s.bodies[op["bi"]] = nb
        elif t in ("fillet", "chamfer"):
            new = list(s.bodies)
            for bi, eis in op["groups"]:
                body = s.bodies[bi]
                if any(e >= len(body.eds) for e in eis): raise RuntimeError("the selected edges no longer exist")
                edges = [body.eds[e] for e in eis]
                if t == "fillet": sh = fillet(body.shape, edges, op["r"], op["r2"] or None)
                else:
                    faces = [body.faces_of_edge(e)[0] for e in edges] if op["mode"] != "eq" else None
                    sh = chamfer(body.shape, edges, op["d"], op["mode"], op["angle"], op["d2"], faces)
                new[bi] = Body(sh)
            s.bodies = new
        elif t == "pattern":
            n, ang = op["n"], op["ang"]; step = math.radians(ang/n if abs(abs(ang) - 360) < 1e-6 else ang/max(n - 1, 1))
            kind, idx = op["obj"]
            if kind == "ent":
                sk = s.sketches[op["sk"]]; g = sk.geo.copy()
                cs = sk.legacy[idx][2] if idx < len(sk.legacy) else [idx]
                for k in range(1, n):
                    M = mat_rot(op["cu"], op["cv"], math.degrees(step*k)); _, nn = geo_copy_curves(g, cs, mat_fn(M))
                g.touch(); sk.geo = g; s.active = sk
            else:
                pl = s.features[idx]["plane"] if kind == "feat" else XY
                org = pl.w(V(op["cu"], op["cv"], 0)); src = s.features[idx]["tool"] if kind == "feat" else s.bodies[idx].shape
                mode = s.features[idx]["op"] if kind == "feat" else "New"
                for k in range(1, n):
                    tool = transformed(src, rot_matrix(org, pl.n, step*k))
                    if mode == "Cut": s.cut_all(tool)
                    elif mode == "Join": s.apply(tool, "Join", True, None)
                    else: s.bodies += [Body(x) for x in solids(tool)]
        elif t == "gear":
            s.bodies.append(Body(spur_gear(op["m"], op["z"], op["a"], op["th"], op["bore"])))
        elif t == "thread":
            if op.get("rod"):
                p = op["p"] or iso_suggest(op["d"])[1]; R = op["d"]/2
                rod = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), R, op["len"]).Shape()
                tool = thread_tool(V(0, 0, 0), V(0, 0, 1), V(1, 0, 0), R, p, op["len"], True)
                s.bodies += [Body(x) for x in solids(clean(boolean(rod, tool, "cut")))]
            else:
                bi, fid = op["face"]; cyl = s.bodies[bi].face_cylinder(fid)
                if not cyl: raise RuntimeError("the threaded face is no longer a cylinder")
                o, a, xd, R, v0, v1, ext = cyl; start = o + a*(v0 + op["off"])
                tool = thread_tool(start, a, xd, R, op["p"], min(op["len"], v1 - v0 - op["off"]), ext)
                s.replace_body(bi, boolean(s.bodies[bi].shape, tool, "cut"))
        elif t == "delete":
            del s.bodies[op["bi"]]
        else:
            raise RuntimeError(f"unknown step type {t}")

    # ---------------- timeline rebuilds (0.7): keep going past failing steps ----------------
    def rebuild_steps(s, i, ops, kinds, strict=None):
        """Re-run ops[i:] on top of state i-1. A step that fails keeps its place (marked with its error) and leaves the
        model as it was; later steps still run. strict = a step index whose failure aborts (raises) instead."""
        old = s.states; s.restore(i - 1); new = old[:i]; fails = []
        for j in range(i, len(ops)):
            op = ops[j]
            try:
                s.exec_op(op); stt = s.snapshot(kinds[j], op)
                if fails and not (op or {}).get("suppressed") and not (op or {}).get("quiet"): stt["warn"] = f"after the failed step {fails[0][0]}"
            except Exception as ex:
                if j == strict: s.states = old; raise
                s.states = new; s.restore(len(new) - 1); s.states = old
                stt = s.snapshot(kinds[j], op); stt["err"] = str(ex) or type(ex).__name__; fails.append((j, kinds[j], ex))
            new.append(stt)
        s.last_fails = fails
        return new, fails

    def replay_edit(s, i, new_op, strict=True):
        """Give step i new parameters and recompute every step after it (also the undone ones).
        The edited step must work (else nothing changes); later steps that break are kept and flagged."""
        old_states, old_pos = s.states, s.pos
        ops = [stt["op"] for stt in old_states]; ops[i] = new_op
        bad = next((j for j in range(i, len(ops)) if ops[j] is None), None)
        if bad is not None: raise RuntimeError(f"step {bad} can't be recomputed")
        try: new_states, fails = s.rebuild_steps(i, ops, [x["kind"] for x in old_states], i if strict else None)
        except Exception as ex:
            traceback.print_exc(); s.states = old_states; s.restore(old_pos)
            raise RuntimeError(f"Step {i} ({old_states[i]['kind']}) could not be rebuilt with the new value:\n{ex}\n\nNothing was changed.")
        s.states = new_states; s.restore(old_pos); return fails

    def replay_ops(s, i, ops, kinds, pos, remap=None):
        """Replace the history from step i on with new ops (reorder / remove / suppress), keeping failures flagged."""
        if any(o is None for o in ops[i:]): raise RuntimeError("a later step can't be recomputed")
        old_states = s.states
        try: new_states, fails = s.rebuild_steps(i, ops, kinds)
        except Exception: s.states = old_states; raise
        s.states = new_states
        if remap: s.tl_remap(remap)
        s.restore(max(0, min(pos, len(new_states) - 1))); return fails

    def replay_remove(s, i):
        """Delete step i and recompute every later step."""
        ops = [x["op"] for x in s.states]; kinds = [x["kind"] for x in s.states]; pos = s.pos
        del ops[i]; del kinds[i]
        return s.replay_ops(i, ops, kinds, pos if pos < i else pos - 1, {j: (j if j < i else None if j == i else j - 1) for j in range(len(s.states))})

    def move_step(s, i, k):
        """Move step i so it becomes step k; everything from the earlier of the two is recomputed."""
        n = len(s.states)
        if not (1 <= i < n and 1 <= k < n) or i == k: return []
        ops = [x["op"] for x in s.states]; kinds = [x["kind"] for x in s.states]
        order = list(range(n)); order.insert(k, order.pop(i))
        remap = {old: new for new, old in enumerate(order)}
        pos = n - 1 if s.pos == n - 1 else remap.get(s.pos, s.pos)
        return s.replay_ops(min(i, k), [ops[j] for j in order], [kinds[j] for j in order], pos, remap)

    def set_suppressed(s, i, on):
        op = dict(s.states[i]["op"]); op["suppressed"] = bool(on)
        if not on: op.pop("suppressed")
        ops = [x["op"] for x in s.states]; ops[i] = op
        return s.replay_ops(i, ops, [x["kind"] for x in s.states], s.pos)

    def rename_step(s, i, name):
        op = dict(s.states[i]["op"])
        if name: op["label"] = name
        else: op.pop("label", None)
        s.states[i] = dict(s.states[i], op=op); s.tl_ver = getattr(s, "tl_ver", 0) + 1; s.changed.emit()

    def load_steps(s, steps, pos):
        """Rebuild a saved design by re-running its steps. Returns the failed steps [(step, kind, error)] (empty if none);
        failed steps stay in the timeline, flagged, and the steps after them still run."""
        s.reset()
        ops = [None] + [stp["op"] for stp in steps]; kinds = ["start"] + [stp["kind"] for stp in steps]
        new, fails = s.rebuild_steps(1, ops, kinds) if steps else ([s.states[0]], [])
        s.states = new; s.restore(min(pos, len(s.states) - 1)); return fails

    # ---- folders (groups of neighbouring steps) ----
    def tl_remap(s, m):
        out = []
        for g in getattr(s, "tl_groups", []):
            idx = [m.get(j) for j in range(g["a"], g["b"] + 1)]; idx = [j for j in idx if j is not None]
            if len(idx) >= 1: out.append(dict(g, a=min(idx), b=max(idx)))
        s.tl_groups = out

    def tl_clean_groups(s):
        n = len(s.states); out = []
        for g in getattr(s, "tl_groups", []):
            b = min(g["b"], n - 1)
            if 1 <= g["a"] <= b: out.append(dict(g, b=b))
        s.tl_groups = out; return out

    def group_of(s, i):
        return next((g for g in getattr(s, "tl_groups", []) if g["a"] <= i <= g["b"]), None)

    def eval_op_exprs(s, op):
        """Feature values typed as expressions (parameter names, other dimensions) are re-evaluated on every rebuild."""
        op = dict(op); env = s.dim_env()
        for k, (text, kind) in op["_x"].items():
            try: val = eval_expr(text, env, kind)
            except ExprError as ex: raise RuntimeError(f"{k} = {text}: {ex}")
            if k in op: op[k] = val
            elif isinstance(op.get("vals"), dict) and k in op["vals"]: op["vals"] = dict(op["vals"]); op["vals"][k] = val
        return op

    def reset(s):
        if s.skedit: s.skedit = None; s.tool = None; s.sk_mode.emit(False)
        s.bodies, s.sketches, s.features, s.active, s.cons, s.canv = [], [], [], None, [], []
        s.asm_init()
        s.ext = s.rev = s.mv = None; s.tool, s.pts = None, []
        s.states, s.pos = [], 0; s.states.append(s.snapshot("start")); s.insp_reset(); s.restore(0)
    def drop_pending(s):
        """Forget a face-click sketch nothing was drawn on."""
        for k in [k for k in s.sketches if k.pending and not k.entities]:
            s.sketches.remove(k)
            if s.active is k: s.active = None
    def restore(s, i):
        i = max(0, min(len(s.states) - 1, i)); stt = s.states[i]; s.pos = i
        s.bodies, s.features, s.cons, s.canv = list(stt["bodies"]), list(stt["feats"]), list(stt.get("cons", [])), list(stt.get("canv", []))
        s.asm_apply(stt.get("asm", {}))
        s.preview = s.handle = s.cmd_hover = None
        s.sketches = [Sketch.restore(t) for t in stt["sk"]]
        s.active = s.sketches[stt["act"]] if stt["act"] is not None else None
        s.ext = s.rev = s.mv = None; s.tool, s.pts = None, []
        s.sel.clear(); s.sel_face = s.sel_body = s.hover = s.hover_edge = s.hover_reg = None
        s.changed.emit(); s.update()
    def undo(s):
        if s.pts: s.pts.pop(); s.update(); return
        if s.pos > 0: s.restore(s.pos - 1)
    def redo(s):
        if s.pos < len(s.states) - 1: s.restore(s.pos + 1)

    # ---- camera ----
    def eye(s):
        cp = math.cos(s.pitch)
        return s.target + V(cp*math.cos(s.yaw), cp*math.sin(s.yaw), math.sin(s.pitch))*s.dist
    def animate_to(s, yaw, pitch, ms=300):
        dy = (yaw - s.yaw + math.pi) % math.tau - math.pi
        s._anim = (s.yaw, s.pitch, s.yaw + dy, max(-1.55, min(1.55, pitch)), time.monotonic(), ms/1000)
        s._timer.start(15)
    def _tick(s):
        if not s._anim: s._timer.stop(); return
        y0, p0, y1, p1, t0, T = s._anim; u = min(1.0, (time.monotonic() - t0)/T); e = u*u*(3 - 2*u)
        s.yaw, s.pitch = y0 + (y1 - y0)*e, p0 + (p1 - p0)*e; s.update()
        if u >= 1: s._anim = None; s._timer.stop()
    def snapped_yaw(s):
        y = s._anim[2] if s._anim else s.yaw
        return round(y/(math.pi/2))*(math.pi/2)
    def turn(s, direction):
        s.animate_to(s.snapped_yaw() + direction*math.pi/2, s._anim[3] if s._anim else s.pitch)
    def look_at_plane(s, n):
        if abs(n.z) > 0.99: s.animate_to(-math.pi/2 if n.z > 0 else math.pi/2, 1.5 if n.z > 0 else -1.5)
        else: s.animate_to(math.atan2(n.y, n.x), max(-1.5, min(1.5, math.asin(max(-1, min(1, n.z))))))
    def set_view(s, yaw, pitch): s.animate_to(yaw, pitch)
    def fit(s):
        lo, hi = [], []
        for b in s.bodies:
            if b.visible and len(b.tv): lo.append(b.tv.reshape(-1, 3).min(0)); hi.append(b.tv.reshape(-1, 3).max(0))
        for sk in s.sketches:
            if not sk.visible: continue
            P = [sk.plane.w(V(q[0], q[1], 0)).t() for ci in range(len(sk.geo.C)) for A in geo_curve_pts(sk.geo, ci) for q in A]
            if P: lo.append(np.min(P, 0)); hi.append(np.max(P, 0))
        if not lo: s.target, s.dist = V(0, 0, 0), 120.0
        else:
            a, b = V(*np.min(lo, 0)), V(*np.max(hi, 0))
            s.target = (a + b)*0.5; s.dist = max(10.0, (b - a).Length*0.5/math.sin(math.radians(22.5))*1.15)
        s.update()
    def home(s):
        if s.home_cam: s.go_cam(s.home_cam); return
        s.target = V(); s.fit(); s.animate_to(-0.8, 0.5)

    def grid_levels(s):
        """Grid squares come in powers of 5 of the base unit (1 mm or 1 inch): each square is split into 5 smaller ones."""
        base = 25.4 if DISPLAY["unit"] in ("in", "ft") else 1.0
        x = math.log(max(s.dist, 1e-6)/(20*base), 5); k0 = math.floor(x)
        return base, k0, x - k0
    def snap_step(s):
        base, k0, t = s.grid_levels(); return base*5.0**(k0 + (0 if t < 0.6 else 1))

    # ---- drawing ----
    def paintGL(s):
        try: s.paint3d()
        except Exception: traceback.print_exc()
        try: s.labels = s.compute_labels()
        except Exception: traceback.print_exc(); s.labels = []
        s.overlay.update()
        if s.cube and s.cube_key != (s.yaw, s.pitch): s.cube_key = (s.yaw, s.pitch); s.cube.update()

    def paint3d(s):
        glClearColor(*BG_BOT, 1); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING); glDisable(GL_CULL_FACE); glDisable(GL_TEXTURE_2D); glUseProgram(0)
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glBegin(GL_QUADS)
        glColor3f(*BG_TOP); glVertex2f(-1, 1); glVertex2f(1, 1)
        glColor3f(*BG_BOT); glVertex2f(1, -1); glVertex2f(-1, -1)
        glEnd()
        glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL); glDepthMask(GL_TRUE)
        vp = glGetIntegerv(GL_VIEWPORT)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        s.apply_projection(vp)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1); glEnable(GL_COLOR_MATERIAL); glEnable(GL_NORMALIZE)
        e, t = s.eye(), s.target
        gluLookAt(e.x, e.y, e.z, t.x, t.y, t.z, 0, 0, 1)
        s.mv_m, s.pj, s.vp = glGetDoublev(GL_MODELVIEW_MATRIX), glGetDoublev(GL_PROJECTION_MATRIX), vp
        s.mvp = np.array(s.pj).reshape(4, 4).T @ np.array(s.mv_m).reshape(4, 4).T
        # studio lighting, fixed in the world (set after gluLookAt): bright key from above, cool fill, soft ambient
        glLightfv(GL_LIGHT0, GL_POSITION, (*LIGHT, 0)); glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.72, 0.71, 0.69, 1)); glLightfv(GL_LIGHT0, GL_SPECULAR, (0.45, 0.45, 0.45, 1))
        glLightfv(GL_LIGHT1, GL_POSITION, (0.4, -0.7, 0.35, 0)); glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.36, 0.36, 0.38, 1)); glLightfv(GL_LIGHT1, GL_SPECULAR, (0, 0, 0, 1))
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.36, 0.36, 0.37, 1))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.28, 0.28, 0.28, 1)); glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 28)
        s.gc_lists()
        glEnable(GL_BLEND); glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)   # keep the framebuffer opaque
        if s.show_grid and not (s.skedit and s.esk and s.esk.plane.is_ground() and not s.sk_opts["grid"]): s.draw_grid(XY, True)
        s.draw_shadow(); s.draw_origin()
        moving = s.mv["bi"] if s.mv and not s.mv["copy"] else None
        shown = s.preview["bodies"] if s.preview else s.bodies; olds = {id(b) for b in s.bodies}
        if s.preview: moving = None
        clip = s.skedit and s.esk and s.sk_opts["slice"]
        if clip:                                                         # Slice: hide what is in front of the sketch plane
            pl = s.esk.plane; sg = 1 if s.eye().dot(pl.n) - pl.o.dot(pl.n) >= 0 else -1
            glClipPlane(GL_CLIP_PLANE0, (-sg*pl.n.x, -sg*pl.n.y, -sg*pl.n.z, sg*pl.n.dot(pl.o) + 1e-6*s.dist)); glEnable(GL_CLIP_PLANE0)
        glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_LIGHT1); glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(1, 1)
        s.draw_body_fills(shown, moving, olds)
        glPolygonOffset(-0.5, -0.5)
        for b in shown:
            if not b.visible or not b.face_colors: continue
            for fid, col in b.face_colors.items():
                m = b.tri_face == fid
                if not m.any() or col is None: continue
                glColor3f(*col); vt = np.ascontiguousarray(b.tv[m].reshape(-1, 3), dtype=np.float32); nt = np.ascontiguousarray(b.vn[m].reshape(-1, 3), dtype=np.float32)
                glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY); glVertexPointer(3, GL_FLOAT, 0, vt); glNormalPointer(GL_FLOAT, 0, nt)
                glDrawArrays(GL_TRIANGLES, 0, len(vt)); glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_VERTEX_ARRAY)
        glPolygonOffset(1, 1); glColor3f(*BODY)
        if s.mv:
            glPushMatrix(); glMultMatrixd(s.move_gl()); glCallList(s.lists(s.bodies[s.mv["bi"]])[0]); glPopMatrix()
        glDisable(GL_POLYGON_OFFSET_FILL); glDisable(GL_LIGHTING)
        s.draw_face_tints(moving)
        glEnable(GL_LINE_SMOOTH)
        s.draw_body_edges(shown, moving)
        if s.mv:
            glPushMatrix(); glMultMatrixd(s.move_gl()); glCallList(s.lists(s.bodies[s.mv["bi"]])[1]); glPopMatrix()
        glDisable(GL_CLIP_PLANE0)
        glColor3f(*SEL); glLineWidth(3.2)
        for bi, ei in s.sel:                                           # selected edges drawn heavier on top
            if bi < len(s.bodies) and s.bodies[bi].visible and ei < len(s.bodies[bi].edges): s.strip(s.bodies[bi].edges[ei])
        if s.hover_edge and s.hover_edge not in s.sel:
            bi, ei = s.hover_edge
            if bi < len(s.bodies) and ei < len(s.bodies[bi].edges):
                glColor4f(*HILITE, 0.9); glLineWidth(3.2); s.strip(s.bodies[bi].edges[ei])
        if s.rev and s.rev.get("body"): s.draw_preview_body(s.rev["body"], s.rev["op"] == "Cut")
        if s.tool == "pick": s.draw_origin_planes()
        if s.skedit and s.esk and s.show_grid and s.sk_opts["grid"] and not s.esk.plane.is_ground():
            s.draw_grid(s.esk.plane, False)
        s.draw_canvases()
        s.draw_sketches(); s.draw_cons()
        if s.preview and len(s.preview.get("cons", [])) > len(s.cons): s.draw_cons(s.preview["cons"][len(s.cons):], True)
        if s.preview:                                                    # sketch geometry a command would add / change
            glDisable(GL_DEPTH_TEST); glColor4f(*SEL, 0.95); glLineWidth(2.0); glEnable(GL_LINE_STIPPLE); glLineStipple(2, 0xAAAA)
            for k, g, g0 in s.preview.get("geos", []):
                if k not in s.sketches: continue
                glPushMatrix(); glMultMatrixd(k.plane.gl())
                for ci in range(len(g.C)):
                    if g.C[ci]["k"] == "point": continue
                    if ci < len(g0.C) and g.C[ci] == g0.C[ci] and all(g.P[i] == g0.P[i] for i in crv_points(g.C[ci]) if i < len(g0.P)): continue
                    for A in geo_polys(g, ci): s.sk_glline(A)
                glPopMatrix()
            glDisable(GL_LINE_STIPPLE); glEnable(GL_DEPTH_TEST)
        s.draw_inspect_3d(); s.draw_joints(); s.draw_bend_lines(); s.draw_cam()
        s.draw_cmd_overlay()
        if s.mv: s.draw_triad()

    def strip(s, pl):
        glBegin(GL_LINE_STRIP)
        for q in pl: glVertex3f(*q)
        glEnd()

    def lists(s, b):
        """Compile a body's smooth-shaded triangles and edge lines into GL display lists once."""
        if b.dl is None:
            t = glGenLists(2); b.dl = (t, t + 1); s._dls[t] = b
            vtx = np.ascontiguousarray(b.tv.reshape(-1, 3), dtype=np.float32); nrm = np.ascontiguousarray(b.vn.reshape(-1, 3), dtype=np.float32)
            glNewList(t, GL_COMPILE)
            if len(vtx):
                glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, vtx); glNormalPointer(GL_FLOAT, 0, nrm)
                glDrawArrays(GL_TRIANGLES, 0, len(vtx))
                glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_VERTEX_ARRAY)
            glEndList()
            seg = np.ascontiguousarray(np.stack([b.sa, b.sb], 1).reshape(-1, 3), dtype=np.float32) if len(b.sa) else None
            glNewList(t + 1, GL_COMPILE)
            if seg is not None:
                glEnableClientState(GL_VERTEX_ARRAY); glVertexPointer(3, GL_FLOAT, 0, seg)
                glDrawArrays(GL_LINES, 0, len(seg)); glDisableClientState(GL_VERTEX_ARRAY)
            glEndList()
        return b.dl

    def gc_lists(s):
        """Free display lists of bodies that are no longer shown (undo history keeps the Body, so just reset its cache)."""
        live = {id(b) for b in s.bodies} | ({id(s.rev["body"])} if s.rev and s.rev.get("body") else set()) | \
               ({id(b) for b in s.preview["bodies"]} if s.preview else set())
        for t, b in list(s._dls.items()):
            if id(b) not in live: glDeleteLists(t, 2); b.dl = None; del s._dls[t]

    def tris_array(s, arr):
        a = np.ascontiguousarray(arr.reshape(-1, 3), dtype=np.float32)
        if not len(a): return
        glEnableClientState(GL_VERTEX_ARRAY); glVertexPointer(3, GL_FLOAT, 0, a)
        glDrawArrays(GL_TRIANGLES, 0, len(a)); glDisableClientState(GL_VERTEX_ARRAY)

    def draw_face_tints(s, skip):
        """Hovered / selected faces: a translucent blue film over the face."""
        glDepthMask(GL_FALSE); glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-2, -2)
        for key, al in ((s.hover, 0.18), (s.sel_face, 0.32)):
            if not key or key[0] >= len(s.bodies) or not s.bodies[key[0]].visible or key[0] == skip: continue
            if key is s.hover and s.sel_face and key[:2] == s.sel_face[:2]: continue
            b = s.bodies[key[0]]; glColor4f(*HILITE, al); s.tris_array(b.tv[b.tri_face == key[1]])
        if s.sel_body is not None and s.sel_body < len(s.bodies) and not s.sel_face and not s.sel and s.sel_body != skip:
            b = s.bodies[s.sel_body]
            if b.visible: glColor4f(*HILITE, 0.14); s.tris_array(b.tv)
        glDisable(GL_POLYGON_OFFSET_FILL); glDepthMask(GL_TRUE)

    def draw_preview_body(s, b, cut):
        glDepthMask(GL_FALSE); glEnable(GL_LIGHTING); glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-1, -1)
        glColor4f(*((0.95, 0.45, 0.2) if cut else (0.25, 0.55, 1.0)), 0.45); glCallList(s.lists(b)[0])
        glDisable(GL_LIGHTING); glDisable(GL_POLYGON_OFFSET_FILL)
        glColor4f(*SKETCH, 0.8); glLineWidth(1.4); glCallList(s.lists(b)[1]); glDepthMask(GL_TRUE)

    def draw_shadow(s):
        """Soft gray ground shadow: flatten the cached triangles onto the model's floor along the key-light direction.
        The stencil test makes overlapping triangles darken each pixel only once."""
        vis = [b for b in s.bodies if b.visible]
        if not vis: return
        h, (lx, ly, lz) = min(b.zmin for b in vis), LIGHT
        glPushMatrix(); glMultMatrixd([1, 0, 0, 0,  0, 1, 0, 0,  -lx/lz, -ly/lz, 0, 0,  lx*h/lz, ly*h/lz, h, 1])
        glDepthMask(GL_FALSE); glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-2, -2)
        glEnable(GL_STENCIL_TEST); glClear(GL_STENCIL_BUFFER_BIT); glStencilFunc(GL_EQUAL, 0, 0xFF); glStencilOp(GL_KEEP, GL_KEEP, GL_INCR)
        glColor4f(0.30, 0.33, 0.38, 0.24)
        for b in vis: glCallList(s.lists(b)[0])
        glDisable(GL_STENCIL_TEST); glDisable(GL_POLYGON_OFFSET_FILL); glDepthMask(GL_TRUE); glPopMatrix()

    def draw_grid(s, plane, ground):
        """Endless-looking grid: three levels of lines centred under the camera target, fading out with distance;
        each level splits a square into 5 and fades in / out smoothly as you zoom so detail grows as you get closer."""
        c = plane.l(s.target); cx, cy = c.x, c.y
        base, k0, t = s.grid_levels()
        glDepthMask(GL_FALSE); glPushMatrix(); glMultMatrixd(plane.gl()); glLineWidth(1)
        col = (0.42, 0.50, 0.58) if ground else (0.25, 0.45, 0.75)
        glBegin(GL_LINES)
        for tier in range(3):
            step = base*5.0**(k0 + tier); a = min(0.5, 0.3*(tier - t + 1))*(1.0 if ground else 0.8)
            if a <= 0.004: continue
            R = step*(22 if ground else 12)
            for axis_c, other_c, horiz in ((cx, cy, False), (cy, cx, True)):
                for i in range(math.ceil((axis_c - R)/step), math.floor((axis_c + R)/step) + 1):
                    if tier < 2 and i % 5 == 0: continue                          # drawn by the coarser level
                    g = i*step; off = abs(g - axis_c)
                    if off >= R: continue
                    h = math.sqrt(R*R - off*off); al = a*(1 - off/R)
                    if ground and i == 0: continue                                # axes are drawn below
                    pts = ((g, other_c - h), (g, other_c), (g, other_c + h)) if not horiz else ((other_c - h, g), (other_c, g), (other_c + h, g))
                    for (p0, a0), (p1, a1) in (((pts[0], 0), (pts[1], al)), ((pts[1], al), (pts[2], 0))):
                        glColor4f(*col, a0); glVertex3f(p0[0], p0[1], 0); glColor4f(*col, a1); glVertex3f(p1[0], p1[1], 0)
        if ground:                                                               # X red, Y green, Z blue
            R = base*5.0**(k0 + 2)*22
            for (dx, dy), rgb, cc in (((1, 0), (0.85, 0.2, 0.2), cx), ((0, 1), (0.3, 0.7, 0.3), cy)):
                glColor4f(*rgb, 0); glVertex3f(dx*(cc - R), dy*(cc - R), 0); glColor4f(*rgb, 0.85); glVertex3f(dx*cc, dy*cc, 0)
                glColor4f(*rgb, 0.85); glVertex3f(dx*cc, dy*cc, 0); glColor4f(*rgb, 0); glVertex3f(dx*(cc + R), dy*(cc + R), 0)
            glColor4f(0.2, 0.4, 0.9, 0.85); glVertex3f(0, 0, 0); glColor4f(0.2, 0.4, 0.9, 0); glVertex3f(0, 0, s.dist*1.5)
        glEnd(); glPopMatrix(); glDepthMask(GL_TRUE)

    def draw_origin(s):
        """Small origin marker lying on the ground plane (keeps the same size on screen as you zoom)."""
        r = s.dist*0.009; n = 32
        glDepthMask(GL_FALSE); glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-3, -3)
        glColor4f(0.66, 0.69, 0.73, 0.9); glBegin(GL_TRIANGLE_FAN); glVertex3f(0, 0, 0)
        for i in range(n + 1): glVertex3f(r*math.cos(i*math.tau/n), r*math.sin(i*math.tau/n), 0)
        glEnd(); glDisable(GL_POLYGON_OFFSET_FILL)
        glColor4f(0.42, 0.45, 0.5, 0.95); glLineWidth(1.4); glBegin(GL_LINE_LOOP)
        for i in range(n): glVertex3f(r*math.cos(i*math.tau/n), r*math.sin(i*math.tau/n), 0)
        glEnd(); glPointSize(4); glBegin(GL_POINTS); glVertex3f(0, 0, 0); glEnd(); glDepthMask(GL_TRUE)

    def plane_size(s): return s.dist*0.2

    def draw_origin_planes(s):
        S = s.plane_size(); glDepthMask(GL_FALSE)
        for name, pl, rgb in ORIGIN_PLANES:
            hot = name == s.hover_plane
            glPushMatrix(); glMultMatrixd(pl.gl())
            glColor4f(*rgb, 0.42 if hot else 0.16); glBegin(GL_QUADS)
            for x, y in ((0, 0), (S, 0), (S, S), (0, S)): glVertex3f(x, y, 0)
            glEnd(); glColor4f(*rgb, 0.95 if hot else 0.6); glLineWidth(2.2 if hot else 1.2); glBegin(GL_LINE_LOOP)
            for x, y in ((0, 0), (S, 0), (S, S), (0, S)): glVertex3f(x, y, 0)
            glEnd(); glPopMatrix()
        glDepthMask(GL_TRUE)

    def draw_sketches(s):
        glDisable(GL_DEPTH_TEST); smooth = glIsEnabled(GL_LINE_SMOOTH); glDisable(GL_LINE_SMOOTH)
        for sk in s.sketches:
            if not sk.visible and sk is not s.esk: continue
            try: s.sk_draw(sk, sk is s.esk)
            except Exception: traceback.print_exc()
            if sk is s.active and (s.ext or s.rev):
                glPushMatrix(); glMultMatrixd(sk.plane.gl())
                if s.ext: s.draw_ext()
                if s.rev: s.draw_rev_axis()
                glPopMatrix()
        if s.calib:
            pl = s.calib["plane"]; glPushMatrix(); glMultMatrixd(pl.gl()); glPointSize(10); glColor3f(0.85, 0.15, 0.15); glBegin(GL_POINTS)
            for q in s.calib["pts"]: glVertex3f(q[0], q[1], 0)
            glEnd(); glPopMatrix()
        if smooth: glEnable(GL_LINE_SMOOTH)
        glEnable(GL_DEPTH_TEST)

    def draw_sketches_old(s):
        glDisable(GL_DEPTH_TEST)
        for sk in s.sketches:
            if not sk.visible: continue
            on = sk is s.active
            glPushMatrix(); glMultMatrixd(sk.plane.gl())
            regs = sk.regions()
            for i, r in enumerate(regs):                                          # closed profiles: blue fill
                sel = i in sk.sel; hov = s.hover_reg == (sk, i)
                al = 0.5 if sel else 0.34 if hov else (0.2 if on else 0.12)
                if s.ext and on and not sk.sel: al = 0.32
                glColor4f(*((0.2, 0.5, 0.95) if sel else FILL), al); s.tris_array(r.tris)
            glColor3f(*(SKETCH if on else (0.45, 0.55, 0.7))); glLineWidth(2 if on else 1.5)
            for kind, pts in sk.entities: s.poly(loop(kind, pts), True)
            if on:
                if s.ext: s.draw_ext()
                if s.rev: s.draw_rev_axis()
                if s.tool in DRAW and s.pts:
                    pre = s.pts + [s.cur]; glColor3f(*SKETCH); glLineWidth(2)
                    s.poly(pre if s.tool == "poly" else loop(s.tool, pre), s.tool != "poly")
                if s.tool in DRAW:
                    glPointSize(9); glColor3f(*SEL); glBegin(GL_POINTS); glVertex3f(s.cur.x, s.cur.y, 0); glEnd()
            glPopMatrix()
        glEnable(GL_DEPTH_TEST)

    def poly(s, pts, closed):
        glBegin(GL_LINE_LOOP if closed else GL_LINE_STRIP)
        for p in pts: glVertex3f(p.x, p.y, p.z)
        glEnd()

    # ---- interactive extrude ----
    def ext_regions(s): return s.active.chosen() if s.active else []
    def draw_ext(s):
        d = s.ext["d"]; regs = s.ext_regions()
        glColor4f(*((0.95, 0.4, 0.2) if d < 0 else (0.2, 0.55, 1)), 0.4); glBegin(GL_QUADS)   # orange = cutting
        for r in regs:
            for pl in r.outline:
                for a, b in zip(pl[:-1], pl[1:]):
                    glVertex3f(a[0], a[1], 0); glVertex3f(b[0], b[1], 0); glVertex3f(b[0], b[1], d); glVertex3f(a[0], a[1], d)
        glEnd()
        for r in regs:
            top = r.tris.copy(); top[:, :, 2] = d; s.tris_array(top)
        glColor3f(*SKETCH); glLineWidth(2)
        for r in regs:
            for pl in r.outline:
                glBegin(GL_LINE_STRIP)
                for q in pl: glVertex3f(q[0], q[1], d)
                glEnd()
        a, b = s.arrow_ends(); h = b.z - a.z; rr = abs(h)*0.2
        glColor3f(*SEL); glLineWidth(4)
        glBegin(GL_LINES); glVertex3f(a.x, a.y, a.z); glVertex3f(a.x, a.y, a.z + h*0.45); glEnd()
        glBegin(GL_TRIANGLE_FAN); glVertex3f(b.x, b.y, b.z)
        for i in range(17): t = i*math.tau/16; glVertex3f(a.x + rr*math.cos(t), a.y + rr*math.sin(t), a.z + h*0.45)
        glEnd()

    def set_ext(s, d):
        if s.ext: s.ext["d"] = d; s.update()
    def ext_axis(s):
        pts = [q for r in s.ext_regions() for pl in r.outline for q in pl]
        if not pts: return V()
        m = np.mean(pts, 0); return V(m[0], m[1], 0)
    def arrow_ends(s):
        c, d = s.ext_axis(), s.ext["d"]
        return V(c.x, c.y, d), V(c.x, c.y, d + (1 if d >= 0 else -1)*s.dist*0.08)
    def arrow_hit(s, p):
        a, b = s.arrow_ends(); pl = s.plane
        return s.screen_seg_dist(p, pl.w(a), pl.w(b)) < 16
    def ray(s, p):
        o = s.unproject(p, 0); return o, s.unproject(p, 1) - o

    def rev_axis_local(s):
        ax = s.rev["axis"]
        if ax == "X": return V(0, 0, 0), V(1, 0, 0)
        if ax == "Y": return V(0, 0, 0), V(0, 1, 0)
        _, a, b = s.rev["segs"][ax]; return a, (b - a).normalize()
    def draw_rev_axis(s):
        o, d = s.rev_axis_local(); L = s.dist*3
        hot = s.rev.get("hot")
        if hot is not None and hot != s.rev["axis"]:                      # sketch line under the mouse
            _, a, b = s.rev["segs"][hot]; glColor4f(1.0, 0.6, 0.1, 0.9); glLineWidth(5)
            glBegin(GL_LINES); glVertex3f(a.x, a.y, 0); glVertex3f(b.x, b.y, 0); glEnd()
        glColor4f(0.85, 0.35, 0.1, 0.9); glLineWidth(1.6); glEnable(GL_LINE_STIPPLE); glLineStipple(3, 0xF0F0)
        glBegin(GL_LINES); glVertex3f(*(o - d*L).t()); glVertex3f(*(o + d*L).t()); glEnd(); glDisable(GL_LINE_STIPPLE)
        if isinstance(s.rev["axis"], int):
            _, a, b = s.rev["segs"][s.rev["axis"]]; glLineWidth(4)
            glBegin(GL_LINES); glVertex3f(a.x, a.y, 0); glVertex3f(b.x, b.y, 0); glEnd()
    def rev_seg_near(s, p):
        pl, best, bd = s.plane, None, 10
        for i, (_, a, b) in enumerate(s.rev.get("segs", [])):
            d = s.screen_seg_dist(p, pl.w(a), pl.w(b))
            if d < bd: best, bd = i, d
        return best

    # ---- move triad ----
    def move_matrix(s): return move_matrix(s.mv["c"], s.mv["d"], s.mv["rot"])
    def move_gl(s):
        M = s.move_matrix()
        return [M[0][0], M[1][0], M[2][0], 0, M[0][1], M[1][1], M[2][1], 0, M[0][2], M[1][2], M[2][2], 0, M[0][3], M[1][3], M[2][3], 1]
    def triad(s):
        o = s.mv["c"] + s.mv["d"]; L = s.dist*0.12
        return o, [(V(1, 0, 0), (0.85, 0.2, 0.2)), (V(0, 1, 0), (0.3, 0.7, 0.3)), (V(0, 0, 1), (0.2, 0.4, 0.9))], L
    def draw_triad(s):
        o, axes, L = s.triad(); glDisable(GL_DEPTH_TEST)
        for i, (a, rgb) in enumerate(axes):
            hot = s.mv.get("hot") == i or s.mv.get("drag") == i
            glColor3f(*((1, 0.75, 0.1) if hot else rgb)); glLineWidth(5 if hot else 3.5)
            b = o + a*L; glBegin(GL_LINES); glVertex3f(*o.t()); glVertex3f(*b.t()); glEnd()
            u = (V(0, 0, 1) if abs(a.z) < 0.9 else V(1, 0, 0)).cross(a).normalize(); w = a.cross(u); rr = L*0.08
            glBegin(GL_TRIANGLE_FAN); glVertex3f(*(o + a*(L*1.25)).t())
            for k in range(17):
                t = k*math.tau/16; glVertex3f(*(b + u*(rr*math.cos(t)) + w*(rr*math.sin(t))).t())
            glEnd()
        for i, (a, rgb) in enumerate(axes):                       # rotation arcs: drag to turn about that axis
            hot = s.mv.get("hot") == ("arc", i) or s.mv.get("rdrag") == i
            glColor3f(*((1, 0.75, 0.1) if hot else rgb)); glLineWidth(6 if hot else 4)
            s.strip([q.t() for q in s.arc_pts(i)])
            if s.mv.get("rdrag") == i:                             # swept angle as a translucent pie
                _, u, w = s.ARC_BASIS[i]; R = L*0.85; a0, a1 = s.mv["ang0"], s.mv["ang0"] + math.radians(s.mv["rot"][i] - s.mv["rot0"])
                glColor4f(1, 0.75, 0.1, 0.25); glBegin(GL_TRIANGLE_FAN); glVertex3f(*o.t())
                for k in range(33):
                    t = a0 + (a1 - a0)*k/32; glVertex3f(*(o + (u*math.cos(t) + w*math.sin(t))*R).t())
                glEnd()
        glPointSize(10); glColor3f(1, 1, 1); glBegin(GL_POINTS); glVertex3f(*o.t()); glEnd()
        glEnable(GL_DEPTH_TEST)
    ARC_BASIS = ((V(1, 0, 0), V(0, 1, 0), V(0, 0, 1)), (V(0, 1, 0), V(0, 0, 1), V(1, 0, 0)), (V(0, 0, 1), V(1, 0, 0), V(0, 1, 0)))
    def arc_pts(s, i, n=24):
        o, _, L = s.triad(); _, u, w = s.ARC_BASIS[i]; R = L*0.85
        return [o + (u*math.cos(t) + w*math.sin(t))*R for t in (0.18 + 1.2*k/(n - 1) for k in range(n))]
    def arc_hit(s, p):
        if s.mvp is None: return None
        px, py = s.mouse_dev(p); k = s.devicePixelRatioF(); best, bd = None, 10*k
        for i in range(3):
            xy, ok = s.project([q.t() for q in s.arc_pts(i)])
            if not ok.all(): continue
            d = min(seg_dist(px, py, *xy[j], *xy[j + 1]) for j in range(len(xy) - 1))
            if d < bd: best, bd = i, d
        return best
    def arc_angle(s, p, i):
        """Angle of the mouse around rotation axis i, measured in the plane of that arc."""
        o, d = s.ray(p); a, u, w = s.ARC_BASIS[i]; c = s.triad()[0]; dn = d.dot(a)
        if abs(dn) < 1e-9: return None
        q = o + d*((c - o).dot(a)/dn) - c
        return math.atan2(q.dot(w), q.dot(u))
    def triad_hit(s, p):
        o, axes, L = s.triad(); best, bd = None, 14
        for i, (a, _) in enumerate(axes):
            d = s.screen_seg_dist(p, o + a*(L*0.15), o + a*(L*1.25))
            if d < bd: best, bd = i, d
        return best

    # ---- projection / picking helpers ----
    def project(s, P):
        """World points (N,3) -> device pixel coords (N,2) with y measured from the bottom, plus a validity mask."""
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        h = np.c_[P, np.ones(len(P))] @ s.mvp.T; w = h[:, 3]; ok = w > 1e-9
        ws = np.where(ok, w, 1.0); nd = h[:, :2]/ws[:, None]; vp = s.vp
        return np.c_[vp[0] + (nd[:, 0] + 1)/2*vp[2], vp[1] + (nd[:, 1] + 1)/2*vp[3]], ok
    def to_logical(s, xy):
        d = s.devicePixelRatioF(); return C.QPointF(xy[0]/d, (s.vp[3] - xy[1])/d)
    def mouse_dev(s, p):
        d = s.devicePixelRatioF(); return p.x()*d, s.vp[3] - p.y()*d
    def screen_seg_dist(s, p, a, b):
        """Distance in logical pixels from the mouse to the projected world segment a-b."""
        if s.mvp is None: return 1e9
        (A, B), ok = s.project([a.t(), b.t()])
        if not ok.all(): return 1e9
        px, py = s.mouse_dev(p)
        return seg_dist(px, py, *A, *B)/s.devicePixelRatioF()
    def unproject(s, p, z):
        d = s.devicePixelRatioF()
        return V(*gluUnProject(p.x()*d, s.vp[3] - p.y()*d, z, s.mv_m, s.pj, s.vp))

    def ground(s, p, snap=True, plane=None):
        if s.mvp is None: return None
        a, b = ((plane or s.plane).l(s.unproject(p, z)) for z in (0, 1)); d = b - a
        if abs(d.z) < 1e-9: return None
        q = a + d*(-a.z/d.z)
        if snap and not (W.QApplication.keyboardModifiers() & C.Qt.ControlModifier):
            g = s.snap_step(); return V(round(q.x/g)*g, round(q.y/g)*g, 0)
        return V(q.x, q.y, 0)

    @property
    def plane(s): return s.active.plane if s.active else XY
    @property
    def entities(s): return s.active.entities if s.active else []

    def new_sketch(s, plane, parent=None, look=True):
        tool, s.pending_tool = s.pending_tool, None
        s.sk_begin_new({"plane": plane}, tool)
    def confirm_pending(s):
        sk = s.active
        if sk and sk.pending: sk.pending = False; s.checkpoint("sketch", sk.pending_op)
    def commit(s, kind, pts):
        if not s.active: s.new_sketch(XY, look=False)
        s.confirm_pending()
        s.do({"t": "ent", "sk": s.sketches.index(s.active), "ent": (kind, list(pts)), "icon": kind})
    def set_entities(s, sk, ents, icon):
        if sk is s.active: s.confirm_pending()
        s.do({"t": "entset", "sk": s.sketches.index(sk), "ents": list(ents), "icon": icon})

    # ---- booleans ----
    def overlaps(s, bi, tool):
        if getattr(s.bodies[bi], "mesh", False): return False
        try: return abs(volume(boolean(s.bodies[bi].shape, tool, "common"))) > 1e-6
        except Exception: return False
    def replace_body(s, bi, shape):
        old = s.bodies[bi]; parts = [Body(x).copy_look(old) for x in solids(clean(shape))]
        s.bodies[bi:bi + 1] = parts
    def cut_all(s, tool):
        """Subtract tool from every visible body it passes through; returns how many were cut."""
        hit = [i for i in range(len(s.bodies)) if s.bodies[i].visible and s.overlaps(i, tool)]
        for i in reversed(hit): s.replace_body(i, boolean(s.bodies[i].shape, tool, "cut"))
        return len(hit)
    def join(s, targets, tool):
        targets = sorted(set(targets)); shape = tool
        for i in targets: shape = boolean(s.bodies[i].shape, shape, "fuse")
        first = targets[0]
        for i in reversed(targets[1:]): del s.bodies[i]
        s.replace_body(first, shape)
    def apply(s, tool, op, positive, parent):
        """Auto: +distance joins the parent body (or makes a new one), -distance cuts holes in whatever it passes through.
        Returns the operation actually performed."""
        auto = op == "Auto"
        if auto: op = "Join" if positive else "Cut"
        if op == "Cut":
            if s.cut_all(tool): return "Cut"
            if not auto: raise RuntimeError("The cut doesn't pass through any body.")
        elif op == "Join":
            targets = [parent] if parent is not None and parent < len(s.bodies) and s.bodies[parent].visible else \
                      [i for i, b in enumerate(s.bodies) if b.visible and s.overlaps(i, tool)]
            if targets: s.join(targets, tool); return "Join"
        s.bodies += [Body(x) for x in solids(tool)]
        return "New"

    # ---- picking ----
    def pick_face(s, p):
        """(body, face, triangle, ray t) of the nearest visible face under the mouse."""
        if s.mvp is None: return None
        o, d = s.ray(p); O, D = np.array(o.t()), np.array(d.t()); best, bt = None, np.inf
        for bi, b in enumerate(s.bodies):
            if not b.visible or not len(b.tv) or not getattr(b, "selectable", True): continue
            t = ray_hits(O, D, b.tv[:, 0], b.tv[:, 1], b.tv[:, 2]); i = int(np.argmin(t))
            if t[i] < bt: best, bt = (bi, int(b.tri_face[i]), i, float(t[i])), t[i]
        return best

    def edge_candidates(s):
        key = (s.yaw, s.pitch, s.dist, s.target.t(), tuple(s.vp), tuple((id(b), b.visible) for b in s.bodies))
        if s._edge_cache and s._edge_cache[0] == key: return s._edge_cache[1]
        data = []
        for bi, b in enumerate(s.bodies):
            if not b.visible or not len(b.sa) or not getattr(b, "selectable", True) or getattr(b, "mesh", False): continue
            A, oka = s.project(b.sa); B, okb = s.project(b.sb)
            data.append((bi, b, A, B, oka & okb))
        s._edge_cache = (key, data); return data

    def edge_near(s, p, tol=9):
        """Nearest *visible* edge within tol logical pixels (edges hidden behind faces are skipped)."""
        if s.mvp is None: return None
        k = s.devicePixelRatioF(); px, py = s.mouse_dev(p); cands = []
        for bi, b, A, B, ok in s.edge_candidates():
            d = B - A; L2 = (d*d).sum(1); L2 = np.where(L2 < 1e-12, 1e-12, L2)
            t = np.clip(((px - A[:, 0])*d[:, 0] + (py - A[:, 1])*d[:, 1])/L2, 0, 1)
            dist = np.hypot(A[:, 0] + t*d[:, 0] - px, A[:, 1] + t*d[:, 1] - py); dist[~ok] = np.inf
            for j in np.argsort(dist)[:6]:
                if dist[j] < tol*k: cands.append((dist[j], bi, int(b.se[j]), b.sa[j] + (b.sb[j] - b.sa[j])*t[j]))
        if not cands: return None
        cands.sort(key=lambda c: c[0]); hit = s.pick_face(p); o, dd = s.ray(p); D = np.array(dd.t())
        for dist, bi, ei, q in cands:
            te = float((q - np.array(o.t())) @ D / (D @ D))
            if hit is None or te <= hit[3] + 0.004*s.dist/math.sqrt(D @ D) + 1e-3*abs(te): return (bi, ei)
        return None

    def pick_origin_plane(s, p):
        o, d = s.ray(p); S = s.plane_size(); best = None
        for name, pl, _ in ORIGIN_PLANES:
            dn = d.dot(pl.n)
            if abs(dn) < 1e-12: continue
            t = (pl.o - o).dot(pl.n)/dn
            if t <= 0: continue
            q = pl.l(o + d*t)
            if 0 <= q.x <= S and 0 <= q.y <= S and (best is None or t < best[0]): best = (t, name, pl)
        return best

    def sketch_on_face(s, hit, pending=False):
        """Start a sketch on a flat face. pending = made by a plain click: it only enters the history once
        something is drawn on it, and quietly disappears if you click elsewhere."""
        bi, fid = hit[0], hit[1]; n = s.bodies[bi].face_normal(fid)
        if n is None: s.msg.emit("That face is curved - pick a flat face."); return False
        s.tool = s.hover = s.sel_face = None; s.drop_pending(); s.sel.clear()
        tool, s.pending_tool = s.pending_tool, None
        s.sk_begin_new({"face": (bi, fid)}, tool); s.sel_body = None
        if not tool: s.msg.emit("Sketching on the face - pick a tool from CREATE (Line L, Rectangle R, Circle C ...). Finish Sketch when done.")
        return True

    def pick_cplane(s, p):
        o, d = s.ray(p); S = s.cons_size(); best = None
        for ci, c in enumerate(s.cons):
            if c["kind"] != "plane" or not c["visible"]: continue
            pl = c["plane"]; dn = d.dot(pl.n)
            if abs(dn) < 1e-12: continue
            t = (pl.o - o).dot(pl.n)/dn; q = pl.l(o + d*t)
            if t > 0 and abs(q.x) <= S/2 and abs(q.y) <= S/2 and (best is None or t < best[0]): best = (t, ci, pl)
        return best

    def pick_plane(s, p):
        hit, pl = s.pick_face(p), s.pick_origin_plane(p); s.hover = s.hover_plane = None
        cp = s.pick_cplane(p)
        if cp and (hit is None or cp[0] < hit[3]) and (pl is None or cp[0] < pl[0]):
            s.tool = None; tool, s.pending_tool = s.pending_tool, None; s.sk_begin_new({"pref": ("cplane", cp[1])}, tool)
            if not tool: s.msg.emit(f"Sketch created on {s.cons[cp[1]]['name']} - pick a tool from CREATE.")
            return
        if pl and (hit is None or pl[0] < hit[3]):
            s.tool = None; tool = s.pending_tool; s.new_sketch(pl[2])
            if not tool: s.msg.emit(f"Sketch created on the {dict(XY='ground (XY)', XZ='front (XZ)', YZ='right (YZ)')[pl[1]]} plane - "
                                    "pick a tool from CREATE (Line L, Rectangle R, Circle C ...).")
        elif hit: s.sketch_on_face(hit)
        else:
            s.tool = None; s.hover_plane = None; s.msg.emit("Click one of the three origin planes, a construction plane or a flat face.")
            s.tool = "pick"

    def region_under(s, p, only=None):
        for sk in reversed(s.sketches):
            if not sk.visible or (only is not None and sk is not only): continue
            g = s.ground(p, False, sk.plane)
            if g is None: continue
            i = sk.region_at(g)
            if i is not None: return sk, i
        return None

    def click_select(s, p):
        """Click with no tool: edge > sketch region > face; empty space clears the selection."""
        e = s.edge_near(p) if "edges" in s.sel_filter else None
        if e:
            s.sel ^= {e}; s.sel_face = None; s.sel_body = e[0]
            if e in s.sel:
                pl = s.bodies[e[0]].edges[e[1]]; L = float(np.linalg.norm(np.diff(pl, axis=0), axis=1).sum())
                s.msg.emit(f"Edge selected - length {flen(L)}. Fillet (F) or Chamfer (H) the selected edges.")
            return
        r = s.region_under(p) if "profiles" in s.sel_filter else None
        if r:
            sk, i = r
            if sk is not s.active: s.active, sk.sel = sk, set()
            sk.sel ^= {i}; s.sel.clear(); s.sel_face = None
            s.msg.emit(f"{len(sk.sel)} profile(s) selected - Extrude (E) or Revolve (V) uses them.")
            return
        hit = s.pick_face(p); s.sel.clear()
        s.drop_pending()
        if hit:
            s.sel_face, s.sel_body = hit[:2], hit[0]
            if getattr(s.bodies[hit[0]], "mesh", False):
                s.sel_face = None; s.msg.emit(f"Mesh body selected ({len(s.bodies[hit[0]].tv):,} triangles) - the MESH tab has tools for it, M moves it.")
                return
            cyl = s.bodies[hit[0]].face_cylinder(hit[1])
            s.msg.emit("Cylindrical face selected - press T to add a thread, F to fillet its edges, M to move the body." if cyl else
                       "Face selected - double-click it (or press S) to sketch on it, M to move the body, F to fillet its edges.")
        else:
            s.sel_face = s.sel_body = None
            if s.active: s.active.sel = set()

    # ---- input ----
    def resizeEvent(s, e):
        super().resizeEvent(e)
        s.overlay.resize(s.size())
        if s.palette: s.palette.adjustSize(); s.palette.move(s.width() - s.palette.width() - 10, 150)
        if s.cube: s.cube.move(s.width() - s.cube.width() - 8, 6)
        if s.navbar: s.navbar.move((s.width() - s.navbar.width())//2, s.height() - s.navbar.height() - 12)

    def mousePressEvent(s, e):
        s.last, s.moved = e.position(), False
        if s.calib: return
        if s.insp_press(e): return
        if s.skedit and not s.cmd and e.button() == C.Qt.LeftButton and not s.nav: s.sk_press(e); return
        if s.cmd:
            if e.button() == C.Qt.LeftButton and s.handle_hit(e.position()):
                o, d = s.ray(e.position()); h = s.handle; t = line_param(o, d, h["o"], h["d"])
                if t is not None: h["drag"], h["off"] = True, h["val"] - t
            return
        if e.button() != C.Qt.LeftButton or s.label_at(e.position()): return
        if s.ext and s.arrow_hit(e.position()):
            o, d = s.ray(e.position()); pl = s.plane
            t = line_param(o, d, pl.w(s.ext_axis()), pl.n)
            s.ext["drag"], s.ext["off"] = t is not None, s.ext["d"] - (t or 0)
        if s.mv:
            i = s.triad_hit(e.position())
            if i is not None:
                o, d = s.ray(e.position()); org, axes, _ = s.triad(); a = axes[i][0]
                t = line_param(o, d, org, a)
                if t is not None: s.mv["drag"], s.mv["t0"], s.mv["d0"] = i, t, s.mv["d"]
            else:
                j = s.arc_hit(e.position()); ang = s.arc_angle(e.position(), j) if j is not None else None
                if ang is not None: s.mv.update(rdrag=j, ang0=ang, last=ang, acc=0.0, rot0=s.mv["rot"][j])

    def mouseMoveEvent(s, e):
        p = e.position(); dx, dy = p.x() - s.last.x(), p.y() - s.last.y(); b = e.buttons()
        if abs(dx) + abs(dy) > 2: s.moved = True
        if s.insp_move(e): s.last = p; return
        if s.skedit and not s.cmd and not s.nav and s.sk_mouse_move(e): s.last = p; return
        if s.cmd and s.handle and s.handle.get("drag") and b & C.Qt.LeftButton:
            h = s.handle; o, d = s.ray(p); t = line_param(o, d, h["o"], h["d"])
            if t is not None:
                g = s.snap_step()/5 if not (W.QApplication.keyboardModifiers() & C.Qt.ControlModifier) else 0
                v = t + h["off"]; v = round(v/g)*g if g else v
                if abs(v) < 1e-9: v = g or 1e-3
                h["val"] = v; s.cmd.set_value(h["key"], v)
            s.last = p; s.update(); return
        if s.ext and s.ext.get("drag") and b & C.Qt.LeftButton:
            o, d = s.ray(p); pl = s.plane; t = line_param(o, d, pl.w(s.ext_axis()), pl.n)
            if t is not None:
                g = s.snap_step()/5 if not (W.QApplication.keyboardModifiers() & C.Qt.ControlModifier) else 0
                v = t + s.ext["off"]; s.ext["d"] = round(v/g)*g if g else v; s.ext_moved.emit(s.ext["d"])
            s.last = p; s.update(); return
        if s.mv and s.mv.get("drag") is not None and b & C.Qt.LeftButton:
            o, d = s.ray(p); i = s.mv["drag"]; a = s.triad()[1][i][0]
            org = s.mv["c"] + s.mv["d0"]; t = line_param(o, d, org, a)
            if t is not None:
                g = s.snap_step()/5; delta = round((t - s.mv["t0"])/g)*g
                s.mv["d"] = s.mv["d0"] + a*delta; s.move_moved.emit()
            s.last = p; s.update(); return
        if s.mv and s.mv.get("rdrag") is not None and b & C.Qt.LeftButton:
            i = s.mv["rdrag"]; ang = s.arc_angle(p, i)
            if ang is not None:
                da = (ang - s.mv["last"] + math.pi) % math.tau - math.pi; s.mv["acc"] += da; s.mv["last"] = ang
                deg = math.degrees(s.mv["acc"])
                if not (W.QApplication.keyboardModifiers() & C.Qt.ControlModifier): deg = round(deg/5)*5   # 5 degree steps
                s.mv["rot"][i] = s.mv["rot0"] + deg; s.move_moved.emit()
            s.last = p; s.update(); return
        nl = bool(b & C.Qt.LeftButton and s.nav)
        nk = s.nav_kind(b, e.modifiers())
        if nk == "orbit" or (nl and s.nav == "orbit"):
            s._anim = None; s.yaw -= dx*0.01; s.pitch = max(-1.55, min(1.55, s.pitch + dy*0.01))
        elif nk == "pan" or (nl and s.nav == "pan"):
            k = s.dist*0.0015
            r = V(-math.sin(s.yaw), math.cos(s.yaw), 0)
            u = V(-math.sin(s.pitch)*math.cos(s.yaw), -math.sin(s.pitch)*math.sin(s.yaw), math.cos(s.pitch))
            s.target = s.target + r*(-dx*k) + u*(dy*k)
        elif nl and s.nav == "zoom": s.dist = max(0.5, s.dist*math.exp(dy*0.01))
        if not b: s.update_hover(p)
        frozen = s.hover_label is not None and len(s.hover_label) > 1 and s.hover_label[1] == "pre"   # don't chase the label
        g = s.ground(p) if s.tool in DRAW and not frozen else None
        if g: s.cur = g
        s.last = p; s.update()

    def update_hover(s, p):
        s.hover = s.hover_edge = s.hover_reg = s.hover_plane = None
        if s.skedit and not s.cmd and not s.nav:
            s.sk_hover_update(p)
            if s.tool is None and not s.sk_hover:
                r = s.region_under(p, s.esk)
                if r and s.sk_opts["profiles"]: s.hover_reg = r
            s.hover_label = ("dim", s.sk_hover[1]) if s.sk_hover and s.sk_hover[0] == "k" else None
            if s.tool is None and not s.sk_hover:
                for lab in reversed(s.labels):
                    if lab["ref"][0] == "auto" and lab["ref"][1] == s.skedit["si"] and lab["rect"].contains(p):
                        s.hover_label = lab["ref"]; s.setCursor(C.Qt.PointingHandCursor); s.setToolTip("Click to type an exact value (adds a dimension)"); break
            return
        if s.cmd:
            s.cmd_hover = s.pick_ref(p, s.cmd.active_kinds())
            s.setCursor(C.Qt.SizeVerCursor if s.handle_hit(p) else C.Qt.PointingHandCursor if s.cmd_hover else C.Qt.ArrowCursor)
            return
        lab = s.label_at(p); s.hover_label = lab["ref"] if lab else None
        if lab: s.setCursor(C.Qt.PointingHandCursor); s.setToolTip("Click to type an exact value"); return
        s.setToolTip("")
        if s.nav: s.setCursor(C.Qt.OpenHandCursor)
        elif s.tool in DRAW: s.setCursor(C.Qt.CrossCursor)
        else: s.unsetCursor()
        if s.mv:
            h = s.triad_hit(p)
            if h is None:
                j = s.arc_hit(p); h = ("arc", j) if j is not None else None
            s.mv["hot"] = h
            return
        if s.tool == "pick":
            hit, pl = s.pick_face(p), s.pick_origin_plane(p)
            if pl and (hit is None or pl[0] < hit[3]): s.hover_plane = pl[1]
            elif hit: s.hover = hit[:2]
            return
        if s.rev: s.rev["hot"] = s.rev_seg_near(p); return
        if s.tool is not None: return
        if s.ext:
            r = s.region_under(p, s.active); s.hover_reg = r; return
        s.hover_edge = s.edge_near(p)
        if s.hover_edge: return
        s.hover_reg = s.region_under(p)
        if s.hover_reg: return
        hit = s.pick_face(p); s.hover = hit[:2] if hit else None

    def mouseReleaseEvent(s, e):
        if s.insp_release(e): return
        if s.calib and e.button() == C.Qt.LeftButton and not s.moved:
            g = s.ground(e.position(), False, s.calib["plane"])
            if g is not None:
                s.calib["pts"].append((g.x, g.y)); s.update()
                if len(s.calib["pts"]) == 2: info = s.calib; s.calib = None; s.calib_done.emit(info)
                else: s.msg.emit("Now click the second point.")
            return
        if s.skedit and not s.cmd:
            if e.button() == C.Qt.RightButton and not s.moved: s.sk_context(e.position()); return
            if e.button() == C.Qt.LeftButton and not s.nav: s.sk_release(e); return
        if s.cmd:
            if s.handle and s.handle.get("drag"): s.handle["drag"] = False; return
            if e.button() == C.Qt.LeftButton and not s.moved:
                ref = s.pick_ref(e.position(), s.cmd.active_kinds())
                if ref is not None: s.cmd.pick(ref)
            s.update(); return
        if s.ext and s.ext.get("drag"): s.ext["drag"] = False; return
        if s.mv and s.mv.get("drag") is not None: s.mv["drag"] = None; return
        if s.mv and s.mv.get("rdrag") is not None: s.mv["rdrag"] = None; return
        if e.button() != C.Qt.LeftButton or s.moved: return
        lab = s.label_at(e.position())
        if lab: s.edit_label(lab); s.changed.emit(); s.update(); return
        if s.ext:                                         # click profiles while extruding to change what is extruded
            r = s.region_under(e.position(), s.active)
            if r: r[0].sel ^= {r[1]}; s.update()
            return
        if s.rev:                                         # click a sketch line to revolve around it
            i = s.rev_seg_near(e.position())
            if i is not None: s.axis_picked.emit(i)
            return
        if s.mv: return
        if s.tool == "pick": s.pick_plane(e.position()); s.changed.emit(); s.update(); return
        if s.tool:
            g = s.ground(e.position())
            if not g: return
            if s.tool == "poly":
                if len(s.pts) > 2 and s.screen_seg_dist(e.position(), s.plane.w(s.pts[0]), s.plane.w(s.pts[0])) < 8:
                    s.commit("poly", s.pts); s.pts = []                                       # click the start point = close
                elif not s.pts or (g - s.pts[-1]).Length > 1e-9: s.pts.append(g)
            else:
                s.pts.append(g)
                if len(s.pts) == 2:
                    a, b = s.pts; s.pts = []
                    ok = (abs(a.x - b.x) > 1e-9 and abs(a.y - b.y) > 1e-9) if s.tool == "rect" else (a - b).Length > 1e-9
                    if ok: s.commit(s.tool, [a, b])
        else: s.click_select(e.position())
        s.changed.emit(); s.update()

    def mouseDoubleClickEvent(s, e):
        """Double-click a sketch curve to edit that sketch; double-click a flat face to sketch on it."""
        if s.skedit and not s.cmd: s.sk_double(e); return
        if s.tool or s.ext or s.mv or s.rev or s.cmd: return
        kk = s.devicePixelRatioF(); px, py = s.mouse_dev(e.position())
        for si in reversed(range(len(s.sketches))):
            sk = s.sketches[si]
            if not sk.visible: continue
            for ci in range(len(sk.geo.C)):
                for A in geo_curve_pts(sk.geo, ci):
                    xy_, ok = s.sk_proj(sk, A)
                    if ok.all() and poly_dist(xy_, px, py)/kk < 6:
                        try: s.sk_edit(si)
                        except Exception as ex: s.msg.emit(str(ex))
                        return
        if True:
            hit = s.pick_face(e.position())
            if hit and s.sketch_on_face(hit): s.changed.emit(); s.update()
            return
        for sk in reversed(s.sketches):
            if not sk.visible: continue
            g = s.ground(e.position(), False, sk.plane)
            if not g: continue
            for k in reversed(range(len(sk.entities))):
                kind, pts = sk.entities[k]
                if kind in ("poly", "line") or not inside(g, loop(kind, pts)): continue
                a, b = pts
                if kind == "rect":
                    r = form_dialog(s, "Rectangle", [("w", "Width", "pos", abs(b.x - a.x)), ("h", "Height", "pos", abs(b.y - a.y))])
                    if r and r["w"] > 0 and r["h"] > 0:
                        b = V(a.x + (1 if b.x >= a.x else -1)*r["w"], a.y + (1 if b.y >= a.y else -1)*r["h"], 0)
                else:
                    r = form_dialog(s, "Circle", [("d", "Diameter", "pos", 2*(b - a).Length)])
                    if r and r["d"] > 0: b = V(a.x + r["d"]/2, a.y, 0)
                    else: r = None
                if r:
                    ents = list(sk.entities); ents[k] = (kind, [a, b]); s.set_entities(sk, ents, kind)
                s.update(); return
        hit = s.pick_face(e.position())                    # double-click a flat face = sketch on it
        if hit and s.sketch_on_face(hit): s.changed.emit(); s.update()

    def wheelEvent(s, e):
        up = (e.angleDelta().y() > 0) != bool(s.zoom_invert); s.dist = max(0.5, s.dist*(0.9 if up else 1.1)); s.update()

    def focusNextPrevChild(s, nxt): return False          # let Tab reach keyPressEvent

    def keyPressEvent(s, e):
        ret = e.key() in (C.Qt.Key_Return, C.Qt.Key_Enter)
        if s.calib and e.key() == C.Qt.Key_Escape: info = s.calib; s.calib = None; s.restore(info["back"]); return
        if s.meas is not None and e.key() in (C.Qt.Key_Escape, C.Qt.Key_Return, C.Qt.Key_Enter): s.window().measure_stop(); return
        if s.skedit and not s.cmd:
            if s.sk_key(e): return
        if s.cmd:
            if s.typing_active() and s.sk_type_key(e): return
            s.type_buf = None
            if ret or e.key() == C.Qt.Key_Escape: s.cmd_key.emit(ret)
            return
        if e.key() == C.Qt.Key_Tab and s.tool in DRAW and s.pts:                     # Tab = type the size
            ref = {"rect": ("w", "pre"), "circle": ("d", "pre")}.get(s.tool, ("seg", "pre", 0))
            s.edit_label({"ref": ref}); s.changed.emit(); s.update(); return
        if s.ext or s.mv or s.rev:
            if ret or e.key() == C.Qt.Key_Escape: s.cmd_done.emit(ret)
            return
        if ret and s.tool == "poly" and len(s.pts) > 2: s.commit("poly", s.pts)
        elif ret and s.tool == "poly" and len(s.pts) == 2: s.commit("line", s.pts)      # open line, e.g. a revolve axis
        if ret or e.key() == C.Qt.Key_Escape:
            s.pts, s.tool, s.hover, s.hover_plane = [], None, None, None
            if e.key() == C.Qt.Key_Escape and not ret: s.sel.clear(); s.sel_face = s.sel_body = None; s.msel = set()
        s.changed.emit(); s.update()

    # ---- on-screen dimensions (click one to type an exact value) ----
    def compute_labels(s):
        if s.mvp is None: return []
        labels = []                                      # (world point, text, strong, pixel offset, edit reference)
        sk = s.active
        if s.cmd:
            sk = None
            if s.handle:
                h = s.handle; labels.append((h["o"] + h["d"]*(h["val"] + s.dist*0.1*(1 if h["val"] >= 0 else -1)), flen(h["val"]), True, (0, -12), ("hdl",)))
            if s.preview and s.preview.get("cells"):
                for i, c in enumerate(s.preview["cells"], 1): labels.append((c, f"cell {i}", True, (0, 0), ("cell", i)))
        s.sk_labels(labels)
        if sk and sk.visible and s.ext:
            pl = sk.plane
            def ent_labels(kind, pts, strong, ref):
                if kind == "rect" and len(pts) == 2:
                    a, b = pts; x0, x1, y0, y1 = min(a.x, b.x), max(a.x, b.x), min(a.y, b.y), max(a.y, b.y)
                    labels.append((pl.w(V((x0 + x1)/2, y0, 0)), flen(x1 - x0), strong, (0, 14), ("w", ref)))
                    labels.append((pl.w(V(x1, (y0 + y1)/2, 0)), flen(y1 - y0), strong, (34, 0), ("h", ref)))
                elif kind == "circle" and len(pts) == 2:
                    a, b = pts; labels.append((pl.w(a + (b - a)*0.5), f"Ø {flen(2*(b - a).Length)}", strong, (0, -12), ("d", ref)))
                elif kind in ("poly", "line"):
                    segs = list(zip(pts, pts[1:] + ([pts[0]] if kind == "poly" and len(pts) > 2 else [])))
                    for j, (q0, q1) in enumerate(segs):
                        L = (q1 - q0).Length
                        if L > 1e-9: labels.append((pl.w((q0 + q1)*0.5), flen(L), strong, (0, -12), ("seg", ref, j)))
            if s.tool in DRAW and s.pts:
                pre = s.pts + [s.cur]
                if s.tool == "poly":
                    L = (pre[-1] - pre[-2]).Length
                    if L > 1e-9: labels.append((pl.w((pre[-1] + pre[-2])*0.5), flen(L), True, (0, -14), ("seg", "pre", 0)))
                else: ent_labels(s.tool, pre, True, "pre")
            if s.ext:
                a, b = s.arrow_ends(); labels.append((pl.w(b), flen(s.ext['d']), True, (0, -16), ("ext",)))
        if s.mv:
            d = s.mv["d"]; o, _, L = s.triad()
            labels.append((o, f"Δ {fmt(d.x/unit_k(), 3)}, {fmt(d.y/unit_k(), 3)}, {fmt(d.z/unit_k(), 3)} {DISPLAY['unit']}", True, (0, 26), ("move",)))
            rot = s.mv["rot"]; j = s.mv.get("rdrag")
            if j is None and isinstance(s.mv.get("hot"), tuple): j = s.mv["hot"][1]
            if j is not None or any(abs(a) > 1e-9 for a in rot):
                j = 2 if j is None else j
                others = any(abs(a) > 1e-9 for k, a in enumerate(rot) if k != j)
                text = f"Rot {fmt(rot[0], 1)}°, {fmt(rot[1], 1)}°, {fmt(rot[2], 1)}°" if others else f"Rotate {'XYZ'[j]} {fmt(rot[j], 1)}°"
                labels.append((s.arc_pts(j)[12], text, True, (0, -14), ("rot",)))
        if not labels: return []
        f = G.QFont(); f.setPixelSize(11); fm = G.QFontMetricsF(f); out = []
        xy, ok = s.project([q.t() for q, *_ in labels])
        for (q, text, strong, off, ref), c, good in zip(labels, xy, ok):
            if not good: continue
            pt = s.to_logical(c); w = fm.horizontalAdvance(text) + 12
            out.append(dict(rect=C.QRectF(pt.x() - w/2 + off[0], pt.y() - 9 + off[1], w, 18), text=text, strong=strong, ref=ref))
        return out

    def label_at(s, p):
        """The editable dimension label under the mouse (committed ones only when no drawing tool is active)."""
        if s.cmd or s.skedit: return None
        for lab in reversed(s.labels):
            if not lab["rect"].adjusted(-2, -2, 2, 2).contains(p): continue
            r = lab["ref"]
            if r[0] in ("ext", "move", "rot") or (len(r) > 1 and r[1] == "pre") or (s.tool is None and not s.ext and not s.mv and not s.rev):
                return lab
        return None

    def edit_label(s, lab):
        r, sk = lab["ref"], s.active
        if r[0] == "auto":                                          # a measured size of a sketch: open it and dimension it
            try: s.sk_edit(r[1])
            except Exception as ex: s.msg.emit(str(ex)); return
            s.auto_dim_place(r[2], r[3]); return
        if r[0] == "odim":                                          # a dimension of a sketch you are not editing: open it
            try: s.sk_edit(r[1])
            except Exception as ex: s.msg.emit(str(ex)); return
            if r[2] < len(s.esk.geo.K): s.dim_edit.emit(r[2])
            return
        if r[0] == "ext":
            v = form_dialog(s, "Extrude distance", [("v", "Distance", "mm", s.ext["d"])], "Negative values cut.")
            if v: s.ext["d"] = v["v"]; s.ext_moved.emit(v["v"])
        elif r[0] == "move":
            d = s.mv["d"]; v = form_dialog(s, "Move", [("x", "X", "mm", d.x), ("y", "Y", "mm", d.y), ("z", "Z", "mm", d.z)])
            if v: s.mv["d"] = V(v["x"], v["y"], v["z"]); s.move_moved.emit()
        elif r[0] == "rot":
            a = s.mv["rot"]; v = form_dialog(s, "Rotate", [("x", "About X", "deg", a[0]), ("y", "About Y", "deg", a[1]), ("z", "About Z", "deg", a[2])],
                                             "Rotation is about the body's centre.")
            if v: s.mv["rot"] = [v["x"], v["y"], v["z"]]; s.move_moved.emit()
        elif r[1] == "pre":                                     # typing the size of the shape being drawn
            a = s.pts[0]
            if s.tool == "rect":
                dx, dy = s.cur.x - a.x, s.cur.y - a.y
                v = form_dialog(s, "Rectangle", [("w", "Width", "pos", abs(dx)), ("h", "Height", "pos", abs(dy))])
                if v and v["w"] > 0 and v["h"] > 0:
                    s.commit("rect", [a, V(a.x + (1 if dx >= 0 else -1)*v["w"], a.y + (1 if dy >= 0 else -1)*v["h"], 0)]); s.pts = []
            elif s.tool == "circle":
                v = form_dialog(s, "Circle", [("d", "Diameter", "pos", 2*(s.cur - a).Length)])
                if v and v["d"] > 0: s.commit("circle", [a, a + V(v["d"]/2, 0, 0)]); s.pts = []
            else:
                last = s.pts[-1]; dvec = s.cur - last; dvec = dvec.normalize() if dvec.Length > 1e-9 else V(1, 0, 0)
                v = form_dialog(s, "Line length", [("l", "Length", "pos", (s.cur - last).Length),
                                                   ("a", "Angle", "deg", math.degrees(math.atan2(dvec.y, dvec.x)))],
                                "Angle is measured from the sketch's X axis.")
                if v and v["l"] > 0:
                    t = math.radians(v["a"]); s.pts.append(last + V(math.cos(t), math.sin(t), 0)*v["l"])
        else:
            kind, pts = sk.entities[r[1]]
            if r[0] in ("w", "h"):
                a, b = pts; cur = abs(b.x - a.x) if r[0] == "w" else abs(b.y - a.y)
                v = form_dialog(s, "Rectangle", [("v", "Width" if r[0] == "w" else "Height", "pos", cur)])
                if not v or v["v"] <= 0: return
                b = V(a.x + (1 if b.x >= a.x else -1)*v["v"], b.y, 0) if r[0] == "w" else V(b.x, a.y + (1 if b.y >= a.y else -1)*v["v"], 0)
                pts = [a, b]
            elif r[0] == "d":
                a, b = pts; v = form_dialog(s, "Circle", [("v", "Diameter", "pos", 2*(b - a).Length)])
                if not v or v["v"] <= 0: return
                pts = [a, a + (b - a).normalize()*(v["v"]/2)]
            else:
                j = r[2]; k = (j + 1) % len(pts); d = pts[k] - pts[j]
                v = form_dialog(s, "Line length", [("v", "Length", "pos", d.Length)], "The end point moves along the line.")
                if not v or v["v"] <= 0: return
                pts = list(pts); pts[k] = pts[j] + d.normalize()*v["v"]
            ents = list(sk.entities); ents[r[1]] = (kind, pts); s.set_entities(sk, ents, kind)
        s.update()

class LabelOverlay(W.QWidget):
    """Transparent layer above the 3D view that paints the live dimension labels."""
    def __init__(s, vp):
        super().__init__(vp); s.vp = vp
        s.setAttribute(C.Qt.WA_TransparentForMouseEvents); s.setAttribute(C.Qt.WA_TranslucentBackground)
    def paintEvent(s, _):
        if not s.vp.labels and not s.vp.glyphs and s.vp.sk_box is None and not (s.vp.skedit and (s.vp.sk_hint or s.vp.sk_cursor)) and s.vp.type_buf is None and s.vp.box is None and s.vp.meas is None: return
        p = G.QPainter(s); p.setRenderHint(G.QPainter.Antialiasing); f = p.font(); f.setPixelSize(11); p.setFont(f)
        vp = s.vp
        vp.draw_box(p); vp.paint_meas_label(p)
        for gl in vp.glyphs:                                               # constraint glyphs
            hot = vp.sk_hover == ("k", gl["ki"]) or ("k", gl["ki"]) in vp.sk_sel
            p.setPen(G.QPen(G.QColor(ACCENT if hot else "#8a929b"), 1.4 if hot else 1)); p.setBrush(G.QColor("#e3f2fc") if hot else G.QColor(255, 255, 255, 235))
            p.drawRoundedRect(gl["rect"], 3, 3)
            f2 = p.font(); f2.setPixelSize(9 if len(gl["sym"]) > 1 else 11); f2.setBold(True); p.setFont(f2)
            p.setPen(G.QColor("#b4b8be" if gl["off"] else "#2b6a2f" if gl["sym"] == "🔒" else "#1d4f7a")); p.drawText(gl["rect"], C.Qt.AlignCenter, gl["sym"] if gl["sym"] != "🔒" else "F")
            if gl["off"]: p.setPen(G.QPen(G.QColor("#b3261e"), 1.2)); p.drawLine(gl["rect"].topLeft(), gl["rect"].bottomRight())
        p.setFont(f)
        if vp.sk_box is not None:                                         # window / crossing selection
            a, b = vp.sk_box; r = C.QRectF(a, b).normalized(); cross = b.x() < a.x()
            pen = G.QPen(G.QColor(ACCENT), 1, C.Qt.DashLine if cross else C.Qt.SolidLine); p.setPen(pen)
            p.setBrush(G.QColor(6, 150, 215, 30 if cross else 45)); p.drawRect(r)
        if vp.skedit and vp.sk_cursor and vp.tool and vp.sk_cursor[2]:
            c = vp.sk_cursor; xy_, ok = vp.sk_proj(vp.esk, [[c[0], c[1]]])
            if ok[0]:
                pt = vp.to_logical(xy_[0]); txt = {"pt": "●", "mid": "△", "on": "◡ on", "vtx": "◆ corner", "hor": "— H", "ver": "| V", "perp": "⊥"}.get(c[2], "")
                if txt:
                    f2 = p.font(); f2.setPixelSize(10); p.setFont(f2); p.setPen(G.QColor("#c46a10")); p.drawText(C.QPointF(pt.x() + 10, pt.y() - 8), txt); p.setFont(f)
        tg = vp.type_target() if vp.type_buf is not None else None
        if tg:                                                             # a size being typed
            if True:
                pt = vp.type_anchor(); f2 = p.font(); f2.setPixelSize(13); p.setFont(f2); fm2 = G.QFontMetricsF(f2)
                txt = f"{tg[1]}:  {vp.type_buf}|"; wd = max(90, fm2.horizontalAdvance(txt) + 18)
                r = C.QRectF(min(pt.x() + 16, s.width() - wd - 6), min(pt.y() + 14, s.height() - 44), wd, 24)
                for ch in vp.children():                                      # keep clear of the command dialog / palette
                    if ch is s or not isinstance(ch, W.QWidget) or not ch.isVisible(): continue
                    gq = C.QRectF(ch.geometry())
                    if gq.width() >= s.width() - 2 or not gq.intersects(r.adjusted(0, 0, 0, 16)): continue
                    if gq.left() > pt.x() - 40: r.moveRight(min(gq.left() - 8, pt.x() - 16))
                    else: r.moveLeft(max(gq.right() + 8, pt.x() + 16))
                    if gq.intersects(r.adjusted(0, 0, 0, 16)): r.moveTop(max(4, gq.top() - 50))
                p.setPen(G.QPen(G.QColor(ACCENT), 2)); p.setBrush(G.QColor("white")); p.drawRoundedRect(r, 3, 3)
                p.setPen(G.QColor("#14324d")); p.drawText(r.adjusted(8, 0, 0, 0), C.Qt.AlignVCenter | C.Qt.AlignLeft, txt)
                f3 = p.font(); f3.setPixelSize(10); p.setFont(f3); p.setPen(G.QColor("#6b7280"))
                p.drawText(C.QPointF(r.left(), r.bottom() + 12), "Enter to apply · Esc to cancel"); p.setFont(f)
        if vp.skedit and vp.sk_hint:
            r = C.QRectF(s.width()/2 - 260, s.height() - 92, 520, 26)
            p.setPen(G.QPen(G.QColor("#b3261e"), 1)); p.setBrush(G.QColor(255, 240, 238, 240)); p.drawRoundedRect(r, 4, 4)
            p.setPen(G.QColor("#7a1b14")); p.drawText(r, C.Qt.AlignCenter, vp.sk_hint)
        for lab in s.vp.labels:
            r, strong, hot = lab["rect"], lab["strong"], lab["ref"] == s.vp.hover_label
            p.setPen(G.QPen(G.QColor(ACCENT if strong or hot else "#9aa4ae"), 1.6 if hot else 1))
            p.setBrush(G.QColor("#e3f2fc") if hot else G.QColor(255, 255, 255, 235 if strong else 205))
            p.drawRoundedRect(r, 3, 3); p.setPen(G.QColor("#14324d" if strong or hot else "#46505a")); p.drawText(r, C.Qt.AlignCenter, lab["text"])

# ---------------------------------------------------------------------------------------------------------------
#  Engineering drawing (3 orthographic views + isometric) via OpenCascade hidden-line removal
# ---------------------------------------------------------------------------------------------------------------
VIEW_DIRS = {"front": ((0, -1, 0), (1, 0, 0)), "top": ((0, 0, 1), (1, 0, 0)), "right": ((1, 0, 0), (0, 1, 0)),
             "left": ((-1, 0, 0), (0, -1, 0)), "bottom": ((0, 0, -1), (1, 0, 0)), "iso": ((1, -1, 1), (1, 1, 0))}
SHEETS = {"A4": (297, 210), "A3": (420, 297), "A2": (594, 420), "Letter": (279.4, 215.9), "Tabloid": (431.8, 279.4)}
SCALES = [20, 10, 5, 4, 2, 1, 0.5, 0.4, 0.25, 0.2, 0.1, 0.05, 0.04, 0.02, 0.01]

def scale_text(k): return f"{fmt(k)}:1" if k >= 1 else f"1:{fmt(1/k)}"

def is_complex(shape):
    """True when a shape has free-form faces (threads, sweeps) - exact hidden-line removal gets very slow on those."""
    from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus
    ok = (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus)
    return any(BRepAdaptor_Surface(f).GetType() not in ok for f in subshapes(shape, TopAbs_FACE))

def model_circles(shapes):
    """Full circular edges (holes, bosses) as (centre, axis, radius) in 3D."""
    out = []
    for sh in shapes:
        for e in subshapes(sh, TopAbs_EDGE):
            try:
                cv = BRepAdaptor_Curve(e)
                if cv.GetType() != GeomAbs_Circle or abs(cv.LastParameter() - cv.FirstParameter() - math.tau) > 1e-6: continue
                ci = cv.Circle(); l, a = ci.Location(), ci.Axis().Direction()
                out.append((V(l.X(), l.Y(), l.Z()), V(a.X(), a.Y(), a.Z()), ci.Radius()))
            except Exception: pass
    return out

def project_views(shapes, names):
    """Hidden-line projections. Exact (analytic) HLR for prismatic parts, mesh-based HLR when there are threads / sweeps."""
    from OCP.HLRBRep import HLRBRep_PolyAlgo, HLRBRep_PolyHLRToShape
    comp = compound(shapes); lo, hi = bbox(comp); diag = max((hi - lo).Length, 1e-3); defl = diag*0.0015
    poly = any(is_complex(sh) for sh in shapes)
    if poly: BRepMesh_IncrementalMesh(comp, diag*0.0008, False, 0.15, True)
    circles = model_circles(shapes); out = {"bbox": (lo, hi), "poly": poly}
    for name in names:
        N, X = VIEW_DIRS[name]
        proj = HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*N), gp_Dir(*X)))
        if poly:
            algo = HLRBRep_PolyAlgo(comp); algo.Projector(proj); algo.Update(); h = HLRBRep_PolyHLRToShape(); h.Update(algo)
        else:
            algo = HLRBRep_Algo(); algo.Add(comp); algo.Projector(proj); algo.Update(); algo.Hide(); h = HLRBRep_HLRToShape(algo)
        res = dict(vis=[], hid=[], smooth=[], circles=[])
        for key, getters in (("vis", ("VCompound", "OutLineVCompound")), ("smooth", ("Rg1LineVCompound",)),
                             ("hid", ("HCompound", "OutLineHCompound"))):
            for g in getters:
                try: c = getattr(h, g)()
                except Exception: continue
                if c is None or c.IsNull(): continue
                for e in subshapes(c, TopAbs_EDGE, unique=False):
                    pts = edge_pts(e, defl)
                    if len(pts) > 1: res[key].append(pts[:, :2])
        n, xv = V(*N).normalize(), V(*X).normalize(); yv = n.cross(xv); seen = set()
        for c, a, r in circles:                           # circles facing this view project to true circles
            if abs(abs(a.dot(n)) - 1) > 1e-6: continue
            key = (round(c.dot(xv), 3), round(c.dot(yv), 3), round(r, 4))
            if key not in seen: seen.add(key); res["circles"].append(key)
        allp = [p for k in ("vis", "hid", "smooth") for p in res[k]]
        if allp:
            a_ = np.concatenate(allp); res["lo"], res["hi"] = a_.min(0), a_.max(0)
        else: res["lo"], res["hi"] = np.zeros(2), np.zeros(2)
        out[name] = res
    return out

def make_sheet(views, o):
    """Lay out the views on a sheet. Returns (width, height, primitives) in millimetres, y up."""
    Wd, Ht = SHEETS[o["sheet"]]; P = []
    third = o["projection"] == "third"
    side = views["right" if third else "left"]; front, top = views["front"], views["top"]
    sz = lambda v: v["hi"] - v["lo"]
    fw, fh = sz(front); tw, th = sz(top); sw, sh = sz(side)
    m, tbw, tbh = 10, (180 if Wd > 300 else 150), 36
    ax0, ay0, ax1, ay1 = m + 6, m + tbh + 6, Wd - m - 6, Ht - m - 6
    gap = 22 if o["dims"] else 14
    need_w, need_h = lambda k: (fw + sw)*k + 3*gap, lambda k: (fh + th)*k + 3*gap
    if o["scale"] == "auto":
        k = next((c for c in SCALES if need_w(c) <= ax1 - ax0 and need_h(c) <= ay1 - ay0), SCALES[-1])
    else: k = float(o["scale"])
    x0 = ax0 + max(0, (ax1 - ax0 - need_w(k))/2) + gap
    y0 = ay0 + max(0, (ay1 - ay0 - need_h(k))/2) + gap
    if third:
        boxes = {"front": (x0, y0), "top": (x0, y0 + fh*k + gap), "side": (x0 + fw*k + gap, y0)}
        iso_cell = (x0 + fw*k + gap, y0 + fh*k + gap, ax1, ay1)
    else:
        boxes = {"top": (x0, y0), "front": (x0, y0 + th*k + gap), "side": (x0 + fw*k + gap, y0 + th*k + gap)}
        iso_cell = (x0 + fw*k + gap, ay0, ax1, y0 + th*k)
    vmap = {"front": front, "top": top, "side": side}

    def place(v, bx, by, kk):
        lo = v["lo"]; f = lambda pts: [(bx + (x - lo[0])*kk, by + (y - lo[1])*kk) for x, y in pts]
        if o["hidden"]:
            for pl in v["hid"]: P.append(("pl", f(pl), "hid"))
        if o["smooth"]:
            for pl in v["smooth"]: P.append(("pl", f(pl), "smooth"))
        for pl in v["vis"]: P.append(("pl", f(pl), "vis"))
        return f
    tf = {n: place(vmap[n], *boxes[n], k) for n in ("front", "top", "side")}
    if o["iso"] and "iso" in views:
        iv = views["iso"]; iw, ih = sz(iv); cx0, cy0, cx1, cy1 = iso_cell
        cw, ch = cx1 - cx0 - 8, cy1 - cy0 - 14
        if cw > 20 and ch > 20 and iw > 0 and ih > 0:
            ki = min(cw/iw, ch/ih, k*1.5)
            iy = cy0 + 10 + (ch - ih*ki)/2; place(iv, cx0 + 4 + (cw - iw*ki)/2, iy, ki)
            P.append(("txt", cx0 + 4 + cw/2, iy - 5, 2.5, "ISOMETRIC (not to scale)", "c", True))
    labels = {"front": "FRONT", "top": "TOP", "side": "RIGHT" if third else "LEFT"}
    for n, (bx, by) in boxes.items():
        v = vmap[n]; w_, h_ = sz(v)
        P.append(("txt", bx + w_*k/2, by + h_*k + (4 if not (o["dims"] and n == "top" and third) else 4), 2.5, labels[n], "c", True))

    def arrow(x, y, dx, dy):
        L = math.hypot(dx, dy) or 1; ux, uy = dx/L, dy/L; a, w = 2.6, 0.8
        P.append(("tri", [(x, y), (x - ux*a - uy*w, y - uy*a + ux*w), (x - ux*a + uy*w, y - uy*a - ux*w)]))
    def dim_h(xa, xb, yref, ydim, text):
        for x in (xa, xb): P.append(("pl", [(x, yref + (1 if ydim > yref else -1)), (x, ydim + (1.5 if ydim > yref else -1.5))], "thin"))
        P.append(("pl", [(xa, ydim), (xb, ydim)], "thin")); arrow(xa, ydim, -1, 0); arrow(xb, ydim, 1, 0)
        P.append(("txt", (xa + xb)/2, ydim + 1.2, 3.0, text, "c", False))
    def dim_v(ya, yb, xref, xdim, text):
        for y in (ya, yb): P.append(("pl", [(xref + (1 if xdim > xref else -1), y), (xdim + (1.5 if xdim > xref else -1.5), y)], "thin"))
        P.append(("pl", [(xdim, ya), (xdim, yb)], "thin")); arrow(xdim, ya, 0, -1); arrow(xdim, yb, 0, 1)
        P.append(("txt", xdim - 1.2, (ya + yb)/2, 3.0, text, "vr", False))
    ku = UNITS[o.get("units", "mm")]; nd = 2 if o.get("units", "mm") == "mm" else 3
    if o["dims"]:
        lo3, hi3 = views["bbox"]; ext = hi3 - lo3
        bx, by = boxes["front"]
        dim_h(bx, bx + fw*k, by, by - 9, fmt(ext.x/ku, nd))
        dim_v(by, by + fh*k, bx, bx - 9, fmt(ext.z/ku, nd))
        sx, sy = boxes["side"]
        dim_h(sx, sx + sw*k, sy, sy - 9, fmt(ext.y/ku, nd))
        seen, notes = {}, 0                                               # hole / boss diameter call-outs
        for n in ("top", "front", "side"):
            v = vmap[n]; slot = 0
            groups = {}
            for cx, cy, r in v["circles"]: groups.setdefault(round(2*r, 3), []).append((cx, cy, r))
            for dia, cs in sorted(groups.items()):
                if dia in seen or notes >= 8: continue
                seen[dia] = n; notes += 1
                cx, cy, r = cs[0]; (px, py), = tf[n]([(cx, cy)]); rr = r*k
                ang = math.radians((45, 135, -45, -135, 20, 160, -20, -160)[slot % 8]); slot += 1
                ca, sa = math.cos(ang), math.sin(ang); sgn = 1 if ca >= 0 else -1
                ex, ey = px + rr*ca, py + rr*sa; tx, ty = ex + 8*ca, ey + 8*sa
                P.append(("pl", [(ex, ey), (tx, ty), (tx + 4*sgn, ty)], "thin")); arrow(ex, ey, -ca, -sa)
                P.append(("txt", tx + 5*sgn, ty - 1, 3.0, (f"{len(cs)}× " if len(cs) > 1 else "") + f"Ø{fmt(dia/ku, nd)}", "l" if sgn > 0 else "r", False))
    # border, zone ticks and title block
    P.append(("pl", [(m, m), (Wd - m, m), (Wd - m, Ht - m), (m, Ht - m), (m, m)], "border"))
    P.append(("pl", [(5, 5), (Wd - 5, 5), (Wd - 5, Ht - 5), (5, Ht - 5), (5, 5)], "thin"))
    tx0, ty0, tx1, ty1 = Wd - m - tbw, m, Wd - m, m + tbh
    rect = lambda a, b, c, d, st_="vis": P.append(("pl", [(a, b), (c, b), (c, d), (a, d), (a, b)], st_))
    rect(tx0, ty0, tx1, ty1, "border"); r1, r2 = ty0 + 11, ty0 + 22
    P.append(("pl", [(tx0, r2), (tx1, r2)], "vis")); P.append(("pl", [(tx0, r1), (tx1, r1)], "vis"))
    cw = tbw/3
    for i in (1, 2): P.append(("pl", [(tx0 + cw*i, ty0), (tx0 + cw*i, r2)], "vis"))
    def cell(x, y, label, value, size=3.5):
        P.append(("txt", x + 1.5, y + 7.6, 1.8, label, "l", False)); P.append(("txt", x + 1.5, y + 2.2, size, value, "l", True))
    P.append(("txt", tx0 + 1.5, ty1 - 3.5, 1.8, "TITLE", "l", False))
    P.append(("txt", tx0 + 1.5, r2 + 3.2, 6, o["title"] or "Untitled", "l", True))
    P.append(("txt", tx1 - 1.5, r2 + 3.2, 2.2, "Fission 0.7", "r", False))
    cell(tx0, r1, "DRAWN BY", o["author"] or "-"); cell(tx0 + cw, r1, "DATE", datetime.date.today().isoformat())
    cell(tx0 + 2*cw, r1, "SCALE", scale_text(k)); cell(tx0, ty0, "UNITS", {"mm": "mm", "in": "inch"}.get(o.get("units", "mm"), o.get("units", "mm"))); cell(tx0 + cw, ty0, "SHEET", f"{o['sheet']}  1/1")
    P.append(("txt", tx0 + 2*cw + 1.5, ty0 + 7.6, 1.8, "THIRD ANGLE" if third else "FIRST ANGLE", "l", False))
    sx_, sy_ = tx0 + 2*cw + 9, ty0 + 3.5                                     # projection symbol
    cone = [(0, 1.2), (6, 2.6), (6, -2.6), (0, -1.2)]
    tx_ = (lambda dx: sx_ + dx) if third else (lambda dx: sx_ + 13 + dx)
    cx_ = sx_ + 13 + 3 if third else sx_ + 3
    P.append(("pl", [(tx_(x), sy_ + y) for x, y in cone + cone[:1]], "thin"))
    P.append(("pl", [(tx_(-1), sy_), (tx_(7), sy_)], "thin"))
    for r in (2.6, 1.2):
        P.append(("pl", [(cx_ + r*math.cos(a/24*math.tau), sy_ + r*math.sin(a/24*math.tau)) for a in range(25)], "thin"))
    return Wd, Ht, P

STYLE_W = {"vis": 0.5, "hid": 0.3, "smooth": 0.25, "thin": 0.18, "border": 0.7}

def render_sheet(p, sheet, k, ox=0.0, oy=0.0):
    """Paint a sheet with QPainter at k device units per mm (used for the preview, PDF and SVG)."""
    Wd, Ht, prims = sheet
    T = lambda x, y: C.QPointF(ox + x*k, oy + (Ht - y)*k)
    p.setRenderHint(G.QPainter.Antialiasing)
    p.fillRect(C.QRectF(ox, oy, Wd*k, Ht*k), G.QColor("white"))
    for pr in prims:
        if pr[0] == "pl":
            w = STYLE_W[pr[2]]; pen = G.QPen(G.QColor("#5a5f66" if pr[2] == "smooth" else "#111"), max(w*k, 0.6))
            pen.setCapStyle(C.Qt.RoundCap); pen.setJoinStyle(C.Qt.RoundJoin)
            if pr[2] == "hid": pen.setDashPattern([2.2/w, 1.2/w])
            elif pr[2] == "center": pen.setDashPattern([8/w, 1.5/w, 1.2/w, 1.5/w]); pen.setCapStyle(C.Qt.FlatCap)
            elif pr[2] == "cut": pen.setDashPattern([10/w, 1.5/w, 2/w, 1.5/w]); pen.setCapStyle(C.Qt.FlatCap)
            p.setPen(pen); p.setBrush(C.Qt.NoBrush); p.drawPolyline(G.QPolygonF([T(*q) for q in pr[1]]))
        elif pr[0] == "tri":
            p.setPen(C.Qt.NoPen); p.setBrush(G.QColor("#111")); p.drawPolygon(G.QPolygonF([T(*q) for q in pr[1]]))
        elif pr[0] == "txt":
            _, x, y, size, text, align, bold = pr
            f = G.QFont("Arial"); f.setPixelSize(max(1, int(round(size*k*1.25)))); f.setBold(bold); p.setFont(f)
            p.setPen(G.QColor("#111")); fm = G.QFontMetricsF(f); w = fm.horizontalAdvance(text); pt = T(x, y)
            if align == "vr":
                p.save(); p.translate(pt); p.rotate(-90); p.drawText(C.QPointF(-w/2, 0), text); p.restore()
            else:
                dx = {"l": 0, "c": -w/2, "r": -w}[align]; p.drawText(C.QPointF(pt.x() + dx, pt.y()), text)

def write_dxf(path, sheet):
    Wd, Ht, prims = sheet; L = ["0", "SECTION", "2", "ENTITIES"]
    layer = {"vis": "VISIBLE", "hid": "HIDDEN", "smooth": "TANGENT", "thin": "DIMENSIONS", "border": "BORDER", "center": "CENTER",
             "hatch": "HATCH", "cut": "CUTTING_PLANE"}
    for pr in prims:
        if pr[0] == "pl":
            for (x1, y1), (x2, y2) in zip(pr[1], pr[1][1:]):
                L += ["0", "LINE", "8", layer[pr[2]], "10", f"{x1:.4f}", "20", f"{y1:.4f}", "11", f"{x2:.4f}", "21", f"{y2:.4f}"]
        elif pr[0] == "txt":
            _, x, y, size, text, align, _b = pr
            L += ["0", "TEXT", "8", "TEXT", "10", f"{x:.4f}", "20", f"{y:.4f}", "40", f"{size:.3f}", "1", text]
            if align == "vr": L += ["50", "90"]
    L += ["0", "ENDSEC", "0", "EOF"]
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(L) + "\n")

class SheetView(W.QWidget):
    def __init__(s):
        super().__init__(); s.sheet, s.zoom, s.pan, s.last = None, 1.0, C.QPointF(0, 0), None; s.setMinimumSize(500, 380)
    def paintEvent(s, _):
        p = G.QPainter(s); p.fillRect(s.rect(), G.QColor("#5f656c"))
        if not s.sheet: return
        Wd, Ht, _ = s.sheet; k = min((s.width() - 30)/Wd, (s.height() - 30)/Ht)*s.zoom
        ox, oy = (s.width() - Wd*k)/2 + s.pan.x(), (s.height() - Ht*k)/2 + s.pan.y()
        p.fillRect(C.QRectF(ox + 4, oy + 4, Wd*k, Ht*k), G.QColor(0, 0, 0, 70)); render_sheet(p, s.sheet, k, ox, oy)
    def wheelEvent(s, e): s.zoom = max(0.5, min(12, s.zoom*(1.15 if e.angleDelta().y() > 0 else 1/1.15))); s.update()
    def mousePressEvent(s, e): s.last = e.position()
    def mouseMoveEvent(s, e):
        if s.last is not None: s.pan += e.position() - s.last; s.last = e.position(); s.update()
    def mouseReleaseEvent(s, e): s.last = None
    def mouseDoubleClickEvent(s, e): s.zoom, s.pan = 1.0, C.QPointF(0, 0); s.update()


# ---------------------------------------------------------------------------------------------------------------
#  Drawings 0.7: chosen views and scale, section and detail views, placed dimensions, centre marks, notes,
#  an editable title block and a parts list
# ---------------------------------------------------------------------------------------------------------------
STYLE_W.update({"center": 0.18, "hatch": 0.13, "cut": 0.6})
VIEW_LABEL = {"front": "FRONT", "top": "TOP", "right": "RIGHT", "left": "LEFT", "bottom": "BOTTOM", "iso": "ISOMETRIC"}

def view_frame(name):
    N, X = VIEW_DIRS[name]; n, x = V(*N).normalize(), V(*X).normalize(); return n, x, n.cross(x)

def hlr_lines(shapes, N, X, poly, defl):
    """Visible / hidden / tangent edges of shapes seen along N (toward the viewer), as 2D polylines in the (X, N×X) frame."""
    from OCP.HLRBRep import HLRBRep_PolyAlgo, HLRBRep_PolyHLRToShape
    comp = compound(shapes); proj = HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*N.t()), gp_Dir(*X.t())))
    if poly:
        algo = HLRBRep_PolyAlgo(comp); algo.Projector(proj); algo.Update(); h = HLRBRep_PolyHLRToShape(); h.Update(algo)
    else:
        algo = HLRBRep_Algo(); algo.Add(comp); algo.Projector(proj); algo.Update(); algo.Hide(); h = HLRBRep_HLRToShape(algo)
    res = dict(vis=[], hid=[], smooth=[], circles=[], hatch=[], cut=[])
    for key, getters in (("vis", ("VCompound", "OutLineVCompound")), ("smooth", ("Rg1LineVCompound",)), ("hid", ("HCompound", "OutLineHCompound"))):
        for g in getters:
            try: c = getattr(h, g)()
            except Exception: continue
            if c is None or c.IsNull(): continue
            for e in subshapes(c, TopAbs_EDGE, unique=False):
                pts = edge_pts(e, defl)
                if len(pts) > 1: res[key].append(pts[:, :2])
    return res

def finish_view(res, shapes, n, xv):
    yv = n.cross(xv); seen = set()
    for c, a, r in model_circles(shapes):
        if abs(abs(a.dot(n)) - 1) > 1e-6: continue
        key = (round(c.dot(xv), 3), round(c.dot(yv), 3), round(r, 4))
        if key not in seen: seen.add(key); res["circles"].append(key)
    allp = [p for k in ("vis", "hid", "smooth", "cut") for p in res[k]]
    if allp: a_ = np.concatenate(allp); res["lo"], res["hi"] = a_.min(0), a_.max(0)
    else: res["lo"], res["hi"] = np.zeros(2), np.zeros(2)
    return res

def project_named(shapes, name):
    lo, hi = bbox(compound(shapes)); diag = max((hi - lo).Length, 1e-3)
    poly = any(is_complex(sh) for sh in shapes)
    if poly: BRepMesh_IncrementalMesh(compound(shapes), diag*0.0008, False, 0.15, True)
    n, xv, _ = view_frame(name)
    return finish_view(hlr_lines(shapes, n, xv, poly, diag*0.0015), shapes, n, xv)

def section_frame(base, axis, flip):
    """Cutting plane normal, section view direction (toward the viewer) and its x axis for a cut line drawn on a base view."""
    n, xv, yv = view_frame(base)
    if axis == "v": pn, N = xv, -xv
    else: pn, N = yv, yv
    if flip: N = -N
    X = (yv.cross(N)) if axis == "v" else xv
    return pn, N, X.normalize()

def section_view(shapes, base, axis, pos, flip=False):
    """Section through a base view along a vertical ("v") or horizontal ("h") cut line at view coordinate pos."""
    pn, N, X = section_frame(base, axis, flip); Y = N.cross(X)
    lo, hi = bbox(compound(shapes)); big = (hi - lo).Length*4 + 100; diag = max((hi - lo).Length, 1e-3)
    P0 = pn*pos; kept, cuts = [], []
    for sh in shapes:
        try:
            k_ = boolean(sh, prism(big_plane_face(P0, N, big), N, big), "cut")
            if subshapes(k_, TopAbs_FACE): kept.append(k_)
            c_ = boolean(sh, big_plane_face(P0, pn, big), "common"); cuts += subshapes(c_, TopAbs_FACE)
        except Exception: pass
    if not kept: raise RuntimeError("the cut line misses the part")
    poly = any(is_complex(sh) for sh in kept)
    if poly: BRepMesh_IncrementalMesh(compound(kept), diag*0.0008, False, 0.15, True)
    res = hlr_lines(kept, N, X, poly, diag*0.0015)
    for f in cuts:                                                   # cut faces: outline + hatch loops
        for loop in face_loops_pts(f, diag*0.001):
            res["cut"].append(np.array([(p.dot(X), p.dot(Y)) for p in loop + loop[:1]]))
    return finish_view(res, kept, N, X)

def hatch_loops(loops, spacing=2.5, ang=45.0):
    """Hatch lines (even-odd fill) for closed 2D loops, all in sheet mm."""
    if not loops: return []
    a = math.radians(ang); R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    L = [np.asarray(l) @ R.T for l in loops]; ys = np.concatenate(L)[:, 1]; out = []
    y = math.floor(ys.min()/spacing)*spacing + spacing/2
    Ri = R.T
    while y < ys.max():
        xs = []
        for l in L:
            for p, q in zip(l, l[1:]):
                if (p[1] - y)*(q[1] - y) < 0: xs.append(p[0] + (y - p[1])*(q[0] - p[0])/(q[1] - p[1]))
        xs.sort()
        for i in range(0, len(xs) - 1, 2): out.append([tuple(np.array([xs[i], y]) @ Ri.T), tuple(np.array([xs[i + 1], y]) @ Ri.T)])
        y += spacing
    return out

def clip_circle(polys, c, r):
    """Pieces of 2D polylines inside a circle."""
    out = []; c = np.asarray(c, float)
    for pl in polys:
        cur = []
        for p, q in zip(pl, pl[1:]):
            p, q = np.asarray(p, float), np.asarray(q, float); d = q - p; f = p - c
            A = d @ d; B = 2*f @ d; Cc = f @ f - r*r
            if A < 1e-18: continue
            disc = B*B - 4*A*Cc
            if disc <= 0: t0, t1 = 1, 0
            else: sq = math.sqrt(disc); t0, t1 = max(0, (-B - sq)/(2*A)), min(1, (-B + sq)/(2*A))
            if t0 >= t1:
                if cur: out.append(np.array(cur)); cur = []
                continue
            a_, b_ = p + d*t0, p + d*t1
            if not cur or np.linalg.norm(np.array(cur[-1]) - a_) > 1e-9:
                if cur: out.append(np.array(cur))
                cur = [a_]
            cur.append(b_)
            if t1 < 1: out.append(np.array(cur)); cur = []
        if len(cur) > 1: out.append(np.array(cur))
    return out

def new_drawing_doc(title=""):
    return {"sheet": "A3", "projection": "third", "scale": "auto", "views": {"front": True, "top": True, "side": True, "iso": True},
            "hidden": True, "smooth": False, "dims": True, "centres": True, "units": DISPLAY["unit"], "parts": True,
            "tb": {"title": title, "pn": "", "material": "", "drawn": "", "checked": "", "company": "", "rev": "A"},
            "sections": [], "details": [], "pdims": [], "notes": []}

def next_label(doc):
    used = {x["label"] for x in doc["sections"] + doc["details"]}
    return next(ch for ch in "ABCDEFGHJKLMNPRSTUVWXYZ" if ch not in used)

def make_sheet2(doc, cache, rows=None):
    """Lay out the drawing. Returns (width, height, primitives, placed views {name: (bx, by, lo, k, w, h)})."""
    o = doc; Wd, Ht = SHEETS[o["sheet"]]; P = []; placed = {}
    third = o["projection"] == "third"; side = "right" if third else "left"
    vw = o["views"]; ku = UNITS[o.get("units", "mm")]; nd = 2 if o.get("units", "mm") == "mm" else 3
    sz = lambda v: v["hi"] - v["lo"]
    m, tbw, tbh = 10, (180 if Wd > 300 else 150), 40
    ax0, ay0, ax1, ay1 = m + 6, m + 6, Wd - m - 6, Ht - m - 6
    gap = 22 if o["dims"] else 14
    main = [n for n in ("front", "top", "side") if vw.get(n, True)] or ["front"]
    get = lambda n: cache[side if n == "side" else n]
    fw, fh = sz(get("front")); tw, th = sz(get("top")) if "top" in main else (0, 0); sw, sh = sz(get("side")) if "side" in main else (0, 0)
    extras = [("sec", s_) for s_ in o["sections"]] + [("det", d_) for d_ in o["details"]] + ([("iso", None)] if vw.get("iso") else []) + ([("xiso", None)] if vw.get("exploded") and "xiso" in cache else [])
    room_w = (ax1 - ax0) - (tbw if (extras or o["parts"]) else 0)*0.35
    need_w = lambda k: (fw + (sw if "side" in main else 0))*k + 3*gap
    need_h = lambda k: (fh + (th if "top" in main else 0))*k + 3*gap + (tbh if not extras else 0)
    if o["scale"] == "auto":
        frac = 0.62 if extras else 1.0
        k = next((c for c in SCALES if need_w(c) <= room_w*frac and need_h(c) <= (ay1 - ay0)*(0.85 if extras else 1.0)), SCALES[-1])
    else: k = float(o["scale"])
    x0, y0 = ax0 + gap, ay0 + gap + (tbh if not extras and need_w(k) > (ax1 - ax0 - tbw) else 0)
    if third: boxes = {"front": (x0, y0), "top": (x0, y0 + fh*k + gap), "side": (x0 + fw*k + gap, y0)}
    else: boxes = {"top": (x0, y0), "front": (x0, y0 + th*k + gap), "side": (x0 + fw*k + gap, y0 + th*k + gap)}
    def place(v, bx, by, kk, clip=None):
        lo = v["lo"]; f = lambda pts: [(bx + (x - lo[0])*kk, by + (y - lo[1])*kk) for x, y in pts]
        def lines(key):
            src = v[key] if clip is None else clip_circle(v[key], clip[0], clip[1])
            return [f(pl) for pl in src]
        if o["hidden"]:
            for pl in lines("hid"): P.append(("pl", pl, "hid"))
        if o["smooth"]:
            for pl in lines("smooth"): P.append(("pl", pl, "smooth"))
        for pl in lines("vis"): P.append(("pl", pl, "vis"))
        if v.get("cut"):
            loops = [f(pl) for pl in v["cut"]]
            for pl in hatch_loops(loops): P.append(("pl", pl, "hatch"))
            for pl in loops: P.append(("pl", pl, "vis"))
        if o["centres"] and clip is None:
            for cx, cy, r in v["circles"]:
                (px, py), = f([(cx, cy)]); rr = r*kk; e = rr + 2
                P.append(("pl", [(px - e, py), (px + e, py)], "center")); P.append(("pl", [(px, py - e), (px, py + e)], "center"))
        return f
    tf = {}
    for n in main:
        v = get(n); bx, by = boxes[n]; tf[n] = place(v, bx, by, k); w_, h_ = sz(v)
        placed[n] = (bx, by, v["lo"], k, w_*k, h_*k)
        P.append(("txt", bx + w_*k/2, by + h_*k + 4, 2.5, VIEW_LABEL[side if n == "side" else n], "c", True))
    main_x1 = max(boxes[n][0] + sz(get(n))[0]*k for n in main); main_y1 = max(boxes[n][1] + sz(get(n))[1]*k for n in main)
    # extra views (sections, details, iso) packed in the free space
    free = [[main_x1 + gap, ay0 + (tbh + 8 if True else 0), ax1, ay1], [ax0, main_y1 + gap, main_x1, ay1]]
    if o["parts"] and rows: free[0][3] -= 8 + 6*(len(rows) + 1)
    def put(w_, h_):
        for fr in free:
            if fr[2] - fr[0] >= w_ and fr[3] - fr[1] >= h_ + 8:
                x, y = fr[0], fr[3] - h_ - 6; fr[3] = y - gap*0.6
                return x, y
        fr = max(free, key=lambda r: (r[2] - r[0])*(r[3] - r[1])); x, y = fr[0], max(fr[1], fr[3] - h_ - 6); fr[3] = y - gap*0.6; return x, y
    for kind, ex in extras:
        if kind == "sec":
            key = ("sec", ex["base"], ex["axis"], round(ex["pos"], 6), ex.get("flip", False))
            v = cache.get(key)
            if v is None or ex["base"] not in tf: continue
            w_, h_ = sz(v); x, y = put(w_*k, h_*k); place(v, x, y, k); placed[f"sec:{ex['label']}"] = (x, y, v["lo"], k, w_*k, h_*k)
            P.append(("txt", x + w_*k/2, y - 5, 3.0, f"SECTION {ex['label']}-{ex['label']}", "c", True))
            bx, by, lo, kk, bw, bh = placed[ex["base"]]                 # the cut line on its base view
            if ex["axis"] == "v":
                px = bx + (ex["pos"] - lo[0])*kk; a, b = (px, by - 4), (px, by + bh + 4); dvec = (-1 if not ex.get("flip") else 1, 0)
                dvec = (1, 0) if ex.get("flip") else (-1, 0)
            else:
                py = by + (ex["pos"] - lo[1])*kk; a, b = (bx - 4, py), (bx + bw + 4, py); dvec = (0, -1) if ex.get("flip") else (0, 1)
            P.append(("pl", [a, b], "cut"))
            for q in (a, b):
                tip = (q[0] + dvec[0]*6, q[1] + dvec[1]*6); P.append(("pl", [q, tip], "thin"))
                L = math.hypot(*dvec); ux, uy = dvec[0]/L, dvec[1]/L
                P.append(("tri", [tip, (tip[0] - ux*2.6 - uy*0.9, tip[1] - uy*2.6 + ux*0.9), (tip[0] - ux*2.6 + uy*0.9, tip[1] - uy*2.6 - ux*0.9)]))
                P.append(("txt", tip[0] + dvec[0]*2.5, tip[1] + dvec[1]*2.5 - 1.2, 3.5, ex["label"], "c", True))
        elif kind == "det":
            if ex["base"] not in placed: continue
            v = get(ex["base"]) if ex["base"] in ("front", "top") else get("side") if ex["base"] == "side" else None
            if v is None: continue
            kk = k*ex.get("mag", 2.0); r = ex["r"]; c = (ex["cx"], ex["cy"])
            w_ = h_ = 2*r*kk; x, y = put(w_ + 4, h_ + 4)
            sub = dict(v); sub["lo"] = np.array([c[0] - r, c[1] - r])
            place(sub, x + 2, y + 2, kk, clip=(c, r))
            P.append(("pl", [(x + 2 + w_/2 + (r*kk)*math.cos(a/48*math.tau), y + 2 + h_/2 + (r*kk)*math.sin(a/48*math.tau)) for a in range(49)], "thin"))
            P.append(("txt", x + 2 + w_/2, y - 5, 3.0, f"DETAIL {ex['label']} ({scale_text(kk)})", "c", True))
            bx, by, lo, bk, _, _ = placed[ex["base"]]; pcx, pcy = bx + (c[0] - lo[0])*bk, by + (c[1] - lo[1])*bk
            P.append(("pl", [(pcx + r*bk*math.cos(a/48*math.tau), pcy + r*bk*math.sin(a/48*math.tau)) for a in range(49)], "thin"))
            P.append(("txt", pcx + r*bk*0.75 + 2, pcy + r*bk*0.75 + 2, 3.5, ex["label"], "l", True))
        elif kind in ("iso", "xiso") and kind in cache:
            v = cache[kind]; w_, h_ = sz(v)
            if w_ <= 0 or h_ <= 0: continue
            ki = min(k, (80 if kind == "iso" else 110)/max(w_, 1e-9), (70 if kind == "iso" else 90)/max(h_, 1e-9)); x, y = put(w_*ki, h_*ki); place(v, x, y, ki)
            P.append(("txt", x + w_*ki/2, y - 5, 2.5, "ISOMETRIC (not to scale)" if kind == "iso" else "EXPLODED VIEW (not to scale)", "c", True))

    def arrow(x, y, dx, dy):
        L = math.hypot(dx, dy) or 1; ux, uy = dx/L, dy/L; a, w = 2.6, 0.8
        P.append(("tri", [(x, y), (x - ux*a - uy*w, y - uy*a + ux*w), (x - ux*a + uy*w, y - uy*a - ux*w)]))
    def dim_line(pa, pb, off, kind, text):
        (xa, ya), (xb, yb) = pa, pb
        if kind == "h": d = (0, 1); ua, ub = (xa, max(ya, yb) + off if off > 0 else min(ya, yb) + off), (xb, max(ya, yb) + off if off > 0 else min(ya, yb) + off)
        elif kind == "v": d = (1, 0); ua, ub = (max(xa, xb) + off if off > 0 else min(xa, xb) + off, ya), (max(xa, xb) + off if off > 0 else min(xa, xb) + off, yb)
        else:
            L = math.hypot(xb - xa, yb - ya) or 1; nx, ny = -(yb - ya)/L, (xb - xa)/L; ua, ub = (xa + nx*off, ya + ny*off), (xb + nx*off, yb + ny*off)
        P.append(("pl", [pa, ua], "thin")); P.append(("pl", [pb, ub], "thin")); P.append(("pl", [ua, ub], "thin"))
        dx, dy = ub[0] - ua[0], ub[1] - ua[1]; arrow(ua[0], ua[1], -dx, -dy); arrow(ub[0], ub[1], dx, dy)
        mx, my = (ua[0] + ub[0])/2, (ua[1] + ub[1])/2
        if kind == "v": P.append(("txt", mx - 1.2, my, 3.0, text, "vr", False))
        else: P.append(("txt", mx, my + 1.2, 3.0, text, "c", False))
    if o["dims"] and "front" in placed:                                  # overall sizes
        lo3, hi3 = cache["bbox"]; ext = hi3 - lo3; bx, by, lo, kk, bw, bh = placed["front"]
        dim_line((bx, by), (bx + bw, by), -9, "h", fmt(ext.x/ku, nd)); dim_line((bx, by), (bx, by + bh), -9, "v", fmt(ext.z/ku, nd))
        if "side" in placed:
            sx, sy, _, _, sw_, _ = placed["side"]; dim_line((sx, sy), (sx + sw_, sy), -9, "h", fmt(ext.y/ku, nd))
        seen = set(); notes = 0
        for n in main:
            v = get(n); groups = {}
            for cx, cy, r in v["circles"]: groups.setdefault(round(2*r, 3), []).append((cx, cy, r))
            for slot, (dia, cs) in enumerate(sorted(groups.items())):
                if dia in seen or notes >= 8: continue
                seen.add(dia); notes += 1; cx, cy, r = cs[0]; (px, py), = tf[n]([(cx, cy)]); rr = r*k
                ang = math.radians((45, 135, -45, -135)[slot % 4]); ca, sa = math.cos(ang), math.sin(ang); sgn = 1 if ca >= 0 else -1
                ex_, ey = px + rr*ca, py + rr*sa; tx, ty = ex_ + 8*ca, ey + 8*sa
                P.append(("pl", [(ex_, ey), (tx, ty), (tx + 4*sgn, ty)], "thin")); arrow(ex_, ey, -ca, -sa)
                P.append(("txt", tx + 5*sgn, ty - 1, 3.0, (f"{len(cs)}× " if len(cs) > 1 else "") + f"Ø{fmt(dia/ku, nd)}", "l" if sgn > 0 else "r", False))
    for d in o["pdims"]:                                                   # placed dimensions (stored in view coordinates)
        if d["view"] not in placed: continue
        bx, by, lo, kk, _, _ = placed[d["view"]]; S = lambda q: (bx + (q[0] - lo[0])*kk, by + (q[1] - lo[1])*kk)
        if d["kind"] == "dia":
            (px, py) = S(d["c"]); rr = d["r"]*kk; ang = math.radians(d.get("ang", 45)); ca, sa = math.cos(ang), math.sin(ang); sgn = 1 if ca >= 0 else -1
            ex_, ey = px + rr*ca, py + rr*sa; tx, ty = ex_ + 8*ca, ey + 8*sa
            P.append(("pl", [(ex_, ey), (tx, ty), (tx + 4*sgn, ty)], "thin")); arrow(ex_, ey, -ca, -sa)
            P.append(("txt", tx + 5*sgn, ty - 1, 3.0, f"Ø{fmt(2*d['r']/ku, nd)}", "l" if sgn > 0 else "r", False)); continue
        a, b = d["a"], d["b"]; val = abs(b[0] - a[0]) if d["kind"] == "h" else abs(b[1] - a[1]) if d["kind"] == "v" else math.hypot(b[0] - a[0], b[1] - a[1])
        dim_line(S(a), S(b), d["off"], d["kind"], fmt(val/ku, nd))
    for nt in o["notes"]:
        for i, line in enumerate(nt["text"].split("\n")): P.append(("txt", nt["x"], nt["y"] - i*5, 3.5, line, "l", False))
    # border, title block, parts list
    P.append(("pl", [(m, m), (Wd - m, m), (Wd - m, Ht - m), (m, Ht - m), (m, m)], "border"))
    P.append(("pl", [(5, 5), (Wd - 5, 5), (Wd - 5, Ht - 5), (5, Ht - 5), (5, 5)], "thin"))
    tb = o["tb"]; tx0, ty0, tx1, ty1 = Wd - m - tbw, m, Wd - m, m + tbh
    rect = lambda a, b, c, d, st_="vis": P.append(("pl", [(a, b), (c, b), (c, d), (a, d), (a, b)], st_))
    rect(tx0, ty0, tx1, ty1, "border"); rows_y = [ty0, ty0 + 10, ty0 + 20, ty1]
    for y in rows_y[1:3]: P.append(("pl", [(tx0, y), (tx1, y)], "vis"))
    cw = tbw/4
    for i in (1, 2, 3): P.append(("pl", [(tx0 + cw*i, ty0), (tx0 + cw*i, ty0 + 20)], "vis"))
    def cell(x, y, label, value, size=3.2):
        P.append(("txt", x + 1.5, y + 7.2, 1.8, label, "l", False)); P.append(("txt", x + 1.5, y + 2.0, size, value or "-", "l", True))
    P.append(("txt", tx0 + 1.5, ty1 - 3.2, 1.8, "TITLE", "l", False)); P.append(("txt", tx0 + 1.5, ty0 + 23, 6, tb.get("title") or "Untitled", "l", True))
    P.append(("txt", tx1 - 1.5, ty1 - 3.2, 1.8, tb.get("company") or "", "r", True))
    cell(tx0, ty0 + 10, "PART NUMBER", tb.get("pn")); cell(tx0 + cw, ty0 + 10, "MATERIAL", tb.get("material"))
    cell(tx0 + 2*cw, ty0 + 10, "DRAWN", tb.get("drawn")); cell(tx0 + 3*cw, ty0 + 10, "CHECKED", tb.get("checked"))
    cell(tx0, ty0, "SCALE", scale_text(k)); cell(tx0 + cw, ty0, "UNITS", {"mm": "mm", "in": "inch"}.get(o.get("units", "mm"), "mm"))
    cell(tx0 + 2*cw, ty0, "DATE", datetime.date.today().isoformat()); cell(tx0 + 3*cw, ty0, "REV / SHEET", f"{tb.get('rev') or '-'}   {o['sheet']}")
    P.append(("txt", tx1 - 1.5, ty0 + 23, 2.0, ("THIRD" if third else "FIRST") + " ANGLE", "r", False))
    if o["parts"] and rows:
        cols = [("ITEM", 10), ("QTY", 10), ("PART NUMBER", 34), ("NAME", 52), ("MATERIAL", tbw - 106)]
        y = ty1 + 6; rh = 6
        for i, r in enumerate(list(reversed(rows)) + [None]):
            yy = y + i*rh; x = tx0
            vals = ["ITEM", "QTY", "PART NUMBER", "NAME", "MATERIAL"] if r is None else \
                   [str(len(rows) - i), str(r["qty"]), r.get("pn") or "", r["name"], r.get("material") or ""]
            for (lab, wcol), val in zip(cols, vals):
                rect(x, yy, x + wcol, yy + rh, "thin"); P.append(("txt", x + 1.2, yy + 1.8, 2.4, val[:28], "l", r is None)); x += wcol
    return Wd, Ht, P, placed

def exact_bbox(shape):
    """Tight bounding box (no triangulation / tolerance padding) - for the sizes printed on drawings."""
    from OCP.BRepBndLib import BRepBndLib
    b = Bnd_Box()
    try: st(BRepBndLib, "AddOptimal")(shape, b, False, False)
    except Exception: return bbox(shape)
    if b.IsVoid(): return bbox(shape)
    lo, hi = b.CornerMin(), b.CornerMax(); return V(lo.X(), lo.Y(), lo.Z()), V(hi.X(), hi.Y(), hi.Z())

class SheetView2(SheetView):
    """Sheet preview that also takes clicks for placing dimensions, notes, section lines and detail circles."""
    def __init__(s, dlg):
        super().__init__(); s.dlg = dlg; s.setMouseTracking(True); s.hover = None; s.down = None
    def geom(s):
        Wd, Ht = s.sheet[0], s.sheet[1]; k = min((s.width() - 30)/Wd, (s.height() - 30)/Ht)*s.zoom
        return k, (s.width() - Wd*k)/2 + s.pan.x(), (s.height() - Ht*k)/2 + s.pan.y()
    def to_sheet(s, pt):
        k, ox, oy = s.geom(); return ((pt.x() - ox)/k, s.sheet[1] - (pt.y() - oy)/k)
    def to_px(s, x, y):
        k, ox, oy = s.geom(); return C.QPointF(ox + x*k, oy + (s.sheet[1] - y)*k)
    def paintEvent(s, e):
        super().paintEvent(e)
        if not s.sheet or not s.dlg.mode: return
        p = G.QPainter(s); p.setRenderHint(G.QPainter.Antialiasing)
        for q in s.dlg.picks:
            c = s.to_px(*q); p.setPen(G.QPen(G.QColor(ACCENT), 2)); p.drawEllipse(c, 4, 4)
        if s.hover:
            c = s.to_px(*s.hover[0]); p.setPen(G.QPen(G.QColor("#e8710a"), 2)); p.drawEllipse(c, 5, 5)
        p.setPen(G.QColor("#ffffff")); f = p.font(); f.setPixelSize(13); p.setFont(f)
        p.drawText(C.QPointF(12, 22), s.dlg.mode_hint())
    def mousePressEvent(s, e):
        if s.dlg.mode and e.button() == C.Qt.LeftButton: s.down = e.position(); return
        super().mousePressEvent(e)
    def mouseMoveEvent(s, e):
        if s.dlg.mode and s.sheet:
            s.hover = s.dlg.snap(s.to_sheet(e.position()), 10/s.geom()[0]); s.update()
        if s.down is None: super().mouseMoveEvent(e)
    def mouseReleaseEvent(s, e):
        if s.down is not None and s.dlg.mode:
            if (e.position() - s.down).manhattanLength() < 5: s.dlg.click(s.to_sheet(e.position()), 10/s.geom()[0])
            s.down = None; s.update(); return
        super().mouseReleaseEvent(e)
    def keyPressEvent(s, e):
        if e.key() == C.Qt.Key_Escape: s.dlg.set_mode(None); return
        super().keyPressEvent(e)


class DrawingDialog2(W.QDialog):
    """Drawing editor: pick the views and scale, add section and detail views, place dimensions and notes,
    fill in the title block; export PDF / SVG / DXF. The drawing is saved with the design."""
    def __init__(s, parent, shapes, doc, rows=None, xshapes=None):
        super().__init__(parent); s.setWindowTitle("Drawing"); s.resize(1280, 820)
        s.shapes, s.doc, s.rows, s.mode, s.picks, s.cache = shapes, doc, rows or [], None, [], {}
        W.QApplication.setOverrideCursor(C.Qt.WaitCursor)
        try:
            for n in ("front", "top", "right", "left", "iso"): s.cache[n] = project_named(shapes, n)
            s.cache["bbox"] = exact_bbox(compound(shapes))
            if xshapes: s.cache["xiso"] = project_named(xshapes, "iso")
            for x in doc["sections"]: s.section_cache(x)
        finally: W.QApplication.restoreOverrideCursor()
        h = W.QHBoxLayout(s); side = W.QScrollArea(); side.setWidgetResizable(True); side.setFixedWidth(300)
        inner = W.QWidget(); f = W.QFormLayout(inner); side.setWidget(inner); h.addWidget(side)
        s.view = SheetView2(s); h.addWidget(s.view, 1); s.form = f
        def combo(items, val):
            b = W.QComboBox()
            for lab, data in items: b.addItem(lab, data)
            b.setCurrentIndex(max(0, b.findData(val))); return b
        s.w_sheet = combo([(k_, k_) for k_ in SHEETS], doc["sheet"]); s.w_proj = combo([("Third angle (ANSI)", "third"), ("First angle (ISO)", "first")], doc["projection"])
        s.w_scale = combo([("Auto", "auto")] + [(scale_text(c), c) for c in SCALES], doc["scale"])
        s.w_units = combo([("Millimetres (mm)", "mm"), ("Inches (in)", "in")], doc.get("units", "mm"))
        f.addRow(W.QLabel("<b>Sheet</b>")); f.addRow("Size", s.w_sheet); f.addRow("Projection", s.w_proj); f.addRow("Scale", s.w_scale); f.addRow("Units", s.w_units)
        f.addRow(W.QLabel("<b>Views</b>")); s.chk = {}
        for key, lab in (("front", "Front"), ("top", "Top"), ("side", "Side (right / left)"), ("iso", "Isometric")) + \
                        ((("exploded", "Exploded view (from the storyboard)"),) if xshapes else ()):
            b = W.QCheckBox(lab); b.setChecked(doc["views"].get(key, True)); s.chk[key] = b; f.addRow(b)
        for key, lab in (("hidden", "Hidden lines"), ("smooth", "Tangent edges"), ("dims", "Overall sizes + hole callouts"), ("centres", "Centre marks"),
                         ("parts", "Parts list")):
            b = W.QCheckBox(lab); b.setChecked(bool(doc.get(key))); s.chk[key] = b; f.addRow(b)
        f.addRow(W.QLabel("<b>Add</b>"))
        for lab, mode in (("Dimension (2 points + place)", "dim"), ("Diameter (click a circle)", "dia"), ("Note", "note"),
                          ("Section - vertical cut line", "secv"), ("Section - horizontal cut line", "sech"), ("Detail view (centre + radius)", "det")):
            b = W.QPushButton(lab); b.clicked.connect(lambda _=False, m_=mode: s.set_mode(m_)); f.addRow(b)
        s.w_mag = W.QDoubleSpinBox(); s.w_mag.setRange(1.1, 20); s.w_mag.setValue(2.0); s.w_mag.setSuffix("×"); f.addRow("Detail scale", s.w_mag)
        rm = W.QPushButton("Remove last added"); rm.clicked.connect(s.remove_last); f.addRow(rm)
        f.addRow(W.QLabel("<b>Title block</b>")); s.tb = {}
        for key, lab in (("title", "Title"), ("pn", "Part number"), ("material", "Material"), ("drawn", "Drawn by"), ("checked", "Checked by"),
                         ("company", "Company"), ("rev", "Revision")):
            e_ = W.QLineEdit(doc["tb"].get(key, "")); s.tb[key] = e_; f.addRow(lab, e_); e_.textChanged.connect(s.rebuild)
        for w_ in (s.w_sheet, s.w_proj, s.w_scale, s.w_units): w_.currentIndexChanged.connect(s.rebuild)
        for b in s.chk.values(): b.toggled.connect(s.rebuild)
        for lab, fn in (("Export PDF...", s.export_pdf), ("Export SVG...", s.export_svg), ("Export DXF...", s.export_dxf)):
            b = W.QPushButton(lab); b.clicked.connect(lambda _=False, fn=fn: fn()); f.addRow(b)
        close = W.QPushButton("Close"); close.clicked.connect(s.accept); f.addRow(close)
        s.rebuild()

    # ---- model ----
    def section_cache(s, x):
        key = ("sec", x["base"], x["axis"], round(x["pos"], 6), x.get("flip", False))
        if key not in s.cache:
            base = {"side": "right" if s.doc["projection"] == "third" else "left"}.get(x["base"], x["base"])
            s.cache[key] = section_view(s.shapes, base, x["axis"], x["pos"], x.get("flip", False))
        return s.cache[key]

    def sync(s):
        d = s.doc; d["sheet"], d["projection"], d["scale"], d["units"] = s.w_sheet.currentData(), s.w_proj.currentData(), s.w_scale.currentData(), s.w_units.currentData()
        for key in ("front", "top", "side", "iso", "exploded"):
            if key in s.chk: d["views"][key] = s.chk[key].isChecked()
        for key in ("hidden", "smooth", "dims", "centres", "parts"): d[key] = s.chk[key].isChecked()
        for key, e_ in s.tb.items(): d["tb"][key] = e_.text()

    def rebuild(s, *_):
        s.sync(); Wd, Ht, P, s.placed = make_sheet2(s.doc, s.cache, s.rows); s.sheet = (Wd, Ht, P); s.view.sheet = s.sheet; s.view.update()

    # ---- interaction ----
    HINTS = {"dim": ["Click the first point", "Click the second point", "Click where the dimension line goes"],
             "dia": ["Click a circle (hole or boss) in a view"], "note": ["Click where the note goes"],
             "secv": ["Click on a view where the vertical cut goes"], "sech": ["Click on a view where the horizontal cut goes"],
             "det": ["Click the centre of the detail", "Click to set its radius"]}
    def mode_hint(s):
        h = s.HINTS.get(s.mode, [""]); return h[min(len(s.picks), len(h) - 1)] + "   (Esc to stop)"
    def set_mode(s, m): s.mode, s.picks = m, []; s.view.setFocus(); s.view.update()

    def view_at(s, q):
        for name, (bx, by, lo, k, w_, h_) in s.placed.items():
            if bx - 3 <= q[0] <= bx + w_ + 3 and by - 3 <= q[1] <= by + h_ + 3: return name
        return None
    def to_view(s, name, q):
        bx, by, lo, k, _, _ = s.placed[name]; return ((q[0] - bx)/k + lo[0], (q[1] - by)/k + lo[1])
    def to_sheetpt(s, name, v):
        bx, by, lo, k, _, _ = s.placed[name]; return (bx + (v[0] - lo[0])*k, by + (v[1] - lo[1])*k)
    def view_data(s, name):
        if name.startswith("sec:"):
            x = next(x for x in s.doc["sections"] if x["label"] == name[4:]); return s.section_cache(x)
        return s.cache[{"side": "right" if s.doc["projection"] == "third" else "left"}.get(name, name)]

    def snap(s, q, tol):
        """(sheet point, view, view point) of the nearest line end / circle centre within tol sheet-mm."""
        name = s.view_at(q)
        if not name: return None
        v = s.view_data(name); best = None
        cands = [tuple(pl[0]) for pl in v["vis"]] + [tuple(pl[-1]) for pl in v["vis"]] + [(cx, cy) for cx, cy, _ in v["circles"]]
        for c in cands:
            sp = s.to_sheetpt(name, c); d = math.hypot(sp[0] - q[0], sp[1] - q[1])
            if d < tol and (best is None or d < best[0]): best = (d, sp, c)
        return (best[1], name, best[2]) if best else None

    def click(s, q, tol):
        d = s.doc; before = {k: len(d[k]) for k in ("pdims", "notes", "details", "sections")}
        try: s.click_(q, tol)
        finally:
            for k, n in before.items():
                if len(d[k]) > n: s.added = getattr(s, "added", []) + [k]
    def click_(s, q, tol):
        d = s.doc; m = s.mode
        if m == "note":
            text, ok = W.QInputDialog.getMultiLineText(s, "Note", "Text:")
            if ok and text.strip(): d["notes"].append({"x": q[0], "y": q[1], "text": text.strip()})
            s.set_mode(None); s.rebuild(); return
        if m == "dia":
            name = s.view_at(q)
            if not name: return
            v = s.view_data(name); vq = s.to_view(name, q); best = None
            for cx, cy, r in v["circles"]:
                dd = abs(math.hypot(vq[0] - cx, vq[1] - cy) - r)
                if best is None or dd < best[0]: best = (dd, cx, cy, r)
            if best and best[0]*s.placed[name][3] < tol*2:
                ang = math.degrees(math.atan2(vq[1] - best[2], vq[0] - best[1]))
                d["pdims"].append({"kind": "dia", "view": name, "c": [best[1], best[2]], "r": best[3], "ang": ang}); s.set_mode(None); s.rebuild()
            return
        if m in ("secv", "sech"):
            name = s.view_at(q)
            if not name or name.startswith(("sec:", "det:")) or name == "iso": return
            vq = s.to_view(name, q); lab = next_label(d)
            x = {"label": lab, "base": name, "axis": "v" if m == "secv" else "h", "pos": vq[0] if m == "secv" else vq[1], "flip": False}
            try:
                W.QApplication.setOverrideCursor(C.Qt.WaitCursor); s.section_cache(x); d["sections"].append(x)
            except Exception as ex: W.QMessageBox.warning(s, "Section", str(ex))
            finally: W.QApplication.restoreOverrideCursor()
            s.set_mode(None); s.rebuild(); return
        if m == "det":
            if not s.picks:
                name = s.view_at(q)
                if not name or name not in ("front", "top", "side"): return
                s.picks = [q]; s._det_view = name; return
            name = s._det_view; c = s.to_view(name, s.picks[0]); r = math.hypot(q[0] - s.picks[0][0], q[1] - s.picks[0][1])/s.placed[name][3]
            if r > 0: d["details"].append({"label": next_label(d), "base": name, "cx": c[0], "cy": c[1], "r": r, "mag": s.w_mag.value()})
            s.set_mode(None); s.rebuild(); return
        if m == "dim":
            if len(s.picks) < 2:
                sn = s.snap(q, tol)
                if not sn: return
                if s.picks and sn[1] != s._dim_view: return
                s._dim_view = sn[1]; s.picks.append(sn[0]); s._dim_pts = getattr(s, "_dim_pts", [])[:len(s.picks) - 1] + [sn[2]]; return
            (xa, ya), (xb, yb) = s.picks
            dx, dy = abs(xb - xa), abs(yb - ya)
            kind = "h" if (q[1] > max(ya, yb) or q[1] < min(ya, yb)) and dx > 1e-6 else "v" if dy > 1e-6 else "h"
            if dx > 1e-6 and dy > 1e-6 and min(ya, yb) <= q[1] <= max(ya, yb) and min(xa, xb) <= q[0] <= max(xa, xb): kind = "a"
            if kind == "h": off = q[1] - (max(ya, yb) if q[1] > max(ya, yb) else min(ya, yb))
            elif kind == "v": off = q[0] - (max(xa, xb) if q[0] > max(xa, xb) else min(xa, xb))
            else:
                L = math.hypot(xb - xa, yb - ya); off = ((q[0] - xa)*(-(yb - ya)) + (q[1] - ya)*(xb - xa))/L
            d["pdims"].append({"kind": kind, "view": s._dim_view, "a": list(s._dim_pts[0]), "b": list(s._dim_pts[1]), "off": off})
            s.set_mode(None); s.rebuild(); return

    def remove_last(s):
        d = s.doc
        n_ = {k: len(d[k]) for k in ("pdims", "notes", "details", "sections")}
        hist = [k for k in getattr(s, "added", []) if d[k]]
        if hist: d[hist[-1]].pop(); s.added = hist[:-1]; s.rebuild(); return
        for key in ("pdims", "notes", "details", "sections"):
            if d[key]: d[key].pop(); break
        s.rebuild()

    # ---- export ----
    def ask(s, filt, ext):
        path, _ = W.QFileDialog.getSaveFileName(s, "Export drawing", (s.doc["tb"].get("title") or "drawing") + ext, filt)
        return path + ext if path and not os.path.splitext(path)[1] else path
    def export_pdf(s, path=None):
        path = path or s.ask("PDF (*.pdf)", ".pdf")
        if not path: return
        Wd, Ht, _ = s.sheet; wr = G.QPdfWriter(path)
        wr.setPageSize(G.QPageSize(C.QSizeF(Wd, Ht), G.QPageSize.Unit.Millimeter, "Fission", G.QPageSize.SizeMatchPolicy.ExactMatch))
        wr.setPageMargins(C.QMarginsF(0, 0, 0, 0)); wr.setResolution(600); wr.setTitle(s.doc["tb"].get("title", ""))
        p = G.QPainter(wr); render_sheet(p, s.sheet, 600/25.4); p.end(); return path
    def export_svg(s, path=None):
        from PySide6.QtSvg import QSvgGenerator
        path = path or s.ask("SVG (*.svg)", ".svg")
        if not path: return
        Wd, Ht, _ = s.sheet; k = 10; g = QSvgGenerator(); g.setFileName(path)
        g.setSize(C.QSize(int(Wd*k), int(Ht*k))); g.setViewBox(C.QRect(0, 0, int(Wd*k), int(Ht*k))); g.setResolution(254)
        p = G.QPainter(g); render_sheet(p, s.sheet, k); p.end(); return path
    def export_dxf(s, path=None):
        path = path or s.ask("DXF (*.dxf)", ".dxf")
        if path: write_dxf(path, s.sheet)
        return path

class DrawingDialog(W.QDialog):
    """FreeCAD-TechDraw-like sheet: front / top / side views + isometric, hidden lines, overall dimensions, title block."""
    def __init__(s, parent, shapes, title):
        super().__init__(parent); s.setWindowTitle("Drawing - 3 view engineering sheet"); s.resize(1180, 760)
        W.QApplication.setOverrideCursor(C.Qt.WaitCursor)
        try: s.views = project_views(shapes, ["front", "top", "right", "left", "iso"])
        finally: W.QApplication.restoreOverrideCursor()
        h = W.QHBoxLayout(s); side = W.QWidget(); side.setFixedWidth(270); f = W.QFormLayout(side); h.addWidget(side)
        s.view = SheetView(); h.addWidget(s.view, 1)
        s.w_sheet = W.QComboBox(); [s.w_sheet.addItem(k) for k in SHEETS]; s.w_sheet.setCurrentText("A3")
        s.w_proj = W.QComboBox(); s.w_proj.addItem("Third angle (ANSI)", "third"); s.w_proj.addItem("First angle (ISO)", "first")
        s.w_scale = W.QComboBox(); s.w_scale.addItem("Auto", "auto")
        for c in SCALES: s.w_scale.addItem(scale_text(c), c)
        s.w_hidden, s.w_smooth, s.w_dims, s.w_iso = (W.QCheckBox() for _ in range(4))
        for b in (s.w_hidden, s.w_dims, s.w_iso): b.setChecked(True)
        s.w_title, s.w_author = W.QLineEdit(title), W.QLineEdit("")
        s.w_units = W.QComboBox(); s.w_units.addItem("Millimetres (mm)", "mm"); s.w_units.addItem("Inches (in)", "in")
        s.w_units.setCurrentIndex(max(0, s.w_units.findData(DISPLAY["unit"])))
        for lab, w in (("Sheet", s.w_sheet), ("Projection", s.w_proj), ("Scale", s.w_scale), ("Hidden lines", s.w_hidden),
                       ("Tangent edges", s.w_smooth), ("Dimensions", s.w_dims), ("Isometric view", s.w_iso),
                       ("Units", s.w_units), ("Title", s.w_title), ("Drawn by", s.w_author)):
            f.addRow(lab, w)
            sig = getattr(w, "currentIndexChanged", None) or getattr(w, "toggled", None) or w.textChanged
            sig.connect(s.rebuild)
        hint = W.QLabel("Wheel = zoom, drag = pan, double-click = fit."); hint.setObjectName("hint"); hint.setWordWrap(True); f.addRow(hint)
        for lab, fn in (("Export PDF...", s.export_pdf), ("Export SVG...", s.export_svg), ("Export DXF...", s.export_dxf)):
            b = W.QPushButton(lab); b.clicked.connect(fn); f.addRow(b)
        close = W.QPushButton("Close"); close.clicked.connect(s.accept); f.addRow(close)
        s.rebuild()

    def opts(s):
        return dict(sheet=s.w_sheet.currentText(), projection=s.w_proj.currentData(), scale=s.w_scale.currentData(),
                    hidden=s.w_hidden.isChecked(), smooth=s.w_smooth.isChecked(), dims=s.w_dims.isChecked(),
                    iso=s.w_iso.isChecked(), title=s.w_title.text(), author=s.w_author.text(), units=s.w_units.currentData())
    def rebuild(s, *_): s.sheet = make_sheet(s.views, s.opts()); s.view.sheet = s.sheet; s.view.update()
    def ask(s, filt, ext):
        path, _ = W.QFileDialog.getSaveFileName(s, "Export drawing", (s.w_title.text() or "drawing") + ext, filt)
        return path + ext if path and not os.path.splitext(path)[1] else path
    def export_pdf(s):
        path = s.ask("PDF (*.pdf)", ".pdf")
        if not path: return
        Wd, Ht, _ = s.sheet; wr = G.QPdfWriter(path)
        wr.setPageSize(G.QPageSize(C.QSizeF(Wd, Ht), G.QPageSize.Unit.Millimeter, "Fission", G.QPageSize.SizeMatchPolicy.ExactMatch))
        wr.setPageMargins(C.QMarginsF(0, 0, 0, 0)); wr.setResolution(600); wr.setTitle(s.w_title.text())
        p = G.QPainter(wr); render_sheet(p, s.sheet, 600/25.4); p.end()
    def export_svg(s):
        from PySide6.QtSvg import QSvgGenerator
        path = s.ask("SVG (*.svg)", ".svg")
        if not path: return
        Wd, Ht, _ = s.sheet; k = 10; g = QSvgGenerator(); g.setFileName(path)
        g.setSize(C.QSize(int(Wd*k), int(Ht*k))); g.setViewBox(C.QRect(0, 0, int(Wd*k), int(Ht*k))); g.setResolution(254)
        g.setTitle(s.w_title.text()); p = G.QPainter(g); render_sheet(p, s.sheet, k); p.end()
    def export_dxf(s):
        path = s.ask("DXF (*.dxf)", ".dxf")
        if path: write_dxf(path, s.sheet)

# ---------------------------------------------------------------------------------------------------------------
#  Main window
# ---------------------------------------------------------------------------------------------------------------

class RibbonFit(W.QScrollArea):
    """Holds one ribbon page so the window can be narrower than the ribbon: first the buttons shrink, then the ribbon
    scrolls sideways (mouse wheel works on it)."""
    def __init__(s, rib):
        super().__init__(); s.rib = rib; s.setWidget(rib); s.setWidgetResizable(True); s.setFrameShape(W.QFrame.NoFrame)
        s.setVerticalScrollBarPolicy(C.Qt.ScrollBarAlwaysOff); s.setHorizontalScrollBarPolicy(C.Qt.ScrollBarAsNeeded)
        s.horizontalScrollBar().setStyleSheet("QScrollBar:horizontal{height:7px;background:transparent}"
                                              "QScrollBar::handle:horizontal{background:#b9c0c8;border-radius:3px;min-width:40px}"
                                              "QScrollBar::add-line,QScrollBar::sub-line{width:0}")
        s.setSizePolicy(W.QSizePolicy.Ignored, W.QSizePolicy.Fixed)
        s.btns = [b for b in rib.findChildren(W.QToolButton) if b.objectName() == "rb"]
        s.compact = False; s.full_w = rib.sizeHint().width(); s.full_h = rib.sizeHint().height()
        s.setFixedHeight(s.full_h + 8)
    def set_compact(s, on):
        if on == s.compact: return
        s.compact = on; n, ic = (36, 26) if on else (46, 34)
        for b in s.btns: b.setFixedSize(n, n); b.setIconSize(C.QSize(ic, ic))
        s.rib.adjustSize()
    def resizeEvent(s, e):
        s.set_compact(s.viewport().width() < s.full_w)
        super().resizeEvent(e)
    def minimumSizeHint(s): return C.QSize(200, s.full_h + 8)
    def sizeHint(s): return C.QSize(s.full_w, s.full_h + 8)
    def wheelEvent(s, e):
        sb = s.horizontalScrollBar()
        if sb.maximum() > 0: sb.setValue(sb.value() - e.angleDelta().y()); e.accept()
        else: super().wheelEvent(e)

class CmdPanel(W.QFrame):
    """Floating Fusion-style command dialog in the top-left corner of the viewport."""
    def __init__(s, parent, title, hint, on_done):
        super().__init__(parent); s.setObjectName("cmd"); s.setFixedWidth(290)
        lay = W.QVBoxLayout(s); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        hd = W.QLabel(title); hd.setObjectName("cmdhdr"); lay.addWidget(hd)
        body = W.QWidget(); bl = W.QVBoxLayout(body); bl.setContentsMargins(12, 12, 12, 12); bl.setSpacing(10); lay.addWidget(body)
        s.form = W.QFormLayout(); s.form.setLabelAlignment(C.Qt.AlignLeft); s.form.setHorizontalSpacing(12); bl.addLayout(s.form)
        h = W.QLabel(hint); h.setObjectName("hint"); h.setWordWrap(True); bl.addWidget(h)
        row = W.QHBoxLayout(); row.addStretch(1); no, ok = W.QPushButton("Cancel"), W.QPushButton("OK"); ok.setObjectName("primary")
        row.addWidget(no); row.addWidget(ok); bl.addLayout(row); s.move(16, 16); s.hide()
        ok.clicked.connect(lambda: on_done(True)); no.clicked.connect(lambda: on_done(False))
    def length(s, label, lo, hi):
        b = LengthSpin(lo, hi); s.form.addRow(label, b); return b
    def spin(s, label, lo, hi, suffix, dec=2):
        b = W.QDoubleSpinBox(); b.setRange(lo, hi); b.setDecimals(dec); b.setSuffix(suffix); s.form.addRow(label, b); return b
    def combo(s, label, items):
        b = W.QComboBox()
        for lab, key in items: b.addItem(lab, key)
        s.form.addRow(label, b); return b

class TLStep(W.QToolButton):
    """Timeline step: click = roll the model to this step, double-click = edit, shift-click = select for a folder,
    drag = move it, right-click = menu."""
    MIME = "application/x-fission-step"
    def __init__(s, on_click, on_double, on_delete=None, idx=None, win=None):
        super().__init__(); s.on_click, s.on_double, s.on_delete, s.idx, s.win = on_click, on_double, on_delete, idx, win
        s._t = C.QTimer(s); s._t.setSingleShot(True); s._t.timeout.connect(lambda: s.on_click()); s._press = None
    def mousePressEvent(s, e):
        s._press = e.position().toPoint() if e.button() == C.Qt.LeftButton else None
        super().mousePressEvent(e)
    def mouseMoveEvent(s, e):
        if s._press is not None and s.idx and (e.position().toPoint() - s._press).manhattanLength() >= W.QApplication.startDragDistance():
            s._press = None; s.setDown(False)
            d = G.QDrag(s); md = C.QMimeData(); md.setData(s.MIME, str(s.idx).encode()); d.setMimeData(md)
            d.setPixmap(s.icon().pixmap(24, 24)); d.exec(C.Qt.MoveAction); return
        super().mouseMoveEvent(e)
    def mouseReleaseEvent(s, e):
        super().mouseReleaseEvent(e)
        if e.button() == C.Qt.LeftButton and s.rect().contains(e.position().toPoint()):
            if e.modifiers() & (C.Qt.ShiftModifier | C.Qt.ControlModifier) and s.win and s.idx: s.win.tl_toggle_sel(s.idx); return
            s._t.start(W.QApplication.doubleClickInterval())
    def mouseDoubleClickEvent(s, e):
        s._t.stop(); C.QTimer.singleShot(0, s.on_double)
    def contextMenuEvent(s, e):
        m = W.QMenu(s); m.addAction("Edit feature...", s.on_double); m.addAction("Roll history here", s.on_click)
        if s.win and s.idx: s.win.tl_step_menu(m, s.idx)
        if s.on_delete: m.addSeparator(); m.addAction("Delete step", s.on_delete)
        m.exec(e.globalPos())


class TLStrip(W.QWidget):
    """The row of timeline steps; accepts a dragged step and moves it to where it was dropped."""
    def __init__(s, win):
        super().__init__(); s.win = win; s.setAcceptDrops(True); s.drop_x = None
    def dragEnterEvent(s, e):
        if e.mimeData().hasFormat(TLStep.MIME): e.acceptProposedAction()
    def dragMoveEvent(s, e):
        if e.mimeData().hasFormat(TLStep.MIME): s.drop_x = e.position().x(); s.update(); e.acceptProposedAction()
    def dragLeaveEvent(s, e): s.drop_x = None; s.update()
    def dropEvent(s, e):
        s.drop_x = None; s.update()
        if not e.mimeData().hasFormat(TLStep.MIME): return
        i = int(bytes(e.mimeData().data(TLStep.MIME)).decode()); e.acceptProposedAction()
        C.QTimer.singleShot(0, lambda: s.win.tl_drop(i, e.position().x()))
    def paintEvent(s, e):
        super().paintEvent(e)
        if s.drop_x is not None:
            p = G.QPainter(s); p.setPen(_pen("#0696d7", 3)); x = s.win.tl_slot_x(s.drop_x); p.drawLine(C.QLineF(x, 2, x, s.height() - 2))


class TimelineWin:
    """Window side of the 0.7 timeline: badges, suppress, rename, drag to reorder, folders."""
    def tl_init(s):
        s.tl_sel = set()

    def tl_icon(s, stt, i, pos):
        ic = icon(stt["kind"], 24); pm = ic.pixmap(48, 48)
        op = stt.get("op") or {}
        if i > pos or op.get("suppressed"): pm = ic.pixmap(48, 48, G.QIcon.Disabled)
        pm = pm.copy(); p = G.QPainter(pm); p.setRenderHint(G.QPainter.Antialiasing); W_ = pm.width()
        if op.get("suppressed"):
            p.setPen(_pen("#8a929b", W_/12)); p.drawLine(C.QLineF(W_*0.1, W_*0.9, W_*0.9, W_*0.1))
        badge = "#d93025" if stt.get("err") else "#f2b705" if stt.get("warn") else None
        if badge:
            r = C.QRectF(W_*0.52, W_*0.52, W_*0.46, W_*0.46); p.setPen(_pen("white", W_/24)); p.setBrush(G.QColor(badge)); p.drawEllipse(r)
            f = p.font(); f.setPixelSize(int(W_*0.36)); f.setBold(True); p.setFont(f); p.setPen(G.QColor("white")); p.drawText(r, C.Qt.AlignCenter, "!")
        p.end(); return G.QIcon(pm)

    def tl_name(s, i):
        v = s.vp; stt = v.states[i]; op = stt.get("op") or {}
        if op.get("label"): return op["label"]
        cm = cmd_for_op(op) if op else None; nm = cm.title.title() if cm else s.NAMES.get(stt['kind'], stt['kind'].title())
        if op.get("t") in ("sk2", "sketch"):
            nn = sum(1 for x in v.states[1:i] if x["op"] and x["op"].get("t") in ("sk2", "sketch")); nm = op.get("name") or f"Sketch{nn + 1}"
        return nm

    def rebuild_timeline(s):
        v = s.vp; groups = v.tl_clean_groups()
        key = (len(v.states), v.pos, id(v.states[-1]), getattr(v, "tl_ver", 0), tuple(sorted(s.tl_sel)),
               tuple((g["a"], g["b"], g.get("open", False), g["name"]) for g in groups), bool(v.skedit))
        if key == s._tl_key: return
        s._tl_key = key
        while s.tl.count():
            w = s.tl.takeAt(0).widget()
            if w: w.deleteLater()
        s.tl_btns = []
        def marker(): mk = W.QFrame(); mk.setObjectName("tlmarker"); mk.setFixedSize(3, 30); s.tl.addWidget(mk)
        if v.pos == 0 and len(v.states) > 1: marker()
        nerr = 0
        for i, stt in enumerate(v.states[1:], 1):
            op = stt.get("op") or {}
            nerr += bool(stt.get("err"))
            g = v.group_of(i)
            if g and not g.get("open"):
                if i == g["a"]:
                    fb = W.QToolButton(); fb.setObjectName("tlstep"); fb.setFixedSize(34, 32); fb.setIconSize(C.QSize(24, 24))
                    errs = sum(1 for j in range(g["a"], g["b"] + 1) if v.states[j].get("err"))
                    fb.setIcon(s.tl_icon({"kind": "tlfolder", "op": {}, "err": errs > 0}, 1, 1))
                    fb.setToolTip(f"{g['name']}: steps {g['a']}-{g['b']}" + (f" ({errs} failed)" if errs else "") + "\nClick to open the folder")
                    fb.clicked.connect(lambda _=False, g=g: s.tl_toggle_group(g)); fb.setContextMenuPolicy(C.Qt.CustomContextMenu)
                    fb.customContextMenuRequested.connect(lambda pt, g=g, fb=fb: s.tl_group_menu(g, fb.mapToGlobal(pt)))
                    s.tl.addWidget(fb); s.tl_btns.append((g["a"], fb))
                if g["a"] <= v.pos <= g["b"] and i == g["b"]: marker()
                continue
            if op.get("quiet"):                                       # renames etc.: undoable, but no icon
                if i == v.pos: marker()
                continue
            if g and i == g["a"]:
                ob = W.QToolButton(); ob.setObjectName("tlgroup"); ob.setFixedSize(14, 32); ob.setText("⌄"); ob.setToolTip(f"{g['name']} - click to close the folder")
                ob.setStyleSheet("QToolButton{border:none;border-left:2px solid #0696d7;color:#0696d7}")
                ob.clicked.connect(lambda _=False, g=g: s.tl_toggle_group(g)); s.tl.addWidget(ob)
            b = TLStep(lambda j=i: s.goto(j), lambda j=i: s.edit_step(j), lambda j=i: s.delete_step(j), i, s)
            b.setObjectName("tlstep"); b.setFixedSize(30, 32); b.setIconSize(C.QSize(24, 24)); b.setIcon(s.tl_icon(stt, i, v.pos))
            if i in s.tl_sel: b.setStyleSheet("QToolButton{background:#cfe8fb;border:1px solid #0696d7;border-radius:3px}")
            nm = s.tl_name(i) + ("  (editing)" if v.skedit and v.skedit.get("i") == i else "")
            tip = f"{i}. {nm}" + ("  (suppressed)" if op.get("suppressed") else "") + ("" if i <= v.pos else "  (undone - click to restore)")
            if stt.get("err"): tip += f"\n⛔ This step failed: {stt['err']}\nEdit it, suppress it or delete it."
            elif stt.get("warn"): tip += f"\n⚠ Rebuilt {stt['warn']} - check it."
            b.setToolTip(tip + "\nDouble-click to edit · drag to move · shift-click to select · right-click for more")
            s.tl.addWidget(b); s.tl_btns.append((i, b))
            if g and i == g["b"]:
                cb = W.QFrame(); cb.setFixedSize(2, 32); cb.setStyleSheet("background:#0696d7"); s.tl.addWidget(cb)
            if i == v.pos: marker()
        s.tl.addStretch(1)
        C.QTimer.singleShot(0, lambda: s.tl_scroll.horizontalScrollBar().setValue(s.tl_scroll.horizontalScrollBar().maximum()))
        s.a_undo.setEnabled(v.pos > 0 or bool(v.pts)); s.a_redo.setEnabled(v.pos < len(v.states) - 1)
        for k in ("first", "prev"): s.tlb[k].setEnabled(v.pos > 0)
        for k in ("next", "last"): s.tlb[k].setEnabled(v.pos < len(v.states) - 1)
        s.tlb["play"].setEnabled(len(v.states) > 1)
        if nerr and getattr(s, "_tl_nerr", 0) != nerr:
            s.statusBar().showMessage(f"{nerr} timeline step(s) failed to rebuild - they're marked red; hover one to see why.", 9000)
        s._tl_nerr = nerr

    # ---- menus ----
    def tl_step_menu(s, m, i):
        v = s.vp; op = v.states[i].get("op") or {}
        m.addAction("Rename...", lambda: s.tl_rename(i))
        m.addAction("Unsuppress" if op.get("suppressed") else "Suppress", lambda: s.tl_suppress(i, not op.get("suppressed")))
        m.addSeparator()
        a = m.addAction("Move earlier", lambda: s.tl_move(i, i - 1)); a.setEnabled(i > 1)
        a = m.addAction("Move later", lambda: s.tl_move(i, i + 1)); a.setEnabled(i < len(v.states) - 1)
        m.addSeparator()
        sel = sorted(s.tl_sel | {i})
        g = v.group_of(i)
        if g: m.addAction(f"Remove folder '{g['name']}'", lambda: s.tl_ungroup(g))
        elif len(sel) > 1: m.addAction(f"Group {len(sel)} steps into a folder", lambda: s.tl_group(sel[0], sel[-1]))
        else: m.addAction("Shift-click steps to select them for a folder").setEnabled(False)

    def tl_group_menu(s, g, pt):
        m = W.QMenu(s); m.addAction("Open folder", lambda: s.tl_toggle_group(g)); m.addAction("Rename folder...", lambda: s.tl_rename_group(g))
        m.addAction("Suppress all", lambda: s.tl_suppress_range(g["a"], g["b"], True)); m.addAction("Unsuppress all", lambda: s.tl_suppress_range(g["a"], g["b"], False))
        m.addSeparator(); m.addAction("Remove folder (keep the steps)", lambda: s.tl_ungroup(g)); m.exec(pt)

    # ---- actions ----
    def tl_run(s, title, fn):
        v = s.vp
        if s.busy(): return None
        s.leave_sketch()
        try:
            W.QApplication.setOverrideCursor(C.Qt.WaitCursor); fails = fn() or []
            if fails: s.statusBar().showMessage(f"{title}: {len(fails)} later step(s) couldn't be rebuilt - shown in red.", 9000)
            else: s.statusBar().showMessage(f"{title}: done.", 5000)
            return fails
        except Exception as ex: traceback.print_exc(); W.QMessageBox.warning(s, title, str(ex))
        finally: W.QApplication.restoreOverrideCursor(); s.tl_sel = set(); s._tl_key = None; s.refresh(); v.update()

    def tl_suppress(s, i, on): return s.tl_run("Suppress" if on else "Unsuppress", lambda: s.vp.set_suppressed(i, on))
    def tl_suppress_range(s, a, b, on):
        def go():
            v = s.vp; ops = [x["op"] for x in v.states]
            for j in range(a, b + 1):
                if ops[j] is None: continue
                o = dict(ops[j]); o.pop("suppressed", None)
                if on: o["suppressed"] = True
                ops[j] = o
            return v.replay_ops(a, ops, [x["kind"] for x in v.states], v.pos)
        return s.tl_run("Suppress" if on else "Unsuppress", go)
    def tl_move(s, i, k): return s.tl_run("Move step", lambda: s.vp.move_step(i, k))
    def tl_rename(s, i, name=None):
        if name is None:
            name, ok = W.QInputDialog.getText(s, "Rename step", "Name:", text=s.tl_name(i))
            if not ok: return
        s.vp.rename_step(i, name.strip()); s._tl_key = None; s.rebuild_timeline()

    def tl_toggle_sel(s, i):
        s.tl_sel ^= {i}; s.rebuild_timeline()
    def tl_group(s, a, b, name=None):
        v = s.vp
        if a > b: a, b = b, a
        if any(not (g["b"] < a or g["a"] > b) for g in v.tl_groups): return s.statusBar().showMessage("Those steps overlap an existing folder.", 5000)
        v.tl_groups.append({"name": name or f"Folder{len(v.tl_groups) + 1}", "a": a, "b": b, "open": False})
        s.tl_sel = set(); s._tl_key = None; s.rebuild_timeline()
    def tl_ungroup(s, g):
        s.vp.tl_groups = [x for x in s.vp.tl_groups if x is not g]; s._tl_key = None; s.rebuild_timeline()
    def tl_toggle_group(s, g):
        g["open"] = not g.get("open"); s._tl_key = None; s.rebuild_timeline()
    def tl_rename_group(s, g, name=None):
        if name is None:
            name, ok = W.QInputDialog.getText(s, "Rename folder", "Name:", text=g["name"])
            if not ok: return
        g["name"] = name.strip() or g["name"]; s._tl_key = None; s.rebuild_timeline()

    # ---- drag & drop ----
    def tl_slots(s):
        """[(x position of the gap before step j, j)] for every visible step button, plus the end."""
        out = [(w.geometry().left() - 1, j) for j, w in s.tl_btns]
        if s.tl_btns: out.append((s.tl_btns[-1][1].geometry().right() + 2, len(s.vp.states)))
        return out
    def tl_slot_x(s, x):
        sl = s.tl_slots()
        return min(sl, key=lambda q: abs(q[0] - x))[0] if sl else x
    def tl_drop(s, i, x):
        sl = s.tl_slots()
        if not sl: return
        j = min(sl, key=lambda q: abs(q[0] - x))[1]                # dropped into the gap before step j
        k = j if j < i else j - 1
        if k != i and k >= 1: s.tl_move(i, k)

    def build_timeline(s):
        bar = W.QFrame(); bar.setObjectName("timeline"); bar.setFixedHeight(48)
        h = W.QHBoxLayout(bar); h.setContentsMargins(10, 4, 10, 4); h.setSpacing(0); s.tlb = {}
        for ic, tip, fn in (("first", "Go to the beginning", lambda: s.goto(0)), ("prev", "Step back (Ctrl+Z)", s.undo),
                            ("play", "Play the history", s.play), ("next", "Step forward (Ctrl+Y)", s.redo),
                            ("last", "Go to the end", lambda: s.goto(len(s.vp.states) - 1))):
            b = s.tbtn("tlbtn", ic, size=18, tip=tip); b.setFixedSize(30, 30); b.clicked.connect(fn); h.addWidget(b); s.tlb[ic] = b
        h.addSpacing(12)
        sa = W.QScrollArea(); sa.setWidgetResizable(True); sa.setFrameShape(W.QFrame.NoFrame)
        sa.setVerticalScrollBarPolicy(C.Qt.ScrollBarAlwaysOff); sa.setStyleSheet("QScrollArea{background:transparent;border:none}")
        sa.viewport().setStyleSheet("background:transparent"); inner = TLStrip(s); s.tl = W.QHBoxLayout(inner); s.tl.setContentsMargins(0, 0, 0, 0); s.tl.setSpacing(3)
        sa.setWidget(inner); h.addWidget(sa, 1); s.tl_scroll = sa; s.tl_strip = inner; s.tl_btns = []
        return bar


class ExprSpinMixin:
    """Number boxes in the command panels also take expressions: parameter names, other dimensions' names and maths
    (width/2 + 3 mm). The expression is kept with the feature and re-evaluated whenever the design rebuilds."""
    def expr_setup(s, form, key, kind):
        s.form_, s.key_, s.kind_ = form, key, kind
        s.setToolTip("Type a value, or an expression using parameter names: width/2, d1 + 3 mm, 2*pi*r")
        s.valueChanged.connect(s._drop_stale)
    def _drop_stale(s, v):
        x = s.exprs().get(s.key_)
        if not x: return
        try: ok = abs(eval_expr(x[0], s.env_(), x[1]) - v) < 1e-9
        except Exception: ok = False
        if not ok: s.exprs().pop(s.key_, None)
    def exprs(s): return s.form_.vals.setdefault("_x", {})
    def env_(s):
        try: return s.form_.app.vp.dim_env()
        except Exception: return {}
    def _plain(s, text):
        t = text.replace("°", "").replace(s.suffix(), "").strip() if s.suffix() else text.replace("°", "").strip()
        return t
    def eval_text(s, text):
        t = s._plain(text)
        if not expr_names(t):
            try: return s.base_value(t), False
            except Exception: pass
        return eval_expr(t, s.env_(), s.kind_), bool(expr_names(t))
    def base_value(s, t):
        if s.kind_ == "len": return parse_len(t)
        return float(eval_expr(t, {}, "ang"))
    def valueFromText(s, text):
        try:
            v, is_expr = s.eval_text(text)
            if is_expr: s.exprs()[s.key_] = [s._plain(text), s.kind_]
            else: s.exprs().pop(s.key_, None)
            return v
        except Exception: return s.value()
    def validate(s, text, pos):
        try: s.eval_text(text); return G.QValidator.State.Acceptable
        except Exception: return G.QValidator.State.Intermediate
    def textFromValue(s, v):
        x = s.exprs().get(s.key_) if hasattr(s, "form_") else None
        if x:
            try:
                if abs(eval_expr(x[0], s.env_(), x[1]) - v) < 1e-9: return x[0]
            except Exception: pass
        return s.base_text(v)
    def fixup(s, text): return s.textFromValue(s.value())

class ExprLenSpin(ExprSpinMixin, LengthSpin):
    def base_text(s, v): return LengthSpin.textFromValue(s, v)

class ExprNumSpin(ExprSpinMixin, W.QDoubleSpinBox):
    def base_text(s, v): return W.QDoubleSpinBox.textFromValue(s, v)

# ---------------------------------------------------------------------------------------------------------------
#  Command framework: every Create / Modify / Construct tool is a list of fields; one floating panel renders it,
#  selection boxes pick references in the viewport, every change re-runs the command on a scratch copy of the
#  model for a live preview, and OK records the operation on the timeline (where double-click reopens it).
# ---------------------------------------------------------------------------------------------------------------
class CmdError(Exception): pass

def Fsel(key, label, kinds, multi=False, req=True, show=None, default=None):
    return dict(type="sel", key=key, label=label, kinds=set(kinds), multi=multi, req=req, show=show, default=default or [])
def Flen(key, label, default, lo=-100000.0, hi=100000.0, show=None): return dict(type="len", key=key, label=label, default=default, lo=lo, hi=hi, show=show)
def Fang(key, label, default, lo=-360.0, hi=360.0, show=None): return dict(type="ang", key=key, label=label, default=default, lo=lo, hi=hi, show=show)
def Fnum(key, label, default, lo, hi, dec=3, show=None, suffix=""): return dict(type="num", key=key, label=label, default=default, lo=lo, hi=hi, dec=dec, show=show, suffix=suffix)
def Fint(key, label, default, lo=1, hi=1000, show=None): return dict(type="int", key=key, label=label, default=default, lo=lo, hi=hi, show=show)
def Fcombo(key, label, default, items, show=None): return dict(type="combo", key=key, label=label, default=default, items=items, show=show)
def Fcheck(key, label, default=False, show=None): return dict(type="check", key=key, label=label, default=default, show=show)
def Ftext(key, label, default="", show=None, ph=""): return dict(type="text", key=key, label=label, default=default, show=show, ph=ph)
def Ffeats(key, label, show=None): return dict(type="feats", key=key, label=label, default=[], show=show)

OPS4 = [("Join", "Join"), ("Cut", "Cut"), ("Intersect", "Intersect"), ("New body", "New")]
OPS_AUTO = [("Auto (+ join / - cut)", "Auto")] + OPS4
SIDES = [("One side", "one"), ("Two sides", "two"), ("Symmetric", "sym")]
EXTENTS = [("Distance", "dist"), ("To object", "to"), ("All", "all")]
POINTS = {"point"}; PLANES = {"plane"}; AXES = {"axis"}; PATHS = {"curve", "edge"}
THREAD_ITEMS = [(f"{std}  {lab}", f"{std}|{lab}") for std, rows in THREADS.items() for lab, _, _ in rows]
OBJTYPES = [("Bodies", "bodies"), ("Features", "features"), ("Sketch shapes", "sketch")]
def objfields():
    return [Fcombo("objtype", "Object type", "bodies", OBJTYPES),
            Fsel("objs", "Objects", {"body"}, True, show=lambda v: v["objtype"] == "bodies"),
            Fsel("sobjs", "Sketch shapes", {"curve"}, True, show=lambda v: v["objtype"] == "sketch"),
            Ffeats("feats", "Features", show=lambda v: v["objtype"] == "features")]
def objs_to_op(v, op):
    op["objs"] = v["objs"] if v["objtype"] == "bodies" else v["sobjs"] if v["objtype"] == "sketch" else []
    op.pop("sobjs", None)
    return op

class Cmd:
    def __init__(s, name, title, icon, fields, op_t=None, hint="", build=None, load=None, handle=None, group="CREATE"):
        s.name, s.title, s.icon, s.fields, s.op_t, s.hint = name, title, icon, fields, op_t or name, hint
        s._build, s._load, s.handle, s.group = build, load, handle, group

    def check(s, v):
        for f in s.fields:
            if f["type"] == "sel" and f["req"] and (f["show"] is None or f["show"](v)) and not v[f["key"]]:
                raise CmdError(f"Select: {f['label']}")

    def build(s, v, vp):
        s.check(v)
        op = {"t": s.op_t, "icon": s.icon}
        for f in s.fields:
            k = f["key"]
            if f["type"] == "sel":
                val = list(v[k])
                if k == "bodies": op[k] = sorted({r[1] for r in val if r[0] == "body"}); continue
                op[k] = val if f["multi"] else (val[0] if val else None)
            else: op[k] = v[k]
        x = {k: list(e) for k, e in (v.get("_x") or {}).items() if any(f["key"] == k for f in s.fields)}
        out = s._build(v, op, vp) if s._build else op
        if x and isinstance(out, dict): out["_x"] = x
        return out

    def load(s, op, vp):
        v = {}
        for f in s.fields:
            k = f["key"]
            if f["type"] == "sel":
                x = op.get(k)
                if k == "bodies": v[k] = [("body", i) for i in (x or [])]
                else: v[k] = list(x) if isinstance(x, list) else ([x] if x else [])
            else: v[k] = op.get(k, f["default"])
        v["_x"] = {k: list(e) for k, e in (op.get("_x") or {}).items()}
        return s._load(op, v, vp) if s._load else v

def construct_cmd(how, title, fields, icon):
    def build(v, op, vp):
        refs = {f["key"]: op[f["key"]] for f in fields if f["type"] == "sel"}
        vals = {f["key"]: op[f["key"]] for f in fields if f["type"] != "sel"}
        return {"t": "construct", "how": how, "refs": refs, "vals": vals, "icon": icon}
    def load(op, v, vp):
        for f in fields:
            k = f["key"]
            if f["type"] == "sel": x = op["refs"].get(k); v[k] = list(x) if f["multi"] else ([x] if x else [])
            else: v[k] = op["vals"].get(k, f["default"])
        return v
    handle = plane_offset_handle if how == "plane_offset" else None
    return Cmd("c_" + how, title, icon, fields, "construct", build=build, load=load, handle=handle, group="CONSTRUCT")

def plane_offset_handle(v, vp):
    """Arrow from the picked plane / face along its normal; dragging it sets the offset distance."""
    if not v.get("plane"): return None
    try:
        pl = vp.ref_plane(v["plane"][0]); o = pl.o
        if v["plane"][0][0] == "oplane": o = pl.w(V(vp.plane_size()/2, vp.plane_size()/2, 0))
        return o, pl.n, "d"
    except Exception: return None

def ext_handle(v, vp):
    if v.get("e1") != "dist" or not v.get("profiles"): return None
    try:
        r = v["profiles"][0]
        if r[0] == "curve": sk = vp.sketches[r[1]]; return sk.plane.w(sk.geo.v(sk.geo.C[r[2]]["p"][0])), sk.plane.n, "d"
        f = vp.ref_face(r); n = vp.sketches[r[1]].plane.n if r[0] == "profile" else face_frame(f)[1]
        return centroid(f), n, "d"
    except Exception: return None

def ext_load(op, v, vp):
    if "profiles" in op: return v
    sk = vp.sketches[op["sk"]]; regs = sk.regions()
    v["profiles"] = [("profile", op["sk"], i) for i in (op.get("sel") or range(len(regs)))]
    for k, dflt in (("dir", "one"), ("e1", "dist"), ("taper", 0.0), ("start", "profile"), ("thin", False), ("thick", 1.0), ("tside", "out")): v[k] = dflt
    v["d"], v["op"] = op["d"], op.get("op", "Auto"); return v

def rev_load(op, v, vp):
    if "profiles" in op: return v
    sk = vp.sketches[op["sk"]]; regs = sk.regions()
    v["profiles"] = [("profile", op["sk"], i) for i in (op.get("sel") or range(len(regs)))]
    ax = op.get("axis", "Y"); v["axis"] = [("skaxis", op["sk"], ax) if ax in ("X", "Y") else ("curve", op["sk"], ax)]
    v["angle"], v["op"], v["dir"], v["extent"] = op.get("angle", 360.0), op.get("op", "Auto"), "one", "angle"
    return v

def edge_load(op, v, vp):
    if "edges" in op: return v
    v["edges"] = [("edge", bi, ei) for bi, eis in op.get("groups", []) for ei in eis]
    if op["t"] == "fillet": v["r"], v["r2"], v["mode"] = op["r"], op.get("r2") or 0.0, "variable" if op.get("r2") else "constant"
    else: v.update(d=op["d"], d2=op.get("d2", op["d"]), angle=op.get("angle", 45.0), mode=op.get("mode", "eq"))
    v["chain"] = False; return v

def make_commands():
    C_ = {}
    def add(c): C_[c.name] = c
    two = lambda v: v["dir"] == "two"
    add(Cmd("extrude", "EXTRUDE", "extrude", [
        Fsel("profiles", "Profiles", {"profile", "pface", "curve"}, True),
        Fcombo("start", "Start", "profile", [("Profile plane", "profile"), ("Offset", "offset"), ("From object", "object")]),
        Flen("soff", "Start offset", 0.0, show=lambda v: v["start"] == "offset"),
        Fsel("sobj", "Start object", PLANES, show=lambda v: v["start"] == "object"),
        Fcombo("dir", "Direction", "one", SIDES),
        Fcombo("e1", "Extent", "dist", EXTENTS),
        Flen("d", "Distance", 10.0, show=lambda v: v["e1"] == "dist"),
        Fcheck("whole", "Distance is whole length", False, show=lambda v: v["dir"] == "sym"),
        Fsel("to", "To object", {"plane", "face", "point", "body"}, show=lambda v: v["e1"] == "to"),
        Fang("taper", "Taper angle", 0.0, -89, 89),
        Fcombo("e2", "Side 2 extent", "dist", EXTENTS, show=two),
        Flen("d2", "Side 2 distance", 5.0, show=lambda v: two(v) and v["e2"] == "dist"),
        Fsel("to2", "Side 2 object", {"plane", "face", "point", "body"}, show=lambda v: two(v) and v["e2"] == "to"),
        Fang("taper2", "Side 2 taper", 0.0, -89, 89, show=two),
        Fcheck("thin", "Thin extrude"),
        Flen("thick", "Wall thickness", 1.0, 0.0, show=lambda v: v["thin"]),
        Fcombo("tside", "Wall side", "out", [("Outside", "out"), ("Inside", "in"), ("Centred", "center")], show=lambda v: v["thin"]),
        Fcombo("op", "Operation", "Auto", OPS_AUTO),
        Fsel("bodies", "Bodies affected", {"body"}, True, req=False, show=lambda v: v["op"] in ("Cut", "Intersect", "Join"))],
        hint="Pick sketch profiles or flat faces (open lines with Thin). Drag the arrow or type a distance; negative cuts.",
        load=ext_load, handle=ext_handle))
    add(Cmd("revolve", "REVOLVE", "revolve", [
        Fsel("profiles", "Profiles", {"profile", "pface"}, True),
        Fsel("axis", "Axis", AXES),
        Fcombo("extent", "Type", "angle", [("Angle", "angle"), ("Full (360°)", "full")]),
        Fcombo("dir", "Direction", "one", SIDES, show=lambda v: v["extent"] == "angle"),
        Fang("angle", "Angle", 360.0, show=lambda v: v["extent"] == "angle"),
        Fang("angle2", "Side 2 angle", 90.0, show=lambda v: v["extent"] == "angle" and v["dir"] == "two"),
        Fcombo("op", "Operation", "Auto", [("Auto (join parent / new)", "Auto")] + OPS4),
        Fsel("bodies", "Bodies affected", {"body"}, True, req=False, show=lambda v: v["op"] in ("Cut", "Intersect", "Join"))],
        hint="Axis: a sketch line, body edge, construction axis, origin axis or cylindrical face.", load=rev_load))
    add(Cmd("sweep", "SWEEP", "sweep", [
        Fsel("profiles", "Profiles", {"profile", "pface"}, True),
        Fcombo("type", "Type", "single", [("Single path", "single"), ("Path + guide rail", "guide")]),
        Fsel("path", "Path", PATHS, True),
        Fsel("guide", "Guide rail", PATHS, True, show=lambda v: v["type"] == "guide"),
        Fnum("dist", "Distance (0-1)", 1.0, 0.01, 1.0, 3),
        Fang("taper", "Taper angle", 0.0, -45, 45),
        Fang("twist", "Twist angle", 0.0, -3600, 3600),
        Fcombo("orient", "Orientation", "perp", [("Perpendicular", "perp"), ("Parallel", "parallel")]),
        Fcombo("op", "Operation", "New", OPS4),
        Fsel("bodies", "Bodies affected", {"body"}, True, req=False, show=lambda v: v["op"] != "New")],
        hint="The profile should sit at the start of the path. Pick every piece of the path (they must join end to end)."))
    add(Cmd("loft", "LOFT", "loft", [
        Fsel("sections", "Profiles (in order)", {"profile", "pface", "point"}, True),
        Fcombo("guide", "Guide type", "none", [("None", "none"), ("Rails", "rails"), ("Centreline", "centerline")]),
        Fsel("rails", "Rails", PATHS, True, show=lambda v: v["guide"] == "rails"),
        Fsel("centerline", "Centreline", PATHS, True, show=lambda v: v["guide"] == "centerline"),
        Fcheck("closed", "Closed (back to the first profile)"),
        Fcheck("ruled", "Straight sections (ruled)"),
        Fcombo("op", "Operation", "New", OPS4),
        Fsel("bodies", "Bodies affected", {"body"}, True, req=False, show=lambda v: v["op"] != "New")],
        hint="Pick profiles on different planes in order; a point can be the first or last section."))
    for nm, title, hint in (("rib", "RIB", "Rib: thin wall from an open sketch line, thickness across the sketch plane, growing until it meets the body."),
                            ("web", "WEB", "Web: thin walls from open sketch lines, thickness in the sketch plane, extended to the body walls.")):
        add(Cmd(nm, title, nm, [
            Fsel("curves", "Sketch lines", {"curve"}, True),
            Flen("thick", "Thickness", 2.0, 0.01),
            Fcombo("side", "Thickness", "sym", [("Symmetric", "sym"), ("One side", "one"), ("Other side", "other")]),
            Fcombo("extent", "Depth", "next", [("To next", "next"), ("Distance", "dist")]),
            Flen("depth", "Depth", 10.0, 0.01, show=lambda v: v["extent"] == "dist"),
            Fcheck("flip", "Flip direction"),
            Fcheck("extend", "Extend ends to the body", True, show=lambda v, nm=nm: nm == "web")], hint=hint))
    add(Cmd("emboss", "EMBOSS", "emboss", [
        Fsel("profiles", "Profiles", {"profile"}, True), Fsel("face", "Face", {"face"}),
        Fcombo("mode", "Type", "emboss", [("Emboss (raise)", "emboss"), ("Deboss (cut)", "deboss")]),
        Flen("depth", "Depth", 1.0, 0.01)], hint="Profiles are projected along the sketch normal onto the chosen face (flat or curved)."))
    add(Cmd("hole", "HOLE", "hole", [
        Fsel("face", "Face", {"face"}, req=False), Fsel("points", "Sketch points", POINTS, True, req=False),
        Fcombo("type", "Hole type", "simple", [("Simple", "simple"), ("Counterbore", "cbore"), ("Countersink", "csink")]),
        Fcombo("tap", "Tap type", "simple", [("Simple", "simple"), ("Clearance", "clearance"), ("Tapped", "tapped")]),
        Fcombo("size", "Thread size", "ISO Metric coarse|M6x1", THREAD_ITEMS, show=lambda v: v["tap"] != "simple"),
        Flen("dia", "Diameter", 6.0, 0.01, show=lambda v: v["tap"] == "simple"),
        Fcombo("extent", "Extent", "dist", [("Distance", "dist"), ("All", "all"), ("To object", "to")]),
        Flen("depth", "Depth", 10.0, 0.01, show=lambda v: v["extent"] == "dist"),
        Fsel("to", "To object", PLANES, show=lambda v: v["extent"] == "to"),
        Fcombo("tip", "Drill point", "angle", [("Angled", "angle"), ("Flat", "flat")], show=lambda v: v["extent"] != "all"),
        Fang("tip_angle", "Drill point angle", 118.0, 1, 179, show=lambda v: v["tip"] == "angle" and v["extent"] != "all"),
        Flen("cb_dia", "Counterbore Ø", 11.0, 0.01, show=lambda v: v["type"] == "cbore"),
        Flen("cb_depth", "Counterbore depth", 6.0, 0.01, show=lambda v: v["type"] == "cbore"),
        Flen("cs_dia", "Countersink Ø", 12.0, 0.01, show=lambda v: v["type"] == "csink"),
        Fang("cs_angle", "Countersink angle", 90.0, 1, 179, show=lambda v: v["type"] == "csink"),
        Fcheck("modeled", "Modeled thread (else cosmetic)", True, show=lambda v: v["tap"] == "tapped"),
        Fcheck("full", "Thread full depth", True, show=lambda v: v["tap"] == "tapped"),
        Flen("tlen", "Thread length", 8.0, 0.01, show=lambda v: v["tap"] == "tapped" and not v["full"]),
        Fsel("bodies", "Bodies affected", {"body"}, True, req=False)],
        hint="Click a face where the hole goes, or pick sketch points (several points = several holes)."))
    add(Cmd("thread", "THREAD", "thread", [
        Fsel("faces", "Faces", {"face"}, True),
        Fcombo("size", "Size", "auto", [("Auto (nearest ISO coarse)", "auto")] + THREAD_ITEMS),
        Fcheck("modeled", "Modeled (else cosmetic)", True), Fcheck("full", "Full length", True),
        Flen("len", "Length", 10.0, 0.01, show=lambda v: not v["full"]), Flen("off", "Offset", 0.0, 0.0, show=lambda v: not v["full"]),
        Fcheck("lh", "Left-handed"), Ftext("cls", "Class", "6g / 6H", ph="e.g. 6g, 2A")], op_t="thread2",
        hint="Pick cylindrical faces of rods (external) or holes (internal). Tables: ISO metric coarse / fine, UNC, UNF, BSP, ACME."))
    xy = [("oplane", "XY")]
    pfields = lambda: [Fsel("plane", "Plane", PLANES, default=xy), Flen("cx", "Centre X", 0.0), Flen("cy", "Centre Y", 0.0)]
    opf = lambda: [Fcombo("op", "Operation", "New", [("New body", "New"), ("Join", "Join"), ("Cut", "Cut"), ("Intersect", "Intersect")]),
                   Fsel("bodies", "Bodies affected", {"body"}, True, req=False, show=lambda v: v["op"] != "New")]
    add(Cmd("box", "BOX", "box", pfields() + [Flen("L", "Length", 20.0, 0.01), Flen("W", "Width", 20.0, 0.01), Flen("H", "Height", 20.0)] + opf(),
            op_t="prim", build=lambda v, op, vp: dict(op, shape="box")))
    add(Cmd("cylinder", "CYLINDER", "cylinder", pfields() + [Flen("D", "Diameter", 20.0, 0.01), Flen("H", "Height", 20.0)] + opf(),
            op_t="prim", build=lambda v, op, vp: dict(op, shape="cyl")))
    add(Cmd("sphere", "SPHERE", "sphere", pfields() + [Flen("D", "Diameter", 20.0, 0.01)] + opf(), op_t="prim", build=lambda v, op, vp: dict(op, shape="sphere")))
    add(Cmd("torus", "TORUS", "torus", pfields() + [Flen("D", "Inner/centre Ø", 40.0, 0.01), Flen("D2", "Torus (tube) Ø", 8.0, 0.01)] + opf(),
            op_t="prim", build=lambda v, op, vp: dict(op, shape="torus")))
    add(Cmd("coil", "COIL", "coil", pfields() + [
        Fcombo("mode", "Type", "rev_height", [("Revolution and height", "rev_height"), ("Revolution and pitch", "rev_pitch"),
                                              ("Height and pitch", "height_pitch"), ("Spiral", "spiral")]),
        Flen("D", "Diameter", 30.0, 0.01), Fnum("revs", "Revolutions", 5.0, 0.1, 500, 2, show=lambda v: v["mode"] != "height_pitch"),
        Flen("H", "Height", 40.0, 0.01, show=lambda v: v["mode"] in ("rev_height", "height_pitch")),
        Flen("pitch", "Pitch", 8.0, 0.01, show=lambda v: v["mode"] != "rev_height"),
        Fang("angle", "Taper angle", 0.0, -60, 60, show=lambda v: v["mode"] != "spiral"),
        Fcombo("section", "Section", "circle", [("Circular", "circle"), ("Square", "square"), ("Triangular (external)", "tri_out"), ("Triangular (internal)", "tri_in")]),
        Fcombo("secpos", "Section position", "center", [("Inside", "inside"), ("On centre", "center"), ("Outside", "outside")]),
        Flen("size", "Section size", 3.0, 0.01)] + opf(), op_t="prim", build=lambda v, op, vp: dict(op, shape="coil")))
    add(Cmd("pipe", "PIPE", "pipe", [Fsel("path", "Path", PATHS, True), Fnum("dist", "Distance (0-1)", 1.0, 0.01, 1.0),
        Fcombo("section", "Section", "circle", [("Circular", "circle"), ("Square", "square"), ("Triangular", "tri_out")]),
        Flen("size", "Section size", 6.0, 0.01), Fcheck("hollow", "Hollow"), Flen("wall", "Wall thickness", 1.0, 0.01, show=lambda v: v["hollow"])] + opf(),
        op_t="prim", build=lambda v, op, vp: dict(op, shape="pipe")))
    add(Cmd("rpattern", "RECTANGULAR PATTERN", "rpattern", objfields() + [
        Fsel("d1", "Direction 1", AXES), Fint("q1", "Quantity 1", 3, 1, 1000), Flen("s1", "Distance 1", 20.0),
        Fcombo("m1", "Distance type 1", "spacing", [("Spacing", "spacing"), ("Extent", "extent")]), Fcheck("sym1", "Symmetric 1"),
        Fsel("d2", "Direction 2", AXES, req=False), Fint("q2", "Quantity 2", 1, 1, 1000), Flen("s2", "Distance 2", 20.0),
        Fcombo("m2", "Distance type 2", "spacing", [("Spacing", "spacing"), ("Extent", "extent")]), Fcheck("sym2", "Symmetric 2"),
        Ftext("skip", "Skip instances", "", ph="e.g. 2, 5"), Fcombo("op", "Bodies", "New", [("New bodies", "New"), ("Join", "Join")])],
        build=lambda v, op, vp: objs_to_op(v, op), hint="Directions: any straight edge, sketch line or axis. Instances count row by row from 1."))
    add(Cmd("cpattern", "CIRCULAR PATTERN", "pattern", objfields() + [
        Fsel("axis", "Axis", AXES), Fint("q", "Quantity", 6, 2, 1000), Fang("angle", "Total angle", 360.0),
        Fcheck("sym", "Symmetric"), Ftext("skip", "Skip instances", "", ph="e.g. 3"), Fcombo("op", "Bodies", "New", [("New bodies", "New"), ("Join", "Join")])],
        build=lambda v, op, vp: objs_to_op(v, op), hint="Axis: a circular edge, cylinder, sketch line, origin axis..."))
    add(Cmd("ppattern", "PATTERN ON PATH", "ppattern", objfields() + [
        Fsel("path", "Path", PATHS, True), Fint("q", "Quantity", 4, 2, 1000), Flen("dist", "Distance", 20.0),
        Fcombo("mode", "Distance type", "spacing", [("Spacing", "spacing"), ("Extent", "extent")]),
        Fcombo("orient", "Orientation", "identical", [("Identical", "identical"), ("Path direction", "path")]),
        Fcheck("flip", "Flip direction"), Ftext("skip", "Skip instances", "", ph="e.g. 2"),
        Fcombo("op", "Bodies", "New", [("New bodies", "New"), ("Join", "Join")])], build=lambda v, op, vp: objs_to_op(v, op)))
    add(Cmd("mirror", "MIRROR", "mirror", objfields() + [Fsel("plane", "Mirror plane", PLANES),
        Fcombo("op", "Bodies", "New", [("New bodies", "New"), ("Join", "Join")])], build=lambda v, op, vp: objs_to_op(v, op)))
    add(Cmd("thicken", "THICKEN", "thicken", [Fsel("faces", "Faces", {"face"}, True), Flen("thk", "Thickness", 2.0, 0.01),
        Fcombo("dir", "Direction", "one", [("One side", "one"), ("Symmetric", "sym")]), Fcheck("flip", "Flip"),
        Fcombo("op", "Operation", "New", [("New body", "New"), ("Join", "Join"), ("Cut", "Cut")])]))
    add(Cmd("bfill", "BOUNDARY FILL", "bfill", [Fsel("tools", "Tools", {"body", "plane"}, True), Ftext("cells", "Cells to keep", "all", ph="all, or e.g. 1, 3"),
        Fcheck("remove_tools", "Remove tool bodies"), Fcombo("op", "Operation", "New", [("New body", "New"), ("Join", "Join"), ("Cut", "Cut")])],
        hint="Pick bodies and planes that together enclose space; each enclosed cell is numbered in the preview."))
    # ---- modify ----
    M = "MODIFY"
    add(Cmd("presspull", "PRESS PULL", "presspull", [Fsel("ref", "Face or edge", {"face", "edge"}), Flen("d", "Distance / radius", 2.0)],
            hint="A face is offset (negative pushes in); an edge is filleted with that radius.", group=M))
    add(Cmd("fillet", "FILLET", "fillet", [
        Fcombo("mode", "Type", "constant", [("Constant radius", "constant"), ("Variable radius", "variable"), ("Rule fillet", "rule"), ("Full round", "fullround")]),
        Fsel("edges", "Edges / faces", {"edge", "face"}, True), Fsel("faces2", "Second faces", {"face"}, True, req=False, show=lambda v: v["mode"] == "rule"),
        Flen("r", "Radius", 2.0, 0.001, show=lambda v: v["mode"] != "fullround"),
        Flen("r2", "End radius", 4.0, 0.0, show=lambda v: v["mode"] == "variable"),
        Fcheck("chain", "Tangent chain", True, show=lambda v: v["mode"] in ("constant", "variable")),
        Fcheck("g2", "Curvature continuous (G2)")], op_t="fillet2", load=edge_load, group=M,
        hint="Rule: fillets every edge where the first faces meet the second faces (or all their edges). Full round: pick the middle face."))
    add(Cmd("chamfer", "CHAMFER", "chamfer", [
        Fsel("edges", "Edges / faces", {"edge", "face"}, True),
        Fcombo("mode", "Type", "eq", [("Equal distance", "eq"), ("Distance and angle", "da"), ("Two distances", "dd")]),
        Flen("d", "Distance", 1.0, 0.001), Fang("angle", "Angle", 45.0, 1, 89, show=lambda v: v["mode"] == "da"),
        Flen("d2", "Distance 2", 2.0, 0.001, show=lambda v: v["mode"] == "dd"), Fcheck("flip", "Flip sides", show=lambda v: v["mode"] != "eq"),
        Fcheck("chain", "Tangent chain", True)], op_t="chamfer2", load=edge_load, group=M))
    add(Cmd("shell", "SHELL", "shell", [Fsel("faces", "Faces to remove", {"face"}, True, req=False), Fsel("bodies", "Or whole bodies", {"body"}, True, req=False),
        Flen("thk", "Thickness", 2.0, 0.01), Fcombo("dir", "Direction", "in", [("Inside", "in"), ("Outside", "out"), ("Both", "both")])], group=M,
        build=lambda v, op, vp: dict(op, bodies=[("body", i) for i in op["bodies"]])))
    add(Cmd("draft", "DRAFT", "draft", [Fsel("pull", "Pull direction (plane)", PLANES), Fsel("faces", "Faces", {"face"}, True),
        Fang("angle", "Angle", 3.0, -60, 60), Fcheck("flip", "Flip pull direction")], group=M,
        hint="The plane fixes the faces' neutral line; faces tilt by the angle away from its normal."))
    add(Cmd("scale", "SCALE", "scale", [Fsel("bodies", "Bodies", {"body"}, True), Fsel("point", "Point", POINTS, req=False),
        Fcombo("type", "Scale type", "uniform", [("Uniform", "uniform"), ("Non-uniform", "nonuniform")]),
        Fnum("s", "Scale factor", 1.0, 0.0001, 10000, 4, show=lambda v: v["type"] == "uniform"),
        Fnum("sx", "X factor", 1.0, 0.0001, 10000, 4, show=lambda v: v["type"] == "nonuniform"),
        Fnum("sy", "Y factor", 1.0, 0.0001, 10000, 4, show=lambda v: v["type"] == "nonuniform"),
        Fnum("sz", "Z factor", 1.0, 0.0001, 10000, 4, show=lambda v: v["type"] == "nonuniform")], group=M,
        build=lambda v, op, vp: dict(op, bodies=[("body", i) for i in op["bodies"]]), hint="Without a point, each body scales about its own centre."))
    add(Cmd("combine", "COMBINE", "combine", [Fsel("target", "Target body", {"body"}), Fsel("tools", "Tool bodies", {"body"}, True),
        Fcombo("op", "Operation", "Join", [("Join", "Join"), ("Cut", "Cut"), ("Intersect", "Intersect")]),
        Fcheck("new", "Result as new body"), Fcheck("keep", "Keep tools")], group=M))
    add(Cmd("offsetface", "OFFSET FACE", "offsetface", [Fsel("faces", "Faces", {"face"}, True), Flen("d", "Distance", 1.0)], group=M,
            hint="Moves faces along their normals - works on flat and curved faces (e.g. make a hole bigger)."))
    add(Cmd("replaceface", "REPLACE FACE", "replaceface", [Fsel("faces", "Faces to replace", {"face"}, True), Fsel("target", "New face / plane", {"plane", "face"})], group=M))
    add(Cmd("splitface", "SPLIT FACE", "splitface", [Fsel("faces", "Faces to split", {"face"}, True), Fsel("tool", "Splitting tool", {"plane", "face", "body"}),
        Fcheck("extend", "Extend the tool", True)], group=M))
    add(Cmd("splitbody", "SPLIT BODY", "splitbody", [Fsel("body", "Body to split", {"body"}), Fsel("tool", "Splitting tool", {"plane", "face", "body"}),
        Fcheck("extend", "Extend the tool", True)], group=M))
    add(Cmd("move2", "MOVE / COPY", "move", [Fsel("bodies", "Bodies", {"body"}, True),
        Fcombo("mode", "Move type", "p2p", [("Point to point", "p2p"), ("Rotate about axis", "rotate"), ("Along axis", "along")]),
        Fsel("from", "From point", POINTS, show=lambda v: v["mode"] == "p2p"), Fsel("to", "To point", POINTS, show=lambda v: v["mode"] == "p2p"),
        Fsel("axis", "Axis", AXES, show=lambda v: v["mode"] != "p2p"), Fang("angle", "Angle", 90.0, -3600, 3600, show=lambda v: v["mode"] == "rotate"),
        Flen("dist", "Distance", 10.0, show=lambda v: v["mode"] == "along"), Fcheck("copy", "Create copy")], group=M,
        build=lambda v, op, vp: dict(op, bodies=[("body", i) for i in op["bodies"]])))
    add(Cmd("align", "ALIGN", "align", [Fsel("from", "From (on the moving body)", {"face", "edge", "point"}), Fsel("to", "To", {"face", "edge", "point", "plane", "axis"}),
        Fcheck("flip", "Flip"), Fcheck("copy", "Create copy")], group=M,
        hint="Face to face, axis to axis or point to point; the first pick's body moves."))
    add(Cmd("material", "PHYSICAL MATERIAL", "material", [Fsel("targets", "Bodies", {"body"}, True),
        Fcombo("mat", "Material", "Aluminium 6061", [(f"{n}  ({d} g/cm³)", n) for n, d in MATERIALS]), Fcheck("recolor", "Colour bodies to match", True)], group=M))
    add(Cmd("appearance", "APPEARANCE", "appearance", [Fsel("targets", "Bodies or faces", {"body", "face"}, True),
        Fcombo("color", "Appearance", (0.78, 0.79, 0.81), [(n, c) for n, c in APPEARANCES])], group=M))
    add(Cmd("defeature", "DELETE FACES", "defeature", [Fsel("faces", "Faces", {"face"}, True)], group=M,
            hint="Removes faces (holes, fillets, bosses...) and heals the body around them."))
    # ---- construct ----
    for how, title, fields, ic in (
        ("plane_offset", "OFFSET PLANE", [Fsel("plane", "Plane / face", PLANES), Flen("d", "Distance", 10.0)], "cplane"),
        ("plane_angle", "PLANE AT ANGLE", [Fsel("line", "Line / edge", AXES), Fang("angle", "Angle", 45.0)], "cplane"),
        ("plane_tangent", "TANGENT PLANE", [Fsel("face", "Cylindrical face", {"face"}), Fang("angle", "Angle", 0.0)], "cplane"),
        ("plane_mid", "MIDPLANE", [Fsel("a", "Plane / face 1", PLANES), Fsel("b", "Plane / face 2", PLANES)], "cplane"),
        ("plane_2edges", "PLANE THROUGH TWO EDGES", [Fsel("a", "Edge 1", AXES), Fsel("b", "Edge 2", AXES)], "cplane"),
        ("plane_3pts", "PLANE THROUGH THREE POINTS", [Fsel("a", "Point 1", POINTS), Fsel("b", "Point 2", POINTS), Fsel("c", "Point 3", POINTS)], "cplane"),
        ("plane_tanpt", "PLANE TANGENT TO FACE AT POINT", [Fsel("face", "Face", {"face"}), Fsel("point", "Point", POINTS)], "cplane"),
        ("plane_path", "PLANE ALONG PATH", [Fsel("path", "Path", PATHS, True), Fnum("t", "Distance (0-1)", 0.5, 0.0, 1.0)], "cplane"),
        ("axis_cyl", "AXIS THROUGH CYLINDER / CONE / TORUS", [Fsel("face", "Face", {"face"})], "caxis"),
        ("axis_perp_pt", "AXIS PERPENDICULAR AT POINT", [Fsel("face", "Face", {"face"}), Fsel("point", "Point", POINTS)], "caxis"),
        ("axis_2planes", "AXIS THROUGH TWO PLANES", [Fsel("a", "Plane 1", PLANES), Fsel("b", "Plane 2", PLANES)], "caxis"),
        ("axis_2pts", "AXIS THROUGH TWO POINTS", [Fsel("a", "Point 1", POINTS), Fsel("b", "Point 2", POINTS)], "caxis"),
        ("axis_edge", "AXIS THROUGH EDGE", [Fsel("edge", "Edge / line", AXES)], "caxis"),
        ("axis_perp_face", "AXIS PERPENDICULAR TO FACE AT POINT", [Fsel("face", "Face", {"face"}), Fsel("point", "Point", POINTS)], "caxis"),
        ("point_vertex", "POINT AT VERTEX", [Fsel("point", "Vertex / point", POINTS)], "cpoint"),
        ("point_2edges", "POINT THROUGH TWO EDGES", [Fsel("a", "Edge 1", AXES), Fsel("b", "Edge 2", AXES)], "cpoint"),
        ("point_3planes", "POINT THROUGH THREE PLANES", [Fsel("a", "Plane 1", PLANES), Fsel("b", "Plane 2", PLANES), Fsel("c", "Plane 3", PLANES)], "cpoint"),
        ("point_center", "POINT AT CENTRE OF CIRCLE / SPHERE / TORUS", [Fsel("obj", "Circle / sphere / torus", {"edge", "curve", "face"})], "cpoint"),
        ("point_edge_plane", "POINT AT EDGE AND PLANE", [Fsel("edge", "Edge / axis", AXES), Fsel("plane", "Plane", PLANES)], "cpoint"),
        ("point_path", "POINT ALONG PATH", [Fsel("path", "Path", PATHS, True), Fnum("t", "Distance (0-1)", 0.5, 0.0, 1.0)], "cpoint")):
        add(construct_cmd(how, title, fields, ic))
    return C_


# ---- sketch-environment commands (run inside the sketch, not recorded as separate timeline steps) ----
def make_sketch_commands():
    out = {}
    def sk_build(fn):
        def b(v, op, vp):
            if not vp.skedit: raise CmdError("start or edit a sketch first")
            op = dict(op); op.update(t="skmod", fn=fn, sk=vp.skedit["si"]); return op
        return b
    def add(name, title, icon, fields, hint):
        out[name] = Cmd(name, title, icon, fields, "skmod", hint=hint, build=sk_build(name[3:]), group="SKETCH")
    add("sk_offset", "OFFSET", "skoffset", [
        Fsel("curves", "Curves", {"curve"}, True), Flen("d", "Offset position", 2.0, 0.0), Fcheck("flip", "Flip side"),
        Fcheck("chain", "Chain selection", True)],
        "Pick a curve (connected curves are included). Positive offsets go to one side; tick Flip for the other.")
    add("sk_mirror", "MIRROR", "skmirror", [
        Fsel("objs", "Objects", {"curve", "point"}, True), Fsel("line", "Mirror line", {"curve"})],
        "Pick the curves to mirror, then a line of this sketch. The copies stay symmetric.")
    add("sk_cpattern", "CIRCULAR PATTERN", "skcpat", [
        Fsel("objs", "Objects", {"curve"}, True), Fsel("center", "Center point", {"point"}), Fint("n", "Quantity", 6, 2, 1000),
        Fang("ang", "Angle", 360.0), Fcheck("sym", "Symmetric")],
        "Pick curves and a centre point (a sketch point, the origin or a body corner).")
    add("sk_rpattern", "RECTANGULAR PATTERN", "skrpat", [
        Fsel("objs", "Objects", {"curve"}, True), Fsel("d1", "Direction 1", {"curve", "axis"}),
        Fcombo("mode1", "Distance type", "spacing", [("Spacing", "spacing"), ("Extent", "extent")]),
        Fint("n1", "Quantity 1", 3, 1, 1000), Flen("s1", "Distance 1", 10.0), Fcheck("sym1", "Symmetric 1"),
        Fsel("d2", "Direction 2", {"curve", "axis"}, req=False), Fcombo("mode2", "Distance type 2", "spacing", [("Spacing", "spacing"), ("Extent", "extent")]),
        Fint("n2", "Quantity 2", 1, 1, 1000), Flen("s2", "Distance 2", 10.0), Fcheck("sym2", "Symmetric 2")],
        "Direction 2 defaults to 90° from direction 1.")
    add("sk_move", "MOVE / COPY", "skmove", [
        Fsel("objs", "Objects", {"curve", "point"}, True),
        Fcombo("mode", "Move type", "free", [("Translate / rotate", "free"), ("Point to point", "ptp")]),
        Flen("dx", "X distance", 0.0, show=lambda v: v["mode"] == "free"), Flen("dy", "Y distance", 0.0, show=lambda v: v["mode"] == "free"),
        Fang("ang", "Angle", 0.0, show=lambda v: v["mode"] == "free"),
        Fsel("pivot", "Rotation centre", {"point"}, req=False, show=lambda v: v["mode"] == "free"),
        Fsel("p0", "From point", {"point"}, show=lambda v: v["mode"] == "ptp"), Fsel("p1", "To point", {"point"}, show=lambda v: v["mode"] == "ptp"),
        Fcheck("copy", "Create copy")],
        "Moves the selection in the sketch's own X / Y; constraints and dimensions still apply afterwards.")
    add("sk_scale", "SKETCH SCALE", "skscale", [
        Fsel("objs", "Objects", {"curve", "point"}, True), Fsel("base", "Point", {"point"}), Fnum("k", "Scale factor", 2.0, 0.0001, 10000.0, 4),
        Fcheck("copy", "Create copy")],
        "Scales the selection about the point; dimensions inside the selection scale too.")
    add("sk_project", "PROJECT", "project", [
        Fcombo("what", "Selection filter", "geom", [("Edges, faces, points", "geom"), ("Bodies (outline)", "body"), ("Other sketches", "sketch")]),
        Fsel("refs", "Geometry", {"edge", "face", "point"}, True, show=lambda v: v["what"] == "geom"),
        Fsel("brefs", "Bodies", {"body"}, True, show=lambda v: v["what"] == "body"),
        Fsel("srefs", "Sketch curves", {"curve"}, True, show=lambda v: v["what"] == "sketch"),
        Fcheck("cons", "Projected as construction")],
        "Projected edges stay linked to the model and update when it changes. They are fixed (purple).")
    add("sk_intersect", "INTERSECT", "intersect", [
        Fsel("refs", "Bodies / faces", {"body", "face"}, True), Fcheck("cons", "As construction")],
        "Adds the curves where the sketch plane cuts the picked bodies or faces.")
    add("sk_include", "INCLUDE 3D GEOMETRY", "include3d", [Fsel("refs", "Edges", {"edge"}, True)],
        "Brings model edges into the sketch where they are in space (not flattened). They stay linked to the model.")
    add("sk_projsurf", "PROJECT TO SURFACE", "projsurf", [Fsel("curves", "Sketch curves", {"curve"}, True), Fsel("faces", "Faces", {"face"}, True)],
        "Projects sketch curves along the sketch normal onto faces (e.g. to wrap a path round a cylinder).")
    for name in ("sk_project",):
        c = out[name]; old = c._build
        def b(v, op, vp, old=old):
            op = old(v, op, vp); w = op.pop("what", "geom")
            op["refs"] = (op.get("refs") or []) if w == "geom" else (op.get("brefs") or []) if w == "body" else (op.get("srefs") or [])
            op.pop("brefs", None); op.pop("srefs", None); return op
        c._build = b
    return out


# ---- SURFACE workspace commands ----
SURF_OPS = ("s_extrude", "s_revolve", "s_sweep", "s_loft", "patch", "ruled", "s_offset", "trim", "untrim", "s_extend", "stitch",
            "unstitch", "revnormal", "s_delface")
CURVES = {"profile", "curve", "edge"}

def _first_edge(v, vp, key):
    r = (v.get(key) or [None])[0]
    if not r or r[0] != "edge": return None
    b = vp.bodies[r[1]]; e = b.eds[r[2]]; fs = faces_of_edge_in(b.shape, e)
    return e, (fs[0] if fs else None)

def s_ext_handle(v, vp):
    if not v.get("profiles"): return None
    try:
        r = v["profiles"][0]; n = vp.surf_dir(v["profiles"], (v.get("dirref") or [None])[0], v.get("chain", True))
        o = centroid(vp.ref_face(r)) if r[0] == "profile" else vp.ref_point(r)
        return o, n, "d"
    except Exception: return None

def s_offset_handle(v, vp):
    r = (v.get("faces") or [None])[0]
    if not r: return None
    try:
        f = vp.bodies[r[1]].faces[0] if r[0] == "body" else vp.ref_face(r)
        p, n = normal_at(f, V(*r[3]) if len(r) > 3 else centroid(f)); return p, n, "d"
    except Exception: return None

def s_edge_handle(v, vp, key="edges"):
    try:
        x = _first_edge(v, vp, key)
        if not x or x[1] is None: return None
        e, f = x; p = _emid(e); out, n = _outward_dir(f, e, p)
        if v.get("kind") in ("normal", "perpendicular"): out = n
        if v.get("flip"): out = -out
        if v.get("kind") == "direction": return None
        return p, out, "d"
    except Exception: return None

def make_surface_commands():
    out = {}
    def add(c): c.group = "SURFACE"; out[c.name] = c
    two = lambda v: v["dir"] == "two"
    chain = lambda: Fcheck("chain", "Chain selection", True)
    add(Cmd("s_extrude", "EXTRUDE (SURFACE)", "sextrude", [
        Fsel("profiles", "Profiles / curves", CURVES, True), chain(),
        Fsel("dirref", "Direction", {"plane", "axis"}, req=False),
        Fcombo("dir", "Direction", "one", SIDES), Flen("d", "Distance", 10.0),
        Flen("d2", "Side 2 distance", 5.0, show=two), Fang("taper", "Taper angle", 0.0, -89, 89)],
        hint="Pick sketch curves, profiles or model edges - they're pulled into an open surface (no end caps). "
             "Edges that aren't flat need a direction. Drag the arrow or type a distance.", handle=s_ext_handle))
    add(Cmd("s_revolve", "REVOLVE (SURFACE)", "srevolve", [
        Fsel("profiles", "Profiles / curves", CURVES, True), Fsel("axis", "Axis", AXES), chain(),
        Fcombo("extent", "Type", "full", [("Full (360°)", "full"), ("Angle", "angle")]),
        Fang("angle", "Angle", 180.0, show=lambda v: v["extent"] == "angle")],
        hint="Spins curves round an axis into a surface of revolution."))
    add(Cmd("s_sweep", "SWEEP (SURFACE)", "ssweep", [
        Fsel("profiles", "Profile curves", CURVES, True), Fsel("path", "Path", PATHS, True), chain(),
        Fcombo("orient", "Orientation", "perp", [("Perpendicular", "perp"), ("Parallel", "parallel")])],
        hint="Drags a curve along a path. The profile should start at the path's beginning."))
    add(Cmd("s_loft", "LOFT (SURFACE)", "sloft", [
        Fsel("sections", "Sections", CURVES | POINTS, True), chain(),
        Fcheck("closed", "Closed (back to the first section)"), Fcheck("ruled", "Straight (ruled) between sections")],
        hint="Pick open or closed curves (or edges) in order; a point can start or end the loft."))
    add(Cmd("patch", "PATCH", "patch", [
        Fsel("boundary", "Boundary", {"curve", "edge"}, True), chain(),
        Fcombo("cont", "Continuity", "G0", [("Connected (G0)", "G0"), ("Tangent (G1)", "G1"), ("Curvature (G2)", "G2")]),
        Fsel("points", "Points to pass through", POINTS, True, req=False)],
        hint="Fills a closed loop of curves or open surface edges (one edge picks its whole loop). Tangent / curvature "
             "blend into the surfaces the edges belong to."))
    add(Cmd("ruled", "RULED SURFACE", "ruled", [
        Fsel("edges", "Edges", {"edge", "curve"}, True), Fcheck("chain", "Tangent chain", False),
        Fcombo("kind", "Type", "tangent", [("Tangent", "tangent"), ("Normal", "normal"), ("Direction", "direction")]),
        Fsel("dirref", "Direction", {"plane", "axis"}, show=lambda v: v["kind"] == "direction"),
        Flen("d", "Distance", 10.0), Fang("ang", "Angle", 0.0, -180, 180), Fcheck("flip", "Flip")],
        hint="A strip hanging off edges: carrying on along the surface (tangent), standing up from it (normal) or in a fixed direction.",
        handle=s_edge_handle))
    add(Cmd("s_offset", "OFFSET (SURFACE)", "soffset", [
        Fsel("faces", "Faces", {"face", "body"}, True), Flen("d", "Distance", 2.0)],
        hint="A new surface a set distance from faces (negative goes the other way).", handle=s_offset_handle))
    add(Cmd("trim", "TRIM", "trim3d", [
        Fsel("tool", "Trimming tool", {"body", "face", "plane", "curve"}), Fsel("remove", "Areas to remove", {"spot"}, True)],
        hint="Pick the tool (a surface, face, plane or sketch curve), then click each part of the surfaces to cut away. "
             "Click parts of the tool surface too to trim it back."))
    add(Cmd("untrim", "UNTRIM", "untrim", [
        Fsel("faces", "Faces", {"face", "body"}, True),
        Fcombo("mode", "Type", "loops", [("Fill holes", "loops"), ("Back to the natural boundary", "all")]),
        Flen("ext", "Extend flat / open faces by", 10.0, 0.0, show=lambda v: v["mode"] == "all")],
        hint="Removes the trimming from surface faces: holes close, or the face grows back to its full surface."))
    add(Cmd("s_extend", "EXTEND", "sextend", [
        Fsel("edges", "Edges", {"edge"}, True), Fcheck("chain", "Tangent chain", True), Flen("d", "Distance", 10.0),
        Fcombo("kind", "Type", "natural", [("Natural", "natural"), ("Tangent", "tangent"), ("Perpendicular", "perpendicular")])],
        hint="Grows a surface past its open edges.", handle=s_edge_handle))
    add(Cmd("stitch", "STITCH", "stitch", [
        Fsel("sbodies", "Surfaces", {"body"}, True), Flen("tol", "Tolerance", 0.01, 1e-6, 100.0)],
        hint="Joins surfaces along edges that touch. A fully closed result becomes a solid body."))
    add(Cmd("unstitch", "UNSTITCH", "unstitch", [Fsel("faces", "Faces / bodies", {"face", "body"}, True)],
        hint="Breaks faces off into their own surface bodies (a whole body splits into one body per face)."))
    add(Cmd("revnormal", "REVERSE NORMAL", "revnormal", [Fsel("targets", "Surfaces / faces", {"body", "face"}, True)],
        hint="Flips which side of a surface is the front (the back shows tinted)."))
    add(Cmd("s_delface", "DELETE FACE", "sdelete", [Fsel("faces", "Faces", {"face"}, True)],
        hint="Removes faces and leaves the hole open - a solid becomes a surface body. (Delete Faces in Solid heals the gap instead.)"))
    return out


# ---- Inspect commands (analyses live in the browser's Analysis folder, not the timeline) ----
def make_inspect_commands():
    out = {}
    def add(c): c.group = "INSPECT"; out[c.name] = c
    add(Cmd("section", "SECTION ANALYSIS", "section", [
        Fsel("plane", "Plane", PLANES), Flen("d", "Offset", 0.0), Fcheck("flip", "Flip (show the other side)"),
        Fcheck("hatch", "Hatch the cut faces", True)],
        hint="Cuts the view along a plane or flat face so you can see inside. Drag the arrow or type an offset. "
             "It stays in the browser (Analysis) - untick it to hide, right-click to edit.", handle=plane_offset_handle))
    add(Cmd("zebra", "ZEBRA ANALYSIS", "zebra", [Fsel("bodies", "Bodies (none = all)", {"body"}, True, req=False)],
        hint="Reflected stripes show how smoothly faces meet: stripes that stay continuous across an edge = tangent / curvature continuous."))
    add(Cmd("draftan", "DRAFT ANALYSIS", "draftan", [
        Fsel("dir", "Pull direction", PLANES | AXES), Fang("angle", "Draft angle needed", 1.0, 0.0, 89.0), Fcheck("flip", "Flip direction"),
        Fsel("bodies", "Bodies (none = all)", {"body"}, True, req=False)],
        hint="Green = enough positive draft, red = enough negative draft, yellow = not enough to release from a mould."))
    add(Cmd("curvmap", "CURVATURE MAP", "curvmap", [
        Fsel("bodies", "Bodies (none = all)", {"body"}, True, req=False), Fnum("max", "Red at curvature (1/mm, 0 = auto)", 0.0, 0.0, 100.0, 4)],
        hint="Colours faces by how tightly they curve: blue = flat, red = sharpest."))
    add(Cmd("comb", "CURVATURE COMB", "comb", [
        Fsel("edges", "Edges / sketch curves", {"edge", "curve"}, True), Fnum("scale", "Comb size", 1.0, 0.05, 20.0, 2)],
        hint="Spikes show curvature along a curve: longer = tighter; smooth comb ends = smooth joins."))
    add(Cmd("interference", "INTERFERENCE", "interference", [Fsel("bodies", "Bodies (none = all)", {"body"}, True, req=False)],
        hint="Finds where solid bodies overlap. Overlaps show red; the list says how much."))
    return out

class MeasurePanel(W.QFrame):
    def __init__(s, win):
        super().__init__(win.vp); s.win, s.vp = win, win.vp
        s.setObjectName("cmd"); s.setFixedWidth(330)
        lay = W.QVBoxLayout(s); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        hd = W.QLabel("MEASURE"); hd.setObjectName("cmdhdr"); lay.addWidget(hd)
        body = W.QWidget(); bl = W.QVBoxLayout(body); bl.setContentsMargins(12, 10, 12, 8); bl.setSpacing(7); lay.addWidget(body)
        form = W.QFormLayout(); form.setHorizontalSpacing(10); bl.addLayout(form)
        s.kind = W.QComboBox(); s.kind.addItem("Faces, edges, vertices", "geom"); s.kind.addItem("Bodies", "body")
        s.dist = W.QComboBox(); s.dist.addItem("Minimum distance", "min"); s.dist.addItem("Centre to centre", "centre"); s.dist.addItem("Maximum distance", "max")
        s.prec = W.QSpinBox(); s.prec.setRange(0, 6); s.prec.setValue(2)
        form.addRow("Select", s.kind); form.addRow("Distance", s.dist); form.addRow("Precision", s.prec)
        s.out = W.QLabel(); s.out.setWordWrap(True); s.out.setTextInteractionFlags(C.Qt.TextSelectableByMouse)
        s.out.setStyleSheet("background:white;border:1px solid #d6d6d6;padding:8px;color:#222"); s.out.setMinimumHeight(90); bl.addWidget(s.out)
        h = W.QLabel("Click one item for its size, two for the distance and angle between them. A third click starts again.")
        h.setObjectName("hint"); h.setWordWrap(True); bl.addWidget(h)
        row = W.QHBoxLayout(); row.setContentsMargins(12, 0, 12, 10); row.addStretch(1)
        rs, cl = W.QPushButton("Restart"), W.QPushButton("Close"); cl.setObjectName("primary"); row.addWidget(rs); row.addWidget(cl); lay.addLayout(row)
        rs.clicked.connect(s.restart); cl.clicked.connect(win.measure_stop)
        for w_ in (s.kind, s.dist): w_.currentIndexChanged.connect(s.changed)
        s.prec.valueChanged.connect(s.changed)
        s.move(16, 16); s.show(); s.raise_(); s.render_out()
    def changed(s, *_):
        m = s.vp.meas
        if m is None: return
        if m.get("kind") != s.kind.currentData(): m["picks"] = []
        m.update(kind=s.kind.currentData(), dist=s.dist.currentData(), prec=s.prec.value()); s.vp.meas_compute()
    def restart(s): s.vp.meas["picks"] = []; s.vp.meas_compute()
    def render_out(s):
        m = s.vp.meas
        if m is None: return
        P = m["picks"]
        names = {"vertex": "Vertex", "edge": "Edge", "face": "Face", "body": "Body", "curve": "Sketch curve", "spoint": "Sketch point", "origin": "Origin", "cpoint": "Point"}
        lines = [f"<b>Selection {i + 1}:</b> {names.get(r[0], r[0])}" for i, r in enumerate(P)] or ["<i>Click something to measure.</i>"]
        prec = m.get("prec", 2)
        def rp(t): return re.sub(r"-?\d+\.\d+", lambda q: fmt(float(q.group(0)), prec), t)
        for k, v in m.get("out", []): lines.append(f"<b>{k}</b>&nbsp;&nbsp;{rp(v)}" if k else f"<span style='color:#b3261e'>{v}</span>")
        s.out.setText("<br>".join(lines))

class InspectWin:
    # ---------------- actions / menus ----------------
    def make_inspect_actions(s, act):
        A = {}
        A["measure"] = act("Measure", "I", s.measure_start, "measure", "Distance, angle, length, area, volume and coordinates.")
        A["section"] = act("Section Analysis", None, lambda: s.open_cmd("section"), "section")
        A["interference"] = act("Interference", None, lambda: s.open_cmd("interference"), "interference")
        A["zebra"] = act("Zebra Analysis", None, lambda: s.open_cmd("zebra"), "zebra")
        A["draftan"] = act("Draft Analysis", None, lambda: s.open_cmd("draftan"), "draftan")
        A["curvmap"] = act("Curvature Map Analysis", None, lambda: s.open_cmd("curvmap"), "curvmap")
        A["comb"] = act("Curvature Comb Analysis", None, lambda: s.open_cmd("comb"), "comb")
        A["com"] = act("Center of Mass", None, s.toggle_com, "com"); A["com"].setCheckable(True)
        A["props"] = act("Properties", None, s.properties_dialog, "props", "Mass, volume, area, centre of mass, inertia.")
        A["clear_ana"] = act("Hide All Analyses", None, s.clear_analyses, "eyeoff")
        A["selfilter"] = act("Selection Filters", None, lambda: None, "selfilter")
        fm = W.QMenu(s)
        s.filter_acts = {}
        for key, label in (("bodies", "Bodies"), ("faces", "Faces"), ("edges", "Edges"), ("profiles", "Sketch profiles")):
            a = fm.addAction(label); a.setCheckable(True); a.setChecked(True); a.toggled.connect(lambda on, k=key: s.set_filter(k, on)); s.filter_acts[key] = a
        fm.addSeparator(); fm.addAction("Select all", lambda: [a.setChecked(True) for a in s.filter_acts.values()])
        A["selfilter"].setMenu(fm); A["selfilter"].triggered.connect(lambda: fm.exec(G.QCursor.pos()))
        A["selall"] = act("Select All Bodies", "Ctrl+A", s.select_all_bodies, "body")
        A["window"] = act("Window Selection (drag on empty space)", None, lambda: s.statusBar().showMessage(
            "Drag on empty space: left-to-right picks what is fully inside, right-to-left what the box touches. Ctrl / Shift adds.", 8000), "selwin")
        A["search"] = act("Command Search", "Ctrl+K", s.focus_search, "search")
        return A

    def inspect_groups(s, A):
        menu = [A["measure"], A["interference"], A["comb"], A["zebra"], A["draftan"], A["curvmap"], A["section"], None, A["com"], A["props"], None, A["clear_ana"]]
        sel = [A["selfilter"], A["selall"], A["window"]]
        return [("INSPECT", (A["measure"], A["section"], A["interference"]), menu), ("SELECT", (A["selfilter"],), sel)]

    def set_filter(s, key, on):
        f = s.vp.sel_filter
        if on: f.add(key)
        else: f.discard(key)
        if not f: s.filter_acts[key].setChecked(True); return
        s.statusBar().showMessage("Selecting: " + ", ".join(sorted(f)), 4000)

    def select_all_bodies(s):
        v = s.vp
        if v.skedit or v.cmd: return
        v.msel = {i for i, b in enumerate(v.bodies) if b.visible and getattr(b, "selectable", True)}
        v.sel_body = min(v.msel) if v.msel else None; v.sel.clear(); v.sel_face = None; v.update(); s.refresh()

    # ---------------- measure ----------------
    def measure_start(s):
        v = s.vp
        if v.cmd: s.cmd_cancel()
        if v.skedit: v.sk_set_tool(None)
        if v.meas is not None: return
        v.meas = dict(picks=[], kind="geom", dist="min", prec=2, out=[])
        s.meas_panel = MeasurePanel(s); v.meas_cb = s.meas_panel.render_out; v.setFocus(); v.update()
        s.statusBar().showMessage("Measure: click faces, edges or vertices. Esc to finish.", 6000)
    def measure_stop(s):
        v = s.vp; v.meas = None; v.meas_cb = None; v.cmd_hover = None
        if getattr(s, "meas_panel", None): s.meas_panel.hide(); s.meas_panel.deleteLater(); s.meas_panel = None
        v.update()

    # ---------------- analyses ----------------
    def analysis_ok(s, op, edit_index=None):
        v = s.vp; v.ana_prev = None
        if op["t"] == "interference": return s.interference_report(op)
        a = {k: x for k, x in op.items() if k != "icon"}; a["visible"] = True
        if a["t"] in ("zebra", "draftan", "curvmap", "interference") and isinstance(a.get("bodies"), list): pass
        ed = getattr(s, "_ana_edit", None); s._ana_edit = None
        if ed is not None and ed < len(v.analyses): a["name"] = v.analyses[ed].get("name"); v.analyses[ed] = a
        else:
            n = sum(1 for x in v.analyses if x["t"] == a["t"]) + 1; a["name"] = f"{ANA_NAMES.get(a['t'], a['t'])}{n}"; v.analyses.append(a)
        v._anacache = {k: x for k, x in v._anacache.items() if k[0] in ("props",)}; v.update(); s.refresh()

    def interference_report(s, op):
        v = s.vp; ids = op.get("bodies") or [i for i, b in enumerate(v.bodies) if b.visible]
        W.QApplication.setOverrideCursor(C.Qt.WaitCursor)
        try: res = v.interference(ids)
        finally: W.QApplication.restoreOverrideCursor()
        v.interf = []
        for i, j, sh, vol in res:
            try: v.interf.append(Body(sh))
            except Exception: pass
        v.update()
        dlg = W.QDialog(s); dlg.setWindowTitle("Interference"); dlg.resize(460, 300); lay = W.QVBoxLayout(dlg)
        if not res: lay.addWidget(W.QLabel(f"No interference between the {len(ids)} bodies checked."))
        else:
            lay.addWidget(W.QLabel(f"{len(res)} overlap(s) found - shown in red in the view."))
            tab = W.QTableWidget(len(res), 3); tab.setHorizontalHeaderLabels(["Body", "Body", "Overlap volume"])
            for r, (i, j, sh, vol) in enumerate(res):
                for c, t in enumerate((v.body_name(i), v.body_name(j), fvol(vol))): tab.setItem(r, c, W.QTableWidgetItem(t))
            tab.horizontalHeader().setStretchLastSection(True); tab.verticalHeader().hide(); lay.addWidget(tab)
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Close); bb.rejected.connect(dlg.reject); bb.accepted.connect(dlg.accept); lay.addWidget(bb)
        dlg.exec(); v.interf = []; v.update()

    def clear_analyses(s):
        v = s.vp
        for a in v.analyses: a["visible"] = False
        v.com_marks = False; v.interf = []
        if "com" in getattr(s, "IA", {}): s.IA["com"].setChecked(False)
        v.update(); s.refresh()

    def toggle_com(s):
        v = s.vp; v.com_marks = not v.com_marks; s.IA["com"].setChecked(v.com_marks); v.update()

    def properties_dialog(s):
        v = s.vp
        if not v.bodies: return s.statusBar().showMessage("There are no bodies to inspect.", 5000)
        dlg = W.QDialog(s); dlg.setWindowTitle("Properties"); dlg.resize(470, 520); lay = W.QVBoxLayout(dlg)
        pick = W.QComboBox()
        for i, b in enumerate(v.bodies): pick.addItem(v.body_name(i), i)
        if v.sel_body is not None and v.sel_body < len(v.bodies): pick.setCurrentIndex(v.sel_body)
        lay.addWidget(pick); tab = W.QTableWidget(0, 2); tab.setHorizontalHeaderLabels(["Property", "Value"])
        tab.horizontalHeader().setStretchLastSection(True); tab.verticalHeader().hide(); tab.setEditTriggers(W.QAbstractItemView.NoEditTriggers); lay.addWidget(tab)
        def fill():
            b = v.bodies[pick.currentData()]; pr = v.body_props(b); rows = []
            rows.append(("Type", "Surface body" if getattr(b, "surface", False) else "Solid body"))
            rows.append(("Material", b.material or "none (set one with Physical Material)"))
            if pr["density"]: rows.append(("Density", f"{fmt(pr['density'], 3)} g/cm³"))
            if pr["mass"] is not None: rows.append(("Mass", f"{fmt(pr['mass'], 3)} g"))
            rows += [("Volume", fvol(pr["volume"])), ("Area", farea(pr["area"]))]
            d = pr["hi"] - pr["lo"]; rows.append(("Bounding box", f"{flen(d.x)} × {flen(d.y)} × {flen(d.z)}"))
            c = pr["com"]; rows.append(("Center of mass", f"{flen(c.x)}, {flen(c.y)}, {flen(c.z)}"))
            if pr.get("inertia") and not getattr(b, "surface", False):
                u = "g·mm²" if pr["density"] else "mm⁵ (no material)"
                rows.append(("Principal moments", ", ".join(fmt(x, 1) for x in pr["inertia"]) + f"  {u}"))
                M = pr["inertia_matrix"]
                for nm, (i, j) in (("Ixx", (0, 0)), ("Iyy", (1, 1)), ("Izz", (2, 2)), ("Ixy", (0, 1)), ("Iyz", (1, 2)), ("Ixz", (0, 2))):
                    rows.append((f"{nm} (about centre of mass)", f"{fmt(M[i][j], 1)} {u}"))
            tab.setRowCount(len(rows))
            for r, (k, x) in enumerate(rows): tab.setItem(r, 0, W.QTableWidgetItem(k)); tab.setItem(r, 1, W.QTableWidgetItem(x))
            tab.resizeColumnToContents(0)
        pick.currentIndexChanged.connect(lambda _: fill()); fill()
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Close); copy = bb.addButton("Copy", W.QDialogButtonBox.ActionRole)
        copy.clicked.connect(lambda: W.QApplication.clipboard().setText("\n".join(f"{tab.item(r, 0).text()}\t{tab.item(r, 1).text()}" for r in range(tab.rowCount()))))
        bb.rejected.connect(dlg.reject); lay.addWidget(bb); dlg.exec()

    # ---------------- browser ----------------
    def insp_browser(s, root, item, folders):
        v = s.vp
        org = item(root, "Origin", "folder", "grp:or", any(v.origin_vis.values()))
        for k, nm in (("XY", "XY plane (ground)"), ("XZ", "XZ plane (front)"), ("YZ", "YZ plane (right)"), ("axes", "Axes")):
            item(org, nm, "cplane" if k != "axes" else "caxis", f"or:{k}", v.origin_vis[k])
        if v.analyses:
            an = item(root, "Analysis", "folder", "grp:an", any(a.get("visible", True) for a in v.analyses)); folders.append(an)
            for i, a in enumerate(v.analyses):
                it = item(an, a.get("name") or ANA_NAMES.get(a["t"], a["t"]), a["t"], f"an:{i}", a.get("visible", True))
                it.setToolTip(0, "Untick to hide · right-click to edit or delete")
        if v.views:
            nv = item(root, "Named Views", "folder", "grp:nv", True); folders.append(nv)
            for name in v.views: item(nv, name, "view", f"nv:{name}", True).setToolTip(0, "Click to go to this view · right-click for more")

    def insp_vis(s, ref, on):
        v = s.vp
        if ref == "grp:or" or ref.startswith("or:"):
            for k in (list(v.origin_vis) if ref == "grp:or" else [ref[3:]]): v.origin_vis[k] = on
        elif ref == "grp:an" or ref.startswith("an:"):
            for i in (range(len(v.analyses)) if ref == "grp:an" else [int(ref[3:])]): v.analyses[i]["visible"] = on
        elif ref == "grp:nv" or ref.startswith("nv:"): pass
        else: return False
        v.update(); C.QTimer.singleShot(0, s.refresh); return True

    def insp_click(s, ref):
        if ref.startswith("nv:") and ref[3:] in s.vp.views: s.vp.go_cam(s.vp.views[ref[3:]]); return True
        return False

    def insp_browser_menu(s, m, ref):
        v = s.vp
        if ref.startswith("bd:"):
            bi = int(ref[3:]); b = v.bodies[bi]
            m.addAction("Rename...", lambda: s.rename_body(bi)); m.addAction("Properties", lambda: (setattr(v, "sel_body", bi), s.properties_dialog()))
            m.addSeparator(); m.addAction("Isolate", lambda: s.isolate(bi)); m.addAction("Show All Bodies", s.unisolate)
            op = m.addMenu("Opacity")
            for pc in (100, 75, 50, 25, 10):
                a = op.addAction(f"{pc}%"); a.setCheckable(True); a.setChecked(abs(getattr(b, "opacity", 1.0)*100 - pc) < 1)
                a.triggered.connect(lambda _=False, pc=pc: s.set_opacity(bi, pc/100))
            a = m.addAction("Selectable"); a.setCheckable(True); a.setChecked(getattr(b, "selectable", True))
            a.triggered.connect(lambda on: s.set_selectable(bi, on))
            m.addSeparator(); m.addAction("Delete", lambda: (setattr(v, "sel_body", bi), v.msel.clear(), s.delete_selected()))
            return True
        if ref.startswith("an:"):
            i = int(ref[3:])
            m.addAction("Edit...", lambda: s.edit_analysis(i)); m.addAction("Delete", lambda: s.delete_analysis(i)); return True
        if ref.startswith("nv:"):
            name = ref[3:]
            m.addAction("Go to view", lambda: v.go_cam(v.views[name])); m.addAction("Update to current view", lambda: s.save_view(name))
            m.addAction("Rename...", lambda: s.rename_view(name)); m.addAction("Delete", lambda: s.delete_view(name)); return True
        if ref == "grp:nv": m.addAction("Save current view...", s.save_view_dialog); return True
        return False

    def edit_analysis(s, i):
        v = s.vp; a = v.analyses[i]; s._ana_edit = i; s.open_cmd(a["t"], vals=COMMANDS[a["t"]].load(a, v))
    def delete_analysis(s, i):
        del s.vp.analyses[i]; s.vp.update(); s.refresh()

    def rename_body(s, bi):
        v = s.vp; cur = v.body_name(bi)
        name, ok = W.QInputDialog.getText(s, "Rename body", "Name:", text=cur)
        if ok and name.strip() and name.strip() != cur: s.run("Rename", {"t": "bodyname", "bi": bi, "name": name.strip(), "quiet": True}); s.refresh()
    def isolate(s, bi):
        for i, b in enumerate(s.vp.bodies): b.visible = i == bi
        s.vp._edge_cache = None; s.vp.update(); s.refresh()
    def unisolate(s):
        for b in s.vp.bodies: b.visible = True
        s.vp._edge_cache = None; s.vp.update(); s.refresh()
    def set_opacity(s, bi, x):
        s.vp.bodies[bi].opacity = x; s.vp.update()
    def set_selectable(s, bi, on):
        s.vp.bodies[bi].selectable = on; s.vp._edge_cache = None; s.vp.update()

    # ---------------- views / display ----------------
    def display_menu(s, m):
        v = s.vp; m.clear()
        vs = m.addMenu("Visual Style"); grp = G.QActionGroup(vs)
        for label, key in VSTYLES:
            a = vs.addAction(label); a.setCheckable(True); a.setChecked(v.vstyle == key); grp.addAction(a)
            a.triggered.connect(lambda _=False, k=key: (setattr(v, "vstyle", k), v.update()))
        cam = m.addMenu("Camera"); g2 = G.QActionGroup(cam)
        for label, on in (("Perspective", False), ("Orthographic", True)):
            a = cam.addAction(label); a.setCheckable(True); a.setChecked(v.ortho == on); g2.addAction(a)
            a.triggered.connect(lambda _=False, on=on: (setattr(v, "ortho", on), v.update()))
        m.addSeparator()
        m.addAction("Look At (selected face or sketch)", lambda: v.look_at_selection() or s.statusBar().showMessage("Select a face first.", 4000))
        m.addAction("Save Named View...", s.save_view_dialog)
        m.addAction("Set Current View as Home", s.set_home); m.addAction("Reset Home", s.reset_home)
        m.addSeparator()
        org = m.addMenu("Origin")
        for k, nm in (("XY", "XY plane"), ("XZ", "XZ plane"), ("YZ", "YZ plane"), ("axes", "Axes")):
            a = org.addAction(nm); a.setCheckable(True); a.setChecked(v.origin_vis[k])
            a.toggled.connect(lambda on, k=k: (v.origin_vis.__setitem__(k, on), v.update(), s.refresh()))

    def cube_menu(s, gpos):
        v = s.vp; m = W.QMenu(s)
        m.addAction("Go Home", v.home); m.addSeparator()
        for label, on in (("Perspective", False), ("Orthographic", True)):
            a = m.addAction(label); a.setCheckable(True); a.setChecked(v.ortho == on)
            a.triggered.connect(lambda _=False, on=on: (setattr(v, "ortho", on), v.update()))
        m.addSeparator(); m.addAction("Set Current View as Home", s.set_home); m.addAction("Reset Home", s.reset_home)
        m.addAction("Save Named View...", s.save_view_dialog); m.exec(gpos)

    def set_home(s): s.vp.home_cam = s.vp.cam_state(); s.statusBar().showMessage("Home view set.", 4000)
    def reset_home(s): s.vp.home_cam = None; s.statusBar().showMessage("Home view reset to the default.", 4000)
    def save_view_dialog(s):
        n = len(s.vp.views) + 1
        name, ok = W.QInputDialog.getText(s, "Named view", "Name:", text=f"View{n}")
        if ok and name.strip(): s.save_view(name.strip())
    def save_view(s, name): s.vp.views[name] = s.vp.cam_state(); s.refresh()
    def rename_view(s, name):
        new, ok = W.QInputDialog.getText(s, "Rename view", "Name:", text=name)
        if ok and new.strip() and new.strip() not in s.vp.views:
            s.vp.views = {(new.strip() if k == name else k): x for k, x in s.vp.views.items()}; s.refresh()
    def delete_view(s, name): s.vp.views.pop(name, None); s.refresh()

    # ---------------- command search ----------------
    def add_command_search(s):
        h = getattr(s, "_topbar_h", None)
        if h is None: return
        s.search_box = W.QLineEdit(); s.search_box.setPlaceholderText("Search commands  (Ctrl+K)"); s.search_box.setFixedWidth(260)
        s.search_box.setClearButtonEnabled(True)
        idx = next((i for i in range(h.count()) if h.itemAt(i).spacerItem() is not None), h.count())
        h.insertWidget(idx, s.search_box)
        s._search_model = G.QStandardItemModel(s)
        comp = W.QCompleter(s._search_model, s); comp.setCaseSensitivity(C.Qt.CaseInsensitive); comp.setFilterMode(C.Qt.MatchContains)
        comp.setCompletionRole(C.Qt.DisplayRole); comp.setMaxVisibleItems(14); s.search_box.setCompleter(comp)
        comp.activated[C.QModelIndex].connect(s.search_run)
        s.search_box.returnPressed.connect(lambda: s.search_first())
    def search_actions(s):
        seen, out = set(), []
        for a in s.findChildren(G.QAction):
            t = a.text().replace("&", "").split("  (")[0].strip()
            if not t or t in seen or a.isSeparator() or t.endswith("..") and len(t) < 4: continue
            seen.add(t); out.append((t, a))
        return sorted(out, key=lambda x: x[0].lower())
    def focus_search(s):
        if not getattr(s, "search_box", None): return
        s._search_map = {}; s._search_model.clear()
        for t, a in s.search_actions():
            it = G.QStandardItem(a.icon(), t); s._search_model.appendRow(it); s._search_map[t] = a
        s.search_box.setFocus(); s.search_box.selectAll()
    def search_run(s, index):
        t = index.data(); a = getattr(s, "_search_map", {}).get(t)
        s.search_box.clear(); s.vp.setFocus()
        if a: C.QTimer.singleShot(0, a.trigger)
    def search_first(s):
        q = s.search_box.text().strip().lower()
        if not q: return
        if not getattr(s, "_search_map", None): s.focus_search()
        hits = [t for t in s._search_map if q in t.lower()]
        if hits:
            a = s._search_map[sorted(hits, key=lambda t: (not t.lower().startswith(q), len(t)))[0]]
            s.search_box.clear(); s.vp.setFocus(); C.QTimer.singleShot(0, a.trigger)

    # ---------------- preferences, recent files, autosave, new window ----------------
    def settings(s): return C.QSettings("Fission", "Fission")
    def load_prefs(s):
        st_ = s.settings(); v = s.vp
        v.nav_preset = st_.value("nav", "fission"); v.zoom_invert = st_.value("zoom_invert", "false") in (True, "true")
        v.ortho = st_.value("ortho", "false") in (True, "true"); v.vstyle = st_.value("vstyle", "edges")
        u = st_.value("units", "mm")
        if u in ("mm", "in"): s.set_units(u)
        return st_
    def prefs_dialog(s):
        st_ = s.settings(); v = s.vp
        fields = [("units", "Default units", "combo", st_.value("units", "mm"), [("Millimetres", "mm"), ("Inches", "in")]),
                  ("nav", "Mouse navigation", "combo", v.nav_preset, [("Fission: right-drag orbit, middle-drag pan", "fission"),
                   ("Fusion: shift+middle orbit, middle pan", "fusion"), ("SolidWorks: middle orbit, ctrl+middle pan", "solidworks")]),
                  ("zoom_invert", "Reverse mouse-wheel zoom", "check", v.zoom_invert),
                  ("ortho", "Orthographic camera by default", "check", st_.value("ortho", "false") in (True, "true")),
                  ("vstyle", "Visual style", "combo", st_.value("vstyle", "edges"), VSTYLES),
                  ("autosave", "Autosave every (minutes, 0 = off)", "int", int(st_.value("autosave", 5)), (0, 120)),
                  ("recent_n", "Recent files to remember", "int", int(st_.value("recent_n", 10)), (0, 30))]
        r = form_dialog(s, "Preferences", fields)
        if not r: return
        for k, x in r.items(): st_.setValue(k, x)
        v.nav_preset, v.zoom_invert, v.vstyle = r["nav"], r["zoom_invert"], r["vstyle"]
        s.set_units(r["units"]); s.autosave_timer.start(max(1, r["autosave"])*60000) if r["autosave"] else s.autosave_timer.stop()
        v.update()

    def add_recent(s, path):
        if not path or os.path.dirname(os.path.abspath(path)) == os.path.abspath(s.autosave_dir()): return
        st_ = s.settings(); lst = [p for p in (st_.value("recent") or []) if isinstance(p, str) and p != path]
        if isinstance(st_.value("recent"), str): lst = []
        lst.insert(0, path); st_.setValue("recent", lst[:int(st_.value("recent_n", 10))]); s.rebuild_recent()
    def rebuild_recent(s):
        m = getattr(s, "recent_menu", None)
        if m is None: return
        m.clear(); lst = s.settings().value("recent") or []
        if isinstance(lst, str): lst = [lst]
        for p in lst:
            a = m.addAction(os.path.basename(p)); a.setToolTip(p); a.triggered.connect(lambda _=False, p=p: s.open_design(p))
        if not lst: a = m.addAction("(none yet)"); a.setEnabled(False)
        else: m.addSeparator(); m.addAction("Clear list", lambda: (s.settings().setValue("recent", []), s.rebuild_recent()))

    def new_window(s):
        args = [sys.argv[0]] if sys.argv and sys.argv[0].lower().endswith(".py") else []
        if not C.QProcess.startDetached(sys.executable, args): W.QMessageBox.warning(s, "New window", "Couldn't start another Fission window.")

    def autosave_dir(s):
        d = os.path.join(C.QStandardPaths.writableLocation(C.QStandardPaths.AppDataLocation) or tempfile.gettempdir(), "autosave")
        os.makedirs(d, exist_ok=True); return d
    def insp_setup(s):
        s.load_prefs(); s.rebuild_recent()
        import uuid
        s._session = uuid.uuid4().hex[:12]; d = s.autosave_dir()
        s._lock = C.QLockFile(os.path.join(d, f"{s._session}.lock")); s._lock.tryLock(0)
        s.autosave_timer = C.QTimer(s); s.autosave_timer.timeout.connect(s.autosave_tick)
        mins = int(s.settings().value("autosave", 5))
        if mins: s.autosave_timer.start(mins*60000)
    def autosave_path(s): return os.path.join(s.autosave_dir(), f"{s._session}.fission")
    def autosave_tick(s):
        v = s.vp
        if len(v.states) <= 1 or not s.dirty(): return
        try:
            data = {"app": "Fission", "format": 2, "units": DISPLAY["unit"], "pos": v.pos, "params": v.params, "autosave_of": s.path,
                    "steps": [{"kind": stt["kind"], "op": enc(stt["op"])} for stt in v.states[1:]]}
            data.update(v.insp_save())
            tmp = s.autosave_path() + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f: json.dump(data, f)
            os.replace(tmp, s.autosave_path())
        except Exception: traceback.print_exc()
    def autosave_cleanup(s):
        try:
            if os.path.exists(s.autosave_path()): os.remove(s.autosave_path())
            s._lock.unlock()
        except Exception: pass
    def check_recovery(s):
        import glob; d = s.autosave_dir(); found = []
        for f in glob.glob(os.path.join(d, "*.fission")):
            sid = os.path.basename(f)[:-8]
            if sid == s._session: continue
            lk = C.QLockFile(os.path.join(d, f"{sid}.lock"))
            if lk.tryLock(0): found.append((f, lk))                       # its window is gone: it crashed or was killed
        for f, lk in found:
            try: info = json.load(open(f, encoding="utf-8"))
            except Exception: info = {}
            when = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
            name = os.path.basename(info.get("autosave_of") or "") or "an unsaved design"
            r = W.QMessageBox.question(s, "Recover design", f"Fission found unsaved work from {name} (autosaved {when}).\n\nOpen it now?",
                                       W.QMessageBox.Open | W.QMessageBox.Discard | W.QMessageBox.Ignore, W.QMessageBox.Open)
            if r == W.QMessageBox.Open:
                s.open_design(f); s.path = None; s._saved = None; s.update_title()
            if r != W.QMessageBox.Ignore:
                try: os.remove(f)
                except Exception: pass
                lk.unlock()
                try: os.remove(os.path.join(d, os.path.basename(f)[:-8] + ".lock"))
                except Exception: pass

# ---- MESH workspace commands, file import / export (0.7) ----
MESH_OPS = ("meshimport", "cadimport", "tessellate", "meshconvert", "mrev", "mrepair", "mcut", "mcombine", "msep", "msmooth", "mreduce", "msection")
QUALITY = [("Low", "low"), ("Medium", "med"), ("High", "high"), ("Custom", "custom")]
UNIT_SCALE = [("Millimetres", 1.0), ("Centimetres", 10.0), ("Metres", 1000.0), ("Inches", 25.4), ("Feet", 304.8)]

def make_mesh_commands():
    out = {}
    def add(c): c.group = "MESH"; out[c.name] = c
    custom = lambda v: v["q"] == "custom"
    add(Cmd("tessellate", "TESSELLATE", "tessellate", [
        Fsel("targets", "Bodies", {"body"}, True), Fcombo("q", "Refinement", "med", QUALITY),
        Flen("defl", "Surface deviation", 0.05, 0.0001, 100, show=custom), Fang("ang", "Normal deviation", 15.0, 1, 90, show=custom),
        Fcheck("keep", "Keep the solid (hidden)")],
        hint="Turns solid or surface bodies into triangle meshes (for mesh editing, or to see what an STL export will look like)."))
    add(Cmd("meshconvert", "CONVERT MESH", "meshconvert", [
        Fsel("targets", "Mesh bodies", {"body"}, True),
        Fcombo("method", "Method", "prismatic", [("Prismatic (merge flat areas)", "prismatic"), ("Faceted (one face per triangle)", "faceted")])],
        hint=f"Makes a B-rep body from a mesh so solid tools work on it. A closed mesh becomes a solid. Prismatic merges coplanar "
             f"triangles into single faces - best for machined parts. Up to {MESH_CONVERT_MAX:,} triangles."))
    add(Cmd("mrev", "REVERSE NORMAL", "mrev", [Fsel("targets", "Mesh bodies", {"body"}, True)],
        hint="Flips which side of the mesh counts as outside."))
    add(Cmd("mrepair", "REPAIR", "mrepair", [Fsel("targets", "Mesh bodies", {"body"}, True), Fcheck("holes", "Fill holes", True)],
        hint="Welds duplicate vertices, drops degenerate and duplicate triangles, makes the normals agree and fills small holes."))
    add(Cmd("mcut", "PLANE CUT", "mcut", [
        Fsel("targets", "Mesh bodies", {"body"}, True), Fsel("plane", "Cutting plane", {"plane"}),
        Fcombo("keep", "Keep", "below", [("Behind the plane", "below"), ("In front of the plane", "above"), ("Both (split)", "both")]),
        Fcheck("cap", "Fill the cut (cap)", True), Fcheck("flip", "Flip the plane")],
        hint="Trims a mesh with a plane. Fill keeps it closed so it can still be printed or converted."))
    add(Cmd("mcombine", "COMBINE (MESH)", "mcombine", [
        Fsel("target", "Target mesh", {"body"}), Fsel("tools", "Tool meshes", {"body"}, True),
        Fcombo("op", "Operation", "merge", [("Merge (just put together)", "merge"), ("Join", "join"), ("Cut", "cut"), ("Intersect", "intersect")]),
        Fcheck("keep", "Keep the tools")],
        hint=f"Merge simply collects the triangles into one body. Join / Cut / Intersect are true booleans (closed meshes under "
             f"{MESH_CONVERT_MAX:,} triangles)."))
    add(Cmd("msep", "SEPARATE", "msep", [Fsel("targets", "Mesh bodies", {"body"}, True)],
        hint="Splits a mesh into one body per connected piece."))
    add(Cmd("msmooth", "SMOOTH", "msmooth", [
        Fsel("targets", "Mesh bodies", {"body"}, True), Fint("iters", "Iterations", 10, 1, 200), Fnum("strength", "Strength", 0.5, 0.05, 1.0, 2)],
        hint="Evens out scan noise without shrinking the part (Taubin smoothing). Open edges stay where they are."))
    add(Cmd("mreduce", "REDUCE", "mreduce", [
        Fsel("targets", "Mesh bodies", {"body"}, True), Fnum("pct", "Keep about", 50.0, 1.0, 99.0, 1, suffix=" %")],
        hint="Fewer triangles: lighter files and faster conversion, at the cost of detail."))
    add(Cmd("msection", "MESH SECTION SKETCH", "msection", [
        Fsel("target", "Body", {"body"}), Fsel("plane", "Plane", {"plane"}), Flen("tol", "Simplify tolerance", 0.05, 0, 100)],
        hint="Slices a mesh (or any body) with a plane and makes a sketch of the cross-section - trace or extrude it to rebuild a scan."))
    return out


class MeshWin:
    """Window side of the mesh workspace and the STEP / IGES / BREP / STL exchange."""
    def make_mesh_actions(s, act, cact):
        M = {n: cact(n, None, COMMANDS[n].title.replace(" (MESH)", "").title()) for n in COMMANDS if COMMANDS[n].group == "MESH"}
        M["insert"] = act("Insert Mesh...", None, s.insert_mesh, "meshimport", "Bring in an STL, OBJ or 3MF mesh (scans, downloaded parts)")
        M["cadimport"] = act("Import CAD File...", None, s.import_cad, "cadimport", "STEP, IGES or BREP solids and surfaces from other CAD programs")
        M["stl"] = act("Export STL...", None, s.export_stl, "export", "Triangle mesh for 3D printing, with refinement options")
        M["step"] = act("Export STEP...", None, lambda: s.export_cad("STEP"), "export", "Exact geometry for other CAD programs")
        M["iges"] = act("Export IGES...", None, lambda: s.export_cad("IGES"), "export")
        M["brep"] = act("Export BREP...", None, lambda: s.export_cad("BREP"), "export")
        M["3mf"] = act("Export 3MF...", None, s.export_3mf, "export"); M["obj"] = act("Export OBJ...", None, s.export_obj, "export")
        return M

    def mesh_ribbon(s, M, T, C3, construct_menu, IA, mv2, aln, scl, mat, app_):
        prep = [M["mrepair"], M["mrev"], M["mreduce"], M["msmooth"]]
        modify = [M["mcut"], M["mcombine"], M["msep"], M["meshconvert"], None, mv2, aln, scl, None, mat, app_]
        return s.build_ribbon((("CREATE", (M["insert"], M["tessellate"], M["msection"]), [M["insert"], M["cadimport"], None, M["tessellate"], M["msection"]]),
                               ("PREPARE", tuple(prep[:2]), prep),
                               ("MODIFY", (M["mcut"], M["mcombine"], M["msep"], M["msmooth"], M["mreduce"], M["meshconvert"]), modify),
                               ("CONSTRUCT", (C3["plane_offset"], C3["plane_mid"], C3["axis_2pts"]), construct_menu),
                               *s.inspect_groups(IA),
                               ("INSERT", (M["insert"], M["cadimport"]), [M["insert"], M["cadimport"]]),
                               ("EXPORT", (M["stl"], M["step"]), [M["stl"], M["step"], M["iges"], M["brep"], None, M["3mf"], M["obj"]])),
                              "MESH")

    # ---------------- import ----------------
    def _last_dir(s):
        try: return s.settings().value("lastdir", "") or ""
        except Exception: return ""
    def _set_last_dir(s, path):
        try: s.settings().setValue("lastdir", os.path.dirname(path))
        except Exception: pass

    def insert_mesh(s, path=None):
        if not path:
            path, _ = W.QFileDialog.getOpenFileName(s, "Insert Mesh", s._last_dir(), "Meshes (*.stl *.obj *.3mf);;All files (*)")
        if not path: return
        s._set_last_dir(path)
        try:
            W.QApplication.setOverrideCursor(C.Qt.WaitCursor); T = read_mesh_file(path)
        except Exception as ex: W.QApplication.restoreOverrideCursor(); return W.QMessageBox.warning(s, "Insert Mesh", f"Couldn't read that file:\n{ex}")
        W.QApplication.restoreOverrideCursor()
        if not len(T): return W.QMessageBox.warning(s, "Insert Mesh", "That file has no triangles in it.")
        P = T.reshape(-1, 3); size = np.ptp(P, 0)
        d = W.QDialog(s); d.setWindowTitle("Insert Mesh"); f = W.QFormLayout(d)
        f.addRow(W.QLabel(f"<b>{os.path.basename(path)}</b><br>{len(T):,} triangles"))
        units = W.QComboBox()
        for lab, k in UNIT_SCALE: units.addItem(lab, k)
        if path.lower().endswith(".3mf"): units.setEnabled(False); units.setToolTip("3MF files say their own units")
        sz = W.QLabel(); upd = lambda: sz.setText(" × ".join(fmt(x*units.currentData(), 2) for x in size) + " mm")
        units.currentIndexChanged.connect(upd); upd()
        yup = W.QCheckBox("File is Y-up (most game / scan tools)"); ctr = W.QCheckBox("Centre on the origin, sitting on the ground"); ctr.setChecked(True)
        f.addRow("File units", units); f.addRow("Size", sz); f.addRow(yup); f.addRow(ctr)
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel); bb.accepted.connect(d.accept); bb.rejected.connect(d.reject); f.addRow(bb)
        if not d.exec(): return
        op = {"t": "meshimport", "icon": "meshimport", "name": os.path.splitext(os.path.basename(path))[0], "data": pack_arr(T),
              "scale": float(units.currentData()), "yup": yup.isChecked(), "center": ctr.isChecked()}
        s.run_file_op(op, "Insert Mesh")

    def import_cad(s, path=None):
        if not path:
            path, _ = W.QFileDialog.getOpenFileName(s, "Import CAD File", s._last_dir(),
                                                    "CAD files (*.step *.stp *.iges *.igs *.brep);;STEP (*.step *.stp);;IGES (*.iges *.igs);;BREP (*.brep)")
        if not path: return
        s._set_last_dir(path)
        try:
            W.QApplication.setOverrideCursor(C.Qt.WaitCursor); sh = read_cad_file(path); txt = shape_to_brep_text(sh)
        except Exception as ex: W.QApplication.restoreOverrideCursor(); return W.QMessageBox.warning(s, "Import", f"Couldn't read that file:\n{ex}")
        W.QApplication.restoreOverrideCursor()
        op = {"t": "cadimport", "icon": "cadimport", "name": os.path.splitext(os.path.basename(path))[0], "brep": pack_text(txt)}
        s.run_file_op(op, "Import")

    def run_file_op(s, op, title):
        v = s.vp
        if v.skedit: v.sk_finish()
        try:
            W.QApplication.setOverrideCursor(C.Qt.WaitCursor); n0 = len(v.bodies); v.do(op)
            s.statusBar().showMessage(f"{title}: added {len(v.bodies) - n0} bod{'y' if len(v.bodies) - n0 == 1 else 'ies'}.", 6000)
            v.fit(); v.update()
        except Exception as ex: traceback.print_exc(); W.QMessageBox.warning(s, title, str(ex))
        finally: W.QApplication.restoreOverrideCursor(); s.refresh()

    # ---------------- export ----------------
    def export_bodies(s):
        v = s.vp; sel = sorted(v.msel | ({v.sel_body} if v.sel_body is not None else set()))
        ids = [i for i in sel if i < len(v.bodies)] or [i for i, b in enumerate(v.bodies) if b.visible]
        return ids

    def export_cad(s, kind):
        ext = {"STEP": "step", "IGES": "igs", "BREP": "brep"}[kind]; v = s.vp
        ids = s.export_bodies()
        if not ids: return W.QMessageBox.information(s, f"Export {kind}", "Nothing to export yet.")
        big = [i for i in ids if getattr(v.bodies[i], "mesh", False) and len(v.bodies[i].tv) > MESH_CONVERT_MAX]
        ids = [i for i in ids if i not in big]
        if not ids: return W.QMessageBox.information(s, f"Export {kind}", "Only large meshes are selected - export them as STL / 3MF, or Reduce and Convert them first.")
        path = s.save_path(f"Export {kind}", f"{s.export_name()}.{ext}", f"{kind} (*.{ext}{' *.stp' if kind == 'STEP' else ' *.iges' if kind == 'IGES' else ''})")
        if not path: return
        try:
            W.QApplication.setOverrideCursor(C.Qt.WaitCursor); write_cad_file(path, [v.bodies[i].shape for i in ids])
            s.statusBar().showMessage(f"Exported {len(ids)} bod{'y' if len(ids) == 1 else 'ies'} to {os.path.basename(path)}" +
                                      (f" ({len(big)} large mesh(es) skipped)" if big else ""), 8000)
        except Exception as ex: W.QMessageBox.warning(s, f"Export {kind}", str(ex))
        finally: W.QApplication.restoreOverrideCursor()

    def export_name(s):
        return os.path.splitext(os.path.basename(s.path))[0] if getattr(s, "path", None) else "fission"

    def export_stl(s):
        v = s.vp; ids = s.export_bodies()
        if not ids: return W.QMessageBox.information(s, "Export STL", "Nothing to export yet.")
        d = W.QDialog(s); d.setWindowTitle("Export STL"); f = W.QFormLayout(d)
        f.addRow(W.QLabel(f"{len(ids)} bod{'y' if len(ids) == 1 else 'ies'} ({'the selection' if v.msel or v.sel_body is not None else 'everything shown'})"))
        q = W.QComboBox()
        for lab, k in QUALITY: q.addItem(lab, k)
        q.setCurrentIndex(1); dev = LengthSpin(0.0001, 100); dev.setValue(0.05); ang = W.QDoubleSpinBox(); ang.setRange(1, 90); ang.setValue(15); ang.setSuffix("°")
        fmt_ = W.QComboBox(); fmt_.addItem("Binary", False); fmt_.addItem("ASCII text", True)
        each = W.QCheckBox("One file per body")
        info = W.QLabel(); info.setStyleSheet("color:#666")
        def upd():
            c = q.currentData() == "custom"; dev.setEnabled(c); ang.setEnabled(c)
            n = 0
            for i in ids:
                b = v.bodies[i]
                if getattr(b, "mesh", False): n += len(b.tv); continue
                dl, an = mesh_quality(b.shape, q.currentData(), dev.value(), ang.value())
                n += len(tessellate(b.shape, dl, math.radians(an)))
            size = 250*n if fmt_.currentData() else 84 + 50*n
            info.setText(f"{n:,} triangles, about {size/1e6:.2f} MB")
        for w in (q, fmt_): w.currentIndexChanged.connect(upd)
        dev.valueChanged.connect(upd); ang.valueChanged.connect(upd)
        f.addRow("Refinement", q); f.addRow("Surface deviation", dev); f.addRow("Normal deviation", ang); f.addRow("Format", fmt_); f.addRow(each); f.addRow(info)
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel); bb.accepted.connect(d.accept); bb.rejected.connect(d.reject); f.addRow(bb)
        upd()
        if not d.exec(): return
        path = s.save_path("Export STL", f"{s.export_name()}.stl", "STL mesh (*.stl)")
        if not path: return
        try:
            W.QApplication.setOverrideCursor(C.Qt.WaitCursor)
            parts = []
            for i in ids:
                b = v.bodies[i]
                if getattr(b, "mesh", False): T = b.tv
                else:
                    dl, an = mesh_quality(b.shape, q.currentData(), dev.value(), ang.value()); T = tessellate(b.shape, dl, math.radians(an))
                parts.append((getattr(b, "name", None) or f"Body{i + 1}", T))
            if each.isChecked() and len(parts) > 1:
                root, ext = os.path.splitext(path)
                for name, T in parts: write_stl(f"{root}_{re.sub(r'[^A-Za-z0-9_-]+', '_', name)}{ext}", T, fmt_.currentData(), name)
            else: write_stl(path, np.concatenate([T for _, T in parts]), fmt_.currentData(), s.export_name())
            s.statusBar().showMessage(f"Exported STL: {sum(len(T) for _, T in parts):,} triangles.", 8000)
        except Exception as ex: W.QMessageBox.warning(s, "Export STL", str(ex))
        finally: W.QApplication.restoreOverrideCursor()


# ---- ASSEMBLE commands and window side (0.7) ----
ASM_OPS = ("comp", "compprops", "joint", "jdrive", "mlink")
JOINT_PICK = {"face", "edge", "point"}

def make_asm_commands():
    out = {}
    def add(c): c.group = "ASSEMBLE"; out[c.name] = c
    add(Cmd("comp", "NEW COMPONENT", "component", [
        Fsel("cbodies", "Bodies", {"body"}, True, req=False), Ftext("name", "Name", "", ph="Component1"),
        Fcheck("ground", "Ground it (it won't move)")],
        hint="Groups bodies into a component - the part that joints move. With no bodies picked, it makes an empty component "
             "and activates it, so new bodies go into it.",
        build=lambda v, op, vp: dict(op, name=(v["name"].strip() or vp.new_comp_name()))))
    lim = []
    for k in range(3):
        lim += [Fcheck(f"lim{k}", f"Limit value {k + 1}", show=lambda v, k=k: len(JOINT_DOF[v["kind"]]) > k),
                Fnum(f"lo{k}", "  Minimum", -90.0, -1e6, 1e6, 3, show=lambda v, k=k: len(JOINT_DOF[v["kind"]]) > k and v[f"lim{k}"]),
                Fnum(f"hi{k}", "  Maximum", 90.0, -1e6, 1e6, 3, show=lambda v, k=k: len(JOINT_DOF[v["kind"]]) > k and v[f"lim{k}"])]
    add(Cmd("joint", "JOINT", "joint", [
        Fsel("a", "Component 1 (moves)", JOINT_PICK), Fsel("b", "Component 2", JOINT_PICK | {"plane"}),
        Fcombo("kind", "Motion", "rigid", JOINT_KINDS), Fang("angle", "Angle", 0.0), Flen("offset", "Offset", 0.0),
        Fcheck("flip", "Flip"), Fcheck("asbuilt", "As-built (keep where they are)"), Ftext("name", "Name", "", ph="Joint1")] + lim,
        hint="Pick a face, circular edge or point on the part that should move, then where it goes on the other part. Faces snap "
             "to their centre, circles and cylinders to their axis. The motion type sets what can still move: Revolute spins, "
             "Slider slides, Cylindrical does both, Pin-slot spins and slides sideways, Planar slides in a plane, Ball swivels.",
        build=lambda v, op, vp: dict(op, name=v["name"].strip() or f"Joint{len(vp.joints) + 1}")))
    return out


class AsmWin:
    """Window side: actions, browser folders, drive / motion link / motion study / BOM dialogs."""
    def make_asm_actions(s, act, cact):
        A = {"comp": cact("comp", None, "New Component"), "joint": cact("joint", "J", "Joint")}
        A["asbuilt"] = act("As-built Joint", None, lambda: s.open_cmd("joint", vals={"asbuilt": True}), "joint",
                           "Joint two parts where they already are")
        A["drive"] = act("Drive Joints...", None, s.drive_dialog, "jdrive", "Set a joint's angle or distance")
        A["link"] = act("Motion Link...", None, s.link_dialog, "mlink", "Make one joint follow another (gears, rack and pinion)")
        A["motion"] = act("Motion Study...", None, s.motion_dialog, "motion", "Animate a joint through its range")
        A["bom"] = act("Bill of Materials...", None, s.bom_dialog, "bom", "Parts list with quantities, materials and mass; save as CSV")
        return A

    def asm_group(s, A):
        return ("ASSEMBLE", (A["comp"], A["joint"], A["drive"], A["motion"]),
                [A["comp"], None, A["joint"], A["asbuilt"], None, A["drive"], A["link"], A["motion"], None, A["bom"]])

    # ---- browser ----
    def asm_browser(s, root, item, folders):
        v = s.vp
        if v.comps:
            cf = item(root, "Components", "folder", "grp:cm", True); folders.append(cf)
            for name, c in v.comps.items():
                ids = v.comp_bodies(name)
                it = item(cf, name + ("  (active)" if v.active_comp == name else "") + ("  📌" if c.get("ground") else ""), "component", f"cm:{name}",
                          any(v.bodies[i].visible for i in ids) if ids else True)
                if v.active_comp == name: f = it.font(0); f.setBold(True); it.setFont(0, f)
                it.setToolTip(0, f"{len(ids)} bod{'y' if len(ids) == 1 else 'ies'}" + (" · grounded" if c.get("ground") else "") +
                              (f"\nPart number {c['pn']}" if c.get("pn") else "") + "\nRight-click: activate, ground, properties")
                for i in ids: item(it, getattr(v.bodies[i], "name", None) or f"Body{i + 1}", "body", f"bd:{i}", v.bodies[i].visible)
        if v.joints:
            jf = item(root, "Joints", "folder", "grp:jt", True); folders.append(jf)
            for j, jt in enumerate(v.joints):
                vals = ", ".join(f"{fmt(x, 2)}{'°' if JOINT_DOF[jt['kind']][k][1] == 'ang' else ' mm'}" for k, x in enumerate(jt["vals"]))
                it = item(jf, f"{jt['name']}  ({dict((b, a) for a, b in JOINT_KINDS)[jt['kind']]})", "joint", f"jt:{j}", True)
                it.setToolTip(0, f"{jt['a']} → {jt['b'] or 'the origin'}" + (f"\n{vals}" if vals else "") + "\nRight-click to drive or animate it")

    def asm_vis(s, ref, on):
        v = s.vp
        if ref.startswith("cm:"):
            for i in v.comp_bodies(ref[3:]): nb = v.bodies[i].restyled(); nb.visible = on; v.bodies[i] = nb
        elif ref in ("grp:cm", "grp:jt") or ref.startswith("jt:"): pass
        else: return False
        v.update(); C.QTimer.singleShot(0, s.refresh); return True

    def asm_click(s, ref):
        v = s.vp
        if ref.startswith("cm:"):
            ids = v.comp_bodies(ref[3:])
            if ids: v.sel_body, v.msel, v.sel_face = ids[0], set(ids), None; v.update()
            return True
        return False

    def asm_browser_menu(s, m, ref):
        v = s.vp
        if ref.startswith("cm:"):
            name = ref[3:]; c = v.comps.get(name, {})
            if v.active_comp == name: m.addAction("Deactivate (new bodies go to the top level)", lambda: s.activate_comp(None))
            else: m.addAction("Activate (new bodies go into it)", lambda: s.activate_comp(name))
            m.addAction("Unground" if c.get("ground") else "Ground", lambda: s.comp_props(name, ground=not c.get("ground")))
            m.addAction("Properties...", lambda: s.comp_dialog(name))
            return True
        if ref.startswith("jt:"):
            j = int(ref[3:]); jt = v.joints[j]
            a = m.addAction("Drive...", lambda: s.drive_dialog(j)); a.setEnabled(bool(JOINT_DOF[jt["kind"]]))
            a = m.addAction("Animate (motion study)...", lambda: s.motion_dialog(j)); a.setEnabled(bool(JOINT_DOF[jt["kind"]]))
            return True
        return False

    def activate_comp(s, name):
        s.vp.active_comp = name; s.refresh()
        s.statusBar().showMessage(f"{name} is active - new bodies go into it." if name else "No active component.", 6000)

    def comp_props(s, comp, **kw):
        return s.run_asm({"t": "compprops", "icon": "component", "comp": comp, **kw}, "Component")

    def run_asm(s, op, title):
        v = s.vp
        if s.busy(): return False
        try:
            W.QApplication.setOverrideCursor(C.Qt.WaitCursor); v.do(op); return True
        except Exception as ex: traceback.print_exc(); W.QMessageBox.warning(s, title, str(ex)); return False
        finally: W.QApplication.restoreOverrideCursor(); s.refresh(); v.update()

    def comp_dialog(s, name):
        v = s.vp; c = v.comps[name]
        d = W.QDialog(s); d.setWindowTitle(f"{name} properties"); f = W.QFormLayout(d)
        nm = W.QLineEdit(name); pn = W.QLineEdit(c.get("pn", "")); de = W.QLineEdit(c.get("desc", "")); gr = W.QCheckBox("Grounded"); gr.setChecked(bool(c.get("ground")))
        f.addRow("Name", nm); f.addRow("Part number", pn); f.addRow("Description", de); f.addRow(gr)
        ids = v.comp_bodies(name); mass = sum((v.body_props(v.bodies[i])["mass"] or 0) for i in ids); vol_ = sum(v.body_props(v.bodies[i])["volume"] for i in ids)
        f.addRow("Bodies", W.QLabel(str(len(ids)))); f.addRow("Volume", W.QLabel(f"{fmt(vol_/1000, 3)} cm³")); f.addRow("Mass", W.QLabel(f"{fmt(mass, 2)} g" if mass else "- (set a material)"))
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel); bb.accepted.connect(d.accept); bb.rejected.connect(d.reject); f.addRow(bb)
        if not d.exec(): return
        s.comp_props(name, name=nm.text().strip() or name, pn=pn.text().strip(), desc=de.text().strip(), ground=gr.isChecked())

    # ---- drive / links / motion ----
    def _joint_combo(s, sel=None, movable=True):
        cb = W.QComboBox()
        for j, jt in enumerate(s.vp.joints):
            if movable and not JOINT_DOF[jt["kind"]]: continue
            cb.addItem(f"{jt['name']} ({jt['a']})", j)
        if sel is not None and cb.findData(sel) >= 0: cb.setCurrentIndex(cb.findData(sel))
        return cb

    def drive_dialog(s, j=None, vals=None):
        v = s.vp
        if not any(JOINT_DOF[jt["kind"]] for jt in v.joints): return W.QMessageBox.information(s, "Drive Joints", "There are no movable joints yet.")
        if vals is not None: return s.run_asm({"t": "jdrive", "icon": "jdrive", "j": j, "vals": list(vals)}, "Drive Joints")
        d = W.QDialog(s); d.setWindowTitle("Drive Joints"); f = W.QFormLayout(d); cb = s._joint_combo(j); f.addRow("Joint", cb)
        boxes = []
        for k in range(3):
            b = W.QDoubleSpinBox(); b.setRange(-1e6, 1e6); b.setDecimals(3); f.addRow(f"Value {k + 1}", b); boxes.append(b)
        def upd():
            jt = v.joints[cb.currentData()]; dof = JOINT_DOF[jt["kind"]]
            for k, b in enumerate(boxes):
                vis = k < len(dof); f.setRowVisible(b, vis)
                if vis:
                    f.labelForField(b).setText(dof[k][0]); b.setSuffix("°" if dof[k][1] == "ang" else " mm"); b.setValue(jt["vals"][k])
                    lim = (jt.get("lim") or {}).get(k) or (jt.get("lim") or {}).get(str(k))
                    b.setToolTip(f"Limited to {lim[0]} … {lim[1]}" if lim else "")
        cb.currentIndexChanged.connect(upd); upd()
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel); bb.accepted.connect(d.accept); bb.rejected.connect(d.reject); f.addRow(bb)
        if not d.exec(): return
        jj = cb.currentData(); n = len(JOINT_DOF[v.joints[jj]["kind"]])
        s.run_asm({"t": "jdrive", "icon": "jdrive", "j": jj, "vals": [boxes[k].value() for k in range(n)]}, "Drive Joints")

    def link_dialog(s, j1=None, j2=None, ratio=None):
        v = s.vp
        if sum(1 for jt in v.joints if JOINT_DOF[jt["kind"]]) < 2: return W.QMessageBox.information(s, "Motion Link", "Link needs two movable joints.")
        if ratio is not None: return s.run_asm({"t": "mlink", "icon": "mlink", "j1": j1, "j2": j2, "k1": 0, "k2": 0, "ratio": ratio}, "Motion Link")
        d = W.QDialog(s); d.setWindowTitle("Motion Link"); f = W.QFormLayout(d)
        a, b = s._joint_combo(j1), s._joint_combo(j2); b.setCurrentIndex(min(1, b.count() - 1))
        k1, k2 = W.QSpinBox(), W.QSpinBox(); k1.setRange(1, 3); k2.setRange(1, 3)
        r = W.QDoubleSpinBox(); r.setRange(-1e4, 1e4); r.setDecimals(4); r.setValue(1.0)
        f.addRow("Joint", a); f.addRow("  value", k1); f.addRow("drives joint", b); f.addRow("  value", k2); f.addRow("Ratio", r)
        f.addRow(W.QLabel("e.g. 0.5 for a 2:1 gear, or mm per degree for a rack (π·module/180·teeth)."))
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel); bb.accepted.connect(d.accept); bb.rejected.connect(d.reject); f.addRow(bb)
        if not d.exec() or a.currentData() == b.currentData(): return
        s.run_asm({"t": "mlink", "icon": "mlink", "j1": a.currentData(), "j2": b.currentData(), "k1": k1.value() - 1, "k2": k2.value() - 1,
                   "ratio": r.value()}, "Motion Link")

    def motion_dialog(s, j=None, rng=None, frames=48, save_dir=None):
        v = s.vp
        if not any(JOINT_DOF[jt["kind"]] for jt in v.joints): return W.QMessageBox.information(s, "Motion Study", "There are no movable joints yet.")
        if rng is None:
            d = W.QDialog(s); d.setWindowTitle("Motion Study"); f = W.QFormLayout(d); cb = s._joint_combo(j); f.addRow("Joint", cb)
            k = W.QSpinBox(); k.setRange(1, 3); lo = W.QDoubleSpinBox(); hi = W.QDoubleSpinBox()
            for x in (lo, hi): x.setRange(-1e6, 1e6); x.setDecimals(2)
            lo.setValue(0); hi.setValue(360); n = W.QSpinBox(); n.setRange(2, 2000); n.setValue(frames)
            sv = W.QCheckBox("Also save the frames as PNG images")
            f.addRow("Value", k); f.addRow("From", lo); f.addRow("To", hi); f.addRow("Frames", n); f.addRow(sv)
            def upd():
                jt = v.joints[cb.currentData()]; lim = (jt.get("lim") or {}).get(0) or (jt.get("lim") or {}).get("0")
                if lim: lo.setValue(lim[0]); hi.setValue(lim[1])
                elif JOINT_DOF[jt["kind"]][0][1] == "len": lo.setValue(0); hi.setValue(50)
            cb.currentIndexChanged.connect(upd); upd()
            bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel); bb.accepted.connect(d.accept); bb.rejected.connect(d.reject); f.addRow(bb)
            if not d.exec(): return
            j, kk, rng, frames = cb.currentData(), k.value() - 1, (lo.value(), hi.value()), n.value()
            if sv.isChecked(): save_dir = W.QFileDialog.getExistingDirectory(s, "Folder for the frames") or None
        else: kk = 0
        jt = v.joints[j]; kk = min(kk, len(JOINT_DOF[jt["kind"]]) - 1)
        W.QApplication.setOverrideCursor(C.Qt.WaitCursor)
        try: seq = v.motion_frames(j, kk, rng[0], rng[1], frames)
        except Exception as ex: W.QApplication.restoreOverrideCursor(); return W.QMessageBox.warning(s, "Motion Study", str(ex))
        W.QApplication.restoreOverrideCursor()
        s._motion = {"seq": seq, "i": 0, "keep": list(v.bodies), "dir": save_dir}
        if not hasattr(s, "_mtimer"): s._mtimer = C.QTimer(s); s._mtimer.timeout.connect(s.motion_tick)
        s._mtimer.start(40); s.statusBar().showMessage("Playing the motion study - Esc or any command stops it.", 4000)
        return seq

    def motion_tick(s):
        m = getattr(s, "_motion", None); v = s.vp
        if not m: s._mtimer.stop(); return
        if m["i"] >= len(m["seq"]) or v.cmd or v.skedit:
            s._mtimer.stop(); v.bodies = m["keep"]; s._motion = None; v.update(); return
        v.bodies = m["seq"][m["i"]]; v.update()
        if m["dir"]:
            v.repaint(); v.grabFramebuffer().save(os.path.join(m["dir"], f"frame_{m['i']:04d}.png"))
        m["i"] += 1

    # ---- bill of materials ----
    def bom_dialog(s, path=None):
        v = s.vp; rows = v.bom_rows()
        if not rows: return W.QMessageBox.information(s, "Bill of Materials", "Nothing to list yet.")
        heads = ["Item", "Qty", "Name", "Part number", "Description", "Material", "Mass (g)", "Volume (cm³)", "Size (mm)"]
        def cells(i, r):
            return [str(i + 1), str(r["qty"]), r["name"], r["pn"], r["desc"], r["material"], "-" if r["mass"] is None else f"{r['mass']:.2f}",
                    f"{r['volume']/1000:.3f}", f"{r['size'].x:.1f} × {r['size'].y:.1f} × {r['size'].z:.1f}"]
        if path:
            import csv
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w_ = csv.writer(fh); w_.writerow(heads)
                for i, r in enumerate(rows): w_.writerow(cells(i, r))
            return rows
        d = W.QDialog(s); d.setWindowTitle("Bill of Materials"); d.resize(820, 360); lay = W.QVBoxLayout(d)
        t = W.QTableWidget(len(rows), len(heads)); t.setHorizontalHeaderLabels(heads); t.verticalHeader().hide()
        for i, r in enumerate(rows):
            for k, c in enumerate(cells(i, r)): t.setItem(i, k, W.QTableWidgetItem(c))
        t.resizeColumnsToContents(); lay.addWidget(t)
        bb = W.QDialogButtonBox(); sv = bb.addButton("Save CSV...", W.QDialogButtonBox.ActionRole); bb.addButton(W.QDialogButtonBox.Close)
        bb.rejected.connect(d.reject); lay.addWidget(bb)
        def save():
            p, _ = W.QFileDialog.getSaveFileName(d, "Save BOM", f"{s.export_name()}_bom.csv", "CSV (*.csv)")
            if p: s.bom_dialog(p if p.lower().endswith(".csv") else p + ".csv"); s.statusBar().showMessage(f"Saved {os.path.basename(p)}", 5000)
        sv.clicked.connect(save); d.exec(); return rows

# ---- SHEET METAL commands + window (0.7) ----
def make_sheet_commands():
    out = {}
    def add(c): c.group = "SHEET METAL"; out[c.name] = c
    add(Cmd("smrule", "SHEET METAL RULES", "smrule", [
        Flen("thick", "Thickness", 1.5, 0.01), Flen("radius", "Bend radius", 1.5, 0.0), Fnum("kfactor", "K-factor", 0.44, 0.0, 1.0, 3),
        Flen("gap", "Relief / hem gap", 0.5, 0.0)],
        hint="Applies to the sheet metal steps after this one. The K-factor says where the neutral layer sits in a bend "
             "(0 = inside, 0.5 = middle) and sets how long the flat pattern is."))
    add(Cmd("sbase", "BASE FLANGE", "sbase", [Fsel("profiles", "Profiles", {"profile"}, True), Fcheck("flip", "Flip side")],
        hint="A flat plate of the rules' thickness from closed sketch profiles - the start of a sheet metal part."))
    add(Cmd("sflange", "FLANGE", "sflange", [
        Fsel("edges", "Edges", {"edge"}, True), Flen("h", "Height", 10.0, 0.0), Fang("angle", "Bend angle", 90.0, 0.0, 180.0),
        Fcheck("flip", "Bend the other way"), Fcheck("custom_r", "Own bend radius"), Flen("r", "Bend radius", 1.5, 0.0, show=lambda v: v["custom_r"])],
        hint="Bends a wall up from the edges of a sheet. Edges that meet get a corner relief so the bends don't collide."))
    add(Cmd("shem", "HEM", "shem", [
        Fsel("edges", "Edges", {"edge"}, True), Flen("h", "Length", 5.0, 0.0), Flen("gap", "Gap", 0.5, 0.0), Fcheck("flip", "Fold the other way")],
        hint="Folds the edge right back on itself (180°) to stiffen it and take the sharp edge off."))
    add(Cmd("sfold", "BEND", "sfold", [
        Fsel("line", "Bend line", {"curve"}), Fang("angle", "Angle", 90.0, 0.0, 180.0), Fcheck("flip", "Bend the other way"),
        Fcheck("side", "Move the other side")],
        hint="Folds a flat sheet along a sketch line drawn on it, with the rules' bend radius; the flat length is kept."))
    add(Cmd("unfold", "UNFOLD", "unfold", [Fsel("targets", "Sheet bodies", {"body"}, True)],
        hint="Lays the part flat in place (Refold puts it back). Use Flat Pattern to keep the folded part and get a flat copy."))
    add(Cmd("refold", "REFOLD", "refold", [Fsel("targets", "Unfolded bodies", {"body"}, True)], hint="Folds an unfolded part back up."))
    add(Cmd("flatpattern", "FLAT PATTERN", "flatpattern", [Fsel("targets", "Sheet bodies", {"body"}, True)],
        hint="Adds the flat blank next to the part, with its bend lines - export it as DXF for a laser or plasma cutter."))
    return out


class SheetWin:
    def make_sheet_actions(s, act, cact):
        Sx = {n: cact(n, None, COMMANDS[n].title.title()) for n in COMMANDS if COMMANDS[n].group == "SHEET METAL"}
        Sx["dxf"] = act("Export Flat Pattern DXF...", None, s.export_flat_dxf, "export", "Outline + bend lines of the selected sheet part")
        return Sx

    def sheet_ribbon(s, Sx, T, C3, construct_menu, IA, fil, mov, mat, app_):
        return s.build_ribbon((("CREATE", (T["pick"], Sx["sbase"], Sx["sflange"], Sx["shem"], Sx["sfold"]),
                                [T["pick"], None, Sx["sbase"], Sx["sflange"], Sx["shem"], Sx["sfold"]]),
                               ("MODIFY", (Sx["unfold"], Sx["refold"], Sx["smrule"]), [Sx["unfold"], Sx["refold"], None, Sx["smrule"], None, fil, mov, mat, app_]),
                               ("CONSTRUCT", (C3["plane_offset"], C3["plane_mid"], C3["axis_2pts"]), construct_menu),
                               *s.inspect_groups(IA),
                               ("FLAT PATTERN", (Sx["flatpattern"], Sx["dxf"]), [Sx["flatpattern"], Sx["dxf"]])), "SHEET METAL")

    def export_flat_dxf(s, path=None, bi=None):
        v = s.vp
        if bi is None: bi = v.sel_body if v.sel_body is not None else next((i for i, b in enumerate(v.bodies) if getattr(b, "sheet", None) and b.visible), None)
        if bi is None: return W.QMessageBox.information(s, "Flat pattern", "Select a sheet metal body first.")
        b = v.bodies[bi]; th = (getattr(b, "sheet", None) or v.sm)["t"]; kk = (getattr(b, "sheet", None) or v.sm)["k"]
        try: flat, lines, frame = unfold_sheet(b.folded if getattr(b, "folded", None) is not None else b.shape, th, kk)
        except Exception as ex: return W.QMessageBox.warning(s, "Flat pattern", f"Couldn't flatten that body: {ex}")
        if not path:
            path, _ = W.QFileDialog.getSaveFileName(s, "Export flat pattern", f"{s.export_name()}_flat.dxf", "DXF (*.dxf)")
            if not path: return
            if not path.lower().endswith(".dxf"): path += ".dxf"
        flat_dxf(path, flat, lines, frame); s.statusBar().showMessage(f"Flat pattern saved: {os.path.basename(path)} ({len(lines)} bend lines)", 8000)
        return path

class RenderDialog(W.QDialog):
    """Local render: pick the look and size, render, save PNG, or render a turntable as a PNG sequence."""
    SIZES = [("1280 × 720", (1280, 720)), ("1920 × 1080", (1920, 1080)), ("2560 × 1440", (2560, 1440)), ("3840 × 2160 (4K)", (3840, 2160)),
             ("1080 × 1080", (1080, 1080)), ("Viewport size", None)]
    def __init__(s, win):
        super().__init__(win); s.win = win; s.setWindowTitle("Render"); s.resize(1100, 700); s.img = None
        h = W.QHBoxLayout(s); side = W.QWidget(); side.setFixedWidth(260); f = W.QFormLayout(side); h.addWidget(side)
        s.pic = W.QLabel("Press Render"); s.pic.setAlignment(C.Qt.AlignCenter); s.pic.setStyleSheet("background:#3c4046;color:#ddd"); h.addWidget(s.pic, 1)
        s.env = W.QComboBox(); [s.env.addItem(lab, k) for lab, k in (("Studio (light)", "studio"), ("Warm", "warm"), ("Dark", "dark"))]
        s.size = W.QComboBox(); [s.size.addItem(lab, d) for lab, d in s.SIZES]
        s.reflect, s.ao, s.shadow = W.QCheckBox("Ground reflection"), W.QCheckBox("Ambient occlusion"), W.QCheckBox("Soft shadow")
        for b in (s.reflect, s.ao, s.shadow): b.setChecked(True)
        s.ss = W.QSpinBox(); s.ss.setRange(1, 4); s.ss.setValue(2); s.ss.setToolTip("Supersampling: renders bigger and scales down for smooth edges")
        f.addRow("Environment", s.env); f.addRow("Size", s.size); f.addRow(s.reflect); f.addRow(s.ao); f.addRow(s.shadow); f.addRow("Anti-aliasing", s.ss)
        rb = W.QPushButton("Render"); rb.setObjectName("primary"); rb.clicked.connect(lambda: s.render()); f.addRow(rb)
        sv = W.QPushButton("Save PNG..."); sv.clicked.connect(lambda: s.save()); f.addRow(sv)
        f.addRow(W.QLabel("<b>Turntable</b>")); s.frames = W.QSpinBox(); s.frames.setRange(4, 720); s.frames.setValue(36); f.addRow("Frames", s.frames)
        tt = W.QPushButton("Render turntable..."); tt.clicked.connect(lambda: s.turntable()); f.addRow(tt)
        f.addRow(W.QLabel("Uses the current camera. Materials (Steel, Brass, ...) render as metal."))
        cl = W.QPushButton("Close"); cl.clicked.connect(s.accept); f.addRow(cl)
    def opts(s): return {"env": s.env.currentData(), "reflect": s.reflect.isChecked(), "ao": s.ao.isChecked(), "shadow": s.shadow.isChecked(), "ss": s.ss.value()}
    def dims(s):
        d = s.size.currentData(); vp = s.win.vp
        return d if d else (max(64, vp.width()), max(64, vp.height()))
    def render(s, size=None):
        W.QApplication.setOverrideCursor(C.Qt.WaitCursor)
        try: s.img = s.win.vp.render_image(*(size or s.dims()), s.opts())
        except Exception as ex: traceback.print_exc(); W.QMessageBox.warning(s, "Render", str(ex)); return None
        finally: W.QApplication.restoreOverrideCursor()
        s.pic.setPixmap(G.QPixmap.fromImage(s.img).scaled(s.pic.size(), C.Qt.KeepAspectRatio, C.Qt.SmoothTransformation)); return s.img
    def save(s, path=None):
        if s.img is None and s.render() is None: return
        if not path:
            path, _ = W.QFileDialog.getSaveFileName(s, "Save render", f"{s.win.export_name()}_render.png", "PNG (*.png)")
            if not path: return
        if not path.lower().endswith(".png"): path += ".png"
        s.img.save(path); return path
    def turntable(s, folder=None, frames=None, size=None):
        folder = folder or W.QFileDialog.getExistingDirectory(s, "Folder for the turntable frames")
        if not folder: return
        vp = s.win.vp; yaw0 = vp.yaw; n = frames or s.frames.value(); out = []
        try:
            for i in range(n):
                vp.yaw = turntable_cam(yaw0, i, n); img = vp.render_image(*(size or s.dims()), s.opts())
                p = os.path.join(folder, f"turntable_{i:04d}.png"); img.save(p); out.append(p)
                if i == 0: s.img = img; s.pic.setPixmap(G.QPixmap.fromImage(img).scaled(s.pic.size(), C.Qt.KeepAspectRatio, C.Qt.SmoothTransformation))
                W.QApplication.processEvents()
        finally: vp.yaw = yaw0; vp.update()
        s.win.statusBar().showMessage(f"Saved {n} turntable frames to {folder}", 8000); return out


class StoryPanel(W.QFrame):
    """Animation workspace: an explode storyboard with a time slider, play, and frame export."""
    def __init__(s, win):
        super().__init__(win.vp); s.win = win; s.setObjectName("cmd"); s.setFixedWidth(300)
        lay = W.QVBoxLayout(s); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        hd = W.QLabel("STORYBOARD"); hd.setObjectName("cmdhdr"); lay.addWidget(hd)
        body = W.QWidget(); f = W.QFormLayout(body); f.setContentsMargins(10, 8, 10, 10); lay.addWidget(body)
        s.factor = W.QDoubleSpinBox(); s.factor.setRange(0.1, 10); s.factor.setValue(1.0); f.addRow("Explode distance", s.factor)
        ab = W.QPushButton("Auto Explode"); ab.clicked.connect(s.auto); f.addRow(ab)
        s.list = W.QListWidget(); s.list.setMaximumHeight(140); f.addRow(s.list)
        s.dur = W.QDoubleSpinBox(); s.dur.setRange(0.1, 30); s.dur.setValue(1.0); s.dur.setSuffix(" s"); f.addRow("Step length", s.dur)
        s.dur.valueChanged.connect(s.set_dur)
        s.slider = W.QSlider(C.Qt.Horizontal); s.slider.setRange(0, 1000); s.slider.valueChanged.connect(s.scrub); f.addRow("Time", s.slider)
        row = W.QHBoxLayout(); s.play_b = W.QPushButton("▶ Play"); s.play_b.clicked.connect(s.play); rs = W.QPushButton("Reset"); rs.clicked.connect(lambda: s.slider.setValue(0))
        row.addWidget(s.play_b); row.addWidget(rs); f.addRow(row)
        ex = W.QPushButton("Export frames..."); ex.clicked.connect(lambda: s.export()); f.addRow(ex)
        s.fps = W.QSpinBox(); s.fps.setRange(5, 60); s.fps.setValue(24); f.addRow("Frames / s", s.fps)
        s.use_render = W.QCheckBox("Use the renderer (slower, nicer)"); f.addRow(s.use_render)
        cl = W.QPushButton("Close"); cl.clicked.connect(s.close_panel); f.addRow(cl)
        s.timer = C.QTimer(s); s.timer.timeout.connect(s.tick); s.adjustSize(); s.move(16, 16); s.refresh_list()
    def steps(s): return s.win.vp.storyboard
    def total(s): return sum(x["dur"] for x in s.steps()) or 1.0
    def refresh_list(s):
        s.list.clear(); v = s.win.vp
        for k, x in enumerate(s.steps()):
            names = ", ".join(getattr(v.bodies[i], "comp", None) or getattr(v.bodies[i], "name", None) or f"Body{i + 1}" for i in x["bodies"] if i < len(v.bodies))
            s.list.addItem(f"{k + 1}. {names}  ({fmt(x['dur'], 1)} s)")
    def auto(s):
        try: s.win.vp.storyboard = s.win.vp.explode_auto(s.factor.value())
        except Exception as ex: return W.QMessageBox.information(s, "Explode", str(ex))
        s.refresh_list(); s.slider.setValue(1000); s.scrub(1000); s.win._saved = None
    def set_dur(s, val):
        r = s.list.currentRow(); st_ = s.steps()
        for k, x in enumerate(st_):
            if r < 0 or k == r: x["dur"] = float(val)
        s.refresh_list(); s.scrub(s.slider.value())
    def scrub(s, val):
        v = s.win.vp; v.disp_off = v.explode_offsets(s.steps(), s.total()*val/1000.0); v.update()
    def play(s):
        if s.timer.isActive(): s.timer.stop(); s.play_b.setText("▶ Play"); return
        if s.slider.value() >= 1000: s.slider.setValue(0)
        s.timer.start(int(1000/s.fps.value())); s.play_b.setText("■ Stop")
    def tick(s):
        step = int(1000/(s.total()*s.fps.value()))
        if s.slider.value() >= 1000: s.timer.stop(); s.play_b.setText("▶ Play"); return
        s.slider.setValue(min(1000, s.slider.value() + max(1, step)))
    def export(s, folder=None, size=None):
        folder = folder or W.QFileDialog.getExistingDirectory(s, "Folder for the frames")
        if not folder: return
        v = s.win.vp; n = max(2, int(s.total()*s.fps.value())); out = []
        for i in range(n + 1):
            v.disp_off = v.explode_offsets(s.steps(), s.total()*i/n)
            if s.use_render.isChecked(): img = v.render_image(*(size or (v.width(), v.height())), {"ss": 1})
            else: v.repaint(); img = v.grabFramebuffer()
            p = os.path.join(folder, f"frame_{i:04d}.png"); img.save(p); out.append(p); W.QApplication.processEvents()
        s.win.statusBar().showMessage(f"Saved {len(out)} frames to {folder}", 8000); return out
    def close_panel(s):
        s.timer.stop(); s.win.vp.disp_off = {}; s.win.vp.update(); s.hide()


class RenderWin:
    WS_PAGE = {"RENDER": 6, "ANIMATION": 7, "MANUFACTURE": 8}
    def set_workspace(s, ws):
        if ws == "DESIGN":
            if s.vp.disp_off and hasattr(s, "story"): s.story.close_panel()
            s.tab_clicked(getattr(s, "design_tab", "SOLID")); return
        if s.cur_tab not in s.WS_PAGE: s.design_tab = s.cur_tab
        s.vp.skedit and s.vp.sk_finish()
        s.tab_clicked(ws)
        if ws == "ANIMATION": s.open_story()
    def render_dialog(s):
        d = RenderDialog(s); s._render_dlg = d; d.exec()
    def open_story(s):
        if not hasattr(s, "story"): s.story = StoryPanel(s)
        s.story.refresh_list(); s.story.show(); s.story.raise_()

# ---- MANUFACTURE commands + window (0.7) ----
CAM_NAMES = {"cam_face": "Face", "cam_contour": "2D Contour", "cam_pocket": "2D Pocket", "cam_drill": "Drill"}
def make_cam_commands():
    out = {}
    def add(c): c.group = "MANUFACTURE"; out[c.name] = c
    tool = lambda d=1: Fint("tool", "Tool number (T)", d, 1, 99)
    add(Cmd("cam_setup", "SETUP", "camsetup", [
        Fcombo("origin", "Work origin (on the stock top)", "corner", [("Front-left corner", "corner"), ("Centre", "center")]),
        Flen("side", "Stock: side offset", 2.0, 0.0), Flen("top", "Stock: top offset", 1.0, 0.0), Flen("bottom", "Stock: bottom offset", 0.0, 0.0),
        Flen("safe", "Safe height (above stock)", 5.0, 0.5), Flen("retract", "Retract height", 2.0, 0.1)],
        hint="The block of material you start from (the part's box plus offsets) and where X0 Y0 Z0 is: the top of the stock."))
    add(Cmd("cam_face", "FACE", "camface", [tool(5), Fnum("stepover", "Stepover", 60.0, 5.0, 95.0, 0, suffix=" %"), Flen("stepdown", "Stepdown (0 = tool's)", 0.0, 0.0)],
        hint="Skims the top of the stock flat, down to the top of the part."))
    add(Cmd("cam_contour", "2D CONTOUR", "camcontour", [
        Fsel("geo", "Flat face or closed edges", {"face", "edge"}, True), tool(1),
        Fcombo("side", "Tool side", "outside", [("Outside (cut a part out)", "outside"), ("Inside (open a hole)", "inside"), ("On the line", "on")]),
        Fcombo("to", "Bottom", "model", [("Bottom of the part", "model"), ("The picked face", "face")]), Flen("extra", "Extra depth", 0.5, 0.0),
        Flen("stepdown", "Stepdown (0 = tool's)", 0.0, 0.0)],
        hint="Follows an outline at several depths: pick the part's top face to cut it out, or the floor of a hole to open it."))
    add(Cmd("cam_pocket", "2D POCKET", "campocket", [
        Fsel("faces", "Pocket floors", {"face"}, True), tool(1), Fnum("stepover", "Stepover", 45.0, 5.0, 95.0, 0, suffix=" %"),
        Flen("stepdown", "Stepdown (0 = tool's)", 0.0, 0.0)],
        hint="Clears everything above flat floors, working outward in rings and down in steps."))
    add(Cmd("cam_drill", "DRILL", "camdrill", [
        Fsel("holes", "Holes (walls or edges)", {"face", "edge"}, True), tool(3), Flen("peck", "Peck depth (0 = none)", 0.0, 0.0),
        Fcheck("tip", "Allow for the drill point", True), Flen("extra", "Extra depth", 0.0, 0.0)],
        hint="Drills every picked hole from its top to its bottom, nearest-row order."))
    return out


class CamWin:
    def make_cam_actions(s, act, cact):
        M = {n: cact(n, None, COMMANDS[n].title.title()) for n in CAM_OPS}
        M["tools"] = act("Tool Library...", None, s.tool_library, "camtools", "End mills and drills with feeds and speeds")
        M["gen"] = act("Generate", None, lambda: s.cam_generate_all(), "compute", "Recompute every toolpath")
        M["sim"] = act("Simulate", None, lambda: s.cam_simulate(), "motion", "Run the tool along the toolpaths")
        M["post"] = act("Post Process...", None, lambda: s.cam_post(), "export", "Write G-code for GRBL or a generic controller")
        return M

    def cam_ribbon(s, M, IA, C3, construct_menu):
        return s.build_ribbon((("SETUP", (M["cam_setup"], M["tools"]), [M["cam_setup"], M["tools"]]),
                               ("2D", (M["cam_face"], M["cam_contour"], M["cam_pocket"], M["cam_drill"]), [M["cam_face"], M["cam_contour"], M["cam_pocket"], M["cam_drill"]]),
                               ("ACTIONS", (M["gen"], M["sim"], M["post"]), [M["gen"], M["sim"], M["post"]]),
                               *s.inspect_groups(IA)), "MANUFACTURE", tab_names=("MANUFACTURE",), ws="MANUFACTURE")

    def cam_ok(s, op):
        v = s.vp; idx = getattr(s, "_cam_edit", None); s._cam_edit = None
        if op["t"] == "cam_setup":
            v.cam["setup"] = {k: op[k] for k in ("origin", "side", "top", "bottom", "safe", "retract")}; errs = v.cam_regen()
        else:
            op = {k: x for k, x in op.items() if k != "icon"}; op.setdefault("name", f"{CAM_NAMES[op['t']]}{sum(1 for o in v.cam['ops'] if o['t'] == op['t']) + 1}")
            if idx is not None and idx < len(v.cam["ops"]): op["name"] = v.cam["ops"][idx].get("name", op["name"]); v.cam["ops"][idx] = op
            else: v.cam["ops"].append(op); idx = len(v.cam["ops"]) - 1
            errs = v.cam_regen(idx)
        s._saved = None
        if errs: W.QMessageBox.warning(s, "Toolpath", "\n".join(f"{v.cam['ops'][i]['name']}: {e}" for i, e in errs.items()))
        else:
            cl, rl, mins = path_stats([m for ms in v.cam_paths.values() for m in ms])
            s.statusBar().showMessage(f"Toolpaths ready: {cl/1000:.2f} m of cutting, about {mins:.1f} min.", 8000)
        s.refresh(); v.update()

    def cam_preview(s, op):
        v = s.vp
        if op["t"] == "cam_setup": return
        try: v.cam_paths[-1] = v.cam_generate(op)
        except Exception as ex: v.cam_paths.pop(-1, None); raise

    def cam_edit(s, i):
        op = s.vp.cam["ops"][i]; cmd = COMMANDS[op["t"]]; vals = cmd.load(op, s.vp); s._cam_edit = i
        s.open_cmd(op["t"], vals=vals)

    def cam_generate_all(s):
        W.QApplication.setOverrideCursor(C.Qt.WaitCursor)
        try: errs = s.vp.cam_regen()
        finally: W.QApplication.restoreOverrideCursor()
        if errs: W.QMessageBox.warning(s, "Toolpath", "\n".join(f"{s.vp.cam['ops'][i]['name']}: {e}" for i, e in errs.items()))
        s.vp.update(); return errs

    def cam_simulate(s, speed=None):
        v = s.vp; moves = [(m, v.cam_tool(v.cam["ops"][i].get("tool", 1))["dia"]/2) for i in sorted(v.cam_paths) if i >= 0 and i < len(v.cam["ops"]) for m in v.cam_paths[i]]
        if not moves: return W.QMessageBox.information(s, "Simulate", "Add a machining operation first.")
        pts = []
        for (m, R), (m2, _) in zip(moves, moves[1:]):
            a, b = np.array(m[1]), np.array(m2[1]); n = max(1, int(np.linalg.norm(b - a)/max(s.vp.span()*0.004, 0.2)))
            pts += [(tuple(a + (b - a)*k/n), R) for k in range(n)]
        s._sim = {"pts": pts, "i": 0, "step": speed or max(1, len(pts)//600)}
        if not hasattr(s, "_simt"): s._simt = C.QTimer(s); s._simt.timeout.connect(s.cam_sim_tick)
        s._simt.start(16); return len(pts)
    def cam_sim_tick(s):
        v = s.vp; m = s._sim
        if m["i"] >= len(m["pts"]) or v.cmd: s._simt.stop(); v.cam_sim = None; v.update(); return
        p, R = m["pts"][m["i"]]; v.cam_sim = {"p": p, "r": R}; m["i"] += m["step"]; v.update()

    def cam_post(s, path=None, flavor=None):
        v = s.vp
        if not v.cam["ops"]: return W.QMessageBox.information(s, "Post Process", "Add a machining operation first.")
        errs = v.cam_regen()
        if errs: return W.QMessageBox.warning(s, "Post Process", "\n".join(f"{v.cam['ops'][i]['name']}: {e}" for i, e in errs.items()))
        if flavor is None:
            items = ["GRBL (hobby CNC routers)", "Generic (Fanuc-style, with tool changes)"]
            it, ok = W.QInputDialog.getItem(s, "Post Process", "Controller:", items, 0, False)
            if not ok: return
            flavor = "grbl" if it.startswith("GRBL") else "generic"
        if not path:
            path, _ = W.QFileDialog.getSaveFileName(s, "Save G-code", f"{s.export_name()}.nc", "G-code (*.nc *.gcode *.tap)")
            if not path: return
        txt = post_gcode(v.cam, v.cam_paths, v.cam_wcs(), flavor, s.export_name())
        open(path, "w", encoding="utf-8").write(txt)
        cl, rl, mins = path_stats([m for ms in v.cam_paths.values() for m in ms])
        s.statusBar().showMessage(f"G-code saved ({txt.count(chr(10))} lines, about {mins:.1f} min).", 8000); return path

    def tool_library(s):
        v = s.vp; d = W.QDialog(s); d.setWindowTitle("Tool Library"); d.resize(860, 320); lay = W.QVBoxLayout(d)
        cols = [("n", "T"), ("name", "Name"), ("type", "Type"), ("dia", "Ø mm"), ("flutes", "Flutes"), ("rpm", "RPM"), ("feed", "Feed mm/min"),
                ("plunge", "Plunge mm/min"), ("stepdown", "Stepdown mm")]
        t = W.QTableWidget(len(v.cam["tools"]), len(cols)); t.setHorizontalHeaderLabels([c[1] for c in cols])
        def fill():
            t.setRowCount(len(v.cam["tools"]))
            for r, tool in enumerate(v.cam["tools"]):
                for c, (k, _) in enumerate(cols): t.setItem(r, c, W.QTableWidgetItem(str(tool[k])))
        fill(); t.resizeColumnsToContents(); lay.addWidget(t)
        row = W.QHBoxLayout(); add = W.QPushButton("Add tool"); rm = W.QPushButton("Remove tool"); row.addWidget(add); row.addWidget(rm); row.addStretch(1); lay.addLayout(row)
        def read():
            out = []
            for r in range(t.rowCount()):
                tool = {}
                for c, (k, _) in enumerate(cols):
                    txt = t.item(r, c).text() if t.item(r, c) else ""
                    tool[k] = txt if k in ("name", "type") else (int(float(txt or 0)) if k in ("n", "flutes", "rpm") else float(txt or 0))
                out.append(tool)
            return out
        def add_tool():
            v.cam["tools"] = read(); n = max([x["n"] for x in v.cam["tools"]] + [0]) + 1
            v.cam["tools"].append({"n": n, "name": "New tool", "type": "flat", "dia": 4.0, "flutes": 2, "rpm": 12000, "feed": 700.0, "plunge": 250.0, "stepdown": 1.0}); fill()
        def rm_tool():
            r = t.currentRow()
            if r >= 0: v.cam["tools"] = read(); del v.cam["tools"][r]; fill()
        add.clicked.connect(add_tool); rm.clicked.connect(rm_tool)
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel); bb.accepted.connect(d.accept); bb.rejected.connect(d.reject); lay.addWidget(bb)
        if d.exec():
            try: v.cam["tools"] = read(); s._saved = None; v.cam_regen(); v.update()
            except Exception as ex: W.QMessageBox.warning(s, "Tool Library", f"Couldn't read the table: {ex}")

    # ---- browser ----
    def cam_browser(s, root, item, folders):
        v = s.vp
        if not v.cam["ops"]: return
        cf = item(root, "Manufacture", "folder", "grp:cam", True); folders.append(cf)
        st_ = v.cam["setup"]; it = item(cf, f"Setup (origin: {st_['origin']})", "camsetup", "camsetup", True)
        for i, op in enumerate(v.cam["ops"]):
            moves = v.cam_paths.get(i) or []; cl, rl, mins = path_stats(moves)
            it = item(cf, f"{op.get('name')}  (T{op.get('tool', 1)})", op["t"].replace("cam_", "cam"), f"cam:{i}", op.get("visible", True))
            it.setToolTip(0, f"{cl/1000:.2f} m cutting · ~{mins:.1f} min" if moves else "No toolpath - right-click to edit")
    def cam_vis(s, ref, on):
        v = s.vp
        if ref == "grp:cam": v.show_paths = on
        elif ref.startswith("cam:"): v.cam["ops"][int(ref[4:])]["visible"] = on
        elif ref == "camsetup": pass
        else: return False
        v.update(); return True
    def cam_browser_menu(s, m, ref):
        v = s.vp
        if ref == "camsetup":
            m.addAction("Edit setup...", lambda: s.open_cmd("cam_setup", vals=dict(v.cam["setup"]))); return True
        if ref.startswith("cam:"):
            i = int(ref[4:])
            m.addAction("Edit...", lambda: s.cam_edit(i)); m.addAction("Regenerate", lambda: (v.cam_regen(i), v.update(), s.refresh()))
            m.addAction("Delete", lambda: s.cam_delete(i)); return True
        return False
    def cam_delete(s, i):
        v = s.vp; del v.cam["ops"][i]; v.cam_paths = {}; v.cam_regen(); s._saved = None; s.refresh(); v.update()
COMMANDS = make_commands()
COMMANDS.update(make_sketch_commands())
COMMANDS.update(make_surface_commands())
COMMANDS.update(make_inspect_commands())
COMMANDS.update(make_mesh_commands())
COMMANDS.update(make_asm_commands())
COMMANDS.update(make_sheet_commands())
COMMANDS.update(make_plastic_commands())
COMMANDS.update(make_cam_commands())
OP_TO_CMD = {"extrude": "extrude", "revolve": "revolve", "sweep": "sweep", "loft": "loft", "rib": "rib", "web": "web", "emboss": "emboss",
             "hole": "hole", "thread2": "thread", "rpattern": "rpattern", "cpattern": "cpattern", "ppattern": "ppattern", "mirror": "mirror",
             "thicken": "thicken", "bfill": "bfill", "presspull": "presspull", "fillet2": "fillet", "chamfer2": "chamfer", "fillet": "fillet",
             "chamfer": "chamfer", "shell": "shell", "draft": "draft", "scale": "scale", "combine": "combine", "offsetface": "offsetface",
             "replaceface": "replaceface", "splitface": "splitface", "splitbody": "splitbody", "move2": "move2", "align": "align",
             "material": "material", "appearance": "appearance", "defeature": "defeature"}
OP_TO_CMD.update({n: n for n in SURF_OPS})
OP_TO_CMD.update({n: n for n in MESH_OPS if n in COMMANDS})
OP_TO_CMD.update({"comp": "comp", "joint": "joint"})
OP_TO_CMD.update({n: n for n in SM_OPS})
OP_TO_CMD.update({n: n for n in PL_OPS})
PRIM_TO_CMD = {"box": "box", "cyl": "cylinder", "sphere": "sphere", "torus": "torus", "coil": "coil", "pipe": "pipe"}

def cmd_for_op(op):
    if not op: return None
    if op["t"] == "construct": return COMMANDS.get("c_" + op["how"])
    if op["t"] == "prim": return COMMANDS.get(PRIM_TO_CMD.get(op.get("shape")))
    return COMMANDS.get(OP_TO_CMD.get(op["t"]))

class CmdForm(W.QFrame):
    """Floating command dialog built from a Cmd's fields."""
    def __init__(s, app, cmd, vals, edit_index=None):
        super().__init__(app.vp); s.app, s.cmd, s.edit_index = app, cmd, edit_index
        s.setObjectName("cmd"); s.setFixedWidth(320)
        lay = W.QVBoxLayout(s); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        hd = W.QLabel(cmd.title + (f"   ·   editing step {edit_index}" if edit_index else "")); hd.setObjectName("cmdhdr"); lay.addWidget(hd)
        sa = W.QScrollArea(); sa.setWidgetResizable(True); sa.setFrameShape(W.QFrame.NoFrame); sa.setHorizontalScrollBarPolicy(C.Qt.ScrollBarAlwaysOff)
        body = W.QWidget(); bl = W.QVBoxLayout(body); bl.setContentsMargins(12, 10, 12, 8); bl.setSpacing(8); sa.setWidget(body); lay.addWidget(sa)
        s.scroll = sa; s.form = W.QFormLayout(); s.form.setHorizontalSpacing(10); s.form.setVerticalSpacing(7); bl.addLayout(s.form)
        s.vals, s.widgets, s.active = {}, {}, None
        s.vals["_x"] = {k: list(x) for k, x in (vals.get("_x") or {}).items()}
        for f in cmd.fields: s.add_field(f, vals.get(f["key"], f["default"]))
        if cmd.hint:
            h = W.QLabel(cmd.hint); h.setObjectName("hint"); h.setWordWrap(True); bl.addWidget(h)
        s.status = W.QLabel(""); s.status.setWordWrap(True); s.status.setStyleSheet("color:#b3261e;font-size:11px"); bl.addWidget(s.status)
        row = W.QHBoxLayout(); row.setContentsMargins(12, 0, 12, 10); row.addStretch(1)
        no, s.ok = W.QPushButton("Cancel"), W.QPushButton("OK"); s.ok.setObjectName("primary")
        row.addWidget(no); row.addWidget(s.ok); lay.addLayout(row)
        no.clicked.connect(app.cmd_cancel); s.ok.clicked.connect(app.cmd_ok)
        first = next((f for f in cmd.fields if f["type"] == "sel" and not s.vals[f["key"]] and s.visible(f)), None) or \
                next((f for f in cmd.fields if f["type"] == "sel" and s.visible(f)), None)
        if first: s.activate(first["key"])
        s.refresh_rows(); s.move(16, 16); s.fit_height(); s.show(); s.raise_()

    # ---- fields ----
    def add_field(s, f, val):
        k, t = f["key"], f["type"]
        if t == "sel":
            w = W.QWidget(); h = W.QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(3)
            b = W.QPushButton("Select"); b.setObjectName("selbtn"); b.setCheckable(True); b.clicked.connect(lambda _=False, k=k: s.activate(k))
            x = W.QToolButton(); x.setText("✕"); x.setObjectName("selx"); x.setToolTip("Clear"); x.clicked.connect(lambda _=False, k=k: s.clear(k))
            h.addWidget(b, 1); h.addWidget(x); s.widgets[k] = (w, b); s.vals[k] = list(val or [])
        elif t in ("len", "ang", "num", "int"):
            x = s.vals["_x"].get(k)
            if x:
                try: val = eval_expr(x[0], s.app.vp.dim_env(), x[1])
                except Exception: s.vals["_x"].pop(k, None)
            if t == "len": b = ExprLenSpin(f["lo"], f["hi"]); b.expr_setup(s, k, "len")
            elif t == "int": b = W.QSpinBox(); b.setRange(f["lo"], f["hi"])
            else:
                b = ExprNumSpin(); b.expr_setup(s, k, "ang"); b.setRange(f["lo"], f["hi"]); b.setDecimals(2 if t == "ang" else f["dec"])
                b.setSuffix(" °" if t == "ang" else f.get("suffix", ""))
            b.setKeyboardTracking(False); b.setValue(val); b.valueChanged.connect(lambda v, k=k: s.set_val(k, v)); s.widgets[k] = (b, b); s.vals[k] = val
        elif t == "combo":
            b = W.QComboBox(); b.setMaxVisibleItems(20)
            for lab, data in f["items"]: b.addItem(lab, data)
            i = next((j for j, (_, d) in enumerate(f["items"]) if d == val), 0); b.setCurrentIndex(i)
            b.currentIndexChanged.connect(lambda i, k=k, b=b: s.set_val(k, b.itemData(i))); s.widgets[k] = (b, b); s.vals[k] = f["items"][i][1]
        elif t == "check":
            b = W.QCheckBox(); b.setChecked(bool(val)); b.toggled.connect(lambda v, k=k: s.set_val(k, v)); s.widgets[k] = (b, b); s.vals[k] = bool(val)
        elif t == "text":
            b = W.QLineEdit(str(val)); b.setPlaceholderText(f.get("ph", "")); b.editingFinished.connect(lambda k=k, b=b: s.set_val(k, b.text()))
            s.widgets[k] = (b, b); s.vals[k] = str(val)
        elif t == "feats":
            b = W.QListWidget(); b.setMaximumHeight(110)
            for i, ft in enumerate(s.app.vp.features):
                it = W.QListWidgetItem(f"{i + 1}. {ft['name']}"); it.setFlags(it.flags() | C.Qt.ItemIsUserCheckable)
                it.setCheckState(C.Qt.Checked if i in (val or []) else C.Qt.Unchecked); b.addItem(it)
            b.itemChanged.connect(lambda _it, k=k, b=b: s.set_val(k, [i for i in range(b.count()) if b.item(i).checkState() == C.Qt.Checked]))
            s.widgets[k] = (b, b); s.vals[k] = list(val or [])
        s.form.addRow(f["label"], s.widgets[k][0])
        if t == "sel": s.update_sel_text(k)

    def field(s, k): return next(f for f in s.cmd.fields if f["key"] == k)
    def visible(s, f): return f["show"] is None or bool(f["show"](s.vals))
    def refresh_rows(s):
        for f in s.cmd.fields:
            try: s.form.setRowVisible(s.widgets[f["key"]][0], s.visible(f))
            except Exception: pass
        if s.active and not s.visible(s.field(s.active)): s.activate(None)
        s.fit_height()
    def fit_height(s):
        vp = s.app.vp; s.scroll.widget().adjustSize()
        want = s.scroll.widget().sizeHint().height() + 2
        s.scroll.setFixedHeight(max(80, min(want, vp.height() - 140))); s.adjustSize()

    def set_val(s, k, v):
        s.vals[k] = v; s.refresh_rows(); s.app.schedule_preview()
    def set_value(s, k, v):
        s.vals.get("_x", {}).pop(k, None)
        w = s.widgets[k][1]; w.blockSignals(True); w.setValue(v); w.blockSignals(False); s.vals[k] = v; s.app.schedule_preview()

    # ---- selection boxes ----
    def update_sel_text(s, k):
        n = len(s.vals[k]); b = s.widgets[k][1]
        b.setText(("1 selected" if n == 1 else f"{n} selected") if n else "Select"); b.setChecked(k == s.active)
    def activate(s, k):
        s.active = k
        for f in s.cmd.fields:
            if f["type"] == "sel": s.update_sel_text(f["key"])
        s.app.vp.setFocus()
    def clear(s, k):
        s.vals[k] = []; s.update_sel_text(k); s.activate(k); s.app.schedule_preview()
    def active_kinds(s):
        if not s.active: return set()
        kinds = set(s.field(s.active)["kinds"])
        if "pface" in kinds: kinds |= {"pface"}
        return kinds
    def wants(s, kinds): return bool(s.active_kinds() & kinds)
    def all_refs(s):
        out = []
        for f in s.cmd.fields:
            if f["type"] == "sel" and s.visible(f): out += s.vals[f["key"]]
        return out
    def pick(s, ref):
        if not s.active or ref is None: return
        f = s.field(s.active); cur = s.vals[s.active]; key = ref_key(ref)
        if f["multi"]:
            hit = next((r for r in cur if ref_key(r) == key), None)
            if hit: cur.remove(hit)
            else: cur.append(ref)
        else:
            s.vals[s.active] = [ref]
            nxt = next((g for g in s.cmd.fields if g["type"] == "sel" and g["key"] != s.active and s.visible(g) and g["req"] and not s.vals[g["key"]]), None)
            s.update_sel_text(s.active)
            if nxt: s.activate(nxt["key"])
        s.update_sel_text(f["key"]); s.app.schedule_preview()

class SolidUI:
    """Command handling for the main window."""
    def open_cmd(s, name, edit_index=None, vals=None):
        v = s.vp
        if s.busy() or s._play.isActive(): return
        if v.meas is not None: s.measure_stop()
        cmd = COMMANDS[name]
        if cmd.group == "SKETCH":
            if not v.skedit: return s.statusBar().showMessage("That tool works inside a sketch - start or edit a sketch first.", 6000)
            si = v.skedit["si"]; pre = {}
            cs = [("curve", si, i[1]) for i in sorted(v.sk_sel) if i[0] == "c"]; ps = [("spoint", si, i[1]) for i in sorted(v.sk_sel) if i[0] == "p"]
            for f in cmd.fields:
                if f["type"] == "sel" and f["multi"] and f["key"] in ("objs", "curves", "srefs") and (cs or ps):
                    pre[f["key"]] = cs + (ps if "point" in f["kinds"] else []); break
            vals = {**pre, **(vals or {})}; v.sk_set_tool(None)
        else:
            if v.skedit: v.sk_finish()
            vals = vals if edit_index else {**s.prefill(cmd), **(vals or {})}
        v.tool, v.pts = None, []; s.show_panel(None)
        s._edit_back = None
        v.cmd = CmdForm(s, cmd, vals, edit_index); v.cmd_hover = None; v.preview = None; v.handle = None
        s.schedule_preview(); s.refresh(); v.setFocus(); v.update()

    def prefill(s, cmd):
        v = s.vp; vals = {}; used = set()
        pools = []
        if v.active and v.active.visible and v.active.entities and v.active in v.sketches:
            si = v.sketches.index(v.active); regs = v.active.regions()
            if regs: pools.append(("profile", [("profile", si, i) for i in (sorted(v.active.sel) or range(len(regs)))]))
        if v.sel: pools.append(("edge", [("edge", bi, ei) for bi, ei in sorted(v.sel)]))
        if v.sel_face:
            bi, fid = v.sel_face; pools.append(("face", [("face", bi, fid)]))
        if v.sel_body is not None and v.sel_body < len(v.bodies):
            pools.append(("body", [("body", i) for i in sorted(v.msel | {v.sel_body}) if i < len(v.bodies)]))
        for f in cmd.fields:
            if f["type"] != "sel" or f["key"] == "bodies": continue
            for kind, refs in pools:
                if kind in used: continue
                ok = kind in f["kinds"] or (kind == "face" and f["kinds"] & {"pface", "plane", "axis"}) or (kind == "profile" and "profile" in f["kinds"])
                if kind == "face" and "pface" in f["kinds"] and not "face" in f["kinds"]:
                    ok = v.bodies[refs[0][1]].face_normal(refs[0][2]) is not None
                if ok:
                    vals[f["key"]] = refs if f["multi"] else refs[:1]; used.add(kind); break
        return vals

    def schedule_preview(s):
        if not hasattr(s, "_pv_timer"):
            s._pv_timer = C.QTimer(s); s._pv_timer.setSingleShot(True); s._pv_timer.timeout.connect(s.update_preview)
        s._pv_timer.start(220)

    def update_preview(s):
        v, form = s.vp, s.vp.cmd
        if not form: return
        try:
            op = form.cmd.build(form.vals, v)
        except CmdError as ex:
            v.preview = None; form.status.setText(str(ex)); form.ok.setEnabled(False); v.handle = None; v.update(); return
        except Exception as ex:
            v.preview = None; form.status.setText(str(ex)); form.ok.setEnabled(False); v.update(); return
        h = form.cmd.handle(form.vals, v) if form.cmd.handle else None
        v.handle = dict(o=h[0], d=h[1], key=h[2], val=form.vals[h[2]]) if h else None
        if op.get("t") in ANALYSIS_OPS:
            v.preview = None; v.analysis_preview(op); form.status.setText(""); form.ok.setEnabled(True); v.update(); return
        if op.get("t") in CAM_OPS:
            v.preview = None
            try: s.cam_preview(op); form.status.setText(""); form.ok.setEnabled(True)
            except Exception as ex: form.status.setText(str(ex)); form.ok.setEnabled(False)
            v.update(); return
        try:
            W.QApplication.setOverrideCursor(C.Qt.BusyCursor); v.preview = v.dry_run(op); form.status.setText(""); form.ok.setEnabled(True)
            if op["t"] == "bfill":
                try:
                    cells = boundary_cells([v.ref_tool_shape(r) for r in op["tools"]])
                    v.preview["cells"] = [centroid(c) for c in cells]
                except Exception: pass
        except Exception as ex:
            traceback.print_exc(); v.preview = None; form.status.setText(f"{ex}"); form.ok.setEnabled(False)
        finally: W.QApplication.restoreOverrideCursor()
        v.update()

    def close_cmd(s):
        v = s.vp
        if v.cmd: v.cmd.hide(); v.cmd.deleteLater()
        v.cmd = v.preview = v.handle = v.cmd_hover = None; v.type_buf = None; v.ana_prev = None; v.cam_paths.pop(-1, None); v.update()

    def cmd_cancel(s):
        v = s.vp; form = v.cmd; s.close_cmd(); s._ana_edit = None; s._cam_edit = None
        if form and form.edit_index and s._edit_back is not None: v.restore(s._edit_back)
        s._edit_back = None; s.refresh()

    def cmd_ok(s):
        v = s.vp; form = v.cmd
        if not form: return
        try: op = form.cmd.build(form.vals, v)
        except Exception as ex: form.status.setText(str(ex)); return
        idx, back = form.edit_index, s._edit_back
        s.close_cmd(); s._edit_back = None
        if op.get("t") in ANALYSIS_OPS: s.analysis_ok(op); s.refresh(); return
        if op.get("t") in CAM_OPS: v.cam_paths.pop(-1, None); s.cam_ok(op); return
        if op.get("t") == "skmod":
            try:
                W.QApplication.setOverrideCursor(C.Qt.WaitCursor); v.exec_op(op); v.sk_commit(v.esk.geo, solve=False)
            except Exception as ex: traceback.print_exc(); W.QMessageBox.warning(s, form.cmd.title.title(), str(ex))
            finally: W.QApplication.restoreOverrideCursor()
            s.refresh(); v.update(); return
        if idx:
            v.restore(back)
            try:
                W.QApplication.setOverrideCursor(C.Qt.WaitCursor); v.replay_edit(idx, op)
                s.statusBar().showMessage(f"Step {idx} updated - {len(v.states) - 1 - idx} later step(s) recomputed.", 7000)
            except Exception as ex: W.QMessageBox.warning(s, "Couldn't apply the change", str(ex))
            finally: W.QApplication.restoreOverrideCursor()
        else:
            v.confirm_pending()
            s.run(form.cmd.title.title(), op)
        v.sel.clear(); v.sel_face = None; s.refresh(); v.update()

    def edit_generic(s, i):
        v = s.vp; op = v.states[i]["op"]; cmd = cmd_for_op(op)
        if not cmd: return False
        back = v.pos; v.restore(i - 1)
        try: vals = cmd.load(op, v)
        except Exception as ex:
            v.restore(back); W.QMessageBox.warning(s, "Edit", f"This step can't be reopened: {ex}"); return True
        s.open_cmd(cmd.name, edit_index=i, vals=vals); s._edit_back = back
        return True

    def compute_all(s):
        v = s.vp
        if len(v.states) < 2 or s.busy(): return
        try:
            W.QApplication.setOverrideCursor(C.Qt.WaitCursor); v.replay_edit(1, v.states[1]["op"], strict=False)
            s.statusBar().showMessage(f"Recomputed {len(v.states) - 1} steps.", 5000)
        except Exception as ex: W.QMessageBox.warning(s, "Compute all", str(ex))
        finally: W.QApplication.restoreOverrideCursor(); s.refresh()

    def delete_step(s, i):
        v = s.vp
        if s.busy() or i <= 0: return
        try:
            W.QApplication.setOverrideCursor(C.Qt.WaitCursor); v.replay_remove(i)
            s.statusBar().showMessage(f"Step {i} deleted and the later steps recomputed.", 6000)
        except Exception as ex: W.QMessageBox.warning(s, "Couldn't delete the step", str(ex))
        finally: W.QApplication.restoreOverrideCursor(); s.refresh()

# ---------------------------------------------------------------------------------------------------------------
#  Sketch environment: window side (contextual ribbon, Sketch Palette, dialogs, parameters, inserts)
# ---------------------------------------------------------------------------------------------------------------
class SketchPalette(W.QFrame):
    """Fusion's Sketch Palette: line type, display toggles, polygon sides / conic rho, status, Finish Sketch."""
    OPTS = (("grid", "Sketch Grid"), ("snap", "Snap"), ("slice", "Slice"), ("profiles", "Show Profile"), ("points", "Show Points"),
            ("dims", "Show Dimensions"), ("cons", "Show Constraints"), ("proj", "Show Projected Geometries"), ("sk3d", "3D Sketch"))
    def __init__(s, win):
        super().__init__(win.vp); s.win = win; vp = win.vp; s.setObjectName("cmd"); s.setFixedWidth(236)
        lay = W.QVBoxLayout(s); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        hd = W.QLabel("SKETCH PALETTE"); hd.setObjectName("cmdhdr"); lay.addWidget(hd)
        body = W.QWidget(); bl = W.QVBoxLayout(body); bl.setContentsMargins(10, 8, 10, 10); bl.setSpacing(5); lay.addWidget(body)
        t = W.QLabel("Options"); t.setStyleSheet("font-weight:bold;color:#444"); bl.addWidget(t)
        row = W.QHBoxLayout(); row.setSpacing(4)
        s.b_cons = s.tb("skcons", "Construction (X): new geometry, or toggle the selected curves", lambda: vp.sk_toggle_flag("cons"))
        s.b_cl = s.tb("skcl", "Centerline: dash-dot lines used as axes", lambda: vp.sk_toggle_flag("cl"))
        s.b_look = s.tb("lookat", "Look At the sketch", lambda: vp.esk and vp.look_at_plane(vp.esk.plane.n), False)
        for b in (s.b_cons, s.b_cl, s.b_look): row.addWidget(b)
        row.addStretch(1); bl.addLayout(row)
        s.checks = {}
        for key, label in s.OPTS:
            cb = W.QCheckBox(label); cb.toggled.connect(lambda on, k=key: s.set_opt(k, on)); bl.addWidget(cb); s.checks[key] = cb
        f = W.QFormLayout(); f.setHorizontalSpacing(8)
        s.sides = W.QSpinBox(); s.sides.setRange(3, 200); s.sides.valueChanged.connect(lambda v: s.set_opt("sides", v, False)); f.addRow("Polygon sides", s.sides)
        s.rho = W.QDoubleSpinBox(); s.rho.setRange(0.01, 0.99); s.rho.setSingleStep(0.05); s.rho.setDecimals(3)
        s.rho.valueChanged.connect(lambda v: s.set_opt("rho", v, False)); f.addRow("Conic rho", s.rho); bl.addLayout(f)
        s.status = W.QLabel(""); s.status.setWordWrap(True); s.status.setObjectName("hint"); bl.addWidget(s.status)
        fin = W.QPushButton("  Finish Sketch"); fin.setIcon(icon("finish", 18)); fin.setObjectName("primary"); fin.clicked.connect(vp.sk_finish)
        bl.addWidget(fin); s.hide()
    def tb(s, ic, tip, fn, check=True):
        b = W.QToolButton(); b.setIcon(icon(ic, 26)); b.setIconSize(C.QSize(26, 26)); b.setToolTip(tip); b.setCheckable(check)
        b.setAutoRaise(True); b.clicked.connect(lambda *_: (fn(), s.sync())); return b
    def set_opt(s, k, v, upd=True):
        vp = s.win.vp; vp.sk_opts[k] = v
        if k == "sk3d": vp.sk_set_tool(vp.tool); s.win.statusBar().showMessage(
            "3D Sketch on: Line and Fit Point Spline now snap to model corners and edges in space." if v else "3D Sketch off.", 6000)
        vp.update()
    def sync(s):
        vp = s.win.vp; o = vp.sk_opts
        for k, cb in s.checks.items(): cb.blockSignals(True); cb.setChecked(bool(o[k])); cb.blockSignals(False)
        for w_, v in ((s.sides, o["sides"]), (s.rho, o["rho"])): w_.blockSignals(True); w_.setValue(v); w_.blockSignals(False)
        s.b_cons.setChecked(o["construction"]); s.b_cl.setChecked(o["centerline"])
        st_ = vp.sk_status()
        if st_ is None: s.status.setText(""); return
        nc = len(vp.esk.geo.C)
        if nc == 0: s.status.setText("Empty sketch - pick a tool from CREATE.")
        elif st_["dof"] == 0: s.status.setText("<span style='color:#1f7a32'><b>Fully constrained</b></span> - the sketch is black.")
        else: s.status.setText(f"<span style='color:#1d5c93'><b>{st_['dof']} degrees of freedom</b></span> - blue geometry can still move. "
                               "Add dimensions (D) or constraints until it turns black.")
        s.adjustSize()

def expr_dialog(parent, title, text, kind, env, name=None, driven=None, note=None):
    """Type a value or an expression (e.g. width/2 + 3 mm).  Returns (text, driven) or None."""
    dlg = W.QDialog(parent); dlg.setWindowTitle(title); dlg.setMinimumWidth(340); f = W.QFormLayout(dlg)
    if note: lb = W.QLabel(note); lb.setWordWrap(True); lb.setObjectName("hint"); f.addRow(lb)
    ed = W.QLineEdit(text); ed.selectAll(); res = W.QLabel(""); f.addRow(name or "Value", ed); f.addRow("", res)
    dv = None
    if driven is not None: dv = W.QCheckBox("Driven (reference) dimension"); dv.setChecked(driven); f.addRow("", dv)
    names = sorted(env)
    if names:
        hint = W.QLabel("You can use: " + ", ".join(names[:24]) + (" ..." if len(names) > 24 else "")); hint.setWordWrap(True); hint.setObjectName("hint"); f.addRow(hint)
    bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel); f.addRow(bb)
    bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
    def upd(*_):
        if dv is not None and dv.isChecked(): res.setText("(measured, not driving)"); bb.button(W.QDialogButtonBox.Ok).setEnabled(True); return
        try:
            v = eval_expr(ed.text(), env, kind); res.setText(f"= {fmt(v, 3)}°" if kind == "ang" else f"= {flen(v)}"); res.setStyleSheet("color:#1f7a32")
            bb.button(W.QDialogButtonBox.Ok).setEnabled(True)
        except ExprError as ex:
            res.setText(str(ex)); res.setStyleSheet("color:#b3261e"); bb.button(W.QDialogButtonBox.Ok).setEnabled(False)
    ed.textChanged.connect(upd)
    if dv is not None: dv.toggled.connect(upd)
    upd(); ed.setFocus()
    if not dlg.exec(): return None
    return ed.text().strip(), (dv.isChecked() if dv is not None else None)

class ParamsDialog(W.QDialog):
    """Change Parameters: user parameters plus every sketch dimension (model parameters)."""
    def __init__(s, win):
        super().__init__(win); s.win = win; vp = win.vp; s.setWindowTitle("Parameters"); s.resize(720, 460)
        lay = W.QVBoxLayout(s)
        lb = W.QLabel("User parameters can be used in any dimension (e.g. width/2 + 3 mm). Changing a value recomputes the whole design.")
        lb.setWordWrap(True); lb.setObjectName("hint"); lay.addWidget(lb)
        s.tab = W.QTableWidget(0, 5); s.tab.setHorizontalHeaderLabels(["Parameter", "Expression", "Value", "Where", "Comment"])
        s.tab.horizontalHeader().setStretchLastSection(True); s.tab.verticalHeader().setVisible(False); lay.addWidget(s.tab, 1)
        row = W.QHBoxLayout(); add = W.QPushButton("+ User parameter"); rem = W.QPushButton("Delete user parameter")
        add.clicked.connect(s.add_user); rem.clicked.connect(s.del_user); row.addWidget(add); row.addWidget(rem); row.addStretch(1)
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel); bb.accepted.connect(s.accept); bb.rejected.connect(s.reject)
        row.addWidget(bb); lay.addLayout(row)
        s.rows = []
        for nm, p in vp.params.items(): s.add_row("user", nm, p.get("x", ""), p.get("v", 0.0), "User", p.get("c", ""))
        for si, sk in enumerate(vp.sketches):
            for ki, k in enumerate(sk.geo.K):
                if k["t"] in DIM_TYPES and k.get("n") and not k.get("drv"):
                    s.add_row(("dim", si, ki), k["n"], k.get("x", ""), k.get("v", 0.0), f"{sk.name or f'Sketch{si+1}'} · {CON_NAMES[k['t']]}", "")
        s.tab.resizeColumnsToContents(); s.tab.setColumnWidth(1, 200)
    def add_row(s, kind, name, expr, val, where, comment):
        r = s.tab.rowCount(); s.tab.insertRow(r)
        for c, txt in enumerate((name, expr, fmt(val, 4), where, comment)):
            it = W.QTableWidgetItem(str(txt))
            if c in (2, 3) or (c == 0 and kind != "user") or (c == 4 and kind != "user"): it.setFlags(it.flags() & ~C.Qt.ItemIsEditable)
            if kind != "user" and c != 1: it.setForeground(G.QColor("#555"))
            s.tab.setItem(r, c, it)
        s.rows.append(kind)
    def add_user(s):
        i = 1
        names = {s.tab.item(r, 0).text() for r in range(s.tab.rowCount())}
        while f"param{i}" in names: i += 1
        s.add_row("user", f"param{i}", "10 mm", 10.0, "User", ""); s.tab.editItem(s.tab.item(s.tab.rowCount() - 1, 0))
    def del_user(s):
        r = s.tab.currentRow()
        if r >= 0 and s.rows[r] == "user": s.tab.removeRow(r); del s.rows[r]
    def result(s):
        users = {}; dims = []
        for r, kind in enumerate(s.rows):
            name, expr = s.tab.item(r, 0).text().strip(), s.tab.item(r, 1).text().strip()
            if kind == "user":
                if not re.fullmatch(r"[A-Za-z_]\w*", name): raise ExprError(f"'{name}' isn't a valid parameter name")
                users[name] = {"x": expr, "c": s.tab.item(r, 4).text()}
            else: dims.append((kind, expr))
        return users, dims

class SketchWin:
    """Window-side sketch environment."""
    SK_CREATE_MENU = [("line",), ("Rectangle", ["rect2", "rect3", "rectc"]), ("Circle", ["circle", "circle2p", "circle3p", "circle2t", "circle3t"]),
                      ("Arc", ["arc3p", "arcc", "arct"]), ("Polygon", ["poly_circ", "poly_insc", "poly_edge"]), ("ellipse",),
                      ("Slot", ["slot_cc", "slot_overall", "slot_center", "slot_arc3", "slot_arcc"]), ("Spline", ["spline", "cspline"]),
                      ("conic",), ("point",), ("text",), None, ("cmd:sk_mirror",), ("cmd:sk_cpattern",), ("cmd:sk_rpattern",), None,
                      ("Project / Include", ["cmd:sk_project", "cmd:sk_intersect", "cmd:sk_include", "cmd:sk_projsurf"]), None, ("dim",)]
    SK_MODIFY_MENU = ["fillet", "chamfer", "trim", "extend", "break", "cmd:sk_scale", "cmd:sk_offset", "cmd:sk_move", None, "params"]

    def setup_sketch_ui(s):
        v = s.vp; s.sk_acts = {}
        def tact(tool):
            title, ic, _, hint = SK_TOOLS[tool]
            a = G.QAction(icon(ic), title, s); a.setCheckable(True); a.setToolTip(f"{title}\n{hint}")
            a.triggered.connect(lambda _=False, t=tool: s.sk_tool(t)); s.sk_acts[tool] = a; return a
        for t in SK_TOOLS: tact(t)
        for t, title, ic in CON_TOOLS:
            a = G.QAction(icon(ic), title, s); a.setCheckable(True); a.setToolTip(f"{title}\n{CON_HINT[t]}")
            a.triggered.connect(lambda _=False, t=t: s.sk_tool("con:" + t)); s.sk_acts["con:" + t] = a
        for name in ("sk_offset", "sk_mirror", "sk_cpattern", "sk_rpattern", "sk_move", "sk_scale", "sk_project", "sk_intersect", "sk_include", "sk_projsurf"):
            c = COMMANDS[name]; a = G.QAction(icon(c.icon), c.title.title().replace("Sketch ", "Sketch "), s)
            a.setToolTip(f"{c.title.title()}\n{c.hint}"); a.triggered.connect(lambda _=False, n=name: s.open_cmd(n)); s.sk_acts["cmd:" + name] = a
        mk = lambda key, ic, title, fn, tip=None: s.sk_acts.__setitem__(key, s._plain_act(ic, title, fn, tip))
        mk("params", "params", "Change Parameters...", s.params_dialog, "User parameters and every dimension, with expressions.")
        mk("dxf", "dxf", "Insert DXF...", lambda: s.insert_file("dxf"), "Import lines, arcs, circles, polylines, splines and ellipses.")
        mk("svg", "svg", "Insert SVG...", lambda: s.insert_file("svg"), "Import paths and shapes (in mm; arcs become splines).")
        mk("canvas", "canvas", "Canvas...", s.insert_canvas, "Put a reference image on a plane or face (calibrate it afterwards).")
        mk("decal", "decal", "Decal...", s.insert_decal, "Stick an image onto a flat face of a body.")
        mk("finish", "finish", "Finish Sketch", lambda: v.sk_finish(), "Leave the sketch (Ctrl+Enter).")
        mk("look", "lookat", "Look At", lambda: v.esk and v.look_at_plane(v.esk.plane.n))
        mk("savedxf", "dxf", "Save Sketch as DXF...", s.export_sketch_dxf)
        s.sk_acts["finish"].setShortcuts([G.QKeySequence("Ctrl+Return"), G.QKeySequence("Ctrl+Enter")]); s.addAction(s.sk_acts["finish"])
        for key, fn in (("L", lambda: s.sk_tool("line")), ("R", lambda: s.sk_tool("rect2")), ("C", lambda: s.sk_tool("circle")),
                        ("D", lambda: s.sk_tool("dim") if v.skedit else s.drawing()), ("T", lambda: s.sk_tool("trim") if v.skedit else s.open_cmd("thread")),
                        ("O", lambda: s.open_cmd("sk_offset") if v.skedit else None), ("X", lambda: v.skedit and (v.sk_toggle_flag("cons"), s.palette.sync())),
                        ("P", lambda: s.open_cmd("sk_project") if v.skedit else s.open_cmd("cpattern"))):
            a = G.QAction(s); a.setShortcut(G.QKeySequence(key)); a.triggered.connect(fn); s.addAction(a)
        s.palette = SketchPalette(s); v.palette = s.palette
        v.sk_mode.connect(s.on_sk_mode); v.dim_place.connect(s.on_dim_place); v.dim_edit.connect(s.on_dim_edit)
        v.text_dialog.connect(s.on_text_dialog); v.corner_dialog.connect(s.on_corner_dialog); v.type_dialog.connect(s.on_type_dialog)
        v.calib_done.connect(s.on_calib_done)
        v.changed.connect(lambda: s.palette.isVisible() and s.palette.sync())

    def _plain_act(s, ic, title, fn, tip=None):
        a = G.QAction(icon(ic), title, s); a.triggered.connect(lambda _=False: fn()); a.setToolTip(f"{title}\n{tip}" if tip else title); return a

    def build_sketch_ribbon(s):
        A = s.sk_acts; rib = W.QFrame(); rib.setObjectName("ribbon"); h = W.QHBoxLayout(rib); h.setContentsMargins(10, 6, 10, 4); h.setSpacing(12)
        mode = s.tbtn("mode", text="DESIGN  ▾"); mode.setToolButtonStyle(C.Qt.ToolButtonTextOnly); h.addWidget(mode, 0, C.Qt.AlignTop)
        right = W.QVBoxLayout(); right.setSpacing(0); h.addLayout(right, 1)
        tabs = W.QHBoxLayout(); tabs.setSpacing(0)
        for name, on in (("SOLID", False), ("SKETCH", True)):
            b = s.tbtn("tab", text=name); b.setCheckable(True); b.setChecked(on)
            if not on: b.clicked.connect(lambda: (s.vp.sk_finish(),)); s.sk_ws_tab = b
            else: b.setStyleSheet("color:#0b4f7a")
            tabs.addWidget(b)
        tabs.addStretch(1); right.addLayout(tabs)
        row = W.QHBoxLayout(); row.setSpacing(0); right.addLayout(row)
        def group(title, acts, menu):
            box = W.QWidget(); vb = W.QVBoxLayout(box); vb.setContentsMargins(8, 4, 8, 0); vb.setSpacing(0)
            icons_ = W.QHBoxLayout(); icons_.setSpacing(2)
            for a in acts:
                b = s.tbtn("rb"); b.setDefaultAction(a); b.setIconSize(C.QSize(34, 34)); b.setFixedSize(46, 46); icons_.addWidget(b)
            vb.addLayout(icons_)
            g = s.tbtn("grp", text=f"{title} ▾"); g.setToolButtonStyle(C.Qt.ToolButtonTextOnly); g.setPopupMode(W.QToolButton.InstantPopup)
            gm = W.QMenu(g); s._fill_menu(gm, menu); g.setMenu(gm); vb.addWidget(g, 0, C.Qt.AlignHCenter); return box
        def sep():
            f = W.QFrame(); f.setObjectName("sep"); f.setFixedWidth(1); return f
        row.addWidget(group("CREATE", [A[k] for k in ("line", "rect2", "circle", "arc3p", "poly_circ", "spline", "slot_cc", "text")] +
                            [A["cmd:sk_mirror"], A["cmd:sk_cpattern"], A["cmd:sk_project"], A["dim"]], s.SK_CREATE_MENU))
        row.addWidget(sep())
        row.addWidget(group("MODIFY", [A["fillet"], A["trim"], A["cmd:sk_offset"]], s.SK_MODIFY_MENU))
        row.addWidget(sep())
        cbox = W.QWidget(); cv = W.QVBoxLayout(cbox); cv.setContentsMargins(8, 2, 8, 0); cv.setSpacing(0); grid = W.QGridLayout(); grid.setSpacing(1)
        for i, (t, title, ic) in enumerate(CON_TOOLS):
            b = s.tbtn("rb"); b.setDefaultAction(A["con:" + t]); b.setIconSize(C.QSize(20, 20)); b.setFixedSize(25, 23); grid.addWidget(b, i // 6, i % 6)
        cv.addLayout(grid)
        g = s.tbtn("grp", text="CONSTRAINTS ▾"); g.setToolButtonStyle(C.Qt.ToolButtonTextOnly); g.setPopupMode(W.QToolButton.InstantPopup)
        gm = W.QMenu(g); s._fill_menu(gm, ["con:" + t for t, _, _ in CON_TOOLS]); g.setMenu(gm); cv.addWidget(g, 0, C.Qt.AlignHCenter)
        row.addWidget(cbox); row.addWidget(sep())
        row.addWidget(group("INSERT", [A["dxf"], A["svg"], A["canvas"]], ["dxf", "svg", None, "canvas", "decal"]))
        row.addWidget(sep())
        fbox = W.QWidget(); fv = W.QVBoxLayout(fbox); fv.setContentsMargins(8, 4, 8, 0); fv.setSpacing(0)
        fb = s.tbtn("rb"); fb.setDefaultAction(A["finish"]); fb.setIconSize(C.QSize(34, 34)); fb.setFixedSize(46, 46); fv.addWidget(fb, 0, C.Qt.AlignHCenter)
        lab = W.QLabel("FINISH SKETCH"); lab.setStyleSheet("color:#2b8a35;font-size:11px;font-weight:bold;padding:3px 4px"); fv.addWidget(lab)
        row.addWidget(fbox); row.addStretch(1)
        return rib

    def _fill_menu(s, m, items):
        A = s.sk_acts
        for it in items:
            if it is None: m.addSeparator()
            elif isinstance(it, str): m.addAction(A[it])
            elif len(it) == 1: m.addAction(A[it[0]])
            else:
                sub = m.addMenu(A[it[1][0]].icon(), it[0])
                for k in it[1]: sub.addAction(A[k])

    # ---- mode switching ----
    def on_sk_mode(s, on):
        s.rib_stack.setCurrentIndex(1 if on else s.TAB_PAGE.get(getattr(s, "cur_tab", "SOLID"), 0)); s.palette.setVisible(on)
        if on: s.palette.sync(); s.place_palette(); s.palette.raise_()
        s.sync_sk_tools(); s.refresh()

    def place_palette(s):
        if hasattr(s, "palette"): s.palette.adjustSize(); s.palette.move(s.vp.width() - s.palette.width() - 10, 150)

    def sync_sk_tools(s):
        v = s.vp
        for k, a in s.sk_acts.items():
            if a.isCheckable(): a.setChecked(bool(v.skedit) and k == v.tool)
        if hasattr(s, "palette") and s.palette.isVisible(): s.palette.sync()

    def sk_tool(s, tool):
        v = s.vp
        if v.cmd: s.cmd_cancel()
        if not v.skedit:
            if s.busy(): return
            v.pending_tool = tool; s.set_tool("pick"); return
        v.sk_set_tool(None if v.tool == tool and not tool.startswith("con:") else tool); s.sync_sk_tools(); v.setFocus()

    def leave_sketch(s):
        if s.vp.skedit: s.vp.sk_finish()

    # ---- dimensions ----
    def on_dim_place(s, k):
        v = s.vp; val = geo_measure(v.esk.geo, k)
        txt = f"{fmt(abs(val), 4)}" if k["t"] == "ang" else fmt_expr(abs(val))
        r = expr_dialog(s, "Sketch Dimension", txt, "ang" if k["t"] == "ang" else "len", v.dim_env(), CON_NAMES[k["t"]])
        if r: v.sk_add_dim(k, r[0])
        v.update()

    def on_dim_edit(s, ki):
        v = s.vp; k = v.esk.geo.K[ki]
        r = expr_dialog(s, f"Edit dimension {k.get('n', '')}", k.get("x") or (fmt(geo_measure(v.esk.geo, k), 4) if k["t"] == "ang" else fmt_expr(geo_measure(v.esk.geo, k))),
                        "ang" if k["t"] == "ang" else "len", {n: x for n, x in v.dim_env().items() if n != k.get("n")}, k.get("n"), bool(k.get("drv")))
        if r: v.sk_set_dim(ki, r[0], r[1])
        v.update()

    # ---- text ----
    def on_text_dialog(s, info):
        v = s.vp; g = v.esk.geo; c = g.C[info["edit"]] if "edit" in info else {}
        dlg = W.QDialog(s); dlg.setWindowTitle("Text"); f = W.QFormLayout(dlg)
        ed = W.QLineEdit(c.get("text", "Text")); font = W.QFontComboBox(); font.setCurrentFont(G.QFont(c.get("font", "Sans Serif")))
        size = LengthSpin(0.001); size.setValue(c.get("size", 5.0)); bold = W.QCheckBox("Bold"); bold.setChecked(c.get("bold", False))
        ital = W.QCheckBox("Italic"); ital.setChecked(c.get("italic", False)); ang = W.QDoubleSpinBox(); ang.setRange(-360, 360); ang.setSuffix(" °"); ang.setValue(c.get("ang", 0.0))
        path = info.get("path", c.get("path")); flip = W.QCheckBox("Flip to the other side"); flip.setChecked(c.get("flip", False))
        pos = W.QDoubleSpinBox(); pos.setRange(0, 1); pos.setSingleStep(0.05); pos.setValue(c.get("pos", 0.0))
        al = W.QComboBox(); al.addItem("From the start", "start"); al.addItem("Centred", "center"); al.setCurrentIndex(1 if c.get("align") == "center" else 0)
        for lab, w_ in (("Text", ed), ("Font", font), ("Height", size), ("", bold), ("", ital)): f.addRow(lab, w_)
        if path is None: f.addRow("Angle", ang)
        else: f.addRow("Along curve", W.QLabel(f"Curve {path + 1}")); f.addRow("Position", pos); f.addRow("Alignment", al); f.addRow("", flip)
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel); bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); f.addRow(bb)
        if not dlg.exec() or not ed.text().strip(): return
        vals = dict(text=ed.text(), font=font.currentFont().family(), size=size.value(), bold=bold.isChecked(), italic=ital.isChecked(), ang=ang.value())
        if path is not None: vals.update(path=path, flip=flip.isChecked(), pos=pos.value(), align=al.currentData())
        if "edit" in info:
            def fn(g2): g2.C[info["edit"]].update(vals)
            v.sk_apply(fn, solve=False)
        elif path is not None:
            def fn(g2):
                cc = g2.C[path]; i = (crv_ends(cc)[0] if crv_ends(cc)[0] is not None else cc["p"][0])
                vals2 = dict(vals); vals2.pop("path"); return build_text(g2, (g2.P[i][0], g2.P[i][1], ("pt", i)), path=path, **{k: vals2[k] for k in ("text", "size", "font", "bold", "italic", "ang", "flip", "pos")})
            ci = v.sk_apply(fn, solve=False)
            if ci is not None: v.esk.geo.C[ci]["align"] = vals["align"]; v.esk.geo.touch()
        else:
            at = info["at"]; v.sk_apply(lambda g2: build_text(g2, v.sk_prep(g2, at), **vals), solve=False)
        v.sk_set_tool(None); s.sync_sk_tools()

    # ---- sketch fillet / chamfer ----
    def on_corner_dialog(s, tool, lines):
        v = s.vp
        if tool == "fillet":
            r = expr_dialog(s, "Sketch Fillet", fmt_expr(getattr(s, "_sk_fr", 2.0)), "len", v.dim_env(), "Radius")
            if not r: return
            try: rad = eval_expr(r[0], v.dim_env(), "len")
            except ExprError as ex: return v.sk_flash(str(ex))
            s._sk_fr = rad
            def fn(g):
                ai = geo_fillet(g, lines[0], lines[1], rad); k = dict(t="rad", e=[C_(ai)], v=rad, x=r[0], n=v.next_dim_name(g))
                k["pos"] = v.dim_default_pos(g, k); g.K.append(k); return ai
            v.sk_apply(fn, msg="Fillet added - its radius is a dimension you can change.")
        else:
            res = form_dialog(s, "Sketch Chamfer", [("mode", "Type", "combo", "eq", (("Equal distance", "eq"), ("Two distances", "dd"), ("Distance and angle", "da"))),
                                                    ("d", "Distance", "pos", 2.0), ("d2", "Second distance", "pos", 3.0), ("a", "Angle", "deg", 45.0)])
            if not res: return
            v.sk_apply(lambda g: geo_chamfer(g, lines[0], lines[1], res["d"], res["d2"] if res["mode"] == "dd" else None,
                                             res["a"] if res["mode"] == "da" else None))

    # ---- typing sizes while drawing (Tab) ----
    def on_type_dialog(s, tool):
        v = s.vp; tp = list(v.tp); cur = v.sk_cursor
        if not tp: return
        a = tp[0]; ax, ay = xy(a); cx, cy = (cur[0], cur[1]) if cur else (ax + 10, ay + 10)
        name = lambda g: v.next_dim_name(g)
        typed = v.type_values; v.type_values = None
        ask = (lambda *a_, **k_: typed) if typed is not None else form_dialog
        def dim(g, t, e, val, **kw):
            k = dict(t=t, e=e, v=val, x=fmt_expr(val) if t != "ang" else fmt(val, 4), n=name(g)); k.update(kw); k["pos"] = v.dim_default_pos(g, k); g.K.append(k)
        if tool == "line":
            last = tp[-1]; lx, ly = xy(last); L0 = math.hypot(cx - lx, cy - ly); a0 = math.degrees(math.atan2(cy - ly, cx - lx))
            r = ask(s, "Line", [("l", "Length", "pos", L0 or 10.0), ("a", "Angle", "deg", a0)], "Angle from the sketch's X axis.")
            if not r or r["l"] <= 0: return
            t = math.radians(r["a"]); end = (lx + math.cos(t)*r["l"], ly + math.sin(t)*r["l"], None)
            prev = v.chain.get("prev") if v.chain else None
            def fn(g):
                ci = build_line(g, v.sk_prep(g, last), end)
                if ci is not None: dim(g, "dist", [list(P_(g.C[ci]["p"][0])), list(P_(g.C[ci]["p"][1]))], r["l"])
                return ci
            ci = v.sk_apply(fn)
            if ci is not None:
                g = v.esk.geo; b_i = g.C[ci]["p"][1]; st0 = v.chain["start"] if v.chain and v.chain.get("start") is not None else g.C[ci]["p"][0]
                v.chain = dict(start=st0, end=b_i, prev=ci); v.tp = [(g.P[b_i][0], g.P[b_i][1], ("pt", b_i))]
            return
        if tool in ("rect2", "rectc"):
            w0, h0 = abs(cx - ax)*(2 if tool == "rectc" else 1), abs(cy - ay)*(2 if tool == "rectc" else 1)
            r = ask(s, "Rectangle", [("w", "Width", "pos", w0 or 10.0), ("h", "Height", "pos", h0 or 10.0)])
            if not r or r["w"] <= 0 or r["h"] <= 0: return
            sx, sy_ = (1 if cx >= ax else -1), (1 if cy >= ay else -1)
            def fn(g):
                if tool == "rect2": L = build_rect2(g, v.sk_prep(g, a), (ax + sx*r["w"], ay + sy_*r["h"], None))
                else: L = build_rectc(g, v.sk_prep(g, a), (ax + r["w"]/2, ay + r["h"]/2, None))
                p0, p1 = g.C[L[0]]["p"]; p2 = g.C[L[1]]["p"][1]
                dim(g, "hdist", [P_(p0), P_(p1)], r["w"], sg=1 if g.P[p1][0] >= g.P[p0][0] else -1)
                dim(g, "vdist", [P_(p1), P_(p2)], r["h"], sg=1 if g.P[p2][1] >= g.P[p1][1] else -1)
            v.sk_apply(fn); v.tp = []; return
        if tool == "circle":
            r = ask(s, "Circle", [("r", "Radius", "pos", math.hypot(cx - ax, cy - ay) or 5.0)])
            if not r or r["r"] <= 0: return
            def fn(g):
                ci = build_circle(g, v.sk_prep(g, a), r["r"]); dim(g, "rad", [C_(ci)], r["r"])
            v.sk_apply(fn); v.tp = []; return
        if tool in ("poly_insc", "poly_circ"):
            r = ask(s, "Polygon", [("d", "Diameter", "pos", 2*math.hypot(cx - ax, cy - ay) or 10.0), ("n", "Sides", "int", v.sk_opts["sides"], (3, 200))])
            if not r or r["d"] <= 0: return
            v.sk_opts["sides"] = r["n"]; t = math.atan2(cy - ay, cx - ax)
            def fn(g):
                L = build_polygon(g, v.sk_prep(g, a), (ax + math.cos(t)*r["d"]/2, ay + math.sin(t)*r["d"]/2, None), r["n"], tool[5:])
                dim(g, "dia", [C_(L[-1])], r["d"])
            v.sk_apply(fn); v.tp = []; return
        if tool in ("slot_cc", "slot_overall", "slot_center") and len(tp) >= 1:
            L0 = math.hypot(cx - ax, cy - ay)
            r = ask(s, "Slot", [("l", "Length (centre to centre)" if tool != "slot_overall" else "Overall length", "pos", L0 or 20.0), ("w", "Width", "pos", 6.0)])
            if not r or r["l"] <= 0 or r["w"] <= 0: return
            t = math.atan2(cy - ay, cx - ax); L = r["l"]/(2 if tool == "slot_center" else 1); b = (ax + math.cos(t)*L, ay + math.sin(t)*L, None)
            def fn(g):
                out = {"slot_cc": build_slot_cc, "slot_overall": build_slot_overall, "slot_center": build_slot_center}[tool](g, v.sk_prep(g, a), b, r["w"])
                if out:
                    ca, cb = g.C[out[-1]]["p"]; dim(g, "dist", [P_(ca), P_(cb)], math.hypot(g.P[cb][0] - g.P[ca][0], g.P[cb][1] - g.P[ca][1]))
                    dim(g, "dia", [C_(out[0])], r["w"])
            v.sk_apply(fn); v.tp = []; return
        if tool == "ellipse" and len(tp) >= 1:
            r = ask(s, "Ellipse", [("a", "Major radius", "pos", math.hypot(cx - ax, cy - ay) or 10.0), ("b", "Minor radius", "pos", 5.0), ("t", "Angle", "deg", 0.0)])
            if not r or r["a"] <= 0 or r["b"] <= 0: return
            t = math.radians(r["t"])
            def fn(g):
                ci = build_ellipse(g, v.sk_prep(g, a), (ax + math.cos(t)*r["a"], ay + math.sin(t)*r["a"], None), (ax - math.sin(t)*r["b"], ay + math.cos(t)*r["b"], None))
                dim(g, "erad", [C_(ci)], r["a"], ax=1); dim(g, "erad", [C_(ci)], r["b"], ax=2)
            v.sk_apply(fn); v.tp = []; return
        v.sk_flash("Type exact sizes with Tab for lines, rectangles, circles, polygons, slots and ellipses - or add a dimension (D) afterwards.")

    # ---- parameters ----
    def params_dialog(s):
        v = s.vp; s.leave_sketch(); dlg = ParamsDialog(s)
        if not dlg.exec(): return
        try: users, dims = dlg.result()
        except ExprError as ex: return W.QMessageBox.warning(s, "Parameters", str(ex))
        env = {}
        for _ in range(len(users) + 1):                                  # parameters may use each other
            for nm, p in users.items():
                try: p["v"] = eval_expr(p["x"], env, "len"); env[nm] = p["v"]
                except ExprError: pass
        for nm, p in users.items():
            if "v" not in p: return W.QMessageBox.warning(s, "Parameters", f"{nm}: can't evaluate '{p['x']}'")
        changed_dims = []
        for (kind, si, ki), expr in dims:
            if v.sketches[si].geo.K[ki].get("x", "") != expr: changed_dims.append((si, ki, expr))
        params_changed = users != {k: {"x": p.get("x"), "c": p.get("c", ""), "v": p.get("v")} for k, p in v.params.items()}
        v.params = users
        if not changed_dims and not params_changed: return
        idx = {}
        n = -1
        for i, stt in enumerate(v.states):
            op = stt.get("op") or {}
            if op.get("t") in ("sk2", "sketch"): n += 1; idx[n] = i
        first = None
        for si, ki, expr in changed_dims:
            i = idx.get(si)
            if i is None or v.states[i]["op"]["t"] != "sk2": continue
            op = dict(v.states[i]["op"]); geo = copy.deepcopy(op["geo"]); geo["con"][ki]["x"] = expr; op["geo"] = geo
            v.states[i] = dict(v.states[i], op=op); first = i if first is None else min(first, i)
        if params_changed: first = 1
        if first is None or len(v.states) < 2: s.refresh(); return
        try:
            W.QApplication.setOverrideCursor(C.Qt.WaitCursor); back = v.pos; v.replay_edit(first, v.states[first]["op"], strict=False); v.restore(back)
            s.statusBar().showMessage("Parameters applied - the design was recomputed.", 6000)
        except Exception as ex: W.QMessageBox.warning(s, "Parameters", str(ex))
        finally: W.QApplication.restoreOverrideCursor(); s.refresh(); v.update()

    # ---- inserting files ----
    def insert_file(s, kind):
        v = s.vp
        path, _ = W.QFileDialog.getOpenFileName(s, f"Insert {kind.upper()}", "", "DXF drawing (*.dxf)" if kind == "dxf" else "SVG image (*.svg)")
        if not path: return
        try: other = dxf_read(path) if kind == "dxf" else svg_read(path)
        except Exception as ex: traceback.print_exc(); return W.QMessageBox.warning(s, "Insert", f"Couldn't read the file:\n{ex}")
        if not other.C: return W.QMessageBox.information(s, "Insert", "No supported geometry found in that file.")
        if not v.skedit:
            if s.busy(): return
            v.sk_begin_new({"plane": XY})
        v.sk_apply(lambda g: geo_merge(g, other), msg=f"Inserted {len(other.C)} curves from {os.path.basename(path)}.", solve=False)
        v.fit()

    def export_sketch_dxf(s, si=None):
        v = s.vp; sk = v.esk if si is None else v.sketches[si]
        if sk is None: return
        path, _ = W.QFileDialog.getSaveFileName(s, "Save sketch as DXF", "sketch.dxf", "DXF drawing (*.dxf)")
        if not path: return
        if not path.lower().endswith(".dxf"): path += ".dxf"
        try: write_sketch_dxf(path, sk.geo); s.statusBar().showMessage(f"Saved {path}", 6000)
        except Exception as ex: W.QMessageBox.warning(s, "Save DXF", str(ex))

    # ---- canvases & decals ----
    def canvas_form(s, c, title, calib=False):
        dlg = W.QDialog(s); dlg.setWindowTitle(title); f = W.QFormLayout(dlg)
        wsp = LengthSpin(0.001); wsp.setValue(c.get("w", 100.0)); xs = LengthSpin(); xs.setValue(c.get("x", 0.0)); ys = LengthSpin(); ys.setValue(c.get("y", 0.0))
        rot = W.QDoubleSpinBox(); rot.setRange(-360, 360); rot.setSuffix(" °"); rot.setValue(c.get("rot", 0.0))
        op = W.QSpinBox(); op.setRange(5, 100); op.setSuffix(" %"); op.setValue(int(round(c.get("op", 0.6)*100)))
        fx = W.QCheckBox("Flip horizontally"); fx.setChecked(c.get("flipx", False)); fy = W.QCheckBox("Flip vertically"); fy.setChecked(c.get("flipy", False))
        for lab, w_ in (("Width", wsp), ("Centre X", xs), ("Centre Y", ys), ("Rotation", rot), ("Opacity", op), ("", fx), ("", fy)): f.addRow(lab, w_)
        res = {"calib": False}
        if calib:
            cb = W.QPushButton("Calibrate (click two points of known distance)..."); f.addRow(cb)
            cb.clicked.connect(lambda: (res.__setitem__("calib", True), dlg.accept()))
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel); bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); f.addRow(bb)
        if not dlg.exec(): return None
        out = dict(c); out.update(w=wsp.value(), x=xs.value(), y=ys.value(), rot=rot.value(), op=op.value()/100, flipx=fx.isChecked(), flipy=fy.isChecked())
        return out, res["calib"]

    def _read_image(s):
        path, _ = W.QFileDialog.getOpenFileName(s, "Choose an image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp)")
        if not path: return None, None
        img = G.QImage(path)
        if img.isNull(): W.QMessageBox.warning(s, "Image", "That image couldn't be opened."); return None, None
        if max(img.width(), img.height()) > 2048: img = img.scaled(2048, 2048, C.Qt.KeepAspectRatio, C.Qt.SmoothTransformation)
        ba = C.QByteArray(); buf = C.QBuffer(ba); buf.open(C.QIODevice.WriteOnly); img.save(buf, "PNG")
        import base64
        return base64.b64encode(bytes(ba)).decode(), os.path.basename(path)

    def insert_canvas(s):
        v = s.vp; s.leave_sketch()
        if s.busy(): return
        how = None
        if v.sel_face and v.bodies[v.sel_face[0]].face_normal(v.sel_face[1]) is not None: how = {"face": tuple(v.sel_face[:2])}
        else:
            r = form_dialog(s, "Canvas", [("pl", "Plane", "combo", "XY", (("Ground (XY)", "XY"), ("Front (XZ)", "XZ"), ("Right (YZ)", "YZ")) +
                                           tuple((c["name"], ("cplane", i)) for i, c in enumerate(v.cons) if c["kind"] == "plane"))],
                            "Tip: select a flat face first to put the canvas on it.")
            if not r: return
            how = {"plane": {"XY": XY, "XZ": XZ, "YZ": YZ}[r["pl"]]} if isinstance(r["pl"], str) else {"pref": r["pl"]}
        data, name = s._read_image()
        if not data: return
        c = dict(t="canvas", img=data, name=name, w=100.0, x=0.0, y=0.0, rot=0.0, op=0.6, icon="canvas"); c.update(how)
        if how.get("face"):
            bi, fid = how["face"]; pl, _ = v.op_plane({"face": how["face"]}); q = pl.l(v.bodies[bi].face_point(fid)); c.update(x=q.x, y=q.y)
        r = s.canvas_form(c, "Canvas")
        if not r: return
        s.run("Canvas", r[0])

    def insert_decal(s):
        v = s.vp; s.leave_sketch()
        if s.busy(): return
        if not v.sel_face or v.bodies[v.sel_face[0]].face_normal(v.sel_face[1]) is None:
            return W.QMessageBox.information(s, "Decal", "Click a flat face of a body first, then choose Decal.")
        data, name = s._read_image()
        if not data: return
        bi, fid = v.sel_face[:2]; pl, _ = v.op_plane({"face": (bi, fid)}); q = pl.l(v.bodies[bi].face_point(fid))
        f = v.bodies[bi].faces[fid]; lo, hi = bbox(f); w = max(min((hi - lo).Length*0.5, 200.0), 5.0)
        c = dict(t="canvas", decal=True, face=(bi, fid), img=data, name=name, w=w, x=q.x, y=q.y, rot=0.0, op=1.0, icon="decal")
        r = s.canvas_form(c, "Decal")
        if r: s.run("Decal", r[0])

    def edit_canvas(s, i):
        v = s.vp; op = v.states[i]["op"]; r = s.canvas_form(op, "Edit canvas" if not op.get("decal") else "Edit decal", calib=True)
        if not r: return
        new, calib = r
        if calib:
            back = v.pos; v.restore(i); c = v.canv[-1] if v.canv else None
            if c is None: return
            v.calib = dict(i=i, plane=c["plane_obj"], pts=[], back=back, op=new)
            s.statusBar().showMessage("Calibrate: click the first point of a distance you know on the canvas.", 15000); return
        try: W.QApplication.setOverrideCursor(C.Qt.WaitCursor); v.replay_edit(i, new)
        except Exception as ex: W.QMessageBox.warning(s, "Canvas", str(ex))
        finally: W.QApplication.restoreOverrideCursor(); s.refresh()

    def on_calib_done(s, info):
        v = s.vp; (x0, y0), (x1, y1) = info["pts"]; d = math.hypot(x1 - x0, y1 - y0)
        if d < 1e-9: return
        r = form_dialog(s, "Calibrate", [("d", "Real distance between the points", "pos", d)], f"Measured on the canvas: {flen(d)}")
        v.restore(info["back"])
        if not r or r["d"] <= 0: return
        k = r["d"]/d; op = dict(info["op"]); op["w"] = op["w"]*k; op["x"] = x0 + (op["x"] - x0)*k; op["y"] = y0 + (op["y"] - y0)*k
        try: W.QApplication.setOverrideCursor(C.Qt.WaitCursor); v.replay_edit(info["i"], op); v.restore(info["back"])
        except Exception as ex: W.QMessageBox.warning(s, "Calibrate", str(ex))
        finally: W.QApplication.restoreOverrideCursor(); s.refresh()

    # ---- browser ----
    def browser_menu(s, pos):
        it = s.tree.itemAt(pos)
        if not it: return
        ref = it.data(0, C.Qt.UserRole) or ""; v = s.vp; m = W.QMenu(s)
        if ref.startswith("sk:"):
            si = int(ref[3:])
            m.addAction("Edit Sketch", lambda: s.edit_sketch(si)); m.addAction("Rename...", lambda: s.rename_sketch(si))
            m.addAction("Look At", lambda: v.look_at_plane(v.sketches[si].plane.n)); m.addAction("Save as DXF...", lambda: s.export_sketch_dxf(si))
        elif ref.startswith("cv:"):
            ci = int(ref[3:]); i = s.canvas_step(ci)
            if i is not None: m.addAction("Edit / Calibrate...", lambda: s.edit_canvas(i))
        elif not (s.insp_browser_menu(m, ref) or s.asm_browser_menu(m, ref) or s.cam_browser_menu(m, ref)): return
        m.exec(s.tree.viewport().mapToGlobal(pos))

    def canvas_step(s, ci):
        n = -1
        for i, stt in enumerate(s.vp.states):
            if (stt.get("op") or {}).get("t") == "canvas":
                n += 1
                if n == ci: return i
        return None

    def browser_double(s, item, _col=0):
        ref = item.data(0, C.Qt.UserRole) or ""
        if ref.startswith("sk:"): s.edit_sketch(int(ref[3:]))
        elif ref.startswith("cv:"):
            i = s.canvas_step(int(ref[3:]))
            if i is not None: s.edit_canvas(i)

    def edit_sketch(s, si):
        v = s.vp
        if v.cmd: s.cmd_cancel()
        try: v.sk_edit(si)
        except Exception as ex: W.QMessageBox.warning(s, "Edit Sketch", str(ex))
        s.refresh()

    def rename_sketch(s, si):
        v = s.vp; sk = v.sketches[si]
        name, ok = W.QInputDialog.getText(s, "Rename sketch", "Name", text=sk.name or f"Sketch{si+1}")
        if not ok or not name.strip(): return
        n = -1
        for i, stt in enumerate(v.states):
            op = stt.get("op") or {}
            if op.get("t") in ("sk2", "sketch"):
                n += 1
                if n == si: stt["op"] = dict(op, name=name.strip())
            sks = stt.get("sk")
            if sks and si < len(sks): sks[si] = tuple(sks[si][:4]) + (name.strip(),)
        sk.name = name.strip(); s.refresh()

def write_sketch_dxf(path, geo):
    """Sketch curves as a minimal DXF (lines, circles, arcs, ellipses; everything else as polylines)."""
    out = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "4", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    def e(*kv): out.extend(str(x) for x in kv)
    for ci, c in enumerate(geo.C):
        k = c["k"]; P = geo.P; lay = "CONSTRUCTION" if c.get("cons") or c.get("cl") else "0"
        if k == "line": (ax, ay), (bx, by) = P[c["p"][0]], P[c["p"][1]]; e("0", "LINE", "8", lay, "10", ax, "20", ay, "11", bx, "21", by)
        elif k == "circle": cx, cy = P[c["p"][0]]; e("0", "CIRCLE", "8", lay, "10", cx, "20", cy, "40", c["r"])
        elif k == "arc":
            a0, a1, r = arc_angles(c, geo.PT); cx, cy = P[c["p"][0]]; e("0", "ARC", "8", lay, "10", cx, "20", cy, "40", r, "50", math.degrees(a0) % 360, "51", math.degrees(a1) % 360)
        elif k == "point": x, y = P[c["p"][0]]; e("0", "POINT", "8", lay, "10", x, "20", y)
        else:
            for A in geo_curve_pts(geo, ci):
                e("0", "LWPOLYLINE", "8", lay, "90", len(A), "70", 0)
                for x, y in A: e("10", x, "20", y)
    out += ["0", "ENDSEC", "0", "EOF"]
    with open(path, "w") as f: f.write("\n".join(out) + "\n")

def light_palette():
    """Fission's own light palette: the UI is designed light, so a dark desktop theme must not leak into
    dialogs, drop-down lists, menus and command panels (black backgrounds with dark text)."""
    p = G.QPalette()
    for grp in (G.QPalette.Active, G.QPalette.Inactive, G.QPalette.Disabled):
        dis = grp == G.QPalette.Disabled
        for role, col in ((G.QPalette.Window, "#f3f3f3"), (G.QPalette.WindowText, "#9a9a9a" if dis else "#333333"),
                          (G.QPalette.Base, "#ffffff"), (G.QPalette.AlternateBase, "#f5f7f9"), (G.QPalette.Text, "#9a9a9a" if dis else "#222222"),
                          (G.QPalette.PlaceholderText, "#9a9a9a"), (G.QPalette.Button, "#f3f3f3"), (G.QPalette.ButtonText, "#9a9a9a" if dis else "#333333"),
                          (G.QPalette.BrightText, "#ffffff"), (G.QPalette.Light, "#ffffff"), (G.QPalette.Midlight, "#e6e6e6"), (G.QPalette.Mid, "#bdbdbd"),
                          (G.QPalette.Dark, "#9e9e9e"), (G.QPalette.Shadow, "#777777"), (G.QPalette.Highlight, "#d6e9f8"), (G.QPalette.HighlightedText, "#1d2a35"),
                          (G.QPalette.ToolTipBase, "#ffffff"), (G.QPalette.ToolTipText, "#333333"), (G.QPalette.Link, ACCENT)):
            p.setColor(grp, role, G.QColor(col))
    return p

class Fission(CamWin, RenderWin, SheetWin, AsmWin, TimelineWin, MeshWin, InspectWin, SketchWin, SolidUI, W.QMainWindow):
    TABS = ("SOLID", "SURFACE", "MESH", "SHEET METAL", "PLASTIC")
    # Other workspaces are hidden while the solid / surface tools are the focus; put them back in TABS to show them again:
    # TABS = ("SOLID", "SURFACE", "MESH", "SHEET METAL", "PLASTIC", "UTILITIES")
    def __init__(s):
        app = W.QApplication.instance()
        if app is not None:
            if app.style().name().lower() != "fusion": app.setStyle("Fusion")
            app.setPalette(light_palette())
            try: G.QGuiApplication.styleHints().setColorScheme(C.Qt.ColorScheme.Light)
            except Exception: pass
        super().__init__(); s.setWindowTitle("Fission"); s.resize(1400, 880); s.setStyleSheet(style(ui_assets()))
        s.vp = Viewport(); s.vp.changed.connect(s.refresh); s._tl_key = None; s.tools = {}; s._play = C.QTimer(s); s.tl_init()
        s._play.timeout.connect(s.play_step)
        s.vp.msg.connect(lambda m: s.statusBar().showMessage(m, 9000)); s.build_panels()
        for name in ("move", "gear", "thread", "pattern", "edge_op", "extrude", "revolve", "drawing", "compute_all", "delete_step",
                     "export_fcstd", "export_3mf", "export_obj"):
            fn = getattr(s, name)
            setattr(s, name, (lambda fn: lambda *a, **k: (s.leave_sketch(), fn(*a, **k))[1])(fn))

        def act(label, key, fn, kind, tip=None):
            a = G.QAction(icon(kind), label, s)
            if key: a.setShortcut(G.QKeySequence(key)); a.setShortcutVisibleInContextMenu(True)
            a.setToolTip(f"{label}  ({key})" + (f"\n{tip}" if tip else "") if key else label); a.triggered.connect(fn); s.addAction(a); return a
        for label, key, tool, kind in (("Create Sketch", "S", "pick", "sketch"), ("Line", None, "line", "poly"),
                                       ("2-Point Rectangle", None, "rect2", "rect"), ("Center Diameter Circle", None, "circle", "circle")):
            a = act(label, key, lambda _=False, t=tool: s.set_tool(t), kind); a.setCheckable(True); s.tools[tool] = a
        T = s.tools
        def cact(name, key=None, label=None, tip=None):
            c = COMMANDS[name]; return act(label or c.title.title(), key, lambda _=False, n=name: s.open_cmd(n), c.icon, tip)
        ext = cact("extrude", "E", "Extrude", "Profiles, flat faces or (thin) open lines. Distance / to object / all, two sides, taper, thin.")
        rev = cact("revolve", "V", "Revolve"); sweep, loft = cact("sweep", None, "Sweep"), cact("loft", None, "Loft")
        rib, web, emb = cact("rib", None, "Rib"), cact("web", None, "Web"), cact("emboss", None, "Emboss")
        hole = cact("hole", None, "Hole"); thr = cact("thread", None, "Thread  (T)", "Pick cylindrical faces of rods or holes.")
        rod = act("Threaded Rod...", None, s.thread, "thread")
        prims = [cact(n) for n in ("box", "cylinder", "sphere", "torus", "coil", "pipe")]
        gear = act("Spur Gear", "G", s.gear, "gear")
        rpat, pat, ppat = cact("rpattern", None, "Rectangular Pattern"), cact("cpattern", None, "Circular Pattern  (P)"), cact("ppattern", None, "Pattern on Path")
        mir, thk, bfl = cact("mirror", None, "Mirror"), cact("thicken", None, "Thicken"), cact("bfill", None, "Boundary Fill")
        pp = cact("presspull", "Q", "Press Pull"); fil, cha = cact("fillet", "F", "Fillet"), cact("chamfer", "H", "Chamfer")
        shl, dft, scl, cmb = cact("shell", None, "Shell"), cact("draft", None, "Draft"), cact("scale", None, "Scale"), cact("combine", None, "Combine")
        ofs, rpf = cact("offsetface", None, "Offset Face"), cact("replaceface", None, "Replace Face")
        spf, spb = cact("splitface", None, "Split Face"), cact("splitbody", None, "Split Body")
        mov = act("Move / Rotate (drag)", "M", s.move, "move", "Select a body (click it or pick it in the browser) first.")
        mv2, aln = cact("move2", None, "Move / Copy (points, axis)"), cact("align", None, "Align")
        dlf, mat, app_ = cact("defeature", None, "Delete Faces"), cact("material", None, "Physical Material"), cact("appearance", None, "Appearance")
        cmp_ = act("Compute All", None, s.compute_all, "compute", "Recompute every timeline step")
        IA = s.make_inspect_actions(act); s.IA = IA
        MA = s.make_mesh_actions(act, cact); s.MA = MA
        AA = s.make_asm_actions(act, cact); s.AA = AA
        SX = s.make_sheet_actions(act, cact); s.SX = SX
        PLx = {n: cact(n, None, COMMANDS[n].title.title()) for n in PL_OPS}; s.PLx = PLx
        RX = {"render": act("Render...", None, s.render_dialog, "render", "High-resolution picture with lighting, reflections and shadows"),
              "turn": act("Turntable...", None, s.render_dialog, "turntable", "PNG frames of the model spinning"),
              "story": act("Storyboard", None, s.open_story, "explode", "Explode the assembly step by step, play it, export frames")}
        CMx = s.make_cam_actions(act, cact); s.CMx = CMx
        cons = {n[2:]: cact(n, None, COMMANDS[n].title.title()) for n in COMMANDS if n.startswith("c_")}
        drw = act("Drawing  (D)", None, s.drawing, "drawing", "3-view engineering drawing (PDF / SVG / DXF).")
        s.path, s._saved = None, None
        s.a_save = act("Save", "Ctrl+S", s.save, "save"); s.a_saveas = act("Save As...", "Ctrl+Shift+S", s.save_as, "save")
        s.a_open = act("Open...", "Ctrl+O", s.open_design, "file"); s.a_new = act("New design", "Ctrl+N", s.new_design, "doc")
        s.a_undo = act("Undo", "Ctrl+Z", s.undo, "undo"); s.a_redo = act("Redo", "Ctrl+Y", s.redo, "redo")
        s.a_redo.setShortcuts([G.QKeySequence("Ctrl+Y"), G.QKeySequence("Ctrl+Shift+Z")])
        act("Delete", "Delete", s.delete_selected, "delete")

        root = W.QWidget(); root.setObjectName("root"); s.setCentralWidget(root)
        col = W.QVBoxLayout(root); col.setContentsMargins(0, 0, 0, 0); col.setSpacing(0)
        col.addWidget(s.build_topbar())
        s.setup_sketch_ui(); SA = s.sk_acts
        C3 = cons
        create_menu = [T["pick"], None, ext, rev, sweep, loft, rib, web, emb, hole, thr, rod, None] + prims + [gear, None,
                       ("Pattern", [rpat, pat, ppat]), mir, thk, bfl]
        modify_menu = [pp, fil, cha, shl, dft, scl, cmb, None, ofs, rpf, spf, spb, None, mov, mv2, aln, None, dlf, mat, app_, None, SA["params"], cmp_]
        construct_menu = [C3[k] for k in ("plane_offset", "plane_angle", "plane_tangent", "plane_mid", "plane_2edges", "plane_3pts", "plane_tanpt", "plane_path")] + [None] + \
                         [C3[k] for k in ("axis_cyl", "axis_perp_pt", "axis_2planes", "axis_2pts", "axis_edge", "axis_perp_face")] + [None] + \
                         [C3[k] for k in ("point_vertex", "point_2edges", "point_3planes", "point_center", "point_edge_plane", "point_path")]
        s.rib_stack = W.QStackedWidget(); s.rib_stack.setSizePolicy(W.QSizePolicy.Preferred, W.QSizePolicy.Maximum)
        s.rib_stack.addWidget(s.build_ribbon((("CREATE", (T["pick"], ext, rev, sweep, loft, hole, mir, pat), create_menu),
                                      ("SKETCH", (T["line"], T["rect2"], T["circle"]), [T["pick"], None, T["line"], T["rect2"], T["circle"], None, SA["dxf"], SA["svg"]]),
                                      ("MODIFY", (pp, fil, cha, shl, cmb, mov), modify_menu),
                                      ("CONSTRUCT", (C3["plane_offset"], C3["plane_mid"], C3["axis_2pts"]), construct_menu),
                                      ("INSERT", (SA["canvas"], SA["decal"]), [SA["canvas"], SA["decal"], None, SA["dxf"], SA["svg"], None, MA["insert"], MA["cadimport"]]),
                                      s.asm_group(AA),
                                      *s.inspect_groups(IA),
                                      ("DRAWING", (drw,), None))))
        s.rib_stack.addWidget(s.build_sketch_ribbon())
        sx = {n: cact(n, None, COMMANDS[n].title.replace(" (SURFACE)", "").title()) for n in SURF_OPS}
        sx["s_delface"].setText("Delete Face")
        s.surf_acts = sx
        s_create_menu = [T["pick"], None, sx["s_extrude"], sx["s_revolve"], sx["s_sweep"], sx["s_loft"], None, sx["patch"], sx["ruled"], sx["s_offset"],
                         None, thk, bfl, None, mir, ("Pattern", [rpat, pat, ppat])]
        s_modify_menu = [pp, fil, cha, None, sx["trim"], sx["untrim"], sx["s_extend"], sx["stitch"], sx["unstitch"], sx["revnormal"], sx["s_delface"],
                         None, rpf, spf, spb, None, mov, mv2, aln, scl, None, mat, app_, None, SA["params"], cmp_]
        s.rib_stack.addWidget(s.build_ribbon((("CREATE", (T["pick"], sx["s_extrude"], sx["s_revolve"], sx["s_sweep"], sx["s_loft"], sx["patch"], sx["ruled"],
                                                      sx["s_offset"], thk, bfl), s_create_menu),
                                      ("MODIFY", (fil, sx["trim"], sx["untrim"], sx["s_extend"], sx["stitch"], sx["unstitch"], sx["revnormal"], sx["s_delface"]),
                                       s_modify_menu),
                                      ("CONSTRUCT", (C3["plane_offset"], C3["plane_mid"], C3["axis_2pts"]), construct_menu),
                                      ("INSERT", (SA["canvas"], SA["decal"]), [SA["canvas"], SA["decal"], None, SA["dxf"], SA["svg"], None, MA["insert"], MA["cadimport"]]), *s.inspect_groups(IA)), "SURFACE"))
        s.rib_stack.addWidget(s.mesh_ribbon(MA, T, C3, construct_menu, IA, mv2, aln, scl, mat, app_))
        s.rib_stack.addWidget(s.sheet_ribbon(SX, T, C3, construct_menu, IA, fil, mov, mat, app_))
        s.rib_stack.addWidget(s.build_ribbon((("CREATE", (T["pick"], ext, PLx["boss"], PLx["lipgroove"], PLx["snapfit"], rib, web),
                                                [T["pick"], None, ext, rev, None, PLx["boss"], PLx["lipgroove"], PLx["snapfit"], rib, web, hole]),
                                               ("MODIFY", (shl, dft, fil, cmb), [shl, dft, fil, cha, cmb, None, mov, mat, app_]),
                                               ("CONSTRUCT", (C3["plane_offset"], C3["plane_mid"], C3["axis_2pts"]), construct_menu),
                                               *s.inspect_groups(IA)), "PLASTIC"))
        s.rib_stack.addWidget(s.build_ribbon((("SETUP", (mat, app_), [mat, app_]), ("RENDER", (RX["render"], RX["turn"]), [RX["render"], RX["turn"]]),
                                               *s.inspect_groups(IA)), "RENDER", tab_names=("RENDER",), ws="RENDER"))
        s.rib_stack.addWidget(s.build_ribbon((("STORYBOARD", (RX["story"],), [RX["story"]]), ("PUBLISH", (RX["render"],), [RX["render"]])),
                                             "ANIMATION", tab_names=("ANIMATION",), ws="ANIMATION"))
        s.rib_stack.addWidget(s.cam_ribbon(CMx, IA, C3, construct_menu))
        pages = [s.rib_stack.widget(i) for i in range(s.rib_stack.count())]
        for p_ in pages: s.rib_stack.removeWidget(p_)
        for p_ in pages: s.rib_stack.addWidget(RibbonFit(p_))
        s.rib_stack.setSizePolicy(W.QSizePolicy.Ignored, W.QSizePolicy.Fixed)
        s.cur_tab = "SOLID"; col.addWidget(s.rib_stack)
        s.vp.cmd_key.connect(lambda ok: s.cmd_ok() if ok else s.cmd_cancel())
        s.split = W.QSplitter(); s.split.setChildrenCollapsible(False)
        s.split.addWidget(s.build_browser()); s.split.addWidget(s.vp)
        s.palette.raise_(); s.split.setStretchFactor(1, 1); s.split.setSizes([260, 1100]); s.split.setHandleWidth(1)
        col.addWidget(s.split, 1); col.addWidget(s.build_timeline())
        s.build_navbar(); s.refresh(); s.add_command_search(); s.insp_setup()
        s.statusBar().showMessage("Tip: press S and click the ground, the front / right plane or a flat face to start a sketch.")

    # ---- chrome ----
    def tbtn(s, name, ic=None, text=None, size=None, tip=None):
        b = W.QToolButton(); b.setObjectName(name)
        if ic: b.setIcon(icon(ic, size or 40)); b.setIconSize(C.QSize(size or 40, size or 40))
        if text: b.setText(text)
        if tip: b.setToolTip(tip)
        return b

    def build_topbar(s):
        bar = W.QFrame(); bar.setObjectName("topbar"); h = W.QHBoxLayout(bar); h.setContentsMargins(10, 5, 10, 0); h.setSpacing(2); s._topbar_h = h
        fb = s.tbtn("topbtn", "file", size=20, tip="File"); fb.setPopupMode(W.QToolButton.InstantPopup)
        m = W.QMenu(fb)
        for a in (s.a_new, s.a_open, s.a_save, s.a_saveas): m.addAction(a)
        m.addSeparator(); m.addAction(icon("meshimport"), "Insert Mesh (STL, OBJ, 3MF)...", s.insert_mesh)
        m.addAction(icon("cadimport"), "Import CAD File (STEP, IGES, BREP)...", s.import_cad)
        m.addSeparator(); m.addAction("New Drawing (3 views)...", s.drawing); m.addSeparator()
        m.addAction("Export STEP...", lambda: s.export_cad("STEP")); m.addAction("Export IGES...", lambda: s.export_cad("IGES"))
        m.addAction("Export BREP...", lambda: s.export_cad("BREP")); m.addAction("Export STL...", s.export_stl)
        m.addAction("Export FreeCAD (.FCStd)...", s.export_fcstd); m.addAction("Export OBJ...", s.export_obj)
        m.addAction("Export 3MF...", s.export_3mf); m.addSeparator()
        m.addAction("New Window", s.new_window); s.recent_menu = m.addMenu("Open Recent"); m.addAction("Preferences...", s.prefs_dialog)
        m.addSeparator(); m.addAction("Quit", s.close); fb.setMenu(m); h.addWidget(fb)
        b = s.tbtn("topbtn", size=20); b.setDefaultAction(s.a_save); b.setIconSize(C.QSize(20, 20)); b.setToolTip("Save  (Ctrl+S)"); h.addWidget(b)
        s.b_undo = s.tbtn("topbtn", size=20); s.b_undo.setDefaultAction(s.a_undo); s.b_undo.setIconSize(C.QSize(20, 20)); h.addWidget(s.b_undo)
        s.b_redo = s.tbtn("topbtn", size=20); s.b_redo.setDefaultAction(s.a_redo); s.b_redo.setIconSize(C.QSize(20, 20)); h.addWidget(s.b_redo)
        h.addSpacing(10)
        tab = W.QFrame(); tab.setObjectName("doctab"); th = W.QHBoxLayout(tab); th.setContentsMargins(12, 0, 16, 0); th.setSpacing(6)
        ic = W.QLabel(); ic.setPixmap(pix("doc", 18)); th.addWidget(ic); s.doc_label = W.QLabel("Untitled"); th.addWidget(s.doc_label)
        h.addWidget(tab); h.addStretch(1)
        lu = W.QLabel("Units"); lu.setStyleSheet("color:#555;font-size:12px;padding-right:4px"); h.addWidget(lu)
        s.units_box = W.QComboBox(); s.units_box.addItem("Millimetres (mm)", "mm"); s.units_box.addItem("Inches (in)", "in")
        s.units_box.setToolTip("Units shown everywhere. You can always type any unit into a number box: cm, m, in, ft ...")
        s.units_box.currentIndexChanged.connect(lambda _: s.set_units(s.units_box.currentData())); h.addWidget(s.units_box)
        h.addSpacing(6)
        return bar

    # ---- units ----
    def set_units(s, u):
        DISPLAY["unit"] = u
        if s.units_box.currentData() != u: s.units_box.blockSignals(True); s.units_box.setCurrentIndex(s.units_box.findData(u)); s.units_box.blockSignals(False)
        for sp in s.findChildren(LengthSpin): sp.refresh_units()
        s.vp.update(); s.statusBar().showMessage(f"Showing lengths in {'millimetres' if u == 'mm' else 'inches'} - "
                                                 "you can still type any unit (cm, m, in, ft ...) into a number box.", 6000)

    # ---- save / open ----
    def doc_name(s): return os.path.splitext(os.path.basename(s.path))[0] if s.path else "Untitled"
    def dirty(s): return s._saved != (id(s.vp.states[-1]), len(s.vp.states)) and len(s.vp.states) > 1
    def update_title(s):
        if not hasattr(s, "doc_label"): return
        name = s.doc_name() + ("*" if s.dirty() else "")
        s.setWindowTitle(f"{name} - Fission"); s.doc_label.setText(name)
    def save(s, *_):
        return s.write_design(s.path) if s.path else s.save_as()
    def save_as(s, *_):
        path, _ = W.QFileDialog.getSaveFileName(s, "Save design as", s.path or "Untitled.fission", "Fission design (*.fission)")
        if not path: return False
        if not path.lower().endswith(".fission"): path += ".fission"
        return s.write_design(path)
    def write_design(s, path):
        v = s.vp; s.cancel_cmd()
        data = {"app": "Fission", "format": 2, "units": DISPLAY["unit"], "pos": v.pos, "params": v.params,
                "camera": [v.yaw, v.pitch, v.dist, list(v.target.t())],
                "steps": [{"kind": stt["kind"], "op": enc(stt["op"])} for stt in v.states[1:]]}
        data.update(v.insp_save())
        try:
            with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=1)
        except Exception as ex:
            W.QMessageBox.warning(s, "Save failed", str(ex)); return False
        s.path, s._saved = path, (id(v.states[-1]), len(v.states)); s.update_title(); s.add_recent(path)
        s.statusBar().showMessage(f"Saved {path}", 6000); return True
    def maybe_save(s):
        if not s.dirty(): return True
        r = W.QMessageBox.question(s, "Fission", f"Save changes to {s.doc_name()}?",
                                   W.QMessageBox.Save | W.QMessageBox.Discard | W.QMessageBox.Cancel, W.QMessageBox.Save)
        return s.save() if r == W.QMessageBox.Save else r == W.QMessageBox.Discard
    def new_design(s, *_):
        if not s.maybe_save(): return
        s.cancel_cmd(); s.vp.reset(); s.vp.params = {}; s.path = None; s._saved = (id(s.vp.states[-1]), len(s.vp.states)); s.refresh()
    def open_design(s, path=None):
        if not s.maybe_save(): return
        if not isinstance(path, str) or not path:
            path, _ = W.QFileDialog.getOpenFileName(s, "Open design", "", "Fission design (*.fission)")
            if not path: return
        try:
            with open(path, encoding="utf-8") as f: data = json.load(f)
            if data.get("app") != "Fission": raise RuntimeError("not a Fission design file")
            steps = [{"kind": stp["kind"], "op": dec(stp["op"])} for stp in data["steps"]]
        except Exception as ex: return W.QMessageBox.warning(s, "Open failed", str(ex))
        s.cancel_cmd(); s.leave_sketch(); s.set_units(data.get("units", "mm")); s.vp.params = data.get("params", {}) or {}
        W.QApplication.setOverrideCursor(C.Qt.WaitCursor)
        try: fail = s.vp.load_steps(steps, data.get("pos", len(steps)))
        finally: W.QApplication.restoreOverrideCursor()
        cam = data.get("camera")
        if cam: v = s.vp; v.yaw, v.pitch, v.dist, v.target = cam[0], cam[1], cam[2], V(*cam[3])
        s.vp.insp_load(data); s.add_recent(path)
        s.path = path; s._saved = (id(s.vp.states[-1]), len(s.vp.states)) if not fail else None
        s.refresh(); s.vp.update()
        if fail: W.QMessageBox.warning(s, "Open", f"{len(fail)} step(s) could not be rebuilt and are marked red in the timeline:\n" +
                                       "\n".join(f"  step {f[0]} ({f[1]}): {f[2]}" for f in fail[:6]) + "\nThe rest of the design was rebuilt.")
    def closeEvent(s, e):
        if s.maybe_save(): s.autosave_cleanup(); e.accept()
        else: e.ignore()

    def build_ribbon(s, groups, cur="SOLID", tab_names=None, ws="DESIGN"):
        rib = W.QFrame(); rib.setObjectName("ribbon"); h = W.QHBoxLayout(rib); h.setContentsMargins(10, 6, 10, 4); h.setSpacing(12)
        mode = s.tbtn("mode", text=f"{ws}  ▾"); mode.setToolButtonStyle(C.Qt.ToolButtonTextOnly); mode.setPopupMode(W.QToolButton.InstantPopup)
        mm = W.QMenu(mode)
        for wn in ("DESIGN", "RENDER", "ANIMATION", "MANUFACTURE"):
            a = mm.addAction(wn.title()); a.setCheckable(True); a.setChecked(wn == ws); a.triggered.connect(lambda _=False, wn=wn: s.set_workspace(wn))
        mode.setMenu(mm); h.addWidget(mode, 0, C.Qt.AlignTop)
        right = W.QVBoxLayout(); right.setSpacing(0); h.addLayout(right, 1)
        tabs = W.QHBoxLayout(); tabs.setSpacing(0); grp = W.QButtonGroup(rib)
        if not hasattr(s, "tab_btns"): s.tab_btns = {}
        for name in (tab_names or s.TABS):
            b = s.tbtn("tab", text=name); b.setCheckable(True); b.setChecked(name == cur); grp.addButton(b); s.tab_btns.setdefault(name, []).append(b)
            b.clicked.connect(lambda _=False, n=name: s.tab_clicked(n)); tabs.addWidget(b)
        tabs.addStretch(1); right.addLayout(tabs)
        row = W.QHBoxLayout(); row.setSpacing(0); right.addLayout(row)
        for i, (title, acts, menu) in enumerate(groups):
            if i: sep = W.QFrame(); sep.setObjectName("sep"); sep.setFixedWidth(1); row.addWidget(sep)
            box = W.QWidget(); v = W.QVBoxLayout(box); v.setContentsMargins(8, 4, 8, 0); v.setSpacing(0)
            icons = W.QHBoxLayout(); icons.setSpacing(2)
            for a in acts:
                b = s.tbtn("rb"); b.setDefaultAction(a); b.setIconSize(C.QSize(34, 34)); b.setFixedSize(46, 46)
                b.setToolButtonStyle(C.Qt.ToolButtonIconOnly); icons.addWidget(b)
            v.addLayout(icons)
            g = s.tbtn("grp", text=f"{title} ▾"); g.setToolButtonStyle(C.Qt.ToolButtonTextOnly); g.setPopupMode(W.QToolButton.InstantPopup)
            gm = W.QMenu(g)
            for a in (menu or acts):
                if a is None: gm.addSeparator()
                elif isinstance(a, tuple):
                    sub = gm.addMenu(a[0]); [sub.addAction(x) for x in a[1]]
                else: gm.addAction(a)
            g.setMenu(gm); v.addWidget(g, 0, C.Qt.AlignHCenter)
            row.addWidget(box)
        row.addStretch(1)
        return rib

    TAB_PAGE = {"SOLID": 0, "SURFACE": 2, "MESH": 3, "SHEET METAL": 4, "PLASTIC": 5, "RENDER": 6, "ANIMATION": 7, "MANUFACTURE": 8}
    def tab_clicked(s, name):
        if name not in s.TAB_PAGE:
            s.statusBar().showMessage(f"{name.title()} tools aren't available in Fission yet.", 5000); name = getattr(s, "cur_tab", "SOLID")
        s.cur_tab = name
        for n, bs in s.tab_btns.items():
            for b in bs: b.setChecked(n == name)
        if hasattr(s, "sk_ws_tab"): s.sk_ws_tab.setText(name)
        if not s.vp.skedit: s.rib_stack.setCurrentIndex(s.TAB_PAGE[name])
        if name == "MESH": s.statusBar().showMessage("Mesh tools: Insert an STL / OBJ / 3MF (or Tessellate a solid), then repair, cut, "
                                                     "smooth, reduce, section or convert it to a solid.", 8000)
        if name == "SURFACE": s.statusBar().showMessage("Surface tools: open (zero-thickness) bodies - extrude curves, loft, patch, trim, "
                                                        "extend and stitch them; Stitch a closed set into a solid or Thicken it.", 8000)

    def build_browser(s):
        s.left = W.QFrame(); s.left.setObjectName("browser"); s.left.setMinimumWidth(180)
        lv = W.QVBoxLayout(s.left); lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(0)
        hdr = W.QFrame(); hdr.setObjectName("panelhdr"); hh = W.QHBoxLayout(hdr); hh.setContentsMargins(6, 5, 8, 5)
        s.bcol = s.tbtn("bcol", text="«", tip="Collapse browser"); s.bcol.clicked.connect(s.toggle_browser)
        s.btitle = W.QLabel("BROWSER"); hh.addWidget(s.bcol); hh.addWidget(s.btitle); hh.addStretch(1)
        s.tree = W.QTreeWidget(); s.tree.setHeaderHidden(True); s.tree.setIndentation(16); s.tree.setIconSize(C.QSize(16, 16))
        s.tree.itemClicked.connect(s.browser_click); s.tree.itemChanged.connect(s.vis_changed)
        s.tree.itemDoubleClicked.connect(lambda it, c: s.browser_double(it, c)); s.tree.setContextMenuPolicy(C.Qt.CustomContextMenu)
        s.tree.customContextMenuRequested.connect(lambda pos: s.browser_menu(pos))
        lv.addWidget(hdr); lv.addWidget(s.tree, 1)
        return s.left

    def toggle_browser(s):
        on = not s.tree.isVisible(); s.tree.setVisible(on); s.btitle.setVisible(on); s.bcol.setText("«" if on else "»")
        s.left.setMinimumWidth(180 if on else 0); s.left.setMaximumWidth(16777215 if on else 38)
        if on: s.split.setSizes([260, max(1, s.width() - 260)])

    def build_timeline_v6(s):
        bar = W.QFrame(); bar.setObjectName("timeline"); bar.setFixedHeight(48)
        h = W.QHBoxLayout(bar); h.setContentsMargins(10, 4, 10, 4); h.setSpacing(0); s.tlb = {}
        for ic, tip, fn in (("first", "Go to the beginning", lambda: s.goto(0)), ("prev", "Step back (Ctrl+Z)", s.undo),
                            ("play", "Play the history", s.play), ("next", "Step forward (Ctrl+Y)", s.redo),
                            ("last", "Go to the end", lambda: s.goto(len(s.vp.states) - 1))):
            b = s.tbtn("tlbtn", ic, size=18, tip=tip); b.setFixedSize(30, 30); b.clicked.connect(fn); h.addWidget(b); s.tlb[ic] = b
        h.addSpacing(12)
        sa = W.QScrollArea(); sa.setWidgetResizable(True); sa.setFrameShape(W.QFrame.NoFrame)
        sa.setVerticalScrollBarPolicy(C.Qt.ScrollBarAlwaysOff); sa.setStyleSheet("QScrollArea{background:transparent;border:none}")
        sa.viewport().setStyleSheet("background:transparent"); inner = W.QWidget(); s.tl = W.QHBoxLayout(inner); s.tl.setContentsMargins(0, 0, 0, 0); s.tl.setSpacing(3)
        sa.setWidget(inner); h.addWidget(sa, 1); s.tl_scroll = sa
        return bar

    NAMES = dict(sketch="Sketch", sk2="Sketch", skgeo="Sketch edit", canvas="Canvas", decal="Decal", poly="Polygon", line="Line", rect="Rectangle", circle="Circle", extrude="Extrude", revolve="Revolve",
                 fillet="Fillet", chamfer="Chamfer", move="Move", pattern="Circular pattern", gear="Spur gear",
                 thread="Thread", delete="Delete")

    def rebuild_timeline_v6(s):
        v = s.vp; key = (len(v.states), v.pos, id(v.states[-1]))
        if key == s._tl_key: return
        s._tl_key = key
        while s.tl.count():
            w = s.tl.takeAt(0).widget()
            if w: w.deleteLater()
        for i, stt in enumerate(v.states[1:], 1):
            if (stt.get("op") or {}).get("quiet"):                     # renames etc.: undoable, but no icon
                if i == v.pos: mk = W.QFrame(); mk.setObjectName("tlmarker"); mk.setFixedSize(3, 30); s.tl.addWidget(mk)
                continue
            b = TLStep(lambda j=i: s.goto(j), lambda j=i: s.edit_step(j), lambda j=i: s.delete_step(j)); b.setObjectName("tlstep"); b.setFixedSize(30, 32); b.setIconSize(C.QSize(24, 24))
            ic = icon(stt["kind"], 24)
            b.setIcon(ic if i <= v.pos else G.QIcon(ic.pixmap(24, 24, G.QIcon.Disabled)))
            cm = cmd_for_op(stt["op"]); nm = cm.title.title() if cm else s.NAMES.get(stt['kind'], stt['kind'].title())
            if stt["op"] and stt["op"].get("t") in ("sk2", "sketch"):
                nn = sum(1 for x in v.states[1:i] if x["op"] and x["op"].get("t") in ("sk2", "sketch"))
                nm = (stt["op"].get("name") or f"Sketch{nn + 1}") + ("  (editing)" if v.skedit and v.skedit["i"] == i else "")
            b.setToolTip(f"{i}. {nm}" + ("" if i <= v.pos else "  (undone - click to restore)")
                         + "\nDouble-click to change its values · right-click for more")
            s.tl.addWidget(b)
            if i == v.pos:
                mk = W.QFrame(); mk.setObjectName("tlmarker"); mk.setFixedSize(3, 30); s.tl.addWidget(mk)
        if v.pos == 0 and len(v.states) > 1:
            mk = W.QFrame(); mk.setObjectName("tlmarker"); mk.setFixedSize(3, 30); s.tl.insertWidget(0, mk)
        s.tl.addStretch(1)
        C.QTimer.singleShot(0, lambda: s.tl_scroll.horizontalScrollBar().setValue(s.tl_scroll.horizontalScrollBar().maximum()))
        s.a_undo.setEnabled(v.pos > 0 or bool(v.pts)); s.a_redo.setEnabled(v.pos < len(v.states) - 1)
        for k in ("first", "prev"): s.tlb[k].setEnabled(v.pos > 0)
        for k in ("next", "last"): s.tlb[k].setEnabled(v.pos < len(v.states) - 1)
        s.tlb["play"].setEnabled(len(v.states) > 1)

    def build_navbar(s):
        nb = W.QFrame(s.vp); nb.setObjectName("navbar"); h = W.QHBoxLayout(nb); h.setContentsMargins(6, 3, 6, 3); h.setSpacing(1); s.navbtn = {}
        def add(kind, tip, check=False):
            b = s.tbtn("nb", kind, size=22, tip=tip); b.setFixedSize(32, 30); b.setCheckable(check); h.addWidget(b); return b
        for kind, tip in (("orbit", "Orbit - drag with the left button"), ("pan", "Pan - drag with the left button"), ("zoom", "Zoom - drag with the left button")):
            b = add(kind, tip, True); b.clicked.connect(lambda _=False, k=kind: s.set_nav(k)); s.navbtn[kind] = b
        sep = W.QFrame(); sep.setObjectName("sep"); sep.setFixedSize(1, 20); h.addWidget(sep)
        add("fit", "Fit to view").clicked.connect(lambda: s.vp.fit())
        g = add("grid", "Show / hide grid", True); g.setChecked(True)
        g.toggled.connect(lambda on: (setattr(s.vp, "show_grid", on), s.vp.update()))
        d = add("display", "Display settings: visual style, camera, named views, origin"); dm = W.QMenu(d)
        dm.aboutToShow.connect(lambda: s.display_menu(dm)); d.setMenu(dm); d.setPopupMode(W.QToolButton.InstantPopup)
        add("lookat", "Look At the selected face").clicked.connect(lambda: s.vp.look_at_selection() or s.statusBar().showMessage("Select a face first.", 4000))
        nb.adjustSize(); s.vp.navbar = nb

    def set_nav(s, mode):
        v = s.vp; v.nav = None if v.nav == mode else mode
        for k, b in s.navbtn.items(): b.setChecked(k == v.nav)
        v.setCursor(C.Qt.OpenHandCursor) if v.nav else v.unsetCursor(); v.setFocus()

    # ---- history ----
    def busy(s): return s.vp.ext or s.vp.rev or s.vp.mv or s.vp.cmd
    def cancel_cmd(s):
        if s.vp.cmd: s.cmd_cancel()
        elif s.busy(): s.finish_cmd(False)
    def undo(s):
        s.stop_play(); s.cancel_cmd()
        if s.vp.skedit: s.vp.sk_undo()
        else: s.vp.undo()
    def redo(s):
        s.stop_play(); s.cancel_cmd()
        if s.vp.skedit: s.vp.sk_redo()
        else: s.vp.redo()
    def goto(s, i): s.stop_play(); s.cancel_cmd(); s.leave_sketch(); s.vp.restore(i)
    def play(s):
        if s._play.isActive(): s.stop_play(); return
        s.cancel_cmd(); s.leave_sketch(); s.vp.restore(0); s.tlb["play"].setIcon(icon("stop", 18)); s._play.start(450)
    def play_step(s):
        if s.vp.pos >= len(s.vp.states) - 1: s.stop_play()
        else: s.vp.restore(s.vp.pos + 1)
    def stop_play(s):
        if s._play.isActive(): s._play.stop()
        s.tlb["play"].setIcon(icon("play", 18))

    # ---- behaviour ----
    def set_tool(s, t):
        v = s.vp
        if t and (t in SK_TOOLS or t.startswith("con:")): return s.sk_tool(t)
        if s.busy(): return
        if t == "pick" and v.skedit: v.sk_finish()
        if t != "pick": v.pending_tool = None
        v.tool, v.pts, v.hover, v.hover_edge, v.hover_plane = t, [], None, None, None
        if t == "pick" and v.sel_face and v.sketch_on_face(v.sel_face): t = None   # face already selected: go straight in
        if t in DRAW and (not v.active or not v.active.visible): v.new_sketch(XY, look=False)
        if t: v.msg.emit({"pick": "Click the ground, the front / right origin plane or a flat face to sketch on it.",
                          "poly": "Line: click points; click the first point (or Enter) to close. With 2 points, Enter makes an open line (e.g. a revolve axis). Ctrl = no snapping.",
                          "rect": "Rectangle: click two opposite corners, or click the size label / press Tab to type it. Ctrl = no snapping.",
                          "circle": "Circle: click the center, then the edge (or press Tab to type the diameter). Ctrl = no snapping."}.get(t, ""))
        for k, a in s.tools.items(): a.setChecked(k == v.tool)
        v.setFocus(); v.update()

    def refresh(s):
        v, t = s.vp, s.tree
        for k, a in s.tools.items(): a.setChecked(k == v.tool)
        t.blockSignals(True); t.clear()
        def item(parent, text, kind, ref, on, bold=False):
            it = W.QTreeWidgetItem(parent, [text]); it.setIcon(0, icon(kind, 16)); it.setData(0, C.Qt.UserRole, ref)
            it.setFlags(it.flags() | C.Qt.ItemIsUserCheckable); it.setCheckState(0, C.Qt.Checked if on else C.Qt.Unchecked)
            if bold: f = it.font(0); f.setBold(True); it.setFont(0, f)
            return it
        root = item(t, s.doc_name() if hasattr(s, "path") else "Untitled", "doc", "root", any(x.visible for x in v.sketches + v.bodies) or not (v.sketches or v.bodies), True)
        sk = item(root, "Sketches", "folder", "grp:sk", any(x.visible for x in v.sketches) or not v.sketches)
        for i, x in enumerate(v.sketches):
            ed = x is v.esk
            it = item(sk, (x.name or f"Sketch{i+1}") + ("  (editing)" if ed else ""), "sketch", f"sk:{i}", x.visible, ed or (x is v.active and not v.skedit))
            nd = sum(1 for k in x.geo.K if k["t"] in DIM_TYPES)
            it.setToolTip(0, f"{len(x.geo.C)} curve(s), {len(x.geo.K) - nd} constraint(s), {nd} dimension(s)\nDouble-click to edit · right-click for more")
        bd = item(root, "Bodies", "folder", "grp:bd", any(x.visible for x in v.bodies) or not v.bodies)
        for i, x in enumerate(v.bodies):
            sf = getattr(x, "surface", False)
            ms = getattr(x, "mesh", False)
            it = item(bd, getattr(x, "name", None) or f"Body{i+1}", "meshbody" if ms else "surfbody" if sf else "sheetbody" if getattr(x, "sheet", None) else "body",
                      f"bd:{i}", x.visible)
            m = None if sf else x.mass()
            tip = (f"Mesh body · {len(x.tv):,} triangles · volume {fmt(abs(mesh_volume(x.tv))/1000, 3)} cm³" if ms else
                   f"Surface body · area {fmt(surface_area(x.shape)/100, 3)} cm²" if sf else f"Volume {fmt(abs(volume(x.shape))/1000, 3)} cm³") + \
                  (f"\n{x.material}: {fmt(m, 2)} g" if m is not None else "")
            it.setToolTip(0, tip)
            if x.color:
                pm = G.QPixmap(12, 12); pm.fill(G.QColor.fromRgbF(*x.color)); it.setIcon(0, G.QIcon(pm))
            if i == v.sel_body: it.setSelected(True)
        folders = [root, sk, bd]
        if v.canv:
            cvf = item(root, "Canvases", "folder", "grp:cv", any(c.get("visible", True) for c in v.canv)); folders.append(cvf)
            for i, c in enumerate(v.canv): item(cvf, c.get("name") or f"Canvas{i+1}", "decal" if c.get("decal") else "canvas", f"cv:{i}", c.get("visible", True))
        if v.cons:
            cn = item(root, "Construction", "folder", "grp:cn", any(c["visible"] for c in v.cons)); folders.append(cn)
            for i, c in enumerate(v.cons): item(cn, c["name"], {"plane": "cplane", "axis": "caxis", "point": "cpoint"}[c["kind"]], f"cn:{i}", c["visible"])
        s.insp_browser(root, item, folders); s.asm_browser(root, item, folders); s.cam_browser(root, item, folders)
        for it in folders: it.setExpanded(True)
        t.blockSignals(False); s.rebuild_timeline(); s.update_title()

    def browser_click(s, item, _col=0):
        ref = item.data(0, C.Qt.UserRole) or ""; v = s.vp
        if s.insp_click(ref) or s.asm_click(ref): return
        if ref.startswith("sk:"): v.active = v.sketches[int(ref[3:])]; v.sel_body = None
        elif ref.startswith("bd:"):
            v.sel_body = int(ref[3:]); v.sel.clear(); v.sel_face = None
            s.statusBar().showMessage(f"Body{v.sel_body+1} selected - M to move / rotate, P to pattern, Delete to remove.", 6000)
        v.update(); C.QTimer.singleShot(0, s.refresh)

    def vis_changed(s, item, col):
        """Eye toggles in the browser: root / folders toggle everything under them."""
        ref, v = item.data(0, C.Qt.UserRole), s.vp
        if not ref or col: return
        on = item.checkState(0) == C.Qt.Checked
        if s.insp_vis(ref, on) or s.asm_vis(ref, on) or s.cam_vis(ref, on): return
        if ref == "grp:cv" or ref.startswith("cv:"):
            sel = range(len(v.canv)) if ref == "grp:cv" else [int(ref[3:])]
            for i in sel: v.canv[i]["visible"] = on
            v.update(); C.QTimer.singleShot(0, s.refresh); return
        if ref == "grp:cn" or ref.startswith("cn:"):
            sel = range(len(v.cons)) if ref == "grp:cn" else [int(ref[3:])]
            for i in sel: v.cons[i] = dict(v.cons[i], visible=on)
            v.update(); C.QTimer.singleShot(0, s.refresh); return
        grp = {"root": v.sketches + v.bodies, "grp:sk": v.sketches, "grp:bd": v.bodies}.get(ref)
        if grp is None: kind, i = ref.split(":"); grp = [(v.sketches if kind == "sk" else v.bodies)[int(i)]]
        for x in grp: x.visible = on
        v._edge_cache = None; v.update(); C.QTimer.singleShot(0, s.refresh)

    def delete_selected(s):
        v = s.vp
        if v.skedit and not v.cmd:
            if v.sk_sel: v.sk_delete_sel()
            return
        if s.busy(): return
        try:
            if v.sel_body is not None and v.sel_body < len(v.bodies):
                ids = sorted(v.msel | {v.sel_body}, reverse=True); v.sel_body = v.sel_face = None; v.sel.clear(); v.msel = set()
                for bi in ids:
                    if bi < len(v.bodies): v.do({"t": "delete", "bi": bi})
            elif v.active and v.active.sel:
                regs = v.active.regions(); keep = []
                for k, p in v.active.entities:                       # remove the shapes that make up the selected regions
                    try: f = make_face(k, p, v.active.plane)
                    except Exception: keep.append((k, p)); continue
                    if not any(abs(volume_area(f, regs[i].face)) > 1e-9 for i in v.active.sel if i < len(regs)): keep.append((k, p))
                v.set_entities(v.active, keep, "delete")
        except Exception as ex: W.QMessageBox.warning(s, "Delete failed", str(ex))
        v.update()

    def run(s, title, op):
        """Run a recorded operation with a wait cursor; returns True on success."""
        try:
            W.QApplication.setOverrideCursor(C.Qt.WaitCursor); s.vp.do(op); return True
        except Exception as ex:
            traceback.print_exc(); W.QMessageBox.warning(s, f"{title} failed", f"{ex}" or traceback.format_exc()); return False
        finally:
            W.QApplication.restoreOverrideCursor(); s.refresh(); s.vp.update()

    # ---- command panels ----
    def build_panels(s):
        s.p_ext = CmdPanel(s.vp, "EXTRUDE", "Drag the arrow or type a value; negative distances cut. Click profiles in the "
                           "sketch to choose which regions are extruded (none chosen = all).", s.finish_cmd)
        s.spin = s.p_ext.length("Distance", -10000, 10000)
        s.opbox = s.p_ext.combo("Operation", (("Auto (+ join / - cut)", "Auto"), ("New body", "New"), ("Join", "Join"), ("Cut", "Cut")))
        s.spin.valueChanged.connect(s.vp.set_ext)
        s.vp.ext_moved.connect(lambda d: (s.spin.blockSignals(True), s.spin.setValue(d), s.spin.blockSignals(False)))
        s.p_rev = CmdPanel(s.vp, "REVOLVE", "Spins the chosen profiles around the dashed orange axis. Click any straight "
                           "line / edge of the sketch to use it as the axis. The profile must not cross the axis.", s.finish_cmd)
        s.rv_axis = s.p_rev.combo("Axis", (("Sketch Y axis (vertical)", "Y"), ("Sketch X axis (horizontal)", "X")))
        s.rv_angle = s.p_rev.spin("Angle", -360, 360, " °"); s.rv_angle.setValue(360)
        s.rv_op = s.p_rev.combo("Operation", (("Auto (join parent / new)", "Auto"), ("New body", "New"), ("Join", "Join"), ("Cut", "Cut")))
        s._rv_timer = C.QTimer(s); s._rv_timer.setSingleShot(True); s._rv_timer.timeout.connect(s.update_revolve)
        for w in (s.rv_axis, s.rv_op): w.currentIndexChanged.connect(lambda *_: s._rv_timer.start(120))
        s.rv_angle.valueChanged.connect(lambda *_: s._rv_timer.start(120))
        s.p_mov = CmdPanel(s.vp, "MOVE / ROTATE", "Drag an arrow to move, drag a curved arc to rotate (5° steps, Ctrl = free), or type values. Rotation is about the body's centre.", s.finish_cmd)
        s.mv_spins = [s.p_mov.length(f"Move {a}", -100000, 100000) for a in "XYZ"]
        s.rot_spins = [s.p_mov.spin(f"Rotate {a}", -360, 360, " °") for a in "XYZ"]
        s.mv_copy = W.QCheckBox("Create a copy"); s.p_mov.form.addRow("", s.mv_copy)
        for b in s.mv_spins + s.rot_spins: b.valueChanged.connect(s.move_from_spins)
        s.mv_copy.toggled.connect(lambda on: (s.vp.mv.__setitem__("copy", on), s.vp.update()) if s.vp.mv else None)
        s.vp.move_moved.connect(s.spins_from_move)
        s.vp.cmd_done.connect(s.finish_cmd); s.vp.axis_picked.connect(s.axis_picked)

    def show_panel(s, panel):
        for p in (s.p_ext, s.p_rev, s.p_mov): p.setVisible(p is panel)
        if panel: panel.raise_()
        s.vp.setFocus()

    def finish_cmd(s, ok):
        v = s.vp
        if v.ext: s.finish_extrude(ok)
        elif v.rev: s.finish_revolve(ok)
        elif v.mv: s.finish_move(ok)
        s.show_panel(None)

    def profile_sketch(s, verb):
        v = s.vp
        if not (v.active and v.active.entities and v.active.visible):
            v.active = next((k for k in reversed(v.sketches) if k.entities and k.visible), None)
        if not v.active:
            W.QMessageBox.information(s, verb, "Draw a closed profile first (S, then L / R / C)."); return None
        if not v.active.regions():
            W.QMessageBox.information(s, verb, "This sketch has no closed profiles."); return None
        return v.active

    def extrude(s, *_):
        v = s.vp
        if s.busy() or not s.profile_sketch("Extrude"): return
        v.tool, v.pts, v.ext = None, [], {"d": 10.0, "drag": False, "off": 0}
        s.spin.blockSignals(True); s.spin.setValue(10); s.spin.blockSignals(False)
        s.show_panel(s.p_ext); s.refresh(); v.update()

    def finish_extrude(s, ok):
        v = s.vp; d, v.ext, sk = v.ext["d"], None, v.active
        if ok and d != 0 and sk:
            v.confirm_pending()
            s.run("Extrude", {"t": "extrude", "sk": v.sketches.index(sk), "sel": sorted(sk.sel), "d": d, "op": s.opbox.currentData()})
        v.sel_face = None; s.refresh(); v.update()

    def revolve(s, *_):
        v = s.vp
        if s.busy() or not s.profile_sketch("Revolve"): return
        segs = sketch_segments(v.active)
        s.rv_axis.blockSignals(True); s.rv_axis.clear()
        s.rv_axis.addItem("Sketch Y axis (vertical)", "Y"); s.rv_axis.addItem("Sketch X axis (horizontal)", "X")
        for i, (name, a, b) in enumerate(segs): s.rv_axis.addItem(name, i)
        lines = [i for i, (n_, _, _) in enumerate(segs) if n_.startswith("Line ")]
        s.rv_axis.setCurrentIndex(2 + lines[-1] if lines else 0)          # a drawn open line is the usual axis
        s.rv_axis.blockSignals(False)
        v.tool, v.pts = None, []; v.rev = {"axis": s.rv_axis.currentData(), "op": "New", "body": None, "tool": None, "segs": segs, "hot": None}
        s.show_panel(s.p_rev); s.update_revolve()
        s.statusBar().showMessage("Revolve: click any straight line or edge of the sketch to spin around it, or pick an axis in the list.", 9000)

    def axis_picked(s, i): s.rv_axis.setCurrentIndex(2 + i)

    def revolve_tool(s):
        v = s.vp; return v.revolve_shape(v.active, sorted(v.active.sel), s.rv_axis.currentData(), s.rv_angle.value())

    def update_revolve(s):
        v = s.vp
        if not v.rev: return
        v.rev["axis"] = s.rv_axis.currentData(); op = s.rv_op.currentData()
        v.rev["op"] = "Cut" if op == "Cut" else "Join"
        try:
            if abs(s.rv_angle.value()) < 1e-6: raise RuntimeError("Angle is zero")
            tool = s.revolve_tool()
            if abs(volume(tool)) < 1e-9: raise RuntimeError("empty")
            v.rev["tool"], v.rev["body"] = tool, Body(tool)
            s.statusBar().clearMessage()
        except Exception:
            v.rev["tool"] = v.rev["body"] = None
            s.statusBar().showMessage("Can't revolve here - the profile probably touches or crosses the axis.", 6000)
        v.update()

    def finish_revolve(s, ok):
        v = s.vp; r, v.rev, sk = v.rev, None, v.active
        if ok and r and r.get("tool") is not None and sk:
            v.confirm_pending()
            s.run("Revolve", {"t": "revolve", "sk": v.sketches.index(sk), "sel": sorted(sk.sel), "axis": s.rv_axis.currentData(),
                              "angle": s.rv_angle.value(), "op": s.rv_op.currentData()})
        s.refresh(); v.update()

    def selected_body(s, verb):
        v = s.vp
        bi = v.sel_body if v.sel_body is not None else (v.sel_face[0] if v.sel_face else None)
        if bi is None and len(v.bodies) == 1: bi = 0
        if bi is None or bi >= len(v.bodies):
            W.QMessageBox.information(s, verb, "Select a body first - click one of its faces in the viewport or click it in the browser.")
            return None
        return bi

    def move(s, *_):
        v = s.vp
        if s.busy(): return
        bi = s.selected_body("Move")
        if bi is None: return
        b = v.bodies[bi]; c = (b.lo + b.hi)*0.5
        v.tool = None; v.mv = {"bi": bi, "d": V(), "rot": [0.0, 0.0, 0.0], "c": c, "drag": None, "copy": False}
        for sp in s.mv_spins + s.rot_spins: sp.blockSignals(True); sp.setValue(0); sp.blockSignals(False)
        s.mv_copy.blockSignals(True); s.mv_copy.setChecked(False); s.mv_copy.blockSignals(False)
        s.show_panel(s.p_mov); v.update()

    def move_from_spins(s, *_):
        m = s.vp.mv
        if not m: return
        m["d"] = V(*(sp.value() for sp in s.mv_spins)); m["rot"] = [sp.value() for sp in s.rot_spins]; s.vp.update()
    def spins_from_move(s):
        m = s.vp.mv
        for sp, val in zip(s.mv_spins + s.rot_spins, list(m["d"].t()) + list(m["rot"])): sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)

    def finish_move(s, ok):
        v = s.vp; m = v.mv; v.mv = None
        if ok and m and (m["d"].Length > 1e-9 or any(abs(a) > 1e-9 for a in m["rot"])):
            v.sel.clear(); v.sel_face = None
            if s.run("Move", {"t": "move", "bi": m["bi"], "d": m["d"].t(), "rot": list(m["rot"]), "copy": m["copy"]}) and m["copy"]:
                v.sel_body = len(v.bodies) - 1
        s.refresh(); v.update()

    def fillet(s, *_): s.edge_op("Fillet")
    def chamfer(s, *_): s.edge_op("Chamfer")

    def edge_op(s, name):
        v = s.vp
        if s.busy(): return
        groups = {}
        for bi, ei in v.sel: groups.setdefault(bi, []).append(ei)
        if not groups and v.sel_face:                               # a face is selected: use all of its edges
            bi, fid = v.sel_face; groups[bi] = v.bodies[bi].face_edges(fid)
        if not groups: return W.QMessageBox.information(s, name, "Click one or more edges (or a face) first.")
        if name == "Fillet":
            r = form_dialog(s, "Fillet", [("r", "Radius", "pos", 1.0), ("r2", "End radius (0 = same)", "pos", 0.0)],
                            "Give the end a different radius for a variable (tapered) fillet along the edge.")
        else:
            r = form_dialog(s, "Chamfer", [("mode", "Type", "combo", "eq", (("Equal distance", "eq"), ("Distance and angle", "da"), ("Two distances", "dd"))),
                                           ("d", "Distance", "pos", 1.0), ("angle", "Angle", "deg", 45.0), ("d2", "Second distance", "pos", 2.0)],
                            "Angle is measured from the first face; 45° gives a symmetric chamfer.")
        if not r: return
        op = {"t": name.lower(), "groups": sorted((bi, sorted(eis)) for bi, eis in groups.items())}; op.update(r)
        v.sel.clear(); v.sel_face = None; s.run(name, op)

    # ---- circular pattern ----
    def pattern(s, *_):
        v = s.vp
        if s.busy(): return
        items = []
        for i in reversed(range(len(v.features))): items.append((f"Feature: {v.features[i]['name']} #{i+1}", ("feat", i)))
        if v.sel_body is not None and v.sel_body < len(v.bodies): items.insert(0, (f"Body: Body{v.sel_body+1}", ("body", v.sel_body)))
        for i, b in enumerate(v.bodies):
            if i != v.sel_body: items.append((f"Body: Body{i+1}", ("body", i)))
        if not items: return W.QMessageBox.information(s, "Circular Pattern", "Extrude a feature or make a body first (sketch shapes: use Circular Pattern inside the sketch).")
        r = form_dialog(s, "Circular Pattern", [("obj", "Pattern", "combo", items[0][1], items), ("n", "Quantity", "int", 6, (2, 360)),
                                                ("ang", "Total angle", "deg", 360.0), ("cu", "Centre X", "mm", 0.0), ("cv", "Centre Y", "mm", 0.0)],
                        "Copies are spread evenly over the angle around an axis normal to the object's sketch plane "
                        "(the ground plane for bodies) through the centre, in that plane's coordinates.")
        if not r: return
        op = {"t": "pattern", "sk": v.sketches.index(v.active) if v.active in v.sketches else None}; op.update(r)
        if op["obj"][0] == "ent": v.confirm_pending(); op["sk"] = v.sketches.index(v.active)
        s.run("Pattern", op)

    # ---- gears & threads ----
    def gear(s, *_):
        if s.busy(): return
        r = form_dialog(s, "Spur Gear", [("m", "Module", "pos", 1.0), ("z", "Teeth", "int", 20, (6, 400)), ("a", "Pressure angle", "deg", 20.0),
                                         ("t", "Thickness", "pos", 8.0), ("bore", "Bore diameter", "pos", 5.0)],
                        "Pitch diameter = module × teeth. Gears that mesh need the same module and pressure angle; "
                        "place them pitch-diameter-sum / 2 apart (use Move). Bore 0 = solid.")
        if not r: return
        if s.run("Gear", {"t": "gear", "m": r["m"], "z": r["z"], "a": r["a"], "th": r["t"], "bore": r["bore"]}):
            s.statusBar().showMessage(f"Gear: pitch Ø{flen(r['m']*r['z'])}, outside Ø{flen(r['m']*(r['z'] + 2))}.", 9000)

    def thread(s, *_):
        v = s.vp
        if s.busy(): return
        cyl = v.bodies[v.sel_face[0]].face_cylinder(v.sel_face[1]) if v.sel_face else None
        if cyl:
            o, a, xd, R, v0, v1, ext = cyl; d, p = iso_suggest(2*R, ext)
            r = form_dialog(s, "Thread", [("p", "Pitch", "pos", p), ("len", "Length", "pos", v1 - v0), ("off", "Start offset", "pos", 0.0)],
                            f"{'External' if ext else 'Internal'} thread on Ø{fmt(2*R)} mm - nearest ISO coarse size is M{fmt(d)} × {fmt(p)}."
                            + ("" if ext else " (For a standard tapped hole, make the hole the tap-drill size: M - pitch.)"))
            if not r or r["p"] <= 0 or r["len"] <= 0: return
            op = {"t": "thread", "face": tuple(v.sel_face), "p": r["p"], "len": r["len"], "off": r["off"]}
        else:
            r = form_dialog(s, "Threaded Rod", [("d", "Diameter", "pos", 10.0), ("len", "Length", "pos", 30.0), ("p", "Pitch (0 = ISO coarse)", "pos", 0.0)],
                            "Tip: select a cylindrical face first (a boss or a hole) to thread it instead.")
            if not r or r["d"] <= 0 or r["len"] <= 0: return
            op = {"t": "thread", "rod": True, "d": r["d"], "len": r["len"], "p": r["p"]}
        v.sel_face = None; s.run("Thread", op)

    # ---- editing old timeline steps ----
    OPS3 = (("Auto", "Auto"), ("New body", "New"), ("Join", "Join"), ("Cut", "Cut"))
    def edit_step(s, i):
        v = s.vp
        if s.busy() or i <= 0 or i >= len(v.states): return
        op = v.states[i].get("op")
        if op and op["t"] in ("sk2", "sketch", "skgeo", "ent", "entset"):
            if op["t"] in ("sk2", "sketch"): si = sum(1 for x in v.states[1:i] if x["op"] and x["op"].get("t") in ("sk2", "sketch"))
            else: si = op["sk"]
            return s.edit_sketch(si)
        if op and op["t"] == "canvas": return s.edit_canvas(i)
        if op and op["t"] not in ("pattern", "thread", "gear") and cmd_for_op(op) and s.edit_generic(i): return
        t = op and op["t"]; title = f"Edit step {i}: {s.NAMES.get(v.states[i]['kind'], v.states[i]['kind'].title())}"
        F = None; note = "Every later step is recomputed with the new value."
        if t == "extrude":
            F = [("d", "Distance", "mm", op["d"]), ("op", "Operation", "combo", op["op"], s.OPS3)]
        elif t == "revolve":
            sk = Sketch.restore(v.states[i - 1]["sk"][op["sk"]])
            axes = [("Sketch Y axis (vertical)", "Y"), ("Sketch X axis (horizontal)", "X")] + [(n, j) for j, (n, _, _) in enumerate(sketch_segments(sk))]
            F = [("angle", "Angle", "deg", op["angle"]), ("axis", "Axis", "combo", op["axis"], axes), ("op", "Operation", "combo", op["op"], s.OPS3)]
        elif t == "fillet":
            F = [("r", "Radius", "pos", op["r"]), ("r2", "End radius (0 = same)", "pos", op["r2"])]
        elif t == "chamfer":
            F = [("mode", "Type", "combo", op["mode"], (("Equal distance", "eq"), ("Distance and angle", "da"), ("Two distances", "dd"))),
                 ("d", "Distance", "pos", op["d"]), ("angle", "Angle", "deg", op["angle"]), ("d2", "Second distance", "pos", op["d2"])]
        elif t == "move":
            F = [(k, lab, "mm", val) for k, lab, val in zip("xyz", ("Move X", "Move Y", "Move Z"), op["d"])] + \
                [(f"r{k}", f"Rotate {k.upper()}", "deg", val) for k, val in zip("xyz", op["rot"])] + [("copy", "Create a copy", "check", op["copy"])]
        elif t == "pattern":
            F = [("n", "Quantity", "int", op["n"], (2, 360)), ("ang", "Total angle", "deg", op["ang"]), ("cu", "Centre X", "mm", op["cu"]), ("cv", "Centre Y", "mm", op["cv"])]
        elif t == "gear":
            F = [("m", "Module", "pos", op["m"]), ("z", "Teeth", "int", op["z"], (6, 400)), ("a", "Pressure angle", "deg", op["a"]),
                 ("th", "Thickness", "pos", op["th"]), ("bore", "Bore diameter", "pos", op["bore"])]
        elif t == "thread":
            F = ([("d", "Diameter", "pos", op["d"]), ("len", "Length", "pos", op["len"]), ("p", "Pitch (0 = ISO coarse)", "pos", op["p"])] if op.get("rod")
                 else [("p", "Pitch", "pos", op["p"]), ("len", "Length", "pos", op["len"]), ("off", "Start offset", "pos", op["off"])])
        elif t == "ent" and op["ent"][0] in ("rect", "circle"):
            kind, (a, b) = op["ent"]
            if kind == "rect":
                F = [("x", "Corner X", "mm", a.x), ("y", "Corner Y", "mm", a.y), ("w", "Width", "pos", abs(b.x - a.x)), ("h", "Height", "pos", abs(b.y - a.y))]
            else:
                F = [("x", "Centre X", "mm", a.x), ("y", "Centre Y", "mm", a.y), ("dia", "Diameter", "pos", 2*(b - a).Length)]
        if not F:
            return W.QMessageBox.information(s, title, "This step has no parameters to change. Double-click an extrude, revolve, "
                                             "fillet, chamfer, move, pattern, gear, thread, rectangle or circle step.")
        r = form_dialog(s, title, F, note)
        if not r: return
        new = dict(op)
        if t == "move":
            new.update(d=(r["x"], r["y"], r["z"]), rot=[r["rx"], r["ry"], r["rz"]], copy=r["copy"])
        elif t == "ent":
            kind, (a, b) = op["ent"]; a2 = V(r["x"], r["y"], 0)
            if kind == "rect":
                if r["w"] <= 0 or r["h"] <= 0: return
                b2 = V(a2.x + (1 if b.x >= a.x else -1)*r["w"], a2.y + (1 if b.y >= a.y else -1)*r["h"], 0)
            else:
                if r["dia"] <= 0: return
                b2 = a2 + V(r["dia"]/2, 0, 0)
            new["ent"] = (kind, [a2, b2])
        else: new.update(r)
        try:
            W.QApplication.setOverrideCursor(C.Qt.WaitCursor); v.replay_edit(i, new)
            s.statusBar().showMessage(f"Step {i} updated - {len(v.states) - 1 - i} later step(s) recomputed.", 7000)
        except Exception as ex: W.QMessageBox.warning(s, "Couldn't apply the change", str(ex))
        finally: W.QApplication.restoreOverrideCursor(); s.refresh(); v.update()

    # ---- drawing & export ----
    def drawing(s, *_):
        v = s.vp; shapes = [b.shape for b in v.bodies if b.visible and not getattr(b, "mesh", False)]
        if not shapes: return W.QMessageBox.information(s, "Drawing", "Nothing to draw yet - make a body first.")
        doc = v.drawing_doc or new_drawing_doc(s.export_name() if getattr(s, "path", None) else "")
        rows = v.bom_rows() if v.comps else [{"qty": 1, "pn": "", "name": getattr(b, "name", None) or f"Body{i + 1}", "material": b.material or ""}
                                             for i, b in enumerate(v.bodies) if b.visible and not getattr(b, "mesh", False)]
        try:
            xs = None
            if v.storyboard:
                offs = v.explode_offsets(v.storyboard, 1e9)
                xs = [transformed(b.shape, [[1, 0, 0, offs.get(i, (0, 0, 0))[0]], [0, 1, 0, offs.get(i, (0, 0, 0))[1]], [0, 0, 1, offs.get(i, (0, 0, 0))[2]]])
                      for i, b in enumerate(v.bodies) if b.visible and not getattr(b, "mesh", False)]
            dlg = DrawingDialog2(s, shapes, doc, rows, xs); s._drawing_dlg = dlg; dlg.exec()
            v.drawing_doc = doc; s._saved = None if doc else s._saved
        except Exception as ex: traceback.print_exc(); W.QMessageBox.warning(s, "Drawing failed", str(ex))

    def save_path(s, title, name, filt):
        if not s.vp.bodies:
            W.QMessageBox.information(s, title, "Nothing to export yet - extrude a body first."); return ""
        path, _ = W.QFileDialog.getSaveFileName(s, title, name, filt)
        return path + os.path.splitext(name)[1] if path and not os.path.splitext(path)[1] else path
    def export_fcstd(s):
        path = s.save_path("Export FreeCAD", "fission.FCStd", "FreeCAD (*.FCStd)")
        if path: write_fcstd(path, [b.shape for b in s.vp.bodies])
    def export_3mf(s):
        path = s.save_path("Export 3MF", "fission.3mf", "3D Manufacturing Format (*.3mf)")
        if path: write_3mf(path, s.vp.bodies)
    def export_obj(s):
        path = s.save_path("Export OBJ", "fission.obj", "Wavefront OBJ (*.obj)")
        if path: write_obj(path, s.vp.bodies)

def volume_area(f1, f2):
    """Area shared by two coplanar faces (used to find which sketch shapes make up a region)."""
    from OCP.BRepGProp import BRepGProp as BG
    try:
        com = boolean(f1, f2, "common"); p = GProp_GProps(); st(BG, "SurfaceProperties")(com, p); return p.Mass()
    except Exception:
        return 0.0

def selftest(out_path=None):
    """Exercise the kernel, meshing, sketch solver and surface tools without a window (used to check packaged builds)."""
    lines, ok = [], True
    def say(m): lines.append(m)
    try:
        import OCP; say(f"OCP {getattr(OCP, '__version__', '?')}")
        b = BRepPrimAPI_MakeBox(20, 10, 5).Shape(); say(f"box volume {volume(b):.3f}")
        body = Body(b); say(f"mesh {len(body.tv)} triangles, {len(body.eds)} edges")
        f = fillet(b, body.eds[:1], 1.0); say(f"fillet volume {volume(f):.3f}")
        g = Geo(); build_rect2(g, (0, 0, None), (10, 5, None)); geo_solve(g); say(f"sketch {len(g.C)} curves solved")
        w1 = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 10)).Edge()).Wire()
        w2 = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(gp_Pnt(0, 0, 20), gp_Dir(0, 0, 1)), 6)).Edge()).Wire()
        s_ = surf_loft([w1, w2]); caps = [surf_patch(subshapes(w, TopAbs_EDGE)) for w in (w1, w2)]
        sol = sew([s_] + caps); say(f"surface loft + patches stitched: solid volume {abs(volume(sol)):.3f}")
        import tempfile as _tf; td = _tf.mkdtemp(); stp = os.path.join(td, "t.step"); stl = os.path.join(td, "t.stl")
        write_cad_file(stp, [b]); rv = abs(volume(read_cad_file(stp))); say(f"STEP round trip volume {rv:.3f}")
        write_stl(stl, tessellate(b)); mv = abs(mesh_volume(read_stl(stl))); say(f"STL round trip volume {mv:.3f}")
        ok = abs(volume(b) - 1000) < 1e-6 and abs(volume(sol)) > 3000 and abs(rv - 1000) < 1e-3 and abs(mv - 1000) < 1e-3
        import OpenGL.GL; say("PyOpenGL ok")
        app_ = W.QApplication.instance() or W.QApplication(sys.argv[:1])
        win = Fission(); say(f"window built: {win.windowTitle()} with {len(COMMANDS)} commands"); win.close()
    except Exception:
        ok = False; say(traceback.format_exc())
    say("SELFTEST " + ("OK" if ok else "FAILED")); text = "\n".join(lines)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh: fh.write(text + "\n")
    else:
        try: print(text)
        except Exception: pass
    return 0 if ok else 1

class FissionApp(W.QApplication):
    """Catches macOS 'open this .fission file' events from Finder / the Dock."""
    def __init__(s, argv):
        super().__init__(argv); s.win, s.pending = None, []
    def event(s, e):
        if e.type() == C.QEvent.FileOpen and e.file().lower().endswith(".fission"):
            if s.win is not None: s.win.open_design(e.file())
            else: s.pending.append(e.file())
            return True
        return super().event(e)

if __name__ == "__main__" and "--selftest" in sys.argv:
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")     # Windows / macOS always have a desktop session
    i = sys.argv.index("--selftest"); out = sys.argv[i + 1] if len(sys.argv) > i + 1 else None
    _app = W.QApplication(sys.argv[:1]); sys.exit(selftest(out))

if __name__ == "__main__":
    if sys.platform == "win32" and os.path.basename(sys.executable).lower() in ("python.exe", "pythonw.exe"):   # run from source: own taskbar button
        try: import ctypes; ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Fission.CAD")
        except Exception: pass
    fmt_ = G.QSurfaceFormat(); fmt_.setSamples(4); fmt_.setDepthBufferSize(24); fmt_.setStencilBufferSize(8)
    fmt_.setProfile(G.QSurfaceFormat.CompatibilityProfile); G.QSurfaceFormat.setDefaultFormat(fmt_)
    app = FissionApp(sys.argv); app.setStyle("Fusion")
    app.setApplicationName("Fission"); app.setDesktopFileName("fission"); app.setWindowIcon(icon("extrude", 64))
    w = Fission()
    try:
        g = app.primaryScreen().availableGeometry()
        if w.width() > g.width()*0.95 or w.height() > g.height()*0.95: w.setGeometry(g.left() + 10, g.top() + 30, int(g.width()*0.95), int(g.height()*0.90))
    except Exception: pass
    w.show(); app.win = w; C.QTimer.singleShot(600, w.check_recovery)
    files = [a for a in sys.argv[1:] if a.lower().endswith(".fission")] + app.pending
    if files: C.QTimer.singleShot(0, lambda: w.open_design(files[-1]))
    sys.exit(app.exec())
