"""
analysis.py
===========
Health Check dataset (mengikuti Roboflow): jumlah gambar & anotasi, rata-rata ukuran,
missing & null, class balance per subset, dimension insights, histogram jumlah objek
per gambar, dan heatmap lokasi anotasi.

Tanpa tkinter, jadi aman dipanggil dari thread latar belakang.
"""

from __future__ import annotations

import statistics
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from project import ALL_SUBSETS, Project

HEATMAP_GRID = 64  # heatmap = grid 64x64 atas koordinat ternormalisasi
SQUARE_TOLERANCE = 0.05  # |w/h - 1| <= 5% dianggap square
ORIENTATION_TAG = 274  # tag EXIF orientasi
PROGRESS_EVERY = 25  # panggil progress tiap N gambar


@dataclass
class HealthReport:
    generated_at: str = ""
    subset_images: dict[str, int] = field(default_factory=dict)
    num_images: int = 0
    num_annotations: int = 0
    avg_megapixels: float = 0.0
    median_width: int = 0
    median_height: int = 0
    median_ratio_label: str = ""  # mis. "640x640 square"
    missing: int = 0  # gambar tanpa file label
    null: int = 0  # file label ada tapi kosong
    class_names: list[str] = field(default_factory=list)
    # "all" + tiap subset -> jumlah anotasi per kelas (urutan = class_names)
    class_counts: dict[str, list[int]] = field(default_factory=dict)
    unknown_class_annotations: int = 0  # id kelas di luar data.yaml
    size_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    aspect_counts: dict[str, int] = field(default_factory=dict)
    objects_hist: dict[int, int] = field(
        default_factory=dict
    )  # jumlah objek -> jumlah gambar
    heatmap: np.ndarray = field(
        default_factory=lambda: np.zeros((HEATMAP_GRID, HEATMAP_GRID))
    )
    errors: int = 0
    first_error: str | None = None
    stopped: bool = False


# ---------- Pembacaan data ----------
def image_size(path: Path) -> tuple[int, int]:
    """(lebar, tinggi) tanpa memuat piksel; memperhitungkan rotasi EXIF
    (sama dengan editor, yang memakai exif_transpose)."""
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(ORIENTATION_TAG) in (5, 6, 7, 8):
            w, h = h, w
    return w, h


def read_annotations(path: Path) -> list[tuple[int, float, float, float, float]] | None:
    """None jika file label tidak ada; [] jika ada tapi kosong; selain itu daftar
    (kelas, x1, y1, x2, y2) ternormalisasi 0-1. Mendukung box dan polygon."""
    if not path.is_file():
        return None
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(parts[0])
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 4:  # box: cx cy w h
            cx, cy, w, h = vals
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon: pasangan x y
            xs, ys = vals[0::2], vals[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        out.append((cls, *(min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))))
    return out


def aspect_label(w: int, h: int) -> str:
    ratio = w / h
    if abs(ratio - 1) <= SQUARE_TOLERANCE:
        return "square"
    return "wide" if ratio > 1 else "tall"


def _paint(heat: np.ndarray, x1: float, y1: float, x2: float, y2: float):
    g = HEATMAP_GRID
    c1, r1 = min(int(x1 * g), g - 1), min(int(y1 * g), g - 1)
    c2, r2 = min(g, int(np.ceil(x2 * g))), min(g, int(np.ceil(y2 * g)))
    heat[r1 : max(r2, r1 + 1), c1 : max(c2, c1 + 1)] += 1


def balance_status(count: int, column: list[int]) -> str:
    """'missing' | 'under' | 'ok' untuk satu sel Class Balance.
    Aturan saya (ambang Roboflow tidak didokumentasikan): kurang dari 50%
    rata-rata anotasi per kelas pada kolom yang sama = under represented."""
    if count == 0:
        return "missing"
    mean = sum(column) / len(column) if column else 0
    return "under" if count < 0.5 * mean else "ok"


# ---------- Analisis utama ----------
def analyze(
    project: Project,
    progress: Callable[[int, int], None] | None = None,
    stop_event: threading.Event | None = None,
) -> HealthReport:
    names = project.read_names()
    n_cls = len(names)
    items = [(s, p) for s in ALL_SUBSETS for p in project.list_images(s)]

    rep = HealthReport(
        generated_at=datetime.now().strftime("%B %d, %Y at %H:%M"),
        class_names=names,
        subset_images={s: 0 for s in ALL_SUBSETS},
        class_counts={k: [0] * n_cls for k in ("all",) + tuple(ALL_SUBSETS)},
    )
    widths, heights, pixels = [], [], []
    sizes: Counter = Counter()
    aspects: Counter = Counter()
    objects: Counter = Counter()

    for n, (subset, path) in enumerate(items, start=1):
        if stop_event is not None and stop_event.is_set():
            rep.stopped = True
            break
        try:
            w, h = image_size(path)
            anns = read_annotations(project.label_path(subset, path))
        except Exception as exc:  # satu file rusak tidak menghentikan analisis
            rep.errors += 1
            rep.first_error = rep.first_error or f"{path.name}: {exc}"
            continue

        rep.num_images += 1
        rep.subset_images[subset] += 1
        widths.append(w)
        heights.append(h)
        pixels.append(w * h)
        sizes[(w, h)] += 1
        aspects[aspect_label(w, h)] += 1

        if anns is None:
            rep.missing += 1
        elif not anns:
            rep.null += 1
        objects[len(anns or [])] += 1

        for cls, x1, y1, x2, y2 in anns or []:
            rep.num_annotations += 1
            _paint(rep.heatmap, x1, y1, x2, y2)
            if 0 <= cls < n_cls:
                rep.class_counts["all"][cls] += 1
                rep.class_counts[subset][cls] += 1
            else:
                rep.unknown_class_annotations += 1

        if progress and (n % PROGRESS_EVERY == 0 or n == len(items)):
            progress(n, len(items))

    if rep.num_images:
        rep.avg_megapixels = sum(pixels) / len(pixels) / 1e6
        rep.median_width = int(round(statistics.median(widths)))
        rep.median_height = int(round(statistics.median(heights)))
        rep.median_ratio_label = (
            f"{rep.median_width}x{rep.median_height} "
            f"{aspect_label(rep.median_width, rep.median_height)}"
        )
    rep.size_counts = dict(sizes)
    rep.aspect_counts = dict(aspects)
    rep.objects_hist = dict(sorted(objects.items()))
    return rep
