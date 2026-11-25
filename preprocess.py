"""
preprocess.py
=============
Preprocessing + augmentasi offline ala Roboflow.

Hasilnya adalah "versi dataset" baru di projects/<project>/versions/vN/ dengan
struktur YOLO lengkap (train/valid/test + data.yaml). Dataset asli tidak diubah.

- Preprocessing: diterapkan ke SEMUA subset (resize, grayscale, kontras).
- Augmentasi   : hanya subset train; membuat N salinan acak per gambar.
  Box ikut ditransformasi (flip, rotasi, shear, crop). Box yang terlalu banyak
  terpotong dibuang.

Gambar selalu dibaca dengan EXIF-orientation diterapkan, sama seperti di editor
label, supaya koordinat label cocok.

Tidak bergantung pada tkinter, jadi bisa dijalankan dari thread latar belakang.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from project import SPLITS
from yolo import Box, load_labels, save_labels

MIN_BOX_PX = 2.0  # box lebih kecil dari ini (setelah transformasi) dibuang

RESIZE_MODES = ("off", "stretch", "fit_gray", "fit_black", "fit_white", "fill")
CONTRAST_MODES = ("off", "stretch", "equalize")
FIT_COLORS = {"fit_gray": (114, 114, 114), "fit_black": (0, 0, 0), "fit_white": (255, 255, 255)}

CHOICES = {"resize_mode": RESIZE_MODES, "contrast_mode": CONTRAST_MODES}

DEFAULT_PREPROCESS = {
    "resize_mode": "off",
    "resize_w": 640,
    "resize_h": 640,
    "grayscale": False,
    "contrast_mode": "off",
    "filter_null": False,
}

DEFAULT_AUGMENT = {
    "enabled": False,
    "copies": 3,
    "include_original": True,
    "seed": 42,
    "min_visibility": 0.3,
    "flip_h": True,
    "flip_v": False,
    "rotate90": False,
    "rotation": 10.0,
    "shear": 0.0,
    "crop": 0.0,
    "brightness": 20.0,
    "contrast": 0.0,
    "saturation": 0.0,
    "hue": 0.0,
    "blur": 0.0,
    "noise": 0.0,
    "gray_pct": 0.0,
    "cutout_count": 0,
    "cutout_size": 10.0,
}

# (kunci, batas bawah, batas atas) untuk validasi angka
_AUG_RANGES = (
    ("copies", 1, 20), ("min_visibility", 0, 1),
    ("rotation", 0, 180), ("shear", 0, 40), ("crop", 0, 60),
    ("brightness", 0, 90), ("contrast", 0, 90), ("saturation", 0, 90),
    ("hue", 0, 180), ("blur", 0, 10), ("noise", 0, 25), ("gray_pct", 0, 100),
    ("cutout_count", 0, 10), ("cutout_size", 1, 50),
)

HELP = {
    # --- preprocessing ---
    "resize_mode": "Ubah ukuran SEMUA gambar (train/valid/test). off = biarkan. stretch = tarik paksa "
                   "ke ukuran target (rasio berubah). fit_* = muat utuh dengan bingkai abu/hitam/putih. "
                   "fill = potong bagian tengah sampai penuh (box di tepi bisa terpotong).",
    "resize_w": "Lebar target (piksel). Dipakai jika resize_mode bukan off.",
    "resize_h": "Tinggi target (piksel). Dipakai jika resize_mode bukan off.",
    "grayscale": "Ubah semua gambar jadi hitam-putih. Tetap 3 kanal agar kompatibel dengan model YOLO.",
    "contrast_mode": "Perbaiki kontras. stretch = regangkan histogram (autocontrast), "
                     "equalize = ratakan histogram.",
    "filter_null": "Buang gambar tanpa anotasi dari versi dataset ini.",
    # --- augmentasi ---
    "enabled": "Aktifkan augmentasi. Hanya diterapkan ke subset train; valid dan test tidak pernah diaugmentasi.",
    "copies": "Jumlah salinan augmentasi per gambar train. Tiap salinan memakai parameter acak sendiri.",
    "include_original": "Sertakan juga gambar train asli (hanya preprocessing) di samping salinan augmentasi.",
    "seed": "Seed random. Seed dan pengaturan yang sama menghasilkan versi yang sama.",
    "min_visibility": "Box yang tersisa kurang dari fraksi ini (0-1) dari ukurannya setelah terpotong akan dibuang.",
    "flip_h": "Cermin horizontal, dengan peluang 50% per salinan.",
    "flip_v": "Cermin vertikal, dengan peluang 50% per salinan.",
    "rotate90": "Putar 90/180/270 derajat secara acak (atau tidak diputar). Gambar non-persegi bertukar lebar-tinggi.",
    "rotation": "Rotasi acak antara -N dan +N derajat. 0 = nonaktif. Box menjadi persegi pembatas objek yang diputar, "
                "jadi sedikit lebih longgar.",
    "shear": "Kemiringan (shear) acak horizontal dan vertikal antara -N dan +N derajat. 0 = nonaktif.",
    "crop": "Zoom/crop acak: gambar dipotong hingga N% lebih kecil lalu diperbesar kembali. 0 = nonaktif.",
    "brightness": "Ubah kecerahan acak sebesar -N% sampai +N%. 0 = nonaktif.",
    "contrast": "Ubah kontras acak sebesar -N% sampai +N%. 0 = nonaktif.",
    "saturation": "Ubah saturasi warna acak sebesar -N% sampai +N%. 0 = nonaktif.",
    "hue": "Geser hue (warna) acak antara -N dan +N derajat. 0 = nonaktif.",
    "blur": "Blur Gaussian dengan radius acak 0 sampai N piksel. 0 = nonaktif.",
    "noise": "Noise garam-merica pada hingga N% piksel. 0 = nonaktif.",
    "gray_pct": "Peluang (%) sebuah salinan diubah jadi hitam-putih. 0 = nonaktif.",
    "cutout_count": "Jumlah kotak tutup (cutout) acak per salinan. Box label tidak diubah. 0 = nonaktif.",
    "cutout_size": "Ukuran maksimum kotak cutout, % dari sisi gambar.",
}


# ============================================================
# Konfigurasi
# ============================================================
def coerce_config(raw: dict, defaults: dict) -> dict:
    """Ubah nilai dari form (string) menjadi tipe sesuai default. ValueError bila tidak valid."""
    out = {}
    for key, default in defaults.items():
        value = raw.get(key, default)
        try:
            if isinstance(default, bool):
                out[key] = bool(value)
            elif isinstance(default, int):
                f = float(value)
                if f != int(f):
                    raise ValueError
                out[key] = int(f)
            elif isinstance(default, float):
                out[key] = float(value)
            else:
                out[key] = str(value).strip()
        except (TypeError, ValueError):
            kind = "bilangan bulat" if isinstance(default, int) and not isinstance(default, bool) else "angka"
            raise ValueError(f"{key} harus berupa {kind}.") from None
    return out


def validate_preprocess(p: dict):
    if p["resize_mode"] not in RESIZE_MODES:
        raise ValueError(f"resize_mode harus salah satu dari: {', '.join(RESIZE_MODES)}.")
    if p["contrast_mode"] not in CONTRAST_MODES:
        raise ValueError(f"contrast_mode harus salah satu dari: {', '.join(CONTRAST_MODES)}.")
    if p["resize_mode"] != "off" and (p["resize_w"] < 32 or p["resize_h"] < 32):
        raise ValueError("resize_w dan resize_h minimal 32 piksel.")


def validate_augment(a: dict):
    if not a["enabled"]:
        return
    for key, lo, hi in _AUG_RANGES:
        if not lo <= a[key] <= hi:
            raise ValueError(f"{key} harus di antara {lo} dan {hi}.")


def active_augmentations(a: dict) -> list[str]:
    """Daftar augmentasi yang benar-benar aktif (untuk memberi peringatan jika kosong)."""
    names = [k for k in ("flip_h", "flip_v", "rotate90") if a[k]]
    names += [k for k in ("rotation", "shear", "crop", "brightness", "contrast", "saturation",
                          "hue", "blur", "noise", "gray_pct") if a[k] > 0]
    if a["cutout_count"] > 0:
        names.append("cutout")
    return names


# ============================================================
# Transformasi geometri (gambar + box)
# ============================================================
def _map_box(b: Box, f: Callable[[float, float], tuple[float, float]]) -> Box:
    """Petakan 4 sudut box dengan f(x, y) lalu ambil persegi pembatasnya."""
    pts = [f(b.x1, b.y1), f(b.x2, b.y1), f(b.x2, b.y2), f(b.x1, b.y2)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return Box(b.cls, min(xs), min(ys), max(xs), max(ys))


def _flip_h(img, boxes):
    w = img.width
    return ImageOps.mirror(img), [_map_box(b, lambda x, y: (w - x, y)) for b in boxes]


def _flip_v(img, boxes):
    h = img.height
    return ImageOps.flip(img), [_map_box(b, lambda x, y: (x, h - y)) for b in boxes]


def _rot90(img, boxes, k):
    """k=1: 90 derajat berlawanan arah jarum jam, 2: 180, 3: 270."""
    w, h = img.size
    if k == 1:
        return img.transpose(Image.Transpose.ROTATE_90), [_map_box(b, lambda x, y: (y, w - x)) for b in boxes]
    if k == 2:
        return img.transpose(Image.Transpose.ROTATE_180), [_map_box(b, lambda x, y: (w - x, h - y)) for b in boxes]
    return img.transpose(Image.Transpose.ROTATE_270), [_map_box(b, lambda x, y: (h - y, x)) for b in boxes]


def _crop_zoom(img, boxes, rng, pct):
    """Potong jendela acak seluas (1-pct%..100%) lalu perbesar ke ukuran semula."""
    w, h = img.size
    frac = rng.uniform(1 - pct / 100, 1)
    cw, ch = w * frac, h * frac
    x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
    out = img.resize((w, h), Image.Resampling.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))
    sx, sy = w / cw, h / ch
    return out, [_map_box(b, lambda x, y: ((x - x0) * sx, (y - y0) * sy)) for b in boxes]


def _random_affine(img, boxes, rng, rotation, shear):
    """Rotasi + shear dalam satu resampling, berpusat di tengah gambar."""
    w, h = img.size
    theta = math.radians(rng.uniform(-rotation, rotation)) if rotation > 0 else 0.0
    shx = math.tan(math.radians(rng.uniform(-shear, shear))) if shear > 0 else 0.0
    shy = math.tan(math.radians(rng.uniform(-shear, shear))) if shear > 0 else 0.0
    if theta == 0 and shx == 0 and shy == 0:
        return img, boxes

    center = np.array([w / 2, h / 2])
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    a = np.array([[1, shx], [shy, 1]]) @ rot            # maju: p' = A (p - c) + c
    t = center - a @ center
    inv = np.linalg.inv(a)
    inv_t = -inv @ t
    out = img.transform(
        (w, h), Image.Transform.AFFINE,
        (inv[0, 0], inv[0, 1], inv_t[0], inv[1, 0], inv[1, 1], inv_t[1]),
        resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0),
    )
    fwd = lambda x, y: (a[0, 0] * x + a[0, 1] * y + t[0], a[1, 0] * x + a[1, 1] * y + t[1])
    return out, [_map_box(b, fwd) for b in boxes]


def _finalize(boxes: list[Box], w: int, h: int, min_vis: float) -> list[Box]:
    """Potong box ke batas gambar; buang yang terlalu kecil / terlalu banyak terpotong."""
    out = []
    for b in boxes:
        area = max(b.x2 - b.x1, 0) * max(b.y2 - b.y1, 0)
        x1, x2 = min(max(b.x1, 0), w), min(max(b.x2, 0), w)
        y1, y2 = min(max(b.y1, 0), h), min(max(b.y2, 0), h)
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_BOX_PX or bh < MIN_BOX_PX or area <= 0:
            continue
        if (bw * bh) / area < min_vis:
            continue
        out.append(Box(b.cls, x1, y1, x2, y2))
    return out


# ============================================================
# Transformasi foto (box tidak berubah)
# ============================================================
def _jitter(rng, pct):
    return 1 + rng.uniform(-pct, pct) / 100


def _shift_hue(img, degrees):
    hsv = np.array(img.convert("HSV"))
    shift = int(round(degrees / 360 * 256))
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
    return Image.fromarray(hsv, "HSV").convert("RGB")


def _salt_pepper(img, rng, max_pct):
    frac = rng.uniform(0, max_pct) / 100
    arr = np.array(img)
    gen = np.random.default_rng(rng.getrandbits(32))
    hit = gen.random(arr.shape[:2]) < frac
    salt = gen.random(arr.shape[:2]) < 0.5
    arr[hit & salt] = 255
    arr[hit & ~salt] = 0
    return Image.fromarray(arr)


def _cutout(img, rng, count, size_pct):
    out = img.copy()
    w, h = img.size
    for _ in range(count):
        cw = max(1, int(w * rng.uniform(0.3, 1.0) * size_pct / 100))
        ch = max(1, int(h * rng.uniform(0.3, 1.0) * size_pct / 100))
        x0, y0 = rng.randint(0, max(w - cw, 0)), rng.randint(0, max(h - ch, 0))
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        out.paste(color, (x0, y0, x0 + cw, y0 + ch))
    return out


# ============================================================
# Pipeline
# ============================================================
def _resize(img, boxes, pre, min_vis):
    """Return (img, boxes, geometri_berubah_selain_stretch)."""
    mode = pre["resize_mode"]
    if mode == "off":
        return img, boxes, False
    tw, th = int(pre["resize_w"]), int(pre["resize_h"])
    w, h = img.size
    if (w, h) == (tw, th):
        return img, boxes, False
    if mode == "stretch":
        sx, sy = tw / w, th / h
        out = img.resize((tw, th), Image.Resampling.LANCZOS)
        return out, [Box(b.cls, b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy) for b in boxes], False

    if mode == "fill":
        scale = max(tw / w, th / h)
        nw, nh = max(tw, round(w * scale)), max(th, round(h * scale))
        resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        ox, oy = (nw - tw) // 2, (nh - th) // 2
        out = resized.crop((ox, oy, ox + tw, oy + th))
        sx, sy = nw / w, nh / h
        moved = [Box(b.cls, b.x1 * sx - ox, b.y1 * sy - oy, b.x2 * sx - ox, b.y2 * sy - oy) for b in boxes]
        return out, _finalize(moved, tw, th, min_vis), True

    # fit_*: muat utuh, sisanya diisi warna bingkai
    scale = min(tw / w, th / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (tw, th), FIT_COLORS[mode])
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    out.paste(resized, (ox, oy))
    sx, sy = nw / w, nh / h
    return out, [Box(b.cls, b.x1 * sx + ox, b.y1 * sy + oy, b.x2 * sx + ox, b.y2 * sy + oy)
                 for b in boxes], True


def preprocess(img: Image.Image, boxes: list[Box], pre: dict, min_vis: float = 0.3):
    """Terapkan preprocessing. Return (img, boxes, geometri_berubah).

    geometri_berubah=True artinya baris label non-box (polygon) tidak lagi valid.
    """
    img, boxes, changed = _resize(img, boxes, pre, min_vis)
    if pre["grayscale"]:
        img = img.convert("L").convert("RGB")
    if pre["contrast_mode"] == "stretch":
        img = ImageOps.autocontrast(img, cutoff=1)
    elif pre["contrast_mode"] == "equalize":
        img = ImageOps.equalize(img)
    return img, boxes, changed


def augment(img: Image.Image, boxes: list[Box], aug: dict, rng: random.Random):
    """Satu salinan augmentasi acak. Return (img, boxes)."""
    if aug["flip_h"] and rng.random() < 0.5:
        img, boxes = _flip_h(img, boxes)
    if aug["flip_v"] and rng.random() < 0.5:
        img, boxes = _flip_v(img, boxes)
    if aug["rotate90"]:
        k = rng.choice((0, 1, 2, 3))
        if k:
            img, boxes = _rot90(img, boxes, k)
    if aug["crop"] > 0:
        img, boxes = _crop_zoom(img, boxes, rng, aug["crop"])
    if aug["rotation"] > 0 or aug["shear"] > 0:
        img, boxes = _random_affine(img, boxes, rng, aug["rotation"], aug["shear"])
    boxes = _finalize(boxes, img.width, img.height, aug["min_visibility"])

    if aug["brightness"] > 0:
        img = ImageEnhance.Brightness(img).enhance(_jitter(rng, aug["brightness"]))
    if aug["contrast"] > 0:
        img = ImageEnhance.Contrast(img).enhance(_jitter(rng, aug["contrast"]))
    if aug["saturation"] > 0:
        img = ImageEnhance.Color(img).enhance(_jitter(rng, aug["saturation"]))
    if aug["hue"] > 0:
        img = _shift_hue(img, rng.uniform(-aug["hue"], aug["hue"]))
    if aug["gray_pct"] > 0 and rng.random() < aug["gray_pct"] / 100:
        img = img.convert("L").convert("RGB")
    if aug["blur"] > 0:
        radius = rng.uniform(0, aug["blur"])
        if radius > 0.1:
            img = img.filter(ImageFilter.GaussianBlur(radius))
    if aug["noise"] > 0:
        img = _salt_pepper(img, rng, aug["noise"])
    if aug["cutout_count"] > 0:
        img = _cutout(img, rng, aug["cutout_count"], aug["cutout_size"])
    return img, boxes


def augment_full(base_img, base_boxes, pre, aug, rng):
    """Augmentasi + resize ulang bila ukuran berubah (mis. rotate90 pada gambar non-persegi)."""
    img, boxes = augment(base_img, base_boxes, aug, rng)
    if pre["resize_mode"] != "off" and img.size != (int(pre["resize_w"]), int(pre["resize_h"])):
        img, boxes, _ = _resize(img, boxes, pre, aug["min_visibility"])
    return img, boxes


def make_preview(img, boxes, pre, aug, n=5, seed=0):
    """Contoh hasil untuk satu gambar: [(keterangan, img, boxes), ...]."""
    base_img, base_boxes, _ = preprocess(img, boxes, pre, aug["min_visibility"])
    out = [("Asli + preprocessing", base_img, base_boxes)]
    if aug["enabled"]:
        for k in range(1, n + 1):
            rng = random.Random(f"preview:{seed}:{k}")
            i2, b2 = augment_full(base_img, base_boxes, pre, aug, rng)
            out.append((f"Augmentasi {k}", i2, b2))
    return out


# ============================================================
# Membuat versi dataset
# ============================================================
def _load_oriented(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _save_image(img: Image.Image, path: Path):
    if path.suffix.lower() in (".jpg", ".jpeg"):
        img.save(path, quality=95)
    else:
        img.save(path)


def _write_version_yaml(root: Path, names: list[str]):
    names_s = "[" + ", ".join(repr(n) for n in names) + "]"
    # Tanpa kunci 'path': Ultralytics memakai folder data.yaml sebagai akar.
    root.joinpath("data.yaml").write_text(
        f"train: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(names)}\nnames: {names_s}\n",
        encoding="utf-8",
    )


def build_version(
    project,
    pre: dict,
    aug: dict,
    stop_event: threading.Event | None = None,
    progress: Callable[[int, int, dict], None] | None = None,
) -> dict:
    """Buat versi dataset baru. Dataset asli tidak diubah.

    pre/aug harus sudah lolos coerce_config + validate_*. Jika dibatalkan atau gagal total,
    folder versi yang setengah jadi dihapus.
    """
    items = [(s, p) for s in SPLITS for p in project.list_images(s)]
    if not any(s == "train" for s, _ in items):
        raise ValueError("Subset train kosong. Split dataset terlebih dahulu.")

    names = project.read_names()
    version = project.next_version_name()
    root = project.versions_dir / version
    for subset in SPLITS:
        (root / subset / "images").mkdir(parents=True, exist_ok=True)
        (root / subset / "labels").mkdir(parents=True, exist_ok=True)

    stats = {
        "version": version, "path": str(root), "total": len(items),
        "counts": {s: 0 for s in SPLITS}, "augmented": 0, "filtered_null": 0,
        "polygon_dropped": 0, "aug_skipped_polygon": 0, "aug_skipped_empty": 0,
        "errors": 0, "first_error": None, "stopped": False,
    }
    stop_event = stop_event or threading.Event()
    done = False
    try:
        for n, (subset, path) in enumerate(items, start=1):
            if stop_event.is_set():
                stats["stopped"] = True
                break
            try:
                _process_one(project, root, subset, path, pre, aug, stats)
            except Exception as exc:  # satu gambar rusak tidak boleh menggagalkan semuanya
                stats["errors"] += 1
                stats["first_error"] = stats["first_error"] or f"{path.name}: {exc}"
            if progress:
                progress(n, len(items), stats)
        done = not stats["stopped"]
        if done and sum(stats["counts"].values()) == 0:
            raise ValueError("Tidak ada gambar yang berhasil ditulis."
                             + (f" Error pertama: {stats['first_error']}" if stats["first_error"] else ""))
        if done:
            _write_version_yaml(root, names)
            root.joinpath("config.json").write_text(
                json.dumps({
                    "version": version,
                    "created_at": datetime.now().astimezone().isoformat(),
                    "classes": names, "preprocess": pre, "augment": aug, "stats": stats,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    finally:
        if not done:
            shutil.rmtree(root, ignore_errors=True)
    if not done:
        stats["version"], stats["path"] = None, None
    return stats


def _process_one(project, root, subset, path, pre, aug, stats):
    img = _load_oriented(path)
    w, h = img.size
    boxes, extra = load_labels(project.label_path(subset, path), w, h)
    if pre["filter_null"] and not boxes and not extra:
        stats["filtered_null"] += 1
        return

    min_vis = aug["min_visibility"]
    base_img, base_boxes, geom_changed = preprocess(img, boxes, pre, min_vis)
    base_extra = extra
    if extra and geom_changed:
        stats["polygon_dropped"] += len(extra)
        base_extra = []

    do_aug = subset == "train" and aug["enabled"]
    if do_aug and extra:
        # Label non-box (polygon) tidak bisa ditransformasi: jangan buat salinan tanpa label.
        stats["aug_skipped_polygon"] += 1
        do_aug = False

    img_dir, lbl_dir = root / subset / "images", root / subset / "labels"
    ext = path.suffix.lower()

    def write(stem, out_img, out_boxes, out_extra):
        _save_image(out_img, img_dir / f"{stem}{ext}")
        save_labels(lbl_dir / f"{stem}.txt", out_boxes, out_extra, *out_img.size)
        stats["counts"][subset] += 1

    if not do_aug or aug["include_original"]:
        write(path.stem, base_img, base_boxes, base_extra)

    if do_aug:
        for k in range(1, aug["copies"] + 1):
            rng = random.Random(f"{aug['seed']}:{subset}:{path.name}:{k}")
            out_img, out_boxes = augment_full(base_img, base_boxes, pre, aug, rng)
            if base_boxes and not out_boxes:
                stats["aug_skipped_empty"] += 1  # semua box hilang: jangan jadi negatif palsu
                continue
            write(f"{path.stem}__aug{k}", out_img, out_boxes, [])
            stats["augmented"] += 1
