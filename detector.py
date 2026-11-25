"""
detector.py
===========
Inferensi model YOLO untuk fitur Label Assisting dan Auto Detect.

Tidak bergantung pada tkinter, jadi bisa dipanggil dari thread latar belakang.
Ultralytics baru di-import saat model dimuat (import-nya lambat).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
from yolo import Box, save_labels

MIN_BOX_PX = 1.0  # box lebih kecil dari ini (setelah clamp) dibuang


@dataclass
class Detection:
    """Satu hasil deteksi mentah dari model (koordinat piksel gambar asli)."""

    name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


class Detector:
    """Pembungkus model YOLO. Satu model per instance; aman dipakai lintas thread."""

    def __init__(self):
        self.lock = threading.Lock()  # satu inferensi pada satu waktu
        self.model = None
        self.path: Path | None = None
        self.names: dict[int, str] = {}

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self, path: Path):
        from ultralytics import YOLO

        model = YOLO(str(path))
        task = getattr(model, "task", "detect")
        if task != "detect":
            raise ValueError(
                f"Model bertipe '{task}'. Label Assist/Auto Detect hanya mendukung model 'detect'."
            )
        names = {int(k): str(v) for k, v in dict(model.names).items()}
        with self.lock:
            self.model, self.path, self.names = model, Path(path), names

    def predict(
        self,
        image: Image.Image,
        conf: float = 0.25,
        iou: float = 0.7,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[Detection] | None:
        """Deteksi satu gambar PIL (RGB). Return None jika dibatalkan sebelum inferensi."""
        with self.lock:
            if self.model is None:
                raise RuntimeError("Model belum dimuat.")
            if should_cancel is not None and should_cancel():
                return None
            result = self.model.predict(
                source=image, conf=conf, iou=iou, verbose=False
            )[0]
            names = self.names

        if result.boxes is None:
            return []
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        return [
            Detection(
                names.get(int(c), str(int(c))),
                float(s),
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            )
            for (x1, y1, x2, y2), c, s in zip(xyxy, cls, scores)
        ]


# ---------- Pemetaan kelas model -> kelas project ----------
def _key(name: str) -> str:
    return str(name).strip().lower()


def build_lookup(project_classes: list[str]) -> dict[str, int]:
    """nama kelas (huruf kecil) -> id kelas di project."""
    return {_key(n): i for i, n in enumerate(project_classes)}


def missing_classes(
    model_names: dict[int, str], project_classes: list[str]
) -> list[str]:
    """Nama kelas model yang belum ada di project (urutan sesuai id model)."""
    lookup = build_lookup(project_classes)
    return [
        model_names[i]
        for i in sorted(model_names)
        if _key(model_names[i]) not in lookup
    ]


def to_boxes(
    dets: list[Detection], lookup: dict[str, int], img_w: int, img_h: int
) -> list[tuple[Box, float]]:
    """Ubah deteksi jadi (Box, conf) dengan id kelas project. Kelas tak dikenal diabaikan."""
    out = []
    for d in dets:
        cls_id = lookup.get(_key(d.name))
        if cls_id is None:
            continue
        x1, x2 = sorted((min(max(d.x1, 0), img_w), min(max(d.x2, 0), img_w)))
        y1, y2 = sorted((min(max(d.y1, 0), img_h), min(max(d.y2, 0), img_h)))
        if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
            continue
        out.append((Box(cls_id, x1, y1, x2, y2), d.conf))
    return out


def box_iou(a: Box, b: Box) -> float:
    iw = min(a.x2, b.x2) - max(a.x1, b.x1)
    ih = min(a.y2, b.y2) - max(a.y1, b.y1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def drop_duplicates(
    candidates: list[tuple[Box, float]], existing: list[Box], threshold: float = 0.5
) -> list[tuple[Box, float]]:
    """Buang saran yang sudah ada boxnya (kelas sama, IoU > threshold)."""
    return [
        (box, conf)
        for box, conf in candidates
        if not any(e.cls == box.cls and box_iou(e, box) > threshold for e in existing)
    ]


def find_latest_best(runs_dir: Path) -> Path | None:
    """best.pt terbaru di runs/*/weights/ (berdasar waktu modifikasi), atau None."""
    try:
        found = list(Path(runs_dir).glob("*/weights/best.pt"))
    except OSError:
        return None
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def has_label(label_path: Path) -> bool:
    """True jika file label ada, termasuk file kosong (gambar ditandai null).
    Jika tak terbaca, dianggap berlabel (aman)."""
    try:
        return label_path.is_file()
    except OSError:
        return True


def run_auto_detect(
    detector: Detector,
    project,
    items: list[tuple[str, Path]],
    lookup: dict[str, int],
    conf: float,
    iou: float,
    stop_event: threading.Event,
    progress: Callable[[int, int, dict], None] | None = None,
) -> dict:
    """Deteksi semua gambar di `items` [(subset, path)] dan tulis label YOLO.

    Kebijakan SKIP: gambar yang sudah punya label (tidak kosong) tidak disentuh.
    Gambar tanpa hasil deteksi tidak dibuat file labelnya (sama seperti save_labels).
    """
    stats = {
        "total": len(items),
        "detected_images": 0,
        "boxes": 0,
        "skipped": 0,
        "no_detection": 0,
        "errors": 0,
        "first_error": None,
        "stopped": False,
    }
    for n, (subset, path) in enumerate(items, start=1):
        if stop_event.is_set():
            stats["stopped"] = True
            break
        label_path = project.label_path(subset, path)
        if has_label(label_path):
            stats["skipped"] += 1
        else:
            try:
                with Image.open(path) as im:
                    img = ImageOps.exif_transpose(im).convert("RGB")
                dets = detector.predict(img, conf, iou, should_cancel=stop_event.is_set)
                if dets is None:
                    stats["stopped"] = True
                    break
                boxes = [b for b, _ in to_boxes(dets, lookup, *img.size)]
                if not boxes:
                    stats["no_detection"] += 1
                elif has_label(label_path):  # cek ulang: mungkin baru dilabel manual
                    stats["skipped"] += 1
                else:
                    save_labels(label_path, boxes, [], *img.size)
                    stats["detected_images"] += 1
                    stats["boxes"] += len(boxes)
            except (
                Exception
            ) as exc:  # satu gambar rusak tidak boleh menghentikan semuanya
                stats["errors"] += 1
                stats["first_error"] = stats["first_error"] or f"{path.name}: {exc}"
        if progress:
            progress(n, len(items), stats)
    return stats
