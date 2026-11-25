from dataclasses import dataclass
from pathlib import Path


@dataclass
class Box:
    """Bounding box dalam koordinat piksel gambar asli (x1<x2, y1<y2)."""

    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def load_labels(path: Path, img_w: int, img_h: int) -> tuple[list[Box], list[str]]:
    """Baca file label YOLO ("cls cx cy w h", ternormalisasi) -> (boxes, extra_lines).

    Baris yang bukan format bounding box (mis. polygon segmentasi) tidak diubah:
    dikembalikan di extra_lines dan ditulis ulang apa adanya saat menyimpan.
    """
    boxes: list[Box] = []
    extra: list[str] = []
    if not path.exists():
        return boxes, extra

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            extra.append(line)
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            extra.append(line)
            continue
        boxes.append(
            Box(
                cls,
                (cx - w / 2) * img_w,
                (cy - h / 2) * img_h,
                (cx + w / 2) * img_w,
                (cy + h / 2) * img_h,
            )
        )
    return boxes, extra


def save_labels(
    path: Path,
    boxes: list[Box],
    extra: list[str],
    img_w: int,
    img_h: int,
    keep_empty: bool = False,
):
    """Tulis label YOLO. Tanpa anotasi: file dihapus, kecuali keep_empty=True
    (gambar ditandai null -> file kosong dibuat/dipertahankan)."""
    lines = []
    for b in boxes:
        cx = (b.x1 + b.x2) / 2 / img_w
        cy = (b.y1 + b.y2) / 2 / img_h
        w = (b.x2 - b.x1) / img_w
        h = (b.y2 - b.y1) / img_h
        lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    lines.extend(extra)

    if not lines:
        if keep_empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        elif path.exists():
            path.unlink()
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)  # tulis atomik: file lama tidak rusak kalau proses terputus
