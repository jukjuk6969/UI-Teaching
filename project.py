import ast
import random
import re
import shutil
from pathlib import Path

# Root semua project: <folder aplikasi>/projects/<nama_project>
PROJECTS_ROOT = Path(__file__).resolve().parent / "projects"

POOL = "unsplit"  # tempat gambar setelah "Tambah ke Dataset", sebelum di-split
SPLITS = ("train", "valid", "test")
ALL_SUBSETS = (POOL,) + SPLITS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

DATA_YAML_TEMPLATE = """train: ../train/images
val: ../valid/images
test: ../test/images

nc: {nc}
names: {names}
"""


class Project:
    """Satu project = satu folder di PROJECTS_ROOT:

    projects/<nama>/
        data.yaml
        unsplit/{images,labels}   <- pool: gambar yang belum dibagi
        train/{images,labels}
        valid/{images,labels}
        test/{images,labels}

    Setelah split, folder train/valid/test siap dipakai sebagai export YOLO.
    """

    def __init__(self, root: Path):
        self.root = root
        self.name = root.name

    # ---------- Buat / buka ----------
    @staticmethod
    def validate_name(name: str) -> str | None:
        """Return pesan error, atau None kalau nama valid."""
        if not name:
            return "Nama project tidak boleh kosong."
        if not NAME_RE.match(name):
            return "Nama hanya boleh huruf, angka, '_' dan '-' (tanpa spasi)."
        return None

    @classmethod
    def create(cls, name: str) -> "Project":
        error = cls.validate_name(name)
        if error:
            raise ValueError(error)
        root = PROJECTS_ROOT / name
        if root.exists():
            raise FileExistsError(f"Project '{name}' sudah ada.")
        project = cls(root)
        project.ensure_structure()
        return project

    @classmethod
    def open(cls, folder: Path) -> "Project":
        root = Path(folder).resolve()
        if root.parent != PROJECTS_ROOT.resolve():
            raise ValueError(
                f"Pilih folder project langsung di dalam:\n{PROJECTS_ROOT}"
            )
        looks_valid = (root / "data.yaml").exists() or any(
            (root / s).is_dir() for s in ALL_SUBSETS
        )
        if not looks_valid:
            raise ValueError(
                "Folder ini bukan project (tidak ada data.yaml atau folder unsplit/train/valid/test)."
            )
        project = cls(root)
        project.ensure_structure()  # lengkapi folder yang kurang, tidak menimpa yang ada
        return project

    def ensure_structure(self):
        for subset in ALL_SUBSETS:
            self.images_dir(subset).mkdir(parents=True, exist_ok=True)
            self.labels_dir(subset).mkdir(parents=True, exist_ok=True)
        if not (self.root / "data.yaml").exists():
            self.write_yaml([])

    # ---------- Path ----------
    def images_dir(self, subset: str) -> Path:
        return self.root / subset / "images"

    def labels_dir(self, subset: str) -> Path:
        return self.root / subset / "labels"

    def label_path(self, subset: str, image_path: Path) -> Path:
        return self.labels_dir(subset) / f"{image_path.stem}.txt"

    # ---------- Gambar ----------
    def list_images(self, subset: str) -> list[Path]:
        folder = self.images_dir(subset)
        return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    def add_to_dataset(self, files) -> tuple[int, int]:
        """Salin file ke unsplit/images. Return (jumlah disalin, jumlah dilewati)."""
        dest = self.images_dir(POOL)
        copied = skipped = 0
        for f in map(Path, files):
            if f.suffix.lower() not in IMAGE_EXTS:
                skipped += 1
                continue
            target = dest / f.name
            n = 1
            while target.exists():  # hindari menimpa file bernama sama
                target = dest / f"{f.stem}_{n}{f.suffix}"
                n += 1
            shutil.copy2(f, target)
            copied += 1
        return copied, skipped

    # ---------- Split ----------
    def split_dataset(
        self, percents: dict[str, float], seed: int = 42
    ) -> dict[str, int]:
        """Acak gambar di unsplit lalu pindahkan (beserta label) ke train/valid/test.

        percents: {"train": 70, "valid": 20, "test": 10}, total harus 100.
        Seed yang sama + isi unsplit yang sama = hasil split yang sama.
        Return jumlah gambar per split.
        """
        values = [percents[s] for s in SPLITS]
        if any(v < 0 for v in values):
            raise ValueError("Persentase tidak boleh negatif.")
        if abs(sum(values) - 100) > 1e-6:
            raise ValueError(f"Total persentase harus 100 (sekarang {sum(values):g}).")

        images = self.list_images(POOL)
        if not images:
            raise ValueError("Tidak ada gambar di dataset (unsplit).")

        random.Random(seed).shuffle(images)
        counts = self._allocate(len(images), values)

        result = {}
        start = 0
        for subset, count in zip(SPLITS, counts):
            for img in images[start : start + count]:
                self._move_pair(img, subset)
            start += count
            result[subset] = count
        return result

    @staticmethod
    def _allocate(n: int, percents: list[float]) -> list[int]:
        """Bagi n item sesuai persentase (largest remainder), jumlah selalu tepat n."""
        raw = [n * p / 100 for p in percents]
        counts = [int(r) for r in raw]
        leftover = n - sum(counts)
        order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
        for i in order[:leftover]:
            counts[i] += 1
        return counts

    def _move_pair(self, image: Path, subset: str):
        """Pindahkan gambar + label-nya (kalau ada) dari unsplit ke subset."""
        img_dir, lbl_dir = self.images_dir(subset), self.labels_dir(subset)
        stem, ext = image.stem, image.suffix
        new_stem, n = stem, 1
        while (img_dir / f"{new_stem}{ext}").exists() or (
            lbl_dir / f"{new_stem}.txt"
        ).exists():
            new_stem = f"{stem}_{n}"
            n += 1

        src_label = self.label_path(POOL, image)
        shutil.move(str(image), str(img_dir / f"{new_stem}{ext}"))
        if src_label.exists():
            shutil.move(str(src_label), str(lbl_dir / f"{new_stem}.txt"))

    # ---------- Training runs ----------
    @property
    def runs_dir(self) -> Path:
        """Folder utama seluruh eksperimen training milik project ini."""
        return self.root / "runs"

    def next_run_name(self, base_name: str = "train") -> str:
        """Cari nama run yang belum dipakai: train, train2, train3, ..."""
        base_name = (
            re.sub(r"[^A-Za-z0-9_-]+", "_", base_name.strip()).strip("_") or "train"
        )
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        candidate = base_name
        index = 2
        while (self.runs_dir / candidate).exists():
            candidate = f"{base_name}{index}"
            index += 1
        return candidate

    # ---------- Versi dataset (hasil preprocessing/augmentasi) ----------
    @property
    def versions_dir(self) -> Path:
        """Folder versi dataset: versions/v1, v2, ... (masing-masing punya data.yaml sendiri)."""
        return self.root / "versions"

    def list_versions(self) -> list[str]:
        """Nama versi yang lengkap (punya data.yaml), diurutkan dari v1 ke terbaru."""
        if not self.versions_dir.is_dir():
            return []
        found = [
            d.name
            for d in self.versions_dir.iterdir()
            if re.fullmatch(r"v\d+", d.name) and (d / "data.yaml").is_file()
        ]
        return sorted(found, key=lambda n: int(n[1:]))

    def next_version_name(self) -> str:
        """v1, v2, ... (nomor tertinggi yang ada + 1, termasuk folder yang setengah jadi)."""
        nums = (
            [
                int(d.name[1:])
                for d in self.versions_dir.iterdir()
                if re.fullmatch(r"v\d+", d.name)
            ]
            if self.versions_dir.is_dir()
            else []
        )
        return f"v{max(nums, default=0) + 1}"

    def version_yaml(self, version: str) -> Path:
        return self.versions_dir / version / "data.yaml"

    # ---------- data.yaml ----------
    def write_yaml(self, class_names: list[str]):
        names = "[" + ", ".join(repr(n) for n in class_names) + "]"
        content = DATA_YAML_TEMPLATE.format(nc=len(class_names), names=names)
        (self.root / "data.yaml").write_text(content, encoding="utf-8")

    def read_names(self) -> list[str]:
        """Baca daftar kelas dari data.yaml (format ['a', 'b'] atau daftar berbaris '- a')."""
        yaml_path = self.root / "data.yaml"
        if not yaml_path.exists():
            return []
        text = yaml_path.read_text(encoding="utf-8")

        inline = re.search(r"^names:\s*(\[.*\])\s*$", text, re.MULTILINE)
        if inline:
            try:
                return [str(n) for n in ast.literal_eval(inline.group(1))]
            except (ValueError, SyntaxError):
                return []

        block = re.search(r"^names:\s*\n((?:[ \t]*-[ \t]*.+\n?)+)", text, re.MULTILINE)
        if block:
            return [
                re.sub(r"^[ \t]*-[ \t]*", "", line).strip().strip("'\"")
                for line in block.group(1).splitlines()
                if line.strip()
            ]
        return []

    def add_class(self, name: str) -> list[str]:
        """Tambah kelas baru di akhir (id kelas lama tidak berubah). Return daftar kelas terbaru."""
        names = self.read_names()
        names.append(name)
        self.write_yaml(names)
        return names

    def remove_class(self, index: int) -> dict[str, int]:
        """Hapus kelas `index`: buang semua anotasinya di semua subset,
        lalu geser id kelas di atasnya turun 1. Return statistik."""
        names = self.read_names()
        if not 0 <= index < len(names):
            raise ValueError("Kelas tidak ditemukan.")

        stats = {"files_changed": 0, "boxes_removed": 0}
        for subset in ALL_SUBSETS:
            for path in self.labels_dir(subset).glob("*.txt"):
                out, changed = [], False
                for line in path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    try:
                        cls = int(parts[0])
                    except (IndexError, ValueError):
                        if parts:
                            out.append(line)  # baris tak dikenal: biarkan apa adanya
                        continue
                    if cls == index:  # anotasi kelas yang dihapus
                        stats["boxes_removed"] += 1
                        changed = True
                    elif cls > index:  # geser id
                        parts[0] = str(cls - 1)
                        out.append(" ".join(parts))
                        changed = True
                    else:
                        out.append(line)
                if not changed:
                    continue
                if out:
                    tmp = path.with_name(path.name + ".tmp")
                    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
                    tmp.replace(path)
                else:
                    path.unlink()  # tidak ada anotasi tersisa
                stats["files_changed"] += 1

        names.pop(index)
        self.write_yaml(names)  # data.yaml diperbarui terakhir
        return stats

    def count_class_boxes(self, index: int) -> int:
        """Jumlah anotasi kelas `index` di semua subset (untuk konfirmasi)."""
        total = 0
        for subset in ALL_SUBSETS:
            for path in self.labels_dir(subset).glob("*.txt"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    try:
                        if parts and int(parts[0]) == index:
                            total += 1
                    except ValueError:
                        pass
        return total
