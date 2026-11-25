"""
hardware.py
===========
Deteksi hardware dan profil pemakaian resource: Light / Medium / High / Extreme / Custom.

Yang diatur profil: device, batch, workers, cache, amp, deterministic.
Nilai dihitung dari hardware yang terdeteksi (VRAM, core CPU, RAM kosong), bukan angka
tetap, sehingga profil yang sama aman di laptop maupun workstation.

Konvensi batch (sama dengan Ultralytics):
    bilangan bulat > 0 = batch tetap, -1 = auto (60% VRAM), pecahan 0-1 = target pemakaian VRAM.

Tanpa tkinter, aman dipanggil dari thread latar belakang. torch di-import lazy
(lambat) dan hasil deteksi GPU di-cache.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from project import IMAGE_EXTS

PROFILE_ORDER = ("light", "medium", "high", "extreme", "custom")

# (judul, keterangan) per profil
PROFILE_TEXT = {
    "light": (
        "Light",
        "Hemat resource. Komputer tetap lancar dipakai kerja lain. Batch kecil (±25% VRAM), "
        "2 worker, tanpa cache. Paling lambat tetapi paling aman dari out-of-memory. "
        "Cocok untuk laptop atau GPU ≤ 4 GB.",
    ),
    "medium": (
        "Medium",
        "Seimbang. ±45% VRAM, worker ¼ core CPU, tanpa cache. Pilihan aman untuk kebanyakan PC; "
        "masih nyaman dipakai browsing ringan selama proses berjalan.",
    ),
    "high": (
        "High",
        "Performa tinggi. ±65% VRAM, worker ½ core CPU, cache di RAM bila muat (jika tidak, di disk), "
        "mode deterministic dimatikan agar lebih cepat. PC akan terasa berat saat berjalan.",
    ),
    "extreme": (
        "Extreme",
        "Maksimal. ±85% VRAM, worker hampir semua core, cache di RAM bila muat ≤60% RAM kosong. "
        "Ada risiko out-of-memory dan layar/PC tersendat. Tutup aplikasi lain dulu.",
    ),
    "custom": (
        "Custom",
        "Atur sendiri parameter hardware. Nilai awal mengikuti profil Medium.",
    ),
}


@dataclass
class HardwareInfo:
    cpu_cores: int = 1
    ram_total_gb: float | None = None
    ram_avail_gb: float | None = None
    accelerator: str = "cpu"  # "cuda" | "mps" | "cpu"
    gpu_name: str = ""
    gpu_count: int = 0
    vram_gb: float | None = None

    def summary(self) -> str:
        if self.accelerator == "cuda":
            count = f" ×{self.gpu_count}" if self.gpu_count > 1 else ""
            gpu = f"GPU: {self.gpu_name}{count}, {self.vram_gb:.0f} GB VRAM"
        elif self.accelerator == "mps":
            gpu = "GPU: Apple (MPS)"
        else:
            gpu = "GPU CUDA tidak terdeteksi, training memakai CPU (jauh lebih lambat)"
        ram = ""
        if self.ram_total_gb:
            ram = f"   |   RAM: {self.ram_total_gb:.0f} GB"
            if self.ram_avail_gb is not None:
                ram += f" (kosong {self.ram_avail_gb:.0f} GB)"
        return f"{gpu}   |   CPU: {self.cpu_cores} core{ram}"


_gpu_cache: dict | None = None
_lock = threading.Lock()


def detect() -> HardwareInfo:
    """Deteksi hardware. Bagian GPU di-cache; RAM kosong dibaca ulang setiap panggilan."""
    global _gpu_cache
    with _lock:
        if _gpu_cache is None:
            gpu = {
                "accelerator": "cpu",
                "gpu_name": "",
                "gpu_count": 0,
                "vram_gb": None,
            }
            try:
                import torch

                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    gpu = {
                        "accelerator": "cuda",
                        "gpu_name": props.name,
                        "gpu_count": torch.cuda.device_count(),
                        "vram_gb": props.total_memory / 1e9,
                    }
                elif (
                    getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available()
                ):
                    gpu = {
                        "accelerator": "mps",
                        "gpu_name": "Apple GPU",
                        "gpu_count": 1,
                        "vram_gb": None,
                    }
            except Exception:
                pass
            _gpu_cache = gpu
        info = HardwareInfo(cpu_cores=os.cpu_count() or 1, **_gpu_cache)
    try:
        import psutil  # sudah menjadi dependensi Ultralytics

        mem = psutil.virtual_memory()
        info.ram_total_gb, info.ram_avail_gb = mem.total / 1e9, mem.available / 1e9
    except Exception:
        pass
    return info


def count_train_images(project, version: str = "", fraction: float = 1.0) -> int:
    """Jumlah gambar train yang akan dipakai (untuk memperkirakan ukuran cache RAM)."""
    folder = (
        project.versions_dir / version / "train" / "images"
        if version
        else project.images_dir("train")
    )
    try:
        n = sum(1 for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    except OSError:
        n = 0
    return int(n * fraction)


def estimate_cache_gb(n_images: int, imgsz: int) -> float:
    """Perkiraan kasar: gambar di-resize (sisi panjang = imgsz), 3 kanal uint8."""
    return n_images * imgsz * imgsz * 3 * 0.8 / 1e9


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def resolve_profile(
    name: str, hw: HardwareInfo, n_images: int = 0, imgsz: int = 640
) -> dict:
    """Nilai konkret {device, batch, workers, cache, amp, deterministic} untuk profil preset."""
    if name not in ("light", "medium", "high", "extreme"):
        raise ValueError(f"Profil '{name}' tidak punya nilai preset.")
    cuda, gpu = hw.accelerator == "cuda", hw.accelerator in ("cuda", "mps")
    device = "0" if cuda else ("mps" if hw.accelerator == "mps" else "cpu")
    cores = hw.cpu_cores
    need, avail = estimate_cache_gb(n_images, imgsz), hw.ram_avail_gb

    def cache_for(limit: float) -> str:
        return (
            "ram"
            if (avail is not None and n_images and need <= avail * limit)
            else "disk"
        )

    batch = {
        "light": 0.25 if cuda else (8 if gpu else 4),
        "medium": 0.45 if cuda else (16 if gpu else 8),
        "high": 0.65 if cuda else (32 if gpu else 16),
        "extreme": 0.85 if cuda else (64 if gpu else 32),
    }[name]
    workers = {
        "light": 2,
        "medium": _clamp(cores // 4, 2, 6),
        "high": _clamp(cores // 2, 4, 12),
        "extreme": _clamp(cores - 2, 4, 16),
    }[name]
    cache = {
        "light": "False",
        "medium": "False",
        "high": cache_for(0.35),
        "extreme": cache_for(0.60),
    }[name]
    return {
        "device": device,
        "batch": batch,
        "workers": workers,
        "cache": cache,
        "amp": cuda,
        "deterministic": name in ("light", "medium"),
    }


def describe(cfg: dict) -> str:
    """Satu baris ringkas nilai konfigurasi hardware."""
    batch = cfg["batch"]
    batch_txt = (
        f"{batch:g} ({batch * 100:.0f}% VRAM)"
        if isinstance(batch, float)
        else str(batch)
    )
    return (
        f"device={cfg['device'] or 'auto'}  batch={batch_txt}  workers={cfg['workers']}  "
        f"cache={cfg['cache']}  amp={cfg['amp']}  deterministic={cfg['deterministic']}"
    )
