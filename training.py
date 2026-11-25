"""
training.py
===========
Mesin training Ultralytics YOLO. Seluruh output berada di:
projects/<project>/runs/<nama_run>/
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import datetime

from project import IMAGE_EXTS

DEFAULT_CONFIG = {
    "model": "yolo26n.pt",
    "run_name": "train",
    "dataset_version": "",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "device": "",
    "workers": 8,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "patience": 100,
    "save_period": -1,
    "cache": "False",
    "amp": True,
    "pretrained": True,
    "cos_lr": False,
    "close_mosaic": 10,
    "seed": 0,
    "deterministic": True,
    "fraction": 1.0,
    "val": True,
    "plots": True,
    "resume_from": "",
}

# Parameter augmentasi online Ultralytics. Tidak tampil di dialog training; diisi
# otomatis dari hasil Auto Tune (best_config.json) dan diteruskan ke model.train().
AUG_KEYS = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
)

CONFIG_HELP = {
    "model": "Bobot awal/model YOLO (.pt), misalnya yolo26n.pt atau path bobot lokal.",
    "run_name": "Nama dasar folder eksperimen. Jika sudah ada, otomatis menjadi nama2, nama3, dan seterusnya.",
    "dataset_version": "Dataset yang dipakai. '(dataset asli)' = train/valid/test di project apa adanya. Atau pilih versi hasil Preprocessing (mis. v1) yang sudah berisi resize/augmentasi. Default: versi terbaru.",
    "epochs": "Jumlah maksimum putaran penuh dataset. Bisa berhenti lebih awal karena patience atau tombol Stop.",
    "imgsz": "Ukuran sisi gambar input dalam piksel. Lebih besar membantu objek kecil tetapi memakai VRAM lebih banyak.",
    "batch": "Gambar per batch. Bilangan bulat = tetap; -1 = auto-batch (60% VRAM); pecahan 0-1 = target pemakaian VRAM (0.7 = 70%). Batch besar memerlukan VRAM lebih besar.",
    "device": "Perangkat: kosong=otomatis, 0=GPU pertama, cpu=CPU, atau 0,1 untuk multi-GPU.",
    "workers": "Jumlah worker pemuat data. Kurangi jika RAM/CPU terbatas atau ada masalah multiprocessing.",
    "optimizer": "Optimizer: auto, SGD, Adam, AdamW, NAdam, RAdam, atau RMSProp.",
    "lr0": "Learning rate awal. Terlalu besar dapat membuat training tidak stabil.",
    "lrf": "Faktor learning rate akhir relatif terhadap lr0. 0.01 berarti sekitar 1% dari lr0.",
    "momentum": "Momentum optimizer, terutama SGD. Mengontrol kontribusi arah update sebelumnya.",
    "weight_decay": "Regularisasi untuk mengurangi overfitting. Terlalu besar dapat menghambat pembelajaran.",
    "warmup_epochs": "Durasi pemanasan learning rate pada awal training.",
    "patience": "Early stopping jika metrik validasi tidak membaik selama N epoch. 0 menonaktifkan.",
    "save_period": "Checkpoint berkala tiap N epoch (epoch1.pt, dst). Default -1 = nonaktif: hanya last.pt (epoch terakhir) dan best.pt yang disimpan.",
    "cache": "False=tanpa cache, True=RAM, disk=cache ke disk. Cache mempercepat loading tetapi memakai ruang.",
    "amp": "Automatic Mixed Precision. Biasanya mempercepat GPU dan menghemat VRAM.",
    "pretrained": "Gunakan bobot pretrained; umumnya disarankan untuk transfer learning.",
    "cos_lr": "Gunakan cosine learning-rate scheduler.",
    "close_mosaic": "Matikan augmentasi mosaic pada N epoch terakhir. 0 tidak mengatur penutupan mosaic.",
    "seed": "Seed random untuk membantu reproduksibilitas.",
    "deterministic": "Aktifkan operasi deterministik jika tersedia; bisa mengurangi performa.",
    "fraction": "Fraksi dataset train yang dipakai, dari >0 sampai 1.0.",
    "val": "Jalankan evaluasi validation selama training.",
    "plots": "Simpan grafik training, confusion matrix, dan visualisasi hasil jika didukung.",
    "resume_from": "Path checkpoint last.pt untuk melanjutkan training. Kosong berarti mulai dari model di atas.",
}


def parse_batch(value):
    """int (batch tetap atau -1) atau float 0<x<1 (fraksi VRAM, semantik Ultralytics)."""
    try:
        number = float(str(value).strip())
    except ValueError:
        raise ValueError("batch harus berupa angka.") from None
    if 0 < number < 1:
        return number
    if number != int(number):
        raise ValueError("batch pecahan harus di antara 0 dan 1 (fraksi VRAM).")
    return int(number)


def validate_config(c):
    if (
        not str(c.get("model", "")).strip()
        and not str(c.get("resume_from", "")).strip()
    ):
        raise ValueError("Model atau checkpoint resume harus diisi.")
    if int(c["epochs"]) < 1 or int(c["imgsz"]) < 32:
        raise ValueError("epochs minimal 1 dan imgsz minimal 32.")
    batch = parse_batch(c["batch"])
    if isinstance(batch, int) and (batch == 0 or batch < -1):
        raise ValueError(
            "batch harus -1, bilangan positif, atau pecahan 0-1 (fraksi VRAM)."
        )
    if int(c["workers"]) < 0 or int(c["patience"]) < 0:
        raise ValueError("workers dan patience tidak boleh negatif.")
    if float(c["lr0"]) <= 0 or float(c["lrf"]) < 0:
        raise ValueError("lr0 harus > 0 dan lrf harus >= 0.")
    if not 0 <= float(c["momentum"]) <= 1:
        raise ValueError("momentum harus di antara 0 dan 1.")
    if float(c["weight_decay"]) < 0 or float(c["warmup_epochs"]) < 0:
        raise ValueError("weight_decay dan warmup_epochs tidak boleh negatif.")
    if int(c["save_period"]) == 0 or int(c["save_period"]) < -1:
        raise ValueError("save_period harus -1 atau bilangan positif.")
    if not 0 < float(c["fraction"]) <= 1:
        raise ValueError("fraction harus > 0 dan maksimal 1.")
    if int(c["close_mosaic"]) < 0:
        raise ValueError("close_mosaic tidak boleh negatif.")
    if str(c["cache"]).lower() not in ("false", "true", "disk", "ram", ""):
        raise ValueError("cache harus False, True, atau disk.")


class EpochProgress:
    """Ubah callback Ultralytics menjadi laporan kemajuan: emit(epoch, epochs, bagian_epoch 0-1).

    Dibatasi sekitar 4 kali per detik supaya antrean UI tidak penuh. Error di emit
    diabaikan agar tidak pernah mengganggu training.
    """

    def __init__(self, emit, min_interval: float = 0.25):
        self.emit, self.min_interval = emit, min_interval
        self.batches, self.last = 0, 0.0

    def attach(self, model):
        model.add_callback("on_train_epoch_start", self._epoch_start)
        model.add_callback("on_train_batch_end", self._batch_end)
        model.add_callback("on_train_epoch_end", self._epoch_end)

    def _epoch_start(self, trainer):
        self.batches = 0
        self._send(trainer, 0.0, force=True)

    def _batch_end(self, trainer):
        self.batches += 1
        try:
            total = len(trainer.train_loader)
        except Exception:
            total = 0
        if total:
            self._send(trainer, self.batches / total)

    def _epoch_end(self, trainer):
        self._send(trainer, 1.0, force=True)

    def _send(self, trainer, inner, force=False):
        now = time.monotonic()
        if not force and now - self.last < self.min_interval:
            return
        self.last = now
        try:
            self.emit(
                int(trainer.epoch) + 1, int(trainer.epochs), min(max(inner, 0.0), 1.0)
            )
        except Exception:
            pass


class TrainingJob:
    """Training background dengan permintaan stop graceful."""

    def __init__(self, project, config, on_status=None, on_done=None, on_progress=None):
        self.project = project
        self.config = dict(config)
        self.on_status = on_status or (lambda _s: None)
        self.on_done = on_done or (lambda _r: None)
        self.on_progress = on_progress or (lambda _p: None)
        self.stop_event = threading.Event()
        self.thread = None
        self.run_dir = None

    def start(self):
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Training masih berjalan.")
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="YOLO-Training"
        )
        self.thread.start()

    def request_stop(self):
        self.stop_event.set()
        self.on_status("Permintaan berhenti diterima. Menunggu checkpoint aman...")

    def _run(self):
        result = {"status": "failed", "run_dir": None, "error": None, "stopped": False}
        try:
            from ultralytics import YOLO

            validate_config(self.config)
            version = str(self.config.get("dataset_version", "")).strip()
            if version:
                data_yaml = self.project.version_yaml(version)
                if not data_yaml.is_file():
                    raise FileNotFoundError(
                        f"Versi dataset '{version}' tidak ditemukan."
                    )
                for split in ("train", "valid"):
                    folder = data_yaml.parent / split / "images"
                    if not any(
                        p.suffix.lower() in IMAGE_EXTS for p in folder.glob("*")
                    ):
                        raise ValueError(f"Versi {version}: subset {split} kosong.")
            else:
                data_yaml = self.project.root / "data.yaml"
                if not data_yaml.is_file():
                    raise FileNotFoundError("data.yaml tidak ditemukan.")
                for split in ("train", "valid"):
                    if not self.project.list_images(split):
                        raise ValueError(
                            f"Dataset {split} kosong. Split dataset terlebih dahulu."
                        )

            run_name = self.project.next_run_name(self.config.get("run_name", "train"))
            self.run_dir = self.project.runs_dir / run_name
            self.run_dir.mkdir(parents=True, exist_ok=False)
            saved = dict(self.config)
            saved.update(
                {
                    "resolved_run_name": run_name,
                    "created_at": datetime.now().astimezone().isoformat(),
                    "data": str(data_yaml.resolve()),
                }
            )
            (self.run_dir / "config.json").write_text(
                json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.on_status(
                f"Training berjalan: {run_name}"
                + (f" (dataset {version})" if version else "")
            )

            resume_path = str(self.config.get("resume_from", "")).strip()
            model = YOLO(resume_path or str(self.config["model"]).strip())

            def stop_callback(trainer):
                if self.stop_event.is_set():
                    trainer.stop = True

            model.add_callback("on_train_batch_end", stop_callback)
            model.add_callback("on_train_epoch_end", stop_callback)

            def report(epoch, epochs, inner):
                self.on_progress(
                    {
                        "kind": "train",
                        "epoch": epoch,
                        "epochs": epochs,
                        "inner": inner,
                        "frac": min(1.0, (epoch - 1 + inner) / max(epochs, 1)),
                    }
                )

            EpochProgress(report).attach(model)

            cache_value = str(self.config["cache"]).lower()
            cache = (
                True
                if cache_value in ("true", "ram")
                else ("disk" if cache_value == "disk" else False)
            )
            args = {
                "data": str(data_yaml.resolve()),
                "epochs": int(self.config["epochs"]),
                "imgsz": int(self.config["imgsz"]),
                "batch": parse_batch(self.config["batch"]),
                "workers": int(self.config["workers"]),
                "optimizer": str(self.config["optimizer"]),
                "lr0": float(self.config["lr0"]),
                "lrf": float(self.config["lrf"]),
                "momentum": float(self.config["momentum"]),
                "weight_decay": float(self.config["weight_decay"]),
                "warmup_epochs": float(self.config["warmup_epochs"]),
                "patience": int(self.config["patience"]),
                "save": True,
                "save_period": int(self.config["save_period"]),
                "cache": cache,
                "amp": bool(self.config["amp"]),
                "pretrained": bool(self.config["pretrained"]),
                "cos_lr": bool(self.config["cos_lr"]),
                "close_mosaic": int(self.config["close_mosaic"]),
                "seed": int(self.config["seed"]),
                "deterministic": bool(self.config["deterministic"]),
                "fraction": float(self.config["fraction"]),
                "val": bool(self.config["val"]),
                "plots": bool(self.config["plots"]),
                "device": str(self.config["device"]).strip() or None,
                "project": str(self.project.runs_dir.resolve()),
                "name": run_name,
                "exist_ok": False,
                "resume": bool(resume_path),
                "verbose": True,
            }
            args.update(
                {k: float(self.config[k]) for k in AUG_KEYS if k in self.config}
            )
            train_result = model.train(**args)
            result["status"] = "stopped" if self.stop_event.is_set() else "completed"
            result["stopped"] = self.stop_event.is_set()
            try:
                result["metrics"] = {
                    str(k): float(v) for k, v in train_result.results_dict.items()
                }
            except Exception:
                result["metrics"] = {}
            self.on_status(
                "Training dihentikan pengguna."
                if result["stopped"]
                else "Training selesai."
            )
        except KeyboardInterrupt:
            result["status"] = "stopped"
            result["stopped"] = True
            self.on_status("Training dihentikan.")
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc()
            self.on_status(f"Training gagal: {exc}")
        finally:
            result["run_dir"] = str(self.run_dir) if self.run_dir else None
            if self.run_dir and self.run_dir.exists():
                (self.run_dir / "result_summary.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            self.on_done(result)
