"""
tuning.py
=========
Auto-tuning config training YOLO dengan Optuna (TPE multivariate + MedianPruner).

Output: projects/<project>/runs/<nama_tune>/
    study.db            <- database Optuna (bisa dilanjutkan jika terputus)
    tune_config.json    <- konfigurasi tuning
    trials.csv          <- semua trial + skor + latensi
    best_config.json    <- config terbaik, SIAP dipakai TrainingJob (epochs penuh)
    result_summary.json

Tanpa tkinter, aman dipanggil dari thread latar belakang. Trial berjalan berurutan
(satu GPU). Dependensi tambahan: pip install optuna
"""

from __future__ import annotations

import gc
import json
import math
import shutil
import threading
import traceback
from datetime import datetime
from pathlib import Path

from project import IMAGE_EXTS, Project
from training import DEFAULT_CONFIG, EpochProgress, parse_batch, validate_config

DEFAULT_TUNE_CONFIG = {
    "tune_name": "tune",
    "tune_resume": "",  # nama folder tune yang ingin dilanjutkan (kosong = baru)
    "n_trials": 100,
    "timeout_hours": 0.0,  # 0 = tanpa batas waktu
    "trial_epochs": 40,  # epoch per trial (pendek)
    "trial_fraction": 0.5,  # fraksi data train per trial
    "trial_patience": 15,
    "sampler_seed": 0,
    "baseline_first": True,  # trial 0 = config Ultralytics default / config Anda
    "keep_trial_dirs": False,
    "search_augmentation": True,
    "models": [],  # >1 pilihan -> model ikut di-tuning (mis. ["yolo26n.pt","yolo26s.pt"])
    "imgsz_choices": [],  # >1 pilihan -> imgsz ikut di-tuning (mis. [512,640,768])
    "speed_weight": 0.02,  # penalti skor per penggandaan latensi (hanya jika model/imgsz di-tuning)
}

# Nilai baseline = default Ultralytics; semuanya berada di dalam rentang pencarian.
BASELINE = {
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "cos_lr": False,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
}


def validate_tune_config(t: dict):
    if int(t["n_trials"]) < 1:
        raise ValueError("n_trials minimal 1.")
    if int(t["trial_epochs"]) < 5:
        raise ValueError("trial_epochs minimal 5.")
    if not 0 < float(t["trial_fraction"]) <= 1:
        raise ValueError("trial_fraction harus > 0 dan maksimal 1.")
    if float(t["timeout_hours"]) < 0 or float(t["speed_weight"]) < 0:
        raise ValueError("timeout_hours dan speed_weight tidak boleh negatif.")


def resolve_data_yaml(project: Project, version: str) -> Path:
    """Sama seperti pemeriksaan di TrainingJob."""
    version = (version or "").strip()
    if version:
        data_yaml = project.version_yaml(version)
        if not data_yaml.is_file():
            raise FileNotFoundError(f"Versi dataset '{version}' tidak ditemukan.")
        for split in ("train", "valid"):
            folder = data_yaml.parent / split / "images"
            if not any(p.suffix.lower() in IMAGE_EXTS for p in folder.glob("*")):
                raise ValueError(f"Versi {version}: subset {split} kosong.")
        return data_yaml
    data_yaml = project.root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError("data.yaml tidak ditemukan.")
    for split in ("train", "valid"):
        if not project.list_images(split):
            raise ValueError(f"Dataset {split} kosong. Split dataset terlebih dahulu.")
    return data_yaml


def fitness_from(d: dict) -> float:
    """Fitness Ultralytics: 0.1*mAP50 + 0.9*mAP50-95 (box)."""
    if d.get("fitness") is not None:
        return float(d["fitness"])
    return 0.1 * float(d.get("metrics/mAP50(B)", 0.0)) + 0.9 * float(
        d.get("metrics/mAP50-95(B)", 0.0)
    )


class TuningJob:
    """Pencarian hyperparameter di background dengan stop graceful."""

    def __init__(
        self,
        project,
        base_config,
        tune_config=None,
        on_status=None,
        on_done=None,
        on_progress=None,
    ):
        self.project = project
        self.config = {**DEFAULT_CONFIG, **base_config}
        self.tcfg = {**DEFAULT_TUNE_CONFIG, **(tune_config or {})}
        self.on_status = on_status or (lambda _s: None)
        self.on_done = on_done or (lambda _r: None)
        self.on_progress = on_progress or (lambda _p: None)
        self._best_score = None
        self._oom_streak = 0
        self._fatal = None
        self.stop_event = threading.Event()
        self.thread = None
        self.tune_dir: Path | None = None

    def start(self):
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Tuning masih berjalan.")
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="YOLO-Tuning"
        )
        self.thread.start()

    def request_stop(self):
        self.stop_event.set()
        self.on_status(
            "Permintaan berhenti diterima. Trial berjalan akan dihentikan..."
        )

    # ---------- ruang pencarian ----------
    def _search_models(self) -> bool:
        return len(self.tcfg["models"]) > 1 or len(self.tcfg["imgsz_choices"]) > 1

    def _suggest(self, trial) -> dict:
        s = trial.suggest_float
        p = {
            # 'auto' sengaja tidak dipakai: ia mengabaikan lr0/momentum.
            "optimizer": trial.suggest_categorical(
                "optimizer", ["SGD", "AdamW", "Adam", "NAdam"]
            ),
            "lr0": s("lr0", 1e-4, 1e-1, log=True),
            "lrf": s("lrf", 0.01, 0.5, log=True),
            "momentum": s("momentum", 0.8, 0.98),
            "weight_decay": s("weight_decay", 1e-5, 1e-3, log=True),
            "warmup_epochs": s("warmup_epochs", 0.0, 5.0),
            "cos_lr": trial.suggest_categorical("cos_lr", [False, True]),
        }
        if self.tcfg["search_augmentation"]:
            p.update(
                {
                    "hsv_h": s("hsv_h", 0.0, 0.1),
                    "hsv_s": s("hsv_s", 0.0, 0.9),
                    "hsv_v": s("hsv_v", 0.0, 0.9),
                    "degrees": s("degrees", 0.0, 20.0),
                    "translate": s("translate", 0.0, 0.3),
                    "scale": s("scale", 0.0, 0.9),
                    "shear": s("shear", 0.0, 10.0),
                    "perspective": s("perspective", 0.0, 0.001),
                    "flipud": s("flipud", 0.0, 0.5),
                    "fliplr": s("fliplr", 0.0, 1.0),
                    "mosaic": s("mosaic", 0.0, 1.0),
                    "mixup": s("mixup", 0.0, 0.5),
                }
            )
        p["close_mosaic"] = trial.suggest_int(
            "close_mosaic", 0, max(1, int(self.tcfg["trial_epochs"]) // 3)
        )
        models, sizes = self.tcfg["models"], self.tcfg["imgsz_choices"]
        p["model"] = (
            trial.suggest_categorical("model", models)
            if len(models) > 1
            else (models[0] if models else str(self.config["model"]).strip())
        )
        p["imgsz"] = (
            trial.suggest_categorical("imgsz", sizes)
            if len(sizes) > 1
            else (int(sizes[0]) if sizes else int(self.config["imgsz"]))
        )
        return p

    def _baseline_params(self) -> dict:
        b = dict(BASELINE)
        for k in ("lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "cos_lr"):
            b[k] = self.config[k]
        if str(self.config["optimizer"]) in ("SGD", "AdamW", "Adam", "NAdam"):
            b["optimizer"] = str(self.config["optimizer"])
        b["close_mosaic"] = min(
            int(self.config["close_mosaic"]),
            max(1, int(self.tcfg["trial_epochs"]) // 3),
        )
        if not self.tcfg["search_augmentation"]:
            for k in list(b):
                if k in (
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
                ):
                    b.pop(k)
        models, sizes = self.tcfg["models"], self.tcfg["imgsz_choices"]
        if len(models) > 1:
            b["model"] = (
                str(self.config["model"]).strip()
                if str(self.config["model"]).strip() in models
                else models[0]
            )
        if len(sizes) > 1:
            b["imgsz"] = (
                int(self.config["imgsz"])
                if int(self.config["imgsz"]) in sizes
                else int(sizes[0])
            )
        return b

    # ---------- satu trial ----------
    def _objective(self, trial, study, data_yaml: Path):
        import optuna
        from ultralytics import YOLO

        params = self._suggest(trial)
        name = f"trial_{trial.number:04d}"
        self.on_status(f"Trial {trial.number + 1}: {name}")
        state = {"pruned": False}
        model = YOLO(str(params["model"]))

        def on_batch(trainer):
            if self.stop_event.is_set():
                trainer.stop = True

        def on_epoch(trainer):
            if self.stop_event.is_set():
                trainer.stop = True
                return
            fit = trainer.fitness
            if fit is None or (isinstance(fit, float) and math.isnan(fit)):
                return
            trial.report(float(fit), int(trainer.epoch))
            if trial.should_prune():
                state["pruned"] = True
                trainer.stop = True

        model.add_callback("on_train_batch_end", on_batch)
        model.add_callback("on_fit_epoch_end", on_epoch)

        n_total = max(1, int(self.tcfg["n_trials"]))

        def report(epoch, epochs, inner):
            overall = (trial.number + (epoch - 1 + inner) / max(epochs, 1)) / n_total
            self.on_progress(
                {
                    "kind": "tune",
                    "trial": trial.number + 1,
                    "n_trials": n_total,
                    "epoch": epoch,
                    "epochs": epochs,
                    "frac": min(1.0, overall),
                    "best": self._best_score,
                }
            )

        EpochProgress(report).attach(model)

        cache = str(self.config["cache"]).lower()
        args = {
            "data": str(data_yaml.resolve()),
            "epochs": int(self.tcfg["trial_epochs"]),
            "imgsz": int(params["imgsz"]),
            "batch": parse_batch(self.config["batch"]),
            "workers": int(self.config["workers"]),
            "device": str(self.config["device"]).strip() or None,
            "patience": int(self.tcfg["trial_patience"]),
            "fraction": float(self.tcfg["trial_fraction"]),
            "cache": True
            if cache in ("true", "ram")
            else ("disk" if cache == "disk" else False),
            "amp": bool(self.config["amp"]),
            "pretrained": bool(self.config["pretrained"]),
            "seed": int(self.config["seed"]),
            "deterministic": bool(self.config["deterministic"]),
            "save": True,
            "plots": False,
            "val": True,
            "verbose": False,
            "project": str(self.tune_dir.resolve()),
            "name": name,
            "exist_ok": True,
        }
        args.update({k: v for k, v in params.items() if k not in ("model", "imgsz")})

        try:
            res = model.train(**args)
        except Exception as exc:
            if (
                "out of memory" in str(exc).lower()
            ):  # batch/imgsz terlalu besar -> buang trial
                self._oom_streak += 1
                if self._oom_streak >= 3:
                    self._fatal = (
                        "GPU kehabisan memori pada 3 trial berturut-turut. Pilih profil "
                        "hardware yang lebih ringan atau kecilkan batch/imgsz."
                    )
                raise optuna.TrialPruned("CUDA OOM")
            raise
        finally:
            self._cleanup(name)

        if self.stop_event.is_set() or state["pruned"]:
            raise optuna.TrialPruned()

        d = dict(getattr(res, "results_dict", {}) or {})
        fitness = fitness_from(d)
        latency = None
        try:
            latency = float(model.trainer.validator.speed["inference"])  # ms/gambar
        except Exception:
            pass
        trial.set_user_attr("fitness", fitness)
        trial.set_user_attr("map50", float(d.get("metrics/mAP50(B)", 0.0)))
        trial.set_user_attr("map50_95", float(d.get("metrics/mAP50-95(B)", 0.0)))
        trial.set_user_attr("latency_ms", latency)
        trial.set_user_attr("params", params)

        score = fitness
        if self._search_models() and latency and latency > 0:
            ref = study.user_attrs.get("ref_latency")
            if ref is None:
                study.set_user_attr("ref_latency", latency)
                ref = latency
            score = fitness - float(self.tcfg["speed_weight"]) * math.log2(
                latency / ref
            )
        trial.set_user_attr("score", score)
        self._oom_streak = 0
        return score

    def _cleanup(self, name: str):
        if not self.tcfg["keep_trial_dirs"]:
            shutil.rmtree(self.tune_dir / name, ignore_errors=True)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---------- hasil ----------
    def _write_best(self, study):
        try:
            best = study.best_trial
        except ValueError:
            return
        params = best.user_attrs.get("params", {})
        cfg = dict(self.config)
        cfg.update({k: v for k, v in params.items()})
        cfg["optimizer"] = params.get("optimizer", cfg["optimizer"])
        cfg["tuned_from"] = {
            "tune_dir": str(self.tune_dir),
            "trial": best.number,
            "score": best.value,
            "fitness": best.user_attrs.get("fitness"),
            "map50_95": best.user_attrs.get("map50_95"),
            "latency_ms": best.user_attrs.get("latency_ms"),
        }
        (self.tune_dir / "best_config.json").write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        try:
            study.trials_dataframe().to_csv(self.tune_dir / "trials.csv", index=False)
        except Exception:
            pass

    def _run(self):
        result = {"status": "failed", "tune_dir": None, "error": None, "stopped": False}
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            validate_config(self.config)
            validate_tune_config(self.tcfg)
            data_yaml = resolve_data_yaml(
                self.project, self.config.get("dataset_version", "")
            )

            resume = str(self.tcfg["tune_resume"]).strip()
            if resume:
                self.tune_dir = self.project.runs_dir / resume
                if not (self.tune_dir / "study.db").is_file():
                    raise FileNotFoundError(f"study.db tidak ada di {self.tune_dir}")
            else:
                name = self.project.next_run_name(self.tcfg.get("tune_name", "tune"))
                self.tune_dir = self.project.runs_dir / name
                self.tune_dir.mkdir(parents=True, exist_ok=False)
            (self.tune_dir / "tune_config.json").write_text(
                json.dumps(
                    {
                        "base": self.config,
                        "tune": self.tcfg,
                        "data": str(data_yaml.resolve()),
                        "created_at": datetime.now().astimezone().isoformat(),
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )

            study = optuna.create_study(
                study_name="yolo_tune",
                direction="maximize",
                load_if_exists=True,
                storage=f"sqlite:///{(self.tune_dir / 'study.db').resolve().as_posix()}",
                sampler=optuna.samplers.TPESampler(
                    seed=int(self.tcfg["sampler_seed"]),
                    multivariate=True,
                    n_startup_trials=15,
                ),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=10,
                    n_warmup_steps=max(5, int(self.tcfg["trial_epochs"]) // 4),
                ),
            )
            if self.tcfg["baseline_first"] and not study.trials:
                study.enqueue_trial(self._baseline_params())

            done = len([t for t in study.trials if t.state.is_finished()])
            remaining = max(0, int(self.tcfg["n_trials"]) - done)
            n_total = max(1, int(self.tcfg["n_trials"]))
            try:
                self._best_score = study.best_value
            except ValueError:
                pass
            self.on_progress(
                {
                    "kind": "tune",
                    "trial": min(done + 1, n_total),
                    "n_trials": n_total,
                    "epoch": 0,
                    "epochs": int(self.tcfg["trial_epochs"]),
                    "frac": min(1.0, done / n_total),
                    "best": self._best_score,
                }
            )
            timeout = float(self.tcfg["timeout_hours"]) * 3600 or None

            def after_trial(st, tr):
                self._write_best(st)
                if self.stop_event.is_set() or self._fatal:
                    st.stop()
                try:
                    self._best_score = st.best_value
                    self.on_status(
                        f"Trial {tr.number + 1}/{self.tcfg['n_trials']} selesai "
                        f"({tr.state.name}). Terbaik: {st.best_value:.4f}"
                    )
                except ValueError:
                    pass
                self.on_progress(
                    {
                        "kind": "tune",
                        "trial": min(tr.number + 2, n_total),
                        "n_trials": n_total,
                        "epoch": 0,
                        "epochs": int(self.tcfg["trial_epochs"]),
                        "frac": min(1.0, (tr.number + 1) / n_total),
                        "best": self._best_score,
                    }
                )

            study.optimize(
                lambda t: self._objective(t, study, data_yaml),
                n_trials=remaining,
                timeout=timeout,
                callbacks=[after_trial],
                gc_after_trial=True,
                catch=(RuntimeError, ValueError),
            )
            self._write_best(study)
            if self._fatal:
                raise RuntimeError(self._fatal)
            result["stopped"] = self.stop_event.is_set()
            result["status"] = "stopped" if result["stopped"] else "completed"
            try:
                result.update(
                    {
                        "best_trial": study.best_trial.number,
                        "best_score": study.best_value,
                        "best_params": study.best_trial.user_attrs.get("params"),
                    }
                )
            except ValueError:
                pass
            self.on_status(
                "Tuning dihentikan pengguna."
                if result["stopped"]
                else "Tuning selesai. Lihat best_config.json."
            )
        except ImportError as exc:
            result["error"] = (
                f"Dependensi belum terpasang ({exc}). Jalankan: pip install optuna"
            )
            self.on_status(result["error"])
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc()
            self.on_status(f"Tuning gagal: {exc}")
        finally:
            result["tune_dir"] = str(self.tune_dir) if self.tune_dir else None
            if self.tune_dir and self.tune_dir.exists():
                (self.tune_dir / "result_summary.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            self.on_done(result)
