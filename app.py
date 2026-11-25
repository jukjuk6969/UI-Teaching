import os
import re
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path as _Path

LOG_FILE = None


def _setup_quiet_mode():
    """Dijalankan lewat pythonw (tanpa jendela konsol), sys.stdout/sys.stderr bernilai None
    sehingga print/tqdm milik Ultralytics bisa error. Arahkan keduanya ke logs/app.log
    supaya semua pesan dan traceback tetap tercatat. Log dibatasi 1000 baris terakhir."""
    global LOG_FILE
    if sys.stdout is not None and sys.stderr is not None:
        return
    log_dir = _Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / "app.log"

    # Rotasi: batasi 1000 baris terakhir
    MAX_LINES = 1000
    if path.exists():
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if len(lines) > MAX_LINES:
                lines = lines[-MAX_LINES:]
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass

    handle = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or handle
    sys.stderr = sys.stderr or handle
    os.environ.setdefault("TQDM_MININTERVAL", "1")
    handle.write(f"\n--- {datetime.now():%Y-%m-%d %H:%M:%S} (app dibuka) ---\n")
    LOG_FILE = str(path)


_setup_quiet_mode()


class ConsoleBuffer:
    """Menampung output konsol (tqdm/logging Ultralytics) untuk ditampilkan di UI."""

    _ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

    def __init__(self, max_lines: int = 500):
        self._lock = threading.Lock()
        self._cur = ""  # teks sejak \r terakhir
        self._live = ""  # baris sebelumnya yang sedang ditimpa
        self._pending = deque(maxlen=max_lines)  # baris selesai, belum dibaca UI

    def feed(self, text: str) -> list[str]:
        """Proses potongan teks; kembalikan baris yang selesai (berakhir \\n)."""
        text = self._ANSI.sub("", text)
        done = []
        with self._lock:
            for ch in text:
                if ch == "\r":
                    if self._cur.strip():
                        self._live = self._cur
                    self._cur = ""
                elif ch == "\n":
                    line = self._cur if self._cur.strip() else self._live
                    if line.strip():
                        done.append(line.rstrip())
                        self._pending.append(line.rstrip())
                    self._cur = self._live = ""
                else:
                    self._cur += ch
        return done

    def snapshot(self) -> tuple[str, list[str]]:
        """(baris live saat ini, baris baru yang selesai sejak panggilan terakhir)."""
        with self._lock:
            live = self._cur if self._cur.strip() else self._live
            lines = list(self._pending)
            self._pending.clear()
        return live, lines


class ConsoleTee:
    """Pengganti sys.stdout/sys.stderr: meneruskan ke tujuan asli + menyalin ke buffer."""

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, inner, buffer: ConsoleBuffer, raw: bool):
        self._inner = inner
        self._buffer = buffer
        self._raw = raw  # True = terminal asli, False = file log

    def write(self, text):
        text = text if isinstance(text, str) else str(text)
        done = self._buffer.feed(text)
        if self._inner is not None:
            try:
                if self._raw:
                    self._inner.write(text)
                elif done:  # file log hanya diisi baris selesai, bukan update \r
                    self._inner.write("\n".join(done) + "\n")
            except Exception:
                pass
        return len(text)

    def flush(self):
        try:
            if self._inner is not None:
                self._inner.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return bool(self._inner is not None and self._inner.isatty())
        except Exception:
            return False

    def __getattr__(self, name):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


CONSOLE = ConsoleBuffer()
sys.stdout = ConsoleTee(sys.stdout, CONSOLE, raw=LOG_FILE is None)
sys.stderr = ConsoleTee(sys.stderr, CONSOLE, raw=LOG_FILE is None)

import dataclasses
import json
import queue
import random
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter import font as tkfont

import numpy as np
from analysis import analyze, balance_status
from detector import (
    Detector,
    build_lookup,
    drop_duplicates,
    find_latest_best,
    missing_classes,
    run_auto_detect,
    to_boxes,
)
from hardware import (
    PROFILE_ORDER,
    PROFILE_TEXT,
    count_train_images,
    describe,
    detect,
    resolve_profile,
)
from PIL import Image, ImageDraw, ImageOps, ImageTk
from preprocess import (
    CHOICES as PREPROCESS_CHOICES,
)
from preprocess import (
    DEFAULT_AUGMENT,
    DEFAULT_PREPROCESS,
    active_augmentations,
    build_version,
    coerce_config,
    make_preview,
    validate_augment,
    validate_preprocess,
)
from preprocess import (
    HELP as PREPROCESS_HELP,
)
from project import ALL_SUBSETS, IMAGE_EXTS, POOL, PROJECTS_ROOT, SPLITS, Project
from training import AUG_KEYS, CONFIG_HELP, DEFAULT_CONFIG, TrainingJob, parse_batch
from tuning import TuningJob
from yolo import Box, load_labels, save_labels

PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#008080",
    "#f032e6",
    "#9a6324",
    "#800000",
    "#808000",
]
MIN_DRAG_PX = 5
MAX_UNDO = 50
HANDLE_DRAW = 4  # setengah ukuran kotak handle (piksel layar)
HANDLE_HIT = 8  # area klik handle (piksel layar)
HANDLE_CURSORS = {
    "nw": "top_left_corner",
    "ne": "top_right_corner",
    "sw": "bottom_left_corner",
    "se": "bottom_right_corner",
    "n": "sb_v_double_arrow",
    "s": "sb_v_double_arrow",
    "e": "sb_h_double_arrow",
    "w": "sb_h_double_arrow",
}
ORIGINAL_DATASET = "(dataset asli)"
YOLO_SIZES = ("n", "s", "m", "l", "x")
YOLO_MODELS = [
    f"{family}{size}.pt"
    for family in ("yolo26", "yolo11", "yolov8")
    for size in YOLO_SIZES
]
HELP_TEXT = (
    "Drag area kosong: buat box\n"
    "Klik box: pilih\n"
    "Drag box terpilih: geser\n"
    "Drag titik sudut/sisi: ubah ukuran\n"
    "Shift+drag: paksa buat box baru\n"
    "Shift+panah: geser box 1 px\n"
    "1-9: pilih kelas aktif\n"
    "R: ubah kelas box terpilih\n"
    "Del / X: hapus box\n"
    "Ctrl+Z: undo\n"
    "Esc: batal pilih\n"
    "A / D atau panah: gambar lain"
)
SHORTCUTS = [
    ("Seret di gambar", "Buat box"),
    ("Klik box", "Pilih box"),
    ("1 – 9", "Pilih kelas aktif"),
    ("R", "Ubah kelas box terpilih"),
    ("Del  /  X", "Hapus box"),
    ("N", "Tandai / batalkan null (gambar tanpa objek)"),
    ("Ctrl + Z", "Undo"),
    ("Esc", "Batal pilih / tolak saran"),
    ("S", "Sarankan box (Label Assist)"),
    ("Enter", "Terima semua saran"),
    ("Klik box saran", "Terima satu saran"),
    ("A / D  atau  ← →", "Gambar sebelumnya / berikutnya"),
]


# ============================================================
# DESIGN SYSTEM
# Netral gelap, satu aksen. Warna jenuh hanya untuk box kelas
# dan status, supaya gambar tetap jadi fokus.
# ============================================================
COLORS = {
    "bg": "#15171c",  # latar jendela
    "surface": "#1b1e24",  # toolbar, panel, input
    "surface_alt": "#23272f",  # kartu, tombol sekunder
    "surface_hover": "#2d323c",
    "border": "#2a2e37",
    "border_strong": "#3a404b",
    "text": "#e8eaed",
    "text_muted": "#9aa3b0",
    "text_subtle": "#6b7381",
    "accent": "#6ea6ff",
    "accent_hover": "#8bb9ff",
    "accent_soft": "#26364f",
    "on_accent": "#0c1626",
    "success": "#3ebd7b",
    "success_hover": "#58cf91",
    "on_success": "#08190f",
    "warning": "#e8a33d",
    "danger": "#e5605a",
    "danger_text": "#ff8a84",
    "danger_soft": "#3a2527",
    "canvas_bg": "#0c0e12",
    "canvas_border": "#23272f",
    "canvas_text": "#7b8490",
    "canvas_sub": "#586170",
}

FONTS = {
    "ui": ("Segoe UI", 10),
    "small": ("Segoe UI", 9),
    "tiny": ("Segoe UI", 8),
    "section": ("Segoe UI", 8, "bold"),
    "header": ("Segoe UI", 13),
    "bold": ("Segoe UI", 9, "bold"),
    "mono": ("Consolas", 9),
}

SP = {"xs": 2, "sm": 4, "md": 8, "lg": 12, "xl": 16}

BUTTON_STYLES = {
    "primary": {
        "bg": COLORS["accent"],
        "fg": COLORS["on_accent"],
        "hover": COLORS["accent_hover"],
    },
    "secondary": {
        "bg": COLORS["surface_alt"],
        "fg": COLORS["text"],
        "hover": COLORS["surface_hover"],
    },
    "success": {
        "bg": COLORS["success"],
        "fg": COLORS["on_success"],
        "hover": COLORS["success_hover"],
    },
    "danger": {
        "bg": COLORS["surface_alt"],
        "fg": COLORS["danger_text"],
        "hover": COLORS["danger_soft"],
    },
    "ghost": {
        "bg": COLORS["surface"],
        "fg": COLORS["text_muted"],
        "hover": COLORS["surface_alt"],
    },
}


def _apply_system_fonts(root):
    """Pakai font terbaik yang ada di sistem (Segoe UI di Windows)."""
    families = set(tkfont.families(root))

    def pick(candidates, fallback):
        return next((f for f in candidates if f in families), fallback)

    ui = pick(
        (
            "Segoe UI",
            "SF Pro Text",
            "Helvetica Neue",
            "Inter",
            "Noto Sans",
            "DejaVu Sans",
        ),
        "Helvetica",
    )
    mono = pick(("Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono"), "Courier")
    for key, spec in FONTS.items():
        FONTS[key] = ((mono if key == "mono" else ui),) + tuple(spec[1:])


def _text_on(hex_color: str) -> str:
    """Warna teks yang terbaca di atas hex_color."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return "#14161a" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "white"


def center_on(win, parent, w=None, h=None):
    win.update_idletasks()
    ww, wh = w or win.winfo_reqwidth(), h or win.winfo_reqheight()
    x = max(parent.winfo_rootx() + (parent.winfo_width() - ww) // 2, 0)
    y = max(parent.winfo_rooty() + (parent.winfo_height() - wh) // 2, 0)
    win.geometry(f"{ww}x{wh}+{x}+{y}" if (w and h) else f"+{x}+{y}")


class StyledButton(tk.Button):
    """Tombol flat dengan hover + state disabled yang rapi."""

    def __init__(
        self, parent, text, command=None, style="primary", padx=14, pady=6, **kw
    ):
        self._style = BUTTON_STYLES.get(style, BUTTON_STYLES["primary"])
        super().__init__(
            parent,
            text=text,
            command=command,
            bg=self._style["bg"],
            fg=self._style["fg"],
            activebackground=self._style["hover"],
            activeforeground=self._style["fg"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=padx,
            pady=pady,
            font=FONTS["ui"],
            cursor="hand2",
            **kw,
        )
        self.bind("<Enter>", self._hover_on)
        self.bind("<Leave>", self._hover_off)
        if str(self["state"]) == "disabled":
            self._apply_state()

    def _hover_on(self, _e):
        if str(self["state"]) != "disabled":
            self.config(bg=self._style["hover"])

    def _hover_off(self, _e):
        if str(self["state"]) != "disabled":
            self.config(bg=self._style["bg"])

    def configure(self, **kw):
        tk.Button.configure(self, **kw)
        if "state" in kw:
            self._apply_state()

    config = configure

    def _apply_state(self):
        if str(self["state"]) == "disabled":
            tk.Button.configure(
                self,
                bg=COLORS["surface"],
                fg=COLORS["text_subtle"],
                activebackground=COLORS["surface"],
                activeforeground=COLORS["text_subtle"],
                cursor="arrow",
            )
        else:
            tk.Button.configure(
                self,
                bg=self._style["bg"],
                fg=self._style["fg"],
                activebackground=self._style["hover"],
                activeforeground=self._style["fg"],
                cursor="hand2",
            )


def vsep(parent, pad=10):
    wrap = tk.Frame(parent, bg=parent["bg"])
    tk.Frame(wrap, bg=COLORS["border_strong"], width=1, height=20).pack(padx=pad)
    return wrap


def hline(parent, bg=None, pady=0):
    f = tk.Frame(parent, bg=bg or COLORS["border"], height=1)
    f.pack(fill=tk.X, pady=pady)
    return f


def lbl(parent, text, **kw):
    kw.setdefault("bg", COLORS["bg"])
    kw.setdefault("fg", COLORS["text"])
    kw.setdefault("font", FONTS["ui"])
    return tk.Label(parent, text=text, **kw)


def section_label(parent, text, bg):
    return tk.Label(
        parent, text=text, bg=bg, fg=COLORS["text_subtle"], font=FONTS["section"]
    )


def make_entry(parent, **kw):
    return tk.Entry(
        parent,
        bg=COLORS["surface_alt"],
        fg=COLORS["text"],
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=COLORS["border_strong"],
        highlightcolor=COLORS["accent"],
        insertbackground=COLORS["text"],
        font=FONTS["ui"],
        selectbackground=COLORS["accent_soft"],
        selectforeground=COLORS["text"],
        **kw,
    )


def make_spinbox(parent, **kw):
    return ttk.Spinbox(parent, style="TSpinbox", font=FONTS["ui"], **kw)


def make_slider(parent, text, var, lo, hi, step, bg):
    """Slider dengan label dan angka di kanan; nilai dibulatkan ke 'step'."""
    wrap = tk.Frame(parent, bg=bg)
    head = tk.Frame(wrap, bg=bg)
    head.pack(fill=tk.X)
    tk.Label(head, text=text, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"]).pack(
        side=tk.LEFT
    )
    value = tk.Label(head, bg=bg, fg=COLORS["text"], font=FONTS["bold"])
    value.pack(side=tk.RIGHT)

    def snap(_v=None):
        current = var.get()
        snapped = round(round(current / step) * step, 2)
        if abs(snapped - current) > 1e-9:
            var.set(snapped)
        value.config(text=f"{snapped:.2f}")

    ttk.Scale(
        wrap, from_=lo, to=hi, variable=var, command=snap, style="Horizontal.TScale"
    ).pack(fill=tk.X, pady=(SP["xs"], 0))
    snap()
    return wrap


# ============================================================
# DIALOGS
# ============================================================
class SplitDialog(tk.Toplevel):
    """Dialog input persentase train/valid/test + seed. Hasil di self.result."""

    DEFAULTS = (70, 20, 10)

    def __init__(self, parent, total_images: int):
        super().__init__(parent)
        self.title("Split Dataset")
        self.resizable(False, False)
        self.transient(parent)
        self.configure(bg=COLORS["bg"])
        self.result: tuple[dict[str, float], int] | None = None

        self.vars = {
            s: tk.StringVar(value=str(v)) for s, v in zip(SPLITS, self.DEFAULTS)
        }
        self.seed_var = tk.StringVar(value="42")

        body = tk.Frame(self, padx=SP["xl"] + 4, pady=SP["xl"], bg=COLORS["bg"])
        body.pack()

        lbl(
            body,
            f"{total_images} gambar di '{POOL}' akan diacak dan\n"
            f"dipindahkan ke train / valid / test.",
            justify="left",
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, SP["lg"]))

        for row, subset in enumerate(SPLITS, start=1):
            lbl(body, subset).grid(row=row, column=0, sticky="w", pady=3)
            make_spinbox(
                body, from_=0, to=100, width=6, textvariable=self.vars[subset]
            ).grid(row=row, column=1, padx=8, pady=3)
            lbl(body, "%", fg=COLORS["text_muted"]).grid(row=row, column=2, sticky="w")

        self.total_label = lbl(body, "", font=FONTS["bold"])
        self.total_label.grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(SP["sm"], 0)
        )

        lbl(body, "Seed").grid(row=5, column=0, sticky="w", pady=(SP["lg"], 0))
        make_entry(body, width=8, textvariable=self.seed_var).grid(
            row=5, column=1, padx=8, pady=(SP["lg"], 0), ipady=3
        )

        buttons = tk.Frame(body, bg=COLORS["bg"])
        buttons.grid(row=6, column=0, columnspan=3, pady=(SP["xl"] + 4, 0), sticky="e")
        StyledButton(buttons, "Batal", self.destroy, style="secondary", width=8).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        StyledButton(buttons, "Split", self._on_ok, style="primary", width=8).pack(
            side=tk.LEFT
        )

        for var in self.vars.values():
            var.trace_add("write", lambda *_: self._update_total())
        self._update_total()

        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Return>", lambda e: self._on_ok())
        center_on(self, parent)
        self.wait_visibility()
        self.grab_set()

    def _parse_percents(self) -> dict[str, float] | None:
        try:
            return {s: float(v.get()) for s, v in self.vars.items()}
        except ValueError:
            return None

    def _update_total(self):
        percents = self._parse_percents()
        if percents is None:
            self.total_label.config(text="Total: -", fg=COLORS["danger_text"])
            return
        total = sum(percents.values())
        ok = abs(total - 100) < 1e-6
        self.total_label.config(
            text=f"Total: {total:g}%",
            fg=COLORS["success"] if ok else COLORS["danger_text"],
        )

    def _on_ok(self):
        percents = self._parse_percents()
        if percents is None:
            messagebox.showerror(
                "Input tidak valid", "Persentase harus berupa angka.", parent=self
            )
            return
        if (
            any(v < 0 for v in percents.values())
            or abs(sum(percents.values()) - 100) > 1e-6
        ):
            messagebox.showerror(
                "Input tidak valid", "Total persentase harus tepat 100.", parent=self
            )
            return
        try:
            seed = int(self.seed_var.get())
        except ValueError:
            messagebox.showerror(
                "Input tidak valid", "Seed harus berupa bilangan bulat.", parent=self
            )
            return
        self.result = (percents, seed)
        self.destroy()


class TrainingConfigDialog(tk.Toplevel):
    """Dialog konfigurasi training. Setiap parameter disertai penjelasan."""

    CHOICES = {
        "model": YOLO_MODELS,
        "optimizer": [
            "auto",
            "SGD",
            "MuSGD",
            "Adam",
            "Adamax",
            "AdamW",
            "NAdam",
            "RAdam",
            "RMSProp",
        ],
        "cache": ["False", "ram", "disk"],
        "save_period": ["-1", "1", "5", "10", "25"],
        "close_mosaic": ["0", "5", "10", "15", "20"],
    }
    # Dropdown yang tetap boleh diketik karena nilainya bisa bebas (mis. fraksi VRAM)
    EDITABLE = {"batch": ["-1", "0.7", "0.9", "4", "8", "16", "32", "64"]}

    @staticmethod
    def _device_choices() -> list[str]:
        values = ["auto", "cpu"]
        try:
            import torch

            n = torch.cuda.device_count()
            values += [str(i) for i in range(n)]
            if n > 1:
                values.append(",".join(str(i) for i in range(n)))
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                values.append("mps")
        except Exception:
            values.append("0")
        return values

    def __init__(
        self,
        parent,
        project,
        title="Konfigurasi Training",
        submit_text="Mulai Training",
        preset=None,
    ):
        super().__init__(parent)
        self.title(f"{title} - {project.name}")
        self.submit_text = submit_text
        self.preset = dict(preset or {})
        self.extra = {k: v for k, v in self.preset.items() if k in AUG_KEYS}
        self.minsize(720, 600)
        self.transient(parent)
        self.configure(bg=COLORS["bg"])
        self.result = None
        self.vars = {}
        self._build(project)
        center_on(self, parent, 860, 770)
        self.grab_set()

    def _build(self, project):
        outer = tk.Frame(self, padx=SP["xl"], pady=SP["lg"], bg=COLORS["bg"])
        outer.pack(fill=tk.BOTH, expand=True)
        lbl(
            outer,
            f"Dataset: {project.root / 'data.yaml'}",
            anchor="w",
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(fill=tk.X, pady=(0, SP["md"]))
        hline(outer)

        # Tombol dipack dulu (side=BOTTOM) supaya tidak terpotong saat jendela mengecil
        footer = tk.Frame(outer, bg=COLORS["bg"])
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        hline(footer, pady=(0, SP["md"]))
        StyledButton(footer, "Batal", self.destroy, style="secondary", width=12).pack(
            side=tk.RIGHT
        )
        StyledButton(
            footer, self.submit_text, self._submit, style="success", width=16
        ).pack(side=tk.RIGHT, padx=8)
        StyledButton(
            footer, "Muat hasil tuning...", self._load_preset, style="ghost"
        ).pack(side=tk.LEFT)

        area = tk.Frame(outer, bg=COLORS["bg"])
        area.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(area, highlightthickness=0, bg=COLORS["bg"])
        scrollbar = ttk.Scrollbar(area, orient="vertical", command=canvas.yview)
        body = tk.Frame(canvas, bg=COLORS["bg"])
        win = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind(
            "<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        body.columnconfigure(2, weight=1)

        def wheel(event):
            if event.num == 4 or event.delta > 0:
                canvas.yview_scroll(-2, "units")
            else:
                canvas.yview_scroll(2, "units")

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind(seq, wheel)

        bool_fields = {"amp", "pretrained", "cos_lr", "deterministic", "val", "plots"}

        def divider(r):
            tk.Frame(body, bg=COLORS["border"], height=1).grid(
                row=r, column=0, columnspan=3, sticky="ew", pady=(SP["md"], 0)
            )

        r = 0
        for key, default in DEFAULT_CONFIG.items():
            if key == "resume_from":
                continue
            default = self.preset.get(key, default)
            lbl(body, key, font=FONTS["bold"], anchor="w").grid(
                row=r, column=0, sticky="nw", padx=(2, 16), pady=(SP["md"], 0)
            )
            if key in bool_fields:
                var = tk.BooleanVar(value=default)
                widget = ttk.Checkbutton(body, variable=var, style="TCheckbutton")
            elif key == "device" or key in self.CHOICES or key in self.EDITABLE:
                if key == "device":
                    values = self._device_choices()
                else:
                    values = list(self.CHOICES.get(key) or self.EDITABLE[key])
                current = str(default)
                if key == "device" and not current:
                    current = "auto"
                if current and current not in values:
                    values.insert(0, current)  # nilai dari preset/config tetap tersedia
                var = tk.StringVar(value=current)
                widget = ttk.Combobox(
                    body,
                    textvariable=var,
                    values=values,
                    state="normal" if key in self.EDITABLE else "readonly",
                    width=24,
                    font=FONTS["ui"],
                )
            else:
                var = tk.StringVar(value=str(default))
                widget = make_entry(body, textvariable=var, width=26)
            if isinstance(widget, ttk.Combobox):
                # Cegah scroll mouse mengubah nilai dropdown secara tidak sengaja
                widget.bind("<MouseWheel>", lambda e: (wheel(e), "break")[1])
            widget.grid(
                row=r,
                column=1,
                sticky="nw",
                pady=(SP["md"], 0),
                ipady=0 if isinstance(widget, (ttk.Checkbutton, ttk.Combobox)) else 3,
            )
            self.vars[key] = var
            lbl(
                body,
                CONFIG_HELP[key],
                justify="left",
                anchor="w",
                wraplength=440,
                fg=COLORS["text_muted"],
                font=FONTS["small"],
            ).grid(row=r, column=2, sticky="nw", padx=12, pady=(SP["md"], 0))
            divider(r + 1)
            r += 2

        lbl(body, "resume_from", font=FONTS["bold"], anchor="w").grid(
            row=r, column=0, sticky="nw", padx=(2, 16), pady=(SP["md"], 0)
        )
        resume_var = tk.StringVar(value="")
        self.vars["resume_from"] = resume_var
        resume_frame = tk.Frame(body, bg=COLORS["bg"])
        resume_frame.grid(row=r, column=1, sticky="nw", pady=(SP["md"], 0))
        make_entry(resume_frame, textvariable=resume_var, width=20).pack(
            side=tk.LEFT, ipady=3
        )

        def pick_checkpoint():
            chosen = filedialog.askopenfilename(
                title="Pilih checkpoint last.pt",
                filetypes=[("Checkpoint PyTorch", "*.pt"), ("Semua file", "*.*")],
            )
            resume_var.set(chosen)

        StyledButton(
            resume_frame,
            "Pilih...",
            pick_checkpoint,
            style="secondary",
            padx=10,
            pady=3,
        ).pack(side=tk.LEFT, padx=(6, 0))
        lbl(
            body,
            CONFIG_HELP["resume_from"],
            justify="left",
            anchor="w",
            wraplength=440,
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).grid(row=r, column=2, sticky="nw", padx=12, pady=(SP["md"], 0))

    def _submit(self):
        config = {}
        for key, var in self.vars.items():
            config[key] = var.get()
        if config["dataset_version"] == ORIGINAL_DATASET:
            config["dataset_version"] = ""
        config.update(self.extra)  # parameter augmentasi dari hasil tuning
        if config.get("device") == "auto":
            config["device"] = ""
        try:
            config["batch"] = parse_batch(config["batch"])
        except ValueError as exc:
            messagebox.showerror(
                "Konfigurasi tidak valid", f"batch tidak valid: {exc}", parent=self
            )
            return
        for key in (
            "epochs",
            "imgsz",
            "workers",
            "patience",
            "save_period",
            "close_mosaic",
            "seed",
        ):
            try:
                config[key] = int(config[key])
            except (TypeError, ValueError):
                messagebox.showerror(
                    "Konfigurasi tidak valid",
                    f"{key} harus berupa bilangan bulat.",
                    parent=self,
                )
                return
        for key in (
            "lr0",
            "lrf",
            "momentum",
            "weight_decay",
            "warmup_epochs",
            "fraction",
        ):
            try:
                config[key] = float(config[key])
            except (TypeError, ValueError):
                messagebox.showerror(
                    "Konfigurasi tidak valid",
                    f"{key} harus berupa angka.",
                    parent=self,
                )
                return
        try:
            from training import validate_config

            validate_config(config)
        except Exception as exc:
            messagebox.showerror("Konfigurasi tidak valid", str(exc), parent=self)
            return
        self.result = config
        self.destroy()

    def _load_preset(self):
        """Isi form dari best_config.json hasil Auto Tune."""
        path = filedialog.askopenfilename(
            title="Pilih best_config.json",
            filetypes=[("Config JSON", "*.json"), ("Semua file", "*.*")],
            parent=self,
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Gagal membaca config", str(exc), parent=self)
            return
        for key, var in self.vars.items():
            if key in ("resume_from",) or key not in data:
                continue
            value = data[key]
            if key == "device" and not value:
                value = "auto"
            if key == "dataset_version":
                value = value or ORIGINAL_DATASET
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(str(value))
        self.extra = {k: data[k] for k in AUG_KEYS if k in data}
        messagebox.showinfo(
            "Config dimuat",
            f"Form diisi dari hasil tuning.\n{len(self.extra)} parameter augmentasi "
            "ikut dipakai saat training.",
            parent=self,
        )


class TuningConfigDialog(tk.Toplevel):
    """Dialog Auto Tune yang disederhanakan. Hanya menanyakan model, imgsz, epochs.
    Hasil di self.result = {"base": {...}, "tune": {...}}."""

    MODELS = YOLO_MODELS
    IMGSZ_CHOICES = [512, 640, 768, 1024]
    EPOCH_CHOICES = [50, 100, 150, 200, 300]

    def __init__(self, parent, project):
        super().__init__(parent)
        self.title(f"Auto Tune - {project.name}")
        self.resizable(False, False)
        self.transient(parent)
        self.configure(bg=COLORS["bg"])
        self.result = None
        self.vars = {}

        body = tk.Frame(self, padx=SP["xl"] + 4, pady=SP["xl"], bg=COLORS["bg"])
        body.pack()

        lbl(
            body,
            "Hyperparameter (learning rate, optimizer, augmentasi) dicari otomatis.\n"
            "Anda hanya perlu memilih model, ukuran gambar, dan jumlah epoch.",
            anchor="w",
            justify="left",
            wraplength=500,
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(fill=tk.X, pady=(0, SP["lg"]))

        # Model
        lbl(body, "Model", font=FONTS["bold"], anchor="w").pack(
            anchor="w", pady=(SP["md"], 0)
        )
        model_var = tk.StringVar(value=self.MODELS[0])
        model_combo = ttk.Combobox(
            body,
            textvariable=model_var,
            values=self.MODELS,
            state="readonly",
            width=24,
            font=FONTS["ui"],
        )
        model_combo.pack(anchor="w", pady=(SP["sm"], SP["md"]))
        self.vars["model"] = model_var

        # imgsz
        lbl(body, "Ukuran Gambar (imgsz)", font=FONTS["bold"], anchor="w").pack(
            anchor="w"
        )
        imgsz_var = tk.StringVar(value=str(640))
        imgsz_combo = ttk.Combobox(
            body,
            textvariable=imgsz_var,
            values=[str(v) for v in self.IMGSZ_CHOICES],
            state="readonly",
            width=24,
            font=FONTS["ui"],
        )
        imgsz_combo.pack(anchor="w", pady=(SP["sm"], SP["md"]))
        self.vars["imgsz"] = imgsz_var

        # Epochs
        lbl(body, "Epoch", font=FONTS["bold"], anchor="w").pack(anchor="w")
        epoch_var = tk.StringVar(value=str(100))
        epoch_combo = ttk.Combobox(
            body,
            textvariable=epoch_var,
            values=[str(v) for v in self.EPOCH_CHOICES],
            state="readonly",
            width=24,
            font=FONTS["ui"],
        )
        epoch_combo.pack(anchor="w", pady=(SP["sm"], SP["xl"]))
        self.vars["epochs"] = epoch_var

        # Buttons
        buttons = tk.Frame(body, bg=COLORS["bg"])
        buttons.pack(fill=tk.X, pady=(SP["lg"], 0))
        StyledButton(buttons, "Batal", self.destroy, style="secondary", width=10).pack(
            side=tk.RIGHT
        )
        StyledButton(
            buttons, "Lanjut...", self._submit, style="success", width=16
        ).pack(side=tk.RIGHT, padx=8)

        self.bind("<Escape>", lambda e: self.destroy())
        center_on(self, parent)
        self.grab_set()

    def _submit(self):
        base = {
            "model": self.vars["model"].get(),
            "imgsz": int(self.vars["imgsz"].get()),
            "epochs": int(self.vars["epochs"].get()),
            "dataset_version": "",  # pakai dataset asli
        }
        tune = {
            "n_trials": 100,
            "timeout_hours": 0.0,
            "trial_epochs": 40,
            "trial_fraction": 1.0,  # FIX: pakai seluruh data
            "trial_patience": 15,
            "search_augmentation": True,
            "models": [],
            "imgsz_choices": [],
            "speed_weight": 0.02,
            "tune_name": "",
            "tune_resume": "",
            "baseline_first": True,
            "keep_trial_dirs": False,
            "sampler_seed": 42,
        }
        self.result = {"base": base, "tune": tune}
        self.destroy()


class HardwareProfileDialog(tk.Toplevel):
    """Pilih profil pemakaian hardware. Hasil di self.result =
    {"profile": nama, "config": {device, batch, workers, cache, amp, deterministic}}."""

    CUSTOM_FIELDS = (
        (
            "device",
            "Perangkat: kosong = otomatis, 0 = GPU pertama, 0,1 = multi-GPU, cpu, atau mps (Apple).",
        ),
        (
            "batch",
            "Gambar per batch. Bilangan bulat = tetap (mis. 16). -1 = auto (60% VRAM). "
            "Pecahan 0-1 = target pemakaian VRAM (0.7 = 70%).",
        ),
        (
            "workers",
            "Jumlah worker pemuat data. Lebih banyak = loading lebih cepat tetapi memakai CPU dan RAM lebih besar.",
        ),
        (
            "cache",
            "False = tanpa cache, ram = simpan gambar di RAM (cepat, butuh RAM besar), "
            "disk = file .npy di samping gambar (cepat, butuh ruang disk).",
        ),
        (
            "amp",
            "Automatic Mixed Precision. Mempercepat GPU dan menghemat VRAM; hanya berlaku di GPU CUDA.",
        ),
        (
            "deterministic",
            "Operasi deterministik untuk hasil yang bisa diulang. Mati = lebih cepat tetapi hasil bisa sedikit berbeda.",
        ),
    )

    def __init__(self, parent, project, base_config, n_images):
        super().__init__(parent)
        self.title(f"Profil Hardware - {project.name}")
        self.resizable(False, False)
        self.transient(parent)
        self.configure(bg=COLORS["bg"])
        self.result = None
        self.base = dict(base_config)
        self.n_images = n_images
        self.hw = None
        self._detected = []
        self.profile = tk.StringVar(value="medium")
        self.cards, self.value_labels, self.custom_vars = {}, {}, {}

        body = tk.Frame(self, padx=SP["xl"] + 4, pady=SP["xl"], bg=COLORS["bg"])
        body.pack()
        section_label(body, "HARDWARE TERDETEKSI", COLORS["bg"]).pack(anchor="w")
        self.hw_label = lbl(
            body,
            "Mendeteksi hardware...",
            fg=COLORS["accent"],
            font=FONTS["small"],
            justify="left",
            wraplength=640,
        )
        self.hw_label.pack(anchor="w", pady=(SP["xs"], SP["lg"]))
        section_label(body, "PROFIL PEMAKAIAN RESOURCE", COLORS["bg"]).pack(
            anchor="w", pady=(0, SP["sm"])
        )
        for name in PROFILE_ORDER:
            self._build_card(body, name, *PROFILE_TEXT[name])

        buttons = tk.Frame(body, bg=COLORS["bg"])
        buttons.pack(fill=tk.X, pady=(SP["lg"], 0))
        StyledButton(buttons, "Batal", self.destroy, style="secondary", width=10).pack(
            side=tk.RIGHT
        )
        StyledButton(
            buttons, "Mulai Tuning", self._submit, style="success", width=16
        ).pack(side=tk.RIGHT, padx=8)

        self.profile.trace_add("write", lambda *_: self._on_profile())
        self._on_profile()
        self.bind("<Escape>", lambda e: self.destroy())
        center_on(self, parent)
        self.wait_visibility()
        self.grab_set()

        threading.Thread(
            target=lambda: self._detected.append(detect()),
            daemon=True,
            name="YOLO-DetectHW",
        ).start()
        self.after(100, self._poll_hw)

    # ---------- UI ----------
    def _build_card(self, parent, name, title, desc):
        card = tk.Frame(
            parent,
            bg=COLORS["bg"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            padx=SP["lg"],
            pady=SP["md"],
        )
        card.pack(fill=tk.X, pady=SP["xs"] + 1)
        self.cards[name] = card
        ttk.Radiobutton(
            card, text=title, value=name, variable=self.profile, style="TRadiobutton"
        ).pack(anchor="w")
        text = lbl(
            card,
            desc,
            justify="left",
            anchor="w",
            wraplength=620,
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        )
        text.pack(fill=tk.X, padx=(22, 0))
        text.bind("<Button-1>", lambda e, n=name: self.profile.set(n))
        if name != "custom":
            value = tk.Label(
                card,
                text="...",
                anchor="w",
                justify="left",
                wraplength=620,
                bg=COLORS["bg"],
                fg=COLORS["text"],
                font=FONTS["mono"],
            )
            value.pack(fill=tk.X, padx=(22, 0), pady=(SP["sm"], 0))
            self.value_labels[name] = value
            return

        self.custom_frame = tk.Frame(card, bg=COLORS["bg"])
        self.custom_frame.columnconfigure(2, weight=1)
        for r, (key, help_text) in enumerate(self.CUSTOM_FIELDS):
            lbl(self.custom_frame, key, font=FONTS["bold"], anchor="w").grid(
                row=r, column=0, sticky="nw", padx=(0, 12), pady=(SP["md"], 0)
            )
            if key in ("amp", "deterministic"):
                var = tk.BooleanVar(value=False)
                widget = ttk.Checkbutton(
                    self.custom_frame, variable=var, style="TCheckbutton"
                )
                ipady = 0
            elif key == "cache":
                var = tk.StringVar(value="False")
                widget = ttk.Combobox(
                    self.custom_frame,
                    textvariable=var,
                    values=["False", "ram", "disk"],
                    state="readonly",
                    width=12,
                    font=FONTS["ui"],
                )
                ipady = 0
            else:
                var = tk.StringVar(value="")
                widget = make_entry(self.custom_frame, textvariable=var, width=14)
                ipady = 3
            widget.grid(row=r, column=1, sticky="nw", pady=(SP["md"], 0), ipady=ipady)
            self.custom_vars[key] = var
            lbl(
                self.custom_frame,
                help_text,
                justify="left",
                anchor="w",
                wraplength=400,
                fg=COLORS["text_muted"],
                font=FONTS["small"],
            ).grid(row=r, column=2, sticky="nw", padx=12, pady=(SP["md"], 0))

    def _on_profile(self):
        selected = self.profile.get()
        for name, card in self.cards.items():
            card.config(
                highlightbackground=COLORS["accent"]
                if name == selected
                else COLORS["border"]
            )
        if selected == "custom":
            self.custom_frame.pack(fill=tk.X, padx=(22, 0), pady=(SP["sm"], 0))
        else:
            self.custom_frame.pack_forget()

    # ---------- Deteksi hardware ----------
    def _poll_hw(self):
        if not self.winfo_exists():
            return
        if not self._detected:
            self.after(100, self._poll_hw)
            return
        self.hw = self._detected[0]
        self.hw_label.config(
            text=self.hw.summary(),
            fg=COLORS["text"] if self.hw.accelerator != "cpu" else COLORS["warning"],
        )
        imgsz = int(self.base.get("imgsz", 640))
        for name, label in self.value_labels.items():
            label.config(
                text=describe(resolve_profile(name, self.hw, self.n_images, imgsz))
            )
        medium = resolve_profile("medium", self.hw, self.n_images, imgsz)
        for key, var in self.custom_vars.items():
            value = medium[key]
            var.set(bool(value) if isinstance(var, tk.BooleanVar) else str(value))

    # ---------- Hasil ----------
    def _read_custom(self) -> dict:
        v = self.custom_vars
        return {
            "device": v["device"].get().strip(),
            "batch": parse_batch(v["batch"].get()),
            "workers": int(v["workers"].get()),
            "cache": v["cache"].get(),
            "amp": bool(v["amp"].get()),
            "deterministic": bool(v["deterministic"].get()),
        }

    def _submit(self):
        if self.hw is None:
            messagebox.showinfo(
                "Tunggu sebentar", "Deteksi hardware belum selesai.", parent=self
            )
            return
        name = self.profile.get()
        try:
            if name == "custom":
                cfg = self._read_custom()
            else:
                cfg = resolve_profile(
                    name, self.hw, self.n_images, int(self.base.get("imgsz", 640))
                )
            from training import validate_config

            validate_config({**self.base, **cfg})
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Konfigurasi tidak valid", str(exc), parent=self)
            return
        self.result = {"profile": name, "config": cfg}
        self.destroy()


class AutoDetectDialog(tk.Toplevel):
    """Dialog Auto Detect. Hasil di self.result = (scope, conf, iou); scope 'view' atau 'all'."""

    def __init__(self, parent, project, current_view: str, default_conf: float):
        super().__init__(parent)
        self.title("Auto Detect")
        self.resizable(False, False)
        self.transient(parent)
        self.configure(bg=COLORS["bg"])
        self.result: tuple[str, float, float] | None = None

        counts = {s: len(project.list_images(s)) for s in ALL_SUBSETS}
        self.scope = tk.StringVar(value="view")
        self.conf_var = tk.StringVar(value=f"{default_conf:.2f}")
        self.iou_var = tk.StringVar(value="0.70")

        body = tk.Frame(self, padx=SP["xl"] + 4, pady=SP["xl"], bg=COLORS["bg"])
        body.pack()
        section_label(body, "CAKUPAN GAMBAR", COLORS["bg"]).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, SP["sm"])
        )
        ttk.Radiobutton(
            body,
            variable=self.scope,
            value="view",
            style="TRadiobutton",
            text=f"Tampilan saat ini: '{current_view}' ({counts[current_view]} gambar)",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Radiobutton(
            body,
            variable=self.scope,
            value="all",
            style="TRadiobutton",
            text=f"Semua subset ({sum(counts.values())} gambar)",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=2)

        section_label(body, "PARAMETER", COLORS["bg"]).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(SP["xl"], SP["sm"])
        )
        for row, (label, var, hint) in enumerate(
            (
                ("Confidence", self.conf_var, "minimal skor deteksi"),
                ("IoU (NMS)", self.iou_var, "gabung box yang tumpang tindih"),
            ),
            start=4,
        ):
            lbl(body, label).grid(row=row, column=0, sticky="w", pady=3)
            make_spinbox(
                body,
                from_=0.01,
                to=0.99,
                increment=0.05,
                format="%.2f",
                width=6,
                textvariable=var,
            ).grid(row=row, column=1, padx=8, pady=3)
            lbl(body, hint, fg=COLORS["text_muted"], font=FONTS["small"]).grid(
                row=row, column=2, sticky="w"
            )

        note = tk.Frame(body, bg=COLORS["surface_alt"])
        note.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(SP["xl"], 0))
        lbl(
            note,
            "Kebijakan: SKIP. Gambar yang sudah punya label (file label tidak kosong) "
            "dilewati dan tidak diubah. Hanya gambar tanpa label yang diisi hasil deteksi.",
            justify="left",
            anchor="w",
            wraplength=380,
            bg=COLORS["surface_alt"],
            fg=COLORS["text_muted"],
            font=FONTS["small"],
            padx=SP["lg"],
            pady=SP["md"],
        ).pack(fill=tk.X)

        buttons = tk.Frame(body, bg=COLORS["bg"])
        buttons.grid(row=7, column=0, columnspan=3, pady=(SP["xl"], 0), sticky="e")
        StyledButton(buttons, "Batal", self.destroy, style="secondary", width=8).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        StyledButton(buttons, "Mulai", self._on_ok, style="primary", width=8).pack(
            side=tk.LEFT
        )

        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Return>", lambda e: self._on_ok())
        center_on(self, parent)
        self.wait_visibility()
        self.grab_set()

    def _on_ok(self):
        try:
            conf, iou = float(self.conf_var.get()), float(self.iou_var.get())
        except ValueError:
            messagebox.showerror(
                "Input tidak valid",
                "Confidence dan IoU harus berupa angka.",
                parent=self,
            )
            return
        if not (0 < conf < 1 and 0 < iou <= 1):
            messagebox.showerror(
                "Input tidak valid",
                "Confidence harus 0-1 dan IoU harus >0 sampai 1.",
                parent=self,
            )
            return
        self.result = (self.scope.get(), conf, iou)
        self.destroy()


class ShortcutsDialog(tk.Toplevel):
    """Daftar pintasan keyboard & mouse."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Pintasan")
        self.resizable(False, False)
        self.transient(parent)
        self.configure(bg=COLORS["bg"])

        body = tk.Frame(self, padx=SP["xl"] + 4, pady=SP["xl"], bg=COLORS["bg"])
        body.pack()
        for row, (keys, action) in enumerate(SHORTCUTS):
            tk.Label(
                body,
                text=keys,
                font=FONTS["mono"],
                anchor="w",
                bg=COLORS["surface_alt"],
                fg=COLORS["text"],
                padx=8,
                pady=2,
            ).grid(row=row, column=0, sticky="ew", pady=3)
            lbl(body, action, fg=COLORS["text_muted"]).grid(
                row=row, column=1, sticky="w", padx=(SP["lg"], 0)
            )
        StyledButton(body, "Tutup", self.destroy, style="secondary", width=8).grid(
            row=len(SHORTCUTS), column=0, columnspan=2, sticky="e", pady=(SP["xl"], 0)
        )

        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Return>", lambda e: self.destroy())
        self.bind("<F1>", lambda e: self.destroy())
        center_on(self, parent)


class PreviewDialog(tk.Toplevel):
    """Contoh hasil preprocessing/augmentasi pada satu gambar (box ikut digambar)."""

    THUMB = (300, 230)

    def __init__(self, parent, samples):
        super().__init__(parent)
        self.title("Pratinjau")
        self.transient(parent)
        self.configure(bg=COLORS["bg"])
        self._photos = []

        body = tk.Frame(self, padx=SP["xl"], pady=SP["xl"], bg=COLORS["bg"])
        body.pack()
        for i, (caption, img, boxes) in enumerate(samples):
            cell = tk.Frame(body, bg=COLORS["bg"])
            cell.grid(
                row=i // 3, column=i % 3, padx=SP["md"], pady=SP["md"], sticky="n"
            )
            shot = img.copy()
            draw = ImageDraw.Draw(shot)
            width = max(2, round(max(shot.size) / 200))
            for b in boxes:
                draw.rectangle(
                    [b.x1, b.y1, b.x2, b.y2],
                    outline=PALETTE[b.cls % len(PALETTE)],
                    width=width,
                )
            shot.thumbnail(self.THUMB, Image.LANCZOS)
            photo = ImageTk.PhotoImage(shot)
            self._photos.append(photo)
            tk.Label(
                cell,
                image=photo,
                bg=COLORS["canvas_bg"],
                width=self.THUMB[0],
                height=self.THUMB[1],
            ).pack()
            lbl(
                cell,
                f"{caption}  ·  {len(boxes)} box",
                fg=COLORS["text_muted"],
                font=FONTS["small"],
            ).pack(anchor="w", pady=(SP["sm"], 0))

        rows = (len(samples) + 2) // 3
        StyledButton(body, "Tutup", self.destroy, style="secondary", width=8).grid(
            row=rows, column=0, columnspan=3, sticky="e", pady=(SP["lg"], 0)
        )
        self.bind("<Escape>", lambda e: self.destroy())
        center_on(self, parent)


class PreprocessDialog(tk.Toplevel):
    """Preprocessing + augmentasi ala Roboflow. Hasil di self.result = (pre, aug)."""

    def __init__(self, parent, project, sample):
        super().__init__(parent)
        self.title(f"Preprocessing & Augmentasi - {project.name}")
        self.minsize(720, 560)
        self.transient(parent)
        self.configure(bg=COLORS["bg"])
        self.result = None
        self.vars = {}
        self.sample = sample  # (gambar PIL, daftar Box) atau None
        self._row = 0
        self._build(project)
        center_on(self, parent, 860, 760)
        self.grab_set()

    def _build(self, project):
        outer = tk.Frame(self, padx=SP["xl"], pady=SP["lg"], bg=COLORS["bg"])
        outer.pack(fill=tk.BOTH, expand=True)

        counts = {s: len(project.list_images(s)) for s in SPLITS}
        info = (
            "Sumber: "
            + "  ·  ".join(f"{s} {n}" for s, n in counts.items())
            + f" gambar.  Hasil disimpan sebagai versions/{project.next_version_name()}; "
            "dataset asli tidak diubah."
        )
        lbl(
            outer,
            info,
            anchor="w",
            justify="left",
            wraplength=800,
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(fill=tk.X, pady=(0, SP["md"]))
        hline(outer)

        footer = tk.Frame(outer, bg=COLORS["bg"])
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        hline(footer, pady=(0, SP["md"]))
        StyledButton(footer, "Pratinjau", self._preview, style="ghost").pack(
            side=tk.LEFT
        )
        StyledButton(footer, "Batal", self.destroy, style="secondary", width=12).pack(
            side=tk.RIGHT
        )
        StyledButton(
            footer, "Buat Versi", self._submit, style="success", width=16
        ).pack(side=tk.RIGHT, padx=8)

        area = tk.Frame(outer, bg=COLORS["bg"])
        area.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(area, highlightthickness=0, bg=COLORS["bg"])
        scrollbar = ttk.Scrollbar(area, orient="vertical", command=canvas.yview)
        body = tk.Frame(canvas, bg=COLORS["bg"])
        win = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind(
            "<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        body.columnconfigure(2, weight=1)

        def wheel(event):
            canvas.yview_scroll(
                -2 if (event.num == 4 or event.delta > 0) else 2, "units"
            )

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind(seq, wheel)

        self._section(body, "PREPROCESSING  ·  semua subset")
        self._fields(body, DEFAULT_PREPROCESS)
        self._section(body, "AUGMENTASI  ·  hanya train")
        self._fields(body, DEFAULT_AUGMENT)

    def _section(self, body, title):
        top = SP["xl"] if self._row else SP["md"]
        section_label(body, title, COLORS["bg"]).grid(
            row=self._row, column=0, columnspan=3, sticky="w", pady=(top, 0)
        )
        self._row += 1

    def _fields(self, body, defaults):
        for key, default in defaults.items():
            r = self._row
            lbl(body, key, font=FONTS["bold"], anchor="w").grid(
                row=r, column=0, sticky="nw", padx=(2, 16), pady=(SP["md"], 0)
            )
            ipady = 3
            if isinstance(default, bool):
                var = tk.BooleanVar(value=default)
                widget = ttk.Checkbutton(body, variable=var, style="TCheckbutton")
                ipady = 0
            elif key in PREPROCESS_CHOICES:
                var = tk.StringVar(value=default)
                widget = ttk.Combobox(
                    body,
                    textvariable=var,
                    values=PREPROCESS_CHOICES[key],
                    state="readonly",
                    width=24,
                    font=FONTS["ui"],
                )
                ipady = 0
            else:
                var = tk.StringVar(value=str(default))
                widget = make_entry(body, textvariable=var, width=26)
            widget.grid(row=r, column=1, sticky="nw", pady=(SP["md"], 0), ipady=ipady)
            self.vars[key] = var
            lbl(
                body,
                PREPROCESS_HELP[key],
                justify="left",
                anchor="w",
                wraplength=420,
                fg=COLORS["text_muted"],
                font=FONTS["small"],
            ).grid(row=r, column=2, sticky="nw", padx=12, pady=(SP["md"], 0))
            tk.Frame(body, bg=COLORS["border"], height=1).grid(
                row=r + 1, column=0, columnspan=3, sticky="ew", pady=(SP["md"], 0)
            )
            self._row += 2

    def _collect(self):
        raw = {k: v.get() for k, v in self.vars.items()}
        pre = coerce_config(raw, DEFAULT_PREPROCESS)
        aug = coerce_config(raw, DEFAULT_AUGMENT)
        validate_preprocess(pre)
        validate_augment(aug)
        if aug["enabled"] and not active_augmentations(aug):
            raise ValueError(
                "Augmentasi aktif, tetapi belum ada jenis augmentasi yang dipilih."
            )
        return pre, aug

    def _preview(self):
        if self.sample is None:
            messagebox.showinfo(
                "Belum ada gambar",
                "Buka sebuah gambar di editor dulu supaya bisa dipratinjau.",
                parent=self,
            )
            return
        try:
            pre, aug = self._collect()
            samples = make_preview(
                *self.sample, pre, aug, n=5, seed=random.randrange(10**6)
            )
        except ValueError as exc:
            messagebox.showerror("Konfigurasi tidak valid", str(exc), parent=self)
            return
        except Exception as exc:
            messagebox.showerror(
                "Pratinjau gagal", f"{type(exc).__name__}: {exc}", parent=self
            )
            return
        PreviewDialog(self, samples)

    def _submit(self):
        try:
            self.result = self._collect()
        except ValueError as exc:
            messagebox.showerror("Konfigurasi tidak valid", str(exc), parent=self)
            return
        self.destroy()


class BulkLabelDialog(tk.Toplevel):
    """Grid view kumpulan gambar untuk labeling cepat ala Roboflow.
    Menampilkan thumbnail, filter subset/status, dan bulk action."""

    THUMB_SIZE = 160
    COLS = 6
    PAGE_SIZE = 60  # kelipatan COLS (6) supaya baris penuh
    STATUS_ALL = "Semua"
    STATUS_UNLABELED = "Belum dilabeli"
    STATUS_LABELED = "Sudah dilabeli"
    STATUS_NULL = "Null"

    def __init__(self, parent, project):
        super().__init__(parent)
        self.title(f"Bulk Label - {project.name}")
        self.minsize(1000, 600)
        self.geometry("1200x780")
        self.transient(parent)
        self.configure(bg=COLORS["bg"])
        self.project = project
        self._photos = []  # referensi agar tidak di-GC
        self._items = []  # list (subset, path, status, box_count)
        self._filtered = []
        self._selected = set()  # index di _filtered
        self._page = 0
        self._loading = False
        self._stop_load = threading.Event()

        self._build_ui()
        self._load_items()
        self.bind("<Escape>", lambda e: self.destroy())
        center_on(self, parent, 1200, 780)
        self.grab_set()

    def _build_ui(self):
        outer = tk.Frame(self, padx=SP["xl"], pady=SP["lg"], bg=COLORS["bg"])
        outer.pack(fill=tk.BOTH, expand=True)

        # ===== Header: filter =====
        header = tk.Frame(outer, bg=COLORS["surface"])
        header.pack(fill=tk.X, pady=(0, SP["md"]))
        header.pack_propagate(False)
        header.config(height=60)

        inner = tk.Frame(header, bg=COLORS["surface"])
        inner.pack(fill=tk.BOTH, expand=True, padx=SP["md"])

        # Subset filter
        lbl(inner, "Subset:", fg=COLORS["text_muted"], font=FONTS["small"]).pack(
            side=tk.LEFT, padx=(0, SP["sm"])
        )
        self.subset_var = tk.StringVar(value=POOL)
        subset_combo = ttk.Combobox(
            inner,
            textvariable=self.subset_var,
            values=[POOL] + list(SPLITS),  # <--- FIX: ubah SPLITS menjadi list
            state="readonly",
            width=10,
            font=FONTS["ui"],
        )
        subset_combo.pack(side=tk.LEFT, padx=(0, SP["lg"]))
        subset_combo.bind("<<ComboboxSelected>>", lambda _e: self._load_items())

        # Status filter
        lbl(inner, "Status:", fg=COLORS["text_muted"], font=FONTS["small"]).pack(
            side=tk.LEFT, padx=(0, SP["sm"])
        )
        self.status_var = tk.StringVar(value=self.STATUS_ALL)
        status_combo = ttk.Combobox(
            inner,
            textvariable=self.status_var,
            values=[
                self.STATUS_ALL,
                self.STATUS_UNLABELED,
                self.STATUS_LABELED,
                self.STATUS_NULL,
            ],
            state="readonly",
            width=18,
            font=FONTS["ui"],
        )
        status_combo.pack(side=tk.LEFT, padx=(0, SP["lg"]))
        status_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_filter())

        # Search
        lbl(inner, "Cari:", fg=COLORS["text_muted"], font=FONTS["small"]).pack(
            side=tk.LEFT, padx=(0, SP["sm"])
        )
        self.search_var = tk.StringVar()
        search_entry = make_entry(inner, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.LEFT, ipady=3)
        search_entry.bind("<KeyRelease>", lambda _e: self._apply_filter())

        # Info count
        self.count_label = lbl(inner, "", fg=COLORS["text_muted"], font=FONTS["small"])
        self.count_label.pack(side=tk.RIGHT)

        # ===== Grid area =====
        grid_outer = tk.Frame(outer, bg=COLORS["bg"])
        grid_outer.pack(fill=tk.BOTH, expand=True)

        self._canvas = canvas = tk.Canvas(
            grid_outer, highlightthickness=0, bg=COLORS["bg"]
        )
        scrollbar = ttk.Scrollbar(grid_outer, orient="vertical", command=canvas.yview)
        self.grid_body = tk.Frame(canvas, bg=COLORS["bg"])
        win = canvas.create_window((0, 0), window=self.grid_body, anchor="nw")
        self.grid_body.bind(
            "<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def wheel(event):
            canvas.yview_scroll(
                -2 if (event.num == 4 or event.delta > 0) else 2, "units"
            )

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind(seq, wheel)

        # ===== Footer: bulk actions =====
        footer = tk.Frame(outer, bg=COLORS["surface"])
        footer.pack(fill=tk.X, pady=(SP["md"], 0))
        footer.pack_propagate(False)
        footer.config(height=56)

        inner_footer = tk.Frame(footer, bg=COLORS["surface"])
        inner_footer.pack(fill=tk.BOTH, expand=True, padx=SP["md"])

        self.selected_label = lbl(
            inner_footer,
            "0 gambar dipilih",
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        )
        self.selected_label.pack(side=tk.LEFT)
        pager = tk.Frame(inner_footer, bg=COLORS["surface"])
        pager.pack(side=tk.LEFT, padx=SP["xl"])
        self.first_btn = StyledButton(
            pager, "«", lambda: self._goto_page(0), style="ghost", padx=8, pady=3
        )
        self.first_btn.pack(side=tk.LEFT)
        self.prev_btn = StyledButton(
            pager,
            "‹",
            lambda: self._goto_page(self._page - 1),
            style="ghost",
            padx=10,
            pady=3,
        )
        self.prev_btn.pack(side=tk.LEFT, padx=(SP["xs"], 0))
        self.page_label = lbl(
            pager,
            "Halaman 1 / 1",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=FONTS["small"],
        )
        self.page_label.pack(side=tk.LEFT, padx=SP["md"])
        self.next_btn = StyledButton(
            pager,
            "›",
            lambda: self._goto_page(self._page + 1),
            style="ghost",
            padx=10,
            pady=3,
        )
        self.next_btn.pack(side=tk.LEFT, padx=(0, SP["xs"]))
        self.last_btn = StyledButton(
            pager,
            "»",
            lambda: self._goto_page(self._page_count() - 1),
            style="ghost",
            padx=8,
            pady=3,
        )
        self.last_btn.pack(side=tk.LEFT)
        StyledButton(
            pager,
            "Pilih halaman",
            self._select_page,
            style="ghost",
            padx=8,
            pady=3,
        ).pack(side=tk.LEFT, padx=(SP["lg"], 0))
        StyledButton(
            pager,
            "Pilih semua",
            self._select_all,
            style="ghost",
            padx=8,
            pady=3,
        ).pack(side=tk.LEFT, padx=(SP["xs"], 0))
        StyledButton(
            pager, "Kosongkan", self._select_none, style="ghost", padx=8, pady=3
        ).pack(side=tk.LEFT, padx=(SP["xs"], 0))
        btn_frame = tk.Frame(inner_footer, bg=COLORS["surface"])
        btn_frame.pack(side=tk.RIGHT)

        StyledButton(
            btn_frame, "Tandai Null", self._bulk_null, style="secondary", padx=10
        ).pack(side=tk.LEFT, padx=(0, SP["sm"]))
        StyledButton(
            btn_frame, "Pindahkan ke...", self._bulk_move, style="secondary", padx=10
        ).pack(side=tk.LEFT, padx=(0, SP["sm"]))
        StyledButton(
            btn_frame, "Hapus", self._bulk_delete, style="danger", padx=10
        ).pack(side=tk.LEFT, padx=(0, SP["sm"]))
        StyledButton(
            btn_frame, "Muat Ulang", self._load_items, style="ghost", padx=10
        ).pack(side=tk.LEFT)

    def _load_items(self):
        """Muat daftar gambar subset terpilih (threaded). Thumbnail TIDAK dibuat di sini."""
        self._stop_load.set()
        self._stop_load = stop = threading.Event()
        self._loading = True
        subset = self.subset_var.get()
        self._items = []
        self._filtered = []
        self._selected.clear()
        self._update_selected_label()
        self._clear_grid("Memuat daftar gambar...")

        def worker():
            items = []
            for path in self.project.list_images(subset):
                if stop.is_set():
                    return
                label_file = self.project.label_path(subset, path)
                if label_file.is_file():
                    try:
                        with open(label_file, "r", encoding="utf-8") as f:
                            lines = [l.strip() for l in f if l.strip()]
                        box_count = sum(1 for l in lines if l and l[0].isdigit())
                        is_null = len(lines) > 0 and box_count == 0
                    except Exception:
                        box_count = 0
                        is_null = False
                    status = self.STATUS_NULL if is_null else self.STATUS_LABELED
                else:
                    box_count = 0
                    status = self.STATUS_UNLABELED
                items.append((subset, path, status, box_count))
            if not stop.is_set():
                self.after(0, lambda: self._on_items_loaded(items))

        threading.Thread(target=worker, daemon=True, name="BulkLabel-Load").start()

    def _on_items_loaded(self, items):
        if not self.winfo_exists():
            return
        self._items = items
        self._loading = False
        self._apply_filter()

    def _apply_filter(self):
        status = self.status_var.get()
        search = self.search_var.get().lower()

        self._filtered = [
            it
            for it in self._items
            if (status == self.STATUS_ALL or it[2] == status)
            and (not search or search in it[1].name.lower())
        ]
        self.count_label.config(text=f"{len(self._filtered)} gambar")
        self._selected.clear()
        self._update_selected_label()
        self._page = 0
        self._render_grid()

    # ---------- Pagination ----------
    def _page_count(self) -> int:
        return max(1, -(-len(self._filtered) // self.PAGE_SIZE))

    def _goto_page(self, page: int):
        page = min(max(page, 0), self._page_count() - 1)
        if page == self._page:
            return
        self._page = page
        self._render_grid()

    def _update_pager(self):
        pages = self._page_count()
        self.page_label.config(text=f"Halaman {self._page + 1} / {pages}")
        at_first, at_last = self._page <= 0, self._page >= pages - 1
        self.first_btn.config(state=tk.DISABLED if at_first else tk.NORMAL)
        self.prev_btn.config(state=tk.DISABLED if at_first else tk.NORMAL)
        self.next_btn.config(state=tk.DISABLED if at_last else tk.NORMAL)
        self.last_btn.config(state=tk.DISABLED if at_last else tk.NORMAL)

    def _page_range(self) -> range:
        start = self._page * self.PAGE_SIZE
        return range(start, min(start + self.PAGE_SIZE, len(self._filtered)))

    def _select_page(self):
        self._selected.update(self._page_range())
        self._update_selected_label()
        self._render_grid(scroll_top=False)

    def _select_all(self):
        self._selected = set(range(len(self._filtered)))
        self._update_selected_label()
        self._render_grid(scroll_top=False)

    def _select_none(self):
        self._selected.clear()
        self._update_selected_label()
        self._render_grid(scroll_top=False)

    # ---------- Render ----------
    def _clear_grid(self, message: str | None = None):
        for child in self.grid_body.winfo_children():
            child.destroy()
        self._photos.clear()
        if message:
            lbl(
                self.grid_body,
                message,
                fg=COLORS["text_muted"],
                font=FONTS["small"],
            ).pack(pady=SP["xl"])

    def _render_grid(self, scroll_top: bool = True):
        """Render HANYA thumbnail pada halaman aktif."""
        self._page = min(max(self._page, 0), self._page_count() - 1)
        self._update_pager()

        if not self._filtered:
            self._clear_grid("Tidak ada gambar yang cocok dengan filter.")
            return
        self._clear_grid()

        indices = self._page_range()
        for row_start in range(indices.start, indices.stop, self.COLS):
            row_frame = tk.Frame(self.grid_body, bg=COLORS["bg"])
            row_frame.pack(fill=tk.X, pady=SP["xs"])
            for idx in range(row_start, min(row_start + self.COLS, indices.stop)):
                cell = tk.Frame(
                    row_frame, bg=COLORS["bg"], padx=SP["xs"], pady=SP["xs"]
                )
                cell.pack(side=tk.LEFT, expand=True, fill=tk.X)
                self._render_cell(cell, idx)

        self.update_idletasks()
        if scroll_top:
            self._canvas.yview_moveto(0)

    def _render_cell(self, parent, idx):
        subset, path, status, box_count = self._filtered[idx]

        cell = tk.Frame(parent, bg=COLORS["surface_alt"], padx=4, pady=4)
        cell.pack(fill=tk.BOTH, expand=True)

        chk_frame = tk.Frame(cell, bg=COLORS["surface_alt"])
        chk_frame.pack(fill=tk.X)
        var = tk.BooleanVar(value=idx in self._selected)
        ttk.Checkbutton(
            chk_frame,
            variable=var,
            style="Card.TCheckbutton",
            command=lambda i=idx, v=var: self._toggle_select(i, v),
        ).pack(side=tk.LEFT)

        thumb_frame = tk.Frame(
            cell, bg=COLORS["canvas_bg"], width=self.THUMB_SIZE, height=self.THUMB_SIZE
        )
        thumb_frame.pack(fill=tk.X, pady=(SP["xs"], 0))
        thumb_frame.pack_propagate(False)

        try:
            with Image.open(path) as src:
                iw, ih = src.size
                src.draft("RGB", (self.THUMB_SIZE * 2, self.THUMB_SIZE * 2))
                img = ImageOps.exif_transpose(src).convert("RGB")
            img.thumbnail((self.THUMB_SIZE, self.THUMB_SIZE), Image.LANCZOS)
            if box_count > 0:
                try:
                    label_file = self.project.label_path(subset, path)
                    if label_file.is_file():
                        boxes, _ = load_labels(label_file, iw, ih)
                        sx, sy = img.size[0] / iw, img.size[1] / ih
                        draw = ImageDraw.Draw(img)
                        for b in boxes:
                            draw.rectangle(
                                [b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy],
                                outline=PALETTE[b.cls % len(PALETTE)],
                                width=2,
                            )
                except Exception:
                    pass
            photo = ImageTk.PhotoImage(img)
            self._photos.append(photo)
            tk.Label(thumb_frame, image=photo, bg=COLORS["canvas_bg"]).pack()
        except Exception:
            tk.Label(
                thumb_frame,
                text="Gagal memuat",
                bg=COLORS["canvas_bg"],
                fg=COLORS["danger_text"],
                font=FONTS["tiny"],
            ).pack(expand=True)

        info_frame = tk.Frame(cell, bg=COLORS["surface_alt"])
        info_frame.pack(fill=tk.X, pady=(SP["xs"], 0))

        name = path.name if len(path.name) <= 22 else path.name[:20] + ".."
        tk.Label(
            info_frame,
            text=name,
            bg=COLORS["surface_alt"],
            fg=COLORS["text"],
            font=FONTS["tiny"],
            anchor="w",
        ).pack(fill=tk.X)

        if status == self.STATUS_NULL:
            badge_color, badge_text = COLORS["warning"], "NULL"
        elif status == self.STATUS_UNLABELED:
            badge_color, badge_text = COLORS["text_subtle"], "Unlabeled"
        else:
            badge_color, badge_text = COLORS["success"], f"{box_count} box"
        tk.Label(
            info_frame,
            text=badge_text,
            bg=badge_color,
            fg=_text_on(badge_color),
            font=FONTS["tiny"],
            anchor="w",
        ).pack(fill=tk.X, pady=(2, 0))

        thumb_frame.bind("<Button-1>", lambda e, i=idx: self._open_in_editor(i))
        for w in thumb_frame.winfo_children():
            w.bind("<Button-1>", lambda e, i=idx: self._open_in_editor(i))

    def _toggle_select(self, idx, var):
        if var.get():
            self._selected.add(idx)
        else:
            self._selected.discard(idx)
        self._update_selected_label()

    def _update_selected_label(self):
        n = len(self._selected)
        self.selected_label.config(
            text=f"{n} gambar dipilih",
            fg=COLORS["accent"] if n > 0 else COLORS["text_muted"],
        )

    def _open_in_editor(self, idx):
        """Buka gambar di editor utama."""
        item = self._filtered[idx]
        subset, path, status, box_count = item
        # Set view di app utama
        self.master.view.set(subset)
        self.master.refresh_images()
        # Cari index gambar
        try:
            new_idx = self.master.image_paths.index(path)
            self.master.index = new_idx
            self.master.load_current()
        except ValueError:
            pass
        self.destroy()

    def _get_selected_paths(self):
        return [self._filtered[i] for i in sorted(self._selected)]

    def _bulk_null(self):
        items = self._get_selected_paths()
        if not items:
            messagebox.showinfo(
                "Tidak ada pilihan", "Pilih gambar terlebih dahulu.", parent=self
            )
            return
        # Hanya bisa tandai null jika belum ada box
        eligible = [it for it in items if it[3] == 0]
        if not eligible:
            messagebox.showinfo(
                "Tidak bisa",
                "Gambar terpilih sudah memiliki box. Hapus box dulu.",
                parent=self,
            )
            return
        if not messagebox.askyesno(
            "Tandai Null",
            f"Tandai {len(eligible)} gambar sebagai null (tanpa objek)?",
            parent=self,
        ):
            return
        for subset, path, _, _ in eligible:
            label_file = self.project.label_path(subset, path)
            try:
                label_file.parent.mkdir(parents=True, exist_ok=True)
                label_file.write_text("", encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("Gagal", f"{path.name}: {exc}", parent=self)
                return
        messagebox.showinfo(
            "Selesai", f"{len(eligible)} gambar ditandai null.", parent=self
        )
        self._load_items()

    def _bulk_move(self):
        items = self._get_selected_paths()
        if not items:
            messagebox.showinfo(
                "Tidak ada pilihan", "Pilih gambar terlebih dahulu.", parent=self
            )
            return
        targets = [s for s in [POOL] + list(SPLITS) if s != self.subset_var.get()]
        if not targets:
            messagebox.showinfo("Tidak bisa", "Tidak ada subset tujuan.", parent=self)
            return

        dialog = tk.Toplevel(self)
        dialog.title("Pindahkan ke Subset")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.configure(bg=COLORS["bg"])
        dialog.grab_set()

        body = tk.Frame(dialog, padx=SP["xl"], pady=SP["xl"], bg=COLORS["bg"])
        body.pack()
        lbl(
            body,
            f"Pindahkan {len(items)} gambar ke:",
            fg=COLORS["text"],
            font=FONTS["ui"],
        ).pack(anchor="w", pady=(0, SP["md"]))
        target_var = tk.StringVar(value=targets[0])
        ttk.Combobox(
            body,
            textvariable=target_var,
            values=targets,
            state="readonly",
            width=20,
            font=FONTS["ui"],
        ).pack(anchor="w", pady=(0, SP["lg"]))

        btn_frame = tk.Frame(body, bg=COLORS["bg"])
        btn_frame.pack(fill=tk.X)
        StyledButton(
            btn_frame, "Batal", dialog.destroy, style="secondary", width=8
        ).pack(side=tk.RIGHT)

        def do_move():
            target = target_var.get()
            moved = 0
            for subset, path, _, _ in items:
                if subset == target:
                    continue
                src_label = self.project.label_path(subset, path)
                dst_img = self.project.image_path(target, path.name)
                dst_label = self.project.label_path(target, path)
                try:
                    dst_img.parent.mkdir(parents=True, exist_ok=True)
                    dst_label.parent.mkdir(parents=True, exist_ok=True)
                    import shutil

                    shutil.move(str(path), str(dst_img))
                    if src_label.is_file():
                        shutil.move(str(src_label), str(dst_label))
                    moved += 1
                except OSError as exc:
                    messagebox.showerror("Gagal", f"{path.name}: {exc}", parent=self)
                    return
            dialog.destroy()
            messagebox.showinfo(
                "Selesai", f"{moved} gambar dipindahkan ke '{target}'.", parent=self
            )
            self._load_items()

        StyledButton(btn_frame, "Pindahkan", do_move, style="primary", width=12).pack(
            side=tk.RIGHT, padx=(SP["sm"], 0)
        )
        center_on(dialog, self)

    def _bulk_delete(self):
        items = self._get_selected_paths()
        if not items:
            messagebox.showinfo(
                "Tidak ada pilihan", "Pilih gambar terlebih dahulu.", parent=self
            )
            return
        if not messagebox.askyesno(
            "Hapus Gambar",
            f"Hapus {len(items)} gambar beserta labelnya?\nTindakan ini tidak bisa dibatalkan.",
            parent=self,
        ):
            return
        deleted = 0
        for subset, path, _, _ in items:
            label_file = self.project.label_path(subset, path)
            try:
                if path.is_file():
                    path.unlink()
                if label_file.is_file():
                    label_file.unlink()
                deleted += 1
            except OSError as exc:
                messagebox.showerror("Gagal", f"{path.name}: {exc}", parent=self)
                return
        messagebox.showinfo("Selesai", f"{deleted} gambar dihapus.", parent=self)
        self._load_items()


class VersionProgressDialog(tk.Toplevel):
    """Progress pembuatan versi dataset. on_cancel dipanggil saat pengguna membatalkan."""

    def __init__(self, parent, total: int, on_cancel, title="Membuat Versi Dataset"):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self.configure(bg=COLORS["bg"])
        self.on_cancel = on_cancel

        body = tk.Frame(self, padx=SP["xl"] + 4, pady=SP["xl"], bg=COLORS["bg"])
        body.pack()
        self.info = lbl(body, "Menyiapkan...", anchor="w")
        self.info.pack(fill=tk.X)
        self.bar = ttk.Progressbar(
            body,
            maximum=max(total, 1),
            length=380,
            mode="determinate",
            style="Horizontal.TProgressbar",
        )
        self.bar.pack(fill=tk.X, pady=(SP["md"], SP["lg"]))
        self.cancel_button = StyledButton(
            body, "Batalkan", self._cancel, style="danger"
        )
        self.cancel_button.pack(anchor="e")

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        center_on(self, parent)
        self.wait_visibility()
        self.grab_set()

    def update_progress(self, n: int, total: int):
        self.bar["value"] = n
        self.info.config(text=f"Memproses gambar {n} dari {total}...")

    def _cancel(self):
        self.cancel_button.config(state=tk.DISABLED)
        self.info.config(text="Membatalkan...", fg=COLORS["warning"])
        self.on_cancel()


class HealthCheckDialog(tk.Toplevel):
    """Health Check dataset (mengikuti Roboflow). Hanya menampilkan HealthReport."""

    BAR_W = 220
    MAX_HIST_BUCKETS = 40

    def __init__(self, parent, report, project_name: str):
        super().__init__(parent)
        self.title(f"Health Check - {project_name}")
        self.minsize(760, 560)
        self.transient(parent)
        self.configure(bg=COLORS["bg"])
        self.rep = report
        self._photo = None
        self._build()
        self.bind("<Escape>", lambda e: self.destroy())
        center_on(self, parent, 940, 780)

    # ---------- Kerangka ----------
    def _build(self):
        outer = tk.Frame(self, padx=SP["xl"], pady=SP["lg"], bg=COLORS["bg"])
        outer.pack(fill=tk.BOTH, expand=True)

        footer = tk.Frame(outer, bg=COLORS["bg"])
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        hline(footer, pady=(0, SP["md"]))
        StyledButton(footer, "Tutup", self.destroy, style="secondary", width=12).pack(
            side=tk.RIGHT
        )

        area = tk.Frame(outer, bg=COLORS["bg"])
        area.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(area, highlightthickness=0, bg=COLORS["bg"])
        scrollbar = ttk.Scrollbar(area, orient="vertical", command=canvas.yview)
        body = tk.Frame(canvas, bg=COLORS["bg"])
        win = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind(
            "<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def wheel(event):
            canvas.yview_scroll(
                -2 if (event.num == 4 or event.delta > 0) else 2, "units"
            )

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind(seq, wheel)

        lbl(body, "Dataset Analytics", font=FONTS["header"]).pack(anchor="w")
        lbl(
            body,
            f"Dibuat pada {self.rep.generated_at}",
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(anchor="w", pady=(0, SP["lg"]))

        self._build_stats(body)
        self._build_class_balance(body)
        self._build_dimensions(body)
        self._build_histogram(body)
        self._build_heatmap(body)

    # ---------- Komponen kecil ----------
    def _title(self, body, title, desc):
        section_label(body, title, COLORS["bg"]).pack(
            anchor="w", pady=(SP["xl"] + SP["sm"], 0)
        )
        hline(body, pady=(SP["xs"], SP["xs"]))
        lbl(body, desc, fg=COLORS["text_muted"], font=FONTS["small"]).pack(
            anchor="w", pady=(0, SP["md"])
        )

    def _stat(self, parent, row, col, title, value, sub=""):
        card = tk.Frame(parent, bg=COLORS["surface_alt"], padx=SP["lg"], pady=SP["md"])
        card.grid(
            row=row, column=col, sticky="nsew", padx=(0, SP["md"]), pady=(0, SP["md"])
        )
        bg = COLORS["surface_alt"]
        tk.Label(
            card, text=title, bg=bg, fg=COLORS["text_muted"], font=FONTS["small"]
        ).pack(anchor="w")
        tk.Label(
            card,
            text=value or "-",
            bg=bg,
            fg=COLORS["text"],
            font=(FONTS["header"][0], 18, "bold"),
        ).pack(anchor="w")
        if sub:
            tk.Label(
                card, text=sub, bg=bg, fg=COLORS["text_subtle"], font=FONTS["tiny"]
            ).pack(anchor="w")

    def _hbar(self, parent, value, vmax, color, width=None):
        w = width or self.BAR_W
        c = tk.Canvas(parent, width=w, height=14, bg=COLORS["bg"], highlightthickness=0)
        c.create_rectangle(0, 2, w, 12, fill=COLORS["surface_alt"], outline="")
        if value > 0 and vmax > 0:
            c.create_rectangle(
                0, 2, max(2, int(w * value / vmax)), 12, fill=color, outline=""
            )
        return c

    def _dist_rows(self, parent, rows, color):
        frame = tk.Frame(parent, bg=COLORS["bg"])
        frame.pack(anchor="w")
        vmax = max(n for _, n in rows)
        for i, (label, n) in enumerate(rows):
            lbl(
                frame,
                label,
                fg=COLORS["text_muted"],
                font=FONTS["small"],
                anchor="w",
                width=14,
            ).grid(row=i, column=0, sticky="w")
            self._hbar(frame, n, vmax, color, 360).grid(
                row=i, column=1, padx=SP["md"], pady=2
            )
            lbl(frame, f"{n:,}", font=FONTS["small"]).grid(row=i, column=2, sticky="w")

    def _histogram(self, parent, buckets, w=820, h=220):
        c = tk.Canvas(parent, width=w, height=h, bg=COLORS["bg"], highlightthickness=0)
        left, bottom, top = 48, 26, 12
        vmax = max(v for _, v in buckets) or 1
        n = len(buckets)
        slot = (w - left - 8) / n
        plot_h = h - bottom - top
        axis = COLORS["border_strong"]
        c.create_line(left, top, left, h - bottom, fill=axis)
        c.create_line(left, h - bottom, w - 8, h - bottom, fill=axis)
        c.create_text(
            left - 6,
            top,
            text=f"{vmax:,}",
            anchor="e",
            fill=COLORS["text_muted"],
            font=FONTS["tiny"],
        )
        c.create_text(
            left - 6,
            h - bottom,
            text="0",
            anchor="e",
            fill=COLORS["text_muted"],
            font=FONTS["tiny"],
        )
        step = 1 if n <= 20 else 5
        for i, (label, v) in enumerate(buckets):
            x1 = left + i * slot + slot * 0.15
            x2 = left + (i + 1) * slot - slot * 0.15
            if v:
                c.create_rectangle(
                    x1,
                    h - bottom - plot_h * v / vmax,
                    x2,
                    h - bottom,
                    fill=COLORS["accent"],
                    outline="",
                )
            if i % step == 0 or label.endswith("+"):
                c.create_text(
                    (x1 + x2) / 2,
                    h - bottom + 4,
                    text=label,
                    anchor="n",
                    fill=COLORS["text_muted"],
                    font=FONTS["tiny"],
                )
        return c

    # ---------- Bagian-bagian ----------
    def _build_stats(self, body):
        r = self.rep
        grid = tk.Frame(body, bg=COLORS["bg"])
        grid.pack(fill=tk.X)
        for c in range(3):
            grid.columnconfigure(c, weight=1, uniform="stat")
        subsets = "  ·  ".join(f"{s} {n}" for s, n in r.subset_images.items() if n)
        cards = [
            ("Number of Images", f"{r.num_images:,}", subsets),
            ("Number of Annotations", f"{r.num_annotations:,}", ""),
            ("Average Image Size", f"{r.avg_megapixels:.2f} mp", ""),
            ("Median Image Ratio", r.median_ratio_label, ""),
            ("Missing Annotations", f"{r.missing:,}", "gambar tanpa file label"),
            ("Null Annotations", f"{r.null:,}", "file label ada tapi kosong"),
        ]
        for i, (title, value, sub) in enumerate(cards):
            self._stat(grid, i // 3, i % 3, title, value, sub)
        if r.errors:
            lbl(
                body,
                f"{r.errors} gambar gagal dibaca (pertama: {r.first_error})",
                fg=COLORS["warning"],
                font=FONTS["small"],
            ).pack(anchor="w")

    def _build_class_balance(self, body):
        r = self.rep
        self._title(
            body,
            "CLASS BALANCE",
            "Jumlah anotasi tiap kelas. Tambah gambar untuk kelas yang kurang terwakili.",
        )
        if not r.class_names:
            lbl(body, "Project belum punya kelas.", fg=COLORS["text_muted"]).pack(
                anchor="w"
            )
            return
        cols = ["all", "train", "valid", "test"]
        if r.subset_images.get(POOL):
            cols.append(POOL)

        table = tk.Frame(body, bg=COLORS["bg"])
        table.pack(fill=tk.X)
        heads = ["Kelas", "Sebaran (all)"] + [
            f"{k} ({r.num_images if k == 'all' else r.subset_images.get(k, 0)})"
            for k in cols
        ]
        for c, text in enumerate(heads):
            lbl(table, text, fg=COLORS["text_subtle"], font=FONTS["section"]).grid(
                row=0, column=c, sticky="w", padx=(0, SP["xl"]), pady=(0, SP["sm"])
            )

        all_counts = r.class_counts["all"]
        vmax = max(all_counts) if all_counts else 0
        for i, name in enumerate(r.class_names):
            row = i + 1
            color = PALETTE[i % len(PALETTE)]
            lbl(table, name, fg=color, font=FONTS["bold"]).grid(
                row=row, column=0, sticky="w", padx=(0, SP["xl"]), pady=SP["xs"]
            )
            self._hbar(table, all_counts[i], vmax, color).grid(
                row=row, column=1, sticky="w", padx=(0, SP["xl"])
            )
            for c, key in enumerate(cols, start=2):
                count = r.class_counts[key][i]
                status = balance_status(count, r.class_counts[key])
                cell = tk.Frame(table, bg=COLORS["bg"])
                cell.grid(row=row, column=c, sticky="w", padx=(0, SP["xl"]))
                lbl(cell, f"{count:,}", font=FONTS["bold"]).pack(anchor="w")
                if status == "missing":
                    lbl(
                        cell, "Missing", fg=COLORS["danger_text"], font=FONTS["tiny"]
                    ).pack(anchor="w")
                elif status == "under":
                    lbl(
                        cell,
                        "Under represented",
                        fg=COLORS["warning"],
                        font=FONTS["tiny"],
                    ).pack(anchor="w")
        if r.unknown_class_annotations:
            lbl(
                body,
                f"{r.unknown_class_annotations} anotasi memakai id kelas yang tidak ada di data.yaml.",
                fg=COLORS["warning"],
                font=FONTS["small"],
            ).pack(anchor="w", pady=(SP["md"], 0))

    def _build_dimensions(self, body):
        r = self.rep
        self._title(
            body, "DIMENSION INSIGHTS", "Ukuran dan rasio aspek gambar asli di dataset."
        )
        lbl(body, "Size Distribution", font=FONTS["bold"]).pack(
            anchor="w", pady=(SP["sm"], SP["xs"])
        )
        sizes = sorted(r.size_counts.items(), key=lambda kv: -kv[1])
        if len(sizes) == 1:
            (w, h), _ = sizes[0]
            lbl(body, f"Semua gambar berukuran {w}x{h}.", fg=COLORS["text_muted"]).pack(
                anchor="w"
            )
        else:
            rows = [(f"{w}x{h}", n) for (w, h), n in sizes[:8]]
            rest = sum(n for _, n in sizes[8:])
            if rest:
                rows.append((f"{len(sizes) - 8} ukuran lain", rest))
            self._dist_rows(body, rows, COLORS["accent"])

        lbl(body, "Aspect Ratio Distribution", font=FONTS["bold"]).pack(
            anchor="w", pady=(SP["lg"], SP["xs"])
        )
        rows = sorted(r.aspect_counts.items(), key=lambda kv: -kv[1])
        self._dist_rows(body, rows, COLORS["accent"])

    def _build_histogram(self, body):
        hist = self.rep.objects_hist
        self._title(
            body,
            "HISTOGRAM OF OBJECT COUNT BY IMAGE",
            "Jumlah gambar menurut banyaknya objek di dalamnya (0 = missing/null).",
        )
        top, cap = max(hist), self.MAX_HIST_BUCKETS
        if top < cap:
            buckets = [(str(i), hist.get(i, 0)) for i in range(top + 1)]
        else:
            buckets = [(str(i), hist.get(i, 0)) for i in range(cap)]
            buckets.append((f"{cap}+", sum(v for k, v in hist.items() if k >= cap)))
        self._histogram(body, buckets).pack(anchor="w")

    def _heat_photo(self):
        r = self.rep
        heat = r.heatmap
        peak = float(heat.max())
        t = heat / peak if peak > 0 else heat
        rgb = np.stack(
            [
                np.clip(1.5 - np.abs(4 * t - 3), 0, 1),
                np.clip(1.5 - np.abs(4 * t - 2), 0, 1),
                np.clip(1.5 - np.abs(4 * t - 1), 0, 1),
            ],
            axis=-1,
        )
        img = Image.fromarray((rgb * 255).astype("uint8"))
        mw, mh = r.median_width or 1, r.median_height or 1
        width = 420
        height = max(120, min(560, round(width * mh / mw)))
        return ImageTk.PhotoImage(img.resize((width, height), Image.BICUBIC))

    def _build_heatmap(self, body):
        r = self.rep
        self._title(
            body,
            "ANNOTATION HEATMAP",
            "Lokasi anotasi pada gambar. Merah = paling banyak anotasi, biru = sedikit.",
        )
        if r.num_annotations == 0:
            lbl(body, "Belum ada anotasi.", fg=COLORS["text_muted"]).pack(anchor="w")
            return
        self._photo = self._heat_photo()
        tk.Label(body, image=self._photo, bg=COLORS["canvas_bg"]).pack(anchor="w")
        lbl(
            body,
            f"Maksimum {int(r.heatmap.max()):,} anotasi menutupi satu titik.",
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(anchor="w", pady=(SP["sm"], SP["xl"]))


# ============================================================
# MAIN APP
# ============================================================
class LabelerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        _apply_system_fonts(self)
        self.title("Labeler")
        self.geometry("1280x760")
        self.minsize(1100, 640)
        self.configure(bg=COLORS["bg"])
        self._setup_theme()

        self.project: Project | None = None
        self.training_job = None
        self.tuning_job = None
        self.training_events = queue.Queue()
        self.training_running = False
        self.view = tk.StringVar(value=POOL)
        self.image_paths: list[Path] = []
        self.index = -1
        self.pil_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None

        # Transformasi gambar asli -> canvas
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        # Labeling
        self.classes: list[str] = []
        self.active_class = 0
        self.boxes: list[Box] = []
        self.extra_lines: list[str] = []
        self.is_null = False  # gambar ini sengaja tanpa objek (file label kosong)
        self.selected: int | None = None
        self.undo_stack: list[list[Box]] = []
        self.drag_start: tuple[float, float] | None = None
        self.drag_rect: int | None = None
        self.edit_mode: str | None = (
            None  # "draw", "move", atau nama handle ("nw", "e", ...)
        )
        self.edit_origin: Box | None = None
        self.edit_snapshot: list[Box] = []
        self.edit_start: tuple[float, float] = (0.0, 0.0)
        self.edit_changed = False
        # Label Assist & Auto Detect
        self.detector = Detector()
        self.assist_events = queue.Queue()
        self.assist_req = 0
        self.suggestions: list[tuple[Box, float]] = []
        self.assist_auto = tk.BooleanVar(value=False)
        self.assist_conf = tk.DoubleVar(value=0.25)
        self.detect_running = False
        self.detect_stop: threading.Event | None = None

        # Versi dataset (preprocessing/augmentasi)
        self.version_events = queue.Queue()
        self.version_running = False
        self.version_stop: threading.Event | None = None
        self.version_dialog = None
        # Health Check
        self.health_events = queue.Queue()
        self.health_running = False
        self.health_stop: threading.Event | None = None
        self.health_dialog = None
        self._build_ui()
        self.bind("n", self._on_key(self.toggle_null))
        self.show_image()

    def report_callback_exception(self, exc, val, tb):
        """Tanpa terminal, error tombol/dialog harus tetap terlihat."""
        import traceback

        traceback.print_exception(exc, val, tb)
        where = f"\n\nDetail ada di:\n{LOG_FILE}" if LOG_FILE else ""
        try:
            messagebox.showerror("Terjadi kesalahan", f"{exc.__name__}: {val}{where}")
        except Exception:
            pass

    # ---------- Theme ----------
    def _setup_theme(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        c = COLORS

        style.configure(
            "TCombobox",
            padding=(8, 5),
            font=FONTS["ui"],
            fieldbackground=c["surface_alt"],
            background=c["surface_alt"],
            foreground=c["text"],
            bordercolor=c["surface_alt"],
            lightcolor=c["surface_alt"],
            darkcolor=c["surface_alt"],
            arrowcolor=c["text_muted"],
            arrowsize=14,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", c["surface_alt"])],
            background=[("active", c["surface_hover"])],
            bordercolor=[("focus", c["accent"])],
            selectbackground=[("readonly", c["surface_alt"])],
            selectforeground=[("readonly", c["text"])],
        )
        # Daftar dropdown combobox adalah Listbox biasa, diatur lewat option database
        self.option_add("*TCombobox*Listbox.background", c["surface_alt"])
        self.option_add("*TCombobox*Listbox.foreground", c["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", c["accent_soft"])
        self.option_add("*TCombobox*Listbox.selectForeground", c["text"])
        self.option_add("*TCombobox*Listbox.font", FONTS["ui"])

        style.configure(
            "TSpinbox",
            padding=(6, 4),
            fieldbackground=c["surface_alt"],
            background=c["surface_alt"],
            foreground=c["text"],
            bordercolor=c["border_strong"],
            lightcolor=c["border_strong"],
            darkcolor=c["border_strong"],
            arrowcolor=c["text_muted"],
            insertcolor=c["text"],
        )
        style.map(
            "TSpinbox",
            bordercolor=[("focus", c["accent"])],
            background=[("active", c["surface_hover"])],
        )

        # Scrollbar tipis tanpa panah
        style.layout(
            "Vertical.TScrollbar",
            [
                (
                    "Vertical.Scrollbar.trough",
                    {
                        "sticky": "ns",
                        "children": [
                            (
                                "Vertical.Scrollbar.thumb",
                                {"expand": "1", "sticky": "nswe"},
                            )
                        ],
                    },
                )
            ],
        )
        style.configure(
            "Vertical.TScrollbar",
            width=10,
            gripcount=0,
            relief="flat",
            background=c["border_strong"],
            troughcolor=c["bg"],
            bordercolor=c["bg"],
            lightcolor=c["border_strong"],
            darkcolor=c["border_strong"],
        )
        style.map("Vertical.TScrollbar", background=[("active", c["text_subtle"])])

        for name in ("TCheckbutton", "TRadiobutton"):
            style.configure(
                name,
                background=c["bg"],
                foreground=c["text"],
                font=FONTS["ui"],
                indicatorbackground=c["surface_alt"],
                indicatorforeground=c["on_accent"],
                upperbordercolor=c["border_strong"],
                lowerbordercolor=c["border_strong"],
                focuscolor=c["bg"],
                padding=(0, 2),
            )
            style.map(
                name,
                background=[("active", c["bg"])],
                foreground=[("active", c["text"])],
                indicatorbackground=[
                    ("selected", c["accent"]),
                    ("active", c["surface_hover"]),
                ],
                upperbordercolor=[("selected", c["accent"])],
                lowerbordercolor=[("selected", c["accent"])],
            )
        # Varian di atas kartu (surface_alt)
        for name in ("Card.TCheckbutton",):
            style.configure(
                name,
                background=c["surface_alt"],
                foreground=c["text"],
                font=FONTS["small"],
                indicatorbackground=c["surface"],
                indicatorforeground=c["on_accent"],
                upperbordercolor=c["border_strong"],
                lowerbordercolor=c["border_strong"],
                focuscolor=c["surface_alt"],
                padding=(0, 2),
            )
            style.map(
                name,
                background=[("active", c["surface_alt"])],
                indicatorbackground=[("selected", c["accent"])],
                upperbordercolor=[("selected", c["accent"])],
                lowerbordercolor=[("selected", c["accent"])],
            )

        style.configure(
            "Horizontal.TScale",
            background=c["accent"],
            troughcolor=c["border_strong"],
            bordercolor=c["surface_alt"],
            lightcolor=c["accent"],
            darkcolor=c["accent"],
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=c["surface_alt"],
            background=c["accent"],
            bordercolor=c["surface_alt"],
            lightcolor=c["accent"],
            darkcolor=c["accent"],
        )

    # ---------- UI ----------
    def _build_ui(self):
        self._build_toolbar()
        self._build_status_bars()
        self._build_class_panel()
        self._build_canvas()
        self.after(200, self._poll_training_events)
        self.after(100, self._poll_assist_events)
        self.after(150, self._poll_version_events)
        self.after(130, self._poll_health_events)

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=COLORS["surface"], height=52)
        bar.pack(side=tk.TOP, fill=tk.X)
        bar.pack_propagate(False)
        tk.Frame(self, bg=COLORS["border"], height=1).pack(side=tk.TOP, fill=tk.X)

        inner = tk.Frame(bar, bg=COLORS["surface"])
        inner.pack(fill=tk.BOTH, expand=True, padx=SP["lg"])
        py = (SP["md"] + 1, SP["md"] + 1)

        StyledButton(inner, "Proyek Baru", self.new_project, style="secondary").pack(
            side=tk.LEFT, pady=py
        )
        StyledButton(inner, "Buka", self.open_project, style="ghost").pack(
            side=tk.LEFT, padx=(SP["sm"], 0), pady=py
        )
        vsep(inner).pack(side=tk.LEFT)

        StyledButton(
            inner, "Tambah ke Dataset", self.add_to_dataset, style="secondary"
        ).pack(side=tk.LEFT, pady=py)
        StyledButton(inner, "Split...", self.split_dataset, style="secondary").pack(
            side=tk.LEFT, padx=(SP["sm"], 0), pady=py
        )
        StyledButton(
            inner, "Preprocessing...", self.open_preprocess, style="secondary"
        ).pack(side=tk.LEFT, padx=(SP["sm"], 0), pady=py)
        vsep(inner).pack(side=tk.LEFT)
        # === TAMBAHKAN TOMBOL BULK LABEL DI SINI ===
        StyledButton(inner, "Bulk Label", self.open_bulk_label, style="secondary").pack(
            side=tk.LEFT, padx=(SP["sm"], 0), pady=py
        )

        vsep(inner).pack(side=tk.LEFT)
        StyledButton(
            inner, "Health Check", self.open_health_check, style="secondary"
        ).pack(side=tk.LEFT, padx=(SP["sm"], 0), pady=py)
        self.train_button = StyledButton(
            inner, "Training YOLO...", self.start_training, style="success"
        )
        self.train_button.pack(side=tk.LEFT, pady=py)
        self.tune_button = StyledButton(
            inner, "Auto Tune...", self.start_tuning, style="secondary"
        )
        self.tune_button.pack(side=tk.LEFT, padx=(SP["sm"], 0), pady=py)
        self.stop_button = StyledButton(
            inner, "Stop", self.stop_training, style="danger"
        )
        self.stop_button.config(state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(SP["sm"], 0), pady=py)
        vsep(inner).pack(side=tk.LEFT)

        tk.Label(
            inner,
            text="Lihat",
            bg=COLORS["surface"],
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(side=tk.LEFT, padx=(0, SP["md"]))
        combo = ttk.Combobox(
            inner,
            textvariable=self.view,
            values=ALL_SUBSETS,
            state="readonly",
            width=9,
            font=FONTS["ui"],
        )
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", self.on_view_change)

        StyledButton(inner, "←", self.prev_image, style="secondary", padx=12).pack(
            side=tk.LEFT, padx=(SP["lg"], 0), pady=py
        )
        StyledButton(inner, "→", self.next_image, style="secondary", padx=12).pack(
            side=tk.LEFT, padx=(SP["sm"], 0), pady=py
        )

        StyledButton(inner, "Pintasan  F1", self.show_shortcuts, style="ghost").pack(
            side=tk.RIGHT, pady=py
        )

    def _build_status_bars(self):
        # Status utama (paling bawah)
        main_box = tk.Frame(self, bg=COLORS["surface"])
        main_box.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Frame(main_box, bg=COLORS["border"], height=1).pack(side=tk.TOP, fill=tk.X)

        main = tk.Frame(main_box, bg=COLORS["surface"], height=28)
        main.pack(fill=tk.X)
        main.pack_propagate(False)
        self.status_project = self._status_seg(main, "—", fg=COLORS["text"])
        self._status_div(main)
        self.status_position = self._status_seg(main, "0 / 0")
        self._status_div(main)
        self.status_file = self._status_seg(main, "")
        self._status_div(main)
        self.status_image = self._status_seg(main, "")
        self._status_div(main)
        self.status_counts = self._status_seg(main, "")

        # Training bar (di atas status utama)
        train_box = tk.Frame(self, bg=COLORS["bg"])
        train_box.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Frame(train_box, bg=COLORS["border"], height=1).pack(side=tk.TOP, fill=tk.X)
        train = tk.Frame(train_box, bg=COLORS["bg"], height=24)
        train.pack(fill=tk.X)
        train.pack_propagate(False)
        self.training_status = tk.Label(
            train,
            text="Training: idle",
            anchor="w",
            bg=COLORS["bg"],
            fg=COLORS["text_subtle"],
            font=FONTS["small"],
            padx=SP["lg"],
        )
        self.training_status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Style(self).configure(
            "Thin.Horizontal.TProgressbar",
            troughcolor=COLORS["surface_alt"],
            background=COLORS["accent"],
            bordercolor=COLORS["surface_alt"],
            lightcolor=COLORS["accent"],
            darkcolor=COLORS["accent"],
            thickness=8,
        )
        self.progress_bar = ttk.Progressbar(
            train,
            length=240,
            maximum=100,
            mode="determinate",
            style="Thin.Horizontal.TProgressbar",
        )
        self.progress_pct = tk.Label(
            train,
            text="",
            width=5,
            anchor="e",
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=FONTS["bold"],
        )
        self.progress_detail = tk.Label(
            train,
            text="",
            anchor="e",
            bg=COLORS["bg"],
            fg=COLORS["text_muted"],
            font=FONTS["small"],
            padx=SP["md"],
        )
        self._progress_ref = None  # (waktu, fraksi) saat laporan pertama, untuk ETA

    def _status_seg(self, parent, text, fg=None):
        lbl_ = tk.Label(
            parent,
            text=text,
            bg=COLORS["surface"],
            fg=fg or COLORS["text_muted"],
            font=FONTS["small"],
            padx=SP["lg"],
        )
        lbl_.pack(side=tk.LEFT, fill=tk.Y)
        return lbl_

    def _status_div(self, parent):
        tk.Frame(parent, bg=COLORS["border"], width=1).pack(
            side=tk.LEFT, fill=tk.Y, pady=7
        )

    def _build_class_panel(self):
        panel = tk.Frame(self, width=300, bg=COLORS["surface"])
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)
        tk.Frame(panel, bg=COLORS["border"], width=1).pack(side=tk.LEFT, fill=tk.Y)
        inner = tk.Frame(
            panel, bg=COLORS["surface"], padx=SP["lg"] + 2, pady=SP["lg"] + 2
        )
        inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_assist_panel(inner)

        section_label(inner, "KELAS", COLORS["surface"]).pack(
            anchor="w", pady=(0, SP["md"])
        )

        list_frame = tk.Frame(inner, bg=COLORS["surface"])
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Border & padding dipindah ke wrapper karena Listbox tidak support padx/pady
        list_wrap = tk.Frame(
            list_frame,
            bg=COLORS["bg"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["border_strong"],
            padx=6,
            pady=6,
        )
        list_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.class_list = tk.Listbox(
            list_wrap,
            exportselection=False,
            activestyle="none",
            bg=COLORS["bg"],
            fg=COLORS["text"],
            selectbackground=COLORS["accent_soft"],
            selectforeground=COLORS["text"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            font=FONTS["ui"],
            selectborderwidth=0,
        )
        scroll = ttk.Scrollbar(
            list_frame, command=self.class_list.yview, style="Vertical.TScrollbar"
        )
        self.class_list.config(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(SP["sm"], 0))
        self.class_list.pack(fill=tk.BOTH, expand=True)
        self.class_list.bind("<<ListboxSelect>>", self.on_class_list_select)

        add_row = tk.Frame(inner, bg=COLORS["surface"])
        add_row.pack(fill=tk.X, pady=(SP["md"], 0))
        self.class_entry = make_entry(add_row)
        self.class_entry.pack(
            side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(0, SP["sm"])
        )
        self.class_entry.bind("<Return>", lambda e: self.add_class())
        StyledButton(
            add_row, "Tambah", self.add_class, style="secondary", padx=12, pady=5
        ).pack(side=tk.LEFT)

        actions = tk.Frame(inner, bg=COLORS["surface"])
        actions.pack(fill=tk.X, pady=(SP["lg"], 0))
        StyledButton(
            actions,
            "Ubah kelas box  (R)",
            self.apply_class_to_selected,
            style="secondary",
        ).pack(fill=tk.X, pady=(0, SP["sm"]))
        StyledButton(
            actions, "Hapus box  (Del)", self.delete_selected, style="secondary"
        ).pack(fill=tk.X)
        self.null_button = StyledButton(
            actions, "Tandai null  (N)", self.toggle_null, style="secondary"
        )
        self.null_button.pack(fill=tk.X, pady=(SP["sm"], 0))
        StyledButton(
            actions, "Hapus kelas terpilih", self.delete_class, style="danger"
        ).pack(fill=tk.X, pady=(SP["sm"], 0))

    def _build_assist_panel(self, parent):
        card = tk.Frame(parent, bg=COLORS["surface_alt"])
        card.pack(side=tk.BOTTOM, fill=tk.X, pady=(SP["lg"], 0))
        inner = tk.Frame(card, bg=COLORS["surface_alt"], padx=SP["lg"], pady=SP["lg"])
        inner.pack(fill=tk.X)
        bg = COLORS["surface_alt"]

        section_label(inner, "LABEL ASSIST", bg).pack(anchor="w", pady=(0, SP["sm"]))

        self.model_label = tk.Label(
            inner,
            text="Model: belum dimuat",
            anchor="w",
            justify="left",
            wraplength=240,
            bg=bg,
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        )
        self.model_label.pack(fill=tk.X)

        StyledButton(
            inner, "Muat Model (.pt)...", self.load_model, style="ghost", padx=8, pady=5
        ).pack(fill=tk.X, pady=(SP["md"], 0))

        make_slider(
            inner, "Confidence saran", self.assist_conf, 0.05, 0.95, 0.05, bg
        ).pack(fill=tk.X, pady=(SP["lg"], 0))

        ttk.Checkbutton(
            inner,
            text="Auto-sarankan tiap gambar",
            variable=self.assist_auto,
            command=self.on_auto_assist_toggle,
            style="Card.TCheckbutton",
        ).pack(fill=tk.X, pady=(SP["md"], 0))

        StyledButton(
            inner,
            "Sarankan box  (S)",
            self.request_suggestions,
            style="primary",
            padx=8,
            pady=6,
        ).pack(fill=tk.X, pady=(SP["md"], 0))

        row = tk.Frame(inner, bg=bg)
        row.pack(fill=tk.X, pady=(SP["sm"], 0))
        StyledButton(
            row,
            "Terima semua",
            self.accept_suggestions,
            style="success",
            padx=8,
            pady=6,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        StyledButton(
            row, "Tolak", self.clear_suggestions, style="ghost", padx=14, pady=6
        ).pack(side=tk.LEFT, padx=(SP["sm"], 0))

        hline(inner, bg=COLORS["border_strong"], pady=SP["lg"])

        self.autodetect_button = StyledButton(
            inner,
            "Auto Detect...",
            self.start_auto_detect,
            style="ghost",
            padx=8,
            pady=6,
        )
        self.autodetect_button.pack(fill=tk.X)

        self.detect_stop_button = StyledButton(
            inner,
            "Batalkan Auto Detect",
            self.stop_auto_detect,
            style="danger",
            padx=8,
            pady=6,
        )
        self.detect_stop_button.config(state=tk.DISABLED)
        self.detect_stop_button.pack(fill=tk.X, pady=(SP["sm"], 0))

        self.assist_info = tk.Label(
            inner,
            text="",
            anchor="w",
            justify="left",
            wraplength=240,
            bg=bg,
            fg=COLORS["accent"],
            font=FONTS["small"],
        )
        self.assist_info.pack(fill=tk.X, pady=(SP["sm"], 0))

    DETAIL_HEIGHT = 230

    def _build_canvas(self):
        self.center = tk.Frame(self, bg=COLORS["bg"])
        self.center.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=SP["lg"], pady=SP["lg"]
        )
        self.canvas = tk.Canvas(
            self.center,
            bg=COLORS["canvas_bg"],
            highlightthickness=1,
            highlightbackground=COLORS["canvas_border"],
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.show_image())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_hover)
        self._build_detail_panel(self.center)

    def _build_detail_panel(self, parent):
        bg = COLORS["surface"]
        self.detail = tk.Frame(
            parent,
            bg=bg,
            height=self.DETAIL_HEIGHT,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self.detail.pack_propagate(False)  # belum di-pack; muncul saat training
        body = tk.Frame(self.detail, bg=bg, padx=SP["lg"], pady=SP["md"])
        body.pack(fill=tk.BOTH, expand=True)

        head = tk.Frame(body, bg=bg)
        head.pack(fill=tk.X)
        section_label(head, "DETAIL PROSES", bg).pack(side=tk.LEFT)
        self.detail_stage = tk.Label(
            head, text="", bg=bg, fg=COLORS["text"], font=FONTS["bold"]
        )
        self.detail_stage.pack(side=tk.RIGHT)

        self.detail_bars = {}
        for key, title in (("epoch", "Epoch"), ("batch", "Batch")):
            row = tk.Frame(body, bg=bg)
            row.pack(fill=tk.X, pady=(SP["sm"], 0))
            tk.Label(
                row,
                text=title,
                width=7,
                anchor="w",
                bg=bg,
                fg=COLORS["text_muted"],
                font=FONTS["small"],
            ).pack(side=tk.LEFT)
            bar = ttk.Progressbar(
                row,
                maximum=100,
                mode="determinate",
                style="Thin.Horizontal.TProgressbar",
            )
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=SP["md"])
            val = tk.Label(
                row,
                text="",
                width=16,
                anchor="e",
                bg=bg,
                fg=COLORS["text"],
                font=FONTS["mono"],
            )
            val.pack(side=tk.LEFT)
            self.detail_bars[key] = (bar, val)

        self.detail_metrics = tk.Label(
            body,
            text="",
            anchor="w",
            bg=bg,
            fg=COLORS["text_muted"],
            font=FONTS["mono"],
        )
        self.detail_metrics.pack(fill=tk.X, pady=(SP["sm"], 0))
        self.detail_live = tk.Label(
            body,
            text="",
            anchor="w",
            bg=bg,
            fg=COLORS["accent"],
            font=FONTS["mono"],
        )
        self.detail_live.pack(fill=tk.X, pady=(SP["xs"], 0))
        log_wrap = tk.Frame(body, bg=COLORS["bg"])
        log_wrap.pack(fill=tk.BOTH, expand=True, pady=(SP["sm"], 0))
        self.detail_log = tk.Text(
            log_wrap,
            height=4,
            bg=COLORS["bg"],
            fg=COLORS["text_muted"],
            font=FONTS["mono"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            wrap="word",
            state="disabled",
            padx=6,
            pady=4,
        )
        scroll = ttk.Scrollbar(
            log_wrap, command=self.detail_log.yview, style="Vertical.TScrollbar"
        )
        self.detail_log.config(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.detail_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _detail_show(self):
        self._detail_reset()
        self.detail.pack(
            side=tk.BOTTOM, fill=tk.X, pady=(SP["md"], 0), before=self.canvas
        )

    def _detail_hide(self):
        self.detail.pack_forget()

    def _detail_reset(self):
        self.detail_stage.config(text="")
        self.detail_metrics.config(text="")
        for bar, val in self.detail_bars.values():
            bar.config(value=0)
            val.config(text="")
        self.detail_log.config(state="normal")
        self.detail_log.delete("1.0", "end")
        self.detail_log.config(state="disabled")

    def _detail_log_add(self, text: str, stamp: bool = True):
        t = self.detail_log
        t.config(state="normal")
        prefix = f"{datetime.now():%H:%M:%S}  " if stamp else ""
        t.insert("end", f"{prefix}{text}\n")
        lines = int(t.index("end-1c").split(".")[0])
        if lines > 300:
            t.delete("1.0", f"{lines - 300}.0")
        t.see("end")
        t.config(state="disabled")

    def _detail_console_update(self):
        live, lines = CONSOLE.snapshot()  # selalu dikosongkan agar tidak menumpuk
        if not self.training_running:
            return
        for line in lines:
            self._detail_log_add(line, stamp=False)
        self.detail_live.config(text=live.strip())

    def _detail_update(self, p: dict):
        epoch, epochs = p.get("epoch"), p.get("epochs")
        inner = p.get("inner")
        batch, batches = p.get("batch"), p.get("batches")

        stage = []
        if p.get("kind") == "tune":
            stage.append(f"Trial {p.get('trial', '?')}/{p.get('n_trials', '?')}")
        if epoch is not None and epochs:
            stage.append(f"Epoch {epoch}/{epochs}")
        if batch is not None and batches:
            stage.append(f"Batch {batch}/{batches}")
        self.detail_stage.config(text="  ·  ".join(stage))

        bar, val = self.detail_bars["epoch"]
        if epoch is not None and epochs:
            done = (epoch - 1 + (inner or 0.0)) / epochs
            bar.config(value=min(done, 1.0) * 100)
            val.config(text=f"{done * 100:.1f}%")

        bar, val = self.detail_bars["batch"]
        if inner is not None:
            bar.config(value=min(inner, 1.0) * 100)
            val.config(
                text=f"{batch}/{batches}"
                if batch is not None and batches
                else f"{inner * 100:.0f}%"
            )

        parts = []
        loss = p.get("loss")
        if isinstance(loss, dict):
            parts += [f"{k} {v:.3f}" for k, v in loss.items()]
        if p.get("it_s") is not None:
            parts.append(f"{p['it_s']:.1f} it/s")
        if p.get("best") is not None:
            parts.append(f"terbaik {p['best']:.4f}")
        self.detail_metrics.config(text="   ".join(parts))

    def _bind_keys(self):
        self.bind("<Left>", self._on_key(self.prev_image))
        self.bind("<Right>", self._on_key(self.next_image))
        self.bind("a", self._on_key(self.prev_image))
        self.bind("d", self._on_key(self.next_image))
        self.bind("<Delete>", self._on_key(self.delete_selected))
        self.bind("<BackSpace>", self._on_key(self.delete_selected))
        self.bind("x", self._on_key(self.delete_selected))
        self.bind("r", self._on_key(self.apply_class_to_selected))
        self.bind("<Escape>", self._on_key(self.deselect))
        self.bind("<Control-z>", self._on_key(self.undo))
        self.bind("s", self._on_key(self.request_suggestions))
        self.bind("<Return>", self._on_key(self.accept_suggestions))
        self.bind("<Control-b>", self._on_key(self.open_bulk_label))
        self.bind("<F1>", self._on_key(self.show_shortcuts))
        self.bind(
            "<Shift-Left>",
            self._on_key(lambda: self.nudge_selected(-1 / self.scale, 0)),
        )
        self.bind(
            "<Shift-Right>",
            self._on_key(lambda: self.nudge_selected(1 / self.scale, 0)),
        )
        self.bind(
            "<Shift-Up>", self._on_key(lambda: self.nudge_selected(0, -1 / self.scale))
        )
        self.bind(
            "<Shift-Down>", self._on_key(lambda: self.nudge_selected(0, 1 / self.scale))
        )
        for n in range(1, 10):
            self.bind(str(n), self._on_key(lambda n=n: self.select_class(n - 1)))

    def _typing(self) -> bool:
        try:
            widget = self.focus_get()
        except KeyError:
            return False
        return isinstance(widget, (tk.Entry, ttk.Entry, ttk.Spinbox))

    def _on_key(self, handler):
        def wrapper(_event):
            if not self._typing():
                handler()

        return wrapper

    def show_shortcuts(self):
        ShortcutsDialog(self)

    # ---------- Project ----------
    def new_project(self):
        name = simpledialog.askstring("Proyek Baru", "Nama project:", parent=self)
        if name is None:
            return
        try:
            project = Project.create(name.strip())
        except (ValueError, FileExistsError) as exc:
            messagebox.showerror("Gagal membuat project", str(exc))
            return
        self.set_project(project)

    def open_project(self):
        PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
        folder = filedialog.askdirectory(
            title="Pilih folder project", initialdir=PROJECTS_ROOT, mustexist=True
        )
        if not folder:
            return
        try:
            project = Project.open(Path(folder))
        except ValueError as exc:
            messagebox.showerror("Bukan project yang valid", str(exc))
            return
        self.set_project(project)

    def set_project(self, project: Project):
        self.project = project
        self.title(f"Labeler - {project.name}")
        self.view.set(POOL)
        self.classes = project.read_names()
        self.active_class = 0
        self.refresh_class_list()
        self.refresh_images()
        if self.detector.loaded:
            self.after_idle(self._sync_model_classes)

    def _require_project(self) -> bool:
        if self.project is None:
            messagebox.showinfo(
                "Belum ada project", "Buat atau buka project terlebih dahulu."
            )
            return False
        return True

    # ---------- Dataset: tambah & split ----------
    def add_to_dataset(self):
        if not self._require_project() or self._busy_detecting():
            return
        files = filedialog.askopenfilenames(
            title="Pilih gambar untuk ditambahkan ke dataset",
            filetypes=[
                ("Gambar", " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))),
                ("Semua file", "*.*"),
            ],
        )
        if not files:
            return
        copied, skipped = self.project.add_to_dataset(files)
        self.view.set(POOL)
        self.refresh_images()
        msg = f"{copied} gambar ditambahkan ke dataset ('{POOL}')."
        if skipped:
            msg += f"\n{skipped} file dilewati (format tidak didukung)."
        messagebox.showinfo("Selesai", msg)

    def split_dataset(self):
        if not self._require_project() or self._busy_detecting():
            return
        total = len(self.project.list_images(POOL))
        if total == 0:
            messagebox.showinfo(
                "Dataset kosong",
                f"Tidak ada gambar di '{POOL}'.\nTambahkan gambar dulu.",
            )
            return

        dialog = SplitDialog(self, total)
        self.wait_window(dialog)
        if dialog.result is None:
            return
        percents, seed = dialog.result

        try:
            counts = self.project.split_dataset(percents, seed)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Split gagal", str(exc))
            return

        self.refresh_images()
        lines = [f"{subset}: {counts[subset]} gambar" for subset in SPLITS]
        messagebox.showinfo("Split selesai", f"Seed {seed}\n\n" + "\n".join(lines))

    # ---------- Training YOLO ----------
    def start_training(self):
        if not self._require_project():
            return
        if self.training_running:
            messagebox.showinfo(
                "Training berjalan",
                "Selesaikan atau hentikan training yang sedang berjalan.",
            )
            return
        dialog = TrainingConfigDialog(self, self.project)
        self.wait_window(dialog)
        if dialog.result is None:
            return

        self.training_running = True
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.training_status.config(text="Training: menyiapkan...", fg=COLORS["accent"])
        self._progress_start()

        def status_callback(message):
            self.training_events.put(("status", message))

        def done_callback(result):
            self.training_events.put(("done", result))

        self.training_job = TrainingJob(
            self.project,
            dialog.result,
            on_status=status_callback,
            on_done=done_callback,
            on_progress=lambda p: self.training_events.put(("progress", p)),
        )
        try:
            self.training_job.start()
        except Exception as exc:
            self.training_running = False
            self.train_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self._progress_hide()
            messagebox.showerror("Training gagal dimulai", str(exc))

    # ---------- Progress bar training / tuning ----------
    def _progress_start(self):
        """Tampilkan bar dalam mode 'menyiapkan' (indeterminate) sampai laporan pertama."""
        self._progress_ref = None
        self.progress_bar.config(mode="indeterminate", value=0)
        self.progress_bar.pack(
            side=tk.RIGHT, padx=(0, SP["lg"]), before=self.training_status
        )
        self.progress_pct.pack(side=tk.RIGHT, before=self.training_status)
        self.progress_detail.pack(side=tk.RIGHT, before=self.training_status)
        self.progress_pct.config(text="")
        self.progress_detail.config(text="menyiapkan...")
        self.progress_bar.start(12)
        self._detail_show()

    def _progress_hide(self):
        self.progress_bar.stop()
        for widget in (self.progress_bar, self.progress_pct, self.progress_detail):
            widget.pack_forget()
        self._detail_hide()

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        seconds = int(max(seconds, 0))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}j {m:02d}m"
        return f"{m}m {s:02d}d" if m else f"{s}d"

    def _progress_update(self, p: dict):
        if str(self.progress_bar.cget("mode")) == "indeterminate":
            self.progress_bar.stop()
            self.progress_bar.config(mode="determinate")
        frac = min(max(float(p.get("frac", 0.0)), 0.0), 1.0)
        self.progress_bar.config(value=frac * 100)
        self.progress_pct.config(text=f"{frac * 100:.0f}%")

        if p.get("kind") == "tune":
            # Tuning: tampilkan trial + epoch di dalam trial + batch di dalam epoch
            trial = p.get("trial", "?")
            n_trials = p.get("n_trials", "?")
            epoch = p.get("epoch")
            epochs = p.get("epochs")
            inner = p.get("inner")
            best = p.get("best")

            detail = f"Trial {trial}/{n_trials}"
            if epoch is not None and epochs is not None:
                detail += f"  ·  Epoch {epoch}/{epochs}"
                if inner is not None:
                    detail += f"  ·  Batch {int(inner * 100)}%"
            if best is not None:
                detail += f"  ·  terbaik {best:.4f}"
        else:
            # Training biasa: epoch + batch di dalam epoch
            epoch = p.get("epoch")
            epochs = p.get("epochs")
            inner = p.get("inner")

            # --- DI SINI LETAK PERBAIKANNYA ---
            if epoch is not None and epochs is not None:
                detail = f"Epoch {epoch}/{epochs}"
                if inner is not None:
                    detail += f"  ·  Batch {int(inner * 100)}%"
            else:
                detail = f"Progress {frac * 100:.0f}%"
            # ---------------------------------

        # ETA
        now = time.monotonic()
        if self._progress_ref is None:
            self._progress_ref = (now, frac)
        t0, f0 = self._progress_ref
        if frac - f0 > 0.01:
            eta = (now - t0) * (1 - frac) / (frac - f0)
            detail += f"  ·  sisa ±{self._fmt_duration(eta)}"

        self.progress_detail.config(text=detail)

    def start_tuning(self):
        if not self._require_project():
            return
        if self.training_running:
            messagebox.showinfo(
                "Proses berjalan",
                "Selesaikan atau hentikan training/tuning yang sedang berjalan.",
            )
            return
        try:
            import optuna  # noqa: F401
        except ImportError:
            messagebox.showerror(
                "Optuna belum terpasang", "Jalankan di terminal:\n\npip install optuna"
            )
            return
        dialog = TuningConfigDialog(self, self.project)
        self.wait_window(dialog)
        if dialog.result is None:
            return
        base_cfg = {**DEFAULT_CONFIG, **dialog.result["base"]}
        tune_cfg = dialog.result["tune"]
        n_images = count_train_images(
            self.project,
            base_cfg.get("dataset_version", ""),
            float(tune_cfg["trial_fraction"]),
        )
        hardware = HardwareProfileDialog(self, self.project, base_cfg, n_images)
        self.wait_window(hardware)
        if hardware.result is None:
            return
        profile = hardware.result["profile"]
        config = {**base_cfg, **hardware.result["config"], "hardware_profile": profile}

        self.training_running = True
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.training_status.config(
            text=f"Tuning: menyiapkan... (profil hardware {profile})",
            fg=COLORS["accent"],
        )
        self._progress_start()
        self.tuning_job = TuningJob(
            self.project,
            config,
            tune_cfg,
            on_status=lambda m: self.training_events.put(("status", f"[Tuning] {m}")),
            on_done=lambda r: self.training_events.put(("tune_done", r)),
            on_progress=lambda p: self.training_events.put(("progress", p)),
        )
        try:
            self.tuning_job.start()
        except Exception as exc:
            self.training_running = False
            self.tuning_job = None
            self.train_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self._progress_hide()
            messagebox.showerror("Tuning gagal dimulai", str(exc))

    def stop_training(self):
        job = self.tuning_job or self.training_job
        if job and self.training_running:
            self.training_status.config(
                text="Training: permintaan stop dikirim; menunggu penyimpanan checkpoint...",
                fg=COLORS["warning"],
            )
            self.stop_button.config(state=tk.DISABLED)
            job.request_stop()

    def _poll_training_events(self):
        latest = None  # <-- baru
        try:
            while True:
                event, payload = self.training_events.get_nowait()
                if event == "status":
                    text = str(payload)
                    color = COLORS["accent"]
                    if "berhenti" in text.lower() or "stop" in text.lower():
                        color = COLORS["warning"]
                    if "selesai" in text.lower():
                        color = COLORS["success"]
                    if "gagal" in text.lower():
                        color = COLORS["danger_text"]
                    self.training_status.config(text=f"Training: {text}", fg=color)
                    self._detail_log_add(text)
                elif event == "progress":
                    latest = payload
                elif event == "done":
                    self._progress_hide()
                    self.training_running = False
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    result = payload
                    run_dir = result.get("run_dir") or "-"
                    if result.get("status") == "completed":
                        messagebox.showinfo(
                            "Training selesai",
                            f"Training berhasil.\n\nHasil tersimpan di:\n{run_dir}",
                        )
                    elif result.get("status") == "stopped":
                        messagebox.showwarning(
                            "Training dihentikan",
                            f"Training dihentikan dengan permintaan pengguna.\n"
                            f"Checkpoint/hasil yang sempat tersimpan berada di:\n{run_dir}",
                        )
                    else:
                        messagebox.showerror(
                            "Training gagal",
                            f"{result.get('error', 'Kesalahan tidak diketahui')}\n\n"
                            f"Folder log/hasil parsial:\n{run_dir}",
                        )
                elif event == "tune_done":
                    self._progress_hide()
                    self.training_running = False
                    self.tuning_job = None
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    folder = payload.get("tune_dir") or "-"
                    if payload.get("error"):
                        messagebox.showerror(
                            "Tuning gagal", f"{payload['error']}\n\nFolder: {folder}"
                        )
                    else:
                        best = payload.get("best_score")
                        score = f"{best:.4f}" if best is not None else "-"
                        title = (
                            "Tuning dihentikan"
                            if payload.get("stopped")
                            else "Tuning selesai"
                        )
                        messagebox.showinfo(
                            title,
                            f"Skor terbaik: {score}\n\nHasil di:\n{folder}\n\n"
                            "Buka 'Training YOLO...' lalu klik 'Muat hasil tuning...' "
                            "dan pilih best_config.json untuk melatih dengan config terbaik.",
                        )
        except queue.Empty:
            pass
        try:
            if latest is not None and self.training_running:
                self._progress_update(latest)
                self._detail_update(latest)
            self._detail_console_update()
        finally:
            self.after(200, self._poll_training_events)

    # ---------- Preprocessing & augmentasi -> versi dataset ----------
    def _busy_version(self) -> bool:
        if self.version_running:
            messagebox.showinfo(
                "Pembuatan versi berjalan",
                "Tunggu atau batalkan pembuatan versi dataset dulu.",
            )
            return True
        return False

    def open_preprocess(self):
        if (
            not self._require_project()
            or self._busy_detecting()
            or self._busy_version()
        ):
            return
        if not self.project.list_images("train"):
            messagebox.showinfo(
                "Subset train kosong",
                "Versi dataset dibuat dari train/valid/test.\nJalankan 'Split...' dulu.",
            )
            return
        sample = (
            (self.pil_image, list(self.boxes)) if self.pil_image is not None else None
        )
        dialog = PreprocessDialog(self, self.project, sample)
        self.wait_window(dialog)
        if dialog.result is None:
            return
        self._start_version_build(*dialog.result)

    def open_bulk_label(self):
        if (
            not self._require_project()
            or self._busy_detecting()
            or self._busy_version()
        ):
            return
        total = sum(len(self.project.list_images(s)) for s in ALL_SUBSETS)
        if total == 0:
            messagebox.showinfo(
                "Dataset kosong",
                "Belum ada gambar. Klik 'Tambah ke Dataset' dulu.",
            )
            return
        dialog = BulkLabelDialog(self, self.project)
        self.wait_window(dialog)
        # Refresh setelah dialog ditutup (mungkin ada perubahan)
        self.refresh_images()

    def _start_version_build(self, pre: dict, aug: dict):
        self.version_running = True
        self.version_stop = threading.Event()
        project, stop = self.project, self.version_stop
        total = sum(len(project.list_images(s)) for s in SPLITS)
        self.version_dialog = VersionProgressDialog(self, total, stop.set)

        def worker():
            try:
                stats = build_version(
                    project,
                    pre,
                    aug,
                    stop,
                    progress=lambda n, t, s: self.version_events.put(
                        ("progress", (n, t))
                    ),
                )
                self.version_events.put(("done", stats))
            except Exception as exc:
                self.version_events.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True, name="YOLO-BuildVersion").start()

    def _close_version_dialog(self):
        self.version_running = False
        if self.version_dialog is not None:
            self.version_dialog.destroy()
            self.version_dialog = None

    def _poll_version_events(self):
        try:
            while True:
                event, data = self.version_events.get_nowait()
                if event == "progress":
                    if self.version_dialog is not None:
                        self.version_dialog.update_progress(*data)
                elif event == "done":
                    self._close_version_dialog()
                    self._show_version_summary(data)
                elif event == "error":
                    self._close_version_dialog()
                    messagebox.showerror("Pembuatan versi gagal", data)
        except queue.Empty:
            pass
        self.after(150, self._poll_version_events)

    def _show_version_summary(self, stats: dict):
        if stats["stopped"]:
            messagebox.showinfo(
                "Dibatalkan",
                "Pembuatan versi dibatalkan. Folder versi sementara dihapus.",
            )
            return
        c = stats["counts"]
        lines = [
            f"train: {c['train']} gambar ({stats['augmented']} hasil augmentasi)",
            f"valid: {c['valid']} gambar",
            f"test: {c['test']} gambar",
        ]
        if stats["filtered_null"]:
            lines.append(f"Dibuang (tanpa label): {stats['filtered_null']} gambar")
        if stats["aug_skipped_polygon"]:
            lines.append(
                f"Tidak diaugmentasi (berlabel polygon): {stats['aug_skipped_polygon']} gambar"
            )
        if stats["polygon_dropped"]:
            lines.append(
                f"Baris label polygon dibuang (tidak valid setelah resize): {stats['polygon_dropped']}"
            )
        if stats["aug_skipped_empty"]:
            lines.append(
                f"Salinan dibuang (semua box hilang): {stats['aug_skipped_empty']}"
            )
        if stats["errors"]:
            lines.append(f"Error: {stats['errors']} (pertama: {stats['first_error']})")
        messagebox.showinfo(
            f"Versi {stats['version']} selesai",
            "\n".join(lines) + "\n\nPilih versi ini di dialog 'Training YOLO...'.",
        )

    # ---------- Health Check ----------
    def open_health_check(self):
        if not self._require_project() or self.health_running:
            return
        total = sum(len(self.project.list_images(s)) for s in ALL_SUBSETS)
        if total == 0:
            messagebox.showinfo(
                "Dataset kosong", "Belum ada gambar. Klik 'Tambah ke Dataset' dulu."
            )
            return
        self.health_running = True
        self.health_stop = threading.Event()
        project, stop = self.project, self.health_stop
        self.health_dialog = VersionProgressDialog(
            self, total, stop.set, title="Menganalisis Dataset"
        )

        def worker():
            try:
                report = analyze(
                    project,
                    progress=lambda n, t: self.health_events.put(("progress", (n, t))),
                    stop_event=stop,
                )
                self.health_events.put(("done", (report, project.name)))
            except Exception as exc:
                self.health_events.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True, name="YOLO-HealthCheck").start()

    def _close_health_dialog(self):
        self.health_running = False
        if self.health_dialog is not None:
            self.health_dialog.destroy()
            self.health_dialog = None

    def _poll_health_events(self):
        try:
            while True:
                event, data = self.health_events.get_nowait()
                if event == "progress":
                    if self.health_dialog is not None:
                        self.health_dialog.update_progress(*data)
                elif event == "done":
                    self._close_health_dialog()
                    report, name = data
                    if report.stopped:
                        messagebox.showinfo("Dibatalkan", "Analisis dibatalkan.")
                    elif report.num_images == 0:
                        messagebox.showinfo(
                            "Tidak ada gambar", "Tidak ada gambar yang bisa dibaca."
                        )
                    else:
                        HealthCheckDialog(self, report, name)
                elif event == "error":
                    self._close_health_dialog()
                    messagebox.showerror("Analisis gagal", data)
        except queue.Empty:
            pass
        self.after(130, self._poll_health_events)

    # ---------- Model (dipakai Label Assist & Auto Detect) ----------
    def _busy_detecting(self) -> bool:
        if self.detect_running:
            messagebox.showinfo(
                "Auto Detect berjalan",
                "Tunggu atau batalkan Auto Detect terlebih dahulu.",
            )
            return True
        return False

    def load_model(self):
        if self._busy_detecting():
            return
        latest = find_latest_best(self.project.runs_dir) if self.project else None
        initial = (
            latest.parent
            if latest
            else (
                self.project.runs_dir
                if self.project and self.project.runs_dir.is_dir()
                else Path.cwd()
            )
        )
        path = filedialog.askopenfilename(
            title="Pilih model YOLO (.pt)",
            initialdir=initial,
            initialfile=latest.name if latest else "",
            filetypes=[("Model YOLO", "*.pt"), ("Semua file", "*.*")],
        )
        if not path:
            return
        self.model_label.config(text="Memuat model...", fg=COLORS["accent"])

        def worker():
            try:
                self.detector.load(Path(path))
                self.assist_events.put(("model_ok", None))
            except Exception as exc:
                self.assist_events.put(("model_err", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True, name="YOLO-LoadModel").start()

    def _sync_model_classes(self):
        if self.project is None or not self.detector.loaded:
            return
        missing = missing_classes(self.detector.names, self.classes)
        if not missing:
            return
        if not messagebox.askyesno(
            "Kelas model belum ada di project",
            "Kelas berikut ada di model tetapi belum ada di project:\n\n"
            + ", ".join(missing)
            + "\n\nTambahkan ke project?\nJika 'Tidak', deteksi kelas tersebut diabaikan.",
        ):
            return
        for name in missing:
            self.classes = self.project.add_class(name)
        self.refresh_class_list()

    # ---------- Label Assisting ----------
    def on_auto_assist_toggle(self):
        if not self.assist_auto.get():
            return
        if not self.detector.loaded:
            messagebox.showinfo(
                "Model belum dimuat", "Muat model dulu lewat 'Muat Model (.pt)...'."
            )
            self.assist_auto.set(False)
            return
        self.request_suggestions(silent=True)

    def request_suggestions(self, silent: bool = False):
        if self.pil_image is None or self.detect_running:
            return
        if not self.detector.loaded:
            if not silent:
                messagebox.showinfo(
                    "Model belum dimuat", "Muat model dulu lewat 'Muat Model (.pt)...'."
                )
            return
        if not self.classes:
            if not silent:
                messagebox.showinfo(
                    "Belum ada kelas", "Tambahkan kelas dulu di panel kanan."
                )
            return

        self.assist_req += 1
        req_id, image, lookup = (
            self.assist_req,
            self.pil_image,
            build_lookup(self.classes),
        )
        conf = float(self.assist_conf.get())
        self.assist_info.config(text="Mencari saran...", fg=COLORS["accent"])

        def worker():
            try:
                dets = self.detector.predict(
                    image, conf, should_cancel=lambda: req_id != self.assist_req
                )
                if dets is not None:
                    self.assist_events.put(("suggest", (req_id, dets, lookup)))
            except Exception as exc:
                self.assist_events.put(
                    ("suggest_err", (req_id, f"{type(exc).__name__}: {exc}"))
                )

        threading.Thread(target=worker, daemon=True, name="YOLO-Assist").start()

    def accept_suggestions(self):
        if not self.suggestions:
            return
        self.push_undo()
        self.boxes.extend(box for box, _ in self.suggestions)
        self.suggestions = []
        self.selected = None
        self.assist_info.config(text="")
        self.commit()

    def accept_suggestion_at(self, ix: float, iy: float) -> bool:
        hits = [i for i, (b, _) in enumerate(self.suggestions) if b.contains(ix, iy)]
        if not hits:
            return False
        i = min(hits, key=lambda k: self.suggestions[k][0].area)
        box, _ = self.suggestions.pop(i)
        self.push_undo()
        self.boxes.append(box)
        self.selected = None
        self.commit()
        return True

    def clear_suggestions(self):
        self.suggestions = []
        self.assist_info.config(text="")
        self.draw_boxes()
        self.update_status()

    # ---------- Auto Detect ----------
    def start_auto_detect(self):
        if not self._require_project() or self.detect_running:
            return
        if not self.detector.loaded:
            messagebox.showinfo(
                "Model belum dimuat", "Muat model dulu lewat 'Muat Model (.pt)...'."
            )
            return
        lookup = build_lookup(self.classes)
        if not lookup:
            messagebox.showinfo(
                "Belum ada kelas",
                "Project belum punya kelas. Tambahkan kelas atau muat ulang model.",
            )
            return

        dialog = AutoDetectDialog(
            self, self.project, self.view.get(), float(self.assist_conf.get())
        )
        self.wait_window(dialog)
        if dialog.result is None:
            return
        scope, conf, iou = dialog.result
        subsets = ALL_SUBSETS if scope == "all" else (self.view.get(),)
        items = [(s, p) for s in subsets for p in self.project.list_images(s)]
        if not items:
            messagebox.showinfo(
                "Tidak ada gambar", "Tidak ada gambar pada cakupan yang dipilih."
            )
            return

        self.assist_req += 1
        self.suggestions = []
        self.draw_boxes()
        self.detect_running = True
        self.detect_stop = threading.Event()
        stop, project, detector = self.detect_stop, self.project, self.detector
        self.autodetect_button.config(state=tk.DISABLED)
        self.detect_stop_button.config(state=tk.NORMAL)
        self.assist_info.config(
            text=f"Auto Detect: 0/{len(items)}", fg=COLORS["accent"]
        )

        def worker():
            try:
                stats = run_auto_detect(
                    detector,
                    project,
                    items,
                    lookup,
                    conf,
                    iou,
                    stop,
                    progress=lambda n, t, s: self.assist_events.put(
                        ("detect_progress", (n, t))
                    ),
                )
                self.assist_events.put(("detect_done", stats))
            except Exception as exc:
                self.assist_events.put(("detect_err", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True, name="YOLO-AutoDetect").start()

    def stop_auto_detect(self):
        if self.detect_running and self.detect_stop is not None:
            self.detect_stop.set()
            self.detect_stop_button.config(state=tk.DISABLED)
            self.assist_info.config(
                text="Membatalkan Auto Detect...", fg=COLORS["warning"]
            )

    def _finish_auto_detect(self):
        self.detect_running = False
        self.autodetect_button.config(state=tk.NORMAL)
        self.detect_stop_button.config(state=tk.DISABLED)
        self.load_current()

    def _poll_assist_events(self):
        try:
            while True:
                event, data = self.assist_events.get_nowait()
                if event == "model_ok":
                    names = list(self.detector.names.values())
                    shown = ", ".join(names[:6]) + (", ..." if len(names) > 6 else "")
                    self.model_label.config(
                        text=f"Model: {self.detector.path.name}\n{len(names)} kelas: {shown}",
                        fg=COLORS["success"],
                    )
                    self._sync_model_classes()
                    if self.assist_auto.get():
                        self.request_suggestions(silent=True)
                elif event == "model_err":
                    self.model_label.config(
                        text="Model: gagal dimuat"
                        if not self.detector.loaded
                        else f"Model: {self.detector.path.name}",
                        fg=COLORS["danger_text"],
                    )
                    messagebox.showerror("Gagal memuat model", data)
                elif event == "suggest":
                    req_id, dets, lookup = data
                    if req_id != self.assist_req or self.pil_image is None:
                        continue
                    iw, ih = self.pil_image.size
                    self.suggestions = drop_duplicates(
                        to_boxes(dets, lookup, iw, ih), self.boxes
                    )
                    self.assist_info.config(
                        text=f"{len(self.suggestions)} saran. "
                        f"Klik box untuk terima, Enter terima semua."
                        if self.suggestions
                        else "Tidak ada saran baru.",
                        fg=COLORS["accent"],
                    )
                    self.draw_boxes()
                    self.update_status()
                elif event == "suggest_err":
                    req_id, msg = data
                    if req_id == self.assist_req:
                        self.assist_info.config(
                            text=f"Saran gagal: {msg}", fg=COLORS["danger_text"]
                        )
                elif event == "detect_progress":
                    n, total = data
                    if self.detect_running:
                        self.assist_info.config(
                            text=f"Auto Detect: {n}/{total}", fg=COLORS["accent"]
                        )
                elif event == "detect_done":
                    self._finish_auto_detect()
                    self.assist_info.config(text="")
                    lines = [
                        f"Gambar diberi label: {data['detected_images']} ({data['boxes']} box)",
                        f"Dilewati (sudah berlabel): {data['skipped']}",
                        f"Tanpa deteksi: {data['no_detection']}",
                    ]
                    if data["errors"]:
                        lines.append(
                            f"Error: {data['errors']} (pertama: {data['first_error']})"
                        )
                    title = (
                        "Auto Detect dibatalkan"
                        if data["stopped"]
                        else "Auto Detect selesai"
                    )
                    messagebox.showinfo(
                        title,
                        f"Diproses dari {data['total']} gambar.\n\n" + "\n".join(lines),
                    )
                elif event == "detect_err":
                    self._finish_auto_detect()
                    self.assist_info.config(text="")
                    messagebox.showerror("Auto Detect gagal", data)
        except queue.Empty:
            pass
        self.after(100, self._poll_assist_events)

    def on_view_change(self, _event=None):
        self.refresh_images()
        self.canvas.focus_set()

    # ---------- Kelas ----------
    @staticmethod
    def _color(cls_id: int) -> str:
        return PALETTE[cls_id % len(PALETTE)]

    def refresh_class_list(self):
        self.class_list.delete(0, tk.END)
        for i, name in enumerate(self.classes):
            self.class_list.insert(tk.END, f"  {i + 1}   {name}")
            self.class_list.itemconfig(
                i,
                foreground=self._color(i),
                selectforeground=self._color(i),
                selectbackground=COLORS["accent_soft"],
            )
        if self.classes:
            self.select_class(min(self.active_class, len(self.classes) - 1))

    def add_class(self):
        if not self._require_project():
            return
        name = self.class_entry.get().strip()
        if not name:
            return
        if name.lower() in (c.lower() for c in self.classes):
            messagebox.showwarning("Kelas sudah ada", f"Kelas '{name}' sudah ada.")
            return
        self.classes = self.project.add_class(name)
        self.class_entry.delete(0, tk.END)
        self.active_class = len(self.classes) - 1
        self.refresh_class_list()
        self.canvas.focus_set()

    def delete_class(self):
        if (
            not self._require_project()
            or self._busy_detecting()
            or self._busy_version()
        ):
            return
        if self.training_running:
            messagebox.showinfo(
                "Training berjalan",
                "Hentikan training dulu sebelum menghapus kelas.",
            )
            return
        if not (0 <= self.active_class < len(self.classes)):
            messagebox.showinfo("Pilih kelas", "Pilih kelas di daftar terlebih dahulu.")
            return

        index = self.active_class
        name = self.classes[index]
        try:
            n_boxes = self.project.count_class_boxes(index)
        except OSError as exc:
            messagebox.showerror("Gagal membaca label", str(exc))
            return

        if not messagebox.askyesno(
            "Hapus kelas",
            f"Hapus kelas '{name}'?\n\n"
            f"{n_boxes} box dengan kelas ini akan dihapus dari seluruh dataset "
            f"(unsplit, train, valid, test).\n"
            f"Id kelas di bawahnya akan bergeser otomatis.\n\n"
            f"Tindakan ini tidak bisa dibatalkan.",
            icon="warning",
        ):
            return

        error = None
        try:
            stats = self.project.remove_class(index)
        except (ValueError, OSError) as exc:
            error, stats = exc, None

        # Muat ulang dari disk (box di memori masih memakai id lama)
        self.classes = self.project.read_names()
        self.active_class = min(index, max(len(self.classes) - 1, 0))
        self.refresh_class_list()
        self.load_current()
        self.canvas.focus_set()

        if error:
            messagebox.showerror("Hapus kelas gagal", str(error))
            return
        messagebox.showinfo(
            "Kelas dihapus",
            f"Kelas '{name}' dihapus.\n"
            f"{stats['boxes_removed']} box dibuang dari {stats['files_changed']} file label.",
        )

    def select_class(self, i: int):
        if 0 <= i < len(self.classes):
            self.active_class = i
            self.class_list.selection_clear(0, tk.END)
            self.class_list.selection_set(i)
            self.class_list.see(i)

    def on_class_list_select(self, _event=None):
        sel = self.class_list.curselection()
        if sel:
            self.active_class = sel[0]
        self.canvas.focus_set()

    # ---------- Daftar gambar & navigasi ----------
    def refresh_images(self):
        self.image_paths = (
            self.project.list_images(self.view.get()) if self.project else []
        )
        self.index = 0 if self.image_paths else -1
        self.load_current()

    def prev_image(self):
        if self.image_paths and self.index > 0:
            self.index -= 1
            self.load_current()

    def next_image(self):
        if self.image_paths and self.index < len(self.image_paths) - 1:
            self.index += 1
            self.load_current()

    def load_current(self):
        self.pil_image = None
        self.boxes = []
        self.extra_lines = []
        self.selected = None
        self.is_null = False
        self.edit_mode = None
        self.suggestions = []
        self.assist_req += 1
        self.undo_stack.clear()

        if self.index >= 0:
            path = self.image_paths[self.index]
            try:
                img = Image.open(path)
                self.pil_image = ImageOps.exif_transpose(img).convert("RGB")
            except Exception as exc:
                messagebox.showerror("Gagal membuka gambar", f"{path.name}\n{exc}")
            if self.pil_image is not None:
                iw, ih = self.pil_image.size
                label_file = self.project.label_path(self.view.get(), path)
                self.boxes, self.extra_lines = load_labels(label_file, iw, ih)
                self.is_null = (
                    label_file.is_file() and not self.boxes and not self.extra_lines
                )
        self.show_image()
        if self.assist_auto.get() and self.pil_image is not None and not self.is_null:
            self.request_suggestions(silent=True)

    # ---------- Render ----------
    def show_image(self):
        self.canvas.delete("all")
        self.drag_rect = None
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        if self.pil_image is None:
            if self.project is None:
                title = "Selamat datang di Labeler"
                sub = "Buat proyek baru atau buka proyek yang sudah ada untuk memulai."
            else:
                title = f"'{self.view.get()}' masih kosong"
                sub = (
                    "Klik 'Tambah ke Dataset' untuk menambahkan gambar."
                    if self.view.get() == POOL
                    else ""
                )
            self.canvas.create_text(
                cw / 2,
                ch / 2 - 14,
                text=title,
                fill=COLORS["canvas_text"],
                font=FONTS["header"],
                justify="center",
            )
            if sub:
                self.canvas.create_text(
                    cw / 2,
                    ch / 2 + 16,
                    text=sub,
                    fill=COLORS["canvas_sub"],
                    font=FONTS["small"],
                    justify="center",
                    width=400,
                )
            self.update_status()
            return

        iw, ih = self.pil_image.size
        self.scale = min(cw / iw, ch / ih)
        new_w, new_h = max(1, int(iw * self.scale)), max(1, int(ih * self.scale))
        self.offset_x = (cw - new_w) / 2
        self.offset_y = (ch - new_h) / 2

        resized = self.pil_image.resize((new_w, new_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(
            self.offset_x, self.offset_y, image=self.tk_image, anchor="nw"
        )

        self.draw_boxes()
        self.update_status()

    def _class_name(self, cls_id: int) -> str:
        return (
            self.classes[cls_id] if 0 <= cls_id < len(self.classes) else f"id {cls_id}"
        )

    def _draw_chip(self, x1: float, y1: float, text: str, color: str):
        """Label kecil di sudut kiri atas box; teks otomatis gelap/terang."""
        if y1 - 20 >= self.offset_y:
            anchor, tx, ty = "sw", x1 + 4, y1 - 3
        else:
            anchor, tx, ty = "nw", x1 + 4, y1 + 3
        text_id = self.canvas.create_text(
            tx,
            ty,
            text=text,
            anchor=anchor,
            fill=_text_on(color),
            font=FONTS["bold"],
            tags="box",
        )
        bx1, by1, bx2, by2 = self.canvas.bbox(text_id)
        bg = self.canvas.create_rectangle(
            bx1 - 3,
            by1 - 1,
            bx2 + 3,
            by2 + 1,
            fill=color,
            outline=color,
            tags="box",
        )
        self.canvas.tag_lower(bg, text_id)

    def draw_boxes(self):
        self.canvas.delete("box")
        if self.pil_image is None:
            return
        for i, b in enumerate(self.boxes):
            x1, y1 = self.image_to_canvas(b.x1, b.y1)
            x2, y2 = self.image_to_canvas(b.x2, b.y2)
            color = self._color(b.cls)
            is_selected = i == self.selected
            if is_selected:
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline="white", width=1, dash=(4, 4), tags="box"
                )
                for hx, hy in self._handle_points(b).values():
                    self.canvas.create_rectangle(
                        hx - HANDLE_DRAW,
                        hy - HANDLE_DRAW,
                        hx + HANDLE_DRAW,
                        hy + HANDLE_DRAW,
                        fill="white",
                        outline=color,
                        width=2,
                        tags="box",
                    )
            else:
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline=color, width=2, tags="box"
                )
            self._draw_chip(x1, y1, self._class_name(b.cls), color)
        if self.is_null and not self.boxes:
            self._draw_chip(
                self.offset_x + 8,
                self.offset_y + 28,
                "NULL · tanpa objek",
                COLORS["warning"],
            )
        self._draw_suggestions()

    def _draw_suggestions(self):
        for box, conf in self.suggestions:
            x1, y1 = self.image_to_canvas(box.x1, box.y1)
            x2, y2 = self.image_to_canvas(box.x2, box.y2)
            color = self._color(box.cls)
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=color, width=2, dash=(7, 4), tags="box"
            )
            self._draw_chip(x1, y1, f"? {self._class_name(box.cls)} {conf:.2f}", color)
        self.null_button.config(
            text="Batalkan null  (N)" if self.is_null else "Tandai null  (N)"
        )

    def update_status(self):
        if self.project is None:
            self.status_project.config(
                text="Belum ada project", fg=COLORS["text_muted"]
            )
            for lbl_ in (
                self.status_position,
                self.status_file,
                self.status_image,
                self.status_counts,
            ):
                lbl_.config(text="")
            return

        self.status_project.config(
            text=f"{self.project.name}  ▸  {self.view.get()}", fg=COLORS["text"]
        )

        if self.pil_image is None:
            self.status_position.config(text="0 / 0")
            self.status_file.config(text="")
            self.status_image.config(text="")
            self.status_counts.config(text="")
            return

        iw, ih = self.pil_image.size
        path = self.image_paths[self.index]
        self.status_position.config(text=f"{self.index + 1} / {len(self.image_paths)}")
        self.status_file.config(text=path.name)
        self.status_image.config(text=f"{iw} × {ih}")
        counts = f"{len(self.boxes)} box"
        if self.is_null and not self.boxes:
            counts += "  ·  null"
        if self.suggestions:
            counts += f"  ·  {len(self.suggestions)} saran"
        self.status_counts.config(text=counts)

    # ---------- Konversi koordinat ----------
    def canvas_to_image(self, cx: float, cy: float) -> tuple[float, float]:
        return (cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale

    def image_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        return ix * self.scale + self.offset_x, iy * self.scale + self.offset_y

    def _clamp_canvas(self, x: float, y: float) -> tuple[float, float]:
        iw, ih = self.pil_image.size
        x = min(max(x, self.offset_x), self.offset_x + iw * self.scale)
        y = min(max(y, self.offset_y), self.offset_y + ih * self.scale)
        return x, y

    # ---------- Mouse: gambar, pilih, geser, resize ----------
    def _handle_points(self, b: Box) -> dict[str, tuple[float, float]]:
        x1, y1 = self.image_to_canvas(b.x1, b.y1)
        x2, y2 = self.image_to_canvas(b.x2, b.y2)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        return {
            "nw": (x1, y1),
            "ne": (x2, y1),
            "se": (x2, y2),
            "sw": (x1, y2),
            "n": (mx, y1),
            "e": (x2, my),
            "s": (mx, y2),
            "w": (x1, my),
        }

    def _handle_at(self, cx: float, cy: float) -> str | None:
        """Nama handle pada box terpilih yang berada di titik canvas ini, atau None."""
        if self.selected is None:
            return None
        for name, (hx, hy) in self._handle_points(self.boxes[self.selected]).items():
            if abs(cx - hx) <= HANDLE_HIT and abs(cy - hy) <= HANDLE_HIT:
                return name
        return None

    def _box_at(self, ix: float, iy: float) -> int | None:
        hits = [i for i, b in enumerate(self.boxes) if b.contains(ix, iy)]
        # Kalau box bertumpuk, pilih yang terkecil (paling mungkin dimaksud)
        return min(hits, key=lambda i: self.boxes[i].area) if hits else None

    def select_at(self, ix: float, iy: float):
        self.selected = self._box_at(ix, iy)
        self.draw_boxes()

    def _begin_edit(self, mode: str, ix: float, iy: float):
        self.edit_mode = mode
        self.edit_origin = dataclasses.replace(self.boxes[self.selected])
        self.edit_snapshot = [dataclasses.replace(b) for b in self.boxes]
        self.edit_start = (ix, iy)
        self.edit_changed = False

    def on_press(self, event):
        self.canvas.focus_set()
        if self.pil_image is None:
            return
        ix, iy = self.canvas_to_image(event.x, event.y)
        force_draw = bool(event.state & 0x0001)  # tahan Shift = paksa gambar box baru

        if not force_draw:
            # 1) handle pada box terpilih -> ubah ukuran
            handle = self._handle_at(event.x, event.y)
            if handle:
                self._begin_edit(handle, ix, iy)
                return
            # 2) klik di dalam box -> pilih, dan siap digeser kalau di-drag
            hit = self._box_at(ix, iy)
            if hit is not None:
                self.selected = hit
                self.draw_boxes()
                self._begin_edit("move", ix, iy)
                return

        # 3) area kosong -> gambar box baru
        self.edit_mode = "draw"
        self.drag_start = self._clamp_canvas(event.x, event.y)

    def on_drag(self, event):
        if self.edit_mode is None:
            return
        if self.edit_mode != "draw":
            self._edit_drag(event)
            return

        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = self._clamp_canvas(event.x, event.y)
        if self.drag_rect is None:
            if abs(x1 - x0) < MIN_DRAG_PX and abs(y1 - y0) < MIN_DRAG_PX:
                return
            self.drag_rect = self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline=self._color(self.active_class),
                width=2,
                dash=(5, 3),
            )
        else:
            self.canvas.coords(self.drag_rect, x0, y0, x1, y1)

    def _edit_drag(self, event):
        """Geser atau ubah ukuran box terpilih, dihitung dari posisi box saat mouse ditekan."""
        iw, ih = self.pil_image.size
        ix, iy = self.canvas_to_image(event.x, event.y)
        dx, dy = ix - self.edit_start[0], iy - self.edit_start[1]
        o = self.edit_origin
        b = self.boxes[self.selected]

        if self.edit_mode == "move":
            w, h = o.x2 - o.x1, o.y2 - o.y1
            b.x1 = min(max(o.x1 + dx, 0), iw - w)
            b.y1 = min(max(o.y1 + dy, 0), ih - h)
            b.x2, b.y2 = b.x1 + w, b.y1 + h
        else:
            x1, y1, x2, y2 = o.x1, o.y1, o.x2, o.y2
            if "w" in self.edit_mode:
                x1 = o.x1 + dx
            if "e" in self.edit_mode:
                x2 = o.x2 + dx
            if "n" in self.edit_mode:
                y1 = o.y1 + dy
            if "s" in self.edit_mode:
                y2 = o.y2 + dy
            x1, x2 = (min(max(v, 0), iw) for v in (x1, x2))
            y1, y2 = (min(max(v, 0), ih) for v in (y1, y2))
            b.x1, b.x2 = (
                min(x1, x2),
                max(x1, x2),
            )  # kalau ditarik melewati sisi seberang, otomatis terbalik
            b.y1, b.y2 = min(y1, y2), max(y1, y2)

        self.edit_changed = True
        self.draw_boxes()

    def _finish_edit(self):
        if not self.edit_changed:
            return
        b = self.boxes[self.selected]
        if b.x2 - b.x1 < 2 or b.y2 - b.y1 < 2:  # terlalu kecil -> batalkan perubahan
            self.boxes = self.edit_snapshot
            self.draw_boxes()
            return
        self.push_undo(self.edit_snapshot)
        self.commit()

    def on_release(self, event):
        mode, self.edit_mode = self.edit_mode, None
        if mode is None:
            return
        if mode != "draw":
            self._finish_edit()
            return

        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = self._clamp_canvas(event.x, event.y)
        self.drag_start = None
        if self.drag_rect is not None:
            self.canvas.delete(self.drag_rect)
            self.drag_rect = None

        if abs(x1 - x0) < MIN_DRAG_PX or abs(y1 - y0) < MIN_DRAG_PX:
            # Klik biasa di area kosong (atau Shift+klik): pilih box di titik itu, atau batal pilih
            ix, iy = self.canvas_to_image(event.x, event.y)
            self.select_at(ix, iy)
            return

        if not self.classes:
            messagebox.showinfo(
                "Belum ada kelas", "Tambahkan kelas dulu di panel kanan."
            )
            self.class_entry.focus_set()
            return

        iw, ih = self.pil_image.size
        ax, ay = self.canvas_to_image(x0, y0)
        bx, by = self.canvas_to_image(x1, y1)
        box = Box(
            self.active_class,
            min(max(min(ax, bx), 0), iw),
            min(max(min(ay, by), 0), ih),
            min(max(max(ax, bx), 0), iw),
            min(max(max(ay, by), 0), ih),
        )
        self.push_undo()
        self.boxes.append(box)
        self.selected = None
        self.commit()

    def on_hover(self, event):
        """Ubah bentuk kursor: handle -> panah ukuran, di dalam box -> geser, lainnya -> crosshair."""
        if self.pil_image is None or self.edit_mode is not None:
            return
        cursor = "crosshair"
        handle = self._handle_at(event.x, event.y)
        if handle:
            cursor = HANDLE_CURSORS[handle]
        else:
            ix, iy = self.canvas_to_image(event.x, event.y)
            if self._box_at(ix, iy) is not None:
                cursor = "fleur"
        self.canvas.config(cursor=cursor)

    def nudge_selected(self, dx: float, dy: float):
        """Geser box terpilih (satuan piksel gambar), tetap di dalam gambar."""
        if self.selected is None or self.pil_image is None:
            return
        iw, ih = self.pil_image.size
        b = self.boxes[self.selected]
        dx = max(-b.x1, min(dx, iw - b.x2))
        dy = max(-b.y1, min(dy, ih - b.y2))
        if dx == 0 and dy == 0:
            return
        self.push_undo()
        b.x1 += dx
        b.x2 += dx
        b.y1 += dy
        b.y2 += dy
        self.commit()

    def deselect(self):
        if self.selected is None and self.suggestions:
            self.suggestions = []
            self.update_status()
        self.selected = None
        self.draw_boxes()

    def push_undo(self, snapshot: list[Box] | None = None):
        if snapshot is None:
            snapshot = [dataclasses.replace(b) for b in self.boxes]
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > MAX_UNDO:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            return
        self.boxes = self.undo_stack.pop()
        self.selected = None
        self.commit()

    def delete_selected(self):
        if self.selected is None:
            return
        self.push_undo()
        del self.boxes[self.selected]
        self.selected = None
        self.commit()

    def apply_class_to_selected(self):
        if self.selected is None or not self.classes:
            return
        if self.boxes[self.selected].cls == self.active_class:
            return
        self.push_undo()
        self.boxes[self.selected].cls = self.active_class
        self.commit()

    def toggle_null(self):
        if self.pil_image is None or self.project is None:
            return
        if not self.is_null and (self.boxes or self.extra_lines):
            messagebox.showinfo(
                "Gambar punya anotasi",
                "Hapus semua box dulu sebelum menandai gambar ini sebagai null.",
            )
            return
        self.is_null = not self.is_null
        self.suggestions = []
        self.assist_req += 1  # batalkan pencarian saran yang sedang jalan
        self.assist_info.config(text="")
        self.commit()

    def commit(self):
        self.save_current_labels()
        self.draw_boxes()
        self.update_status()

    def save_current_labels(self):
        if self.project is None or self.pil_image is None:
            return
        path = self.image_paths[self.index]
        iw, ih = self.pil_image.size
        try:
            save_labels(
                self.project.label_path(self.view.get(), path),
                self.boxes,
                self.extra_lines,
                iw,
                ih,
                keep_empty=self.is_null,
            )
        except OSError as exc:
            messagebox.showerror("Gagal menyimpan label", str(exc))


if __name__ == "__main__":
    LabelerApp().mainloop()
