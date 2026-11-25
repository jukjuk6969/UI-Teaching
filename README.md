# UI-Teaching — Labeling Dataset & Training YOLO

Aplikasi desktop Python untuk membuat dan mengelola dataset deteksi objek, memberi anotasi format YOLO, memeriksa dataset, melakukan preprocessing/augmentasi, dan melatih model YOLO melalui Ultralytics.

## Persyaratan

- Windows, macOS, atau Linux.
- Python 3.10–3.12 disarankan.
- Koneksi internet saat instalasi pertama.
- GPU NVIDIA opsional. CPU juga didukung, tetapi jauh lebih lambat.

## Instalasi pertama

Jalankan semua perintah dari folder hasil clone repository ini.

### 1. Unduh repository

```bash
git clone https://github.com/jukjuk6969/UI-Teaching.git
cd UI-Teaching
```

Alternatif: unduh ZIP dari GitHub, ekstrak, lalu buka terminal di folder `UI-Teaching`.

### 2. Buat virtual environment

**Windows (PowerShell):**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Jika aktivasi diblokir, jalankan `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` lalu aktifkan ulang.

**Windows (Command Prompt):**

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Pasang dependensi

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optuna juga dipasang melalui `requirements.txt` untuk fitur **Auto Tune**.

### 4. Pasang dukungan GPU secara opsional

Untuk GPU NVIDIA, pasang build PyTorch yang sesuai dengan sistem operasi, driver, dan versi CUDA:

https://pytorch.org/get-started/locally/

Jalankan perintah PyTorch yang dipilih dari terminal virtual environment `.venv` yang aktif. Lalu cek:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Jika hasilnya `True`, Ultralytics dapat memakai CUDA. Jika tidak, aplikasi tetap dapat berjalan memakai CPU.

### 5. Jalankan aplikasi

```bash
python app.py
```

Di Windows, `app.bat` dapat dijalankan dengan double click. Launcher mencari `.venv` atau `venv` di dalam folder aplikasi, lalu mengambil fallback Python dari `PATH`. `app.vbs` menjalankan aplikasi tanpa jendela terminal.

## Dari pengisian data sampai training

1. Jalankan aplikasi dan pilih **Proyek Baru**.
2. Masukkan nama proyek tanpa spasi. Nama hanya boleh huruf, angka, `_`, atau `-`.
3. Klik **Tambah ke Dataset** dan pilih gambar.
4. Tambahkan kelas melalui panel kanan, lalu buat bounding box pada objek.
5. Gambar yang tidak memiliki objek dapat ditandai sebagai **null**.
6. Gunakan **Bulk Label** untuk mempercepat peninjauan dan pengelolaan label banyak gambar.
7. Klik **Split...** dan bagi dataset. Contoh: 70% `train`, 20% `valid`, 10% `test`. Persentase harus total 100%.
8. Pastikan subset `train` dan `valid` berisi gambar serta label.
9. Opsional: klik **Health Check** untuk memeriksa kualitas dataset.
10. Opsional: klik **Preprocessing...** untuk membuat versi dataset baru; dataset asli tidak akan ditimpa.
11. Klik **Training YOLO...**.
12. Pilih model, misalnya `yolo26n.pt` untuk percobaan ringan.
13. Pilih `(dataset asli)` atau versi dataset dari folder `versions`.
14. Isi epoch, ukuran gambar, dan batch. Mulai dengan batch kecil jika GPU memiliki memori terbatas.
15. Klik **Mulai Training**. Gunakan **Stop** untuk meminta penghentian aman.
16. Hasil training tersimpan di:

```text
projects/<nama-proyek>/runs/<nama-run>/
```

Bobot penting berada di `weights/best.pt` dan `weights/last.pt`.

## Struktur data aplikasi

Aplikasi menyimpan semua project relatif terhadap lokasi file `project.py`. Lokasi lama milik pembuat komputer tidak digunakan.

```text
UI-Teaching/
├── app.py
├── requirements.txt
├── projects/
│   └── <nama-proyek>/
│       ├── data.yaml
│       ├── unsplit/
│       ├── train/
│       ├── valid/
│       ├── test/
│       ├── versions/
│       └── runs/
└── logs/
```

Untuk membuka proyek lama, pilih **Buka** dan pilih folder langsung di dalam `projects/<nama-proyek>/`.

## Pembacaan dan penyimpanan label

Format label YOLO adalah:

```text
class_id x_center y_center width height
```

Seluruh koordinat ternormalisasi antara `0` hingga `1`. ID kelas harus sesuai urutan pada `data.yaml`. Gambar bertanda null disimpan sebagai file label kosong.

## Fitur utama

- Labeling bounding box dengan mouse dan keyboard.
- Undo, pindah subset, hapus gambar, tandai null, dan Bulk Label.
- Label Assist dan Auto Detect memakai model `.pt`.
- Health Check untuk statistik dataset.
- Preprocessing dan augmentasi offline.
- Training dan Auto Tune Ultralytics YOLO.
- Hardware profile untuk CPU/GPU, batch, worker, cache, AMP, dan deterministic mode.
- Resume training dari `weights/last.pt`.

## Troubleshooting

Jika terjadi error instalasi:

```bash
python -m pip show ultralytics
python -c "import ultralytics, torch, PIL, numpy; print('OK')"
```

Jika error saat aplikasi dijalankan, jalankan dari terminal:

```bash
python app.py
```

Pesan aplikasi juga dapat dilihat di `logs/app.log` saat dijalankan melalui `pythonw` atau `app.vbs`.
