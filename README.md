# UI-Teaching — Labeling Dataset & Training YOLO

Aplikasi desktop Python untuk membuat dan mengelola dataset deteksi objek, memberi anotasi format YOLO, memeriksa dataset, melakukan preprocessing/augmentasi, dan melatih model YOLO melalui Ultralytics.

## Persyaratan

- Windows, macOS, atau Linux.
- Python 3.10–3.12 disarankan. Pastikan Python tersedia di terminal (`python --version` atau `python3 --version`).
- Koneksi internet saat instalasi pertama: pip mungkin perlu mengunduh paket dan bobot model YOLO.
- Untuk training, gunakan GPU NVIDIA jika tersedia. CPU tetap dapat dipakai, tetapi training biasanya jauh lebih lambat.

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

Jika PowerShell menolak aktivasi, jalankan `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` pada terminal tersebut, lalu aktifkan ulang.

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

### 3. Perbarui pip dan pasang dependensi

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optuna disertakan untuk fitur **Auto Tune**. Jika hanya ingin melabeli dan training biasa, fitur tersebut boleh diabaikan.

### 4. (Opsional) Siapkan GPU NVIDIA

Untuk GPU NVIDIA, pasang PyTorch yang sesuai dengan sistem operasi, driver, dan versi CUDA komputer Anda dengan mengikuti petunjuk resmi PyTorch: <https://pytorch.org/get-started/locally/>. Lakukan ini di virtual environment `.venv` yang sama. Setelah itu, pastikan `python -c "import torch; print(torch.cuda.is_available())"` menampilkan `True`.

Jangan memilih paket CUDA secara tebakan. Jika tidak ada GPU yang cocok atau PyTorch tidak mendeteksinya, aplikasi dapat berjalan menggunakan CPU.

### 5. Jalankan aplikasi

Dengan virtual environment aktif:

```bash
python app.py
```

Di Windows, `app.bat` juga dapat digunakan. Launcher akan mencari virtual environment `.venv` atau `venv` di dalam folder repository; jika tidak menemukannya, launcher mencoba Python yang tersedia di `PATH`. `app.vbs` menjalankan aplikasi tanpa jendela terminal, terutama berguna setelah dependensi berhasil dipasang.

## Alur pertama kali sampai training

1. Jalankan aplikasi dan pilih **Proyek Baru**. Nama proyek hanya boleh berisi huruf, angka, `_`, dan `-`.
2. Klik **Tambah ke Dataset** untuk memasukkan gambar. Data proyek disimpan di `projects/<nama-proyek>/` di dalam folder aplikasi.
3. Tambahkan nama kelas pada panel kelas, lalu anotasi objek dengan membuat bounding box. Simpan label sesuai kelas. Untuk gambar yang memang tidak mengandung objek, tandai sebagai **null**.
4. Ulangi anotasi sampai gambar siap. Fitur **Bulk Label** membantu meninjau banyak gambar sekaligus.
5. Klik **Split...** untuk membagi data, misalnya `70% train`, `20% valid`, `10% test` (total harus 100%). Training memerlukan gambar pada subset `train` dan `valid`.
6. (Opsional) Jalankan **Health Check** untuk melihat gambar/label yang hilang, distribusi kelas, dan masalah dataset.
7. (Opsional) Klik **Preprocessing...** untuk membuat versi dataset baru. Dataset asli tidak ditimpa. Versi hasil tersimpan di `projects/<nama-proyek>/versions/`.
8. Klik **Training YOLO...**. Pilih model awal (contoh `yolo26n.pt`), jumlah epoch, ukuran gambar, dan perangkat. Model/bobot yang belum tersedia dapat diunduh oleh Ultralytics saat pertama digunakan.
9. Pilih `(dataset asli)` atau versi preprocessing yang ingin dipakai. Untuk percobaan awal, pilih model kecil seperti `yolo26n.pt`, nilai batch yang ringan, dan jumlah epoch secukupnya. Kurangi batch jika memori GPU habis; di CPU, gunakan ekspektasi waktu training yang lebih lama.
10. Klik **Mulai Training** dan pantau status/progress. Tombol **Stop** meminta penghentian aman.
11. Hasil training, termasuk bobot `best.pt` dan `last.pt`, disimpan di `projects/<nama-proyek>/runs/<nama-run>/`. `best.pt` biasanya menjadi pilihan untuk inferensi atau melanjutkan eksperimen.

## Struktur proyek dan path

Path data dibuat relatif terhadap lokasi kode aplikasi (`project.py`), bukan lokasi tertentu di komputer pembuat. Dengan demikian clone dapat berada di folder mana pun. Struktur yang dibuat aplikasi:

```text
UI-Teaching/
├── app.py
├── requirements.txt
├── projects/                    # dibuat saat membuat proyek pertama
│   └── <nama-proyek>/
│       ├── data.yaml
│       ├── unsplit/              # gambar yang belum dibagi
│       ├── train/
│       ├── valid/
│       ├── test/
│       ├── versions/             # versi preprocessing (opsional)
│       └── runs/                 # hasil training/tuning
└── logs/                         # log aplikasi, bila tersedia
```

Jangan memindahkan hanya file `data.yaml` secara terpisah dari folder dataset. Jika membuka dataset dari aplikasi, gunakan **Buka** dan pilih folder proyek di dalam `projects/`.

## Catatan

- Label deteksi menggunakan format YOLO: `class_id x_center y_center width height`, dengan koordinat ternormalisasi 0–1. ID kelas dimulai dari 0 dan harus sesuai urutan kelas di `data.yaml`.
- `valid` dipakai untuk validasi selama training. `test` tidak otomatis dievaluasi dalam training standar.
- Auto Tune bersifat opsional dan dapat memerlukan waktu serta resource komputasi yang besar.
- Jika instalasi atau training gagal, jalankan melalui `python app.py` di terminal agar pesan error terlihat. Pastikan virtual environment aktif dan periksa bahwa perintah Python/pip berasal dari `.venv`.
