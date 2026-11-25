# Labeler + YOLO Training

## Instalasi
Jalankan dari folder ini:

```bash
python -m pip install ultralytics pillow numpy
python app.py
```

Untuk fitur **Auto Tune**, pasang juga Optuna:

```bash
python -m pip install optuna
```

Jika memakai GPU NVIDIA, pasang PyTorch build CUDA yang sesuai dengan versi driver/CUDA di komputer terlebih dahulu. Jangan menganggap `pip install ultralytics` otomatis memasang build CUDA yang tepat untuk semua perangkat.

Jika aplikasi dijalankan lewat `pythonw` (tanpa jendela konsol), semua pesan dan traceback dicatat di `logs/app.log` (dibatasi 1000 baris terakhir). Error pada tombol/dialog juga ditampilkan sebagai pesan, lengkap dengan lokasi log.

## Cara menggunakan
1. Klik **file app.bat** biar tidak terlalu lama
2. Klik **Proyek Baru** atau **Buka**.
3. Klik **Tambah ke Dataset**, tambahkan kelas di panel kanan, lalu anotasi gambar (satu per satu, atau cepat lewat **Bulk Label**).
4. Klik **Split...** untuk membagi dataset menjadi train/valid/test.
5. (Opsional) Klik **Health Check** untuk memeriksa kualitas dataset.
6. (Opsional) Klik **Preprocessing...** untuk membuat versi dataset dengan resize/augmentasi.
7. (Opsional) Klik **Auto Tune...** untuk mencari hyperparameter terbaik.
8. Klik **Training YOLO...**.
9. Isi konfigurasi dan pilih `dataset_version`. Setiap parameter memiliki penjelasan di sisi kanan.
10. Klik **Mulai Training**.
11. Pantau status dan progress bar di bagian bawah jendela. Klik **Stop** untuk meminta penghentian aman.

## Anotasi
Seret di area kosong gambar untuk membuat box. Klik box untuk memilihnya, seret box terpilih untuk menggesernya, dan seret titik sudut/sisi untuk mengubah ukuran. Tahan **Shift** saat menyeret untuk memaksa membuat box baru (berguna di atas box yang sudah ada). Perubahan disimpan otomatis ke file label setiap kali Anda menambah, mengubah, atau menghapus box.

Gambar yang memang tidak berisi objek bisa ditandai **null** (`N` atau tombol **Tandai null**). Gambar null disimpan dengan file label kosong, sehingga dibedakan dari gambar yang belum dilabeli. Gambar yang masih punya box tidak bisa ditandai null; hapus box-nya dulu.

Tekan **F1** (atau tombol **Pintasan** di toolbar) untuk melihat daftar pintasan. Ringkasannya:

| Tombol | Fungsi |
|---|---|
| Seret di area kosong | Buat box |
| Klik box | Pilih box |
| Seret box terpilih | Geser box |
| Seret titik sudut/sisi | Ubah ukuran box |
| `Shift` + seret | Paksa buat box baru |
| `Shift` + panah | Geser box terpilih 1 piksel layar |
| `1`-`9` | Pilih kelas aktif |
| `R` | Ubah kelas box terpilih |
| `Del` / `X` / `Backspace` | Hapus box |
| `N` | Tandai / batalkan null |
| `Ctrl+Z` | Undo (maksimal 50 langkah) |
| `Esc` | Batal pilih / tolak saran |
| `S` | Sarankan box (Label Assist) |
| `Enter` | Terima semua saran |
| `A` / `D` atau `←` / `→` | Gambar sebelumnya / berikutnya |
| `Ctrl+B` | Buka Bulk Label |
| `F1` | Daftar pintasan |

Pintasan keyboard tidak aktif saat kursor berada di kolom input teks.

### Mengelola kelas
Tambahkan kelas lewat kolom di panel kanan. Tombol **Hapus kelas terpilih** menghapus kelas beserta **semua box** dengan kelas itu di seluruh dataset (dataset, train, valid, test), dan id kelas di bawahnya bergeser otomatis. Tindakan ini tidak bisa dibatalkan, dan tidak bisa dilakukan saat training berjalan.

## Bulk Label
Tampilan grid thumbnail untuk meninjau dan mengelola banyak gambar sekaligus. Buka lewat tombol **Bulk Label** di toolbar atau `Ctrl+B`.

**Filter**
- **Subset**: `dataset`, `train`, `valid`, atau `test`. Mengganti subset memuat ulang daftar gambar.
- **Status**: Semua, Belum dilabeli, Sudah dilabeli, atau Null.
- **Cari**: filter berdasarkan nama file.

**Pagination**
Hanya satu halaman thumbnail yang dibuat pada satu waktu (default **60 gambar per halaman**, diatur lewat `PAGE_SIZE` di `BulkLabelDialog`; gunakan kelipatan 6 agar baris penuh). Ini menjaga dialog tetap cepat dan hemat memori pada dataset besar, karena thumbnail halaman lain tidak dimuat sampai dibuka. Daftar nama file dan status label tetap dibaca untuk seluruh gambar di latar belakang, tetapi itu hanya membaca file teks kecil.

Navigasi halaman ada di footer:

| Kontrol | Fungsi |
|---|---|
| `«` / `‹` | Halaman pertama / sebelumnya |
| `›` / `»` | Halaman berikutnya / terakhir |
| **Pilih halaman** | Centang semua gambar di halaman ini |
| **Pilih semua** | Centang semua gambar hasil filter, lintas halaman |
| **Kosongkan** | Batalkan semua pilihan |

**Pilihan lintas halaman**
Centang tersimpan saat berpindah halaman, sehingga aksi massal berlaku untuk semua gambar terpilih di semua halaman. Pilihan direset saat filter Status atau pencarian berubah, saat subset diganti, dan setelah aksi massal selesai (daftar dimuat ulang dan kembali ke halaman 1).

**Aksi massal**
- **Tandai Null**: menandai gambar terpilih sebagai null. Hanya gambar yang belum punya box yang diproses.
- **Pindahkan ke...**: memindahkan gambar terpilih beserta file labelnya ke subset lain.
- **Hapus**: menghapus gambar dan labelnya dari disk. Tidak bisa dibatalkan.
- **Muat Ulang**: membaca ulang daftar gambar dari disk.

Klik thumbnail untuk membuka gambar itu di editor utama.

## Label Assist
Memakai model YOLO (`.pt`) untuk menyarankan box pada gambar yang sedang dibuka.

1. Klik **Muat Model (.pt)...** di panel **Label Assist**. File `best.pt` dari run terbaru di `runs/` dipilih sebagai tawaran awal.
2. Jika model punya kelas yang belum ada di project, Anda akan ditanya apakah kelas itu ditambahkan. Bila memilih **Tidak**, deteksi untuk kelas tersebut diabaikan.
3. Atur **Confidence saran**, lalu klik **Sarankan box (S)**. Centang **Auto-sarankan tiap gambar** agar saran muncul otomatis saat berpindah gambar (dilewati untuk gambar null).
4. Saran tampil sebagai box putus-putus berawalan `?` beserta skor. Klik satu box untuk menerimanya, **Terima semua** (atau `Enter`) untuk menerima semuanya, **Tolak** (atau `Esc`) untuk membuangnya.

Saran yang tumpang tindih dengan box yang sudah ada tidak ditampilkan. Saran belum tersimpan sampai diterima.

## Auto Detect
Mengisi label otomatis untuk banyak gambar sekaligus. Butuh model yang sudah dimuat dan minimal satu kelas di project.

1. Klik **Auto Detect...**.
2. Pilih cakupan: **tampilan saat ini** atau **semua subset**.
3. Atur **Confidence** (skor minimal) dan **IoU (NMS)** (ambang penggabungan box yang tumpang tindih).
4. Klik **Mulai**. Gunakan **Batalkan Auto Detect** untuk menghentikan.

Kebijakannya **SKIP**: gambar yang sudah punya label (file label tidak kosong) dilewati dan tidak diubah. Hanya gambar tanpa label yang diisi hasil deteksi. Setelah selesai, ringkasan menampilkan jumlah gambar yang diberi label, yang dilewati, yang tanpa deteksi, dan error bila ada. Selama Auto Detect berjalan, **Tambah ke Dataset**, **Split...**, **Preprocessing...**, **Bulk Label**, dan **Muat Model** dikunci.

## Health Check
Klik **Health Check** untuk menganalisis seluruh dataset (progress bisa dibatalkan). Hasilnya ditampilkan dalam satu jendela:

- **Ringkasan**: jumlah gambar dan anotasi, ukuran rata-rata, rasio median, serta jumlah gambar tanpa file label (*missing*) dan dengan file label kosong (*null*).
- **Class Balance**: jumlah anotasi tiap kelas per subset, dengan penanda *Missing* dan *Under represented*.
- **Dimension Insights**: sebaran ukuran dan rasio aspek gambar.
- **Histogram** jumlah objek per gambar.
- **Annotation Heatmap**: lokasi anotasi pada gambar.

Jika ada anotasi dengan id kelas yang tidak ada di `data.yaml`, atau gambar yang gagal dibaca, jumlahnya ditampilkan sebagai peringatan.

## Preprocessing & Augmentasi
Mirip Roboflow: hasilnya adalah **versi dataset** baru di `versions/v1`, `v2`, dan seterusnya. Dataset asli (`train/`, `valid/`, `test/`) tidak pernah diubah, dan tiap versi punya `data.yaml` sendiri.

Klik **Preprocessing...** (butuh subset train yang sudah di-split), atur opsi, lalu **Buat Versi**. Tombol **Pratinjau** menampilkan contoh hasil pada gambar yang sedang dibuka, lengkap dengan box, supaya Anda bisa memeriksa sebelum membuat versi.

**Preprocessing** (semua subset: train, valid, test)

| Opsi | Fungsi |
|---|---|
| `resize_mode` | `off`, `stretch`, `fit_gray` / `fit_black` / `fit_white` (muat utuh dengan bingkai), atau `fill` (potong tengah) |
| `resize_w`, `resize_h` | Ukuran target |
| `grayscale` | Hitam-putih (tetap 3 kanal) |
| `contrast_mode` | `stretch` (autocontrast) atau `equalize` |
| `filter_null` | Buang gambar tanpa label |

**Augmentasi** (hanya train; valid dan test tidak pernah diaugmentasi). Tiap gambar train menghasilkan `copies` salinan dengan parameter acak sendiri, dan gambar aslinya ikut disimpan jika `include_original` aktif.

| Jenis | Opsi |
|---|---|
| Geometri | `flip_h`, `flip_v`, `rotate90`, `rotation`, `shear`, `crop` |
| Warna | `brightness`, `contrast`, `saturation`, `hue`, `gray_pct` |
| Gangguan | `blur`, `noise`, `cutout_count` + `cutout_size` |

Hal yang perlu diketahui:
- Box ikut ditransformasi. Pada rotasi dan shear, box menjadi persegi pembatas objek yang diputar, jadi sedikit lebih longgar. Box yang tersisa kurang dari `min_visibility` setelah terpotong dibuang.
- Salinan yang semua box-nya hilang tidak disimpan, supaya tidak menjadi gambar negatif palsu.
- Label non-box (polygon segmentasi) tidak bisa ditransformasi. Gambar dengan label polygon tidak diaugmentasi, dan barisnya dibuang jika resize mengubah geometri (`fit_*`, `fill`). Ringkasan di akhir proses menampilkan jumlahnya.
- Gambar selalu dibaca dengan orientasi EXIF diterapkan, sama seperti di editor label.
- Seed dan pengaturan yang sama menghasilkan versi yang sama. Pengaturan lengkap tersimpan di `versions/vN/config.json`.
- Ultralytics tetap menjalankan augmentasi online bawaannya (mosaic, flip, HSV, dll.) saat training. Augmentasi offline di sini bersifat tambahan, jadi jangan berlebihan.
- Jika Anda mengubah label setelah membuat versi, versi lama tidak ikut berubah. Buat versi baru.

Di dialog Training, pilih `dataset_version`: versi terbaru dipilih otomatis, atau `(dataset asli)` untuk melatih dari dataset tanpa preprocessing.

## Auto Tune (tidak disarankan)
Mencari hyperparameter (learning rate, optimizer, augmentasi) secara otomatis memakai Optuna. Butuh `pip install optuna`.

1. Klik **Auto Tune...**.
2. Pilih **Model** (`yolo26n.pt` sampai `yolo26x.pt`), **imgsz** (512, 640, 768, 1024), dan **Epoch** (50 sampai 300) untuk training akhir.
3. Klik **Lanjut...**, lalu pilih **profil hardware** (lihat di bawah) dan klik **Mulai Tuning**.

Tuning memakai dataset asli, dengan nilai bawaan: 100 trial, 40 epoch per trial, patience 15 per trial, dan pencarian parameter augmentasi aktif. Nilai ini ada di `TuningConfigDialog._submit` jika ingin diubah.

Setelah selesai, ringkasan menampilkan skor terbaik dan folder hasil. Untuk melatih dengan konfigurasi terbaik, buka **Training YOLO...**, klik **Muat hasil tuning...**, dan pilih `best_config.json`. Form terisi otomatis, dan parameter augmentasi dari hasil tuning ikut dipakai saat training. **Stop** juga bisa menghentikan tuning.

### Profil hardware
Aplikasi mendeteksi GPU/CPU dan menawarkan profil pemakaian resource. Tiap profil menampilkan nilai yang akan dipakai (`device`, `batch`, `workers`, `cache`, `amp`, `deterministic`). Profil **custom** memungkinkan Anda mengisi sendiri semua nilai tersebut; isian awalnya diambil dari profil `medium`.

## Struktur hasil
```text
projects/<project>/
  data.yaml
  dataset/                      # gambar yang belum di-split
  train/
  valid/
  test/
  versions/
    v1/                         # hasil Preprocessing
      data.yaml
      config.json               # pengaturan + statistik
      train/{images,labels}
      valid/{images,labels}
      test/{images,labels}
  runs/
    train/
      config.json
      args.yaml                 # dibuat oleh Ultralytics
      results.csv               # metrik per epoch
      weights/
        best.pt
        last.pt
      results.png               # jika plots aktif
      confusion_matrix.png      # jika tersedia
      result_summary.json
    train2/
    ...
logs/
  app.log                       # hanya saat dijalankan via pythonw
```

Nama folder eksperimen otomatis bertambah jika nama dasar telah digunakan. Folder eksperimen yang ada tidak ditimpa.

## Resume
Pilih `runs/<nama_run>/weights/last.pt` pada field `resume_from` (tombol **Pilih...** membuka pemilih file). Saat resume aktif, Ultralytics melanjutkan state training dari checkpoint dan sebagian besar argumen training akan mengikuti checkpoint tersebut.

## Penghentian di tengah training
Tombol **Stop** mengirim permintaan penghentian ke callback Ultralytics. Ini bukan pemutusan proses secara paksa: penghentian dilakukan pada titik aman yang diperiksa trainer, sehingga bisa ada jeda sebelum status berubah. `save=True` dan `save_period=1` membuat checkpoint disimpan berkala; `last.pt` biasanya menjadi checkpoint resume. Jika proses Python/komputer dimatikan paksa, penyimpanan checkpoint pada batch yang sedang berjalan tidak dijamin.

## Catatan
- `valid` dipakai sebagai validation split melalui `data.yaml`.
- `test` tidak otomatis dievaluasi saat training standar; gunakan evaluasi terpisah setelah training bila ingin metrik test.
- Pastikan setiap gambar punya file label YOLO yang benar dan ID kelas berada pada rentang `0..nc-1`.
- Menu **Lihat** di toolbar berpindah antara `dataset` (gambar yang belum di-split) dan subset `train`, `valid`, `test`.
- Training dan tuning tidak bisa berjalan bersamaan; selesaikan atau hentikan salah satunya dulu.
- Untuk pemilihan config terbaik yang bisa digunakan pada laptop engineer adalah:
- yolo26n:
- Batch = 48
- Imgz = 640 --> ini untuk detail keakuratan dalam deteksi
- Worker = 8
- Cache = "ram"
- yolo26s:
- Batch = 30
- Imgz = 640 --> ini untuk detail keakuratan dalam deteksi
- Worker = 8
- Cache = "ram"
- Semakin tinggi model yolo, semakin berat training dan semakin berat saat mendeteksi
- Yolo dimulai dari janis "n" untuk mencari keringanan dan kecepatan, semakin naik ke "x", maka semakin berat untuk training dan kecepatan pendeteksi turun
