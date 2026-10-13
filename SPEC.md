# SPEC — Manipulator Lab: Kinematics and Trajectory Planning

**Owner:** Farid Akhmad
**Status:** v1.0.0 (semua minggu selesai dan terverifikasi di mesin pengguna; satu butir DoD masih terbuka: kontribusi pribadi, ditulis oleh penulis)
**Tanggal:** 12 Oktober 2026
**Posisi dalam roadmap:** Project 1, setelah Numerical Method Toolkit (NMT)
**Estimasi:** 4–6 minggu (5–7 jam/minggu)

> Semua angka toleransi dan target di dokumen ini adalah **target yang diusulkan**, bukan hasil terverifikasi. Hasil pengukuran baru boleh dicantumkan di README setelah dijalankan dan direproduksi.

---

## 1. Tujuan dan masalah yang diselesaikan

**Tujuan.** Membangun simulator lengan robot planar 2-link yang menghitung forward kinematics (FK), inverse kinematics (IK), mendeteksi target di luar jangkauan, menganalisis singularitas, dan menghasilkan trajektori joint yang mulus, lengkap dengan tes otomatis dan visualisasi.

**Masalah.** Diberikan target posisi ujung lengan (x, y), tentukan sudut joint (θ₁, θ₂) yang mencapainya, putuskan apakah target dapat dicapai, lalu gerakkan lengan ke sana dengan profil gerak yang halus.

**Mengapa penting untuk portofolio.** Proyek ini menjembatani NMT (matematika numerik) ke robotika: model matematika, pembanding analitik vs numerik, analisis conditioning di dekat singularitas, dan validasi otomatis. Proyek ini juga menjadi modul kinematika yang dipakai ulang oleh Project 3 (Vision-Guided Capstone).

**Kemampuan yang dibuktikan:** pemodelan matematis, penurunan solusi analitik, penggunaan NMT sebagai dependensi nyata, analisis stabilitas numerik, testing berbasis invariant, dan komunikasi teknis (diagram, plot, demo).

**Pembeda dari contoh IK di NMT.** NMT sudah memiliki IK planar 2-link. Proyek ini harus menambah nilai: baseline analitik, perbandingan metode, analisis workspace dan singularitas, trajektori, dan visualisasi. Bukan mengulang contoh yang sama.

---

## 2. Scope

### In scope (MVP)
1. FK planar 2-link.
2. IK analitik (kedua cabang: elbow-up dan elbow-down).
3. IK numerik memakai Newton dari NMT, dibandingkan dengan solusi analitik.
4. Pemeriksaan reachability dengan alasan penolakan yang eksplisit.
5. Visualisasi: lengan, target, batas workspace, target tak terjangkau.
6. Analisis singularitas dan conditioning Jacobian.
7. Trajektori joint-space: linear dan cubic (rest-to-rest).
8. Suite tes otomatis dan evaluasi terukur yang reproducible.

### Out of scope (versi pertama)
- Perangkat keras fisik, ROS 2, Gazebo.
- Lengan 3D atau lebih dari 2 link, dinamika (torsi, massa, gesekan).
- Antarmuka web atau GUI interaktif.
- Penghindaran tabrakan (stretch goal).

### Stretch goals (hanya setelah MVP lulus semua gate)
- Trajektori quintic dan perbandingan tiga profil.
- Batas joint (joint limits) dan pemilihan cabang IK yang valid.
- Pemeriksaan tabrakan sederhana dengan lingkaran/kotak.

---

## 3. Model matematika

### 3.1 Konvensi
- Satuan: panjang dalam **meter**, sudut dalam **radian**, waktu dalam **detik**.
- Frame dunia {0} di pangkal lengan; sumbu x ke kanan, y ke atas, sudut positif berlawanan arah jarum jam.
- Parameter: panjang link l₁, l₂ > 0. **Default: l₁ = 1.0 m, l₂ = 0.8 m** (jangkauan 0.2 ≤ r ≤ 1.8 m; l₁ ≠ l₂ agar lubang tak terjangkau di tengah terlihat di demo). **Durasi demo trajektori: T = 2.0 s.** Semua dapat diubah lewat argumen.
- θ₁ diukur dari sumbu x ke link 1; θ₂ adalah sudut relatif link 2 terhadap link 1.

### 3.2 Forward kinematics
```
x = l₁ cos θ₁ + l₂ cos(θ₁ + θ₂)
y = l₁ sin θ₁ + l₂ sin(θ₁ + θ₂)
```

### 3.3 Inverse kinematics analitik
Misalkan r² = x² + y².
```
cos θ₂ = (r² − l₁² − l₂²) / (2 l₁ l₂)
θ₂ = ± arccos(cos θ₂)                      # + : elbow-down, − : elbow-up (konvensi dicatat di kode)
θ₁ = atan2(y, x) − atan2(l₂ sin θ₂, l₁ + l₂ cos θ₂)
```
**Reachability:** target dapat dicapai jika |l₁ − l₂| ≤ r ≤ l₁ + l₂. Karena cos θ₂ harus berada di [−1, 1], implementasi harus menangani galat floating-point di batas (clamp dengan toleransi eksplisit, bukan clamp diam-diam).

### 3.4 Jacobian dan singularitas
```
J(θ) = [ −l₁ sin θ₁ − l₂ sin(θ₁+θ₂)    −l₂ sin(θ₁+θ₂) ]
       [  l₁ cos θ₁ + l₂ cos(θ₁+θ₂)     l₂ cos(θ₁+θ₂) ]

det J = l₁ l₂ sin θ₂
```
Singularitas terjadi saat sin θ₂ = 0 (lengan terentang penuh atau terlipat penuh). Di dekat titik itu, J ill-conditioned: iterasi Newton memerlukan solusi sistem linear dengan matriks hampir singular, sehingga langkah iterasi bisa membesar dan konvergensi memburuk. Ini harus diperlihatkan lewat eksperimen (nomor kondisi vs θ₂, iterasi Newton vs jarak ke batas workspace).

**Nomor kondisi (diturunkan dan diverifikasi minggu 2).** Dengan F = ‖J‖²_F = l₁² + 2l₂² + 2l₁l₂ cos θ₂ dan D = |det J| = l₁l₂ |sin θ₂|, nilai singular memenuhi σ₁² + σ₂² = F dan σ₁σ₂ = D. Maka cond(J) = σ₁²/D, dengan σ₁² = ½(F + √(F² − 4D²)). Nilainya **hanya bergantung pada θ₂** (tidak pada θ₁ maupun cabang IK), sehingga petanya berupa cincin konsentris. Implementasi: `condition_number` (SVD) dan `condition_from_theta2` (rumus tertutup), keduanya `inf` bila σ_min ≤ 10·ε·σ_max. Ambang bendera singularitas: |sin θ₂| < 1e-6 (`SINGULARITY_TOL`).

### 3.5 Trajektori joint-space
Untuk setiap joint dari q₀ ke q_f dalam durasi T:

- **Linear:** q(t) = q₀ + (q_f − q₀)·t/T. Posisi kontinu, tetapi kecepatan melompat di t = 0 dan t = T.
- **Cubic rest-to-rest** (q̇(0) = q̇(T) = 0):
  ```
  q(t) = q₀ + 3Δ (t/T)² − 2Δ (t/T)³,   Δ = q_f − q₀
  ```
  Bentuk umum q(t) = a₀ + a₁t + a₂t² + a₃t³ dengan syarat batas posisi dan kecepatan menghasilkan sistem linear 4×4. Sistem itu dapat diselesaikan dengan Gaussian elimination NMT sebagai **pemeriksa silang** terhadap rumus tertutup di atas.
- **Quintic** (stretch): menambah syarat percepatan nol di ujung (kontinuitas C²).

---

## 4. Arsitektur

```
manipulator-lab/
├── pyproject.toml
├── README.md
├── SPEC.md
├── src/manipulator_lab/
│   ├── __init__.py
│   ├── arm.py            # dataclass PlanarArm(l1, l2), validasi parameter
│   ├── kinematics.py     # fk(), ik_analytic(), jacobian(), is_reachable()
│   ├── ik_numeric.py     # adapter ke Newton NMT + pembanding
│   ├── workspace.py      # sampling workspace, batas jangkauan, peta conditioning
│   ├── trajectory.py     # linear, cubic (quintic stretch)
│   └── viz.py            # plot dan animasi Matplotlib (tanpa logika matematika)
├── tests/
│   ├── test_kinematics.py
│   ├── test_reachability.py
│   ├── test_trajectory.py
│   └── test_ik_numeric.py
├── experiments/
│   ├── run_ik_comparison.py
│   └── run_trajectory_profiles.py
└── docs/
    ├── frames.png
    └── results/          # plot, tabel, dan metadata lingkungan
```

**Aliran data:** `target (x, y)` → `is_reachable` → `ik_analytic / ik_numeric` → `q_goal` → `trajectory` → `q(t)` → `fk` → `viz`.

**Prinsip desain:**
- Logika matematika tidak boleh bergantung pada Matplotlib (visualisasi adalah lapisan terluar).
- Fungsi murni dengan input tervalidasi; error dilaporkan lewat exception atau hasil bertipe yang eksplisit (mis. `reachable=False` + alasan), bukan nilai NaN diam-diam.
- Pemakaian NMT hanya lewat `ik_numeric.py` (dan pemeriksa silang kubik), agar versi dependensi mudah dipin dan diganti.

**Dependensi NMT (dikonfirmasi dari kode v1.0.0, commit `e9e6a4e`).**

- Nama distribusi: `numerical-methods-toolkit`; nama import: `numerical_toolkit`.
- Pin di `pyproject.toml`:
  ```toml
  dependencies = [
      "numpy>=2.0",
      "numerical-methods-toolkit @ git+https://github.com/faridakhmad437-bit/Numerical-Method-Toolkit.git@v1.0.0",
  ]
  ```
- API yang dipakai (semua diekspor dari `numerical_toolkit`):

| Kebutuhan | Fungsi | Signature | Keluaran |
|---|---|---|---|
| IK numerik (utama) | `newton_system` | `newton_system(F, J, x0, *, options=None)` | `SolverResult` |
| IK numerik (pembanding bawaan) | `planar_2link_ik` | `planar_2link_ik(target, l1, l2, initial=(0.0, 0.0), *, options=None)` | `SolverResult` |
| Pemeriksa silang kubik | `gaussian_elimination` | `gaussian_elimination(A, b)` | `np.ndarray` (solusi `x`) |
| Konfigurasi | `SolverOptions` | `xtol=1e-10, rtol=1e-8, ftol=1e-10, maxiter=100, store_history=True` (frozen dataclass) | |
| Hasil | `SolverResult` | `solution, converged, iterations, function_calls, residual, step_error, message, history` | |

- Pakai `newton_system` dengan `F` dan `J` milik Manipulator Lab (bukan `planar_2link_ik`), supaya Jacobian yang dianalisis di proyek ini adalah Jacobian yang sama dengan yang dipakai solver. `planar_2link_ik` dipakai sebagai pembanding.
- Perilaku NMT yang harus ditangani adapter `ik_numeric.py` (terverifikasi lewat smoke test, bukan asumsi):
  1. **Cabang ditentukan oleh tebakan awal.** Target FK(0.6, −0.9) dengan tebakan (0.3, −0.5) menghasilkan θ₂ = −0.9 (elbow-up); tebakan (0.3, +0.5) menghasilkan θ₂ = +0.9 (elbow-down).
  2. **Tebakan awal default (0, 0) singular** (sin θ₂ = 0) dan melempar `InvalidInputError("Jacobian is singular.")`. Adapter harus menyediakan tebakan awal sendiri dan tidak memakai default NMT.
  3. **Target di luar jangkauan:** `planar_2link_ik` mengembalikan `converged=False`, `iterations=0`, `residual=None` (tidak melempar). Pemeriksaan jangkauannya memakai perbandingan ketat tanpa toleransi, sehingga `is_reachable` milik proyek ini mendefinisikan toleransinya sendiri.
  4. `newton_system` hanya melempar error bila `np.linalg.solve` gagal pada matriks singular tepat. Matriks hampir singular lolos dan menghasilkan langkah besar; ini justru perilaku yang dieksperimenkan di Kelompok D.
  5. `SolverResult` memvalidasi `residual` bertipe real, finite, ≥ 0, atau `None`.

**Stack:** Python 3.11+, NumPy, Matplotlib, pytest. Semua gratis.

---

## 5. Requirements

| ID | Requirement |
|---|---|
| FR-1 | `fk(arm, theta)` mengembalikan posisi ujung lengan; menerima array sudut dan menolak input tak valid. |
| FR-2 | `ik_analytic(arm, target, branch)` mengembalikan (θ₁, θ₂) untuk cabang yang diminta, atau hasil "tidak terjangkau" dengan alasan. |
| FR-3 | `is_reachable(arm, target)` berlaku konsisten dengan FR-2 di seluruh batas workspace. |
| FR-4 | `jacobian(arm, theta)` dan `condition_number` tersedia; singularitas terdeteksi dengan ambang eksplisit. |
| FR-5 | IK numerik (Newton NMT) menerima tebakan awal dan melaporkan iterasi, residual, dan status konvergensi. |
| FR-6 | `trajectory.linear` dan `trajectory.cubic` menghasilkan q(t), q̇(t) pada grid waktu yang diberikan. |
| FR-7 | Visualisasi menampilkan lengan, target, batas workspace, dan penanda target tak terjangkau. |
| FR-8 | Eksperimen dapat dijalankan ulang dengan satu perintah, dengan seed dan konfigurasi tercatat. |
| NFR-1 | Tidak ada klaim performa tanpa pengukuran yang tercatat kondisinya. |
| NFR-2 | Seluruh test suite lulus sebelum checkpoint/commit. |

---

## 6. Rencana pengujian

Tes ditulis lebih dulu (test-first) untuk tiap kelompok.

### Kelompok A — Konsistensi FK ↔ IK
- Untuk grid sudut (θ₁, θ₂) yang valid: `ik_analytic(fk(θ))` menghasilkan konfigurasi yang jika dimasukkan ke FK kembali ke titik semula dalam toleransi. Perbandingan dilakukan di ruang posisi, bukan sudut, karena sudut ill-conditioned di dekat batas workspace.
- Kedua cabang (elbow-up/down) menghasilkan posisi ujung yang sama.
- Kasus khusus: θ₂ = 0 dan θ₂ = π (singular), θ₁ di sekitar ±π (wrap-around sudut).

### Kelompok B — Reachability
- Titik di dalam, tepat di batas luar (r = l₁ + l₂), tepat di batas dalam (r = |l₁ − l₂|), dan sedikit di luar: hasil harus benar dan konsisten.
- Kasus l₁ = l₂ (batas dalam menjadi titik asal) ditangani tanpa pembagian nol atau NaN.
- Input tidak valid (NaN, inf, panjang link ≤ 0) ditolak dengan error jelas.

### Kelompok C — Trajektori
- Endpoint: q(0) = q₀ dan q(T) = q_f.
- Cubic: q̇(0) = q̇(T) = 0; linear: kecepatan konstan.
- Hasil rumus tertutup kubik sama dengan solusi sistem 4×4 via NMT (dalam toleransi).
- Posisi dan kecepatan kontinu pada grid waktu; T ≤ 0 ditolak.

### Kelompok D — IK numerik vs analitik
- Dari tebakan awal yang wajar, Newton konvergen ke salah satu dari dua solusi analitik.
- Di dekat singularitas, perilaku (iterasi bertambah, gagal konvergen, atau cabang berbeda) dicatat sebagai hasil eksperimen, bukan disembunyikan.
- Tebakan awal tepat di singularitas (θ₂ = 0) menghasilkan `InvalidInputError` dari NMT; adapter harus mengubahnya menjadi hasil terstruktur (`converged=False` + alasan) atau error proyek yang jelas.
- Cabang hasil Newton ditentukan tebakan awal; tes memastikan tebakan dengan θ₂ < 0 mendarat di elbow-up dan θ₂ > 0 di elbow-down untuk target yang jelas di dalam workspace.

**Toleransi (ditetapkan v0.3):**

| Tes | Toleransi | Status |
|---|---|---|
| FK(IK analitik(target)) vs target | 1e-9 m (ruang posisi) | Dipakai di tes A–B; galat teramati 1.3e-15 m (grid 25×25 × 2 cabang) |
| Cek jangkauan (`REACH_TOL`) | 1e-9 m pada radius | Dipakai; hanya menyerap noise float |
| Sudut θ₂ di batas workspace | 1e-6 rad | Dipakai; galat teramati 0 |
| cond(J): SVD vs rumus tertutup | relatif 1e-9 (di luar |sin θ₂| < 1e-3 pada tes) | Dipakai; galat relatif maksimum teramati 5.0e-14 pada 102.564 sel peta (eksperimen, sandbox) |
| Ambang singularitas | |sin θ₂| < 1e-6 | Dipakai |
| Newton vs analitik (sudut) | 1e-6 rad, hanya jauh dari singularitas | Dipakai; selisih maksimum teramati 8.9e-16 rad pada 936 target (eksperimen v2, sandbox) |
| Residual Newton | ≤ 1e-9 m (default NMT `ftol` = 1e-10) | Dipakai; maksimum teramati 7.8e-16 m; iterasi median 5, maksimum 6 (eksperimen v2, sandbox) |
| Endpoint trajektori (posisi, kecepatan) | 1e-12 | Dipakai; galat endpoint maksimum teramati 4.4e-16 rad, kecepatan ujung kubik 0 (eksperimen W4, 1000 lintasan, sandbox) |
| Koefisien kubik: rumus tertutup vs sistem 4×4 NMT | relatif 1e-9 | Dipakai; selisih relatif maksimum teramati 4.7e-16 (sandbox) |

Aturan: toleransi tidak boleh dilonggarkan agar tes lulus. Perubahan harus disertai pengukuran galat dan alasannya. Kelonggaran saat ini jauh di atas galat teramati (1e-9 vs 1.3e-15); boleh diperketat setelah dibandingkan di mesin lain.

---

## 7. Protokol evaluasi (untuk bukti di README)

1. **Grid target:** sampel titik reachable dan unreachable pada grid tetap (resolusi dicatat), plus subset acak dengan seed tetap.
2. **Metrik IK:** galat posisi akhir, galat vs solusi analitik, jumlah iterasi Newton, tingkat konvergensi, dan runtime (median dari beberapa kali run; dicatat Python/NumPy/OS/CPU).
3. **Metrik singularitas:** nomor kondisi J vs θ₂; iterasi Newton vs jarak ke batas workspace.
4. **Metrik trajektori:** galat endpoint, kecepatan maksimum, dan lonjakan kecepatan/percepatan pada linear vs cubic.
5. **Aturan:** semua metode dibandingkan pada target dan tebakan awal yang sama; definisi "berhasil" ditulis sebelum eksperimen dijalankan; kegagalan dilaporkan.

### Lingkungan pengukuran (tercatat 12 Oktober 2026, mesin pengembangan)

| Item | Nilai |
|---|---|
| OS | Windows 11 (Windows-11-10.0.26200-SP0) |
| Python | 3.14.3 (MSC v.1944, 64-bit AMD64) |
| NumPy | 2.5.4 |
| CPU | 13th Gen Intel Core i5-13450HX |

Angka runtime di README hanya boleh berasal dari mesin ini. Tes yang dijalankan di sandbox Linux (Python 3.13) hanya dipakai untuk verifikasi fungsional, bukan untuk angka performa. Skrip eksperimen perlu mencetak ulang versi Python/NumPy/OS/CPU ke `docs/results/` setiap kali dijalankan.

### Riwayat pra-registrasi eksperimen IK (minggu 3)

Dicatat apa adanya, termasuk kegagalan:

- **v1 (pra-registrasi, margin 0,05 m dari batas): GAGAL.** Tebakan awal default tidak mendarat di cabang yang diminta pada seluruh target (1080 konvergen, 1008 di cabang yang diminta). Penyebabnya diselidiki lewat bagian A2 (pemindaian cabang terhadap jarak ke batas): di dekat batas dalam, cabang yang dicapai bergantian 0%/100% menurut radius (basin saling berselang-seling).
- **v2 (post hoc, margin 0,30 m, grid baru): LULUS di sandbox** (936/936 konvergen di cabang yang diminta). Kriteria v2 dibuat setelah melihat kegagalan v1, sehingga bukan konfirmasi independen; v1 tetap dipertahankan di skrip dan laporan.
- **H1** (konvergensi lebih lambat mendekati singularitas): terkonfirmasi hanya untuk batas luar (iterasi 6, 8, 9, 11, 12, 14, 16, 17 untuk celah 1e-1 … 1e-8). Di batas dalam polanya tidak monoton.
- **H2** (konvergensi linier pada akar singular, celah 0): terkonfirmasi; rasio langkah tepat 0,5 dan 28 iterasi, dibanding 6 di interior.
- **H3** (besaran galat sudut mengikuti kondisi): besarnya tidak didukung data.
- Start acak: 88% konvergen (seed 0, N = 2000; 836 elbow-up, 924 elbow-down, 240 mencapai batas iterasi).

### Temuan eksperimen trajektori (minggu 4)

Kriteria C1–C7 ditulis di docstring `experiments/run_trajectory.py` sebelum dijalankan (1000 pasangan awal/tujuan acak, seed 0, T = 2 s, 201 sampel). Hasil di sandbox dan di mesin pengguna: semua kriteria terpenuhi (galat endpoint 4.4e-16 rad; rasio kecepatan puncak kubik/linear 1.5 dalam 2e-16; kedua profil menempuh segmen yang sama di ruang sendi; θ₂ ≤ 0 sepanjang lintasan elbow-up ke elbow-up, sehingga tidak pernah melewati singularitas θ₂ = 0). Temuan deskriptif (tanpa kriteria):

- Linear dan kubik menempuh **lintasan yang sama**; yang berbeda hanya pewaktuannya. Kubik membayar kecepatan puncak 1.5× dan percepatan puncak 6|Δ|/T² untuk kecepatan nol di kedua ujung.
- Interpolasi sendi **tidak lurus di ruang kerja**: simpangan maksimum jalur ujung terhadap garis lurus median 0,52 m, persentil 95 sebesar 1,98 m, maksimum 2,96 m pada sampel ini.
- Selisih sendi dipakai apa adanya (tanpa jalan terpendek): pada sampel acak ini 25,3% lintasan memutar θ₁ lebih dari π. Angka ini bergantung pada distribusi awal yang dipilih (θ₁ seragam), bukan sifat umum; batasan itu dicatat di README.
- Sepanjang lintasan, |sin θ₂| minimum 6.0e-4 dan cond(J) maksimum 1.7e3: lintasan bisa mendekati, tetapi pada sampel ini tidak mencapai, ambang singularitas 1e-6.

---

## 8. Deliverables

- Repositori `manipulator-lab` dengan struktur di atas, `pyproject.toml`, lisensi, dan CI/lint hanya jika benar-benar dijalankan.
- Test suite (kelompok A–D) yang lulus.
- Dua skrip eksperimen yang reproducible dengan hasil tersimpan di `docs/results/`.
- Diagram frame koordinat dan diagram arsitektur.
- GIF/video pendek: lengan mengikuti trajektori ke target, termasuk satu contoh target tak terjangkau.
- Plot: posisi, kecepatan joint, dan galat; nomor kondisi vs θ₂.
- README: tujuan, instalasi, quick-start, hasil beserta kondisi pengukuran, batasan.
- Satu engineering note: "Mengapa IK numerik memburuk di dekat singularitas."

---

## 9. Kriteria keberhasilan (Definition of Done)

- [x] Pengguna baru dapat menginstal, menjalankan simulator, mengubah target, dan melihat hasil reachable/unreachable. (Bukti: instalasi venv bersih oleh penulis di Windows dan simulasi di sandbox, contoh README dijalankan, demo reachable/unreachable. Belum diuji oleh orang lain.)
- [x] Seluruh tes kelompok A–D lulus pada commit rilis. (279 tes lulus dan `ruff` bersih di mesin pengguna pada pohon kerja yang menjadi tag `v1.0.0`.)
- [x] IK analitik dan numerik dibandingkan pada target yang sama, hasil terdokumentasi. (`docs/results/ik_comparison_summary.json`, bagian 7.)
- [x] Perilaku di dekat singularitas dijelaskan dengan bukti (plot/tabel). (`docs/ENGINEERING_NOTE.md`, plot `ik_*.png` dan `conditioning_*.png`.)
- [x] Trajektori cubic terbukti memenuhi syarat batas dan dibandingkan dengan linear.
- [x] Angka di README berasal dari run yang tercatat (lingkungan, seed, perintah). (Setiap angka README ditelusuri ke file di `docs/results/` atau ke keluaran tes di mesin pengguna.)
- [x] Batasan dan hal yang di luar scope tertulis jelas. (README "Findings and limitations", bagian 5.)
- [ ] Reviewer dapat melihat kontribusi pribadi dengan jelas.

---

## 10. Milestone (estimasi, bisa bergeser)

| Minggu | Fokus | Gate keluar |
|---|---|---|
| 1 | Setup repo, spec final, tes kelompok A–B (merah), FK + IK analitik | Tes A–B hijau |
| 2 | Reachability, Jacobian, workspace sampling | Peta workspace + conditioning |
| 3 | IK numerik via NMT, eksperimen perbandingan | Tabel hasil + tes D |
| 4 | Trajektori linear dan cubic, tes C | Tes C hijau, plot profil |
| 5 | Visualisasi dan animasi, README, diagram | Demo GIF berjalan |
| 6 (buffer) | Engineering note, rapikan, verifikasi dari environment bersih | Checklist DoD terpenuhi |

Jika waktu mingguan hanya 2–3 jam, perpanjang jadwal, jangan pangkas tes dan dokumentasi.

---

## 11. Risiko dan mitigasi

| Risiko | Mitigasi |
|---|---|
| API NMT kurang nyaman dipakai | Isolasi di `ik_numeric.py`; catat temuan sebagai umpan balik untuk NMT |
| Scope melebar (GUI, kolisi, 3D) | Semua masuk daftar stretch; kerjakan hanya setelah DoD MVP |
| Galat di batas workspace (floating-point) | Toleransi eksplisit dan tes kelompok B |
| Klaim performa tanpa dasar | Aturan NFR-1 dan protokol evaluasi bagian 7 |
| Animasi memakan waktu | Visualisasi ditunda ke minggu 5; logika diuji tanpa plot |
| Tebakan awal default tidak andal memilih cabang di dekat batas dalam (terlipat), sekitar 0,3 m dari batas | Terukur di minggu 3 (cabang yang dicapai bergantian menurut radius). Di zona itu berikan `initial=` secara eksplisit; jangan mengandalkan `default_initial_guess` |

---

## 12. Keputusan terbuka (semua terisi pada v0.3)

1. ~~Konvensi cabang~~ **Selesai:** elbow-up = θ₂ negatif, elbow-down = θ₂ positif.
2. ~~Nilai default~~ **Selesai:** l₁ = 1.0 m, l₂ = 0.8 m, T = 2.0 s.
3. ~~Nama dan signature fungsi NMT~~ **Selesai:** lihat bagian 4 (tag `v1.0.0`).
4. ~~Toleransi~~ **Selesai:** tabel di bagian 6.
5. ~~Lingkungan pengukuran~~ **Selesai:** tabel di bagian 7.

---

## 13. Status verifikasi

| Item | Status |
|---|---|
| Spesifikasi ini | Draft, belum direview |
| Implementasi | Sebagian: `arm.py`, `kinematics.py` (FK, IK analitik, reachability, Jacobian, kondisi, singularitas), `workspace.py` (batas, peta kondisi), `ik_numeric.py` (adapter Newton NMT), `trajectory.py` (linear, kubik, pemeriksa silang NMT 4×4, `plan_to_target`), `viz.py` (pose lengan, animasi GIF, snapshot, target tak terjangkau; matplotlib, tidak diimpor oleh `__init__`), `experiments/run_demo.py`, `experiments/trace_newton.py` (diagnostik), `docs/ENGINEERING_NOTE.md` (catatan teknik: angka dari file hasil mesin pengguna; mekanisme langkah Newton yang meledak ditelusuri di satu kasus dan ditandai sebagai interpretasi) |
| Tes | Minggu 2: 141 tes lulus di mesin Windows pengguna (commit `97b7bff`). Minggu 3 (kelompok D): 180 tes total lulus di sandbox Linux/Python 3.13, `ruff` bersih; 6 uji mutasi, 1 awalnya lolos (penghapusan wrap sudut) lalu ditutup dengan tes baru; dan 180 tes lulus juga di mesin Windows pengguna (commit `e86cc11`, Python 3.14.3, `ruff` bersih). Minggu 4 (kelompok C): 252 tes total lulus di sandbox Linux/Python 3.13, `ruff` bersih; 17 uji mutasi pada `trajectory.py`, 1 awalnya lolos (`plan_to_target` mengabaikan profil) lalu ditutup dengan tes baru; dan 252 tes lulus juga di mesin Windows pengguna (commit `69292d6`, Python 3.14.3, `ruff` bersih). Minggu 5 (`test_viz.py`, 27 tes): 279 tes total lulus di sandbox, `ruff` bersih; 21 uji mutasi pada `viz.py`, 18 terdeteksi dan 3 ekuivalen (penjaga lain di fungsi yang sama menolak masukan yang sama, sehingga perilakunya tidak berubah); dan 279 tes lulus juga di mesin Windows pengguna (commit `1cb227d`, Python 3.14.3, `ruff` bersih) |
| Hasil eksperimen | Conditioning: PASS di mesin pengguna (selisih relatif maksimum 4.7e-14, 102.564 sel). IK numerik, mesin pengguna (Windows, Python 3.14.3, commit `e86cc11`): v1 GAGAL, v2 LULUS, RESULT: PASS; file hasil resmi ada di `docs/results/` (riwayat di bagian 7). Waktu proses bervariasi sekitar 30% antar-run (median per panggilan, dua run: analitik 5,8 dan 7,5 µs; Newton 238 dan 232 µs), sehingga rasio kecepatan tidak dikutip sebagai klaim tunggal. Trajektori (W4), mesin pengguna (Windows, Python 3.14.3, commit `69292d6`): C1–C7 terpenuhi, RESULT: PASS (galat endpoint 4.4e-16 rad; selisih relatif koefisien NMT 4.6e-16); file hasil di `docs/results/trajectory_*`. Demo (W5): mesin pengguna (Windows, Python 3.14.3, commit `1cb227d`): D1–D4 terpenuhi, angka sama dengan sandbox (galat frame terakhir 5.6e-16 m, panjang lengan di semua frame 2.2e-16 m, GIF 316.558 byte, 60 frame; ukuran sama persis dengan sandbox); GIF dan PNG dilihat langsung, satu cacat tampilan (teks memotong lingkaran jangkauan) diperbaiki; GIF dan PNG diperiksa visual hanya di sandbox; file hasil di `docs/results/demo_*` |
| API NMT v1.0.0 (nama, signature, perilaku IK dan Gaussian) | Dibaca dari kode dan di-smoke-test (install bersih, IK, target di luar jangkauan, sistem kubik 4×4) |
| NMT v1.0.0: test suite dan lint | Terverifikasi di sesi ini (clone `e9e6a4e`, Python venv bersih): `pytest` 470 lulus, `ruff check` bersih |
| NMT v1.0.0: build paket dan status CI GitHub | Belum diverifikasi |
