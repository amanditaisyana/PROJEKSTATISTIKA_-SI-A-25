# =============================================================
#  REGRESI LINEAR BERGANDA (Multiple Linear Regression)
#  Dataset : Tingkat Pengangguran Terbuka (TPT) Berdasarkan
#            Tingkat Pendidikan, 2015–2025
#  Tujuan  : Memprediksi nilai TPT (%) berdasarkan:
#              - Tahun
#              - Semester (Februari / Agustus)
#              - Tingkat Pendidikan
# =============================================================

# ── 1. IMPORT LIBRARY ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder


# ── 2. LOAD DATA ───────────────────────────────────────────────
# Ganti path di bawah sesuai lokasi file CSV kamu
df = pd.read_csv("TPT_Berdasarkan_Tingkat_Pendidikan_2015-2025_Lengkap.csv")

print("=" * 60)
print("  PREVIEW DATA (5 baris pertama)")
print("=" * 60)
print(df.head())
print(f"\nJumlah baris  : {len(df)}")
print(f"Jumlah kolom  : {df.shape[1]}")
print(f"\nTipe data tiap kolom:\n{df.dtypes}")


# ── 3. CARA UBAH DATA MENJADI ANGKA (ENCODING) ─────────────────
#
#  Model regresi hanya bisa membaca ANGKA, bukan teks.
#  Karena kolom "Semester" dan "Tingkat Pendidikan" berisi teks,
#  kita perlu mengubahnya terlebih dahulu.
#
#  Ada 2 cara encoding yang umum dipakai:
#
#  [A] Label Encoding  → ubah kategori jadi angka urutan (0, 1, 2, ...)
#      Cocok untuk data ORDINAL (ada urutan maknanya)
#      Contoh: SD=0, SMP=1, SMA=2, S1=3 ...
#
#  [B] One-Hot Encoding → tiap kategori jadi kolom baru (0 atau 1)
#      Cocok untuk data NOMINAL (tidak ada urutan/ranking)
#      Contoh: Februari → kolom "Semester_Februari" bernilai 1 atau 0
#
#  Pada kasus ini kita pakai ONE-HOT ENCODING karena jenjang
#  pendidikan tidak benar-benar "berurutan" secara pengaruh terhadap TPT.

print("\n" + "=" * 60)
print("  ENCODING DATA KATEGORIKAL → ANGKA")
print("=" * 60)

# Sebelum encoding
print("\n[SEBELUM] Kolom 'Semester' (contoh nilai):")
print(df['Semester'].value_counts().to_string())

print("\n[SEBELUM] Kolom 'Tingkat Pendidikan' (contoh nilai):")
print(df['Tingkat Pendidikan'].value_counts().to_string())

# One-Hot Encoding dengan pd.get_dummies()
# drop_first=True → hapus 1 kolom dummy untuk menghindari multikolinearitas
df_encoded = pd.get_dummies(
    df,
    columns=['Semester', 'Tingkat Pendidikan'],
    drop_first=True
)

# Hapus kolom 'Jenis Data' (tidak relevan untuk model)
df_encoded = df_encoded.drop(columns=['Jenis Data'])

print("\n[SESUDAH] Kolom setelah encoding:")
for col in df_encoded.columns:
    print(f"  - {col}")

print(f"\nContoh 3 baris pertama setelah encoding:")
print(df_encoded.head(3).to_string())


# ── 4. PISAHKAN FITUR (X) DAN TARGET (y) ──────────────────────
#
#  X = variabel bebas / fitur → yang kita pakai untuk memprediksi
#  y = variabel terikat / target → yang ingin kita prediksi (TPT %)

X = df_encoded.drop(columns=['TPT (%)'])   # semua kolom kecuali TPT
y = df_encoded['TPT (%)']                  # kolom yang diprediksi

print("\n" + "=" * 60)
print("  PEMBAGIAN FITUR (X) DAN TARGET (y)")
print("=" * 60)
print(f"Fitur (X) shape : {X.shape}  → {X.shape[0]} baris, {X.shape[1]} kolom")
print(f"Target (y) shape: {y.shape}  → {y.shape[0]} nilai TPT")
print(f"\nNama kolom fitur: {X.columns.tolist()}")


# ── 5. BAGI DATA: TRAINING SET & TEST SET ─────────────────────
#
#  Training set (80%) → dipakai untuk "melatih" / mengajari model
#  Test set    (20%) → dipakai untuk "menguji" seberapa akurat model
#
#  Analoginya seperti belajar soal ujian (training),
#  lalu ujian dengan soal baru yang belum pernah dilihat (test).

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% untuk test
    random_state=42      # agar hasil bisa direproduksi
)

print("\n" + "=" * 60)
print("  PEMBAGIAN TRAINING & TEST SET")
print("=" * 60)
print(f"Total data   : {len(X)} baris")
print(f"Training set : {len(X_train)} baris (80%)")
print(f"Test set     : {len(X_test)} baris (20%)")


# ── 6. BUAT DAN LATIH MODEL ────────────────────────────────────
#
#  Model regresi linear berganda mencari persamaan:
#
#    TPT = β₀ + β₁×Tahun + β₂×Semester_Februari
#            + β₃×Pendidikan_SMA_Kejuruan + ... + ε
#
#  Dimana:
#    β₀ = intercept (konstanta)
#    β₁, β₂, ... = koefisien (bobot tiap fitur)
#    ε = error / residu

model = LinearRegression()
model.fit(X_train, y_train)   # ← proses "pembelajaran" terjadi di sini

print("\n" + "=" * 60)
print("  MODEL SUDAH DILATIH")
print("=" * 60)
print(f"\nIntercept (β₀)  : {model.intercept_:.4f}")
print("\nKoefisien tiap fitur:")
for fitur, koef in zip(X.columns, model.coef_):
    tanda = "↑" if koef > 0 else "↓"
    print(f"  {tanda} {fitur:<55} : {koef:>8.4f}")


# ── 7. PREDIKSI ────────────────────────────────────────────────

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)
y_pred_all   = model.predict(X)


# ── 8. EVALUASI MODEL ──────────────────────────────────────────
#
#  R²    (R-squared)    → seberapa besar variasi data yang bisa dijelaskan model
#                          0 = jelek, 1 = sempurna
#  MAE   (Mean Abs Err) → rata-rata selisih absolut prediksi vs aktual (satuan %)
#  RMSE  (Root MSE)     → seperti MAE tapi lebih sensitif terhadap error besar

def evaluasi(nama, y_aktual, y_prediksi):
    r2   = r2_score(y_aktual, y_prediksi)
    mae  = mean_absolute_error(y_aktual, y_prediksi)
    mse  = mean_squared_error(y_aktual, y_prediksi)
    rmse = np.sqrt(mse)
    print(f"\n  [{nama}]")
    print(f"    R²   : {r2:.4f}  → model menjelaskan {r2*100:.1f}% variasi data")
    print(f"    MAE  : {mae:.4f}% → rata-rata error prediksi")
    print(f"    RMSE : {rmse:.4f}%")
    return r2, mae, rmse

print("\n" + "=" * 60)
print("  EVALUASI PERFORMA MODEL")
print("=" * 60)
r2_tr, mae_tr, rmse_tr = evaluasi("TRAINING SET", y_train, y_pred_train)
r2_te, mae_te, rmse_te = evaluasi("TEST SET    ", y_test,  y_pred_test)


# ── 9. TABEL PERBANDINGAN AKTUAL VS PREDIKSI ──────────────────

hasil = X_test.copy()
hasil['TPT_Aktual']  = y_test.values
hasil['TPT_Prediksi'] = y_pred_test.round(2)
hasil['Residual']    = (y_test.values - y_pred_test).round(2)
hasil['Tahun']       = hasil['Tahun'].astype(int)

print("\n" + "=" * 60)
print("  SAMPEL HASIL PREDIKSI (10 baris pertama test set)")
print("=" * 60)
tampil = hasil[['Tahun', 'TPT_Aktual', 'TPT_Prediksi', 'Residual']].head(10)
tampil.index = range(1, len(tampil)+1)
print(tampil.to_string())


# ── 10. PREDIKSI DATA BARU ────────────────────────────────────
#
#  Contoh: memprediksi TPT tahun 2026, semester Februari, jenjang SMA Kejuruan
#
#  Kita harus membuat baris input dengan format yang SAMA dengan X_train:
#  kolom-kolomnya harus sama persis (hasil one-hot encoding).

print("\n" + "=" * 60)
print("  PREDIKSI DATA BARU")
print("=" * 60)

# Buat template baris input dengan nilai 0 semua
kolom_fitur = X.columns.tolist()

def buat_input(tahun, semester, tingkat_pendidikan):
    """
    Buat 1 baris input untuk prediksi.

    Parameter:
        tahun              : int, misal 2026
        semester           : str, 'Februari' atau 'Agustus'
        tingkat_pendidikan : str, salah satu dari:
                             'Diploma I/II/III', 'SMA Kejuruan', 'SMA umum',
                             'SMP', 'Tidak/Belum Pernah Sekolah/Belum Tamat & Tamat SD',
                             'Universitas'
    Catatan:
        Kategori referensi (drop_first=True):
          - Semester     → 'Agustus' (tidak ada kolomnya, nilai 0)
          - Pendidikan   → 'Diploma I/II/III' (tidak ada kolomnya, nilai 0)
    """
    row = {col: 0 for col in kolom_fitur}
    row['Tahun'] = tahun

    # Semester
    if semester == 'Februari':
        row['Semester_Februari'] = 1
    # (Agustus = referensi, semua kolom semester tetap 0)

    # Tingkat Pendidikan
    # Peta nama kolom setelah get_dummies
    peta_pendidikan = {
        'SMA Kejuruan'  : 'Tingkat Pendidikan_SMA Kejuruan',
        'SMA umum'      : 'Tingkat Pendidikan_SMA umum',
        'SMP'           : 'Tingkat Pendidikan_SMP',
        'Tidak/Belum Pernah Sekolah/Belum Tamat & Tamat SD':
                          'Tingkat Pendidikan_Tidak/Belum Pernah Sekolah/Belum Tamat & Tamat SD',
        'Universitas'   : 'Tingkat Pendidikan_Universitas',
        # 'Diploma I/II/III' → referensi, nilai 0
    }
    if tingkat_pendidikan in peta_pendidikan:
        row[peta_pendidikan[tingkat_pendidikan]] = 1

    return pd.DataFrame([row])


# ── Contoh prediksi ──
skenario = [
    (2026, 'Februari', 'SMA Kejuruan'),
    (2026, 'Agustus',  'Universitas'),
    (2027, 'Februari', 'SMP'),
    (2027, 'Agustus',  'Tidak/Belum Pernah Sekolah/Belum Tamat & Tamat SD'),
    (2030, 'Februari', 'SMA umum'),
]

print(f"\n{'No':<4} {'Tahun':<6} {'Semester':<12} {'Tingkat Pendidikan':<54} {'Prediksi TPT (%)'}")
print("-" * 100)
for i, (th, sem, edu) in enumerate(skenario, 1):
    inp    = buat_input(th, sem, edu)
    hasil_pred = model.predict(inp)[0]
    print(f"{i:<4} {th:<6} {sem:<12} {edu:<54} {hasil_pred:.2f}%")


# ── 11. VISUALISASI ───────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Regresi Linear Berganda — TPT Berdasarkan Tingkat Pendidikan 2015–2025",
    fontsize=14, fontweight='bold', y=1.01
)

warna_aktual  = '#378ADD'
warna_prediksi = '#D85A30'

# ── Plot 1: Aktual vs Prediksi (seluruh data) ──
ax1 = axes[0, 0]
ax1.scatter(y, y_pred_all, alpha=0.55, color=warna_aktual, edgecolors='none', s=35, label='Data')
batas = [y.min() - 0.5, y.max() + 0.5]
ax1.plot(batas, batas, '--', color='gray', linewidth=1, label='Prediksi sempurna')
ax1.set_xlabel("TPT Aktual (%)")
ax1.set_ylabel("TPT Prediksi (%)")
ax1.set_title("Aktual vs Prediksi (seluruh data)")
ax1.legend(fontsize=9)
ax1.text(0.05, 0.92, f"R² = {r2_score(y, y_pred_all):.3f}", transform=ax1.transAxes,
         fontsize=10, color='#3C3489', fontweight='bold')

# ── Plot 2: Residual Plot ──
ax2 = axes[0, 1]
resid_all = y.values - y_pred_all
warna_resid = ['#D85A30' if r > 0 else '#0F6E56' for r in resid_all]
ax2.bar(range(len(resid_all)), resid_all, color=warna_resid, alpha=0.65, width=0.8)
ax2.axhline(0, color='gray', linewidth=1, linestyle='--')
ax2.set_xlabel("Indeks observasi")
ax2.set_ylabel("Residual (%)")
ax2.set_title("Residual (Aktual − Prediksi)")
patch_pos = mpatches.Patch(color='#D85A30', alpha=0.65, label='Over-prediksi (+)')
patch_neg = mpatches.Patch(color='#0F6E56', alpha=0.65, label='Under-prediksi (−)')
ax2.legend(handles=[patch_pos, patch_neg], fontsize=9)

# ── Plot 3: TPT rata-rata per Tahun (aktual vs prediksi) ──
ax3 = axes[1, 0]
df_plot = df.copy()
df_plot['Prediksi'] = y_pred_all
grouped = df_plot.groupby('Tahun').agg(
    Aktual=('TPT (%)', 'mean'),
    Prediksi=('Prediksi', 'mean')
).reset_index()
ax3.plot(grouped['Tahun'], grouped['Aktual'],   'o-', color=warna_aktual,   linewidth=1.8, markersize=5, label='Aktual')
ax3.plot(grouped['Tahun'], grouped['Prediksi'], 's--', color=warna_prediksi, linewidth=1.8, markersize=5, label='Prediksi')
ax3.set_xlabel("Tahun")
ax3.set_ylabel("Rata-rata TPT (%)")
ax3.set_title("Tren Rata-rata TPT per Tahun")
ax3.legend(fontsize=9)
ax3.set_xticks(grouped['Tahun'])
ax3.tick_params(axis='x', rotation=45)

# ── Plot 4: Koefisien (bobot tiap fitur) ──
ax4 = axes[1, 1]
label_singkat = [
    'Tahun', 'Sem: Februari',
    'Pend: SMA Kejuruan', 'Pend: SMA umum',
    'Pend: SMP', 'Pend: SD/Blm Skolah', 'Pend: Universitas'
]
koef_vals = model.coef_
warna_koef = ['#378ADD' if k > 0 else '#D85A30' for k in koef_vals]
bars = ax4.barh(label_singkat, koef_vals, color=warna_koef, alpha=0.8)
ax4.axvline(0, color='gray', linewidth=0.8, linestyle='--')
ax4.set_xlabel("Nilai Koefisien")
ax4.set_title("Koefisien Regresi Tiap Fitur")
for bar, val in zip(bars, koef_vals):
    ax4.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va='center', ha='left' if val >= 0 else 'right', fontsize=9)
ax4.tick_params(axis='y', labelsize=9)

plt.tight_layout()
plt.savefig("hasil_regresi_tpt.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nGrafik disimpan sebagai 'hasil_regresi_tpt.png'")


# ── 12. RINGKASAN AKHIR ────────────────────────────────────────
print("\n" + "=" * 60)
print("  RINGKASAN MODEL")
print("=" * 60)
print(f"""
  Persamaan regresi:
  TPT = {model.intercept_:.2f}
        + ({model.coef_[0]:.4f}) × Tahun
        + ({model.coef_[1]:.4f}) × Semester_Februari
        + ({model.coef_[2]:.4f}) × Pendidikan_SMA_Kejuruan
        + ({model.coef_[3]:.4f}) × Pendidikan_SMA_umum
        + ({model.coef_[4]:.4f}) × Pendidikan_SMP
        + ({model.coef_[5]:.4f}) × Pendidikan_SD
        + ({model.coef_[6]:.4f}) × Pendidikan_Universitas

  Performa model:
    R² Train : {r2_tr:.4f}  ({r2_tr*100:.1f}% variasi terjelaskan)
    R² Test  : {r2_te:.4f}  ({r2_te*100:.1f}% variasi terjelaskan)
    MAE Test : {mae_te:.4f}%
    RMSE Test: {rmse_te:.4f}%

  Interpretasi singkat:
    - Setiap +1 tahun  → TPT turun ~{abs(model.coef_[0]):.2f}% (tren menurun)
    - Semester Februari → TPT lebih tinggi ~{model.coef_[1]:.2f}% vs Agustus
    - SMA Kejuruan paling tinggi TPT-nya (+{model.coef_[2]:.2f}% vs Diploma)
    - SD/Belum Sekolah paling rendah TPT ({model.coef_[5]:.2f}% vs Diploma)
""")
print("=" * 60)
print("  SELESAI")
print("=" * 60)