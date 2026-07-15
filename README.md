# Eksperimen SML - Prediksi Pembayaran Pajak Daerah

Repo eksperimen untuk submission akhir kelas Dicoding **"Membangun Sistem Machine Learning" (MSML)**.
Sistem ML yang dibangun adalah **klasifikasi biner** untuk memprediksi apakah sebuah tagihan
pajak daerah akan **dibayar (paid)** atau **belum dibayar (unpaid)**.

## Deskripsi Dataset

- **Sumber:** SIPD (Sistem Informasi Pajak Daerah) - sistem pemungutan pajak daerah.
- **Ukuran:** 76.205 baris data tagihan pajak daerah.
- **Berkas mentah:** `pajak_daerah_raw.csv`.
- **Target:** kolom `label` (1 = dibayar/paid, 0 = belum dibayar/unpaid).

### Daftar Fitur

**NUMERIC** (dibiarkan numerik, NaN diimputasi median oleh pipeline):

```
total_tax, omzet, fare, period_days, deadline_lead_days,
registration_age_days, collection_year, collection_month, quarter
```

**CATEGORICAL** (di-cast ke string, NaN diisi: `advertisement_type` -> "NONE", sisanya -> "UNKNOWN"):

```
tax_code, tax_name, advertisement_type, upt, district_name, business_name
```

**Kolom yang di-DROP** (tidak dipakai):

```
collection_id (identifier), volume (100% null), rental_value (100% null), npa (99% null)
```

## Pembersihan & Preprocessing

Alur pembersihan yang dipakai di seluruh berkas:

1. Buang kolom DROP (`collection_id`, `volume`, `rental_value`, `npa`).
2. Cast semua kolom CATEGORICAL ke string dan isi NaN
   (`advertisement_type` -> `"NONE"`, kategorikal lain -> `"UNKNOWN"`).
3. Kolom NUMERIC dibiarkan numerik; NaN diimputasi median di dalam pipeline.

Preprocessor berupa `ColumnTransformer`:

- **num:** `SimpleImputer(strategy="median")` -> `StandardScaler`.
- **cat:** `SimpleImputer(strategy="constant", fill_value="UNKNOWN")` -> `OneHotEncoder(handle_unknown="ignore")`.

## Cara Menjalankan

### 1. Install dependency

```bash
pip install -r requirements.txt
```

### 2. Menjalankan notebook eksperimen

Buka notebook eksperimen (EDA + eksperimen preprocessing) di `preprocessing/Eksperimen_Dwi-Agni.ipynb`
dengan Jupyter (notebook membaca `../pajak_daerah_raw.csv` relatif terhadap lokasinya):

```bash
jupyter notebook preprocessing/Eksperimen_Dwi-Agni.ipynb
```

### 3. Menjalankan skrip automate preprocessing

Skrip ini membaca `pajak_daerah_raw.csv`, melakukan pembersihan, split stratified
(`test_size=0.2`, `random_state=42`), lalu menyimpan hasil ke folder
`pajak_daerah_preprocessing/`:

```bash
python preprocessing/automate_Dwi-Agni.py
```

**Output yang dihasilkan (folder `pajak_daerah_preprocessing/`):**

- `train.csv` -> kolom = NUMERIC + CATEGORICAL + `label` (belum di-encode, kategorikal sudah string & terisi).
- `test.csv`  -> sama seperti train, hasil split stratified.
- `preprocessor.joblib` -> `ColumnTransformer` yang sudah di-fit pada data train.

## Struktur Folder

```
repo_eksperimen/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # CI: jalankan preprocessing + upload artifact
├── pajak_daerah_raw.csv                # Dataset mentah (76.205 baris)
├── preprocessing/
│   ├── Eksperimen_Dwi-Agni.ipynb       # Notebook eksperimen (EDA + eksperimen preprocessing)
│   ├── automate_Dwi-Agni.py            # Skrip automate preprocessing
│   └── pajak_daerah_preprocessing/     # Output: train.csv, test.csv, preprocessor.joblib
├── requirements.txt
├── README.md
└── .gitignore
```

## Catatan CI

- Workflow **Preprocessing CI** (`.github/workflows/preprocessing.yml`) berjalan otomatis
  pada setiap `push` dan bisa dipicu manual lewat `workflow_dispatch`.
- Runner: `ubuntu-latest`, Python `3.12`.
- Langkah CI: checkout -> setup Python -> install dependency
  (`pandas==2.2.3`, `scikit-learn==1.5.2`, `numpy==1.26.4`, `joblib==1.4.2`) ->
  jalankan `preprocessing/automate_Dwi-Agni.py` ->
  unggah folder `preprocessing/pajak_daerah_preprocessing` sebagai artifact `preprocessing-output`.
