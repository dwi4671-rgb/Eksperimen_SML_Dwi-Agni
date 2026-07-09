"""
automate_Dwi-Agni.py
=========================
Script preprocessing otomatis & reusable untuk proyek MSML
"Prediksi Pembayaran Tagihan Pajak Daerah" (klasifikasi biner).

Standalone: TIDAK bergantung pada config.py, semua konstanta didefinisikan lokal.

Alur run():
  load_raw -> clean -> train_test_split(stratify=label) ->
  simpan train.csv & test.csv -> fit ColumnTransformer pada X_train ->
  joblib.dump preprocessor.joblib.

Jalankan:
  python automate_Dwi-Agni.py
  python automate_Dwi-Agni.py --raw path/ke/pajak_daerah_raw.csv --out folder/output
"""

import os
import argparse

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# ==========================================================================
# KONTRAK FITUR (harus PERSIS sama di semua berkas proyek)
# ==========================================================================
NUMERIC = [
    "total_tax", "omzet", "fare", "period_days", "deadline_lead_days",
    "registration_age_days", "collection_year", "collection_month", "quarter",
]
CATEGORICAL = [
    "tax_code", "tax_name", "advertisement_type", "upt",
    "district_name", "business_name",
]
TARGET = "label"                       # 1 = dibayar/paid, 0 = belum dibayar/unpaid
DROP_COLS = ["collection_id", "volume", "rental_value", "npa"]

# Direktori tempat script ini berada (dipakai untuk path default)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================================================
# 1. LOAD
# ==========================================================================
def load_raw(path):
    """Baca CSV mentah menjadi DataFrame."""
    df = pd.read_csv(path)
    return df


# ==========================================================================
# 2. CLEAN
# ==========================================================================
def clean(df):
    """
    Pembersihan sesuai kontrak:
      - Buang DROP_COLS.
      - Cast kolom kategorikal ke string; isi NaN
        (advertisement_type -> "NONE", sisanya -> "UNKNOWN").
      - Pastikan kolom numerik bertipe numeric (coerce NaN bila gagal parse).
    """
    df = df.copy()

    # Buang kolom yang tidak dipakai (abaikan bila kolom tidak ada)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Kategorikal -> string + isi NaN
    for col in CATEGORICAL:
        if col in df.columns:
            fill_value = "NONE" if col == "advertisement_type" else "UNKNOWN"
            # pakai nullable string agar NaN tetap NA (bukan literal "nan")
            s = df[col].astype("string").fillna(fill_value)
            df[col] = s.astype(str)

    # Numerik -> pastikan numeric (NaN diimputasi nanti oleh pipeline)
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ==========================================================================
# 3. PREPROCESSOR (KONTRAK)
# ==========================================================================
def build_preprocessor():
    """Bangun ColumnTransformer sesuai kontrak (belum di-fit)."""
    numeric_pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC),
        ("cat", categorical_pipeline, CATEGORICAL),
    ])
    return preprocessor


# ==========================================================================
# 4. RUN (orkestrasi penuh)
# ==========================================================================
def run(raw_csv=None, out_dir=None):
    """
    Jalankan seluruh alur preprocessing dan simpan artefak.

    Default:
      raw_csv = <dir script>/../pajak_daerah_raw.csv
      out_dir = <dir script>/pajak_daerah_preprocessing
    """
    if raw_csv is None:
        raw_csv = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "pajak_daerah_raw.csv"))
    if out_dir is None:
        out_dir = os.path.join(SCRIPT_DIR, "pajak_daerah_preprocessing")

    os.makedirs(out_dir, exist_ok=True)

    # --- Load & clean ---
    print(f"[1/5] Memuat data mentah: {raw_csv}")
    df = load_raw(raw_csv)
    print(f"      Bentuk mentah: {df.shape}")

    print("[2/5] Membersihkan data ...")
    df = clean(df)
    print(f"      Bentuk setelah clean: {df.shape}")

    # --- Pisahkan fitur & target ---
    feature_cols = NUMERIC + CATEGORICAL
    X = df[feature_cols]
    y = df[TARGET]

    # --- Split stratified ---
    print("[3/5] Split train/test (stratify=label, test_size=0.2, random_state=42) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Simpan train.csv & test.csv (fitur + label, belum di-encode) ---
    train_df = X_train.copy()
    train_df[TARGET] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values

    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[4/5] Menyimpan: {train_path} ({train_df.shape[0]} baris)")
    print(f"      Menyimpan: {test_path} ({test_df.shape[0]} baris)")

    # --- Fit preprocessor pada X_train & simpan ---
    print("[5/5] Fit ColumnTransformer pada X_train ...")
    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    n_features_out = X_train_enc.shape[1]

    prep_path = os.path.join(out_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, prep_path)

    # --- Ringkasan ---
    print("\n=== RINGKASAN PREPROCESSING ===")
    print(f"Baris train           : {train_df.shape[0]}")
    print(f"Baris test            : {test_df.shape[0]}")
    print(f"Fitur mentah          : {len(feature_cols)} "
          f"({len(NUMERIC)} numerik + {len(CATEGORICAL)} kategorikal)")
    print(f"Fitur setelah OHE     : {n_features_out}")
    print(f"Preprocessor disimpan : {prep_path}")
    print("================================")

    return {
        "train_path": train_path,
        "test_path": test_path,
        "preprocessor_path": prep_path,
        "n_features_out": n_features_out,
    }


# ==========================================================================
# CLI
# ==========================================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing otomatis dataset pajak daerah (MSML)."
    )
    parser.add_argument("--raw", default=None, help="Path CSV mentah (opsional).")
    parser.add_argument("--out", default=None, help="Folder output (opsional).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(raw_csv=args.raw, out_dir=args.out)
