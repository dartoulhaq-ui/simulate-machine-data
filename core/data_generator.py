"""Generator dataset sintetis (Easy, Medium, Hard), pemuat dataset sklearn, dan ringkasan EDA."""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, make_classification, make_regression
import streamlit as st


@st.cache_data(show_spinner=False)
def generate_easy_dataset(
    source_name: str,
    n_samples: int = 800,
    n_features: int = 5,
    noise: float = 15.0,
    missing_pct: float = 5.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Membuat dataset untuk modul Easy.
    Mengembalikan: (DataFrame, nama_kolom_target, tipe_tugas ['regression'|'classification']).
    """
    rng = np.random.default_rng(seed)

    if source_name == "Dataset Bawaan: Iris (Klasifikasi)":
        raw = load_iris(as_frame=True)
        df = raw.frame.copy()
        target_col = "target"
        task_type = "classification"
    elif source_name == "Dataset Bawaan: Diabetes (Regresi)":
        raw = load_diabetes(as_frame=True)
        df = raw.frame.copy()
        target_col = "target"
        task_type = "regression"
    elif source_name == "Sintetis: Klasifikasi 2 Kelas":
        n_informative = max(2, min(n_features - 1, int(n_features * 0.7)))
        n_redundant = max(0, n_features - n_informative - 1)
        flip_y = min(0.4, max(0.01, noise / 100.0))
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            flip_y=flip_y,
            random_state=seed,
        )
        cols = [f"fitur_{i+1}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df["target_kelas"] = y
        target_col = "target_kelas"
        task_type = "classification"
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(1, n_features - 1),
            noise=noise,
            random_state=seed,
        )
        cols = [f"fitur_{i+1}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df["target_nilai"] = np.round(y, 2)
        target_col = "target_nilai"
        task_type = "regression"

    if missing_pct > 0:
        feature_cols = [c for c in df.columns if c != target_col]
        total_cells = len(df) * len(feature_cols)
        n_missing = int(total_cells * (missing_pct / 100.0))
        if n_missing > 0:
            row_idx = rng.integers(0, len(df), size=n_missing)
            col_idx = rng.integers(0, len(feature_cols), size=n_missing)
            for r, c in zip(row_idx, col_idx):
                df.iat[r, df.columns.get_loc(feature_cols[c])] = np.nan

    return df, target_col, task_type


@st.cache_data(show_spinner=False)
def generate_medium_dataset(
    task_type: str = "classification",
    n_samples: int = 1000,
    missing_pct: float = 8.0,
    outlier_pct: float = 4.0,
    noise: float = 12.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Membuat dataset campuran (numerik + kategorikal + tanggal) dengan missing value & outlier
    untuk modul Medium.
    """
    rng = np.random.default_rng(seed)

    usia = rng.integers(18, 65, size=n_samples).astype(float)
    pendapatan_juta = np.round(rng.normal(12.0, 4.5, size=n_samples).clip(2.0, 45.0), 2)
    skor_aktivitas = np.round(rng.uniform(10.0, 100.0, size=n_samples), 2)
    lama_langganan_bln = rng.integers(1, 72, size=n_samples).astype(float)

    wilayah = rng.choice(
        ["Jawa", "Sumatera", "Kalimantan", "Sulawesi", "Bali_Nusa"],
        size=n_samples,
        p=[0.4, 0.25, 0.15, 0.1, 0.1],
    )
    segmen = rng.choice(
        ["Reguler", "Silver", "Gold", "Platinum"],
        size=n_samples,
        p=[0.45, 0.3, 0.18, 0.07],
    )
    pendidikan = rng.choice(["SMA", "D3", "S1", "S2"], size=n_samples, p=[0.3, 0.2, 0.4, 0.1])

    base_dates = pd.date_range("2025-01-01", periods=365, freq="D")
    tanggal_transaksi = rng.choice(base_dates, size=n_samples)

    segmen_bonus = np.where(segmen == "Platinum", 15.0, np.where(segmen == "Gold", 8.0, 0.0))
    linear_signal = (
        0.35 * pendapatan_juta
        + 0.08 * skor_aktivitas
        + 0.05 * lama_langganan_bln
        - 0.04 * usia
        + segmen_bonus
        + rng.normal(0, max(0.5, noise / 5.0), size=n_samples)
    )

    df = pd.DataFrame(
        {
            "tanggal_transaksi": pd.to_datetime(tanggal_transaksi).strftime("%Y-%m-%d"),
            "usia": usia,
            "pendapatan_juta": pendapatan_juta,
            "skor_aktivitas": skor_aktivitas,
            "lama_langganan_bln": lama_langganan_bln,
            "wilayah": wilayah,
            "segmen": segmen,
            "pendidikan": pendidikan,
        }
    )

    if task_type == "classification":
        threshold = np.median(linear_signal)
        df["status_loyal"] = (linear_signal > threshold).astype(int)
        target_col = "status_loyal"
    else:
        df["nilai_belanja"] = np.round((linear_signal * 10.0) + 50.0, 2)
        target_col = "nilai_belanja"

    num_cols = ["usia", "pendapatan_juta", "skor_aktivitas", "lama_langganan_bln"]
    if outlier_pct > 0:
        n_outliers = max(1, int(n_samples * (outlier_pct / 100.0)))
        out_rows = rng.choice(n_samples, size=n_outliers, replace=False)
        df.loc[out_rows, "pendapatan_juta"] = df.loc[out_rows, "pendapatan_juta"] * rng.uniform(5.0, 9.0, size=n_outliers)
        df.loc[out_rows, "skor_aktivitas"] = df.loc[out_rows, "skor_aktivitas"] * rng.uniform(3.5, 6.0, size=n_outliers)

    if missing_pct > 0:
        candidate_cols = num_cols + ["wilayah", "segmen", "pendidikan"]
        total_cells = n_samples * len(candidate_cols)
        n_missing = int(total_cells * (missing_pct / 100.0))
        if n_missing > 0:
            r_idx = rng.integers(0, n_samples, size=n_missing)
            c_idx = rng.integers(0, len(candidate_cols), size=n_missing)
            for r, c in zip(r_idx, c_idx):
                col_name = candidate_cols[c]
                df.loc[r, col_name] = np.nan

    return df, target_col, task_type


@st.cache_data(show_spinner=False)
def generate_hard_dataset(
    task_type: str = "classification",
    n_samples: int = 1200,
    n_features: int = 12,
    minority_ratio: float = 0.15,
    noise: float = 10.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Membuat dataset untuk modul Hard (Imbalanced Classification atau Regresi Kompleks
    dengan fitur informatif + fitur redundan/noise untuk diuji pada seleksi fitur RFE).
    """
    n_informative = max(3, int(n_features * 0.5))
    n_redundant = max(1, int(n_features * 0.2))

    if task_type == "classification":
        majority_weight = max(0.5, min(0.95, 1.0 - minority_ratio))
        flip_y = min(0.25, max(0.01, noise / 100.0))
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            weights=[majority_weight, 1.0 - majority_weight],
            flip_y=flip_y,
            class_sep=1.1,
            random_state=seed,
        )
        cols = [
            f"fitur_inti_{i+1}" if i < n_informative
            else (f"fitur_redundan_{i+1}" if i < n_informative + n_redundant else f"fitur_noise_{i+1}")
            for i in range(n_features)
        ]
        df = pd.DataFrame(np.round(X, 4), columns=cols)
        df["target_fraud"] = y
        target_col = "target_fraud"
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=seed,
        )
        cols = [
            f"fitur_inti_{i+1}" if i < n_informative else f"fitur_noise_{i+1}"
            for i in range(n_features)
        ]
        df = pd.DataFrame(np.round(X, 4), columns=cols)
        df["target_skor"] = np.round(y, 2)
        target_col = "target_skor"

    return df, target_col, task_type


def load_uploaded_csv(uploaded_file: Any, max_rows: int = 20000) -> pd.DataFrame:
    """Membaca file CSV yang diunggah pengguna dengan batas keamanan baris."""
    df = pd.read_csv(uploaded_file)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Menghitung ringkasan EDA: shape, tipe data, persentase missing, dan statistik deskriptif."""
    n_rows, n_cols = df.shape
    missing_count = df.isna().sum()
    missing_pct = (missing_count / max(1, n_rows) * 100.0).round(2)

    info_df = pd.DataFrame(
        {
            "Kolom": df.columns,
            "Tipe Data": [str(dt) for dt in df.dtypes],
            "Jumlah Kosong (Missing)": missing_count.values,
            "Missing (%)": missing_pct.values,
            "Nilai Unik": [df[col].nunique(dropna=True) for col in df.columns],
        }
    )

    describe_num = df.describe().T.round(3)
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_missing": int(missing_count.sum()),
        "total_missing_pct": round(float(missing_count.sum()) / max(1, n_rows * n_cols) * 100.0, 2),
        "info_df": info_df,
        "describe_num": describe_num,
    }