"""Modul imputasi missing value, penanganan outlier, feature engineering, dan ColumnTransformer."""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)


def impute_dataframe(
    df: pd.DataFrame,
    target_col: str,
    strategy: str = "median",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Melakukan imputasi pada fitur DataFrame dan mengembalikan:
    (df_bersih, tabel_perbandingan_sebelum_sesudah).
    Pilihan strategy: 'mean', 'median', 'most_frequent', 'knn', 'drop'.
    """
    df_clean = df.copy()
    before_missing = df_clean.isna().sum()

    # Pastikan baris yang targetnya kosong selalu dihapus terlebih dahulu
    df_clean = df_clean.dropna(subset=[target_col]).reset_index(drop=True)

    feature_cols = [c for c in df_clean.columns if c != target_col]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_clean[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df_clean[c])]

    if strategy == "drop":
        df_clean = df_clean.dropna().reset_index(drop=True)
    elif strategy == "knn":
        if num_cols:
            knn_imp = KNNImputer(n_neighbors=5)
            df_clean[num_cols] = knn_imp.fit_transform(df_clean[num_cols])
        if cat_cols:
            cat_imp = SimpleImputer(strategy="most_frequent")
            df_clean[cat_cols] = cat_imp.fit_transform(df_clean[cat_cols])
    else:
        num_strategy = strategy if strategy in ["mean", "median", "most_frequent"] else "median"
        if num_cols:
            num_imp = SimpleImputer(strategy=num_strategy)
            df_clean[num_cols] = num_imp.fit_transform(df_clean[num_cols])
        if cat_cols:
            cat_imp = SimpleImputer(strategy="most_frequent")
            df_clean[cat_cols] = cat_imp.fit_transform(df_clean[cat_cols])

    after_missing = df_clean.isna().sum()
    comparison_df = pd.DataFrame(
        {
            "Kolom": df.columns,
            "Missing Sebelum": [int(before_missing[col]) for col in df.columns],
            "Missing Sesudah": [int(after_missing.get(col, 0)) for col in df.columns],
        }
    )
    return df_clean, comparison_df


def detect_and_handle_outliers(
    df: pd.DataFrame,
    target_col: str,
    method: str = "iqr",
    action: str = "clip",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mendeteksi outlier pada kolom numerik menggunakan 'iqr' (1.5 * IQR) atau 'zscore' (|Z| > 3).
    Pilihan action: 'none' (hanya deteksi), 'clip' (pangkas ke batas bawah/atas), 'remove' (hapus baris).
    """
    df_out = df.copy()
    num_cols = [
        c for c in df_out.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df_out[c])
    ]

    report_rows: List[Dict[str, Any]] = []
    rows_to_drop = pd.Series(False, index=df_out.index)

    for col in num_cols:
        series = df_out[col].dropna()
        if len(series) < 5:
            continue

        if method == "zscore":
            mean_val = float(series.mean())
            std_val = float(series.std()) if float(series.std()) > 0 else 1.0
            lower_bound = mean_val - 3.0 * std_val
            upper_bound = mean_val + 3.0 * std_val
        else:
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

        is_outlier = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
        outlier_count = int(is_outlier.sum())

        if action == "clip" and outlier_count > 0:
            df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
        elif action == "remove" and outlier_count > 0:
            rows_to_drop = rows_to_drop | is_outlier

        report_rows.append(
            {
                "Kolom Numerik": col,
                "Metode": method.upper(),
                "Batas Bawah": round(lower_bound, 2),
                "Batas Atas": round(upper_bound, 2),
                "Jumlah Outlier Terdeteksi": outlier_count,
                "Tindakan": action.upper(),
            }
        )

    if action == "remove":
        df_out = df_out.loc[~rows_to_drop].reset_index(drop=True)

    report_df = pd.DataFrame(report_rows)
    return df_out, report_df


def apply_feature_engineering(
    df: pd.DataFrame,
    target_col: str,
    use_interaction: bool = True,
    use_ratio: bool = True,
    use_binning: bool = False,
    use_date_extract: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Membuat fitur-fitur baru (Feature Engineering) secara aman:
    - Ekstraksi tanggal (bulan, hari dalam minggu, akhir pekan) lalu membuang string tanggal mentah.
    - Interaksi perkalian 2 kolom numerik utama.
    - Rasio pembagian 2 kolom numerik utama.
    - Binning (pengelompokan kuartil) pada kolom numerik pertama.
    """
    df_fe = df.copy()
    created_cols: List[str] = []

    # 1. Ekstraksi fitur tanggal jika ada kolom bertipe tanggal / bernama tanggal
    date_candidates = [
        c for c in df_fe.columns
        if c != target_col and ("tanggal" in c.lower() or "date" in c.lower())
    ]
    for dcol in date_candidates:
        parsed = pd.to_datetime(df_fe[dcol], errors="coerce")
        if parsed.notna().sum() > 0:
            if use_date_extract:
                df_fe[f"{dcol}_bulan"] = parsed.dt.month.fillna(1).astype(int)
                df_fe[f"{dcol}_hari_minggu"] = parsed.dt.dayofweek.fillna(0).astype(int)
                df_fe[f"{dcol}_akhir_pekan"] = (parsed.dt.dayofweek >= 5).astype(int)
                created_cols.extend([f"{dcol}_bulan", f"{dcol}_hari_minggu", f"{dcol}_akhir_pekan"])
            # Hapus kolom tanggal string mentah agar tidak membuat OneHotEncoder membengkak
            df_fe = df_fe.drop(columns=[dcol])

    num_cols = [
        c for c in df_fe.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df_fe[c])
    ]

    # 2. Fitur interaksi perkalian
    if use_interaction and len(num_cols) >= 2:
        col_a, col_b = num_cols[0], num_cols[1]
        new_name = f"interaksi_{col_a}_x_{col_b}"
        df_fe[new_name] = (df_fe[col_a] * df_fe[col_b]).round(3)
        created_cols.append(new_name)

    # 3. Fitur rasio pembagian
    if use_ratio and len(num_cols) >= 2:
        col_a, col_b = num_cols[1], num_cols[0]
        new_name = f"rasio_{col_a}_per_{col_b}"
        denom = np.where(np.abs(df_fe[col_b]) < 1e-6, 1.0, df_fe[col_b])
        df_fe[new_name] = (df_fe[col_a] / denom).round(3)
        created_cols.append(new_name)

    # 4. Fitur binning (pengelompokan kategori kuartil)
    if use_binning and len(num_cols) >= 1:
        col_first = num_cols[0]
        new_name = f" grup_{col_first}".strip()
        filled_series = df_fe[col_first].fillna(df_fe[col_first].median())
        try:
            df_fe[new_name] = pd.qcut(
                filled_series,
                q=4,
                labels=["Rendah", "Menengah_Bawah", "Menengah_Atas", "Tinggi"],
                duplicates="drop",
            ).astype(str)
            created_cols.append(new_name)
        except ValueError:
            pass

    return df_fe, created_cols


def build_column_transformer(
    num_cols: List[str],
    cat_cols: List[str],
    num_imputer_type: str = "median",
    scaler_type: str = "standard",
    encoder_type: str = "onehot",
    task_type: str = "classification",
    seed: int = 42,
) -> ColumnTransformer:
    """
    Membangun scikit-learn ColumnTransformer yang menggabungkan:
    - Pipeline Numerik: Imputer (Mean/Median/KNN) + Scaler (Standard/MinMax/Robust)
    - Pipeline Kategorikal: Imputer (Modus) + Encoder (OneHot/Ordinal/Target)
    """
    if num_imputer_type == "knn":
        num_imputer = KNNImputer(n_neighbors=5)
    elif num_imputer_type == "mean":
        num_imputer = SimpleImputer(strategy="mean")
    else:
        num_imputer = SimpleImputer(strategy="median")

    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    num_pipeline = Pipeline(
        steps=[
            ("imputer", num_imputer),
            ("scaler", scaler),
        ]
    )

    if encoder_type == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    elif encoder_type == "target":
        target_type = "continuous" if task_type == "regression" else "auto"
        encoder = TargetEncoder(target_type=target_type, random_state=seed)
    else:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")