"""Modul utilitas untuk mengekspor hasil simulasi (CSV metrik/leaderboard, model .joblib, dan notebook .ipynb)."""

import io
from typing import Dict, Any, List, Tuple
import joblib
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
import pandas as pd


def metrics_to_csv_bytes(metrics: Dict[str, Any], model_name: str, seed: int) -> bytes:
    """Mengubah dictionary metrik evaluasi menjadi bytes CSV yang siap diunduh."""
    row = {"Model": model_name, "Global_Seed": seed}
    row.update(metrics)
    df_metrics = pd.DataFrame([row])
    return df_metrics.to_csv(index=False).encode("utf-8")


def leaderboard_to_csv_bytes(leaderboard_df: pd.DataFrame, seed: int) -> bytes:
    """Mengekspor tabel leaderboard multi-model menjadi bytes CSV."""
    df_out = leaderboard_df.copy()
    df_out["Global_Seed"] = seed
    return df_out.to_csv(index=False).encode("utf-8")


def model_to_joblib_bytes(model_obj: Any) -> bytes:
    """Menyimpan objek model / pipeline scikit-learn ke dalam buffer bytes .joblib."""
    buffer = io.BytesIO()
    joblib.dump(model_obj, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def snippet_to_ipynb_bytes(
    title: str,
    summary_markdown: str,
    python_code: str,
    metrics: Dict[str, Any],
    seed: int,
) -> bytes:
    """Membuat file Jupyter Notebook (.ipynb) dari potongan kode simulasi."""
    nb = new_notebook()
    metrics_lines = "\n".join([f"- **{k}**: `{v}`" for k, v in metrics.items()])

    nb.cells.append(
        new_markdown_cell(
            f"# {title}\n"
            f"*Diekspor otomatis dari **SIMULATE MACHINE DATA** (`GLOBAL_SEED = {seed}`)*\n\n"
            f"## Ringkasan Eksperimen\n{summary_markdown}\n\n"
            f"### Hasil Metrik Evaluasi\n{metrics_lines}"
        )
    )
    nb.cells.append(new_markdown_cell("## Kode Reproduksi Pipeline & Model"))
    nb.cells.append(new_code_cell(python_code))

    notebook_str = nbformat.writes(nb)
    return notebook_str.encode("utf-8")


def build_structured_notebook_bytes(
    lang: str,
    project_title: str,
    team_name: str,
    competition_goal: str,
    selected_sections: List[str],
    target_col: str,
    model_choice: str,
    seed: int,
) -> Tuple[bytes, List[Dict[str, str]]]:
    """
    Generator kerangka Jupyter Notebook (.ipynb) lengkap dengan pilihan Bahasa Indonesia / English
    untuk halaman Notebook Builder (Target Pembelajaran #3).
    Mengembalikan (bytes_ipynb, list_pratinjau_sel).
    """
    nb = new_notebook()
    preview_cells: List[Dict[str, str]] = []

    def add_md(content: str) -> None:
        nb.cells.append(new_markdown_cell(content))
        preview_cells.append({"type": "markdown", "content": content})

    def add_code(content: str) -> None:
        nb.cells.append(new_code_cell(content))
        preview_cells.append({"type": "code", "content": content})

    is_id = lang.lower().startswith("id")

    if is_id:
        add_md(
            f"# 🏆 {project_title}\n"
            f"**Tim:** {team_name}  \n"
            f"**Global Random Seed:** `{seed}`  \n"
            f"**Tujuan Kompetisi:** {competition_goal}\n\n---"
        )
    else:
        add_md(
            f"# 🏆 {project_title}\n"
            f"**Team:** {team_name}  \n"
            f"**Global Random Seed:** `{seed}`  \n"
            f"**Competition Goal:** {competition_goal}\n\n---"
        )

    if "Pendahuluan" in selected_sections:
        if is_id:
            add_md(
                "## 1. Pendahuluan & Konfigurasi Reproduksibilitas\n"
                "Pada bagian ini, kami mengunci *random seed* global dan menampilkan versi pustaka (*library*) "
                "agar seluruh eksperimen dapat direproduksi 100% oleh juri maupun anggota tim lain."
            )
        else:
            add_md(
                "## 1. Introduction & Reproducibility Setup\n"
                "In this section, we lock the global random seed and log library versions "
                "so that all experiments are 100% reproducible."
            )
        add_code(
            f"""import sys
import numpy as np
import pandas as pd
import sklearn

GLOBAL_SEED = {seed}
np.random.seed(GLOBAL_SEED)

print("Python version:", sys.version.split()[0])
print("Pandas version:", pd.__version__)
print("Scikit-Learn version:", sklearn.__version__)"""
        )

    if "EDA" in selected_sections:
        if is_id:
            add_md(
                "## 2. Exploratory Data Analysis (EDA)\n"
                "Memeriksa dimensi dataset, tipe data, persentase *missing values*, distribusi target, "
                "serta mendeteksi adanya *outlier*."
            )
        else:
            add_md(
                "## 2. Exploratory Data Analysis (EDA)\n"
                "Inspecting dataset dimensions, data types, missing value percentages, target distribution, "
                "and potential outliers."
            )
        add_code(
            f"""train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
TARGET_COL = "{target_col}"

print("Train shape:", train_df.shape, "| Test shape:", test_df.shape)
display(train_df.head())
display(train_df.describe().T)

missing_pct = (train_df.isna().sum() / len(train_df) * 100).round(2)
display(missing_pct[missing_pct > 0])"""
        )

    if "Preprocessing" in selected_sections:
        if is_id:
            add_md(
                "## 3. Preprocessing & Feature Engineering\n"
                "Membangun `ColumnTransformer` dan `Pipeline` untuk menangani *missing values*, "
                "*encoding* kolom kategorikal, serta *scaling* numerik tanpa kebocoran data (*data leakage*)."
            )
        else:
            add_md(
                "## 3. Preprocessing & Feature Engineering\n"
                "Building a `ColumnTransformer` and `Pipeline` to handle missing values, "
                "categorical encoding, and numerical scaling without data leakage."
            )
        add_code(
            """from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

feature_cols = [c for c in train_df.columns if c != TARGET_COL]
num_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_cols)
])"""
        )

    if "Modeling" in selected_sections:
        if is_id:
            add_md(
                f"## 4. Pengembangan Model & Cross-Validation ({model_choice})\n"
                "Melatih model di dalam kerangka *K-Fold Cross-Validation* agar estimasi skor lokal "
                "sejalan dengan *Private Leaderboard*."
            )
        else:
            add_md(
                f"## 4. Model Development & Cross-Validation ({model_choice})\n"
                "Training the model inside a *K-Fold Cross-Validation* scheme so local validation "
                "aligns reliably with the *Private Leaderboard*."
            )
        add_code(
            f"""from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

X = train_df[feature_cols]
y = train_df[TARGET_COL]

model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.08,
    random_state=GLOBAL_SEED,
    eval_metric="logloss"
)

clf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
cv_scores = cross_val_score(clf_pipeline, X, y, cv=cv, scoring="f1")
print(f"5-Fold CV Mean: {{cv_scores.mean():.4f}} +/- {{cv_scores.std():.4f}}")"""
        )

    if "Evaluasi" in selected_sections:
        if is_id:
            add_md(
                "## 5. Evaluasi Model, Analisis Error & Submission\n"
                "Memeriksa fitur paling berpengaruh (*Feature Importance*) dan mengekspor file `submission.csv`."
            )
        else:
            add_md(
                "## 5. Model Evaluation, Error Analysis & Submission\n"
                "Inspecting top feature importances and exporting the final `submission.csv` file."
            )
        add_code(
            """clf_pipeline.fit(X, y)
test_preds = clf_pipeline.predict(test_df[feature_cols])

submission = pd.DataFrame({
    "id": test_df.index,
    TARGET_COL: test_preds
})
submission.to_csv("submission.csv", index=False)
print("File submission.csv berhasil dibuat!")"""
        )

    if "Kesimpulan" in selected_sections:
        if is_id:
            add_md(
                "## 6. Kesimpulan & Langkah Pengembangan Berikutnya\n"
                "- **Temuan Utama EDA**: Beberapa fitur numerik memiliki kemiringan (*skewness*) sehingga imputasi *median* lebih stabil.\n"
                "- **Hasil Validasi**: Skor *5-Fold Cross-Validation* konsisten dengan deviasi antar-fold yang rendah.\n"
                "- **Rencana Selanjutnya**: Menambahkan *Stacking Ensemble* dan *Feature Interaction* pada fitur terpenting."
            )
        else:
            add_md(
                "## 6. Conclusion & Future Improvements\n"
                "- **Key EDA Findings**: Several numerical features are skewed, making median imputation more robust.\n"
                "- **Validation Results**: 5-Fold Cross-Validation scores are stable with low variance across folds.\n"
                "- **Next Steps**: Explore Stacking Ensembles and domain-specific feature interactions."
            )

    return nbformat.writes(nb).encode("utf-8"), preview_cells