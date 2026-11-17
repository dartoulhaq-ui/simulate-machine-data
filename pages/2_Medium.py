"""Halaman Modul 2: MEDIUM - Bersihkan & Bangun Fitur (Outlier, FE, ColumnTransformer & K-Fold CV)."""

import pandas as pd
import streamlit as st

from core.data_generator import generate_medium_dataset, load_uploaded_csv
from core.evaluation import (
    plot_boxplot_with_insight,
    plot_categorical_bar,
    plot_confusion_matrix_chart,
    plot_feature_importance_chart,
    plot_residuals_chart,
)
from core.export import metrics_to_csv_bytes, model_to_joblib_bytes, snippet_to_ipynb_bytes
from core.models import train_medium_pipeline
from core.preprocessing import apply_feature_engineering, detect_and_handle_outliers
from core.utils import (
    get_global_seed,
    init_session_state,
    load_css,
    render_page_header,
    render_sidebar_controls,
)

st.set_page_config(page_title="Medium — Bersihkan & Bangun Fitur", page_icon="🟡", layout="wide")
init_session_state()
load_css()
render_sidebar_controls()

global_seed = get_global_seed()

# 1. HEADER
render_page_header(
    title="MODUL 2: BERSIHKAN & BANGUN FITUR",
    badge_text="🟡 LEVEL 2: MEDIUM",
    badge_class="badge-medium",
    est_time="15 Menit",
    subtitle="Deteksi outlier, Feature Engineering, Encoding, Scaling, dan Pipeline + K-Fold Cross-Validation.",
)

# 2. TUJUAN BELAJAR
st.markdown(
    """
    <div class="neo-card">
        <h3>🎯 Tujuan Belajar Modul Medium</h3>
        <ul>
            <li>Mendeteksi dan menangani <strong>Outlier</strong> menggunakan metode <strong>IQR</strong> dan <strong>Z-Score</strong>.</li>
            <li>Menciptakan fitur baru (<strong>Feature Engineering</strong>: interaksi, rasio, binning, dan ekstraksi waktu) untuk mendongkrak akurasi model.</li>
            <li>Membangun <code>Pipeline</code> + <code>ColumnTransformer</code> (Imputasi KNN, Encoding Kategori, Scaling Numerik) yang dievaluasi dengan <strong>K-Fold Cross-Validation</strong>.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# 3. TEORI SINGKAT (Expander default tertutup)
with st.expander("📖 Penjelasan Lengkap: Outlier, Encoding, Scaling, Pipeline & K-Fold CV", expanded=False):
    st.markdown(
        """
        ### 1. Deteksi Outlier: IQR vs Z-Score
        - **IQR (Interquartile Range)**: Menghitung jarak kuartil tengah ($Q_3 - Q_1$). Nilai di luar $[Q_1 - 1.5 \\times IQR, Q_3 + 1.5 \\times IQR]$ dianggap outlier. Sangat tangguh untuk data yang miring.
        - **Z-Score**: Mengukur berapa standar deviasi suatu titik menjauh dari rata-rata ($|Z| > 3$).
        - **Tips Kaggle**: Jangan buru-buru menghapus baris outlier (karena pada data test Kaggle kita tidak boleh menghapus baris!). Gunakan **Clipping (Winsorizing)** atau **RobustScaler**.

        ### 2. Mengapa Harus Memakai `Pipeline` + `ColumnTransformer`?
        Jika Anda melakukan `StandardScaler` atau `TargetEncoder` pada seluruh tabel *sebelum* `train_test_split` atau `K-Fold`, maka informasi dari data ujian bocor ke data latihan (**Data Leakage**). Akibatnya skor di laptop terlihat 99%, tapi hancur saat di-submit ke Kaggle. `Pipeline` memastikan transformasi hanya di-*fit* pada lipatan data latihan.

        ### 3. K-Fold Cross-Validation (K-Fold CV)
        Daripada hanya mengandalkan satu kali *split*, K-Fold membagi data menjadi $K$ bagian (misal 5 lipatan) dan menguji model secara bergiliran. Tim Kaggle yang baik selalu berpatokan pada **rata-rata skor CV**, bukan semata-mata skor Public Leaderboard.
        """
    )

st.markdown("### 🎛️ 4. Panel Kontrol Preprocessing, Feature Engineering & Pipeline")
uploaded_csv = st.file_uploader(
    "📂 Opsional: Upload file CSV sendiri (Maks 10 MB) — kosongkan untuk memakai Dataset Campuran Sintetis",
    type=["csv"],
    key="medium_csv_upload",
)

# 4 & 5. PANEL KONTROL DALAM st.form + TOMBOL JALANKAN SIMULASI
with st.form("medium_simulation_form"):
    c_data, c_prep, c_fe, c_mod = st.columns(4)

    with c_data:
        st.markdown("**A. Dataset Campuran**")
        task_choice = st.selectbox(
            "Tipe Masalah",
            options=["classification", "regression"],
            format_func=lambda x: "Klasifikasi (Status Loyal)" if x == "classification" else "Regresi (Nilai Belanja)",
            disabled=(uploaded_csv is not None),
        )
        n_samples = st.slider("Jumlah Baris", 300, 15000, 1000, step=100)
        missing_pct = st.slider("% Missing Value", 0.0, 25.0, 8.0, step=1.0)
        outlier_pct = st.slider("% Outlier Ekstrem", 0.0, 15.0, 4.0, step=1.0)

    with c_prep:
        st.markdown("**B. Outlier & Transformasi**")
        outlier_method = st.selectbox("Metode Deteksi Outlier", ["iqr", "zscore"], format_func=str.upper)
        outlier_action = st.selectbox(
            "Tindakan Outlier",
            ["clip", "remove", "none"],
            format_func=lambda x: {
                "clip": "Clip (Pangkas ke Batas Aman)",
                "remove": "Remove (Hapus Baris Outlier)",
                "none": "Biarkan (Hanya Deteksi)",
            }[x],
        )
        num_imputer_type = st.selectbox(
            "Imputasi Numerik",
            ["median", "mean", "knn"],
            format_func=lambda x: "KNN Imputer (5 Tetangga)" if x == "knn" else f"SimpleImputer ({x.title()})",
        )
        scaler_type = st.selectbox(
            "Scaling Numerik",
            ["standard", "minmax", "robust"],
            format_func=lambda x: {
                "standard": "StandardScaler (Z-Score)",
                "minmax": "MinMaxScaler (0 s.d. 1)",
                "robust": "RobustScaler (Tahan Outlier)",
            }[x],
        )
        encoder_type = st.selectbox(
            "Encoding Kategori",
            ["onehot", "ordinal", "target"],
            format_func=lambda x: {
                "onehot": "One-Hot Encoding",
                "ordinal": "Ordinal Encoding",
                "target": "Target Encoding",
            }[x],
        )

    with c_fe:
        st.markdown("**C. Feature Engineering**")
        fe_date = st.checkbox("📅 Ekstrak Tanggal (Bulan/Hari/Weekend)", value=True)
        fe_inter = st.checkbox("✖️ Fitur Interaksi (Perkalian 2 Fitur)", value=True)
        fe_ratio = st.checkbox("➗ Fitur Rasio (Pembagian 2 Fitur)", value=True)
        fe_bin = st.checkbox("📊 Fitur Binning (Kuartil Kategori)", value=False)

    with c_mod:
        st.markdown("**D. Model & K-Fold CV**")
        model_choice = st.selectbox(
            "Pilih Algoritma",
            ["Random Forest", "Decision Tree", "KNN", "SVM"],
        )
        cv_folds = st.slider("Jumlah Lipatan K-Fold CV", 3, 10, 5)
        n_estimators = st.slider("n_estimators (Khusus RF)", 50, 200, 100, step=25)
        max_depth = st.slider("max_depth (Tree / RF)", 2, 16, 8)
        n_neighbors = st.slider("n_neighbors (Khusus KNN)", 3, 21, 5, step=2)
        svm_c = st.select_slider("Parameter C (Khusus SVM)", options=[0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)

    run_medium = st.form_submit_button("▶ Jalankan Simulasi Medium", use_container_width=True)

if run_medium or "medium_sim_state" not in st.session_state:
    try:
        with st.spinner("Mendeteksi outlier, membangun fitur baru, dan menjalankan K-Fold Cross-Validation..."):
            if uploaded_csv is not None:
                df_raw = load_uploaded_csv(uploaded_csv)
                target_col = df_raw.columns[-1]
                task_type = "classification" if df_raw[target_col].nunique(dropna=True) <= 15 else "regression"
            else:
                df_raw, target_col, task_type = generate_medium_dataset(
                    task_type=task_choice,
                    n_samples=n_samples,
                    missing_pct=missing_pct,
                    outlier_pct=outlier_pct,
                    seed=global_seed,
                )

            df_outlier_handled, outlier_report_df = detect_and_handle_outliers(
                df_raw, target_col=target_col, method=outlier_method, action=outlier_action
            )

            df_fe, created_cols = apply_feature_engineering(
                df_outlier_handled,
                target_col=target_col,
                use_interaction=fe_inter,
                use_ratio=fe_ratio,
                use_binning=fe_bin,
                use_date_extract=fe_date,
            )

            hyperparams = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "n_neighbors": n_neighbors,
                "C": svm_c,
            }

            pipeline_res = train_medium_pipeline(
                df=df_fe,
                target_col=target_col,
                task_type=task_type,
                model_choice=model_choice,
                num_imputer_type=num_imputer_type,
                scaler_type=scaler_type,
                encoder_type=encoder_type,
                cv_folds=cv_folds,
                test_size=0.2,
                hyperparams=hyperparams,
                seed=global_seed,
            )

            st.session_state["medium_sim_state"] = {
                "df_raw": df_raw,
                "df_outlier_handled": df_outlier_handled,
                "df_fe": df_fe,
                "target_col": target_col,
                "task_type": task_type,
                "outlier_report_df": outlier_report_df,
                "created_cols": created_cols,
                "pipeline_res": pipeline_res,
                "seed_used": global_seed,
                "cv_folds": cv_folds,
            }
            st.session_state["progress"]["medium_prep"] = True
            st.session_state["progress"]["medium_pipeline"] = True
    except Exception as exc:
        st.error(f"Gagal menjalankan pipeline Medium: {exc}")

# 6. HASIL SIMULASI
if "medium_sim_state" in st.session_state:
    sim = st.session_state["medium_sim_state"]
    df_raw: pd.DataFrame = sim["df_raw"]
    df_fe: pd.DataFrame = sim["df_fe"]
    target_col: str = sim["target_col"]
    task_type: str = sim["task_type"]
    res = sim["pipeline_res"]

    st.markdown("---")
    st.markdown("## 📊 6. Hasil Pembersihan, Feature Engineering & Evaluasi Pipeline")

    m_cols = st.columns(len(res["metrics"]))
    for idx, (k, v) in enumerate(res["metrics"].items()):
        m_cols[idx].metric(k.replace("_", " "), f"{v}")

    tab_out, tab_viz, tab_eval = st.tabs(
        ["🧹 1. Laporan Outlier & Fitur Baru", "📦 2. Visualisasi Boxplot & Kategori", "🏆 3. Evaluasi K-Fold & Feature Importance"]
    )

    with tab_out:
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.markdown("#### Laporan Deteksi & Penanganan Outlier")
            st.dataframe(sim["outlier_report_df"], use_container_width=True, hide_index=True)
        with col_o2:
            st.markdown(f"#### Fitur Baru yang Berhasil Dibuat ({len(sim['created_cols'])} kolom)")
            if sim["created_cols"]:
                st.success("Kolom baru: " + ", ".join([f"`{c}`" for c in sim["created_cols"]]))
                st.dataframe(df_fe[sim["created_cols"]].head(8), use_container_width=True)
            else:
                st.warning("Tidak ada fitur baru yang diaktifkan pada panel Feature Engineering.")

    with tab_viz:
        num_cols_raw = [c for c in df_raw.columns if c != target_col and pd.api.types.is_numeric_dtype(df_raw[c])]
        cat_cols_raw = [
            c for c in df_raw.columns
            if c != target_col and not pd.api.types.is_numeric_dtype(df_raw[c]) and "tanggal" not in c.lower()
        ]

        v1, v2 = st.columns(2)
        with v1:
            if num_cols_raw:
                sel_num = st.selectbox("Pilih Kolom Numerik untuk Boxplot (Data Setelah Outlier Action):", num_cols_raw)
                fig_box, insight_box = plot_boxplot_with_insight(sim["df_outlier_handled"], sel_num)
                st.plotly_chart(fig_box, use_container_width=True)
                st.info(insight_box)
        with v2:
            if cat_cols_raw:
                sel_cat = st.selectbox("Pilih Kolom Kategorikal untuk Grafik Frekuensi:", cat_cols_raw)
                fig_bar, insight_bar = plot_categorical_bar(df_raw, sel_cat)
                st.plotly_chart(fig_bar, use_container_width=True)
                st.info(insight_bar)

    with tab_eval:
        st.markdown(
            f"**Skor Tiap Lipatan ({sim['cv_folds']}-Fold Cross-Validation):** "
            + " | ".join([f"Fold {i+1}: `{s:.4f}`" for i, s in enumerate(res["cv_scores"])])
        )
        ev1, ev2 = st.columns(2)
        with ev1:
            if task_type == "classification":
                fig_cm, insight_cm = plot_confusion_matrix_chart(res["y_test"], res["y_pred"])
            else:
                fig_cm, insight_cm = plot_residuals_chart(res["y_test"], res["y_pred"], seed=sim["seed_used"])
            st.plotly_chart(fig_cm, use_container_width=True)
            st.info(insight_cm)
        with ev2:
            fig_fi, insight_fi = plot_feature_importance_chart(res["feature_names"], res["importances"])
            st.plotly_chart(fig_fi, use_container_width=True)
            st.info(insight_fi)

    # 7. PENJELASAN LANGKAH DEMI LANGKAH (Expander)
    with st.expander("🔍 7. Penjelasan Langkah demi Langkah (Mengapa Pipeline Ini Aman dari Data Leakage?)", expanded=False):
        st.markdown(
            f"""
            1. **Penanganan Outlier**: Data diperiksa menggunakan metode yang dipilih sehingga nilai ekstrem tidak merusak skala fitur.
            2. **Feature Engineering**: Sistem menambahkan **{len(sim['created_cols'])} kolom baru** untuk menangkap pola non-linear dan waktu.
            3. **ColumnTransformer di dalam K-Fold**: Saat `{sim['cv_folds']}-Fold CV` berjalan, rata-rata imputer, parameter scaler, dan kamus encoder hanya dihitung dari $K-1$ lipatan latihan, lalu diterapkan ke 1 lipatan validasi.
            4. **Evaluasi Akhir**: Model **{res['model_name']}** diuji pada **{res['n_test']} baris data test** yang belum pernah dilihat sama sekali.
            """
        )

    # 8. KODE YANG SETARA
    st.markdown("### 💻 8. Kode Python Setara (`Pipeline` + `ColumnTransformer`)")
    st.code(res["code"], language="python")

    # 9. LATIHAN MANDIRI
    st.markdown(
        """
        <div class="neo-card">
            <h3>✍ 9. Tantangan Latihan Mandiri</h3>
            <ol>
                <li>Pilih algoritma <strong>KNN</strong> atau <strong>SVM</strong>, lalu bandingkan skornya saat <em>Tindakan Outlier = Biarkan</em> dengan <em>Scaling = MinMaxScaler</em> vs saat <em>Tindakan Outlier = Clip</em> dengan <em>Scaling = RobustScaler</em>.</li>
                <li>Matikan semua centang <strong>Feature Engineering</strong>, catat skor <code>CV Mean</code> untuk <strong>Random Forest</strong>, lalu nyalakan kembali semua centang dan lihat apakah fitur interaksi baru masuk ke jajaran atas grafik <em>Feature Importance</em>!</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 10. EKSPOR HASIL
    st.markdown("### ⬇ 10. Ekspor Artefak Pipeline Medium")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.download_button(
            "📄 Unduh Metrik CV & Test (.csv)",
            data=metrics_to_csv_bytes(res["metrics"], res["model_name"], sim["seed_used"]),
            file_name="medium_pipeline_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ex2:
        st.download_button(
            "🤖 Unduh Full Pipeline (.joblib)",
            data=model_to_joblib_bytes(res["pipeline"]),
            file_name="medium_full_pipeline.joblib",
            mime="application/octet-stream",
            use_container_width=True,
        )
    with ex3:
        ipynb_bytes = snippet_to_ipynb_bytes(
            title=f"Modul Medium - Pipeline {res['model_name']}",
            summary_markdown=f"Pipeline ColumnTransformer + {sim['cv_folds']}-Fold CV pada target `{target_col}`.",
            python_code=res["code"],
            metrics=res["metrics"],
            seed=sim["seed_used"],
        )
        st.download_button(
            "📒 Unduh Notebook (.ipynb)",
            data=ipynb_bytes,
            file_name="medium_pipeline_experiment.ipynb",
            mime="application/x-ipynb+json",
            use_container_width=True,
        )