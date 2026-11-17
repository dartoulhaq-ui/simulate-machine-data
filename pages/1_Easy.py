"""Halaman Modul 1: EASY - Kenali Datamu (EDA Dasar, Missing Value & Model Linear/Logistik)."""

import pandas as pd
import streamlit as st

from core.data_generator import generate_easy_dataset, load_uploaded_csv, summarize_dataset
from core.evaluation import (
    plot_confusion_matrix_chart,
    plot_correlation_heatmap,
    plot_feature_importance_chart,
    plot_histogram_with_insight,
    plot_residuals_chart,
    plot_scatter_with_insight,
)
from core.export import metrics_to_csv_bytes, model_to_joblib_bytes, snippet_to_ipynb_bytes
from core.models import train_easy_model
from core.preprocessing import impute_dataframe
from core.utils import (
    get_environment_info,
    get_global_seed,
    get_requirements_text,
    init_session_state,
    load_css,
    render_page_header,
    render_sidebar_controls,
)

st.set_page_config(page_title="Easy — Kenali Datamu", page_icon="🟢", layout="wide")
init_session_state()
load_css()
render_sidebar_controls()

global_seed = get_global_seed()

# 1. HEADER
render_page_header(
    title="MODUL 1: KENALI DATAMU (EDA & MODEL DASAR)",
    badge_text="🟢 LEVEL 1: EASY",
    badge_class="badge-easy",
    est_time="10 Menit",
    subtitle="Memahami kondisi awal dataset, menangani nilai kosong (missing value), dan melatih model dasar.",
)

# 2. TUJUAN BELAJAR
st.markdown(
    """
    <div class="neo-card">
        <h3>🎯 Tujuan Belajar Modul Easy</h3>
        <ul>
            <li>Membaca anatomi dataset baru menggunakan <code>shape</code>, <code>dtypes</code>, <code>describe()</code>, dan persentase <em>missing value</em>.</li>
            <li>Membandingkan dampak pengisian nilai kosong (<strong>Mean vs Median vs Modus vs Drop Baris</strong>) sebelum dan sesudah imputasi.</li>
            <li>Melatih model baseline <strong>Linear Regression</strong> (untuk angka kontinu) dan <strong>Logistic Regression</strong> (untuk klasifikasi 2 kelas) secara <em>reproducible</em>.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# 3. TEORI SINGKAT (Expander default tertutup)
with st.expander("📖 Penjelasan Lengkap: Teori EDA, Imputasi & Baseline Model", expanded=False):
    st.markdown(
        """
        ### 1. Mengapa Harus EDA Terlebih Dahulu?
        Kesalahan terbesar tim pemula di lomba Kaggle adalah langsung memasukkan data ke model tanpa mengecek isinya.
        Lewat **EDA (Exploratory Data Analysis)**, kita memeriksa:
        - Apakah ada kolom yang sebagian besar isinya kosong (*missing value*)?
        - Apakah persebaran angkanya normal atau miring (*skewed*)?

        ### 2. Kapan Memakai Mean vs Median untuk Missing Value?
        - **Mean (Rata-rata)**: Cocok jika distribusi data simetris dan tidak punya nilai ekstrem (*outlier*).
        - **Median (Nilai Tengah)**: Jauh lebih aman jika data memiliki *outlier* (misalnya data gaji atau harga rumah).
        - **Drop Baris**: Hanya boleh dipakai jika jumlah baris yang kosong sangat sedikit (< 3%).

        ### 3. Mengapa Mulai dari Linear / Logistic Regression?
        Dalam lomba Kaggle, kita wajib membuat **Baseline Model** yang cepat (< 2 detik) sebagai patokan skor awal
        sebelum beralih ke model kompleks seperti XGBoost.
        """
    )

# Pilihan Sumber Data (Sintetis / Bawaan / Upload CSV)
st.markdown("### 🎛️ 4. Panel Kontrol Simulasi")
uploaded_csv = st.file_uploader(
    "📂 Opsional: Upload file CSV sendiri (Maks 10 MB) — kosongkan jika ingin memakai Generator Dataset Sintetis",
    type=["csv"],
)

# 4 & 5. PANEL KONTROL DALAM st.form + TOMBOL JALANKAN SIMULASI
with st.form("easy_simulation_form"):
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

    with col_cfg1:
        st.markdown("**A. Pengaturan Dataset**")
        dataset_source = st.selectbox(
            "Pilih Dataset",
            options=[
                "Sintetis: Regresi Linear",
                "Sintetis: Klasifikasi 2 Kelas",
                "Dataset Bawaan: Iris (Klasifikasi)",
                "Dataset Bawaan: Diabetes (Regresi)",
            ],
            disabled=(uploaded_csv is not None),
        )
        n_samples = st.slider(
            "Jumlah Baris Data",
            min_value=200,
            max_value=20000,
            value=800,
            step=200,
            help="Default <= 5.000 baris agar tetap instan.",
        )
        n_features = st.slider("Jumlah Fitur Numerik", min_value=3, max_value=12, value=5)

    with col_cfg2:
        st.markdown("**B. Tingkat Tantangan Data**")
        noise_level = st.slider(
            "Tingkat Noise / Gangguan Acak",
            min_value=1.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
        )
        missing_pct = st.slider(
            "Persentase Missing Value (%)",
            min_value=0.0,
            max_value=30.0,
            value=6.0,
            step=1.0,
        )
        impute_strategy = st.selectbox(
            "Metode Penanganan Missing Value",
            options=["median", "mean", "most_frequent", "drop"],
            format_func=lambda x: {
                "median": "Imputasi Median (Tahan Outlier)",
                "mean": "Imputasi Mean (Rata-rata)",
                "most_frequent": "Imputasi Modus (Nilai Terbanyak)",
                "drop": "Drop / Hapus Baris yang Kosong",
            }[x],
        )

    with col_cfg3:
        st.markdown("**C. Split & Hyperparameter Model**")
        test_size = st.slider(
            "Proporsi Data Uji (Test Split)",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
        )
        c_param = st.select_slider(
            "Kekuatan Regularisasi C (Khusus Klasifikasi)",
            options=[0.01, 0.1, 1.0, 5.0, 10.0],
            value=1.0,
        )
        st.caption(f"🎲 Menggunakan **GLOBAL_SEED = {global_seed}** (ubah di sidebar kiri).")

    run_clicked = st.form_submit_button("▶ Jalankan Simulasi Easy", use_container_width=True)

# Eksekusi atau perbarui state saat tombol ditekan (atau saat pertama kali halaman dibuka)
if run_clicked or "easy_sim_state" not in st.session_state:
    try:
        with st.spinner("Sedang membuat dataset, menjalankan EDA, dan melatih model baseline..."):
            if uploaded_csv is not None:
                df_raw = load_uploaded_csv(uploaded_csv)
                target_col = df_raw.columns[-1]
                nunique_target = df_raw[target_col].nunique(dropna=True)
                task_type = "classification" if nunique_target <= 15 else "regression"
            else:
                df_raw, target_col, task_type = generate_easy_dataset(
                    source_name=dataset_source,
                    n_samples=n_samples,
                    n_features=n_features,
                    noise=noise_level,
                    missing_pct=missing_pct,
                    seed=global_seed,
                )

            eda_summary = summarize_dataset(df_raw)
            df_imputed, impute_compare_df = impute_dataframe(
                df_raw, target_col=target_col, strategy=impute_strategy
            )
            model_res = train_easy_model(
                df=df_raw,
                target_col=target_col,
                task_type=task_type,
                impute_strategy=impute_strategy,
                test_size=test_size,
                c_param=float(c_param),
                seed=global_seed,
            )

            st.session_state["easy_sim_state"] = {
                "df_raw": df_raw,
                "df_imputed": df_imputed,
                "target_col": target_col,
                "task_type": task_type,
                "eda_summary": eda_summary,
                "impute_compare_df": impute_compare_df,
                "impute_strategy": impute_strategy,
                "model_res": model_res,
                "seed_used": global_seed,
            }
            st.session_state["progress"]["easy_eda"] = True
            st.session_state["progress"]["easy_model"] = True
    except Exception as exc:
        st.error(f"Terjadi kendala saat memproses dataset: {exc}. Pastikan data memiliki kolom numerik yang valid.")

# 6. HASIL SIMULASI
if "easy_sim_state" in st.session_state:
    sim = st.session_state["easy_sim_state"]
    df_raw: pd.DataFrame = sim["df_raw"]
    df_imputed: pd.DataFrame = sim["df_imputed"]
    target_col: str = sim["target_col"]
    task_type: str = sim["task_type"]
    eda = sim["eda_summary"]
    model_res = sim["model_res"]

    st.markdown("---")
    st.markdown("## 📊 6. Hasil Eksplorasi Data (EDA) & Evaluasi Model")

    # Baris Metrik Ringkasan Data
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📏 Ukuran Data (Shape)", f"{eda['n_rows']} baris × {eda['n_cols']} kolom")
    c2.metric("❓ Total Sel Kosong", f"{eda['total_missing']} sel ({eda['total_missing_pct']}%)")
    c3.metric("🎯 Kolom Target", f"{target_col} ({task_type.upper()})")
    c4.metric("🧹 Baris Setelah Imputasi", f"{len(df_imputed)} baris")

    tab_eda, tab_viz, tab_model = st.tabs(
        ["📋 1. Ringkasan Data & Imputasi", "📈 2. Visualisasi EDA + Insight", "🤖 3. Performa Model Baseline"]
    )

    with tab_eda:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Contoh 5 Baris Pertama (`df.head()`) — Sebelum Imputasi")
            st.dataframe(df_raw.head(8), use_container_width=True)
            st.markdown("#### Tipe Data & Persentase Missing per Kolom")
            st.dataframe(eda["info_df"], use_container_width=True, hide_index=True)

        with col_right:
            st.markdown("#### Contoh 5 Baris Pertama — Sesudah Imputasi")
            st.dataframe(df_imputed.head(8), use_container_width=True)
            st.markdown("#### Perbandingan Missing Value (Sebelum vs Sesudah)")
            st.dataframe(sim["impute_compare_df"], use_container_width=True, hide_index=True)

        st.markdown("#### Statistik Deskriptif (`df.describe()`)")
        st.dataframe(eda["describe_num"], use_container_width=True)

    with tab_viz:
        num_cols = [c for c in df_imputed.columns if pd.api.types.is_numeric_dtype(df_imputed[c])]
        if len(num_cols) >= 2:
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                selected_hist_col = st.selectbox("Pilih Kolom untuk Histogram:", options=num_cols, index=0)
                fig_hist, insight_hist = plot_histogram_with_insight(df_imputed, selected_hist_col)
                st.plotly_chart(fig_hist, use_container_width=True)
                st.info(insight_hist)

            with vcol2:
                x_scatter = st.selectbox("Sumbu X Scatter Plot:", options=num_cols, index=0)
                y_scatter = st.selectbox(
                    "Sumbu Y Scatter Plot:",
                    options=num_cols,
                    index=min(1, len(num_cols) - 1),
                )
                color_arg = target_col if task_type == "classification" else None
                fig_scat, insight_scat = plot_scatter_with_insight(
                    df_imputed, x_scatter, y_scatter, color_col=color_arg, seed=sim["seed_used"]
                )
                st.plotly_chart(fig_scat, use_container_width=True)
                st.info(insight_scat)

            fig_corr, insight_corr = plot_correlation_heatmap(df_imputed)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.info(insight_corr)

    with tab_model:
        st.markdown(f"#### Hasil Pelatihan: **{model_res['model_name']}** (`GLOBAL_SEED = {sim['seed_used']}`)")
        m_cols = st.columns(len(model_res["metrics"]))
        for idx, (m_name, m_val) in enumerate(model_res["metrics"].items()):
            m_cols[idx].metric(m_name, f"{m_val}")

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            if task_type == "classification":
                fig_eval, insight_eval = plot_confusion_matrix_chart(model_res["y_test"], model_res["y_pred"])
            else:
                fig_eval, insight_eval = plot_residuals_chart(
                    model_res["y_test"], model_res["y_pred"], seed=sim["seed_used"]
                )
            st.plotly_chart(fig_eval, use_container_width=True)
            st.info(insight_eval)

        with g_col2:
            fig_imp, insight_imp = plot_feature_importance_chart(
                model_res["feature_names"], model_res["importances"]
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.info(insight_imp)

    # 7. PENJELASAN LANGKAH DEMI LANGKAH (Expander)
    with st.expander("🔍 7. Penjelasan Langkah demi Langkah (Apa yang Baru Saja Terjadi?)", expanded=False):
        st.markdown(
            f"""
            1. **Pembuatan / Pemuatan Data (`seed={sim['seed_used']}`)**: Sistem menyiapkan **{eda['n_rows']} baris** data dengan **{eda['total_missing']} sel kosong** ({eda['total_missing_pct']}%).
            2. **Penanganan Missing Value (`{sim['impute_strategy']}`)**: Seluruh nilai kosong ditangani sehingga tersisa **0 sel kosong** pada data bersih.
            3. **Pembagian Train/Test Split**: Data dibagi menjadi **{model_res['n_train']} baris latihan (train)** dan **{model_res['n_test']} baris ujian (test)**.
            4. **Pelatihan Baseline (`{model_res['model_name']}`)**: Model dilatih di dalam `Pipeline` (Imputer + `StandardScaler` + Model) agar nilai rata-rata dari data uji (*test set*) tidak bocor ke data latihan.
            """
        )

    # 8. KODE YANG SETARA
    st.markdown("### 💻 8. Kode Python Setara (Siap Salin ke Notebook Kaggle)")
    st.code(model_res["code"], language="python")

    # 9. LATIHAN MANDIRI
    st.markdown(
        """
        <div class="neo-card">
            <h3>✍ 9. Tantangan Latihan Mandiri</h3>
            <ol>
                <li>Ubah <strong>Persentase Missing Value</strong> ke <strong>25%</strong>, lalu bandingkan skor model antara metode <em>Drop Baris</em> vs <em>Imputasi Median</em>. Mengapa menghapus baris menurunkan keandalan data?</li>
                <li>Pada dataset <strong>Klasifikasi 2 Kelas</strong>, naikkan <strong>Tingkat Noise</strong> dari 10 ke 40, lalu amati perubahan angka di luar diagonal pada <em>Confusion Matrix</em>.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 10. EKSPOR HASIL & REPRODUKSIBILITAS
    st.markdown("### ⬇ 10. Ekspor Artefak & Reproduksibilitas")
    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)

    with exp_col1:
        st.download_button(
            "📄 Unduh Metrik (.csv)",
            data=metrics_to_csv_bytes(model_res["metrics"], model_res["model_name"], sim["seed_used"]),
            file_name="easy_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp_col2:
        st.download_button(
            "🤖 Unduh Model (.joblib)",
            data=model_to_joblib_bytes(model_res["pipeline"]),
            file_name="easy_baseline_model.joblib",
            mime="application/octet-stream",
            use_container_width=True,
        )
    with exp_col3:
        ipynb_bytes = snippet_to_ipynb_bytes(
            title=f"Modul Easy - {model_res['model_name']}",
            summary_markdown=f"Eksperimen baseline pada dataset `{target_col}` menggunakan imputasi `{sim['impute_strategy']}`.",
            python_code=model_res["code"],
            metrics=model_res["metrics"],
            seed=sim["seed_used"],
        )
        st.download_button(
            "📒 Unduh Snippet (.ipynb)",
            data=ipynb_bytes,
            file_name="easy_experiment.ipynb",
            mime="application/x-ipynb+json",
            use_container_width=True,
        )
    with exp_col4:
        st.download_button(
            "📦 Unduh requirements.txt",
            data=get_requirements_text().encode("utf-8"),
            file_name="requirements.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with st.expander("🧪 Lihat Detail Versi Environment (Bukti Reproduksibilitas)"):
        st.json(get_environment_info())