"""Halaman Modul 3: HARD - Optimalkan Model (Imbalanced Data, XGBoost, RFE, Tuning, Leaderboard & Ensembling)."""

import pandas as pd
import streamlit as st

from core.data_generator import generate_hard_dataset, load_uploaded_csv
from core.evaluation import (
    plot_confusion_matrix_chart,
    plot_feature_importance_chart,
    plot_leaderboard_bar_with_insight,
    plot_learning_curve_with_insight,
    plot_pr_curve_with_insight,
    plot_residuals_chart,
    plot_roc_curve_with_insight,
)
from core.export import leaderboard_to_csv_bytes, model_to_joblib_bytes, snippet_to_ipynb_bytes
from core.models import run_hard_optimization
from core.utils import (
    get_global_seed,
    init_session_state,
    load_css,
    render_page_header,
    render_sidebar_controls,
)

st.set_page_config(page_title="Hard — Optimalkan Model", page_icon="🟠", layout="wide")
init_session_state()
load_css()
render_sidebar_controls()

global_seed = get_global_seed()

# 1. HEADER
render_page_header(
    title="MODUL 3: OPTIMALKAN MODEL & ENSEMBLING",
    badge_text="🟠 LEVEL 3: HARD",
    badge_class="badge-hard",
    est_time="20 Menit",
    subtitle="Menangani Imbalanced Data (SMOTE), Seleksi Fitur (RFE), Gradient Boosting (XGBoost), Tuning (Optuna), dan Ensembling.",
)

# 2. TUJUAN BELAJAR
st.markdown(
    """
    <div class="neo-card">
        <h3>🎯 Tujuan Belajar Modul Hard</h3>
        <ul>
            <li>Menangani <strong>Dataset Tidak Seimbang (Imbalanced)</strong> menggunakan <strong>SMOTE</strong> dan <code>class_weight</code> agar kelas minoritas tidak diabaikan model.</li>
            <li>Menyaring fitur terbaik dengan <strong>RFE (Recursive Feature Elimination)</strong> dan mengoptimalkan <strong>XGBoost</strong> lewat <strong>Optuna / RandomizedSearch / GridSearch</strong>.</li>
            <li>Membandingkan banyak algoritma dalam satu <strong>Tabel Leaderboard</strong> serta menggabungkan model dengan <strong>Voting & Stacking Ensemble</strong>.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# 3. TEORI SINGKAT (Expander default tertutup)
with st.expander("📖 Penjelasan Lengkap: Imbalanced Data, XGBoost, Optuna Tuning & Ensembling", expanded=False):
    st.markdown(
        """
        ### 1. Bahaya Imbalanced Data & Jebakan Akurasi
        Bayangkan dataset deteksi kecurangan (*fraud*) yang berisi **95% transaksi normal** dan hanya **5% transaksi fraud**.
        Jika model malas menebak "Semua Normal", akurasinya **95%** tetapi ia gagal menangkap satu pun pelaku fraud (`Recall = 0%`)!
        - **SMOTE (*Synthetic Minority Over-sampling Technique*)**: Membuat titik data sintetis baru pada kelas minoritas di data latihan.
        - **Class Weight / `scale_pos_weight`**: Memberi hukuman (*penalti*) berkali-kali lipat jika model salah menebak kelas minoritas.

        ### 2. Seleksi Fitur (RFE vs Feature Importance)
        Terlalu banyak kolom sampah (*noise*) membuat model lambat dan mudah *overfitting*. **RFE** secara bertahap memangkas fitur paling lemah hingga tersisa fitur inti terbaik.

        ### 3. Hyperparameter Tuning & Ensembling (Senjata Utama Juara Kaggle)
        - **Optuna**: Mencari kombinasi parameter terbaik secara cerdas (*Bayesian Optimization*) sehingga jauh lebih cepat daripada mencoba satu per satu (*GridSearch*).
        - **Voting & Stacking Ensemble**: Menggabungkan prediksi dari beberapa model berbeda (misal *Random Forest + Extra Trees + XGBoost*) agar kesalahan satu model ditutupi oleh model lainnya.
        """
    )

st.markdown("### 🎛️ 4. Panel Kontrol Optimasi & Leaderboard")
uploaded_csv = st.file_uploader(
    "📂 Opsional: Upload file CSV sendiri (Maks 10 MB) — kosongkan untuk memakai Dataset Imbalanced Sintetis",
    type=["csv"],
    key="hard_csv_upload",
)

# 4 & 5. PANEL KONTROL DALAM st.form + TOMBOL JALANKAN SIMULASI
with st.form("hard_simulation_form"):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**A. Dataset & Ketimpangan**")
        task_choice = st.selectbox(
            "Tipe Kompetisi",
            ["classification", "regression"],
            format_func=lambda x: "Klasifikasi Imbalanced (Fraud)" if x == "classification" else "Regresi Kompleks",
            disabled=(uploaded_csv is not None),
        )
        n_samples = st.slider("Jumlah Baris Data", 400, 10000, 1200, step=200)
        n_features = st.slider("Total Fitur Awal (Inti + Noise)", 6, 20, 12)
        minority_ratio = st.slider(
            "Rasio Kelas Minoritas (Klasifikasi)",
            min_value=0.05,
            max_value=0.50,
            value=0.15,
            step=0.05,
            help="0.15 berarti 15% kelas positif (minoritas) dan 85% kelas negatif (mayoritas).",
        )

    with c2:
        st.markdown("**B. Imbalance & Seleksi Fitur**")
        imbalance_method = st.selectbox(
            "Penanganan Imbalanced Data",
            ["smote", "class_weight", "none"],
            format_func=lambda x: {
                "smote": "SMOTE (Oversampling Sintetis)",
                "class_weight": "Class Weight / Scale Pos Weight",
                "none": "Tanpa Perlakuan (Raw)",
            }[x],
        )
        feature_selection_method = st.selectbox(
            "Metode Seleksi Fitur",
            ["rfe", "importance", "all"],
            format_func=lambda x: {
                "rfe": "RFE (Recursive Feature Elimination)",
                "importance": "Top Feature Importance",
                "all": "Gunakan Semua Fitur",
            }[x],
        )
        n_selected_features = st.slider("Jumlah Fitur Dipertahankan", 3, 18, 8)

    with c3:
        st.markdown("**C. Hyperparameter Tuning**")
        tuning_method = st.selectbox(
            "Engine Tuning XGBoost",
            ["optuna", "random", "grid"],
            format_func=lambda x: {
                "optuna": "Optuna (Bayesian TPE - Cepat & Cerdas)",
                "random": "RandomizedSearchCV",
                "grid": "GridSearchCV",
            }[x],
        )
        n_trials = st.slider(
            "Jumlah Trial Tuning (Maks 30)",
            min_value=4,
            max_value=30,
            value=10,
            step=2,
            help="Dibatasi <= 30 trial agar simulasi selesai < 20 detik.",
        )

    with c4:
        st.markdown("**D. Validasi & Ensembling**")
        cv_folds = st.slider("Lipatan Stratified K-Fold", 3, 5, 4)
        ensemble_type = st.selectbox(
            "Metode Ensembling",
            ["voting", "stacking"],
            format_func=lambda x: "Voting Ensemble (RF + ET + XGB)" if x == "voting" else "Stacking Ensemble (Meta-Learner)",
        )
        noise_val = st.slider("Tingkat Noise Data", 1.0, 35.0, 10.0)
        st.caption(f"🎲 Seed Aktif: **{global_seed}**")

    run_hard = st.form_submit_button("▶ Jalankan Optimasi & Leaderboard Hard", use_container_width=True)

if run_hard or "hard_sim_state" not in st.session_state:
    try:
        with st.spinner("Menjalankan seleksi fitur, SMOTE, tuning XGBoost, dan membandingkan seluruh model..."):
            if uploaded_csv is not None:
                df_raw = load_uploaded_csv(uploaded_csv)
                target_col = df_raw.columns[-1]
                task_type = "classification" if df_raw[target_col].nunique(dropna=True) <= 15 else "regression"
            else:
                df_raw, target_col, task_type = generate_hard_dataset(
                    task_type=task_choice,
                    n_samples=n_samples,
                    n_features=n_features,
                    minority_ratio=minority_ratio,
                    noise=noise_val,
                    seed=global_seed,
                )

            hard_res = run_hard_optimization(
                df=df_raw,
                target_col=target_col,
                task_type=task_type,
                imbalance_method=imbalance_method,
                feature_selection_method=feature_selection_method,
                n_selected_features=n_selected_features,
                tuning_method=tuning_method,
                n_trials=n_trials,
                cv_folds=cv_folds,
                ensemble_type=ensemble_type,
                seed=global_seed,
            )

            st.session_state["hard_sim_state"] = {
                "df_raw": df_raw,
                "target_col": target_col,
                "task_type": task_type,
                "imbalance_method": imbalance_method,
                "tuning_method": tuning_method,
                "res": hard_res,
                "seed_used": global_seed,
            }
            st.session_state["progress"]["hard_imbalance"] = True
            st.session_state["progress"]["hard_tuning"] = True
    except Exception as exc:
        st.error(f"Terjadi kendala saat menjalankan optimasi Hard: {exc}")

# 6. HASIL SIMULASI
if "hard_sim_state" in st.session_state:
    sim = st.session_state["hard_sim_state"]
    res = sim["res"]
    lb_df: pd.DataFrame = res["leaderboard_df"]
    top_row = lb_df.iloc[0]

    st.markdown("---")
    st.markdown("## 🏆 6. Hasil Leaderboard Model, Tuning & Kurva Diagnostik")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("🥇 Juara Leaderboard", f"{res['best_model_name']}")
    mc2.metric(f"🎯 Skor Terbaik ({res['sort_col']})", f"{top_row[res['sort_col']]}")
    mc3.metric("🧬 Fitur Terpilih", f"{len(res['selected_cols'])} dari {len(res['all_cols'])} fitur")
    mc4.metric("⚡ Waktu Latih Juara", f"{top_row['Waktu_Detik']} detik")

    tab_lb, tab_curves, tab_tune = st.tabs(
        [
            "🏆 1. Tabel Leaderboard & Perbandingan Model",
            "📈 2. Kurva ROC, Precision-Recall & Learning Curve",
            "🔧 3. Detail Hyperparameter Tuning & Seleksi Fitur",
        ]
    )

    with tab_lb:
        st.markdown("#### Peringkat Perbandingan Model (Diurutkan dari Skor Ujian Tertinggi)")
        st.dataframe(lb_df, use_container_width=True, hide_index=True)

        fig_lb, insight_lb = plot_leaderboard_bar_with_insight(lb_df, res["sort_col"])
        st.plotly_chart(fig_lb, use_container_width=True)
        st.info(insight_lb)

        if sim["task_type"] == "classification":
            st.markdown(
                f"**Distribusi Kelas Data Latihan Sebelum `{sim['imbalance_method'].upper()}`:** `{res['class_counts_before']}`  \n"
                f"**Distribusi Kelas Data Latihan Sesudah `{sim['imbalance_method'].upper()}`:** `{res['class_counts_after']}`"
            )

    with tab_curves:
        if sim["task_type"] == "classification" and res["best_y_prob"] is not None:
            col_roc, col_pr = st.columns(2)
            with col_roc:
                fig_roc, insight_roc = plot_roc_curve_with_insight(res["y_test"], res["best_y_prob"])
                st.plotly_chart(fig_roc, use_container_width=True)
                st.info(insight_roc)
            with col_pr:
                fig_pr, insight_pr = plot_pr_curve_with_insight(res["y_test"], res["best_y_prob"])
                st.plotly_chart(fig_pr, use_container_width=True)
                st.info(insight_pr)

        col_lc, col_cm = st.columns(2)
        with col_lc:
            fig_lc, insight_lc = plot_learning_curve_with_insight(
                res["lc_sizes"],
                res["lc_train_mean"],
                res["lc_val_mean"],
                metric_label=res["scoring_name"],
            )
            st.plotly_chart(fig_lc, use_container_width=True)
            st.info(insight_lc)

        with col_cm:
            if sim["task_type"] == "classification":
                fig_eval, insight_eval = plot_confusion_matrix_chart(res["y_test"], res["best_y_pred"])
            else:
                fig_eval, insight_eval = plot_residuals_chart(
                    res["y_test"], res["best_y_pred"], seed=sim["seed_used"]
                )
            st.plotly_chart(fig_eval, use_container_width=True)
            st.info(insight_eval)

    with tab_tune:
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.markdown(f"#### Parameter Terbaik Hasil `{sim['tuning_method'].upper()}`")
            st.json(res["best_params"])
            st.markdown("#### Riwayat Percobaan Trial Tuning")
            st.dataframe(res["tuning_history_df"], use_container_width=True, hide_index=True)

        with tcol2:
            st.markdown(f"#### Daftar Fitur Terpilih ({len(res['selected_cols'])} kolom)")
            st.success(", ".join([f"`{c}`" for c in res["selected_cols"]]))
            fig_fi, insight_fi = plot_feature_importance_chart(res["selected_cols"], res["importances"])
            st.plotly_chart(fig_fi, use_container_width=True)
            st.info(insight_fi)

    # 7. PENJELASAN LANGKAH DEMI LANGKAH (Expander)
    with st.expander("🔍 7. Penjelasan Langkah demi Langkah (Bagaimana Optimasi Ini Bekerja?)", expanded=False):
        st.markdown(
            f"""
            1. **Seleksi Fitur**: Dari **{len(res['all_cols'])} fitur awal**, algoritma memilih **{len(res['selected_cols'])} fitur paling informatif** sehingga fitur *noise* terbuang.
            2. **Penanganan Imbalance (`{sim['imbalance_method'].upper()}`)**: Hanya diterapkan pada **data latihan (`X_train`)**, sedangkan data ujian (`X_test`) dibiarkan murni sesuai proporsi asli agar evaluasi tidak curang.
            3. **Hyperparameter Tuning (`{sim['tuning_method'].upper()}`)**: Menjalankan pencarian parameter terbaik `XGBoost` dengan validasi silang dan menemukan kombinasi `{res['best_params']}`.
            4. **Ensembling & Leaderboard**: Seluruh model tunggal dibandingkan langsung dengan model gabungan (**Voting / Stacking**), menghasilkan juara **`{res['best_model_name']}`**.
            """
        )

    # 8. KODE YANG SETARA
    st.markdown("### 💻 8. Kode Python Setara (SMOTE + Tuning XGBoost + Seleksi Fitur)")
    st.code(res["code"], language="python")

    # 9. LATIHAN MANDIRI
    st.markdown(
        """
        <div class="neo-card">
            <h3>✍ 9. Tantangan Latihan Mandiri</h3>
            <ol>
                <li>Atur <strong>Rasio Kelas Minoritas = 0.05 (5%)</strong>, lalu bandingkan skor <code>Test_F1</code> dan <code>Test_Recall</code> saat <em>Penanganan Imbalanced = Tanpa Perlakuan</em> vs saat menggunakan <strong>SMOTE</strong>!</li>
                <li>Ganti <strong>Metode Seleksi Fitur</strong> menjadi <em>RFE</em> dengan 6 fitur, lalu amati apakah fitur bernama <code>fitur_noise_...</code> berhasil disingkirkan secara otomatis.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 10. EKSPOR HASIL
    st.markdown("### ⬇ 10. Ekspor Artefak Leaderboard & Model Juara")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.download_button(
            "📊 Unduh Tabel Leaderboard (.csv)",
            data=leaderboard_to_csv_bytes(lb_df, sim["seed_used"]),
            file_name="hard_leaderboard_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ex2:
        st.download_button(
            "🤖 Unduh Model Juara (.joblib)",
            data=model_to_joblib_bytes(res["best_model_obj"]),
            file_name="hard_champion_model.joblib",
            mime="application/octet-stream",
            use_container_width=True,
        )
    with ex3:
        top_metrics = top_row.to_dict()
        ipynb_bytes = snippet_to_ipynb_bytes(
            title=f"Modul Hard - Juara: {res['best_model_name']}",
            summary_markdown=f"Optimasi model menggunakan `{sim['imbalance_method']}` dan tuning `{sim['tuning_method']}`.",
            python_code=res["code"],
            metrics=top_metrics,
            seed=sim["seed_used"],
        )
        st.download_button(
            "📒 Unduh Notebook Eksperimen (.ipynb)",
            data=ipynb_bytes,
            file_name="hard_optimization.ipynb",
            mime="application/x-ipynb+json",
            use_container_width=True,
        )