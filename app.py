"""Halaman Beranda SIMULATE MACHINE DATA."""

import streamlit as st
from core.utils import (
    GLOSSARY_AWAM,
    init_session_state,
    load_css,
    render_page_header,
    render_sidebar_controls,
)

st.set_page_config(
    page_title="SIMULATE MACHINE DATA — Latihan Kaggle Interaktif",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
load_css()
render_sidebar_controls()

render_page_header(
    title="SIMULATE MACHINE DATA",
    badge_text="🏠 BERANDA UTAMA",
    badge_class="badge-easy",
    est_time="Mulai Cepat 5 Menit",
    subtitle=(
        "Laboratorium latihan cepat EDA, Preprocessing, Modeling, Tuning, dan Manajemen "
        "Submission Kaggle untuk tim lomba — interaktif & tanpa login."
    ),
)

# Bagian 1: Ringkasan Metrik & Progress Tim
progress_dict = st.session_state["progress"]
completed_count = sum(1 for v in progress_dict.values() if v)
total_count = len(progress_dict)
progress_pct = int((completed_count / total_count) * 100) if total_count else 0

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("🎲 Global Seed Aktif", f"{st.session_state['global_seed']}")
with col_m2:
    st.metric("✅ Progress Kurikulum", f"{completed_count} / {total_count} Selesai ({progress_pct}%)")
with col_m3:
    today_sub_count = len(st.session_state.get("submissions_log", []))
    st.metric("📮 Total Catatan Submission", f"{today_sub_count} tercatat")

st.progress(progress_pct / 100.0)

# Bagian 2: Peta Jalan Belajar (Easy -> Legend)
st.markdown("## 🗺️ Peta Jalan Kurikulum Lomba (Easy → Legend)")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="neo-card">
            <span class="neo-badge badge-easy">🟢 LEVEL 1: EASY</span>
            <span class="neo-badge badge-time">⏱ 10 Menit</span>
            <h3>Kenali Datamu</h3>
            <p>Latihan membuat dataset sintetis, membaca statistik dasar (EDA), mengatasi nilai kosong sederhana, hingga melatih Linear & Logistic Regression.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Easy.py", label="👉 Buka Modul 1: Easy", icon="🟢")

    st.markdown(
        """
        <div class="neo-card" style="margin-top: 1rem;">
            <span class="neo-badge badge-hard">🟠 LEVEL 3: HARD</span>
            <span class="neo-badge badge-time">⏱ 20 Menit</span>
            <h3>Optimalkan Model</h3>
            <p>Menangani data timpang (SMOTE/Class Weight), Gradient Boosting (XGBoost), Hyperparameter Tuning (Optuna/RandomizedSearch), dan Ensembling.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/3_Hard.py", label="👉 Buka Modul 3: Hard", icon="🟠")

with col2:
    st.markdown(
        """
        <div class="neo-card">
            <span class="neo-badge badge-medium">🟡 LEVEL 2: MEDIUM</span>
            <span class="neo-badge badge-time">⏱ 15 Menit</span>
            <h3>Bersihkan & Bangun Fitur</h3>
            <p>Mendeteksi outlier (IQR/Z-Score), encoding kategori, scaling numerik, Feature Engineering, dan membangun Pipeline anti bocor data + K-Fold CV.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_Medium.py", label="👉 Buka Modul 2: Medium", icon="🟡")

    st.markdown(
        """
        <div class="neo-card" style="margin-top: 1rem;">
            <span class="neo-badge badge-legend">🔴 LEVEL 4: LEGEND</span>
            <span class="neo-badge badge-time">⏱ 25 Menit</span>
            <h3>Mode Lomba Kaggle</h3>
            <p>Simulasi Public vs Private Leaderboard (efek shake-up), Neural Network (MLP), demo Pretrained Model, dan ekspor artefak lomba lengkap.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/4_Legend.py", label="👉 Buka Modul 4: Legend", icon="🔴")

st.markdown("---")

# Bagian 3: Checklist Progress & Glosarium Panduan Awam
col_chk, col_info = st.columns([1, 1])

with col_chk:
    st.markdown("### ✅ Checklist Target Belajar Tim")
    st.caption("Centang modul yang sudah dipelajari tim Anda. Data dapat diekspor lewat menu di sidebar kiri.")
    labels_map = {
        "easy_eda": "🟢 Easy: Eksplorasi Data Dasar & Imputasi",
        "easy_model": "🟢 Easy: Train/Test Split & Regresi/Klasifikasi Dasar",
        "medium_prep": "🟡 Medium: Deteksi Outlier, Encoding & Feature Engineering",
        "medium_pipeline": "🟡 Medium: Scikit-Learn Pipeline & Cross-Validation",
        "hard_imbalance": "🟠 Hard: Penanganan Imbalanced Data & XGBoost",
        "hard_tuning": "🟠 Hard: Hyperparameter Tuning & Perbandingan Model",
        "legend_sim": "🔴 Legend: Simulasi Public vs Private Leaderboard",
        "legend_notebook": "🔴 Legend: Dokumentasi Notebook & Manajemen Kuota Submission",
    }
    for key, label in labels_map.items():
        st.session_state["progress"][key] = st.checkbox(
            label,
            value=bool(st.session_state["progress"].get(key, False)),
            key=f"chk_{key}",
        )

with col_info:
    st.markdown("### 🧰 Peralatan Lomba Tambahan")
    st.page_link("pages/5_Notebook_Builder.py", label="📒 Generator Kerangka Notebook (.ipynb)", icon="📒")
    st.page_link("pages/6_Submission_Manager.py", label="📮 Pencatat Kuota Submission (Maks 5/hari)", icon="📮")
    st.page_link("pages/7_Kalkulator_Skor.py", label="🧮 Simulator Skor Akhir (60% LB + 40% Notebook)", icon="🧮")

    with st.expander("📖 Penjelasan Lengkap: Glosarium & Istilah Penting (Panduan Awam)", expanded=st.session_state.get("panduan_awam", True)):
        for term, desc in GLOSSARY_AWAM.items():
            st.markdown(f"**🔹 {term}**  \n{desc}")