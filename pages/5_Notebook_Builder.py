"""Halaman Modul 5: NOTEBOOK BUILDER - Generator Kerangka Jupyter Notebook (.ipynb) Bahasa Indonesia & Inggris."""

import streamlit as st
from core.export import build_structured_notebook_bytes
from core.utils import (
    get_global_seed,
    init_session_state,
    load_css,
    render_page_header,
    render_sidebar_controls,
)

st.set_page_config(page_title="Notebook Builder (.ipynb)", page_icon="📒", layout="wide")
init_session_state()
load_css()
render_sidebar_controls()

global_seed = get_global_seed()

render_page_header(
    title="GENERATOR KERANGKA NOTEBOOK (.IPYNB)",
    badge_text="📒 DOKUMENTASI STANDAR LOMBA",
    badge_class="badge-medium",
    est_time="5 Menit",
    subtitle="Buat kerangka Jupyter Notebook rapi (Pendahuluan, EDA, Preprocessing, Modeling, Evaluasi, Kesimpulan) dalam Bahasa Indonesia atau Inggris.",
)

st.markdown(
    """
    <div class="neo-card">
        <h3>🎯 Mengapa Dokumentasi Notebook Sangat Penting?</h3>
        <p>Dalam penilaian lomba ini, <strong>Kualitas Notebook menyumbang bobot 40% dari nilai akhir!</strong> Juri menilai kerapian narasi Markdown, kejelasan langkah EDA, pencegahan <em>data leakage</em>, pencatatan <em>random seed</em>, serta kesimpulan akhir.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("📖 Penjelasan Lengkap: Standar Struktur Notebook Juara Kaggle", expanded=False):
    st.markdown(
        """
        1. **Pendahuluan & Environment**: Menjelaskan masalah yang diselesaikan, metrik evaluasi yang dipakai, dan mengunci `GLOBAL_SEED`.
        2. **Exploratory Data Analysis (EDA)**: Menampilkan ringkasan tabel, visualisasi disertai kalimat **Insight** di bawah tiap grafik.
        3. **Preprocessing & Feature Engineering**: Menjelaskan alasan pemilihan metode imputasi dan fitur baru yang dibuat.
        4. **Modeling & Cross-Validation**: Menampilkan perbandingan model baseline vs model utama menggunakan *K-Fold CV*.
        5. **Evaluasi & Analisis Error**: Membahas *Confusion Matrix* / *Feature Importance* dan pembuatan `submission.csv`.
        6. **Kesimpulan**: Merangkum temuan terpenting dalam poin-poin singkat yang mudah dibaca juri.
        """
    )

with st.form("notebook_builder_form"):
    col1, col2 = st.columns(2)

    with col1:
        lang_choice = st.selectbox(
            "🌐 Bahasa Dokumentasi Notebook",
            options=["id", "en"],
            format_func=lambda x: "🇮🇩 Bahasa Indonesia" if x == "id" else "🇬🇧 English (International Standard)",
        )
        project_title = st.text_input(
            "📌 Judul Notebook Lomba",
            value="Eksperimen End-to-End Machine Learning & Analisis Fitur",
        )
        team_name = st.text_input("👥 Nama Tim Kaggle", value="Tim Simulate Machine Data")
        target_col = st.text_input("🎯 Nama Kolom Target di Dataset", value="target")

    with col2:
        model_choice = st.selectbox(
            "🤖 Algoritma Utama di Template Kode",
            options=["XGBoost Classifier / Regressor", "Random Forest Pipeline", "Stacking Ensemble"],
        )
        competition_goal = st.text_area(
            "📝 Deskripsi Singkat Tujuan Kompetisi",
            value="Memprediksi variabel target secara akurat dan stabil di Private Leaderboard menggunakan validasi silang 5-Fold.",
            height=100,
        )
        selected_sections = st.multiselect(
            "📑 Pilih Bagian Bab yang Ingin Dimasukkan",
            options=["Pendahuluan", "EDA", "Preprocessing", "Modeling", "Evaluasi", "Kesimpulan"],
            default=["Pendahuluan", "EDA", "Preprocessing", "Modeling", "Evaluasi", "Kesimpulan"],
        )

    generate_nb = st.form_submit_button("▶ Buat & Pratinjau Kerangka Notebook (.ipynb)", use_container_width=True)

if generate_nb or "nb_builder_state" not in st.session_state:
    if not selected_sections:
        st.warning("Pilih minimal satu bagian bab untuk dimasukkan ke dalam notebook.")
    else:
        ipynb_bytes, preview_cells = build_structured_notebook_bytes(
            lang=lang_choice,
            project_title=project_title,
            team_name=team_name,
            competition_goal=competition_goal,
            selected_sections=selected_sections,
            target_col=target_col,
            model_choice=model_choice,
            seed=global_seed,
        )
        st.session_state["nb_builder_state"] = {
            "ipynb_bytes": ipynb_bytes,
            "preview_cells": preview_cells,
            "lang": lang_choice,
            "title": project_title,
        }
        st.session_state["progress"]["legend_notebook"] = True

if "nb_builder_state" in st.session_state:
    nb_state = st.session_state["nb_builder_state"]

    st.markdown("---")
    dl_col1, dl_col2 = st.columns([2, 1])
    with dl_col1:
        st.success(
            f"✅ Kerangka Notebook **{nb_state['title']}** siap diunduh! "
            f"Total terdapat **{len(nb_state['preview_cells'])} sel** (Markdown + Kode Python)."
        )
    with dl_col2:
        st.download_button(
            label="⬇ Unduh File Template_Lomba.ipynb",
            data=nb_state["ipynb_bytes"],
            file_name="Template_Notebook_Kaggle.ipynb",
            mime="application/x-ipynb+json",
            use_container_width=True,
        )

    with st.expander("🔍 Pratinjau Isi Sel Notebook (Markdown & Kode)", expanded=True):
        for idx, cell in enumerate(nb_state["preview_cells"], start=1):
            if cell["type"] == "markdown":
                st.markdown(
                    f"<div class='neo-card'><small><b>Sel #{idx} [MARKDOWN]</b></small><hr style='margin:0.4rem 0;'>",
                    unsafe_allow_html=True,
                )
                st.markdown(cell["content"])
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.caption(f"💻 Sel #{idx} [PYTHON CODE]")
                st.code(cell["content"], language="python")