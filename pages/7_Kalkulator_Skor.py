"""Halaman Modul 7: KALKULATOR SKOR & CHECKLIST KUALITAS NOTEBOOK (Bobot 60% Kaggle + 40% Notebook)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.utils import (
    init_session_state,
    load_css,
    render_page_header,
    render_sidebar_controls,
)

st.set_page_config(page_title="Kalkulator Skor & Checklist Notebook", page_icon="🧮", layout="wide")
init_session_state()
load_css()
render_sidebar_controls()

render_page_header(
    title="KALKULATOR SKOR AKHIR & CHECKLIST NOTEBOOK",
    badge_text="🧮 RUMUS RESMI: 60% KAGGLE + 40% NOTEBOOK",
    badge_class="badge-legend",
    est_time="5 Menit",
    subtitle="Hitung proyeksi nilai akhir tim Anda (15% Public LB + 45% Private LB + 40% Kualitas Notebook) dan periksa kelengkapan notebook.",
)

st.markdown(
    """
    <div class="neo-card">
        <h3>📐 Rumus Penilaian Resmi Kompetisi</h3>
        <p>Nilai akhir tim tidak hanya ditentukan oleh skor <em>Public Leaderboard</em>, melainkan gabungan dari:</p>
        <ul>
            <li><strong>60% Skor Kaggle Leaderboard</strong> — terbagi menjadi:
                <ul>
                    <li><strong>15% Skor Public Leaderboard</strong> (data uji awal yang terlihat selama lomba).</li>
                    <li><strong>45% Skor Private Leaderboard</strong> (data uji akhir untuk menguji ketahanan model terhadap <em>overfitting</em>).</li>
                </ul>
            </li>
            <li><strong>40% Skor Kualitas & Dokumentasi Notebook (<code>.ipynb</code>)</strong> — dinilai dari kelengkapan EDA, kerapian Markdown, reproduksibilitas (seed & versi library), visualisasi ber-insight, dan kesimpulan.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

col_chk, col_calc = st.columns([1, 1])

with col_chk:
    st.markdown("### ✅ 1. Checklist Kualitas Notebook Tim (Bobot 40%)")
    st.caption("Centang kriteria yang sudah dipenuhi di dalam file `.ipynb` tim Anda:")

    c_eda = st.checkbox(
        "1. EDA Lengkap (shape, dtypes, describe, analisis missing value & outlier)",
        value=True,
    )
    c_viz = st.checkbox(
        "2. Visualisasi Informatif (disertai kalimat penjelasan 'Insight' di bawah tiap grafik)",
        value=True,
    )
    c_md = st.checkbox(
        "3. Dokumentasi Markdown Rapi (memiliki bab Pendahuluan, Preprocessing, Modeling, Evaluasi)",
        value=True,
    )
    c_pipe = st.checkbox(
        "4. Bebas Data Leakage (menggunakan Pipeline / ColumnTransformer & K-Fold CV)",
        value=True,
    )
    c_seed = st.checkbox(
        "5. Reproduksibilitas Penuh (GLOBAL_SEED dikunci & versi Python/library ditampilkan)",
        value=True,
    )
    c_conc = st.checkbox(
        "6. Kesimpulan & Alasan Pemilihan Model Akhir tertulis jelas",
        value=True,
    )

    checklist_items = [c_eda, c_viz, c_md, c_pipe, c_seed, c_conc]
    auto_nb_score = round((sum(1 for x in checklist_items if x) / len(checklist_items)) * 100.0, 1)
    st.info(f"📋 **Skor Kesiapan Notebook Berdasarkan Checklist:** **`{auto_nb_score} / 100`**")

with col_calc:
    st.markdown("### 🧮 2. Simulator Perhitungan Nilai Akhir (Skala 0 – 100)")

    pub_score_input = st.slider(
        "📢 Skor Public Leaderboard Tim (Skala 0 - 100)",
        min_value=0.0,
        max_value=100.0,
        value=84.0,
        step=0.5,
    )
    priv_score_input = st.slider(
        "🔒 Estimasi Skor Private Leaderboard Tim (Skala 0 - 100)",
        min_value=0.0,
        max_value=100.0,
        value=83.0,
        step=0.5,
    )
    use_checklist_score = st.toggle("Gunakan Skor dari Checklist Notebook di Samping", value=True)
    if use_checklist_score:
        nb_score_input = auto_nb_score
        st.caption(f"📒 Skor Kualitas Notebook: **{nb_score_input}** (otomatis dari checklist)")
    else:
        nb_score_input = st.slider(
            "📒 Skor Kualitas Notebook Manual (Skala 0 - 100)",
            min_value=0.0,
            max_value=100.0,
            value=88.0,
            step=0.5,
        )

# Perhitungan Bobot: 15% Public + 45% Private + 40% Notebook
contrib_pub = round(0.15 * pub_score_input, 2)
contrib_priv = round(0.45 * priv_score_input, 2)
contrib_kaggle_total = round(contrib_pub + contrib_priv, 2)
contrib_nb = round(0.40 * nb_score_input, 2)
final_total_score = round(contrib_kaggle_total + contrib_nb, 2)

st.markdown("---")
st.markdown("## 🏆 Hasil Perhitungan Skor Akhir Tim Anda")

m1, m2, m3, m4 = st.columns(4)
m1.metric("📢 Kontribusi Public LB (15%)", f"{contrib_pub:.2f} poin", f"Dari skor {pub_score_input}")
m2.metric("🔒 Kontribusi Private LB (45%)", f"{contrib_priv:.2f} poin", f"Dari skor {priv_score_input}")
m3.metric("📒 Kontribusi Notebook (40%)", f"{contrib_nb:.2f} poin", f"Dari skor {nb_score_input}")
m4.metric("🌟 TOTAL SKOR AKHIR", f"{final_total_score:.2f} / 100")

st.markdown("---")
st.markdown("### ⚖️ Simulasi Mengapa Tim yang Overfit ke Public LB Bisa Kalah")
st.caption(
    "Tabel di bawah membandingkan **Tim Anda** dengan tim lawan yang hanya mengejar skor Public LB (overfitting) "
    "tetapi mengabaikan dokumentasi Notebook dan kestabilan Private LB:"
)

sim_compare_df = pd.DataFrame(
    [
        {
            "Tim": "⭐ Tim Anda (Validasi CV Stabil + Notebook Rapi)",
            "Public LB (15%)": pub_score_input,
            "Private LB (45%)": priv_score_input,
            "Notebook (40%)": nb_score_input,
            "Nilai Akhir Total": final_total_score,
        },
        {
            "Tim": "Tim Pemburu Public LB (Overfit Public + Notebook Kosong)",
            "Public LB (15%)": 94.0,
            "Private LB (45%)": 71.0,
            "Notebook (40%)": 55.0,
            "Nilai Akhir Total": round(0.15 * 94.0 + 0.45 * 71.0 + 0.40 * 55.0, 2),
        },
        {
            "Tim": "Tim Seimbang Standar",
            "Public LB (15%)": 80.0,
            "Private LB (45%)": 80.0,
            "Notebook (40%)": 78.0,
            "Nilai Akhir Total": round(0.15 * 80.0 + 0.45 * 80.0 + 0.40 * 78.0, 2),
        },
    ]
).sort_values("Nilai Akhir Total", ascending=False).reset_index(drop=True)

st.dataframe(sim_compare_df, use_container_width=True, hide_index=True)

fig_comp = go.Figure()
fig_comp.add_trace(
    go.Bar(
        name="15% Public LB",
        x=sim_compare_df["Tim"],
        y=sim_compare_df["Public LB (15%)"] * 0.15,
        marker_color="#0E7C86",
    )
)
fig_comp.add_trace(
    go.Bar(
        name="45% Private LB",
        x=sim_compare_df["Tim"],
        y=sim_compare_df["Private LB (45%)"] * 0.45,
        marker_color="#FF6B1A",
    )
)
fig_comp.add_trace(
    go.Bar(
        name="40% Kualitas Notebook",
        x=sim_compare_df["Tim"],
        y=sim_compare_df["Notebook (40%)"] * 0.40,
        marker_color="#111111",
    )
)
fig_comp.update_layout(
    title="Komposisi Penyusun Nilai Akhir (Stacked Bar)",
    barmode="stack",
    yaxis_title="Kontribusi Poin (Total Maks 100)",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F5F0E6",
)
st.plotly_chart(fig_comp, use_container_width=True)