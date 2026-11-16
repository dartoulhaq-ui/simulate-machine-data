"""Halaman Modul 7: Kalkulator Skor & Evaluasi Notebook (Kerangka Fase 0, diimplementasikan penuh pada Fase 3)."""

import streamlit as st
from core.utils import init_session_state, load_css, render_page_header, render_sidebar_controls

st.set_page_config(page_title="Kalkulator Skor Lomba", page_icon="🧮", layout="wide")
init_session_state()
load_css()
render_sidebar_controls()

render_page_header(
    title="KALKULATOR SKOR & CHECKLIST NOTEBOOK",
    badge_text="🧮 EVALUASI LOMBA",
    badge_class="badge-legend",
    est_time="5 Menit",
    subtitle="Simulasi bobot penilaian akhir: 60% Kaggle (15% Public + 45% Private) & 40% Kualitas Notebook.",
)

st.info(
    "🚧 **Status Fase 0 (Setup & Navigasi Aktif)**: Halaman **Kalkulator Skor** sudah terhubung "
    "dan akan diaktifkan penuh pada **Fase 3**."
)