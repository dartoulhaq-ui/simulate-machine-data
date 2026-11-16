"""Halaman Modul 1: EASY - Kenali Datamu (Kerangka Fase 0, diimplementasikan penuh pada Fase 1)."""

import streamlit as st
from core.utils import init_session_state, load_css, render_page_header, render_sidebar_controls

st.set_page_config(page_title="Easy — Kenali Datamu", page_icon="🟢", layout="wide")
init_session_state()
load_css()
render_sidebar_controls()

render_page_header(
    title="MODUL 1: KENALI DATAMU",
    badge_text="🟢 EASY",
    badge_class="badge-easy",
    est_time="10 Menit",
    subtitle="Eksplorasi data awal (EDA), penanganan missing value sederhana, dan model dasar.",
)

st.info(
    "🚧 **Status Fase 0 (Setup & Navigasi Aktif)**: Kerangka halaman **Easy** sudah terhubung. "
    "Generator dataset sintetis, visualisasi EDA interaktif, dan pelatihan Linear/Logistic Regression akan diisi penuh pada **Fase 1**."
)