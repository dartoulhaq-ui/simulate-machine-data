"""Halaman Modul 4: LEGEND - Mode Lomba Kaggle (Kerangka Fase 0, diimplementasikan penuh pada Fase 3)."""

import streamlit as st
from core.utils import init_session_state, load_css, render_page_header, render_sidebar_controls

st.set_page_config(page_title="Legend — Mode Lomba Kaggle", page_icon="🔴", layout="wide")
init_session_state()
load_css()
render_sidebar_controls()

render_page_header(
    title="MODUL 4: MODE LOMBA KAGGLE",
    badge_text="🔴 LEGEND",
    badge_class="badge-legend",
    est_time="25 Menit",
    subtitle="Simulasi penuh Public vs Private Leaderboard (Shake-up), Neural Network (MLP), dan Pretrained Model.",
)

st.info(
    "🚧 **Status Fase 0 (Setup & Navigasi Aktif)**: Kerangka halaman **Legend** sudah terhubung. "
    "Simulasi kompetisi penuh dan demo pretrained model akan diisi penuh pada **Fase 3**."
)