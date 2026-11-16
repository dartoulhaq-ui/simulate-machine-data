"""Halaman Modul 6: Manajemen Submission Kaggle (Kerangka Fase 0, diimplementasikan penuh pada Fase 3)."""

import streamlit as st
from core.utils import init_session_state, load_css, render_page_header, render_sidebar_controls

st.set_page_config(page_title="Submission Manager", page_icon="📮", layout="wide")
init_session_state()
load_css()
render_sidebar_controls()

render_page_header(
    title="MANAJEMEN SUBMISSION KAGGLE",
    badge_text="📮 KUOTA 5 / HARI",
    badge_class="badge-hard",
    est_time="5 Menit",
    subtitle="Pencatat kuota submission harian tim (maksimal 5/hari, reset pukul 00:00 UTC).",
)

st.info(
    "🚧 **Status Fase 0 (Setup & Navigasi Aktif)**: Halaman **Submission Manager** sudah terhubung "
    "dan akan diaktifkan penuh pada **Fase 3**."
)