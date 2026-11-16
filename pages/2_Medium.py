"""Halaman Modul 2: MEDIUM - Bersihkan & Bangun Fitur (Kerangka Fase 0, diimplementasikan penuh pada Fase 1)."""

import streamlit as st
from core.utils import init_session_state, load_css, render_page_header, render_sidebar_controls

st.set_page_config(page_title="Medium — Bersihkan & Bangun Fitur", page_icon="🟡", layout="wide")
init_session_state()
load_css()
render_sidebar_controls()

render_page_header(
    title="MODUL 2: BERSIHKAN & BANGUN FITUR",
    badge_text="🟡 MEDIUM",
    badge_class="badge-medium",
    est_time="15 Menit",
    subtitle="Deteksi outlier, encoding, scaling, feature engineering, dan Pipeline + Cross-Validation.",
)

st.info(
    "🚧 **Status Fase 0 (Setup & Navigasi Aktif)**: Kerangka halaman **Medium** sudah terhubung. "
    "Fitur deteksi outlier, ColumnTransformer Pipeline, dan K-Fold CV akan diisi penuh pada **Fase 1**."
)