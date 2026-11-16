"""Halaman Modul 5: Generator Kerangka Notebook (Kerangka Fase 0, diimplementasikan penuh pada Fase 2)."""

import streamlit as st
from core.utils import init_session_state, load_css, render_page_header, render_sidebar_controls

st.set_page_config(page_title="Notebook Builder", page_icon="📒", layout="wide")
init_session_state()
load_css()
render_sidebar_controls()

render_page_header(
    title="NOTEBOOK BUILDER (.IPYNB)",
    badge_text="📒 DOKUMENTASI",
    badge_class="badge-medium",
    est_time="5 Menit",
    subtitle="Generator kerangka Jupyter Notebook rapi dalam Bahasa Indonesia & Inggris.",
)

st.info(
    "🚧 **Status Fase 0 (Setup & Navigasi Aktif)**: Halaman **Notebook Builder** sudah terhubung "
    "dan akan diaktifkan penuh pada **Fase 2**."
)