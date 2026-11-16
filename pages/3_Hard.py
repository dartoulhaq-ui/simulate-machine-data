"""Halaman Modul 3: HARD - Optimalkan Model (Kerangka Fase 0, diimplementasikan penuh pada Fase 2)."""

import streamlit as st
from core.utils import init_session_state, load_css, render_page_header, render_sidebar_controls

st.set_page_config(page_title="Hard — Optimalkan Model", page_icon="🟠", layout="wide")
init_session_state()
load_css()
render_sidebar_controls()

render_page_header(
    title="MODUL 3: OPTIMALKAN MODEL",
    badge_text="🟠 HARD",
    badge_class="badge-hard",
    est_time="20 Menit",
    subtitle="Penanganan imbalanced data (SMOTE), Gradient Boosting (XGBoost), Hyperparameter Tuning, dan Ensembling.",
)

st.info(
    "🚧 **Status Fase 0 (Setup & Navigasi Aktif)**: Kerangka halaman **Hard** sudah terhubung. "
    "Fitur leaderboard perbandingan model, tuning Optuna/RandomizedSearch, dan ekspor model akan diisi penuh pada **Fase 2**."
)