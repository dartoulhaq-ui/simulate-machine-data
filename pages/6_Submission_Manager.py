"""Halaman Modul 6: MANAJEMEN SUBMISSION KAGGLE (Kuota Maks 5/Hari, Reset 00:00 UTC, Log & Ekspor JSON/CSV)."""

from datetime import datetime, timezone
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.utils import (
    MAX_DAILY_SUBMISSIONS,
    add_submission_record,
    get_daily_submission_status,
    get_global_seed,
    get_utc_today_str,
    init_session_state,
    load_css,
    render_page_header,
    render_sidebar_controls,
)

st.set_page_config(page_title="Submission Manager Kaggle", page_icon="📮", layout="wide")
init_session_state()
load_css()
render_sidebar_controls()

global_seed = get_global_seed()

render_page_header(
    title="MANAJEMEN SUBMISSION KAGGLE TIM",
    badge_text="📮 KUOTA MAKS 5 / HARI (RESET 00:00 UTC)",
    badge_class="badge-hard",
    est_time="5 Menit",
    subtitle="Pantau sisa kuota harian, catat skor CV vs Public Leaderboard, dan cegah pemborosan submission tim Anda.",
)

used_today, rem_today, today_logs = get_daily_submission_status()
utc_now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# Panel Indikator Kuota
k1, k2, k3, k4 = st.columns(4)
k1.metric("🕒 Waktu Server Kaggle (UTC)", utc_now_str.split()[1] + " UTC")
k2.metric("📊 Terpakai Hari Ini", f"{used_today} / {MAX_DAILY_SUBMISSIONS}")
k3.metric("🟢 Sisa Kuota Hari Ini", f"{rem_today} submission")
k4.metric("📚 Total Riwayat Log Tim", f"{len(st.session_state['submissions_log'])} tercatat")

st.progress(min(1.0, used_today / float(MAX_DAILY_SUBMISSIONS)))

if rem_today == 0:
    st.error(
        "🚨 **PERINGATAN KUOTA HABIS (5/5)**: Tim Anda telah mencapai batas maksimal **5 submission** untuk tanggal "
        f"`{get_utc_today_str()} (UTC)`. Kuota baru akan terbuka otomatis pada pukul **00:00 UTC** (07:00 WIB). "
        "Gunakan waktu tersisa untuk memperbaiki dokumentasi Notebook (bobot 40%)!"
    )
elif rem_today == 1:
    st.warning(
        "⚠️ **Sisa 1 Kuota Terakhir Hari Ini!** Pastikan Anda hanya mengirim model dengan skor *K-Fold CV* terbaik."
    )
else:
    st.info(
        f"✅ Kuota masih aman (**{rem_today} submission tersisa** untuk hari UTC `{get_utc_today_str()}`). "
        "Catat setiap pengiriman agar seluruh anggota tim mengetahui progres skor terbaru."
    )

col_form, col_chart = st.columns([1, 1])

with col_form:
    st.markdown("### ➕ Catat Submission Baru ke Log Tim")
    with st.form("add_submission_manual_form"):
        model_desc = st.text_input(
            "Nama Model / Pipeline yang Disubmit",
            value="XGBoost Tuned (Optuna 15 Trials) + 5-Fold CV",
        )
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            cv_score = st.number_input(
                "Skor Validasi Lokal (K-Fold CV)",
                min_value=0.0,
                max_value=1.0,
                value=0.8450,
                step=0.001,
                format="%.4f",
            )
        with c_s2:
            public_score = st.number_input(
                "Skor Public Leaderboard Kaggle",
                min_value=0.0,
                max_value=1.0,
                value=0.8390,
                step=0.001,
                format="%.4f",
            )
        notes = st.text_area(
            "Catatan Perubahan Fitur / Hyperparameter",
            value="Menambahkan fitur rasio pendapatan dan imputasi KNN (k=5).",
            height=80,
        )
        submit_btn = st.form_submit_button(
            "📮 Simpan Catatan Submission (Pakai 1 Kuota)",
            disabled=(rem_today <= 0),
            use_container_width=True,
        )

    if submit_btn:
        ok, msg = add_submission_record(
            model_desc=model_desc,
            cv_score=cv_score,
            public_score=public_score,
            notes=notes,
        )
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

with col_chart:
    st.markdown("### 📈 Grafik Tren Skor: Local CV vs Public LB")
    all_logs = st.session_state["submissions_log"]
    if len(all_logs) >= 1:
        df_log = pd.DataFrame(all_logs)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_log["id"],
                y=df_log["cv_score"],
                mode="lines+markers",
                name="Skor Local K-Fold CV",
                line=dict(color="#0E7C86", width=3),
                marker=dict(size=9),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_log["id"],
                y=df_log["public_score"],
                mode="lines+markers",
                name="Skor Public Leaderboard",
                line=dict(color="#FF6B1A", width=3),
                marker=dict(size=9),
            )
        )
        fig.update_layout(
            title="Perkembangan Skor dari Submission ke Submission",
            xaxis_title="Submission #",
            yaxis_title="Skor Metrik",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F5F0E6",
            margin=dict(l=30, r=20, t=45, b=35),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "💡 **Tips Anti-Shakeup**: Pilih *Final Submission* yang memiliki **Skor Local CV** dan **Public LB** "
            "sama-sama tinggi serta celah (*gap*) paling rapat."
        )
    else:
        st.info("Belum ada catatan submission. Tambahkan catatan di sebelah kiri atau jalankan simulasi di halaman **Legend**.")

st.markdown("---")
st.markdown("### 📋 Tabel Riwayat Log Submission Tim")

if st.session_state["submissions_log"]:
    df_table = pd.DataFrame(st.session_state["submissions_log"])
    df_table["Selisih (Public - CV)"] = (df_table["public_score"] - df_table["cv_score"]).round(4)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        st.download_button(
            "📄 Ekspor Log ke CSV (.csv)",
            data=df_table.to_csv(index=False).encode("utf-8"),
            file_name="kaggle_submissions_log.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with b2:
        json_log = json.dumps(st.session_state["submissions_log"], indent=2).encode("utf-8")
        st.download_button(
            "🔄 Ekspor Log ke JSON (.json)",
            data=json_log,
            file_name="kaggle_submissions_log.json",
            mime="application/json",
            use_container_width=True,
        )
    with b3:
        if st.button("🗑️ Reset Log Simulasi (Kosongkan Kuota Latihan)", use_container_width=True):
            st.session_state["submissions_log"] = []
            st.rerun()
else:
    st.caption("Log masih kosong.")