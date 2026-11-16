"""Modul utilitas untuk seed global, tema CSS, info environment, dan state tim."""

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Dict, Any
import numpy as np
import streamlit as st


DEFAULT_MODULES_PROGRESS: Dict[str, bool] = {
    "easy_eda": False,
    "easy_model": False,
    "medium_prep": False,
    "medium_pipeline": False,
    "hard_imbalance": False,
    "hard_tuning": False,
    "legend_sim": False,
    "legend_notebook": False,
}

GLOSSARY_AWAM: Dict[str, str] = {
    "EDA (Exploratory Data Analysis)": (
        "Proses mengenali kondisi awal data (seperti mengecek jumlah baris, nilai kosong, "
        "dan persebaran angka) sebelum dimasukkan ke dalam model Machine Learning."
    ),
    "Random Seed": (
        "Angka pengunci keacakan (default: 42). Jika seluruh anggota tim memakai angka seed "
        "yang sama, maka pembagian data dan hasil model akan selalu 100% identik."
    ),
    "Missing Value": (
        "Data yang kosong atau tidak tercatat pada tabel (biasanya tertulis NaN/Null). "
        "Harus diisi (imputasi) atau dihapus agar model tidak error."
    ),
    "Outlier": (
        "Nilai ekstrem yang menyimpang jauh dari mayoritas data lainnya (misalnya umur 250 tahun)."
    ),
    "Overfitting": (
        "Kondisi ketika model terlalu menghafal data latihan (atau terlalu menyesuaikan diri dengan "
        "Public Leaderboard) sehingga nilainya anjlok saat diuji pada data baru di Private Leaderboard."
    ),
    "Pipeline": (
        "Rangkaian langkah otomatis yang menggabungkan pembersihan data dan pelatihan model "
        "menjadi satu kesatuan agar tidak terjadi kebocoran data (data leakage)."
    ),
    "Cross-Validation (K-Fold)": (
        "Teknik menguji model secara bergiliran pada beberapa potongan data untuk memastikan "
        "skor model benar-benar stabil dan bukan kebetulan."
    ),
    "Shake-up": (
        "Perubahan peringkat besar-besaran di akhir lomba Kaggle saat skor Private Leaderboard dibuka."
    ),
}


def init_session_state() -> None:
    """Inisialisasi seluruh variabel session_state default jika belum ada."""
    if "global_seed" not in st.session_state:
        st.session_state["global_seed"] = 42

    if " panduan_awam" not in st.session_state:
        st.session_state["panduan_awam"] = True

    if "progress" not in st.session_state:
        st.session_state["progress"] = DEFAULT_MODULES_PROGRESS.copy()

    if "submissions_log" not in st.session_state:
        st.session_state["submissions_log"] = []


def load_css() -> None:
    """Membaca file assets/style.css dan menerapkannya ke halaman Streamlit."""
    css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def get_global_seed() -> int:
    """Mengambil nilai GLOBAL_SEED aktif dari session_state."""
    init_session_state()
    return int(st.session_state.get("global_seed", 42))


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Membuat generator angka acak NumPy berbasis GLOBAL_SEED."""
    active_seed = seed if seed is not None else get_global_seed()
    return np.random.default_rng(active_seed)


def get_environment_info() -> Dict[str, str]:
    """Mengambil versi Python dan library utama menggunakan importlib.metadata."""
    libraries = [
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "plotly",
        "scipy",
        "xgboost",
        "optuna",
        "imbalanced-learn",
        "nbformat",
        "joblib",
    ]
    env_info: Dict[str, str] = {
        "python": sys.version.split()[0],
    }
    for lib in libraries:
        try:
            env_info[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            env_info[lib] = "belum terpasang"
    return env_info


def get_requirements_text() -> str:
    """Membaca isi file requirements.txt untuk fitur unduh reproduksibilitas."""
    req_path = Path(__file__).resolve().parent.parent / "requirements.txt"
    if req_path.exists():
        return req_path.read_text(encoding="utf-8")
    env_info = get_environment_info()
    lines = [
        f"{pkg}=={ver}"
        for pkg, ver in env_info.items()
        if pkg != "python" and ver != "belum terpasang"
    ]
    return "\n".join(lines)


def render_page_header(title: str, badge_text: str, badge_class: str, est_time: str, subtitle: str) -> None:
    """Menampilkan banner header bergaya Neo Brutalism di bagian atas modul."""
    st.markdown(
        f"""
        <div class="neo-header-banner">
            <div>
                <span class="neo-badge {badge_class}">{badge_text}</span>
                <span class="neo-badge badge-time">⏱ {est_time}</span>
            </div>
            <h1 style="margin-top: 0.6rem; margin-bottom: 0.3rem;">{title}</h1>
            <p style="font-size: 1rem; opacity: 0.95;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_controls() -> None:
    """Menampilkan panel kontrol global di sidebar (Seed, Panduan Awam, Ekspor/Impor JSON, Environment)."""
    init_session_state()

    with st.sidebar:
        st.markdown("### ⚙️ Kontrol Global Tim")

        seed_val = st.number_input(
            "🎲 Global Random Seed",
            min_value=0,
            max_value=999999,
            value=int(st.session_state["global_seed"]),
            step=1,
            help="Angka kunci agar hasil pembagian data & pelatihan model selalu konsisten di semua laptop.",
        )
        st.session_state["global_seed"] = int(seed_val)

        st.session_state["panduan_awam"] = st.toggle(
            "🧭 Mode Panduan Awam",
            value=bool(st.session_state.get("panduan_awam", True)),
            help="Tampilkan petunjuk istilah teknis dengan bahasa sehari-hari.",
        )

        utc_now = datetime.now(timezone.utc).strftime("%H:%M UTC")
        st.caption(f"🕒 Waktu Server Kaggle: **{utc_now}** (Reset kuota: 00:00 UTC)")

        st.markdown("---")
        with st.expander("🔄 Simpan / Muat Data Tim (JSON)"):
            st.caption(
                "Karena aplikasi ini tanpa login, gunakan fitur ini untuk membagikan progres "
                "dan catatan submission ke anggota tim lain."
            )
            export_payload: Dict[str, Any] = {
                "global_seed": st.session_state["global_seed"],
                "progress": st.session_state["progress"],
                "submissions_log": st.session_state["submissions_log"],
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            json_bytes = json.dumps(export_payload, indent=2).encode("utf-8")
            st.download_button(
                label="⬇ Ekspor State Tim (.json)",
                data=json_bytes,
                file_name="simulate_machine_data_state.json",
                mime="application/json",
                use_container_width=True,
            )

            uploaded_json = st.file_uploader("⬆ Impor State Tim (.json)", type=["json"])
            if uploaded_json is not None:
                try:
                    loaded_data = json.loads(uploaded_json.read().decode("utf-8"))
                    if "global_seed" in loaded_data:
                        st.session_state["global_seed"] = int(loaded_data["global_seed"])
                    if "progress" in loaded_data and isinstance(loaded_data["progress"], dict):
                        st.session_state["progress"].update(loaded_data["progress"])
                    if "submissions_log" in loaded_data and isinstance(loaded_data["submissions_log"], list):
                        st.session_state["submissions_log"] = loaded_data["submissions_log"]
                    st.success("Data tim berhasil dimuat!")
                except Exception as exc:
                    st.error(f"File JSON tidak valid: {exc}")

        with st.expander("🧪 Info Environment & Library"):
            env_info = get_environment_info()
            for pkg, ver in env_info.items():
                st.markdown(f"- **{pkg}**: `{ver}`")
            st.download_button(
                label="⬇ Unduh requirements.txt",
                data=get_requirements_text().encode("utf-8"),
                file_name="requirements.txt",
                mime="text/plain",
                use_container_width=True,
            )