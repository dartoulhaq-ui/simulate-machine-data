"""Visualisasi data & evaluasi model (Easy, Medium, Hard) dengan Plotly + kalimat Insight otomatis."""

from typing import List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

NEO_COLORS = ["#0E7C86", "#FF6B1A", "#111111", "#2B9348", "#E63946", "#6C757D"]


def _apply_neo_layout(fig: go.Figure, title: str) -> go.Figure:
    """Menerapkan gaya visual Neo Brutalism yang bersih pada grafik Plotly."""
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk, sans-serif", size=18, color="#111111")),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F5F0E6",
        font=dict(family="Inter, sans-serif", color="#111111"),
        margin=dict(l=40, r=30, t=55, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#DED8CE", linecolor="#111111", linewidth=2)
    fig.update_yaxes(showgrid=True, gridcolor="#DED8CE", linecolor="#111111", linewidth=2)
    return fig


def plot_histogram_with_insight(df: pd.DataFrame, col: str) -> Tuple[go.Figure, str]:
    """Membuat histogram untuk satu kolom numerik beserta kalimat insight otomatis."""
    series = df[col].dropna()
    fig = px.histogram(
        df,
        x=col,
        nbins=30,
        color_discrete_sequence=["#0E7C86"],
    )
    fig.update_traces(marker_line_color="#111111", marker_line_width=1.5)
    fig = _apply_neo_layout(fig, f"Distribusi Histogram: {col}")

    if len(series) > 2:
        mean_v = float(series.mean())
        median_v = float(series.median())
        skew_v = float(series.skew())
        if abs(skew_v) < 0.5:
            shape_desc = "cukup simetris (mendekati distribusi normal)"
        elif skew_v >= 0.5:
            shape_desc = "menceng ke kanan (positively skewed — ada ekor nilai tinggi/outlier di kanan)"
        else:
            shape_desc = "menceng ke kiri (negatively skewed — ada ekor nilai rendah di kiri)"
        insight = (
            f"💡 **Insight Otomatis**: Kolom `{col}` memiliki rata-rata **{mean_v:.2f}** dan median **{median_v:.2f}**. "
            f"Bentuk sebarannya **{shape_desc}** (skewness = `{skew_v:.2f}`). Jika rata-rata dan median terpaut jauh, "
            f"gunakan **Median** saat mengisi nilai kosong (imputasi)."
        )
    else:
        insight = f"💡 **Insight Otomatis**: Data pada kolom `{col}` terlalu sedikit untuk dihitung kemiringannya."
    return fig, insight


def plot_boxplot_with_insight(
    df: pd.DataFrame,
    num_col: str,
    cat_col: str | None = None,
) -> Tuple[go.Figure, str]:
    """Membuat Boxplot untuk mendeteksi outlier secara visual beserta insight otomatis."""
    fig = px.box(
        df,
        x=cat_col if cat_col else None,
        y=num_col,
        color=cat_col if cat_col else None,
        color_discrete_sequence=NEO_COLORS,
    )
    fig = _apply_neo_layout(fig, f"Boxplot & Deteksi Outlier: {num_col}")

    series = df[num_col].dropna()
    if len(series) >= 4:
        q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
        iqr = q3 - q1
        outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
        insight = (
            f"💡 **Insight Otomatis**: Rentang tengah (Q1 s.d. Q3) untuk `{num_col}` berada di antara **{q1:.2f}** "
            f"hingga **{q3:.2f}**. Terdeteksi **{len(outliers)} titik di luar garis pagar (outlier)**. "
            f"Titik-titik di luar garis kumis (whiskers) dapat menarik garis model regresi menjadi miring."
        )
    else:
        insight = f"💡 **Insight Otomatis**: Boxplot menampilkan batas kuartil dari `{num_col}`."
    return fig, insight


def plot_scatter_with_insight(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    seed: int = 42,
) -> Tuple[go.Figure, str]:
    """Membuat Scatter Plot (sampling maks 2.000 titik agar tidak lag) + insight korelasi."""
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    sampled_note = ""
    if len(plot_df) > 2000:
        plot_df = plot_df.sample(n=2000, random_state=seed)
        sampled_note = " (ditampilkan 2.000 sampel acak agar grafik tetap ringan)"

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col if (color_col and color_col in plot_df.columns) else None,
        color_discrete_sequence=NEO_COLORS,
        opacity=0.8,
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="#111111")))
    fig = _apply_neo_layout(fig, f"Scatter Plot: {x_col} vs {y_col}{sampled_note}")

    if pd.api.types.is_numeric_dtype(plot_df[x_col]) and pd.api.types.is_numeric_dtype(plot_df[y_col]):
        corr_val = float(plot_df[x_col].corr(plot_df[y_col]))
        if np.isnan(corr_val):
            corr_val = 0.0
        abs_c = abs(corr_val)
        strength = "kuat" if abs_c >= 0.6 else ("sedang" if abs_c >= 0.3 else "lemah")
        direction = "positif (searah)" if corr_val >= 0 else "negatif (berlawanan arah)"
        insight = (
            f"💡 **Insight Otomatis**: Hubungan linear antara `{x_col}` dan `{y_col}` tergolong **{strength} & {direction}** "
            f"(koefisien korelasi Pearson $r = {corr_val:.2f}$). Semakin rapat titik-titik membentuk garis miring, "
            f"semakin mudah model Linear Regression memprediksinya."
        )
    else:
        insight = f"💡 **Insight Otomatis**: Scatter plot memetakan persebaran `{x_col}` terhadap `{y_col}`."
    return fig, insight


def plot_correlation_heatmap(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Membuat Heatmap Korelasi antar kolom numerik + insight pasangan fitur tertinggi."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        fig = go.Figure()
        return _apply_neo_layout(fig, "Heatmap Korelasi (Minimal 2 Kolom Numerik)"), "💡 Tambahkan kolom numerik."

    corr_matrix = num_df.corr().fillna(0.0).round(2)
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale=["#FF6B1A", "#F5F0E6", "#0E7C86"],
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig = _apply_neo_layout(fig, "Heatmap Korelasi Antar Fitur Numerik")

    # Gunakan .to_numpy(copy=True) agar array bersifat writable pada Pandas 2.2+ / NumPy 2.0+
    abs_corr_arr = corr_matrix.abs().to_numpy(copy=True)
    np.fill_diagonal(abs_corr_arr, 0.0)
    max_pos = np.unravel_index(np.argmax(abs_corr_arr), abs_corr_arr.shape)
    col_1 = corr_matrix.index[int(max_pos[0])]
    col_2 = corr_matrix.columns[int(max_pos[1])]
    best_r = float(corr_matrix.loc[col_1, col_2])

    insight = (
        f"💡 **Insight Otomatis**: Pasangan kolom dengan hubungan terkuat adalah **`{col_1}`** dan **`{col_2}`** "
        f"(korelasi $r = {best_r:.2f}$). Nilai mendekati `+1` atau `-1` menandakan fitur tersebut sangat informatif "
        f"(atau saling duplikat/multikolinear jika keduanya adalah sesama fitur input)."
    )
    return fig, insight


def plot_categorical_bar(df: pd.DataFrame, cat_col: str) -> Tuple[go.Figure, str]:
    """Membuat grafik batang frekuensi untuk kolom kategorikal + insight ketimpangan kategori."""
    counts = df[cat_col].fillna("MISSING").astype(str).value_counts().reset_index()
    counts.columns = [cat_col, "Jumlah"]

    fig = px.bar(
        counts,
        x=cat_col,
        y="Jumlah",
        text="Jumlah",
        color=cat_col,
        color_discrete_sequence=NEO_COLORS,
    )
    fig.update_traces(marker_line_color="#111111", marker_line_width=2)
    fig = _apply_neo_layout(fig, f"Distribusi Kategori: {cat_col}")

    top_cat = counts.iloc[0][cat_col]
    top_pct = (counts.iloc[0]["Jumlah"] / max(1, len(df))) * 100.0
    insight = (
        f"💡 **Insight Otomatis**: Kategori terbanyak pada `{cat_col}` adalah **`{top_cat}`** (**{top_pct:.1f}%** dari total baris). "
        f"Perhatikan apakah ada kategori langka (kurang dari 5%) karena kategori yang terlalu sedikit bisa memicu overfitting saat di-One-Hot Encode."
    )
    return fig, insight


def plot_confusion_matrix_chart(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[go.Figure, str]:
    """Membuat Confusion Matrix untuk evaluasi model klasifikasi beserta penjelasan awam."""
    labels = sorted(list(set(np.unique(y_true)).union(set(np.unique(y_pred)))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    str_labels = [f"Kelas {lbl}" for lbl in labels]

    fig = px.imshow(
        cm,
        x=str_labels,
        y=str_labels,
        text_auto=True,
        color_continuous_scale=["#F5F0E6", "#0E7C86"],
        labels=dict(x="Prediksi Model", y="Kenyataan (Aktual)", color="Jumlah"),
    )
    fig = _apply_neo_layout(fig, "Confusion Matrix (Kenyataan vs Prediksi)")

    correct = int(np.trace(cm))
    total = int(cm.sum())
    wrong = total - correct
    insight = (
        f"💡 **Insight Cara Membaca**: Kotak diagonal dari kiri-atas ke kanan-bawah adalah **tebakan benar** "
        f"(total **{correct} dari {total}** data uji tepat). Sedangkan **{wrong} data** di luar diagonal adalah "
        f"kesalahan tebak (*False Positive* / *False Negative*)."
    )
    return fig, insight


def plot_residuals_chart(y_true: np.ndarray, y_pred: np.ndarray, seed: int = 42) -> Tuple[go.Figure, str]:
    """Membuat Plot Residual (Prediksi vs Selisih Error) untuk model regresi."""
    residuals = y_true - y_pred
    res_df = pd.DataFrame({"Prediksi": y_pred, "Residual (Aktual - Prediksi)": residuals})
    if len(res_df) > 2000:
        res_df = res_df.sample(n=2000, random_state=seed)

    fig = px.scatter(
        res_df,
        x="Prediksi",
        y="Residual (Aktual - Prediksi)",
        color_discrete_sequence=["#FF6B1A"],
        opacity=0.8,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#111111", line_width=2)
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="#111111")))
    fig = _apply_neo_layout(fig, "Plot Residual (Garis Putus-Putus = Error Nol)")

    mean_abs_err = float(np.mean(np.abs(residuals)))
    insight = (
        f"💡 **Insight Cara Membaca**: Titik yang menempel dekat dengan garis putus-putus horizontal (`0`) berarti "
        f"tebakan model sangat akurat. Rata-rata simpangan meleset (MAE) saat ini adalah **{mean_abs_err:.3f}**. "
        f"Jika titik menyebar acak di atas & bawah garis 0 tanpa membentuk pola melengkung, berarti model sudah sehat."
    )
    return fig, insight


def plot_feature_importance_chart(
    feature_names: List[str],
    importances: np.ndarray,
) -> Tuple[go.Figure, str]:
    """Membuat grafik Feature Importance / Koefisien terpenting."""
    imp_df = pd.DataFrame(
        {
            "Fitur": feature_names,
            "Kontribusi": np.abs(importances),
        }
    ).sort_values("Kontribusi", ascending=True).tail(15)

    fig = px.bar(
        imp_df,
        x="Kontribusi",
        y="Fitur",
        orientation="h",
        color_discrete_sequence=["#0E7C86"],
    )
    fig.update_traces(marker_line_color="#111111", marker_line_width=1.5)
    fig = _apply_neo_layout(fig, "Top Fitur Paling Berpengaruh (Feature Importance)")

    top_feat = imp_df.iloc[-1]["Fitur"] if not imp_df.empty else "-"
    insight = (
        f"💡 **Insight Otomatis**: Fitur **`{top_feat}`** memberikan kontribusi terbesar bagi keputusan model. "
        f"Pada lomba Kaggle, fitur dengan kontribusi paling tinggi adalah kandidat terbaik untuk dieksplorasi "
        f"lebih lanjut menjadi fitur interaksi atau rasio baru."
    )
    return fig, insight


def plot_roc_curve_with_insight(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[go.Figure, str]:
    """Membuat grafik ROC Curve beserta nilai AUC dan penjelasan cara membacanya."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {roc_auc:.4f})",
            line=dict(color="#0E7C86", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Tebakan Acak (AUC = 0.50)",
            line=dict(color="#111111", width=2, dash="dash"),
        )
    )
    fig.update_xaxes(title_text="False Positive Rate (1 - Specificity)")
    fig.update_yaxes(title_text="True Positive Rate (Recall)")
    fig = _apply_neo_layout(fig, f"Kurva ROC (Area Under Curve = {roc_auc:.4f})")

    quality = "Sangat Baik" if roc_auc >= 0.85 else ("Cukup Baik" if roc_auc >= 0.72 else "Perlu Optimasi")
    insight = (
        f"💡 **Insight Cara Membaca ROC-AUC**: Luas area di bawah kurva (**ROC-AUC**) adalah **`{roc_auc:.4f}`** ({quality}). "
        f"Semakin melengkung kurva hijau mendekati pojok kiri-atas `(0, 1)`, semakin pintar model membedakan kelas positif dan negatif."
    )
    return fig, insight


def plot_pr_curve_with_insight(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[go.Figure, str]:
    """Membuat grafik Precision-Recall Curve (sangat krusial untuk data tidak seimbang/imbalanced)."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = float(average_precision_score(y_true, y_prob))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"PR Curve (AP = {ap_score:.4f})",
            line=dict(color="#FF6B1A", width=3),
        )
    )
    fig.update_xaxes(title_text="Recall (Sensitivitas Kelas Minoritas)")
    fig.update_yaxes(title_text="Precision (Ketepatan Tebakan Positif)")
    fig = _apply_neo_layout(fig, f"Precision-Recall Curve (PR-AUC = {ap_score:.4f})")

    insight = (
        f"💡 **Insight Cara Membaca PR Curve**: Pada dataset timpang (*imbalanced*), kurva **Precision-Recall** "
        f"(skor **`{ap_score:.4f}`**) jauh lebih jujur daripada ROC-AUC. Kurva yang bertahan tinggi di pojok kanan-atas "
        f"menunjukkan model mampu menangkap kelas minoritas (*Recall* tinggi) tanpa banyak salah tuduh (*Precision* tinggi)."
    )
    return fig, insight


def plot_learning_curve_with_insight(
    train_sizes: np.ndarray,
    train_scores_mean: np.ndarray,
    val_scores_mean: np.ndarray,
    metric_label: str = "Skor",
) -> Tuple[go.Figure, str]:
    """Membuat grafik Learning Curve untuk mendiagnosis overfitting vs underfitting."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_scores_mean,
            mode="lines+markers",
            name="Skor Data Latihan (Train)",
            line=dict(color="#111111", width=3),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=val_scores_mean,
            mode="lines+markers",
            name="Skor Validasi (Cross-Validation)",
            line=dict(color="#0E7C86", width=3),
            marker=dict(size=8),
        )
    )
    fig.update_xaxes(title_text="Jumlah Sampel Data Latihan")
    fig.update_yaxes(title_text=metric_label)
    fig = _apply_neo_layout(fig, "Learning Curve (Diagnosis Overfitting vs Underfitting)")

    gap = float(train_scores_mean[-1] - val_scores_mean[-1])
    if gap > 0.12:
        diag = "terindikasi **Overfitting** (jarak garis Train dan Validasi cukup lebar). Kurangi `max_depth` atau tambah regularisasi."
    elif float(val_scores_mean[-1]) < 0.65:
        diag = "terindikasi **Underfitting** (kedua garis masih rendah). Gunakan model Boosting atau tambah fitur baru."
    else:
        diag = "**Konvergen & Sehat** (garis Validasi naik mendekati garis Train dengan celah yang wajar)."

    insight = (
        f"💡 **Insight Learning Curve**: Selisih akhir antara skor Train dan Validasi adalah **`{gap:.4f}`**, sehingga model {diag}"
    )
    return fig, insight


def plot_leaderboard_bar_with_insight(
    leaderboard_df: pd.DataFrame,
    score_col: str,
) -> Tuple[go.Figure, str]:
    """Membuat grafik batang perbandingan skor antar model pada Leaderboard."""
    plot_df = leaderboard_df.sort_values(score_col, ascending=True)
    fig = px.bar(
        plot_df,
        x=score_col,
        y="Model",
        orientation="h",
        text=score_col,
        color="Kategori",
        color_discrete_sequence=NEO_COLORS,
    )
    fig.update_traces(marker_line_color="#111111", marker_line_width=1.5)
    fig = _apply_neo_layout(fig, f"Perbandingan Leaderboard Model ({score_col})")

    best_row = leaderboard_df.iloc[0]
    insight = (
        f"💡 **Insight Leaderboard**: Peringkat #1 ditempati oleh **`{best_row['Model']}`** dengan `{score_col}` sebesar "
        f"**`{best_row[score_col]}`**. Perhatikan bahwa model **Ensemble / Tuned** umumnya lebih stabil dibanding model tunggal."
    )
    return fig, insight