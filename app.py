import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "resnet_96x1024.keras"
MAPPING_PATH = BASE / "model" / "class_mapping.json"
CSV_PATH = BASE / "mahasiswa.csv"

OUT_H, OUT_W, WORK = 96, 1024, 2048
THRESHOLD = 0.40


@st.cache_resource
def load():
    if not MODEL_PATH.exists():
        st.error(f"Model tidak ditemukan: {MODEL_PATH}")
        st.stop()
    if not MAPPING_PATH.exists():
        st.error(f"Mapping tidak ditemukan: {MAPPING_PATH}")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]
    return model, classes


def load_mahasiswa():
    """Membaca mapping kode NIM (nim) -> (nama, nim_resmi) dari mahasiswa.csv."""
    data_mhs = {}
    if not CSV_PATH.exists():
        return data_mhs

    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Bersihkan spasi pada nama kolom dan isinya
            clean_row = {k.strip().lower(): (v.strip() if v else "") for k, v in row.items() if k}
            kode = clean_row.get("nim", "")
            nim_resmi = clean_row.get("nim_resmi", "")
            nama = clean_row.get("nama", "")
            if kode:
                data_mhs[kode] = {
                    "nama": nama if nama else kode,
                    "nim_resmi": nim_resmi,
                }
    return data_mhs


def preprocess(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = min(1.0, WORK / max(g.shape))
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = cv2.divide(g, cv2.medianBlur(g, 51), scale=255)
    _, ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(ink > 0)
    if len(xs) > 100:
        g = g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = g.shape
    s = min(OUT_H / h, OUT_W / w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    pad = np.full((OUT_H, OUT_W), 255, dtype=np.uint8)
    y0, x0 = (OUT_H - nh) // 2, (OUT_W - nw) // 2
    pad[y0 : y0 + nh, x0 : x0 + nw] = g
    return pad


# --- KONFIGURASI HALAMAN & CUSTOM CSS ---
st.set_page_config(
    page_title="Identifikasi Tulisan Tangan",
    page_icon="✍️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 780px;
    }
    .header-box {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 22px;
    }
    .header-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 6px 0;
        line-height: 1.2;
    }
    .header-sub {
        font-size: 0.95rem;
        color: #475569;
        margin: 0;
    }
    .result-card {
        border-radius: 16px;
        padding: 22px 26px;
        margin: 18px 0 24px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }
    .result-confident {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1.5px solid #86efac;
    }
    .result-uncertain {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1.5px solid #fcd34d;
    }
    .res-badge {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
    }
    .badge-ok {
        background-color: #bbf7d0;
        color: #166534;
    }
    .badge-warn {
        background-color: #fde68a;
        color: #92400e;
    }
    .res-nama {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        line-height: 1.25;
    }
    .res-nim {
        font-size: 0.92rem;
        font-weight: 500;
        color: #475569;
        margin-top: 4px;
        font-family: monospace;
    }
    .res-right {
        text-align: right;
        padding-left: 16px;
    }
    .res-persen {
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .persen-ok {
        color: #15803d;
    }
    .persen-warn {
        color: #b45309;
    }
    .res-persen-label {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HEADER ---
st.markdown(
    """
    <div class="header-box">
        <div class="header-title">✍️ Identifikasi Penulis Tulisan Tangan</div>
        <p class="header-sub">Upload foto tulisan tangan 1 kalimat untuk memprediksi identitas penulis (49 kelas).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, classes = load()
data_mhs = load_mahasiswa()

up = st.file_uploader(
    "Upload foto tulisan tangan",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if up:
    bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Gagal membaca gambar.")
        st.stop()

    proc = preprocess(bgr)

    c1, c2 = st.columns(2)
    with c1:
        st.image(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            caption="Gambar Asli (Input)",
            use_container_width=True,
        )
    with c2:
        st.image(
            proc,
            caption=f"Yang dilihat model ({OUT_H}x{OUT_W})",
            use_container_width=True,
        )

    if st.button("🔍 Identifikasi Penulis", type="primary", use_container_width=True):
        x = np.repeat(proc[None, :, :, None], 3, axis=-1).astype("float32")
        prob = model.predict(x, verbose=0)[0]

        # Ambil Top-5 (1 jawaban utama + 4 kandidat lainnya di tabel)
        top_indices = prob.argsort()[::-1][:5]

        # --- JAWABAN UTAMA (RANK 1) ---
        best_idx = top_indices[0]
        best_kode = classes[best_idx]
        info_best = data_mhs.get(best_kode, {"nama": best_kode, "nim_resmi": ""})
        best_nama = info_best["nama"]
        best_nim_display = (
            f"{info_best['nim_resmi']} ({best_kode})"
            if info_best["nim_resmi"]
            else best_kode
        )
        best_conf = prob[best_idx] * 100
        is_confident = prob[best_idx] >= THRESHOLD

        card_cls = "result-confident" if is_confident else "result-uncertain"
        badge_cls = "badge-ok" if is_confident else "badge-warn"
        badge_txt = "✓ Teridentifikasi" if is_confident else "⚠️ Confidence Rendah (Tebakan Terdekat)"
        persen_cls = "persen-ok" if is_confident else "persen-warn"

        st.markdown(
            f"""
            <div class="result-card {card_cls}">
                <div>
                    <span class="res-badge {badge_cls}">{badge_txt}</span>
                    <div class="res-nama">{best_nama}</div>
                    <div class="res-nim">NIM: {best_nim_display}</div>
                </div>
                <div class="res-right">
                    <div class="res-persen {persen_cls}">{best_conf:.1f}%</div>
                    <div class="res-persen-label">Tingkat Keyakinan</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- TABEL KANDIDAT LAINNYA (TANPA JAWABAN UTAMA) ---
        st.markdown("##### 📊 Kemungkinan Kandidat Lainnya")
        other_rows = []
        for rank, idx in enumerate(top_indices[1:], start=2):
            kode = classes[idx]
            info = data_mhs.get(kode, {"nama": "-", "nim_resmi": ""})
            nim_str = f"{info['nim_resmi']} ({kode})" if info["nim_resmi"] else kode
            conf = float(prob[idx] * 100)
            other_rows.append(
                {
                    "Peringkat": f"#{rank}",
                    "Nama Mahasiswa": info["nama"],
                    "NIM": nim_str,
                    "Kecocokan": conf,
                }
            )

        df_others = pd.DataFrame(other_rows)
        st.dataframe(
            df_others,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Peringkat": st.column_config.TextColumn("Rank", width="small"),
                "Nama Mahasiswa": st.column_config.TextColumn("Nama", width="large"),
                "NIM": st.column_config.TextColumn("NIM", width="medium"),
                "Kecocokan": st.column_config.ProgressColumn(
                    "Persentase",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
        )