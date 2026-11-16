import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "resnet_96x1024.keras"
MAPPING_PATH = BASE / "model" / "class_mapping.json"

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
    classes = json.load(open(MAPPING_PATH))["classes"]
    return model, classes


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


st.set_page_config(page_title="Identifikasi Tulisan Tangan", page_icon="✍️")
st.title("✍️ Identifikasi Penulis Tulisan Tangan")
st.caption("Upload foto tulisan tangan 1 kalimat → prediksi penulis (49 kelas).")

model, classes = load()

up = st.file_uploader("Foto tulisan tangan", type=["png", "jpg", "jpeg", "bmp", "webp"])

if up:
    bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Gagal membaca gambar.")
        st.stop()

    proc = preprocess(bgr)

    c1, c2 = st.columns(2)
    with c1:
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
    with c2:
        st.image(proc, caption=f"Yang dilihat model ({OUT_H}x{OUT_W})", use_container_width=True)

    if st.button("Identifikasi", type="primary"):
        x = np.repeat(proc[None, :, :, None], 3, axis=-1).astype("float32")
        prob = model.predict(x, verbose=0)[0]
        top = prob.argsort()[::-1][:3]

        if prob[top[0]] < THRESHOLD:
            st.warning(
                f"⚠️ Model tidak yakin. Tebakan terdekat: **{classes[top[0]]}** "
                f"({prob[top[0]] * 100:.1f}%)"
            )
        else:
            st.success(f"Prediksi: **{classes[top[0]]}** ({prob[top[0]] * 100:.1f}%)")

        st.subheader("Top-3")
        for i in top:
            st.write(f"**{classes[i]}** — {prob[i] * 100:.1f}%")
            st.progress(float(min(prob[i], 1.0)))
