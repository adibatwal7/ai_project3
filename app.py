#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py  —  Streamlit English → Spanish Translator

Usage:
    streamlit run app.py
    streamlit run app.py -- --model_dir /path/to/models/translation
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch

from inference import load_model, translate, DEFAULT_MODEL_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="English → Spanish · AI Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* ── Background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #1a0533 0%, #0a0a1a 50%, #001a2e 100%);
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem !important; max-width: 1200px !important; }

/* ── Animated top banner ── */
.top-banner {
    width: 100%;
    background: linear-gradient(90deg, #7c3aed22, #3b82f622, #06b6d422);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 0.55rem 2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.78rem;
    color: #6b7280;
    margin-bottom: 0;
}
.top-banner span { color: #a78bfa; font-weight: 600; }

/* ── Hero ── */
.hero-section {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1.1;
    color: #ffffff; /* Use solid white for maximum reliability */
    text-shadow: 0 0 20px rgba(167, 139, 250, 0.5), 0 0 40px rgba(124, 58, 237, 0.3);
    margin: 0 0 1rem;
    letter-spacing: -0.02em;
    display: block;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.1rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.01em;
}
.hero-pills {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 1.2rem;
}
.pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    color: #94a3b8;
    font-weight: 500;
}

/* ── Main translation panel ── */
.translator-grid {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 0;
    align-items: stretch;
    margin: 1.5rem 0;
}

.panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 0;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: border-color 0.3s ease;
}
.panel:hover { border-color: rgba(255,255,255,0.15); }

.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.4rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.lang-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.lang-label.en { color: #93c5fd; }
.lang-label.es { color: #6ee7b7; }
.lang-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
}
.lang-dot.en { background: #3b82f6; box-shadow: 0 0 8px #3b82f6; }
.lang-dot.es { background: #10b981; box-shadow: 0 0 8px #10b981; }
.char-count { font-size: 0.72rem; color: #4b5563; }

/* arrow divider */
.arrow-divider {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 0 1rem;
    color: #4b5563;
}
.arrow-icon {
    font-size: 1.6rem;
    line-height: 1;
}

/* ── textarea override — use Streamlit's own data-testid to beat specificity ── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextArea"] > div > div > textarea {
    background: rgba(15, 15, 35, 0.95) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #ffffff !important; /* Pure white for maximum visibility */
    font-size: 1.2rem !important;
    font-weight: 400 !important;
    line-height: 1.6 !important;
    resize: none !important;
    padding: 1.2rem !important;
    box-shadow: none !important;
    caret-color: #a78bfa !important;
    -webkit-text-fill-color: #ffffff !important; /* Ensure text isn't transparent */
}
[data-testid="stTextArea"] textarea::placeholder {
    color: #4b5563 !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(124,58,237,0.7) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
    outline: none !important;
}

/* ── Output text ── */
.output-body {
    padding: 1.2rem 1.4rem;
    min-height: 160px;
    font-size: 1.2rem;
    line-height: 1.7;
    color: #d1fae5;
    font-weight: 400;
}
.output-placeholder { color: #1f2937; font-style: italic; }

/* ── Translate button (only the main translate button, keyed) ── */
[data-testid="stButton"] > button {
    background: rgba(255,255,255,0.04) !important;
    color: #9ca3af !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 999px !important;
    padding: 0.28rem 0.8rem !important;
    transition: all 0.18s ease !important;
    box-shadow: none !important;
    white-space: nowrap !important;
    min-height: unset !important;
    height: auto !important;
    line-height: 1.4 !important;
}
[data-testid="stButton"] > button:hover {
    background: rgba(124,58,237,0.15) !important;
    border-color: rgba(124,58,237,0.4) !important;
    color: #c4b5fd !important;
    transform: none !important;
    box-shadow: none !important;
}
/* Translate main button override via key class */
.translate-btn [data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 50%, #3b82f6 100%) !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 1.5rem !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.35) !important;
}
.translate-btn [data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(124,58,237,0.5) !important;
    background: linear-gradient(135deg, #6d28d9 0%, #4338ca 50%, #2563eb 100%) !important;
}

/* ── Example chips ── */
.examples-section { margin: 0.5rem 0 1rem; }
.examples-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 0.6rem;
}
.chips { display: flex; flex-wrap: wrap; gap: 0.45rem; }
.chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.8rem;
    color: #9ca3af;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}
.chip:hover { background: rgba(124,58,237,0.15); border-color: rgba(124,58,237,0.4); color: #c4b5fd; }

/* ── History ── */
.history-section { margin-top: 1.5rem; }
.history-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.history-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.history-item:hover { border-color: rgba(255,255,255,0.12); }
.history-en { font-size: 0.9rem; color: #93c5fd; margin-bottom: 0.3rem; }
.history-es { font-size: 0.95rem; color: #d1fae5; font-weight: 500; }
.history-time { font-size: 0.7rem; color: #374151; margin-top: 0.3rem; }

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1rem 1.5rem;
    margin-top: 1rem;
}
.stat-item { display: flex; flex-direction: column; gap: 0.15rem; }
.stat-label { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #4b5563; }
.stat-value { font-size: 0.92rem; font-weight: 600; color: #e2e8f0; }
.stat-value.green { color: #34d399; }
.stat-value.purple { color: #a78bfa; }
.stat-value.blue { color: #60a5fa; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    color: #1f2937;
    font-size: 0.78rem;
}
.footer a { color: #374151; text-decoration: none; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #7c3aed !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
    color: #6b7280 !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model dir
# ─────────────────────────────────────────────────────────────────────────────
def _get_model_dir() -> str:
    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
        args, _ = parser.parse_known_args()
        return args.model_dir
    except Exception:
        return DEFAULT_MODEL_DIR


MODEL_DIR = _get_model_dir()


# ─────────────────────────────────────────────────────────────────────────────
# Cached model loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(model_dir: str):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (*load_model(model_dir, dev), dev)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "last_translation" not in st.session_state:
    st.session_state.last_translation = None
if "translate_time" not in st.session_state:
    st.session_state.translate_time = None

# ─────────────────────────────────────────────────────────────────────────────
# Top micro-banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
    <span>🧠 Transformer NMT</span> · Trained from scratch on 118k sentence pairs ·
    rui.torch.transformer · d_emb=128 · n_layers=2 · n_heads=8 · d_ff=512
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">English → Spanish</h1>
    <p class="hero-sub">Neural machine translation · Transformer · No APIs · Trained from scratch</p>
    <div class="hero-pills">
        <span class="pill">⚡ Greedy decoding</span>
        <span class="pill">🔤 15k vocabulary</span>
        <span class="pill">📚 118k training pairs</span>
        <span class="pill">🏗 6.3M parameters</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    try:
        model, src_vec, tgt_vec, cfg, device = get_model(MODEL_DIR)
        model_loaded = True
    except FileNotFoundError as e:
        model_loaded = False
        load_error = str(e)

if not model_loaded:
    st.error(f"⚠️ Model not found: {load_error}")
    st.info("Run the **translater.ipynb** notebook (cells 1–11) to train and save the model first.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Example chips — clicking one sets the input
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    "Hello, how are you?",
    "I love learning new languages.",
    "Where is the nearest hospital?",
    "The weather is beautiful today.",
    "What time does the train leave?",
    "Thank you very much for your help.",
    "Can you speak more slowly?",
    "I would like a table for two.",
    "My name is Maria.",
    "Do you speak English?",
    "How much does this cost?",
    "I am lost.",
]

st.markdown('<div class="examples-section"><div class="examples-title">💡 Try an example — click to fill the input</div>', unsafe_allow_html=True)
# Use rows of 4 columns to give more space for the text
rows = [EXAMPLES[i:i+4] for i in range(0, len(EXAMPLES), 4)]
for r_idx, row in enumerate(rows):
    cols = st.columns(4)
    for c_idx, ex in enumerate(row):
        with cols[c_idx]:
            if st.button(ex, key=f"chip_{r_idx}_{c_idx}", use_container_width=True):
                st.session_state.input_text = ex
                st.rerun()
    st.write("") # Add spacing between rows
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Translation panel — two columns with arrow in between
# ─────────────────────────────────────────────────────────────────────────────
left_col, mid_col, right_col = st.columns([10, 1, 10], gap="small")

with left_col:
    st.markdown("""
    <div class="panel">
        <div class="panel-header">
            <div class="lang-label en">
                <div class="lang-dot en"></div>
                🇬🇧 &nbsp;English
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    english_input = st.text_area(
        label="English",
        value=st.session_state.input_text,
        placeholder="Type or paste an English sentence…",
        height=180,
        label_visibility="collapsed",
        key="english_input",
    )
    char_count = len(english_input)
    st.caption(f"{char_count} characters")

with mid_col:
    st.markdown("<div style='height:120px; display:flex; align-items:center; justify-content:center;'><span style='font-size:1.8rem; color:#374151;'>→</span></div>", unsafe_allow_html=True)

with right_col:
    st.markdown("""
    <div class="panel">
        <div class="panel-header">
            <div class="lang-label es">
                <div class="lang-dot es"></div>
                🇪🇸 &nbsp;Spanish
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.last_translation:
        st.markdown(
            f'<div class="output-body">{st.session_state.last_translation}</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.translate_time:
            st.caption(f"⏱ {st.session_state.translate_time:.2f}s")
    else:
        st.markdown(
            '<div class="output-body output-placeholder">Translation will appear here…</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Translate button (full width, centred)
# ─────────────────────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([3, 4, 3])
with btn_col:
    st.markdown('<div class="translate-btn">', unsafe_allow_html=True)
    translate_clicked = st.button("🌐  Translate", use_container_width=True, key="translate_main")
    st.markdown('</div>', unsafe_allow_html=True)

if translate_clicked:
    text = english_input.strip()
    if not text:
        st.warning("Please type an English sentence first.")
    else:
        with st.spinner("Translating…"):
            t0 = time.time()
            result = translate(text, model, src_vec, tgt_vec, cfg, device)
            elapsed = time.time() - t0

        st.session_state.last_translation = result
        st.session_state.translate_time = elapsed
        st.session_state.history.insert(0, {
            "en": text,
            "es": result,
            "time": elapsed,
            "ts": time.strftime("%H:%M:%S"),
        })
        # Keep last 10
        st.session_state.history = st.session_state.history[:10]
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Stats bar
# ─────────────────────────────────────────────────────────────────────────────
dev_label = "GPU 🚀" if device.type == "cuda" else "CPU"
translations_done = len(st.session_state.history)
avg_time = (
    sum(h["time"] for h in st.session_state.history) / translations_done
    if translations_done else 0.0
)

st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-label">Device</div>
        <div class="stat-value blue">{dev_label}</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">Architecture</div>
        <div class="stat-value">Transformer Enc–Dec</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">d_emb · n_layers · n_heads · d_ff</div>
        <div class="stat-value purple">{cfg['d_emb']} · {cfg['n_layers']} · {cfg['n_heads']} · {cfg['d_ff']}</div>
    </div>
    <div class="stat-item">
        <div class="stat-label">Translations this session</div>
        <div class="stat-value green">{translations_done}</div>
    </div>
    {"<div class='stat-item'><div class='stat-label'>Avg latency</div><div class='stat-value'>{:.3f}s</div></div>".format(avg_time) if translations_done else ""}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Translation history
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("""
    <div class="history-section">
        <div class="history-title">🕘 Recent translations</div>
    </div>
    """, unsafe_allow_html=True)
    for h in st.session_state.history:
        st.markdown(f"""
        <div class="history-item">
            <div class="history-en">🇬🇧 {h['en']}</div>
            <div class="history-es">🇪🇸 {h['es']}</div>
            <div class="history-time">{h['ts']} · {h['time']:.3f}s</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑 Clear history", key="clear_history"):
        st.session_state.history = []
        st.session_state.last_translation = None
        st.session_state.translate_time = None
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with PyTorch · rui.torch.transformer · Streamlit &nbsp;·&nbsp;
    No external translation APIs used
</div>
""", unsafe_allow_html=True)
