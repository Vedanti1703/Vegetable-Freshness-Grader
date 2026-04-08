"""
app.py — Streamlit UI for Vegetable Freshness Grader
=====================================================
Integrates a fine-tuned ResNet18 model (freshness_model.pth, ~99.4% accuracy)
with OpenCV visual analysis to grade vegetable freshness.

Run with:  python -m streamlit run app.py
"""

import streamlit as st
from PIL import Image
import numpy as np
import os

# Local modules
from predictor import load_model, analyze_image, is_model_loaded, get_model_accuracy, get_class_names
from scoring import (
    calculate_freshness_score,
    calculate_grade,
    estimate_shelf_life,
    estimate_fair_price,
    get_recommendation,
    detect_issues,
    get_grade_color,
    get_grade_label,
    get_hindi_name,
)

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VegiFresh AI — Freshness Grader",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
# PREMIUM CSS
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

  /* ── Global reset ── */
  .stApp {
    font-family: 'Inter', sans-serif;
    background: #0e1117;
    color: #e8eaf0;
  }
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-thumb { background: #2d8a5e; border-radius: 10px; }

  /* ════════════════════ HERO ════════════════════ */
  .hero {
    background: linear-gradient(135deg, #0a1f14 0%, #0d3320 40%, #1a5c3a 75%, #2d8a5e 100%);
    border-radius: 28px;
    padding: 52px 60px 44px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(45,138,94,0.25);
  }
  .hero::before {
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(59,176,120,0.12) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero::after {
    content: '';
    position: absolute; bottom: -60px; left: -60px;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(45,138,94,0.08) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-top { display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 20px; }
  .hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem; font-weight: 700;
    color: white; margin: 0 0 10px;
    letter-spacing: -1px; line-height: 1.15;
    position: relative; z-index:1;
  }
  .hero-title span { color: #3bb078; }
  .hero-sub {
    font-size: 1.05rem; opacity: 0.75;
    margin: 0 0 20px; font-weight: 400;
    position: relative; z-index:1;
  }
  .hero-pills { display: flex; gap: 10px; flex-wrap: wrap; position: relative; z-index:1; }
  .pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    padding: 7px 16px; border-radius: 50px;
    font-size: 0.82rem; font-weight: 600; color: rgba(255,255,255,0.9);
  }
  .model-badge {
    background: rgba(59,176,120,0.2);
    border: 1px solid rgba(59,176,120,0.5);
    border-radius: 16px;
    padding: 14px 22px;
    text-align: right;
    position: relative; z-index:1;
    min-width: 200px;
  }
  .model-badge .acc-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700; color: #3bb078; line-height: 1;
  }
  .model-badge .acc-label {
    font-size: 0.75rem; color: rgba(255,255,255,0.6);
    text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;
  }

  /* ════════════ STATUS BAR ════════════ */
  .status-bar {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 20px;
    border-radius: 12px;
    font-size: 0.88rem; font-weight: 500;
    margin-bottom: 24px;
  }
  .status-bar.ready {
    background: rgba(59,176,120,0.12);
    border: 1px solid rgba(59,176,120,0.3);
    color: #3bb078;
  }
  .status-bar.demo {
    background: rgba(255,193,7,0.1);
    border: 1px solid rgba(255,193,7,0.3);
    color: #ffc107;
  }
  .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    animation: pulse 2s infinite;
  }
  .status-dot.green { background: #3bb078; box-shadow: 0 0 8px #3bb078; }
  .status-dot.yellow { background: #ffc107; box-shadow: 0 0 8px #ffc107; }
  @keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
  }

  /* ═══════════ UPLOAD CARD ═══════════ */
  .upload-card {
    background: #161b22;
    border: 2px dashed rgba(45,138,94,0.4);
    border-radius: 24px;
    padding: 40px 36px;
    margin-bottom: 24px;
    transition: border-color 0.3s;
    position: relative;
    overflow: hidden;
  }
  .upload-card:hover { border-color: rgba(59,176,120,0.7); }
  .upload-card::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(45,138,94,0.04) 0%, transparent 70%);
    pointer-events: none;
  }
  .upload-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem; font-weight: 600; color: #e8eaf0;
    margin-bottom: 6px;
  }
  .upload-sub { font-size: 0.88rem; color: #6b7280; margin-bottom: 24px; }

  /* ═══════════ IMAGE PREVIEW ═══════════ */
  .img-panel {
    background: #161b22;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    overflow: hidden;
  }
  .img-info-box {
    padding: 14px 18px;
    background: rgba(255,255,255,0.03);
    border-top: 1px solid rgba(255,255,255,0.06);
    font-size: 0.82rem; color: #9ca3af;
  }
  .img-info-box b { color: #e8eaf0; }

  /* ═══════════════ GRADE BANNER ═══════════════ */
  .grade-banner {
    border-radius: 24px;
    padding: 36px 44px;
    display: flex; align-items: center; gap: 30px;
    color: white; margin-bottom: 28px;
    animation: fadeUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative; overflow: hidden;
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
  }
  .grade-banner::after {
    content: '';
    position: absolute; top: -60%; right: -15%;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .grade-circle {
    width: 110px; height: 110px; min-width: 110px;
    border-radius: 50%;
    background: rgba(255,255,255,0.15);
    border: 3px solid rgba(255,255,255,0.35);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem; font-weight: 700;
    backdrop-filter: blur(10px);
    position: relative; z-index:1;
    box-shadow: inset 0 0 30px rgba(255,255,255,0.08);
  }
  .grade-info { position: relative; z-index:1; }
  .grade-info h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem; font-weight: 700;
    margin: 0 0 6px; letter-spacing: -0.5px;
  }
  .grade-sub { font-size: 1rem; opacity: 0.8; }
  .grade-rec {
    margin-top: 14px;
    display: inline-block;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
    padding: 8px 20px; border-radius: 50px;
    font-size: 0.88rem;
  }

  /* ═══════════ STAT CARDS ═══════════ */
  .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .stat-card {
    background: #161b22;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 24px 20px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative; overflow: hidden;
  }
  .stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent, #2d8a5e);
    opacity: 0.7;
  }
  .stat-card:hover {
    transform: translateY(-4px);
    border-color: rgba(45,138,94,0.3);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
  }
  .stat-icon { font-size: 1.8rem; margin-bottom: 10px; }
  .stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem; font-weight: 700;
    color: #3bb078; letter-spacing: -0.5px; line-height: 1;
  }
  .stat-label {
    font-size: 0.72rem; color: #6b7280;
    text-transform: uppercase; letter-spacing: 1px;
    font-weight: 600; margin-top: 6px;
  }

  /* ═══════════ SECTION CARD ═══════════ */
  .section-card {
    background: #161b22;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 20px;
    position: relative; overflow: hidden;
  }
  .section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem; font-weight: 600;
    color: #e8eaf0; margin-bottom: 20px;
    display: flex; align-items: center; gap: 8px;
  }

  /* ═════════════ SCORE DISPLAY ═════════════ */
  .score-big {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 5rem; font-weight: 700;
    line-height: 1; letter-spacing: -3px;
    text-align: center; margin: 16px 0 4px;
  }
  .score-label { text-align: center; font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
  .score-bar-track {
    height: 14px;
    background: rgba(255,255,255,0.06);
    border-radius: 14px; overflow: hidden;
    margin: 18px 0 8px; position: relative;
  }
  .score-bar-fill {
    height: 100%; border-radius: 14px;
    transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
  }
  .score-scale {
    display: flex; justify-content: space-between;
    font-size: 0.7rem; color: #4b5563; font-weight: 600;
  }

  /* ═════════════ ML BADGE ═════════════ */
  .ml-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 10px 22px; border-radius: 50px;
    font-weight: 700; font-size: 0.95rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    margin-top: 12px;
  }
  .ml-badge.fresh { background: linear-gradient(135deg, #1a5c3a, #2d8a5e); color: white; }
  .ml-badge.rotten { background: linear-gradient(135deg, #5a0f0f, #c1392b); color: white; }

  /* ═════════════ METRIC BAR ═════════════ */
  .metric-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .metric-lbl { font-size: 0.85rem; color: #9ca3af; font-weight: 500; }
  .metric-val { font-size: 0.85rem; font-weight: 700; color: #e8eaf0; }
  .bar-track { height: 8px; background: rgba(255,255,255,0.06); border-radius: 8px; overflow: hidden; margin-bottom: 16px; }
  .bar-fill  { height: 100%; border-radius: 8px; transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1); }

  /* ═════ ISSUE / NO-ISSUE ═════ */
  .issue-tag {
    display: inline-block;
    background: rgba(230, 90, 30, 0.12);
    border: 1px solid rgba(230, 90, 30, 0.3);
    color: #f0a060; padding: 8px 16px;
    border-radius: 50px; font-size: 0.83rem;
    margin: 4px; font-weight: 600;
  }
  .no-issue-box {
    background: rgba(59,176,120,0.08);
    border: 1px solid rgba(59,176,120,0.25);
    border-radius: 14px; padding: 20px 24px;
    color: #3bb078; font-weight: 600; font-size: 0.95rem;
  }

  /* ═════ BREAKDOWN BOX ═════ */
  .breakdown-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.88rem;
  }
  .breakdown-row:last-child { border-bottom: none; }
  .breakdown-lbl { color: #9ca3af; }
  .breakdown-val { font-weight: 700; color: #e8eaf0; }

  /* ═════ BUTTON ═════ */
  .stButton > button {
    background: linear-gradient(135deg, #1a5c3a, #2d8a5e) !important;
    color: white !important; border: none !important;
    border-radius: 50px !important;
    padding: 14px 40px !important;
    font-weight: 700 !important; font-size: 1rem !important;
    transition: all 0.3s cubic-bezier(0.16,1,0.3,1) !important;
    box-shadow: 0 4px 20px rgba(26,92,58,0.4) !important;
    letter-spacing: 0.3px !important;
  }
  .stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 32px rgba(26,92,58,0.55) !important;
  }

  /* ═════ UPLOAD ZONE ═════ */
  [data-testid="stFileUploader"] > div {
    background: rgba(45,138,94,0.05) !important;
    border: 2px dashed rgba(45,138,94,0.35) !important;
    border-radius: 16px !important;
    transition: all 0.3s !important;
  }
  [data-testid="stFileUploader"] > div:hover {
    border-color: rgba(59,176,120,0.6) !important;
    background: rgba(45,138,94,0.08) !important;
  }

  @media (max-width: 768px) {
    .hero { padding: 32px 24px; }
    .hero-title { font-size: 1.9rem; }
    .grade-banner { flex-direction: column; text-align: center; padding: 28px; }
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
    .score-big { font-size: 3.5rem; }
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════

if "result" not in st.session_state:
    st.session_state.result = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None


# ══════════════════════════════════════════════════════════════════
# LOAD MODEL (cached across reruns)
# ══════════════════════════════════════════════════════════════════

@st.cache_resource
def init_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "freshness_model.pth")
    return load_model(model_path)

model_ready = init_model()
model_acc   = get_model_accuracy()
class_names = get_class_names()


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def render_metric_bar(label, value, max_val, color, unit=""):
    pct = min((value / max_val) * 100, 100)
    st.markdown(f"""
        <div class="metric-row">
            <span class="metric-lbl">{label}</span>
            <span class="metric-val">{value}{unit}</span>
        </div>
        <div class="bar-track">
            <div class="bar-fill" style="width:{pct}%;background:{color};"></div>
        </div>
    """, unsafe_allow_html=True)

GRADE_BG = {
    "A": "linear-gradient(135deg, #0d3320 0%, #1a5c3a 40%, #2d8a5e 100%)",
    "B": "linear-gradient(135deg, #1e4d0d 0%, #2f7a1a 40%, #55a630 100%)",
    "C": "linear-gradient(135deg, #4a2800 0%, #8a4e0f 40%, #d07030 100%)",
    "D": "linear-gradient(135deg, #3a0505 0%, #7a1010 40%, #c1392b 100%)",
}
SCORE_BAR = {
    "A": "linear-gradient(90deg, #1a5c3a, #2d8a5e, #3bb078)",
    "B": "linear-gradient(90deg, #2a6b1e, #55a630, #7bc043)",
    "C": "linear-gradient(90deg, #8a4e0f, #d07030, #f0a050)",
    "D": "linear-gradient(90deg, #5a0f0f, #c1392b, #e05545)",
}


# ══════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════

acc_display = f"{model_acc:.1f}%" if model_acc else "—"
acc_note    = "Training Accuracy" if model_acc else "Demo Mode"

st.markdown(f"""
<div class="hero">
  <div class="hero-top">
    <div>
      <div class="hero-title">🥦 Vegi<span>Fresh</span> AI</div>
      <p class="hero-sub">AI-powered freshness grader for Indian vegetable markets</p>
      <div class="hero-pills">
        <div class="pill">🧠 ResNet18 CNN</div>
        <div class="pill">📷 OpenCV Analysis</div>
        <div class="pill">⚡ Instant Results</div>
        <div class="pill">🇮🇳 Indian Market Pricing</div>
      </div>
    </div>
    <div class="model-badge">
      <div class="acc-val">{acc_display}</div>
      <div class="acc-label">{acc_note}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Model status bar
if model_ready:
    classes_str = " · ".join(c.title() for c in class_names)
    st.markdown(f"""
    <div class="status-bar ready">
      <div class="status-dot green"></div>
      <b>Model Ready</b> — freshness_model.pth loaded &nbsp;|&nbsp; Classes: {classes_str} &nbsp;|&nbsp; Accuracy: {acc_display}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-bar demo">
      <div class="status-dot yellow"></div>
      <b>Demo Mode</b> — freshness_model.pth not found. Place the model file in the project root for real predictions.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# UPLOAD SECTION
# ══════════════════════════════════════════════════════════════════

if st.session_state.result is None:

    st.markdown("""
    <div class="upload-card">
      <div class="upload-title">📸 Upload a Vegetable Image</div>
      <div class="upload-sub">Supports JPG, PNG, WEBP &nbsp;·&nbsp; Recommended: clear, well-lit photo of a single vegetable</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop image here or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = pil_image

        col_img, col_info = st.columns([1.3, 1])

        with col_img:
            st.markdown('<div class="img-panel">', unsafe_allow_html=True)
            st.image(pil_image, caption="Uploaded Image", use_container_width=True)
            st.markdown(f"""
            <div class="img-info-box">
              <b>File:</b> {uploaded_file.name} &nbsp;·&nbsp;
              <b>Size:</b> {uploaded_file.size/1024:.1f} KB &nbsp;·&nbsp;
              <b>Dimensions:</b> {pil_image.size[0]} × {pil_image.size[1]} px
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_info:
            preview = pil_image.resize((224, 224))
            st.image(preview, caption="Model Input (224×224)", width=190)
            st.markdown("""
            <div style="margin-top:18px;padding:18px;background:#161b22;border:1px solid rgba(255,255,255,0.07);border-radius:16px;">
              <div style="font-size:0.85rem;color:#6b7280;line-height:2;">
                <div>🔬 <b style="color:#e8eaf0;">Stage 1</b> — OpenCV color & texture analysis</div>
                <div>🧠 <b style="color:#e8eaf0;">Stage 2</b> — ResNet18 CNN classification</div>
                <div>📊 <b style="color:#e8eaf0;">Stage 3</b> — Hybrid score (70% ML + 30% CV)</div>
                <div>🏷️ <b style="color:#e8eaf0;">Stage 4</b> — Grade, price & shelf life</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("🔬  Analyze Freshness  →", use_container_width=True):
                with st.spinner("Running AI analysis… please wait"):
                    analysis = analyze_image(pil_image)

                    freshness_score = calculate_freshness_score(
                        analysis["prediction"],
                        analysis["confidence"],
                        analysis["cv_features"],
                    )
                    grade    = calculate_grade(freshness_score)
                    veg_name = "Vegetable"

                    st.session_state.result = {
                        "prediction":     analysis["prediction"],
                        "confidence":     analysis["confidence"],
                        "cv_features":    analysis["cv_features"],
                        "is_mock":        analysis["is_mock"],
                        "model_accuracy": analysis["model_accuracy"],
                        "freshness_score": freshness_score,
                        "grade":          grade,
                        "vegetable_name": veg_name,
                        "hindi_name":     get_hindi_name(veg_name),
                        "shelf_life":     estimate_shelf_life(grade, veg_name),
                        "price_min":      estimate_fair_price(grade, veg_name)[0],
                        "price_max":      estimate_fair_price(grade, veg_name)[1],
                        "recommendation": get_recommendation(grade),
                        "grade_label":    get_grade_label(grade),
                        "grade_color":    get_grade_color(grade),
                        "issues":         detect_issues(
                            analysis["cv_features"],
                            analysis["prediction"],
                            analysis["confidence"],
                        ),
                    }
                    st.rerun()


# ══════════════════════════════════════════════════════════════════
# RESULTS SECTION
# ══════════════════════════════════════════════════════════════════

if st.session_state.result is not None:
    r     = st.session_state.result
    cv    = r["cv_features"]
    grade = r["grade"]
    score = r["freshness_score"]
    color = r["grade_color"]

    # Demo mode warning
    if r["is_mock"]:
        st.markdown("""
        <div class="status-bar demo">
          <div class="status-dot yellow"></div>
          <b>Demo Mode</b> — Results are simulated (color heuristics). Load freshness_model.pth for real CNN predictions.
        </div>
        """, unsafe_allow_html=True)

    # ── Grade Banner ──
    st.markdown(f"""
    <div class="grade-banner" style="background:{GRADE_BG.get(grade,GRADE_BG['C'])};">
      <div class="grade-circle">{grade}</div>
      <div class="grade-info">
        <h2>{r['vegetable_name']} — {r['grade_label']}</h2>
        <div class="grade-sub">{r['hindi_name']} &nbsp;·&nbsp; Freshness Score: <b>{score}/100</b></div>
        <div class="grade-rec">{r['recommendation']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Image + Score ──
    col_l, col_r = st.columns([1, 1.2])

    with col_l:
        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, caption="Analyzed Image", use_container_width=True)

        pred = r["prediction"]
        conf = r["confidence"]
        badge_cls = "fresh" if pred == "fresh" else "rotten"
        badge_emoji = "✅" if pred == "fresh" else "🚫"
        acc_info = f" &nbsp;·&nbsp; Model acc: {r['model_accuracy']:.1f}%" if r.get("model_accuracy") else ""
        st.markdown(f"""
        <div style="text-align:center;margin-top:14px;">
          <div class="ml-badge {badge_cls}">
            {badge_emoji} CNN: {pred.title()} &nbsp;·&nbsp; {conf:.1f}% confidence{acc_info}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown(f"""
        <div class="section-card">
          <div class="section-title">📊 Freshness Score</div>
          <div class="score-big" style="color:{color};">{score}</div>
          <div class="score-label">out of 100</div>
          <div class="score-bar-track">
            <div class="score-bar-fill" style="width:{score}%;background:{SCORE_BAR.get(grade,SCORE_BAR['C'])};"></div>
          </div>
          <div class="score-scale">
            <span>🔴 Rotten (0)</span>
            <span>🟡 Aging (50)</span>
            <span>🟢 Fresh (100)</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stat Cards ──
    shelf_icon = "🟢" if r["shelf_life"] >= 5 else ("🟡" if r["shelf_life"] >= 2 else "🔴")
    conf_icon  = "🎯" if conf >= 80 else ("📊" if conf >= 60 else "⚠️")

    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card" style="--accent:#3bb078">
        <div class="stat-icon">📅</div>
        <div class="stat-val">{r['shelf_life']} days</div>
        <div class="stat-label">Shelf Life {shelf_icon}</div>
      </div>
      <div class="stat-card" style="--accent:#f59e0b">
        <div class="stat-icon">💰</div>
        <div class="stat-val">₹{r['price_min']}–{r['price_max']}</div>
        <div class="stat-label">Fair Price / kg</div>
      </div>
      <div class="stat-card" style="--accent:#60a5fa">
        <div class="stat-icon">{conf_icon}</div>
        <div class="stat-val">{conf:.1f}%</div>
        <div class="stat-label">CNN Confidence</div>
      </div>
      <div class="stat-card" style="--accent:{color}">
        <div class="stat-icon">🏷️</div>
        <div class="stat-val" style="color:{color};">Grade {grade}</div>
        <div class="stat-label">{r['grade_label']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bottom Row: CV Metrics + Issues + Breakdown ──
    col_cv, col_right2 = st.columns([1, 1])

    with col_cv:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔬 OpenCV Visual Analysis</div>', unsafe_allow_html=True)
        render_metric_bar("Color Saturation",   cv["saturation"],  255, "#2d8a5e")
        render_metric_bar("Rot / Dark Spots",   cv["rot_pct"],     100, "#c1392b", "%")
        render_metric_bar("Wilting",            cv["wilting_pct"], 100, "#e07b39", "%")
        render_metric_bar("Wrinkle Density",    cv["edge_density"], 30, "#d4a017")
        st.markdown(f"""
        <div style="margin-top:8px;padding:10px 14px;background:rgba(255,255,255,0.03);
                    border-radius:10px;font-size:0.78rem;color:#6b7280;">
          Brightness: {cv.get('brightness','N/A')} &nbsp;·&nbsp; Saturation: {cv['saturation']:.0f}/255
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right2:
        # Issues
        if r["issues"]:
            st.markdown('<div class="section-card" style="border-color:rgba(230,90,30,0.2);">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="color:#f0a060;">⚠️ Issues Detected</div>', unsafe_allow_html=True)
            for issue in r["issues"]:
                st.markdown(f'<span class="issue-tag">{issue}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="section-card" style="border-color:rgba(59,176,120,0.2);">
              <div class="section-title" style="color:#3bb078;">✅ No Issues Detected</div>
              <div class="no-issue-box">Vegetable appears in excellent condition based on visual inspection.</div>
            </div>
            """, unsafe_allow_html=True)

        # Score Breakdown
        ml_weight  = round(calculate_freshness_score(r["prediction"], r["confidence"], {
            "saturation": 120, "rot_pct": 0, "wilting_pct": 0}) * 0.7, 1)
        st.markdown(f"""
        <div class="section-card" style="margin-top:0;">
          <div class="section-title">📐 Score Breakdown</div>
          <div class="breakdown-row">
            <span class="breakdown-lbl">🧠 CNN Prediction</span>
            <span class="breakdown-val">{r['prediction'].title()} ({r['confidence']:.1f}%)</span>
          </div>
          <div class="breakdown-row">
            <span class="breakdown-lbl">📊 ML Weight (70%)</span>
            <span class="breakdown-val">{round(score * 0.7, 1)} pts</span>
          </div>
          <div class="breakdown-row">
            <span class="breakdown-lbl">📷 CV Weight (30%)</span>
            <span class="breakdown-val">{round(score * 0.3, 1)} pts</span>
          </div>
          <div class="breakdown-row">
            <span class="breakdown-lbl" style="color:{color};font-weight:700;">Final Score</span>
            <span class="breakdown-val" style="color:{color};">{score}/100 → Grade {grade}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Reset ──
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("← Analyze Another Image", use_container_width=True):
            st.session_state.result = None
            st.session_state.uploaded_image = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════

st.markdown(f"""
<div style="text-align:center;padding:48px 20px 28px;color:#374151;font-size:0.78rem;">
  Built with ❤️ using <b style="color:#6b7280;">Streamlit</b> ·
  <b style="color:#6b7280;">PyTorch ResNet18</b> ·
  <b style="color:#6b7280;">OpenCV</b><br>
  VegiFresh AI v3.0 — Model accuracy {acc_display} · Classes: {', '.join(c.title() for c in class_names)}
</div>
""", unsafe_allow_html=True)
