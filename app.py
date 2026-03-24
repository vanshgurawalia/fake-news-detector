import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from transformers import BertTokenizer, BertForSequenceClassification
import time
import re
from utils import (
    preprocess_text,
    get_confidence_label,
    highlight_suspicious_words,
    get_credibility_score,
    SUSPICIOUS_PATTERNS,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeShield — AI Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e2e;
}

/* Main header */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #7b2fff 50%, #ff2d78 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.main-header p {
    color: #8888aa;
    font-size: 1.05rem;
    font-weight: 300;
}

/* Cards */
.card {
    background: #111120;
    border: 1px solid #1e1e35;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Result badges */
.result-real {
    background: linear-gradient(135deg, #00ff87 0%, #00d4a0 100%);
    color: #001a0d;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.4rem;
    display: inline-block;
}
.result-fake {
    background: linear-gradient(135deg, #ff2d78 0%, #ff6b35 100%);
    color: #fff;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.4rem;
    display: inline-block;
}
.result-uncertain {
    background: linear-gradient(135deg, #ffd700 0%, #ff9500 100%);
    color: #1a1000;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.4rem;
    display: inline-block;
}

/* Metric boxes */
.metric-box {
    background: #0f0f1e;
    border: 1px solid #252545;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
}
.metric-label {
    font-size: 0.8rem;
    color: #666688;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Text area */
.stTextArea textarea {
    background: #0f0f1e !important;
    border: 1px solid #252545 !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7b2fff) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.7rem 2.5rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.35) !important;
}

/* Highlighted text */
.highlight-suspicious {
    background: rgba(255, 45, 120, 0.25);
    border-bottom: 2px solid #ff2d78;
    border-radius: 3px;
    padding: 0 2px;
}
.highlight-neutral {
    color: #e8e8f0;
}

/* Progress bar override */
.stProgress > div > div {
    background: linear-gradient(90deg, #00d4ff, #7b2fff) !important;
}

/* Divider */
hr { border-color: #1e1e35 !important; }

/* Info/warning boxes */
.info-box {
    background: rgba(0, 212, 255, 0.08);
    border-left: 3px solid #00d4ff;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.9rem;
    color: #aaccff;
}
.warn-box {
    background: rgba(255, 45, 120, 0.08);
    border-left: 3px solid #ff2d78;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.9rem;
    color: #ffaabb;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #8888cc;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load BERT model and tokenizer."""
    model_name = "jy46604790/Fake-News-Bert-Detect"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model, max_len: int = 512):
    """Run inference and return (label, real_prob, fake_prob)."""
    cleaned = preprocess_text(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        max_length=max_len,
        truncation=True,
        padding="max_length",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        probs = probs.squeeze().numpy()

    # probs[0] = REAL, probs[1] = FAKE (standard ordering; adjust if fine-tuned otherwise)
    real_prob, fake_prob = float(probs[0]), float(probs[1])
    label = "FAKE" if fake_prob > real_prob else "REAL"
    return label, real_prob, fake_prob


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FakeShield")
    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    max_len = st.slider("Max Token Length", 64, 512, 256, 32)
    show_tokens = st.checkbox("Show token analysis", value=False)
    show_patterns = st.checkbox("Highlight suspicious patterns", value=True)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    <div class="info-box">
    <b>Architecture:</b> BERT-base-uncased<br>
    <b>Parameters:</b> ~110M<br>
    <b>Task:</b> Binary Classification<br>
    <b>Labels:</b> REAL / FAKE
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 How It Works")
    st.markdown("""
    1. Text is tokenized via WordPiece  
    2. BERT encodes contextual embeddings  
    3. [CLS] token → classification head  
    4. Softmax → confidence scores  
    """)

    st.markdown("---")
    st.markdown("### 🧪 Sample Headlines")
    samples = {
        "🟢 Likely Real": "NASA scientists discover evidence of water ice in permanently shadowed craters near Moon's south pole.",
        "🔴 Likely Fake": "BREAKING: Secret government document EXPOSES massive cover-up — they don't want you to know THIS!",
        "🟡 Borderline": "New study suggests coffee may have surprising health benefits that experts are still debating.",
    }
    for label, text in samples.items():
        if st.button(label, use_container_width=True):
            st.session_state["sample_text"] = text


# ─── Main UI ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ FakeShield</h1>
    <p>BERT-powered fake news detection — paste any article, headline, or claim below</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Loading BERT model..."):
    tokenizer, model = load_model()

st.markdown("---")

# Input area
col_input, col_info = st.columns([3, 1])

with col_input:
    default_text = st.session_state.get("sample_text", "")
    user_text = st.text_area(
        "📝 Enter news article or headline",
        value=default_text,
        height=160,
        placeholder="Paste a news headline, paragraph, or full article here...",
    )

with col_info:
    st.markdown("<div class='section-title'>Quick Tips</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    ✅ Works best with English text<br><br>
    ✅ Longer context = more accurate<br><br>
    ✅ Headlines alone may be ambiguous<br><br>
    ⚠️ This is a demo model — not fine-tuned on a news dataset
    </div>
    """, unsafe_allow_html=True)

# Analyze button
col_btn, _ = st.columns([1, 3])
with col_btn:
    analyze = st.button("🔍 Analyze", use_container_width=True)

# ─── Results ──────────────────────────────────────────────────────────────────
if analyze:
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Running BERT inference..."):
            time.sleep(0.4)  # slight UX pause
            label, real_prob, fake_prob = predict(user_text, tokenizer, model, max_len)
            conf_label, conf_color = get_confidence_label(
                fake_prob if label == "FAKE" else real_prob
            )
            cred_score = get_credibility_score(user_text, real_prob)

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # ── Top result row ──
        r1, r2, r3, r4 = st.columns(4)

        verdict_html = (
            f'<div class="result-{label.lower()}">{label}</div>'
            if label in ("REAL", "FAKE")
            else f'<div class="result-uncertain">{label}</div>'
        )

        with r1:
            st.markdown("<div class='metric-label'>VERDICT</div>", unsafe_allow_html=True)
            st.markdown(verdict_html, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{fake_prob*100:.1f}%</div>
                <div class="metric-label">Fake Probability</div>
            </div>""", unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{real_prob*100:.1f}%</div>
                <div class="metric-label">Real Probability</div>
            </div>""", unsafe_allow_html=True)

        with r4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{cred_score}/10</div>
                <div class="metric-label">Credibility Score</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts row ──
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='section-title'>Confidence Breakdown</div>", unsafe_allow_html=True)
            fig_bar = go.Figure(go.Bar(
                x=["REAL", "FAKE"],
                y=[real_prob * 100, fake_prob * 100],
                marker_color=["#00ff87", "#ff2d78"],
                text=[f"{real_prob*100:.1f}%", f"{fake_prob*100:.1f}%"],
                textposition="outside",
                textfont=dict(color="#e8e8f0", size=13),
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8e8f0",
                yaxis=dict(range=[0, 115], showgrid=True, gridcolor="#1e1e35"),
                xaxis=dict(showgrid=False),
                showlegend=False,
                margin=dict(t=20, b=10),
                height=260,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            st.markdown("<div class='section-title'>Probability Distribution</div>", unsafe_allow_html=True)
            fig_pie = go.Figure(go.Pie(
                labels=["REAL", "FAKE"],
                values=[real_prob, fake_prob],
                hole=0.55,
                marker=dict(colors=["#00ff87", "#ff2d78"],
                            line=dict(color="#0a0a0f", width=3)),
                textfont=dict(color="#e8e8f0"),
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e8e8f0",
                showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=20, b=10),
                height=260,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Gauge ──
        st.markdown("<div class='section-title'>Fake News Confidence Gauge</div>", unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fake_prob * 100,
            number={"suffix": "%", "font": {"color": "#e8e8f0", "size": 28}},
            delta={"reference": 50, "increasing": {"color": "#ff2d78"}, "decreasing": {"color": "#00ff87"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#555577"},
                "bar": {"color": "#7b2fff"},
                "steps": [
                    {"range": [0, 30], "color": "rgba(0,255,135,0.15)"},
                    {"range": [30, 60], "color": "rgba(255,215,0,0.15)"},
                    {"range": [60, 100], "color": "rgba(255,45,120,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "#1e1e35",
            },
            title={"text": "Fake Probability %", "font": {"color": "#8888aa"}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e8e8f0",
            height=230,
            margin=dict(t=30, b=0),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Pattern Analysis ──
        if show_patterns:
            st.markdown("---")
            st.markdown("<div class='section-title'>🔍 Suspicious Pattern Analysis</div>", unsafe_allow_html=True)

            found_patterns = []
            text_lower = user_text.lower()
            for category, patterns in SUSPICIOUS_PATTERNS.items():
                hits = [p for p in patterns if re.search(p, text_lower)]
                if hits:
                    found_patterns.append((category, hits))

            if found_patterns:
                p1, p2 = st.columns(2)
                cols = [p1, p2]
                for i, (cat, hits) in enumerate(found_patterns):
                    with cols[i % 2]:
                        st.markdown(f"""
                        <div class="warn-box">
                        <b>⚠️ {cat}</b><br>
                        Matched: {', '.join(f'<code>{h}</code>' for h in hits[:3])}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">✅ No common sensationalism or clickbait patterns detected.</div>
                """, unsafe_allow_html=True)

        # ── Token Analysis ──
        if show_tokens:
            st.markdown("---")
            st.markdown("<div class='section-title'>🔤 Tokenization Preview</div>", unsafe_allow_html=True)
            tokens = tokenizer.tokenize(user_text[:400])
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_df_data = {"Token": tokens[:40], "ID": token_ids[:40]}
            st.dataframe(token_df_data, use_container_width=True, height=200)
            st.caption(f"Total tokens (truncated to {max_len}): **{len(tokenizer.tokenize(user_text))}**")

        # ── Interpretation ──
        st.markdown("---")
        st.markdown("<div class='section-title'>💡 Interpretation</div>", unsafe_allow_html=True)
        if label == "FAKE" and fake_prob > 0.7:
            msg = f"🔴 **High likelihood of misinformation.** BERT assigned {fake_prob*100:.1f}% fake probability. The text shows strong linguistic signals associated with fabricated or misleading content."
            box = "warn-box"
        elif label == "FAKE":
            msg = f"🟠 **Possibly misleading.** Fake probability is {fake_prob*100:.1f}%. The model leans toward fake but with moderate confidence — verify with trusted sources."
            box = "warn-box"
        elif label == "REAL" and real_prob > 0.7:
            msg = f"🟢 **Likely credible.** BERT assigned {real_prob*100:.1f}% real probability. The text exhibits patterns consistent with factual reporting."
            box = "info-box"
        else:
            msg = f"🟡 **Inconclusive.** Confidence is low ({max(real_prob, fake_prob)*100:.1f}%). This could be ambiguous phrasing, opinion content, or satire. Cross-check with multiple sources."
            box = "info-box"

        st.markdown(f'<div class="{box}">{msg}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("⚠️ **Disclaimer:** This is a demonstration using a base (non-fine-tuned) BERT model. For production use, fine-tune on a labeled fake news dataset like LIAR or FakeNewsNet.")
