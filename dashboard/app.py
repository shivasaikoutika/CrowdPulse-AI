# CrowdPulse AI — Live Dashboard
# Real-time Crowd Density & Sentiment Monitoring

import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.risk_engine.crowd_risk_score import calculate_risk_score, get_zone_summary
from src.sentiment_analysis.fer_inference import analyze_frame_sentiment, get_crowd_mood
from src.crowd_counting.csrnet_inference import load_model, estimate_crowd, get_zone_density

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CrowdPulse AI",
    page_icon="🚨",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .risk-safe     { color: #00ff88; font-size: 2rem; font-weight: bold; }
    .risk-watch    { color: #ffcc00; font-size: 2rem; font-weight: bold; }
    .risk-critical { color: #ff4444; font-size: 2rem; font-weight: bold; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .title-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
    }
    .subtitle-text {
        font-size: 1rem;
        color: #aaaaaa;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<p class="title-text">🚨 CrowdPulse AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Real-time Crowd Density & Sentiment Monitoring for Safer Public Spaces</p>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/AMD-Slingshot%202026-ED1C24?style=for-the-badge")
    st.markdown("## ⚙️ Controls")

    source = st.radio("📷 Video Source", ["Webcam", "Upload Video", "Demo Mode"])
    density_weight  = st.slider("Density Weight",  0.0, 1.0, 0.6, 0.1)
    sentiment_weight = st.slider("Sentiment Weight", 0.0, 1.0, 0.4, 0.1)

    st.markdown("## 🚨 Alert Thresholds")
    watch_threshold    = st.slider("Watch Threshold",    0, 100, 40)
    critical_threshold = st.slider("Critical Threshold", 0, 100, 70)

    st.markdown("## 📊 About")
    st.info("CrowdPulse AI combines crowd counting + sentiment analysis into a unified CrowdRisk Score to prevent public safety disasters.")

# ─────────────────────────────────────────────
# Session State for History
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []

# ─────────────────────────────────────────────
# Layout — 3 Columns for Metrics
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

crowd_placeholder     = col1.empty()
sentiment_placeholder = col2.empty()
risk_placeholder      = col3.empty()
mood_placeholder      = col4.empty()

# ─────────────────────────────────────────────
# Layout — Video + Heatmap
# ─────────────────────────────────────────────
st.markdown("### 📷 Live Feed & Density Heatmap")
vid_col, heat_col = st.columns(2)
video_placeholder   = vid_col.empty()
heatmap_placeholder = heat_col.empty()

# ─────────────────────────────────────────────
# Layout — Zone Analysis + Trend Chart
# ─────────────────────────────────────────────
st.markdown("### 📊 Zone Analysis & Risk Trend")
zone_col, trend_col = st.columns(2)
zone_placeholder  = zone_col.empty()
trend_placeholder = trend_col.empty()

# ─────────────────────────────────────────────
# Layout — Alert Log
# ─────────────────────────────────────────────
st.markdown("### 🚨 Alert Log")
alert_placeholder = st.empty()

# ─────────────────────────────────────────────
# Demo Mode — Simulated Data
# ─────────────────────────────────────────────
def get_demo_data(t):
    """Generate realistic simulated data for demo mode."""
    crowd_count   = int(150 + 80 * np.sin(t / 10) + np.random.randint(-10, 10))
    density_score = min((crowd_count / 500) * 100, 100)
    sentiment_score = max(0, min(100, 20 + 30 * np.sin(t / 8) + np.random.randint(-5, 5)))
    return crowd_count, round(density_score, 2), round(sentiment_score, 2)


def generate_demo_heatmap(crowd_count):
    """Generate a fake density heatmap for demo."""
    heatmap = np.zeros((480, 640), dtype=np.float32)
    num_clusters = max(1, crowd_count // 30)
    for _ in range(num_clusters):
        cx = np.random.randint(50, 590)
        cy = np.random.randint(50, 430)
        intensity = np.random.uniform(0.3, 1.0)
        for y in range(max(0, cy - 60), min(480, cy + 60)):
            for x in range(max(0, cx - 60), min(640, cx + 60)):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                heatmap[y, x] += intensity * np.exp(-dist / 30)
    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8) if heatmap.max() > 0 else heatmap.astype(np.uint8)
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


# ─────────────────────────────────────────────
# Risk Color Helper
# ─────────────────────────────────────────────
def risk_color_class(level):
    if level == "SAFE":
        return "risk-safe"
    elif level == "WATCH":
        return "risk-watch"
    return "risk-critical"


# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────
start = st.button("▶️ Start Monitoring", type="primary", use_container_width=True)
stop  = st.button("⏹️ Stop",             use_container_width=True)

if start:
    t = 0
    while not stop:
        t += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        # ── Get Data ──
        if source == "Demo Mode":
            crowd_count, density_score, sentiment_score = get_demo_data(t)
            heatmap_img = generate_demo_heatmap(crowd_count)
            demo_frame  = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(demo_frame, f"DEMO MODE", (220, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
            cv2.putText(demo_frame, f"People Detected: {crowd_count}", (180, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            annotated = demo_frame
            mood = "CALM" if sentiment_score < 30 else "TENSE" if sentiment_score < 60 else "PANIC"

        # ── Risk Score ──
        weights = {"density": density_weight, "sentiment": sentiment_weight}
        risk_score, risk_level, _ = calculate_risk_score(
            density_score, sentiment_score, weights
        )

        # ── Update History ──
        st.session_state.history.append({
            "time": timestamp,
            "crowd": crowd_count,
            "density": density_score,
            "sentiment": sentiment_score,
            "risk": risk_score
        })
        if len(st.session_state.history) > 60:
            st.session_state.history.pop(0)

        # ── Alert Log ──
        if risk_level == "CRITICAL":
            st.session_state.alert_log.append({
                "time": timestamp,
                "risk_score": risk_score,
                "level": risk_level,
                "crowd": crowd_count
            })

        # ── Metric Cards ──
        crowd_placeholder.markdown(f"""
            <div class="metric-card">
                <div style="color:#aaa;">👥 Crowd Count</div>
                <div style="color:#fff;font-size:2rem;font-weight:bold;">{crowd_count}</div>
                <div style="color:#aaa;font-size:0.8rem;">people detected</div>
            </div>""", unsafe_allow_html=True)

        sentiment_placeholder.markdown(f"""
            <div class="metric-card">
                <div style="color:#aaa;">😟 Distress Level</div>
                <div style="color:#fff;font-size:2rem;font-weight:bold;">{sentiment_score}%</div>
                <div style="color:#aaa;font-size:0.8rem;">sentiment score</div>
            </div>""", unsafe_allow_html=True)

        css_class = risk_color_class(risk_level)
        risk_placeholder.markdown(f"""
            <div class="metric-card">
                <div style="color:#aaa;">🧠 CrowdRisk Score</div>
                <div class="{css_class}">{risk_score}/100</div>
                <div style="color:#aaa;font-size:0.8rem;">{risk_level}</div>
            </div>""", unsafe_allow_html=True)

        mood_placeholder.markdown(f"""
            <div class="metric-card">
                <div style="color:#aaa;">🎭 Crowd Mood</div>
                <div style="color:#fff;font-size:2rem;font-weight:bold;">{mood}</div>
                <div style="color:#aaa;font-size:0.8rem;">overall sentiment</div>
            </div>""", unsafe_allow_html=True)

        # ── Video & Heatmap ──
        video_placeholder.image(
            annotated, channels="BGR",
            caption="Live Camera Feed", use_container_width=True
        )
        heatmap_placeholder.image(
            heatmap_img, channels="BGR",
            caption="Crowd Density Heatmap", use_container_width=True
        )

        # ── Zone Bar Chart ──
        with zone_placeholder.container():
            zones_data = {
                "Zone 1": np.random.randint(20, 80),
                "Zone 2": np.random.randint(20, 80),
                "Zone 3": np.random.randint(20, 80),
                "Zone 4": np.random.randint(20, 80),
            }
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor("#1e2130")
            ax.set_facecolor("#1e2130")
            colors = ["#00ff88" if v < 40 else "#ffcc00" if v < 70 else "#ff4444"
                      for v in zones_data.values()]
            ax.bar(zones_data.keys(), zones_data.values(), color=colors)
            ax.set_ylabel("Risk Score", color="white")
            ax.set_title("Zone-wise Risk", color="white")
            ax.tick_params(colors="white")
            ax.axhline(y=watch_threshold,    color="#ffcc00", linestyle="--", alpha=0.7, label="Watch")
            ax.axhline(y=critical_threshold, color="#ff4444", linestyle="--", alpha=0.7, label="Critical")
            ax.legend(facecolor="#1e2130", labelcolor="white")
            st.pyplot(fig)
            plt.close()

        # ── Trend Chart ──
        with trend_placeholder.container():
            if len(st.session_state.history) > 1:
                df = pd.DataFrame(st.session_state.history)
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                fig2.patch.set_facecolor("#1e2130")
                ax2.set_facecolor("#1e2130")
                ax2.plot(df["time"], df["risk"],      color="#ff4444", label="Risk Score")
                ax2.plot(df["time"], df["sentiment"], color="#ffcc00", label="Sentiment")
                ax2.plot(df["time"], df["density"],   color="#00aaff", label="Density")
                ax2.set_title("Risk Trend Over Time", color="white")
                ax2.tick_params(colors="white", labelsize=6)
                ax2.legend(facecolor="#1e2130", labelcolor="white", fontsize=7)
                st.pyplot(fig2)
                plt.close()

        # ── Alert Log Table ──
        if st.session_state.alert_log:
            alert_df = pd.DataFrame(st.session_state.alert_log).tail(5)
            alert_placeholder.dataframe(
                alert_df, use_container_width=True, hide_index=True
            )
        else:
            alert_placeholder.success("✅ No critical alerts — all zones are safe!")

        time.sleep(1)
