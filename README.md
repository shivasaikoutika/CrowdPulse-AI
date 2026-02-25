# 🚨 CrowdPulse AI
### Real-time Crowd Density & Sentiment Monitoring for Safer Public Spaces

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b)
![AMD](https://img.shields.io/badge/Optimized-AMD%20Ryzen%20AI-ED1C24)
![License](https://img.shields.io/badge/License-MIT-green)

> Built for AMD Slingshot Hackathon 2026 | Theme: AI for Smart Cities

---

## 🌟 Overview

CrowdPulse AI is a real-time intelligent crowd monitoring platform 
that combines deep learning-based crowd density estimation with 
facial sentiment & distress detection to prevent crowd-related 
disasters in public spaces.

It transforms passive CCTV infrastructure into an active, 
intelligent early-warning system — monitoring crowd density, 
emotional state, and risk levels simultaneously, and alerting 
authorities before situations escalate.

---

## 🚨 Problem Statement

Every year, crowd crushes and stampedes claim hundreds of lives 
in India — at railway stations, religious gatherings, and stadiums. 
The 2024 Hathras stampede claimed 120+ lives. Current systems are 
entirely reactive. CrowdPulse AI makes them proactive.

---

## ✨ Features

### ✅ Currently Implemented
- 📷 **Real-time Crowd Counting** — CSRNet-based density estimation
- 😟 **Sentiment & Distress Detection** — FER+/DeepFace on detected faces
- 🧠 **CrowdRisk Score Engine** — Unified 0-100 risk index (🟢🟡🔴)
- 📊 **Live Streamlit Dashboard** — Heatmaps, trends, per-zone scores
- 📱 **Instant Authority Alerts** — WhatsApp/SMS via Twilio
- 📄 **Post-Event PDF Reports** — Auto-generated safety summaries

### 🔮 Upcoming Features
- ⚡ **Predictive Surge Engine** — LSTM-based 5-10 min crowd surge forecasting
- 🌊 **Flow Anomaly Detection** — Optical flow for stampede pattern recognition  
- 🗺️ **Dynamic Exit Routing** — Real-time safest evacuation path suggestions
- 🔊 **Audio Panic Detection** — Multi-modal distress sensing

---

## 🏗️ Architecture
```
📷 Camera Feed (CCTV / Webcam)
        ↓
┌─────────────────────────────────┐
│       CrowdPulse AI Engine      │
│  ┌──────────────┐ ┌───────────┐ │
│  │ Crowd Count  │ │ Sentiment │ │
│  │  (CSRNet)    │ │ (FER+)    │ │
│  └──────────────┘ └───────────┘ │
│       ↓                ↓        │
│    🧠 CrowdRisk Score Engine    │
└─────────────────────────────────┘
        ↓                 ↓
  📊 Dashboard      📱 Alerts + 📄 Reports
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Crowd Counting | CSRNet (PyTorch) |
| Sentiment Detection | DeepFace / FER+ |
| Video Processing | OpenCV |
| Risk Engine | Custom weighted scoring |
| Dashboard | Streamlit |
| Backend | FastAPI |
| Alerts | Twilio (WhatsApp/SMS) |
| Reports | ReportLab |
| Edge Optimization | ONNX Runtime + AMD ROCm |

---

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CrowdPulse-AI.git
cd CrowdPulse-AI

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard/app.py
```

---

## 📊 Demo

![Dashboard Screenshot](assets/demo_screenshot.png)

> 🎥 Demo Video: [Watch here](#)

---

## 👥 Team

| Name | Role |
|------|------|
| K.Shivasai | ML Engineer |

---

## 🏆 Hackathon

Built for **AMD Slingshot 2026** by Hack2Skill  
Theme: **AI for Smart Cities**  
Platform: [amdslingshot.in](https://amdslingshot.in)

---

## 📄 License
MIT License — feel free to use and build on this!
