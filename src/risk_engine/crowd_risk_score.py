# CrowdPulse AI — Risk Engine
# Combines crowd density + sentiment into unified CrowdRisk Score

def calculate_risk_score(density_score, sentiment_score, weights=None):
    """
    Calculate unified CrowdRisk Score (0-100)
    
    Args:
        density_score: float (0-100) — crowd density level
        sentiment_score: float (0-100) — distress/negative sentiment level
        weights: dict with 'density' and 'sentiment' keys (default 60/40)
    
    Returns:
        risk_score: float (0-100)
        risk_level: str — 'SAFE', 'WATCH', or 'CRITICAL'
        color: str — 'green', 'yellow', or 'red'
    """
    if weights is None:
        weights = {"density": 0.6, "sentiment": 0.4}

    risk_score = (
        weights["density"] * density_score +
        weights["sentiment"] * sentiment_score
    )
    risk_score = round(min(max(risk_score, 0), 100), 2)

    if risk_score < 40:
        return risk_score, "SAFE", "green"
    elif risk_score < 70:
        return risk_score, "WATCH", "yellow"
    else:
        return risk_score, "CRITICAL", "red"


def get_zone_summary(zones: dict):
    """
    Summarize risk across multiple zones.
    
    Args:
        zones: dict of {zone_name: {"density": float, "sentiment": float}}
    
    Returns:
        summary: list of zone risk results
    """
    summary = []
    for zone, values in zones.items():
        score, level, color = calculate_risk_score(
            values["density"], values["sentiment"]
        )
        summary.append({
            "zone": zone,
            "risk_score": score,
            "risk_level": level,
            "color": color
        })
    return summary
```

**Commit new file** ✅

---

## 🚀 Step 2 — Create Alerts Module

**"Add file"** → **"Create new file"**

Filename:
```
src/alerts/twilio_alert.py
# CrowdPulse AI — Alert System
# Sends WhatsApp/SMS alerts via Twilio when risk is CRITICAL

from twilio.rest import Client
import os

def send_alert(zone, risk_score, risk_level, snapshot_url=None):
    """
    Send WhatsApp/SMS alert to authority when risk is CRITICAL.
    
    Args:
        zone: str — name of the affected zone
        risk_score: float — current CrowdRisk Score
        risk_level: str — SAFE / WATCH / CRITICAL
        snapshot_url: str — optional camera snapshot URL
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token  = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM")
    to_number   = os.getenv("TWILIO_TO")

    if not all([account_sid, auth_token, from_number, to_number]):
        print("[ALERT] Twilio credentials not set. Skipping alert.")
        return

    client = Client(account_sid, auth_token)

    message_body = (
        f"🚨 CROWDPULSE AI ALERT 🚨\n"
        f"Zone: {zone}\n"
        f"Risk Level: {risk_level}\n"
        f"CrowdRisk Score: {risk_score}/100\n"
        f"Action Required: Deploy security personnel immediately.\n"
        f"{'Snapshot: ' + snapshot_url if snapshot_url else ''}"
    )

    message = client.messages.create(
        body=message_body,
        from_=from_number,
        to=to_number
    )
    print(f"[ALERT] Sent successfully. SID: {message.sid}")
```

**Commit new file** ✅

---

## 🚀 Step 3 — Create a .env.example file

**"Add file"** → **"Create new file"**

Filename:
```
.env.example
  # Twilio Credentials
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_FROM=whatsapp:+14155238886
TWILIO_TO=whatsapp:+91XXXXXXXXXX
