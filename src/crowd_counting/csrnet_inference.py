# CrowdPulse AI — Crowd Counting Module
# Uses CSRNet architecture for real-time crowd density estimation

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class CSRNet(nn.Module):
    """Simplified CSRNet for crowd density estimation."""

    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, dilation=2, padding=2),  nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


# Image transform pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(weights_path=None, device="cpu"):
    """Load CSRNet model with optional pretrained weights."""
    model = CSRNet().to(device)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[CrowdCount] Loaded weights from {weights_path}")
    model.eval()
    return model


def estimate_crowd(frame, model, device="cpu"):
    """
    Estimate crowd count from a single video frame.

    Args:
        frame: numpy array (BGR from OpenCV)
        model: loaded CSRNet model
        device: 'cpu' or 'cuda'

    Returns:
        count: int — estimated number of people
        density_map: numpy array — heatmap of crowd density
        density_score: float (0-100) — normalized density score
    """
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).resize((640, 480))

    # Preprocess
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        density_map = model(input_tensor)

    density_map = density_map.squeeze().cpu().numpy()
    count = int(density_map.sum())

    # Normalize to 0-100 score (assuming max crowd = 500 people)
    density_score = min((count / 500) * 100, 100)

    return count, density_map, round(density_score, 2)


def get_zone_density(frame, model, zones=4, device="cpu"):
    """
    Split frame into zones and estimate density per zone.

    Args:
        frame: numpy array
        model: CSRNet model
        zones: number of horizontal zones to split into

    Returns:
        zone_counts: list of (zone_id, count, density_score)
    """
    h, w = frame.shape[:2]
    zone_width = w // zones
    zone_counts = []

    for i in range(zones):
        zone_frame = frame[:, i * zone_width:(i + 1) * zone_width]
        count, _, score = estimate_crowd(zone_frame, model, device)
        zone_counts.append({
            "zone": f"Zone {i + 1}",
            "count": count,
            "density_score": score
        })

    return zone_counts
```

**Commit new file** ✅

---

## 🚀 Step 5 — Sentiment Analysis Module

**"Add file"** → **"Create new file"**

Filename:
```
src/sentiment_analysis/fer_inference.py
