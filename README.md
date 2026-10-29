# Real-Time Driver Unwanted Pose & Distraction Detection System

A lightweight, real-time Driver Monitoring System (DMS) designed to detect driver drowsiness, yawning, poor posture (slouching/leaning), and hand-held distractions[cite: 1]. The model is developed locally in Python to be lightweight and mobile-deployment ready (via TFLite).

## 🚀 Key Features & Architecture

* **Posture & Slouch Tracking (MediaPipe Pose):** Tracks shoulder alignment and nose-to-shoulder vertical distance to detect lateral leaning and forward neck slouching in real time.
* **Yawn Detection (MediaPipe Face Mesh):** Calculates the Mouth Aspect Ratio (MAR) using facial landmarks (`13, 14, 78, 308`) to trigger yawning alerts[cite: 1, 2].
* **Eye Region Extraction (MediaPipe Face Mesh + LightCNN):** Automatically extracts and crops the left and right eye bounding boxes in real time to feed into a custom LightCNN classifier (trained on the MRL Eye dataset) for open/closed eye state detection[cite: 1, 2].
* **In-Hand Distraction Detection (Wrist-Gated YOLOv11n):** *(In Progress)* Monitors wrist coordinates (`15, 16`) and crops the hand region only when raised, detecting cell phones, cups, and bottles with minimal CPU overhead[cite: 2, 3].

---

## 📂 Project Structure

```text
code/
│
├── utils/
│   └── geometry.py        # Euclidean distance math for MAR, Eye ROI cropping, and Pose metrics
├── main.py                # Main webcam loop running MediaPipe Face Mesh & Pose
├── requirements.txt       # Required Python libraries
└── README.md              # Project documentation