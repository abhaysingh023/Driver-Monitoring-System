# Real-Time Driver Pose & Distraction Detection System

A lightweight Driver Monitoring System (DMS) built with Python, OpenCV, and MediaPipe to detect poor posture (slouching/leaning), yawning (MAR), and extract eye regions for drowsiness analysis[cite: 1, 2].

## 🚀 Features
- **Posture Tracking (MediaPipe Pose):** Detects sideways leaning and forward neck slouching[cite: 1, 2].
- **Yawn Detection (MediaPipe Face Mesh):** Computes Mouth Aspect Ratio (MAR) to trigger yawning alerts[cite: 1, 2].
- **Eye Region Extraction:** Crops left/right eye bounding boxes in real time for LightCNN open/closed classification[cite: 1, 2].
- **In-Hand Object Detection *(Upcoming)*:** Wrist-gated YOLOv11n to detect phones, cups, and bottles[cite: 2, 3].

## 📂 Project Structure
```text
Driver-Monitoring-System/
├── utils/
│   └── geometry.py      # Math for MAR, Eye cropping, and Pose metrics
├── main.py              # Main webcam loop
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

## 🛠️ How to Run

**Prerequisites:** Python **3.10 or 3.11** and a working webcam[cite: 1, 2].

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/abhaysingh023/Driver-Monitoring-System.git](https://github.com/abhaysingh023/Driver-Monitoring-System.git)
   cd Driver-Monitoring-System
   ```

2. **Create & activate virtual environment:**
   ```bash
   python -m venv .venv
   # Windows (CMD): .venv\Scripts\activate
   # Windows (PowerShell): .venv\Scripts\Activate.ps1
   # Mac/Linux: source .venv/bin/activate
   ```
   *(PowerShell script error fix: `Set-ExecutionPolicy Unrestricted -Scope Process`)*

3. **Install dependencies & run:**
   ```bash
   pip install -r requirements.txt
   python main.py
   ```
   Press **`q`** on the webcam window to quit.

## ⚙️ Threshold Tuning (`main.py`)
Adjust these values at the top of `main.py` based on your camera distance:
- `MAR_THRESHOLD = 0.50`: Increase (e.g., `0.60`) if talking triggers yawn alerts.
- `LEAN_THRESHOLD = 45`: Shoulder height difference (px)[cite: 2].
- `SLOUCH_THRESHOLD = 110`: Nose-to-shoulder vertical distance (px)[cite: 2]. Set this ~25px below your normal sitting `Neck` value shown on screen.

## 🔧 Quick Fixes
- **Black screen / Webcam error:** Close other camera apps or change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `main.py`.
- **`ModuleNotFoundError`:** Ensure `(.venv)` is active in your terminal before running `python main.py`.
