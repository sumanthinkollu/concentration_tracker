# 🎯 Concentration Tracker (Real-Time)

A real-time concentration monitoring system using **MediaPipe** and **OpenCV** — built with a standard webcam. This project tracks **eye blinks**, **gaze direction**, and **head orientation** to measure attentiveness and live concentration percentage.

> Developed by **Sumanth Inkollu** 

---

## 🔍 Features

- 👀 **Eye Blink Detection** using Eye Aspect Ratio (EAR)
- 🎯 **Gaze Detection** to detect if user is looking at screen
- 📐 **Head Pose Estimation** based on nose landmark distance
- 🧠 **Concentration Score** (0–100%) with visual bar
- 🚨 **Distraction Counter** and real-time alerts
- 🖥️ **Full-sized webcam display** (1280x720 resolution)
- ⚡ **FPS counter** for performance monitoring

---

## 🖼 Sample UI

- ✅ ACTIVE / DISTRACTED / NO FACE indicator (top left)
- 📊 Live **concentration bar**
- 🔢 **Distraction counter**
- ⚡ **FPS display**

---

## 📦 Tech Stack

- Python 3.x
- OpenCV
- MediaPipe
- NumPy

---

## 🚀 Setup & Run

### 🔧 1. Clone the repository
git clone https://github.com/<your-username>/concentration_tracker.git
cd concentration_tracker

---

### 🐍 2. Create & activate a virtual environment
python -m venv venv
# For CMD:
venv\Scripts\activate
# For PowerShell:
.\venv\Scripts\Activate.ps1

---

📥 3. Install dependencies
pip install -r requirements.txt

---

▶️ 4. Run the tracker
python concentration_tracker.py
