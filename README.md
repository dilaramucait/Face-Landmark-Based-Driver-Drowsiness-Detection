# 👁️ Face Landmark-Based Driver Drowsiness Detection

## 📌 Overview
This project is developed for the **Digital Image Processing course**. It is a real-time computer vision system that detects driver drowsiness using facial landmarks. The project uses **MediaPipe Face Mesh** and **OpenCV** to track eye and mouth movements and calculate **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** for fatigue detection.

When signs of drowsiness or yawning are detected, the system triggers visual warnings and an audible alarm to improve driver safety.

---

## 🚀 Features
- Real-time face landmark detection using MediaPipe Face Mesh
- 468 facial landmark tracking
- Eye Aspect Ratio (EAR) based blink and eye-closure detection
- Mouth Aspect Ratio (MAR) based yawn detection
- Drowsiness state monitoring
- Audible alert system (system beep alarm)
- Real-time dashboard UI overlay

---

## 🧠 How It Works
1. Captures live video from webcam
2. Detects facial landmarks using MediaPipe Face Mesh
3. Extracts key points for eyes and mouth
4. Computes:
   - Eye Aspect Ratio (EAR)
   - Mouth Aspect Ratio (MAR)
5. Determines:
   - Drowsiness based on prolonged eye closure
   - Yawning based on mouth opening
6. Triggers alerts when thresholds are exceeded

---

## 🛠️ Technologies Used
- Python
- OpenCV
- MediaPipe
- NumPy

---

## 📊 Key Concepts
- Face Landmark Detection (468-point facial mesh)
- Eye Aspect Ratio (EAR) for eye closure detection
- Mouth Aspect Ratio (MAR) for yawning detection
- Real-time video stream processing
