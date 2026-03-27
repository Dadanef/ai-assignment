
---

 📄 TECHNICAL REPORT
---
# Facial Emotion Recognition Project Report

## 1. Introduction

This project implements a complete facial emotion recognition system. The system includes preprocessing, face detection, real-time webcam interaction, emotion classification, visualization, and batch accuracy evaluation.

---

## 2. Methods

### 2.1 Preprocessing

Each image is processed using the following steps:
- Resize to a fixed resolution
- Convert to grayscale
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)

This improves contrast and stability for face detection.

---

### 2.2 Face Detection and Tuning

Face detection is performed using Haar Cascade. Multiple parameter combinations were tested:

- Different scaleFactor values
- Different minNeighbors values
- Different minSize values

Results were recorded in a CSV table to compare performance.

---

### 2.3 Real-time Webcam

A real-time webcam system was implemented with:
- FPS counter
- Frame skipping (every N frames)
- Screenshot functionality (key: s)
- Quit functionality (key: q)

This improves performance and usability.

---

### 2.4 Emotion Classification

Emotion classification is performed using the FER library. For each detected face, the system returns:

- All 7 emotion scores:
  angry, disgust, fear, happy, sad, surprise, neutral
- Top emotion
- Confidence score

---

### 2.5 Visualization

The system provides rich visualization:
- Color-coded bounding boxes
- Emotion label with confidence percentage
- Bar chart for each face
- Summary figure combining all faces

---

### 2.6 Batch Accuracy Analysis

A labeled dataset with 20+ images was used. The system computes:

- Overall accuracy
- Per-emotion accuracy
- Confusion matrix

All results are saved to CSV and image files.

---

## 3. Results

- Preprocessing improved image clarity
- Parameter tuning showed that moderate scaleFactor and minNeighbors values perform best
- Emotion classification worked reliably on clear frontal faces
- Batch evaluation produced accuracy metrics and confusion matrix

---

## 4. Error Analysis

Several limitations were observed:

- Low lighting reduces detection accuracy
- Small or blurry faces are harder to detect
- Side faces are less accurate than frontal faces
- Emotions such as sad and neutral are often confused
- Fear and surprise can overlap

---

## 5. Conclusion

The system successfully integrates preprocessing, face detection, emotion classification, visualization, and evaluation into a single pipeline.

The results demonstrate that the system works effectively under normal conditions, with some limitations in challenging scenarios.

Future improvements may include:
- Using more advanced face detectors
- Improving model robustness
- Expanding dataset size