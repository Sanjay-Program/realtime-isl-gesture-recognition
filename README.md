# 🇮🇳 Real-Time Indian Sign Language Gesture Recognition

This project is a real-time hand gesture recognition system for Indian Sign Language (ISL) using deep learning and computer vision techniques. It detects 36 different classes (A–Z and 0–9) using a webcam and provides accurate predictions with background removal using GrabCut.

---

## 🚀 Features

- ✋ Detects hand gestures for A–Z and 0–9
- 🎥 Real-time webcam-based recognition
- 🧠 Deep learning with MobileNetV2
- 🎨 Background removal using GrabCut
- 📊 Accuracy & loss graphs, confusion matrix, and misclassified samples
- 🧼 Clean UI with bounding box and confidence display

---

## 🧠 Model Architecture

- **Base Model:** MobileNetV2 (`imagenet` weights)
- **Custom Layers:** GlobalAveragePooling → Dense(512, ReLU) → Dropout(0.5) → Dense(36, Softmax)
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam (lr=0.0001)
- **Training Strategy:**  
  - Image Augmentation  
  - EarlyStopping  
  - ReduceLROnPlateau  

---

---
## dataset
-data/
- ├── train/
- │   ├── A/
- │   ├── B/
- │   └── ...
- ├── validation/
- │   ├── A/
- │   ├── B/
- │   └── ...

## 🧪 Usage

### 🔧 Setup

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/realtime-isl-gesture-recognition.git
cd realtime-isl-gesture-recognition 
