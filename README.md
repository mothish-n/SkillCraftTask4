# SkillCraftTask4
# ✋ Hand Gesture Recognition using CNN and OpenCV

This project builds a hand gesture recognition model that can classify different static hand gestures from images or real-time video input. It enables intuitive **human-computer interaction** and **gesture-based control systems** using computer vision and deep learning.

## 📌 Features

- Recognizes hand gestures like Thumbs Up 👍, Peace ✌️, Stop ✋, Fist ✊, etc.
- Real-time gesture prediction using webcam
- Clean UI in terminal or display window
- Preprocessing using OpenCV
- Classification using Convolutional Neural Networks (CNN)

## 🧠 Model Overview

- Model: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Input: 64x64 grayscale images of hand gestures
- Output: Class label (gesture name)

## 🗂 Dataset

We used the [Kaggle Hand Gesture Dataset](https://www.kaggle.com/gti-upm/leapgestrecog)  
You can replace this with any labeled gesture dataset (ASL, custom images, etc.)

## 🚀 How to Run

1. Clone the repo
2. Install dependencies  
3. Train the model (`train_model.py`)  
4. Run real-time prediction (`predict_video.py`)

```bash
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition
pip install -r requirements.txt
python train_model.py
python predict_video.py
