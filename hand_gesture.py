import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === Configuration ===
DATASET_PATH = r"C:\Users\mothi\Downloads\archive (4)\dataset\train"
IMG_SIZE = 64
CATEGORIES = [
    '01_palm',
    '02_l',
    '03_fist',
    '04_fist_moved',
    '05_thumb',
    '06_index',
    '07_ok',
    '08_palm_moved',
    '09_c',
    '10_down'
]

def load_data():
    data = []
    for idx, category in enumerate(CATEGORIES):
        path = os.path.join(DATASET_PATH, category)
        for img in tqdm(os.listdir(path), desc=category):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized, idx])
            except Exception:
                continue
    return data

# Load and preprocess data
data = load_data()
X = []
y = []
for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = to_categorical(y, num_classes=len(CATEGORIES))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(CATEGORIES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save
model.save("gesture_model.h5")
print("✅ Model saved as gesture_model.h5")
