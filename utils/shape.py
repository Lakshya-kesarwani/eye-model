import os
# path = 'dataset/images'
# count =0
# for filename in os.listdir(path):
#     count +=1
# print(f"Total images in dataset: {count}")
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import mediapipe as mp
import base64
from flask import jsonify
import cv2

def extract_eye_region(image, landmarks, eye_landmarks, padding=30):
    h, w, _ = image.shape
    eye_points = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_landmarks]
    x_coords, y_coords = zip(*eye_points)
    x_min = max(min(x_coords) - padding, 0)
    x_max = min(max(x_coords) + padding, w)
    y_min = max(min(y_coords) - padding, 0)
    y_max = min(max(y_coords) + padding, h)
    return image[y_min:y_max, x_min:x_max] if (x_max - x_min > 0 and y_max - y_min > 0) else None

def preprocess_image(image_path, target_size=(200, 50)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array
MODEL_PATH = 'saved_models/best_model.h5'  # Update with your best model path
model = tf.keras.models.load_model(MODEL_PATH)

# Grid configuration
GRID_CONFIG = {
    'rows': 9,
    'cols': 9,
}
IMAGE_SIZE = (200, 50)  # Update this to match your model's input size
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158, 157, 173, 246]
RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 384, 398, 466]

path = os.listdir('dataset/images/')
image_b64 = load_img(path[np.random.randint(0, len(path))])
if not image_b64:
    print(jsonify({'error': 'No image provided'}), 400)

# Decode the base64 image
image_data = base64.b64decode(image_b64)
np_image = np.frombuffer(image_data, np.uint8)
img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

# Convert image to RGB for Mediapipe
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = face_mesh.process(image_rgb)
cv2.imshow('image', img)

# Extract landmarks
landmarks = results.multi_face_landmarks[0].landmark

# Extract left and right eye regions
left_eye = extract_eye_region(img, landmarks, LEFT_EYE_LANDMARKS)
right_eye = extract_eye_region(img, landmarks, RIGHT_EYE_LANDMARKS)

if left_eye is None or right_eye is None:
    print(jsonify({'error': 'Failed to extract eye regions'}), 400)

# Resize both eyes to the same height and concatenate them
target_height = min(left_eye.shape[0], right_eye.shape[0])
left_eye = cv2.resize(left_eye, (left_eye.shape[1], target_height))
right_eye = cv2.resize(right_eye, (right_eye.shape[1], target_height))
combined_eyes = cv2.hconcat([left_eye, right_eye])

# Preprocess the combined eye image (resize and normalize)
img_resized = cv2.resize(combined_eyes, IMAGE_SIZE)
img_normalized = img_resized / 255.0
img_input = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

# Predict the coordinates
predicted_coords = model.predict(img_input)[0]  # Get the first prediction
x_coord, y_coord = predicted_coords
print(f"Predicted coordinates: ({x_coord}, {y_coord})")

# Map the coordinates to a label (0 to 80)
row = int((y_coord / 792) * GRID_CONFIG['rows'])  # Normalize y to grid row
col = int((x_coord / 356) * GRID_CONFIG['cols'])  # Normalize x to grid column

# Ensure row and col are within bounds
row = min(max(row, 0), GRID_CONFIG['rows'] - 1)
col = min(max(col, 0), GRID_CONFIG['cols'] - 1)
print(f"Row: {row}, Column: {col}")
label = row * GRID_CONFIG['cols'] + col  # Calculate the label (0 to 80)
print(f"Predicted label: {label}")