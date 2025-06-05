from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from datetime import datetime
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from utils.predict import predict_image,extract_eye_region, map_coordinates_to_label
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
app = Flask(__name__)
from keras.losses import MeanSquaredError

CORS(app)  # Allow requests from mobile app

DATASET_DIR = 'dataset'
LOG_FILE = 'upload_log.txt'

# Load the trained model
MODEL_PATH = os.getcwd()+'/saved_models/Model_1_Simple_CNN_200_50_20250604_160555.h5'  # Update with your best model path
MODEL = tf.keras.models.load_model(MODEL_PATH, custom_objects={'mse': MeanSquaredError()},compile=False)


# Grid configuration
GRID_CONFIG = {
    'rows': 9,
    'cols': 9,
}
IMAGE_SIZE = (200, 50)  # Update this to match your model's input size

# Mediapipe setup for face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158, 157, 173, 246]
RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 384, 398, 466]

def extract_eye_region(image, landmarks, eye_landmarks, padding=30):
    h, w, _ = image.shape
    eye_points = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_landmarks]
    x_coords, y_coords = zip(*eye_points)
    x_min = max(min(x_coords) - padding, 0)
    x_max = min(max(x_coords) + padding, w)
    y_min = max(min(y_coords) - padding, 0)
    y_max = min(max(y_coords) + padding, h)
    return image[y_min:y_max, x_min:x_max] if (x_max - x_min > 0 and y_max - y_min > 0) else None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the base64-encoded image from the request
        data = request.get_json()
        image_data = data.get('image')
        os.makedirs('temp', exist_ok=True)

        # Generate unique filename
        filename = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.jpg'
        filepath = os.path.join('temp/', filename)

        # Decode and save the image
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_data))

        # Decode the base64 image
        result = predict_image(filepath,MODEL)
        print(result)
        label = result['predicted_label']
        
        # Return the prediction
        return jsonify({
            'predictedIndex': label
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.get_json()
        image_data = data.get('image')
        x = data.get('x')
        y = data.get('y')
        index = data.get('index')
        if not image_data or not x or not y:
            return jsonify({'status': 'fail', 'message': 'Missing image or label'}), 400

        # Prepare directory for label
        save_dir = os.path.join(DATASET_DIR, 'images')
        os.makedirs(save_dir, exist_ok=True)

        # Generate unique filename
        filename = str(index)+str(x)+'_'+str(y)+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.jpg'
        filepath = os.path.join(save_dir, filename)

        # Decode and save the image
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_data))

        # Log the upload
        log_entry = f"{datetime.now()} - coordinates: {x}, {y}, File: {filepath}\n"
        with open(LOG_FILE, 'a') as log:
            log.write(log_entry)

        print(f"Saved image for coordinates '{x},{y}' to {filepath}")
        return jsonify({'status': 'success', 'message': f'Saved to {filepath}'})

    except Exception as e:
        error_msg = f"Error processing upload: {str(e)}"
        with open(LOG_FILE, 'a') as log:
            log.write(f"{datetime.now()} - {error_msg}\n")
        return jsonify({'status': 'error', 'message': error_msg}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000, debug=True)