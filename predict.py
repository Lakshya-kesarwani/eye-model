import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from keras.losses import MeanSquaredError

# Load the trained model
# MODEL_PATH = 'saved_models\Model_1_Simple_CNN_200_50_20250604_160555.h5'
# model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'mse': MeanSquaredError()})

# Grid configuration
GRID_CONFIG = {
    'rows': 9,
    'cols': 9,
}
IMAGE_SIZE = (200, 50)  # Update this to match your model's input size

# Path to the original dataset
DATASET_DIR = 'dataset/images'

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

def map_coordinates_to_label(x_coord, y_coord):
    # Map the coordinates to a label (0 to 80)
    row = int((y_coord / 792) * GRID_CONFIG['rows'])  # Normalize y to grid row
    col = int((x_coord / 356) * GRID_CONFIG['cols'])  # Normalize x to grid column

    # Ensure row and col are within bounds
    # row = min(max(row, 0), GRID_CONFIG['rows'] - 1)
    # col = min(max(col, 0), GRID_CONFIG['cols'] - 1)

    label = row * GRID_CONFIG['cols'] + col  # Calculate the label (0 to 80)
    return label

def test_on_original_dataset():
    files = os.listdir(DATASET_DIR)
    files = [files[np.random.randint(0,len(files))] for i in range(30) ]  # Take the first 30 images for testing

    results = []

    for file in files:
        file_path = os.path.join(DATASET_DIR, file)
        image = cv2.imread(file_path)
        if image is None:
            print(f"Skipping unreadable image: {file_path}")
            continue

        # Convert image to RGB for Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_mp = face_mesh.process(image_rgb)

        if not results_mp.multi_face_landmarks:
            print(f"No face detected in: {file_path}")
            continue

        # Extract landmarks
        landmarks = results_mp.multi_face_landmarks[0].landmark

        # Extract left and right eye regions
        left_eye = extract_eye_region(image, landmarks, LEFT_EYE_LANDMARKS)
        right_eye = extract_eye_region(image, landmarks, RIGHT_EYE_LANDMARKS)

        if left_eye is None or right_eye is None:
            print(f"Failed to extract eye regions for: {file_path}")
            continue

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
        print(f"Predicted coordinates for {file}: ({x_coord}, {y_coord})")
        x_coord = x_coord*354
        y_coord = y_coord*792

        # Map the predicted coordinates to a label
        predicted_label = map_coordinates_to_label(x_coord, y_coord)

        # Extract the actual label from the filename
        x_actual = float(file.split('_')[0])  # Assuming the filename starts with the label
        y_actual = float(file.split('_')[1])
        # Store the results
        results.append({
            'filename': file,
            'actual_label': {'x': x_actual, 'y': y_actual},
            'predicted_coordinates': {'x': x_coord, 'y': y_coord},
            'predicted_label': predicted_label
        })

    # Print the results
    for result in results:
        print(f"Filename: {result['filename']}")
        print(f"  Actual Label: {result['actual_label']}")
        print(f"  Predicted Coordinates: {result['predicted_coordinates']}")
        print(f"  Predicted Label: {result['predicted_label']}")
        print()


def predict_image(image_path,model):
    print(f"Predicting for image: {image_path}")

    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    results_mp = face_mesh.process(image_rgb)
    print(f"Processing image:")

    if not results_mp.multi_face_landmarks:
        print(f"No face detected in: {image_path}")

    # Extract landmarks
    landmarks = results_mp.multi_face_landmarks[0].landmark

    # Extract left and right eye regions
    left_eye = extract_eye_region(image, landmarks, LEFT_EYE_LANDMARKS)
    right_eye = extract_eye_region(image, landmarks, RIGHT_EYE_LANDMARKS)

    if left_eye is None or right_eye is None:
        print(f"Failed to extract eye regions for: {image_path}")

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

    print(f"Predicting coordinates for image: {image_path}")

    predicted_coords = model.predict(img_input)[0]  # Get the first prediction
    x_coord, y_coord = predicted_coords
    x_coord = x_coord * 354  # Scale to grid width
    y_coord = y_coord * 792  # Scale to grid height
    print(f"Predicted coordinates: ({x_coord}, {y_coord})")

    # Map the predicted coordinates to a label
    predicted_label = map_coordinates_to_label(x_coord, y_coord)
    return {
        'filename': os.path.basename(image_path),
        'predicted_coordinates': {'x': x_coord, 'y': y_coord},
        'predicted_label': predicted_label
    }

if __name__ == "__main__":
#     # Test on the original dataset
    test_on_original_dataset()

    # Example usage of predict_image function
    