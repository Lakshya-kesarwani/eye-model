from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow requests from mobile app

DATASET_DIR = 'dataset'
LOG_FILE = 'upload_log.txt'

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