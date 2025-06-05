# test_model.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from models import all_models  # Import models from models.py
import warnings
from datetime import datetime 
warnings.filterwarnings("ignore")
# Set

DATA_DIR = 'dataset_cropped/images'
LOGS_DIR = 'logs'
MODELS_DIR = 'saved_models'
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(img_size):
    X, y = [], []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.jpg'):
            try:
                parts = filename.split('_')
                x = float(parts[1])
                y_coord = float(parts[2])
                path = os.path.join(DATA_DIR, filename)
                img = cv2.imread(path)
                img = cv2.resize(img, (img_size[0], img_size[1]))
                # print(f"Loaded {filename} with coordinates ({x}, {y_coord})")
                if img is None:
                    raise ValueError(f"Image {filename} could not be loaded.")
                X.append(img)
                y.append([x/354, y_coord/792])
            except Exception as e:
                print(f"Skipping {filename}: {e}")
    X = np.array(X) / 255.0
    y = np.array(y) 
    return X, y

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

from tensorflow.keras.callbacks import CSVLogger, Callback

class PrintValAccuracy(Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_mae = logs.get('val_mae')
        print(f"Epoch {epoch + 1}: val_mae = {val_mae:.4f}")

def train_and_evaluate(model_fn, model_name, img_size, X_train, X_test, y_train, y_test):
    model = model_fn(input_shape=(img_size[0], img_size[1], 3))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    #
    # another optimizer
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae']) 
    

    log_path = os.path.join(LOGS_DIR, f'{model_name}_{img_size}.csv')
    csv_logger = CSVLogger(log_path)

    print_callback = PrintValAccuracy()

    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_data=(X_test, y_test),
                        verbose=0, callbacks=[csv_logger,print_callback])

    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    # Predict on test data
    y_pred_scaled = model.predict(X_test)

    y_pred = y_pred_scaled   # Rescale predictions to original coordinates

    # Calculate R-squared
    r2 = calculate_r2(y_test, y_pred)

    # Save model'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model.save(os.path.join(MODELS_DIR, f'{model_name}_{img_size[0]}_{img_size[1]}_{timestamp}.h5'))

    # Plot MSE loss
    plt.plot(history.history['loss'], label='Train MSE')
    plt.plot(history.history['val_loss'], label='Val MSE')
    plt.title(f'{model_name} - Image Size {img_size[0]}x{img_size[1]}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()
    plot_path = os.path.join(LOGS_DIR, f'{model_name}_{img_size}_mse.png')
    plt.savefig(plot_path)
    plt.clf()

    return loss, mae, r2, model  # Return the trained model

def plot_results(results):
    """
    Generate and save a graph comparing MSE and MAE for all models and image sizes.
    """
    mse_values = []
    mae_values = []
    labels = []

    for name, img_size, mse, mae in results:
        labels.append(f"{name}\n({img_size[0]}x{img_size[1]})")
        mse_values.append(mse)
        mae_values.append(mae)

    x = np.arange(len(labels))

    plt.figure(figsize=(12, 8))  # Increased figure size for better readability
    bar_width = 0.35  # Adjusted bar width for better spacing

    # Plot MSE and MAE side by side
    plt.bar(x - bar_width / 2, mse_values, width=bar_width, label='MSE', color='blue')
    plt.bar(x + bar_width / 2, mae_values, width=bar_width, label='MAE', color='orange')

    # Improve label formatting
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=10)
    plt.xlabel('Models and Image Sizes', fontsize=12)
    plt.ylabel('Error Values', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(LOGS_DIR, 'model_comparison.png')
    plt.savefig(plot_path)
    plt.clf()
    print(f"\n📊 Comparison graph saved at: {plot_path}")

def main():
    image_sizes = [[200,50],[400,100]]
    results = []

    for img_size in image_sizes:
        print(f"\n=== Processing image size: {img_size} ===")
        X, y = load_data(img_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # arr_train ={}
        # for i in range(len(y_train)):
        #     arr_train[tuple(y_train[i])] = arr_train.get(tuple(y_train[i]), 0) + 1
        # arr_test={}
        # for i in range(len(y_test)):
        #     arr_test[tuple(y_test[i])] = arr_test.get(tuple(y_test[i]), 0) + 1
        # # Combine unique coordinates from training and testing sets
        # unique_coordinates = set(arr_train.keys()).union(set(arr_test.keys()))

        # for coord in unique_coordinates:
        #     print(f"Coordinates {coord} - Train: {arr_train.get(coord, 0)}, Test: {arr_test.get(coord, 0)} test_percent = {arr_test.get(coord, 0) /(arr_test.get(coord, 1) + arr_train.get(coord, 0) +0.00001)}")
            
        for name, model_fn in all_models:
            try:
                print(f"\nTraining {name} at image size {img_size}...")
                # print(y_train[:10]/[354, 792])
                loss, mae, r2, trained_model = train_and_evaluate(model_fn, name.replace(' ', '_').replace(':', ''), img_size, X_train, X_test, y_train, y_test)
                results.append((name, img_size, loss, mae, r2))
                print(f"→ MSE: {loss:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
                # Use the trained model for predictions
                y_pred = trained_model.predict(X_test)
                print(f"R2 Score (calculated again): {calculate_r2(y_test, y_pred):.4f}")
            except Exception as e:
                print(f"Error training {name} at image size {img_size}: {e}")
                results.append((name, img_size, float('inf'), float('inf'), float('-inf')))

    # Find best model
    best_model = min(results, key=lambda x: x[2])
    print("\n=== Results ===")
    for name, img_size, mse, mae, r2 in results:
        print(f"Model: {name}, Image Size: {img_size}, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    #save result in csv
    results_file = os.path.join(LOGS_DIR, 'results.csv')
    with open(results_file, 'w') as f:
        f.write("Model,Image Size,MSE,MAE,R2\n")
        for name, img_size, mse, mae, r2 in results:
            f.write(f"{name},{img_size},{mse:.4f},{mae:.4f},{r2:.4f}\n") 
    print(f"\n✅ Best model: {best_model[0]} at image size {best_model[1]} with MSE: {best_model[2]:.4f}, MAE: {best_model[3]:.4f}, R2: {best_model[4]:.4f}")

if __name__ == '__main__':
    main()