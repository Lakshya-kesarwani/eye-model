# models.py
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2


def model_1(input_shape):  # Simple CNN
    return models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='sigmoid')
    ])

def model_2(input_shape):  # BatchNorm + Dropout
    return models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='sigmoid')
    ])

def model_3(input_shape):  # Large kernels
    return models.Sequential([
        layers.Conv2D(32, (5,5), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (5,5), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='sigmoid')
    ])


def model_4(input_shape):  # L2 Regularization
    l2 = regularizers.l2(0.01)
    return models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2, input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=l2),
        layers.Dense(2, activation='sigmoid', kernel_regularizer=l2)
    ])

def model_5(input_shape):  # Lightweight
    return models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='sigmoid')
    ])

def model_6(input_shape):  # Deep CNN
    return models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dense(2,activation = "sigmoid")
    ])

def model_7(input_shape):  # SeparableConv
    return models.Sequential([
        layers.SeparableConv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.SeparableConv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2,activation = "sigmoid")
    ])

# List of model names and functions
all_models = [
    ("Model 1: Simple CNN", model_1),
    ("Model 2: BatchNorm + Dropout", model_2),
    ("Model 3: Large Kernels", model_3),
    ("Model 4: L2 Regularization", model_4),
    ("Model 5: Lightweight", model_5),
    ("Model 6: Deep CNN", model_6),
    ("Model 7: SeparableConv", model_7),
]
