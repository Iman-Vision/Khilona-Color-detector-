import os
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    Input, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from pillow_heif import register_heif_opener

register_heif_opener()
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "src", "cleaned_dataset")
IMG_SIZE = 64
CLASSES = ["blue", "purple", "yellow"]
BELTS = ["A", "B", "C"]
BELT_MAP = {"blue": "A", "yellow": "B", "purple": "C"}


def load_data(root, classes, img_size):
    X, y = [], []
    for idx, c in enumerate(classes):
        cls_dir = Path(root) / c
        files = [f for f in cls_dir.glob("*.*") if f.suffix.lower() in
                 {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".heic", ".heif"}]
        print(f"Class '{c}': {len(files)} images")
        for p in files:
            try:
                img = Image.open(p).convert("RGB").resize((img_size, img_size))
                X.append(np.asarray(img, dtype=np.float32) / 255.0)
                y.append(idx)
            except Exception as e:
                print(f"  Skipping {p.name}: {e}")
    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=len(classes))
    print(f"\nTotal loaded: {len(X)} images")
    return X, y


def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    lr_schedule = CosineDecay(initial_learning_rate=1e-3, decay_steps=1000, alpha=1e-5)
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_history(history, save_path="training_history.png"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'o-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 's-', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'o-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 's-', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to: {save_path}")


def main():
    print("=" * 60)
    print("  KHILONA COLOR DETECTOR - Model Training")
    print("=" * 60)

    X, y = load_data(DATA_DIR, CLASSES, IMG_SIZE)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    model = build_model((IMG_SIZE, IMG_SIZE, 3), len(CLASSES))
    model.summary()

    save_dir = os.path.dirname(__file__)
    model_path = os.path.join(save_dir, "models", "khilona_model.keras")
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print("\nTraining with data augmentation...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=16),
        steps_per_epoch=len(X_train) // 16,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n{'='*40}")
    print(f"  TRAINING ACCURACY:   {train_acc:.4f}")
    print(f"  VALIDATION ACCURACY: {val_acc:.4f}")
    print(f"  TESTING ACCURACY:    {test_acc:.4f}")
    print(f"{'='*40}")

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=CLASSES))

    plot_history(history, os.path.join(save_dir, "training_history.png"))

    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    meta = {
        "classes": CLASSES,
        "belts": BELTS,
        "belt_map": BELT_MAP,
        "img_size": IMG_SIZE,
        "test_accuracy": float(test_acc)
    }
    meta_path = os.path.join(save_dir, "models", "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
