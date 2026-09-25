import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_DIR = "leapGestRecog"   # folder containing 00, 01, ..., 09 subject folders
IMG_SIZE = (64, 64)
RANDOM_STATE = 42
EPOCHS = 10
BATCH_SIZE = 32

os.makedirs("outputs", exist_ok=True)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ----------------------------------------------------------------------
# 1. Load images and labels
# ----------------------------------------------------------------------
print("Scanning dataset...")

images, labels = [], []
label_names = []

subject_folders = sorted(os.listdir(DATA_DIR))
subject_folders = [f for f in subject_folders if os.path.isdir(os.path.join(DATA_DIR, f))]

# Gesture class folders are the same across every subject folder;
# grab the list of gesture names from the first subject.
first_subject_path = os.path.join(DATA_DIR, subject_folders[0])
gesture_folders = sorted(os.listdir(first_subject_path))
gesture_folders = [g for g in gesture_folders if os.path.isdir(os.path.join(first_subject_path, g))]
label_names = gesture_folders

print(f"Found {len(subject_folders)} subjects and {len(gesture_folders)} gesture classes:")
for g in label_names:
    print(f"  - {g}")

for subject in subject_folders:
    for class_idx, gesture in enumerate(gesture_folders):
        folder = os.path.join(DATA_DIR, subject, gesture)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith(".png"):
                continue
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path).convert("L").resize(IMG_SIZE)
                images.append(np.array(img) / 255.0)
                labels.append(class_idx)
            except Exception as e:
                print(f"Skipping {path}: {e}")

X = np.array(images).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
y = np.array(labels)

print(f"\nTotal images loaded: {X.shape[0]}")

# ----------------------------------------------------------------------
# 2. Train/test split
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ----------------------------------------------------------------------
# 3. Build CNN model
# ----------------------------------------------------------------------
num_classes = len(label_names)

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ----------------------------------------------------------------------
# 4. Train
# ----------------------------------------------------------------------
print("\nTraining CNN...")
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)

# ----------------------------------------------------------------------
# 5. Evaluate
# ----------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.3f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=label_names))

# ----------------------------------------------------------------------
# 6. Training curves
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/training_curves.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# 7. Confusion matrix
# ----------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
plt.title(f"Confusion Matrix (Accuracy: {test_acc:.1%})")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# 8. Sample predictions
# ----------------------------------------------------------------------
sample_idx = np.random.choice(len(X_test), size=8, replace=False)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for ax, idx in zip(axes.ravel(), sample_idx):
    ax.imshow(X_test[idx].squeeze(), cmap="gray")
    true_l = label_names[y_test[idx]]
    pred_l = label_names[y_pred[idx]]
    color = "green" if true_l == pred_l else "red"
    ax.set_title(f"True: {true_l}\nPred: {pred_l}", color=color, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_predictions.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# 9. Save model
# ----------------------------------------------------------------------
model.save("outputs/gesture_model.keras")
print("Model saved to outputs/gesture_model.keras")

print("\nDone. Outputs saved to outputs/")
