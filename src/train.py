"""
train.py -- DigitSense CNN training pipeline.

Trains a Convolutional Neural Network on the MNIST handwritten digit
dataset, evaluates on the held-out test set, and saves:
    - Trained model  -> model/digit_cnn.h5
    - Metrics        -> results/metrics.txt
    - Confusion matrix -> results/confusion_matrix.png
    - Training curves  -> results/training_curves.png
    - Sample predictions grid -> results/sample_predictions.png
    - 5 sample MNIST test images -> results/sample_digit_0..4.png
"""

# -- Reproducibility seeds (must be set before any TF/Keras import) --
import numpy as np
import random
import os

np.random.seed(42)
random.seed(42)

# Headless matplotlib -- must be set before importing pyplot
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
tf.random.set_seed(42)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from PIL import Image

# -- Paths ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    # -- 1. Load MNIST ------------------------------------------------
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    print(f"  X_train.shape = {X_train.shape}")  # (60000, 28, 28)
    print(f"  X_test.shape  = {X_test.shape}")    # (10000, 28, 28)
    assert X_train.shape == (60000, 28, 28), f"Unexpected X_train shape: {X_train.shape}"
    assert X_test.shape == (10000, 28, 28), f"Unexpected X_test shape: {X_test.shape}"

    # -- Keep raw uint8 test images BEFORE normalizing (for PNG saving) -
    X_test_raw = X_test.copy()  # uint8, values 0-255

    # -- 2. Save 5 raw MNIST test images as PNGs ---------------------
    print("Saving 5 sample MNIST test images (raw uint8)...")
    for i in range(5):
        img = Image.fromarray(X_test_raw[i])
        save_path = os.path.join(RESULTS_DIR, f"sample_digit_{i}.png")
        img.save(save_path)
        # Verify the saved image is not a black square
        reopened = Image.open(save_path)
        arr = np.array(reopened)
        assert arr.max() > 0, f"sample_digit_{i}.png is blank (all zeros)!"
        print(f"  Saved sample_digit_{i}.png -- label={y_test[i]}, "
              f"max_pixel={arr.max()}, shape={arr.shape}")

    # -- 3. Preprocess ------------------------------------------------
    # Normalize to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Reshape to (N, 28, 28, 1) for Conv2D
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # -- 4. Train / Validation split (90/10) --------------------------
    val_split = 0.1
    n_val = int(len(X_train) * val_split)
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train_split, y_train_split = X_train[train_idx], y_train[train_idx]

    print(f"  Training set:   {X_train_split.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set:       {X_test.shape[0]} samples")

    # -- 5. Build CNN -------------------------------------------------
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.summary()

    # -- 6. Compile ---------------------------------------------------
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # -- 7. Train -----------------------------------------------------
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    print("\nTraining CNN...")
    history = model.fit(
        X_train_split, y_train_split,
        epochs=7,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
    )

    # -- 8. Evaluate on test set --------------------------------------
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test loss:     {test_loss:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")

    # Get predictions (probabilities -> class labels)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # -- 9. Compute sklearn metrics -----------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n{'='*50}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (macro)")
    print(f"  Recall:    {recall:.4f} (macro)")
    print(f"  F1 Score:  {f1:.4f} (macro)")
    print(f"{'='*50}")

    # Save metrics to file
    metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("DigitSense -- CNN Test Set Metrics\n")
        f.write("=" * 45 + "\n")
        f.write(f"Test Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision:      {precision:.4f} (macro-averaged)\n")
        f.write(f"Recall:         {recall:.4f} (macro-averaged)\n")
        f.write(f"F1 Score:       {f1:.4f} (macro-averaged)\n")
        f.write("=" * 45 + "\n\n")
        f.write("Classification Report (per-class):\n")
        f.write(classification_report(y_test, y_pred, digits=4))
    print(f"  Metrics saved to {metrics_path}")

    # Sanity check
    if accuracy < 0.95:
        print("WARNING: Test accuracy is below 95% -- check normalization/reshaping!")

    # -- 10. Confusion matrix -----------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=range(10), yticklabels=range(10), ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title("Confusion Matrix -- DigitSense CNN (MNIST Test Set)", fontsize=15)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to {cm_path}")

    # -- Find most-confused digit pairs programmatically --------------
    cm_copy = cm.copy().astype(float)
    np.fill_diagonal(cm_copy, 0)  # zero out the diagonal (correct preds)
    # Find top 5 off-diagonal (most confused) pairs
    top_k = 5
    flat_indices = np.argsort(cm_copy.ravel())[::-1][:top_k]
    print(f"\n  Top {top_k} most-confused digit pairs (true -> predicted, count):")
    confused_pairs = []
    for idx in flat_indices:
        row, col = divmod(idx, 10)
        count = int(cm_copy[row, col])
        if count > 0:
            confused_pairs.append((int(row), int(col), count))
            print(f"    {row} -> {col}: {count} misclassifications")

    # Append to metrics.txt
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(f"\nMost Confused Digit Pairs (top {top_k} off-diagonal):\n")
        f.write("-" * 45 + "\n")
        for true_d, pred_d, cnt in confused_pairs:
            f.write(f"  True {true_d} -> Predicted {pred_d}: {cnt} errors\n")

    # -- 11. Training curves ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curves
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", marker="o")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", marker="s")
    axes[0].set_title("Accuracy per Epoch", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss curves
    axes[1].plot(history.history["loss"], label="Train Loss", marker="o")
    axes[1].plot(history.history["val_loss"], label="Val Loss", marker="s")
    axes[1].set_title("Loss per Epoch", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("DigitSense -- Training Curves", fontsize=16, y=1.02)
    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    fig.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved to {curves_path}")

    # -- 12. Sample predictions grid (3x3) ----------------------------
    # Find misclassified indices
    misclassified = np.where(y_pred != y_test)[0]
    correct = np.where(y_pred == y_test)[0]

    # Pick at least 1 misclassified, rest correct
    n_mis = min(3, len(misclassified))  # up to 3 misclassified
    n_cor = 9 - n_mis
    chosen_mis = np.random.choice(misclassified, n_mis, replace=False) if n_mis > 0 else np.array([], dtype=int)
    chosen_cor = np.random.choice(correct, n_cor, replace=False)
    sample_indices = np.concatenate([chosen_mis, chosen_cor])
    np.random.shuffle(sample_indices)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        idx = sample_indices[i]
        img = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        ax.imshow(img, cmap="gray")
        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"True: {true_label}  Pred: {pred_label}",
                     color=color, fontsize=12, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Sample Predictions -- DigitSense CNN", fontsize=15)
    plt.tight_layout()
    preds_path = os.path.join(RESULTS_DIR, "sample_predictions.png")
    fig.savefig(preds_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sample predictions saved to {preds_path}")

    # -- 13. Save model -----------------------------------------------
    model_path = os.path.join(MODEL_DIR, "digit_cnn.h5")
    model.save(model_path)
    print(f"  Model saved to {model_path}")

    # -- 14. Generate non-MNIST test image (PIL-drawn digit) ----------
    from PIL import ImageDraw, ImageFont
    # Create a black-on-white digit "7" using PIL
    drawn_img = Image.new("L", (100, 100), 255)  # white background
    draw = ImageDraw.Draw(drawn_img)
    try:
        font = ImageFont.truetype("arial.ttf", 72)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((20, 5), "7", fill=0, font=font)  # black digit
    drawn_path = os.path.join(RESULTS_DIR, "test_drawn_digit.png")
    drawn_img.save(drawn_path)
    print(f"  PIL-drawn test digit saved to {drawn_path}")

    print("\n[OK] Training pipeline complete!")
    print(f"   Model:   {model_path}")
    print(f"   Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
