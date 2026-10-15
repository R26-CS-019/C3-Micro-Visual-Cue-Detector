"""
19_predict_single_screenshot.py

Classify one screenshot at a time with the retained EfficientNetV2B0 model.
The source screenshot and selected model are never modified.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(r"C:\Users\oshan\Desktop\model 1 login pp2")

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "model1_efficientnetv2b0_top_viewport"
    / "model1_efficientnetv2b0_selected_best.keras"
)

EXPECTED_MODEL_SHA256 = (
    "6be68e8492246210f323b99a89fdc6107d7abfef30db7bb06820b5005497f1af"
)

OUTPUT_DIR = PROJECT_DIR / "reports" / "19_single_screenshot_predictions"
PREVIEW_DIR = OUTPUT_DIR / "processed_previews"
PREDICTIONS_CSV = OUTPUT_DIR / "single_image_predictions.csv"

TARGET_WIDTH = 384
TARGET_HEIGHT = 160
DECISION_THRESHOLD = 0.4660

ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".webp", ".tif", ".tiff",
}


# =============================================================================
# VALIDATION AND PREPROCESSING
# =============================================================================

def calculate_sha256(file_path):
    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def validate_and_load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Selected model not found:\n{MODEL_PATH}")

    model_hash = calculate_sha256(MODEL_PATH)

    if model_hash.lower() != EXPECTED_MODEL_SHA256.lower():
        raise RuntimeError(
            "Selected model SHA256 mismatch.\n"
            f"Expected: {EXPECTED_MODEL_SHA256}\n"
            f"Actual  : {model_hash}\n"
            "Prediction stopped to avoid using the wrong model."
        )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise RuntimeError(f"Unexpected model inputs: {input_shape}")
        input_shape = input_shape[0]

    expected_shape = (None, TARGET_HEIGHT, TARGET_WIDTH, 3)

    if tuple(input_shape) != expected_shape:
        raise RuntimeError(
            f"Expected model input {expected_shape}, but found {input_shape}."
        )

    return model, model_hash


def validate_image_path(image_path):
    image_path = Path(image_path).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Screenshot not found:\n{image_path}")

    if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise RuntimeError(
            f"Unsupported extension: {image_path.suffix}\n"
            f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    return image_path


def preprocess_image(image_path):
    """
    Match UNIFORM_WIDTH_SCALE_TOP_VIEWPORT:
    resize width to 384 while preserving aspect ratio, then take top 160 pixels.
    If an unusual image becomes shorter than 160 pixels, pad its bottom white.
    """

    try:
        with Image.open(image_path) as image:
            image.load()
            image = ImageOps.exif_transpose(image).convert("RGB")
            original_width, original_height = image.size

            if original_width < 1 or original_height < 1:
                raise RuntimeError(f"Invalid image dimensions: {image.size}")

            scale = TARGET_WIDTH / float(original_width)
            resized_height = max(1, round(original_height * scale))

            resized = image.resize(
                (TARGET_WIDTH, resized_height),
                Image.Resampling.LANCZOS,
            )

            if resized_height >= TARGET_HEIGHT:
                prepared = resized.crop((0, 0, TARGET_WIDTH, TARGET_HEIGHT))
                action = "TOP_CROP"
                changed_pixels = resized_height - TARGET_HEIGHT
            else:
                prepared = Image.new(
                    "RGB",
                    (TARGET_WIDTH, TARGET_HEIGHT),
                    (255, 255, 255),
                )
                prepared.paste(resized, (0, 0))
                action = "BOTTOM_WHITE_PAD"
                changed_pixels = TARGET_HEIGHT - resized_height

    except UnidentifiedImageError as error:
        raise RuntimeError(f"Unreadable image:\n{image_path}") from error

    return prepared, {
        "original_width": original_width,
        "original_height": original_height,
        "resized_height": resized_height,
        "vertical_action": action,
        "removed_or_padded_pixels": changed_pixels,
    }


# =============================================================================
# PREDICTION AND REPORTING
# =============================================================================

def extract_malicious_score(raw_prediction):
    prediction = np.asarray(raw_prediction)

    if prediction.shape == (1, 1):
        score = float(prediction[0, 0])
    elif prediction.shape == (1, 2):
        score = float(prediction[0, 1])
    elif prediction.shape == (1,):
        score = float(prediction[0])
    else:
        raise RuntimeError(
            f"Unexpected prediction output shape: {prediction.shape}"
        )

    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(f"Invalid model output: {score}")

    return score


def append_csv(row):
    file_exists = PREDICTIONS_CSV.exists()

    with open(
        PREDICTIONS_CSV,
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def predict_image(model, model_hash, selected_path):
    image_path = validate_image_path(selected_path)
    image_hash = calculate_sha256(image_path)
    prepared, details = preprocess_image(image_path)

    model_input = np.expand_dims(
        np.asarray(prepared, dtype=np.float32),
        axis=0,
    )

    malicious_score = extract_malicious_score(
        model.predict(model_input, verbose=0)
    )

    predicted_label = int(malicious_score >= DECISION_THRESHOLD)
    predicted_class = "MALICIOUS" if predicted_label else "NOT_MALICIOUS"
    class_score = malicious_score if predicted_label else 1.0 - malicious_score

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in image_path.stem
    )[:80]

    preview_path = PREVIEW_DIR / f"{timestamp}_{safe_stem}_384x160.png"
    prepared.save(preview_path, format="PNG", optimize=False)

    row = {
        "prediction_time": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "source_image_path": str(image_path),
        "source_filename": image_path.name,
        "source_sha256": image_hash,
        "original_width": details["original_width"],
        "original_height": details["original_height"],
        "height_after_width_scale": details["resized_height"],
        "prepared_width": TARGET_WIDTH,
        "prepared_height": TARGET_HEIGHT,
        "vertical_action": details["vertical_action"],
        "removed_or_padded_pixels": details["removed_or_padded_pixels"],
        "processed_preview_path": str(preview_path),
        "model_path": str(MODEL_PATH),
        "model_sha256": model_hash,
        "decision_threshold": f"{DECISION_THRESHOLD:.4f}",
        "malicious_score": f"{malicious_score:.10f}",
        "predicted_numeric_label": predicted_label,
        "predicted_class": predicted_class,
        "predicted_class_score": f"{class_score:.10f}",
    }

    append_csv(row)
    return row


def format_result(row):
    return (
        f"Prediction        : {row['predicted_class']}\n"
        f"Malicious score   : {float(row['malicious_score']):.6f}\n"
        f"Fixed threshold   : {DECISION_THRESHOLD:.4f}\n"
        f"Class score       : {float(row['predicted_class_score']):.6f}\n"
        f"Original size     : {row['original_width']}x{row['original_height']}\n"
        f"Prepared size     : {TARGET_WIDTH}x{TARGET_HEIGHT}\n"
        f"Vertical action   : {row['vertical_action']}\n"
        f"Preview saved to  : {row['processed_preview_path']}\n\n"
        "This visual prediction is supporting evidence, not proof that a page "
        "is safe or malicious."
    )


# =============================================================================
# FILE-SELECTION WINDOW
# =============================================================================

def run_file_dialog(model, model_hash):
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError as error:
        raise RuntimeError(
            "Tkinter is unavailable. Run the script with an image path."
        ) from error

    root = tk.Tk()
    root.withdraw()

    try:
        while True:
            selected_path = filedialog.askopenfilename(
                title="Select a screenshot to classify",
                filetypes=[
                    (
                        "Image files",
                        "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff",
                    ),
                    ("All files", "*.*"),
                ],
            )

            if not selected_path:
                print("No screenshot selected. Exiting.")
                break

            try:
                row = predict_image(model, model_hash, selected_path)
                result_text = format_result(row)

                print("\n" + "=" * 80)
                print("SINGLE-SCREENSHOT PREDICTION")
                print("=" * 80)
                print(result_text)
                print("=" * 80)

                again = messagebox.askyesno(
                    f"Prediction: {row['predicted_class']}",
                    result_text + "\n\nCheck another screenshot?",
                )

                if not again:
                    break

            except Exception as error:
                print(f"\nPREDICTION FAILED\n{error}")

                if not messagebox.askretrycancel(
                    "Prediction failed",
                    f"{error}\n\nSelect another screenshot?",
                ):
                    break
    finally:
        root.destroy()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classify one screenshot with the retained visual model."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Optional image path; omit it to open the file-selection window.",
    )
    arguments = parser.parse_args()

    print("=" * 80)
    print("EFFICIENTNETV2B0 SINGLE-SCREENSHOT PREDICTION")
    print("=" * 80)
    print(f"TensorFlow          : {tf.__version__}")
    print(f"Model               : {MODEL_PATH}")
    print(f"Fixed threshold     : {DECISION_THRESHOLD:.4f}")
    print(f"Target dimensions   : {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print("Preprocessing       : UNIFORM_WIDTH_SCALE_TOP_VIEWPORT")
    print(f"Output directory    : {OUTPUT_DIR}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    print("Verifying and loading retained model...")
    model, model_hash = validate_and_load_model()
    print(f"Model SHA256        : {model_hash}")
    print(f"Model input         : {model.input_shape}\n")

    if arguments.image_path:
        row = predict_image(model, model_hash, arguments.image_path)
        print("=" * 80)
        print("SINGLE-SCREENSHOT PREDICTION")
        print("=" * 80)
        print(format_result(row))
        print("=" * 80)
    else:
        print("Opening screenshot selection window...")
        run_file_dialog(model, model_hash)

    print(f"\nPrediction log saved to: {PREDICTIONS_CSV}")
    print("No training or threshold tuning was performed.")
    print("The selected source screenshot was not modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 80)
        print("SINGLE-SCREENSHOT PREDICTION FAILED")
        print("=" * 80)
        print(error)
        print("\nThe model and source screenshot were not modified.")
        sys.exit(1)
