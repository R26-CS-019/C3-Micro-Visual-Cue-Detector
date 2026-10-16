from pathlib import Path
import os
import sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
sys.modules.setdefault("sklearn", None)
from datetime import datetime

import numpy as np
import tensorflow as tf

from PIL import (
    Image,
    ImageOps,
    UnidentifiedImageError,
)

from playwright.sync_api import sync_playwright

from page_type_router import detect_page_type
from checkpoint_loader import load_checkpoint


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

SCREENSHOT_DIR = (
    RESULTS_DIR / "screenshots"
)

HTML_DIR = (
    RESULTS_DIR / "html"
)

PREVIEW_DIR = (
    RESULTS_DIR / "processed_previews"
)


# ============================================================
# MODEL 1
# Login-page phishing detector
# ============================================================

MODEL1_PATH = PROJECT_ROOT.parent / "model 1 login pp2/models/model1_efficientnetv2b0_top_viewport/model1_efficientnetv2b0_selected_best.keras"  # R26 hosting fix

MODEL1_THRESHOLD = 0.4660

MODEL1_WIDTH = 384
MODEL1_HEIGHT = 160


# ============================================================
# MODEL 2
# General webpage phishing detector
# ============================================================

MODEL2_PATH = PROJECT_ROOT.parent / "new model 2 pp2/models/model2_effnetv2b0/stage2/model2_effnetv2b0_stage2_best.keras"

MODEL2_THRESHOLD = 0.392

MODEL2_WIDTH = 384
MODEL2_HEIGHT = 224
MODEL2_CHANNELS = 3


# ============================================================
# CREATE RESULT FOLDERS
# ============================================================

for folder in [
    SCREENSHOT_DIR,
    HTML_DIR,
    PREVIEW_DIR,
]:

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    print("=" * 80)
    print("LOADING VISUAL PHISHING MODELS")
    print("=" * 80)

    print()

    if not MODEL1_PATH.is_file():

        raise FileNotFoundError(
            "\nMODEL 1 was not found:\n"
            f"{MODEL1_PATH}"
        )


    if not MODEL2_PATH.is_file():

        raise FileNotFoundError(
            "\nMODEL 2 was not found:\n"
            f"{MODEL2_PATH}"
        )


    print("Loading Model 1...")

    model1 = load_checkpoint(
        MODEL1_PATH,
        compile=False,
    )

    print(
        f"Model 1 input : "
        f"{model1.input_shape}"
    )


    print()

    print("Loading Model 2...")

    model2 = load_checkpoint(
        MODEL2_PATH,
        compile=False,
    )

    print(
        f"Model 2 input : "
        f"{model2.input_shape}"
    )


    # ========================================================
    # VERIFY INPUT SHAPES
    # ========================================================

    expected_model1_shape = (
        None,
        MODEL1_HEIGHT,
        MODEL1_WIDTH,
        3,
    )

    expected_model2_shape = (
        None,
        MODEL2_HEIGHT,
        MODEL2_WIDTH,
        3,
    )


    if tuple(model1.input_shape) != expected_model1_shape:

        raise RuntimeError(
            "\nUnexpected Model 1 input shape.\n"
            f"Expected: {expected_model1_shape}\n"
            f"Found   : {model1.input_shape}"
        )


    if tuple(model2.input_shape) != expected_model2_shape:

        raise RuntimeError(
            "\nUnexpected Model 2 input shape.\n"
            f"Expected: {expected_model2_shape}\n"
            f"Found   : {model2.input_shape}"
        )


    print()

    print("Both models loaded successfully.")

    print()

    return model1, model2


# ============================================================
# MODEL 1 PREPROCESSING
# ============================================================
#
# Exact idea used during Model 1 preparation:
#
# 1. RGB
# 2. Resize width to 384 preserving aspect ratio
# 3. Keep TOP 160 pixels
# 4. If shorter than 160, white-pad bottom
#
# ============================================================

def preprocess_model1(
    screenshot_path,
):

    try:

        with Image.open(
            screenshot_path
        ) as image:

            image.load()

            image = ImageOps.exif_transpose(
                image
            ).convert("RGB")


            original_width, original_height = (
                image.size
            )


            if (
                original_width < 1
                or
                original_height < 1
            ):

                raise RuntimeError(
                    "Invalid screenshot dimensions."
                )


            # ------------------------------------------------
            # Resize width to 384
            # ------------------------------------------------

            scale = (
                MODEL1_WIDTH
                /
                float(original_width)
            )


            resized_height = max(
                1,
                round(
                    original_height
                    * scale
                ),
            )


            resized = image.resize(
                (
                    MODEL1_WIDTH,
                    resized_height,
                ),
                Image.Resampling.LANCZOS,
            )


            # ------------------------------------------------
            # TOP VIEWPORT
            # ------------------------------------------------

            if resized_height >= MODEL1_HEIGHT:

                prepared = resized.crop(
                    (
                        0,
                        0,
                        MODEL1_WIDTH,
                        MODEL1_HEIGHT,
                    )
                )

                action = "TOP_CROP"


            else:

                prepared = Image.new(
                    "RGB",
                    (
                        MODEL1_WIDTH,
                        MODEL1_HEIGHT,
                    ),
                    (
                        255,
                        255,
                        255,
                    ),
                )

                prepared.paste(
                    resized,
                    (0, 0),
                )

                action = (
                    "BOTTOM_WHITE_PAD"
                )


            # ------------------------------------------------
            # Save preview
            # ------------------------------------------------

            preview_path = (
                PREVIEW_DIR
                /
                "model1_prepared.png"
            )

            prepared.save(
                preview_path
            )


            # EfficientNetV2 built-in preprocessing
            # expects RGB float values in [0, 255].
            model_input = np.expand_dims(
                np.asarray(
                    prepared,
                    dtype=np.float32,
                ),
                axis=0,
            )


            return (
                model_input,
                action,
                preview_path,
            )


    except UnidentifiedImageError as error:

        raise RuntimeError(
            "Model 1 could not read screenshot."
        ) from error


# ============================================================
# MODEL 2 PREPROCESSING
# ============================================================
#
# Exact training/inference idea:
#
# Resize while keeping aspect ratio so image fits inside:
#
#     384 x 224
#
# Then centered white padding.
#
# ============================================================

def preprocess_model2(
    screenshot_path,
):

    image_bytes = tf.io.read_file(
        str(screenshot_path)
    )


    image = tf.io.decode_image(
        image_bytes,
        channels=MODEL2_CHANNELS,
        expand_animations=False,
    )


    image.set_shape(
        [
            None,
            None,
            MODEL2_CHANNELS,
        ]
    )


    image = tf.cast(
        image,
        tf.float32,
    )


    # ========================================================
    # ORIGINAL SIZE
    # ========================================================

    original_shape = tf.cast(
        tf.shape(image)[:2],
        tf.float32,
    )


    original_height = (
        original_shape[0]
    )

    original_width = (
        original_shape[1]
    )


    # ========================================================
    # SCALE WITHOUT DISTORTION
    # ========================================================

    scale = tf.minimum(

        MODEL2_HEIGHT
        /
        original_height,

        MODEL2_WIDTH
        /
        original_width,
    )


    new_height = tf.maximum(

        1,

        tf.cast(
            tf.round(
                original_height
                * scale
            ),
            tf.int32,
        ),
    )


    new_width = tf.maximum(

        1,

        tf.cast(
            tf.round(
                original_width
                * scale
            ),
            tf.int32,
        ),
    )


    # ========================================================
    # RESIZE
    # ========================================================

    image = tf.image.resize(
        image,
        [
            new_height,
            new_width,
        ],
        method="bilinear",
        antialias=True,
    )


    # ========================================================
    # CALCULATE WHITE PADDING
    # ========================================================

    height_padding = (
        MODEL2_HEIGHT
        - new_height
    )

    width_padding = (
        MODEL2_WIDTH
        - new_width
    )


    top_padding = (
        height_padding // 2
    )

    bottom_padding = (
        height_padding
        - top_padding
    )

    left_padding = (
        width_padding // 2
    )

    right_padding = (
        width_padding
        - left_padding
    )


    # ========================================================
    # WHITE PAD
    # ========================================================

    image = tf.pad(

        image,

        paddings=[
            [
                top_padding,
                bottom_padding,
            ],
            [
                left_padding,
                right_padding,
            ],
            [
                0,
                0,
            ],
        ],

        mode="CONSTANT",

        constant_values=255.0,
    )


    image.set_shape(
        [
            MODEL2_HEIGHT,
            MODEL2_WIDTH,
            MODEL2_CHANNELS,
        ]
    )


    # ========================================================
    # SAVE MODEL 2 PREVIEW
    # ========================================================

    preview_path = (
        PREVIEW_DIR
        /
        "model2_prepared.png"
    )


    preview_array = tf.cast(
        tf.clip_by_value(
            image,
            0,
            255,
        ),
        tf.uint8,
    ).numpy()


    Image.fromarray(
        preview_array
    ).save(
        preview_path
    )


    # Add batch dimension
    model_input = tf.expand_dims(
        image,
        axis=0,
    )


    return (
        model_input,
        "ASPECT_RESIZE_WHITE_PADDING",
        preview_path,
    )


# ============================================================
# EXTRACT MODEL SCORE
# ============================================================

def extract_probability(
    raw_prediction,
):

    prediction = np.asarray(
        raw_prediction
    )


    if prediction.shape == (1, 1):

        probability = float(
            prediction[0, 0]
        )


    elif prediction.shape == (1,):

        probability = float(
            prediction[0]
        )


    elif prediction.shape == (1, 2):

        # Positive class is index 1
        probability = float(
            prediction[0, 1]
        )


    else:

        raise RuntimeError(
            "Unexpected model output shape: "
            f"{prediction.shape}"
        )


    if not (
        np.isfinite(probability)
        and
        0.0 <= probability <= 1.0
    ):

        raise RuntimeError(
            "Invalid model probability: "
            f"{probability}"
        )


    return probability


# ============================================================
# MODEL 1 PREDICTION
# ============================================================

def predict_model1(
    model,
    screenshot_path,
):

    (
        model_input,
        preprocessing_action,
        preview_path,
    ) = preprocess_model1(
        screenshot_path
    )


    raw_prediction = model.predict(
        model_input,
        verbose=0,
    )


    malicious_probability = (
        extract_probability(
            raw_prediction
        )
    )


    legitimate_probability = (
        1.0
        - malicious_probability
    )


    prediction = (

        "PHISHING"

        if malicious_probability
        >= MODEL1_THRESHOLD

        else "LEGITIMATE"
    )


    return {

        "model":
            "MODEL_1",

        "model_name":
            "EfficientNetV2B0 Login Page Detector",

        "threshold":
            MODEL1_THRESHOLD,

        "phishing_probability":
            malicious_probability,

        "legitimate_probability":
            legitimate_probability,

        "prediction":
            prediction,

        "preprocessing":
            preprocessing_action,

        "preview_path":
            str(preview_path),
    }


# ============================================================
# MODEL 2 PREDICTION
# ============================================================

def predict_model2(
    model,
    screenshot_path,
):

    (
        model_input,
        preprocessing_action,
        preview_path,
    ) = preprocess_model2(
        screenshot_path
    )


    raw_prediction = model.predict(
        model_input,
        verbose=0,
    )


    phishing_probability = (
        extract_probability(
            raw_prediction
        )
    )


    legitimate_probability = (
        1.0
        - phishing_probability
    )


    prediction = (

        "PHISHING"

        if phishing_probability
        >= MODEL2_THRESHOLD

        else "LEGITIMATE"
    )


    return {

        "model":
            "MODEL_2",

        "model_name":
            "EfficientNetV2B0 Webpage Detector",

        "threshold":
            MODEL2_THRESHOLD,

        "phishing_probability":
            phishing_probability,

        "legitimate_probability":
            legitimate_probability,

        "prediction":
            prediction,

        "preprocessing":
            preprocessing_action,

        "preview_path":
            str(preview_path),
    }


# ============================================================
# CAPTURE WEBPAGE
# ============================================================



# ============================================================
# PUBLIC-ADDRESS GUARD  (R26 hosting fix)
# The hosted browser may only open public websites, never
# private, local or cloud-internal addresses.
# ============================================================

import ipaddress
import socket
from urllib.parse import urlsplit


def _is_public_url(url):
    parts = urlsplit(url)
    if parts.scheme in ("data", "blob", "about"):
        return True
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    try:
        infos = socket.getaddrinfo(
            parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)
        )
    except (socket.gaierror, UnicodeError, ValueError):
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        mapped = getattr(address, "ipv4_mapped", None)
        if not address.is_global or (mapped and not mapped.is_global):
            return False
    return True


def _guard_route(route):
    if _is_public_url(route.request.url):
        route.continue_()
    else:
        route.abort()


def require_public_url(url):
    if not _is_public_url(url):
        raise ValueError("Use a public http(s) website address.")

def capture_webpage(
    url,
):

    require_public_url(url)  # R26 hosting fix

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    screenshot_path = (
        SCREENSHOT_DIR
        /
        f"webpage_{timestamp}.png"
    )


    html_path = (
        HTML_DIR
        /
        f"webpage_{timestamp}.html"
    )


    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=os.environ.get("C3_HEADLESS", "1") != "0",  # R26 hosting fix
        )


        page = browser.new_page(

            viewport={
                "width": 1366,
                "height": 768,
            }
        )


        try:

            print()
            print("Opening webpage...")


            page.route("**/*", _guard_route)  # R26 hosting fix

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000,
            )


            # Allow dynamic content to render
            page.wait_for_timeout(
                7000
            )


            # =================================================
            # HTML
            # =================================================

            html = page.content()


            html_path.write_text(
                html,
                encoding="utf-8",
                errors="ignore",
            )


            # =================================================
            # SCREENSHOT
            # =================================================

            page.screenshot(
                path=str(
                    screenshot_path
                ),
                full_page=False,
            )


            return (
                html,
                screenshot_path,
                html_path,
            )


        finally:

            browser.close()


# ============================================================
# COMPLETE VISUAL COMPONENT
# ============================================================

def analyze_url(
    url,
    model1,
    model2,
):

    print()
    print("=" * 80)
    print("MICRO VISUAL CUE DETECTION COMPONENT")
    print("=" * 80)

    print()

    print(
        f"URL: {url}"
    )


    # ========================================================
    # CAPTURE WEBPAGE
    # ========================================================

    (
        html,
        screenshot_path,
        html_path,
    ) = capture_webpage(
        url
    )


    print(
        "Rendered HTML collected."
    )

    print(
        "Screenshot captured."
    )


    # ========================================================
    # PAGE-TYPE ROUTER
    # ========================================================

    router_result = detect_page_type(
        html,
        url
    )


    page_type = (
        router_result[
            "page_type"
        ]
    )


    print()
    print("=" * 80)
    print("PAGE TYPE ROUTER")
    print("=" * 80)

    print()

    print(
        f"Page type        : "
        f"{page_type}"
    )

    print(
        f"Router confidence: "
        f"{router_result['confidence']}"
    )

    print(
        f"Router score     : "
        f"{router_result['score']}"
    )


    # ========================================================
    # MODEL ROUTING
    # ========================================================

    if page_type == "LOGIN":

        print()
        print(
            "Routing screenshot → MODEL 1"
        )

        prediction_result = (
            predict_model1(
                model1,
                screenshot_path,
            )
        )


    else:

        print()
        print(
            "Routing screenshot → MODEL 2"
        )

        prediction_result = (
            predict_model2(
                model2,
                screenshot_path,
            )
        )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 80)
    print("VISUAL DETECTION RESULT")
    print("=" * 80)

    print()

    print(
        f"Page type            : "
        f"{page_type}"
    )

    print(
        f"Model used           : "
        f"{prediction_result['model']}"
    )

    print(
        f"Model                : "
        f"{prediction_result['model_name']}"
    )

    print()

    print(
        f"Phishing probability : "
        f"{prediction_result['phishing_probability']:.6f}"
    )

    print(
        f"Legitimate probability: "
        f"{prediction_result['legitimate_probability']:.6f}"
    )

    print(
        f"Decision threshold   : "
        f"{prediction_result['threshold']:.4f}"
    )

    print()

    print(
        f"FINAL PREDICTION     : "
        f"{prediction_result['prediction']}"
    )

    print()

    print(
        f"Screenshot           : "
        f"{screenshot_path}"
    )

    print(
        f"Rendered HTML        : "
        f"{html_path}"
    )

    print(
        f"Prepared preview     : "
        f"{prediction_result['preview_path']}"
    )

    print()

    print("=" * 80)


    # ========================================================
    # RETURN FOR FUTURE API / GROUP FUSION
    # ========================================================

    return {

        "component":
            "MICRO_VISUAL_CUE",

        "url":
            url,

        "page_type":
            page_type,

        "router_confidence":
            router_result[
                "confidence"
            ],

        "router_score":
            router_result[
                "score"
            ],

        "model_used":
            prediction_result[
                "model"
            ],

        "model_name":
            prediction_result[
                "model_name"
            ],

        "phishing_probability":
            prediction_result[
                "phishing_probability"
            ],

        "legitimate_probability":
            prediction_result[
                "legitimate_probability"
            ],

        "threshold":
            prediction_result[
                "threshold"
            ],

        "prediction":
            prediction_result[
                "prediction"
            ],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MICRO VISUAL CUE COMPONENT STARTUP")
    print("=" * 80)

    print()

    print(
        f"Model 1 threshold : "
        f"{MODEL1_THRESHOLD:.4f}"
    )

    print(
        f"Model 2 threshold : "
        f"{MODEL2_THRESHOLD:.3f}"
    )

    print()


    # ========================================================
    # LOAD BOTH MODELS ONCE
    # ========================================================

    model1, model2 = (
        load_models()
    )


    # ========================================================
    # URL INPUT
    # ========================================================

    url = input(
        "\nEnter website URL: "
    ).strip()


    if not url.startswith(
        (
            "http://",
            "https://",
        )
    ):

        url = (
            "https://" + url
        )


    # ========================================================
    # ANALYZE
    # ========================================================

    analyze_url(
        url,
        model1,
        model2,
    )


if __name__ == "__main__":

    try:

        main()


    except Exception as error:

        print()

        print("=" * 80)
        print("COMPONENT ERROR")
        print("=" * 80)

        print()

        print(error)

        print()

        print("=" * 80)