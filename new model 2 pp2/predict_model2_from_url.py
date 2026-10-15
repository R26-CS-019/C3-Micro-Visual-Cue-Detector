"""Capture a webpage screenshot from a URL and evaluate it with Model 2."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import tensorflow as tf
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from predict_model2_single_external_image import (
    MODEL_PATH,
    load_and_prepare_image,
    load_locked_threshold,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parent

CAPTURE_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "model2_effnetv2b0"
    / "url_capture"
)

VIEWPORT_WIDTH = 1366
VIEWPORT_HEIGHT = 768
PAGE_TIMEOUT_MS = 30_000
PAGE_SETTLE_TIME_MS = 2_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a screenshot and HTML from a webpage URL, "
            "then evaluate the screenshot using finalized Model 2."
        )
    )

    parser.add_argument(
        "url",
        help="Full webpage URL beginning with http:// or https://",
    )

    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show the Chromium window while capturing the webpage.",
    )

    parser.add_argument(
        "--full-page",
        action="store_true",
        help="Capture the complete webpage instead of the visible viewport.",
    )

    return parser.parse_args()


def validate_url(url: str) -> str:
    url = url.strip()

    parsed = urlparse(url)

    if parsed.scheme.lower() not in {"http", "https"}:
        raise SystemExit(
            "\nERROR: The URL must begin with http:// or https://"
        )

    if not parsed.hostname:
        raise SystemExit("\nERROR: The URL does not contain a valid hostname.")

    return url


def create_capture_folder(url: str) -> Path:
    hostname = urlparse(url).hostname or "website"

    safe_hostname = re.sub(
        pattern=r"[^A-Za-z0-9._-]+",
        repl="_",
        string=hostname,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    capture_folder = CAPTURE_ROOT / f"{timestamp}_{safe_hostname}"
    capture_folder.mkdir(parents=True, exist_ok=False)

    return capture_folder


def capture_webpage(
    url: str,
    capture_folder: Path,
    show_browser: bool,
    full_page: bool,
) -> dict[str, object]:

    screenshot_path = capture_folder / "webpage_screenshot.png"
    html_path = capture_folder / "webpage_source.html"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not show_browser,
        )

        context = browser.new_context(
            viewport={
                "width": VIEWPORT_WIDTH,
                "height": VIEWPORT_HEIGHT,
            },
            device_scale_factor=1,
            accept_downloads=False,
            service_workers="block",
        )

        page = context.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)

        response = None
        timed_out = False

        try:
            response = page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=PAGE_TIMEOUT_MS,
            )
        except PlaywrightTimeoutError:
            timed_out = True
            print(
                "WARNING: The webpage did not completely load within "
                "30 seconds. Capturing the available content."
            )

        page.wait_for_timeout(PAGE_SETTLE_TIME_MS)

        page.screenshot(
            path=str(screenshot_path),
            full_page=full_page,
        )

        html_content = page.content()
        html_path.write_text(
            html_content,
            encoding="utf-8",
        )

        capture_information = {
            "requested_url": url,
            "final_url": page.url,
            "page_title": page.title(),
            "http_status": response.status if response else None,
            "page_load_timed_out": timed_out,
            "screenshot_path": str(screenshot_path),
            "html_path": str(html_path),
        }

        context.close()
        browser.close()

    return capture_information


def predict_screenshot(
    screenshot_path: Path,
) -> dict[str, object]:

    if not MODEL_PATH.is_file():
        raise SystemExit(
            "\nERROR: Final Model 2 was not found:\n"
            f"{MODEL_PATH}"
        )

    threshold, selection_rule = load_locked_threshold()

    image_batch = load_and_prepare_image(screenshot_path)

    model = tf.keras.models.load_model(MODEL_PATH)

    phishing_probability = float(
        model.predict(
            image_batch,
            verbose=0,
        ).reshape(-1)[0]
    )

    legitimate_probability = 1.0 - phishing_probability

    predicted_label = (
        "PHISHING"
        if phishing_probability >= threshold
        else "LEGITIMATE"
    )

    return {
        "predicted_label": predicted_label,
        "phishing_probability": phishing_probability,
        "legitimate_probability": legitimate_probability,
        "locked_threshold": threshold,
        "threshold_selection_rule": selection_rule,
        "model_path": str(MODEL_PATH),
        "screenshot_sha256": sha256_file(screenshot_path),
    }


def main() -> None:
    args = parse_arguments()
    url = validate_url(args.url)

    capture_folder = create_capture_folder(url)

    print("=" * 80)
    print("MODEL 2 — AUTOMATIC URL SCREENSHOT PREDICTION")
    print("=" * 80)
    print(f"URL: {url}")
    print("Opening webpage using Playwright Chromium...")
    print()

    capture_information = capture_webpage(
        url=url,
        capture_folder=capture_folder,
        show_browser=args.show_browser,
        full_page=args.full_page,
    )

    screenshot_path = Path(
        str(capture_information["screenshot_path"])
    )

    print("Webpage capture completed.")
    print(f"Screenshot: {screenshot_path}")
    print(f"HTML file: {capture_information['html_path']}")
    print()
    print("Sending screenshot to Model 2...")

    prediction = predict_screenshot(screenshot_path)

    complete_result = {
        "capture_time": datetime.now().astimezone().isoformat(),
        **capture_information,
        **prediction,
    }

    result_path = capture_folder / "visual_prediction_result.json"

    with result_path.open("w", encoding="utf-8") as file_handle:
        json.dump(
            complete_result,
            file_handle,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("PREDICTION RESULT")
    print("-" * 80)
    print(
        f"Predicted label       : "
        f"{prediction['predicted_label']}"
    )
    print(
        f"Phishing probability  : "
        f"{prediction['phishing_probability']:.6f}"
    )
    print(
        f"Legitimate probability: "
        f"{prediction['legitimate_probability']:.6f}"
    )
    print(
        f"Locked threshold      : "
        f"{prediction['locked_threshold']:.3f}"
    )

    print()
    print("OUTPUT FILES")
    print("-" * 80)
    print(f"Screenshot: {screenshot_path}")
    print(f"HTML source: {capture_information['html_path']}")
    print(f"JSON result: {result_path}")

    print()
    print("IMPORTANT")
    print("-" * 80)
    print("The finalized model and threshold were not changed.")
    print("Use an isolated testing environment for suspicious URLs.")
    print("=" * 80)


if __name__ == "__main__":
    main()