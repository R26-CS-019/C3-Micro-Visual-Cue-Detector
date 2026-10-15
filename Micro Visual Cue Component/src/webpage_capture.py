from pathlib import Path
from datetime import datetime

from playwright.sync_api import sync_playwright

from page_type_router import detect_page_type


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\oshan\Desktop\Micro Visual Cue Component"
)

RESULTS_DIR = PROJECT_ROOT / "results"

SCREENSHOT_DIR = RESULTS_DIR / "screenshots"

HTML_DIR = RESULTS_DIR / "html"


# Create folders automatically
SCREENSHOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HTML_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CAPTURE WEBPAGE
# ============================================================

def capture_webpage(url):

    print("=" * 70)
    print("MICRO VISUAL CUE - WEBPAGE CAPTURE")
    print("=" * 70)

    print()
    print("URL:")
    print(url)
    print()

    # --------------------------------------------------------
    # Create unique filename
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    screenshot_path = (
        SCREENSHOT_DIR /
        f"webpage_{timestamp}.png"
    )

    html_path = (
        HTML_DIR /
        f"webpage_{timestamp}.html"
    )


    # ========================================================
    # OPEN BROWSER
    # ========================================================

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            viewport={
                "width": 1366,
                "height": 768
            }
        )

        try:

            print("Opening webpage...")

            # ------------------------------------------------
            # Load webpage
            # ------------------------------------------------

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000
            )

            # Allow dynamic content to load
            page.wait_for_timeout(
                3000
            )


            # =================================================
            # GET RENDERED HTML
            # =================================================

            html = page.content()

            html_path.write_text(
                html,
                encoding="utf-8",
                errors="ignore"
            )

            print(
                "Rendered HTML collected."
            )


            # =================================================
            # CAPTURE SCREENSHOT
            # =================================================

            page.screenshot(
                path=str(
                    screenshot_path
                ),
                full_page=True
            )

            print(
                "Screenshot captured."
            )


            # =================================================
            # ROUTE PAGE
            # =================================================

            router_result = detect_page_type(
                html
            )


            print()
            print("=" * 70)
            print("PAGE ROUTING RESULT")
            print("=" * 70)

            print()

            print(
                f"Page type  : "
                f"{router_result['page_type']}"
            )

            print(
                f"Confidence : "
                f"{router_result['confidence']}"
            )

            print(
                f"Router score: "
                f"{router_result['score']}"
            )

            print()

            print(
                f"Password input : "
                f"{router_result['password_input']}"
            )

            print(
                f"Username input : "
                f"{router_result['username_input']}"
            )

            print(
                f"Login text     : "
                f"{router_result['login_text']}"
            )

            print(
                f"Form present   : "
                f"{router_result['form_present']}"
            )


            print()
            print("=" * 70)

            # =================================================
            # SELECT MODEL
            # =================================================

            if (
                router_result["page_type"]
                == "LOGIN"
            ):

                model_to_use = "MODEL_1"

                print(
                    "ROUTE TO → MODEL 1"
                )

                print(
                    "EfficientNetV2B0 "
                    "Login Page Detector"
                )

            else:

                model_to_use = "MODEL_2"

                print(
                    "ROUTE TO → MODEL 2"
                )

                print(
                    "EfficientNetV2B0 "
                    "Webpage Detector"
                )


            print("=" * 70)

            print()
            print("Screenshot saved:")
            print(screenshot_path)

            print()
            print("HTML saved:")
            print(html_path)


            # =================================================
            # RETURN DATA FOR FUTURE CNN CONNECTION
            # =================================================

            return {

                "url":
                    url,

                "page_type":
                    router_result[
                        "page_type"
                    ],

                "router_confidence":
                    router_result[
                        "confidence"
                    ],

                "router_score":
                    router_result[
                        "score"
                    ],

                "model_to_use":
                    model_to_use,

                "screenshot_path":
                    str(
                        screenshot_path
                    ),

                "html_path":
                    str(
                        html_path
                    ),
            }


        except Exception as error:

            print()
            print("=" * 70)
            print("ERROR")
            print("=" * 70)

            print()
            print(error)

            return None


        finally:

            browser.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    url = input(
        "Enter website URL: "
    ).strip()


    # Add https automatically
    if not url.startswith(
        (
            "http://",
            "https://"
        )
    ):

        url = (
            "https://" + url
        )


    result = capture_webpage(
        url
    )


    if result is not None:

        print()
        print("=" * 70)
        print("CAPTURE COMPLETE")
        print("=" * 70)

        print()

        print(
            f"Page type    : "
            f"{result['page_type']}"
        )

        print(
            f"Selected model: "
            f"{result['model_to_use']}"
        )

        print()

        print(
            "The screenshot is now ready "
            "for EfficientNetV2B0 prediction."
        )

        print("=" * 70)