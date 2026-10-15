from playwright.sync_api import sync_playwright

from page_type_router import detect_page_type


def test_url(url):

    print("=" * 70)
    print("MICRO VISUAL CUE - URL PAGE TYPE TEST")
    print("=" * 70)

    print()
    print("URL:")
    print(url)
    print()

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

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000
            )

            # Small wait for dynamic page content
            page.wait_for_timeout(
                3000
            )

            # Get rendered HTML automatically
            html = page.content()

            print("HTML collected successfully.")

            # Run your existing router
            result = detect_page_type(
                html
            )

            print()
            print("=" * 70)
            print("PAGE TYPE RESULT")
            print("=" * 70)

            print()
            print(
                f"Page type  : "
                f"{result['page_type']}"
            )

            print(
                f"Confidence : "
                f"{result['confidence']}"
            )

            print(
                f"Score      : "
                f"{result['score']}"
            )

            print()

            print(
                f"Password input : "
                f"{result['password_input']}"
            )

            print(
                f"Username input : "
                f"{result['username_input']}"
            )

            print(
                f"Login text     : "
                f"{result['login_text']}"
            )

            print(
                f"Form present   : "
                f"{result['form_present']}"
            )

            print()

            print("Reasons:")

            for reason in result["reasons"]:

                print(
                    f"  - {reason}"
                )

            print()
            print("=" * 70)

            if result["page_type"] == "LOGIN":

                print(
                    "ROUTE TO → MODEL 1"
                )

                print(
                    "EfficientNetV2B0 Login Page Detector"
                )

            else:

                print(
                    "ROUTE TO → MODEL 2"
                )

                print(
                    "EfficientNetV2B0 Webpage Detector"
                )

            print("=" * 70)


        except Exception as error:

            print()
            print(
                "ERROR while opening webpage:"
            )

            print(error)


        finally:

            browser.close()


if __name__ == "__main__":

    url = input(
        "Enter website URL: "
    ).strip()

    if not url.startswith(
        ("http://", "https://")
    ):

        url = (
            "https://" + url
        )

    test_url(
        url
    )