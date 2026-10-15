"""Route webpage screenshots to the correct visual model."""

from __future__ import annotations

import re
from html import unescape
from pathlib import Path
from urllib.parse import urlparse


# ============================================================
# ROUTER TERMS
# ============================================================

# "Login" and "Sign in" visible text are intentionally not used
# because general homepages commonly contain these navigation links.
LOGIN_TEXT_TERMS: list[str] = []

PASSWORD_TERMS = [
    "password",
    "passcode",
    "forgot password",
    "forgot your password",
]

USERNAME_TERMS = [
    "username",
    "user name",
    "user id",
    "userid",
    "email",
    "e-mail",
    "phone",
    "skype",
    "registration",
    "registration no",
    "registration number",
    "reg no",
    "student id",
    "student number",
]

AUTH_TERMS = [
    "remember me",
    "remember user",
    "verify your identity",
    "verify identity",
    "authentication",
    "access your account",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def contains_any(
    text: str,
    terms: list[str],
) -> bool:
    """Return True when the text contains one of the terms."""

    return any(
        term in text
        for term in terms
    )


def get_visible_text(
    html: str,
) -> str:
    """Extract approximate visible text from HTML."""

    if not isinstance(html, str):
        return ""

    cleaned_html = re.sub(
        r"<script\b[^>]*>.*?</script>",
        " ",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    cleaned_html = re.sub(
        r"<style\b[^>]*>.*?</style>",
        " ",
        cleaned_html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    cleaned_html = re.sub(
        r"<!--.*?-->",
        " ",
        cleaned_html,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        cleaned_html,
    )

    text = unescape(text)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()


def get_input_tags(
    html: str,
) -> list[str]:
    """Extract input elements from HTML."""

    if not isinstance(html, str):
        return []

    return re.findall(
        r"<input\b[^>]*>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )


def is_password_input(
    input_tag: str,
) -> bool:
    """Check whether an input element is a password field."""

    return bool(
        re.search(
            r"\btype\s*=\s*[\"']?\s*password\b",
            input_tag,
            flags=re.IGNORECASE,
        )
    )


def is_username_input(
    input_tag: str,
) -> bool:
    """Check whether an input collects a user identifier."""

    tag_lower = input_tag.lower()

    # Explicit email input can still act as a user identifier.
    # Email is NOT required for the final router decision.
    if re.search(
        r"\btype\s*=\s*[\"']?\s*email\b",
        tag_lower,
        flags=re.IGNORECASE,
    ):
        return True

    # Username or email autocomplete attribute
    if re.search(
        r"\bautocomplete\s*=\s*[\"'][^\"']*"
        r"(username|email)[^\"']*[\"']",
        tag_lower,
        flags=re.IGNORECASE,
    ):
        return True

    # Check name, ID, placeholder and other attributes
    return contains_any(
        tag_lower,
        USERNAME_TERMS,
    )


# ============================================================
# URL LOGIN DETECTION
# ============================================================

def detect_login_from_url(
    url: str,
) -> bool:
    """
    Detect clear login/sign-in indicators in the URL.

    Examples detected:

        https://example.com/login
        https://example.com/login/
        https://example.com/login/index.php
        https://example.com/login.php
        https://example.com/user/login
        https://example.com/auth/login
        https://example.com/signin
        https://login.example.com/

    We intentionally avoid searching for the word "login"
    anywhere in the URL because a normal article such as:

        /blog/login-security-guide

    should not automatically become a LOGIN page.
    """

    if not url:
        return False

    try:
        parsed = urlparse(
            url.strip()
        )

        hostname = (
            parsed.hostname or ""
        ).lower()

        path = (
            parsed.path or ""
        ).lower()

        # ====================================================
        # 1. HOSTNAME
        # ====================================================

        hostname_parts = [
            part
            for part in hostname.split(".")
            if part
        ]

        login_hostname_terms = {
            "login",
            "signin",
            "sign-in",
            "log-in",
        }

        if any(
            part in login_hostname_terms
            for part in hostname_parts
        ):
            return True

        # ====================================================
        # 2. URL PATH
        # ====================================================
        #
        # Detect:
        #
        # /login
        # /login/
        # /login.php
        # /login.html
        # /user/login
        # /auth/signin
        #
        # But avoid:
        #
        # /blog/login-security-guide
        # ====================================================

        login_path_pattern = re.compile(
            r"(?:^|/)"
            r"(?:login|log-in|signin|sign-in)"
            r"(?:/|\.|$)",
            flags=re.IGNORECASE,
        )

        if login_path_pattern.search(
            path
        ):
            return True

        return False

    except Exception:
        return False


# ============================================================
# PAGE TYPE DETECTION
# ============================================================

def detect_page_type(
    html: str,
    url: str = "",
) -> dict[str, object]:
    """
    Detect whether the webpage should use Model 1 or Model 2.

    Routing order:

    1. Clear login indicator in URL
       -> Model 1

    2. Otherwise inspect rendered HTML.

    The router does NOT determine whether the website is
    phishing. It only selects the appropriate visual model.
    """

    if html is None:
        html = ""

    if not isinstance(html, str):
        html = str(html)

    if url is None:
        url = ""

    if not isinstance(url, str):
        url = str(url)

    html_lower = html.lower()

    visible_text = get_visible_text(
        html
    )

    input_tags = get_input_tags(
        html
    )

    score = 0

    reasons: list[str] = []


    # ========================================================
    # 1. URL LOGIN CHECK
    # ========================================================

    url_login_detected = (
        detect_login_from_url(
            url
        )
    )

    if url_login_detected:

        score += 10

        reasons.append(
            "login_indicator_in_url"
        )


    # ========================================================
    # 2. PASSWORD INPUT
    # ========================================================

    password_input = any(
        is_password_input(tag)
        for tag in input_tags
    )

    if password_input:

        score += 6

        reasons.append(
            "password_input"
        )


    # ========================================================
    # 3. USERNAME / ACCOUNT IDENTIFIER INPUT
    # ========================================================

    username_input = any(
        is_username_input(tag)
        for tag in input_tags
    )

    if username_input:

        score += 2

        reasons.append(
            "username_or_email_input"
        )


    # ========================================================
    # 4. LOGIN TEXT
    # ========================================================
    #
    # This remains False with the current configuration.
    #
    # Ordinary "Login" and "Sign in" navigation text should
    # NOT influence routing.
    # ========================================================

    login_text = contains_any(
        visible_text,
        LOGIN_TEXT_TERMS,
    )


    # ========================================================
    # 5. PASSWORD TEXT
    # ========================================================

    password_text = contains_any(
        visible_text,
        PASSWORD_TERMS,
    )

    if password_text:

        score += 1

        reasons.append(
            "password_text"
        )


    # ========================================================
    # 6. AUTHENTICATION TEXT
    # ========================================================

    authentication_text = contains_any(
        visible_text,
        AUTH_TERMS,
    )

    if authentication_text:

        score += 1

        reasons.append(
            "authentication_text"
        )


    # ========================================================
    # 7. FORM ELEMENT
    # ========================================================

    form_present = bool(
        re.search(
            r"<form\b",
            html_lower,
            flags=re.IGNORECASE,
        )
    )

    if form_present:

        reasons.append(
            "form_present"
        )


    # ========================================================
    # 8. HTML LOGIN FORM
    # ========================================================
    #
    # HTML selects Model 1 only when ALL are present:
    #
    # 1. Form
    # 2. Password input
    # 3. Username/account identifier
    #
    # This prevents normal homepages containing hidden
    # account elements from easily becoming LOGIN.
    # ========================================================

    login_form_present = (
        form_present
        and password_input
        and username_input
    )


    # ========================================================
    # FINAL ROUTING DECISION
    # ========================================================


    # --------------------------------------------------------
    # RULE 1:
    # Clear login URL
    # --------------------------------------------------------

    if url_login_detected:

        page_type = "LOGIN"

        confidence = "HIGH"


    # --------------------------------------------------------
    # RULE 2:
    # Strong HTML login form
    # --------------------------------------------------------

    elif login_form_present:

        page_type = "LOGIN"

        confidence = "HIGH"


    # --------------------------------------------------------
    # OTHERWISE:
    # General webpage
    # --------------------------------------------------------

    else:

        page_type = "GENERAL_WEBPAGE"

        if score == 0:
            confidence = "HIGH"

        else:
            confidence = "MEDIUM"


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "page_type":
            page_type,

        "confidence":
            confidence,

        "score":
            score,

        "url_login_detected":
            url_login_detected,

        "password_input":
            password_input,

        "username_input":
            username_input,

        "login_text":
            login_text,

        "password_text":
            password_text,

        "authentication_text":
            authentication_text,

        "form_present":
            form_present,

        "login_form_present":
            login_form_present,

        "reasons":
            reasons,
    }


# ============================================================
# LOCAL HTML / URL TEST
# ============================================================

def main() -> None:

    print("=" * 70)

    print(
        "MICRO VISUAL CUE - PAGE TYPE ROUTER"
    )

    print("=" * 70)

    print()


    # ========================================================
    # OPTIONAL URL
    # ========================================================

    url = input(
        "Enter webpage URL (optional): "
    ).strip()


    # ========================================================
    # HTML FILE
    # ========================================================

    html_path_text = input(
        "Enter HTML file path: "
    ).strip().strip('"')


    html_path = Path(
        html_path_text
    )


    if not html_path.is_file():

        raise SystemExit(
            "\nERROR: HTML file does not exist."
        )


    html = html_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


    # ========================================================
    # ROUTE
    # ========================================================

    result = detect_page_type(
        html,
        url,
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print()

    print("=" * 70)

    print(
        "PAGE TYPE RESULT"
    )

    print("=" * 70)

    print()


    print(
        f"Page type           : "
        f"{result['page_type']}"
    )


    print(
        f"Confidence          : "
        f"{result['confidence']}"
    )


    print(
        f"Score               : "
        f"{result['score']}"
    )


    print(
        f"URL login detected  : "
        f"{result['url_login_detected']}"
    )


    print(
        f"Password input      : "
        f"{result['password_input']}"
    )


    print(
        f"Username input      : "
        f"{result['username_input']}"
    )


    print(
        f"Login text          : "
        f"{result['login_text']}"
    )


    print(
        f"Password text       : "
        f"{result['password_text']}"
    )


    print(
        f"Authentication text : "
        f"{result['authentication_text']}"
    )


    print(
        f"Form present        : "
        f"{result['form_present']}"
    )


    print(
        f"Login form present  : "
        f"{result['login_form_present']}"
    )


    print()

    print(
        "Reasons:"
    )


    if result["reasons"]:

        for reason in result["reasons"]:

            print(
                f"  - {reason}"
            )

    else:

        print(
            "  - No login evidence"
        )


    print()

    print("=" * 70)


    if (
        result["page_type"]
        == "LOGIN"
    ):

        print(
            "ROUTE TO -> MODEL 1"
        )

        print(
            "EfficientNetV2B0 Login Page Detector"
        )

    else:

        print(
            "ROUTE TO -> MODEL 2"
        )

        print(
            "EfficientNetV2B0 General Webpage Detector"
        )


    print("=" * 70)


if __name__ == "__main__":
    main()