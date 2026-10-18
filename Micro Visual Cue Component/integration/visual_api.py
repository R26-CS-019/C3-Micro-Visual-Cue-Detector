from pathlib import Path
from contextlib import asynccontextmanager
from threading import Lock
import sys

import os

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware  # R26 hosting fix
from pydantic import BaseModel


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR)
    )


# ============================================================
# IMPORT EXISTING VISUAL COMPONENT
# ============================================================

from visual_component import (
    load_models,
    analyze_url,
    require_public_url,  # R26 hosting fix
)


from hosting_diagnostics import (
    CaptureQualityError, CaptureNavigationError, deployment_info, evidence_file,
    model_identity,
)
from visual_component import MODEL1_PATH, MODEL2_PATH, MODEL1_THRESHOLD, MODEL2_THRESHOLD, RESULTS_DIR
from gradcam_evidence import attach_gradcam, capture_review_visual_evidence

# ============================================================
# GLOBAL MODEL VARIABLES
# ============================================================

model1 = None
model2 = None


# Prevent multiple visual predictions from running at
# exactly the same time during initial integration testing.
prediction_lock = Lock()


# ============================================================
# API STARTUP / SHUTDOWN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model1
    global model2

    print("=" * 80)
    print("STARTING MICRO VISUAL CUE API")
    print("=" * 80)

    print()
    print("Loading Model 1 and Model 2...")

    model1, model2 = load_models()

    print()
    print("Models loaded successfully.")
    print("Visual API is ready.")

    yield

    print()
    print("Shutting down Visual API...")


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Micro Visual Cue Phishing Detection API",

    description=(
        "Visual phishing detection using page-type routing "
        "and two EfficientNetV2B0 models."
    ),

    version="1.0",

    lifespan=lifespan,
)


# R26 hosting fix: let the group website call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("C3_ALLOWED_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ============================================================
# REQUEST FORMAT
# ============================================================

class URLRequest(BaseModel):

    url: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    ready = model1 is not None and model2 is not None
    return {"component": "micro_visual_cue", "status": "online", "ready": ready,
            "model1_loaded": model1 is not None, "model2_loaded": model2 is not None,
            "deployment": deployment_info(),
            "models": {"1": model_identity(1, MODEL1_PATH, MODEL1_THRESHOLD),
                       "2": model_identity(2, MODEL2_PATH, MODEL2_THRESHOLD)} if ready else {}}


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: URLRequest):

    global model1
    global model2

    # --------------------------------------------------------
    # CHECK MODELS
    # --------------------------------------------------------

    if (
        model1 is None
        or
        model2 is None
    ):

        raise HTTPException(
            status_code=503,
            detail="Visual models are not loaded."
        )


    # --------------------------------------------------------
    # CLEAN URL
    # --------------------------------------------------------

    url = request.url.strip()


    if not url:

        raise HTTPException(
            status_code=400,
            detail="URL cannot be empty."
        )


    # Automatically add HTTPS when user enters:
    #
    # example.com
    #
    # instead of:
    #
    # https://example.com
    #

    if not url.startswith(
        (
            "http://",
            "https://",
        )
    ):

        url = (
            "https://" + url
        )


    # --------------------------------------------------------
    # RUN EXISTING VISUAL COMPONENT
    # --------------------------------------------------------

    try:

        try:
            require_public_url(url)  # R26 hosting fix
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))

        with prediction_lock:

            result = analyze_url(
                url,
                model1,
                model2,
            )
            selected_model = model1 if result["model_used"] == "MODEL_1" else model2
            result = attach_gradcam(result, selected_model, RESULTS_DIR)


        # ----------------------------------------------------
        # CONVERT TEXT LABEL TO NUMERIC LABEL
        # ----------------------------------------------------

        predicted_label = (

            1

            if result["prediction"]
            == "PHISHING"

            else 0
        )


        # ----------------------------------------------------
        # STANDARD API RESPONSE
        # ----------------------------------------------------

        return {

            "component":
                "micro_visual_cue",

            "prediction":
                result["prediction"],

            "predicted_label":
                predicted_label,

            "phishing_probability":
                result[
                    "phishing_probability"
                ],

            "legitimate_probability":
                result[
                    "legitimate_probability"
                ],

            "page_type":
                result["page_type"],

            "router_confidence":
                result[
                    "router_confidence"
                ],

            "router_score":
                result[
                    "router_score"
                ],

            "model_used":
                result["model_used"],

            "model_name":
                result["model_name"],

            "threshold":
                result["threshold"],

            "requested_url":
                request.url,

            "processed_url":
                url,

            "diagnostics": result["diagnostics"],
            "run_id": result["run_id"],
            "evidence": result["evidence"],
            "visual_evidence": result["visual_evidence"],
            "status":
                "success",
        }


    except CaptureQualityError as error:
        detail = dict(error.public_detail)
        detail["visual_evidence"] = capture_review_visual_evidence(RESULTS_DIR, detail["run_id"])
        raise HTTPException(status_code=422, detail=detail)

    except CaptureNavigationError as error:
        raise HTTPException(status_code=502, detail=error.public_detail)

    except HTTPException:

        raise


    except Exception as error:
        print(f"Visual component failed: {error!r}")  # R26 hosting fix

        raise HTTPException(

            status_code=500,

            detail="Visual component failed.",  # R26 hosting fix
        )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "component":
            "Micro Visual Cue Detection",

        "api":
            "running",

        "predict_endpoint":
            "/predict",

        "health_endpoint":
            "/health",

        "documentation":
            "/docs",
    }

@app.get("/evidence/{run_id}/{filename}", include_in_schema=False)
def download_evidence(run_id: str, filename: str, x_evidence_token: str | None = Header(default=None)):
    import secrets
    expected = os.environ.get("C3_EVIDENCE_TOKEN", "")
    if not expected or not x_evidence_token or not secrets.compare_digest(expected, x_evidence_token):
        raise HTTPException(status_code=403, detail="Evidence access is disabled or not authorised.")
    try:
        path = evidence_file(RESULTS_DIR, run_id, filename)
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="Evidence not found on this replica.")
    return FileResponse(path, filename=filename, headers={"Cache-Control": "no-store"})
