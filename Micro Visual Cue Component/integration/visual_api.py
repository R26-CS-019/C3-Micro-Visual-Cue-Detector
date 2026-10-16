from pathlib import Path
from contextlib import asynccontextmanager
from threading import Lock
import sys

import os

from fastapi import FastAPI, HTTPException
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

    models_loaded = (
        model1 is not None
        and
        model2 is not None
    )

    return {

        "component":
            "micro_visual_cue",

        "status":
            "online",

        "model1_loaded":
            model1 is not None,

        "model2_loaded":
            model2 is not None,

        "ready":
            models_loaded,
    }


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

            "status":
                "success",
        }


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