# C3 · Micro Visual Cue Detection

**Component 3 of R26-CS-019, an adaptive multi-layer phishing detection framework.**
Author: M. A. O. Rashmika (IT22546234), SLIIT.

C3 judges a website **by how it looks**. It opens the URL in a browser, takes a 1366 × 768
screenshot, decides whether the page is a login page, and sends the screenshot to one of two
EfficientNetV2B0 CNNs.

| | |
|---|---|
| **Live API** | https://c3-visual.whitewater-a91f119b.koreacentral.azurecontainerapps.io (Azure Container Apps) |
| **Group website** | https://r26-cs-019.onrender.com (C3 is the "Visual appearance" layer) |
| **API docs** | `/docs` on the live API |

## How it works

```
URL ─► public-address guard ─► browser capture (profile c3-capture-v2) ─► capture quality gate
                                                                               │
              ┌──────────── page-type router (rendered HTML + URL) ◄───────────┘
              ▼                                          ▼
   LOGIN ─► Model 1 (login pages)              GENERAL ─► Model 2 (general webpages)
            384 × 160 top viewport                        384 × 224 aspect-fit, white pad
            threshold 0.466                               threshold 0.392
              └──────────────► PHISHING / LEGITIMATE + score + evidence ◄──────┘
```

1. **Public-address guard.** Only public http(s) sites are opened. Private, local and
   cloud-internal addresses are refused for the page and for every file it loads.
2. **Capture.** See [Capture profile](#capture-profile) below.
3. **Quality gate.** If the browser received an HTTP error, a bot-protection or verification
   page, or an almost blank screen, C3 **does not guess**. It answers "capture needs review"
   (HTTP 422) and the group fusion skips C3 for that URL.
4. **Router.** Login URL patterns or a login form in the rendered HTML route to Model 1;
   everything else routes to Model 2.
5. **Model and threshold.** The frozen checkpoints are verified by SHA-256 at start-up. The
   service refuses to start if either file differs from the approved one.

| Model | Checkpoint (SHA-256 prefix) | Input | Threshold |
|---|---|---|---|
| 1 · login pages | `model1_efficientnetv2b0_selected_best.keras` (`6be68e84`) | 384 × 160, width-scale, top crop / bottom white pad (Lanczos) | 0.466 |
| 2 · general pages | `model2_effnetv2b0_stage2_best.keras` (`466ef586`) | 384 × 224, bilinear aspect fit, centred white pad | 0.392 |

## Capture profile

The hosted API **and** the local interface call the same function,
`hosting_diagnostics.capture_with_diagnostics`, so a URL is photographed the same way on
Azure and on the laptop.

| Setting | Value |
|---|---|
| Browser | Playwright full Chromium, new headless mode |
| Identity | standard desktop Chrome user agent, `en-US` locale and `Accept-Language`, `Asia/Colombo` time zone |
| Viewport | 1366 × 768, device scale 1, viewport screenshot (not full page) |
| Timing | `domcontentloaded`, 30 s navigation timeout, then 10 s render wait |
| Fonts (Docker) | Playwright font set plus `fonts-noto-core` (Sinhala, Tamil and other scripts) |

Each value can be overridden with an environment variable (`C3_CAPTURE_WAIT_MS`,
`C3_BROWSER_CHANNEL`, `C3_USER_AGENT`, `C3_LOCALE`, `C3_TIMEZONE`, `C3_HEADLESS`). The
active profile is reported by `/health` and recorded in every run.

**Why this profile matters.** The first hosted version used Playwright's headless-shell
build, which identifies itself as `HeadlessChrome`. From a cloud address, many large sites
refused it (HTTP 403) or served a challenge page with HTTP 200, and the CNN then classified
the challenge page instead of the website (for example, Amazon's robot check scored 0.997).
On 57 URLs re-tested on 18 Oct 2026 against laptop results:

| | Before (`26139e4`) | After (`c3-capture-v2`) |
|---|---|---|
| URLs the hosted C3 could assess | 32 / 57 | 43 / 57 |
| Same verdict as the laptop, where both answered | 27 / 31 (87 %) | 41 / 43 (95 %) |
| Legitimate sites called phishing because of a block page | 2 | 0 |

Some sites still block cloud addresses by IP reputation (for example Oracle). C3 reports
those as "capture needs review" rather than guessing.

## API

`POST /predict` with `{"url": "https://example.com"}`

| HTTP | Meaning | Body |
|---|---|---|
| 200 | Classified | `prediction`, `predicted_label` (1 = phishing), `phishing_probability`, `page_type`, `router_confidence`, `model_used`, `threshold`, `run_id`, `diagnostics` |
| 400 | Invalid, private or non-existent address | `detail` message |
| 422 | Capture needs review (HTTP error, block/verification page, blank) | `status: capture_needs_review`, `capture.warnings` |
| 502 | Browser could not load the page | `status: capture_failed` |
| 503 | Models not loaded yet | |

`GET /health` returns readiness, build revision, library versions, capture settings and the
verified checkpoint hashes. `GET /evidence/{run_id}/{file}` returns the screenshot, model
input and result for a run, only when the `C3_EVIDENCE_TOKEN` secret is configured and sent
in the `X-Evidence-Token` header.

## Run locally

```powershell
cd "full  pp2 module"
.\.venv\Scripts\python.exe -m playwright install chromium   # once
# API (same as Azure)
cd "Micro Visual Cue Component"
..\.venv\Scripts\python.exe -m uvicorn integration.visual_api:app --port 8000
# or the research interface with Grad-CAM evidence (see interface/README.md)
```

## Deployment

`git push` to `main` → GitHub Actions (`.github/workflows/build-c3-image.yml`) builds the
Docker image and publishes `ghcr.io/r26-cs-019/c3-micro-visual-cue-detector:<commit SHA>` →
Azure Container App `c3-visual` (2 vCPU, 4 GiB, scale 0–1) is updated to that tag as a new
revision. `/health` → `deployment.build_revision` shows which commit is live.

## Results (from saved evaluation artefacts)

| Evaluation | Accuracy | Recall | FPR | F1 |
|---|---|---|---|---|
| Model 2, fixed Phish360 test (2,765 images) | 91.36 % | 95.66 % | 12.27 % | 0.9102 |
| Model 2, preliminary external Phish-Iris (2,449 images) | 81.91 % | 92.68 % | 26.57 % | 0.8187 |
| Model 1 (deployed `6be68e84`), diagnostic login split (428 images) | 81.07 % | 75.70 % | 13.55 % | 0.8000 |

## Limitations

- On the login split, a classifier using only the original image dimensions reaches 100 %,
  which exposes a dataset shortcut and limits claims about learned visual cues.
- The login test split has already been inspected, so it is diagnostic, not a fresh holdout.
- End-to-end routing accuracy, calibration and unseen-brand generalisation are not established.
- Websites change (pop-up adverts, banners, regional content), so a live score can differ
  from an earlier scan of the same URL.
- The address guard checks DNS before navigation but does not pin connections.

See `research_paper/` for the full write-up and `research_suite/` for the evidence tools.
