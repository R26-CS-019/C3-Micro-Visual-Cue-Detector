# Connect the existing C3 evidence button

This is a small addition based on repository commit `4479a3e5e60e495d95765e5854b120073421d17c`. It uses the capture diagnostics already present in that commit. It does **not** replace capture settings, routing, preprocessing, trained checkpoints, thresholds, the Dockerfile, or the other components.

## Commit these backend files

1. `Micro Visual Cue Component/src/gradcam_evidence.py` (new)
2. `Micro Visual Cue Component/src/hosting_diagnostics.py` (only adds explanation filenames to the existing download allowlist)
3. `Micro Visual Cue Component/integration/visual_api.py` (generates explanation under the existing prediction lock and adds the `evidence` response field)

This document may also be committed. Existing dependencies already include TensorFlow, NumPy and Pillow; no dependency or model downloads are needed for this addition.

## What happens for each successful prediction

The existing pipeline captures the webpage once and saves the exact float32 input. The API reuses the selected **already-loaded model** and this saved tensor to generate both class overlays. Before generating maps, it checks that inference on that tensor reproduces the returned score. It also verifies the explanation forward path against the model output. There is no new browser capture, classifier training or weight update.

Files are stored in the existing run folder:

```text
Micro Visual Cue Component/results/runs/<run_id>/
    screenshot.png
    model_input.png
    model_input_float.npy
    phishing_overlay.png
    legitimate_overlay.png
    phishing_heatmap.npy
    legitimate_heatmap.npy
    explanation.json
    result.json
```

If explanation generation fails, the classifier result is retained and `diagnostics.gradcam.status` is `unavailable`. Do not display a stale map from another run. A disabled explanation (`C3_GRADCAM_ENABLED=0`) has status `disabled`. The default is enabled.

## New response fields for the existing button

All existing successful prediction fields remain. The response adds:

```json
{
  "run_id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "evidence": {
    "run_id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "requires_header": "X-Evidence-Token",
    "screenshot": "/evidence/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/screenshot.png",
    "model_input": "/evidence/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/model_input.png",
    "gradcam_phishing": "/evidence/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/phishing_overlay.png",
    "gradcam_legitimate": "/evidence/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/legitimate_overlay.png",
    "explanation": "/evidence/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/explanation.json"
  }
}
```

These are paths on the **Azure API**, not the Render website. The ready/unavailable status and explanation metadata are in `diagnostics.gradcam`. When unavailable or disabled, the Grad-CAM paths are null; screenshot and input paths remain available.

## Static-site integration (inline images; preferred for the current group site)

The successful `/predict` response now includes `visual_evidence`, so a static website does not need a server-side proxy or `X-Evidence-Token` to display the images from its prediction:

```json
{
  "visual_evidence": {
    "screenshot_jpeg_base64": "...plain base64...",
    "model_input_png_base64": "...plain base64...",
    "gradcam": {
      "status": "ready",
      "phishing_overlay_png_base64": "...plain base64...",
      "legitimate_overlay_png_base64": "...plain base64..."
    }
  }
}
```

The API returns no `data:` prefix. The existing button handler should read `visual_evidence` from the C3 response and set its image sources like this (use your actual DOM elements):

```javascript
const v = c3Response.visual_evidence;
screenshotImage.src = `data:image/jpeg;base64,${v.screenshot_jpeg_base64}`;
inputImage.src = `data:image/png;base64,${v.model_input_png_base64}`;
if (v.gradcam.status === 'ready') {
  phishingImage.src = `data:image/png;base64,${v.gradcam.phishing_overlay_png_base64}`;
  legitimateImage.src = `data:image/png;base64,${v.gradcam.legitimate_overlay_png_base64}`;
}
```

Check fields for null before setting an image source. For `unavailable` or `disabled`, both overlay values are null; retain the screenshot/input view and show the explanation status. If the site uses a Content Security Policy, its image policy must permit `data:` images. Never insert the strings with `innerHTML`.

The existing protected `evidence` paths and download endpoint are unchanged. No evidence token or secret is included in `visual_evidence`. The static frontend should use the inline data rather than call protected download URLs. If an aggregator forwards only scalar fields, it must forward `visual_evidence` too.

For HTTP **422 capture_needs_review**, the existing FastAPI error envelope is preserved; the review image is at `response.detail.visual_evidence.screenshot_jpeg_base64`. This response still has no classifier verdict. **502 behaviour is unchanged** and no inline images are added to it.

Screenshot copies are JPEG quality 70, maximum 900px wide. Overlay copies are PNG, maximum 600px wide. Additional downscaling keeps each display image under 225,000 binary bytes (and its base64 under 300,000 characters). The model-input PNG is returned byte-for-byte, at its original 384px width and below 300,000 binary bytes; base64 adds its standard one-third transport overhead. Saved research images, tensors and model outputs are never resized or overwritten by this display encoding.

## Optional server-backed integration using protected downloads

The following alternative is only for sites that have a backend. The current static group website should use the inline fields above instead. The main website source was not supplied, so its click handler is not modified by this patch. A server-backed integration would:

1. Preserve C3's `run_id`, `evidence` and `diagnostics.gradcam` fields in the group analysis response. An aggregator that copies only the old scalar fields must forward these new fields too.
2. Set a long random `C3_EVIDENCE_TOKEN` as a server secret on **Azure** and in the **Render backend** that retrieves images. Keep this secret out of browser JavaScript, GitHub commits and URLs. If an evidence token is already configured, preserve it and configure the corresponding Render server secret.
3. Add/adjust the Render backend evidence proxy. For example, the browser asks its own backend for `/api/c3/evidence/<run_id>/screenshot.png`; the Render backend requests the corresponding Azure `/evidence/<run_id>/screenshot.png` with `X-Evidence-Token`, then streams the image back. Validate the run ID and use a filename allowlist; don't accept arbitrary target URLs from the browser. Apply the platform's existing access rules to this route.
4. Have the existing button open its current modal/panel. Display the **Captured screenshot** and the **Phishing Grad-CAM** from the same run. Optionally include **Actual model input** and **Legitimate Grad-CAM** tabs.
5. Render images using the same-origin proxy paths. A direct browser `<img src="https://AZURE/evidence/...">` cannot supply the required secret header. Don't remove API protection or embed the secret in a query string to work around this.
6. Show a clear loading/error state. For status `unavailable`/`disabled`, retain the screenshot and show that Grad-CAM is unavailable. Do not treat this as a failed classifier prediction. Existing HTTP 422/502 capture errors remain unchanged and have no CNN explanation.

An image response can be PNG; the manifest/metadata response is JSON. Preserve the successful content type and return `Cache-Control: no-store` from the proxy. Keep remote error responses separate from image responses so an error body isn't displayed as a broken PNG.

The API currently permits cross-origin request headers only for prediction JSON. The proxy approach does not require widening that policy or sending evidence secrets to the public browser.

## Deploy and verify

1. Review/copy the three changed files into the repository and commit. The patch archive is based on the commit above; merge rather than overwrite if newer source changes exist.
2. Wait for the existing image build to succeed. Deploy the image tagged with the new full commit SHA to Azure. A push/build alone does not change a running Azure revision. Verify the commit via `/health`.
3. Keep the existing working capture/model environment settings. `C3_GRADCAM_ENABLED` defaults to `1`; no other new Azure variable is required except the evidence token if not already configured.
4. Allow extra request time for two Grad-CAM passes, especially on the first call. Review the aggregator's per-component timeout and Azure CPU/memory; use a bounded timeout large enough for the existing browser capture plus explanation time. Validate the deployed timing instead of assuming the earlier C3 timings still apply.
5. Make one prediction and inspect the returned `evidence` and `diagnostics.gradcam` fields. Download all three PNGs through the proxy and confirm their run ID matches the result card.

Investigator-only PowerShell example (token comes from the environment):

```powershell
$c3Api = 'https://YOUR-C3-AZURE-API'
$c3Run = 'PASTE_RUN_ID'
$c3Headers = @{ 'X-Evidence-Token' = $env:C3_EVIDENCE_TOKEN }
Invoke-WebRequest -Uri "$c3Api/evidence/$c3Run/screenshot.png" -Headers $c3Headers -OutFile '.\captured.png'
Invoke-WebRequest -Uri "$c3Api/evidence/$c3Run/phishing_overlay.png" -Headers $c3Headers -OutFile '.\gradcam.png'
```

## Storage and interpretation

Evidence remains on the container's local filesystem, as in the current backend. It is ephemeral and not shared across replicas. A download can return 404 after a restart or on a different replica. Use the deployment's shared storage strategy for durable/multi-replica access, or retain a single replica for an agreed investigation. This patch doesn't change scaling, storage accounts or traffic settings. The service owner must manage retention and archive evidence that matters.

Grad-CAM is a coarse, class-specific sensitivity visualisation. Maps are normalised separately; colours are not probabilities. An empty map can occur and is recorded in `explanation.json`. These maps do not prove the model learned phishing cues or avoided brand memorisation. They explain only the captured input associated with their run.
