# PP2 research evidence suite

This suite adds reproducible research experiments around the two existing deployed EfficientNetV2B0 checkpoints. It does not retrain them. Model 1 now uses the Step 15 baseline retained by the validation-controlled continuation comparison; older continuation experiments remain separately identified by their checkpoint hashes. `models.json` fixes checkpoint hashes, thresholds and preprocessing. Paths resolve relative to the full project root; use `--project-root` or `PP2_PROJECT_ROOT` when needed.

## Start here

Open `results/index.html` for links to saved evidence. From PowerShell in `full  pp2 module`:

```powershell
.\research_suite\run.cmd --help
.\research_suite\run.cmd evaluate --manifest .\research_suite\examples\login_pages.csv --exploratory --gradcam
.\research_suite\run.cmd mask --manifest .\research_suite\examples\smoke.csv --regions .\research_suite\examples\regions.json
.\research_suite\run.cmd baseline --manifest .\research_suite\examples\model1_dataset.csv --model 1
```

The launcher uses the existing TensorFlow environment at `Desktop\MicroVisualPhishingDetector\venv`. Set `PP2_PYTHON` to another Python executable if needed. `requirements.txt` records the tested libraries. It does not change your PyTorch environment. The older tested Keras environment needs a narrow metadata compatibility loader for these newer archives; all stored weights are restored from unmodified originals and adaptation is recorded.

All new runs get unique directories. Models are read-only. Generated plots/maps are explanations of frozen models, not training improvements.

## 1. Integrated Grad-CAM and evaluation

`evaluate --gradcam` exports full source screenshots, actual cropped/padded inputs, exact float tensors, both binary class heatmaps and overlays, checkpoint/software details, predictions.csv, per_brand.csv, confusion_matrices.csv, report.json, report.md, and an HTML gallery. The explanation graph must reproduce the ordinary checkpoint score. Empty class maps are explicitly recorded. Map colour intensity is normalized within each image and cannot be compared across samples. Grad-CAM uses signed binary logits to avoid sigmoid saturation. Reference: https://keras.io/examples/vision/grad_cam/

The manifest columns are in `core.FIELDS`. Required fields are image, model (1/2), label (LEGITIMATE/PHISHING). Source labels, brand identities and provenance need review. Use `--exploratory` for unverified material; results retain that limitation. Do not include training examples as independent evaluation evidence.

## 2. Reproducible controlled masking

`mask` reads a manifest and JSON region definitions keyed by sample_id. Coordinates are `[left, top, right-exclusive, bottom-exclusive]` on the actual model input, not the full original screenshot. Each target has one or more equal-size controls. The tool rejects out-of-bounds rectangles, mismatched control sizes and controls overlapping their target. Regions should be fixed before reviewing scores. Choose controls unrelated to all relevant cues; geometry checks cannot verify their meaning.

Masks modify exact float tensors independently, preserving the baseline without PNG quantization effects. Output includes coordinates, changed pixel counts, paired target-versus-control changes, images and provenance. Weak blank-background controls are flagged. White masking introduces artificial content and cannot by itself establish brand memorization or independence. Repeat across brands, correct predictions, errors and control placements. Preview PNGs are rounded; exact inputs are retained for Grad-CAM runs.

## 3. Same-brand evaluation

Use `evaluate --mode same-brand`. Supply multiple legitimate and phishing examples per reviewed canonical brand. Missing classes are flagged. Report recall, false-positive rate, precision, F1 and counts rather than only accuracy. Wilson intervals are descriptive and assume independent samples; repeated templates make those assumptions inappropriate. The included 20-image login set is exploratory and has only one legitimate image per brand. It manually selects Model 1; it does not test URL/HTML routing.

## 4. Unseen-brand evaluation

First prepare an inventory:

```powershell
.\research_suite\run.cmd bootstrap --images 'C:\path\to\login pages'
```

Review **every** training/validation brand in the generated inventory, setting brand_verified only after checking. Use consistent canonical names and handle aliases (for example Microsoft/Office/Outlook) explicitly. Preserve the `.checkpoint.json` sidecar and inventory_sha256 values. Review evaluation labels/brands too.

Run `audit --manifest EVALUATION.csv --reference TRAINING_INVENTORY.csv`. Resolve/exclude candidate overlaps and rerun. Then:

```powershell
.\research_suite\run.cmd evaluate --mode unseen --manifest EVALUATION.csv --training-inventory TRAINING_INVENTORY.csv --audit-report AUDIT_FOLDER\audit.json --gradcam
```

The tool refuses incomplete inventories, checkpoint mismatches, coverage mismatches, overlapping brands, exact source overlap and unresolved audit candidates. This depends on honest, complete metadata; it is not an automatic discovery of every brand. Existing datasets currently lack that complete inventory, so no unseen-brand performance claim is supplied.

`split-brands --manifest DATA.csv --test-brands BRAND_A --validation-brands BRAND_B` creates a proposal with disjoint brands and checks known duplicate/template groups and label presence. **It does not retrain the CNN.** To make a new split applicable to a CNN, train a new checkpoint on that split and register its full training manifest, hash, threshold and inventory in a separate config. Do not use an old checkpoint to claim it was trained on a newly proposed split.

## 5. Baseline comparison

`baseline` compares the registered CNN with two simple logistic classifiers: spatial RGB means/colour histograms, and original width/height/aspect ratio. Both fit standardization on TRAIN only and select their thresholds by validation F1. The CNN uses its locked configured threshold. No test tuning occurs. Classifier weights and thresholds are saved as NPZ.

It requires the exact configured checkpoint's original TRAIN and VALIDATION membership and rejects known hash/template overlap. The TEST images are shared by all methods. Augmentation, class weighting and model selection differ and are reported as limitations. The included Model 1 test split was already inspected and is diagnostic, **not fresh final evaluation evidence**. Model 2 data paths may still point to the original external dataset; resolve their location before running its baseline. Other baseline settings are fixed rather than optimized on test data.

## 6. Leakage and shortcut audit

`audit` recomputes source hashes, exact prepared-pixel hashes and dHash candidates on model inputs; checks manifest hashes; and exports source/dimension-by-label counts and known brand overlap. dHash candidates require visual review, not automatic deletion. No candidate does not prove independence. Different page versions, browser decorations, blank/error pages, cookies, cropping, padding and collection sources require human review. Dimensions-only baseline performance exposes metadata predictability; it does not prove the CNN uses the original dimensions.

## 7. Research report exporter

Evaluation and masking commands export CSV, JSON, Markdown and HTML, including confusion matrices, per-brand metrics, hashes, settings and limitations. Baseline reports include method comparison tables; audits include candidate and association spreadsheets. Use these artifacts for the viva, including negative findings. Keep prior runs intact and never tune against the reserved final test set.

## Verification

`test_research.py` tests confusion matrices, missing-class handling, logistic fitting, template-split leakage rejection and unseen-inventory requirements. `verification.json` records actual integration runs and numerical parity with previously archived scores. Passing these checks validates the tools; it does not establish generalization or eliminate dataset bias.
