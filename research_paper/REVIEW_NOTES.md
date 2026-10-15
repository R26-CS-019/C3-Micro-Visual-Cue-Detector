# Research paper review notes

The Word draft uses the supplied conference-template-a4.docx, its Times New Roman styles, A4 geometry, a full-width title/author block, and two-column body. The original template has not been changed. The paper includes three tables, two original figures, and five verified primary references. Heading numbering comes from the template. Figure PNG and vector PDF files are separate reusable assets.

Author: M.A Oshan Rashmika. Affiliation: Sri Lanka Institute of Information Technology, Sri Lanka. Email: Oshanrashmika180@gmail.com. No department or city was invented. No page limit was supplied.

## Items to resolve before submission

- The deployed login checkpoint differs from the checkpoint retained by validation selection. This discrepancy is disclosed in the paper; the implementation was not changed.
- The login test split has already been inspected and its dimensions-only baseline is perfectly predictive. Login metrics are diagnostic, not fresh final evaluation evidence.
- Phish-Iris results are preliminary. The archived evaluation ran despite a local cleaning note requiring overlap checking first; the result JSON explicitly records cross-dataset near-duplicate checking as incomplete. Complete exact and perceptual overlap checks before promoting these results to independent external evidence.
- Same-brand labels and brands need review, and six overlap candidates remain unresolved. Do not claim unseen-brand generalization.
- Document the original login collection, source URLs, acquisition procedure, dates, label provenance, and dataset permissions. Local counts differ from published dataset totals and are described as local audited copies.
- End-to-end routing accuracy, a matched single-model comparison, calibration, and latency are not available. They are not claimed.
- No research novelty or superiority over published systems is established by this draft. The contributions are scoped to the implemented component and its diagnostic evidence.
- Inspect the DOCX in Microsoft Word for final column flow, figure positioning, table wrapping, and conference-specific details before exporting a submission PDF. This build verifies document XML and package structure; it does not claim a Word rendering or final page count.

## Rebuild

From the module root: `.\.venv\Scripts\python.exe research_paper\build_paper.py`.

`research_paper.md` is an editable content copy. `build_paper.py` currently contains the content used to regenerate the DOCX. Existing training/evaluation artifacts and models are read only.

## Evidence map

- m1_training: `model 1 login pp2/reports/15_efficientnetv2b0_top_viewport_training/training_configuration.json`
- m1_selection: `model 1 login pp2/reports/17_efficientnetv2b0_continued_finetuning/continuation_comparison.json`
- m1_selected_test: `model 1 login pp2/reports/18_efficientnetv2b0_diagnostic_test/diagnostic_test_metrics.json`
- baseline: `research_suite/results/baseline_20261011T084658Z_aa75e1/report.json`
- m2_test: `new model 2 pp2/reports/model2_effnetv2b0/final_test/final_test_results.json`
- m2_external: `new model 2 pp2/reports/model2_effnetv2b0/external_evaluation/phish_iris/evaluation/phish_iris_external_results.json`
- same_brand: `research_suite/results/same-brand_20261011T084659Z_ed55d1/report.json`
- masking: `research_suite/results/masking_20261011T085645Z_43da93/report.json`
- verification: `research_suite/verification.json`
- registry: `research_suite/models.json`

Additional methods are supported by `Micro Visual Cue Component/src/visual_component.py`, `Micro Visual Cue Component/src/page_type_router.py`, `Micro Visual Cue Component/integration/visual_api.py`, both model training scripts, and the local split and cleaning summaries. `verification.json` records source hashes and recomputation of metrics from confusion counts, plus comparison against the archived Model 2 prediction CSVs.

## Verified external references

- Phishpedia: https://www.usenix.org/conference/usenixsecurity21/presentation/lin
- EfficientNetV2: https://proceedings.mlr.press/v139/tan21a.html
- Phish360 and CrossPhire citation: https://web.cs.hacettepe.edu.tr/~selman/phish360-dataset/
- Phish-Iris release: https://data.mendeley.com/datasets/tk4fkswtj5/1
- Grad-CAM: https://arxiv.org/abs/1610.02391
