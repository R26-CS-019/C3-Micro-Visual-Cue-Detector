"""Manifest validation, statistics, simple baselines and report export."""
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

FIELDS = ["sample_id", "image", "model", "label", "brand", "brand_verified", "label_verified", "label_evidence", "split", "template_group", "capture_source", "original_width", "original_height", "sha256", "inventory_sha256"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0]) if rows else fields or FIELDS
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest is empty")
    for i, row in enumerate(rows):
        for key in FIELDS:
            row.setdefault(key, "")
        row["sample_id"] = row["sample_id"] or f"sample_{i:05d}"
        row["model"] = int(row["model"])
        if row["model"] not in (1, 2) or row["label"] not in ("LEGITIMATE", "PHISHING"):
            raise ValueError(f"Invalid model or label: {row['sample_id']}")
        row["split"] = row["split"].upper()
        row["brand"] = row["brand"].strip().lower()
        image = Path(row["image"])
        row["image"] = str((path.parent / image).resolve() if not image.is_absolute() else image)
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id values must be unique")
    return rows


def verified(value):
    return str(value).lower() in {"true", "yes", "1"}


def metrics(labels, scores, threshold):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    if not len(y) or len(y) != len(p) or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)) or not set(y) <= {0, 1} or not 0 <= threshold <= 1:
        raise ValueError("Scores must be finite probabilities matching nonempty labels")
    predicted = p >= threshold
    tp = int(np.sum((y == 1) & predicted)); fn = int(np.sum((y == 1) & ~predicted))
    tn = int(np.sum((y == 0) & ~predicted)); fp = int(np.sum((y == 0) & predicted))
    ratio = lambda a, b: a/b if b else None
    precision = ratio(tp, tp+fp); recall = ratio(tp, tp+fn); specificity = ratio(tn, tn+fp)
    def wilson(successes, total):
        if not total:
            return None
        z = 1.959963984540054; rate = successes/total
        denominator = 1+z*z/total
        centre = (rate+z*z/(2*total))/denominator
        radius = z*np.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/denominator
        return [max(0.0, float(centre-radius)), min(1.0, float(centre+radius))]
    return {"images": len(y), "threshold": threshold, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": (tp+tn)/len(y), "precision": precision, "recall": recall,
            "specificity": specificity, "false_positive_rate": ratio(fp, fp+tn),
            "f1": ratio(2*tp, 2*tp+fp+fn),
            "balanced_accuracy": (recall+specificity)/2 if recall is not None and specificity is not None else None,
            "brier_score": float(np.mean((p-y)**2)),
            "recall_wilson_95": wilson(tp, tp+fn), "fpr_wilson_95": wilson(fp, fp+tn),
            "interval_note": "Wilson intervals assume independent observations; repeated templates violate this assumption."}


def grouped_metrics(rows):
    result = []
    keys = sorted({(r["model"], r.get("brand") or "UNKNOWN") for r in rows})
    for model, brand in keys:
        group = [r for r in rows if r["model"] == model and (r.get("brand") or "UNKNOWN") == brand]
        m = metrics([r["label"] == "PHISHING" for r in group], [r["score"] for r in group], group[0]["threshold"])
        result.append({"model": model, "brand": brand, **m})
    return result


def dhash(array):
    gray = np.asarray(Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), "big")


def features(array):
    """96 spatial RGB means + 24 colour histogram features; no CNN features."""
    array = np.clip(array, 0, 255).astype(np.uint8)
    spatial = np.asarray(Image.fromarray(array).resize((8, 4), Image.Resampling.BOX), dtype=float).reshape(-1)/255
    hist = np.concatenate([np.histogram(array[..., c], bins=8, range=(0, 256))[0]/array.shape[0]/array.shape[1] for c in range(3)])
    return np.concatenate([spatial, hist])


def fit_logistic(x, y, iterations=600, l2=0.01):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if set(y) != {0, 1}:
        raise ValueError("Baseline training requires both labels")
    mean = x.mean(axis=0); scale = x.std(axis=0); scale[scale < 1e-8] = 1
    x = (x-mean)/scale
    weights = np.zeros(x.shape[1]); bias = 0.0
    step = 0.5/(np.linalg.norm(x, 2)**2/(4*len(x)) + l2 + 0.25)
    for _ in range(iterations):
        p = 1/(1+np.exp(-np.clip(x @ weights+bias, -40, 40)))
        residual = p-y
        weights -= step*(x.T @ residual/len(y)+l2*weights)
        bias -= step*residual.mean()
    return {"mean": mean, "scale": scale, "weights": weights, "bias": np.array(bias)}


def logistic_scores(fit, x):
    z = (np.asarray(x)-fit["mean"])/fit["scale"] @ fit["weights"]+fit["bias"]
    return 1/(1+np.exp(-np.clip(z, -40, 40)))


def validation_threshold(y, scores):
    if set(y) != {0, 1}:
        raise ValueError("Threshold selection requires both validation labels")
    candidates = np.unique(np.concatenate([[0, 0.5, 1], scores]))
    return float(max(candidates, key=lambda t: (metrics(y, scores, t)["f1"] or 0, -abs(t-0.5))))


def export(out, title, rows, metadata, cards=""):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    per_brand = grouped_metrics(rows) if rows else []
    overall = []
    for model in sorted({r["model"] for r in rows}):
        group = [r for r in rows if r["model"] == model]
        overall.append({"model": model, **metrics([r["label"] == "PHISHING" for r in group], [r["score"] for r in group], group[0]["threshold"])})
    report = {"title": title, "metadata": metadata, "metrics": overall, "per_brand": per_brand}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(out / "predictions.csv", rows)
    write_csv(out / "per_brand.csv", per_brand)
    write_csv(out / "confusion_matrices.csv", overall, ["model", "tn", "fp", "fn", "tp"])
    limitations = metadata.get("limitations", [])
    body = ''.join(f'<li>{html.escape(str(x))}</li>' for x in limitations)
    summary = html.escape(json.dumps(overall, indent=2))
    page = '<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font-family:Arial;margin:24px}figure{display:inline-block;vertical-align:top;margin:8px}img{max-width:100%}article{border-bottom:1px solid #aaa;padding:16px 0}pre{white-space:pre-wrap}</style><h1>'+html.escape(title)+'</h1><ul>'+body+'</ul><h2>Confusion matrices and metrics</h2><pre>'+summary+'</pre>'+cards
    (out / "index.html").write_text(page, encoding="utf-8")
    lines = [title, "="*len(title), "", json.dumps(overall, indent=2), "", "Limitations:"]+["- "+str(x) for x in limitations]
    (out / "report.md").write_text('\n'.join(lines), encoding="utf-8")
