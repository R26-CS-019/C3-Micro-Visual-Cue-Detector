"""PP2 research evidence CLI. Run --help for evaluation and experiment commands."""
import argparse
import html
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import numpy as np
from PIL import Image
from core import FIELDS, read_csv, write_csv, verified, sha, metrics, features, fit_logistic, logistic_scores, validation_threshold, dhash, export

HERE = Path(__file__).resolve().parent


def configuration(args):
    return json.loads(Path(args.config).read_text(encoding="utf-8"))["models"]


def output(args, name):
    root = Path(args.output or HERE / "results")
    out = root / (name+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:6])
    out.mkdir(parents=True)
    return out


def metadata(args, manifest=None):
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "command": args.command,
            "project": str(Path(args.project_root).resolve()), "config_sha256": sha(args.config),
            "manifest_sha256": sha(manifest) if manifest else None,
            "limitations": ["Exploratory unless explicitly supported by reserved data and verified provenance.",
                            "Heatmaps and masks do not prove brand independence; maps are coarse and normalized per image."]}


def original_manifest(args, number):
    spec = configuration(args)[str(number)]
    import csv
    with (Path(args.project_root) / spec["training_manifest"]).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for r in rows:
        split = r["split"] if number == 1 else r["model_split"]
        if split not in {"TRAIN", "VALIDATION", "TEST"}: continue
        if number == 1:
            path = Path(args.project_root) / "model 1 login pp2/data/prepared_login_uniform_top_viewport_384x160" / r["prepared_relative_path"]
            label = "PHISHING" if r["numeric_label"] == "1" else "LEGITIMATE"
            identity = r["original_sha256"]; encoded_sha = r["output_sha256"]
            brand = ""; group = ""
            width, height = r["original_width"], r["original_height"]
        else:
            path = Path(r["screenshot_path"]); label = r["label"]
            identity = encoded_sha = r["sha256"]; group = r["leakage_group_id"]
            brand = r["brand"] if r["brand"] != "LEGITIMATE" else ""
            width = height = ""
        result.append(dict(sample_id=f"model{number}_{len(result):05d}", image=str(path), model=number, label=label,
                           brand=brand, brand_verified="FALSE", label_verified="TRUE", label_evidence="Existing dataset manifest; retain original source provenance",
                           split=split, template_group=group, capture_source=f"model{number}_original_dataset",
                           original_width=width, original_height=height, sha256=encoded_sha, inventory_sha256=identity))
    return result


def bootstrap(args):
    out = output(args, "manifests")
    for number in (1, 2):
        rows = original_manifest(args, number)
        write_csv(out / f"model{number}_dataset.csv", rows, FIELDS)
        inventory = [r for r in rows if r["split"] in {"TRAIN", "VALIDATION"}]
        path = out / f"model{number}_training_inventory.csv"
        write_csv(path, inventory, FIELDS)
        spec = configuration(args)[str(number)]
        Path(str(path)+".checkpoint.json").write_text(json.dumps({"checkpoint_sha256": spec["sha256"],
            "training_manifest_sha256": sha(Path(args.project_root)/spec["training_manifest"]),
            "note": "Review every training/validation brand before unseen-brand evaluation."}, indent=2), encoding="utf-8")
    if args.images:
        rows = []
        for image in sorted(Path(args.images).rglob("*")):
            if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}: continue
            label = "LEGITIMATE" if image.parent.name.lower() in {"leg", "legitimate"} else "PHISHING" if image.parent.name.lower() in {"mal", "phishing"} else None
            if label is None: continue
            import re
            brand = re.sub(r"\s*\(.*?\)|\s+login", "", image.stem.lower()).replace("linked in", "linkedin").strip()
            with Image.open(image) as source: dimensions = source.size
            rows.append(dict(sample_id=f"external_{len(rows):04d}", image=str(image.resolve()), model=args.model, label=label,
                              brand=brand, brand_verified="FALSE", label_verified="FALSE", label_evidence="Folder label; verify source before final claims",
                              split="EXTERNAL", template_group="", capture_source="user_login_folder", original_width=dimensions[0],
                              original_height=dimensions[1], sha256=sha(image), inventory_sha256=""))
        write_csv(out / "same_brand.csv", rows, FIELDS)
    print("Prepared editable manifests:", out)


def unseen_guard(args, rows):
    if not args.training_inventory:
        raise ValueError("Unseen-brand evaluation requires --training-inventory")
    inventory = read_csv(args.training_inventory)
    sidecar = json.loads(Path(str(args.training_inventory)+".checkpoint.json").read_text(encoding="utf-8"))
    models = {r["model"] for r in rows}
    if len(models) != 1:
        raise ValueError("Use one checkpoint/inventory per unseen-brand evaluation")
    number = next(iter(models)); spec = configuration(args)[str(number)]
    if sidecar["checkpoint_sha256"] != spec["sha256"]:
        raise ValueError("Training inventory belongs to another checkpoint")
    if sidecar["training_manifest_sha256"] != sha(Path(args.project_root)/spec["training_manifest"]):
        raise ValueError("Training manifest changed since inventory creation")
    if any(not r["brand"] or not verified(r["brand_verified"]) for r in inventory):
        raise ValueError("Training/validation brand inventory is incomplete or unverified")
    expected = {r["inventory_sha256"] for r in original_manifest(args, number) if r["split"] in {"TRAIN", "VALIDATION"}}
    if {r["inventory_sha256"] for r in inventory} != expected or any(r["model"] != number or r["split"] not in {"TRAIN", "VALIDATION"} for r in inventory):
        raise ValueError("Inventory does not cover the configured checkpoint's training/validation samples")
    if any(not r["brand"] or not verified(r["brand_verified"]) for r in rows):
        raise ValueError("Evaluation brands must be verified")
    overlap = {r["brand"] for r in rows} & {r["brand"] for r in inventory}
    if overlap: raise ValueError("Brands overlap training/validation: "+", ".join(sorted(overlap)))
    training_hashes = {r["sha256"] for r in inventory} | expected
    if any(sha(r["image"]) in training_hashes for r in rows):
        raise ValueError("Evaluation image exactly overlaps training/validation")
    if not args.audit_report:
        raise ValueError("Unseen evaluation requires --audit-report for these manifests")
    audit_result = json.loads(Path(args.audit_report).read_text(encoding="utf-8"))
    if audit_result["metadata"]["manifest_sha256"] != sha(args.manifest) or audit_result["reference_sha256"] != sha(args.training_inventory):
        raise ValueError("Audit report belongs to different evaluation/reference manifests")
    if audit_result["exact_match_pairs"] or audit_result["near_duplicate_candidates"]:
        raise ValueError("Resolve/exclude overlap candidates and rerun audit before unseen evaluation")
    return {"inventory_sha256": sha(args.training_inventory), "held_out_brands": sorted({r["brand"] for r in rows}),
            "note": "Depends on complete, correctly reviewed canonical brand labels. Duplicate independence must also be audited."}


def evaluate(args):
    rows = read_csv(args.manifest)
    if args.mode == "unseen":
        if args.exploratory: raise ValueError("Unseen-brand claims cannot bypass verification with --exploratory")
        guard = unseen_guard(args, rows)
    else: guard = None
    if not args.exploratory and any(not verified(r["label_verified"]) or not r["label_evidence"] or not verified(r["brand_verified"]) for r in rows):
        raise ValueError("Verify brands, labels and provenance, or explicitly use --exploratory")
    from engine import Engine
    engine = Engine(args.project_root, args.config); out = output(args, args.mode)
    predictions, cards = [], []
    for i, r in enumerate(rows):
        if r["sha256"] and sha(r["image"]) != r["sha256"]: raise ValueError("Image changed: "+r["image"])
        array = engine.prepare(r["image"], r["model"]); score = engine.score(array, r["model"])
        folder = out / f"sample_{i:04d}"; folder.mkdir()
        Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(folder / "model_input.png")
        with Image.open(r["image"]) as original:
            original.convert("RGB").save(folder / "original.png")
        maps = engine.explain(array, r["model"], folder) if args.gradcam else None
        threshold = engine.spec(r["model"])["threshold"]
        prediction = "PHISHING" if score >= threshold else "LEGITIMATE"
        predictions.append({**r, "score": score, "threshold": threshold, "prediction": prediction, "correct": prediction == r["label"],
                            "checkpoint_sha256": engine.spec(r["model"])["sha256"], "explanation": maps})
        names = [("original", "Full screenshot"), ("model_input", "Actual model input")]
        if maps: names += [("phishing_overlay", "PHISHING evidence"), ("legitimate_overlay", "LEGITIMATE evidence")]
        cards.append('<article><h2>'+html.escape(f'{r["brand"]}: {prediction} | expected {r["label"]} | score {score:.4f}')+'</h2>'+''.join(f'<figure><img width="384" src="{folder.name}/{name}.png"><figcaption>{caption}</figcaption></figure>' for name,caption in names)+'</article>')
        print(r["sample_id"], prediction, round(score, 4), flush=True)
    meta = metadata(args, args.manifest)
    meta.update(checkpoints=[engine.metadata(n) for n in sorted({r["model"] for r in rows})], brand_guard=guard,
                exploratory=args.exploratory, label_counts=dict(Counter(r["label"] for r in rows)))
    if args.mode == "same-brand":
        missing = [brand for brand in {r["brand"] for r in rows} if {r["label"] for r in rows if r["brand"] == brand} != {"LEGITIMATE", "PHISHING"}]
        meta["brands_missing_one_class"] = missing
        if missing: meta["limitations"].append("Brands missing one class cannot establish same-brand discrimination: "+str(missing))
    export(out, "PP2 "+args.mode+" evaluation", predictions, meta, ''.join(cards)); print("Report:", out / "index.html")


def masks(args):
    rows = read_csv(args.manifest); regions = json.loads(Path(args.regions).read_text(encoding="utf-8"))
    from engine import Engine
    engine = Engine(args.project_root, args.config); out = output(args, "masking")
    results, comparisons, cards = [], [], []
    for r in rows:
        if r["sample_id"] not in regions: continue
        if r["sha256"] and sha(r["image"]) != r["sha256"]: raise ValueError("Image changed")
        array = engine.prepare(r["image"], r["model"]); baseline = engine.score(array, r["model"])
        folder = out / r["sample_id"]
        # sample IDs are labels, never filesystem paths.
        if folder.parent.resolve() != out.resolve(): raise ValueError("Unsafe sample_id for output")
        folder.mkdir()
        Image.fromarray(np.clip(array,0,255).astype(np.uint8)).save(folder / "original.png")
        np.save(folder / "original_float.npy",array)
        for target in regions[r["sample_id"]]:
            name = target["name"]
            if not name.replace("_", "").isalnum(): raise ValueError("Region names must be alphanumeric")
            boxes = [target["box"]]+target["controls"]
            if len(boxes) < 2: raise ValueError("Each target requires a control")
            left,top,right,bottom = boxes[0]; dimensions = (right-left,bottom-top)
            group = []
            for i, box in enumerate(boxes):
                l,t,rr,b = box
                if not all(isinstance(v, int) for v in box) or not (0 <= l < rr <= array.shape[1] and 0 <= t < b <= array.shape[0]): raise ValueError("Invalid mask coordinates")
                if (rr-l,b-t) != dimensions: raise ValueError("Control dimensions must match target")
                if i and max(l,left)<min(rr,right) and max(t,top)<min(b,bottom): raise ValueError("Control overlaps the target")
                masked = array.copy(); masked[t:b,l:rr] = 255
                score = engine.score(masked, r["model"]); version = name if i == 0 else f"{name}_control_{i}"
                Image.fromarray(np.clip(masked,0,255).astype(np.uint8)).save(folder / f"{version}.png")
                np.save(folder / f"{version}_float.npy",masked)
                changed = int(np.any(masked != array, axis=-1).sum()); area = (rr-l)*(b-t)
                item = {"sample_id": r["sample_id"], "brand": r["brand"], "label": r["label"], "model": r["model"],
                        "region": name, "version": version, "box": box, "mask_area": area, "changed_pixels": changed,
                        "baseline_score": baseline, "score": score, "score_change": score-baseline,
                        "threshold": engine.spec(r["model"])["threshold"],
                        "weak_control": bool(i and changed < 0.05*area)}
                item["prediction"] = "PHISHING" if score >= item["threshold"] else "LEGITIMATE"
                results.append(item); group.append(item)
            control_mean = float(np.mean([x["score_change"] for x in group[1:]]))
            comparisons.append({"sample_id": r["sample_id"], "region": name, "target_change": group[0]["score_change"],
                                "mean_control_change": control_mean, "target_minus_control": group[0]["score_change"]-control_mean,
                                "weak_control": any(x["weak_control"] for x in group)})
            cards.append('<article><h2>'+html.escape(r["sample_id"]+" / "+name)+'</h2><figure><img width="384" src="'+folder.name+'/original.png"><figcaption>Original: '+f'{baseline:.4f}'+'</figcaption></figure>'+''.join(f'<figure><img width="384" src="{folder.name}/{x["version"]}.png"><figcaption>{x["version"]}: {x["score"]:.4f} ({x["score_change"]:+.4f})</figcaption></figure>' for x in group)+'</article>')
    if not results: raise ValueError("No region sample IDs matched the manifest")
    write_csv(out / "paired_comparisons.csv", comparisons)
    meta = metadata(args, args.manifest); meta.update(regions_sha256=sha(args.regions), regions=regions,
          checkpoints=[engine.metadata(n) for n in sorted({r["model"] for r in results})])
    meta["limitations"] += ["Masks are white perturbations of exact float model inputs; PNG previews are rounded.",
                            "Controls with few changed pixels are flagged. Region selection and source labels require review.",
                            "Mask variants are correlated; their confusion matrices are descriptive, not independent performance estimates."]
    export(out, "Controlled masking experiments", results, meta, ''.join(cards)); print("Report:", out / "index.html")


def split_brands(args):
    rows = read_csv(args.manifest)
    if any(not r["brand"] or not verified(r["brand_verified"]) for r in rows): raise ValueError("Review all canonical brand labels first")
    test = set(args.test_brands); val = set(args.validation_brands)
    if test & val or not test or not val: raise ValueError("Test and validation brands must be nonempty and disjoint")
    available = {r["brand"] for r in rows}
    if not test | val <= available: raise ValueError("Requested brand absent from manifest")
    for r in rows: r["split"] = "TEST" if r["brand"] in test else "VALIDATION" if r["brand"] in val else "TRAIN"
    for group_field in ("template_group", "sha256"):
        groups = defaultdict(set)
        for r in rows:
            if r[group_field]: groups[r[group_field]].add(r["split"])
        if any(len(v)>1 for v in groups.values()): raise ValueError("Duplicate/template group crosses proposed brand split")
    for split in ("TRAIN", "VALIDATION", "TEST"):
        if {r["label"] for r in rows if r["split"] == split} != {"LEGITIMATE", "PHISHING"}: raise ValueError("Each split needs both labels")
    out = output(args, "brand_split"); write_csv(out / "split_manifest.csv", rows, FIELDS)
    (out / "protocol.json").write_text(json.dumps({"test_brands": sorted(test), "validation_brands": sorted(val),
        "manifest_sha256": sha(args.manifest), "note": "Split proposal only. Existing CNNs were not retrained; do not claim their test brands were held out."}, indent=2), encoding="utf-8")
    print("Brand split proposal:", out)


def audit(args):
    rows = read_csv(args.manifest); references = read_csv(args.reference) if args.reference else []
    from engine import Engine
    engine = Engine(args.project_root, args.config); out = output(args, "audit")
    indexed = []
    for kind, data in [("evaluation",rows),("reference",references)]:
        for r in data:
            digest = sha(r["image"])
            if r["sha256"] and digest != r["sha256"]: raise ValueError("Manifest image hash changed: "+r["image"])
            array = engine.prepare(r["image"],r["model"])
            with Image.open(r["image"]) as im: width,height=im.size
            indexed.append({**r,"audit_set":kind,"actual_sha256":digest,"prepared_pixel_sha256":__import__('hashlib').sha256(array.tobytes()).hexdigest(),
                            "dhash":dhash(array),"actual_width":width,"actual_height":height})
    leaks = []
    previews=[]
    for a in [x for x in indexed if x["audit_set"] == "evaluation"]:
        for b in indexed:
            if a is b or (b["audit_set"] == "evaluation" and b["sample_id"] <= a["sample_id"]): continue
            if a["model"] != b["model"]: continue
            distance = (a["dhash"] ^ b["dhash"]).bit_count()
            raw = a["actual_sha256"] == b["actual_sha256"]; pixel=a["prepared_pixel_sha256"]==b["prepared_pixel_sha256"]
            if raw or pixel or distance <= args.distance:
                leaks.append({"sample":a["sample_id"],"other":b["sample_id"],"sample_image":a["image"],"other_image":b["image"],"other_set":b["audit_set"],"exact_file":raw,
                              "exact_prepared_pixels":pixel,"dhash_distance":distance,"status":"exact_match" if raw or pixel else "visual_review_required"})
                names=[]
                for suffix,item in [('sample',a),('other',b)]:
                    name=f'candidate_{len(leaks):04d}_{suffix}.png'
                    Image.fromarray(np.clip(engine.prepare(item['image'],item['model']),0,255).astype(np.uint8)).save(out/name)
                    names.append('<figure style="display:inline-block"><img width="384" src="'+name+'"><figcaption>'+html.escape(item['sample_id'])+'</figcaption></figure>')
                previews.append('<article><h2>Candidate '+str(len(leaks))+' (dHash distance '+str(distance)+')</h2>'+''.join(names)+'</article>')
    groups = Counter((r["label"],r["capture_source"],r["original_width"] or r["actual_width"],r["original_height"] or r["actual_height"]) for r in indexed if r["audit_set"]=="evaluation")
    associations=[dict(label=k[0],source=k[1],width=k[2],height=k[3],images=v) for k,v in groups.items()]
    write_csv(out/"overlap_candidates.csv",leaks,["sample","other","sample_image","other_image","other_set","exact_file","exact_prepared_pixels","dhash_distance","status"])
    write_csv(out/"source_dimension_associations.csv",associations)
    exact=sum(x["status"]=="exact_match" for x in leaks)
    report={"metadata":metadata(args,args.manifest),"reference_sha256":sha(args.reference) if args.reference else None,
            "evaluation_images":len(rows),"reference_images":len(references),"exact_match_pairs":exact,
            "near_duplicate_candidates":len(leaks)-exact,"brand_overlap":sorted({r["brand"] for r in rows if r["brand"]}&{r["brand"] for r in references if r["brand"]}),
            "unverified_brand_counts":{"evaluation":sum(not r["brand"] or not verified(r["brand_verified"]) for r in rows),"reference":sum(not r["brand"] or not verified(r["brand_verified"]) for r in references)},
            "limitations":["dHash candidates require visual review; low distance is not proof of duplication.","No match does not prove independence. Check template groups and collection provenance.","Dimension/source correlations suggest possible shortcuts, not proof that CNNs use them."]}
    report["limitations"].append("An empty brand-overlap list cannot establish disjoint brands when inventory labels are missing or unverified.")
    (out/"audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><h1>Leakage and shortcut audit</h1><p><a href="overlap_candidates.csv">Overlap candidates</a> | <a href="source_dimension_associations.csv">Source/dimension associations</a></p><pre>'+html.escape(json.dumps(report,indent=2))+'</pre>'+''.join(previews),encoding="utf-8")
    (out/"report.md").write_text('# Leakage and shortcut audit\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n',encoding="utf-8")
    print(json.dumps(report,indent=2));print("Audit:",out)


def baseline(args):
    rows = [r for r in read_csv(args.manifest) if r["model"] == args.model]
    train=[r for r in rows if r["split"]=="TRAIN"]; val=[r for r in rows if r["split"]=="VALIDATION"]; test=[r for r in rows if r["split"]=="TEST"]
    if not train or not val or not test: raise ValueError("Baseline requires TRAIN, VALIDATION and TEST")
    expected=original_manifest(args,args.model)
    for split in ("TRAIN","VALIDATION"):
        supplied=Counter(r["inventory_sha256"] or r["sha256"] for r in rows if r["split"]==split)
        original=Counter(r["inventory_sha256"] for r in expected if r["split"]==split)
        if supplied!=original: raise ValueError("Fair CNN comparison requires the configured checkpoint's identical training/validation split")
    for field in ("sha256","inventory_sha256","template_group"):
        seen=defaultdict(set)
        for r in rows:
            if r[field]: seen[r[field]].add(r["split"])
        if any(len(v)>1 for v in seen.values()): raise ValueError("Baseline split leakage in "+field)
    from engine import Engine
    engine=Engine(args.project_root,args.config);out=output(args,"baseline")
    xs=[];dims=[];ys=[]
    for i,r in enumerate(rows):
        if r["sha256"] and sha(r["image"])!=r["sha256"]:raise ValueError("Image hash changed")
        xs.append(features(engine.prepare(r["image"],r["model"])))
        if not r["original_width"] or not r["original_height"]:
            with Image.open(r["image"]) as im:w,h=im.size
        else:w,h=float(r["original_width"]),float(r["original_height"])
        dims.append([np.log(w),np.log(h),w/h]);ys.append(int(r["label"]=="PHISHING"))
        if i%500==0:print("Baseline features",i,"/",len(rows),flush=True)
    xs=np.asarray(xs);dims=np.asarray(dims);ys=np.asarray(ys)
    ix={split:np.array([i for i,r in enumerate(rows) if r["split"]==split]) for split in ("TRAIN","VALIDATION","TEST")}
    comparisons=[];test_rows=[]
    for name,x in [("colour_layout_logistic",xs),("dimensions_only_logistic",dims)]:
        fitted=fit_logistic(x[ix["TRAIN"]],ys[ix["TRAIN"]],args.iterations)
        threshold=validation_threshold(ys[ix["VALIDATION"]],logistic_scores(fitted,x[ix["VALIDATION"]]))
        np.savez(out/(name+".npz"),**fitted,threshold=threshold)
        scores=logistic_scores(fitted,x[ix["TEST"]])
        comparisons.append({"method":name,**metrics(ys[ix["TEST"]],scores,threshold)})
        for index,score in zip(ix["TEST"],scores): test_rows.append({"method":name,**rows[index],"score":float(score),"threshold":threshold})
    cnn=[]
    for i in ix["TEST"]:
        score=engine.score(engine.prepare(rows[i]["image"],args.model),args.model);cnn.append(score)
        test_rows.append({"method":"cnn",**rows[i],"score":score,"threshold":engine.spec(args.model)["threshold"]})
    comparisons.append({"method":"cnn",**metrics(ys[ix["TEST"]],cnn,engine.spec(args.model)["threshold"])})
    write_csv(out/"comparison.csv",comparisons);write_csv(out/"test_predictions.csv",test_rows)
    report={"metadata":metadata(args,args.manifest),"checkpoint":engine.metadata(args.model),"methods":comparisons,
            "feature_definition":"8x4 RGB spatial means and 8-bin RGB histograms; dimensions-only uses original width, height and ratio",
            "training":"Train-only standardization; fixed L2=0.01; validation-only maximum-F1 threshold; no test tuning",
            "split_counts":{k:len(v) for k,v in ix.items()},"limitations":["Original Model 1 TEST was already inspected and is diagnostic, not untouched final evidence.","Simple baselines are controls, not exhaustive competing methods.","CNN may also have used augmentation, class weighting and checkpoint selection; identical splits alone do not equalize all training choices."]}
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (out/"report.md").write_text('# Baseline comparison\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n',encoding="utf-8")
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><h1>Baseline comparison</h1><pre>'+html.escape(json.dumps(report,indent=2))+'</pre>',encoding="utf-8")
    print("Baseline report:",out/"index.html")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",default=os.environ.get("PP2_PROJECT_ROOT",str(HERE.parent if (HERE.parent/"model 1 login pp2").exists() else Path(r"C:\Users\oshan\Desktop\full  pp2 module"))))
    parser.add_argument("--config",default=str(HERE/"models.json"));parser.add_argument("--output")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("bootstrap");p.add_argument("--images");p.add_argument("--model",type=int,choices=[1,2],default=1)
    p=sub.add_parser("evaluate");p.add_argument("--manifest",required=True);p.add_argument("--mode",choices=["same-brand","unseen"],default="same-brand");p.add_argument("--training-inventory");p.add_argument("--audit-report");p.add_argument("--exploratory",action="store_true");p.add_argument("--gradcam",action="store_true")
    p=sub.add_parser("mask");p.add_argument("--manifest",required=True);p.add_argument("--regions",required=True)
    p=sub.add_parser("split-brands");p.add_argument("--manifest",required=True);p.add_argument("--test-brands",nargs="+",required=True);p.add_argument("--validation-brands",nargs="+",required=True)
    p=sub.add_parser("audit");p.add_argument("--manifest",required=True);p.add_argument("--reference");p.add_argument("--distance",type=int,default=6)
    p=sub.add_parser("baseline");p.add_argument("--manifest",required=True);p.add_argument("--model",type=int,choices=[1,2],required=True);p.add_argument("--iterations",type=int,default=600)
    args=parser.parse_args()
    {"bootstrap":bootstrap,"evaluate":evaluate,"mask":masks,"split-brands":split_brands,"audit":audit,"baseline":baseline}[args.command](args)
    from reporting import build_index
    build_index(args.output or HERE/"results")


if __name__=="__main__":main()
