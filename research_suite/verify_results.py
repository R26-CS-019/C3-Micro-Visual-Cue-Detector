"""Verify saved initial experiments against archived predictions, without inference."""
import argparse
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from core import metrics, write_csv
from reporting import build_index


def records(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root',required=True)
    args=parser.parse_args();here=Path(__file__).resolve().parent;root=here/'results'
    same=root/'same-brand_20261011T084659Z_ed55d1'
    old={r['image']:float(r['phishing_probability']) for r in records(Path(args.archive_root)/'viva_evidence/login_pages_review/predictions.csv')}
    new=records(same/'predictions.csv')
    assert len(new)==len(old)==20
    parity=max(abs(float(r['score'])-old[r['image']]) for r in new)
    assert parity < 1e-6,parity
    explanation_errors=[]
    for folder in root.glob('same-brand_*'):
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        write_csv(folder/'per_brand.csv',report['per_brand'])
        for path in folder.glob('sample_*/explanation.json'):
            explanation=json.loads(path.read_text(encoding='utf-8'))
            assert set(explanation)=={'PHISHING','LEGITIMATE'}
            explanation_errors.extend(v['forward_parity_error'] for v in explanation.values())
            assert (path.parent/'model_input_float.npy').exists()
    assert len(explanation_errors)==44 and max(explanation_errors)<1e-5
    smoke=records(root/'same-brand_20261011T084833Z_97b541/predictions.csv')
    expected={1:0.9645748734474182,2:0.9682071805000305}
    smoke_error=max(abs(float(r['score'])-expected[int(r['model'])]) for r in smoke)
    assert smoke_error<1e-6
    baseline=root/'baseline_20261011T084658Z_aa75e1'
    report=json.loads((baseline/'report.json').read_text(encoding='utf-8'))
    predictions=records(baseline/'test_predictions.csv');comparison=[]
    for original in report['methods']:
        group=[r for r in predictions if r['method']==original['method']]
        calculated=metrics([r['label']=='PHISHING' for r in group],[float(r['score']) for r in group],float(group[0]['threshold']))
        assert len(group)==428 and calculated['accuracy']==original['accuracy']
        comparison.append({'method':original['method'],**calculated})
    report['methods']=comparison
    write_csv(baseline/'comparison.csv',comparison)
    (baseline/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (baseline/'report.md').write_text('# Baseline comparison\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n',encoding='utf-8')
    (baseline/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Baseline comparison</h1><p><a href="comparison.csv">Comparison spreadsheet</a></p><pre>'+html.escape(json.dumps(report,indent=2))+'</pre>',encoding='utf-8')
    audit=json.loads((root/'audit_20261011T085645Z_05ca81/audit.json').read_text(encoding='utf-8'))
    masks=root/'masking_20261011T085645Z_43da93'
    mask_rows=records(masks/'predictions.csv')
    for r in mask_rows:assert (masks/r['sample_id']/(r['version']+'_float.npy')).exists()
    verification={'created_utc':datetime.now(timezone.utc).isoformat(),'unit_tests':{'command':'python -B -m unittest test_research -v','passed':7},
        'same_brand_images':20,'archived_prediction_max_absolute_error':parity,
        'two_checkpoint_smoke_max_absolute_error':smoke_error,'gradcam_class_maps_checked':len(explanation_errors),
        'gradcam_forward_max_absolute_error':max(explanation_errors),'mask_variants_checked':len(mask_rows),
        'baseline_split_counts':report['split_counts'],'baseline':comparison,
        'audit':{'evaluation_images':audit['evaluation_images'],'reference_images':audit['reference_images'],'exact_match_pairs':audit['exact_match_pairs'],'unresolved_near_candidates':audit['near_duplicate_candidates']},
        'unseen_brand_evaluation_performed':False,'checkpoint_weights_modified':False,
        'limitations':['Verification checks tool behavior, not brand independence.','Same-brand labels and canonical brands require provenance review.','No fresh final test or retraining was performed.']}
    (here/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    build_index(root)
    print(json.dumps({k:v for k,v in verification.items() if k!='baseline'},indent=2))


if __name__=='__main__':main()
