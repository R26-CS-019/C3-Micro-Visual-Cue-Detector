"""Build a local evidence index after each completed experiment."""
import html
import json
from pathlib import Path


def build_index(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    cards=[]
    for folder in sorted(root.iterdir(),reverse=True):
        if not folder.is_dir():continue
        report_path=next((folder/name for name in ('report.json','audit.json') if (folder/name).exists()),None)
        if report_path:
            report=json.loads(report_path.read_text(encoding='utf-8'))
            links=[]
            for filename in ('index.html','predictions.csv','per_brand.csv','confusion_matrices.csv','paired_comparisons.csv','comparison.csv','overlap_candidates.csv','source_dimension_associations.csv','report.md',report_path.name):
                if (folder/filename).exists():
                    links.append(f'<a href="{folder.name}/{filename}">{filename}</a>')
            cards.append('<article><h2>'+html.escape(report.get('title',folder.name))+'</h2><p>'+html.escape(folder.name)+'</p><p>'+' | '.join(links)+'</p></article>')
        elif folder.name.startswith('manifests'):
            links=[f'<a href="{folder.name}/{p.name}">{p.name}</a>' for p in sorted(folder.glob('*.csv'))]
            cards.append('<article><h2>Manifests for review</h2><p>'+' | '.join(links)+'</p></article>')
    page='''<!doctype html><meta charset="utf-8"><title>PP2 research evidence</title>
<style>body{font:16px Arial;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.6}article{border:1px solid #ddd;padding:12px 20px;margin:16px 0}a{color:#0645ad}.notice{background:#fff1d6;padding:16px}</style>
<h1>PP2 research evidence</h1><p>Frozen Model 1 and Model 2 checkpoints: explanations, controlled experiments and reproducible reports.</p>
<div class="notice"><strong>Research limits:</strong> the included same-brand set has only one legitimate image per brand. Masking examples validate the tool and require wider reviewed experiments. No unseen-brand result is claimed: complete verified training-brand records are required. Grad-CAM and masking alone do not prove brand independence.</div>
<p><strong>Model 1 diagnostic test (428 images):</strong> CNN accuracy 81.54%; colour/layout logistic 66.36%; original-dimensions logistic 100%. The dimension result exposes a dataset shortcut risk. It does not establish which cues the CNN uses. This test split was previously inspected.</p>
<p><a href="../README.md">Instructions</a> | <a href="../verification.json">Verification</a></p>'''+''.join(cards)
    (root/'index.html').write_text(page,encoding='utf-8')
