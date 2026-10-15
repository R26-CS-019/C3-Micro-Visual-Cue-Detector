"""Checks for metrics, leakage guards, baseline fitting and split semantics."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from core import metrics, fit_logistic, logistic_scores, validation_threshold, write_csv, export, FIELDS
import csv
from unittest.mock import patch
import types
from core import sha
from research import masks
from research import split_brands, unseen_guard


class ResearchTests(unittest.TestCase):
    def test_confusion_and_missing_class(self):
        m=metrics([0,0,1,1],[0.1,0.9,0.3,0.8],0.5)
        self.assertEqual([m[k] for k in ('tn','fp','fn','tp')],[1,1,1,1])
        self.assertEqual(m['f1'],0.5)
        self.assertIsNone(metrics([0],[0.1],0.5)['recall'])
        with self.assertRaises(ValueError):metrics([1],[float('nan')],0.5)

    def test_baseline_generalization_and_train_scaler(self):
        x=np.array([[-2],[-1],[1],[2]],dtype=float);y=np.array([0,0,1,1])
        fitted=fit_logistic(x,y,300)
        self.assertEqual(float(fitted['mean'][0]),0)
        scores=logistic_scores(fitted,[[-1.5],[1.5]])
        self.assertLess(scores[0],0.5);self.assertGreater(scores[1],0.5)
        threshold=validation_threshold([0,1],scores)
        self.assertEqual(metrics([0,1],scores,threshold)['f1'],1)

    def test_brand_split_rejects_template_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'data.csv';rows=[]
            for brand in ['a','b','c']:
                for label in ['LEGITIMATE','PHISHING']:
                    rows.append(dict(sample_id=brand+label,image='image.png',model=1,label=label,brand=brand,brand_verified='TRUE',split='',template_group='shared' if label=='PHISHING' else brand))
            write_csv(manifest,rows,FIELDS)
            args=argparse.Namespace(manifest=str(manifest),test_brands=['a'],validation_brands=['b'],output=tmp)
            with self.assertRaisesRegex(ValueError,'crosses'):split_brands(args)

    def test_unseen_requires_inventory(self):
        with self.assertRaisesRegex(ValueError,'training-inventory'):
            unseen_guard(argparse.Namespace(training_inventory=None),[])

    def test_export_preserves_per_brand_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[dict(model=1,brand='a',label='LEGITIMATE',score=0.1,threshold=0.5),dict(model=1,brand='a',label='PHISHING',score=0.9,threshold=0.5)]
            export(tmp,'Example',rows,{'limitations':[]})
            with (Path(tmp)/'per_brand.csv').open(encoding='utf-8-sig') as handle:
                record=next(csv.DictReader(handle))
            self.assertEqual(record['recall'],'1.0')
            self.assertEqual(record['false_positive_rate'],'0.0')
            self.assertEqual(record['tp'],'1')
            self.assertTrue(all(0 <= value <= 1 for value in metrics([1]*214,[1]*214,0.5)['recall_wilson_95']))

    def test_unseen_rejects_shared_brand(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source.csv';source.write_text('original manifest')
            config=root/'models.json';config.write_text(json.dumps({'models':{'1':{'sha256':'checkpoint','training_manifest':'source.csv'}}}))
            inventory=root/'inventory.csv';row=dict(sample_id='train',image='example.png',model=1,label='PHISHING',brand='a',brand_verified='TRUE',split='TRAIN',inventory_sha256='original')
            write_csv(inventory,[row],FIELDS)
            Path(str(inventory)+'.checkpoint.json').write_text(json.dumps({'checkpoint_sha256':'checkpoint','training_manifest_sha256':sha(source)}))
            args=argparse.Namespace(training_inventory=str(inventory),config=str(config),project_root=tmp)
            with patch('research.original_manifest',return_value=[row]):
                with self.assertRaisesRegex(ValueError,'Brands overlap'):
                    unseen_guard(args,[dict(row,sample_id='eval')])

    def test_mask_controls_reject_size_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'data.csv';regions=root/'regions.json'
            write_csv(manifest,[dict(sample_id='one',image='image.png',model=1,label='PHISHING')],FIELDS)
            fake=types.SimpleNamespace(Engine=lambda *unused:types.SimpleNamespace(prepare=lambda *unused:np.zeros((20,20,3)),score=lambda *unused:0.7,spec=lambda *unused:{'threshold':0.5}))
            args=argparse.Namespace(manifest=str(manifest),regions=str(regions),project_root=tmp,config='unused',output=tmp)
            for control,error in [([10,10,12,13],'dimensions'),([2,2,6,6],'overlaps')]:
                regions.write_text(json.dumps({'one':[{'name':'logo','box':[1,1,5,5],'controls':[control]}]}))
                with patch.dict('sys.modules',{'engine':fake}):
                    with self.assertRaisesRegex(ValueError,error):masks(args)


if __name__=='__main__':unittest.main()
