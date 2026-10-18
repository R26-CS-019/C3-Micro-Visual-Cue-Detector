"""Run with: python -m unittest discover -s tests -p test_inline_evidence.py"""
import base64
import io
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Micro Visual Cue Component/src'))
from gradcam_evidence import build_visual_evidence, capture_review_visual_evidence, _resized_image_base64


class InlineEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.folder=self.root/'runs'/('a'*32)
        self.folder.mkdir(parents=True)
        Image.new('RGB',(1366,768),'#438879').save(self.folder/'screenshot.png')
        Image.new('RGB',(384,224),'#438879').save(self.folder/'model_input.png')

    def test_ready_plain_base64_and_preserved_input(self):
        for name in ('phishing_overlay.png','legitimate_overlay.png'):
            Image.new('RGB',(384,224),'#b98438').save(self.folder/name)
        original=(self.folder/'model_input.png').read_bytes()
        result=build_visual_evidence(self.folder,{'status':'ready'})
        self.assertEqual(set(result),{'screenshot_jpeg_base64','model_input_png_base64','gradcam'})
        self.assertEqual(base64.b64decode(result['model_input_png_base64'],validate=True),original)
        self.assertEqual(result['gradcam']['status'],'ready')
        for name in ('phishing_overlay_png_base64','legitimate_overlay_png_base64'):
            self.assertTrue(base64.b64decode(result['gradcam'][name],validate=True).startswith(b'\x89PNG'))
        with Image.open(io.BytesIO(base64.b64decode(result['screenshot_jpeg_base64'],validate=True))) as image:
            self.assertEqual(image.format,'JPEG');self.assertLessEqual(image.width,900)

    def test_not_ready_never_reuses_stale_overlays(self):
        for name in ('phishing_overlay.png','legitimate_overlay.png'):
            Image.new('RGB',(384,224),'red').save(self.folder/name)
        for status in ('disabled','unavailable'):
            result=build_visual_evidence(self.folder,{'status':status})
            self.assertEqual(result['gradcam'],{'status':status,'phishing_overlay_png_base64':None,'legitimate_overlay_png_base64':None})
            self.assertTrue(result['screenshot_jpeg_base64']);self.assertTrue(result['model_input_png_base64'])

    def test_large_images_are_bounded_without_altering_source(self):
        noise=self.folder/'noise.png'
        Image.frombytes('RGB',(1600,1200),random.Random(42).randbytes(1600*1200*3)).save(noise)
        original=noise.read_bytes()
        for fmt,width in [('JPEG',900),('PNG',600)]:
            encoded=_resized_image_base64(noise,fmt,width)
            raw=base64.b64decode(encoded,validate=True)
            self.assertLessEqual(len(raw),225000);self.assertLessEqual(len(encoded),300000)
            with Image.open(io.BytesIO(raw)) as image:
                self.assertEqual(image.format,fmt);self.assertLessEqual(image.width,width)
        self.assertEqual(noise.read_bytes(),original)

    def test_review_capture_only_exposes_inline_screenshot(self):
        result=capture_review_visual_evidence(self.root,self.folder.name)
        self.assertEqual(set(result),{'screenshot_jpeg_base64'})
        self.assertTrue(base64.b64decode(result['screenshot_jpeg_base64']).startswith(b'\xff\xd8'))
        self.assertIsNone(capture_review_visual_evidence(self.root,'invalid')['screenshot_jpeg_base64'])

    def test_response_never_contains_evidence_secret(self):
        secret='not-for-inline-response-9f10b7d264'
        with patch.dict(os.environ,{'C3_EVIDENCE_TOKEN':secret}):
            body=json.dumps(build_visual_evidence(self.folder,{'status':'disabled'}))
        self.assertNotIn(secret,body);self.assertNotIn('C3_EVIDENCE_TOKEN',body)


if __name__=='__main__':unittest.main()
