"""Capture evidence and deployment identity for the existing frozen CNNs.

No classifier, threshold, router or preprocessing changes are made here.
"""
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

EXPECTED_HASHES = {
    1: '6be68e8492246210f323b99a89fdc6107d7abfef30db7bb06820b5005497f1af',
    2: '466ef5861994427a15af742dd68cdc4685b1ee972b8e77dfcc071f88b63e742c',
}
ALLOWED_FILES = {'screenshot.png', 'model_input.png', 'model_input_float.npy', 'capture.json', 'result.json'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def capture_settings():
    wait = int(os.environ.get('C3_CAPTURE_WAIT_MS', '7000'))
    if not 1000 <= wait <= 30000:
        raise ValueError('C3_CAPTURE_WAIT_MS must be between 1000 and 30000')
    return {'headless': os.environ.get('C3_HEADLESS', '1') != '0',
            'viewport': {'width': 1366, 'height': 768}, 'device_scale_factor': 1,
            'wait_until': 'domcontentloaded', 'navigation_timeout_ms': 30000,
            'render_wait_ms': wait, 'full_page': False}


def deployment_info():
    versions = {}
    for name in ('tensorflow', 'keras', 'Pillow', 'playwright', 'numpy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed'
    return {'build_revision': os.environ.get('C3_BUILD_REVISION', 'unknown'),
            'python': platform.python_version(), 'system': platform.system(),
            'versions': versions, 'capture_settings': capture_settings()}


@lru_cache(maxsize=4)
def model_identity(number, checkpoint, threshold):
    checkpoint = Path(checkpoint)
    digest = sha256(checkpoint)
    if digest != EXPECTED_HASHES[number]:
        raise ValueError(f'Model {number} checkpoint hash does not match the approved frozen checkpoint')
    return {'model_number': number, 'checkpoint_filename': checkpoint.name,
            'checkpoint_sha256': digest, 'threshold': float(threshold),
            'input_shape': [160 if number == 1 else 224, 384, 3],
            'preprocessing': 'width_scale_top_crop_bottom_white_pad_lanczos' if number == 1 else 'bilinear_aspect_fit_center_white_pad',
            'input_range': [0, 255], 'input_dtype': 'float32'}


def verify_checkpoints(model1, model2):
    model_identity(1, model1, 0.466)
    model_identity(2, model2, 0.392)


class CaptureQualityError(RuntimeError):
    def __init__(self, capture):
        self.public_detail = {'status': 'capture_needs_review', 'prediction': None,
                              'predicted_label': None, 'phishing_probability': None,
                              'run_id': capture['run_id'], 'capture': capture,
                              'message': 'The captured image may not represent the intended webpage. No CNN prediction was performed.'}
        super().__init__(self.public_detail['message'])


class CaptureNavigationError(RuntimeError):
    def __init__(self, capture):
        self.public_detail = {'status': 'capture_failed', 'prediction': None,
                              'predicted_label': None, 'phishing_probability': None,
                              'run_id': capture['run_id'], 'capture': capture,
                              'message': 'The browser could not complete capture. No website classification is available.'}
        super().__init__(self.public_detail['message'])


def quality_warnings(title, text, status, screenshot):
    from PIL import Image, ImageStat
    warnings = []
    if status is not None and status >= 400:
        warnings.append(f'http_error_{status}')
    clues = ('access denied', 'verify you are human', 'checking your browser',
             "verify you're human", 'just a moment', 'request blocked')
    # Avoid triggering on a passing reference deep in a long article.
    probe = title.lower() + (' ' + text.lower() if len(text) < 1500 else '')
    if any(term in probe for term in clues):
        warnings.append('possible_access_block_or_verification')
    with Image.open(screenshot) as image:
        sample = image.convert('RGB').resize((128, 72))
        stats = ImageStat.Stat(sample)
        if max(stats.stddev) < 2 and min(stats.mean) > 245:
            warnings.append('near_blank_capture')
    return warnings


def capture_with_diagnostics(url, results_dir, guard_route):
    from playwright.sync_api import sync_playwright
    settings = capture_settings()
    folder = Path(results_dir) / 'runs' / uuid.uuid4().hex
    folder.mkdir(parents=True)
    record = {'run_id': folder.name, 'captured_at_utc': datetime.now(timezone.utc).isoformat(),
              'requested_url': url, 'settings': settings, 'deployment': deployment_info(),
              'warnings': [], 'capture_completed': False, 'valid_for_assessment': False}
    screenshot = folder / 'screenshot.png'; html_path = folder / 'page.html'
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=settings['headless'])
            try:
                record['browser_version'] = browser.version
                page = browser.new_page(viewport=settings['viewport'], device_scale_factor=1)
                page.route('**/*', guard_route)
                response = page.goto(url, wait_until=settings['wait_until'], timeout=settings['navigation_timeout_ms'])
                page.wait_for_timeout(settings['render_wait_ms'])
                # A fixed delay cannot guarantee readiness; record what actually loaded.
                html = page.content(); html_path.write_text(html, encoding='utf-8')
                page.screenshot(path=str(screenshot), full_page=False)
                title = page.title()
                body = page.locator('body')
                text = body.inner_text(timeout=5000) if body.count() else ''
                status = response.status if response else None
                image_state = page.locator('img').evaluate_all('(images) => ({total: images.length, loaded: images.filter(i => i.complete && i.naturalWidth > 0).length})')
                warnings = quality_warnings(title, text, status, screenshot)
                record.update(final_url=page.url, title=title, http_status=status,
                              visible_text_length=len(text.strip()), image_load_state=image_state,
                              user_agent=page.evaluate('() => navigator.userAgent'),
                              document_ready_state=page.evaluate('() => document.readyState'),
                              screenshot_sha256=sha256(screenshot), html_sha256=sha256(html_path),
                              capture_completed=True, warnings=warnings,
                              valid_for_assessment=not warnings)
                save_json(folder / 'capture.json', record)
                return html, screenshot, html_path
            finally:
                browser.close()
    except Exception as exc:
        # Preserve the failed run without pretending it received a model verdict.
        record.update(error_type=type(exc).__name__, error=str(exc), valid_for_assessment=False)
        save_json(folder / 'capture.json', record)
        raise CaptureNavigationError(record) from exc


def read_capture(screenshot):
    return json.loads((Path(screenshot).parent / 'capture.json').read_text(encoding='utf-8'))


def check_capture(record):
    if not record['valid_for_assessment']:
        raise CaptureQualityError(record)


def attach_evidence(result, screenshot, array, identity, capture):
    import numpy as np
    from PIL import Image
    folder = Path(screenshot).parent
    array = np.asarray(array, dtype=np.float32)
    np.save(folder / 'model_input_float.npy', array)
    Image.fromarray(np.clip(array, 0, 255).astype('uint8')).save(folder / 'model_input.png')
    diagnostics = {'deployment': deployment_info(), 'capture': capture, 'model': identity,
                   'input_tensor_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
                   'model_input_png_sha256': sha256(folder / 'model_input.png'),
                   'model_input_npy_sha256': sha256(folder / 'model_input_float.npy'),
                   'evidence_base_path': f'/evidence/{folder.name}',
                   'evidence_requires_header': 'X-Evidence-Token'}
    result = {**result, 'run_id': folder.name, 'diagnostics': diagnostics}
    save_json(folder / 'result.json', result)
    return result


def evidence_file(results_dir, run_id, filename):
    if not re.fullmatch('[a-f0-9]{32}', run_id) or filename not in ALLOWED_FILES:
        raise ValueError('Invalid evidence path')
    root = (Path(results_dir) / 'runs').resolve()
    path = root / run_id / filename
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise FileNotFoundError('Evidence not found on this replica')
    return path
