"""Class-specific Grad-CAM for the exact input of an existing prediction.

Reuses the loaded checkpoint. No recapture, training or threshold changes.
"""
import json
import logging
import os
import base64
import io
from pathlib import Path


# Binary image cap; base64 transport is about one third larger. Screenshot and
# overlay encoders use a tighter target to keep their base64 strings <=300 KB.
MAX_IMAGE_BYTES = 300_000
MAX_RESIZED_IMAGE_BYTES = 225_000


def _resized_image_base64(path, image_format, max_width):
    """Encode a display copy only; never overwrite saved research evidence."""
    from PIL import Image, ImageOps

    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert('RGB')
        if image.width > max_width:
            height = max(1, round(image.height * max_width / image.width))
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        while True:
            buffer = io.BytesIO()
            if image_format == 'JPEG':
                image.save(buffer, format='JPEG', quality=70, optimize=True)
            else:
                image.save(buffer, format='PNG', optimize=True)
            data = buffer.getvalue()
            if len(data) <= MAX_RESIZED_IMAGE_BYTES:
                return base64.b64encode(data).decode('ascii')
            if image.width == 1 and image.height == 1:
                raise ValueError('Image cannot be encoded within the inline size limit')
            image = image.resize((max(1, int(image.width * .8)), max(1, int(image.height * .8))),
                                 Image.Resampling.LANCZOS)


def _model_input_base64(path):
    """Preserve the exact saved model-input PNG rather than resize its pixels."""
    data = Path(path).read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Saved model input exceeds inline image size limit')
    return base64.b64encode(data).decode('ascii')


def _optional_image(encode, *args):
    try:
        return encode(*args)
    except Exception:
        logging.exception('Inline visual evidence unavailable; classification retained')
        return None


def build_visual_evidence(folder, explanation):
    folder = Path(folder)
    ready = explanation['status'] == 'ready'
    return {
        'screenshot_jpeg_base64': _optional_image(_resized_image_base64, folder / 'screenshot.png', 'JPEG', 900),
        'model_input_png_base64': _optional_image(_model_input_base64, folder / 'model_input.png'),
        'gradcam': {
            'status': explanation['status'],
            'phishing_overlay_png_base64': _optional_image(_resized_image_base64, folder / 'phishing_overlay.png', 'PNG', 600) if ready else None,
            'legitimate_overlay_png_base64': _optional_image(_resized_image_base64, folder / 'legitimate_overlay.png', 'PNG', 600) if ready else None,
        },
    }


def capture_review_visual_evidence(results_dir, run_id):
    from hosting_diagnostics import evidence_file

    try:
        screenshot = evidence_file(results_dir, run_id, 'screenshot.png')
        encoded = _optional_image(_resized_image_base64, screenshot, 'JPEG', 900)
    except (ValueError, FileNotFoundError):
        encoded = None
    return {'screenshot_jpeg_base64': encoded}


def generate_maps(model, array, output, expected_score):
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    output = Path(output)
    array = np.asarray(array)
    if array.dtype != np.float32 or tuple(array.shape) != tuple(model.input_shape[1:]):
        raise ValueError('Saved tensor does not match the selected checkpoint input')
    if not np.isfinite(array).all():
        raise ValueError('Saved tensor contains non-finite values')
    reference = float(np.asarray(model(array[None, ...], training=False))[0, 0])
    if not np.isclose(reference, expected_score, atol=1e-5, rtol=1e-5):
        raise ValueError('Saved tensor score does not match the original prediction')
    backbone = next(layer for layer in model.layers
                    if isinstance(layer, tf.keras.Model) and 'efficientnet' in layer.name.lower())
    conv = next(layer for layer in reversed(backbone.layers) if isinstance(layer, tf.keras.layers.Conv2D))
    probe = tf.keras.Model(backbone.inputs, [conv.output, backbone.output])
    index = model.layers.index(backbone)
    prepared = np.clip(array, 0, 255).astype(np.uint8)
    maps = {}
    for target in ('PHISHING', 'LEGITIMATE'):
        x = tf.convert_to_tensor(array[None, ...])
        with tf.GradientTape() as tape:
            tape.watch(x)
            for layer in model.layers[1:index]:
                x = layer(x, training=False)
            features, head = probe(x, training=False)
            for layer in model.layers[index + 1:-1]:
                head = layer(head, training=False)
            final = model.layers[-1]
            logit = tf.matmul(head, final.kernel)
            if final.use_bias:
                logit += final.bias
            objective = logit[:, 0] if target == 'PHISHING' else -logit[:, 0]
        gradient = tape.gradient(objective, features)
        if gradient is None or not bool(tf.reduce_all(tf.math.is_finite(gradient))):
            raise ValueError('Invalid Grad-CAM gradients')
        reconstructed = float(tf.sigmoid(logit)[0, 0])
        if not np.isclose(reference, reconstructed, atol=1e-5, rtol=1e-5):
            raise ValueError('Grad-CAM forward path differs from checkpoint inference')
        weights = tf.reduce_mean(gradient, axis=(0, 1, 2))
        raw = tf.nn.relu(tf.reduce_sum(features[0] * weights, axis=-1))
        maximum = float(tf.reduce_max(raw))
        heat = raw / maximum if maximum > 0 else tf.zeros_like(raw)
        np.save(output / f'{target.lower()}_heatmap.npy', heat.numpy())
        resized = tf.image.resize(heat[..., None], array.shape[:2]).numpy()[..., 0]
        anchors = np.array([[15, 20, 80], [0, 150, 220], [255, 220, 0], [230, 20, 20]], dtype=float)
        colour = np.stack([np.interp(resized, [0, 1/3, 2/3, 1], anchors[:, c]) for c in range(3)], axis=-1)
        alpha = (0.55 * resized)[..., None]
        overlay = np.clip(prepared * (1 - alpha) + colour * alpha, 0, 255).astype(np.uint8)
        Image.fromarray(overlay).save(output / f'{target.lower()}_overlay.png')
        maps[target] = {'layer': conv.name, 'target': target, 'target_score': 'signed_binary_logit',
                        'feature_shape': list(features.shape), 'empty_heatmap': maximum == 0,
                        'forward_parity_error': abs(reference - reconstructed)}
    metadata = {'status': 'ready', 'maps': maps, 'original_prediction_score': expected_score,
                'saved_input_score': reference, 'input_score_difference': abs(reference - expected_score),
                'interpretation': 'Maps are independently normalised sensitivity visualisations, not probabilities or proof of causal features.'}
    (output / 'explanation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def attach_gradcam(result, model, results_dir):
    """Keep successful classification even if explanation generation is unavailable."""
    import numpy as np
    from hosting_diagnostics import evidence_file, save_json

    run_id = result['run_id']
    # Validate the path before reading a tensor; allow_pickle=False is intentional.
    folder = evidence_file(results_dir, run_id, 'result.json').parent
    base = f'/evidence/{run_id}'
    evidence = {'run_id': run_id, 'requires_header': 'X-Evidence-Token',
                'screenshot': f'{base}/screenshot.png', 'model_input': f'{base}/model_input.png',
                'gradcam_phishing': None, 'gradcam_legitimate': None, 'explanation': None}
    if os.environ.get('C3_GRADCAM_ENABLED', '1') == '0':
        explanation = {'status': 'disabled'}
    else:
        try:
            tensor = evidence_file(results_dir, run_id, 'model_input_float.npy')
            explanation = generate_maps(model, np.load(tensor, allow_pickle=False), folder,
                                        float(result['phishing_probability']))
            evidence.update(gradcam_phishing=f'{base}/phishing_overlay.png',
                            gradcam_legitimate=f'{base}/legitimate_overlay.png',
                            explanation=f'{base}/explanation.json')
        except Exception:
            logging.exception('Grad-CAM unavailable for run %s; prediction retained', run_id)
            explanation = {'status': 'unavailable', 'message': 'The prediction is available, but its explanation could not be generated.'}
            save_json(folder / 'explanation.json', explanation)
    result['evidence'] = evidence
    result['diagnostics']['gradcam'] = explanation
    result['visual_evidence'] = build_visual_evidence(folder, explanation)
    save_json(folder / 'result.json', result)
    return result
