"""Frozen-checkpoint inference, exact preprocessing and binary Grad-CAM."""
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from core import sha

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
# These inference tools do not use Keras' optional scikit-learn wrappers.
sys.modules.setdefault("sklearn", None)
import tensorflow as tf
from checkpoint_loader import load_checkpoint


class Engine:
    def __init__(self, project, config):
        self.project = Path(project).resolve()
        self.config = json.loads(Path(config).read_text(encoding="utf-8"))["models"]
        self.loaded = {}

    def spec(self, number):
        return self.config[str(number)]

    def model(self, number):
        if number not in self.loaded:
            spec = self.spec(number); checkpoint = self.project / spec["checkpoint"]
            if sha(checkpoint) != spec["sha256"]:
                raise ValueError("Checkpoint hash differs from configuration; explicitly register the intended checkpoint")
            model = load_checkpoint(checkpoint, compile=False)
            if tuple(model.input_shape) != (None, spec["height"], spec["width"], 3):
                raise ValueError("Checkpoint input shape differs from configuration")
            self.loaded[number] = model
        return self.loaded[number]

    def prepare(self, path, number):
        spec = self.spec(number); width = spec["width"]; height = spec["height"]
        if number == 1:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                resized_height = max(1, round(image.height*width/image.width))
                image = image.resize((width, resized_height), Image.Resampling.LANCZOS)
                if resized_height >= height:
                    image = image.crop((0, 0, width, height))
                else:
                    canvas = Image.new("RGB", (width, height), "white"); canvas.paste(image, (0, 0)); image = canvas
                return np.asarray(image, dtype=np.float32)
        image = tf.cast(tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False), tf.float32)
        shape = tf.cast(tf.shape(image)[:2], tf.float32)
        scale = tf.minimum(height/shape[0], width/shape[1])
        dimensions = tf.maximum(1, tf.cast(tf.round(shape*scale), tf.int32))
        image = tf.image.resize(image, dimensions, method="bilinear", antialias=True)
        pad_h = height-dimensions[0]; pad_w = width-dimensions[1]
        return tf.pad(image, [[pad_h//2, pad_h-pad_h//2], [pad_w//2, pad_w-pad_w//2], [0, 0]], constant_values=255).numpy()

    def score(self, array, number):
        p = float(np.asarray(self.model(number)(array[None, ...], training=False))[0, 0])
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid checkpoint probability")
        return p

    def metadata(self, number):
        spec = self.spec(number)
        return {"model": number, **spec, "checkpoint_absolute": str(self.project / spec["checkpoint"]),
                "tensorflow": tf.__version__, "keras": tf.keras.__version__,
                "compatibility": getattr(self.model(number), "compatibility_metadata", None)}

    def explain(self, array, number, out):
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        model = self.model(number)
        backbone = next(l for l in model.layers if isinstance(l, tf.keras.Model) and "efficientnet" in l.name.lower())
        conv = next(l for l in reversed(backbone.layers) if isinstance(l, tf.keras.layers.Conv2D))
        probe = tf.keras.Model(backbone.inputs, [conv.output, backbone.output])
        index = model.layers.index(backbone)
        prepared = np.clip(array, 0, 255).astype(np.uint8)
        Image.fromarray(prepared).save(out / "model_input.png")
        np.save(out / "model_input_float.npy", array)
        maps = {}
        for target in ("PHISHING", "LEGITIMATE"):
            x = tf.convert_to_tensor(array[None, ...])
            with tf.GradientTape() as tape:
                tape.watch(x)
                for layer in model.layers[1:index]: x = layer(x, training=False)
                features, head = probe(x, training=False)
                for layer in model.layers[index+1:-1]: head = layer(head, training=False)
                final = model.layers[-1]
                logit = tf.matmul(head, final.kernel)
                if final.use_bias: logit += final.bias
                score = logit[:, 0] if target == "PHISHING" else -logit[:, 0]
            gradient = tape.gradient(score, features)
            if gradient is None or not bool(tf.reduce_all(tf.math.is_finite(gradient))):
                raise ValueError("Invalid Grad-CAM gradients")
            reconstructed = float(tf.sigmoid(logit)[0, 0]); reference = self.score(array, number)
            if not np.isclose(reference, reconstructed, atol=1e-5, rtol=1e-5):
                raise ValueError("Explanation forward path differs from checkpoint prediction")
            weights = tf.reduce_mean(gradient, axis=(0, 1, 2))
            raw = tf.nn.relu(tf.reduce_sum(features[0]*weights, axis=-1))
            maximum = float(tf.reduce_max(raw)); heat = raw/maximum if maximum > 0 else tf.zeros_like(raw)
            np.save(out / f"{target.lower()}_heatmap.npy", heat.numpy())
            resized = tf.image.resize(heat[..., None], array.shape[:2]).numpy()[..., 0]
            anchors = np.array([[15,20,80], [0,150,220], [255,220,0], [230,20,20]], dtype=float)
            colour = np.stack([np.interp(resized, [0,1/3,2/3,1], anchors[:, c]) for c in range(3)], axis=-1)
            alpha = (0.55*resized)[..., None]
            Image.fromarray(np.clip(prepared*(1-alpha)+colour*alpha,0,255).astype(np.uint8)).save(out / f"{target.lower()}_overlay.png")
            maps[target] = {"layer": conv.name, "target": target, "target_score": "signed_binary_logit",
                            "feature_shape": list(features.shape), "empty_heatmap": maximum == 0,
                            "forward_parity_error": abs(reference-reconstructed)}
        (out / "explanation.json").write_text(json.dumps(maps, indent=2), encoding="utf-8")
        return maps
