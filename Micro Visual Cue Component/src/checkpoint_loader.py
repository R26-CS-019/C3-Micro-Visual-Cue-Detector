"""Load original weights; handle newer initializer metadata in older Keras."""
import json
import warnings
import zipfile
import tensorflow as tf


def load_checkpoint(path, compile=False):
    try:
        return tf.keras.models.load_model(path, compile=compile)
    except TypeError as error:
        if "unexpected keyword argument 'input_axes'" not in str(error):
            raise
    # Axes only configure random initialization. These checkpoints store None,
    # and every model weight is restored from the original archive afterward.
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
    removed = []

    def visit(node):
        if isinstance(node, dict):
            if node.get("class_name") == "BatchNormalization" and "renorm" in node.get("config", {}):
                if node["config"]["renorm"] is not False:
                    raise ValueError("Cannot adapt enabled batch renormalization")
                for key in ("renorm", "renorm_clipping", "renorm_momentum"):
                    if key in node["config"]:
                        del node["config"][key]
                        removed.append(key)
            if node.get("class_name") == "Dense" and node.get("config", {}).get("quantization_config", "absent") is None:
                del node["config"]["quantization_config"]
                removed.append("quantization_config")
            if node.get("module") == "keras.initializers":
                for key in ("input_axes", "output_axes"):
                    if key in node["config"]:
                        if node["config"][key] is not None:
                            raise ValueError("Cannot adapt a non-default initializer axis")
                        del node["config"][key]
                        removed.append(key)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(config)
    config["compile_config"] = {}
    model = tf.keras.models.model_from_json(json.dumps(config))
    model.load_weights(path)
    model.compatibility_metadata = {"removed_default_initializer_fields": len(removed),
                                    "original_archive_unchanged": True}
    warnings.warn("Loaded original checkpoint weights after adapting default initializer metadata for older Keras.")
    return model
