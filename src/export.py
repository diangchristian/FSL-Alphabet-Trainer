"""
Step 3: export the trained model in the app's "fsl-conv1d-bilstm-v1" release format.

releases/<version>/model.json  metadata the app validates + weight specs
releases/<version>/weights.bin float32 weights in Keras layer order (what loadWeights expects)
releases/<version>/parity.json 3 inputs -> Keras outputs; the app's model test must reproduce them
fixtures/normalize.json        raw landmarks + frame size -> expected features, for the app's normalizeFrame test
"""
import hashlib
import json

import keras
import mediapipe as mp
import numpy as np

import config
import dataset
from features import to_frame


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_normalize_fixture(data) -> None:
    """Real landmarks, re-expressed under several frame sizes, incl. non-square camera frames."""
    rng = np.random.default_rng(7)
    cases = []
    for i in rng.choice(len(data["labels"]), 6, replace=False):
        present = data["present"][i]
        for width, height in [tuple(map(float, data["sizes"][i])), (1280.0, 720.0), (640.0, 480.0)]:
            raw = np.zeros(128, dtype=np.float32)
            for slot in (0, 1):
                if present[slot]:
                    raw[slot * 63:(slot + 1) * 63] = data["raw"][i][slot].reshape(-1)
                    raw[126 + slot] = 1
            hands = [(slot, data["raw"][i][slot]) for slot in (0, 1) if present[slot]]
            cases.append({"raw": raw.tolist(), "width": width, "height": height,
                          "expected": to_frame(hands, width, height).tolist()})
    # A frame with no hands stays all zeros.
    cases.append({"raw": [0.0] * 128, "width": 640.0, "height": 480.0, "expected": [0.0] * 128})
    config.FIXTURES.mkdir(exist_ok=True)
    (config.FIXTURES / "normalize.json").write_text(json.dumps({"normalization": config.NORMALIZATION, "cases": cases}))


def main() -> None:
    model = keras.models.load_model(config.CACHE / "model.keras")
    frames, labels, groups, letters = dataset.load()
    split = np.load(config.CACHE / "split.npz")
    metrics = json.loads((config.REPORTS / "metrics.json").read_text())

    specs, chunks, offset = [], [], 0
    for layer in model.layers:
        for weight in layer.weights:
            array = np.asarray(weight.numpy(), dtype="<f4")
            blob = array.tobytes()
            specs.append({"name": weight.path, "shape": list(array.shape), "dtype": "float32", "offset": offset, "bytes": len(blob)})
            chunks.append(blob)
            offset += len(blob)
    weights = b"".join(chunks)

    metadata = {
        "format": config.FORMAT,
        "category": "alphabet",
        "inputShape": [config.FRAMES, config.FEATURES],
        "classes": letters,
        "preprocessing": {
            "frames": config.FRAMES,
            "features": config.FEATURES,
            "layout": ["left_xyz_landmarks_0_to_20", "right_xyz_landmarks_0_to_20", "left_present", "right_present"],
            "normalization": config.NORMALIZATION,
            "coordinates": "pixels relative to the wrist (x*width, y*height, z*width), divided by the wrist-to-middle-knuckle length",
            "interpolation": "linear coordinates, nearest binary flags, absent coordinates zeroed",
            "seed": config.SEED,
            "validation_fraction": config.VALIDATION_FRACTION,
            "test_fraction": config.TEST_FRACTION,
            "static_method": f"repeat + per-timestep Gaussian noise ({dataset.NOISE_STD} palm lengths)",
            "augmentation": f"rotation ±{dataset.ROTATION_DEGREES}°, mirror p={dataset.MIRROR_PROBABILITY} (x negated, hand slots swapped)",
            "grouping": "source path with Roboflow .rf. siblings grouped",
            "extraction": {
                "backend": "MediaPipe HandLandmarker",
                "mediapipe_version": mp.__version__,
                "model_sha256": config.HAND_MODEL_SHA256,
                "model_url": config.HAND_MODEL_URL,
                "num_hands": config.NUM_HANDS,
                "detection_confidence": config.DETECTION_CONFIDENCE,
                "swap_hands": config.SWAP_HANDS,
                "image_orientation": "EXIF transpose; no horizontal flip",
            },
        },
        "evaluation": {k: metrics[k] for k in ("test_accuracy", "test_samples", "correct_and_confident_at_0.8", "wrong_but_confident_at_0.8")},
        "weights": specs,
        "weightBytes": len(weights),
        "weightSha256": sha256(weights),
    }
    # Release files are served as immutable, so the version must change whenever *any* content
    # changes (weights or metadata), never just overwrite files at an already-cached URL.
    version = sha256(weights + json.dumps(metadata, sort_keys=True).encode())[:16]
    metadata = {"format": metadata.pop("format"), "category": metadata.pop("category"), "version": version, **metadata}

    out = config.RELEASES / version
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.bin").write_bytes(weights)
    model_json = json.dumps(metadata, indent=1).encode()
    (out / "model.json").write_bytes(model_json)

    probe = dataset.fixed(frames[split["test"][:3]], np.random.default_rng(3))
    expected = model.predict(probe, verbose=0)
    (out / "parity.json").write_text(json.dumps({"inputs": probe.tolist(), "expected": expected.tolist()}))

    hand = config.HAND_MODEL.read_bytes()
    entry = {
        "id": "alphabet", "name": "Alphabet", "available": True, "version": version,
        "modelUrl": f"./models/alphabet/{version}/model.json", "weightsUrl": f"./models/alphabet/{version}/weights.bin",
        "modelSha256": sha256(model_json), "weightsSha256": sha256(weights), "bytes": len(model_json) + len(weights),
        "handUrl": f"./shared/{config.HAND_MODEL_SHA256}.task", "handSha256": config.HAND_MODEL_SHA256, "handBytes": len(hand),
        "classes": len(letters),
    }
    (out / "catalog-entry.json").write_text(json.dumps(entry, indent=2))

    write_normalize_fixture(np.load(config.CACHE / "landmarks.npz"))
    print(f"exported release {version} ({len(specs)} weight tensors, {len(weights)} bytes) to {out}")


if __name__ == "__main__":
    main()
