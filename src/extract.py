"""
Step 1: run MediaPipe HandLandmarker over every dataset image and cache the raw landmarks.

Settings mirror the app and the previous release: IMAGE mode, 2 hands, detection
confidence 0.5, EXIF transpose, no horizontal flip, the same hand_landmarker.task.
Output: cache/landmarks.npz (+ reports/extract.json with no-hand counts per letter).
"""
import hashlib
import json
import re
import sys

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageOps

import config

# Roboflow copies are named "<original>_jpg.rf.<hash>.jpg"; group them with "<original>.jpg".
GROUP = re.compile(r"^(.*?)(?:_jpg\.rf\.[0-9a-f]+)?\.jpe?g$", re.IGNORECASE)


def group_of(letter: str, name: str) -> str:
    match = GROUP.match(name)
    return f"{letter}/{match.group(1) if match else name}"


def main() -> None:
    digest = hashlib.sha256(config.HAND_MODEL.read_bytes()).hexdigest()
    if digest != config.HAND_MODEL_SHA256:
        sys.exit(f"hand_landmarker.task hash mismatch: {digest}")

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.HAND_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=config.NUM_HANDS,
        min_hand_detection_confidence=config.DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.DETECTION_CONFIDENCE,
    )
    letters = sorted(p.name for p in config.DATASET.iterdir() if p.is_dir())
    raw, slots, sizes, labels, groups, paths = [], [], [], [], [], []
    report = {}
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for label, letter in enumerate(letters):
            files = sorted(p for p in (config.DATASET / letter).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            missing = 0
            for path in files:
                image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
                width, height = image.size
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image)))
                hands = np.zeros((2, 21, 3), dtype=np.float32)
                present = np.zeros(2, dtype=np.float32)
                best = [-1.0, -1.0]
                for points, handedness in zip(result.hand_landmarks, result.handedness):
                    category = handedness[0]
                    name = (category.category_name or category.display_name or "").lower()
                    if name not in ("left", "right"):
                        continue
                    slot = (0 if name == "left" else 1) ^ int(config.SWAP_HANDS)
                    if category.score <= best[slot]:
                        continue
                    best[slot] = category.score
                    hands[slot] = [[p.x, p.y, p.z] for p in points]
                    present[slot] = 1
                if not present.any():
                    missing += 1
                    continue
                raw.append(hands)
                slots.append(present)
                sizes.append((width, height))
                labels.append(label)
                groups.append(group_of(letter, path.name))
                paths.append(f"{letter}/{path.name}")
            report[letter] = {"images": len(files), "no_hand": missing}
            print(f"{letter}: {len(files) - missing}/{len(files)} with a hand", flush=True)

    config.CACHE.mkdir(exist_ok=True)
    np.savez_compressed(
        config.CACHE / "landmarks.npz",
        raw=np.array(raw), present=np.array(slots), sizes=np.array(sizes, dtype=np.float32),
        labels=np.array(labels), groups=np.array(groups), paths=np.array(paths), letters=np.array(letters),
    )
    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "extract.json").write_text(json.dumps({"mediapipe": mp.__version__, "letters": report}, indent=2))
    print(f"saved {len(raw)} samples, mediapipe {mp.__version__}")


if __name__ == "__main__":
    main()
