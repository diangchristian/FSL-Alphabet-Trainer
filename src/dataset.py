"""
Step 2 input: turn cached landmarks into training sequences.

- Group-aware split: an original photo and its Roboflow copies always land in the
  same split, so test accuracy isn't measured on near-duplicates of training images.
- A still image becomes a 32-frame sequence by repetition (the app does the same with
  one camera frame) plus small per-timestep noise, so the model tolerates jitter.
- Training-only augmentation: in-plane rotation and left/right mirroring.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import config
from features import mirror, rotate, to_frame

NOISE_STD = 0.03          # in palm lengths; live landmarks wobble frame to frame
ROTATION_DEGREES = 15.0
MIRROR_PROBABILITY = 0.5


def load():
    data = np.load(config.CACHE / "landmarks.npz")
    frames = np.stack([
        to_frame([(slot, data["raw"][i][slot]) for slot in (0, 1) if data["present"][i][slot]], *data["sizes"][i])
        for i in range(len(data["labels"]))
    ])
    keep = frames[:, 126:].any(axis=1)  # a degenerate (zero palm) detection leaves no hand
    return frames[keep], data["labels"][keep], data["groups"][keep], [str(x) for x in data["letters"]]


def split(labels, groups):
    """Returns train/val/test index arrays, ~70/15/15, stratified by letter, grouped by photo."""
    folds = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=config.SEED)
    chunks = [test for _, test in folds.split(np.zeros(len(labels)), labels, groups)]
    n_test = round(20 * config.TEST_FRACTION)
    n_val = round(20 * config.VALIDATION_FRACTION)
    test = np.concatenate(chunks[:n_test])
    val = np.concatenate(chunks[n_test:n_test + n_val])
    train = np.concatenate(chunks[n_test + n_val:])
    return train, val, test


def sequence(frame, rng, augment):
    if augment:
        if rng.random() < MIRROR_PROBABILITY:
            frame = mirror(frame)
        frame = rotate(frame, rng.uniform(-ROTATION_DEGREES, ROTATION_DEGREES))
    seq = np.repeat(frame[None, :], config.FRAMES, axis=0)
    noise = rng.normal(0, NOISE_STD, size=(config.FRAMES, 126)).astype(np.float32)
    present = np.repeat(seq[:, 126:], 63, axis=1)  # only perturb present hands
    seq[:, :126] += noise * present
    return seq


def batches(frames, labels, batch_size, rng, augment):
    """Endless generator of (x, y) batches for Keras."""
    while True:
        order = rng.permutation(len(frames))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            yield np.stack([sequence(frames[i], rng, augment) for i in idx]), labels[idx]


def fixed(frames, rng):
    """Deterministic evaluation sequences (noise, no augmentation)."""
    return np.stack([sequence(f, rng, False) for f in frames])
