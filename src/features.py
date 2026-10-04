"""
Landmark features, "hand-relative-v1".

The app must compute exactly the same thing (client normalizeFrame), checked by
fixtures/normalize.json. Each present hand is expressed relative to its wrist,
in pixels (so the camera's aspect ratio doesn't squash the hand), divided by the
palm length (wrist -> middle-finger knuckle) so hand size and distance don't matter.

Frame layout (128 floats, same as the app): left hand 21x(x,y,z), right hand
21x(x,y,z), left_present, right_present. Absent hands are all zeros.
"""
import numpy as np

WRIST, MIDDLE_MCP = 0, 9


def normalize_hand(points: np.ndarray, width: float, height: float) -> np.ndarray | None:
    """points: (21,3) MediaPipe image landmarks (x/width, y/height, z at x scale)."""
    px = np.asarray(points, dtype=np.float64) * np.array([width, height, width])
    origin = px[WRIST]
    palm = float(np.hypot(*(px[MIDDLE_MCP, :2] - origin[:2])))
    if not np.isfinite(palm) or palm < 1e-6:
        return None
    return ((px - origin) / palm).astype(np.float32)


def to_frame(hands: list[tuple[int, np.ndarray]], width: float, height: float) -> np.ndarray:
    """hands: [(slot, (21,3) raw landmarks)] with slot 0=left, 1=right."""
    frame = np.zeros(128, dtype=np.float32)
    for slot, points in hands:
        normalized = normalize_hand(points, width, height)
        if normalized is None:
            continue
        frame[slot * 63:(slot + 1) * 63] = normalized.reshape(-1)
        frame[126 + slot] = 1.0
    return frame


def mirror(frame: np.ndarray) -> np.ndarray:
    """Left/right mirror image: negate x and swap the hand slots (a left-handed signer)."""
    out = np.zeros_like(frame)
    for slot in (0, 1):
        hand = frame[slot * 63:(slot + 1) * 63].reshape(21, 3).copy()
        hand[:, 0] *= -1
        other = 1 - slot
        out[other * 63:(other + 1) * 63] = hand.reshape(-1)
        out[126 + other] = frame[126 + slot]
    return out


def rotate(frame: np.ndarray, degrees: float) -> np.ndarray:
    """In-plane rotation about each wrist (the origin after normalization)."""
    out = frame.copy()
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    for slot in (0, 1):
        if not frame[126 + slot]:
            continue
        hand = out[slot * 63:(slot + 1) * 63].reshape(21, 3)
        x, y = hand[:, 0].copy(), hand[:, 1].copy()
        hand[:, 0], hand[:, 1] = c * x - s * y, s * x + c * y
    return out
