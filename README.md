# fsl-alphabet-model

Trains Hudyat's FSL alphabet recognizer: the paper's **conv1d + BiLSTM** network on
**hand-relative landmarks**, exported in the app's `fsl-conv1d-bilstm-v1` release format.

Compared with the previous release (trained on the same dataset with raw image
coordinates), landmarks are made relative to the wrist and scaled by palm length,
in pixels. Where the hand is in the camera, how big it is, and the camera's aspect ratio
no longer change the model's input. Training adds rotation and left/right mirror
augmentation, and the train/val/test split keeps each photo with its Roboflow copies.

## Setup (Windows, Python 3.12 — MediaPipe/TensorFlow don't support 3.14 yet)

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Dataset: `C:\Users\chris\Downloads\archive\Collated\<letter>\*.jpg` (set in `src/config.py`).
`assets/hand_landmarker.task` is the same hand model the app downloads (sha256 checked).

## Run

```
.venv\Scripts\python src\extract.py    # MediaPipe landmarks -> cache/landmarks.npz, reports/extract.json
.venv\Scripts\python src\train.py      # train -> cache/model.keras, reports/metrics.json + confusion matrix
.venv\Scripts\python src\export.py     # release -> releases/<version>/ + fixtures/normalize.json
.venv\Scripts\python src\publish.py    # copy into fsl-pwa as the *candidate* alphabet model
```

## Test in the app, then publish

1. In fsl-pwa (`client`): `npm test` — `tests/normalize.test.mjs` checks the browser's
   `normalizeFrame` against `fixtures/normalize.json` from this project.
2. Open a practice page with `?lab=1` → **Alphabet model: retrained candidate**, and compare against
   **published** on laptop/phone/tablet (time to recognize, wrong guesses, lock resets).
3. If it wins: in `fsl-pwa/server/model-assets/catalog.json`, replace the alphabet entry with
   `releases/<version>/catalog-entry.json`, then from `fsl-pwa/server` run
   `npx tsx prisma/scripts/registerModels.ts`. The previous release stays on disk for rollback.

`releases/<version>/model.json` records the training settings and test-set evaluation
(including how often a correct letter clears the app's 0.8 confidence gate).
