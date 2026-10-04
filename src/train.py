"""
Step 2: train the conv1d + BiLSTM alphabet model on hand-relative landmarks.
Writes cache/model.keras, cache/split.npz and reports/{metrics.json,confusion.csv,confusion.png}.
"""
import json

import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import config
import dataset
from model import build

BATCH = 64
EPOCHS = 80


def main() -> None:
    keras.utils.set_random_seed(config.SEED)
    frames, labels, groups, letters = dataset.load()
    train, val, test = dataset.split(labels, groups)
    print(f"samples train {len(train)} / val {len(val)} / test {len(test)}; letters {len(letters)}")

    rng = np.random.default_rng(config.SEED)
    x_val = dataset.fixed(frames[val], np.random.default_rng(1))
    x_test = dataset.fixed(frames[test], np.random.default_rng(2))

    model = build(len(letters))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        dataset.batches(frames[train], labels[train], BATCH, rng, augment=True),
        steps_per_epoch=int(np.ceil(len(train) / BATCH)),
        validation_data=(x_val, labels[val]),
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
        ],
        verbose=2,
    )

    probs = model.predict(x_test, verbose=0)
    predicted = probs.argmax(axis=1)
    top = probs.max(axis=1)
    accuracy = float((predicted == labels[test]).mean())
    matrix = confusion_matrix(labels[test], predicted, labels=range(len(letters)))
    confused = sorted(
        ((int(matrix[i, j]), letters[i], letters[j]) for i in range(len(letters)) for j in range(len(letters)) if i != j and matrix[i, j]),
        reverse=True,
    )[:10]
    metrics = {
        "test_accuracy": accuracy,
        "test_samples": int(len(test)),
        # The app accepts a frame only at >= 0.8 confidence; this is how often a correct sign clears it.
        "correct_and_confident_at_0.8": float(((predicted == labels[test]) & (top >= 0.8)).mean()),
        "wrong_but_confident_at_0.8": float(((predicted != labels[test]) & (top >= 0.8)).mean()),
        "most_confused": [f"{a}->{b} x{n}" for n, a, b in confused],
        "per_letter": classification_report(labels[test], predicted, target_names=letters, output_dict=True, zero_division=0),
    }
    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savetxt(config.REPORTS / "confusion.csv", matrix, fmt="%d", delimiter=",", header=",".join(letters))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(letters)), letters)
        ax.set_yticks(range(len(letters)), letters)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        fig.savefig(config.REPORTS / "confusion.png", dpi=120, bbox_inches="tight")
    except Exception as error:  # the plot is a convenience; metrics.json is the record
        print("confusion plot skipped:", error)

    model.save(config.CACHE / "model.keras")
    np.savez(config.CACHE / "split.npz", train=train, val=val, test=test)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_letter"}, indent=2))


if __name__ == "__main__":
    main()
