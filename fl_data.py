"""fl_data.py - turns the Hugging Face FEMNIST dataset into per-writer FL clients (numpy only, no torch)."""
import numpy as np


def _to_array(im):
    im = im.convert("L")
    if im.size != (28, 28):
        im = im.resize((28, 28))
    return np.asarray(im, dtype=np.uint8)


def load_femnist_clients(ds, num_clients=100, min_samples=50, test_frac=0.2, seed=0):
    """Pick `num_clients` writers with at least `min_samples` images; each writer is one client.
    Returns (clients, X_test, y_test, num_classes). Images are uint8 arrays of shape (N, 28, 28).
    Each client keeps (1 - test_frac) of its images for training; the rest go to one shared test set."""
    rng = np.random.default_rng(seed)
    writers = np.asarray(list(ds["writer_id"]))                  # reads one column; does not decode images
    uniq, counts = np.unique(writers, return_counts=True)
    eligible = uniq[counts >= min_samples]
    if len(eligible) == 0:
        raise ValueError(f"no writer has at least {min_samples} images")
    chosen = rng.choice(eligible, size=min(num_clients, len(eligible)), replace=False)
    idx = np.where(np.isin(writers, chosen))[0]
    sub = ds.select(idx.tolist())

    sub_writers = writers[idx]
    chars = np.asarray(list(sub["character"]))
    classes, y_all = np.unique(chars, return_inverse=True)
    X_all = np.stack([_to_array(im) for im in sub["image"]])

    clients, Xte, yte = [], [], []
    for cid, w in enumerate(chosen):
        rows = np.where(sub_writers == w)[0]
        rng.shuffle(rows)
        n_test = max(1, int(round(len(rows) * test_frac)))
        te, tr = rows[:n_test], rows[n_test:]
        clients.append({"id": cid, "writer": str(w), "X": X_all[tr], "y": y_all[tr].astype(np.int64)})
        Xte.append(X_all[te]); yte.append(y_all[te])
    X_test, y_test = np.concatenate(Xte), np.concatenate(yte).astype(np.int64)
    return clients, X_test, y_test, len(classes)