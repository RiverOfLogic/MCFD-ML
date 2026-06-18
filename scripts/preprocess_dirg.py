"""
DIRG dataset preprocessing — temporal split + train-stat normalization.
1. Each file: temporal split — first 70% signal → train_raw, last 30% → test_pool_raw
2. Compute per-channel mean/std from ALL train_raw segments (global train stats)
3. Normalize all segments (train + test_pool) with train stats
4. Sliding window each segment
5. Random 50/50 split test_pool windows into val/test
"""

import os
import numpy as np
import scipy.io as sio
from pathlib import Path


def extract_signal(mat, expected_channels=6):
    ignore = {'__header__', '__version__', '__globals__'}
    keys = [k for k in mat.keys() if k not in ignore]
    if not keys:
        raise ValueError("MAT file contains no data variables")
    for k in keys:
        if hasattr(mat[k], 'shape') and expected_channels in mat[k].shape:
            return mat[k]
    return mat[keys[0]]


def parse_filename(file_name):
    """C{label}A_{speed}_{load}_{condition}.mat -> (label, speed, load)"""
    parts = file_name.replace('.mat', '').split('_')
    label = int(parts[0][1])
    speed = int(parts[1])
    load = int(parts[2])
    return label, speed, load


def main(raw_dir, save_dir, window_size=2048, stride=2048):
    raw_dir = Path(raw_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(f for f in os.listdir(raw_dir) if f.endswith('.mat') and f.startswith('C'))
    if not all_files:
        print(f"No .mat files found in {raw_dir}")
        return

    rng = np.random.RandomState(42)

    train_x, train_y, train_info = [], [], []
    val_x, val_y, val_info = [], [], []
    test_x, test_y, test_info = [], [], []

    for fname in all_files:
        label, speed, load = parse_filename(fname)
        mat = sio.loadmat(os.path.join(raw_dir, fname))
        signal = extract_signal(mat)
        if signal.shape[1] == 6:
            signal = signal.T
        if signal.shape[0] != 6:
            continue

        L = signal.shape[1]
        n_train_t = int(L * 0.7)
        if n_train_t < window_size:
            continue

        train_raw = signal[:, :n_train_t]
        test_pool_raw = signal[:, n_train_t:]

        # Per-file train statistics (no data leakage)
        tr_mean = train_raw.mean(axis=1, keepdims=True)
        tr_std = train_raw.std(axis=1, keepdims=True)

        # Normalize with this file's train stats
        train_norm = (train_raw - tr_mean) / (tr_std + 1e-8)
        test_norm = (test_pool_raw - tr_mean) / (tr_std + 1e-8)

        # Window train
        train_windows = []
        L = train_norm.shape[1]
        for start in range(0, L - window_size + 1, stride):
            train_windows.append(train_norm[:, start:start + window_size])

        # Window test_pool
        test_pool_windows = []
        L = test_norm.shape[1]
        for start in range(0, L - window_size + 1, stride):
            test_pool_windows.append(test_norm[:, start:start + window_size])

        if not train_windows or len(test_pool_windows) < 2:
            continue

        # Random 50/50 val/test from test_pool
        idx = rng.permutation(len(test_pool_windows))
        n_val = len(test_pool_windows) // 2
        val_wins = [test_pool_windows[i] for i in idx[:n_val]]
        test_wins = [test_pool_windows[i] for i in idx[n_val:]]

        for w in train_windows:
            train_x.append(w)
            train_y.append(label)
            train_info.append([speed, load])
        for w in val_wins:
            val_x.append(w)
            val_y.append(label)
            val_info.append([speed, load])
        for w in test_wins:
            test_x.append(w)
            test_y.append(label)
            test_info.append([speed, load])

    print(f"Total files: {len(all_files)}")
    for split_name, (xs, ys, inf) in [
        ('train', (train_x, train_y, train_info)),
        ('val', (val_x, val_y, val_info)),
        ('test', (test_x, test_y, test_info))
    ]:
        X = np.array(xs, dtype=np.float32)
        Y = np.array(ys, dtype=np.int64)
        I = np.array(inf, dtype=np.int32)
        np.save(save_dir / f"{split_name}_x.npy", X)
        np.save(save_dir / f"{split_name}_y.npy", Y)
        np.save(save_dir / f"{split_name}_info.npy", I)
        print(f"{split_name}: {len(X)} samples, labels {np.unique(Y)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess DIRG dataset")
    parser.add_argument("--raw", default="raw_data/DIRG")
    parser.add_argument("--save", default="data/DIRG")
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    args = parser.parse_args()

    ROOT = Path(__file__).parent.parent
    main(ROOT / args.raw, ROOT / args.save, args.window, args.stride)
