"""
MAFAULDA dataset preprocessing — temporal split + per-file train-stat normalization.
Each file normalizes using its own train portion mean/std.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def collect_files(root_dir):
    entries = []
    for label_name in sorted(os.listdir(root_dir)):
        label_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        try:
            label = int(label_name)
        except ValueError:
            continue
        for load_name in os.listdir(label_path):
            load_path = os.path.join(label_path, load_name)
            if not os.path.isdir(load_path):
                continue
            try:
                load = int(load_name.replace('g', ''))
            except ValueError:
                continue
            for file_name in os.listdir(load_path):
                if not file_name.endswith('.csv'):
                    continue
                try:
                    speed_part = os.path.splitext(file_name)[0]
                    speed = int(float(speed_part))
                except ValueError:
                    continue
                full_path = os.path.join(load_path, file_name)
                entries.append((full_path, label, speed, load))
    return entries


def main(raw_dir, save_dir, window_size=2048, stride=2048, expected_channels=8):
    raw_dir = Path(raw_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    entries = collect_files(raw_dir)
    if not entries:
        print(f"No CSV files found in {raw_dir}")
        return

    rng = np.random.RandomState(42)
    train_x, train_y, train_info = [], [], []
    val_x, val_y, val_info = [], [], []
    test_x, test_y, test_info = [], [], []

    for fpath, label, speed, load in entries:
        try:
            df = pd.read_csv(fpath, header=None)
            signal = df.values
            if signal.ndim != 2:
                continue
            if signal.shape[1] == expected_channels:
                signal = signal.T
            elif signal.shape[0] != expected_channels:
                continue

            L = signal.shape[1]
            n_train_t = int(L * 0.7)
            if n_train_t < window_size:
                continue

            train_raw = signal[:, :n_train_t]
            test_pool_raw = signal[:, n_train_t:]

            tr_mean = train_raw.mean(axis=1, keepdims=True)
            tr_std = train_raw.std(axis=1, keepdims=True)

            train_norm = (train_raw - tr_mean) / (tr_std + 1e-8)
            test_norm = (test_pool_raw - tr_mean) / (tr_std + 1e-8)

            train_windows = []
            for start in range(0, train_norm.shape[1] - window_size + 1, stride):
                train_windows.append(train_norm[:, start:start + window_size])

            test_pool_windows = []
            for start in range(0, test_norm.shape[1] - window_size + 1, stride):
                test_pool_windows.append(test_norm[:, start:start + window_size])

            if not train_windows or len(test_pool_windows) < 2:
                continue

            idx = rng.permutation(len(test_pool_windows))
            n_val = len(test_pool_windows) // 2
            val_wins = [test_pool_windows[i] for i in idx[:n_val]]
            test_wins = [test_pool_windows[i] for i in idx[n_val:]]

            for w in train_windows:
                train_x.append(w); train_y.append(label); train_info.append([speed, load])
            for w in val_wins:
                val_x.append(w); val_y.append(label); val_info.append([speed, load])
            for w in test_wins:
                test_x.append(w); test_y.append(label); test_info.append([speed, load])
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            continue

    print(f"Total files: {len(entries)}")
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
        print(f"{split_name}: {len(X)} samples")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess MAFAULDA dataset")
    parser.add_argument("--raw", default="raw_data/MAFDATA")
    parser.add_argument("--save", default="data/MAFAULDA")
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    args = parser.parse_args()
    ROOT = Path(__file__).parent.parent
    main(ROOT / args.raw, ROOT / args.save, args.window, args.stride)
