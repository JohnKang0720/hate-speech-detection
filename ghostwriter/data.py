from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "ai_human.csv"


def load_dataset(path=DATA):
    return pd.read_csv(path).dropna(subset=["text"]).reset_index(drop=True)


def split(df, seed=0, val=0.15, test=0.15):
    """Stratified train/val/test split by label."""
    rng = np.random.default_rng(seed)
    parts = {0: [], 1: [], 2: []}
    for _, g in df.groupby("label"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        nt, nv = int(len(idx) * test), int(len(idx) * val)
        parts[2].append(idx[:nt]); parts[1].append(idx[nt:nt + nv]); parts[0].append(idx[nt + nv:])
    return [df.loc[np.concatenate(parts[s])].sample(frac=1, random_state=seed).reset_index(drop=True)
            for s in (0, 1, 2)]
