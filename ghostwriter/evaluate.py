"""Metrics, and a light paraphrase attack for robustness."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .features import STOPWORDS

FILLERS = ["basically", "honestly", "actually", "kind of", "you know", "sort of"]


def score(y, proba, thr=0.5):
    pred = (np.asarray(proba) >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, proba)}


def attack(texts, seed=0, rate=0.06):
    """Swap adjacent words, drop a stopword, or inject a filler — meaning intact."""
    rng = np.random.default_rng(seed)
    out = []
    for text in texts:
        words = text.split()
        new, i = [], 0
        while i < len(words):
            r, w = rng.random(), words[i]
            if r < rate and i + 1 < len(words):
                new += [words[i + 1], w]; i += 2
            elif r < 2 * rate and w.lower() in STOPWORDS:
                i += 1
            elif r < 3 * rate:
                new += [FILLERS[rng.integers(len(FILLERS))], w]; i += 1
            else:
                new.append(w); i += 1
        out.append(" ".join(new))
    return out
