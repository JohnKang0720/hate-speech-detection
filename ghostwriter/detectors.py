"""Three detectors: logistic, gradient boosting, and a BiLSTM with attention."""
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, TrigramLM, feature_matrix, tokenize

PAD, UNK = 0, 1


class StyloLogistic:
    def __init__(self):
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    def fit(self, texts, y):
        self.pipe.fit(feature_matrix(texts), y)
        return self

    def predict_proba(self, texts):
        return self.pipe.predict_proba(feature_matrix(texts))[:, 1]

    def export(self):
        scaler, clf = self.pipe.steps[0][1], self.pipe.steps[1][1]
        return {"features": FEATURE_NAMES, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}


class StyloGB:
    """Stylometric features + n-gram perplexity, gradient boosted.

    Perplexity is cross-fit: the LM scoring a human text was never trained on it,
    otherwise held-out humans look 'unfamiliar' and get flagged as AI.
    """

    def __init__(self, folds=4, seed=0):
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=4, random_state=seed)
        self.folds, self.seed = folds, seed

    def fit(self, texts, y):
        texts, y = list(texts), np.asarray(y)
        self.lm = TrigramLM().fit([t for t, l in zip(texts, y) if l == 0])
        ppl = np.zeros(len(texts))
        human = np.where(y == 0)[0]
        rng = np.random.default_rng(self.seed)
        rng.shuffle(human)
        for fold in np.array_split(human, self.folds):
            lm = TrigramLM().fit([texts[i] for i in np.setdiff1d(human, fold)])
            ppl[fold] = [lm.perplexity(texts[i]) for i in fold]
        ai = np.where(y == 1)[0]
        ppl[ai] = [self.lm.perplexity(texts[i]) for i in ai]
        self.clf.fit(np.hstack([feature_matrix(texts), np.log1p(ppl)[:, None]]), y)
        return self

    def _matrix(self, texts):
        ppl = np.log1p([self.lm.perplexity(t) for t in texts])
        return np.hstack([feature_matrix(texts), ppl[:, None]])

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._matrix(list(texts)))[:, 1]

    @property
    def feature_names(self):
        return FEATURE_NAMES + ["log_perplexity"]


class AttnLSTM(nn.Module):
    def __init__(self, vocab, emb=64, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden, 1)
        self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hidden, 2))

    def forward(self, x, return_attn=False):
        mask = x != PAD
        h, _ = self.lstm(self.embedding(x))
        scores = self.attn(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = scores.softmax(1)
        logits = self.out((weights.unsqueeze(1) @ h).squeeze(1))
        return (logits, weights) if return_attn else logits


class AttentionDetector:
    """BiLSTM + additive attention — the mechanism carried over from hate-speech."""

    def __init__(self, maxlen=200, seed=0):
        self.maxlen, self.seed = maxlen, seed

    def _encode(self, texts):
        rows = []
        for x in texts:
            ids = [self.vocab.get(t, UNK) for t in tokenize(x)][:self.maxlen]
            rows.append(ids + [PAD] * (self.maxlen - len(ids)))
        return torch.tensor(rows)

    def fit(self, texts, y, epochs=4, batch_size=64, lr=1e-3):
        torch.manual_seed(self.seed)
        texts, y = list(texts), np.asarray(y)
        counts = Counter(t for x in texts for t in tokenize(x))
        self.vocab = {"<pad>": PAD, "<unk>": UNK}
        for tok, c in counts.most_common(20_000):
            if c >= 2:
                self.vocab[tok] = len(self.vocab)

        self.model = AttnLSTM(len(self.vocab))
        opt = torch.optim.Adam(self.model.parameters(), lr)
        loss_fn = nn.CrossEntropyLoss()
        X, yt = self._encode(texts), torch.tensor(y)
        rng = np.random.default_rng(self.seed)
        for _ in range(epochs):
            self.model.train()
            for b in np.array_split(rng.permutation(len(texts)), len(texts) // batch_size):
                opt.zero_grad()
                loss_fn(self.model(X[b]), yt[b]).backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, texts):
        self.model.eval()
        X = self._encode(texts)
        return torch.cat([self.model(X[s:s + 128]).softmax(1)[:, 1] for s in range(0, len(X), 128)]).numpy()

    @torch.no_grad()
    def attention(self, text):
        self.model.eval()
        toks = tokenize(text)[:self.maxlen]
        _, w = self.model(self._encode([text]), return_attn=True)
        return toks, w[0, :len(toks)].numpy()
