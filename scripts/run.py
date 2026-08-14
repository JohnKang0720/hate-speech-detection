"""Train the detectors, print metrics, save figures, export the web demo."""
import json
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance

from ghostwriter import data, evaluate, viz
from ghostwriter.detectors import AttentionDetector, StyloGB, StyloLogistic
from ghostwriter.features import FEATURE_NAMES, feature_matrix

REPO = Path(__file__).resolve().parent.parent
FIG, DOCS = REPO / "figures", REPO / "docs"


def main():
    df = data.load_dataset()
    train, val, test = data.split(df, seed=0)
    Xtr, ytr = train.text.tolist(), train.label.to_numpy()
    Xte, yte = test.text.tolist(), test.label.to_numpy()
    print(f"{len(df):,} texts — train {len(Xtr)} / test {len(Xte)}\n")

    logi = StyloLogistic().fit(Xtr, ytr)
    gb = StyloGB().fit(Xtr, ytr)
    attn = AttentionDetector().fit(Xtr, ytr)

    results = {"Stylo-Logistic": (yte, logi.predict_proba(Xte)),
               "Stylo-GB": (yte, gb.predict_proba(Xte)),
               "BiLSTM-Attention": (yte, attn.predict_proba(Xte))}
    metrics = {name: evaluate.score(y, p) for name, (y, p) in results.items()}

    print(f"{'detector':<20}{'Acc':>7}{'F1':>7}{'AUC':>7}")
    for name, m in metrics.items():
        print(f"{name:<20}{m['accuracy']:>7.3f}{m['f1']:>7.3f}{m['auc']:>7.3f}")

    ai = [t for t, l in zip(Xte, yte) if l == 1]
    before = float((gb.predict_proba(ai) >= 0.5).mean())
    after = float((gb.predict_proba(evaluate.attack(ai)) >= 0.5).mean())
    metrics["_adversarial"] = {"before": before, "after": after}
    print(f"\nStylo-GB on AI text: {before:.3f} → {after:.3f} after a light paraphrase attack")

    feats = feature_matrix(Xte)
    burst = feats[:, FEATURE_NAMES.index("std_sentence_len")]
    viz.distribution(burst[yte == 0], burst[yte == 1], "sentence-length std (burstiness)",
                     "Humans write burstier sentences", FIG / "burstiness.png")
    logppl = np.log1p([gb.lm.perplexity(t) for t in Xte])
    viz.distribution(logppl[yte == 0], logppl[yte == 1], "log perplexity (human-trained LM)",
                     "AI text is 'smoother' to a human LM", FIG / "perplexity.png")
    viz.roc(results, FIG / "roc.png")
    imp = permutation_importance(gb.clf, gb._matrix(Xte), yte, n_repeats=5, random_state=0)
    viz.feature_importance(gb.feature_names, imp.importances_mean, FIG / "feature_importance.png")
    viz.calibration(yte, results["BiLSTM-Attention"][1], FIG / "calibration.png")
    viz.adversarial(before, after, FIG / "adversarial.png")
    toks, w = attn.attention(ai[0][:600])
    viz.attention_example(toks, w, float(attn.predict_proba([ai[0][:600]])[0]), FIG / "attention.png")
    (FIG / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved figures -> {FIG.relative_to(REPO)}/")

    export_web(df, {k: v for k, v in metrics.items() if not k.startswith("_")}, metrics["_adversarial"])


def export_web(df, metrics, adversarial):
    logi = StyloLogistic().fit(df.text.tolist(), df.label.to_numpy())
    ex = lambda label: df[(df.label == label) & df.text.str.len().between(220, 700)].text.head(3).tolist()
    (DOCS / "figures").mkdir(parents=True, exist_ok=True)
    (DOCS / "data.json").write_text(json.dumps({
        "model": logi.export(), "metrics": metrics, "adversarial": adversarial,
        "examples": {"human": ex(0), "ai": ex(1)}}, separators=(",", ":")))
    for f in ["attention", "roc", "burstiness", "perplexity", "feature_importance", "adversarial"]:
        (DOCS / "figures" / f"{f}.png").write_bytes((FIG / f"{f}.png").read_bytes())
    print(f"Exported demo -> {DOCS.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
