from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, roc_curve

INK, MUTE, GRID = "#141414", "#8a8a8a", "#e6e6e6"
HUMAN, AI = "#4c6ef5", "#e8590c"


def style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save_fig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def distribution(human, ai, xlabel, title, path):
    fig, ax = plt.subplots(figsize=(7, 4)); style_axis(ax)
    bins = np.histogram_bin_edges(np.concatenate([human, ai]), bins=40)
    ax.hist(human, bins=bins, color=HUMAN, alpha=0.6, label="Human", density=True)
    ax.hist(ai, bins=bins, color=AI, alpha=0.6, label="AI", density=True)
    ax.set_xlabel(xlabel); ax.set_ylabel("density")
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    save_fig(fig, path)


def roc(results, path):
    fig, ax = plt.subplots(figsize=(6.2, 5)); style_axis(ax)
    for (name, (y, p)), col in zip(results.items(), [INK, "#5c5c5c", "#a0a0a0"]):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color=MUTE, ls="--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — human vs AI", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    save_fig(fig, path)


def feature_importance(names, importances, path):
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 5)); style_axis(ax)
    ax.barh(np.array(names)[order], np.array(importances)[order], color=INK)
    ax.set_xlabel("Permutation importance")
    ax.set_title("What gives AI text away", color=INK, fontsize=12, loc="left")
    save_fig(fig, path)


def calibration(y, proba, path):
    frac, mean = calibration_curve(y, proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5)); style_axis(ax)
    ax.plot(mean, frac, marker="o", color=INK, lw=1.8, label="Detector")
    ax.plot([0, 1], [0, 1], color=MUTE, ls="--", lw=1, label="Perfect")
    ax.set_xlabel("Predicted P(AI)"); ax.set_ylabel("Observed fraction AI")
    ax.set_title("Calibration", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    save_fig(fig, path)


def adversarial(before, after, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2)); style_axis(ax)
    ax.bar(["Original", "Lightly edited"], [before, after], color=[INK, AI], width=0.55)
    for i, v in enumerate([before, after]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", color=INK, fontsize=11)
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy on AI text")
    ax.set_title("Robustness to a light paraphrase attack", color=INK, fontsize=12, loc="left")
    save_fig(fig, path)


def attention_example(tokens, weights, prob_ai, path, max_chars=72):
    """Tokens shaded by attention weight, flowing by width so nothing overlaps."""
    w = np.asarray(weights, float)
    w /= w.max() + 1e-9
    lines, cur, width = [], [], 0
    for tok, a in zip(tokens, w):
        if width + len(tok) + 1 > max_chars and cur:
            lines.append(cur); cur, width = [], 0
        cur.append((tok, a)); width += len(tok) + 1
    lines.append(cur)

    fig, ax = plt.subplots(figsize=(max_chars * 0.13, 0.42 * len(lines) + 0.9))
    ax.axis("off"); ax.set_xlim(0, max_chars); ax.set_ylim(0, len(lines) + 1)
    ax.set_title(f"Where the model looks   ·   P(AI) = {prob_ai:.2f}", color=INK, fontsize=12, loc="left")
    for r, line in enumerate(lines):
        x = 0
        for tok, a in line:
            ax.text(x, len(lines) - r, tok, fontsize=11, family="monospace", va="center", ha="left",
                    color=INK if a < 0.55 else "white",
                    bbox=dict(boxstyle="square,pad=0.25", facecolor=(0.08, 0.08, 0.08, float(a)), edgecolor="none"))
            x += len(tok) + 1
    save_fig(fig, path)
