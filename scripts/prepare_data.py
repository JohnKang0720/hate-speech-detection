"""Build a balanced human-vs-AI dataset from HC3.

    curl -sL https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl -o data/hc3_all.jsonl
    python scripts/prepare_data.py
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PER_CLASS, MIN_WORDS, MAX_CHARS = 3000, 30, 1500


def main():
    human, ai = [], []
    for line in open(REPO / "data" / "hc3_all.jsonl"):
        row = json.loads(line)
        human += [(t.strip()[:MAX_CHARS], 0, row.get("source", "")) for t in row.get("human_answers", []) if len(t.split()) >= MIN_WORDS]
        ai += [(t.strip()[:MAX_CHARS], 1, row.get("source", "")) for t in row.get("chatgpt_answers", []) if len(t.split()) >= MIN_WORDS]

    cols = ["text", "label", "source"]
    df = pd.concat([pd.DataFrame(human, columns=cols).sample(PER_CLASS, random_state=7),
                    pd.DataFrame(ai, columns=cols).sample(PER_CLASS, random_state=7)]).sample(frac=1, random_state=7)
    out = REPO / "data" / "ai_human.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} texts -> {out.relative_to(REPO)} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
