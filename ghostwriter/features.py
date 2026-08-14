"""Turn text into stylometric features, and score its perplexity."""
import math
import re
from collections import Counter

import numpy as np

STOPWORDS = {"the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
             "with", "as", "at", "by", "is", "are", "was", "were", "be", "been", "it",
             "this", "that", "these", "those", "i", "you", "he", "she", "they", "we",
             "his", "her", "their", "our", "its", "not", "no", "do", "does", "did",
             "so", "than", "then", "there", "here", "which", "who", "what", "when"}

FEATURE_NAMES = ["mean_sentence_len", "std_sentence_len", "mean_word_len", "std_word_len",
                 "type_token_ratio", "stopword_ratio", "comma_ratio", "period_ratio",
                 "question_ratio", "exclaim_ratio", "digit_ratio", "upper_word_ratio",
                 "bigram_repeat_ratio", "commas_per_sentence"]

WORD = re.compile(r"[A-Za-z']+")
TOKEN = re.compile(r"[a-z']+|[.,!?;:]")


def tokenize(text):
    return TOKEN.findall(text.lower())


def extract_features(text):
    text = (text or "").strip()
    chars = len(text) or 1
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    sent_lens = [len(WORD.findall(s)) for s in sentences] or [0]
    words = WORD.findall(text)
    n = len(words) or 1
    lower = [w.lower() for w in words]
    word_lens = [len(w) for w in words] or [0]
    bigrams = list(zip(lower, lower[1:]))
    return {
        "mean_sentence_len": np.mean(sent_lens),
        "std_sentence_len": np.std(sent_lens),
        "mean_word_len": np.mean(word_lens),
        "std_word_len": np.std(word_lens),
        "type_token_ratio": len(set(lower)) / n,
        "stopword_ratio": sum(w in STOPWORDS for w in lower) / n,
        "comma_ratio": text.count(",") / n,
        "period_ratio": text.count(".") / n,
        "question_ratio": text.count("?") / n,
        "exclaim_ratio": text.count("!") / n,
        "digit_ratio": sum(c.isdigit() for c in text) / chars,
        "upper_word_ratio": sum(w.isupper() and len(w) > 1 for w in words) / n,
        "bigram_repeat_ratio": 1 - len(set(bigrams)) / max(len(bigrams), 1),
        "commas_per_sentence": text.count(",") / max(len(sentences), 1),
    }


def feature_matrix(texts):
    return np.array([list(extract_features(t).values()) for t in texts])


class TrigramLM:
    """Word-trigram language model; perplexity is used as a detector feature."""

    def __init__(self, k=0.5, max_vocab=20_000):
        self.k, self.max_vocab = k, max_vocab

    def fit(self, texts):
        self.tri, self.bi, unigrams = Counter(), Counter(), Counter()
        for t in texts:
            toks = ["<s>", "<s>"] + tokenize(t) + ["</s>"]
            unigrams.update(toks)
            for g in zip(toks, toks[1:], toks[2:]):
                self.tri[g] += 1
                self.bi[g[:2]] += 1
        self.vocab = {w for w, _ in unigrams.most_common(self.max_vocab)}
        return self

    def perplexity(self, text):
        toks = ["<s>", "<s>"] + tokenize(text) + ["</s>"]
        grams = list(zip(toks, toks[1:], toks[2:]))
        V = len(self.vocab)
        ll = sum(math.log((self.tri[g] + self.k) / (self.bi[g[:2]] + self.k * V)) for g in grams) / len(grams)
        return math.exp(-ll)
