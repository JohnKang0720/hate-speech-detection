import numpy as np

from ghostwriter.detectors import StyloLogistic
from ghostwriter.evaluate import attack
from ghostwriter.features import FEATURE_NAMES, TrigramLM, extract_features

HUMAN = ["ok so basically the sun is hot. really hot!! nobody knows why exactly lol"] * 6
AI = ["The Sun is a star composed primarily of hydrogen and helium. It generates "
      "energy through nuclear fusion, converting hydrogen into helium in its core."] * 6


def test_features_complete():
    f = extract_features("Hello there. This is a test! Really?")
    assert list(f) == FEATURE_NAMES


def test_perplexity_higher_out_of_domain():
    lm = TrigramLM().fit(["the cat sat on the mat"] * 20)
    assert lm.perplexity("the cat sat on the mat") < lm.perplexity("quantum entropy zebra")


def test_logistic_runs_and_exports():
    logi = StyloLogistic().fit(HUMAN + AI, [0] * 6 + [1] * 6)
    p = logi.predict_proba(HUMAN + AI)
    assert p.shape == (12,) and np.all((p >= 0) & (p <= 1))
    assert len(logi.export()["coef"]) == len(FEATURE_NAMES)


def test_attack_preserves_most_words():
    out = attack(["the quick brown fox jumps over the lazy dog again and again"], seed=1)[0]
    assert len(out.split()) >= 6
