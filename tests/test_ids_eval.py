import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ids_eval import (confusion_matrix, f1_score, most_missed_attack,  # noqa: E402
                      precision, predict, recall)


def test_precision():
    cm = {"tp": 8, "fp": 2, "fn": 4, "tn": 16}
    assert precision(cm) == 0.8
    assert abs(recall(cm) - 8 / 12) < 1e-9
    assert precision({"tp": 0, "fp": 0, "fn": 3, "tn": 5}) == 0.0


def test_threshold():
    assert predict(0.9) == "attack"
    assert predict(0.5) == "attack"
    assert predict(0.49) == "normal"
    assert predict(0.95, threshold=0.95) == "attack"
    assert predict(0.9, threshold=0.95) == "normal"


def test_most_missed_attack():
    flows = [
        {"label": "attack", "attack_type": "DoS", "attack_prob": 0.9},
        {"label": "attack", "attack_type": "DoS", "attack_prob": 0.2},
        {"label": "attack", "attack_type": "PortScan", "attack_prob": 0.1},
        {"label": "attack", "attack_type": "PortScan", "attack_prob": 0.3},
        {"label": "normal", "attack_type": "none", "attack_prob": 0.1},
    ]
    assert most_missed_attack(flows) == ("PortScan", 2)
    assert most_missed_attack([]) == (None, 0)


def test_confusion_matrix_and_f1():
    labels = ["attack", "attack", "normal", "normal"]
    preds = ["attack", "normal", "attack", "normal"]
    assert confusion_matrix(labels, preds) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert f1_score(0.5, 0.5) == 0.5
