"""Evaluate an intrusion detection model.

Each network flow has a true label ("attack" or "normal") and the model's
predicted probability that the flow is an attack. These functions turn the
probabilities into predictions and measure how good the model is.
"""


def predict(prob, threshold=0.5):
    """Return "attack" if the probability reaches the threshold, else "normal".

    A probability exactly equal to the threshold counts as an attack.
    """
    if prob > threshold:
        return "attack"
    return "normal"


def confusion_matrix(labels, predictions):
    """Count true/false positives and negatives. "attack" is the positive class."""
    cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for actual, predicted in zip(labels, predictions):
        if actual == "attack" and predicted == "attack":
            cm["tp"] += 1
        elif actual == "normal" and predicted == "attack":
            cm["fp"] += 1
        elif actual == "attack" and predicted == "normal":
            cm["fn"] += 1
        else:
            cm["tn"] += 1
    return cm


def precision(cm):
    """Of all flows flagged as attacks, the fraction that really were attacks."""
    flagged = cm["tp"] + cm["fn"]
    return cm["tp"] / flagged if flagged else 0.0


def recall(cm):
    """Of all real attacks, the fraction the model caught."""
    attacks = cm["tp"] + cm["fn"]
    return cm["tp"] / attacks if attacks else 0.0


def f1_score(p, r):
    """Harmonic mean of precision and recall."""
    return 2 * p * r / (p + r) if (p + r) else 0.0


def most_missed_attack(flows, threshold=0.5):
    """Return (attack_type, count) for the attack type with the most missed flows.

    A missed flow is a real attack that the model predicted as normal.
    `flows` is a list of dicts with keys "label", "attack_type", "attack_prob".
    """
    missed = {}
    for flow in flows:
        if flow["label"] != "attack":
            continue
        missed.setdefault(flow["attack_type"], 0)
        if predict(flow["attack_prob"], threshold) == "normal":
            missed[flow["attack_type"]] += 1
    if not missed:
        return None, 0
    worst = min(missed, key=missed.get)
    return worst, missed[worst]
