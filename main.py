"""Print an evaluation report for the IDS model predictions in data/flows.csv."""

import csv
from pathlib import Path

from ids_eval import (confusion_matrix, f1_score, most_missed_attack,
                      precision, predict, recall)

DATA_FILE = Path(__file__).parent / "data" / "flows.csv"
THRESHOLD = 0.4  # catch more attacks


def load_flows(path):
    flows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["attack_prob"] = float(row["attack_prob"])
            flows.append(row)
    return flows


def main():
    flows = load_flows(DATA_FILE)
    labels = [f["label"] for f in flows]
    preds = [predict(f["attack_prob"], THRESHOLD) for f in flows]
    cm = confusion_matrix(labels, preds)
    p, r = precision(cm), recall(cm)

    print(f"IDS evaluation report (decision threshold = {THRESHOLD:.2f})")
    print(f"Flows: {len(flows)}  (attacks: {labels.count('attack')}, "
          f"normal: {labels.count('normal')})")
    print()
    print(f"{'':16}{'pred. attack':>14}{'pred. normal':>14}")
    print(f"{'actual attack':16}{cm['tp']:>14}{cm['fn']:>14}")
    print(f"{'actual normal':16}{cm['fp']:>14}{cm['tn']:>14}")
    print()
    print(f"Precision: {p:.2f}")
    print(f"Recall:    {r:.2f}")
    print(f"F1 score:  {f1_score(p, r):.2f}")
    attack_type, count = most_missed_attack(flows, THRESHOLD)
    print(f"Most missed attack type: {attack_type} ({count} missed)")


if __name__ == "__main__":
    main()
