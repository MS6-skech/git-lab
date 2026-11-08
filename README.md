# Intrusion Detecton Evaluator

A small Python program that evaluates a machine learning intrusion detection
system (IDS). `data/flows.csv` contains network flows with their true label
(`attack` or `normal`), the attack type, and the probability of attack predicted
by a trained classifier. The program turns those probabilities into predictions
with a decision threshold and prints the confusion matrix, precision, recall,
F1 score and the attack type the model misses most.

The report it prints is wrong. In this lab you fix it while practising the
whole Git and GitHub workflow. Follow the assignment sheet on Moodle.

## Project structure

```
ids_eval.py          # prediction and evaluation functions
main.py              # prints the evaluation report
data/flows.csv       # labelled flows with model predictions (synthetic data)
tests/               # tests for ids_eval.py (do not edit)
```

## Author

<!-- Replace this line with your full name -->
