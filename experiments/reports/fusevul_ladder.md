# FuSEVul component ladder — results vs stated targets

L1 = GraphCodeBERT code only · L2 = +RoBERTa explanation (self-attn fusion) · L3 = +22 quality features. Val split. Threshold picked on a held-out tune slice (non-circular); argmax@0.5 shown for direct comparability. ROC/PR-AUC are threshold-free (the fair ladder-contribution measure).

## reveal  (stated: acc 91.68, f1 46.76)

| Rung | ROC-AUC | PR-AUC | Acc@0.5 | F1@0.5 | Acc(tuned) | F1(tuned) | Acc? | F1? |
|---|---|---|---|---|---|---|---|---|
| L1 | 86.87 | 49.50 | 84.95 | 45.89 | 84.95 | 45.89 | LOSE -6.73 | LOSE -0.87 |
| L2 | 84.92 | 48.11 | 87.20 | 43.50 | 85.44 | 42.03 | LOSE -6.24 | LOSE -4.73 |
| L3 | 82.23 | 49.47 | 88.87 | 47.40 | 87.73 | 47.26 | LOSE -3.95 | WIN +0.50 |

## devign  (stated: acc 60.39, f1 55.91)

| Rung | ROC-AUC | PR-AUC | Acc@0.5 | F1@0.5 | Acc(tuned) | F1(tuned) | Acc? | F1? |
|---|---|---|---|---|---|---|---|---|
| L1 | 63.21 | 59.00 | 59.26 | 50.33 | 49.01 | 62.34 | LOSE -11.38 | WIN +6.43 |
| L2 | 63.16 | 58.05 | 60.54 | 31.07 | 50.37 | 62.77 | LOSE -10.02 | WIN +6.86 |
| L3 | 62.72 | 58.79 | 57.87 | 55.37 | 48.32 | 61.82 | LOSE -12.07 | WIN +5.91 |

