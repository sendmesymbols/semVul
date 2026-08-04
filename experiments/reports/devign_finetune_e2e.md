# End-to-end fine-tune (GraphCodeBERT, full FT) -- devign held-out TEST

Same seed-2025 80/10/10 fold as holdout_test_eval.py. Encoder+classifier fine-tuned JOINTLY, best epoch by val ROC-AUC. NOT FuSEVul's exact fold.

best val ROC-AUC=63.52

FuSEVul (Devign test): Acc=60.39, F1=55.91

| Policy | Acc | F1 | Prec | Rec | ROC-AUC | PR-AUC | Acc? | F1? |
|---|---|---|---|---|---|---|---|---|
| fixed_0.5 | 59.41 | 53.32 | 56.40 | 50.56 | 63.20 | 60.96 | LOSE -0.98 | LOSE -2.59 |
| max_bal_acc | 57.66 | 62.11 | 52.68 | 75.65 | 63.20 | 60.96 | LOSE -2.73 | WIN +6.20 |
| max_f1 | 49.89 | 64.00 | 47.73 | 97.13 | 63.20 | 60.96 | LOSE -10.50 | WIN +8.09 |
