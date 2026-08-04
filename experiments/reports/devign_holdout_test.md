# Held-out TEST-fold evaluation -- devign (mean +/- std over 5 split seeds)

Stratified 80/10/10 per seed over the deduped train+val pool (n=23542). Threshold on VAL, metrics on TEST. NOT FuSEVul's exact fold -- same dataset & protocol, different split.

FuSEVul (Devign test): Acc=60.39, F1=55.91

| Config | Policy | Acc | F1 | ROC-AUC | Acc vs base | F1 vs base |
|---|---|---|---|---|---|---|
| code_only (GCB) | fixed_0.5 | 59.38 ± 0.78 | 56.69 ± 0.80 | 64.75 ± 0.91 | LOSE -1.01 | WIN +0.78 |
| code_only (GCB) | max_bal_acc | 58.19 ± 1.41 | 59.61 ± 3.55 | 64.75 ± 0.91 | LOSE -2.20 | WIN +3.70 |
| code_only (GCB) | max_f1 | 53.41 ± 1.64 | 64.66 ± 0.40 | 64.75 ± 0.91 | LOSE -6.98 | WIN +8.75 |
| code + qual | fixed_0.5 | 58.77 ± 0.76 | 55.93 ± 0.56 | 64.32 ± 0.88 | LOSE -1.62 | TIE |
| code + qual | max_bal_acc | 58.11 ± 1.27 | 58.30 ± 3.29 | 64.32 ± 0.88 | LOSE -2.28 | WIN +2.39 |
| code + qual | max_f1 | 52.51 ± 1.20 | 64.83 ± 0.32 | 64.32 ± 0.88 | LOSE -7.88 | WIN +8.92 |
| full (code+expl+qual) | fixed_0.5 | 58.34 ± 0.98 | 55.33 ± 1.01 | 62.85 ± 1.43 | LOSE -2.05 | LOSE -0.58 |
| full (code+expl+qual) | max_bal_acc | 57.26 ± 0.60 | 58.47 ± 2.68 | 62.85 ± 1.43 | LOSE -3.13 | WIN +2.56 |
| full (code+expl+qual) | max_f1 | 52.16 ± 0.83 | 64.46 ± 0.61 | 62.85 ± 1.43 | LOSE -8.23 | WIN +8.55 |
