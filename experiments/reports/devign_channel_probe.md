# Channel-isolation probe -- devign (READ-ONLY, cached embeddings)

train n=21837 (pos 45.8%), val n=2732 (pos 43.4%). Probes trained on TRAIN, AUC reported on VAL. Threshold-independent.

| Feature set | dim | LR ROC | LR PR | MLP ROC | MLP PR |
|---|---|---|---|---|---|
| code_L1 + qual | 790 | 62.76 | 58.46 | 64.73 | 61.19 |
| code_L1 (GCB) | 768 | 62.78 | 58.43 | 64.46 | 60.98 |
| code_L2 (GCB+UniX) | 1536 | 62.66 | 58.10 | 63.78 | 60.40 |
| code_L1 + expl + qual (=full L1) | 1174 | 62.07 | 57.95 | 63.65 | 59.92 |
| code_L2 + expl + qual (=full L2) | 1942 | 61.98 | 57.42 | 63.14 | 59.96 |
| code_L1 + expl | 1152 | 62.13 | 57.96 | 62.84 | 59.33 |
| expl + qual | 406 | 54.81 | 48.72 | 55.88 | 48.77 |
| expl_only | 384 | 54.80 | 48.56 | 55.70 | 48.22 |
| qual_only | 22 | 52.84 | 45.50 | 53.14 | 45.91 |
