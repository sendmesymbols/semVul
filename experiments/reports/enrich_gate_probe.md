# Enriched-explanation gate probe (frozen embeddings)

Probe trained on TRAIN, ROC/PR-AUC on VAL. Same protocol as devign_channel_probe.md. 'enr' = static-v1 enriched channel.

## devign

| Feature set | dim | LR ROC | LR PR | MLP ROC | MLP PR |
|---|---|---|---|---|---|
| code_L1 | 768 | 62.78 | 58.43 | 64.46 | 60.98 |
| code_L1 + expl_orig | 1152 | 62.12 | 57.95 | 62.84 | 59.33 |
| code_L1 + expl_enr | 1152 | 62.05 | 58.04 | 63.43 | 60.04 |
| expl_orig only | 384 | 54.81 | 48.55 | 55.70 | 48.22 |
| expl_enr only | 384 | 54.35 | 47.42 | 57.00 | 49.83 |
| qual_v2 only | 22 | 53.22 | 46.17 | 53.81 | 47.36 |
| code_L1 + qual_v2 | 790 | 62.83 | 58.58 | 64.53 | 61.02 |
| code_L1 + expl_enr + qual_v2 | 1174 | 62.08 | 58.15 | 64.01 | 60.28 |
| code_L1 + expl_orig + qual (old full) | 1174 | 62.08 | 57.96 | 63.65 | 59.92 |

## reveal

| Feature set | dim | LR ROC | LR PR | MLP ROC | MLP PR |
|---|---|---|---|---|---|
| code_L1 | 768 | 83.42 | 28.97 | 85.22 | 35.72 |
| code_L1 + expl_orig | 1152 | 82.35 | 27.30 | 84.75 | 34.65 |
| code_L1 + expl_enr | 1152 | 82.48 | 28.97 | 84.43 | 34.01 |
| expl_orig only | 384 | 75.18 | 26.70 | 78.74 | 29.33 |
| expl_enr only | 384 | 77.24 | 28.67 | 80.09 | 28.87 |
| qual_v2 only | 22 | 68.39 | 21.88 | 71.25 | 23.02 |
| code_L1 + qual_v2 | 790 | 83.41 | 28.81 | 85.14 | 34.82 |
| code_L1 + expl_enr + qual_v2 | 1174 | 82.45 | 28.51 | 84.41 | 33.96 |
| code_L1 + expl_orig + qual (old full) | 1174 | 82.28 | 27.40 | 84.67 | 35.22 |

