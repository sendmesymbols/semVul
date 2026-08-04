# Component ablation -- devign at L1

Each row uses the SAME cached embeddings; only the head varies.
'delta' columns = tag - full (positive = the tag helps).

## fixed 0.5 threshold

| Config | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| full | 55.41  (+0.00) | 51.51  (+0.00) | 49.35  (+0.00) | 53.88  (+0.00) |
| no_expl | 57.18  (+1.76) | 53.37  (+1.86) | 51.19  (+1.84) | 55.75  (+1.87) |
| no_qual | 56.31  (+0.90) | 53.43  (+1.91) | 50.28  (+0.93) | 57.00  (+3.12) |
| concat | 56.24  (+0.82) | 51.81  (+0.30) | 50.21  (+0.86) | 53.52  (-0.36) |

## max balanced-accuracy threshold (honest headline)

| Config | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| full | 56.94  (+0.00) | 47.21  (+0.00) | 51.20  (+0.00) | 43.80  (+0.00) |
| no_expl | 57.18  (+0.24) | 53.37  (+6.16) | 51.19  (-0.01) | 55.75  (+11.95) |
| no_qual | 57.73  (+0.78) | 47.05  (-0.16) | 52.35  (+1.15) | 42.73  (-1.07) |
| concat | 56.63  (-0.31) | 46.57  (-0.64) | 50.79  (-0.41) | 43.00  (-0.80) |

## max-F1 threshold

| Config | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| full | 49.45  (+0.00) | 61.65  (+0.00) | 46.25  (+0.00) | 92.42  (+0.00) |
| no_expl | 51.84  (+2.39) | 62.03  (+0.38) | 47.47  (+1.22) | 89.47  (-2.94) |
| no_qual | 49.45  (+0.00) | 62.63  (+0.98) | 46.39  (+0.14) | 96.34  (+3.93) |
| concat | 50.90  (+1.45) | 61.60  (-0.05) | 46.94  (+0.69) | 89.56  (-2.85) |


FuSEVul target for reference: Acc=60.39, F1=55.91
