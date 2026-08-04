# Component ablation -- devign at L2

Each row uses the SAME cached embeddings; only the head varies.
'delta' columns = tag - full (positive = the tag helps).

## fixed 0.5 threshold

| Config | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| full | 56.63  (+0.00) | 55.48  (+0.00) | 50.55  (+0.00) | 61.46  (+0.00) |
| no_expl | 57.69  (+1.06) | 52.82  (-2.65) | 51.80  (+1.25) | 53.88  (-7.58) |
| no_qual | 57.18  (+0.55) | 54.99  (-0.49) | 51.11  (+0.56) | 59.50  (-1.96) |
| concat | 57.76  (+1.14) | 54.50  (-0.98) | 51.77  (+1.22) | 57.54  (-3.93) |

## max balanced-accuracy threshold (honest headline)

| Config | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| full | 56.63  (+0.00) | 55.48  (+0.00) | 50.55  (+0.00) | 61.46  (+0.00) |
| no_expl | 57.22  (+0.59) | 54.45  (-1.03) | 51.18  (+0.63) | 58.16  (-3.30) |
| no_qual | 58.12  (+1.49) | 47.95  (-7.52) | 52.85  (+2.30) | 43.89  (-17.57) |
| concat | 57.06  (+0.43) | 56.01  (+0.53) | 50.95  (+0.40) | 62.18  (+0.71) |

## max-F1 threshold

| Config | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| full | 50.35  (+0.00) | 62.48  (+0.00) | 46.78  (+0.00) | 94.02  (+0.00) |
| no_expl | 51.25  (+0.90) | 62.55  (+0.07) | 47.22  (+0.44) | 92.60  (-1.43) |
| no_qual | 50.16  (-0.20) | 62.61  (+0.13) | 46.71  (-0.07) | 94.92  (+0.89) |
| concat | 50.78  (+0.43) | 61.91  (-0.57) | 46.92  (+0.14) | 90.99  (-3.03) |


FuSEVul target for reference: Acc=60.39, F1=55.91
