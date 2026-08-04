# Component ablation -- reveal at L2

Each row uses the SAME cached embeddings; only the head varies.
'delta' columns = tag - full (positive = the tag helps).

## fixed 0.5 threshold

| Config | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| full | 86.32  (+0.00) | 43.14  (+0.00) | 35.01  (+0.00) | 56.19  (+0.00) |
| no_expl | 86.10  (-0.22) | 42.34  (-0.81) | 34.32  (-0.70) | 55.24  (-0.95) |
| no_qual | 85.79  (-0.53) | 42.63  (-0.52) | 33.99  (-1.02) | 57.14  (+0.95) |
| concat | 86.36  (+0.04) | 42.59  (-0.55) | 34.85  (-0.17) | 54.76  (-1.43) |

## max balanced-accuracy threshold (honest headline)

| Config | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| full | 79.37  (+0.00) | 39.64  (+0.00) | 27.16  (+0.00) | 73.33  (+0.00) |
| no_expl | 79.89  (+0.53) | 40.10  (+0.47) | 27.67  (+0.51) | 72.86  (-0.48) |
| no_qual | 81.43  (+2.07) | 40.56  (+0.92) | 28.80  (+1.64) | 68.57  (-4.76) |
| concat | 83.02  (+3.65) | 41.34  (+1.70) | 30.36  (+3.20) | 64.76  (-8.57) |

## max-F1 threshold

| Config | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| full | 87.64  (+0.00) | 42.77  (+0.00) | 37.37  (+0.00) | 50.00  (+0.00) |
| no_expl | 88.03  (+0.40) | 42.13  (-0.64) | 38.08  (+0.71) | 47.14  (-2.86) |
| no_qual | 88.08  (+0.44) | 42.95  (+0.18) | 38.49  (+1.12) | 48.57  (-1.43) |
| concat | 88.39  (+0.75) | 42.36  (-0.41) | 39.11  (+1.75) | 46.19  (-3.81) |


FuSEVul target for reference: Acc=91.68, F1=46.76
