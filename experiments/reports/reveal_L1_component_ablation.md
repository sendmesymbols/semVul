# Component ablation -- reveal at L1

Each row uses the SAME cached embeddings; only the head varies.
'delta' columns = tag - full (positive = the tag helps).

## fixed 0.5 threshold

| Config | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| full | 85.44  (+0.00) | 41.83  (+0.00) | 33.15  (+0.00) | 56.67  (+0.00) |
| no_expl | 86.10  (+0.66) | 43.17  (+1.34) | 34.68  (+1.53) | 57.14  (+0.48) |
| no_qual | 86.45  (+1.01) | 43.38  (+1.55) | 35.33  (+2.18) | 56.19  (-0.48) |
| concat | 84.82  (-0.62) | 41.82  (-0.01) | 32.38  (-0.77) | 59.05  (+2.38) |

## max balanced-accuracy threshold (honest headline)

| Config | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| full | 80.38  (+0.00) | 41.01  (+0.00) | 28.39  (+0.00) | 73.81  (+0.00) |
| no_expl | 77.39  (-2.99) | 38.66  (-2.34) | 25.80  (-2.59) | 77.14  (+3.33) |
| no_qual | 82.93  (+2.55) | 42.77  (+1.77) | 30.98  (+2.59) | 69.05  (-4.76) |
| concat | 79.15  (-1.23) | 40.45  (-0.55) | 27.47  (-0.91) | 76.67  (+2.86) |

## max-F1 threshold

| Config | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| full | 88.43  (+0.00) | 42.45  (+0.00) | 39.27  (+0.00) | 46.19  (+0.00) |
| no_expl | 88.96  (+0.53) | 43.08  (+0.63) | 41.13  (+1.85) | 45.24  (-0.95) |
| no_qual | 87.81  (-0.62) | 42.65  (+0.20) | 37.73  (-1.54) | 49.05  (+2.86) |
| concat | 86.85  (-1.58) | 42.61  (+0.16) | 35.69  (-3.58) | 52.86  (+6.67) |


FuSEVul target for reference: Acc=91.68, F1=46.76
