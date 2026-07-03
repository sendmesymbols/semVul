[transformers] RobertaModel LOAD REPORT from: microsoft/graphcodebert-base
Key                       | Status     | Details
--------------------------+------------+--------
lm_head.dense.bias        | UNEXPECTED |        
lm_head.layer_norm.weight | UNEXPECTED |        
lm_head.decoder.weight    | UNEXPECTED |        
lm_head.dense.weight      | UNEXPECTED |        
lm_head.bias              | UNEXPECTED |        
lm_head.layer_norm.bias   | UNEXPECTED |        
lm_head.decoder.bias      | UNEXPECTED |        
pooler.dense.weight       | MISSING    |        
pooler.dense.bias         | MISSING    |        

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 499M/499M [01:50<00:00, 4.51MB/s]
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 197/197 [00:00<00:00, 9143.88it/s]
[transformers] RobertaModel LOAD REPORT from: roberta-base                                                                                                                                                 | 0/197 [00:00<?, ?it/s]
Key                       | Status     | Details
--------------------------+------------+--------
        

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 499M/499M [03:22<00:00, 2.46MB/s]
[reveal_L1] ep1/5 loss=0.0334 val acc=89.00 f1=39.90 p=40.29 r=39.52
[reveal_L1] ep2/5 loss=0.0265 val acc=84.51 f1=45.00 p=33.49 r=68.57
[reveal_L1] ep3/5 loss=0.0210 val acc=89.49 f1=48.60 p=44.31 r=53.81
[reveal_L1] ep4/5 loss=0.0165 val acc=87.24 f1=48.58 p=38.70 r=65.24
[reveal_L1] ep5/5 loss=0.0134 val acc=87.86 f1=44.13 p=38.38 r=51.90
[reveal_L1] DONE best@ep3: acc=89.49 f1=48.60 (stated {'acc': 91.68, 'f1': 46.76, 'prec': 57.24, 'rec': 39.52}) in 31.3 min

===== reveal_L2 =====
[reveal_L2] device=cuda amp=bf16 fusion=self batch=4x8
[data] reveal: train=17905 (pos 9.3%)  val=2273 (pos 9.2%)  qual_dim=22
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 197/197 [00:00<00:00, 36936.87it/s]
[transformers] RobertaModel LOAD REPORT from: microsoft/graphcodebert-base
Key                       | Status     | Details
--------------------------+------------+--------
lm_head.dense.bias        | UNEXPECTED |        
lm_head.layer_norm.weight | UNEXPECTED |        
lm_head.decoder.weight    | UNEXPECTED |        
lm_head.dense.weight      | UNEXPECTED |        
lm_head.bias              | UNEXPECTED |        
lm_head.layer_norm.bias   | UNEXPECTED |        
lm_head.decoder.bias      | UNEXPECTED |        
pooler.dense.weight       | MISSING    |        
pooler.dense.bias         | MISSING    |        

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 197/197 [00:00<00:00, 15614.31it/s]
[transformers] RobertaModel LOAD REPORT from: roberta-base
Key                       | Status     | Details
--------------------------+------------+--------
        

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
[reveal_L2] ep1/5 loss=0.0342 val acc=89.09 f1=44.39 p=41.95 r=47.14
[reveal_L2] ep2/5 loss=0.0270 val acc=90.85 f1=48.51 p=50.52 r=46.67
[reveal_L2] ep3/5 loss=0.0213 val acc=84.21 f1=44.85 p=33.11 r=69.52
[reveal_L2] ep4/5 loss=0.0168 val acc=86.36 f1=45.42 p=36.03 r=61.43
[reveal_L2] ep5/5 loss=0.0140 val acc=85.92 f1=46.31 p=35.75 r=65.71
[reveal_L2] DONE best@ep2: acc=90.85 f1=48.51 (stated {'acc': 91.68, 'f1': 46.76, 'prec': 57.24, 'rec': 39.52}) in 51.0 min

===== reveal_L3 =====
[reveal_L3] device=cuda amp=bf16 fusion=self batch=4x8
[data] reveal: train=17905 (pos 9.3%)  val=2273 (pos 9.2%)  qual_dim=22
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 197/197 [00:00<?, ?it/s]
[transformers] RobertaModel LOAD REPORT from: microsoft/graphcodebert-base
Key                       | Status     | Details
--------------------------+------------+--------
lm_head.dense.bias        | UNEXPECTED |        
lm_head.layer_norm.weight | UNEXPECTED |        
lm_head.decoder.weight    | UNEXPECTED |        
lm_head.dense.weight      | UNEXPECTED |        
lm_head.bias              | UNEXPECTED |        
lm_head.layer_norm.bias   | UNEXPECTED |        
lm_head.decoder.bias      | UNEXPECTED |        
pooler.dense.weight       | MISSING    |        
pooler.dense.bias         | MISSING    |        

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 197/197 [00:00<00:00, 5901.94it/s]
[transformers] RobertaModel LOAD REPORT from: roberta-base
Key                       | Status     | Details
--------------------------+------------+--------
        

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
[reveal_L3] ep1/5 loss=0.0688 val acc=88.56 f1=39.81 p=38.74 r=40.95
[reveal_L3] ep2/5 loss=0.0554 val acc=83.81 f1=42.14 p=31.46 r=63.81
[reveal_L3] ep3/5 loss=0.0425 val acc=90.15 f1=48.62 p=46.90 r=50.48
[reveal_L3] ep4/5 loss=0.0331 val acc=86.93 f1=46.49 p=37.39 r=61.43
[reveal_L3] ep5/5 loss=0.0256 val acc=88.12 f1=47.06 p=40.00 r=57.14
[reveal_L3] DONE best@ep3: acc=90.15 f1=48.62 (stated {'acc': 91.68, 'f1': 46.76, 'prec': 57.24, 'rec': 39.52}) in 50.8 min

===== devign_L1 =====
[devign_L1] device=cuda amp=bf16 fusion=self batch=4x8
[data] devign: train=20837 (pos 46.2%)  val=2732 (pos 43.4%)  qual_dim=22
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 197/197 [00:00<00:00, 22663.21it/s]
[transformers] RobertaModel LOAD REPORT from: microsoft/graphcodebert-base
Key                       | Status     | Details
--------------------------+------------+--------
lm_head.dense.bias        | UNEXPECTED |        
lm_head.layer_norm.weight | UNEXPECTED |        
lm_head.decoder.weight    | UNEXPECTED |        
lm_head.dense.weight      | UNEXPECTED |        
lm_head.bias              | UNEXPECTED |        
lm_head.layer_norm.bias   | UNEXPECTED |        
lm_head.decoder.bias      | UNEXPECTED |        
pooler.dense.weight       | MISSING    |        
pooler.dense.bias         | MISSING    |        

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 197/197 [00:00<00:00, 14401.36it/s]
[transformers] RobertaModel LOAD REPORT from: roberta-base
Key                       | Status     | Details
--------------------------+------------+--------
        

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
[devign_L1] ep1/5 loss=0.6852 val acc=54.50 f1=50.54 p=47.89 r=53.50
[devign_L1] ep2/5 loss=0.6728 val acc=54.87 f1=50.97 p=48.27 r=54.00
[devign_L1] ep3/5 loss=0.6500 val acc=60.94 f1=34.82 p=63.33 r=24.01
[devign_L1] ep4/5 loss=0.6171 val acc=59.30 f1=45.28 p=54.44 r=38.75
[devign_L1] ep5/5 loss=0.5690 val acc=60.94 f1=43.27 p=58.65 r=34.29
[devign_L1] DONE best@ep2: acc=54.87 f1=50.97 (stated {'acc': 60.39, 'f1': 55.91}) in 31.1 min

===== devign_L2 =====
[devign_L2] device=cuda amp=bf16 fusion=self batch=4x8
[data] devign: train=20837 (pos 46.2%)  val=2732 (pos 43.4%)  qual_dim=22
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 197/197 [00:00<00:00, 13035.04it/s]
[transformers] RobertaModel LOAD REPORT from: microsoft/graphcodebert-base
Key                       | Status     | Details
--------------------------+------------+--------
lm_head.dense.bias        | UNEXPECTED |        
lm_head.layer_norm.weight | UNEXPECTED |        
lm_head.decoder.weight    | UNEXPECTED |        
lm_head.dense.weight      | UNEXPECTED |        
lm_head.bias              | UNEXPECTED |        
lm_head.layer_norm.bias   | UNEXPECTED |        
lm_head.decoder.bias      | UNEXPECTED |        
pooler.dense.weight       | MISSING    |        
pooler.dense.bias         | MISSING    |        

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 197/197 [00:00<00:00, 9065.13it/s]
[transformers] RobertaModel LOAD REPORT from: roberta-base
Key                       | Status     | Details
--------------------------+------------+--------
        

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
[devign_L2] ep1/5 loss=0.6879 val acc=56.55 f1=48.14 p=50.00 r=46.42
[devign_L2] ep2/5 loss=0.6692 val acc=60.07 f1=37.41 p=58.63 r=27.46
[devign_L2] ep3/5 loss=0.6462 val acc=59.08 f1=48.14 p=53.56 r=43.72
[devign_L2] ep4/5 loss=0.6105 val acc=61.13 f1=44.80 p=58.48 r=36.31
[devign_L2] ep5/5 loss=0.5610 val acc=61.86 f1=47.64 p=59.03 r=39.93
[devign_L2] DONE best@ep3: acc=59.08 f1=48.14 (stated {'acc': 60.39, 'f1': 55.91}) in 58.9 min

===== devign_L3 =====
[devign_L3] device=cuda amp=bf16 fusion=self batch=4x8
[data] devign: train=20837 (pos 46.2%)  val=2732 (pos 43.4%)  qual_dim=22
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 197/197 [00:00<?, ?it/s]
[transformers] RobertaModel LOAD REPORT from: microsoft/graphcodebert-base
Key                       | Status     | Details
--------------------------+------------+--------
lm_head.dense.bias        | UNEXPECTED |        
lm_head.layer_norm.weight | UNEXPECTED |        
lm_head.decoder.weight    | UNEXPECTED |        
lm_head.dense.weight      | UNEXPECTED |        
lm_head.bias              | UNEXPECTED |        
lm_head.layer_norm.bias   | UNEXPECTED |        
lm_head.decoder.bias      | UNEXPECTED |        
pooler.dense.weight       | MISSING    |        
pooler.dense.bias         | MISSING    |        

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 197/197 [00:00<00:00, 9743.61it/s]
[transformers] RobertaModel LOAD REPORT from: roberta-base
Key                       | Status     | Details
--------------------------+------------+--------
        

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
[devign_L3] ep1/5 loss=0.8267 val acc=56.66 f1=1.66 p=58.82 r=0.84
[devign_L3] ep2/5 loss=0.7625 val acc=58.35 f1=39.72 p=53.50 r=31.59
[devign_L3] ep3/5 loss=0.7208 val acc=59.08 f1=34.92 p=56.50 r=25.27
[devign_L3] ep4/5 loss=0.6873 val acc=57.65 f1=47.34 p=51.49 r=43.81
[devign_L3] ep5/5 loss=0.6391 val acc=60.76 f1=49.43 p=56.16 r=44.14
[devign_L3] DONE best@ep5: acc=60.76 f1=49.43 (stated {'acc': 60.39, 'f1': 55.91}) in 59.0 min
# FuSEVul component ladder — results vs stated targets

L1 = CodeT5+ code only · L2 = +RoBERTa explanation (self-attn fusion) · L3 = +22 quality features. Reported on the benchmark val split.

## reveal  (stated: acc 91.68, f1 46.76)

| Rung | Acc | F1 | Prec | Rec | Acc? | F1? |
|---|---|---|---|---|---|---|
| L1 | 89.49 | 48.60 | 44.31 | 53.81 | LOSE -2.19 | WIN +1.84 |
| L2 | 90.85 | 48.51 | 50.52 | 46.67 | LOSE -0.83 | WIN +1.75 |
| L3 | 90.15 | 48.62 | 46.90 | 50.48 | LOSE -1.53 | WIN +1.86 |

## devign  (stated: acc 60.39, f1 55.91)

| Rung | Acc | F1 | Prec | Rec | Acc? | F1? |
|---|---|---|---|---|---|---|
| L1 | 54.87 | 50.97 | 48.27 | 54.00 | LOSE -5.52 | LOSE -4.94 |
| L2 | 59.08 | 48.14 | 53.56 | 43.72 | LOSE -1.31 | LOSE -7.77 |
| L3 | 60.76 | 49.43 | 56.16 | 44.14 | WIN +0.37 | LOSE -6.48 |


FuSEVul component ladder — results vs stated targets
L1 = CodeT5+ code only · L2 = +RoBERTa explanation (self-attn fusion) · L3 = +22 quality features. Reported on the benchmark val split.

reveal (stated: acc 91.68, f1 46.76)
Rung	Acc	F1	Prec	Rec	Acc?	F1?
L1	89.49	48.60	44.31	53.81	LOSE -2.19	WIN +1.84
L2	90.85	48.51	50.52	46.67	LOSE -0.83	WIN +1.75
L3	90.15	48.62	46.90	50.48	LOSE -1.53	WIN +1.86
devign (stated: acc 60.39, f1 55.91)
Rung	Acc	F1	Prec	Rec	Acc?	F1?
L1	54.87	50.97	48.27	54.00	LOSE -5.52	LOSE -4.94
L2	59.08	48.14	53.56	43.72	LOSE -1.31	LOSE -7.77
L3	60.76	49.43	56.16	44.14	WIN +0.37	LOSE -6.48