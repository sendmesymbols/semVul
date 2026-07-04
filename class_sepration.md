 found something important — before talking about prompt tweaks, I can now say why the explanations aren't helping, with numbers. I ran a class-separability check over all 21,837 Devign train explanations, and the explanation channel carries almost no signal that distinguishes vulnerable from safe code:

Signal	label=0 (safe)	label=1 (vulnerable)
mean # risky_operations	1.97	2.09
mean # missing_checks	2.02	2.12
mentions "bounds"	82.7%	84.5%
mentions "pointer arithmetic"	72.7%	74.1%
mentions "null check"	68.2%	71.4%
says "appears safe"/"benign"	~0%	~0%
Qwen produces essentially the same "this function has pointer arithmetic, missing bounds checks, and no null check" template for every function. A safe remove() wrapper in the val set gets "buffer overflow via snprintf" listed as a risky operation. When both classes get near-identical text, RoBERTa has nothing to learn from — the fusion model correctly learns to ignore the channel. The explanations aren't a weak feature; they're close to a constant.