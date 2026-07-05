1. Your pipeline is NOT the same as FuSEVul

There are several major differences.

(a) Your model isn't actually using the explanations

This is the first thing I checked because your ablation results were strange.

In model.py:

output = self.attention(code_output, text_output)[:, 0, :]
output = self.fc1(code_output)

The very next line overwrites output.

Everything computed by

self.attention(code_output, text_output)

is discarded.

Then you continue with

fc1(code_output)

which uses only the code encoder output.

That means the explanation branch is effectively ignored.

So if this is the code behind the paper you're comparing against, then your explanation modality contributes almost nothing.

This is not an interpretation.

It is literally overwritten.

2. The attention module itself is unusual

Your attention computes

queries(code)
keys(code)

attention = QKᵀ

attention = attention × assist(text)

instead of standard cross-attention

Q(code)

K(text)

V(text)

or

Q(code)

K(code)

V(text)

This isn't what Transformer cross-attention does.

It may still work, but it isn't a conventional fusion mechanism.

3. You select checkpoints by accuracy, not F1

In run.py:

if eval_acc >= best_metrics['acc']:
    best_metrics = metrics

That's it.

No F1.

No recall.

No validation loss.

Only accuracy.

Yet later you compare F1.

Those objectives are inconsistent.

4. No early stopping

You simply train

100 epochs

and keep whichever epoch has the highest validation accuracy.

That's different from many papers.

5. No seed control

I noticed

#set_seed(42)

is commented out.

So every run is different.

6. No scheduler

You commented out

scheduler.step()

and warmup.

Again, different training dynamics.

7. Default CrossEntropy

You're using

CrossEntropyLoss()

No focal loss.

No class weights.

Nothing.

8. No threshold tuning

Prediction is simply

pred = argmax(logits)

Always.

That means threshold = 0.5.

Exactly like most baseline papers.

Now let's answer the real question.

Why did explanations help FuSEVul?

There are several possibilities.

Possibility 1 (most likely)

Their fusion actually uses explanations.

Yours doesn't.

Remember this line:

output = self.attention(...)

immediately followed by

output = self.fc1(code_output)

That effectively throws away the explanation representation.

If this isn't a transcription mistake, it's the biggest issue I've seen.

Possibility 2

Their GPT explanations are much higher quality.

Your explanations come from

gpt-3.5-turbo-0125

with a single prompt:

Explain the function and security risks.

FuSEVul may have used:

richer prompting,
filtering,
regeneration,
or different explanation formats.

Those details matter.

Possibility 3

CodeT5 already encodes much of the semantics

This is a real phenomenon.

Modern code LMs often already capture:

API semantics,
control flow,
vulnerability cues.

The explanation may therefore be redundant.

Possibility 4

Devign is noisy

Devign is notorious.

Many papers report:

high recall,
unstable accuracy,
unstable F1.

A modality helping on Reveal but not Devign would not be surprising.

Are your flagship ideas hurting?

This is where I disagree with your conclusion.

Your observation tables show changes like

61.44

61.56

61.47

These are tiny.

They are within the reported standard deviations.

Scientifically, that's not strong evidence that the components are harmful.

It's evidence that they don't produce a measurable improvement on this dataset under your evaluation.

Those are different conclusions.

The committee's question

You said:

They'll ask, "If explanations are worse than code, what's the point?"

I would answer it differently.

The purpose of your research is not necessarily to prove that explanations always outperform code.

Instead, your research asks whether LLM-generated semantic information can complement source-code representations.

If, after careful experimentation, the answer is "not consistently on Devign, but perhaps on other datasets or under different operating points," that's still a valid scientific result.

However, before defending that conclusion, I think you need to resolve the issue in model.py. If the explanation branch is genuinely being overwritten and ignored, then your current implementation cannot fairly evaluate whether explanations help. That's the first thing I would investigate before making any claims about the effectiveness of explanations.