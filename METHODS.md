# Experimental protocol

## Signal processing

Common-average reference over 64 electrodes; Butterworth bandpass designed with
passband 0.5–20 Hz, stopband 0.1–40 Hz, passband loss 1.5 dB and stopband attenuation
20 dB at 240 Hz. Per-character-record `sosfiltfilt` uses odd padding of 2880 samples.
Each event uses 192 raw samples (0.8 s), with 150 samples of context on each side,
reflection at record boundaries, and polyphase resampling by 4/15 using a
301-tap Kaiser(5) FIR. The retained input is 64 × 52 × 1 float32.
This is offline, noncausal filtering. Truncated repetition curves do not measure
real-time latency. Character/event validation rejects incomplete trials rather
than silently dropping them.

Normalization is one scalar mean/std computed in float64, applied to float32
arrays using std + 1e-6. Only 61 fit characters determine CV normalization; all 85
Train characters determine final normalization. Transfer uses source statistics.

## Model

Input → temporal Conv2D(8, 1×32) → BatchNorm → spatial depthwise convolution
(64×1, multiplier 2, max-norm 1) → BatchNorm → ELU → average pool(1×4) → dropout
→ separable Conv2D(16, 1×16) → BatchNorm → ELU → average pool(1×4) → dropout.
The feature tensor is 1×3×16. Channel attention uses global pooling and bias-free
Dense(4, ReLU), Dense(16, sigmoid). Temporal attention uses Conv1D(16, 3, ReLU)
and Conv1D(16, 1, sigmoid). The residual formulations are:

- none: x
- channel: x + x*C(x)
- temporal: x + x*T(x)
- full: x + x*C(x)*T(x)

Flattened features have dimension 48. The sigmoid output has max-norm 0.25.
Channel attention acts on learned feature channels, not directly on electrode
identities. Removing a branch also changes parameter count and gate amplitude;
this is not a parameter-matched mechanism experiment.

| Model | Total parameters |
|---|---:|
| none | 2001 |
| channel | 2129 |
| temporal | 3057 |
| full | 3185 |

The full model has 3105 trainable and 80 nontrainable parameters.

## CV and training

Fixed outer split seed 42; five held-out groups of 17 characters. Each remaining
68-character development partition is split into 61 fit and seven inner-validation
characters using seed 1000 + fold number. Each Train character is outer-held-out
exactly once. Model/shuffle seeds are 42, 123 and 2026. Model seeds never alter
the outer split. All variants use identical partitions.

Adam at initial LR 0.001, batch 32, binary cross-entropy and equal class weights.
Maximum 200 epochs; early stopping on inner BCE with patience20 and min_delta0.
ReduceLROnPlateau uses factor0.5, patience5, min_delta0.0001 and min_lr0.
The earliest strictly minimum inner BCE checkpoint is used. Both dropout layers
use 0.25 in both the primary analysis and the retained matched four-way ablation.
The old core `--profile ablation` option uses 0.5 and is retained only for historical compatibility; use `run.py ablation` for the reported matched analysis.

Final primary training uses the median of 15 best epochs (three seeds × five
folds) and the epoch-wise median of their recorded learning rates, extending the
last LR when needed. It refits from scratch on Train85. The supplied primary
epochs are A-none34, A-full23, B-none21, B-full15, all batch32. Test is never used
by the training functions.

The separate transfer weights use A-none batch32/34 epochs, A-full batch64/35,
B-none batch64/33 and B-full batch32/15. Batch choice used mean inner BCE pooled
over the 15 histories, ties favoring32. These are retained historical weights;
the sensitivity search itself is outside this compact release. This pooled-inner
selection is exploratory, not an independent nested-CV performance estimate.

## Decoding and interpretation

For repetitions1…15, accumulate predicted target probabilities for each stimulus
code; choose the highest-scoring row and column. Primary endpoint: correct
characters at repetition15. Binary threshold is0.5. All three seeds are reported;
sample SD describes seed variation, not uncertainty across a participant
population. Seeds share the same test characters and are not independent subjects.
The same Test data were inspected in prior development; repeated evaluation
does not provide fresh confirmation. There are only two subjects. Positive and
negative comparisons must both be retained in manuscript interpretation.

## Supplementary analyses

The integrated analysis/ directory includes matched d25 ablation, extraction of
48-dimensional post-attention flattened features, t-SNE and supplementary metrics.
Feature sampling uses 1200 held-out events (200 target, 1000 non-target), identical
within each subject/fold across variants/seeds. Original-space silhouette is
reported separately from 2D silhouette. Illustrative t-SNE uses seed42/fold1,
perplexity30, PCA initialization, automatic learning rate and 1000 iterations;
no favorable plot selection is performed. paper_assets rebuilds archived figures,
not new embeddings. Feature extraction can recompute embeddings from CV checkpoints.

Binary metrics use the same saved event probabilities and fixed threshold0.5.
Wilson intervals apply to each model's 100 character outcomes, with binomial
assumptions; they do not describe between-subject generalization or independence
of repeated seeds. Paired character errors retain both improvements and regressions.
The nominal first temporal kernel spans 32/64=0.5s. The attention-grid stride is
16/64=0.25s; it is not the receptive-field width or evidence of physiological
localization. Temporal attention operates on three positions and 16 feature maps.

Competition II is outside this release's fresh validation scope. Its historical
short report is not validated by the present Competition III workflow. Historical
complexity evidence is preserved separately; timing is hardware/protocol-specific,
and twice kernel MACs is not a complete count of all floating-point operations.
