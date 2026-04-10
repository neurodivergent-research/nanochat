# Experiment Config Rationale

Current config: `fictional_knowledge_config.yaml`

```yaml
seed: 10
warmup_steps: 1000
step_between_injections: 100
injection_mode: all
step_2_start_saving: 13910
steps_between_checkpoints: 219
```

---

## Model and training scale

The depth-20 model has ~561M parameters and trains for **21,400 optimizer steps** on **~11.2B tokens** (Chinchilla-optimal 20:1 token-to-param ratio). Each optimizer step processes 524,288 tokens across `device_batch_size=32` sequences with `grad_accum_steps=8`.

---

## Injection mode: `all`

The fictional knowledge dataset has 130 entries total:
- **40 entries** have 9 pre-generated paraphrases → 10 variants each (original + 9)
- **90 entries** have no paraphrases → 1 variant each (original only)
- **490 total unique texts** across all entries

`all` mode randomly samples one variant per entry at each injection step using a deterministic per-step seed, so all GPUs agree on which variant to use. Entries without paraphrases always receive their original text.

This is the most diverse injection scenario: the model sees the same fact expressed in different surface forms across training, which the paper (Chang et al., 2024) shows leads to slower forgetting and better semantic generalization compared to duplication.

---

## Why `step_between_injections: 100`

### The imbalance problem

With `all` mode, the 130 entries are not equivalent:
- **90 no-para entries**: only 1 variant → every injection is the same text
- **40 para entries**: 10 variants → each variant appears ~1/10th as often

At `step_between_injections=200` (102 injections), para variants would only be seen ~10 times each — too few for reliable memorization according to the paper's acquisition dynamics. A minimum of ~20 exposures per variant is a safer target.

Setting `step_between_injections=100` gives **204 total injections**, which yields:

| Entry type | Exposures per unique text |
|---|---|
| 90 no-para entries | 204× (original only) |
| 40 para entries | ~20× per variant (×10 variants) |

This brings para variant exposure to ~20×, closer to the no-para baseline while keeping fictional tokens at **0.484%** of total training tokens — still a negligible fraction of the gradient signal.

### Alignment with the paper

Chang et al. inject every 100 training steps in their experiments. Our `step_between_injections` is in optimizer step units, so this is a direct match to the paper's injection cadence. With 22 full paraphrase cycles across training, the model accumulates micro-acquisitions of each fact in progressively different surface forms.

The paper's duplication scenario uses **10 injections per fact** as the baseline for reliable acquisition. With 9 paraphrases per entry, 10 full cycles through all paraphrases maps directly to that baseline — each paraphrase is seen ~10 times, matching the number of repetitions the paper found sufficient for memorization. For `all` mode specifically, 102 random draws over 10 variants gives each variant ~10 appearances per entry on average, preserving the same total exposure budget as the paper's duplication design while distributing it across surface forms. At `step_between_injections=100` this doubles to ~20 appearances per variant, providing headroom to compensate for the imbalance between no-para entries (204 exposures) and para entries (~20 per variant).

### Total injection budget

```
injections = range(1000, 21400, 100) = 204 steps
fictional tokens = 204 × 130 sequences × 2048 tokens = 54.3M
% of total tokens = 54.3M / 11.2B = 0.484%
sequences per opt step: 130 fictional out of 256 total (at injection steps only)
```

---

## Checkpoint config: `step_2_start_saving: 18947`, `steps_between_checkpoints: 219`

### Why 219 steps between checkpoints

The merging experiments found that **~3% of total tokens** is the optimal spacing between checkpoints used in a merge (*"el minimo de espacio optimo entre los pasos es de aproximadamente 3% de total de los tokenes"*). For our scale:

```
3% × 11.2B = 336M tokens → 642 optimizer steps
```

Rather than saving exactly at 3% intervals, we save **every 115M tokens (219 steps)** — roughly 1/3 of the optimal 3% window. This gives finer-grained control when selecting which checkpoints to merge without being locked into a coarser grid.

### Why start at step 18947 (12 checkpoints)

The merging experiments also established that N=6 is the practical upper limit where adding more models stops improving performance meaningfully (*"por lo observado mergear mas de 6 modelos no mejora el performance considerablemente"*). The primary merge target is therefore 6 checkpoints, but saving **12** keeps the option open to experiment with larger N without re-running training.

Starting at step 18,947 places exactly 12 interval checkpoints between that step and the end of training, all spaced 219 steps (~115M tokens) apart:

```
checkpoints: 18947, 19166, 19385, 19604, 19823, 20042,
             20261, 20480, 20699, 20918, 21137, 21356  (+21400 final)
total saved: 12 interval + 1 final (always saved by base_train.py) = 13
coverage:    steps 18947–21400 = 88.5%–100% of training
spacing:     ~115M tokens (~1/3 of optimal 3% window)
storage:     13 × ~7.85GB ≈ 102GB (weights + Adam states)
```

This satisfies the 65% floor constraint (*"No queremos que el modelo mas viejo tenga menos de 65% de tokens de entrenamiento"*) — all 12 checkpoints are well above that threshold — while keeping storage costs manageable at ~102GB versus the ~275GB a full 35-checkpoint sweep would require.

### The 65% floor and the learnability threshold

The paper establishes a *learnability threshold*: facts presented with repetition intervals longer than this threshold are never reliably learned. Early checkpoints, being undertrained, have a higher learnability threshold (more parameters are still noisy), making merges from them unstable. Staying above 65% of training ensures all checkpoint pairs are within the stable acquisition regime.

---

## Summary

| Parameter | Value | Reasoning |
|---|---|---|
| `injection_mode` | `all` | Maximum surface-form diversity; slower forgetting than duplication |
| `step_between_injections` | 100 | Gives ~20× exposure per para variant; matches paper's 100-step cadence |
| `warmup_steps` | 1000 | Avoids injecting into an unstable early-training loss landscape |
| `step_2_start_saving` | 18947 | Saves exactly 12 checkpoints; primary merge target is N=6, 12 gives room to go higher |
| `steps_between_checkpoints` | 219 | ~115M tokens; 3× denser than optimal 3% spacing for flexibility |
