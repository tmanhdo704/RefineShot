# Experimental results

This is the curated portfolio snapshot. Scores from different protocols should
only be compared within the same experiment group.

## Main thesis result

The main comparison follows the original AutoShot protocol: choose the
best-performing decision threshold independently for each dataset.

| Dataset | Best-threshold F1 |
|---|---:|
| SHOT | **0.8607** |
| BBC | 0.9656 |
| ClipShots | 0.7706 |

For SHOT, the selected threshold is `0.12`, with precision `0.8408`, recall
`0.8816`, and F1 `0.8607`.

## Fixed-deployment reference

The following values use one fixed deployment operating point. They are kept
for reproducibility but are not the headline best-threshold result.

| Dataset | Threshold | F1 | Precision | Recall | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| SHOT | 0.10 | 0.8545 | 0.8554 | 0.8537 | 2148 | 363 | 368 |
| BBC | 0.10 | 0.9656 | 0.9750 | 0.9564 | 4633 | 119 | 211 |
| ClipShots | 0.10 | 0.7529 | 0.6661 | 0.8657 | 6241 | 3129 | 968 |

## Selected comparisons

| Experiment | SHOT F1 | BBC F1 | ClipShots F1 |
|---|---:|---:|---:|
| Original AutoShot baseline | 0.8405 | 0.9554 | 0.7649 |
| Phase-2 BCE control | 0.8378 | 0.9570 | 0.6983 |
| Temperature + Gaussian | 0.8540 | 0.9570 | 0.7441 |
| Full Phase-2 candidate | 0.8542 | 0.9551 | 0.7409 |

The fixed-deployment table comes from the committed experiment snapshot.
Datasets and cached logits are external; see [`DATA.md`](DATA.md).
