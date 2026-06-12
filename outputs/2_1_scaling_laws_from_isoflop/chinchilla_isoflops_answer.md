# Problem: `chinchilla_isoflops`

## (a) Compute-Optimal Model Size

**Plot:** `chinchilla_isoflops_model_size.svg`

The IsoFLOPs optima used to fit the model-size scaling law were:

| Compute budget C (FLOPs) | N_opt(C) |
|---:|---:|
| 6.000e18 | 762,093,419 |
| 1.000e19 | 806,647,749 |
| 3.000e19 | 1,536,852,354 |
| 6.000e19 | 1,952,041,776 |
| 1.000e20 | 3,253,402,960 |
| 3.000e20 | 5,903,836,027 |
| 6.000e20 | 6,971,055,968 |
| 1.000e21 | 6,859,328,563 |
| 3.000e21 | 12,148,905,329 |

Fitting a power law in log-log space gives:

```text
N_opt(C) = 1.163 * C^0.468683
```

**One-sentence response:** The fitted IsoFLOPs scaling law predicts an optimal model size of approximately **7.005e10 parameters** for a budget of **1e23 FLOPs** and approximately **2.061e11 parameters** for a budget of **1e24 FLOPs**.

## (b) Compute-Optimal Dataset Size

**Plot:** `chinchilla_isoflops_dataset_size.svg`

For each IsoFLOPs optimum above, I computed:

```text
D_opt(C) = C / (6 * N_opt(C))
```

The dataset-size points used to fit the scaling law were:

| Compute budget C (FLOPs) | D_opt(C) tokens |
|---:|---:|
| 6.000e18 | 1,312,175,089 |
| 1.000e19 | 2,066,164,157 |
| 3.000e19 | 3,253,402,962 |
| 6.000e19 | 5,122,841,182 |
| 1.000e20 | 5,122,841,182 |
| 3.000e20 | 8,469,069,902 |
| 6.000e20 | 14,345,028,997 |
| 1.000e21 | 24,297,810,658 |
| 3.000e21 | 41,155,971,378 |

Fitting a power law in log-log space gives:

```text
D_opt(C) = 0.1433 * C^0.531317
```

**One-sentence response:** The fitted IsoFLOPs scaling law predicts an optimal dataset size of approximately **2.379e11 tokens** for a budget of **1e23 FLOPs** and approximately **8.086e11 tokens** for a budget of **1e24 FLOPs**.

