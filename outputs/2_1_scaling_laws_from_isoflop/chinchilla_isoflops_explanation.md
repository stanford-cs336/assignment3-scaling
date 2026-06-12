# Understanding the `chinchilla_isoflops` Solution

This note explains what we did for the IsoFLOPs scaling-law problem, step by step.

## Goal

The assignment asks us to reproduce the IsoFLOPs method from the Chinchilla paper.

The main question is:

> Given a compute budget `C`, what model size `N` and dataset size `D` should we use?

The compute approximation from the assignment is:

```text
C = 6 * N * D
```

where:

- `C` is the training compute budget in FLOPs.
- `N` is the number of model parameters.
- `D` is the number of training tokens.

So once we know `C` and `N`, we can solve for `D`:

```text
D = C / (6 * N)
```

## The Data

The file `data/isoflops_curves.json` contains many synthetic training runs.

Each run has:

```json
{
  "parameters": 49999999,
  "compute_budget": 6e18,
  "final_loss": 7.192784500319437
}
```

This means:

- A model with about `50M` parameters was trained.
- The run used `6e18` FLOPs.
- The final training loss was about `7.19`.

For each compute budget, there are multiple runs with different model sizes.

## Step 1: Group Runs By Compute Budget

The IsoFLOPs method compares runs that used the same amount of compute.

For example, suppose we have several runs with:

```text
C = 6e18 FLOPs
```

Each run uses a different model size `N`. Since `C` is fixed, changing `N` also changes the dataset size `D`.

For each fixed compute budget, we ask:

> Which model size produced the lowest final loss?

That model size becomes `N_opt(C)`, the optimal model size for that compute budget.

## Step 2: Pick The Lowest-Loss Run For Each Budget

The assignment says we do not need to fit a quadratic curve inside each IsoFLOPs profile. Instead, we can simply take the run with the lowest `final_loss`.

So for every compute budget `C_i`, we pick:

```text
N_opt(C_i) = parameters from the run with lowest final_loss
```

Then we compute:

```text
D_opt(C_i) = C_i / (6 * N_opt(C_i))
```

This gives us pairs like:

```text
<C_i, N_opt(C_i)>
<C_i, D_opt(C_i)>
```

These are the data points used to fit the scaling laws.

## Step 3: Fit Power Laws

The Chinchilla-style IsoFLOPs method assumes the optimal model size and dataset size follow power laws:

```text
N_opt(C) = a * C^alpha
D_opt(C) = b * C^beta
```

Power laws are easier to fit if we take logs.

For model size:

```text
N_opt(C) = a * C^alpha
```

Taking the natural log of both sides:

```text
log(N_opt) = log(a) + alpha * log(C)
```

This is now a straight-line fit:

```text
y = intercept + slope * x
```

where:

```text
x = log(C)
y = log(N_opt)
slope = alpha
intercept = log(a)
```

We do the same thing for dataset size:

```text
log(D_opt) = log(b) + beta * log(C)
```

## Step 4: Our Fitted Scaling Laws

Using the lowest-loss IsoFLOPs points, the script fit:

```text
N_opt(C) = 1.163 * C^0.468683
D_opt(C) = 0.1433 * C^0.531317
```

The exponents are important:

- `N_opt` scales approximately as `C^0.469`.
- `D_opt` scales approximately as `C^0.531`.

These add to about `1.0`, which makes sense because:

```text
C = 6 * N * D
```

If `N` grows like `C^0.469` and `D` grows like `C^0.531`, then:

```text
N * D grows like C^(0.469 + 0.531) = C^1
```

That matches the compute formula.

## Step 5: Predictions

The assignment asks for predictions at:

```text
C = 1e23 FLOPs
C = 1e24 FLOPs
```

Using the fitted model-size law:

```text
N_opt(C) = 1.163 * C^0.468683
```

we get:

```text
C = 1e23 FLOPs -> N_opt ≈ 70.1B parameters
C = 1e24 FLOPs -> N_opt ≈ 206.1B parameters
```

Using the fitted dataset-size law:

```text
D_opt(C) = 0.1433 * C^0.531317
```

we get:

```text
C = 1e23 FLOPs -> D_opt ≈ 237.9B tokens
C = 1e24 FLOPs -> D_opt ≈ 808.6B tokens
```

## Data Points Used For The Fit

These are the IsoFLOPs optima found by the script:

| Compute budget `C` | `N_opt` | `D_opt = C / (6N_opt)` | Final loss |
|---:|---:|---:|---:|
| `6.000e18` | 762,093,419 | 1,312,175,089 | 5.899930 |
| `1.000e19` | 806,647,749 | 2,066,164,157 | 5.617943 |
| `3.000e19` | 1,536,852,354 | 3,253,402,962 | 5.107177 |
| `6.000e19` | 1,952,041,776 | 5,122,841,182 | 4.830586 |
| `1.000e20` | 3,253,402,960 | 5,122,841,182 | 4.652893 |
| `3.000e20` | 5,903,836,027 | 8,469,069,902 | 4.311219 |
| `6.000e20` | 6,971,055,968 | 14,345,028,997 | 4.121241 |
| `1.000e21` | 6,859,328,563 | 24,297,810,658 | 4.002835 |
| `3.000e21` | 12,148,905,329 | 41,155,971,378 | 3.773188 |

## What The Plots Show

There are two plots:

```text
outputs/chinchilla_isoflops_model_size.svg
outputs/chinchilla_isoflops_dataset_size.svg
```

The orange dots are the optimal points we found from the data.

The teal line is the fitted power law.

The line extends beyond the observed data up to at least `1e24` FLOPs, because the assignment asks us to extrapolate to larger compute budgets.

## Final One-Sentence Answers

For model size:

> The fitted IsoFLOPs scaling law predicts about `7.005e10` parameters at `1e23` FLOPs and `2.061e11` parameters at `1e24` FLOPs.

For dataset size:

> The fitted IsoFLOPs scaling law predicts about `2.379e11` tokens at `1e23` FLOPs and `8.086e11` tokens at `1e24` FLOPs.

## Big Picture

The whole procedure is:

1. For each compute budget, find the best run.
2. Treat that run's parameter count as `N_opt`.
3. Compute `D_opt` from the compute formula.
4. Fit power laws in log-log space.
5. Use those fitted laws to extrapolate to larger compute budgets.

This is exactly the IsoFLOPs scaling-law workflow the assignment asks us to reproduce.
