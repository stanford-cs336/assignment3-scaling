# Learning Notes For `chinchilla_isoflops`

This file summarizes the main concepts we clarified while solving the IsoFLOPs scaling-law problem.

## 1. FLOPs vs FLOP/s

In this assignment, `FLOPs` means **total floating-point operations spent during training**.

It does **not** mean floating-point operations per second.

The distinction is:

```text
FLOPs  = total operations
FLOP/s = operations per second, hardware speed
```

So when the assignment says:

```text
C = 1e23 FLOPs
```

it means the training run is allowed to spend about `1e23` total floating-point operations.

In real hardware terms:

```text
training time ≈ total FLOPs / achieved FLOP/s
```

But the scaling-law math abstracts away hardware utilization and focuses on total theoretical training compute.

## 2. What Is A Compute Budget?

A compute budget `C` is the amount of training compute we are allowed to spend.

Even though we could always keep training longer, in practice we have limits:

- GPU time
- money
- wall-clock time
- API limits
- assignment limits

So the question becomes:

> Given a fixed compute budget `C`, how should we spend it?

Using the approximation:

```text
C = 6 * N * D
```

we can spend compute in two main ways:

- Bigger model: larger `N`, smaller `D`
- More training data: smaller `N`, larger `D`

The IsoFLOPs method asks:

> For a fixed `C`, which choice of `N` and `D` gives the lowest loss?

## 3. Hardware Utilization Caveat

The assignment treats `C` as theoretical useful training compute.

In real life, two runs with the same theoretical `C` might take different wall-clock times if one uses the GPU less efficiently.

For example:

```text
same theoretical FLOPs, high utilization -> faster
same theoretical FLOPs, low utilization  -> slower
```

But for this problem, we ignore that and compare runs by total FLOPs.

So inside this assignment:

```text
same C = same idealized training budget
```

## 4. Why `C = 6 * N * D`?

The formula:

```text
C = 6 * N * D
```

is a rule-of-thumb estimate for dense Transformer training compute.

Where:

- `C` = total training compute in FLOPs
- `N` = number of model parameters
- `D` = number of training tokens

The intuition is:

```text
cost per token ≈ proportional to N
number of tokens = D
total cost ≈ N * D
```

The factor `6` comes from the rough cost of training:

- forward pass
- backward pass
- gradient-related computation

So the approximation says:

```text
training FLOPs per token ≈ 6 * N
```

and therefore:

```text
total training FLOPs ≈ D * 6N = 6ND
```

This is an approximation. It does not perfectly account for all attention costs, sequence length details, or hardware effects.

## 5. What Is `D`, The Number Of Tokens?

`D` is the total number of training token positions processed.

It is not the number of sentences.

During language-model training, text is tokenized and split into fixed-length blocks, such as:

```text
sequence length = 2048 tokens
```

If:

```text
batch size = 32
sequence length = 2048
training steps = 1000
```

then:

```text
D = 32 * 2048 * 1000
```

The model predicts the next token at each position.

For a block:

```text
tokens:  x1, x2, x3, ..., x2048
```

the targets are shifted:

```text
input positions: x1, x2, x3, ..., x2047
targets:         x2, x3, x4, ..., x2048
```

Conceptually:

```text
see x1                 -> predict x2
see x1, x2             -> predict x3
see x1, x2, x3         -> predict x4
...
see x1 ... x2047       -> predict x2048
```

This happens in parallel during training.

## 6. Does Predicting Later Tokens Cost More?

Yes, in a detailed Transformer compute accounting, later positions can involve more attention work because they can attend to more previous tokens.

But the formula:

```text
C = 6ND
```

is a simplified average-cost approximation.

It treats each token as costing about:

```text
6N FLOPs
```

This is good enough for the scaling-law exercise, but not an exact per-token compute formula.

## 7. What The IsoFLOPs Procedure Does

For each compute budget `C_i`, the data contains several runs with different model sizes `N`.

For each fixed `C_i`, we choose the run with the lowest final loss.

That gives:

```text
N_opt(C_i)
```

Then we compute:

```text
D_opt(C_i) = C_i / (6 * N_opt(C_i))
```

So each compute budget gives us two points:

```text
<C_i, N_opt(C_i)>
<C_i, D_opt(C_i)>
```

Then we fit curves through these points and extrapolate to larger compute budgets.

## 8. Why Chinchilla Matters

Once we have points like:

```text
(C_1, N_opt(C_1))
(C_2, N_opt(C_2))
...
```

we need to choose what kind of curve to fit.

Many curves could fit the observed points:

- linear
- quadratic
- polynomial
- exponential
- power law

But they might extrapolate very differently.

Chinchilla gives us a scientific prior:

> The compute-optimal model size and dataset size approximately follow power laws.

So instead of fitting an arbitrary curve, we fit:

```text
N_opt(C) = A * C^a
D_opt(C) = B * C^b
```

Chinchilla gives us the curve shape. The assignment data gives us the actual constants and exponents.

## 9. What Is A Power Law?

A power law has the form:

```text
y = A * x^b
```

In our problem:

```text
N_opt(C) = A * C^a
D_opt(C) = B * C^b
```

The exponent controls how fast the quantity grows.

For example, if:

```text
N_opt(C) ∝ C^0.47
```

then increasing compute by `10x` increases optimal model size by:

```text
10^0.47 ≈ 3x
```

So model size grows with compute, but slower than compute itself.

## 10. Why Take Logs?

The power-law equation:

```text
N = A * C^a
```

is curved in normal coordinates.

Taking logs turns it into a straight line:

```text
log(N) = log(A * C^a)
```

Using log rules:

```text
log(N) = log(A) + a * log(C)
```

Now define:

```text
Y = log(N)
X = log(C)
B = log(A)
```

Then:

```text
Y = B + aX
```

That is a linear equation.

So fitting a line in log-log space is equivalent to fitting a power law in normal space.

## 11. Why Fit In Log Space Instead Of Fitting `N = A * C^a` Directly?

We can fit the power law directly, but fitting in log space is usually easier and more stable.

Direct fitting:

```text
N = A * C^a
```

is nonlinear because `a` appears in the exponent.

Log-space fitting:

```text
log(N) = log(A) + a * log(C)
```

is ordinary linear regression.

It also makes sense statistically for scaling laws because the values span many orders of magnitude. Log fitting cares more about relative error than absolute error.

Example:

```text
actual N = 1e9, prediction = 2e9
```

This is a `2x` error.

```text
actual N = 1e12, prediction = 1.001e12
```

This has the same absolute error, `1e9`, but only a tiny relative error.

Log-space fitting treats the first error as much more serious, which is usually what we want for scaling laws.

## 12. The Key Insight

The whole method is:

1. Use experiments to find the best `N` for each compute budget.
2. Compute the corresponding `D`.
3. Assume the optimal `N` and `D` follow power laws.
4. Fit those power laws in log-log space.
5. Extrapolate to larger compute budgets.

In short:

```text
Chinchilla gives us the form of the curve.
The IsoFLOPs data gives us the fitted numbers.
The fitted curve gives us predictions for larger C.
```

