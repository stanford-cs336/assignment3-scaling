# chinchilla_isoflops Results

IsoFLOPs optima used for fitting:

| C (FLOPs) | N_opt | D_opt = C / (6 N_opt) | final loss |
|---:|---:|---:|---:|
| 6.000e18 | 762,093,419 | 1,312,175,089 | 5.899930 |
| 1.000e19 | 806,647,749 | 2,066,164,157 | 5.617943 |
| 3.000e19 | 1,536,852,354 | 3,253,402,962 | 5.107177 |
| 6.000e19 | 1,952,041,776 | 5,122,841,182 | 4.830586 |
| 1.000e20 | 3,253,402,960 | 5,122,841,182 | 4.652893 |
| 3.000e20 | 5,903,836,027 | 8,469,069,902 | 4.311219 |
| 6.000e20 | 6,971,055,968 | 14,345,028,997 | 4.121241 |
| 1.000e21 | 6,859,328,563 | 24,297,810,658 | 4.002835 |
| 3.000e21 | 12,148,905,329 | 41,155,971,378 | 3.773188 |

Model-size fit: N_opt(C) = 1.163e0 * C^0.468683.
Dataset-size fit: D_opt(C) = 1.433e-1 * C^0.531317.

For C = 1e23 FLOPs: N_opt = 70,054,233,905, D_opt = 237,910,911,842.
For C = 1e24 FLOPs: N_opt = 206,118,539,185, D_opt = 808,596,195,789.

One-sentence responses:
For model size, the fitted IsoFLOPs scaling law predicts about 7.005e10 parameters at 1e23 FLOPs and 2.061e11 parameters at 1e24 FLOPs.
For dataset size, the fitted IsoFLOPs scaling law predicts about 2.379e11 tokens at 1e23 FLOPs and 8.086e11 tokens at 1e24 FLOPs.
