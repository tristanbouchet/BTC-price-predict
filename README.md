Python code that fits historical prices of BTC with a model to make prediction of future prices.

The price can be predicted over 3 orders of magnitude, with only 7 free parameters.

# Requirements

Required libraries: numpy, pandas, matplotlib, lmfit, sklearn

Optional libraries: yfinance (v1.0), curl_cffi (for API updated data), PyQt5 (for interactive plots)

# Model explanation

The global trend, but mainly the support, follow a power-law (PL), as was noted by many others. The reference date is naturally taken as the genesis block, $t_0$:
```math
S = K * (t/t_0)^a
```
The support is found with a quantile regression, fitted in log-log space.

To fit the cycles, or "bull-runs", more work is required. We assume that they are induced by the BTC halving, which happen approximately every 4 years.

First, the price during cycles is normalized by the PL support:
```math
r = P/S - 1
```
The shape of this ratio is erratic, and chosen empirically as a periodic [Lorentzian](https://en.wikipedia.org/wiki/Cauchy_distribution) function. The change in BTC creation decrease [geometricaly](https://en.wikipedia.org/wiki/Geometric_progression) every halving, so the amplitude is assumed to decrease geometrically as well.
```math
r(t) = A_k *  Lorentzian(x = t; x_0 = t_k + d, \gamma = \sigma_c) + C
```

where $A_k = A_0 * \mu^k$, with $A_0$: initial cycle amplitude, $\mu$: the decay factor, $k$: halving number.

$t_k$: the halving starting dates, $d$: the delay between halving start and cycle peak, $\sigma_c$: width of the cycle, i.e. the average duration of bull-runs.

An offset $C$ is also added to adjust the support.

