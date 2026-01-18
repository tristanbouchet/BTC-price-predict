Python notebook that fits previous prices of BTC with a model to make prediction of future price.

The price is predicted over 3 orders of magnitude, with only 7 free parameters.

# Requirements

Required libraries: numpy, pandas, matplotlib, lmfit, sklearn

Optional libraries: yfinance (v1.0), curl_cffi (for API updated data), PyQt5 (for interactive plots)

# Model explanation

The global trend, but mainly the support, follow a power-law (PL), as was noted by many others. The reference date is naturally taken as the genesis block, t0:

S = K * (t/t0)^a

To fit the cycles, or "bull-runs", more work is required. We assume that they are induced by the BTC halving, happening every 4 years approximately.

The price during cycles is normalized by the PL support (S): r = P/S - 1 

The shape of this ratio is erratic, and chosen empirically as a [Lorentzian](https://en.wikipedia.org/wiki/Cauchy_distribution) function. The change in BTC creation decrease [geometricaly](https://en.wikipedia.org/wiki/Geometric_progression) every halving, so the amplitude is assumed to decrease geometrically as well.

r(t) = A_k * Lorentzian(x = t; x_0 = d, gamma = sigma_c)

where A_k = A_0 * mu^k, with A_0: initial cycle amplitude, mu: the decay factor, k: halving number.

d: the delay between halving start and cycle peak, sigma_c: width of the cycle, i.e. the duration of bull-run

