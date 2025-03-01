Python notebook that fits previous prices of BTC with a model to make prediction of future price.

The global trend is a powerlaw (PL), with an index either fitted or fixed at 6 (assuming a node growth index of 3 and Metcalfe's law).

The cycles variations (c) are defined as: price = PL * (1 + c). They are fitted with a [Lorentzian](https://en.wikipedia.org/wiki/Cauchy_distribution) that decays [geometricaly](https://en.wikipedia.org/wiki/Geometric_progression) every halving:

c(t) = A_k * Lorentzian(x = t; x_0 = d, gamma = sigma_c)

where A_k = A_0 * mu^k, with A_0: initial cycle amplitude, mu: decay factor, k: halving number.

and d: the delay between halving start and cycle peak, sigma_c: width of the cycle ~ duration of bull-run

The fit can be bounded until 2024 or earlier so that the model can be tested against reality for later prices.
