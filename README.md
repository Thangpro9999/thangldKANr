# thangldKANr

Kolmogorov-Arnold Networks (KAN) for regression, graphs and time series in R on top of R 'torch'
(no Python needed), with **one engine and interchangeable edge bases**:

| `basis` | edge function | reference implementation compared against |
|---|---|---|
| `"chebyshev"` | Chebyshev expansion of tanh(x) | SynodicMonth/ChebyKAN `5f7efdd` |
| `"rbf"` | Gaussian RBF + SiLU base (FastKAN) | ZiyaoLi/fast-kan `17b6540` |
| `"legendre"` | Legendre expansion of tanh(x) | OrthogPolyKANs `3b2760a` |
| `"wavelet"` | Mexican-hat wavelet (Wav-KAN) | zavareh1/Wav-KAN `f7ab521` |
| `"rational"` | safe Pade rational function | ml-research/rational_activations `2496d91` |
| `"hybrid"` | SiLU + Chebyshev + RBF | ChebyKAN + fast-kan components |
| `"jacobi"` | Jacobi expansion of tanh(x) | SpaceLearner/JacobiKAN `51d160c` |
| `"relu"` | squared ReLU bumps (ReLU-KAN) | quiqi/relu_kan `79eab7a` |

Each layer may use a different basis, and other packages can add bases
with `register_kan_basis()`. A base R oracle with analytic
gradients is also a training backend without libtorch.

```r
library(thangldKANr)
kan_bases()
m <- thangld_kan(c(2, 4, 1), basis = "chebyshev", degree = 4, seed = 1)
fit <- thangld_kan_fit(m, x, y, validation_split = 0.2, epochs = 300,
                     batch_size = 32, lr = 0.02, patience = 50, seed = 1)
thangld_kan(c(2, 4, 1), basis = c("rbf", "jacobi"), degree = 3, seed = 1)
thangld_kan_validate("both")

# graphs: node regression with neighbourhood aggregation
g <- thangld_graph_kan_simulate(n = 150, seed = 1)
gm <- thangld_graph_kan(c(2, 4, 1), basis = "chebyshev", self_branch = TRUE, seed = 1)
# time series: one-step forecasts from lag windows
tm <- thangld_temporal_kan(lags = 1:12, hidden = 4, basis = "legendre", seed = 1)
```

Graph KANs and lag-window KANs use the same layers, so every basis works
there too (`thangld_graph_kan_validate()`, `thangld_temporal_kan_validate()`).

Status: 0.1.0; regression, one output, numeric inputs, CPU.

Validation: `thangld_kan_validate()`, `tests/testthat`, `inst/validation/<basis>`
(Python comparisons at the pinned commits above).

License: GPL-3. Citation: `citation("thangldKANr")`; third-party references: `inst/NOTICE`.
