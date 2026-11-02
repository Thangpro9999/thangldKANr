# thangldKANr 0.1.0

* First release.
* `thangld_kan(widths, basis = ...)`: Kolmogorov-Arnold networks with
  Chebyshev, RBF (FastKAN), Legendre, wavelet, rational, hybrid, Jacobi and
  ReLU-KAN edges; per-layer bases; basis-specific arguments in `...` or
  `options`.
* `thangld_graph_kan()`: graph KANs for transductive node regression
  (aggregate with the normalized adjacency, optional self branch), with
  fit, predict, save/load and `thangld_graph_kan_validate()`.
* `thangld_temporal_kan()`: lag-window KANs for one-step forecasts, with
  fit, predict, rolling-origin evaluation and
  `thangld_temporal_kan_validate()`.
* Both accept every edge basis.
* `register_kan_basis()`, `unregister_kan_basis()`, `kan_bases()`,
  `kan_layer_bases()`: add edge bases from other packages.
* `thangld_kan_validate(backend, basis, mixed)`: numerical checks of every
  basis and of a network mixing bases.
* `inst/validation/<name>`: comparisons with the Python reference of each
  basis, of the graph KAN and of the lag-window KAN at pinned commits.
