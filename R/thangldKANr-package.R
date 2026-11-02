#' thangldKANr: Kolmogorov-Arnold Networks with Interchangeable Edge Bases
#' Using 'torch'
#'
#' One network engine, many edge types. [thangld_kan()] builds a network whose
#' edge functions use one of the built-in bases (Chebyshev, Gaussian RBF as
#' in FastKAN, Legendre, Mexican-hat wavelet, safe Pade rational, hybrid
#' SiLU + Chebyshev + RBF, Jacobi, squared-ReLU bumps), possibly a different
#' basis in each layer. Other packages add bases with
#' [register_kan_basis()]. The same layers are used by
#' [thangld_graph_kan()] (node regression on a graph) and
#' [thangld_temporal_kan()] (one-step forecasts from lag windows).
#'
#' Main entry points: [thangld_kan()], [kan_bases()], [thangld_kan_fit()],
#' [predict.thangld_kan()], [thangld_kan_metrics()], [thangld_kan_edge()],
#' [plot.thangld_kan()], [thangld_kan_save()], [thangld_kan_load()],
#' [thangld_kan_forward_r()], [thangld_kan_gradient_r()],
#' [thangld_kan_validate()], [register_kan_basis()]; graphs:
#' [thangld_graph_kan()], [thangld_graph_kan_fit()],
#' [thangld_graph_kan_adjacency()], [thangld_graph_kan_validate()]; time
#' series: [thangld_temporal_kan()], [thangld_temporal_kan_fit()],
#' [thangld_temporal_kan_rolling()], [thangld_temporal_kan_validate()].
#'
#' @references
#' Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljacic, M.,
#' Hou, T. Y. and Tegmark, M. (2024). KAN: Kolmogorov-Arnold Networks.
#' \doi{10.48550/arXiv.2404.19756}
#' @keywords internal
"_PACKAGE"
