#' Create a Kolmogorov-Arnold network with a chosen edge basis
#'
#' Each layer computes \eqn{h^{(l+1)}_j = \sum_i \phi_{j,i}(h^{(l)}_i)}, where
#' the edge functions \eqn{\phi_{j,i}} are expansions in the chosen basis.
#' `basis` may differ between layers. The built-in bases follow published KAN
#' variants (see [kan_bases()]); other
#' bases are added with [register_kan_basis()].
#'
#' Basis-specific arguments are passed by name in `...`. A per-layer
#' argument (marked "per layer") may be one value or one value per layer of
#' the network; in a mixed network the values for layers of another basis
#' are ignored. An argument that no selected basis uses is an error.
#' `options` gives basis-specific values that override `...`, for example
#' `options = list(jacobi = list(degree = 3))`.
#'
#' \describe{
#'   \item{`"chebyshev"`}{\eqn{\sum_k c_{ijk} T_k(\tanh x)}. `degree` (per
#'     layer, default 4), `init_sd` (`NULL`: \eqn{1/(\mathrm{in}(K+1))}).
#'     Same layout and initialization as ChebyKAN.}
#'   \item{`"rbf"`}{FastKAN: optional layer normalization, Gaussian radial
#'     basis functions on a fixed grid and a SiLU base branch. `num_grids`
#'     (8), `grid_min` (-2), `grid_max` (2), `denominator` (`NULL`: grid
#'     spacing), `layernorm` (`TRUE`), `base_update` (`TRUE`), `ln_eps`
#'     (1e-5), `spline_init_sd` (0.1).}
#'   \item{`"legendre"`}{\eqn{\sum_k c_{ijk} P_k(\tanh x)}. `degree` (per
#'     layer, 4), `init_sd` (`NULL`).}
#'   \item{`"wavelet"`}{Wav-KAN with the Mexican-hat wavelet, trainable
#'     translation and scale per edge. `scale_param` (`"log"` or `"raw"`),
#'     `bias` (`TRUE`).}
#'   \item{`"rational"`}{Safe Pade form \eqn{P(x)/(1+|Q(x)|)} (no poles).
#'     `num_degree` (per layer, 3), `den_degree` (per layer, 2, not
#'     above `num_degree`), `init_sd` (`NULL`), `den_init_sd` (0.1).}
#'   \item{`"hybrid"`}{Sum of SiLU, Chebyshev and RBF parts. `bases` (subset
#'     of `"silu"`, `"chebyshev"`, `"rbf"`), `degree` (per layer, 3),
#'     `num_grids` (per layer, 8), `grid_min` (-2), `grid_max` (2),
#'     `denominator` (`NULL`), `rbf_init_sd` (0.1). See
#'     [hybrid_kan_components()].}
#'   \item{`"jacobi"`}{\eqn{\sum_k c_{ijk} P_k^{(\alpha,\beta)}(\tanh x)}.
#'     `degree` (per layer, 4), `alpha` (1), `beta` (1), `init_sd` (`NULL`).}
#'   \item{`"relu"`}{ReLU-KAN squared ReLU bumps with trainable phases.
#'     `grid_size` (per layer, 5), `k` (per layer, 3), `grid_range`
#'     (`c(-2, 2)`).}
#' }
#'
#' @param widths Integer layer widths; the last must be 1.
#' @param basis Basis name, one value or one per layer (see [kan_bases()]).
#' @param ... Named basis-specific arguments (see Details).
#' @param options `NULL` or a named list (names are bases) of named lists of
#'   basis-specific arguments.
#' @param seed Optional seed for the initialization.
#' @return An object of class `thangld_kan`.
#' @examples
#' thangld_kan(c(2, 3, 1), basis = "chebyshev", degree = 4, seed = 1)
#' thangld_kan(c(2, 3, 1), basis = "relu", grid_size = 4, seed = 1)
#' # one basis per layer
#' thangld_kan(c(2, 3, 1), basis = c("rbf", "legendre"), degree = 3, seed = 1)
#' @export
thangld_kan <- function(widths, basis = "chebyshev", ..., options = NULL,
                      seed = NULL) {
  widths <- check_widths(widths)
  if (!is.null(seed)) seed <- check_count(seed, "seed", min = 0L)
  specs <- build_layer_specs(widths, basis, list(...), options)
  new_model(widths, specs, seed)
}

#' Layer specifications from basis names and basis-specific arguments
#'
#' `widths[l]` is the number of inputs of layer `l` as seen by the basis.
#' @noRd
build_layer_specs <- function(widths, basis, dots, options) {
  nl <- length(widths) - 1L
  if (!is.character(basis) || !(length(basis) %in% c(1L, nl)) ||
      anyNA(basis)) {
    stop(sprintf("`basis` must be one basis name or %d names.", nl),
         call. = FALSE)
  }
  basis <- rep_len(basis, nl)
  ub <- unique(basis)
  impls <- stats::setNames(lapply(ub, basis_impl), ub)
  if (length(dots) && (is.null(names(dots)) || any(names(dots) == ""))) {
    stop("Basis-specific arguments in `...` must be named.", call. = FALSE)
  }
  if (!is.null(options)) {
    if (!is.list(options) || is.null(names(options)) ||
        !all(names(options) %in% ub)) {
      stop("`options` must be a list named by bases used in `basis`.",
           call. = FALSE)
    }
  }
  specs <- vector("list", nl)
  used <- character()
  for (b in ub) {
    fm <- setdiff(names(formals(impls[[b]]$layer_specs)), "widths")
    a <- dots[names(dots) %in% fm]
    used <- c(used, names(a))
    o <- options[[b]]
    if (length(o)) {
      if (!is.list(o) || is.null(names(o)) || !all(names(o) %in% fm)) {
        stop(sprintf("`options$%s` must be a named list of arguments of ",
                     b), sprintf("basis \"%s\" (%s).", b,
                                 paste(fm, collapse = ", ")), call. = FALSE)
      }
      a[names(o)] <- o
    }
    sp <- do.call(impls[[b]]$layer_specs, c(list(widths = widths), a))
    if (!is.list(sp) || length(sp) != nl) {
      stop(sprintf("Basis \"%s\": layer_specs() must return %d specifications.",
                   b, nl), call. = FALSE)
    }
    for (l in which(basis == b)) {
      specs[[l]] <- c(list(basis = b), sp[[l]])
    }
  }
  unused <- setdiff(names(dots), used)
  if (length(unused)) {
    stop(sprintf("Argument(s) %s not used by basis %s.",
                 paste0("`", unused, "`", collapse = ", "),
                 paste0("\"", ub, "\"", collapse = ", ")), call. = FALSE)
  }
  specs
}

#' Bases used by the layers of a model
#'
#' @param model A `thangld_kan` object.
#' @return Character vector with one basis name per layer.
#' @examples
#' kan_layer_bases(thangld_kan(c(2, 3, 1), basis = c("rbf", "jacobi"), seed = 1))
#' @export
kan_layer_bases <- function(model) {
  validate_model(model)
  vapply(model$specs, function(s) s$basis, "")
}
