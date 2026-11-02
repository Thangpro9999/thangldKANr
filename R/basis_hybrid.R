# basis = "hybrid" ----------------------------------------
# Hybrid KAN layer -----------------------------------------------------------------
#
# phi[j,i](x) = base[i,j] silu(x)
#             + sum_k cheb[i,j,k] T_k(tanh x)
#             + sum_m rbf[i,j,m] exp(-((x - mu_m) / d)^2)
# out[b, j] = sum_i phi[j,i](h[b, i]); any non-empty subset of the three parts.

.hybrid_hybrid_bases <- c("silu", "chebyshev", "rbf")

#' Basis functions of a hybrid KAN edge
#'
#' `hybrid_basis()` returns the features one edge combines: \eqn{\mathrm{silu}(x)
#' = x / (1 + e^{-x})}, the Chebyshev polynomials \eqn{T_0, \dots, T_K} of
#' \eqn{\tanh(x)} (as in ChebyKAN) and the Gaussian radial basis
#' functions \eqn{\exp(-((x - \mu_m)/d)^2)} on a fixed grid (as in FastKAN).
#' `hybrid_basis_deriv()` returns their derivatives with respect to
#' \eqn{x}, using \eqn{T'_{k+1} = 2 T_k + 2 z T'_k - T'_{k-1}} (no
#' singularity at \eqn{z = \pm 1}) and the chain rule through \eqn{\tanh}.
#'
#' @param x Numeric vector.
#' @param degree Chebyshev degree \eqn{K \ge 0}.
#' @param num_grids Number of RBF centers \eqn{G \ge 2}.
#' @param grid_min,grid_max Range of the RBF centers.
#' @param denominator RBF width \eqn{d}; `NULL` gives the center spacing.
#' @return A matrix with columns `silu`, `T0..TK` and `rbf1..rbfG`.
#' @examples
#' hybrid_basis(c(-1, 0, 1), degree = 2, num_grids = 3)
#' @export
hybrid_basis <- function(x, degree = 3L, num_grids = 8L, grid_min = -2,
                         grid_max = 2, denominator = NULL) {
  x <- .hybrid_check_z(x)
  g <- .hybrid_rbf_grid_spec(num_grids, grid_min, grid_max, denominator)
  degree <- check_count(degree, "degree", min = 0L)
  z <- tanh(x)
  t <- .hybrid_cheb_array(matrix(z, ncol = 1L), degree)
  out <- cbind(.hybrid_silu(x), matrix(t, length(x)), matrix(.hybrid_rbf_array(matrix(x, ncol = 1L), g), length(x)))
  colnames(out) <- c("silu", paste0("T", 0:degree), paste0("rbf", seq_len(g$num_grids)))
  out
}

#' @rdname hybrid_basis
#' @export
hybrid_basis_deriv <- function(x, degree = 3L, num_grids = 8L, grid_min = -2,
                               grid_max = 2, denominator = NULL) {
  x <- .hybrid_check_z(x)
  g <- .hybrid_rbf_grid_spec(num_grids, grid_min, grid_max, denominator)
  degree <- check_count(degree, "degree", min = 0L)
  z <- tanh(x)
  dt <- .hybrid_cheb_array(matrix(z, ncol = 1L), degree, deriv = TRUE)
  out <- cbind(.hybrid_silu_deriv(x), matrix(dt, length(x)) * (1 - z^2),
               matrix(.hybrid_rbf_array(matrix(x, ncol = 1L), g, deriv = TRUE), length(x)))
  colnames(out) <- c("silu", paste0("T", 0:degree), paste0("rbf", seq_len(g$num_grids)))
  out
}

#' @noRd
.hybrid_check_z <- function(z) {
  if (!is.numeric(z) || anyNA(z) || any(!is.finite(z))) {
    stop("`x` must be a finite numeric vector.", call. = FALSE)
  }
  as.numeric(z)
}

#' @noRd
.hybrid_silu <- function(x) x / (1 + exp(-x))

#' @noRd
.hybrid_silu_deriv <- function(x) {
  s <- 1 / (1 + exp(-x))
  s * (1 + x * (1 - s))
}

#' @noRd
.hybrid_rbf_grid_spec <- function(num_grids, grid_min, grid_max, denominator) {
  num_grids <- check_count(num_grids, "num_grids", min = 2L)
  grid_min <- check_number(grid_min, "grid_min")
  grid_max <- check_number(grid_max, "grid_max")
  if (grid_max <= grid_min) stop("`grid_max` must exceed `grid_min`.", call. = FALSE)
  centers <- seq(grid_min, grid_max, length.out = num_grids)
  d <- if (is.null(denominator)) (grid_max - grid_min) / (num_grids - 1L) else
    check_number(denominator, "denominator", min = 0, strict = TRUE)
  list(num_grids = num_grids, grid_min = grid_min, grid_max = grid_max,
       centers = centers, denominator = d)
}

#' Chebyshev T_k(z) (or dT_k/dz) for an n x p matrix: n x p x (K+1)
#' @noRd
.hybrid_cheb_array <- function(z, degree, deriv = FALSE) {
  b <- array(0, c(nrow(z), ncol(z), degree + 1L))
  b[, , 1L] <- 1
  if (degree >= 1L) b[, , 2L] <- z
  if (degree >= 2L) for (k in 2:degree) b[, , k + 1L] <- 2 * z * b[, , k] - b[, , k - 1L]
  if (!deriv) return(b)
  d <- array(0, dim(b))
  if (degree >= 1L) d[, , 2L] <- 1
  if (degree >= 2L) {
    for (k in 2:degree) d[, , k + 1L] <- 2 * b[, , k] + 2 * z * d[, , k] - d[, , k - 1L]
  }
  d
}

#' Gaussian RBF (or derivative wrt x) for an n x p matrix: n x p x G
#' @noRd
.hybrid_rbf_array <- function(h, g, deriv = FALSE) {
  G <- length(g$centers)
  b <- array(0, c(nrow(h), ncol(h), G))
  for (m in seq_len(G)) {
    u <- (h - g$centers[m]) / g$denominator
    b[, , m] <- if (deriv) -2 * u / g$denominator * exp(-u^2) else exp(-u^2)
  }
  b
}

#' @noRd
.hybrid_layer_describe <- function(spec) {
  parts <- c(silu = "SiLU", chebyshev = sprintf("Chebyshev K=%d", spec$degree),
             rbf = sprintf("RBF G=%d", spec$grid$num_grids))
  paste(parts[spec$bases], collapse = " + ")
}

#' @noRd
.hybrid_layer_param_dims <- function(p, q, spec) {
  d <- list(base = c(p, q), cheb = c(p, q, spec$degree + 1L),
            rbf = c(p, q, spec$grid$num_grids))
  d[c(silu = "base", chebyshev = "cheb", rbf = "rbf")[spec$bases]]
}

#' @noRd
.hybrid_layer_init <- function(p, q, spec) {
  out <- list()
  if ("silu" %in% spec$bases) {
    out$base <- array(stats::runif(p * q, -1 / sqrt(p), 1 / sqrt(p)), c(p, q))
  }
  if ("chebyshev" %in% spec$bases) {
    k1 <- spec$degree + 1L
    out$cheb <- array(stats::rnorm(p * q * k1, 0, 1 / (p * k1)), c(p, q, k1))
  }
  if ("rbf" %in% spec$bases) {
    G <- spec$grid$num_grids
    out$rbf <- array(stats::rnorm(p * q * G, 0, spec$rbf_init_sd), c(p, q, G))
  }
  out
}

#' @noRd
.hybrid_layer_check_params <- function(par, spec) invisible(TRUE)

#' Feature blocks of one layer: list of (feature matrix n x (p*K), K, coef name)
#' @noRd
.hybrid_hybrid_blocks <- function(h, spec, deriv = FALSE) {
  n <- nrow(h)
  p <- ncol(h)
  bl <- list()
  if ("silu" %in% spec$bases) {
    bl$base <- list(f = .hybrid_silu(h), df = if (deriv) .hybrid_silu_deriv(h), k = 1L)
  }
  if ("chebyshev" %in% spec$bases) {
    z <- tanh(h)
    k1 <- spec$degree + 1L
    bl$cheb <- list(f = matrix(.hybrid_cheb_array(z, spec$degree), n, p * k1),
                    df = if (deriv) matrix(.hybrid_cheb_array(z, spec$degree, TRUE) *
                                             as.vector(1 - z^2), n, p * k1),
                    k = k1)
  }
  if ("rbf" %in% spec$bases) {
    G <- spec$grid$num_grids
    bl$rbf <- list(f = matrix(.hybrid_rbf_array(h, spec$grid), n, p * G),
                   df = if (deriv) matrix(.hybrid_rbf_array(h, spec$grid, TRUE), n, p * G),
                   k = G)
  }
  bl
}

#' coefficient array `[p, q, K]` (or `[p, q]`) -> `(p*K) x q` matrix
#' @noRd
.hybrid_coef_matrix <- function(a, k) {
  d <- dim(a)
  if (length(d) == 2L) return(a)
  matrix(aperm(a, c(1L, 3L, 2L)), d[1L] * k, d[2L])
}

#' @noRd
.hybrid_layer_forward_r <- function(h, par, spec) {
  bl <- .hybrid_hybrid_blocks(h, spec, deriv = TRUE)
  out <- 0
  for (nm in names(bl)) {
    bl[[nm]]$cmat <- .hybrid_coef_matrix(par[[nm]], bl[[nm]]$k)
    out <- out + bl[[nm]]$f %*% bl[[nm]]$cmat
  }
  list(out = out, cache = list(bl = bl, n = nrow(h), p = ncol(h)))
}

#' @noRd
.hybrid_layer_backward_r <- function(cache, g, par, spec) {
  n <- cache$n
  p <- cache$p
  q <- ncol(g)
  grads <- list()
  dh <- matrix(0, n, p)
  for (nm in names(cache$bl)) {
    b <- cache$bl[[nm]]
    k <- b$k
    gc <- crossprod(b$f, g)
    grads[[nm]] <- if (nm == "base") gc else aperm(array(gc, c(p, k, q)), c(1L, 3L, 2L))
    db <- (g %*% t(b$cmat)) * b$df
    dh <- dh + rowSums(array(db, c(n, p, k)), dims = 2L)
  }
  list(grads = grads[names(par)], dh = dh)
}

#' @noRd
.hybrid_layer_forward_torch <- function(h, par, spec) {
  out <- NULL
  add <- function(a, b) if (is.null(a)) b else a + b
  if (!is.null(par$base)) {
    out <- add(out, torch::torch_matmul(torch::nnf_silu(h), par$base))
  }
  if (!is.null(par$cheb)) {
    k1 <- par$cheb$size(3L)
    z <- torch::torch_tanh(h)
    tl <- vector("list", k1)
    tl[[1L]] <- torch::torch_ones_like(z)
    if (k1 >= 2L) tl[[2L]] <- z
    if (k1 >= 3L) for (k in 3:k1) tl[[k]] <- 2 * z * tl[[k - 1L]] - tl[[k - 2L]]
    out <- add(out, torch::torch_einsum("bid,iod->bo",
                                        list(torch::torch_stack(tl, dim = 3L), par$cheb)))
  }
  if (!is.null(par$rbf)) {
    cen <- r_tensor(spec$grid$centers, dtype = h$dtype)
    u <- (h$unsqueeze(3L) - cen) / spec$grid$denominator
    out <- add(out, torch::torch_einsum("bid,iod->bo", list(torch::torch_exp(-u^2), par$rbf)))
  }
  out
}

#' @noRd
.hybrid_layer_edge_r <- function(x, par, spec, i, j) {
  out <- data.frame(z = x)
  tot <- 0
  if (!is.null(par$base)) {
    out$silu <- par$base[i, j] * .hybrid_silu(x)
    tot <- tot + out$silu
  }
  if (!is.null(par$cheb)) {
    out$chebyshev <- drop(matrix(.hybrid_cheb_array(matrix(tanh(x)), spec$degree), length(x)) %*%
                            par$cheb[i, j, ])
    tot <- tot + out$chebyshev
  }
  if (!is.null(par$rbf)) {
    out$rbf <- drop(matrix(.hybrid_rbf_array(matrix(x), spec$grid), length(x)) %*% par$rbf[i, j, ])
    tot <- tot + out$rbf
  }
  out$phi <- tot
  out
}

#' Independent implementation: explicit loops, cos(k acos z), explicit exp
#' @noRd
.hybrid_layer_naive_forward <- function(h, par, spec) {
  q <- if (!is.null(par$base)) ncol(par$base) else dim(par[[1L]])[2L]
  out <- matrix(0, nrow(h), q)
  for (b in seq_len(nrow(h))) for (j in seq_len(q)) {
    s <- 0
    for (i in seq_len(ncol(h))) {
      x <- h[b, i]
      if (!is.null(par$base)) s <- s + par$base[i, j] * x / (1 + exp(-x))
      if (!is.null(par$cheb)) {
        th <- acos(tanh(x))
        for (k in 0:spec$degree) s <- s + par$cheb[i, j, k + 1L] * cos(k * th)
      }
      if (!is.null(par$rbf)) {
        for (m in seq_along(spec$grid$centers)) {
          s <- s + par$rbf[i, j, m] *
            exp(-((x - spec$grid$centers[m]) / spec$grid$denominator)^2)
        }
      }
    }
    out[b, j] <- s
  }
  out
}

#' @noRd
.hybrid_layer_basis_checks <- function() {
  x <- c(-3, -1.2, -0.3, 0, 0.4, 1.1, 2.7)
  b <- hybrid_basis(x, degree = 5, num_grids = 6)
  z <- tanh(x)
  exT <- outer(acos(z), 0:5, function(t, k) cos(k * t))
  cen <- seq(-2, 2, length.out = 6)
  exR <- exp(-(outer(x, cen, `-`) / 0.8)^2)
  h <- 1e-5
  fd <- (hybrid_basis(x + h, 5, 6) - hybrid_basis(x - h, 5, 6)) / (2 * h)
  list(
    list(check = "SiLU vs x * plogis(x)", error = max(abs(b[, 1L] - x * stats::plogis(x))),
         tol = 1e-12),
    list(check = "Chebyshev part vs cos(k acos(tanh x))", error = max(abs(b[, 2:7] - exT)),
         tol = 1e-10),
    list(check = "RBF part vs explicit exp (6 centers on [-2, 2], d = 0.8)",
         error = max(abs(b[, 8:13] - exR)), tol = 1e-12),
    list(check = "basis derivatives vs central difference (h = 1e-5)",
         error = max(abs(hybrid_basis_deriv(x, 5, 6) - fd)), tol = 1e-6)
  )
}

#' @noRd
.hybrid_validation_fixture <- function(...) {
  model <- thangld_kan(basis = "hybrid", c(3, 4, 2, 1), degree = c(4, 3, 5), num_grids = c(5, 6, 4), seed = 11, ...)
  model$params <- with_seed(21, lapply(model$params, function(pl) {
    lapply(pl, function(a) {
      a[] <- stats::rnorm(length(a), sd = 0.4)
      a
    })
  }))
  x <- with_seed(12, matrix(stats::rnorm(18, sd = 1.5), 6, 3))
  y <- with_seed(13, stats::rnorm(6))
  list(model = model, x = x, y = y)
}


#' Layer specifications for all layers (seed handled by thangld_kan())
#' @noRd
.hybrid_layer_specs <- function(widths, bases = c("silu", "chebyshev", "rbf"),
                       degree = 3L, num_grids = 8L, grid_min = -2, grid_max = 2,
                       denominator = NULL, rbf_init_sd = 0.1) {
  widths <- check_widths(widths)
  nl <- length(widths) - 1L
  if (!is.character(bases) || !length(bases) || anyNA(bases) ||
      !all(bases %in% .hybrid_hybrid_bases) || anyDuplicated(bases)) {
    stop("`bases` must be a non-empty subset of \"silu\", \"chebyshev\", \"rbf\".",
         call. = FALSE)
  }
  bases <- .hybrid_hybrid_bases[.hybrid_hybrid_bases %in% bases]
  chk <- function(v, nm, lo) {
    if (!is.numeric(v) || !(length(v) %in% c(1L, nl)) || anyNA(v) ||
        any(v != round(v)) || any(v < lo)) {
      stop(sprintf("`%s` must be one integer >= %d or %d integers.", nm, lo, nl),
           call. = FALSE)
    }
    rep_len(as.integer(v), nl)
  }
  degree <- chk(degree, "degree", 0L)
  num_grids <- chk(num_grids, "num_grids", 2L)
  rbf_init_sd <- check_number(rbf_init_sd, "rbf_init_sd", min = 0, strict = TRUE)
  specs <- lapply(seq_len(nl), function(l) list(
    bases = bases, degree = degree[l],
    grid = .hybrid_rbf_grid_spec(num_grids[l], grid_min, grid_max, denominator),
    rbf_init_sd = rbf_init_sd))
  specs
}

#' Size of each basis part on data
#'
#' For every edge of one layer, evaluates the SiLU, Chebyshev and RBF parts
#' at the layer inputs produced by `x` and reports the standard deviation of
#' each part and its share of the summed standard deviations. This is a
#' description of the fitted coefficients (the parts are not orthogonal; see
#' [thangld_kan()]), not a variable-importance or causal measure.
#'
#' @param model A (fitted) `thangld_kan` whose layer `layer` uses `basis = "hybrid"`.
#' @param x Predictor matrix or data frame on the original scale (columns as
#'   in the fit).
#' @param layer Layer index.
#' @return A data frame with one row per edge and part: `input`, `output`,
#'   `part`, `sd`, `share`.
#' @examples
#' set.seed(1)
#' x <- matrix(runif(200, -2, 2), 100, 2, dimnames = list(NULL, c("a", "b")))
#' fit <- thangld_kan_fit(thangld_kan(basis = "hybrid", c(2, 1), seed = 1), x, sin(x[, 1]) + x[, 2],
#'                       epochs = 20, backend = "r", seed = 1)
#' hybrid_kan_components(fit, x)
#' @export
hybrid_kan_components <- function(model, x, layer = 1L) {
  validate_model(model)
  if (!isTRUE(model$fitted)) stop("The model is not fitted.", call. = FALSE)
  layer <- check_count(layer, "layer")
  if (layer > length(model$params)) stop("`layer` exceeds the number of layers.", call. = FALSE)
  xs <- apply_preprocess_x(model$preprocess, align_columns(
    model$preprocess, as_numeric_matrix(x, "x"), model$widths[1L]))
  h <- if (layer == 1L) xs else
    engine_forward(model$params, model$specs, xs)$outs[[layer - 1L]]
  spec <- model$specs[[layer]]
  if (!identical(spec$basis, "hybrid")) {
    stop(sprintf("Layer %d does not use the \"hybrid\" basis.", layer),
         call. = FALSE)
  }
  par <- model$params[[layer]]
  parts <- intersect(c("silu", "chebyshev", "rbf"), spec$bases)
  rows <- list()
  for (i in seq_len(ncol(h))) for (j in seq_len(model$widths[layer + 1L])) {
    e <- .hybrid_layer_edge_r(h[, i], par, spec, i, j)
    s <- vapply(parts, function(pp) stats::sd(e[[pp]]), numeric(1))
    rows[[length(rows) + 1L]] <- data.frame(input = i, output = j, part = parts,
      sd = unname(s), share = unname(s / sum(s)))
  }
  do.call(rbind, rows)
}