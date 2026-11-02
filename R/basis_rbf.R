# basis = "rbf" ----------------------------------------
# Gaussian RBF (FastKAN-type) layer ------------------------------------------------
#
# h -> u = LayerNorm(h) (optional; never for a layer with one input)
# out[b, j] = sum_i sum_m w[i, j, m] exp(-((u[b, i] - c_m) / d)^2)
#             + (optional) sum_i base_weight[j, i] silu(h[b, i]) + base_bias[j]

#' Gaussian radial basis functions
#'
#' `rbf_basis()` evaluates \eqn{b_m(u) = \exp(-((u - c_m)/d)^2)} for centers
#' `centers` and width (denominator) `d`; `rbf_basis_deriv()` returns
#' \eqn{db_m/du = -2 (u - c_m) / d^2 \, b_m(u)}. `rbf_grid()` returns the
#' FastKAN default grid: `num_grids` equally spaced centers on
#' `[grid_min, grid_max]` and `d = (grid_max - grid_min) / (num_grids - 1)`.
#'
#' @param u Numeric vector.
#' @param centers Numeric vector of centers.
#' @param d Positive width.
#' @param grid_min,grid_max,num_grids Grid definition.
#' @return `rbf_basis()`/`rbf_basis_deriv()`: a `length(u)` by
#'   `length(centers)` matrix. `rbf_grid()`: a list with `centers` and `d`.
#' @examples
#' g <- rbf_grid(-2, 2, 8)
#' g
#' round(rbf_basis(c(-2, 0, 2), g$centers, g$d), 3)
#' @export
rbf_basis <- function(u, centers, d) {
  u <- .rbf_check_vec(u, "u")
  centers <- .rbf_check_vec(centers, "centers")
  d <- check_number(d, "d", min = 0, strict = TRUE)
  exp(-(outer(u, centers, "-") / d)^2)
}

#' @rdname rbf_basis
#' @export
rbf_basis_deriv <- function(u, centers, d) {
  b <- rbf_basis(u, centers, d)
  -2 * outer(u, centers, "-") / d^2 * b
}

#' @rdname rbf_basis
#' @export
rbf_grid <- function(grid_min = -2, grid_max = 2, num_grids = 8L) {
  grid_min <- check_number(grid_min, "grid_min")
  grid_max <- check_number(grid_max, "grid_max")
  num_grids <- check_count(num_grids, "num_grids", min = 2L)
  if (grid_max <= grid_min) stop("`grid_max` must exceed `grid_min`.",
                                 call. = FALSE)
  list(centers = seq(grid_min, grid_max, length.out = num_grids),
       d = (grid_max - grid_min) / (num_grids - 1))
}

#' @noRd
.rbf_check_vec <- function(z, name) {
  if (!is.numeric(z) || anyNA(z) || any(!is.finite(z))) {
    stop(sprintf("`%s` must be a finite numeric vector.", name), call. = FALSE)
  }
  as.numeric(z)
}

#' @noRd
.rbf_silu_r <- function(x) x / (1 + exp(-x))

#' @noRd
.rbf_silu_deriv_r <- function(x) {
  s <- 1 / (1 + exp(-x))
  s * (1 + x * (1 - s))
}

#' Layer specification (one per layer)
#' @noRd
.rbf_fast_layer_spec <- function(p, centers, d, layernorm, base_update, ln_eps,
                            spline_init_sd) {
  list(centers = centers, d = d, layernorm = layernorm && p > 1L,
       ln_eps = ln_eps, base_update = base_update,
       spline_init_sd = spline_init_sd)
}

#' @noRd
.rbf_layer_describe <- function(spec) {
  sprintf("RBF G=%d d=%.3g%s%s", length(spec$centers), spec$d,
          if (spec$layernorm) " +LN" else "",
          if (spec$base_update) " +base" else "")
}

#' @noRd
.rbf_layer_param_dims <- function(p, q, spec) {
  out <- list(weight = c(p, q, length(spec$centers)))
  if (spec$layernorm) out <- c(out, list(ln_weight = p, ln_bias = p))
  if (spec$base_update) out <- c(out, list(base_weight = c(q, p),
                                           base_bias = q))
  out
}

#' Initialization following FastKAN: spline weights N(0, 0.1^2) truncated
#' to `[-2, 2]`; base layer as torch nn.Linear (U(-1/sqrt(p), 1/sqrt(p)));
#' LayerNorm weight 1, bias 0.
#' @noRd
.rbf_layer_init <- function(p, q, spec) {
  g <- length(spec$centers)
  w <- stats::rnorm(p * q * g, 0, spec$spline_init_sd)
  w <- pmin(pmax(w, -2), 2)
  out <- list(weight = array(w, c(p, q, g)))
  if (spec$layernorm) {
    out$ln_weight <- array(1, p)
    out$ln_bias <- array(0, p)
  }
  if (spec$base_update) {
    bound <- 1 / sqrt(p)
    out$base_weight <- array(stats::runif(q * p, -bound, bound), c(q, p))
    out$base_bias <- array(stats::runif(q, -bound, bound), q)
  }
  out
}

#' @noRd
.rbf_layer_check_params <- function(par, spec) invisible(TRUE)

#' Layer normalization over the features of each row
#' @noRd
.rbf_layernorm_r <- function(h, gamma, beta, eps) {
  mu <- rowMeans(h)
  xc <- h - mu
  rstd <- 1 / sqrt(rowMeans(xc^2) + eps)
  xhat <- xc * rstd
  u <- sweep(sweep(xhat, 2L, as.numeric(gamma), "*"), 2L, as.numeric(beta), "+")
  list(u = u, xhat = xhat, rstd = rstd)
}

#' @noRd
.rbf_layer_forward_r <- function(h, par, spec) {
  n <- nrow(h)
  p <- ncol(h)
  w <- par$weight
  q <- dim(w)[2L]
  g <- dim(w)[3L]
  ln <- if (spec$layernorm) .rbf_layernorm_r(h, par$ln_weight, par$ln_bias,
                                        spec$ln_eps) else NULL
  u <- if (spec$layernorm) ln$u else h
  diff <- array(u, c(n, p, g)) - rep(spec$centers, each = n * p)
  b <- exp(-(diff / spec$d)^2)
  bm <- matrix(b, n, p * g)
  wm <- matrix(aperm(w, c(1L, 3L, 2L)), p * g, q)
  out <- bm %*% wm
  if (spec$base_update) {
    out <- out + .rbf_silu_r(h) %*% t(par$base_weight)
    out <- sweep(out, 2L, as.numeric(par$base_bias), "+")
  }
  list(out = out, cache = list(h = h, ln = ln, diff = diff, b = b, bm = bm,
                               wm = wm))
}

#' @noRd
.rbf_layer_backward_r <- function(cache, g, par, spec) {
  h <- cache$h
  n <- nrow(h)
  p <- ncol(h)
  d3 <- dim(par$weight)
  q <- d3[2L]
  gg <- d3[3L]
  grads <- list()
  gw <- crossprod(cache$bm, g)
  grads$weight <- aperm(array(gw, c(p, gg, q)), c(1L, 3L, 2L))
  db <- array(g %*% t(cache$wm), c(n, p, gg))
  dbdu <- -2 * cache$diff / spec$d^2 * cache$b
  du <- rowSums(db * dbdu, dims = 2L)
  if (spec$layernorm) {
    ln <- cache$ln
    grads$ln_weight <- array(colSums(du * ln$xhat), p)
    grads$ln_bias <- array(colSums(du), p)
    dxhat <- sweep(du, 2L, as.numeric(par$ln_weight), "*")
    dh <- ln$rstd * (dxhat - rowMeans(dxhat) - ln$xhat * rowMeans(dxhat * ln$xhat))
  } else {
    dh <- du
  }
  if (spec$base_update) {
    grads$base_weight <- crossprod(g, .rbf_silu_r(h))
    grads$base_bias <- array(colSums(g), q)
    dh <- dh + (g %*% par$base_weight) * .rbf_silu_deriv_r(h)
  }
  list(grads = grads, dh = dh)
}

#' @noRd
.rbf_layer_forward_torch <- function(h, par, spec) {
  u <- if (spec$layernorm) {
    torch::nnf_layer_norm(h, h$size(2L), weight = par$ln_weight,
                          bias = par$ln_bias, eps = spec$ln_eps)
  } else h
  centers <- r_tensor(spec$centers, dtype = h$dtype)
  b <- torch::torch_exp(-((u$unsqueeze(3L) - centers) / spec$d)^2)
  out <- torch::torch_einsum("bim,iom->bo", list(b, par$weight))
  if (spec$base_update) {
    out <- out + torch::nnf_linear(torch::nnf_silu(h), par$base_weight,
                                   par$base_bias)
  }
  out
}

#' @noRd
.rbf_layer_edge_r <- function(x, par, spec, i, j) {
  rbf <- drop(rbf_basis(x, spec$centers, spec$d) %*% par$weight[i, j, ])
  base <- if (spec$base_update) par$base_weight[j, i] * .rbf_silu_r(x) else 0 * x
  data.frame(rbf = rbf, base = base, phi = rbf + base)
}

#' Independent loop implementation of one layer
#' @noRd
.rbf_layer_naive_forward <- function(h, par, spec) {
  n <- nrow(h)
  p <- ncol(h)
  q <- dim(par$weight)[2L]
  out <- matrix(0, n, q)
  for (b in seq_len(n)) {
    u <- h[b, ]
    if (spec$layernorm) {
      m <- sum(u) / p
      v <- sum((u - m)^2) / p
      u <- (u - m) / sqrt(v + spec$ln_eps) * par$ln_weight + par$ln_bias
    }
    for (j in seq_len(q)) {
      s <- 0
      for (i in seq_len(p)) {
        for (m in seq_along(spec$centers)) {
          s <- s + par$weight[i, j, m] *
            exp(-((u[i] - spec$centers[m]) / spec$d)^2)
        }
        if (spec$base_update) {
          s <- s + par$base_weight[j, i] * h[b, i] / (1 + exp(-h[b, i]))
        }
      }
      if (spec$base_update) s <- s + par$base_bias[j]
      out[b, j] <- s
    }
  }
  out
}

#' Basis-specific acceptance checks
#' @noRd
.rbf_layer_basis_checks <- function() {
  g <- rbf_grid(-2, 2, 8)
  u <- c(-3.1, -2, -0.7, 0, 0.3, 1.9, 2.5)
  b <- rbf_basis(u, g$centers, g$d)
  explicit <- outer(u, g$centers, function(a, c) exp(-((a - c) / (4 / 7))^2))
  hh <- 1e-6
  fd <- (rbf_basis(u + hh, g$centers, g$d) - rbf_basis(u - hh, g$centers, g$d)) /
    (2 * hh)
  hm <- with_seed(3, matrix(stats::rnorm(12), 4, 3))
  ln <- .rbf_layernorm_r(hm, rep(1, 3), rep(0, 3), 0)
  list(
    list(check = "RBF vs explicit exp(-((u-c)/d)^2), FastKAN grid",
         error = max(abs(b - explicit)), tol = 1e-10),
    list(check = "RBF value 1 at centers and exp(-1) at center +- d",
         error = max(abs(c(diag(rbf_basis(g$centers, g$centers, g$d)) - 1,
                           rbf_basis(g$centers[3] + g$d, g$centers[3], g$d) -
                             exp(-1)))), tol = 1e-10),
    list(check = "RBF symmetry b(c + t) = b(c - t)",
         error = max(abs(rbf_basis(0.3 + u, 0.3, 0.8) -
                           rbf_basis(0.3 - u, 0.3, 0.8))), tol = 1e-10),
    list(check = "RBF derivative vs central difference",
         error = max(abs(rbf_basis_deriv(u, g$centers, g$d) - fd)), tol = 1e-6),
    list(check = "layer norm rows have mean 0 and variance 1 (eps = 0)",
         error = max(abs(c(rowMeans(ln$u), rowMeans(ln$u^2) - 1))), tol = 1e-10)
  )
}

#' Fixture used by thangld_kan_validate(): 3 -> 4 -> 2 -> 1 with layer norm and
#' base update, non-trivial layer norm parameters
#' @noRd
.rbf_validation_fixture <- function() {
  model <- thangld_kan(basis = "rbf", c(3, 4, 2, 1), num_grids = 5, seed = 11)
  model$params <- with_seed(21, lapply(model$params, function(pl) {
    lapply(pl, function(a) a + array(stats::rnorm(length(a), sd = 0.2), dim(a)))
  }))
  x <- with_seed(12, matrix(stats::rnorm(18, sd = 1.5), 6, 3))
  y <- with_seed(13, stats::rnorm(6))
  list(model = model, x = x, y = y)
}


#' Layer specifications for all layers (seed handled by thangld_kan())
#' @noRd
.rbf_layer_specs <- function(widths, num_grids = 8L, grid_min = -2, grid_max = 2,
                     denominator = NULL, layernorm = TRUE, base_update = TRUE,
                     ln_eps = 1e-5, spline_init_sd = 0.1) {
  widths <- check_widths(widths)
  g <- rbf_grid(grid_min, grid_max, num_grids)
  d <- if (is.null(denominator)) g$d else
    check_number(denominator, "denominator", min = 0, strict = TRUE)
  layernorm <- check_flag(layernorm, "layernorm")
  base_update <- check_flag(base_update, "base_update")
  ln_eps <- check_number(ln_eps, "ln_eps", min = 0)
  spline_init_sd <- check_number(spline_init_sd, "spline_init_sd", min = 0,
                                 strict = TRUE)
  specs <- lapply(seq_len(length(widths) - 1L), function(l) {
    .rbf_fast_layer_spec(widths[l], g$centers, d, layernorm, base_update, ln_eps,
                    spline_init_sd)
  })
  specs
}