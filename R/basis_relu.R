# basis = "relu" ----------------------------------------
# ReLU-KAN layer -----------------------------------------------------------------
#
# u = relu(x_i - low[i, m]), v = relu(high[i, m] - x_i), b = (r u v)^2,
# out[b, j] = bias[j] + sum_i sum_m weight[i, j, m] * b[i, m](x_i)

#' ReLU-KAN basis functions and their derivatives
#'
#' `relu_basis()` evaluates the \eqn{G = g + k} ReLU-KAN basis functions
#' \deqn{R_m(x) = \{r\,\mathrm{ReLU}(x - s_m)\,\mathrm{ReLU}(e_m - x)\}^2,
#'   \quad m = 1, \dots, G,}
#' on an equally spaced grid over `grid_range = c(a, b)` with \eqn{L = b - a}:
#' \eqn{s_m = a + L(m - 1 - k)/g}, \eqn{e_m = s_m + L(k + 1)/g} and
#' \eqn{r = 4g^2/\{(k + 1)^2 L^2\}}, so each function is a smooth bump with
#' support \eqn{[s_m, e_m]} and peak value 1 at the center. With
#' `grid_range = c(0, 1)` the grid and \eqn{r} are those of the original
#' ReLU-KAN layer (Qiu et al. 2024). `relu_basis_deriv()` returns
#' \eqn{dR_m/dx = 2r^2 uv(v - u)} with \eqn{u = \mathrm{ReLU}(x - s_m)},
#' \eqn{v = \mathrm{ReLU}(e_m - x)}.
#'
#' @param x Numeric vector.
#' @param grid_size Number of grid intervals \eqn{g \ge 1}.
#' @param k Support width in grid intervals minus one, \eqn{k \ge 0}.
#' @param grid_range Numeric `c(min, max)`, `min < max`.
#' @return A `length(x)` by `grid_size + k` matrix.
#' @examples
#' relu_basis(c(-1, 0, 1), grid_size = 5, k = 3)
#' @export
relu_basis <- function(x, grid_size = 5L, k = 3L, grid_range = c(-2, 2)) {
  g <- .relu_relu_grid(grid_size, k, grid_range)
  x <- .relu_check_x(x)
  out <- matrix(.relu_relu_array(matrix(x, ncol = 1L), g$low, g$high, g$r), length(x))
  colnames(out) <- paste0("R", seq_len(ncol(out)))
  out
}

#' @rdname relu_basis
#' @export
relu_basis_deriv <- function(x, grid_size = 5L, k = 3L, grid_range = c(-2, 2)) {
  g <- .relu_relu_grid(grid_size, k, grid_range)
  x <- .relu_check_x(x)
  out <- matrix(.relu_relu_array(matrix(x, ncol = 1L), g$low, g$high, g$r, deriv = "x"),
                length(x))
  colnames(out) <- paste0("dR", seq_len(ncol(out)))
  out
}

#' Grid (low, high as 1 x G matrices) and r
#' @noRd
.relu_relu_grid <- function(grid_size, k, grid_range) {
  grid_size <- check_count(grid_size, "grid_size", min = 1L)
  k <- check_count(k, "k", min = 0L)
  .relu_check_range(grid_range)
  L <- grid_range[2L] - grid_range[1L]
  low <- grid_range[1L] + L * seq.int(-k, grid_size - 1L) / grid_size
  list(low = matrix(low, 1L), high = matrix(low + L * (k + 1) / grid_size, 1L),
       r = 4 * grid_size^2 / ((k + 1)^2 * L^2), g = grid_size, k = k)
}

#' @noRd
.relu_check_range <- function(grid_range) {
  if (!is.numeric(grid_range) || length(grid_range) != 2L ||
      any(!is.finite(grid_range)) || grid_range[1L] >= grid_range[2L]) {
    stop("`grid_range` must be two finite numbers with min < max.", call. = FALSE)
  }
  invisible(TRUE)
}

#' @noRd
.relu_check_x <- function(x) {
  if (!is.numeric(x) || anyNA(x) || any(!is.finite(x))) {
    stop("`x` must be a finite numeric vector.", call. = FALSE)
  }
  as.numeric(x)
}

#' Basis (or a derivative) for an n x p matrix: n x p x G
#' low, high: p x G; deriv "none", "x", "low" or "high"
#' @noRd
.relu_relu_array <- function(h, low, high, r, deriv = "none") {
  n <- nrow(h)
  p <- ncol(h)
  G <- ncol(low)
  xa <- array(h, c(n, p, G))
  la <- array(rep(low, each = n), c(n, p, G))
  ha <- array(rep(high, each = n), c(n, p, G))
  u <- pmax(xa - la, 0)
  v <- pmax(ha - xa, 0)
  switch(deriv,
         none = (r * u * v)^2,
         x = 2 * r^2 * u * v * (v - u),
         low = -2 * r^2 * u * v^2,
         high = 2 * r^2 * u^2 * v)
}

#' @noRd
.relu_layer_describe <- function(spec) {
  sprintf("ReLU-KAN g=%d, k=%d on [%g, %g]", spec$grid_size, spec$k,
          spec$grid_min, spec$grid_max)
}

#' @noRd
.relu_layer_param_dims <- function(p, q, spec) {
  G <- spec$grid_size + spec$k
  list(weight = c(p, q, G), bias = q, low = c(p, G), high = c(p, G))
}

#' @noRd
.relu_layer_init <- function(p, q, spec) {
  G <- spec$grid_size + spec$k
  g <- .relu_relu_grid(spec$grid_size, spec$k, c(spec$grid_min, spec$grid_max))
  bound <- 1 / sqrt(p * G)   # Conv2d default (fan_in = in * G)
  list(weight = array(stats::runif(p * q * G, -bound, bound), c(p, q, G)),
       bias = array(stats::runif(q, -bound, bound), q),
       low = matrix(g$low, p, G, byrow = TRUE),
       high = matrix(g$high, p, G, byrow = TRUE))
}

#' @noRd
.relu_layer_check_params <- function(par, spec) invisible(TRUE)

#' @noRd
.relu_layer_forward_r <- function(h, par, spec) {
  n <- nrow(h)
  d <- dim(par$weight)
  b <- matrix(.relu_relu_array(h, par$low, par$high, spec$r), n, d[1L] * d[3L])
  wmat <- matrix(aperm(par$weight, c(1L, 3L, 2L)), d[1L] * d[3L], d[2L])
  out <- b %*% wmat + rep(par$bias, each = n)
  list(out = out, cache = list(h = h, b = b, wmat = wmat))
}

#' @noRd
.relu_layer_backward_r <- function(cache, g, par, spec) {
  d <- dim(par$weight)
  n <- nrow(g)
  gw <- aperm(array(crossprod(cache$b, g), c(d[1L], d[3L], d[2L])), c(1L, 3L, 2L))
  db <- array(g %*% t(cache$wmat), c(n, d[1L], d[3L]))
  dd <- function(w) .relu_relu_array(cache$h, par$low, par$high, spec$r, deriv = w)
  list(grads = list(weight = gw, bias = array(colSums(g), d[2L]),
                    low = colSums(db * dd("low"), dims = 1L),
                    high = colSums(db * dd("high"), dims = 1L)),
       dh = rowSums(db * dd("x"), dims = 2L))
}

#' @noRd
.relu_layer_forward_torch <- function(h, par, spec) {
  x <- h$unsqueeze(3L)
  u <- torch::nnf_relu(x - par$low)
  v <- torch::nnf_relu(par$high - x)
  b <- (u * v * spec$r)^2   # tensor first: `scalar * tensor` rounds the scalar to float32
  torch::torch_einsum("bim,iom->bo", list(b, par$weight)) + par$bias
}

#' @noRd
.relu_layer_edge_r <- function(x, par, spec, i, j) {
  b <- matrix(.relu_relu_array(matrix(x, ncol = 1L), par$low[i, , drop = FALSE],
                         par$high[i, , drop = FALSE], spec$r), length(x))
  data.frame(phi = drop(b %*% par$weight[i, j, ]))
}

#' Independent implementation: explicit loops and an explicit piecewise rule
#' (zero outside the support, normalized quartic bump inside); bias added once
#' @noRd
.relu_layer_naive_forward <- function(h, par, spec) {
  d <- dim(par$weight)
  out <- matrix(0, nrow(h), d[2L])
  for (b in seq_len(nrow(h))) for (j in seq_len(d[2L])) {
    s <- par$bias[j]
    for (i in seq_len(d[1L])) for (m in seq_len(d[3L])) {
      lo <- par$low[i, m]
      hi <- par$high[i, m]
      x <- h[b, i]
      if (x > lo && x < hi) {
        s <- s + par$weight[i, j, m] * spec$r^2 * ((x - lo) * (hi - x))^2
      }
    }
    out[b, j] <- s
  }
  out
}

#' @noRd
.relu_layer_basis_checks <- function() {
  x <- c(seq(-2.7, 2.7, by = 0.0137), -2, 0, 2)
  g <- .relu_relu_grid(5L, 3L, c(-2, 2))
  B <- relu_basis(x, 5, 3, c(-2, 2))
  w <- g$high - g$low
  expl <- sapply(seq_len(ncol(B)), function(m) {
    t <- (x - g$low[m]) / w[m]
    ifelse(t > 0 & t < 1, 16 * t^2 * (1 - t)^2, 0)
  })
  h <- 1e-6
  fdx <- (relu_basis(x + h, 5, 3, c(-2, 2)) - relu_basis(x - h, 5, 3, c(-2, 2))) / (2 * h)
  hm <- matrix(x, ncol = 1L)
  lo <- g$low[1, 4]
  hi <- g$high[1, 4]
  fdp <- function(dl, dh) (.relu_relu_array(hm, matrix(lo + dl), matrix(hi + dh), g$r) -
                             .relu_relu_array(hm, matrix(lo - dl), matrix(hi - dh), g$r)) / (2 * h)
  dlow <- max(abs(drop(.relu_relu_array(hm, matrix(lo), matrix(hi), g$r, "low")) - drop(fdp(h, 0))))
  dhigh <- max(abs(drop(.relu_relu_array(hm, matrix(lo), matrix(hi), g$r, "high")) - drop(fdp(0, h))))
  area <- vapply(seq_len(ncol(B)), function(m) stats::integrate(function(t)
    relu_basis(t, 5, 3, c(-2, 2))[, m], g$low[m], g$high[m], rel.tol = 1e-12)$value,
    numeric(1))
  orig <- .relu_relu_grid(5L, 3L, c(0, 1))
  mid <- (g$low + g$high) / 2
  list(
    list(check = "basis vs explicit piecewise quartic 16 t^2 (1 - t)^2 (g = 5, k = 3)",
         error = max(abs(B - expl)), tol = 1e-10),
    list(check = "peak value 1 at the support midpoints",
         error = max(abs(diag(relu_basis(drop(mid), 5, 3, c(-2, 2))) - 1)), tol = 1e-12),
    list(check = "derivative in x vs central difference (h = 1e-6)",
         error = max(abs(relu_basis_deriv(x, 5, 3, c(-2, 2)) - fdx)), tol = 1e-6),
    list(check = "derivatives in low and high vs central difference (h = 1e-6)",
         error = max(dlow, dhigh), tol = 1e-6),
    list(check = "integral of each basis function = 8 w / 15 (integrate)",
         error = max(abs(area - 8 * drop(w) / 15)), tol = 1e-8),
    list(check = "grid_range c(0, 1) reproduces the original arange(-k, g)/g and r = 4g^2/(k+1)^2",
         error = max(abs(orig$low - (-3:4) / 5), abs(orig$high - ((-3:4) / 5 + 4 / 5)),
                     abs(orig$r - 4 * 25 / 16)), tol = 1e-14)
  )
}

#' @noRd
.relu_validation_fixture <- function(...) {
  model <- thangld_kan(basis = "relu", c(3, 4, 2, 1), grid_size = c(4, 3, 5), k = c(2, 3, 1), seed = 11, ...)
  # move phases and the scale away from the grid so the trainable low and high
  # differ per input
  model$params <- with_seed(14, lapply(model$params, function(par) {
    par$low <- par$low + matrix(stats::rnorm(length(par$low), sd = 0.1), nrow(par$low))
    par$high <- par$high + matrix(stats::rnorm(length(par$high), sd = 0.1), nrow(par$high))
    par
  }))
  x <- with_seed(12, matrix(stats::rnorm(18, sd = 1), 6, 3))
  y <- with_seed(13, stats::rnorm(6))
  list(model = model, x = x, y = y)
}


#' Layer specifications for all layers (seed handled by thangld_kan())
#' @noRd
.relu_layer_specs <- function(widths, grid_size = 5L, k = 3L, grid_range = c(-2, 2)) {
  widths <- check_widths(widths)
  nl <- length(widths) - 1L
  chk <- function(v, nm, min) {
    if (!is.numeric(v) || !(length(v) %in% c(1L, nl)) || anyNA(v) ||
        any(v != round(v)) || any(v < min)) {
      stop(sprintf("`%s` must be one integer >= %d or %d integers.", nm, min, nl),
           call. = FALSE)
    }
    rep_len(as.integer(v), nl)
  }
  grid_size <- chk(grid_size, "grid_size", 1L)
  k <- chk(k, "k", 0L)
  .relu_check_range(grid_range)
  specs <- lapply(seq_len(nl), function(l) {
    g <- .relu_relu_grid(grid_size[l], k[l], grid_range)
    list(grid_size = grid_size[l], k = k[l], grid_min = as.numeric(grid_range[1L]),
         grid_max = as.numeric(grid_range[2L]), r = g$r)
  })
  specs
}