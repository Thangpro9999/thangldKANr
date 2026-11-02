# basis = "chebyshev" ----------------------------------------
# Chebyshev KAN layer --------------------------------------------------------------
#
# z = tanh(h); out[b, j] = sum_i sum_k coef[i, j, k] T_k(z[b, i])
# (the layer of ChebyKAN, SynodicMonth, commit 5f7efdd)

#' Chebyshev polynomials and their derivatives
#'
#' `chebyshev_basis()` evaluates \eqn{T_0(z), \dots, T_K(z)} by the
#' recurrence \eqn{T_0 = 1}, \eqn{T_1 = z},
#' \eqn{T_{k+1}(z) = 2 z T_k(z) - T_{k-1}(z)}.
#' `chebyshev_basis_deriv()` returns \eqn{T'_k(z) = k U_{k-1}(z)} with the
#' Chebyshev polynomials of the second kind \eqn{U}, which has no
#' singularity at \eqn{z = \pm 1} (unlike the \eqn{\arccos} form).
#'
#' @param z Numeric vector.
#' @param degree Maximum degree \eqn{K \ge 0}.
#' @return A `length(z)` by `degree + 1` matrix (column `k + 1` = degree `k`).
#' @examples
#' chebyshev_basis(c(-1, 0, 0.5, 1), 3)
#' chebyshev_basis_deriv(c(-1, 0, 1), 3)
#' @export
chebyshev_basis <- function(z, degree) {
  degree <- check_count(degree, "degree", min = 0L)
  z <- .chebyshev_check_z(z)
  out <- matrix(.chebyshev_cheby_basis_array(matrix(z, ncol = 1L), degree), length(z))
  colnames(out) <- paste0("T", 0:degree)
  out
}

#' @rdname chebyshev_basis
#' @export
chebyshev_basis_deriv <- function(z, degree) {
  degree <- check_count(degree, "degree", min = 0L)
  z <- .chebyshev_check_z(z)
  out <- matrix(.chebyshev_cheby_basis_array(matrix(z, ncol = 1L), degree, deriv = TRUE), length(z))
  colnames(out) <- paste0("dT", 0:degree)
  out
}

#' @noRd
.chebyshev_check_z <- function(z) {
  if (!is.numeric(z) || anyNA(z) || any(!is.finite(z))) {
    stop("`z` must be a finite numeric vector.", call. = FALSE)
  }
  as.numeric(z)
}

#' Chebyshev polynomials T_k and derivatives as n x p x (K+1) arrays
#' @noRd
.chebyshev_cheby_basis_array <- function(z, degree, deriv = FALSE) {
  n <- nrow(z)
  p <- ncol(z)
  out <- array(0, c(n, p, degree + 1L))
  if (!deriv) {
    out[, , 1L] <- 1
    if (degree >= 1L) out[, , 2L] <- z
    if (degree >= 2L) for (k in 2:degree) {
      out[, , k + 1L] <- 2 * z * out[, , k] - out[, , k - 1L]
    }
    return(out)
  }
  if (degree >= 1L) {
    u_prev <- matrix(1, n, p)
    u_curr <- 2 * z
    out[, , 2L] <- u_prev
    if (degree >= 2L) for (k in 2:degree) {
      out[, , k + 1L] <- k * u_curr
      u_next <- 2 * z * u_curr - u_prev
      u_prev <- u_curr
      u_curr <- u_next
    }
  }
  out
}

#' @noRd
.chebyshev_layer_describe <- function(spec) sprintf("Chebyshev K=%d", spec$degree)

#' @noRd
.chebyshev_layer_param_dims <- function(p, q, spec) list(coef = c(p, q, spec$degree + 1L))

#' @noRd
.chebyshev_layer_init <- function(p, q, spec) {
  k1 <- spec$degree + 1L
  sd <- if (is.null(spec$init_sd)) 1 / (p * k1) else spec$init_sd
  list(coef = array(stats::rnorm(p * q * k1, 0, sd), c(p, q, k1)))
}

#' @noRd
.chebyshev_layer_check_params <- function(par, spec) invisible(TRUE)

#' @noRd
.chebyshev_layer_forward_r <- function(h, par, spec) {
  n <- nrow(h)
  d <- dim(par$coef)
  p <- d[1L]
  q <- d[2L]
  k1 <- d[3L]
  z <- tanh(h)
  b <- matrix(.chebyshev_cheby_basis_array(z, k1 - 1L), n, p * k1)
  cmat <- matrix(aperm(par$coef, c(1L, 3L, 2L)), p * k1, q)
  list(out = b %*% cmat, cache = list(z = z, b = b, cmat = cmat))
}

#' @noRd
.chebyshev_layer_backward_r <- function(cache, g, par, spec) {
  d <- dim(par$coef)
  p <- d[1L]
  q <- d[2L]
  k1 <- d[3L]
  n <- nrow(g)
  gc <- crossprod(cache$b, g)
  grad <- aperm(array(gc, c(p, k1, q)), c(1L, 3L, 2L))
  db <- g %*% t(cache$cmat)
  dbdz <- matrix(.chebyshev_cheby_basis_array(cache$z, k1 - 1L, deriv = TRUE), n, p * k1)
  dz <- rowSums(array(db * dbdz, c(n, p, k1)), dims = 2L)
  list(grads = list(coef = grad), dh = dz * (1 - cache$z^2))
}

#' @noRd
.chebyshev_layer_forward_torch <- function(h, par, spec) {
  coef <- par$coef
  k1 <- coef$size(3L)
  z <- torch::torch_tanh(h)
  basis <- vector("list", k1)
  basis[[1L]] <- torch::torch_ones_like(z)
  if (k1 >= 2L) basis[[2L]] <- z
  if (k1 >= 3L) for (k in 3:k1) basis[[k]] <- z * 2 * basis[[k - 1L]] - basis[[k - 2L]]
  b <- torch::torch_stack(basis, dim = 3L)
  torch::torch_einsum("bid,iod->bo", list(b, coef))
}

#' @noRd
.chebyshev_layer_edge_r <- function(x, par, spec, i, j) {
  z <- tanh(x)
  data.frame(z = z, phi = drop(chebyshev_basis(z, spec$degree) %*%
                                 par$coef[i, j, ]))
}

#' Independent implementation: explicit loops and T_k(z) = cos(k acos z)
#' @noRd
.chebyshev_layer_naive_forward <- function(h, par, spec) {
  d <- dim(par$coef)
  out <- matrix(0, nrow(h), d[2L])
  for (b in seq_len(nrow(h))) for (j in seq_len(d[2L])) {
    s <- 0
    for (i in seq_len(d[1L])) {
      z <- tanh(h[b, i])
      for (k in seq_len(d[3L])) s <- s + par$coef[i, j, k] * cos((k - 1) * acos(z))
    }
    out[b, j] <- s
  }
  out
}

#' @noRd
.chebyshev_layer_basis_checks <- function() {
  z <- c(-1, -0.999, -0.73, -0.5, -0.1, 0, 0.2, 0.5, 0.88, 0.999, 1)
  zi <- z[abs(z) < 1]
  tr <- outer(acos(z), 0:8, function(t, k) cos(k * t))
  dtr <- outer(acos(zi), 0:8, function(t, k) k * sin(k * t) / sin(t))
  gl <- .chebyshev_gauss_legendre_nodes(20)
  w <- function(x) 1 / sqrt(1 - x^2)
  gram <- outer(0:4, 0:4, Vectorize(function(m, n) stats::integrate(function(x)
    w(x) * chebyshev_basis(x, 4)[, m + 1] * chebyshev_basis(x, 4)[, n + 1], -1, 1,
    rel.tol = 1e-12)$value))
  list(
    list(check = "recurrence T_0..T_8 vs cos(k acos z)",
         error = max(abs(chebyshev_basis(z, 8) - tr)), tol = 1e-12),
    list(check = "derivative k U_{k-1} vs k sin(k t) / sin(t)",
         error = max(abs(chebyshev_basis_deriv(zi, 8) - dtr)), tol = 1e-9),
    list(check = "boundary T_k(1) = 1, T_k(-1) = (-1)^k, T'_k(1) = k^2",
         error = max(abs(c(chebyshev_basis(1, 8) - 1, chebyshev_basis(-1, 8) - (-1)^(0:8),
                           chebyshev_basis_deriv(1, 8) - (0:8)^2))), tol = 1e-10),
    list(check = "orthogonality with weight 1/sqrt(1 - x^2) (integrate)",
         error = max(abs(gram - diag(c(pi, rep(pi / 2, 4))))), tol = 1e-6)
  )
}

#' Gauss-Legendre nodes and weights (Golub-Welsch), base R
#' @noRd
.chebyshev_gauss_legendre_nodes <- function(n) {
  k <- seq_len(n - 1L)
  beta <- k / sqrt(4 * k^2 - 1)
  jm <- matrix(0, n, n)
  jm[cbind(k, k + 1L)] <- beta
  jm[cbind(k + 1L, k)] <- beta
  e <- eigen(jm, symmetric = TRUE)
  list(x = e$values, w = 2 * e$vectors[1L, ]^2)
}

#' @noRd
.chebyshev_validation_fixture <- function() {
  model <- thangld_kan(basis = "chebyshev", c(3, 4, 2, 1), degree = c(4, 3, 5), init_sd = 0.4,
                        seed = 11)
  x <- with_seed(12, matrix(stats::rnorm(18, sd = 1.5), 6, 3))
  y <- with_seed(13, stats::rnorm(6))
  list(model = model, x = x, y = y)
}


#' Layer specifications for all layers (seed handled by thangld_kan())
#' @noRd
.chebyshev_layer_specs <- function(widths, degree = 4L, init_sd = NULL) {
  widths <- check_widths(widths)
  nl <- length(widths) - 1L
  if (!is.numeric(degree) || !(length(degree) %in% c(1L, nl)) ||
      anyNA(degree) || any(degree != round(degree)) || any(degree < 0)) {
    stop(sprintf("`degree` must be one integer >= 0 or %d integers.", nl),
         call. = FALSE)
  }
  degree <- rep_len(as.integer(degree), nl)
  init_sd <- check_number(init_sd, "init_sd", min = 0, strict = TRUE,
                          allow_null = TRUE)
  specs <- lapply(degree, function(k) list(degree = k, init_sd = init_sd))
  specs
}