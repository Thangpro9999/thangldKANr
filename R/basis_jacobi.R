# basis = "jacobi" ----------------------------------------
# Jacobi KAN layer -----------------------------------------------------------------
#
# z = tanh(h); out[b, j] = sum_i sum_k coef[i, j, k] P_k^(alpha, beta)(z[b, i])

#' Jacobi polynomials and their derivatives
#'
#' `jacobi_basis()` evaluates \eqn{P_0^{(\alpha,\beta)}(z), \dots,
#' P_K^{(\alpha,\beta)}(z)} by the three-term recurrence
#' \deqn{2n(n+\alpha+\beta)(2n+\alpha+\beta-2) P_n = (2n+\alpha+\beta-1)
#'   \{(2n+\alpha+\beta)(2n+\alpha+\beta-2) z + \alpha^2-\beta^2\} P_{n-1}
#'   - 2(n+\alpha-1)(n+\beta-1)(2n+\alpha+\beta) P_{n-2},}
#' with \eqn{P_0 = 1} and \eqn{P_1 = \{(\alpha-\beta) + (\alpha+\beta+2) z\}/2}
#' (standard normalization, \eqn{P_n(1) = \binom{n+\alpha}{n}}).
#' `jacobi_basis_deriv()` uses
#' \eqn{\frac{d}{dz} P_n^{(\alpha,\beta)} = \frac{n+\alpha+\beta+1}{2}
#' P_{n-1}^{(\alpha+1,\beta+1)}}. Special cases: \eqn{\alpha = \beta = 0}
#' gives Legendre polynomials; \eqn{\alpha = \beta = -1/2} gives Chebyshev
#' polynomials up to a constant factor per degree.
#'
#' @param z Numeric vector.
#' @param degree Maximum degree \eqn{K \ge 0}.
#' @param alpha,beta Jacobi parameters, both \eqn{> -1}.
#' @return A `length(z)` by `degree + 1` matrix.
#' @examples
#' jacobi_basis(c(-1, 0, 1), 3, alpha = 1, beta = 1)
#' @export
jacobi_basis <- function(z, degree, alpha = 1, beta = 1) {
  degree <- check_count(degree, "degree", min = 0L)
  .jacobi_check_ab(alpha, beta)
  z <- .jacobi_check_z(z)
  out <- matrix(.jacobi_jacobi_array(matrix(z, ncol = 1L), degree, alpha, beta), length(z))
  colnames(out) <- paste0("P", 0:degree)
  out
}

#' @rdname jacobi_basis
#' @export
jacobi_basis_deriv <- function(z, degree, alpha = 1, beta = 1) {
  degree <- check_count(degree, "degree", min = 0L)
  .jacobi_check_ab(alpha, beta)
  z <- .jacobi_check_z(z)
  out <- matrix(.jacobi_jacobi_array(matrix(z, ncol = 1L), degree, alpha, beta, deriv = TRUE),
                length(z))
  colnames(out) <- paste0("dP", 0:degree)
  out
}

#' @noRd
.jacobi_check_ab <- function(alpha, beta) {
  for (v in list(alpha, beta)) {
    if (!is.numeric(v) || length(v) != 1L || !is.finite(v) || v <= -1) {
      stop("`alpha` and `beta` must be single numbers > -1.", call. = FALSE)
    }
  }
  invisible(TRUE)
}

#' @noRd
.jacobi_check_z <- function(z) {
  if (!is.numeric(z) || anyNA(z) || any(!is.finite(z))) {
    stop("`z` must be a finite numeric vector.", call. = FALSE)
  }
  as.numeric(z)
}

#' Jacobi basis (or derivative) for an n x p matrix: n x p x (K+1)
#' @noRd
.jacobi_jacobi_array <- function(z, degree, a, b, deriv = FALSE) {
  if (deriv) {
    d <- array(0, c(nrow(z), ncol(z), degree + 1L))
    if (degree >= 1L) {
      s <- .jacobi_jacobi_array(z, degree - 1L, a + 1, b + 1)
      for (n in 1:degree) d[, , n + 1L] <- (n + a + b + 1) / 2 * s[, , n]
    }
    return(d)
  }
  p <- array(0, c(nrow(z), ncol(z), degree + 1L))
  p[, , 1L] <- 1
  if (degree >= 1L) p[, , 2L] <- ((a - b) + (a + b + 2) * z) / 2
  if (degree >= 2L) {
    for (n in 2:degree) {
      c0 <- 2 * n * (n + a + b) * (2 * n + a + b - 2)
      c1 <- (2 * n + a + b - 1) * (2 * n + a + b) * (2 * n + a + b - 2)
      c2 <- (2 * n + a + b - 1) * (a^2 - b^2)
      c3 <- 2 * (n + a - 1) * (n + b - 1) * (2 * n + a + b)
      p[, , n + 1L] <- ((c1 * z + c2) * p[, , n] - c3 * p[, , n - 1L]) / c0
    }
  }
  p
}

#' @noRd
.jacobi_layer_describe <- function(spec) {
  sprintf("Jacobi K=%d (alpha=%g, beta=%g)", spec$degree, spec$alpha, spec$beta)
}

#' @noRd
.jacobi_layer_param_dims <- function(p, q, spec) list(coef = c(p, q, spec$degree + 1L))

#' @noRd
.jacobi_layer_init <- function(p, q, spec) {
  k1 <- spec$degree + 1L
  sd <- if (is.null(spec$init_sd)) 1 / (p * k1) else spec$init_sd
  list(coef = array(stats::rnorm(p * q * k1, 0, sd), c(p, q, k1)))
}

#' @noRd
.jacobi_layer_check_params <- function(par, spec) invisible(TRUE)

#' @noRd
.jacobi_layer_forward_r <- function(h, par, spec) {
  n <- nrow(h)
  d <- dim(par$coef)
  z <- tanh(h)
  b <- matrix(.jacobi_jacobi_array(z, d[3L] - 1L, spec$alpha, spec$beta), n, d[1L] * d[3L])
  cmat <- matrix(aperm(par$coef, c(1L, 3L, 2L)), d[1L] * d[3L], d[2L])
  list(out = b %*% cmat, cache = list(z = z, b = b, cmat = cmat))
}

#' @noRd
.jacobi_layer_backward_r <- function(cache, g, par, spec) {
  d <- dim(par$coef)
  n <- nrow(g)
  grad <- aperm(array(crossprod(cache$b, g), c(d[1L], d[3L], d[2L])), c(1L, 3L, 2L))
  db <- g %*% t(cache$cmat)
  dbdz <- matrix(.jacobi_jacobi_array(cache$z, d[3L] - 1L, spec$alpha, spec$beta, deriv = TRUE),
                 n, d[1L] * d[3L])
  dz <- rowSums(array(db * dbdz, c(n, d[1L], d[3L])), dims = 2L)
  list(grads = list(coef = grad), dh = dz * (1 - cache$z^2))
}

#' @noRd
.jacobi_layer_forward_torch <- function(h, par, spec) {
  coef <- par$coef
  k1 <- coef$size(3L)
  a <- spec$alpha
  b <- spec$beta
  z <- torch::torch_tanh(h)
  basis <- vector("list", k1)
  basis[[1L]] <- torch::torch_ones_like(z)
  if (k1 >= 2L) basis[[2L]] <- (z * (a + b + 2) + (a - b)) / 2
  if (k1 >= 3L) {
    for (n in 2:(k1 - 1L)) {
      c0 <- 2 * n * (n + a + b) * (2 * n + a + b - 2)
      c1 <- (2 * n + a + b - 1) * (2 * n + a + b) * (2 * n + a + b - 2)
      c2 <- (2 * n + a + b - 1) * (a^2 - b^2)
      c3 <- 2 * (n + a - 1) * (n + b - 1) * (2 * n + a + b)
      # tensor first: `scalar * tensor` rounds the scalar to float32 in R torch
      basis[[n + 1L]] <- ((z * c1 + c2) * basis[[n]] - basis[[n - 1L]] * c3) / c0
    }
  }
  torch::torch_einsum("bid,iod->bo", list(torch::torch_stack(basis, dim = 3L), coef))
}

#' @noRd
.jacobi_layer_edge_r <- function(x, par, spec, i, j) {
  z <- tanh(x)
  data.frame(z = z, phi = drop(jacobi_basis(z, spec$degree, spec$alpha, spec$beta) %*%
                                 par$coef[i, j, ]))
}

#' Explicit sum formula
#' P_n(z) = sum_s choose(n+a, n-s) choose(n+b, s) ((z-1)/2)^s ((z+1)/2)^(n-s)
#' @noRd
.jacobi_jacobi_explicit <- function(z, n, a, b) {
  gch <- function(x, k) exp(lgamma(x + 1) - lgamma(k + 1) - lgamma(x - k + 1))
  s <- 0
  for (m in 0:n) s <- s + gch(n + a, n - m) * gch(n + b, m) * ((z - 1) / 2)^m * ((z + 1) / 2)^(n - m)
  s
}

#' Independent implementation: explicit loops and the explicit sum formula
#' @noRd
.jacobi_layer_naive_forward <- function(h, par, spec) {
  d <- dim(par$coef)
  out <- matrix(0, nrow(h), d[2L])
  for (b in seq_len(nrow(h))) for (j in seq_len(d[2L])) {
    s <- 0
    for (i in seq_len(d[1L])) {
      z <- tanh(h[b, i])
      for (k in seq_len(d[3L])) {
        s <- s + par$coef[i, j, k] * .jacobi_jacobi_explicit(z, k - 1L, spec$alpha, spec$beta)
      }
    }
    out[b, j] <- s
  }
  out
}

#' @noRd
.jacobi_layer_basis_checks <- function() {
  z <- c(-1, -0.999, -0.73, -0.5, -0.1, 0, 0.2, 0.5, 0.88, 0.999, 1)
  ab <- list(c(1, 1), c(0.5, -0.3), c(2, 0), c(-0.5, -0.5))
  ex <- max(vapply(ab, function(v) max(abs(jacobi_basis(z, 6, v[1], v[2]) -
    sapply(0:6, function(n) .jacobi_jacobi_explicit(z, n, v[1], v[2])))), numeric(1)))
  h <- 1e-6
  zz <- z[abs(z) < 0.99]
  fd <- max(vapply(ab, function(v) max(abs(jacobi_basis_deriv(zz, 6, v[1], v[2]) -
    (jacobi_basis(zz + h, 6, v[1], v[2]) - jacobi_basis(zz - h, 6, v[1], v[2])) / (2 * h))),
    numeric(1)))
  leg <- cbind(1, z, (3 * z^2 - 1) / 2, (5 * z^3 - 3 * z) / 2, (35 * z^4 - 30 * z^2 + 3) / 8)
  tn <- outer(acos(z), 0:5, function(t, k) cos(k * t))
  cn <- sapply(0:5, function(n) exp(lgamma(2 * n + 1) - 2 * n * log(2) - 2 * lgamma(n + 1)))
  # orthogonality with weight (1 - x)^a (1 + x)^b, a = 1, b = 0.5
  w <- function(x) (1 - x)^1 * (1 + x)^0.5
  gram <- outer(0:3, 0:3, Vectorize(function(m, n) stats::integrate(function(x)
    w(x) * jacobi_basis(x, 3, 1, 0.5)[, m + 1] * jacobi_basis(x, 3, 1, 0.5)[, n + 1],
    -1, 1, rel.tol = 1e-12)$value))
  list(
    list(check = "recurrence vs explicit sum formula (4 parameter pairs, K = 6)",
         error = ex, tol = 1e-10),
    list(check = "derivative formula vs central difference (h = 1e-6)", error = fd, tol = 1e-6),
    list(check = "alpha = beta = 0 equals Legendre P_0..P_4",
         error = max(abs(jacobi_basis(z, 4, 0, 0) - leg)), tol = 1e-12),
    list(check = "alpha = beta = -1/2 equals c_n T_n (Chebyshev)",
         error = max(abs(jacobi_basis(z, 5, -0.5, -0.5) - sweep(tn, 2, cn, `*`))), tol = 1e-12),
    list(check = "P_n(1) = choose(n + alpha, n)",
         error = max(abs(jacobi_basis(1, 6, 1.5, 0.2) - choose(1.5 + 0:6, 0:6))), tol = 1e-10),
    list(check = "orthogonality with weight (1-x)^alpha (1+x)^beta (integrate)",
         error = max(abs(gram[upper.tri(gram)])), tol = 1e-8)
  )
}

#' @noRd
.jacobi_validation_fixture <- function(...) {
  model <- thangld_kan(basis = "jacobi", c(3, 4, 2, 1), degree = c(4, 3, 5), alpha = 1, beta = 1,
                      init_sd = 0.2, seed = 11, ...)
  x <- with_seed(12, matrix(stats::rnorm(18, sd = 1.5), 6, 3))
  y <- with_seed(13, stats::rnorm(6))
  list(model = model, x = x, y = y)
}


#' Layer specifications for all layers (seed handled by thangld_kan())
#' @noRd
.jacobi_layer_specs <- function(widths, degree = 4L, alpha = 1, beta = 1, init_sd = NULL) {
  widths <- check_widths(widths)
  nl <- length(widths) - 1L
  if (!is.numeric(degree) || !(length(degree) %in% c(1L, nl)) ||
      anyNA(degree) || any(degree != round(degree)) || any(degree < 0)) {
    stop(sprintf("`degree` must be one integer >= 0 or %d integers.", nl), call. = FALSE)
  }
  degree <- rep_len(as.integer(degree), nl)
  .jacobi_check_ab(alpha, beta)
  init_sd <- check_number(init_sd, "init_sd", min = 0, strict = TRUE, allow_null = TRUE)
  specs <- lapply(degree, function(k) list(degree = k, alpha = as.numeric(alpha),
                                           beta = as.numeric(beta), init_sd = init_sd))
  specs
}