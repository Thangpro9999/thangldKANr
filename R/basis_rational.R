# basis = "rational" ----------------------------------------
# Rational KAN layer ---------------------------------------------------------------
#
# out[b, j] = sum_i P_ij(h[b, i]) / (1 + |Q_ij(h[b, i])|)
# P_ij(x) = sum_{k=0}^{m} num[i, j, k + 1] x^k
# Q_ij(x) = sum_{k=1}^{n} den[i, j, k] x^k

#' Safe Pade rational function
#'
#' Evaluates the "safe" Pade form used by rational activation functions
#' (Molina, Schramowski and Kersting, 2020, version B) and by the group
#' rational KAN of Yang and Wang (2025):
#' \deqn{R(x) = \frac{a_0 + a_1 x + \dots + a_m x^m}{1 + |b_1 x + \dots + b_n x^n|}.}
#' The denominator is at least 1, so \eqn{R} has no poles. For
#' \eqn{m = n + 1} the function grows linearly in \eqn{|x|}.
#' `rational_deriv()` returns \eqn{R'(x)} (with \eqn{d|u|/du = \mathrm{sign}(u)},
#' 0 at \eqn{u = 0}, as in torch).
#'
#' @param x Numeric vector.
#' @param num Numerator coefficients \eqn{a_0, \dots, a_m}.
#' @param den Denominator coefficients \eqn{b_1, \dots, b_n} (`n <= m`).
#' @return Numeric vector of `length(x)`.
#' @examples
#' rational(c(-2, 0, 2), num = c(0, 1, 0, 0.5), den = c(0, 0.5))
#' rational_deriv(c(-2, 0, 2), num = c(0, 1, 0, 0.5), den = c(0, 0.5))
#' @export
rational <- function(x, num, den) {
  .rational_check_coefs(num, den)
  x <- .rational_check_z(x)
  v <- outer(x, seq_along(num) - 1L, `^`)
  p <- drop(v %*% num)
  q <- drop(v[, 1L + seq_along(den), drop = FALSE] %*% den)
  p / (1 + abs(q))
}

#' @rdname rational
#' @export
rational_deriv <- function(x, num, den) {
  .rational_check_coefs(num, den)
  x <- .rational_check_z(x)
  m <- length(num) - 1L
  v <- outer(x, 0:m, `^`)
  dv <- cbind(0, sweep(v[, seq_len(m), drop = FALSE], 2L, seq_len(m), `*`))
  p <- drop(v %*% num)
  dp <- drop(dv %*% num)
  idx <- 1L + seq_along(den)
  q <- drop(v[, idx, drop = FALSE] %*% den)
  dq <- drop(dv[, idx, drop = FALSE] %*% den)
  d <- 1 + abs(q)
  dp / d - p * sign(q) * dq / d^2
}

#' @noRd
.rational_check_coefs <- function(num, den) {
  if (!is.numeric(num) || length(num) < 1L || anyNA(num) || any(!is.finite(num)) ||
      !is.numeric(den) || anyNA(den) || any(!is.finite(den)) ||
      length(den) > length(num) - 1L) {
    stop("`num` (length m + 1) and `den` (length n <= m) must be finite numeric vectors.",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' @noRd
.rational_check_z <- function(z) {
  if (!is.numeric(z) || anyNA(z) || any(!is.finite(z))) {
    stop("`x` must be a finite numeric vector.", call. = FALSE)
  }
  as.numeric(z)
}

#' Powers h^0..h^m as an n x p x (m+1) array
#' @noRd
.rational_power_array <- function(h, m) {
  v <- array(1, c(nrow(h), ncol(h), m + 1L))
  if (m >= 1L) for (k in 1:m) v[, , k + 1L] <- v[, , k] * h
  v
}

#' @noRd
.rational_layer_describe <- function(spec) sprintf("rational [%d/%d]", spec$num_degree, spec$den_degree)

#' @noRd
.rational_layer_param_dims <- function(p, q, spec) {
  list(num = c(p, q, spec$num_degree + 1L), den = c(p, q, spec$den_degree))
}

#' @noRd
.rational_layer_init <- function(p, q, spec) {
  m1 <- spec$num_degree + 1L
  sd <- if (is.null(spec$init_sd)) 1 / (p * m1) else spec$init_sd
  list(num = array(stats::rnorm(p * q * m1, 0, sd), c(p, q, m1)),
       den = array(stats::rnorm(p * q * spec$den_degree, 0, spec$den_init_sd),
                   c(p, q, spec$den_degree)))
}

#' @noRd
.rational_layer_check_params <- function(par, spec) invisible(TRUE)

#' Per-edge P, Q and their derivatives: n x p x q arrays
#' @noRd
.rational_rational_parts <- function(h, par, deriv = FALSE) {
  n <- nrow(h)
  dn <- dim(par$num)
  p <- dn[1L]
  q <- dn[2L]
  m <- dn[3L] - 1L
  nd <- dim(par$den)[3L]
  v <- .rational_power_array(h, m)
  P <- Q <- array(0, c(n, p, q))
  dP <- dQ <- if (deriv) array(0, c(n, p, q)) else NULL
  for (i in seq_len(p)) {
    vi <- matrix(v[, i, ], n, m + 1L)
    P[, i, ] <- vi %*% t(matrix(par$num[i, , ], q, m + 1L))
    if (nd > 0L) {
      Q[, i, ] <- vi[, 1L + seq_len(nd), drop = FALSE] %*%
        t(matrix(par$den[i, , ], q, nd))
    }
    if (deriv) {
      dvi <- cbind(0, sweep(vi[, seq_len(m), drop = FALSE], 2L, seq_len(m), `*`))
      dP[, i, ] <- dvi %*% t(matrix(par$num[i, , ], q, m + 1L))
      if (nd > 0L) {
        dQ[, i, ] <- dvi[, 1L + seq_len(nd), drop = FALSE] %*%
          t(matrix(par$den[i, , ], q, nd))
      }
    }
  }
  list(v = v, P = P, Q = Q, dP = dP, dQ = dQ)
}

#' @noRd
.rational_layer_forward_r <- function(h, par, spec) {
  r <- .rational_rational_parts(h, par, deriv = TRUE)
  D <- 1 + abs(r$Q)
  out <- colSums(aperm(r$P / D, c(2L, 1L, 3L)))
  if (is.null(dim(out))) out <- matrix(out, nrow(h))
  list(out = out, cache = c(r, list(D = D)))
}

#' @noRd
.rational_layer_backward_r <- function(cache, g, par, spec) {
  dn <- dim(par$num)
  p <- dn[1L]
  q <- dn[2L]
  m1 <- dn[3L]
  nd <- dim(par$den)[3L]
  n <- nrow(g)
  D <- cache$D
  s <- sign(cache$Q)
  gnum <- array(0, dn)
  gden <- array(0, dim(par$den))
  dh <- matrix(0, n, p)
  for (i in seq_len(p)) {
    vi <- matrix(cache$v[, i, ], n, m1)
    Di <- matrix(D[, i, ], n, q)
    Pi <- matrix(cache$P[, i, ], n, q)
    si <- matrix(s[, i, ], n, q)
    gnum[i, , ] <- t(crossprod(vi, g / Di))
    wq <- -g * Pi * si / Di^2
    if (nd > 0L) gden[i, , ] <- t(crossprod(vi[, 1L + seq_len(nd), drop = FALSE], wq))
    dphi <- matrix(cache$dP[, i, ], n, q) / Di -
      Pi * si * matrix(cache$dQ[, i, ], n, q) / Di^2
    dh[, i] <- rowSums(g * dphi)
  }
  list(grads = list(num = gnum, den = gden), dh = dh)
}

#' @noRd
.rational_layer_forward_torch <- function(h, par, spec) {
  num <- par$num
  den <- par$den
  m1 <- num$size(3L)
  nd <- den$size(3L)
  pw <- vector("list", m1)
  pw[[1L]] <- torch::torch_ones_like(h)
  if (m1 >= 2L) for (k in 2:m1) pw[[k]] <- pw[[k - 1L]] * h
  v <- torch::torch_stack(pw, dim = 3L)
  P <- torch::torch_einsum("bik,ijk->bij", list(v, num))
  Q <- torch::torch_einsum("bik,ijk->bij", list(v[, , 2:(nd + 1L)], den))
  (P / (1 + torch::torch_abs(Q)))$sum(dim = 2L)
}

#' @noRd
.rational_layer_edge_r <- function(x, par, spec, i, j) {
  data.frame(z = x, phi = rational(x, par$num[i, j, ], par$den[i, j, ]))
}

#' Independent implementation: explicit loops, Horner evaluation
#' @noRd
.rational_layer_naive_forward <- function(h, par, spec) {
  dn <- dim(par$num)
  horner <- function(cf, x) {
    s <- 0
    for (k in rev(seq_along(cf))) s <- s * x + cf[k]
    s
  }
  out <- matrix(0, nrow(h), dn[2L])
  for (b in seq_len(nrow(h))) for (j in seq_len(dn[2L])) {
    s <- 0
    for (i in seq_len(dn[1L])) {
      x <- h[b, i]
      pn <- horner(par$num[i, j, ], x)
      qd <- x * horner(par$den[i, j, ], x)
      s <- s + pn / (1 + abs(qd))
    }
    out[b, j] <- s
  }
  out
}

#' @noRd
.rational_layer_basis_checks <- function() {
  x <- c(-3, -1.7, -0.4, 0.1, 0.9, 2.5)
  a <- c(0.3, -1.1, 0.7, 0.25)
  b <- c(0.5, -0.8)
  ex <- (0.3 - 1.1 * x + 0.7 * x^2 + 0.25 * x^3) / (1 + abs(0.5 * x - 0.8 * x^2))
  h <- 1e-5
  fd <- (rational(x + h, a, b) - rational(x - h, a, b)) / (2 * h)
  big <- c(1e3, 1e4)
  slope <- rational(big, a, b) / big
  list(
    list(check = "rational vs explicit [3/2] formula",
         error = max(abs(rational(x, a, b) - ex)), tol = 1e-10),
    list(check = "derivative vs central difference (h = 1e-5)",
         error = max(abs(rational_deriv(x, a, b) - fd)), tol = 1e-6),
    list(check = "denominator >= 1: no poles on a fine grid",
         error = as.numeric(any(!is.finite(rational(seq(-50, 50, by = 0.01), a, b)))),
         tol = 0),
    list(check = "[3/2] grows linearly: R(x)/x -> a_3/|b_2|",
         error = abs(slope[2L] - 0.25 / 0.8), tol = 1e-3)
  )
}

#' @noRd
.rational_validation_fixture <- function(...) {
  model <- thangld_kan(basis = "rational", c(3, 4, 2, 1), num_degree = c(3, 4, 2),
                        den_degree = c(2, 3, 1), init_sd = 0.2,
                        den_init_sd = 0.5, seed = 11, ...)
  x <- with_seed(12, matrix(stats::rnorm(18, sd = 1.5), 6, 3))
  y <- with_seed(13, stats::rnorm(6))
  list(model = model, x = x, y = y)
}


#' Layer specifications for all layers (seed handled by thangld_kan())
#' @noRd
.rational_layer_specs <- function(widths, num_degree = 3L, den_degree = 2L,
                         init_sd = NULL, den_init_sd = 0.1) {
  widths <- check_widths(widths)
  nl <- length(widths) - 1L
  chk <- function(v, nm, lo) {
    if (!is.numeric(v) || !(length(v) %in% c(1L, nl)) || anyNA(v) ||
        any(v != round(v)) || any(v < lo)) {
      stop(sprintf("`%s` must be one integer >= %d or %d integers.", nm, lo, nl),
           call. = FALSE)
    }
    rep_len(as.integer(v), nl)
  }
  num_degree <- chk(num_degree, "num_degree", 1L)
  den_degree <- chk(den_degree, "den_degree", 1L)
  if (any(den_degree > num_degree)) {
    stop("`den_degree` must not exceed `num_degree`.", call. = FALSE)
  }
  init_sd <- check_number(init_sd, "init_sd", min = 0, strict = TRUE,
                          allow_null = TRUE)
  den_init_sd <- check_number(den_init_sd, "den_init_sd", min = 0, strict = TRUE)
  specs <- lapply(seq_len(nl), function(l) list(
    num_degree = num_degree[l], den_degree = den_degree[l], init_sd = init_sd,
    den_init_sd = den_init_sd))
  specs
}