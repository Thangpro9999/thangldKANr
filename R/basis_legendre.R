# basis = "legendre" ----------------------------------------
# Legendre KAN layer --------------------------------------------------------------
#
# z = tanh(h); out[b, j] = sum_i sum_k coef[i, j, k] P_k(z[b, i])

#' Legendre polynomials and their derivatives
#'
#' `legendre_basis()` evaluates \eqn{P_0(z), \dots, P_K(z)} by Bonnet's
#' recurrence \eqn{P_0 = 1}, \eqn{P_1 = z},
#' \eqn{(k+1) P_{k+1}(z) = (2k+1) z P_k(z) - k P_{k-1}(z)}.
#' `legendre_basis_deriv()` returns \eqn{P'_k(z)} from
#' \eqn{P'_{k+1} = P'_{k-1} + (2k+1) P_k}, \eqn{P'_0 = 0}, \eqn{P'_1 = 1},
#' which has no singularity at \eqn{z = \pm 1}.
#'
#' Base R only; this is the numerical reference for the torch layer.
#' The polynomials are orthogonal on \eqn{[-1, 1]} with unit weight; that
#' does not make the features \eqn{P_k(\tanh(x))} orthogonal on a sample.
#'
#' @param z Numeric vector.
#' @param degree Maximum degree \eqn{K \ge 0}.
#' @return A `length(z)` by `degree + 1` matrix (column `k + 1` = degree `k`).
#' @examples
#' legendre_basis(c(-1, 0, 0.5, 1), 3)
#' legendre_basis_deriv(c(-1, 0, 1), 3)
#' @export
legendre_basis <- function(z, degree) {
  degree <- check_count(degree, "degree", min = 0L)
  z <- .legendre_check_z(z)
  out <- matrix(0, length(z), degree + 1L,
                dimnames = list(NULL, paste0("P", 0:degree)))
  out[, 1L] <- 1
  if (degree >= 1L) out[, 2L] <- z
  if (degree >= 2L) {
    for (k in 1:(degree - 1L)) {
      out[, k + 2L] <- ((2 * k + 1) * z * out[, k + 1L] - k * out[, k]) / (k + 1)
    }
  }
  out
}

#' @rdname legendre_basis
#' @export
legendre_basis_deriv <- function(z, degree) {
  degree <- check_count(degree, "degree", min = 0L)
  z <- .legendre_check_z(z)
  p <- legendre_basis(z, degree)
  out <- matrix(0, length(z), degree + 1L,
                dimnames = list(NULL, paste0("dP", 0:degree)))
  if (degree >= 1L) out[, 2L] <- 1
  if (degree >= 2L) {
    for (k in 1:(degree - 1L)) {
      out[, k + 2L] <- out[, k] + (2 * k + 1) * p[, k + 1L]
    }
  }
  out
}

#' @noRd
.legendre_check_z <- function(z) {
  if (!is.numeric(z) || anyNA(z) || any(!is.finite(z))) {
    stop("`z` must be a finite numeric vector.", call. = FALSE)
  }
  as.numeric(z)
}

#' Basis (or derivative) for an n x p matrix, as an n x p x (K+1) array
#' @noRd
.legendre_legendre_basis_array <- function(z, degree, deriv = FALSE) {
  n <- nrow(z)
  p <- ncol(z)
  b <- array(0, c(n, p, degree + 1L))
  b[, , 1L] <- 1
  if (degree >= 1L) b[, , 2L] <- z
  if (degree >= 2L) {
    for (k in 1:(degree - 1L)) {
      b[, , k + 2L] <- ((2 * k + 1) * z * b[, , k + 1L] - k * b[, , k]) / (k + 1)
    }
  }
  if (!deriv) return(b)
  d <- array(0, c(n, p, degree + 1L))
  if (degree >= 1L) d[, , 2L] <- 1
  if (degree >= 2L) {
    for (k in 1:(degree - 1L)) d[, , k + 2L] <- d[, , k] + (2 * k + 1) * b[, , k + 1L]
  }
  d
}

#' @noRd
.legendre_layer_describe <- function(spec) sprintf("Legendre K=%d", spec$degree)

#' @noRd
.legendre_layer_param_dims <- function(p, q, spec) list(coef = c(p, q, spec$degree + 1L))

#' @noRd
.legendre_layer_init <- function(p, q, spec) {
  k1 <- spec$degree + 1L
  sd <- if (is.null(spec$init_sd)) 1 / (p * k1) else spec$init_sd
  list(coef = array(stats::rnorm(p * q * k1, 0, sd), c(p, q, k1)))
}

#' @noRd
.legendre_layer_check_params <- function(par, spec) invisible(TRUE)

#' @noRd
.legendre_layer_forward_r <- function(h, par, spec) {
  n <- nrow(h)
  d <- dim(par$coef)
  p <- d[1L]
  q <- d[2L]
  k1 <- d[3L]
  z <- tanh(h)
  b <- matrix(.legendre_legendre_basis_array(z, k1 - 1L), n, p * k1)
  cmat <- matrix(aperm(par$coef, c(1L, 3L, 2L)), p * k1, q)
  list(out = b %*% cmat, cache = list(z = z, b = b, cmat = cmat))
}

#' @noRd
.legendre_layer_backward_r <- function(cache, g, par, spec) {
  d <- dim(par$coef)
  p <- d[1L]
  q <- d[2L]
  k1 <- d[3L]
  n <- nrow(g)
  gc <- crossprod(cache$b, g)
  grad <- aperm(array(gc, c(p, k1, q)), c(1L, 3L, 2L))
  db <- g %*% t(cache$cmat)
  dbdz <- matrix(.legendre_legendre_basis_array(cache$z, k1 - 1L, deriv = TRUE), n, p * k1)
  dz <- rowSums(array(db * dbdz, c(n, p, k1)), dims = 2L)
  list(grads = list(coef = grad), dh = dz * (1 - cache$z^2))
}

#' @noRd
.legendre_layer_forward_torch <- function(h, par, spec) {
  coef <- par$coef
  k1 <- coef$size(3L)
  z <- torch::torch_tanh(h)
  basis <- vector("list", k1)
  basis[[1L]] <- torch::torch_ones_like(z)
  if (k1 >= 2L) basis[[2L]] <- z
  if (k1 >= 3L) {
    for (k in 1:(k1 - 2L)) {
      basis[[k + 2L]] <- ((2 * k + 1) * z * basis[[k + 1L]] - k * basis[[k]]) /
        (k + 1)
    }
  }
  b <- torch::torch_stack(basis, dim = 3L)
  torch::torch_einsum("bid,iod->bo", list(b, coef))
}

#' @noRd
.legendre_layer_edge_r <- function(x, par, spec, i, j) {
  z <- tanh(x)
  data.frame(z = z, phi = drop(legendre_basis(z, spec$degree) %*%
                                 par$coef[i, j, ]))
}

#' Independent implementation: explicit loops and explicit low-degree
#' polynomials (degree <= 6) or the Rodrigues-free closed form sum
#' @noRd
.legendre_layer_naive_forward <- function(h, par, spec) {
  explicit <- list(function(z) 1, function(z) z, function(z) (3 * z^2 - 1) / 2,
                   function(z) (5 * z^3 - 3 * z) / 2,
                   function(z) (35 * z^4 - 30 * z^2 + 3) / 8,
                   function(z) (63 * z^5 - 70 * z^3 + 15 * z) / 8,
                   function(z) (231 * z^6 - 315 * z^4 + 105 * z^2 - 5) / 16)
  d <- dim(par$coef)
  if (d[3L] > length(explicit)) stop("naive check supports degree <= 6")
  out <- matrix(0, nrow(h), d[2L])
  for (b in seq_len(nrow(h))) for (j in seq_len(d[2L])) {
    s <- 0
    for (i in seq_len(d[1L])) {
      z <- tanh(h[b, i])
      for (k in seq_len(d[3L])) s <- s + par$coef[i, j, k] * explicit[[k]](z)
    }
    out[b, j] <- s
  }
  out
}

#' @noRd
.legendre_layer_basis_checks <- function() {
  z <- c(-1, -0.999, -0.73, -0.5, -0.1, 0, 0.2, 0.5, 0.88, 0.999, 1)
  ex <- cbind(1, z, (3 * z^2 - 1) / 2, (5 * z^3 - 3 * z) / 2,
              (35 * z^4 - 30 * z^2 + 3) / 8, (63 * z^5 - 70 * z^3 + 15 * z) / 8)
  dex <- cbind(0, 1, 3 * z, (15 * z^2 - 3) / 2, (140 * z^3 - 60 * z) / 8,
               (315 * z^4 - 210 * z^2 + 15) / 8)
  sgn <- matrix((-1)^(0:9), length(z), 10, byrow = TRUE)
  gl <- .legendre_gauss_legendre_nodes(20)
  pb <- legendre_basis(gl$x, 8)
  gram <- crossprod(pb * gl$w, pb)
  list(
    list(check = "P_0..P_5 vs explicit polynomials",
         error = max(abs(legendre_basis(z, 5) - ex)), tol = 1e-10),
    list(check = "derivatives P'_0..P'_5 vs explicit",
         error = max(abs(legendre_basis_deriv(z, 5) - dex)), tol = 1e-10),
    list(check = "symmetry P_k(-z) = (-1)^k P_k(z)",
         error = max(abs(legendre_basis(-z, 9) - sgn * legendre_basis(z, 9))),
         tol = 1e-10),
    list(check = "boundary P_k(1) = 1, P_k(-1) = (-1)^k, P'_k(1) = k(k+1)/2",
         error = max(abs(c(legendre_basis(1, 9) - 1,
                           legendre_basis(-1, 9) - (-1)^(0:9),
                           legendre_basis_deriv(1, 9) - (0:9) * (1:10) / 2))),
         tol = 1e-10),
    list(check = "orthogonality: int P_m P_n = 2/(2n+1) delta (Gauss-Legendre 20)",
         error = max(abs(gram - diag(2 / (2 * (0:8) + 1)))), tol = 1e-10)
  )
}

#' Gauss-Legendre nodes and weights (Golub-Welsch), base R
#' @noRd
.legendre_gauss_legendre_nodes <- function(n) {
  k <- seq_len(n - 1L)
  beta <- k / sqrt(4 * k^2 - 1)
  jm <- matrix(0, n, n)
  jm[cbind(k, k + 1L)] <- beta
  jm[cbind(k + 1L, k)] <- beta
  e <- eigen(jm, symmetric = TRUE)
  list(x = e$values, w = 2 * e$vectors[1L, ]^2)
}

#' @noRd
.legendre_validation_fixture <- function() {
  model <- thangld_kan(basis = "legendre", c(3, 4, 2, 1), degree = c(4, 3, 5), init_sd = 0.4,
                        seed = 11)
  x <- with_seed(12, matrix(stats::rnorm(18, sd = 1.5), 6, 3))
  y <- with_seed(13, stats::rnorm(6))
  list(model = model, x = x, y = y)
}


#' Layer specifications for all layers (seed handled by thangld_kan())
#' @noRd
.legendre_layer_specs <- function(widths, degree = 4L, init_sd = NULL) {
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