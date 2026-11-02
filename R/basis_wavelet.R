# basis = "wavelet" ----------------------------------------
# Wavelet (Mexican hat) KAN layer -------------------------------------------------
#
# u[b, i, j] = (h[b, i] - t[i, j]) / s[i, j],  s = exp(log_scale) (or raw)
# out[b, j]  = sum_i w[i, j] psi(u[b, i, j]) + bias[j]
# psi(u)     = C (1 - u^2) exp(-u^2 / 2),  C = 2 / (sqrt(3) pi^(1/4))

#' Mexican hat (Ricker) wavelet
#'
#' `mexican_hat()` evaluates \eqn{\psi(u) = C (1 - u^2) e^{-u^2/2}} with
#' \eqn{C = 2/(\sqrt{3}\pi^{1/4})}, so that \eqn{\int \psi^2 = 1} and
#' \eqn{\int \psi = 0}; `mexican_hat_deriv()` returns
#' \eqn{\psi'(u) = C e^{-u^2/2} (u^3 - 3u)}.
#'
#' On an edge the wavelet is used as \eqn{w\,\psi((x - t)/s)} without the
#' \eqn{1/\sqrt{s}} factor of the continuous wavelet transform (the same
#' convention as Wav-KAN); the learned weight \eqn{w} absorbs the amplitude.
#' Wav-KAN codes the wavelet with the opposite sign,
#' \eqn{C (u^2 - 1) e^{-u^2/2}}, which corresponds to \eqn{-w}.
#'
#' @param u Numeric vector.
#' @return Numeric vector.
#' @examples
#' mexican_hat(c(-1, 0, 1))
#' stats::integrate(function(u) mexican_hat(u)^2, -Inf, Inf)$value
#' @export
mexican_hat <- function(u) {
  u <- .wavelet_check_vec(u, "u")
  .wavelet_mh_const() * (1 - u^2) * exp(-u^2 / 2)
}

#' @rdname mexican_hat
#' @export
mexican_hat_deriv <- function(u) {
  u <- .wavelet_check_vec(u, "u")
  .wavelet_mh_const() * exp(-u^2 / 2) * (u^3 - 3 * u)
}

#' @noRd
.wavelet_mh_const <- function() 2 / (sqrt(3) * pi^0.25)

#' @noRd
.wavelet_check_vec <- function(z, name) {
  if (!is.numeric(z) || anyNA(z) || any(!is.finite(z))) {
    stop(sprintf("`%s` must be a finite numeric vector.", name), call. = FALSE)
  }
  as.numeric(z)
}

#' @noRd
.wavelet_layer_describe <- function(spec) {
  sprintf("Mexican hat (%s scale%s)", spec$scale_param,
          if (spec$bias) ", bias" else "")
}

#' @noRd
.wavelet_scale_name <- function(spec) if (spec$scale_param == "log") "log_scale" else "scale"

#' @noRd
.wavelet_layer_param_dims <- function(p, q, spec) {
  out <- list(weight = c(p, q), translation = c(p, q))
  out[[.wavelet_scale_name(spec)]] <- c(p, q)
  if (spec$bias) out$bias <- q
  out
}

#' Initialization as in Wav-KAN: scale 1, translation 0, weights
#' U(-1/sqrt(in), 1/sqrt(in)); bias 0.
#' @noRd
.wavelet_layer_init <- function(p, q, spec) {
  bound <- 1 / sqrt(p)
  out <- list(weight = array(stats::runif(p * q, -bound, bound), c(p, q)),
              translation = array(0, c(p, q)))
  out[[.wavelet_scale_name(spec)]] <- array(if (spec$scale_param == "log") 0 else 1,
                                   c(p, q))
  if (spec$bias) out$bias <- array(0, q)
  out
}

#' @noRd
.wavelet_get_scale <- function(par, spec) {
  if (spec$scale_param == "log") exp(par$log_scale) else par$scale
}

#' @noRd
.wavelet_layer_check_params <- function(par, spec) {
  if (spec$scale_param == "raw" && any(par$scale <= 0)) {
    stop("A wavelet scale became non-positive during training; use ",
         "scale_param = \"log\" (default) or a smaller `lr`.", call. = FALSE)
  }
  invisible(TRUE)
}

#' @noRd
.wavelet_layer_forward_r <- function(h, par, spec) {
  n <- nrow(h)
  p <- ncol(h)
  q <- ncol(par$weight)
  s <- .wavelet_get_scale(par, spec)
  hx <- array(h, c(n, p, q))
  u <- (hx - rep(par$translation, each = n)) / rep(s, each = n)
  e <- exp(-u^2 / 2)
  psi <- .wavelet_mh_const() * (1 - u^2) * e
  out <- rowSums(aperm(psi * rep(par$weight, each = n), c(1L, 3L, 2L)),
                 dims = 2L)
  out <- matrix(out, n, q)
  if (spec$bias) out <- sweep(out, 2L, as.numeric(par$bias), "+")
  list(out = out, cache = list(u = u, e = e, psi = psi, s = s))
}

#' @noRd
.wavelet_layer_backward_r <- function(cache, g, par, spec) {
  n <- nrow(g)
  p <- dim(cache$u)[2L]
  q <- dim(cache$u)[3L]
  ga <- aperm(array(g, c(n, q, p)), c(1L, 3L, 2L))
  dpsi <- .wavelet_mh_const() * cache$e * (cache$u^3 - 3 * cache$u)
  a <- ga * rep(par$weight, each = n) * dpsi
  srep <- rep(cache$s, each = n)
  grads <- list(weight = colSums(ga * cache$psi, dims = 1L),
                translation = colSums(-a / srep, dims = 1L))
  if (spec$scale_param == "log") {
    grads$log_scale <- colSums(-a * cache$u, dims = 1L)
  } else {
    grads$scale <- colSums(-a * cache$u / srep, dims = 1L)
  }
  grads <- lapply(grads, function(m) matrix(m, p, q))
  if (spec$bias) grads$bias <- array(colSums(g), q)
  dh <- rowSums(a / srep, dims = 2L)
  list(grads = grads, dh = matrix(dh, n, p))
}

#' @noRd
.wavelet_layer_forward_torch <- function(h, par, spec) {
  s <- if (spec$scale_param == "log") torch::torch_exp(par$log_scale) else
    par$scale
  u <- (h$unsqueeze(3L) - par$translation) / s
  psi <- (1 - u^2) * torch::torch_exp(-u^2 / 2) * .wavelet_mh_const()  # tensor first: keeps the R double in float64
  out <- (psi * par$weight)$sum(dim = 2L)
  if (spec$bias) out <- out + par$bias
  out
}

#' @noRd
.wavelet_layer_edge_r <- function(x, par, spec, i, j) {
  s <- .wavelet_get_scale(par, spec)[i, j]
  u <- (x - par$translation[i, j]) / s
  data.frame(u = u, phi = par$weight[i, j] * mexican_hat(u))
}

#' @noRd
.wavelet_layer_naive_forward <- function(h, par, spec) {
  s <- .wavelet_get_scale(par, spec)
  q <- ncol(par$weight)
  out <- matrix(0, nrow(h), q)
  for (b in seq_len(nrow(h))) for (j in seq_len(q)) {
    v <- 0
    for (i in seq_len(ncol(h))) {
      u <- (h[b, i] - par$translation[i, j]) / s[i, j]
      v <- v + par$weight[i, j] * 2 / (sqrt(3) * pi^0.25) * (1 - u^2) *
        exp(-u^2 / 2)
    }
    out[b, j] <- v + if (spec$bias) par$bias[j] else 0
  }
  out
}

#' @noRd
.wavelet_layer_basis_checks <- function() {
  u <- c(-4, -1.7, -1, -0.3, 0, 0.5, 1, 2.2, 5)
  h <- 1e-6
  fx <- .wavelet_validation_fixture()
  g <- thangld_kan_gradient_r(fx$model, fx$x, fx$y)
  fd <- thangld_kan_fd_gradient_r(fx$model, fx$x, fx$y, step = 1e-5, order = 4)
  per <- lapply(c("weight", "translation", "log_scale", "bias"), function(nm) {
    a <- unlist(lapply(g$params, `[[`, nm))
    b <- unlist(lapply(fd$params, `[[`, nm))
    list(check = sprintf("gradient wrt %s vs central difference O4 (h=1e-5)", nm),
         error = scaled_error(a, b), tol = 1e-5)
  })
  c(list(
    list(check = "psi vs explicit C(1-u^2)exp(-u^2/2)",
         error = max(abs(mexican_hat(u) -
                           2 / (sqrt(3) * pi^0.25) * (1 - u^2) * exp(-u^2 / 2))),
         tol = 1e-10),
    list(check = "psi(+-1) = 0, psi(0) = C, symmetry",
         error = max(abs(c(mexican_hat(c(-1, 1)), mexican_hat(0) - .wavelet_mh_const(),
                           mexican_hat(u) - mexican_hat(-u)))), tol = 1e-10),
    list(check = "int psi = 0 and int psi^2 = 1 (stats::integrate)",
         error = max(abs(c(
           stats::integrate(mexican_hat, -Inf, Inf, rel.tol = 1e-12)$value,
           stats::integrate(function(v) mexican_hat(v)^2, -Inf, Inf,
                            rel.tol = 1e-12)$value - 1))), tol = 1e-8),
    list(check = "psi' vs central difference",
         error = max(abs(mexican_hat_deriv(u) -
                           (mexican_hat(u + h) - mexican_hat(u - h)) / (2 * h))),
         tol = 1e-6)
  ), per)
}

#' Fixture for thangld_kan_validate(): 3 -> 4 -> 2 -> 1 with spread
#' translations and scales
#' @noRd
.wavelet_validation_fixture <- function(scale_param = "log", bias = TRUE) {
  model <- thangld_kan(basis = "wavelet", c(3, 4, 2, 1), scale_param = scale_param, bias = bias,
                    seed = 11)
  model$params <- with_seed(21, lapply(model$params, function(pl) {
    pl$translation[] <- stats::runif(length(pl$translation), -1.5, 1.5)
    if (!is.null(pl$log_scale)) pl$log_scale[] <- stats::runif(length(pl$log_scale), -0.5, 0.7)
    if (!is.null(pl$scale)) pl$scale[] <- stats::runif(length(pl$scale), 0.6, 2)
    if (!is.null(pl$bias)) pl$bias[] <- stats::rnorm(length(pl$bias), sd = 0.3)
    pl
  }))
  x <- with_seed(12, matrix(stats::rnorm(18, sd = 1.5), 6, 3))
  y <- with_seed(13, stats::rnorm(6))
  list(model = model, x = x, y = y)
}


#' Layer specifications for all layers (seed handled by thangld_kan())
#' @noRd
.wavelet_layer_specs <- function(widths, scale_param = c("log", "raw"), bias = TRUE) {
  widths <- check_widths(widths)
  scale_param <- match.arg(scale_param)
  bias <- check_flag(bias, "bias")
  specs <- lapply(seq_len(length(widths) - 1L), function(l) {
    list(wavelet = "mexican_hat", scale_param = scale_param, bias = bias)
  })
  specs
}