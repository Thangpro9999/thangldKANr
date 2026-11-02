# Numerical validation suite ------------------------------------------------------

#' Run the numerical validation suite
#'
#' Runs the acceptance checks of thangldKANr on small fixtures in float64, for
#' each requested basis, and returns one row per check: basis-specific
#' checks, the forward pass against an independent loop implementation,
#' analytic gradients against fourth-order central differences (steps
#' `1e-4`, `1e-5`, `1e-6`) for a multi-layer network, save/load, and (when
#' the torch runtime is installed) the torch forward pass, autograd
#' gradients and a short Adam run against the base R backend. With
#' `mixed = TRUE` the same forward, gradient and torch checks also run on a
#' network whose layers use different bases. Missing checks (no torch
#' runtime, or a registered basis without a naive implementation) are
#' reported as skipped (`pass = NA`), never as passed. Errors are scaled
#' errors `max(abs(a - b) / pmax(1, abs(a), abs(b)))` unless stated
#' otherwise.
#'
#' @param backend `"both"`, `"r"` or `"torch"`.
#' @param basis Bases to validate (default: all available, see
#'   [kan_bases()]).
#' @param mixed Also validate a network that mixes bases across layers.
#' @return A data frame with `basis`, `group`, `check`, `error`, `tol`,
#'   `pass`; attributes `all_pass` and `n_skipped`.
#' @examples
#' res <- thangld_kan_validate("r", basis = "chebyshev", mixed = FALSE)
#' res[, c("check", "pass")]
#' @export
thangld_kan_validate <- function(backend = c("both", "r", "torch"),
                               basis = kan_bases()$name, mixed = TRUE) {
  backend <- match.arg(backend)
  if (!is.character(basis) || !length(basis) || anyNA(basis)) {
    stop("`basis` must be a character vector of basis names.", call. = FALSE)
  }
  for (b in basis) basis_impl(b)
  mixed <- check_flag(mixed, "mixed")
  rows <- list()
  add <- function(b, group, check, error, tol) {
    rows[[length(rows) + 1L]] <<- data.frame(
      basis = b, group = group, check = check, error = error, tol = tol,
      pass = if (is.na(error)) NA else error <= tol)
  }
  for (b in basis) {
    validate_one_basis(b, backend, function(...) add(b, ...))
  }
  if (mixed) validate_mixed(backend, function(...) add("mixed", ...))
  res <- do.call(rbind, rows)
  attr(res, "n_skipped") <- sum(is.na(res$pass))
  attr(res, "all_pass") <- all(!is.na(res$pass)) && all(res$pass)
  res
}

#' @noRd
validate_core <- function(model, x, y, backend, add, naive = TRUE) {
  if (backend %in% c("r", "both")) {
    nv <- x
    for (l in seq_along(model$params)) {
      if (is.null(nv)) break
      nv <- layer_naive_forward(nv, model$params[[l]], model$specs[[l]])
    }
    add("forward", "oracle vs naive edge-by-edge loops",
        if (is.null(nv)) NA_real_ else
          scaled_error(thangld_kan_forward_r(model, x), nv), 1e-8)
    g <- thangld_kan_gradient_r(model, x, y)
    for (h in c(1e-4, 1e-5, 1e-6)) {
      fd <- thangld_kan_fd_gradient_r(model, x, y, step = h, order = 4L)
      add("gradient", sprintf("parameters vs central difference O4 (h=%g)", h),
          scaled_error(unlist(g$params), unlist(fd$params)), 1e-5)
      add("gradient", sprintf("input vs central difference O4 (h=%g)", h),
          scaled_error(g$x, fd$x), 1e-5)
    }
  }
  if (backend %in% c("torch", "both")) {
    checks <- c("forward torch vs oracle", "autograd parameters vs oracle",
                "autograd input vs oracle")
    if (!torch_ready()) {
      for (ck in checks) add("torch", ck, NA_real_, 1e-8)
    } else {
      add("torch", checks[1],
          scaled_error(engine_forward_torch(model$params, model$specs, x),
                       thangld_kan_forward_r(model, x)), 1e-8)
      g <- thangld_kan_gradient_r(model, x, y)
      gt <- engine_gradient_torch(model$params, model$specs, x, y)
      add("torch", checks[2], scaled_error(unlist(gt$params),
                                           unlist(g$params)), 1e-8)
      add("torch", checks[3], scaled_error(gt$x, g$x), 1e-8)
    }
  }
}

#' @noRd
validate_training <- function(make_model, backend, add) {
  if (backend %in% c("r", "both")) {
    xs <- with_seed(14, matrix(stats::runif(40, -2, 2), 20, 2,
                               dimnames = list(NULL, c("a", "b"))))
    fit <- thangld_kan_fit(make_model(1), xs, xs[, 1] * xs[, 2],
                         epochs = 5, backend = "r", seed = 1)
    f <- tempfile(fileext = ".rds")
    on.exit(unlink(f), add = TRUE)
    thangld_kan_save(fit, f)
    add("save/load", "predictions before vs after reload",
        scaled_error(predict(fit, xs), predict(thangld_kan_load(f), xs)), 1e-8)
  }
  if (backend %in% c("torch", "both")) {
    checks <- c("Adam training torch vs r backend (parameters)",
                "Adam training torch vs r backend (loss history)")
    if (!torch_ready()) {
      for (ck in checks) add("torch", ck, NA_real_, 1e-8)
    } else {
      xs <- with_seed(15, matrix(stats::runif(60, -2, 2), 30, 2))
      ys <- sin(xs[, 1]) + xs[, 2]^2
      m0 <- make_model(2)
      fr <- thangld_kan_fit(m0, xs, ys, epochs = 20, batch_size = 8, lr = 0.02,
                          seed = 3, backend = "r")
      ft <- thangld_kan_fit(m0, xs, ys, epochs = 20, batch_size = 8, lr = 0.02,
                          seed = 3, backend = "torch")
      add("torch", checks[1], scaled_error(unlist(ft$params),
                                           unlist(fr$params)), 1e-8)
      add("torch", checks[2], scaled_error(ft$history$train_loss,
                                           fr$history$train_loss), 1e-8)
    }
  }
}

#' @noRd
validate_one_basis <- function(b, backend, add) {
  impl <- basis_impl(b)
  if (backend %in% c("r", "both")) {
    if (is.null(impl$basis_checks)) {
      add("basis", "basis-specific checks", NA_real_, 1e-10)
    } else {
      for (r in impl$basis_checks()) add("basis", r$check, r$error, r$tol)
    }
  }
  fx <- validation_fixture(b)
  validate_core(fx$model, fx$x, fx$y, backend, add)
  validate_training(function(s) thangld_kan(c(2, 3, 1), basis = b, seed = s),
                    backend, add)
}

#' Network mixing the built-in bases across layers
#' @noRd
mixed_fixture <- function() {
  model <- thangld_kan(c(3, 4, 3, 2, 1),
                     basis = c("rbf", "chebyshev", "relu", "rational"),
                     degree = 3, grid_size = 3, seed = 31)
  x <- with_seed(32, matrix(stats::rnorm(18, sd = 1.2), 6, 3))
  y <- with_seed(33, stats::rnorm(6))
  list(model = model, x = x, y = y)
}

#' @noRd
validate_mixed <- function(backend, add) {
  fx <- mixed_fixture()
  validate_core(fx$model, fx$x, fx$y, backend, add)
  validate_training(function(s) {
    thangld_kan(c(2, 3, 1), basis = c("legendre", "wavelet"), seed = s)
  }, backend, add)
}
