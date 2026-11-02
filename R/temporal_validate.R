#' Run the numerical validation suite
#'
#' Checks the lag-window model with Chebyshev edges (forward pass
#' against an independent `cos(k acos z)` loop implementation, analytic
#' gradients against fourth-order central differences for steps `1e-4`,
#' `1e-5`, `1e-6`), the lag-window construction against a direct loop, and
#' (with the torch runtime) the torch forward pass, autograd and a short Adam
#' run against the base R backend, and the gradients (and torch pass) of the
#' model with every other built-in basis. Skipped torch checks are reported with
#' `pass = NA`, never as passed.
#'
#' @param backend `"both"`, `"r"` or `"torch"`.
#' @return A data frame with `group`, `check`, `error`, `tol`, `pass`;
#'   attributes `all_pass` and `n_skipped`.
#' @examples
#' thangld_temporal_kan_validate("r")[, c("check", "pass")]
#' @export
thangld_temporal_kan_validate <- function(backend = c("both", "r", "torch")) {
  backend <- match.arg(backend)
  rows <- list()
  add <- function(group, check, error, tol) {
    rows[[length(rows) + 1L]] <<- data.frame(
      group = group, check = check, error = error, tol = tol,
      pass = if (is.na(error)) NA else error <= tol)
  }
  m <- thangld_temporal_kan(lags = c(1, 2, 5), hidden = 3, degree = c(4, 3), seed = 11)
  m$params <- with_seed(21, lapply(m$params, function(pl) {
    pl$coef[] <- stats::rnorm(length(pl$coef), sd = 0.4)
    pl
  }))
  x <- with_seed(12, matrix(stats::rnorm(18, sd = 1.5), 6, 3))
  y <- with_seed(13, stats::rnorm(6))
  if (backend %in% c("r", "both")) {
    naive <- x
    for (l in seq_along(m$params)) naive <- layer_naive_forward(naive, m$params[[l]], m$specs[[l]])
    add("forward", "oracle vs naive loops (cos(k acos z))",
        scaled_error(engine_forward(m$params, m$specs, x)$out, naive), 1e-8)
    g <- engine_gradient(m$params, m$specs, x, y)
    for (h in c(1e-4, 1e-5, 1e-6)) {
      fd <- engine_fd_gradient(m$params, m$specs, x, y, h, 4L)
      add("gradient", sprintf("coefficients vs central difference O4 (h=%g)", h),
          scaled_error(unlist(g$params), unlist(fd$params)), 1e-5)
      add("gradient", sprintf("input vs central difference O4 (h=%g)", h),
          scaled_error(g$x, fd$x), 1e-5)
    }
    s <- with_seed(3, cumsum(stats::rnorm(40)))
    for (dif in c(FALSE, TRUE)) {
      w <- thangld_temporal_kan_embed(s, c(1, 3, 4), difference = dif)
      z <- if (dif) c(NA, diff(s)) else s
      direct <- t(vapply(w$index, function(t) z[t - c(1, 3, 4)], numeric(3)))
      add("windows", sprintf("lag windows vs direct loop (difference = %s)", dif),
          max(abs(w$x - direct)) + max(abs(w$target - z[w$index])), 0)
    }
  }
  if (backend %in% c("torch", "both")) {
    ck <- c("forward torch vs oracle", "autograd vs oracle",
            "Adam training torch vs r backend")
    if (!torch_ready()) {
      for (k in ck) add("torch", k, NA_real_, 1e-8)
    } else {
      add("torch", ck[1], scaled_error(engine_forward_torch(m$params, m$specs, x),
                                       engine_forward(m$params, m$specs, x)$out), 1e-8)
      g <- engine_gradient(m$params, m$specs, x, y)
      gt <- engine_gradient_torch(m$params, m$specs, x, y)
      add("torch", ck[2], scaled_error(c(unlist(gt$params), gt$x),
                                       c(unlist(g$params), g$x)), 1e-8)
      s <- with_seed(5, as.numeric(stats::arima.sim(list(ar = 0.6), 120)))
      m0 <- thangld_temporal_kan(1:3, hidden = 2, seed = 2)
      fr <- thangld_temporal_kan_fit(m0, s, n_val = 20, epochs = 15, lr = 0.02, seed = 3,
                             backend = "r")
      ft <- thangld_temporal_kan_fit(m0, s, n_val = 20, epochs = 15, lr = 0.02, seed = 3,
                             backend = "torch")
      add("torch", ck[3], scaled_error(c(unlist(ft$params), ft$history$train_loss),
                                       c(unlist(fr$params), fr$history$train_loss)),
          1e-8)
    }
  }
  # every other built-in basis in the lag-window model
  for (b in setdiff(builtin_basis_table()$name, "chebyshev")) {
    mb <- thangld_temporal_kan(c(1, 2, 5), hidden = 3, basis = b, seed = 7)
    if (backend %in% c("r", "both")) {
      g <- engine_gradient(mb$params, mb$specs, x, y)
      fd <- engine_fd_gradient(mb$params, mb$specs, x, y, 1e-5, 4L)
      add("bases", sprintf("basis \"%s\": gradients vs central difference O4 (h=1e-05)", b),
          max(scaled_error(unlist(g$params), unlist(fd$params)),
              scaled_error(g$x, fd$x)), 1e-5)
    }
    if (backend %in% c("torch", "both")) {
      ckb <- sprintf("basis \"%s\": torch forward and autograd vs oracle", b)
      if (!torch_ready()) {
        add("bases", ckb, NA_real_, 1e-8)
      } else {
        g <- engine_gradient(mb$params, mb$specs, x, y)
        gt <- engine_gradient_torch(mb$params, mb$specs, x, y)
        add("bases", ckb, max(
          scaled_error(engine_forward_torch(mb$params, mb$specs, x),
                       engine_forward(mb$params, mb$specs, x)$out),
          scaled_error(c(unlist(gt$params), gt$x), c(unlist(g$params), g$x))), 1e-8)
      }
    }
  }
  res <- do.call(rbind, rows)
  attr(res, "n_skipped") <- sum(is.na(res$pass))
  attr(res, "all_pass") <- all(!is.na(res$pass)) && all(res$pass)
  res
}

#' Save and load a Temporal KAN
#'
#' @param model A `thangld_temporal_kan`.
#' @param file Path of the `.rds` file.
#' @return `thangld_temporal_kan_save()` returns `file` invisibly;
#'   `thangld_temporal_kan_load()` returns the model.
#' @examples
#' m <- thangld_temporal_kan(1:3, seed = 1)
#' f <- tempfile(fileext = ".rds")
#' thangld_temporal_kan_save(m, f)
#' identical(thangld_temporal_kan_load(f)$params, m$params)
#' @export
thangld_temporal_kan_save <- function(model, file) {
  validate_tkan(model)
  if (!is.character(file) || length(file) != 1L) {
    stop("`file` must be a single path.", call. = FALSE)
  }
  saveRDS(list(format = "thangldKANr_model", format_version = 1L,
               package_version = as.character(utils::packageVersion("thangldKANr")),
               r_version = R.version.string, model = unclass(model)), file)
  invisible(file)
}

#' @rdname thangld_temporal_kan_save
#' @export
thangld_temporal_kan_load <- function(file) {
  if (!is.character(file) || length(file) != 1L || !file.exists(file)) {
    stop("File not found: ", file, call. = FALSE)
  }
  obj <- readRDS(file)
  if (!is.list(obj) || !identical(obj$format, "thangldKANr_model")) {
    stop("Not a thangldKANr model file.", call. = FALSE)
  }
  validate_tkan(structure(obj$model, class = "thangld_temporal_kan"))
}
