# Temporal KAN: architecture, lag windows, fit, one-step forecasts ---------------

#' Create a lag-window Temporal KAN
#'
#' A Temporal KAN here is a Kolmogorov-Arnold network that receives a window
#' of lagged values of a univariate series and predicts the next value
#' (one-step-ahead forecasting). The KAN part is a stack of edge layers of
#' any basis of [thangld_kan()] (default Chebyshev,
#' \eqn{\phi_{j,i}(x) = \sum_k c_{i,j,k} T_k(\tanh x)}); with
#' `hidden = integer()` the forecast is an additive function of the lags (a
#' nonlinear additive autoregression), and hidden layers allow interactions
#' between lags. It is not recurrent or attention-based.
#'
#' @param lags Positive integer lags used as inputs, e.g. `1:12` or
#'   `c(1, 2, 12)`.
#' @param hidden Integer widths of hidden layers (`integer()` for none).
#' @param basis Edge basis, one value or one per layer (see [kan_bases()]).
#' @param ... Named basis-specific arguments, as in [thangld_kan()] (for
#'   example `degree` for `"chebyshev"`, default 4).
#' @param difference If `TRUE`, the network models first differences
#'   \eqn{y_t - y_{t-1}} and forecasts are integrated back.
#' @param log If `TRUE`, the series is log-transformed first (must be
#'   positive) and forecasts are exponentiated back (median forecast).
#' @param options `NULL` or a named list of basis-specific arguments per
#'   basis, as in [thangld_kan()].
#' @param seed Optional seed for the initialization.
#' @return An object of class `thangld_temporal_kan`.
#' @examples
#' thangld_temporal_kan(lags = 1:12, hidden = 4, seed = 1)
#' thangld_temporal_kan(lags = 1:6, basis = "rbf", seed = 1)
#' @export
thangld_temporal_kan <- function(lags = 1:12, hidden = integer(),
                                 basis = "chebyshev", ..., difference = FALSE,
                                 log = FALSE, options = NULL, seed = NULL) {
  if (!is.numeric(lags) || !length(lags) || anyNA(lags) ||
      any(lags != round(lags)) || any(lags < 1) || anyDuplicated(lags)) {
    stop("`lags` must be distinct positive integers.", call. = FALSE)
  }
  lags <- sort(as.integer(lags))
  if (length(hidden) && (!is.numeric(hidden) || anyNA(hidden) ||
                         any(hidden != round(hidden)) || any(hidden < 1))) {
    stop("`hidden` must be positive integers (or integer()).", call. = FALSE)
  }
  widths <- c(length(lags), as.integer(hidden), 1L)
  nl <- length(widths) - 1L
  difference <- check_flag(difference, "difference")
  log <- check_flag(log, "log")
  if (!is.null(seed)) seed <- check_count(seed, "seed", min = 0L)
  specs <- build_layer_specs(widths, basis, list(...), options)
  params <- with_seed(seed, lapply(seq_len(nl), function(l) {
    layer_init(widths[l], widths[l + 1L], specs[[l]])
  }))
  structure(list(lags = lags, widths = widths, specs = specs, params = params,
                 difference = difference, log = log, init = list(seed = seed),
                 fitted = FALSE), class = "thangld_temporal_kan")
}

#' @noRd
validate_tkan <- function(model) {
  if (!inherits(model, "thangld_temporal_kan")) {
    stop("`model` must be a thangld_temporal_kan object (see thangld_temporal_kan()).",
         call. = FALSE)
  }
  check_layer_params(model$params, model$specs, model$widths)
  invisible(model)
}

#' @export
print.thangld_temporal_kan <- function(x, ...) {
  cat("<thangld_temporal_kan> lag-window KAN, one-step forecasts\n")
  cat("  layers : ", paste(vapply(x$specs, layer_describe, ""), collapse = ", "), "\n", sep = "")
  cat("  lags   : ", paste(x$lags, collapse = ", "), "\n", sep = "")
  cat("  widths : ", paste(x$widths, collapse = " -> "), "\n", sep = "")
  cat("  target : ", if (x$log) "log " else "", if (x$difference)
    "first difference" else "level", "\n", sep = "")
  cat("  fitted : ", if (isTRUE(x$fitted)) sprintf(
    "yes (%s, %d epochs, %d windows, last time index %d)", x$train$backend,
    nrow(x$history), x$train$n_windows, x$train$last_index) else "no",
    "\n", sep = "")
  invisible(x)
}

#' Lag-window design matrix
#'
#' Builds, for every time index `t` with all lags available, the row
#' \eqn{(z_{t-l_1}, \dots, z_{t-l_p})} and the target \eqn{z_t}, where `z`
#' is the (optionally log-transformed and differenced) series. Only values
#' before `t` enter row `t`.
#'
#' @param y Numeric vector or `ts`.
#' @param lags Positive integer lags.
#' @param difference,log Transformations as in [thangld_temporal_kan()].
#' @return A list with `x` (matrix, columns `lag1`, ...), `target`, `index`
#'   (time index of each target in `y`) and `z` (transformed series aligned
#'   with `y`, `NA` where undefined).
#' @examples
#' w <- thangld_temporal_kan_embed(1:10, lags = c(1, 3))
#' cbind(w$x, target = w$target, t = w$index)
#' @export
thangld_temporal_kan_embed <- function(y, lags, difference = FALSE, log = FALSE) {
  y <- check_series(y)
  if (log) {
    if (any(y <= 0)) stop("`log = TRUE` needs a positive series.", call. = FALSE)
    y <- base::log(y)
  }
  z <- if (difference) c(NA_real_, diff(y)) else y
  start <- max(lags) + 1L + as.integer(difference)
  n <- length(y)
  if (start > n) {
    stop("The series is too short for these lags.", call. = FALSE)
  }
  idx <- start:n
  x <- vapply(lags, function(l) z[idx - l], numeric(length(idx)))
  if (is.null(dim(x))) x <- matrix(x, nrow = length(idx))
  colnames(x) <- paste0("lag", lags)
  list(x = x, target = z[idx], index = idx, z = z)
}

#' @noRd
check_series <- function(y) {
  if (!is.numeric(y) || !is.null(dim(y)) && NCOL(y) != 1L) {
    stop("`y` must be a univariate numeric series.", call. = FALSE)
  }
  y <- as.numeric(y)
  if (anyNA(y) || any(!is.finite(y))) {
    stop("`y` contains NA or non-finite values.", call. = FALSE)
  }
  y
}

#' Fit a Temporal KAN on a series
#'
#' Builds lag windows from `y`, estimates the standardization (mean and
#' standard deviation of the transformed series) from the training targets
#' only, holds out the last `n_val` windows (in time order) for validation
#' and early stopping, and trains by mini-batch Adam on the mean squared
#' one-step error. Mini-batches are drawn from training windows; every
#' window contains only values before its target, so shuffling windows does
#' not use future information. The validation windows come after all
#' training windows.
#'
#' @param model A [thangld_temporal_kan()] object.
#' @param y Numeric vector or `ts` (the full history available at the
#'   forecast origin).
#' @param n_val Number of final windows used for validation (0 for none).
#' @param epochs,batch_size,lr,betas,eps,patience,min_delta Training
#'   settings (see [thangld_kan_fit()] for their meaning).
#' @param seed Optional seed for the batch order.
#' @param backend `"torch"` (default) or `"r"` (base R, no libtorch).
#' @param dtype `"float64"` or `"float32"` (torch backend).
#' @param verbose Print progress.
#' @return The fitted `thangld_temporal_kan` with `scaling`, `history`, `train` and
#'   the stored series `y` (used by [predict.thangld_temporal_kan()]).
#' @examples
#' y <- as.numeric(datasets::lynx)
#' m <- thangld_temporal_kan(lags = 1:3, seed = 1)
#' fit <- thangld_temporal_kan_fit(m, log(y), n_val = 10, epochs = 50, lr = 0.02,
#'                         backend = "r", seed = 1)
#' predict(fit)
#' @export
thangld_temporal_kan_fit <- function(model, y, n_val = 0L, epochs = 200L,
                             batch_size = 32L, lr = 0.01,
                             betas = c(0.9, 0.999), eps = 1e-8,
                             patience = NULL, min_delta = 0, seed = NULL,
                             backend = c("torch", "r"),
                             dtype = c("float64", "float32"), verbose = FALSE) {
  validate_tkan(model)
  backend <- match.arg(backend)
  dtype <- match.arg(dtype)
  if (backend == "torch") require_torch()
  y <- check_series(y)
  w <- thangld_temporal_kan_embed(y, model$lags, model$difference, model$log)
  n_val <- check_count(n_val, "n_val", min = 0L)
  nw <- nrow(w$x)
  if (nw - n_val < 2L) {
    stop("Too few training windows after holding out `n_val`.", call. = FALSE)
  }
  epochs <- check_count(epochs, "epochs")
  lr <- check_number(lr, "lr", min = 0, strict = TRUE)
  if (!is.null(patience)) patience <- check_count(patience, "patience")
  if (!is.null(seed)) seed <- check_count(seed, "seed", min = 0L)
  tr <- seq_len(nw - n_val)
  va <- if (n_val > 0L) (nw - n_val + 1L):nw else integer()
  sc <- if (isTRUE(model$fitted)) model$scaling else
    list(center = mean(w$target[tr]),
         scale = if (stats::sd(w$target[tr]) > 0) stats::sd(w$target[tr]) else 1)
  xs <- (w$x - sc$center) / sc$scale
  ys <- (w$target - sc$center) / sc$scale
  batch_size <- min(check_count(batch_size, "batch_size"), length(tr))
  schedule <- with_seed(seed, make_batch_schedule(length(tr), batch_size, epochs))
  args <- list(model$params, model$specs, xs[tr, , drop = FALSE], ys[tr],
               if (length(va)) xs[va, , drop = FALSE] else NULL,
               if (length(va)) ys[va] else NULL, schedule, lr, betas, eps,
               patience, min_delta, verbose)
  t0 <- proc.time()[["elapsed"]]
  res <- if (backend == "r") do.call(train_r, args) else
    do.call(train_torch, c(args, list(dtype)))
  model$params <- res$params
  model$history <- res$history
  model$scaling <- sc
  model$y <- y
  model$fitted <- TRUE
  model$train <- list(backend = backend, epochs = epochs, batch_size = batch_size,
                      lr = lr, seed = seed, n_windows = nw, n_val = n_val,
                      best_epoch = res$best_epoch, last_index = length(y),
                      elapsed_sec = proc.time()[["elapsed"]] - t0)
  model
}

#' One-step-ahead forecast
#'
#' Forecasts the value following the end of `newdata` (default: the series
#' used in fitting) with the fitted network and the stored training
#' standardization. Only one-step forecasts are supported.
#'
#' @param object A fitted `thangld_temporal_kan`.
#' @param newdata Optional series (history up to the forecast origin); at
#'   least `max(lags)` (+1 with `difference`) values.
#' @param h Forecast horizon; must be 1.
#' @param backend `"r"` (default) or `"torch"`.
#' @param ... Unused.
#' @return A single forecast on the original scale of the series.
#' @examples
#' y <- as.numeric(datasets::lynx)
#' fit <- thangld_temporal_kan_fit(thangld_temporal_kan(1:3, seed = 1), log(y[1:100]),
#'                         epochs = 30, lr = 0.02, backend = "r", seed = 1)
#' c(forecast = predict(fit, log(y[1:100])), actual = log(y[101]))
#' @export
predict.thangld_temporal_kan <- function(object, newdata = NULL, h = 1L,
                                 backend = c("r", "torch"), ...) {
  validate_tkan(object)
  if (!isTRUE(object$fitted)) stop("The model is not fitted.", call. = FALSE)
  if (!identical(as.integer(h), 1L)) {
    stop("Only one-step forecasts (h = 1) are supported.",
         call. = FALSE)
  }
  backend <- match.arg(backend)
  y <- if (is.null(newdata)) object$y else check_series(newdata)
  w <- thangld_temporal_kan_embed(c(y, mean(y)), object$lags, object$difference,
                          object$log)
  # the appended placeholder is only the (unused) target of the last row
  xr <- w$x[nrow(w$x), , drop = FALSE]
  xs <- (xr - object$scaling$center) / object$scaling$scale
  out <- if (backend == "r") drop(engine_forward(object$params, object$specs,
                                                 xs)$out) else
    engine_forward_torch(object$params, object$specs, xs)
  zhat <- out * object$scaling$scale + object$scaling$center
  last <- y[length(y)]
  if (object$log) last <- base::log(last)
  val <- if (object$difference) last + zhat else zhat
  if (object$log) exp(val) else val
}
