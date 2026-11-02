# Rolling-origin evaluation and baselines --------------------------------------------

#' Rolling-origin one-step evaluation
#'
#' For every forecast origin \eqn{t_0} from `initial` to `length(y) - 1`, the
#' value \eqn{y_{t_0+1}} is forecast using only \eqn{y_1, \dots, y_{t_0}}.
#' The Temporal KAN is (re)fitted on \eqn{y_{1:t_0}} at the first origin and
#' then every `refit_every` origins (expanding window); between refits the
#' last fitted network and its standardization (estimated before that
#' refit's origin) are applied to the current history, so no value after an
#' origin is ever used for that origin's forecast. Validation windows for
#' early stopping are the last `n_val` windows before each refit origin.
#'
#' Baselines computed at the same origins: naive (\eqn{\hat y_{t_0+1} =
#' y_{t_0}}) and seasonal naive (\eqn{\hat y_{t_0+1} = y_{t_0+1-m}}) when
#' `period = m > 1`. MASE scales by the in-sample mean absolute one-step
#' seasonal-naive (or naive) error of the series up to the first origin.
#'
#' @param model An unfitted [thangld_temporal_kan()] (its architecture and seed are
#'   reused at every refit).
#' @param y Numeric vector or `ts`.
#' @param initial First forecast origin (number of observations available).
#' @param refit_every Refit frequency in origins.
#' @param period Seasonal period for the seasonal naive baseline and MASE
#'   (default `frequency(y)` for a `ts`, else 1).
#' @param ... Passed to [thangld_temporal_kan_fit()] (e.g. `n_val`, `epochs`, `lr`,
#'   `seed`, `backend`).
#' @return A list with `forecasts` (data frame: `origin`, `index`, `actual`,
#'   `kan`, `naive`, `snaive`, `refit`) and `metrics` (RMSE, MAE, MASE per
#'   method).
#' @examples
#' y <- log(as.numeric(datasets::lynx))
#' r <- thangld_temporal_kan_rolling(thangld_temporal_kan(1:3, seed = 1), y, initial = 100,
#'                           refit_every = 7, n_val = 10, epochs = 30,
#'                           lr = 0.02, backend = "r", seed = 1)
#' r$metrics
#' @export
thangld_temporal_kan_rolling <- function(model, y, initial, refit_every = 1L,
                                 period = NULL, ...) {
  validate_tkan(model)
  if (isTRUE(model$fitted)) {
    stop("Pass an unfitted thangld_temporal_kan(); it is refitted at each refit origin.",
         call. = FALSE)
  }
  if (is.null(period)) period <- if (stats::is.ts(y)) stats::frequency(y) else 1L
  period <- check_count(period, "period")
  y <- check_series(y)
  n <- length(y)
  initial <- check_count(initial, "initial", min = 2L)
  if (initial >= n) stop("`initial` must be smaller than length(y).",
                         call. = FALSE)
  refit_every <- check_count(refit_every, "refit_every")
  origins <- initial:(n - 1L)
  rows <- vector("list", length(origins))
  fit <- NULL
  for (k in seq_along(origins)) {
    t0 <- origins[k]
    hist <- y[seq_len(t0)]
    refit <- (k - 1L) %% refit_every == 0L
    if (refit) fit <- thangld_temporal_kan_fit(model, hist, ...)
    rows[[k]] <- data.frame(
      origin = t0, index = t0 + 1L, actual = y[t0 + 1L],
      kan = predict(fit, hist), naive = y[t0],
      snaive = if (period > 1L && t0 + 1L - period >= 1L) y[t0 + 1L - period]
        else NA_real_, refit = refit)
  }
  fc <- do.call(rbind, rows)
  m <- if (period > 1L) period else 1L
  y0 <- y[seq_len(initial)]
  scale <- if (length(y0) > m) mean(abs(diff(y0, lag = m))) else NA_real_
  meth <- c("kan", "naive", if (period > 1L) "snaive")
  metrics <- do.call(rbind, lapply(meth, function(mm) {
    e <- fc[[mm]] - fc$actual
    data.frame(method = mm, rmse = sqrt(mean(e^2, na.rm = TRUE)),
               mae = mean(abs(e), na.rm = TRUE),
               mase = mean(abs(e), na.rm = TRUE) / scale,
               n = sum(!is.na(e)))
  }))
  list(forecasts = fc, metrics = metrics, period = period, initial = initial)
}
