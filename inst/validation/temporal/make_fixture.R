# Shared fixture for thangldKANr <-> ChebyKAN-on-lag-windows comparison ----------
# Usage: Rscript make_fixture.R <out_dir>
# Writes raw series, lags, n_val, initial coefficients and batch schedules.
# Python builds the lag windows and the standardization itself from the raw
# series, so the windowing mechanism is checked independently.
suppressPackageStartupMessages(library(thangldKANr))
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[1] else "fixture"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
fmt <- function(df) {
  for (j in seq_along(df)) if (is.double(df[[j]])) df[[j]] <- sprintf("%.17g", df[[j]])
  df
}
wcsv <- function(df, path) utils::write.csv(fmt(df), path, row.names = FALSE, quote = FALSE)
par_long <- function(params) do.call(rbind, lapply(seq_along(params), function(l) {
  a <- params[[l]]$coef
  data.frame(layer = l - 1L, name = "coef", pos = seq_along(a) - 1L, value = as.vector(a))
}))
par_dims <- function(params) do.call(rbind, lapply(seq_along(params), function(l) {
  data.frame(layer = l - 1L, name = "coef", dims = paste(dim(params[[l]]$coef), collapse = " "))
}))
sched_long <- function(s) do.call(rbind, lapply(seq_along(s), function(e) {
  do.call(rbind, lapply(seq_along(s[[e]]), function(b) {
    data.frame(epoch = e - 1L, batch = b - 1L, idx = s[[e]][[b]] - 1L)
  }))
}))
sim_nlar <- function(n, seed) {
  set.seed(seed)
  y <- numeric(n)
  for (t in 3:n) y[t] <- 0.5 * y[t - 1] - 0.8 * tanh(2 * y[t - 2]) + stats::rnorm(1, sd = 0.3)
  y
}
series <- list(
  nlar = list(y = sim_nlar(260, 3), lags = 1:2, hidden = 3L),
  lynx_log = list(y = log(as.numeric(datasets::lynx)), lags = 1:3, hidden = integer()),
  airpass_logdiff = list(y = as.numeric(diff(log(datasets::AirPassengers))),
                         lags = c(1, 2, 12), hidden = 2L)
)
settings <- data.frame(epochs = 150L, batch_size = 32L, lr = 0.02, beta1 = 0.9,
                       beta2 = 0.999, eps = 1e-8, n_val = 20L)
wcsv(settings, file.path(out, "train_settings.csv"))
runs <- list()
for (nm in names(series)) {
  s <- series[[nm]]
  for (seed in 1:5) {
    id <- sprintf("%s_s%d", nm, seed)
    dd <- file.path(out, "train", id)
    dir.create(dd, recursive = TRUE, showWarnings = FALSE)
    m <- thangld_temporal_kan(s$lags, hidden = s$hidden, seed = seed)
    w <- thangld_temporal_kan_embed(s$y, s$lags)
    ntr <- nrow(w$x) - settings$n_val
    sched <- thangldKANr:::with_seed(seed, thangldKANr:::make_batch_schedule(
      ntr, settings$batch_size, settings$epochs))
    wcsv(data.frame(y = s$y), file.path(dd, "series.csv"))
    utils::write.csv(data.frame(lags = paste(s$lags, collapse = " "),
                                widths = paste(m$widths, collapse = " ")),
                     file.path(dd, "meta.csv"), row.names = FALSE, quote = FALSE)
    wcsv(par_long(m$params), file.path(dd, "init_params.csv"))
    utils::write.csv(par_dims(m$params), file.path(dd, "init_dims.csv"),
                     row.names = FALSE, quote = FALSE)
    utils::write.csv(sched_long(sched), file.path(dd, "schedule.csv"),
                     row.names = FALSE, quote = FALSE)
    runs[[id]] <- data.frame(id = id, series = nm, seed = seed)
  }
}
utils::write.csv(do.call(rbind, runs), file.path(out, "train_runs.csv"),
                 row.names = FALSE, quote = FALSE)
cat("Fixture written to", normalizePath(out), "\n")
