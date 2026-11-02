# Shared fixture for the R <-> Python comparison ----------------------------------
# Usage: Rscript make_fixture.R <out_dir>
# Parameters are written in long format (layer, name, pos, value) where `pos`
# is the 0-based column-major position in the R array; dims are in
# *_dims.csv. Requires the installed thangldKANr package; no torch runtime needed.

suppressPackageStartupMessages(library(thangldKANr))
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[1] else "fixture"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

fmt <- function(df) {
  for (j in seq_along(df)) if (is.double(df[[j]])) {
    df[[j]] <- sprintf("%.17g", df[[j]])
  }
  df
}
wcsv <- function(df, path) utils::write.csv(fmt(df), path, row.names = FALSE,
                                            quote = FALSE)
mat_long <- function(m) data.frame(row = rep(seq_len(nrow(m)) - 1L, ncol(m)),
                                   col = rep(seq_len(ncol(m)) - 1L,
                                             each = nrow(m)),
                                   value = as.vector(m))
par_long <- function(params) do.call(rbind, lapply(seq_along(params), function(l) {
  do.call(rbind, lapply(names(params[[l]]), function(nm) {
    a <- params[[l]][[nm]]
    data.frame(layer = l - 1L, name = nm, pos = seq_along(a) - 1L,
               value = as.vector(a))
  }))
}))
par_dims <- function(params) do.call(rbind, lapply(seq_along(params), function(l) {
  do.call(rbind, lapply(names(params[[l]]), function(nm) {
    d <- dim(params[[l]][[nm]])
    if (is.null(d)) d <- length(params[[l]][[nm]])
    data.frame(layer = l - 1L, name = nm, dims = paste(d, collapse = " "))
  }))
}))
sched_long <- function(s) do.call(rbind, lapply(seq_along(s), function(e) {
  do.call(rbind, lapply(seq_along(s[[e]]), function(b) {
    data.frame(epoch = e - 1L, batch = b - 1L, idx = s[[e]][[b]] - 1L)
  }))
}))
write_model <- function(model, prefix, dir = out) {
  wcsv(par_long(model$params), file.path(dir, paste0(prefix, "_params.csv")))
  utils::write.csv(par_dims(model$params),
                   file.path(dir, paste0(prefix, "_dims.csv")),
                   row.names = FALSE, quote = FALSE)
  utils::write.csv(data.frame(widths = paste(model$widths, collapse = " ")),
                   file.path(dir, paste0(prefix, "_meta.csv")),
                   row.names = FALSE, quote = FALSE)
}

# 1. forward / gradient fixture -------------------------------------------------
model <- thangldKANr:::validation_fixture("chebyshev")$model
set.seed(7)
x <- matrix(stats::rnorm(10 * model$widths[1], sd = 1.5), 10, model$widths[1])
y <- stats::rnorm(10)
write_model(model, "net")
wcsv(mat_long(x), file.path(out, "net_x.csv"))
wcsv(data.frame(y = y), file.path(out, "net_y.csv"))

# 2. Adam fixture: 5 steps with explicit mini-batches ---------------------------
adam_sched <- list(list(c(1, 2, 3, 4)), list(c(5, 6, 7)), list(c(8, 9, 10, 1)),
                   list(c(2, 4, 6, 8, 10)), list(1:10))
utils::write.csv(sched_long(adam_sched), file.path(out, "adam_schedule.csv"),
                 row.names = FALSE, quote = FALSE)
wcsv(data.frame(lr = 0.02, beta1 = 0.9, beta2 = 0.999, eps = 1e-8),
     file.path(out, "adam_settings.csv"))

# 3. training fixture: targets x 5 seeds ------------------------------------------
targets <- list(
  linear = list(f = function(x) 1 + 2 * x[, 1] - x[, 2], widths = c(2, 1)),
  additive = list(f = function(x) sin(x[, 1]) + 0.5 * x[, 2]^2,
                  widths = c(2, 1)),
  interaction = list(f = function(x) x[, 1] * x[, 2], widths = c(2, 4, 1))
)
ts <- data.frame(epochs = 200L, batch_size = 32L, lr = 0.02, beta1 = 0.9,
                 beta2 = 0.999, eps = 1e-8, n_train = 240L, n_val = 60L,
                 n_test = 500L, noise_sd = 0.05)
wcsv(ts, file.path(out, "train_settings.csv"))
sim <- function(f, n, seed) {
  set.seed(seed)
  x <- matrix(stats::runif(2 * n, -2, 2), n, 2)
  list(x = x, y = f(x) + stats::rnorm(n, sd = 0.05))
}
runs <- list()
for (nm in names(targets)) {
  tg <- targets[[nm]]
  for (seed in 1:5) {
    id <- sprintf("%s_s%d", nm, seed)
    d <- sim(tg$f, 300, seed)
    te <- sim(tg$f, 500, seed + 100)
    pp <- thangldKANr:::fit_preprocess(d$x[1:240, ], d$y[1:240])
    dd <- file.path(out, "train", id)
    dir.create(dd, recursive = TRUE, showWarnings = FALSE)
    wcsv(mat_long(thangldKANr:::apply_preprocess_x(pp, d$x[1:240, ])),
         file.path(dd, "x_train.csv"))
    wcsv(data.frame(y = thangldKANr:::apply_preprocess_y(pp, d$y[1:240])),
         file.path(dd, "y_train.csv"))
    wcsv(mat_long(thangldKANr:::apply_preprocess_x(pp, d$x[241:300, ])),
         file.path(dd, "x_val.csv"))
    wcsv(data.frame(y = thangldKANr:::apply_preprocess_y(pp, d$y[241:300])),
         file.path(dd, "y_val.csv"))
    wcsv(mat_long(thangldKANr:::apply_preprocess_x(pp, te$x)),
         file.path(dd, "x_test.csv"))
    wcsv(data.frame(y = te$y), file.path(dd, "y_test_original.csv"))
    wcsv(data.frame(y_center = pp$y_center, y_scale = pp$y_scale),
         file.path(dd, "y_scaling.csv"))
    widths <- tg$widths
    m <- thangld_kan(widths, basis = "chebyshev", seed = seed)
    write_model(m, "init", dd)
    sched <- thangldKANr:::with_seed(seed, thangldKANr:::make_batch_schedule(240L, 32L, 200L))
    utils::write.csv(sched_long(sched), file.path(dd, "schedule.csv"),
                     row.names = FALSE, quote = FALSE)
    runs[[id]] <- data.frame(id = id, target = nm, seed = seed)
  }
}
utils::write.csv(do.call(rbind, runs), file.path(out, "train_runs.csv"),
                 row.names = FALSE, quote = FALSE)
cat("Fixture written to", normalizePath(out), "\n")
