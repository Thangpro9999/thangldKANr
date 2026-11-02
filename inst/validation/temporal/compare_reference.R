# Compare thangldKANr with ChebyKAN on lag windows -----------------------------------
# Usage: Rscript compare_reference.R <fixture_dir> <python_out_dir> <report_dir>
# Thresholds fixed before the first run: windows and scaling identical up to
# 1e-12; loss histories, best-epoch coefficients and the final one-step
# forecast: scaled error <= 1e-6.
suppressPackageStartupMessages(library(thangldKANr))
args <- commandArgs(trailingOnly = TRUE)
fix <- args[1]; py <- args[2]; rep_dir <- args[3]
dir.create(rep_dir, showWarnings = FALSE, recursive = TRUE)
use_torch <- thangldKANr:::torch_ready()
se <- thangldKANr:::scaled_error
rd <- function(...) utils::read.csv(file.path(...), stringsAsFactors = FALSE)
rows <- list()
add <- function(level, check, variant, error, tol, note = "") {
  rows[[length(rows) + 1]] <<- data.frame(level = level, check = check,
    variant = variant, error = error, tol = tol,
    status = if (is.na(error)) "not run" else if (error <= tol) "pass" else "FAIL",
    note = note)
}
st <- rd(fix, "train_settings.csv")
runs <- rd(fix, "train_runs.csv")
met <- list()
tim <- rd(py, "train_timing.csv")
for (k in seq_len(nrow(runs))) {
  id <- runs$id[k]
  d <- file.path(fix, "train", id)
  y <- rd(d, "series.csv")$y
  meta <- rd(d, "meta.csv")
  lags <- as.integer(strsplit(meta$lags, " ")[[1]])
  widths <- as.integer(strsplit(meta$widths, " ")[[1]])
  m0 <- thangld_temporal_kan(lags, hidden = widths[-c(1, length(widths))], seed = runs$seed[k])
  pd <- file.path(py, "train", id)
  w <- thangld_temporal_kan_embed(y, lags)
  pw <- rd(pd, "windows.csv")
  add("windows", paste(id, "lag windows and targets, R vs Python"), "numpy",
      max(abs(cbind(w$x, w$target) - as.matrix(pw[, -1]))), 1e-12)
  for (b in c("r", if (use_torch) "torch")) {
    fit <- thangld_temporal_kan_fit(m0, y, n_val = st$n_val, epochs = st$epochs,
                            batch_size = st$batch_size, lr = st$lr,
                            betas = c(st$beta1, st$beta2), eps = st$eps,
                            seed = runs$seed[k], backend = b)
    psc <- rd(pd, "scaling.csv")
    add("windows", paste(id, "training-only scaling, R vs Python"), b,
        se(c(fit$scaling$center, fit$scaling$scale), c(psc$center, psc$scale)), 1e-12)
    fc <- predict(fit)
    met[[length(met) + 1]] <- data.frame(id = id, impl = paste0("thangldKANr-", b),
      best_epoch = fit$train$best_epoch, fit_sec = fit$train$elapsed_sec, forecast = fc)
    for (v in c("opt", "unopt")) {
      h <- rd(pd, sprintf("%s_history.csv", v))
      bp <- utils::read.csv(file.path(pd, sprintf("%s_best_params.csv", v)))
      tm <- tim[tim$id == id & tim$variant == v, ]
      add("training", sprintf("%s (R %s) train/val loss histories", id, b), v,
          se(c(fit$history$train_loss, fit$history$val_loss), c(h$train_loss, h$val_loss)), 1e-6)
      add("training", sprintf("%s (R %s) best-epoch coefficients", id, b), v,
          se(unlist(lapply(fit$params, function(p) as.vector(p$coef))), bp$value), 1e-6,
          sprintf("best epoch R %d / Python %d", fit$train$best_epoch, tm$best_epoch))
      pf <- rd(pd, sprintf("%s_forecast.csv", v))$forecast
      add("forecast", sprintf("%s (R %s) one-step forecast after last value", id, b), v,
          se(fc, pf), 1e-6)
      if (b == "r") met[[length(met) + 1]] <- data.frame(id = id, impl = paste0("ChebyKAN-", v),
        best_epoch = tm$best_epoch, fit_sec = tm$fit_sec, forecast = pf)
    }
  }
  if (!use_torch) add("training", paste(id, "(R torch backend)"), "all", NA, 1e-6,
                      "torch runtime not installed")
}
res <- do.call(rbind, rows)
utils::write.csv(res, file.path(rep_dir, "comparison_checks.csv"), row.names = FALSE)
utils::write.csv(do.call(rbind, met), file.path(rep_dir, "forecasts.csv"), row.names = FALSE)
sink(file.path(rep_dir, "comparison_summary.txt"))
cat("thangldKANr vs ChebyKAN-on-lag-windows comparison\nDate:", format(Sys.time()), "\n")
cat(R.version.string, "| torch runtime:", use_torch, "\n")
cat("Python env:", paste(readLines(file.path(py, "python_env.json"), warn = FALSE), collapse = " "), "\n\n")
print(table(res$level, res$status))
cat("\nFailures / not run:\n")
print(res[res$status != "pass", c("level", "check", "variant", "error", "note")], row.names = FALSE)
ok <- !is.na(res$error) & is.finite(res$error)
cat("\nMax error per level and variant:\n")
print(stats::aggregate(error ~ level + variant, data = res[ok, ], FUN = max))
sink()
cat(readLines(file.path(rep_dir, "comparison_summary.txt")), sep = "\n")
