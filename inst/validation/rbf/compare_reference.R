# Compare thangldKANr with the Python reference -------------------------------------------
# Usage: Rscript compare_reference.R <fixture_dir> <python_out_dir> <report_dir>
# Base R oracle/backend always; R torch backend when its runtime is installed
# (otherwise those rows are "not run"). Thresholds fixed before the first run:
# forward, gradients, 5 Adam steps: scaled error <= 1e-8; training (200
# epochs): loss histories, best-epoch parameters <= 1e-6, test metrics
# |diff| <= 1e-6.

suppressPackageStartupMessages(library(thangldKANr))
args <- commandArgs(trailingOnly = TRUE)
fix <- args[1]
py <- args[2]
rep_dir <- args[3]
dir.create(rep_dir, showWarnings = FALSE, recursive = TRUE)
use_torch <- thangldKANr:::torch_ready()
se <- thangldKANr:::scaled_error
rd <- function(...) utils::read.csv(file.path(...), stringsAsFactors = FALSE)
read_matrix <- function(path) {
  d <- utils::read.csv(path)
  m <- matrix(0, max(d$row) + 1, max(d$col) + 1)
  m[cbind(d$row + 1, d$col + 1)] <- d$value
  m
}
read_params <- function(path, like) {
  d <- utils::read.csv(path, stringsAsFactors = FALSE)
  lapply(seq_along(like), function(l) {
    out <- lapply(names(like[[l]]), function(nm) {
      s <- d[d$layer == l - 1 & d$name == nm, ]
      a <- like[[l]][[nm]] * 0
      a[s$pos + 1] <- as.numeric(s$value)
      a
    })
    stats::setNames(out, names(like[[l]]))
  })
}
read_sched <- function(path) {
  d <- utils::read.csv(path)
  lapply(sort(unique(d$epoch)), function(e) {
    s <- d[d$epoch == e, ]
    lapply(sort(unique(s$batch)), function(b) s$idx[s$batch == b] + 1)
  })
}
rows <- list()
add <- function(level, check, variant, error, tol, note = "") {
  rows[[length(rows) + 1]] <<- data.frame(
    level = level, check = check, variant = variant, error = error, tol = tol,
    status = if (is.na(error)) "not run" else if (error <= tol) "pass" else "FAIL",
    note = note)
}
variants <- strsplit(readLines(file.path(py, "variants.txt"), warn = FALSE), " ")[[1]]

net_template <- function() thangldKANr:::validation_fixture("rbf")$model
model <- net_template()
model$params <- read_params(file.path(fix, "net_params.csv"), model$params)
x <- read_matrix(file.path(fix, "net_x.csv"))
y <- rd(fix, "net_y.csv")$y
outs <- thangld_kan_forward_r(model, x, layers = TRUE)
g <- thangld_kan_gradient_r(model, x, y)
for (h in c(1e-4, 1e-5, 1e-6)) {
  fd <- thangld_kan_fd_gradient_r(model, x, y, step = h, order = 4)
  add("gradient", sprintf("R analytic vs central difference O4 h=%g (params)", h),
      "R oracle", se(unlist(g$params), unlist(fd$params)), 1e-5)
  add("gradient", sprintf("R analytic vs central difference O4 h=%g (input)", h),
      "R oracle", se(g$x, fd$x), 1e-5)
}
if (use_torch) {
  gt <- thangldKANr:::engine_gradient_torch(model$params, model$specs, x, y)
  ft <- thangldKANr:::engine_forward_torch(model$params, model$specs, x)
}
for (v in variants) {
  for (l in seq_along(outs)) {
    pm <- read_matrix(file.path(py, sprintf("net_%s_layer%d_out.csv", v, l - 1)))
    add("forward", sprintf("layer %d output, R oracle vs Python", l), v,
        se(outs[[l]], pm), 1e-8)
  }
  pg <- read_params(file.path(py, sprintf("net_%s_grad_params.csv", v)),
                    model$params)
  px <- read_matrix(file.path(py, sprintf("net_%s_grad_x.csv", v)))
  add("gradient", "parameter gradient, R oracle vs Python", v,
      se(unlist(g$params), unlist(pg)), 1e-8)
  add("gradient", "input gradient, R oracle vs Python", v, se(g$x, px), 1e-8)
  if (use_torch) {
    add("forward", "output, R torch vs Python", v, se(ft, pm), 1e-8)
    add("gradient", "parameter gradient, R torch vs Python", v,
        se(unlist(gt$params), unlist(pg)), 1e-8)
    add("gradient", "input gradient, R torch vs Python", v, se(gt$x, px), 1e-8)
  } else {
    add("forward", "R torch vs Python", v, NA, 1e-8,
        "torch runtime not installed")
  }
}

# Adam ---------------------------------------------------------------------------
st <- rd(fix, "adam_settings.csv")
sched <- read_sched(file.path(fix, "adam_schedule.csv"))
par <- model$params
states <- lapply(par, function(pl) lapply(pl, function(a) list(t = 0L, m = a * 0,
                                                               v = a * 0)))
r_par <- list()
r_v <- list()
r_loss <- numeric(0)
for (s in seq_along(sched)) {
  idx <- sched[[s]][[1]]
  gg <- thangldKANr:::engine_gradient(par, model$specs, x[idx, , drop = FALSE], y[idx])
  r_loss[s] <- gg$loss
  for (l in seq_along(par)) for (nm in names(par[[l]])) {
    u <- thangldKANr:::adam_step_r(par[[l]][[nm]], gg$params[[l]][[nm]],
                             states[[l]][[nm]], st$lr, c(st$beta1, st$beta2),
                             st$eps)
    par[[l]][[nm]] <- u$param
    states[[l]][[nm]] <- u$state
  }
  r_par[[s]] <- par
  r_v[[s]] <- lapply(states, function(sl) lapply(sl, `[[`, "v"))
}
for (v in variants) {
  pl <- rd(py, sprintf("adam_%s_loss.csv", v))$loss
  add("adam", "5 steps: batch losses", v, se(r_loss, pl), 1e-8)
  pp <- lapply(seq_along(sched), function(s) read_params(
    file.path(py, sprintf("adam_%s_step%d_params.csv", v, s - 1)), model$params))
  pv <- lapply(seq_along(sched), function(s) read_params(
    file.path(py, sprintf("adam_%s_step%d_exp_avg_sq.csv", v, s - 1)),
    model$params))
  add("adam", "5 steps: parameters after each step", v,
      se(unlist(r_par), unlist(pp)), 1e-8)
  add("adam", "5 steps: exp_avg_sq after each step", v,
      se(unlist(r_v), unlist(pv)), 1e-8)
}

# training -------------------------------------------------------------------------
ts <- rd(fix, "train_settings.csv")
runs <- rd(fix, "train_runs.csv")
backends <- c("r", if (use_torch) "torch")
met <- list()
tim <- rd(py, "train_timing.csv")
for (k in seq_len(nrow(runs))) {
  id <- runs$id[k]
  d <- file.path(fix, "train", id)
  xtr <- read_matrix(file.path(d, "x_train.csv"))
  ytr <- rd(d, "y_train.csv")$y
  xva <- read_matrix(file.path(d, "x_val.csv"))
  yva <- rd(d, "y_val.csv")$y
  xte <- read_matrix(file.path(d, "x_test.csv"))
  yte <- rd(d, "y_test_original.csv")$y
  ysc <- rd(d, "y_scaling.csv")
  widths <- as.integer(strsplit(rd(d, "init_meta.csv")$widths, " ")[[1]])
  seed <- runs$seed[k]
  m0 <- thangld_kan(basis = "rbf", widths, seed = seed)
  m0$params <- read_params(file.path(d, "init_params.csv"), m0$params)
  fits <- list()
  for (b in backends) {
    fits[[b]] <- thangld_kan_fit(m0, xtr, ytr, validation_data = list(x = xva, y = yva),
                         epochs = ts$epochs, batch_size = ts$batch_size,
                         lr = ts$lr, betas = c(ts$beta1, ts$beta2),
                         eps = ts$eps, standardize_x = FALSE,
                         standardize_y = FALSE, seed = seed, backend = b)
    p <- predict(fits[[b]], xte) * ysc$y_scale + ysc$y_center
    met[[length(met) + 1]] <- data.frame(
      id = id, target = runs$target[k], seed = seed,
      impl = paste0("thangldKANr-", b), best_epoch = fits[[b]]$train$best_epoch,
      fit_sec = fits[[b]]$train$elapsed_sec, t(thangld_kan_metrics(yte, p)))
  }
  for (v in variants) {
    pdir <- file.path(py, "train", id)
    h <- rd(pdir, sprintf("%s_history.csv", v))
    bp <- read_params(file.path(pdir, sprintf("%s_best_params.csv", v)),
                      m0$params)
    ppo <- rd(pdir, sprintf("%s_pred_test_scaled.csv", v))$pred *
      ysc$y_scale + ysc$y_center
    tm <- tim[tim$id == id & tim$variant == v, ]
    met[[length(met) + 1]] <- data.frame(
      id = id, target = runs$target[k], seed = seed, impl = paste0("ref-", v),
      best_epoch = tm$best_epoch, fit_sec = tm$fit_sec,
      t(thangld_kan_metrics(yte, ppo)))
    for (b in backends) {
      f <- fits[[b]]
      lab <- sprintf("%s (R %s backend)", id, b)
      add("training", paste(lab, "train loss history"), v,
          se(f$history$train_loss, h$train_loss), 1e-6)
      add("training", paste(lab, "val loss history"), v,
          se(f$history$val_loss, h$val_loss), 1e-6)
      add("training", paste(lab, "best-epoch parameters"), v,
          se(unlist(f$params), unlist(bp)), 1e-6,
          sprintf("best epoch R %d / Python %d", f$train$best_epoch,
                  tm$best_epoch))
      pr <- predict(f, xte) * ysc$y_scale + ysc$y_center
      add("training", paste(lab, "test RMSE/MAE/R2 |diff|"), v,
          max(abs(thangld_kan_metrics(yte, pr) - thangld_kan_metrics(yte, ppo))), 1e-6)
    }
  }
  if (!use_torch) add("training", sprintf("%s (R torch backend)", id), "all",
                      NA, 1e-6, "torch runtime not installed")
}
res <- do.call(rbind, rows)
metrics <- do.call(rbind, met)
utils::write.csv(res, file.path(rep_dir, "comparison_checks.csv"),
                 row.names = FALSE)
utils::write.csv(metrics, file.path(rep_dir, "training_metrics.csv"),
                 row.names = FALSE)
agg <- stats::aggregate(cbind(rmse, mae, r2, fit_sec) ~ target + impl,
                        data = metrics, FUN = stats::median)
sink(file.path(rep_dir, "comparison_summary.txt"))
cat("thangldKANr vs Python reference comparison\n")
cat("Date:", format(Sys.time()), "\n")
cat(R.version.string, "| thangldKANr", as.character(packageVersion("thangldKANr")),
    "| torch R pkg", as.character(packageVersion("torch")),
    "| torch runtime:", use_torch, "\n")
cat("Python env:", paste(readLines(file.path(py, "python_env.json"), warn = FALSE),
                         collapse = " "), "\n\n")
cat("Status counts:\n")
print(table(res$level, res$status))
cat("\nFailures / not run:\n")
print(res[res$status != "pass", c("level", "check", "variant", "error", "note")],
      row.names = FALSE)
cat("\nMax error per level and variant:\n")
ok <- !is.na(res$error) & is.finite(res$error)
print(stats::aggregate(error ~ level + variant, data = res[ok, ], FUN = max))
cat("\nTest metrics (original scale), median over 5 seeds:\n")
print(agg, row.names = FALSE)
sink()
cat(readLines(file.path(rep_dir, "comparison_summary.txt")), sep = "\n")
