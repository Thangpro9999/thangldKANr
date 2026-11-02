# Compare thangldKANr with the Python graph KAN reference ----------------------------
# Usage: Rscript compare_reference.R <fixture_dir> <python_out_dir> <report_dir>
# Thresholds fixed before the first run: normalized adjacency 1e-12; forward,
# gradients and 5 Adam steps 1e-8; training (loss histories, best-epoch
# coefficients, predictions for all nodes, scaling) scaled error 1e-6.
suppressPackageStartupMessages(library(thangldKANr))
args <- commandArgs(trailingOnly = TRUE)
fix <- args[1]; py <- args[2]; rep_dir <- args[3]
dir.create(rep_dir, showWarnings = FALSE, recursive = TRUE)
use_torch <- thangldKANr:::torch_ready()
se <- thangldKANr:::scaled_error
rd <- function(...) utils::read.csv(file.path(...), stringsAsFactors = FALSE)
vec <- function(...) rd(...)[[1]]
rows <- list()
add <- function(level, check, variant, error, tol, note = "") {
  rows[[length(rows) + 1]] <<- data.frame(level = level, check = check,
    variant = variant, error = error, tol = tol,
    status = if (is.na(error)) "not run" else if (error <= tol) "pass" else "FAIL",
    note = note)
}
read_case <- function(d) {
  meta <- rd(d, "meta.csv")
  widths <- as.integer(strsplit(meta$widths, " ")[[1]])
  sb <- isTRUE(meta$self_branch == 1L)
  m <- thangld_graph_kan(widths, degree = 0, self_branch = sb)
  dims <- rd(d, "dims.csv")
  pr <- rd(d, "params.csv")
  m$params <- lapply(seq_len(nrow(dims)), function(l) {
    dd <- as.integer(strsplit(dims$dims[l], " ")[[1]])
    list(coef = array(pr$value[pr$layer == l - 1L][order(pr$pos[pr$layer == l - 1L])], dd))
  })
  m$specs <- lapply(m$params, function(p) list(basis = "chebyshev",
                                               degree = dim(p$coef)[3] - 1L,
                                               self_branch = sb))
  e <- rd(d, "edges.csv")
  xl <- rd(d, "x.csv")
  x <- matrix(0, meta$n, widths[1])
  x[cbind(xl$row + 1L, xl$col + 1L)] <- xl$value
  nd <- rd(d, "nodes.csv")
  list(model = m, edges = cbind(e$from, e$to) + 1L, n = meta$n, x = x, y = rd(d, "y.csv")$y,
       a = sort(nd$node[nd$set == "a"]) + 1L, b = sort(nd$node[nd$set == "b"]) + 1L)
}
variants <- c("opt", "unopt")

# 1. net fixtures: plain ("net", Python output in py/) and self branch
#    ("net_self", Python output in py/self/)
net_checks <- function(case, pyd, tag) {
  add <- function(level, check, ...) add0(level, paste0(tag, check), ...)
  f <- read_case(file.path(fix, case))
  ah <- thangld_graph_kan_adjacency(f$edges, n = f$n)
  add("graph", "normalized adjacency, R vs NumPy", "numpy",
      max(abs(as.vector(ah) - vec(pyd, "net_a_hat.csv"))), 1e-12)
  outs <- thangld_graph_kan_forward_r(f$model, f$x, ah, layers = TRUE)
  g <- thangld_graph_kan_gradient_r(f$model, f$x, ah, f$y, f$a)
  st <- lapply(f$model$params, function(p) list(t = 0L, m = 0 * p$coef, v = 0 * p$coef))
  pars <- f$model$params
  adam <- list()
  for (s in 1:5) {
    gg <- thangldKANr:::graph_gradient(pars, f$model$specs, f$x, ah, f$y, f$a)
    for (l in seq_along(pars)) {
      r <- thangldKANr:::adam_step_r(pars[[l]]$coef, gg$params[[l]]$coef, st[[l]], 0.02,
                                   c(0.9, 0.999), 1e-8)
      pars[[l]]$coef <- r$param
      st[[l]] <- r$state
    }
    adam[[s]] <- unlist(lapply(pars, function(p) as.vector(p$coef)))
  }
  tm <- if (use_torch) list(
    fwd = thangldKANr:::graph_forward_torch(f$model$params, f$model$specs, f$x, ah),
    grad = thangldKANr:::graph_gradient_torch(f$model$params, f$model$specs, f$x, ah, f$y, f$a))
  for (v in variants) {
    for (l in seq_along(outs)) {
      add("forward", sprintf("layer %d output, R oracle vs Python", l), v,
          se(as.vector(outs[[l]]), vec(pyd, sprintf("net_%s_layer%d.csv", v, l - 1L))), 1e-8)
    }
    add("gradient", "coefficient gradient, R oracle vs Python", v,
        se(unlist(lapply(g$params, function(p) as.vector(p$coef))),
           vec(pyd, sprintf("net_%s_grad_params.csv", v))), 1e-8)
    add("gradient", "node-feature gradient, R oracle vs Python", v,
        se(as.vector(g$x), vec(pyd, sprintf("net_%s_grad_x.csv", v))), 1e-8)
    ap <- rd(pyd, sprintf("adam_%s_params.csv", v))
    add("adam", "5 full-batch Adam steps, R vs torch.optim.Adam", v,
        max(vapply(1:5, function(s) se(adam[[s]], ap$value[ap$step == s - 1L][
          order(ap$pos[ap$step == s - 1L])]), numeric(1))), 1e-8)
    if (use_torch) {
      add("forward", "R torch vs Python (last layer)", v,
          se(tm$fwd, vec(pyd, sprintf("net_%s_layer%d.csv", v, length(outs) - 1L))), 1e-8)
      add("gradient", "R torch autograd vs Python", v,
          se(unlist(lapply(tm$grad$params, function(p) as.vector(p$coef))),
             vec(pyd, sprintf("net_%s_grad_params.csv", v))), 1e-8)
    } else {
      add("forward", "R torch vs Python", v, NA, 1e-8, "torch runtime not installed")
    }
  }

}
add0 <- add
net_checks("net", py, "")
if (dir.exists(file.path(fix, "net_self"))) net_checks("net_self", file.path(py, "self"), "self branch: ")

# 2. training runs
st <- rd(fix, "train_settings.csv")
runs <- rd(fix, "train_runs.csv")
tim <- rd(py, "train_timing.csv")
met <- list()
for (k in seq_len(nrow(runs))) {
  id <- runs$id[k]
  cs <- read_case(file.path(fix, "train", id))
  pd <- file.path(py, "train", id)
  for (b in c("r", if (use_torch) "torch")) {
    fit <- thangld_graph_kan_fit(cs$model, cs$x, cs$edges, cs$y, train = cs$a, val = cs$b,
                         epochs = st$epochs, lr = st$lr, betas = c(st$beta1, st$beta2),
                         eps = st$eps, patience = st$patience, backend = b)
    sc <- rd(pd, "scaling.csv")
    pp <- fit$preprocess
    add("training", sprintf("%s (R %s) training-node scaling", id, b), "numpy",
        se(c(pp$x_center, pp$x_scale, pp$y_center, pp$y_scale), sc$value), 1e-12)
    pr <- predict(fit)
    te <- setdiff(seq_len(cs$n), c(cs$a, cs$b))
    for (v in variants) {
      h <- rd(pd, sprintf("%s_history.csv", v))
      tv <- tim[tim$id == id & tim$variant == v, ]
      add("training", sprintf("%s (R %s) train/val loss histories", id, b), v,
          if (nrow(h) != nrow(fit$history)) Inf else
            se(c(fit$history$train_loss, fit$history$val_loss), c(h$train_loss, h$val_loss)),
          1e-6, sprintf("epochs R %d / Python %d", nrow(fit$history), nrow(h)))
      add("training", sprintf("%s (R %s) best-epoch coefficients", id, b), v,
          se(unlist(lapply(fit$params, function(p) as.vector(p$coef))),
             vec(pd, sprintf("%s_best_params.csv", v))), 1e-6,
          sprintf("best epoch R %d / Python %d", fit$train$best_epoch, tv$best_epoch))
      pv <- vec(pd, sprintf("%s_predictions.csv", v))
      add("training", sprintf("%s (R %s) predictions, all nodes", id, b), v, se(pr, pv), 1e-6)
      if (b == "r") met[[length(met) + 1]] <- data.frame(id = id, impl = paste0("ref-", v),
        best_epoch = tv$best_epoch, fit_sec = tv$fit_sec,
        rmse_test = sqrt(mean((pv[te] - cs$y[te])^2)))
    }
    met[[length(met) + 1]] <- data.frame(id = id, impl = paste0("thangldKANr-", b),
      best_epoch = fit$train$best_epoch, fit_sec = fit$train$elapsed_sec,
      rmse_test = sqrt(mean((pr[te] - cs$y[te])^2)))
  }
  if (!use_torch) add("training", paste(id, "(R torch backend)"), "all", NA, 1e-6,
                      "torch runtime not installed")
}
res <- do.call(rbind, rows)
utils::write.csv(res, file.path(rep_dir, "comparison_checks.csv"), row.names = FALSE)
met <- do.call(rbind, met)
utils::write.csv(met, file.path(rep_dir, "train_metrics.csv"), row.names = FALSE)
sink(file.path(rep_dir, "comparison_summary.txt"))
cat("thangldKANr vs Python (NumPy A_hat + ChebyKAN + torch Adam) comparison\nDate:",
    format(Sys.time()), "\n")
cat(R.version.string, "| torch runtime:", use_torch, "\n")
cat("Python env:", paste(readLines(file.path(py, "python_env.json"), warn = FALSE),
                         collapse = " "), "\n\n")
print(table(res$level, res$status))
cat("\nFailures / not run:\n")
print(res[res$status != "pass", c("level", "check", "variant", "error", "note")], row.names = FALSE)
ok <- !is.na(res$error) & is.finite(res$error)
cat("\nMax error per level and variant:\n")
print(stats::aggregate(error ~ level + variant, data = res[ok, ], FUN = max))
cat("\nTest-node RMSE (24 test nodes), median over 5 seeds:\n")
met$target <- sub("_s[0-9]+$", "", met$id)
print(stats::aggregate(cbind(rmse_test, fit_sec) ~ target + impl, data = met, FUN = stats::median))
sink()
cat(readLines(file.path(rep_dir, "comparison_summary.txt")), sep = "\n")
