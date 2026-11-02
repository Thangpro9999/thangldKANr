# Shared fixture for the thangldKANr <-> Python (NumPy + ChebyKAN) comparison -------
# Usage: Rscript make_fixture.R <out_dir>
# Writes edge lists (not the normalized adjacency: Python builds it itself),
# node features, targets, node masks and initial coefficients.
suppressPackageStartupMessages(library(thangldKANr))
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[1] else "fixture"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
fmt <- function(df) {
  for (j in seq_along(df)) if (is.double(df[[j]])) df[[j]] <- sprintf("%.17g", df[[j]])
  df
}
wcsv <- function(df, path) utils::write.csv(fmt(df), path, row.names = FALSE, quote = FALSE)
mat_long <- function(m) data.frame(row = rep(seq_len(nrow(m)) - 1L, ncol(m)),
                                   col = rep(seq_len(ncol(m)) - 1L, each = nrow(m)),
                                   value = as.vector(m))
write_case <- function(dir, model, edges, n, x, y, nodes_a, nodes_b = integer()) {
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  wcsv(do.call(rbind, lapply(seq_along(model$params), function(l) {
    a <- model$params[[l]]$coef
    data.frame(layer = l - 1L, name = "coef", pos = seq_along(a) - 1L, value = as.vector(a))
  })), file.path(dir, "params.csv"))
  utils::write.csv(do.call(rbind, lapply(seq_along(model$params), function(l)
    data.frame(layer = l - 1L, name = "coef",
               dims = paste(dim(model$params[[l]]$coef), collapse = " ")))),
    file.path(dir, "dims.csv"), row.names = FALSE, quote = FALSE)
  utils::write.csv(data.frame(widths = paste(model$widths, collapse = " "), n = n,
                              self_branch = as.integer(isTRUE(model$specs[[1]]$self_branch))),
                   file.path(dir, "meta.csv"), row.names = FALSE, quote = FALSE)
  utils::write.csv(data.frame(from = edges[, 1] - 1L, to = edges[, 2] - 1L),
                   file.path(dir, "edges.csv"), row.names = FALSE, quote = FALSE)
  wcsv(mat_long(x), file.path(dir, "x.csv"))
  wcsv(data.frame(y = y), file.path(dir, "y.csv"))
  utils::write.csv(data.frame(node = c(nodes_a, nodes_b) - 1L,
                              set = c(rep("a", length(nodes_a)), rep("b", length(nodes_b)))),
                   file.path(dir, "nodes.csv"), row.names = FALSE, quote = FALSE)
}

# 1. forward / gradient / Adam fixture (standardized features given directly)
f <- thangldKANr:::graph_validation_fixture()
write_case(file.path(out, "net"), f$model, f$edges, 7L, f$x, f$y, f$nodes)
fs <- thangldKANr:::graph_validation_fixture(self_branch = TRUE)
write_case(file.path(out, "net_self"), fs$model, fs$edges, 7L, fs$x, fs$y, fs$nodes)

# 2. training runs: 3 targets x 5 seeds on SBM graphs (Python standardizes itself)
settings <- data.frame(epochs = 200L, lr = 0.02, beta1 = 0.9, beta2 = 0.999,
                       eps = 1e-8, patience = 50L)
wcsv(settings, file.path(out, "train_settings.csv"))
runs <- list()
# *_self: the same data with the two-layer self-branch architecture
for (tg in c("one_hop", "two_hop", "own_features", "one_hop_self", "own_features_self"))
  for (s in 1:5) {
  g <- thangld_graph_kan_simulate(120, p_in = 0.1, p_out = 0.01, seed = 100 + s)
  a <- thangld_graph_kan_adjacency(g$adjacency)
  noise <- thangldKANr:::with_seed(200 + s, stats::rnorm(120, sd = 0.05))
  u1 <- a %*% g$x
  sb <- grepl("_self$", tg)
  y <- switch(sub("_self$", "", tg),
    one_hop = sin(1.5 * u1[, 1]) + 0.5 * u1[, 2]^2,
    two_hop = tanh((a %*% u1)[, 1] * u1[, 2]),
    own_features = g$x[, 1] * g$x[, 2]) + noise
  widths <- if (tg == "one_hop") c(2, 1) else c(2, 4, 1)
  m <- thangld_graph_kan(widths, degree = 3, self_branch = sb, seed = s)
  id <- sprintf("%s_s%d", tg, s)
  write_case(file.path(out, "train", id), m, g$edges, 120L, g$x, y, 1:72, 73:96)
  runs[[length(runs) + 1L]] <- data.frame(id = id, target = tg, seed = s)
}
utils::write.csv(do.call(rbind, runs), file.path(out, "train_runs.csv"),
                 row.names = FALSE, quote = FALSE)
cat("Fixture written to", out, "\n")
