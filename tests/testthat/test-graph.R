test_that("normalized adjacency: formula, inputs, isolated nodes, errors", {
  e <- rbind(c(1, 2), c(2, 3), c(2, 3), c(3, 3))
  a <- thangld_graph_kan_adjacency(e, n = 4)
  a1 <- matrix(0, 4, 4)
  a1[1, 2] <- a1[2, 1] <- a1[2, 3] <- a1[3, 2] <- 1
  a1 <- a1 + diag(4)
  d <- rowSums(a1)
  expect_equal(a, a1 / sqrt(outer(d, d)))
  expect_equal(a[4, ], c(0, 0, 0, 1))                 # isolated node keeps itself
  expect_equal(thangld_graph_kan_adjacency(a1 - diag(4)), a)
  expect_equal(thangld_graph_kan_adjacency(as.data.frame(e), n = 4), a)
  b <- thangld_graph_kan_adjacency(e, n = 4, self_loops = FALSE)
  expect_equal(b[4, ], rep(0, 4))
  expect_true(isSymmetric(b))
  expect_error(thangld_graph_kan_adjacency(matrix(c(0, 1, 0, 0), 2)), "symmetric")
  expect_error(thangld_graph_kan_adjacency(matrix(-1, 3, 3)), "non-negative")
  expect_error(thangld_graph_kan_adjacency(cbind(c(0, 1, 2), c(1, 2, 3))), "indices")
  expect_error(thangld_graph_kan_adjacency(cbind(1:3, 2:4), n = 3), "beyond")
  expect_error(thangld_graph_kan_adjacency(matrix(NA_real_, 3, 3)), "numeric")
})

test_that("simulator is reproducible and leaves the caller's RNG alone", {
  set.seed(99)
  before <- stats::runif(1)
  set.seed(99)
  g1 <- thangld_graph_kan_simulate(40, seed = 5)
  after <- stats::runif(1)
  expect_identical(before, after)
  g2 <- thangld_graph_kan_simulate(40, seed = 5)
  expect_identical(g1, g2)
  expect_true(isSymmetric(g1$adjacency))
  expect_equal(sum(diag(g1$adjacency)), 0)
})

test_that("architecture and fit argument checks", {
  expect_error(thangld_graph_kan(c(2, 2)), "last width")
  expect_error(thangld_graph_kan(3), "two positive")
  expect_error(thangld_graph_kan(c(2, 3, 1), degree = c(1, 2, 3)), "degree")
  g <- thangld_graph_kan_simulate(20, seed = 1)
  m <- thangld_graph_kan(c(2, 1), seed = 1)
  y <- rnorm(20)
  expect_error(thangld_graph_kan_fit(m, g$x[, 1, drop = FALSE], g$adjacency, y, 1:10, backend = "r"),
               "columns")
  expect_error(thangld_graph_kan_fit(m, g$x, g$adjacency, y[-1], 1:10, backend = "r"), "one value")
  expect_error(thangld_graph_kan_fit(m, g$x, g$adjacency, y, 1, backend = "r"), "two training")
  expect_error(thangld_graph_kan_fit(m, g$x, g$adjacency, y, 1:10, 10:12, backend = "r"), "overlap")
  expect_error(thangld_graph_kan_fit(m, g$x, g$adjacency, y, c(1, 1, 2), backend = "r"), "distinct")
  y2 <- y
  y2[3] <- NA
  expect_error(thangld_graph_kan_fit(m, g$x, g$adjacency, y2, 1:10, backend = "r"), "finite")
  x2 <- g$x
  x2[1, 1] <- Inf
  expect_error(thangld_graph_kan_fit(m, x2, g$adjacency, y, 1:10, backend = "r"))
  expect_error(thangld_graph_kan_fit(m, g$x, g$adjacency[1:10, 1:10], y, 1:10, backend = "r"),
               "does not match")
  expect_error(predict(m), "not fitted")
})

test_that("labels outside train/val are never used; scaling uses training nodes", {
  g <- thangld_graph_kan_simulate(40, seed = 2)
  y <- g$x[, 1] + rnorm(40, sd = 0.1)
  m <- thangld_graph_kan(c(2, 3, 1), seed = 1)
  f1 <- thangld_graph_kan_fit(m, g$x, g$adjacency, y, 1:25, 26:32, epochs = 20, lr = 0.02,
                      backend = "r")
  y2 <- y
  y2[33:40] <- c(1e6, NA, -5, 0, 3, NA, 2, 7)
  f2 <- thangld_graph_kan_fit(m, g$x, g$adjacency, y2, 1:25, 26:32, epochs = 20, lr = 0.02,
                      backend = "r")
  expect_identical(f1$params, f2$params)
  expect_identical(predict(f1), predict(f2))
  x3 <- g$x
  x3[33:40, ] <- x3[33:40, ] * 50
  f3 <- thangld_graph_kan_fit(m, x3, g$adjacency, y, 1:25, 26:32, epochs = 20, lr = 0.02,
                      backend = "r")
  expect_identical(f3$preprocess, f1$preprocess)
  expect_equal(f1$preprocess$y_center, mean(y[1:25]))
})

test_that("constant feature is centered with a warning; one-feature graphs work", {
  g <- thangld_graph_kan_simulate(30, seed = 3)
  x <- cbind(g$x[, 1], 2)
  expect_warning(f <- thangld_graph_kan_fit(thangld_graph_kan(c(2, 1), seed = 1), x, g$edges,
                                    g$x[, 1], 1:20, epochs = 5, backend = "r"), "Constant")
  expect_true(all(is.finite(predict(f))))
  f1 <- thangld_graph_kan_fit(thangld_graph_kan(c(1, 1), seed = 1), g$x[, 1], g$edges, g$x[, 2],
                      1:20, epochs = 5, backend = "r")
  expect_length(predict(f1), 30)
})

test_that("predictions are permutation equivariant and match a new-graph predict", {
  g <- thangld_graph_kan_simulate(30, seed = 4)
  y <- g$x[, 2]
  f <- thangld_graph_kan_fit(thangld_graph_kan(c(2, 2, 1), seed = 1), g$x, g$adjacency, y, 1:20,
                     epochs = 10, lr = 0.02, backend = "r")
  p <- sample(30)
  expect_equal(predict(f, g$x[p, ], g$adjacency[p, p]), predict(f)[p], tolerance = 1e-12)
  expect_equal(predict(f, g$x, g$edges), predict(f), tolerance = 1e-12)
})

test_that("without a graph the model is a node-wise KAN", {
  g <- thangld_graph_kan_simulate(25, seed = 5)
  f <- thangld_graph_kan_fit(thangld_graph_kan(c(2, 1), seed = 1), g$x, NULL, g$x[, 1], 1:15,
                     epochs = 10, backend = "r")
  expect_false(f$train$graph)
  expect_equal(f$a_hat, diag(25))
  xs <- sweep(sweep(g$x, 2, f$preprocess$x_center), 2, f$preprocess$x_scale, `/`)
  manual <- drop(thangldKANr:::layer_naive_forward(xs, f$params[[1]], f$specs[[1]]))
  expect_equal(predict(f), manual * f$preprocess$y_scale + f$preprocess$y_center)
})

test_that("early stopping restores the best validation epoch; save/load", {
  g <- thangld_graph_kan_simulate(60, seed = 6)
  y <- sin(g$x[, 1]) + rnorm(60, sd = 0.3)
  f <- thangld_graph_kan_fit(thangld_graph_kan(c(2, 4, 1), seed = 1), g$x, g$adjacency, y, 1:30, 31:45,
                     epochs = 300, lr = 0.05, patience = 20, backend = "r")
  h <- f$history
  expect_equal(f$train$best_epoch, which.min(h$val_loss))
  expect_lt(nrow(h), 300)
  tf <- tempfile(fileext = ".rds")
  thangld_graph_kan_save(f, tf)
  expect_identical(predict(thangld_graph_kan_load(tf)), predict(f))
  saveRDS(list(a = 1), tf)
  expect_error(thangld_graph_kan_load(tf), "Not a thangldKANr")
  expect_output(print(f), "Fitted on 60 nodes")
})

# Threshold fixed before the first run (2026-10-30): on an SBM graph whose
# target is an additive function of one-hop neighborhood means, a ONE-layer
# graph KAN (which is exactly that model class) reaches test R^2 >= 0.85 and
# test RMSE below 0.8 x the RMSE of the same KAN without the graph.
# History: the first pre-registered version of this check used a TWO-layer
# graph KAN c(2, 4, 1) and failed (test R^2 0.823 < 0.85); it is kept, with
# its threshold, as a recorded result in inst/benchmarks/neighborhood_check.R
# and in the verification report, not deleted or loosened.
test_that("one-hop target: 1-layer graph KAN R2 >= 0.85 and beats the graph-free KAN", {
  g <- thangld_graph_kan_simulate(300, p_in = 0.08, p_out = 0.005, seed = 1)
  a <- thangld_graph_kan_adjacency(g$adjacency)
  u <- a %*% g$x
  y <- sin(1.5 * u[, 1]) + 0.5 * u[, 2]^2 + stats::rnorm(300, sd = 0.05)
  fitg <- function(graph) thangld_graph_kan_fit(thangld_graph_kan(c(2, 1), degree = 5, seed = 1), g$x,
                                        graph, y, train = 1:180, val = 181:240,
                                        epochs = 600, lr = 0.02, patience = 60,
                                        backend = "r")
  te <- 241:300
  mg <- thangld_graph_kan_metrics(y[te], predict(fitg(g$adjacency))[te])
  m0 <- thangld_graph_kan_metrics(y[te], predict(fitg(NULL))[te])
  expect_gte(mg[["r2"]], 0.85)
  expect_lt(mg[["rmse"]], 0.8 * m0[["rmse"]])
})

# Pre-registered 2026-10-31 BEFORE the first run, after adding `self_branch`:
# the original two-layer threshold (R^2 >= 0.85, RMSE < 0.8 x graph-free KAN),
# unchanged, now for the two-layer self-branch graph KAN; and R^2 >= 0.85 on a
# target that uses the node's own features only.
test_that("self branch: shapes, input [H, A_hat H], argument check", {
  m <- thangld_graph_kan(c(3, 4, 1), degree = 2, self_branch = TRUE, seed = 1)
  expect_equal(dim(m$params[[1]]$coef), c(6L, 4L, 3L))
  expect_equal(dim(m$params[[2]]$coef), c(8L, 1L, 3L))
  expect_error(thangld_graph_kan(c(2, 1), self_branch = NA), "self_branch")
  g <- thangld_graph_kan_simulate(10, seed = 4)
  a <- thangld_graph_kan_adjacency(g$adjacency)
  m1 <- thangld_graph_kan(c(2, 1), degree = 3, self_branch = TRUE, seed = 2)
  ex <- thangldKANr:::layer_forward_r(cbind(g$x, a %*% g$x), m1$params[[1]], m1$specs[[1]])$out
  expect_lt(max(abs(thangld_graph_kan_forward_r(m1, g$x, a) - drop(ex))), 1e-12)
  expect_output(print(m1), "\\[H, A_hat H\\]")
})

test_that("self branch: 2-layer graph KAN meets the original one-hop threshold", {
  g <- thangld_graph_kan_simulate(300, p_in = 0.08, p_out = 0.005, seed = 1)
  a <- thangld_graph_kan_adjacency(g$adjacency)
  u <- a %*% g$x
  y <- sin(1.5 * u[, 1]) + 0.5 * u[, 2]^2 +
    thangldKANr:::with_seed(42, stats::rnorm(300, sd = 0.05))
  fitg <- function(graph) thangld_graph_kan_fit(thangld_graph_kan(c(2, 4, 1), degree = 3, self_branch = TRUE,
                                                  seed = 1), g$x, graph, y, train = 1:180,
                                        val = 181:240, epochs = 600, lr = 0.02,
                                        patience = 60, backend = "r")
  te <- 241:300
  mg <- thangld_graph_kan_metrics(y[te], predict(fitg(g$adjacency))[te])
  m0 <- thangld_graph_kan_metrics(y[te], predict(fitg(NULL))[te])
  expect_gte(mg[["r2"]], 0.85)
  expect_lt(mg[["rmse"]], 0.8 * m0[["rmse"]])
})

test_that("self branch: 2-layer graph KAN fits an own-feature target (R2 >= 0.85)", {
  g <- thangld_graph_kan_simulate(300, p_in = 0.08, p_out = 0.005, seed = 1)
  y <- g$x[, 1] * g$x[, 2] + thangldKANr:::with_seed(43, stats::rnorm(300, sd = 0.05))
  fit <- thangld_graph_kan_fit(thangld_graph_kan(c(2, 4, 1), degree = 3, self_branch = TRUE, seed = 1),
                       g$x, g$adjacency, y, train = 1:180, val = 181:240, epochs = 600,
                       lr = 0.02, patience = 60, backend = "r")
  expect_gte(thangld_graph_kan_metrics(y[241:300], predict(fit)[241:300])[["r2"]], 0.85)
})

test_that("validation suite: base R checks pass", {
  v <- thangld_graph_kan_validate("r")
  expect_true(all(v$pass))
})

test_that("torch backend equals the r backend (needs runtime)", {
  skip_if_no_torch()
  v <- thangld_graph_kan_validate("torch")
  expect_true(all(v$pass %in% TRUE))
  expect_equal(attr(v, "n_skipped"), 0)
})
