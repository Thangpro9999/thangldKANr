# Acceptance thresholds fixed before the first run (see VERIFICATION_REPORT.md):
# independent test sets of 500 rows, base R backend.
sim <- function(f, n, seed) {
  set.seed(seed)
  x <- matrix(runif(2 * n, -2, 2), n, 2, dimnames = list(NULL, c("x1", "x2")))
  list(x = x, y = f(x) + rnorm(n, sd = 0.05))
}
run_case <- function(f, widths, epochs, seed) {
  tr <- sim(f, 300, seed)
  te <- sim(f, 500, seed + 100)
  m <- thangld_kan(widths, seed = 1)
  fit <- thangld_kan_fit(m, tr$x[1:240, ], tr$y[1:240],
                 validation_data = list(x = tr$x[241:300, ], y = tr$y[241:300]),
                 epochs = epochs, batch_size = 32, lr = 0.02, patience = 50,
                 seed = 1, backend = "r")
  list(fit = fit, metrics = thangld_kan_metrics(te$y, predict(fit, te$x)))
}

test_that("linear target: test R2 >= 0.99", {
  r <- run_case(function(x) 1 + 2 * x[, 1] - x[, 2], c(2, 1), 300, 1)
  expect_gte(r$metrics[["r2"]], 0.99)
})

test_that("additive target: test R2 >= 0.95", {
  r <- run_case(function(x) sin(x[, 1]) + 0.5 * x[, 2]^2, c(2, 1), 300, 2)
  expect_gte(r$metrics[["r2"]], 0.95)
})

test_that("smooth interaction: two layers reach test R2 >= 0.90", {
  r <- run_case(function(x) x[, 1] * x[, 2], c(2, 4, 1), 400, 3)
  expect_gte(r$metrics[["r2"]], 0.90)
})

test_that("training loss decreases and history is recorded", {
  tr <- sim(function(x) sin(x[, 1]), 100, 5)
  fit <- thangld_kan_fit(thangld_kan(c(2, 1), seed = 1), tr$x, tr$y, epochs = 50, lr = 0.02,
                 backend = "r", seed = 1)
  h <- fit$history
  expect_equal(nrow(h), 50L)
  expect_lt(h$train_loss[50], h$train_loss[1])
  expect_null(fit$train$best_epoch)
})

test_that("early stopping restores the best validation epoch", {
  tr <- sim(function(x) sin(x[, 1]), 60, 6)
  va <- sim(function(x) sin(x[, 1]), 60, 7)
  fit <- thangld_kan_fit(thangld_kan(c(2, 4, 1), seed = 1), tr$x, tr$y,
                 validation_data = list(x = va$x, y = va$y), epochs = 3000,
                 lr = 0.05, patience = 20, seed = 1, backend = "r")
  h <- fit$history
  expect_lt(nrow(h), 3000L)
  expect_equal(fit$train$best_epoch, which.min(h$val_loss))
  expect_equal(nrow(h) - fit$train$best_epoch, 20L)
  pp <- fit$preprocess
  vs <- mean((thangld_kan_forward_r(fit, thangldKANr:::apply_preprocess_x(pp, va$x)) -
                thangldKANr:::apply_preprocess_y(pp, va$y))^2)
  expect_equal(vs, min(h$val_loss), tolerance = 1e-12)
})

test_that("same seed -> identical fit; caller RNG untouched", {
  tr <- sim(function(x) x[, 1]^2, 80, 8)
  m <- thangld_kan(c(2, 1), seed = 1)
  set.seed(42)
  before <- .Random.seed
  a <- thangld_kan_fit(m, tr$x, tr$y, epochs = 10, batch_size = 16, seed = 9,
               backend = "r")
  expect_identical(.Random.seed, before)
  b <- thangld_kan_fit(m, tr$x, tr$y, epochs = 10, batch_size = 16, seed = 9,
               backend = "r")
  expect_identical(a$params, b$params)
})

test_that("Adam first step equals p - lr * g / (|g| + eps)", {
  s <- thangldKANr:::adam_step_r(c(1, -2), c(0.5, 0.1),
                           list(t = 0L, m = c(0, 0), v = c(0, 0)),
                           lr = 0.1, betas = c(0.9, 0.999), eps = 1e-8)
  expect_equal(s$param, c(1, -2) - 0.1 * c(0.5, 0.1) / (c(0.5, 0.1) + 1e-8))
})

test_that("fit argument checks", {
  tr <- sim(function(x) x[, 1], 20, 9)
  m <- thangld_kan(c(2, 1), seed = 1)
  expect_error(thangld_kan_fit(m, tr$x, tr$y, lr = 0, backend = "r"), "lr")
  expect_error(thangld_kan_fit(m, tr$x, tr$y, validation_split = 1, backend = "r"),
               "validation_split")
  expect_error(thangld_kan_fit(m, tr$x, tr$y, validation_data = list(x = tr$x),
                       backend = "r"), "validation_data")
})
