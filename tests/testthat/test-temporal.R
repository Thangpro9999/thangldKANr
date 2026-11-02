test_that("lag windows use only past values, with and without differencing", {
  w <- thangld_temporal_kan_embed(1:10, lags = c(1, 3))
  expect_equal(w$index, 4:10)
  expect_equal(unname(w$x[1, ]), c(3, 1))
  expect_equal(w$target, 4:10)
  y <- c(2, 5, 4, 8, 7, 11)
  wd <- thangld_temporal_kan_embed(y, lags = 1, difference = TRUE)
  expect_equal(wd$index, 3:6)
  expect_equal(unname(wd$x[, 1]), diff(y)[1:4])
  expect_equal(wd$target, diff(y)[2:5])
  wl <- thangld_temporal_kan_embed(exp(1:6), lags = 2, log = TRUE)
  expect_equal(wl$target, 3:6)
  expect_error(thangld_temporal_kan_embed(c(1, -1, 2, 3), 1, log = TRUE), "positive")
  expect_error(thangld_temporal_kan_embed(1:3, lags = 5), "too short")
  expect_error(thangld_temporal_kan_embed(c(1, NA, 3), 1), "NA")
})

test_that("architecture and argument checks", {
  m <- thangld_temporal_kan(lags = c(12, 1, 2), hidden = c(4, 3), seed = 1)
  expect_equal(m$lags, c(1L, 2L, 12L))
  expect_equal(m$widths, c(3L, 4L, 3L, 1L))
  expect_error(thangld_temporal_kan(lags = c(1, 1)), "distinct")
  expect_error(thangld_temporal_kan(lags = 0), "positive")
  expect_error(thangld_temporal_kan(hidden = -1), "hidden")
  expect_output(print(m), "lag-window")
})

sim_nlar <- function(n, seed) {
  set.seed(seed)
  y <- numeric(n)
  for (t in 3:n) y[t] <- 0.5 * y[t - 1] - 0.8 * tanh(2 * y[t - 2]) + rnorm(1, sd = 0.3)
  y
}

test_that("scaling is estimated on training windows only (no leakage)", {
  y <- sim_nlar(120, 1)
  fit <- thangld_temporal_kan_fit(thangld_temporal_kan(1:2, seed = 1), y, n_val = 20, epochs = 3,
                          backend = "r", seed = 1)
  w <- thangld_temporal_kan_embed(y, 1:2)
  tr <- seq_len(nrow(w$x) - 20)
  expect_equal(fit$scaling$center, mean(w$target[tr]))
  expect_equal(fit$scaling$scale, sd(w$target[tr]))
  # changing the validation part does not change the scaling
  y2 <- y
  y2[110:120] <- y2[110:120] + 100
  fit2 <- thangld_temporal_kan_fit(thangld_temporal_kan(1:2, seed = 1), y2, n_val = 20,
                           epochs = 1, backend = "r", seed = 1)
  expect_equal(fit2$scaling, fit$scaling)
})

test_that("rolling-origin forecasts never use values after their origin", {
  y <- sim_nlar(90, 2)
  args <- list(thangld_temporal_kan(1:2, seed = 1), initial = 70, refit_every = 5,
               n_val = 10, epochs = 5, lr = 0.02, backend = "r", seed = 1)
  r1 <- do.call(thangld_temporal_kan_rolling, c(list(y = y), args))
  y2 <- y
  y2[81:90] <- rnorm(10, 50)            # change the future after origin 80
  r2 <- do.call(thangld_temporal_kan_rolling, c(list(y = y2), args))
  keep <- r1$forecasts$origin <= 80
  expect_equal(r1$forecasts$kan[keep], r2$forecasts$kan[keep])
  expect_equal(r1$forecasts$naive[keep], r2$forecasts$naive[keep])
  expect_equal(nrow(r1$forecasts), 20L)
  expect_equal(sum(r1$forecasts$refit), 4L)
})

test_that("naive and seasonal naive baselines and MASE are correct", {
  y <- ts(sin(2 * pi * (1:60) / 12) + (1:60) / 30, frequency = 12)
  r <- thangld_temporal_kan_rolling(thangld_temporal_kan(c(1, 12), seed = 1), y, initial = 48,
                            refit_every = 12, epochs = 2, backend = "r", seed = 1)
  f <- r$forecasts
  yy <- as.numeric(y)
  expect_equal(f$naive, yy[f$origin])
  expect_equal(f$snaive, yy[f$index - 12])
  sc <- mean(abs(diff(yy[1:48], lag = 12)))
  expect_equal(r$metrics$mase[r$metrics$method == "snaive"],
               mean(abs(f$snaive - f$actual)) / sc)
  expect_equal(r$period, 12L)
})

test_that("forecast inverts log and differencing; only h = 1", {
  y <- exp(cumsum(c(1, rep(0.05, 39))))
  fit <- thangld_temporal_kan_fit(thangld_temporal_kan(1:2, difference = TRUE, log = TRUE,
                                       seed = 1), y, epochs = 200, lr = 0.05,
                          backend = "r", seed = 1)
  expect_equal(predict(fit), y[40] * exp(0.05), tolerance = 1e-3)
  expect_error(predict(fit, h = 2), "one-step")
  expect_error(predict(thangld_temporal_kan(1:2)), "not fitted")
})

# Threshold fixed before the first run: nonlinear AR(2), noise sd 0.3;
# rolling-origin RMSE of the KAN <= 0.40 and below the naive baseline.
test_that("nonlinear AR: rolling one-step RMSE <= 0.40 and beats naive", {
  y <- sim_nlar(260, 3)
  r <- thangld_temporal_kan_rolling(thangld_temporal_kan(1:2, hidden = 3, seed = 1), y,
                            initial = 200, refit_every = 20, n_val = 30,
                            epochs = 150, batch_size = 32, lr = 0.02,
                            patience = 30, backend = "r", seed = 1)
  mt <- r$metrics
  expect_lte(mt$rmse[mt$method == "kan"], 0.40)
  expect_lt(mt$rmse[mt$method == "kan"], mt$rmse[mt$method == "naive"])
})

test_that("save/load round trip and validation suite", {
  y <- sim_nlar(80, 4)
  fit <- thangld_temporal_kan_fit(thangld_temporal_kan(1:2, seed = 1), y, epochs = 5,
                          backend = "r", seed = 1)
  f <- tempfile(fileext = ".rds")
  on.exit(unlink(f))
  thangld_temporal_kan_save(fit, f)
  expect_equal(predict(thangld_temporal_kan_load(f)), predict(fit))
  saveRDS(1, f)
  expect_error(thangld_temporal_kan_load(f), "Not a thangldKANr")
  res <- thangld_temporal_kan_validate("r")
  expect_true(all(res$pass))
})

test_that("seed reproducibility and caller RNG untouched", {
  y <- sim_nlar(80, 5)
  set.seed(42)
  before <- .Random.seed
  a <- thangld_temporal_kan_fit(thangld_temporal_kan(1:2, seed = 1), y, epochs = 5,
                        batch_size = 8, backend = "r", seed = 2)
  expect_identical(.Random.seed, before)
  b <- thangld_temporal_kan_fit(thangld_temporal_kan(1:2, seed = 1), y, epochs = 5,
                        batch_size = 8, backend = "r", seed = 2)
  expect_identical(a$params, b$params)
})

test_that("torch backend equals the r backend (needs runtime)", {
  if (!thangldKANr:::torch_ready()) {
    expect_error(thangld_temporal_kan_fit(thangld_temporal_kan(1:2), sim_nlar(50, 1)), "torch")
  }
  skip_if_no_torch()
  res <- thangld_temporal_kan_validate("both")
  expect_equal(attr(res, "n_skipped"), 0L)
  expect_true(attr(res, "all_pass"))
  y <- sim_nlar(100, 6)
  fit <- thangld_temporal_kan_fit(thangld_temporal_kan(1:2, hidden = 2, seed = 1), y, n_val = 20,
                          epochs = 20, lr = 0.02, backend = "r", seed = 1)
  ft <- thangld_temporal_kan_fit(thangld_temporal_kan(1:2, hidden = 2, seed = 1), y, n_val = 20,
                         epochs = 20, lr = 0.02, backend = "torch", seed = 1)
  expect_lt(serr(predict(ft, backend = "torch"), predict(fit)), 1e-8)
})
