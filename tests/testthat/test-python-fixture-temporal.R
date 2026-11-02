# Bundled ChebyKAN-on-lag-windows reference (log lynx, seed 1, lags 1:3, 150
# Adam epochs). The Python side built the windows, the training-only scaling
# and the training loop independently from the raw series. Thresholds were
# fixed before the first run: windows 1e-12, loss history / coefficients /
# forecast scaled error 1e-6.
ref <- function(f) utils::read.csv(test_path("fixtures", "temporal", f))

reference_fit <- function(backend) {
  st <- ref("train_settings.csv")
  y <- ref("series.csv")$y
  m0 <- thangld_temporal_kan(1:3, seed = 1)
  thangld_temporal_kan_fit(m0, y, n_val = st$n_val, epochs = st$epochs,
                   batch_size = st$batch_size, lr = st$lr,
                   betas = c(st$beta1, st$beta2), eps = st$eps,
                   seed = 1, backend = backend, verbose = FALSE)
}

check_against_python <- function(fit) {
  se <- thangldKANr:::scaled_error
  h <- ref("opt_history.csv")
  expect_lte(se(c(fit$history$train_loss, fit$history$val_loss),
                c(h$train_loss, h$val_loss)), 1e-6)
  expect_lte(se(unlist(lapply(fit$params, function(p) as.vector(p$coef))),
                ref("opt_best_params.csv")$value), 1e-6)
  expect_lte(se(predict(fit), ref("opt_forecast.csv")$forecast), 1e-6)
  sc <- ref("scaling.csv")
  expect_lte(se(c(fit$scaling$center, fit$scaling$scale), c(sc$center, sc$scale)), 1e-12)
}

test_that("lag windows match the Python construction", {
  w <- thangld_temporal_kan_embed(ref("series.csv")$y, 1:3)
  pw <- ref("windows.csv")
  expect_lte(max(abs(cbind(w$x, w$target) - as.matrix(pw[, -1]))), 1e-12)
})

test_that("base R training matches ChebyKAN on lag windows", {
  check_against_python(reference_fit("r"))
})

test_that("torch training matches ChebyKAN on lag windows", {
  skip_if_no_torch()
  check_against_python(reference_fit("torch"))
})
