# Need the torch runtime; skipped (and counted as skipped) without it.
fx <- thangldKANr:::validation_fixture()

test_that("namespace loads; torch-free paths work without the runtime", {
  if (!thangldKANr:::torch_ready()) {
    expect_error(thangld_kan_fit(fx$model, fx$x, fx$y, backend = "torch"), "torch")
  }
  expect_length(thangld_kan_forward_r(fx$model, fx$x), nrow(fx$x))
})

test_that("torch forward and autograd equal the oracle (float64, 1e-8)", {
  skip_if_no_torch()
  m <- fx$model
  expect_lt(serr(thangldKANr:::engine_forward_torch(m$params, m$specs, fx$x),
                 thangld_kan_forward_r(m, fx$x)), 1e-8)
  g <- thangld_kan_gradient_r(m, fx$x, fx$y)
  gt <- thangldKANr:::engine_gradient_torch(m$params, m$specs, fx$x, fx$y)
  expect_lt(abs(gt$loss - g$loss), 1e-12)
  expect_lt(serr(unlist(gt$params), unlist(g$params)), 1e-8)
  expect_lt(serr(gt$x, g$x), 1e-8)
})

test_that("torch and r backends give the same Adam trajectory", {
  skip_if_no_torch()
  set.seed(15)
  xs <- matrix(runif(60, -2, 2), 30, 2)
  ys <- sin(xs[, 1]) + xs[, 2]^2
  m0 <- thangld_kan(c(2, 3, 1), seed = 2)
  args <- list(m0, xs, ys, epochs = 25, batch_size = 7, lr = 0.02, seed = 3,
               validation_split = 0.2, patience = 100)
  fr <- do.call(thangld_kan_fit, c(args, backend = "r"))
  ft <- do.call(thangld_kan_fit, c(args, backend = "torch"))
  expect_lt(serr(unlist(ft$params), unlist(fr$params)), 1e-8)
  expect_lt(serr(ft$history$train_loss, fr$history$train_loss), 1e-8)
  expect_lt(serr(ft$history$val_loss, fr$history$val_loss), 1e-8)
  expect_equal(ft$train$best_epoch, fr$train$best_epoch)
  expect_lt(serr(predict(ft, xs, backend = "torch"), predict(fr, xs)), 1e-8)
})

test_that("float32 torch forward is close to float64", {
  skip_if_no_torch()
  m <- fx$model
  expect_lt(serr(thangldKANr:::engine_forward_torch(m$params, m$specs, fx$x,
                                              "float32"),
                 thangld_kan_forward_r(m, fx$x)), 1e-4)
})

test_that("validation suite with torch reports no skipped checks", {
  skip_if_no_torch()
  res <- thangld_kan_validate("both")
  expect_equal(attr(res, "n_skipped"), 0L)
  expect_true(attr(res, "all_pass"))
})
