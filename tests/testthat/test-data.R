make_xy <- function(n = 40, seed = 1) {
  set.seed(seed)
  x <- matrix(runif(2 * n, -2, 2), n, 2, dimnames = list(NULL, c("a", "b")))
  list(x = x, y = sin(x[, "a"]) + 0.3 * x[, "b"])
}
mk <- function(widths, seed = 1) thangld_kan(widths, seed = seed)

test_that("NA, Inf, wrong sizes and non-numeric input are rejected clearly", {
  d <- make_xy()
  m <- mk(c(2, 1))
  x_na <- d$x
  x_na[3, 1] <- NA
  expect_error(thangld_kan_fit(m, x_na, d$y, backend = "r"), "NA")
  x_inf <- d$x
  x_inf[2, 2] <- Inf
  expect_error(thangld_kan_fit(m, x_inf, d$y, backend = "r"), "infinite")
  y_na <- d$y
  y_na[1] <- NaN
  expect_error(thangld_kan_fit(m, d$x, y_na, backend = "r"), "NA")
  expect_error(thangld_kan_fit(m, d$x, d$y[-1], backend = "r"), "length")
  expect_error(thangld_kan_fit(m, d$x[, 1, drop = FALSE], d$y, backend = "r"),
               "expects 2")
  expect_error(thangld_kan_fit(m, data.frame(a = letters[1:40], b = 1:40), d$y,
                       backend = "r"), "numeric")
  expect_error(thangld_kan_fit(m, d$x, cbind(d$y, d$y), backend = "r"), "one column")
})

test_that("one variable and one-row prediction work", {
  set.seed(2)
  x <- runif(30, -1, 1)
  fit <- thangld_kan_fit(mk(c(1, 1)), x, x^2, epochs = 10, backend = "r")
  expect_length(predict(fit, c(-0.5, 0.5)), 2L)
  d <- make_xy()
  fit2 <- thangld_kan_fit(mk(c(2, 1)), d$x, d$y, epochs = 5, backend = "r")
  expect_length(predict(fit2, d$x[1, , drop = FALSE]), 1L)
  expect_length(predict(fit2, data.frame(a = 0.1, b = -0.2)), 1L)
})

test_that("one training row gives a warning and a usable model", {
  expect_warning(
    fit <- thangld_kan_fit(mk(c(2, 1)), matrix(c(1, 2), 1), 3, epochs = 5,
                   backend = "r"), "one training row")
  expect_true(is.finite(predict(fit, matrix(c(1, 2), 1))))
})

test_that("constant column is fixed at its training value and warned", {
  d <- make_xy()
  x <- cbind(d$x, c = 5)
  expect_warning(fit <- thangld_kan_fit(mk(c(3, 1)), x, d$y, epochs = 10,
                                backend = "r"), "Constant predictor")
  expect_equal(fit$preprocess$constant, c(FALSE, FALSE, TRUE))
  x2 <- x
  x2[, "c"] <- 100
  expect_equal(predict(fit, x2), predict(fit, x))
})

test_that("columns are matched by name; missing names are errors", {
  d <- make_xy()
  fit <- thangld_kan_fit(mk(c(2, 1)), d$x, d$y, epochs = 10, backend = "r")
  p <- predict(fit, d$x)
  expect_equal(predict(fit, d$x[, c("b", "a")]), p)
  expect_equal(predict(fit, as.data.frame(d$x)[, c("b", "a")]), p)
  expect_equal(predict(fit, cbind(extra = 1, d$x)), p)
  expect_error(predict(fit, d$x[, "a", drop = FALSE]), "missing columns")
  expect_warning(p2 <- predict(fit, unname(d$x)), "no column names")
  expect_equal(p2, p)
})

test_that("preprocessing uses training rows only; y is back-transformed", {
  d <- make_xy(60)
  fit <- thangld_kan_fit(mk(c(2, 1)), d$x[1:40, ], d$y[1:40],
                 validation_data = list(x = d$x[41:60, ], y = d$y[41:60]),
                 epochs = 5, backend = "r")
  expect_equal(fit$preprocess$x_center, unname(colMeans(d$x[1:40, ])))
  expect_equal(fit$preprocess$y_scale, sd(d$y[1:40]))
  xs <- thangldKANr:::apply_preprocess_x(fit$preprocess, d$x)
  raw <- thangld_kan_forward_r(fit, xs)
  expect_equal(predict(fit, d$x), raw * sd(d$y[1:40]) + mean(d$y[1:40]))
  f2 <- thangld_kan_fit(mk(c(2, 1)), d$x, d$y, validation_split = 0.25, epochs = 3,
                backend = "r", seed = 4)
  expect_equal(f2$train$n_train, 45L)
})

test_that("metrics, including undefined R2", {
  m <- thangld_kan_metrics(c(1, 2, 3), c(1, 2, 4))
  expect_equal(unname(m["r2"]), 0.5)
  expect_true(is.na(thangld_kan_metrics(c(2, 2), c(1, 3))["r2"]))
})
