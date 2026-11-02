u <- c(-4, -1.7, -1, -0.3, 0, 0.5, 1, 2.2, 5)

test_that("Mexican hat: explicit form, zeros, normalization, derivative", {
  C <- 2 / (sqrt(3) * pi^0.25)
  expect_lt(max(abs(mexican_hat(u) - C * (1 - u^2) * exp(-u^2 / 2))), 1e-10)
  expect_equal(mexican_hat(c(-1, 1)), c(0, 0))
  expect_equal(mexican_hat(u), mexican_hat(-u))
  expect_equal(stats::integrate(function(v) mexican_hat(v)^2, -Inf, Inf)$value,
               1, tolerance = 1e-8)
  expect_lt(abs(stats::integrate(mexican_hat, -Inf, Inf)$value), 1e-8)
  h <- 1e-6
  expect_lt(max(abs(mexican_hat_deriv(u) -
                      (mexican_hat(u + h) - mexican_hat(u - h)) / (2 * h))), 1e-6)
  expect_error(mexican_hat(NA_real_), "finite")
})

test_that("gradients wrt weight, translation, scale (log and raw), bias, input", {
  for (sp in c("log", "raw")) {
    fx <- thangldKANr:::validation_fixture("wavelet", scale_param = sp)
    g <- thangld_kan_gradient_r(fx$model, fx$x, fx$y)
    for (h in c(1e-4, 1e-5, 1e-6)) {
      fd <- thangld_kan_fd_gradient_r(fx$model, fx$x, fx$y, step = h, order = 4)
      for (nm in names(g$params[[1]])) {
        expect_lt(serr(unlist(lapply(g$params, `[[`, nm)),
                       unlist(lapply(fd$params, `[[`, nm))), 1e-5)
      }
      expect_lt(serr(g$x, fd$x), 1e-5)
    }
  }
})

test_that("log and raw scale parameterizations give the same function", {
  a <- thangldKANr:::validation_fixture("wavelet", "log")
  b <- a$model
  b$specs <- lapply(b$specs, function(s) { s$scale_param <- "raw"; s })
  b$params <- lapply(b$params, function(pl) {
    pl$scale <- exp(pl$log_scale)
    pl$log_scale <- NULL
    pl
  })
  expect_lt(serr(thangld_kan_forward_r(a$model, a$x), thangld_kan_forward_r(b, a$x)),
            1e-12)
  # chain rule: dL/dlog s = s dL/ds
  ga <- thangld_kan_gradient_r(a$model, a$x, a$y)
  gb <- thangld_kan_gradient_r(b, a$x, a$y)
  expect_lt(serr(ga$params[[1]]$log_scale,
                 gb$params[[1]]$scale * b$params[[1]]$scale), 1e-12)
})

test_that("raw scale reaching zero stops training with a clear error", {
  m <- thangld_kan(basis = "wavelet", c(1, 1), scale_param = "raw", seed = 1)
  m$params[[1]]$scale[] <- -1
  expect_error(thangldKANr:::validate_model(m), "non-positive")
})

test_that("initialization and layout follow Wav-KAN", {
  m <- thangld_kan(basis = "wavelet", c(30, 4, 1), seed = 1)
  p <- m$params[[1]]
  expect_equal(dim(p$weight), c(30L, 4L))
  expect_true(all(abs(p$weight) <= 1 / sqrt(30)))
  expect_equal(range(p$translation), c(0, 0))
  expect_equal(range(p$log_scale), c(0, 0))
  expect_named(thangld_kan(basis = "wavelet", c(2, 1), bias = FALSE)$params[[1]],
               c("weight", "translation", "log_scale"))
})

test_that("edge function equals w psi((x - t)/s)", {
  fx <- thangldKANr:::validation_fixture("wavelet")
  pl <- fx$model$params[[1]]
  e <- thangld_kan_edge(fx$model, 1, 2, 3, x = c(-1, 0, 2))
  s <- exp(pl$log_scale[2, 3])
  expect_equal(e$phi, pl$weight[2, 3] * mexican_hat((c(-1, 0, 2) -
                                                      pl$translation[2, 3]) / s))
})

# Local-variation target, threshold fixed before the first run.
test_that("localized oscillation: test R2 >= 0.90", {
  f <- function(x) sin(3 * x[, 1]) * exp(-x[, 1]^2) + 0.3 * x[, 2]
  sim <- function(n, seed) {
    set.seed(seed)
    x <- matrix(runif(2 * n, -2, 2), n, 2)
    list(x = x, y = f(x) + rnorm(n, sd = 0.05))
  }
  tr <- sim(300, 21)
  te <- sim(500, 121)
  fit <- thangld_kan_fit(thangld_kan(basis = "wavelet", c(2, 6, 1), seed = 1), tr$x[1:240, ],
                      tr$y[1:240],
                      validation_data = list(x = tr$x[241:300, ],
                                             y = tr$y[241:300]),
                      epochs = 400, batch_size = 32, lr = 0.02, patience = 50,
                      seed = 1, backend = "r")
  expect_gte(thangld_kan_metrics(te$y, predict(fit, te$x))[["r2"]], 0.90)
})
