test_that("safe Pade function: explicit form, derivative, no poles", {
  x <- seq(-4, 4, length.out = 41)
  a <- c(0.2, 1, -0.3, 0.4)
  b <- c(-0.6, 0.9)
  ex <- (0.2 + x - 0.3 * x^2 + 0.4 * x^3) / (1 + abs(-0.6 * x + 0.9 * x^2))
  expect_lt(max(abs(rational(x, a, b) - ex)), 1e-12)
  xs <- x[abs(-0.6 * x + 0.9 * x^2) > 1e-3]   # away from the |Q| kink
  h <- 1e-5
  fd <- (rational(xs + h, a, b) - rational(xs - h, a, b)) / (2 * h)
  expect_lt(max(abs(rational_deriv(xs, a, b) - fd)), 1e-6)
  expect_true(all(is.finite(rational(seq(-1e3, 1e3, by = 0.5), a, b))))
  expect_error(rational(1, c(1, 2), c(1, 2)), "den")
  expect_error(rational(NA_real_, a, b), "finite")
})

test_that("rational_kan argument checks and parameter layout", {
  m <- thangld_kan(basis = "rational", c(3, 2, 1), num_degree = c(4, 3), den_degree = c(3, 2), seed = 1)
  expect_equal(dim(m$params[[1]]$num), c(3L, 2L, 5L))
  expect_equal(dim(m$params[[1]]$den), c(3L, 2L, 3L))
  expect_equal(dim(m$params[[2]]$den), c(2L, 1L, 2L))
  expect_true(all(m$params[[1]]$den != 0))
  expect_error(thangld_kan(basis = "rational", c(2, 1), num_degree = 2, den_degree = 3), "exceed")
  expect_error(thangld_kan(basis = "rational", c(2, 1), num_degree = 0), "num_degree")
  expect_error(thangld_kan(basis = "rational", c(2, 1), den_init_sd = 0), "den_init_sd")
})

test_that("edge function equals the rational formula with its coefficients", {
  m <- thangld_kan(basis = "rational", c(2, 1), seed = 3)
  e <- thangld_kan_edge(m, layer = 1, input = 2, output = 1,
                         x = seq(-2, 2, length.out = 7))
  p <- m$params[[1]]
  expect_equal(e$phi, rational(e$z, p$num[2, 1, ], p$den[2, 1, ]))
})

test_that("denominator coefficients receive gradient (nonzero init)", {
  f <- thangldKANr:::validation_fixture("rational")
  g <- thangld_kan_gradient_r(f$model, f$x, f$y)
  expect_gt(max(abs(g$params[[1]]$den)), 0)
})
