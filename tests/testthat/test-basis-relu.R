test_that("ReLU basis: support, peak, original grid, derivative", {
  B <- relu_basis(seq(-3, 3, by = 0.01), grid_size = 5, k = 3, grid_range = c(-2, 2))
  expect_equal(ncol(B), 8L)
  expect_true(all(B >= 0) && all(B <= 1 + 1e-12))
  # support of R_1 is [-2 - 4 * 3/5, -2 + 4/5] = [-4.4, -1.2]
  expect_equal(unname(relu_basis(c(-1.2, -1.19), 5, 3, c(-2, 2))[, 1]), c(0, 0))
  expect_equal(unname(relu_basis(-2.8, 5, 3, c(-2, 2))[1, 1]), 1)
  # grid_range c(0, 1): grid and r of the original ReLUKANLayer(g = 5, k = 3)
  low <- (-3:4) / 5
  x <- c(0.13, 0.5, 0.91)
  r <- 4 * 25 / 16
  ex <- sapply(seq_along(low), function(m) (r * pmax(x - low[m], 0) * pmax(low[m] + 4 / 5 - x, 0))^2)
  expect_lt(max(abs(relu_basis(x, 5, 3, c(0, 1)) - ex)), 1e-12)
  h <- 1e-6
  xx <- c(-1.733, -0.411, 0.257, 1.399)
  fd <- (relu_basis(xx + h, 4, 2) - relu_basis(xx - h, 4, 2)) / (2 * h)
  expect_lt(max(abs(relu_basis_deriv(xx, 4, 2) - fd)), 1e-6)
  expect_error(relu_basis(0, grid_range = c(1, 1)), "grid_range")
  expect_error(relu_basis(c(0, NA)), "finite")
})

test_that("architecture: parameter shapes, grid init, Conv2d-style bounds", {
  m <- thangld_kan(basis = "relu", c(3, 2, 1), grid_size = c(4, 6), k = c(2, 1), seed = 1)
  p1 <- m$params[[1]]
  expect_equal(dim(p1$weight), c(3L, 2L, 6L))
  expect_equal(dim(p1$low), c(3L, 6L))
  expect_equal(length(p1$bias), 2L)
  expect_equal(dim(m$params[[2]]$weight), c(2L, 1L, 7L))
  expect_equal(p1$low[2, ], -2 + 4 * (-2:3) / 4)
  expect_equal(p1$high - p1$low, matrix(4 * 3 / 4, 3, 6))
  expect_true(all(abs(p1$weight) <= 1 / sqrt(3 * 6)))
  expect_equal(m$specs[[1]]$r, 4 * 16 / (9 * 16))
  expect_error(thangld_kan(basis = "relu", c(2, 1), k = -1), "k")
  expect_error(thangld_kan(basis = "relu", c(2, 1), grid_size = 0), "grid_size")
  expect_error(thangld_kan(basis = "relu", c(2, 1), grid_range = c(2, -2)), "grid_range")
})

test_that("phases outside the data give zero output except the bias", {
  m <- thangld_kan(basis = "relu", c(1, 1), grid_size = 3, k = 1, seed = 3)
  x <- matrix(c(10, -10), 2, 1)
  expect_equal(drop(thangld_kan_forward_r(m, x)), rep(m$params[[1]]$bias[1], 2))
  g <- thangld_kan_gradient_r(m, x, c(0.5, -0.5))
  expect_equal(max(abs(g$params[[1]]$low)), 0)
})
