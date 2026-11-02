g <- rbf_grid(-2, 2, 8)
u <- c(-3.1, -2, -0.7, 0, 0.3, 1.9, 2.5)

test_that("FastKAN default grid and explicit Gaussian values", {
  expect_equal(g$centers, seq(-2, 2, length.out = 8))
  expect_equal(g$d, 4 / 7)
  b <- rbf_basis(u, g$centers, g$d)
  expect_lt(max(abs(b - exp(-(outer(u, g$centers, "-") / (4 / 7))^2))), 1e-10)
  expect_equal(diag(rbf_basis(g$centers, g$centers, g$d)), rep(1, 8))
  expect_true(all(b > 0 & b <= 1))
})

test_that("RBF derivative matches central differences", {
  h <- 1e-6
  fd <- (rbf_basis(u + h, g$centers, g$d) - rbf_basis(u - h, g$centers, g$d)) /
    (2 * h)
  expect_lt(max(abs(rbf_basis_deriv(u, g$centers, g$d) - fd)), 1e-6)
})

test_that("argument checks", {
  expect_error(rbf_basis(c(1, NA), 0, 1), "finite")
  expect_error(rbf_basis(1, 0, 0), "d")
  expect_error(rbf_grid(1, 0, 3), "exceed")
  expect_error(rbf_grid(0, 1, 1), "num_grids")
})

test_that("layer norm is never applied to a one-input layer", {
  m <- thangld_kan(basis = "rbf", c(1, 3, 1), seed = 1)
  expect_false(m$specs[[1]]$layernorm)
  expect_true(m$specs[[2]]$layernorm)
  expect_null(m$params[[1]]$ln_weight)
  expect_length(m$params[[2]]$ln_weight, 3L)
})

test_that("parameter layout and initialization follow FastKAN", {
  m <- thangld_kan(basis = "rbf", c(30, 40, 1), seed = 2)
  p <- m$params[[1]]
  expect_equal(dim(p$weight), c(30L, 40L, 8L))
  expect_equal(dim(p$base_weight), c(40L, 30L))
  expect_equal(sd(p$weight), 0.1, tolerance = 0.05)
  expect_true(all(abs(p$base_weight) <= 1 / sqrt(30)))
  expect_equal(as.numeric(p$ln_weight), rep(1, 30))
  m2 <- thangld_kan(basis = "rbf", c(2, 1), layernorm = FALSE, base_update = FALSE, seed = 1)
  expect_named(m2$params[[1]], "weight")
})

test_that("without layer norm and base, one layer is a sum of RBF edges", {
  m <- thangld_kan(basis = "rbf", c(2, 1), layernorm = FALSE, base_update = FALSE, seed = 3)
  x <- matrix(c(-1, 0.5, 2, -0.3, 0.2, 1.1), 3, 2)
  s <- thangld_kan_edge(m, 1, 1, 1, x = x[, 1])$phi +
    thangld_kan_edge(m, 1, 2, 1, x = x[, 2])$phi
  expect_equal(s, thangld_kan_forward_r(m, x))
})

test_that("layer norm over two inputs keeps only the sign of their difference", {
  # documented property: LN of (a, b) is (+-1, -+1) * gamma + beta
  m <- thangld_kan(basis = "rbf", c(2, 1), base_update = FALSE, ln_eps = 0, seed = 1)
  x1 <- matrix(c(0.1, 0.2), 1)
  x2 <- matrix(c(-3, 5), 1)
  expect_equal(thangld_kan_forward_r(m, x1), thangld_kan_forward_r(m, x2))
})
