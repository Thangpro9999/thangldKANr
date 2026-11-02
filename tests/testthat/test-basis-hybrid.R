test_that("hybrid basis: SiLU, Chebyshev and RBF parts and derivatives", {
  x <- seq(-3, 3, length.out = 25)
  b <- hybrid_basis(x, degree = 4, num_grids = 5)
  expect_equal(colnames(b), c("silu", paste0("T", 0:4), paste0("rbf", 1:5)))
  expect_lt(max(abs(b[, "silu"] - x * stats::plogis(x))), 1e-12)
  expect_lt(max(abs(b[, "T3"] - cos(3 * acos(tanh(x))))), 1e-10)
  expect_lt(max(abs(b[, "rbf2"] - exp(-((x + 1) / 1)^2))), 1e-12)
  h <- 1e-5
  fd <- (hybrid_basis(x + h, 4, 5) - hybrid_basis(x - h, 4, 5)) / (2 * h)
  expect_lt(max(abs(hybrid_basis_deriv(x, 4, 5) - fd)), 1e-6)
  expect_error(hybrid_basis(x, num_grids = 1), "num_grids")
  expect_error(hybrid_basis(x, grid_min = 2, grid_max = 1), "grid_max")
})

test_that("bases selects parameter blocks; order is canonical", {
  m <- thangld_kan(basis = "hybrid", c(3, 2, 1), bases = c("rbf", "silu"), num_grids = 4, seed = 1)
  expect_equal(names(m$params[[1]]), c("base", "rbf"))
  expect_equal(dim(m$params[[1]]$rbf), c(3L, 2L, 4L))
  m2 <- thangld_kan(basis = "hybrid", c(2, 1), bases = "chebyshev", degree = 2, seed = 1)
  expect_equal(names(m2$params[[1]]), "cheb")
  expect_error(thangld_kan(basis = "hybrid", c(2, 1), bases = character()), "bases")
  expect_error(thangld_kan(basis = "hybrid", c(2, 1), bases = "spline"), "bases")
  expect_error(thangld_kan(basis = "hybrid", c(2, 1), bases = c("rbf", "rbf")), "bases")
})

test_that("a Chebyshev-only hybrid equals the explicit Chebyshev KAN formula", {
  m <- thangld_kan(basis = "hybrid", c(2, 1), bases = "chebyshev", degree = 3, seed = 2)
  x <- matrix(c(-1.5, 0.2, 0.7, 2), 2, 2)
  cf <- m$params[[1]]$cheb
  ex <- sapply(1:2, function(b) sum(sapply(1:2, function(i)
    sum(cf[i, 1, ] * cos((0:3) * acos(tanh(x[b, i])))))))
  expect_lt(max(abs(thangld_kan_forward_r(m, x) - ex)), 1e-12)
})

test_that("edge parts add up to the edge and components report shares", {
  set.seed(1)
  x <- matrix(runif(160, -2, 2), 80, 2, dimnames = list(NULL, c("a", "b")))
  y <- sin(2 * x[, 1]) + 0.5 * x[, 2]
  fit <- thangld_kan_fit(thangld_kan(basis = "hybrid", c(2, 1), seed = 1), x, y, epochs = 30,
                        lr = 0.02, backend = "r", seed = 1)
  e <- thangld_kan_edge(fit, layer = 1, input = 1, output = 1)
  expect_equal(e$phi, e$silu + e$chebyshev + e$rbf)
  cmp <- hybrid_kan_components(fit, x)
  expect_equal(nrow(cmp), 2 * 1 * 3)
  expect_equal(as.vector(tapply(cmp$share, cmp$input, sum)), c(1, 1))
  expect_true(all(cmp$sd >= 0))
  expect_error(hybrid_kan_components(thangld_kan(basis = "hybrid", c(2, 1)), x), "not fitted")
  expect_error(hybrid_kan_components(fit, x, layer = 2), "exceeds")
})
