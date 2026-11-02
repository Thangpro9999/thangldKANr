test_that("Jacobi basis: special cases, boundary value, derivative", {
  z <- seq(-1, 1, length.out = 21)
  expect_lt(max(abs(jacobi_basis(z, 3, 0, 0) -
                      cbind(1, z, (3 * z^2 - 1) / 2, (5 * z^3 - 3 * z) / 2))), 1e-12)
  expect_lt(max(abs(jacobi_basis(z, 2, 1, 1)[, 3] - (15 * z^2 - 3) / 4)), 1e-12)
  expect_equal(unname(jacobi_basis(1, 5, 2, 0.5)[1, ]), choose(2 + 0:5, 0:5))
  # P_n(-z; a, b) = (-1)^n P_n(z; b, a)
  expect_lt(max(abs(jacobi_basis(-z, 5, 1.3, 0.4) -
                      sweep(jacobi_basis(z, 5, 0.4, 1.3), 2, (-1)^(0:5), `*`))), 1e-12)
  h <- 1e-6
  zz <- seq(-0.9, 0.9, length.out = 11)
  fd <- (jacobi_basis(zz + h, 5, 0.7, -0.2) - jacobi_basis(zz - h, 5, 0.7, -0.2)) / (2 * h)
  expect_lt(max(abs(jacobi_basis_deriv(zz, 5, 0.7, -0.2) - fd)), 1e-6)
  expect_error(jacobi_basis(z, 3, -1, 0), "alpha")
  expect_error(jacobi_basis(c(0, NA), 3), "finite")
})

test_that("alpha = beta = 0 network equals an explicit Legendre KAN", {
  m <- thangld_kan(basis = "jacobi", c(2, 1), degree = 3, alpha = 0, beta = 0, seed = 2)
  x <- matrix(c(-1.2, 0.3, 0.8, 2.1), 2, 2)
  cf <- m$params[[1]]$coef
  leg <- function(z) c(1, z, (3 * z^2 - 1) / 2, (5 * z^3 - 3 * z) / 2)
  ex <- sapply(1:2, function(b) sum(sapply(1:2, function(i) sum(cf[i, 1, ] * leg(tanh(x[b, i]))))))
  expect_lt(max(abs(thangld_kan_forward_r(m, x) - ex)), 1e-12)
})

test_that("architecture stores alpha and beta; argument checks", {
  m <- thangld_kan(basis = "jacobi", c(3, 2, 1), degree = c(2, 5), alpha = 0.5, beta = 2, seed = 1)
  expect_equal(dim(m$params[[2]]$coef), c(2L, 1L, 6L))
  expect_equal(m$specs[[1]]$alpha, 0.5)
  expect_equal(m$specs[[2]]$beta, 2)
  expect_error(thangld_kan(basis = "jacobi", c(2, 1), alpha = c(1, 2)), "alpha")
  expect_error(thangld_kan(basis = "jacobi", c(2, 1), beta = -2), "alpha")
  expect_error(thangld_kan(basis = "jacobi", c(2, 1), degree = -1), "degree")
})
