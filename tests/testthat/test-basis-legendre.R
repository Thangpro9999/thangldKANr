z <- c(-1, -0.999, -0.73, -0.5, -0.1, 0, 0.2, 0.5, 0.88, 0.999, 1)

test_that("Bonnet recurrence matches explicit P_0..P_6 (1e-10)", {
  ex <- cbind(1, z, (3 * z^2 - 1) / 2, (5 * z^3 - 3 * z) / 2,
              (35 * z^4 - 30 * z^2 + 3) / 8, (63 * z^5 - 70 * z^3 + 15 * z) / 8,
              (231 * z^6 - 315 * z^4 + 105 * z^2 - 5) / 16)
  expect_lt(max(abs(unname(legendre_basis(z, 6)) - ex)), 1e-10)
})

test_that("derivatives: explicit, finite differences and boundary", {
  dex <- cbind(0, 1, 3 * z, (15 * z^2 - 3) / 2, (140 * z^3 - 60 * z) / 8)
  expect_lt(max(abs(unname(legendre_basis_deriv(z, 4)) - dex)), 1e-10)
  zi <- seq(-0.9, 0.9, by = 0.1)
  h <- 1e-6
  fd <- (legendre_basis(zi + h, 8) - legendre_basis(zi - h, 8)) / (2 * h)
  expect_lt(max(abs(unname(legendre_basis_deriv(zi, 8)) - unname(fd))), 1e-6)
  expect_equal(unname(drop(legendre_basis_deriv(1, 7))), (0:7) * (1:8) / 2)
})

test_that("symmetry, boundary, |P_k| <= 1 and orthogonality", {
  s <- matrix((-1)^(0:8), length(z), 9, byrow = TRUE)
  expect_lt(max(abs(legendre_basis(-z, 8) - s * legendre_basis(z, 8))), 1e-10)
  expect_equal(unname(drop(legendre_basis(-1, 8))), (-1)^(0:8))
  expect_true(all(abs(legendre_basis(seq(-1, 1, by = 0.01), 10)) <= 1 + 1e-12))
  gl <- thangldKANr:::.legendre_gauss_legendre_nodes(20)
  pb <- legendre_basis(gl$x, 8)
  expect_lt(max(abs(crossprod(pb * gl$w, pb) - diag(2 / (2 * (0:8) + 1)))),
            1e-10)
  # the identity int_{-1}^{1} P_2^2 = 2/5 also with stats::integrate
  i2 <- stats::integrate(function(t) legendre_basis(t, 2)[, 3]^2, -1, 1)$value
  expect_equal(i2, 2 / 5, tolerance = 1e-8)
})

test_that("degree 0/1 Legendre equals Chebyshev T_0/T_1 and input checks", {
  expect_equal(unname(legendre_basis(z, 1)), unname(cbind(1, z)))
  expect_error(legendre_basis(c(0, NA), 2), "finite")
  expect_error(legendre_basis(0, -1), "degree")
  expect_error(thangld_kan(basis = "legendre", c(2, 1), degree = c(1, 2)), "degree")
})

test_that("layout and initialization follow the ChebyKAN convention", {
  m <- thangld_kan(basis = "legendre", c(40, 50, 1), degree = 4, seed = 2)
  expect_equal(dim(m$params[[1]]$coef), c(40L, 50L, 5L))
  expect_equal(sd(m$params[[1]]$coef), 1 / (40 * 5), tolerance = 0.05)
})

test_that("saturated tanh keeps gradients finite", {
  m <- thangld_kan(basis = "legendre", c(2, 3, 1), degree = 5, init_sd = 0.5, seed = 4)
  g <- thangld_kan_gradient_r(m, matrix(c(30, -30), 1), 0)
  expect_true(all(is.finite(unlist(g$params))))
  expect_equal(drop(g$x), c(0, 0))
})
