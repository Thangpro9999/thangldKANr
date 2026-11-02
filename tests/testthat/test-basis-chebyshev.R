z <- c(-1, -0.999, -0.73, -0.5, -0.1, 0, 0.2, 0.5, 0.88, 0.999, 1)

test_that("recurrence matches cos(k acos z) and explicit T_0..T_4 (1e-10)", {
  ex <- cbind(1, z, 2 * z^2 - 1, 4 * z^3 - 3 * z, 8 * z^4 - 8 * z^2 + 1)
  expect_lt(max(abs(unname(chebyshev_basis(z, 4)) - ex)), 1e-10)
  k <- 0:9
  expect_lt(max(abs(unname(chebyshev_basis(z, 9)) - outer(acos(z), k,
                                                         function(a, kk) cos(kk * a)))),
            1e-10)
})

test_that("derivatives: k U_{k-1}, finite differences, boundary k^2", {
  zi <- seq(-0.9, 0.9, by = 0.1)
  h <- 1e-6
  fd <- (chebyshev_basis(zi + h, 8) - chebyshev_basis(zi - h, 8)) / (2 * h)
  expect_lt(max(abs(unname(chebyshev_basis_deriv(zi, 8)) - unname(fd))), 1e-6)
  expect_equal(unname(drop(chebyshev_basis_deriv(1, 7))), (0:7)^2)
  expect_equal(unname(drop(chebyshev_basis_deriv(-1, 7))), (-1)^(1:8) * (0:7)^2)
})

test_that("input checks and layout follow the ChebyKAN convention", {
  expect_error(chebyshev_basis(c(0, NA), 2), "finite")
  expect_error(chebyshev_basis(0, -1), "degree")
  expect_error(thangld_kan(c(2, 1), basis = "chebyshev", degree = c(1, 2)), "degree")
  m <- thangld_kan(c(40, 50, 1), basis = "chebyshev", degree = 4, seed = 2)
  expect_equal(dim(m$params[[1]]$coef), c(40L, 50L, 5L))
  expect_equal(sd(m$params[[1]]$coef), 1 / (40 * 5), tolerance = 0.05)
})

test_that("saturated tanh keeps gradients finite (no acos singularity)", {
  m <- thangld_kan(c(2, 3, 1), basis = "chebyshev", degree = 5, init_sd = 0.5,
                 seed = 4)
  g <- thangld_kan_gradient_r(m, matrix(c(30, -30), 1), 0)
  expect_true(all(is.finite(unlist(g$params))))
  expect_equal(drop(g$x), c(0, 0))
})
