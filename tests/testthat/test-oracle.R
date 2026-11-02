fx <- thangldKANr:::validation_fixture()

test_that("forward pass matches independent loops (1e-8)", {
  naive <- fx$x
  for (l in seq_along(fx$model$params)) {
    naive <- thangldKANr:::layer_naive_forward(naive, fx$model$params[[l]],
                                         fx$model$specs[[l]])
  }
  expect_lt(serr(thangld_kan_forward_r(fx$model, fx$x), naive), 1e-8)
  outs <- thangld_kan_forward_r(fx$model, fx$x, layers = TRUE)
  expect_equal(length(outs), length(fx$model$params))
  expect_equal(drop(outs[[length(outs)]]), thangld_kan_forward_r(fx$model, fx$x))
})

test_that("analytic gradients match central differences", {
  g <- thangld_kan_gradient_r(fx$model, fx$x, fx$y)
  for (h in c(1e-4, 1e-5, 1e-6)) {
    fd <- thangld_kan_fd_gradient_r(fx$model, fx$x, fx$y, step = h, order = 4)
    expect_lt(serr(unlist(g$params), unlist(fd$params)), 1e-5)
    expect_lt(serr(g$x, fd$x), 1e-5)
  }
  fd2 <- thangld_kan_fd_gradient_r(fx$model, fx$x, fx$y, step = 1e-6, order = 2)
  expect_lt(serr(unlist(g$params), unlist(fd2$params)), 1e-5)
  expect_equal(g$loss, mean((thangld_kan_forward_r(fx$model, fx$x) - fx$y)^2))
})

test_that("oracle input checks", {
  expect_error(thangld_kan_forward_r(fx$model, fx$x[, 1:2]), "expects 3")
  expect_error(thangld_kan_forward_r(fx$model, rbind(fx$x, NA)), "NA")
  expect_error(thangld_kan_gradient_r(fx$model, fx$x, fx$y[-1]), "length")
  expect_error(thangld_kan_forward_r(list(), fx$x), "thangld_kan")
  expect_error(thangld_kan_fd_gradient_r(fx$model, fx$x, fx$y, order = 3), "order")
})

test_that("seed gives reproducible init without touching the user's RNG", {
  set.seed(99)
  before <- .Random.seed
  a <- thangld_kan(c(2, 3, 1), seed = 5)
  expect_identical(.Random.seed, before)
  expect_identical(a$params, thangld_kan(c(2, 3, 1), seed = 5)$params)
  expect_error(thangld_kan(c(2, 3, 2)), "single output")
  expect_error(thangld_kan(3), "widths")
  expect_output(print(a), "2 -> 3 -> 1")
})
