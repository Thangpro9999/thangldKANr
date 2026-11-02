test_that("all built-in bases are listed and build networks", {
  kb <- kan_bases()
  expect_setequal(kb$name[kb$builtin], c("chebyshev", "rbf", "legendre",
    "wavelet", "rational", "hybrid", "jacobi", "relu"))
  for (b in kb$name[kb$builtin]) {
    m <- thangld_kan(c(2, 3, 1), basis = b, seed = 1)
    expect_s3_class(m, "thangld_kan")
    expect_equal(kan_layer_bases(m), c(b, b))
    expect_length(thangld_kan_forward_r(m, matrix(0.3, 2, 2)), 2L)
  }
})

test_that("argument routing: unknown basis, unused or unnamed arguments", {
  expect_error(thangld_kan(c(2, 1), basis = "nope"), "Unknown basis")
  expect_error(thangld_kan(c(2, 1), basis = "relu", degree = 3), "not used")
  expect_error(thangld_kan(c(2, 1), basis = "chebyshev", 3), "named")
  expect_error(thangld_kan(c(2, 3, 1), basis = c("rbf", "relu", "jacobi")),
               "basis")
  expect_error(thangld_kan(c(2, 1), basis = "chebyshev",
                         options = list(relu = list(k = 2))), "options")
  expect_error(thangld_kan(c(2, 1), basis = "chebyshev",
                         options = list(chebyshev = list(k = 2))), "options")
})

test_that("per-basis arguments in a mixed network go to their layers", {
  m <- thangld_kan(c(2, 3, 2, 1), basis = c("chebyshev", "relu", "jacobi"),
                 degree = c(5, 1, 2), grid_size = 4,
                 options = list(jacobi = list(alpha = 0.5)), seed = 3)
  expect_equal(m$specs[[1]]$degree, 5L)
  expect_equal(m$specs[[2]]$grid_size, 4L)
  expect_null(m$specs[[2]]$degree)
  expect_equal(m$specs[[3]]$degree, 2L)
  expect_equal(m$specs[[3]]$alpha, 0.5)
})

test_that("a mixed network passes the gradient check and save/load", {
  m <- thangld_kan(c(3, 4, 3, 2, 1),
                 basis = c("rbf", "wavelet", "hybrid", "legendre"), seed = 5)
  x <- thangldKANr:::with_seed(6, matrix(stats::rnorm(15), 5, 3))
  y <- thangldKANr:::with_seed(7, stats::rnorm(5))
  g <- thangld_kan_gradient_r(m, x, y)
  fd <- thangld_kan_fd_gradient_r(m, x, y, step = 1e-5, order = 4L)
  expect_lt(serr(unlist(g$params), unlist(fd$params)), 1e-5)
  expect_lt(serr(g$x, fd$x), 1e-5)
  fit <- thangld_kan_fit(m, x, y, epochs = 3, backend = "r", seed = 1)
  f <- tempfile(fileext = ".rds")
  on.exit(unlink(f))
  thangld_kan_save(fit, f)
  expect_lt(serr(predict(fit, x), predict(thangld_kan_load(f), x)), 1e-8)
})

toy_linear <- function() list(
  label = "linear (toy)",
  layer_specs = function(widths, scale = 0.3) {
    rep(list(list(scale = scale)), length(widths) - 1L)
  },
  describe = function(spec) "linear",
  param_dims = function(p, q, spec) list(w = c(p, q)),
  init = function(p, q, spec) {
    list(w = matrix(stats::rnorm(p * q, sd = spec$scale), p, q))
  },
  check_params = function(par, spec) invisible(TRUE),
  forward_r = function(h, par, spec) list(out = h %*% par$w, cache = h),
  backward_r = function(cache, g, par, spec) {
    list(grads = list(w = crossprod(cache, g)), dh = g %*% t(par$w))
  },
  forward_torch = function(h, par, spec) torch::torch_matmul(h, par$w),
  edge_r = function(x, par, spec, i, j) data.frame(phi = par$w[i, j] * x),
  naive_forward = function(h, par, spec) h %*% par$w)

test_that("register_kan_basis plugs a new basis into every function", {
  on.exit(unregister_kan_basis("toy_linear"), add = TRUE)
  register_kan_basis("toy_linear", toy_linear())
  expect_true("toy_linear" %in% kan_bases()$name)
  expect_equal(kan_bases()$arguments[kan_bases()$name == "toy_linear"], "scale")
  expect_error(register_kan_basis("toy_linear", toy_linear()), "already")
  expect_error(register_kan_basis("chebyshev", toy_linear()), "built-in")
  expect_error(register_kan_basis("bad", list(describe = identity)), "lacks")
  m <- thangld_kan(c(2, 3, 1), basis = c("toy_linear", "chebyshev"),
                 scale = 0.5, seed = 2)
  x <- matrix(c(0.1, -0.4, 0.7, 0.2), 2)
  expect_equal(thangld_kan_forward_r(m, x, layers = TRUE)[[1]],
               x %*% m$params[[1]]$w)
  res <- thangld_kan_validate("r", basis = "toy_linear", mixed = FALSE)
  expect_true(all(is.na(res$pass[res$group == "basis"])))
  expect_true(all(res$pass[res$group != "basis"]))
  expect_equal(attr(res, "n_skipped"), 1L)
  expect_true(unregister_kan_basis("toy_linear"))
  expect_error(thangld_kan(c(2, 1), basis = "toy_linear"), "Unknown basis")
})

test_that("a registered basis also runs on the torch backend", {
  skip_if_no_torch()
  on.exit(unregister_kan_basis("toy_linear"), add = TRUE)
  register_kan_basis("toy_linear", toy_linear())
  res <- thangld_kan_validate("torch", basis = "toy_linear", mixed = FALSE)
  expect_true(attr(res, "all_pass"))
})

test_that("hybrid_kan_components rejects layers of another basis", {
  x <- matrix(stats::runif(40, -2, 2), 20, 2)
  fit <- thangld_kan_fit(thangld_kan(c(2, 1), basis = "relu", seed = 1), x,
                       x[, 1], epochs = 2, backend = "r", seed = 1)
  expect_error(hybrid_kan_components(fit, x), "hybrid")
})
