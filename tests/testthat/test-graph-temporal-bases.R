test_that("graph KAN works with every built-in basis and mixed bases", {
  g <- thangld_graph_kan_simulate(n = 30, seed = 1)
  y <- g$x[, 1] + stats::rnorm(30, sd = 0.1)
  for (b in kan_bases()$name[kan_bases()$builtin]) {
    m <- thangld_graph_kan(c(2, 3, 1), basis = b, self_branch = TRUE, seed = 1)
    fit <- thangld_graph_kan_fit(m, g$x, g$edges, y, train = 1:20, val = 21:25,
                                 epochs = 3, backend = "r")
    expect_true(all(is.finite(predict(fit))))
    f <- tempfile(fileext = ".rds")
    thangld_graph_kan_save(fit, f)
    expect_equal(predict(thangld_graph_kan_load(f)), predict(fit))
    unlink(f)
  }
  m <- thangld_graph_kan(c(2, 3, 1), basis = c("rbf", "chebyshev"), seed = 2)
  expect_equal(kan_layer_bases(structure(m, class = "thangld_kan")), c("rbf", "chebyshev"))
  expect_equal(m$specs[[2]]$degree, 3L)
  expect_error(thangld_graph_kan(c(2, 1), basis = "relu", degree = 2), "not used")
})

test_that("temporal KAN works with every built-in basis", {
  s <- thangldKANr:::with_seed(3, as.numeric(stats::arima.sim(list(ar = 0.5), 80)))
  for (b in kan_bases()$name[kan_bases()$builtin]) {
    m <- thangld_temporal_kan(1:3, hidden = 2, basis = b, seed = 1)
    fit <- thangld_temporal_kan_fit(m, s, n_val = 10, epochs = 3, seed = 1,
                                    backend = "r")
    expect_true(all(is.finite(predict(fit))))
  }
  expect_error(thangld_temporal_kan(1:3, basis = "wavelet", degree = 2), "not used")
})
