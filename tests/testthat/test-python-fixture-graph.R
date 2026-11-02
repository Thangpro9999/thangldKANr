# Bundled Python reference (NumPy normalized adjacency + ChebyKAN recurrence
# layers + torch.optim.Adam, full batch). Thresholds fixed before the first
# run: adjacency 1e-12, forward/gradient 1e-8, training 1e-6.
ref <- function(...) test_path("fixtures", "graph", ...)
rd <- function(...) utils::read.csv(ref(...))
read_case <- function(d) {
  meta <- rd(d, "meta.csv")
  widths <- as.integer(strsplit(meta$widths, " ")[[1]])
  m <- thangld_graph_kan(widths, degree = 0)
  dims <- rd(d, "dims.csv")
  pr <- rd(d, "params.csv")
  m$params <- lapply(seq_len(nrow(dims)), function(l) {
    dd <- as.integer(strsplit(dims$dims[l], " ")[[1]])
    list(coef = array(pr$value[pr$layer == l - 1L][order(pr$pos[pr$layer == l - 1L])], dd))
  })
  m$specs <- lapply(m$params, function(p) list(basis = "chebyshev", degree = dim(p$coef)[3] - 1L))
  e <- rd(d, "edges.csv")
  xl <- rd(d, "x.csv")
  x <- matrix(0, meta$n, widths[1])
  x[cbind(xl$row + 1L, xl$col + 1L)] <- xl$value
  nd <- rd(d, "nodes.csv")
  list(model = m, edges = cbind(e$from, e$to) + 1L, n = meta$n, x = x, y = rd(d, "y.csv")$y,
       a = sort(nd$node[nd$set == "a"]) + 1L, b = sort(nd$node[nd$set == "b"]) + 1L)
}

test_that("adjacency, forward and gradients match the Python reference", {
  f <- read_case("net")
  ah <- thangld_graph_kan_adjacency(f$edges, n = f$n)
  expect_lte(max(abs(as.vector(ah) - rd("net_a_hat.csv")$value)), 1e-12)
  outs <- thangld_graph_kan_forward_r(f$model, f$x, ah, layers = TRUE)
  for (l in seq_along(outs)) {
    expect_lte(serr(as.vector(outs[[l]]), rd(sprintf("net_unopt_layer%d.csv", l - 1))$value),
               1e-8)
  }
  g <- thangld_graph_kan_gradient_r(f$model, f$x, ah, f$y, f$a)
  expect_lte(serr(unlist(lapply(g$params, function(p) as.vector(p$coef))),
                  rd("net_unopt_grad_params.csv")$value), 1e-8)
  expect_lte(serr(as.vector(g$x), rd("net_unopt_grad_x.csv")$value), 1e-8)
})

check_training <- function(backend) {
  cs <- read_case(file.path("train", "one_hop_s1"))
  st <- rd("train_settings.csv")
  fit <- thangld_graph_kan_fit(cs$model, cs$x, cs$edges, cs$y, train = cs$a, val = cs$b,
                       epochs = st$epochs, lr = st$lr, betas = c(st$beta1, st$beta2),
                       eps = st$eps, patience = st$patience, backend = backend)
  h <- rd("py_one_hop_s1", "unopt_history.csv")
  expect_equal(nrow(fit$history), nrow(h))
  expect_lte(serr(c(fit$history$train_loss, fit$history$val_loss),
                  c(h$train_loss, h$val_loss)), 1e-6)
  expect_lte(serr(unlist(lapply(fit$params, function(p) as.vector(p$coef))),
                  rd("py_one_hop_s1", "unopt_best_params.csv")$value), 1e-6)
  expect_lte(serr(predict(fit), rd("py_one_hop_s1", "unopt_predictions.csv")$pred), 1e-6)
  pp <- fit$preprocess
  expect_lte(serr(c(pp$x_center, pp$x_scale, pp$y_center, pp$y_scale),
                  rd("py_one_hop_s1", "scaling.csv")$value), 1e-12)
}

test_that("base R full-batch training matches the Python reference", {
  check_training("r")
})

test_that("torch full-batch training matches the Python reference", {
  skip_if_no_torch()
  check_training("torch")
})
