## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")

## -----------------------------------------------------------------------------
library(thangldKANr)

## -----------------------------------------------------------------------------
kan_bases()[, c("name", "label")]

## -----------------------------------------------------------------------------
set.seed(1)
n <- 400
x <- matrix(runif(2 * n, -2, 2), n, 2, dimnames = list(NULL, c("x1", "x2")))
f <- function(x) x[, "x1"] * x[, "x2"] + sin(x[, "x1"])
y <- f(x) + rnorm(n, sd = 0.05)
train <- 1:240
valid <- 241:300
test <- 301:400
fit_one <- function(basis, ...) {
  thangld_kan_fit(thangld_kan(c(2, 4, 1), basis = basis, ..., seed = 1),
                x[train, ], y[train],
                validation_data = list(x = x[valid, ], y = y[valid]),
                epochs = 150, batch_size = 32, lr = 0.02, patience = 30,
                seed = 1, backend = "r")
}
fits <- list(chebyshev = fit_one("chebyshev", degree = 4),
             rbf = fit_one("rbf"),
             relu = fit_one("relu"))
t(sapply(fits, function(m) thangld_kan_metrics(y[test], predict(m, x[test, ]))))

## ----fig.width = 6, fig.height = 4--------------------------------------------
plot(fits$chebyshev, type = "loss")
plot(fits$rbf, layer = 1, input = 1, output = 1, scale = "original")

## -----------------------------------------------------------------------------
m <- thangld_kan(c(2, 4, 3, 1), basis = c("rbf", "chebyshev", "jacobi"),
               degree = 3, options = list(jacobi = list(alpha = 0.5)),
               seed = 2)
m
kan_layer_bases(m)

## -----------------------------------------------------------------------------
lin <- list(
  label = "linear (toy)",
  layer_specs = function(widths) rep(list(list()), length(widths) - 1L),
  describe = function(spec) "linear",
  param_dims = function(p, q, spec) list(w = c(p, q)),
  init = function(p, q, spec) list(w = matrix(rnorm(p * q, sd = 0.3), p, q)),
  check_params = function(par, spec) invisible(TRUE),
  forward_r = function(h, par, spec) list(out = h %*% par$w, cache = h),
  backward_r = function(cache, g, par, spec)
    list(grads = list(w = crossprod(cache, g)), dh = g %*% t(par$w)),
  forward_torch = function(h, par, spec) torch::torch_matmul(h, par$w),
  edge_r = function(x, par, spec, i, j) data.frame(phi = par$w[i, j] * x))
register_kan_basis("toy_linear", lin)
res <- thangld_kan_validate("r", basis = "toy_linear", mixed = FALSE)
res[, c("group", "check", "pass")]
unregister_kan_basis("toy_linear")

## -----------------------------------------------------------------------------
f <- tempfile(fileext = ".rds")
thangld_kan_save(fits$relu, f)
all.equal(predict(fits$relu, x[test, ]), predict(thangld_kan_load(f), x[test, ]))
res <- thangld_kan_validate("r")
aggregate(pass ~ basis, res, function(p) sprintf("%d/%d", sum(p), length(p)))

