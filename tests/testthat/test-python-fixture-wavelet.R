# basis = "wavelet". Fixture produced by inst/validation/python (Python reference at a pinned
# commit, PyTorch, CPU float64) from inputs written by
# inst/validation/wavelet/make_fixture.R. No Python is needed to run this file.
fx <- function(...) testthat::test_path("fixtures", "wavelet", ...)
rmat <- function(f) {
  d <- utils::read.csv(fx(f))
  m <- matrix(0, max(d$row) + 1, max(d$col) + 1)
  m[cbind(d$row + 1, d$col + 1)] <- d$value
  m
}
rpar <- function(f, like) {
  d <- utils::read.csv(fx(f), stringsAsFactors = FALSE)
  lapply(seq_along(like), function(l) {
    out <- lapply(names(like[[l]]), function(nm) {
      s <- d[d$layer == l - 1 & d$name == nm, ]
      a <- like[[l]][[nm]] * 0
      a[s$pos + 1] <- as.numeric(s$value)
      a
    })
    stats::setNames(out, names(like[[l]]))
  })
}
model <- thangldKANr:::validation_fixture("wavelet", scale_param = "raw", bias = FALSE)$model
model$params <- rpar("net_params.csv", model$params)
x <- rmat("net_x.csv")
y <- utils::read.csv(fx("net_y.csv"))$y
variants <- strsplit(readLines(fx("variants.txt"), warn = FALSE), " ")[[1]]

test_that("fixture comes from the pinned reference commit", {
  env <- paste(readLines(fx("python_env.json"), warn = FALSE), collapse = " ")
  expect_match(env, "\"commit_ok\": true")
})

for (v in variants) {
  test_that(sprintf("forward layers match the reference (%s, 1e-8)", v), {
    outs <- thangld_kan_forward_r(model, x, layers = TRUE)
    for (l in seq_along(outs)) {
      expect_lt(serr(outs[[l]], rmat(sprintf("net_%s_layer%d_out.csv", v, l - 1))),
                1e-8)
    }
  })
  test_that(sprintf("gradients match reference autograd (%s, 1e-8)", v), {
    g <- thangld_kan_gradient_r(model, x, y)
    pg <- rpar(sprintf("net_%s_grad_params.csv", v), model$params)
    expect_lt(serr(unlist(g$params), unlist(pg)), 1e-8)
    expect_lt(serr(g$x, rmat(sprintf("net_%s_grad_x.csv", v))), 1e-8)
    skip_if_no_torch()
    gt <- thangldKANr:::engine_gradient_torch(model$params, model$specs, x, y)
    expect_lt(serr(unlist(gt$params), unlist(pg)), 1e-8)
    expect_lt(serr(gt$x, rmat(sprintf("net_%s_grad_x.csv", v))), 1e-8)
  })
  test_that(sprintf("5 Adam steps match torch.optim.Adam (%s)", v), {
    st <- utils::read.csv(fx("adam_settings.csv"))
    sd <- utils::read.csv(fx("adam_schedule.csv"))
    par <- model$params
    states <- lapply(par, function(pl) {
      lapply(pl, function(a) list(t = 0L, m = a * 0, v = a * 0))
    })
    for (s in sort(unique(sd$epoch))) {
      idx <- sd$idx[sd$epoch == s] + 1
      g <- thangldKANr:::engine_gradient(par, model$specs, x[idx, , drop = FALSE],
                                   y[idx])
      for (l in seq_along(par)) for (nm in names(par[[l]])) {
        u <- thangldKANr:::adam_step_r(par[[l]][[nm]], g$params[[l]][[nm]],
                                 states[[l]][[nm]], st$lr,
                                 c(st$beta1, st$beta2), st$eps)
        par[[l]][[nm]] <- u$param
        states[[l]][[nm]] <- u$state
      }
      ref <- rpar(sprintf("adam_%s_step%d_params.csv", v, s), model$params)
      expect_lt(serr(unlist(par), unlist(ref)), 1e-8)
    }
  })
}
