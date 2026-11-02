set.seed(3)
x <- matrix(runif(120, -2, 2), 60, 2, dimnames = list(NULL, c("a", "b")))
y <- sin(x[, "a"]) + x[, "b"]^2 / 2
fit <- thangld_kan_fit(thangld_kan(c(2, 3, 1), seed = 1), x, y, epochs = 20, lr = 0.02,
               backend = "r", seed = 1)

test_that("edge function: grid, original scale, index checks", {
  e <- thangld_kan_edge(fit, layer = 1, input = 2, output = 3, n = 11)
  expect_equal(nrow(e), 11L)
  expect_equal(range(e$x), fit$layer_input_range[[1]][, 2])
  e2 <- thangld_kan_edge(fit, 1, 1, 1, x = c(-1, 0, 1), scale = "original")
  pp <- fit$preprocess
  expect_equal(e2$h, (c(-1, 0, 1) - pp$x_center[1]) / pp$x_scale[1])
  expect_error(thangld_kan_edge(fit, 2, 1, 1, scale = "original"), "layer 1")
  expect_error(thangld_kan_edge(fit, 3), "layer")
  expect_error(thangld_kan_edge(fit, 1, input = 3), "input")
})

test_that("plots draw without error", {
  f <- tempfile(fileext = ".pdf")
  grDevices::pdf(f)
  on.exit({
    grDevices::dev.off()
    unlink(f)
  })
  expect_invisible(plot(fit, layer = 2, input = 3))
  expect_invisible(plot(fit, scale = "original"))
  expect_invisible(plot(fit, type = "loss"))
  expect_error(plot(thangld_kan(c(1, 1)), type = "loss"), "not fitted")
})

test_that("save/load: same predictions and metadata", {
  f <- tempfile(fileext = ".rds")
  on.exit(unlink(f))
  thangld_kan_save(fit, f)
  g <- thangld_kan_load(f)
  expect_lt(serr(predict(fit, x), predict(g, x)), 1e-8)
  expect_identical(g$params, fit$params)
  expect_identical(g$preprocess, fit$preprocess)
  expect_equal(attr(g, "saved")$package_version,
               as.character(packageVersion("thangldKANr")))
  saveRDS(list(a = 1), f)
  expect_error(thangld_kan_load(f), "Not a thangldKANr")
  expect_error(thangld_kan_load(tempfile()), "not found")
  expect_error(predict(thangld_kan(c(2, 1)), x), "not fitted")
})

test_that("continued training keeps the preprocessing", {
  g <- thangld_kan_fit(fit, x[1:10, ] * 10, y[1:10], epochs = 2, backend = "r")
  expect_identical(g$preprocess, fit$preprocess)
})

test_that("validation suite (base R part) passes", {
  res <- thangld_kan_validate("r")
  expect_true(all(res$pass))
})
