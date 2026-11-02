# User-facing functions (generic over the layer type) -----------------------------

#' Validate the structure of a model object
#' @noRd
validate_model <- function(model) {
  if (!inherits(model, "thangld_kan")) {
    stop("`model` must be a thangld_kan object (see thangld_kan()).", call. = FALSE)
  }
  w <- model$widths
  if (length(model$params) != length(w) - 1L ||
      length(model$specs) != length(w) - 1L) {
    stop("Corrupt model: number of layers does not match widths.",
         call. = FALSE)
  }
  for (l in seq_along(model$params)) {
    dims <- layer_param_dims(w[l], w[l + 1L], model$specs[[l]])
    if (!identical(sort(names(dims)), sort(names(model$params[[l]])))) {
      stop(sprintf("Corrupt model: parameters of layer %d do not match.", l),
           call. = FALSE)
    }
    for (nm in names(dims)) {
      a <- model$params[[l]][[nm]]
      if (!identical(as.integer(dim(a)), as.integer(dims[[nm]]))) {
        stop(sprintf("Corrupt model: `%s` of layer %d has wrong dimensions.",
                     nm, l), call. = FALSE)
      }
      if (any(!is.finite(a))) {
        stop(sprintf("Parameters `%s` of layer %d are not finite.", nm, l),
             call. = FALSE)
      }
    }
    layer_check_params(model$params[[l]], model$specs[[l]])
  }
  invisible(model)
}

#' @noRd
n_parameters <- function(model) {
  sum(vapply(model$params, function(pl) sum(lengths(pl)), numeric(1)))
}

#' @noRd
check_widths <- function(widths) {
  if (!is.numeric(widths) || length(widths) < 2L || anyNA(widths) ||
      any(widths != round(widths)) || any(widths < 1)) {
    stop("`widths` must be an integer vector of length >= 2 with values >= 1.",
         call. = FALSE)
  }
  widths <- as.integer(widths)
  if (widths[length(widths)] != 1L) {
    stop("thangldKANr supports a single output: the last element of ",
         "`widths` must be 1.", call. = FALSE)
  }
  widths
}

#' @noRd
new_model <- function(widths, specs, seed, init_info = list()) {
  params <- with_seed(seed, lapply(seq_along(specs), function(l) {
    layer_init(widths[l], widths[l + 1L], specs[[l]])
  }))
  structure(list(widths = widths, specs = specs, params = params,
                 init = c(list(seed = seed), init_info), fitted = FALSE),
            class = "thangld_kan")
}

#' @export
print.thangld_kan <- function(x, ...) {
  cat("<thangld_kan> Kolmogorov-Arnold network\n")
  cat("  widths : ", paste(x$widths, collapse = " -> "), "\n", sep = "")
  cat("  layers : ", paste(vapply(x$specs, layer_describe, ""),
                           collapse = "; "), "\n", sep = "")
  cat("  params : ", n_parameters(x), "\n", sep = "")
  if (isTRUE(x$fitted)) {
    h <- x$history
    cat("  fitted : yes (", x$train$backend, ", ", nrow(h), " epochs",
        if (!is.null(x$train$best_epoch))
          paste0(", best epoch ", x$train$best_epoch) else "",
        ")\n", sep = "")
    cat("  final train loss (scaled MSE): ",
        format(h$train_loss[nrow(h)], digits = 4), "\n", sep = "")
  } else {
    cat("  fitted : no\n")
  }
  invisible(x)
}

#' Check a model-scale input matrix
#' @noRd
check_model_input <- function(model, x) {
  x <- as_numeric_matrix(x, "x")
  if (ncol(x) != model$widths[1L]) {
    stop(sprintf("`x` has %d columns but the model expects %d inputs.",
                 ncol(x), model$widths[1L]), call. = FALSE)
  }
  x
}

#' Forward pass and gradients in base R (numerical oracle)
#'
#' `thangld_kan_forward_r()` evaluates the network on model-scale inputs (already
#' standardized). `thangld_kan_gradient_r()` returns the mean squared error
#' \eqn{L = n^{-1}\sum_b (\hat y_b - y_b)^2} and its analytic gradient with
#' respect to every parameter array and to the inputs.
#' `thangld_kan_fd_gradient_r()` approximates the same gradients by central
#' differences of order 2 (truncation error \eqn{O(h^2)}) or 4
#' (\eqn{O(h^4)}); validation only. These functions use base R only and are
#' the reference for the torch backend.
#'
#' @param model A [thangld_kan()] object.
#' @param x Numeric matrix of model-scale inputs (`n` by `widths[1]`).
#' @param y Numeric vector of length `n` (model scale).
#' @param layers If `TRUE`, return the list of layer outputs.
#' @param step Central-difference step.
#' @param order Order of the central-difference stencil, 2 or 4.
#' @return `thangld_kan_forward_r()`: numeric vector (or list of layer outputs).
#'   The gradient functions: a list with `loss`, `params` (list per layer of
#'   named gradient arrays shaped like `model$params`) and `x`.
#' @examples
#' m <- thangld_kan(c(2, 3, 1), seed = 1)
#' x <- matrix(c(-1, 0.5, 2, -0.2), 2)
#' thangld_kan_forward_r(m, x)
#' g <- thangld_kan_gradient_r(m, x, c(0, 1))
#' fd <- thangld_kan_fd_gradient_r(m, x, c(0, 1))
#' max(abs(unlist(g$params) - unlist(fd$params)))
#' @export
thangld_kan_forward_r <- function(model, x, layers = FALSE) {
  validate_model(model)
  x <- check_model_input(model, x)
  f <- engine_forward(model$params, model$specs, x)
  if (layers) return(f$outs)
  drop(f$out)
}

#' @rdname thangld_kan_forward_r
#' @export
thangld_kan_gradient_r <- function(model, x, y) {
  validate_model(model)
  x <- check_model_input(model, x)
  y <- as_response(y, nrow(x))
  engine_gradient(model$params, model$specs, x, y)
}

#' @rdname thangld_kan_forward_r
#' @export
thangld_kan_fd_gradient_r <- function(model, x, y, step = 1e-5, order = 2L) {
  validate_model(model)
  x <- check_model_input(model, x)
  y <- as_response(y, nrow(x))
  step <- check_number(step, "step", min = 0, strict = TRUE)
  if (!order %in% c(2, 4)) stop("`order` must be 2 or 4.", call. = FALSE)
  engine_fd_gradient(model$params, model$specs, x, y, step, as.integer(order))
}

#' Fit a Kolmogorov-Arnold network for regression
#'
#' Trains all parameters of a [thangld_kan()] network by mini-batch Adam on the
#' mean squared error. Inputs (and optionally the response) are
#' standardized with constants estimated on the training rows only;
#' predictions are returned on the original scale of `y`.
#'
#' **Backends.** `backend = "torch"` (default) uses R 'torch' autograd and
#' `torch::optim_adam()`. `backend = "r"` uses the base R analytic gradient
#' ([thangld_kan_gradient_r()]) and an Adam update that follows `torch.optim.Adam`;
#' it needs no libtorch. Both backends receive the same mini-batch schedule
#' (drawn in R from `seed`), so in float64 they follow the same trajectory
#' up to rounding.
#'
#' **Validation and early stopping.** Supply `validation_data` (a list with
#' `x` and `y`) or `validation_split`. The validation loss is recorded every
#' epoch; with `patience`, training stops after `patience` epochs without an
#' improvement larger than `min_delta`. With validation data the parameters
#' of the epoch with the lowest validation loss are restored. Never use test
#' data here.
#'
#' **Constant columns** in the training rows are fixed at their training
#' value on the model scale during fitting and prediction (with a warning).
#'
#' @param model A [thangld_kan()] object whose first width equals `ncol(x)`. If it
#'   was already fitted, training continues and keeps its preprocessing.
#' @param x Numeric matrix or data frame of predictors.
#' @param y Numeric response vector.
#' @param validation_data Optional `list(x = , y = )` on the original scale.
#' @param validation_split Fraction in `[0, 1)` of rows held out for
#'   validation when `validation_data` is `NULL`.
#' @param epochs Maximum number of epochs.
#' @param batch_size Mini-batch size; `NULL` means full batch.
#' @param lr,betas,eps Adam settings.
#' @param patience `NULL` (no early stopping) or number of epochs.
#' @param min_delta Minimum improvement of the validation loss.
#' @param standardize_x,standardize_y Standardize with training mean and
#'   standard deviation.
#' @param seed Optional seed for the batch order and the validation split;
#'   the caller's random number stream is restored afterwards.
#' @param backend `"torch"` or `"r"`.
#' @param dtype `"float64"` (default) or `"float32"` (torch backend only).
#' @param verbose Print progress.
#' @return The fitted model with `preprocess`, `history` (`epoch`,
#'   `train_loss`, `val_loss` on the standardized scale) and `train`.
#' @examples
#' set.seed(1)
#' x <- matrix(runif(200, -2, 2), ncol = 2, dimnames = list(NULL, c("a", "b")))
#' y <- sin(x[, "a"]) + 0.5 * x[, "b"]^2 + rnorm(100, sd = 0.05)
#' fit <- thangld_kan_fit(thangld_kan(c(2, 1), seed = 1), x, y, epochs = 50, lr = 0.02,
#'               backend = "r", validation_split = 0.2, seed = 1)
#' fit
#' head(predict(fit, x))
#' @export
thangld_kan_fit <- function(model, x, y, validation_data = NULL,
                    validation_split = 0, epochs = 200L, batch_size = NULL,
                    lr = 0.01, betas = c(0.9, 0.999), eps = 1e-8,
                    patience = NULL, min_delta = 0, standardize_x = TRUE,
                    standardize_y = TRUE, seed = NULL,
                    backend = c("torch", "r"),
                    dtype = c("float64", "float32"), verbose = FALSE) {
  validate_model(model)
  backend <- match.arg(backend)
  dtype <- match.arg(dtype)
  if (backend == "torch") require_torch()
  x <- as_numeric_matrix(x, "x")
  y <- as_response(y, nrow(x), "y")
  if (ncol(x) != model$widths[1L]) {
    stop(sprintf("`x` has %d columns but `model` expects %d inputs (widths[1]).",
                 ncol(x), model$widths[1L]), call. = FALSE)
  }
  epochs <- check_count(epochs, "epochs")
  lr <- check_number(lr, "lr", min = 0, strict = TRUE)
  eps <- check_number(eps, "eps", min = 0, strict = TRUE)
  if (!is.numeric(betas) || length(betas) != 2L || any(!is.finite(betas)) ||
      any(betas < 0) || any(betas >= 1)) {
    stop("`betas` must be two numbers in [0, 1).", call. = FALSE)
  }
  if (!is.null(patience)) patience <- check_count(patience, "patience")
  min_delta <- check_number(min_delta, "min_delta", min = 0)
  standardize_x <- check_flag(standardize_x, "standardize_x")
  standardize_y <- check_flag(standardize_y, "standardize_y")
  verbose <- check_flag(verbose, "verbose")
  validation_split <- check_number(validation_split, "validation_split",
                                   min = 0)
  if (validation_split >= 1) {
    stop("`validation_split` must be in [0, 1).", call. = FALSE)
  }
  if (!is.null(seed)) seed <- check_count(seed, "seed", min = 0L)

  x_val <- NULL
  y_val <- NULL
  if (!is.null(validation_data)) {
    if (!is.list(validation_data) || is.null(validation_data$x) ||
        is.null(validation_data$y)) {
      stop("`validation_data` must be a list with elements `x` and `y`.",
           call. = FALSE)
    }
    x_val <- as_numeric_matrix(validation_data$x, "validation_data$x")
    y_val <- as_response(validation_data$y, nrow(x_val), "validation_data$y")
    if (!is.null(colnames(x)) && !is.null(colnames(x_val))) {
      x_val <- align_columns(list(col_names = colnames(x)), x_val, ncol(x))
    } else if (ncol(x_val) != ncol(x)) {
      stop("`validation_data$x` must have the same columns as `x`.",
           call. = FALSE)
    }
  } else if (validation_split > 0) {
    n_val <- floor(validation_split * nrow(x))
    if (n_val < 1L || n_val >= nrow(x)) {
      stop("`validation_split` leaves no rows for training or validation.",
           call. = FALSE)
    }
    val_idx <- with_seed(seed, sample.int(nrow(x), n_val))
    x_val <- x[val_idx, , drop = FALSE]
    y_val <- y[val_idx]
    x <- x[-val_idx, , drop = FALSE]
    y <- y[-val_idx]
  }
  n <- nrow(x)
  if (is.null(batch_size)) batch_size <- n
  batch_size <- min(check_count(batch_size, "batch_size"), n)

  pp <- if (isTRUE(model$fitted)) model$preprocess else
    fit_preprocess(x, y, standardize_x, standardize_y)
  if (!isTRUE(model$fitted)) {
    if (any(pp$constant) && n > 1L) {
      warning("Constant predictor(s) in the training data are fixed at their ",
              "training value: ",
              paste(if (is.null(pp$col_names)) which(pp$constant) else
                pp$col_names[pp$constant], collapse = ", "), call. = FALSE)
    }
    if (n == 1L) {
      warning("Only one training row: preprocessing uses scale 1 and the fit ",
              "is degenerate.", call. = FALSE)
    } else if (pp$y_constant) {
      warning("The response is constant in the training data.", call. = FALSE)
    }
  }
  xs <- apply_preprocess_x(pp, x)
  ys <- apply_preprocess_y(pp, y)
  xvs <- if (!is.null(x_val)) apply_preprocess_x(pp, x_val) else NULL
  yvs <- if (!is.null(y_val)) apply_preprocess_y(pp, y_val) else NULL

  schedule <- with_seed(seed, make_batch_schedule(n, batch_size, epochs))
  t0 <- proc.time()[["elapsed"]]
  res <- switch(backend,
    r = train_r(model$params, model$specs, xs, ys, xvs, yvs, schedule, lr,
                betas, eps, patience, min_delta, verbose),
    torch = train_torch(model$params, model$specs, xs, ys, xvs, yvs, schedule,
                        lr, betas, eps, patience, min_delta, verbose, dtype)
  )
  elapsed <- proc.time()[["elapsed"]] - t0
  model$params <- res$params
  model$preprocess <- pp
  model$history <- res$history
  model$fitted <- TRUE
  model$layer_input_range <- layer_input_ranges(model$params, model$specs, xs)
  model$train <- list(backend = backend,
                      dtype = if (backend == "r") "float64" else dtype,
                      epochs = epochs, batch_size = batch_size, lr = lr,
                      betas = betas, eps = eps, patience = patience,
                      min_delta = min_delta, seed = seed,
                      validation = !is.null(x_val), n_train = n,
                      n_val = if (is.null(x_val)) 0L else nrow(x_val),
                      best_epoch = res$best_epoch,
                      last_epoch = nrow(res$history), elapsed_sec = elapsed)
  validate_model(model)
}

#' Predict with a fitted Kolmogorov-Arnold network
#'
#' Applies the stored training preprocessing, evaluates the network and
#' returns predictions on the original scale of the response. Columns of
#' `newdata` are matched by name when the model was trained with names
#' (order does not matter, extra columns are ignored, missing ones are an
#' error).
#'
#' @param object A fitted model.
#' @param newdata Numeric matrix, data frame or vector of predictors.
#' @param backend `"r"` (default, base R, no libtorch needed) or `"torch"`.
#' @param dtype Floating point type for `backend = "torch"`.
#' @param ... Unused.
#' @return Numeric vector of predictions.
#' @examples
#' set.seed(2)
#' x <- matrix(runif(100, -1, 1), ncol = 1, dimnames = list(NULL, "x1"))
#' fit <- thangld_kan_fit(thangld_kan(c(1, 1), seed = 1), x, x[, 1]^3, epochs = 20,
#'               lr = 0.02, backend = "r")
#' predict(fit, data.frame(x1 = c(-0.5, 0, 0.5)))
#' @export
predict.thangld_kan <- function(object, newdata, backend = c("r", "torch"),
                        dtype = c("float64", "float32"), ...) {
  validate_model(object)
  if (!isTRUE(object$fitted)) {
    stop("The model is not fitted; use thangld_kan_fit() first.", call. = FALSE)
  }
  backend <- match.arg(backend)
  dtype <- match.arg(dtype)
  x <- as_numeric_matrix(newdata, "newdata")
  x <- align_columns(object$preprocess, x, object$widths[1L])
  xs <- apply_preprocess_x(object$preprocess, x)
  out <- switch(backend,
    r = drop(engine_forward(object$params, object$specs, xs)$out),
    torch = engine_forward_torch(object$params, object$specs, xs, dtype)
  )
  invert_preprocess_y(object$preprocess, out)
}

#' Regression metrics
#'
#' @param truth,estimate Numeric vectors of equal length.
#' @return Named vector `rmse`, `mae`, `r2`; `r2` is `NA` when `truth` has
#'   zero variance.
#' @examples
#' thangld_kan_metrics(c(1, 2, 3), c(1.1, 1.9, 3.2))
#' @export
thangld_kan_metrics <- function(truth, estimate) {
  if (!is.numeric(truth) || !is.numeric(estimate) ||
      length(truth) != length(estimate) || !length(truth)) {
    stop("`truth` and `estimate` must be numeric vectors of equal length.",
         call. = FALSE)
  }
  e <- estimate - truth
  sst <- sum((truth - mean(truth))^2)
  c(rmse = sqrt(mean(e^2)), mae = mean(abs(e)),
    r2 = if (sst > 0) 1 - sum(e^2) / sst else NA_real_)
}

#' Evaluate a learned edge function
#'
#' Returns the edge function of layer `layer` from input node `input` to
#' output node `output` on a grid. An edge function is one component of the
#' fitted network: in a multi-layer network it is not the partial effect of
#' a predictor, and it is never a causal effect.
#'
#' @param model A model (fitted or not).
#' @param layer,input,output Edge indices (1-based).
#' @param x Optional grid of edge inputs; by default `n` points over the
#'   range of this edge's input in the training data (or `c(-3, 3)`).
#' @param n Number of grid points.
#' @param scale `"model"` (layer-input scale) or `"original"` (layer 1 of a
#'   fitted model: original units of the predictor).
#' @return A data frame with `x`, `h` (model-scale layer input), the
#'   basis-specific components and `phi` (edge output on the model scale).
#' @examples
#' m <- thangld_kan(c(2, 3, 1), seed = 1)
#' head(thangld_kan_edge(m, layer = 1, input = 2, output = 3, n = 5))
#' @export
thangld_kan_edge <- function(model, layer = 1L, input = 1L, output = 1L, x = NULL,
                     n = 101L, scale = c("model", "original")) {
  validate_model(model)
  scale <- match.arg(scale)
  layer <- check_count(layer, "layer")
  if (layer > length(model$params)) {
    stop("`layer` exceeds the number of layers.", call. = FALSE)
  }
  input <- check_count(input, "input")
  output <- check_count(output, "output")
  if (input > model$widths[layer]) stop("`input` out of range.", call. = FALSE)
  if (output > model$widths[layer + 1L]) {
    stop("`output` out of range.", call. = FALSE)
  }
  if (scale == "original" && (layer != 1L || !isTRUE(model$fitted))) {
    stop("`scale = \"original\"` is available only for layer 1 of a fitted ",
         "model.", call. = FALSE)
  }
  pp <- model$preprocess
  if (is.null(x)) {
    n <- check_count(n, "n", min = 2L)
    rng <- if (scale == "original") c(pp$x_min[input], pp$x_max[input]) else
      if (isTRUE(model$fitted)) model$layer_input_range[[layer]][, input] else
        c(-3, 3)
    if (rng[1L] == rng[2L]) rng <- rng + c(-1, 1)
    x <- seq(rng[1L], rng[2L], length.out = n)
  } else {
    if (!is.numeric(x) || any(!is.finite(x))) {
      stop("`x` must be a finite numeric vector.", call. = FALSE)
    }
  }
  h <- if (scale == "original") (x - pp$x_center[input]) / pp$x_scale[input] else x
  comp <- layer_edge_r(h, model$params[[layer]], model$specs[[layer]],
                       input, output)
  structure(data.frame(x = x, h = h, comp),
            edge = c(layer = layer, input = input, output = output),
            scale = scale)
}

#' Plot a Kolmogorov-Arnold network
#'
#' `type = "edge"` draws one learned edge function (see [thangld_kan_edge()]);
#' `type = "loss"` draws the loss history of a fitted model.
#'
#' @param x A model.
#' @param type `"edge"` or `"loss"`.
#' @param layer,input,output,scale Passed to [thangld_kan_edge()].
#' @param ... Graphical parameters passed to [graphics::plot()].
#' @return `x`, invisibly.
#' @examples
#' plot(thangld_kan(c(1, 1), seed = 1))
#' @export
plot.thangld_kan <- function(x, type = c("edge", "loss"), layer = 1L, input = 1L,
                     output = 1L, scale = c("model", "original"), ...) {
  type <- match.arg(type)
  scale <- match.arg(scale)
  if (type == "loss") {
    if (!isTRUE(x$fitted)) stop("The model is not fitted.", call. = FALSE)
    h <- x$history
    yl <- range(c(h$train_loss, h$val_loss), na.rm = TRUE)
    graphics::plot(h$epoch, h$train_loss, type = "l", log = "y", ylim = yl,
                   xlab = "epoch", ylab = "MSE (standardized scale)", ...)
    if (any(!is.na(h$val_loss))) {
      graphics::lines(h$epoch, h$val_loss, lty = 2)
      graphics::legend("topright", c("train", "validation"), lty = 1:2,
                       bty = "n")
    }
    return(invisible(x))
  }
  e <- thangld_kan_edge(x, layer, input, output, scale = scale)
  lab <- if (scale == "original" && !is.null(x$preprocess$col_names))
    x$preprocess$col_names[input] else
      sprintf("h[%d] (layer %d input)", input, layer)
  graphics::plot(e$x, e$phi, type = "l", xlab = lab,
                 ylab = sprintf("phi[%d,%d]", output, input),
                 main = sprintf("Edge: layer %d, %d -> %d", layer, input,
                                output), ...)
  invisible(x)
}

#' Save and load a Kolmogorov-Arnold network
#'
#' A model is plain R data, saved with [saveRDS()] together with metadata
#' (format version, package version, R version, time); it can be reloaded
#' with or without libtorch.
#'
#' @param model A model.
#' @param file Path of the `.rds` file.
#' @return `thangld_kan_save()` returns `file` invisibly; `thangld_kan_load()` returns the
#'   model, with saving metadata in `attr(, "saved")`.
#' @examples
#' m <- thangld_kan(c(2, 1), seed = 1)
#' f <- tempfile(fileext = ".rds")
#' thangld_kan_save(m, f)
#' identical(m$params, thangld_kan_load(f)$params)
#' @export
thangld_kan_save <- function(model, file) {
  validate_model(model)
  if (!is.character(file) || length(file) != 1L || is.na(file)) {
    stop("`file` must be a single path.", call. = FALSE)
  }
  obj <- list(format = "thangldKANr_model", format_version = 1L,
              package_version = as.character(utils::packageVersion("thangldKANr")),
              r_version = R.version.string,
              saved_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
              model = unclass(model))
  saveRDS(obj, file)
  invisible(file)
}

#' @rdname thangld_kan_save
#' @export
thangld_kan_load <- function(file) {
  if (!is.character(file) || length(file) != 1L || !file.exists(file)) {
    stop("File not found: ", file, call. = FALSE)
  }
  obj <- readRDS(file)
  if (!is.list(obj) || !identical(obj$format, "thangldKANr_model")) {
    stop("Not a thangldKANr model file.", call. = FALSE)
  }
  if (!is.numeric(obj$format_version) || obj$format_version > 1L) {
    stop("The model was saved by a newer thangldKANr version.", call. = FALSE)
  }
  model <- structure(obj$model, class = "thangld_kan")
  validate_model(model)
  attr(model, "saved") <- obj[c("format_version", "package_version",
                                "r_version", "saved_at")]
  model
}
