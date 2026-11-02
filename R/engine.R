# Generic KAN engine -----------------------------------------------------------
#
# A model stores, for each layer l, a named list of parameter arrays
# `params[[l]]` and a layer specification `specs[[l]]`. Each basis (see
# registry.R) provides:
#   layer_param_dims(p, q, spec)          named list of array dims
#   layer_init(p, q, spec)                named list of initial arrays
#   layer_forward_r(h, par, spec)         list(out = n x q, cache)
#   layer_backward_r(cache, g, par, spec) list(grads = named list, dh = n x p)
#   layer_forward_torch(h, par, spec)     tensor n x q (par: named tensors)
#   layer_check_params(par, spec)         error if parameters left their domain
#   layer_edge_r(x, par, spec, i, j)      data.frame of edge components
#   layer_naive_forward(h, par, spec)     independent loop implementation
# Everything below works on the model scale (already preprocessed inputs).

#' Forward pass in base R
#' @noRd
engine_forward <- function(params, specs, x, keep = FALSE) {
  h <- x
  outs <- vector("list", length(params))
  caches <- if (keep) vector("list", length(params)) else NULL
  for (l in seq_along(params)) {
    r <- layer_forward_r(h, params[[l]], specs[[l]])
    h <- r$out
    outs[[l]] <- h
    if (keep) caches[[l]] <- r$cache
  }
  list(out = h, outs = outs, caches = caches)
}

#' MSE loss and analytic gradients (parameters and input)
#' @noRd
engine_gradient <- function(params, specs, x, y) {
  n <- nrow(x)
  f <- engine_forward(params, specs, x, keep = TRUE)
  resid <- drop(f$out) - y
  g <- matrix(2 * resid / n, n, 1L)
  grads <- vector("list", length(params))
  for (l in rev(seq_along(params))) {
    b <- layer_backward_r(f$caches[[l]], g, params[[l]], specs[[l]])
    grads[[l]] <- b$grads[names(params[[l]])]
    g <- b$dh
  }
  list(loss = mean(resid^2), params = grads, x = g)
}

#' Central-difference gradients (validation only)
#'
#' order 2: (f(t+h) - f(t-h)) / 2h, truncation O(h^2).
#' order 4: (-f(t+2h) + 8 f(t+h) - 8 f(t-h) + f(t-2h)) / 12h, O(h^4).
#' @noRd
engine_fd_gradient <- function(params, specs, x, y, step = 1e-5, order = 2L) {
  loss_at <- function(pp, xx) {
    mean((drop(engine_forward(pp, specs, xx)$out) - y)^2)
  }
  deriv <- function(f) {
    if (order == 2L) return((f(step) - f(-step)) / (2 * step))
    (-f(2 * step) + 8 * f(step) - 8 * f(-step) + f(-2 * step)) / (12 * step)
  }
  grads <- lapply(params, function(pl) lapply(pl, function(a) a * 0))
  for (l in seq_along(params)) {
    for (nm in names(params[[l]])) {
      for (j in seq_along(params[[l]][[nm]])) {
        grads[[l]][[nm]][j] <- deriv(function(e) {
          pp <- params
          pp[[l]][[nm]][j] <- pp[[l]][[nm]][j] + e
          loss_at(pp, x)
        })
      }
    }
  }
  gx <- x * 0
  for (j in seq_along(x)) {
    gx[j] <- deriv(function(e) {
      xx <- x
      xx[j] <- xx[j] + e
      loss_at(params, xx)
    })
  }
  list(loss = loss_at(params, x), params = grads, x = gx)
}

#' Adam update (same arithmetic as torch.optim.Adam, no weight decay)
#' @noRd
adam_step_r <- function(param, grad, state, lr, betas, eps) {
  state$t <- state$t + 1L
  state$m <- betas[1L] * state$m + (1 - betas[1L]) * grad
  state$v <- betas[2L] * state$v + (1 - betas[2L]) * grad * grad
  bc1 <- 1 - betas[1L]^state$t
  bc2 <- 1 - betas[2L]^state$t
  denom <- sqrt(state$v) / sqrt(bc2) + eps
  list(param = param - (lr / bc1) * state$m / denom, state = state)
}

#' Mini-batch schedule: list (epochs) of lists (batches) of row indices
#' @noRd
make_batch_schedule <- function(n, batch_size, epochs) {
  lapply(seq_len(epochs), function(e) {
    perm <- if (batch_size >= n) seq_len(n) else sample.int(n)
    split(perm, ceiling(seq_along(perm) / batch_size))
  })
}

#' Early stopping / history bookkeeping shared by both backends
#' @noRd
new_tracker <- function(n_epochs, has_val, patience, min_delta) {
  env <- new.env(parent = emptyenv())
  env$train_loss <- rep(NA_real_, n_epochs)
  env$val_loss <- rep(NA_real_, n_epochs)
  env$best <- Inf
  env$best_epoch <- NULL
  env$best_params <- NULL
  env$wait <- 0L
  env$last <- 0L
  env$update <- function(epoch, train_loss, val_loss, params_fun) {
    env$train_loss[epoch] <- train_loss
    env$val_loss[epoch] <- val_loss
    env$last <- epoch
    if (!has_val) return(FALSE)
    if (val_loss < env$best - min_delta || is.null(env$best_epoch)) {
      env$best <- val_loss
      env$best_epoch <- epoch
      env$best_params <- params_fun()
      env$wait <- 0L
    } else {
      env$wait <- env$wait + 1L
    }
    !is.null(patience) && env$wait >= patience
  }
  env$history <- function() {
    k <- env$last
    data.frame(epoch = seq_len(k), train_loss = env$train_loss[seq_len(k)],
               val_loss = env$val_loss[seq_len(k)])
  }
  env
}

#' @noRd
report_progress <- function(verbose, epoch, n_epochs, tr, va) {
  if (verbose && (epoch %% max(1L, n_epochs %/% 10L) == 0L || epoch == 1L)) {
    message(sprintf("epoch %d: train %.6g%s", epoch, tr,
                    if (is.na(va)) "" else sprintf(", val %.6g", va)))
  }
}

#' @noRd
check_finite_loss <- function(trl, epoch) {
  if (!is.finite(trl)) {
    stop(sprintf("Training diverged at epoch %d (non-finite loss); ", epoch),
         "try a smaller `lr`.", call. = FALSE)
  }
}

#' Training loop, base R backend
#' @noRd
train_r <- function(params, specs, xs, ys, xvs, yvs, schedule, lr, betas, eps,
                    patience, min_delta, verbose) {
  states <- lapply(params, function(pl) {
    lapply(pl, function(a) list(t = 0L, m = a * 0, v = a * 0))
  })
  tr <- new_tracker(length(schedule), !is.null(xvs), patience, min_delta)
  loss_on <- function(xx, yy) {
    mean((drop(engine_forward(params, specs, xx)$out) - yy)^2)
  }
  for (epoch in seq_along(schedule)) {
    for (idx in schedule[[epoch]]) {
      g <- engine_gradient(params, specs, xs[idx, , drop = FALSE], ys[idx])
      for (l in seq_along(params)) {
        for (nm in names(params[[l]])) {
          s <- adam_step_r(params[[l]][[nm]], g$params[[l]][[nm]],
                           states[[l]][[nm]], lr, betas, eps)
          params[[l]][[nm]] <- s$param
          states[[l]][[nm]] <- s$state
        }
        layer_check_params(params[[l]], specs[[l]])
      }
    }
    trl <- loss_on(xs, ys)
    val <- if (is.null(xvs)) NA_real_ else loss_on(xvs, yvs)
    check_finite_loss(trl, epoch)
    report_progress(verbose, epoch, length(schedule), trl, val)
    if (tr$update(epoch, trl, val, function() params)) break
  }
  best <- if (!is.null(tr$best_params)) tr$best_params else params
  list(params = best, history = tr$history(), best_epoch = tr$best_epoch)
}

#' torch module holding all layer parameters
#' @noRd
engine_torch_module <- function(params, specs, dtype = "float64") {
  require_torch()
  td <- torch_dtype_from(dtype)
  gen <- torch::nn_module(
    "kan_engine_module",
    initialize = function(params, specs) {
      self$specs <- specs
      self$pnames <- lapply(params, names)
      for (l in seq_along(params)) {
        for (nm in names(params[[l]])) {
          self[[paste0("l", l, "_", nm)]] <- torch::nn_parameter(
            r_tensor(params[[l]][[nm]], dtype = td))
        }
      }
    },
    layer_params = function(l) {
      out <- lapply(self$pnames[[l]], function(nm) self[[paste0("l", l, "_", nm)]])
      stats::setNames(out, self$pnames[[l]])
    },
    forward = function(x) {
      h <- x
      for (l in seq_along(self$specs)) {
        h <- layer_forward_torch(h, self$layer_params(l), self$specs[[l]])
      }
      h
    }
  )
  gen(params, specs)
}

#' Copy torch parameters back to plain R arrays (original dims kept)
#' @noRd
torch_module_params <- function(net, like) {
  lapply(seq_along(like), function(l) {
    out <- lapply(names(like[[l]]), function(nm) {
      a <- as_r_array(net[[paste0("l", l, "_", nm)]])
      array(a, dim(like[[l]][[nm]]))
    })
    stats::setNames(out, names(like[[l]]))
  })
}

#' Forward pass with torch (model scale)
#' @noRd
engine_forward_torch <- function(params, specs, x, dtype = "float64") {
  net <- engine_torch_module(params, specs, dtype)
  xt <- r_tensor(x, dtype = torch_dtype_from(dtype))
  torch::with_no_grad(out <- net(xt))
  drop(as_r_array(out))
}

#' Loss and autograd gradients with torch (model scale)
#' @noRd
engine_gradient_torch <- function(params, specs, x, y, dtype = "float64") {
  td <- torch_dtype_from(dtype)
  net <- engine_torch_module(params, specs, dtype)
  xt <- r_tensor(x, dtype = td, requires_grad = TRUE)
  yt <- r_tensor(matrix(y, ncol = 1L), dtype = td)
  loss <- torch::nnf_mse_loss(net(xt), yt)
  loss$backward()
  grads <- lapply(seq_along(params), function(l) {
    out <- lapply(names(params[[l]]), function(nm) {
      gt <- net[[paste0("l", l, "_", nm)]]$grad
      if (is.null(gt) || inherits(gt, "NULL")) return(params[[l]][[nm]] * 0)
      array(as_r_array(gt), dim(params[[l]][[nm]]))
    })
    stats::setNames(out, names(params[[l]]))
  })
  list(loss = loss$item(), params = grads, x = as_r_array(xt$grad))
}

#' Training loop, torch backend
#' @noRd
train_torch <- function(params, specs, xs, ys, xvs, yvs, schedule, lr, betas,
                        eps, patience, min_delta, verbose, dtype) {
  td <- torch_dtype_from(dtype)
  net <- engine_torch_module(params, specs, dtype)
  opt <- torch::optim_adam(net$parameters, lr = lr, betas = betas, eps = eps)
  xt <- r_tensor(xs, dtype = td)
  yt <- r_tensor(matrix(ys, ncol = 1L), dtype = td)
  xvt <- if (is.null(xvs)) NULL else r_tensor(xvs, dtype = td)
  yvt <- if (is.null(yvs)) NULL else
    r_tensor(matrix(yvs, ncol = 1L), dtype = td)
  tr <- new_tracker(length(schedule), !is.null(xvs), patience, min_delta)
  eval_loss <- function(a, b) {
    torch::with_no_grad(v <- torch::nnf_mse_loss(net(a), b)$item())
    v
  }
  for (epoch in seq_along(schedule)) {
    for (idx in schedule[[epoch]]) {
      opt$zero_grad()
      loss <- torch::nnf_mse_loss(net(xt[idx, , drop = FALSE]),
                                  yt[idx, , drop = FALSE])
      loss$backward()
      opt$step()
    }
    cur <- torch_module_params(net, params)
    for (l in seq_along(cur)) layer_check_params(cur[[l]], specs[[l]])
    trl <- eval_loss(xt, yt)
    val <- if (is.null(xvt)) NA_real_ else eval_loss(xvt, yvt)
    check_finite_loss(trl, epoch)
    report_progress(verbose, epoch, length(schedule), trl, val)
    if (tr$update(epoch, trl, val, function() cur)) break
  }
  best <- if (!is.null(tr$best_params)) tr$best_params else
    torch_module_params(net, params)
  list(params = best, history = tr$history(), best_epoch = tr$best_epoch)
}

#' Range of each layer's inputs on the training data (for edge plots)
#' @noRd
layer_input_ranges <- function(params, specs, xs) {
  h <- xs
  out <- vector("list", length(params))
  for (l in seq_along(params)) {
    out[[l]] <- apply(h, 2L, range)
    h <- layer_forward_r(h, params[[l]], specs[[l]])$out
  }
  out
}
