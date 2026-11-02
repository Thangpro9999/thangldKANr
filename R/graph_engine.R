# Graph KAN engine ------------------------------------------------------------------
#
# Layer l: H_l = KAN_l(U_l), H_0 = standardized node features, with
# U_l = A_hat %*% H_{l-1} (default) or U_l = [H_{l-1}, A_hat %*% H_{l-1}] with the
# self branch (the node's own state next to its neighborhood mean).
# Loss: mean squared error over the training nodes only (all nodes are
# propagated; labels of other nodes are never used). Full-batch Adam: one
# step per epoch, then train and validation losses at the new parameters.

#' Layer input: neighborhood mean, optionally next to the node's own state
#' @noRd
graph_layer_input <- function(h, a_hat, spec) {
  ah <- a_hat %*% h
  if (isTRUE(spec$self_branch)) cbind(h, ah) else ah
}

#' Gradient with respect to H_{l-1} from the gradient with respect to U_l
#' @noRd
graph_layer_input_backward <- function(du, a_hat, spec) {
  if (!isTRUE(spec$self_branch)) return(crossprod(a_hat, du))
  p <- ncol(du) %/% 2L
  du[, seq_len(p), drop = FALSE] + crossprod(a_hat, du[, p + seq_len(p), drop = FALSE])
}

#' Forward pass in base R
#' @noRd
graph_forward <- function(params, specs, x, a_hat, keep = FALSE) {
  h <- x
  outs <- vector("list", length(params))
  caches <- if (keep) vector("list", length(params)) else NULL
  for (l in seq_along(params)) {
    r <- layer_forward_r(graph_layer_input(h, a_hat, specs[[l]]), params[[l]], specs[[l]])
    h <- r$out
    outs[[l]] <- h
    if (keep) caches[[l]] <- r$cache
  }
  list(out = drop(h), outs = outs, caches = caches)
}

#' Masked MSE and analytic gradients (parameters and node features)
#' @noRd
graph_gradient <- function(params, specs, x, a_hat, y, nodes) {
  f <- graph_forward(params, specs, x, a_hat, keep = TRUE)
  r <- f$out[nodes] - y[nodes]
  loss <- mean(r^2)
  g <- matrix(0, nrow(x), 1L)
  g[nodes, 1L] <- 2 * r / length(nodes)
  grads <- vector("list", length(params))
  for (l in rev(seq_along(params))) {
    b <- layer_backward_r(f$caches[[l]], g, params[[l]], specs[[l]])
    grads[[l]] <- b$grads
    g <- graph_layer_input_backward(b$dh, a_hat, specs[[l]])
  }
  list(loss = loss, params = grads, x = g)
}

#' @noRd
graph_loss <- function(params, specs, x, a_hat, y, nodes) {
  mean((graph_forward(params, specs, x, a_hat)$out[nodes] - y[nodes])^2)
}

#' Central differences (order 2 or 4) of the masked loss
#' @noRd
graph_fd_gradient <- function(params, specs, x, a_hat, y, nodes, step = 1e-5,
                              order = 2L) {
  w <- if (order == 4L) list(c(-2, -1, 1, 2), c(1, -8, 8, -1) / 12) else
    list(c(-1, 1), c(-1, 1) / 2)
  d1 <- function(fun) sum(vapply(seq_along(w[[1L]]), function(s)
    w[[2L]][s] * fun(w[[1L]][s] * step), numeric(1))) / step
  gp <- lapply(seq_along(params), function(l) {
    lapply(stats::setNames(names(params[[l]]), names(params[[l]])), function(nm) {
      a <- params[[l]][[nm]]
      out <- a
      for (k in seq_along(a)) {
        out[k] <- d1(function(e) {
          pp <- params
          pp[[l]][[nm]][k] <- a[k] + e
          graph_loss(pp, specs, x, a_hat, y, nodes)
        })
      }
      out
    })
  })
  gx <- x
  for (k in seq_along(x)) {
    gx[k] <- d1(function(e) {
      xx <- x
      xx[k] <- x[k] + e
      graph_loss(params, specs, xx, a_hat, y, nodes)
    })
  }
  list(params = gp, x = gx)
}




#' Full-batch Adam in base R
#' @noRd
graph_train_r <- function(params, specs, xs, a_hat, ys, train, val, epochs, lr,
                          betas, eps, patience, min_delta, verbose) {
  state <- lapply(params, function(pl) lapply(pl, function(a)
    list(t = 0L, m = array(0, dim(a)), v = array(0, dim(a)))))
  tr <- new_tracker(epochs, length(val) > 0L, patience, min_delta)
  for (epoch in seq_len(epochs)) {
    g <- graph_gradient(params, specs, xs, a_hat, ys, train)
    for (l in seq_along(params)) for (nm in names(params[[l]])) {
      s <- adam_step_r(params[[l]][[nm]], g$params[[l]][[nm]], state[[l]][[nm]],
                       lr, betas, eps)
      params[[l]][[nm]] <- s$param
      state[[l]][[nm]] <- s$state
    }
    out <- graph_forward(params, specs, xs, a_hat)$out
    trl <- mean((out[train] - ys[train])^2)
    vl <- if (length(val)) mean((out[val] - ys[val])^2) else NA_real_
    check_finite_loss(trl, epoch)
    report_progress(verbose, epoch, epochs, trl, vl)
    cur <- params
    if (tr$update(epoch, trl, vl, function() cur)) break
  }
  list(params = if (!is.null(tr$best_params)) tr$best_params else params,
       history = tr$history(), best_epoch = tr$best_epoch)
}

#' torch module: parameters named l{l}_{name}; forward(x, a_hat)
#' @noRd
graph_torch_module <- function(params, specs, dtype = "float64") {
  require_torch()
  td <- torch_dtype_from(dtype)
  gen <- torch::nn_module(
    "thangld_graph_kan_module",
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
    forward = function(x, a_hat) {
      h <- x
      for (l in seq_along(self$specs)) {
        ah <- torch::torch_matmul(a_hat, h)
        u <- if (isTRUE(self$specs[[l]]$self_branch)) torch::torch_cat(list(h, ah), dim = 2L) else ah
        h <- layer_forward_torch(u, self$layer_params(l), self$specs[[l]])
      }
      h
    }
  )
  gen(params, specs)
}


#' @noRd
graph_forward_torch <- function(params, specs, x, a_hat, dtype = "float64") {
  td <- torch_dtype_from(dtype)
  net <- graph_torch_module(params, specs, dtype)
  torch::with_no_grad(out <- net(r_tensor(x, dtype = td),
                                 r_tensor(a_hat, dtype = td)))
  drop(as_r_array(out))
}

#' @noRd
graph_gradient_torch <- function(params, specs, x, a_hat, y, nodes, dtype = "float64") {
  td <- torch_dtype_from(dtype)
  net <- graph_torch_module(params, specs, dtype)
  xt <- r_tensor(x, dtype = td, requires_grad = TRUE)
  at <- r_tensor(a_hat, dtype = td)
  idx <- r_tensor(as.integer(nodes), dtype = torch::torch_long())
  yt <- r_tensor(matrix(y[nodes], ncol = 1L), dtype = td)
  loss <- torch::nnf_mse_loss(net(xt, at)[idx, , drop = FALSE], yt)
  loss$backward()
  grads <- lapply(seq_along(params), function(l) {
    out <- lapply(names(params[[l]]), function(nm) {
      array(as_r_array(net[[paste0("l", l, "_", nm)]]$grad), dim(params[[l]][[nm]]))
    })
    stats::setNames(out, names(params[[l]]))
  })
  list(loss = loss$item(), params = grads, x = as_r_array(xt$grad))
}

#' Full-batch Adam with torch autograd and torch::optim_adam
#' @noRd
graph_train_torch <- function(params, specs, xs, a_hat, ys, train, val, epochs,
                              lr, betas, eps, patience, min_delta, verbose,
                              dtype = "float64") {
  td <- torch_dtype_from(dtype)
  net <- graph_torch_module(params, specs, dtype)
  opt <- torch::optim_adam(net$parameters, lr = lr, betas = betas, eps = eps)
  xt <- r_tensor(xs, dtype = td)
  at <- r_tensor(a_hat, dtype = td)
  it <- r_tensor(as.integer(train), dtype = torch::torch_long())
  yt <- r_tensor(matrix(ys[train], ncol = 1L), dtype = td)
  tr <- new_tracker(epochs, length(val) > 0L, patience, min_delta)
  for (epoch in seq_len(epochs)) {
    opt$zero_grad()
    loss <- torch::nnf_mse_loss(net(xt, at)[it, , drop = FALSE], yt)
    loss$backward()
    opt$step()
    torch::with_no_grad(out <- drop(as_r_array(net(xt, at))))
    trl <- mean((out[train] - ys[train])^2)
    vl <- if (length(val)) mean((out[val] - ys[val])^2) else NA_real_
    check_finite_loss(trl, epoch)
    report_progress(verbose, epoch, epochs, trl, vl)
    cur <- torch_module_params(net, params)
    if (tr$update(epoch, trl, vl, function() cur)) break
  }
  list(params = if (!is.null(tr$best_params)) tr$best_params else
         torch_module_params(net, params),
       history = tr$history(), best_epoch = tr$best_epoch)
}
