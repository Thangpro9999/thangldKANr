#' Create a graph KAN architecture
#'
#' A graph KAN layer first averages node states over neighborhoods with the
#' normalized adjacency matrix \eqn{\hat A} ([thangld_graph_kan_adjacency()]) and then
#' applies a Kolmogorov-Arnold layer to every node,
#' \deqn{H^{(l)} = \mathrm{KAN}_l(\hat A H^{(l-1)}), \qquad
#'   \mathrm{KAN}_l(U)_{v,j} = \sum_i \sum_{k=0}^{K} c_{i,j,k} T_k(\tanh U_{v,i}),}
#' which is the "aggregate, then KAN" architecture of graph KANs (Kiamari,
#' Kiamari and Krishnamachari, 2024; the GraphKAN prototype of Zhang, 2024,
#' uses the same order with Fourier edges). By default edges are Chebyshev
#' expansions (layout `c(in, out, K + 1)`, no bias, degree 3); any basis of
#' [thangld_kan()] can be used instead (`basis`, `...`, `options`). With
#' `L` layers a node's prediction depends on its `L`-hop neighborhood. With
#' no graph (identity \eqn{\hat A}) the model reduces to a Chebyshev KAN on
#' the node's own features, which is the natural baseline.
#'
#' With `self_branch = TRUE` every layer receives the node's own state next
#' to its neighborhood mean,
#' \deqn{H^{(l)} = \mathrm{KAN}_l([H^{(l-1)}, \hat A H^{(l-1)}]),}
#' so a layer has \eqn{2 \times} the inputs (the "concatenate self and
#' neighbors" update of GraphSAGE, Hamilton, Ying and Leskovec, 2017). The
#' network can then use the node's own features, its one-hop mean or both
#' in any layer, so a two-layer model can still fit a target that depends on
#' one hop only, or on the node's own features only. The default (`FALSE`)
#' is the plain "aggregate, then KAN" architecture above.
#'
#' Task: transductive node regression (one numeric target per node,
#' one graph, full-batch training).
#'
#' @param widths Integer layer widths; the first is the number of node
#'   features, the last must be 1.
#' @param basis Edge basis, one value or one per layer (see [kan_bases()]).
#' @param ... Named basis-specific arguments, as in [thangld_kan()]. For
#'   `"chebyshev"` the `degree` defaults to 3 here.
#' @param options `NULL` or a named list of basis-specific arguments per
#'   basis, as in [thangld_kan()].
#' @param self_branch Logical; feed `[H, A_hat H]` instead of `A_hat H` to
#'   every layer (see Details).
#' @param seed Optional seed for the initialization.
#' @return An object of class `thangld_graph_kan`.
#' @references
#' Kiamari, M., Kiamari, M. and Krishnamachari, B. (2024). GKAN: Graph
#' Kolmogorov-Arnold Networks. \doi{10.48550/arXiv.2406.06470}
#'
#' Hamilton, W. L., Ying, R. and Leskovec, J. (2017). Inductive
#' representation learning on large graphs. Advances in Neural Information
#' Processing Systems 30. \doi{10.48550/arXiv.1706.02216}
#'
#' Kipf, T. N. and Welling, M. (2017). Semi-supervised classification with
#' graph convolutional networks. International Conference on Learning
#' Representations. \doi{10.48550/arXiv.1609.02907}
#' @examples
#' thangld_graph_kan(c(2, 4, 1), seed = 1)
#' thangld_graph_kan(c(2, 4, 1), self_branch = TRUE, seed = 1)
#' thangld_graph_kan(c(2, 4, 1), basis = "rbf", seed = 1)
#' @export
thangld_graph_kan <- function(widths, basis = "chebyshev", ...,
                              self_branch = FALSE, options = NULL,
                              seed = NULL) {
  if (!is.numeric(widths) || length(widths) < 2L || anyNA(widths) ||
      any(widths != round(widths)) || any(widths < 1)) {
    stop("`widths` must be at least two positive integers.", call. = FALSE)
  }
  widths <- as.integer(widths)
  if (widths[length(widths)] != 1L) {
    stop("The last width must be 1 (one target per node).", call. = FALSE)
  }
  nl <- length(widths) - 1L
  self_branch <- check_flag(self_branch, "self_branch")
  if (!is.null(seed)) seed <- check_count(seed, "seed", min = 0L)
  dots <- list(...)
  if ("chebyshev" %in% basis && is.null(dots$degree) &&
      is.null(options$chebyshev$degree)) {
    options$chebyshev$degree <- 3L
  }
  inputs <- widths[-length(widths)] * (1L + self_branch)
  specs <- build_layer_specs(c(inputs, 1L), basis, dots, options)
  specs <- lapply(specs, function(s) c(s, list(self_branch = self_branch)))
  params <- with_seed(seed, lapply(seq_len(nl), function(l)
    layer_init(n_inputs(widths[l], specs[[l]]), widths[l + 1L], specs[[l]])))
  structure(list(widths = widths, specs = specs, params = params, seed = seed,
                 fitted = FALSE), class = "thangld_graph_kan")
}

#' Inputs of a layer: width, doubled with the self branch
#' @noRd
n_inputs <- function(width, spec) width * (1L + isTRUE(spec$self_branch))

#' @noRd
validate_gkan <- function(model) {
  if (!inherits(model, "thangld_graph_kan")) {
    stop("`model` must be a thangld_graph_kan object.", call. = FALSE)
  }
  w <- model$widths
  inputs <- vapply(seq_len(length(w) - 1L), function(l)
    n_inputs(w[l], model$specs[[l]]), 1L)
  check_layer_params(model$params, model$specs, w, inputs)
  invisible(model)
}

#' @export
print.thangld_graph_kan <- function(x, ...) {
  cat("Graph KAN: widths", paste(x$widths, collapse = " -> "), "| layer input",
      if (isTRUE(x$specs[[1L]]$self_branch)) "[H, A_hat H]" else "A_hat H",
      "with A_hat = D^-1/2 (A + I) D^-1/2 | edges:",
      paste(vapply(x$specs, layer_describe, ""), collapse = ", "), "\n")
  if (isTRUE(x$fitted)) {
    cat("Fitted on", x$train$n_nodes, "nodes:", length(x$train$train), "train,",
        length(x$train$val), "validation | backend", x$train$backend,
        "| best epoch", if (is.null(x$train$best_epoch)) "-" else x$train$best_epoch,
        "of", nrow(x$history), "\n")
  } else cat("Not fitted.\n")
  invisible(x)
}

#' @noRd
check_nodes <- function(idx, n, name) {
  if (is.null(idx)) return(integer())
  if (is.logical(idx)) {
    if (length(idx) != n || anyNA(idx)) {
      stop(sprintf("Logical `%s` must have one non-missing value per node.", name),
           call. = FALSE)
    }
    return(which(idx))
  }
  if (!is.numeric(idx) || anyNA(idx) || any(idx != round(idx)) || any(idx < 1) ||
      any(idx > n) || anyDuplicated(idx)) {
    stop(sprintf("`%s` must be distinct node indices in 1..%d.", name, n), call. = FALSE)
  }
  sort(as.integer(idx))
}

#' Fit a graph KAN for node regression
#'
#' Transductive node regression on one graph: all nodes and their features are
#' propagated, the loss is the mean squared error over the `train` nodes, and
#' the `val` nodes are used only for early stopping (the epoch with the lowest
#' validation loss is restored). Labels of other nodes are never read; they may
#' be `NA`. Feature and target standardization constants are estimated on the
#' training nodes only. Training is full batch: one Adam step per epoch, after
#' which the training and validation losses are recorded.
#'
#' @param model A [thangld_graph_kan()] object.
#' @param x Node feature matrix or data frame (\eqn{n} rows, finite values).
#' @param graph Adjacency matrix or edge list (see [thangld_graph_kan_adjacency()]), or
#'   `NULL` for no graph (identity propagation; the node-wise KAN baseline).
#' @param y Numeric target, one value per node (`NA` allowed outside
#'   `train` and `val`).
#' @param train,val Node indices (or logical vectors) of training and
#'   validation nodes; they must not overlap.
#' @param epochs Maximum number of epochs (= Adam steps).
#' @param lr,betas,eps Adam settings (as `torch::optim_adam`).
#' @param patience Early-stopping patience in epochs (`NULL` for none).
#' @param min_delta Minimum validation improvement.
#' @param self_loops Passed to [thangld_graph_kan_adjacency()].
#' @param backend `"torch"` (autograd) or `"r"` (analytic gradients in base R;
#'   no libtorch needed).
#' @param dtype `"float64"` or `"float32"` (torch backend).
#' @param verbose Print progress.
#' @return The fitted `thangld_graph_kan` with `history`, `preprocess`, `a_hat`, `x`
#'   and `train` information.
#' @examples
#' g <- thangld_graph_kan_simulate(60, seed = 1)
#' y <- sin(g$x[, 1]) + drop(thangld_graph_kan_adjacency(g$adjacency) %*% g$x[, 2])
#' fit <- thangld_graph_kan_fit(thangld_graph_kan(c(2, 3, 1), seed = 1), g$x, g$adjacency, y,
#'                      train = 1:40, val = 41:50, epochs = 50, lr = 0.02,
#'                      backend = "r")
#' fit
#' @export
thangld_graph_kan_fit <- function(model, x, graph, y, train, val = NULL, epochs = 300L,
                          lr = 0.01, betas = c(0.9, 0.999), eps = 1e-8,
                          patience = NULL, min_delta = 0, self_loops = TRUE,
                          backend = c("torch", "r"), dtype = c("float64", "float32"),
                          verbose = FALSE) {
  validate_gkan(model)
  backend <- match.arg(backend)
  dtype <- match.arg(dtype)
  if (backend == "torch") require_torch()
  x <- as_numeric_matrix(x, "x")
  n <- nrow(x)
  if (ncol(x) != model$widths[1L]) {
    stop(sprintf("`x` has %d columns; the model expects %d.", ncol(x), model$widths[1L]),
         call. = FALSE)
  }
  a_hat <- if (is.null(graph)) diag(n) else thangld_graph_kan_adjacency(graph, n, self_loops)
  if (nrow(a_hat) != n) stop("`graph` and `x` have different numbers of nodes.", call. = FALSE)
  if (!is.numeric(y) || length(y) != n) {
    stop("`y` must be numeric with one value per node.", call. = FALSE)
  }
  train <- check_nodes(train, n, "train")
  val <- check_nodes(val, n, "val")
  if (length(train) < 2L) stop("At least two training nodes are needed.", call. = FALSE)
  if (length(intersect(train, val))) stop("`train` and `val` overlap.", call. = FALSE)
  if (any(!is.finite(y[c(train, val)]))) {
    stop("`y` must be finite on the training and validation nodes.", call. = FALSE)
  }
  epochs <- check_count(epochs, "epochs")
  lr <- check_number(lr, "lr", min = 0, strict = TRUE)
  if (!is.numeric(betas) || length(betas) != 2L || any(betas < 0 | betas >= 1)) {
    stop("`betas` must be two numbers in [0, 1).", call. = FALSE)
  }
  eps <- check_number(eps, "eps", min = 0)
  if (!is.null(patience)) patience <- check_count(patience, "patience")
  min_delta <- check_number(min_delta, "min_delta", min = 0)
  verbose <- check_flag(verbose, "verbose")
  pp <- graph_preprocess(x[train, , drop = FALSE], y[train])
  xs <- sweep(sweep(x, 2L, pp$x_center), 2L, pp$x_scale, `/`)
  ys <- (y - pp$y_center) / pp$y_scale
  ys[!seq_len(n) %in% c(train, val)] <- 0          # never used; avoids NA arithmetic
  args <- list(model$params, model$specs, xs, a_hat, ys, train, val, epochs, lr,
               betas, eps, patience, min_delta, verbose)
  t0 <- proc.time()[["elapsed"]]
  res <- if (backend == "r") do.call(graph_train_r, args) else
    do.call(graph_train_torch, c(args, list(dtype)))
  model$params <- res$params
  model$history <- res$history
  model$preprocess <- pp
  model$a_hat <- a_hat
  model$x <- x
  model$fitted <- TRUE
  model$train <- list(backend = backend, epochs = epochs, lr = lr, train = train,
                      val = val, n_nodes = n, self_loops = self_loops,
                      graph = !is.null(graph), best_epoch = res$best_epoch,
                      elapsed_sec = proc.time()[["elapsed"]] - t0)
  model
}

#' @noRd
graph_preprocess <- function(x, y) {
  sx <- apply(x, 2L, stats::sd)
  const <- !is.finite(sx) | sx == 0
  if (any(const)) {
    warning("Constant feature(s) on the training nodes: ",
            paste(which(const), collapse = ", "), "; they are only centered.",
            call. = FALSE)
  }
  sy <- stats::sd(y)
  list(x_center = colMeans(x), x_scale = ifelse(const, 1, sx),
       y_center = mean(y), y_scale = if (is.finite(sy) && sy > 0) sy else 1)
}

#' Predict node targets
#'
#' @param object A fitted [thangld_graph_kan_fit()] model.
#' @param newdata `NULL` (the fitted graph and features: predictions for all
#'   nodes, transductive) or a new node feature matrix, used with `graph`.
#' @param graph Graph for `newdata` (adjacency matrix, edge list or `NULL` for
#'   no graph); ignored when `newdata` is `NULL`.
#' @param backend `"r"` or `"torch"`.
#' @param ... Unused.
#' @return Numeric vector, one prediction per node, on the original scale.
#' @export
predict.thangld_graph_kan <- function(object, newdata = NULL, graph = NULL,
                              backend = c("r", "torch"), ...) {
  validate_gkan(object)
  if (!isTRUE(object$fitted)) stop("The model is not fitted.", call. = FALSE)
  backend <- match.arg(backend)
  if (is.null(newdata)) {
    x <- object$x
    a_hat <- object$a_hat
  } else {
    x <- as_numeric_matrix(newdata, "newdata")
    if (ncol(x) != object$widths[1L]) stop("`newdata` has the wrong number of columns.",
                                           call. = FALSE)
    a_hat <- if (is.null(graph)) diag(nrow(x)) else
      thangld_graph_kan_adjacency(graph, nrow(x), object$train$self_loops)
  }
  pp <- object$preprocess
  xs <- sweep(sweep(x, 2L, pp$x_center), 2L, pp$x_scale, `/`)
  out <- if (backend == "r") graph_forward(object$params, object$specs, xs, a_hat)$out else
    graph_forward_torch(object$params, object$specs, xs, a_hat)
  out * pp$y_scale + pp$y_center
}

#' Regression metrics
#'
#' @param truth,estimate Numeric vectors of equal length.
#' @return Named vector `rmse`, `mae`, `r2`.
#' @examples
#' thangld_graph_kan_metrics(c(1, 2, 3), c(1.1, 1.9, 3.2))
#' @export
thangld_graph_kan_metrics <- function(truth, estimate) {
  if (!is.numeric(truth) || !is.numeric(estimate) || length(truth) != length(estimate) ||
      any(!is.finite(c(truth, estimate)))) {
    stop("`truth` and `estimate` must be finite numeric vectors of equal length.",
         call. = FALSE)
  }
  e <- truth - estimate
  ss <- sum((truth - mean(truth))^2)
  c(rmse = sqrt(mean(e^2)), mae = mean(abs(e)), r2 = if (ss > 0) 1 - sum(e^2) / ss else NA)
}

#' Forward pass and gradients in base R
#'
#' `thangld_graph_kan_forward_r()` evaluates the network on standardized features
#' `x` with propagation matrix `a_hat`. `thangld_graph_kan_gradient_r()` returns the
#' masked mean squared error over `nodes` and its analytic gradients with
#' respect to every coefficient and to `x`; `thangld_graph_kan_fd_gradient_r()` the
#' same by central differences (order 2 or 4), for checking.
#'
#' @param model A `thangld_graph_kan`.
#' @param x Standardized node features.
#' @param a_hat Propagation matrix ([thangld_graph_kan_adjacency()]).
#' @param layers Return every layer's output.
#' @param y Target (model scale).
#' @param nodes Indices of the nodes in the loss.
#' @param step,order Finite-difference step and order.
#' @return See Description.
#' @examples
#' g <- thangld_graph_kan_simulate(8, seed = 2)
#' m <- thangld_graph_kan(c(2, 2, 1), seed = 1)
#' a <- thangld_graph_kan_adjacency(g$adjacency)
#' thangld_graph_kan_forward_r(m, g$x, a)
#' thangld_graph_kan_gradient_r(m, g$x, a, rnorm(8), 1:5)$loss
#' @export
thangld_graph_kan_forward_r <- function(model, x, a_hat, layers = FALSE) {
  validate_gkan(model)
  f <- graph_forward(model$params, model$specs, as_numeric_matrix(x, "x"), a_hat)
  if (layers) f$outs else f$out
}

#' @rdname thangld_graph_kan_forward_r
#' @export
thangld_graph_kan_gradient_r <- function(model, x, a_hat, y, nodes) {
  validate_gkan(model)
  graph_gradient(model$params, model$specs, as_numeric_matrix(x, "x"), a_hat, y,
                 check_nodes(nodes, nrow(a_hat), "nodes"))
}

#' @rdname thangld_graph_kan_forward_r
#' @export
thangld_graph_kan_fd_gradient_r <- function(model, x, a_hat, y, nodes, step = 1e-5, order = 2L) {
  validate_gkan(model)
  graph_fd_gradient(model$params, model$specs, as_numeric_matrix(x, "x"), a_hat, y,
                    check_nodes(nodes, nrow(a_hat), "nodes"), step, order)
}
