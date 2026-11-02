# Validation suite and persistence ---------------------------------------------------

#' @noRd
graph_validation_fixture <- function(self_branch = FALSE) {
  m <- thangld_graph_kan(c(3, 4, 2, 1), degree = c(4, 3, 5), self_branch = self_branch, seed = 11)
  m$params <- with_seed(21, lapply(m$params, function(pl) {
    pl$coef[] <- stats::rnorm(length(pl$coef), sd = 0.4)
    pl
  }))
  edges <- rbind(c(1, 2), c(2, 3), c(3, 4), c(4, 1), c(4, 5), c(5, 6), c(2, 6), c(7, 7))
  x <- with_seed(12, matrix(stats::rnorm(21, sd = 1.5), 7, 3))
  y <- with_seed(13, stats::rnorm(7))
  list(model = m, edges = edges, a_hat = thangld_graph_kan_adjacency(edges, n = 7), x = x, y = y,
       nodes = c(1L, 2L, 4L, 5L, 7L))
}

#' Run the numerical validation suite
#'
#' Checks the normalized adjacency against an explicit loop, the Chebyshev
#' forward pass against an independent loop implementation (explicit
#' neighbor sums and \eqn{\cos(k \arccos z)}), analytic gradients of the
#' masked loss against fourth-order central differences (\eqn{h = 10^{-4},
#' 10^{-5}, 10^{-6}}), for both the plain and the self-branch architecture,
#' permutation equivariance, the identity-graph
#' reduction, save/load, and, with the torch runtime, torch forward, autograd
#' and full-batch Adam training against the base R oracle. These checks use
#' Chebyshev edges; for every other built-in basis the gradients (and, with
#' torch, the torch forward pass and autograd) of a self-branch graph model
#' are also checked. Torch checks that
#' cannot run are reported with `pass = NA` and counted in `n_skipped`; they
#' are never counted as passed.
#'
#' @param backend `"both"`, `"r"` or `"torch"`.
#' @return A data frame (`group`, `check`, `error`, `tol`, `pass`) with
#'   attributes `all_pass` and `n_skipped`.
#' @examples
#' v <- thangld_graph_kan_validate("r")
#' v[, c("check", "pass")]
#' @export
thangld_graph_kan_validate <- function(backend = c("both", "r", "torch")) {
  backend <- match.arg(backend)
  rows <- list()
  add <- function(group, check, error, tol) {
    rows[[length(rows) + 1L]] <<- data.frame(
      group = group, check = check, error = error, tol = tol,
      pass = if (is.na(error)) NA else error <= tol)
  }
  f <- graph_validation_fixture()
  m <- f$model
  if (backend %in% c("r", "both")) {
    n <- nrow(f$x)
    a <- matrix(0, n, n)
    for (e in seq_len(nrow(f$edges))) {
      i <- f$edges[e, 1L]
      j <- f$edges[e, 2L]
      if (i != j) a[i, j] <- a[j, i] <- 1
    }
    a1 <- a + diag(n)
    ex <- matrix(0, n, n)
    for (i in seq_len(n)) for (j in seq_len(n)) {
      ex[i, j] <- a1[i, j] / sqrt(sum(a1[i, ]) * sum(a1[j, ]))
    }
    add("graph", "normalized adjacency vs explicit loop (edge list input)",
        max(abs(f$a_hat - ex)), 1e-14)
    add("graph", "edge list and adjacency matrix inputs agree",
        max(abs(f$a_hat - thangld_graph_kan_adjacency(a))), 0)
    z <- c(-0.99, -0.5, 0, 0.3, 0.97)
    add("basis", "Chebyshev recurrence vs cos(k acos z)",
        max(abs(matrix(.chebyshev_cheby_basis_array(matrix(z), 6L), 5L) -
                  outer(acos(z), 0:6, function(t, k) cos(k * t)))), 1e-12)
    naive_forward <- function(m) {
      naive <- f$x
      for (l in seq_along(m$params)) {
        u <- matrix(0, n, ncol(naive))
        for (v in seq_len(n)) for (w in seq_len(n)) {
          if (ex[v, w] != 0) u[v, ] <- u[v, ] + ex[v, w] * naive[w, ]
        }
        if (isTRUE(m$specs[[l]]$self_branch)) u <- cbind(naive, u)
        naive <- layer_naive_forward(u, m$params[[l]], m$specs[[l]])
      }
      naive
    }
    grad_checks <- function(m, what) {
      g <- graph_gradient(m$params, m$specs, f$x, f$a_hat, f$y, f$nodes)
      for (h in c(1e-4, 1e-5, 1e-6)) {
        fd <- graph_fd_gradient(m$params, m$specs, f$x, f$a_hat, f$y, f$nodes, h, 4L)
        add("gradient", sprintf("%scoefficients vs central difference O4 (h=%g)", what, h),
            scaled_error(unlist(g$params), unlist(fd$params)), 1e-5)
        add("gradient", sprintf("%snode features vs central difference O4 (h=%g)", what, h),
            scaled_error(g$x, fd$x), 1e-5)
      }
    }
    out <- graph_forward(m$params, m$specs, f$x, f$a_hat)$out
    add("forward", "oracle vs naive loops (neighbor sums, cos(k acos z))",
        scaled_error(out, naive_forward(m)), 1e-8)
    grad_checks(m, "")
    ms <- graph_validation_fixture(self_branch = TRUE)$model
    add("forward", "self branch [H, A_hat H]: oracle vs naive loops",
        scaled_error(graph_forward(ms$params, ms$specs, f$x, f$a_hat)$out,
                     naive_forward(ms)), 1e-8)
    grad_checks(ms, "self branch: ")
    perm <- c(5L, 2L, 7L, 1L, 3L, 6L, 4L)
    outp <- graph_forward(m$params, m$specs, f$x[perm, ], f$a_hat[perm, perm])$out
    add("graph", "permutation equivariance f(PX, PAP') = P f(X, A)",
        scaled_error(outp, out[perm]), 1e-12)
    m1 <- thangld_graph_kan(c(3, 1), degree = 4, seed = 3)
    add("graph", "identity graph = node-wise Chebyshev KAN",
        scaled_error(graph_forward(m1$params, m1$specs, f$x, diag(n))$out,
                     drop(layer_naive_forward(f$x, m1$params[[1]], m1$specs[[1]]))), 1e-12)
    m2 <- thangld_graph_kan(c(3, 1), degree = 4, self_branch = TRUE, seed = 3)
    add("graph", "self branch, identity graph = node-wise KAN on [X, X]",
        scaled_error(graph_forward(m2$params, m2$specs, f$x, diag(n))$out,
                     drop(layer_naive_forward(cbind(f$x, f$x), m2$params[[1]],
                                              m2$specs[[1]]))), 1e-12)
    m3 <- ms
    m3$params[[1]]$coef[4:6, , ] <- 0
    m3$params[[2]]$coef[5:8, , ] <- 0
    m3$params[[3]]$coef[3:4, , ] <- 0
    add("graph", "self branch with zero neighbor coefficients = graph-free KAN",
        scaled_error(graph_forward(m3$params, m3$specs, f$x, f$a_hat)$out,
                     graph_forward(m3$params, m3$specs, f$x, diag(n))$out), 1e-12)
    fit <- thangld_graph_kan_fit(thangld_graph_kan(c(3, 2, 1), seed = 1), f$x, f$edges, f$y,
                         train = 1:5, val = 6:7, epochs = 5, backend = "r")
    tf <- tempfile(fileext = ".rds")
    on.exit(unlink(tf), add = TRUE)
    thangld_graph_kan_save(fit, tf)
    add("save/load", "predictions before vs after reload",
        max(abs(predict(fit) - predict(thangld_graph_kan_load(tf)))), 1e-8)
  }
  if (backend %in% c("torch", "both")) {
    ck <- c("forward torch vs oracle", "autograd parameters vs oracle",
            "autograd node features vs oracle",
            "full-batch Adam training torch vs r backend (parameters and losses)",
            "self branch: torch forward and autograd vs oracle")
    if (!torch_ready()) {
      for (k in ck) add("torch", k, NA_real_, 1e-8)
    } else {
      out <- graph_forward(m$params, m$specs, f$x, f$a_hat)$out
      add("torch", ck[1], scaled_error(graph_forward_torch(m$params, m$specs, f$x, f$a_hat),
                                       out), 1e-8)
      g <- graph_gradient(m$params, m$specs, f$x, f$a_hat, f$y, f$nodes)
      gt <- graph_gradient_torch(m$params, m$specs, f$x, f$a_hat, f$y, f$nodes)
      add("torch", ck[2], scaled_error(unlist(gt$params), unlist(g$params)), 1e-8)
      add("torch", ck[3], scaled_error(gt$x, g$x), 1e-8)
      ms <- graph_validation_fixture(self_branch = TRUE)$model
      gs <- graph_gradient(ms$params, ms$specs, f$x, f$a_hat, f$y, f$nodes)
      gst <- graph_gradient_torch(ms$params, ms$specs, f$x, f$a_hat, f$y, f$nodes)
      add("torch", ck[5], max(
        scaled_error(graph_forward_torch(ms$params, ms$specs, f$x, f$a_hat),
                     graph_forward(ms$params, ms$specs, f$x, f$a_hat)$out),
        scaled_error(unlist(gst$params), unlist(gs$params)), scaled_error(gst$x, gs$x)), 1e-8)
      a <- thangld_graph_kan_fit(m, f$x, f$edges, f$y, 1:5, 6:7, epochs = 25, lr = 0.02,
                         backend = "r")
      b <- thangld_graph_kan_fit(m, f$x, f$edges, f$y, 1:5, 6:7, epochs = 25, lr = 0.02,
                         backend = "torch")
      add("torch", ck[4], max(scaled_error(unlist(a$params), unlist(b$params)),
                              scaled_error(unlist(a$history[, -1]), unlist(b$history[, -1]))),
          1e-8)
    }
  }
  # every other built-in basis inside the graph engine (self branch on)
  others <- setdiff(builtin_basis_table()$name, "chebyshev")
  for (b in others) {
    mb <- thangld_graph_kan(c(3, 4, 1), basis = b, self_branch = TRUE, seed = 5)
    if (backend %in% c("r", "both")) {
      g <- graph_gradient(mb$params, mb$specs, f$x, f$a_hat, f$y, f$nodes)
      fd <- graph_fd_gradient(mb$params, mb$specs, f$x, f$a_hat, f$y, f$nodes,
                              1e-5, 4L)
      add("bases", sprintf("basis \"%s\": gradients vs central difference O4 (h=1e-05)", b),
          max(scaled_error(unlist(g$params), unlist(fd$params)),
              scaled_error(g$x, fd$x)), 1e-5)
    }
    if (backend %in% c("torch", "both")) {
      ckb <- sprintf("basis \"%s\": torch forward and autograd vs oracle", b)
      if (!torch_ready()) {
        add("bases", ckb, NA_real_, 1e-8)
      } else {
        g <- graph_gradient(mb$params, mb$specs, f$x, f$a_hat, f$y, f$nodes)
        gt <- graph_gradient_torch(mb$params, mb$specs, f$x, f$a_hat, f$y, f$nodes)
        add("bases", ckb, max(
          scaled_error(graph_forward_torch(mb$params, mb$specs, f$x, f$a_hat),
                       graph_forward(mb$params, mb$specs, f$x, f$a_hat)$out),
          scaled_error(unlist(gt$params), unlist(g$params)),
          scaled_error(gt$x, g$x)), 1e-8)
      }
    }
  }
  res <- do.call(rbind, rows)
  attr(res, "all_pass") <- all(res$pass %in% TRUE)
  attr(res, "n_skipped") <- sum(is.na(res$pass))
  res
}

#' Save and load a graph KAN
#'
#' Models are plain R objects (no torch pointers), stored with `saveRDS()`
#' together with a format tag and the package and R versions.
#'
#' @param model A `thangld_graph_kan`.
#' @param file Path.
#' @return `thangld_graph_kan_save()`: `file`, invisibly. `thangld_graph_kan_load()`: the model.
#' @examples
#' f <- tempfile(fileext = ".rds")
#' thangld_graph_kan_save(thangld_graph_kan(c(2, 1), seed = 1), f)
#' thangld_graph_kan_load(f)
#' @export
thangld_graph_kan_save <- function(model, file) {
  validate_gkan(model)
  if (!is.character(file) || length(file) != 1L) {
    stop("`file` must be a single path.", call. = FALSE)
  }
  saveRDS(list(format = "thangldKANr_model", format_version = 1L,
               package_version = as.character(utils::packageVersion("thangldKANr")),
               r_version = R.version.string, model = unclass(model)), file)
  invisible(file)
}

#' @rdname thangld_graph_kan_save
#' @export
thangld_graph_kan_load <- function(file) {
  if (!is.character(file) || length(file) != 1L || !file.exists(file)) {
    stop("File not found: ", file, call. = FALSE)
  }
  obj <- readRDS(file)
  if (!is.list(obj) || !identical(obj$format, "thangldKANr_model")) {
    stop("Not a thangldKANr model file.", call. = FALSE)
  }
  validate_gkan(structure(obj$model, class = "thangld_graph_kan"))
}
