# Graph utilities -------------------------------------------------------------------

#' Normalized adjacency matrix of an undirected graph
#'
#' Returns \eqn{\hat A = D^{-1/2} (A + I) D^{-1/2}} (Kipf and Welling, 2017),
#' where \eqn{D} is the degree matrix of \eqn{A + I}; with
#' `self_loops = FALSE` the identity is not added. This is the propagation
#' operator of every graph KAN layer.
#'
#' @param graph Either an \eqn{n \times n} non-negative symmetric adjacency
#'   matrix (weights allowed, diagonal ignored) or a two-column matrix or data
#'   frame of node indices (one undirected edge per row; duplicates and
#'   self-edges are dropped). A \eqn{2 \times 2} matrix is read as an
#'   adjacency matrix unless `n` is given and differs from 2.
#' @param n Number of nodes (needed for an edge list when the highest-index
#'   nodes are isolated; default: the largest index).
#' @param self_loops Add \eqn{I} before normalizing.
#' @return A dense \eqn{n \times n} matrix.
#' @examples
#' edges <- cbind(c(1, 2, 3), c(2, 3, 4))
#' thangld_graph_kan_adjacency(edges)
#' @export
thangld_graph_kan_adjacency <- function(graph, n = NULL, self_loops = TRUE) {
  self_loops <- check_flag(self_loops, "self_loops")
  a <- as_adjacency(graph, n)
  diag(a) <- 0
  if (self_loops) diag(a) <- 1
  deg <- rowSums(a)
  dinv <- ifelse(deg > 0, 1 / sqrt(deg), 0)
  a * outer(dinv, dinv)
}

#' @noRd
as_adjacency <- function(graph, n = NULL) {
  if (is.data.frame(graph)) graph <- as.matrix(graph)
  if (!is.numeric(graph) || !is.matrix(graph) || anyNA(graph) ||
      any(!is.finite(graph))) {
    stop("`graph` must be a numeric adjacency matrix or a two-column edge list.",
         call. = FALSE)
  }
  is_edges <- ncol(graph) == 2L &&
    (nrow(graph) != 2L || (!is.null(n) && !identical(as.numeric(n), 2)))
  if (is_edges) {
    if (any(graph < 1) || any(graph != round(graph))) {
      stop("Edge list entries must be node indices >= 1.", call. = FALSE)
    }
    nn <- if (is.null(n)) max(graph) else check_count(n, "n")
    if (max(graph) > nn) stop("Edge list refers to nodes beyond `n`.", call. = FALSE)
    a <- matrix(0, nn, nn)
    a[graph] <- 1
    a[graph[, 2:1, drop = FALSE]] <- 1
    return(a)
  }
  if (nrow(graph) != ncol(graph)) {
    stop("An adjacency matrix must be square.", call. = FALSE)
  }
  if (any(graph < 0)) stop("Adjacency weights must be non-negative.", call. = FALSE)
  if (max(abs(graph - t(graph))) > 1e-12 * max(1, max(graph))) {
    stop("The adjacency matrix must be symmetric (undirected graph).", call. = FALSE)
  }
  if (!is.null(n) && check_count(n, "n") != nrow(graph)) {
    stop("`n` does not match the adjacency matrix.", call. = FALSE)
  }
  unname(graph)
}

#' Simulate a stochastic block model graph with node features
#'
#' Nodes are split into `blocks` equal communities; an edge appears with
#' probability `p_in` inside a community and `p_out` between communities.
#' Features are independent \eqn{N(0, 1)} plus a community mean shift of
#' `shift`. Used in examples, tests and benchmarks (no external data).
#'
#' @param n Number of nodes.
#' @param p Number of features.
#' @param blocks Number of communities.
#' @param p_in,p_out Edge probabilities.
#' @param shift Size of the community mean shift of the features.
#' @param seed Optional seed (the caller's random number stream is restored).
#' @return A list with `x` (\eqn{n \times p}), `adjacency` (0/1 matrix),
#'   `edges` (two-column matrix) and `block` (community labels).
#' @examples
#' g <- thangld_graph_kan_simulate(30, seed = 1)
#' dim(g$adjacency)
#' @export
thangld_graph_kan_simulate <- function(n = 150L, p = 2L, blocks = 3L, p_in = 0.1,
                               p_out = 0.01, shift = 1, seed = NULL) {
  n <- check_count(n, "n", min = 2L)
  p <- check_count(p, "p")
  blocks <- check_count(blocks, "blocks")
  p_in <- check_number(p_in, "p_in", min = 0)
  p_out <- check_number(p_out, "p_out", min = 0)
  if (!is.null(seed)) seed <- check_count(seed, "seed", min = 0L)
  with_seed(seed, {
    block <- rep_len(seq_len(blocks), n)
    same <- outer(block, block, `==`)
    u <- matrix(stats::runif(n * n), n, n)
    a <- (u < ifelse(same, p_in, p_out)) * 1
    a[lower.tri(a, diag = TRUE)] <- 0
    a <- a + t(a)
    mu <- matrix(stats::rnorm(blocks * p, sd = shift), blocks, p)
    x <- matrix(stats::rnorm(n * p), n, p) + mu[block, , drop = FALSE]
    colnames(x) <- paste0("x", seq_len(p))
    e <- which(upper.tri(a) & a > 0, arr.ind = TRUE)
    list(x = x, adjacency = a, edges = unname(e[order(e[, 1L], e[, 2L]), , drop = FALSE]),
         block = block)
  })
}
