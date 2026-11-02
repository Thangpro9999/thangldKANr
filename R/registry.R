# Basis registry -------------------------------------------------------------------
#
# Every edge type ("basis") is a list of functions with the layer interface
# used by the generic engine (see engine.R). The built-in bases live in
# basis_<name>.R; other packages (for example splineKANr, as basis
# "bspline") add theirs with register_kan_basis(). Each layer specification
# carries its basis name in `spec$basis`, so one network can mix bases.

.kan_registry <- new.env(parent = emptyenv())

#' @noRd
kan_required_fields <- c("layer_specs", "describe", "param_dims", "init",
                         "check_params", "forward_r", "backward_r",
                         "forward_torch", "edge_r")

#' @noRd
kan_optional_fields <- c("label", "naive_forward", "basis_checks",
                         "validation_fixture")

#' Built-in bases (functions looked up in this namespace by prefix)
#' @noRd
builtin_basis_table <- function() {
  data.frame(
    name = c("chebyshev", "rbf", "legendre", "wavelet", "rational", "hybrid",
             "jacobi", "relu"),
    label = c("Chebyshev polynomials (ChebyKAN)",
              "Gaussian radial basis functions (FastKAN)",
              "Legendre polynomials",
              "Mexican-hat wavelets (Wav-KAN)",
              "Safe Pade rational functions",
              "SiLU + Chebyshev + RBF (hybrid)",
              "Jacobi polynomials (JacobiKAN)",
              "Squared ReLU bumps (ReLU-KAN)"),
    stringsAsFactors = FALSE)
}

#' @noRd
builtin_impl <- function(name, label) {
  ns <- environment(builtin_impl)
  f <- function(x) get(paste0(".", name, "_", x), envir = ns, inherits = FALSE)
  list(label = label, layer_specs = f("layer_specs"),
       describe = f("layer_describe"), param_dims = f("layer_param_dims"),
       init = f("layer_init"), check_params = f("layer_check_params"),
       forward_r = f("layer_forward_r"), backward_r = f("layer_backward_r"),
       forward_torch = f("layer_forward_torch"), edge_r = f("layer_edge_r"),
       naive_forward = f("layer_naive_forward"),
       basis_checks = f("layer_basis_checks"),
       validation_fixture = f("validation_fixture"), builtin = TRUE)
}

#' @noRd
kan_registry_init <- function() {
  if (isTRUE(.kan_registry$.ready)) return(invisible())
  tb <- builtin_basis_table()
  for (i in seq_len(nrow(tb))) {
    assign(tb$name[i], builtin_impl(tb$name[i], tb$label[i]),
           envir = .kan_registry)
  }
  assign(".ready", TRUE, envir = .kan_registry)
  invisible()
}

#' Implementation of one basis (error if unknown)
#' @noRd
basis_impl <- function(name) {
  kan_registry_init()
  if (!is.character(name) || length(name) != 1L || is.na(name)) {
    stop("Corrupt model: a layer has no basis name.", call. = FALSE)
  }
  impl <- get0(name, envir = .kan_registry, inherits = FALSE)
  if (is.null(impl)) {
    stop(sprintf(paste0("Unknown basis \"%s\". Available: %s. Other bases ",
                        "are added with register_kan_basis()."),
                 name, paste(kan_bases()$name, collapse = ", ")),
         call. = FALSE)
  }
  impl
}

#' List the available edge bases
#'
#' @return A data frame with one row per basis: `name` (the value for the
#'   `basis` argument of [thangld_kan()]), `label`, `builtin` and `arguments`
#'   (the basis-specific arguments accepted by [thangld_kan()]).
#' @examples
#' kan_bases()
#' @export
kan_bases <- function() {
  kan_registry_init()
  nms <- setdiff(ls(.kan_registry, all.names = FALSE), character())
  tb <- builtin_basis_table()
  nms <- c(tb$name, sort(setdiff(nms, tb$name)))
  rows <- lapply(nms, function(nm) {
    impl <- get(nm, envir = .kan_registry, inherits = FALSE)
    data.frame(name = nm,
               label = if (is.null(impl$label)) nm else as.character(impl$label),
               builtin = isTRUE(impl$builtin),
               arguments = paste(setdiff(names(formals(impl$layer_specs)),
                                         "widths"), collapse = ", "),
               stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
}

#' Register an additional edge basis
#'
#' Adds an edge type to the registry so that `thangld_kan(basis = name)`
#' builds networks with it. Every other function of the package (fitting
#' with both backends, prediction, edge plots, save/load and
#' [thangld_kan_validate()]) then works with the new basis unchanged. This is
#' how other packages plug in their basis (for example a B-spline basis as
#' `"bspline"`), typically from their `.onLoad()`.
#'
#' `impl` is a named list of functions. Required:
#' \describe{
#'   \item{`layer_specs(widths, ...)`}{Checks the basis arguments and returns
#'     a list with one specification (a list) per layer. Its formal
#'     arguments other than `widths` are the arguments [thangld_kan()]
#'     accepts for this basis.}
#'   \item{`describe(spec)`}{One-line description of a layer.}
#'   \item{`param_dims(p, q, spec)`}{Named list of parameter array dimensions
#'     for a layer with `p` inputs and `q` outputs.}
#'   \item{`init(p, q, spec)`}{Named list of initial parameter arrays (uses
#'     the R random number generator; the seed is set by [thangld_kan()]).}
#'   \item{`check_params(par, spec)`}{Error if the parameters left their
#'     domain; otherwise invisible `TRUE`.}
#'   \item{`forward_r(h, par, spec)`}{Base R forward pass of an `n` x `p`
#'     matrix: `list(out = n x q matrix, cache = anything)`.}
#'   \item{`backward_r(cache, g, par, spec)`}{Analytic backward pass given
#'     the `n` x `q` gradient of the loss with respect to the layer output:
#'     `list(grads = named list like par, dh = n x p matrix)`.}
#'   \item{`forward_torch(h, par, spec)`}{The same forward pass with 'torch'
#'     tensors (`par` is a named list of tensors); must be differentiable by
#'     autograd. Write `tensor * scalar`, not `scalar * tensor`, so that R
#'     doubles are not rounded to float32.}
#'   \item{`edge_r(x, par, spec, i, j)`}{Data frame describing edge
#'     \eqn{i \to j} at inputs `x`, with the edge output in column `phi`.}
#' }
#' Optional, used by [thangld_kan_validate()] (a missing one is reported as a
#' skipped check, never as passed): `naive_forward(h, par, spec)` (an
#' independent loop implementation), `basis_checks()` (list of
#' `list(check, error, tol)`), `validation_fixture()` (returns
#' `list(model, x, y)`), and `label`.
#'
#' Models store only the basis name, so a model with a registered basis can
#' be reloaded only after the basis is registered again.
#'
#' @param name Basis name: lower-case letters, digits and `_`, starting with
#'   a letter. Built-in names cannot be replaced.
#' @param impl Named list of functions described in Details.
#' @param overwrite Replace an earlier registration of `name`.
#' @return `register_kan_basis()` returns `name` invisibly;
#'   `unregister_kan_basis()` returns `TRUE` invisibly if a basis was
#'   removed.
#' @examples
#' # a toy linear basis: phi(x) = w * x + b on every edge
#' lin <- list(
#'   label = "linear (toy)",
#'   layer_specs = function(widths) rep(list(list()), length(widths) - 1L),
#'   describe = function(spec) "linear",
#'   param_dims = function(p, q, spec) list(w = c(p, q)),
#'   init = function(p, q, spec) list(w = matrix(rnorm(p * q, sd = 0.3), p, q)),
#'   check_params = function(par, spec) invisible(TRUE),
#'   forward_r = function(h, par, spec) list(out = h %*% par$w, cache = h),
#'   backward_r = function(cache, g, par, spec)
#'     list(grads = list(w = crossprod(cache, g)), dh = g %*% t(par$w)),
#'   forward_torch = function(h, par, spec) torch::torch_matmul(h, par$w),
#'   edge_r = function(x, par, spec, i, j) data.frame(phi = par$w[i, j] * x))
#' register_kan_basis("toy_linear", lin)
#' m <- thangld_kan(c(2, 3, 1), basis = c("toy_linear", "chebyshev"), seed = 1)
#' m
#' unregister_kan_basis("toy_linear")
#' @export
register_kan_basis <- function(name, impl, overwrite = FALSE) {
  kan_registry_init()
  if (!is.character(name) || length(name) != 1L || is.na(name) ||
      !grepl("^[a-z][a-z0-9_]*$", name)) {
    stop("`name` must be one lower-case name (letters, digits, `_`).",
         call. = FALSE)
  }
  if (name %in% builtin_basis_table()$name) {
    stop(sprintf("\"%s\" is a built-in basis and cannot be replaced.", name),
         call. = FALSE)
  }
  if (!isTRUE(overwrite) && exists(name, envir = .kan_registry,
                                   inherits = FALSE)) {
    stop(sprintf("Basis \"%s\" is already registered (use overwrite = TRUE).",
                 name), call. = FALSE)
  }
  if (!is.list(impl)) stop("`impl` must be a named list.", call. = FALSE)
  miss <- setdiff(kan_required_fields, names(impl))
  if (length(miss)) {
    stop("`impl` lacks: ", paste(miss, collapse = ", "), ".", call. = FALSE)
  }
  for (nm in intersect(c(kan_required_fields, kan_optional_fields[-1L]),
                       names(impl))) {
    if (!is.function(impl[[nm]])) {
      stop(sprintf("`impl$%s` must be a function.", nm), call. = FALSE)
    }
  }
  if (!"widths" %in% names(formals(impl$layer_specs))) {
    stop("`impl$layer_specs` must have a `widths` argument.", call. = FALSE)
  }
  impl$builtin <- FALSE
  assign(name, impl, envir = .kan_registry)
  invisible(name)
}

#' @rdname register_kan_basis
#' @export
unregister_kan_basis <- function(name) {
  kan_registry_init()
  if (!is.character(name) || length(name) != 1L || is.na(name)) {
    stop("`name` must be one basis name.", call. = FALSE)
  }
  if (name %in% builtin_basis_table()$name) {
    stop(sprintf("\"%s\" is a built-in basis.", name), call. = FALSE)
  }
  if (!exists(name, envir = .kan_registry, inherits = FALSE)) {
    return(invisible(FALSE))
  }
  rm(list = name, envir = .kan_registry)
  invisible(TRUE)
}

# Generic layer interface used by the engine: dispatch on spec$basis ----------------

#' @noRd
layer_describe <- function(spec) basis_impl(spec$basis)$describe(spec)
#' @noRd
layer_param_dims <- function(p, q, spec) {
  basis_impl(spec$basis)$param_dims(p, q, spec)
}
#' @noRd
layer_init <- function(p, q, spec) basis_impl(spec$basis)$init(p, q, spec)
#' @noRd
layer_check_params <- function(par, spec) {
  basis_impl(spec$basis)$check_params(par, spec)
}
#' @noRd
layer_forward_r <- function(h, par, spec) {
  basis_impl(spec$basis)$forward_r(h, par, spec)
}
#' @noRd
layer_backward_r <- function(cache, g, par, spec) {
  basis_impl(spec$basis)$backward_r(cache, g, par, spec)
}
#' @noRd
layer_forward_torch <- function(h, par, spec) {
  basis_impl(spec$basis)$forward_torch(h, par, spec)
}
#' @noRd
layer_edge_r <- function(x, par, spec, i, j) {
  basis_impl(spec$basis)$edge_r(x, par, spec, i, j)
}
#' @noRd
layer_naive_forward <- function(h, par, spec) {
  f <- basis_impl(spec$basis)$naive_forward
  if (is.null(f)) return(NULL)
  f(h, par, spec)
}

#' Validation fixture of one basis: list(model, x, y)
#' @noRd
validation_fixture <- function(basis = "chebyshev", ...) {
  f <- basis_impl(basis)$validation_fixture
  if (!is.null(f)) return(f(...))
  model <- thangld_kan(c(3, 4, 2, 1), basis = basis, seed = 11)
  x <- with_seed(12, matrix(stats::rnorm(18, sd = 1.5), 6, 3))
  y <- with_seed(13, stats::rnorm(6))
  list(model = model, x = x, y = y)
}

#' Check parameter arrays against the layer specifications
#'
#' `inputs[l]` is the number of inputs of layer `l` (differs from
#' `widths[l]` when a model feeds extra inputs, as the graph self branch).
#' @noRd
check_layer_params <- function(params, specs, widths,
                               inputs = widths[-length(widths)]) {
  nl <- length(widths) - 1L
  if (length(params) != nl || length(specs) != nl) {
    stop("Corrupt model: number of layers does not match widths.",
         call. = FALSE)
  }
  for (l in seq_len(nl)) {
    dims <- layer_param_dims(inputs[l], widths[l + 1L], specs[[l]])
    if (!identical(sort(names(dims)), sort(names(params[[l]])))) {
      stop(sprintf("Corrupt model: parameters of layer %d do not match.", l),
           call. = FALSE)
    }
    for (nm in names(dims)) {
      a <- params[[l]][[nm]]
      if (!identical(as.integer(dim(a)), as.integer(dims[[nm]]))) {
        stop(sprintf("Corrupt model: `%s` of layer %d has wrong dimensions.",
                     nm, l), call. = FALSE)
      }
      if (any(!is.finite(a))) {
        stop(sprintf("Parameters `%s` of layer %d are not finite.", nm, l),
             call. = FALSE)
      }
    }
    layer_check_params(params[[l]], specs[[l]])
  }
  invisible(TRUE)
}
