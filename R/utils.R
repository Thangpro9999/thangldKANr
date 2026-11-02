# Internal helpers --------------------------------------------------------------

#' Check a single integer >= min
#' @noRd
check_count <- function(x, name, min = 1L) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || x != round(x) ||
      x < min) {
    stop(sprintf("`%s` must be a single integer >= %d.", name, min),
         call. = FALSE)
  }
  as.integer(x)
}

#' Check a single finite number
#' @noRd
check_number <- function(x, name, min = -Inf, strict = FALSE,
                         allow_null = FALSE) {
  if (allow_null && is.null(x)) return(NULL)
  ok <- is.numeric(x) && length(x) == 1L && is.finite(x) &&
    (if (strict) x > min else x >= min)
  if (!ok) {
    stop(sprintf("`%s` must be a single finite number %s %s.", name,
                 if (strict) ">" else ">=", format(min)), call. = FALSE)
  }
  as.numeric(x)
}

#' Check a single logical flag
#' @noRd
check_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("`%s` must be TRUE or FALSE.", name), call. = FALSE)
  }
  x
}

#' Coerce predictors to a finite numeric matrix
#'
#' Accepts a numeric matrix, a data frame with numeric columns, or a numeric
#' vector (one column). Rejects NA, NaN and Inf.
#' @noRd
as_numeric_matrix <- function(x, name = "x") {
  if (is.data.frame(x)) {
    is_num <- vapply(x, is.numeric, logical(1))
    if (!all(is_num)) {
      stop(sprintf("All columns of `%s` must be numeric (offending: %s).",
                   name, paste(names(x)[!is_num], collapse = ", ")),
           call. = FALSE)
    }
    x <- as.matrix(x)
  }
  if (is.null(dim(x)) && is.numeric(x)) x <- matrix(x, ncol = 1L)
  if (!is.matrix(x) || !is.numeric(x)) {
    stop(sprintf("`%s` must be a numeric matrix, data frame or vector.",
                 name), call. = FALSE)
  }
  if (nrow(x) < 1L || ncol(x) < 1L) {
    stop(sprintf("`%s` must have at least one row and one column.", name),
         call. = FALSE)
  }
  if (anyNA(x)) {
    stop(sprintf("`%s` contains NA/NaN values; remove or impute them first.",
                 name), call. = FALSE)
  }
  if (any(!is.finite(x))) {
    stop(sprintf("`%s` contains infinite values.", name), call. = FALSE)
  }
  storage.mode(x) <- "double"
  x
}

#' Coerce a response to a finite numeric vector of a given length
#' @noRd
as_response <- function(y, n, name = "y") {
  if (is.data.frame(y) || is.matrix(y)) {
    if (NCOL(y) != 1L) {
      stop(sprintf("`%s` must have exactly one column (single output).",
                   name), call. = FALSE)
    }
    y <- if (is.data.frame(y)) y[[1L]] else y[, 1L]
  }
  if (!is.numeric(y)) {
    stop(sprintf("`%s` must be numeric.", name), call. = FALSE)
  }
  if (length(y) != n) {
    stop(sprintf("`%s` has length %d but `x` has %d rows.", name,
                 length(y), n), call. = FALSE)
  }
  if (anyNA(y) || any(!is.finite(y))) {
    stop(sprintf("`%s` contains NA, NaN or infinite values.", name),
         call. = FALSE)
  }
  as.numeric(y)
}

#' Evaluate an expression with a fixed RNG seed, restoring the caller's
#' RNG state afterwards (so the package never changes the user's stream).
#' @noRd
with_seed <- function(seed, expr) {
  if (is.null(seed)) return(expr)
  has_seed <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  if (has_seed) old <- get(".Random.seed", envir = globalenv())
  on.exit({
    if (has_seed) {
      assign(".Random.seed", old, envir = globalenv())
    } else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      rm(".Random.seed", envir = globalenv())
    }
  })
  set.seed(seed)
  expr
}

#' Scaled error used by all numerical checks
#'
#' `max(abs(a - b) / pmax(1, abs(a), abs(b)))`.
#' @noRd
scaled_error <- function(a, b) {
  a <- as.numeric(a)
  b <- as.numeric(b)
  if (length(a) != length(b)) stop("Length mismatch.", call. = FALSE)
  if (!length(a)) return(0)
  max(abs(a - b) / pmax(1, abs(a), abs(b)))
}

#' TRUE when the torch runtime (libtorch + lantern) is usable
#' @noRd
torch_ready <- function() {
  requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()
}

#' Fail clearly when the torch backend is unusable
#' @noRd
require_torch <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("Package 'torch' is not installed. Install it with ",
         "install.packages(\"torch\"), or use backend = \"r\".",
         call. = FALSE)
  }
  if (!torch::torch_is_installed()) {
    stop("The torch runtime (libtorch/lantern) is not installed. Run ",
         "torch::install_torch() and restart R, or use backend = \"r\".",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' torch dtype from a string
#' @noRd
torch_dtype_from <- function(dtype) {
  switch(dtype,
    float32 = torch::torch_float32(),
    float64 = torch::torch_float64(),
    stop("`dtype` must be \"float32\" or \"float64\".", call. = FALSE)
  )
}

#' Convert a tensor to a plain R double array (detached, on CPU)
#' @noRd
as_r_array <- function(t) {
  a <- torch::as_array(t$detach()$cpu())
  storage.mode(a) <- "double"
  a
}

#' Tensor that owns a copy of its R data
#'
#' `torch::torch_tensor()` can share memory with the R object it is built
#' from; when that object is a temporary (for example `matrix(y, ncol = 1L)`)
#' R may reuse the memory after garbage collection and the tensor then holds
#' garbage. Found 2026-10-31 when the torch training checks were first run:
#' targets of later mini-batches were overwritten. Cloning gives the tensor
#' its own storage.
#' @noRd
r_tensor <- function(x, dtype = NULL, requires_grad = FALSE) {
  t <- torch::torch_tensor(x, dtype = dtype)$clone()
  if (requires_grad) t$requires_grad_(TRUE)
  t
}
