# Preprocessing ----------------------------------------------------------------
#
# Every constant is estimated on the training rows only and stored in the
# model, so predict() applies exactly the same transformation.

#' Estimate the preprocessing from training data
#' @noRd
fit_preprocess <- function(x, y, standardize_x = TRUE, standardize_y = TRUE) {
  p <- ncol(x)
  n <- nrow(x)
  col_sd <- if (n > 1L) apply(x, 2L, stats::sd) else rep(0, p)
  constant <- col_sd == 0 | !is.finite(col_sd)
  if (standardize_x) {
    x_center <- colMeans(x)
    x_scale <- ifelse(constant, 1, col_sd)
  } else {
    x_center <- rep(0, p)
    x_scale <- rep(1, p)
  }
  # value a constant column always takes on the model scale
  const_value <- ifelse(constant, (x[1L, ] - x_center) / x_scale, NA_real_)
  y_sd <- if (n > 1L) stats::sd(y) else 0
  y_constant <- !is.finite(y_sd) || y_sd == 0
  if (standardize_y) {
    y_center <- mean(y)
    y_scale <- if (y_constant) 1 else y_sd
  } else {
    y_center <- 0
    y_scale <- 1
  }
  list(col_names = colnames(x), x_center = unname(x_center),
       x_scale = unname(x_scale), constant = unname(constant),
       const_value = unname(const_value),
       x_min = unname(apply(x, 2L, min)), x_max = unname(apply(x, 2L, max)),
       y_center = y_center, y_scale = y_scale, y_constant = y_constant,
       standardize_x = standardize_x, standardize_y = standardize_y)
}

#' Align new data with the training columns (by name when available)
#' @noRd
align_columns <- function(pp, x, p) {
  if (!is.null(pp$col_names)) {
    if (is.null(colnames(x))) {
      if (ncol(x) != p) {
        stop(sprintf("`newdata` has %d columns but the model was trained on %d.",
                     ncol(x), p), call. = FALSE)
      }
      warning("`newdata` has no column names; columns are used in the ",
              "training order (", paste(pp$col_names, collapse = ", "), ").",
              call. = FALSE)
      return(x)
    }
    miss <- setdiff(pp$col_names, colnames(x))
    if (length(miss)) {
      stop("`newdata` is missing columns used in training: ",
           paste(miss, collapse = ", "), call. = FALSE)
    }
    return(x[, pp$col_names, drop = FALSE])
  }
  if (ncol(x) != p) {
    stop(sprintf("`newdata` has %d columns but the model was trained on %d.",
                 ncol(x), p), call. = FALSE)
  }
  x
}

#' Map raw inputs to the model scale
#' @noRd
apply_preprocess_x <- function(pp, x) {
  z <- sweep(sweep(x, 2L, pp$x_center, "-"), 2L, pp$x_scale, "/")
  if (any(pp$constant)) {
    idx <- which(pp$constant)
    z[, idx] <- rep(pp$const_value[idx], each = nrow(z))
  }
  z
}

#' @noRd
apply_preprocess_y <- function(pp, y) (y - pp$y_center) / pp$y_scale

#' @noRd
invert_preprocess_y <- function(pp, y) y * pp$y_scale + pp$y_center
