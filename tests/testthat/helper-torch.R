skip_if_no_torch <- function() {
  testthat::skip_if_not(thangldKANr:::torch_ready(),
                        "torch runtime (libtorch/lantern) not installed")
}
serr <- function(a, b) thangldKANr:::scaled_error(a, b)
