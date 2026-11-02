## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")

## -----------------------------------------------------------------------------
library(thangldKANr)

## -----------------------------------------------------------------------------
y <- log(as.numeric(datasets::lynx))
m <- thangld_temporal_kan(lags = 1:3, degree = 4, seed = 1)
m
fit <- thangld_temporal_kan_fit(m, y[1:100], n_val = 15, epochs = 150, lr = 0.02,
                        batch_size = 32, patience = 40, seed = 1, backend = "r")
fit$train$best_epoch
predict(fit)   # forecast of y[101]
y[101]

## -----------------------------------------------------------------------------
r <- thangld_temporal_kan_rolling(thangld_temporal_kan(1:3, seed = 1), y, initial = 100,
                          refit_every = 5, n_val = 15, epochs = 100,
                          lr = 0.02, patience = 40, seed = 1, backend = "r")
r$metrics

## ----fig.width = 6, fig.height = 3.5------------------------------------------
fc <- r$forecasts
plot(fc$index, fc$actual, type = "b", pch = 16, xlab = "t", ylab = "log lynx")
lines(fc$index, fc$kan, type = "b", col = 2)
lines(fc$index, fc$naive, type = "b", col = 4, lty = 2)
legend("topleft", c("actual", "temporal KAN", "naive"), col = c(1, 2, 4),
       lty = c(1, 1, 2), bty = "n")

## -----------------------------------------------------------------------------
v <- thangld_temporal_kan_validate("r")
table(v$status)

