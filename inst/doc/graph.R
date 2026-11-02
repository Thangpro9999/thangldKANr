## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")

## -----------------------------------------------------------------------------
library(thangldKANr)

## -----------------------------------------------------------------------------
g <- thangld_graph_kan_simulate(300, p_in = 0.08, p_out = 0.005, seed = 1)
a <- thangld_graph_kan_adjacency(g$adjacency)
u <- a %*% g$x
set.seed(2)
y <- sin(1.5 * u[, 1]) + 0.5 * u[, 2]^2 + rnorm(300, sd = 0.05)
train <- 1:180; val <- 181:240; test <- 241:300

## -----------------------------------------------------------------------------
fit_g <- thangld_graph_kan_fit(thangld_graph_kan(c(2, 1), degree = 5, seed = 1), g$x, g$adjacency, y,
                       train, val, epochs = 600, lr = 0.02, patience = 60, backend = "r")
fit_0 <- thangld_graph_kan_fit(thangld_graph_kan(c(2, 1), degree = 5, seed = 1), g$x, NULL, y,
                       train, val, epochs = 600, lr = 0.02, patience = 60, backend = "r")
fit_g
rbind(graph = thangld_graph_kan_metrics(y[test], predict(fit_g)[test]),
      no_graph = thangld_graph_kan_metrics(y[test], predict(fit_0)[test]))

## ----fig.width = 6, fig.height = 3.5------------------------------------------
h <- fit_g$history
plot(h$epoch, h$train_loss, type = "l", log = "y", xlab = "epoch", ylab = "MSE (model scale)")
lines(h$epoch, h$val_loss, col = 2)
legend("topright", c("train nodes", "validation nodes"), col = 1:2, lty = 1, bty = "n")

## -----------------------------------------------------------------------------
fit2 <- function(sb) thangld_graph_kan_fit(thangld_graph_kan(c(2, 4, 1), degree = 3, self_branch = sb,
                                             seed = 1), g$x, g$adjacency, y, train, val,
                                   epochs = 600, lr = 0.02, patience = 60, backend = "r")
rbind(plain_2layer = thangld_graph_kan_metrics(y[test], predict(fit2(FALSE))[test]),
      self_branch_2layer = thangld_graph_kan_metrics(y[test], predict(fit2(TRUE))[test]))

## -----------------------------------------------------------------------------
v <- thangld_graph_kan_validate("r")
v[, c("check", "pass")]

