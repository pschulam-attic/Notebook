suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(ggplot2)
})
source("~/Git/nips15-model/functions.R")
options(width = 120, dplyr.width = 120)

pfvc  <- read_csv("data/benchmark_pfvc.csv")
data  <- pfvc %>% group_by(ptid) %>% do(fold = .$fold[1], datum = make_datum(.))

model_files <- lapply(1:10, function(k) sprintf("models/jmlr/folds/pfvc/%02d/model.rds", k))
models      <- lapply(model_files, readRDS)

truncate_datum <- function(datum, censor_time) {
  keep <- datum[["x"]] <= censor_time
  datum[["x"]] <- datum[["x"]][keep]
  datum[["y"]] <- datum[["y"]][keep]
  datum
}

predict_datum <- function(datum, censor_time, model) {
  observed <- truncate_datum(datum, censor_time)
  xnew <- datum[["x"]][datum[["x"]] > censor_time]
  if (length(xnew) == 0) return(numeric(0))
  ynew <- apply_model(observed, model, xnew)$ynew_hat
  list(x = xnew, y = ynew)
}

predict_grid <- function(datum, censor_time, model) {
  observed <- truncate_datum(datum, censor_time)
  xnew <- seq(0, 25, 0.25)
  ynew <- apply_model(observed, model, xnew)$ynew_hat
  list(x = xnew, y = ynew)
}

args            <- commandArgs(trailingOnly = TRUE)
censor_time     <- as.numeric(args[1])
posteriors_file <- args[2]

posteriors <- read_csv(posteriors_file)
data       <- data %>% inner_join(posteriors, "ptid")

predictions <- vector("list", nrow(data))
prediction_grids <- vector("list", nrow(data))
observations <- vector("list", nrow(data))

for (i in seq_along(predictions)) {
  m <- models[[data$fold[[i]]]]

  p <- predict_datum(data$datum[[i]], censor_time, m)
  if (length(p) == 0) next
  w <- unlist(data[i, -c(1, 2, 3)])
  x <- p$x
  y <- p$y %*% w
  predictions[[i]] <- data.frame(ptid = data$datum[[i]]$ptid, x = x, yhat = y)

  g <- predict_grid(data$datum[[i]], censor_time, m)
  w <- unlist(data[i, -c(1, 2, 3)])
  x <- g$x
  rank <- order(w, decreasing=TRUE)
  y1 <- g$y[, rank[1]]
  w1 <- w[rank[1]]
  y2 <- g$y[, rank[2]]
  w2 <- w[rank[2]]

  pg0 <- data.frame(ptid = data$datum[[i]]$ptid, x = x, y = g$y %*% w, rank = "PP", weight = 1)
  pg1 <- data.frame(ptid = data$datum[[i]]$ptid, x = x, y = y1, rank = "1", weight = w1)
  pg2 <- data.frame(ptid = data$datum[[i]]$ptid, x = x, y = y2, rank = "2", weight = w2)
  prediction_grids[[i]] <- rbind(pg0, pg1, pg2)

  observations[[i]] <- data.frame(ptid = data$datum[[i]]$ptid, x = data$datum[[i]]$x, y = data$datum[[i]]$y, observed = data$datum[[i]]$x <= censor_time)
}

predictions <- do.call("rbind", predictions) %>% tbl_df
prediction_grids <- do.call("rbind", prediction_grids) %>% tbl_df
observations <- do.call("rbind", observations) %>% tbl_df

bn <- basename(posteriors_file)
fn <- str_replace(bn, "csv", "pdf")
fn <- file.path("plots", fn)

pdf(fn, width = 12, height = 9)
chunk_size <- 25
num_chunks <- ceiling(nrow(data) / chunk_size)
for (chunk in 1:chunk_size) {
  i1 <- (chunk - 1) * chunk_size + 1
  i2 <- chunk * chunk_size
  sub_group <- data$ptid[i1:i2]
  sub_grids <- filter(prediction_grids, ptid %in% sub_group)
  sub_obs <- filter(observations, ptid %in% sub_group)

  p <- ggplot() + xlim(0, 25) + ylim(0, 120)
  p <- p + geom_line(aes(x, y, group = rank, color = rank, alpha = weight), data = sub_grids)
  p <- p + geom_point(aes(x, y, color = observed), data = sub_obs)
  p <- p + facet_wrap(~ ptid)
  print(p)
}
dev.off()

## breaks <- c(1, 2, 4, 8, 25)
## breaks <- c(1, 2, 3, 4, 8, 25)
## predictions$bin <- cut(predictions$x, breaks)
## predictions$y <- filter(pfvc, years_seen_full > censor_time)$pfvc_sm

## predictions %>%
##   mutate(err = abs(y - yhat)) %>%
##   group_by(bin) %>%
##   summarize(mae = round(mean(err), 2)) %>%
##   print

## predictions %>%
##   mutate(err = abs(y - yhat)) %>%
##   group_by(true_subtype, bin) %>%
##   summarize(mae = mean(err)) %>%
##   spread(bin, mae) %>%
##   round(2)

## predictions %>%
##   mutate(err = abs(y - yhat)) %>%
##   group_by(true_subtype, subtype) %>%
##   summarize(mae = mean(err)) %>%
##   spread(subtype, mae) %>%
##   round(2)

## bins <- levels(predictions$bin)
## for (b in bins) {
##   b_pred <- filter(predictions, bin == b)
##   cat(b, "\n")
##   b_summ <- b_pred %>%
##     mutate(err = abs(y - yhat)) %>%
##     group_by(true_subtype, subtype) %>%
##     summarize(mae = mean(err)) %>%
##     spread(subtype, mae)
##   b_summ %>% round(2) %>% print
## }
