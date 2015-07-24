suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
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

args            <- commandArgs(trailingOnly = TRUE)
censor_time     <- as.numeric(args[1])
posteriors_file <- args[2]

posteriors <- read_csv(posteriors_file)
data       <- data %>% inner_join(posteriors, "ptid")

predictions <- vector("list", nrow(data))
for (i in seq_along(predictions)) {
  m <- models[[data$fold[[i]]]]
  p <- predict_datum(data$datum[[i]], censor_time, m)
  if (length(p) == 0) next
  w <- unlist(data[i, -c(1, 2, 3)])
  x <- p$x
  y <- p$y %*% w
  predictions[[i]] <- data.frame(ptid = data$datum[[i]]$ptid, x = x, yhat = y)
}
predictions <- do.call("rbind", predictions) %>% tbl_df

breaks <- c(1, 2, 4, 8, 25)
predictions$bin <- cut(predictions$x, breaks)
predictions$y <- filter(pfvc, years_seen_full > censor_time)$pfvc_sm

predictions %>%
  mutate(err = abs(y - yhat)) %>%
  group_by(bin) %>%
  summarize(mae = round(mean(err), 2)) %>%
  print

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
