suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
})
source("~/Git/nips15-model/functions.R")
options(width = 120, dplyr.width = 120)

pfvc  <- read_csv("data/benchmark_pfvc.csv")
data  <- pfvc %>% group_by(ptid) %>% do(fold = .$fold[1], datum = make_datum(.))
model <- readRDS("benchmark_pfvc_model.rds")

truncate_datum <- function(datum, censor_time) {
  keep <- datum[["x"]] <= censor_time
  datum[["x"]] <- datum[["x"]][keep]
  datum[["y"]] <- datum[["y"]][keep]
  datum
}

predict_datum <- function(datum, censor_time, subtype, model) {
  observed <- truncate_datum(datum, censor_time)
  xnew <- datum[["x"]][datum[["x"]] > censor_time]
  if (length(xnew) == 0) return(numeric(0))
  ynew <- apply_model(observed, model, xnew)$ynew_hat[, subtype]
  list(x = xnew, y = ynew)
}

censor_time <- 1.0
true_subtypes_file <- "benchmark_pfvc_subtypes.csv"
pred_subtypes_file <- commandArgs(trailingOnly = TRUE)
#pred_subtypes_file <- "benchmark_pfvc_1y_subtypes.csv"

true_subtypes <- read_csv(true_subtypes_file) %>% select(ptid = ptid, true_subtype = subtype)
pred_subtypes <- read_csv(pred_subtypes_file)
data <- data %>% inner_join(true_subtypes, "ptid") %>% inner_join(pred_subtypes, "ptid")

predictions <- vector("list", nrow(data))
for (i in seq_along(predictions)) {
  pred <- predict_datum(data$datum[[i]], censor_time, data$subtype[i], model)
  if (length(pred) == 0) next
  predictions[[i]] <- data.frame(ptid = data$datum[[i]]$ptid, x = pred$x, yhat = pred$y)
}
predictions <- do.call("rbind", predictions) %>% tbl_df

breaks <- c(1, 2, 4, 8, 25)
predictions$bin <- cut(predictions$x, breaks)
predictions$y <- filter(pfvc, years_seen_full > censor_time)$pfvc_sm
predictions <- predictions %>% inner_join(true_subtypes, "ptid") %>% inner_join(pred_subtypes, "ptid")

predictions %>%
  mutate(err = abs(y - yhat)) %>%
  group_by(bin) %>%
  summarize(mae = round(mean(err), 2))

predictions %>%
  mutate(err = abs(y - yhat)) %>%
  group_by(true_subtype, bin) %>%
  summarize(mae = mean(err)) %>%
  spread(bin, mae) %>%
  round(2)

predictions %>%
  mutate(err = abs(y - yhat)) %>%
  group_by(true_subtype, subtype) %>%
  summarize(mae = mean(err)) %>%
  spread(subtype, mae) %>%
  round(2)

bins <- levels(predictions$bin)
for (b in bins) {
  b_pred <- filter(predictions, bin == b)
  cat(b, "\n")
  b_summ <- b_pred %>%
    mutate(err = abs(y - yhat)) %>%
    group_by(true_subtype, subtype) %>%
    summarize(mae = mean(err)) %>%
    spread(subtype, mae)
  b_summ %>% round(2) %>% print
}
