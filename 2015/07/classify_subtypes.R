library(dplyr)
library(tidyr)
source("~/Git/nips15-model/functions.R")

set.seed(1)

pick_subtype <- function(inf) {
  which.max(inf$posterior)
}

truncate_datum <- function(datum, censor_time) {
  keep <- datum[["x"]] <= censor_time
  datum[["x"]] <- datum[["x"]][keep]
  datum[["y"]] <- datum[["y"]][keep]
  datum
}

pfvc  <- read_csv("data/benchmark_pfvc.csv")
data  <- pfvc %>% group_by(ptid) %>% do(fold = .$fold[1], datum = make_datum(.))
model <- readRDS("benchmark_pfvc_model.rds")

inferences   <- lapply(data$datum, apply_model, model)
subtypes     <- vapply(inferences, pick_subtype, numeric(1))
subtypes     <- data.frame(ptid = data$ptid, subtype = subtypes)
write_csv(subtypes, "benchmark_pfvc_subtypes.csv")

truncated  <- lapply(data$datum, truncate_datum, censor_time=1.0)
inferences <- lapply(truncated, apply_model, model)
subtypes   <- vapply(inferences, pick_subtype, numeric(1))
subtypes   <- data.frame(ptid = data$ptid, subtype = subtypes)
write_csv(subtypes, "benchmark_pfvc_1y_subtypes.csv")

posteriors <- do.call("rbind", lapply(inferences, "[[", "posterior"))
posteriors <- data.frame(posteriors)
names(posteriors) <- paste0("p", 1:ncol(posteriors))
posteriors <- cbind(data.frame(ptid = data$ptid) , posteriors)
write_csv(posteriors, "benchmark_pfvc_1y_posteriors.csv")

truncated  <- lapply(data$datum, truncate_datum, censor_time=2.0)
inferences <- lapply(truncated, apply_model, model)
subtypes   <- vapply(inferences, pick_subtype, numeric(1))
subtypes   <- data.frame(ptid = data$ptid, subtype = subtypes)
write_csv(subtypes, "benchmark_pfvc_2y_subtypes.csv")

posteriors <- do.call("rbind", lapply(inferences, "[[", "posterior"))
posteriors <- data.frame(posteriors)
names(posteriors) <- paste0("p", 1:ncol(posteriors))
posteriors <- cbind(data.frame(ptid = data$ptid) , posteriors)
write_csv(posteriors, "benchmark_pfvc_2y_posteriors.csv")
