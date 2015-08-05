suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
})
source("~/Git/nips15-model/functions.R")

truncate_datum <- function(datum, censor_time) {
  obs <- datum[["x"]] <= censor_time
  datum[["x"]] <- datum[["x"]][obs]
  datum[["y"]] <- datum[["y"]][obs]
  datum
}

censor_time <- 2.0

job_info <- list(

  list(
    aux_marker = "pdc"
  ),

  list(
    aux_marker = "pv1"
  ),

  list(
    aux_marker = "hrt"
  ),

  list(
    aux_marker = "gi"
  ),

  list(
    aux_marker = "rp"
  )

)

for (job in job_info) {
  aux_marker  <- job$aux_marker
  aux_model   <- sprintf("%s_model_and_data.rds", aux_marker) %>% readRDS

  models               <- aux_model[[1]]
  prepared_data        <- aux_model[[2]]
  truncated_data       <- prepared_data
  truncated_data$datum <- lapply(truncated_data$datum, truncate_datum, censor_time)

  nobs           <- vapply(truncated_data$datum, function(d) length(d$x), integer(1))
  truncated_data <- truncated_data[nobs > 0L, ]

  subtypes <- read_csv("benchmark_pfvc_subtypes.csv")
  aux_match <- match(truncated_data$ptid, subtypes$ptid)

  loglik <- matrix(0, nrow(subtypes), length(models))

  for (i in seq_along(models)) {
    m <- models[[i]]
    inferences <- lapply(truncated_data$datum, apply_model, m)
    loglik[aux_match, i] <- vapply(inferences, "[[", numeric(1), "likelihood")
    loglik[-aux_match, i] <- 0.0
  }

  ratios <- loglik - loglik[, 1]
  ratio_file <- sprintf("%s_%.1f_ratios.dat", aux_marker, censor_time) %>% file.path("param", .)
  write.table(ratios, ratio_file, row.names=FALSE, col.names=FALSE)
}
