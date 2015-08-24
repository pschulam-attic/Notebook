library(dplyr)
library(readr)

as.severity <- function(x) {
  factor(x, levels = c(0, 1, 2, 3, 4))
}

estimate_likelihood <- function(counts, smooth=1) {
  counts <- counts + 1
  denom <- rowSums(counts)
  counts / denom
}

all_censor_times <- c(1.0, 2.0, 3.0, 4.0, 8.0)

for (censor_time in all_censor_times) {
  severity_file <- "data/hrt.csv"
  marker_name <- "heart"
  no_active_cutoff <- 3
  folds <- read_csv("data/benchmark_pfvc.csv") %>% group_by(ptid) %>% summarize(fold = fold[1]) %>% ungroup
  output_file <- sprintf("data/hrt_likelihoods_%.01f.dat", censor_time)

  severities <- read_csv(severity_file)
  i <- severities$ptid
  x <- severities$years_seen
  y <- as.integer(severities[[marker_name]] / 25)

  normalized <- data.frame(ptid = i, x = x, y = y) %>% tbl_df %>% inner_join(folds, "ptid")
  normalized <- normalized %>% group_by(ptid) %>% mutate(z = (max(y) > no_active_cutoff) + 1) %>% ungroup
  normalized <- filter(normalized, x <= censor_time)

  likel_factor <- matrix(0, nrow(folds), 2)
  rownames(likel_factor) <- folds$ptid
  logliks <- vector("list", 10)

  for (k in 1:10) {
    train_data <- filter(normalized, fold != k)
    counts <- with(train_data, table(z, as.severity(y)))
    loglik <- log(estimate_likelihood(counts))
    logliks[[k]] <- loglik

    test_data <- filter(normalized, fold == k)
    test_ptid <- unique(test_data$ptid)
    for (ptid in test_ptid) {
      y <- test_data$y[test_data$ptid == ptid]
      if (length(y) < 1) next
      y <- as.character(y)
      for (z in 1:2) {
        ll <- sum(loglik[z, ][y])
        likel_factor[as.character(ptid), z] <- ll
      }
    }
  }

  y_true <- group_by(normalized, ptid) %>% summarize(z = z[1])
  y_true <- y_true$z

  has_true <- as.integer(rownames(likel_factor)) %in% normalized$ptid
  y_pred <- apply(likel_factor[has_true, ], 1, which.max)

  print(table(y_true, y_pred))

  write.table(likel_factor, output_file, col.names=FALSE, row.names=FALSE)
}
