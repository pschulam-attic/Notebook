library(dplyr)
library(tidyr)
source("~/Git/nips15-model/functions.R")

set.seed(1)

num_subtypes <- 8

pfvc  <- read_csv("data/benchmark_pfvc.csv")
data  <- pfvc %>% group_by(ptid) %>% do(fold = .$fold[1], datum = make_datum(.))
model <- fit_model(data$datum, c(num_subtypes = num_subtypes, max_iter = 100, tol = 1e-6, xlo = -1, xhi = 23))
saveRDS(model, sprintf("benchmark_pfvc_model_%02d.rds", num_subtypes))
