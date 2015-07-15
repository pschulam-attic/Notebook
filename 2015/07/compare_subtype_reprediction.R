suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
})

true_subtypes <- read_csv("benchmark_pfvc_subtypes.csv")
pred_subtypes <- read_csv("benchmark_pfvc_1y_subtypes.csv")
repred_subtypes <- read_csv(commandArgs(trailingOnly = TRUE)[1])

org_correct <- true_subtypes$subtype == pred_subtypes$subtype
new_correct <- true_subtypes$subtype == repred_subtypes$subtype

pred <- pred_subtypes$subtype[new_correct & !org_correct]
repred <- repred_subtypes$subtype[new_correct & !org_correct]

table(pred, repred)
