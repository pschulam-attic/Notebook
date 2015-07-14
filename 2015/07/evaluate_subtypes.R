library(caret)
library(dplyr)
library(readr)

subtypes <- read_csv("benchmark_pfvc_subtypes.csv")

subtypes_1y      <- read_csv("benchmark_pfvc_1y_subtypes.csv")
subtypes_1y_base <- read_csv("benchmark_pfvc_1y_subtypes_base_adjustment.csv")
subtypes_1y_coef <- read_csv("benchmark_pfvc_1y_subtypes_coef_adjustment.csv")
subtypes_1y_coef_true <- read_csv("benchmark_pfvc_1y_subtypes_coef_true_adjustment.csv")
subtypes_1y_val  <- read_csv("benchmark_pfvc_1y_subtypes_val_adjustment.csv")

subtypes_2y <- read_csv("benchmark_pfvc_2y_subtypes.csv")

cat("\nSubtype Predictions After *1* Year\n")
confusionMatrix(table(subtypes_1y$subtype, subtypes$subtype))

cat("\nSubtype Predictions After *1* Year w/ Base Adjustment\n")
confusionMatrix(table(subtypes_1y_base$subtype, subtypes$subtype))

cat("\nSubtype Predictions After *1* Year w/ Coef Adjustment\n")
confusionMatrix(table(subtypes_1y_coef$subtype, subtypes$subtype))

cat("\nSubtype Predictions After *1* Year w/ True Coef Adjustment\n")
confusionMatrix(table(subtypes_1y_coef_true$subtype, subtypes$subtype))

## cat("\nSubtype Predictions After *1* Year w/ Value Adjustment\n")
## confusionMatrix(table(subtypes_1y_val$subtype, subtypes$subtype))

## cat("\nSubtype Predictions After *2* Years\n")
## confusionMatrix(table(subtypes_2y$subtype, subtypes$subtype))
