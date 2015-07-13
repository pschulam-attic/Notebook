library(caret)
library(dplyr)
library(readr)

subtypes    <- read_csv("benchmark_pfvc_subtypes.csv")
subtypes_1y <- read_csv("benchmark_pfvc_1y_subtypes.csv")
subtypes_2y <- read_csv("benchmark_pfvc_2y_subtypes.csv")

cat("\nSubtype Predictions After *1* Year\n")
confusionMatrix(table(subtypes_1y$subtype, subtypes$subtype))

cat("\nSubtype Predictions After *2* Years\n")
confusionMatrix(table(subtypes_2y$subtype, subtypes$subtype))
