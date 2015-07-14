library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)

pfvc <- read_csv("data/benchmark_pfvc.csv")
args <- commandArgs(trailingOnly = TRUE)
true_subtypes <- args[1] %>% read_csv %>% select(ptid = ptid, true_subtype = subtype)
pred_subtypes <- args[2] %>% read_csv
targ_subtype  <- as.integer(args[3])

pfvc <- pfvc %>% inner_join(true_subtypes, "ptid") %>% inner_join(pred_subtypes, "ptid")
pfvc <- pfvc %>% filter(true_subtype == targ_subtype, subtype != targ_subtype)

pdf(sprintf("misclassifications_%02d.pdf", targ_subtype), width = 24, height = 18)
p <- ggplot(pfvc) + xlim(0, 20) + ylim(0, 120)
p <- p + geom_point(aes(years_seen_full, pfvc))
p <- p + facet_wrap(~ ptid)
print(p)
dev.off()
