library(dplyr)
library(readr)
library(ggplot2)

severity_files <- c("gen.csv", "hrt.csv", "gi.csv", "kid.csv", "msc.csv", "rp.csv")
severity_files <- file.path("data", severity_files)

gen <- read_csv(severity_files[1]) %>%
  group_by(ptid) %>% mutate(max_general = max(general)) %>%
  filter(years_seen <= 1.0) %>% ungroup

p <- ggplot(gen) + geom_histogram(aes(general)) + facet_wrap(~ max_general)
print(p)


hrt <- read_csv(severity_files[2]) %>%
  group_by(ptid) %>% mutate(max_heart = max(heart)) %>%
  filter(years_seen <= 1.0) %>% ungroup

p <- ggplot(hrt) + geom_histogram(aes(heart)) + facet_wrap(~ max_heart)
print(p)


gi <- read_csv(severity_files[3]) %>%
  group_by(ptid) %>% mutate(max_gi = max(gi)) %>%
  filter(years_seen <= 1.0) %>% ungroup

p <- ggplot(gi) + geom_histogram(aes(gi)) + facet_wrap(~ max_gi)
print(p)


kid <- read_csv(severity_files[4]) %>%
  group_by(ptid) %>% mutate(max_kidney = max(kidney)) %>%
  filter(years_seen <= 1.0) %>% ungroup

p <- ggplot(kid) + geom_histogram(aes(kidney)) + facet_wrap(~ max_kidney)
print(p)


msc <- read_csv(severity_files[5]) %>%
  group_by(ptid) %>% mutate(max_muscle = max(muscle)) %>%
  filter(years_seen <= 1.0) %>% ungroup

p <- ggplot(msc) + geom_histogram(aes(muscle)) + facet_wrap(~ max_muscle)
print(p)


rp <- read_csv(severity_files[6]) %>%
  group_by(ptid) %>% mutate(max_rp = max(rp)) %>%
  filter(years_seen <= 1.0) %>% ungroup

p <- ggplot(rp) + geom_histogram(aes(rp)) + facet_wrap(~ max_rp)
print(p)

