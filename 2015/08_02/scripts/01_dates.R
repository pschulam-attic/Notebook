library(dplyr, warn.c = FALSE)
library(readr)
library(lubridate)
library(ggplot2)
library(magrittr)

DESTINATION <- "data/derived"
OUTPUT <- "dates.csv"

main <- function()
{
  ptdata <- read_csv("data/original/ptdata.csv")

  first_seen <- ptdata %>% select(ptid=ingPtID, first_seen=strFrstSeen)

  first_sick <- ptdata %>% select(ptid=ingPtID, first_sick=strDoFrstSx)

  dates <- left_join(first_seen, first_sick, "ptid")

  write_csv(dates, file.path(DESTINATION, OUTPUT))
}

main()
