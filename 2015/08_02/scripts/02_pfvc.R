library(readr)
library(dplyr, warn.c=FALSE)
library(lubridate)

DESTINATION <- "data/derived"
OUTPUT <- "pfvc.csv"

main <- function()
{
  pft <- read_csv("data/original/pft.csv")
  pfvc <- pft %>% select(ptid=ingPtID, date=STUDY_DATE, pfvc=stppFVCpred)
  write_csv(pfvc, file.path(DESTINATION, OUTPUT))
}

main()
