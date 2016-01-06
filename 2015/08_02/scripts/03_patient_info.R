library(dplyr)
library(readr)

DESTINATION <- "data/derived"
OUTPUT <- "patient_info.csv"

features <- c(
  female = "strFemale",
  race = "strRaceId1",
  transplant = "ysntransplant",
  smoking_status = "ingSmokStatus",
  started_smoking = "lngAgeStart",
  stopped_smoking = "lngAgeStop",
  dead = "sngDead"
)

main <- function()
{
  ptdata <- read_csv("data/original/ptdata.csv")

  info <- ptdata[c("ingPtID", features)]
  colnames(info) <- c("ptid", names(features))

  write_csv(info, file.path(DESTINATION, OUTPUT))
}

main()
