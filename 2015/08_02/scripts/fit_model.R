library(readr)
library(dplyr, warn.c=FALSE)
source("scripts/ZZ_common.R")

main <- function()
{
  pfvc <- read_csv("data/derived/pfvc.csv")

  dates <- read_csv("data/derived/dates.csv")

  pfvc <- align_longitudinal_data(pfvc, dates, "ptid", "date", "first_sick")
}

align_longitudinal_data <- function(tbl, dates, key, timestamp, base)
{
  ## Check that longitudinal has the key.
  k1 <- tbl[[key]]
  if (is.null(k1))
    err("`tbl` does not contain key variable \"%s\".", key)

  ## Check that the dates table has the key.
  k2 <- dates[[key]]
  if (is.null(k2))
    err("`dates` does not contain key variable \"%s\".", key)

  ## Match longitudinal observations to baseline data.
  ax <- match(k1, k2)

  no_match <- unique(k1[is.na(ax)])
  no_match_str <- paste(no_match, collapse = ", ")
  if (length(no_match) > 0)
    wrn("%d unit(s) are not in the dates table: [%s]", length(no_match), no_match_str)

  tbl$base <- dates[[base]][ax]

  tbl
}

main()
