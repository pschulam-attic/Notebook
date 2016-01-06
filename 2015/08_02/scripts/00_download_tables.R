library(RODBC)
library(dplyr, warn.c=FALSE)
library(readr, warn.c=FALSE)

USERNAME <- "WIN-T312MF37MBJ\\pschulam"
PASSWORD <- "Bism@Rck13"

DESTINATION <- "data/original"

TABLES_TO_DOWNLOAD <- c(
  ptdata = "tPtData",
  visit = "tVisit",
  pft = "tPFT",
  echo = "tECHO",
  med = "tMeds"
)

msg <- function(m, ...)
{
  m <- sprintf(m, ...)
  message(m)
}

msg("Opening connection...")
sclerodata_conn <- odbcConnect("sclerodata", uid=USERNAME, pwd=PASSWORD)
msg("Done.")

for (i in seq_along(TABLES_TO_DOWNLOAD))
{
  basename <- names(TABLES_TO_DOWNLOAD)[i]
  filename <- paste0(basename, ".csv")
  filename <- file.path(DESTINATION, filename)
  tablename <- TABLES_TO_DOWNLOAD[i]

  msg("Downloading %s...", tablename)
  query <- sprintf("SELECT * FROM %s", tablename)
  tbl <- tbl_df(sqlQuery(sclerodata_conn, query))
  msg("Done.")

  msg("Writing %s to %s...", tablename, filename)
  write_csv(tbl, filename)
  msg("Done.")
}

msg("Closing connection...")
odbcClose(sclerodata_conn)
msg("Done.")
