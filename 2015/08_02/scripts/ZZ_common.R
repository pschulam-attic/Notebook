msg <- function(m, ...)
{
  m <- sprintf(m, ...)
  message(m)
}

wrn <- function(m, ...)
{
  m <- sprintf(m, ...)
  warning(m, call. = FALSE)
}

err <- function(m, ...)
{
  m <- sprintf(m, ...)
  stop(m, call. = FALSE)
}
