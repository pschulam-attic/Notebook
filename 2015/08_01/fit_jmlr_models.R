suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
})
source("~/Git/nips15-model/functions.R")

set.seed(1)

steps <- c()
steps <- c(steps, "train_seeds")
steps <- c(steps, "train_folds")


datum_factory <- function(time, marker, pop_form, sub_form) {
  function(patient_tbl) {
    list(
      ptid     = patient_tbl[["ptid"]][1]
    , x        = patient_tbl[[time]]
    , y        = patient_tbl[[marker]]
    , pop_feat = model.matrix(pop_form, patient_tbl)[1, ]
    , sub_feat = model.matrix(sub_form, patient_tbl)[1, ]
    )
  }
}

seed_config <- list(

  ## (1)
  list(
    short_name = "pfvc",
    time_name  = "years_seen_full",
    var_name   = "pfvc",
    data_file  = "data/benchmark_pfvc.csv",
    pop_form   = ~ female + afram - 1,
    sub_form   = ~ female + afram + aca + scl - 1,
    opts = c(
      num_subtypes = 9L,
      xlo          = -1,
      xhi          = 25,
      num_coef     = 6,
      degree       = 2,
      v_const      = 16.0,
      v_ou         = 36.0,
      l_ou         = 2.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-3
    )
  ),

  ## (2)
  list(
    short_name = "pdc",
    time_name  = "years_seen",
    var_name   = "pdlco",
    data_file  = "data/benchmark_pdc.csv",
    pop_form   = ~ female + afram - 1,
    sub_form   = ~ female + afram - 1,
    opts = c(
      num_subtypes = 6L,
      xlo          = -1,
      xhi          = 25,
      num_coef     = 6,
      degree       = 2,
      v_const      = 16.0,
      v_ou         = 36.0,
      l_ou         = 2.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-3
    )
  ),

  ## (3)
  list(
    short_name = "tss",
    time_name  = "years_seen",
    var_name   = "tss",
    data_file  = "data/benchmark_tss.csv",
    pop_form   = ~ female + afram - 1,
    sub_form   = ~ female + afram - 1,
    opts = c(
      num_subtypes = 4L,
      xlo          = -1,
      xhi          = 25,
      num_coef     = 6,
      degree       = 2,
      v_const      = 9.0,
      v_ou         = 16.0,
      l_ou         = 2.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-3
    )
  ),

  ## (4)
  list(
    short_name = "pv1",
    time_name  = "years_seen",
    var_name   = "pfev1",
    data_file  = "data/benchmark_pv1.csv",
    pop_form   = ~ female + afram - 1,
    sub_form   = ~ female + afram - 1,
    opts = c(
      num_subtypes = 4L,
      xlo          = -1,
      xhi          = 25,
      num_coef     = 6,
      degree       = 2,
      v_const      = 25.0,
      v_ou         = 16.0,
      l_ou         = 2.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-3
    )
  ),

  ## (5)
  list(
    short_name = "sp",
    time_name  = "years_seen",
    var_name   = "rvsp",
    data_file  = "data/benchmark_sp.csv",
    pop_form   = ~ female + afram - 1,
    sub_form   = ~ female + afram - 1,
    opts = c(
      num_subtypes = 4L,
      xlo          = -1,
      xhi          = 25,
      num_coef     = 6,
      degree       = 2,
      v_const      = 9.0,
      v_ou         = 16.0,
      l_ou         = 2.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-3
    )
  )

)

seed_config <- seed_config[1]

seed_dir <- "models/jmlr/seeds"

if ("train_seeds" %in% steps)
{
  for (config in seed_config)
  {
    model_dir <- file.path(seed_dir, config$short_name)
    param_dir <- file.path(model_dir, "param")

    dir.create(model_dir, recursive = TRUE)
    dir.create(param_dir, recursive = TRUE)

    make_datum <- datum_factory(config$time_name, config$var_name, config$pop_form, config$sub_form)
    data_tbl <- read_csv(config$data_file) %>% group_by(ptid)
    data_tbl <- do(data_tbl, datum = make_datum(.))

    model <- fit_model(data_tbl$datum, config$opts)
    saveRDS(model, file.path(model_dir, "model.rds"))
    dump_model(model, param_dir)

    pdf(file.path(model_dir, "subtypes.pdf"), width = 8, height = 6)
    plot_model(model)
    dev.off()
  }
}

num_folds <- 10
fold_dir <- "models/jmlr/folds"

if ("train_folds" %in% steps)
{
  for (config in seed_config)
  {
    make_datum <- datum_factory(config$time_name, config$var_name, config$pop_form, config$sub_form)
    data_tbl <- read_csv(config$data_file) %>% group_by(ptid)
    data_tbl <- do(data_tbl, datum = make_datum(.), fold = .$fold[1])

    seed_model <- readRDS(file.path(seed_dir, config$short_name, "model.rds"))

    for (fold in 1:num_folds)
    {
      model_dir <- file.path(fold_dir, sprintf("%s/%02d", config$short_name, fold))
      param_dir <- file.path(model_dir, "param")

      dir.create(model_dir, recursive = TRUE)
      dir.create(param_dir, recursive = TRUE)

      train <- data_tbl$fold != fold
      opts  <- config$opts
      opts$tol <- 1e-4
      model <- fit_model(data_tbl$datum[train], opts, seed_model)
      saveRDS(model, file.path(model_dir, "model.rds"))
      dump_model(model, param_dir)

      pdf(file.path(model_dir, "subtypes.pdf"), width = 8, height = 6)
      plot_model(model)
      dev.off()
    }
  }
}
