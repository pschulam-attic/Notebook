suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
})
source("~/Git/nips15-model/functions.R")

make_datum <- function(patient_tbl) {
  list(
    ptid     = patient_tbl[["ptid"]][1]
  , x        = patient_tbl[["years_seen"]]
  , y        = patient_tbl[[marker_var]]
  , sub_feat = model.matrix(~ female+afram-1, patient_tbl)[1, ]
  , pop_feat = model.matrix(~ female+afram-1, patient_tbl)[1, ]
  )
}

plot_model <- function(m, opts, ...) {
  x <- seq(opts[["xlo"]], opts[["xhi"]], 0.1)
  X <- m$basis(x)
  Y <- X %*% m$param$B
  matplot(x, Y, ...)
}

set.seed(1)

job_info <- list(

  list(
    aux_marker = "tss",
    marker_var = "tss",
    opts = c(
      num_subtypes = 4L,
      xlo          = 0,
      xhi          = 25,
      num_coef     = 6,
      degree       = 2,
      v_const      = 9.0,
      v_ou         = 16.0,
      l_ou         = 2.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-4
    )
  ), ## End pDLCO

  list(
    aux_marker = "pdc",
    marker_var = "pdlco",
    opts = c(
      num_subtypes = 8L,
      xlo          = 0,
      xhi          = 25,
      num_coef     = 6,
      degree       = 2,
      v_const      = 36.0,
      v_ou         = 25.0,
      l_ou         = 2.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-4
    )
  ), ## End pDLCO

  list(
    aux_marker = "pv1",
    marker_var = "pfev1",
    opts = c(
      num_subtypes = 4L,
      xlo          = 0,
      xhi          = 25,
      num_coef     = 6,
      degree       = 2,
      v_const      = 25.0,
      v_ou         = 16.0,
      l_ou         = 2.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-4
    )
  ), # End pFEV1

  list(
    aux_marker = "ef",
    marker_var = "ef",
    opts = c(
      num_subtypes = 3L,
      xlo          = 0,
      xhi          = 25,
      num_coef     = 6,
      degree       = 2,
      v_const      = 16.0,
      v_ou         = 9.0,
      l_ou         = 2.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-4
    )
  ), # End EF

  list(
    aux_marker = "sp",
    marker_var = "rvsp",
    opts = c(
      num_subtypes = 3L,
      xlo          = 0,
      xhi          = 25,
      num_coef     = 6,
      degree       = 2,
      v_const      = 16.0,
      v_ou         = 9.0,
      l_ou         = 2.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-4
    )
  ), # End EF

  list(
    aux_marker = "hrt",
    marker_var = "heart",
    opts = c(
      num_subtypes = 4L,
      xlo          = 0,
      xhi          = 25,
      num_coef     = 6,
      degree       = 2,
      v_const      = 0.0,
      v_ou         = 0.0,
      l_ou         = 1.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-4
    )
  ), ## End heart

  list(
    aux_marker = "gi",
    marker_var = "gi",
    opts = c(
      num_subtypes = 2L,
      xlo          = 0,
      xhi          = 25,
      num_coef     = 3,
      degree       = 1,
      v_const      = 25.0,
      v_ou         = 0.0,
      l_ou         = 1.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-6
    )
  ), ## End GI

  list(
    aux_marker = "rp",
    marker_var = "rp",
    opts = c(
      num_subtypes = 2L,
      xlo          = 0,
      xhi          = 25,
      num_coef     = 3,
      degree       = 1,
      v_const      = 25.0,
      v_ou         = 0.0,
      l_ou         = 1.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-6
    )
  ), ## End RP

  list(
    aux_marker = "msc",
    marker_var = "muscle",
    opts = c(
      num_subtypes = 2L,
      xlo          = 0,
      xhi          = 25,
      num_coef     = 3,
      degree       = 1,
      v_const      = 25.0,
      v_ou         = 0.0,
      l_ou         = 1.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-6
    )
  ), ## End Muscle

  list(
    aux_marker = "kid",
    marker_var = "kidney",
    opts = c(
      num_subtypes = 2L,
      xlo          = 0,
      xhi          = 25,
      num_coef     = 3,
      degree       = 1,
      v_const      = 25.0,
      v_ou         = 0.0,
      l_ou         = 1.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-6
    )
  ), ## End Kidney

  list(
    aux_marker = "wgt",
    marker_var = "weight",
    opts = c(
      num_subtypes = 2L,
      xlo          = 0,
      xhi          = 25,
      num_coef     = 3,
      degree       = 1,
      v_const      = 25.0,
      v_ou         = 0.0,
      l_ou         = 1.0,
      v_noise      = 1.0,
      max_iter     = 100,
      tol          = 1e-6
    )
  )  ## End Weight

)  ## End job info

for (job in job_info) {

  aux_marker <- job$aux_marker
  marker_var <- job$marker_var
  opts       <- job$opts

  aux_data  <- sprintf("%s.csv", aux_marker) %>% file.path("data", .) %>% read_csv
  aux_data  <- aux_data %>% filter(years_seen <= opts[["xhi"]])
  aux_model <- sprintf("%s_model_and_data.rds", aux_marker)

  features <- read_csv("data/benchmark_pfvc.csv") %>% select(ptid, female, afram, fold)
  features <- features[!duplicated(features$ptid), ]
  subtypes <- read_csv("benchmark_pfvc_subtypes.csv")

  aux_data <- inner_join(aux_data, features, "ptid")
  write_csv(aux_data, file.path("data", sprintf("benchmark_%s.csv", aux_marker)))
  ## aux_data <- inner_join(aux_data, subtypes, "ptid")

  ## prepared_data <- aux_data %>% group_by(ptid) %>% do(datum = make_datum(.))
  ## prepared_data <- inner_join(prepared_data, subtypes, "ptid")

  ## model <- fit_model(prepared_data$datum, opts)

  ## saveRDS(list(model, prepared_data), file.path("models", aux_model))

  ## dump_model(model, file.path("models", aux_marker))

  ## num_subtypes <- length(unique(prepared_data$subtype))
  ## models       <- vector("list", num_subtypes)

  ## for (k in 1:num_subtypes) {
  ##   train_ix    <- prepared_data$subtype == k
  ##   train       <- prepared_data$datum[train_ix]
  ##   models[[k]] <- fit_model(train, opts)
  ## }

  ## saveRDS(list(models, prepared_data), aux_model)
}
