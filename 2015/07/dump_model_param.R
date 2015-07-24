library(nnet)
source("~/Git/nips15-model/functions.R")
m <- readRDS("benchmark_pfvc_model.rds")
dump_model(m, "param")

## bknots <- environment(m$basis)$boundary_knots
## iknots <- environment(m$basis)$interior_knots
## degree <- environment(m$basis)$degree
## ncoef  <- length(iknots) + degree + 1
## basis_param <- matrix(c(bknots, degree, ncoef), nrow=1)
## write.table(basis_param, "param/basis.dat", row.names=FALSE, col.names=FALSE)

## v_const <- environment(m$kernel)$v_const
## v_ou    <- environment(m$kernel)$v_ou
## l_ou    <- environment(m$kernel)$l_ou
## if (exists("v_noise", environment(m$kernel))) {
##   v_noise <- environment(m$kernel)$v_noise
## } else {
##   v_noise <- 1.0
## }
## kernel_param <- matrix(c(v_const, v_ou, l_ou, v_noise), nrow=1)
## write.table(kernel_param, "param/kernel.dat", row.names=FALSE, col.names=FALSE)

## b <- matrix(m$param$b, nrow=1)
## B <- t(m$param$B)
## W <- unname(coef(m$param$m))
## write.table(b, "param/pop.dat", row.names=FALSE, col.names=FALSE)
## write.table(B, "param/subpop.dat", row.names=FALSE, col.names=FALSE)
## write.table(W, "param/marginal.dat", row.names=FALSE, col.names=FALSE)
