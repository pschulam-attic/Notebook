library(splines)
m <- readRDS("benchmark_pfvc_model.rds")
x <- seq(0, 20, 0.5)
X <- m$basis(x)
Y <- X %*% m$param$B
pdf("pfvc_subtypes.pdf", width = 8, height = 6)
matplot(x, Y)
dev.off()
