#!/usr/bin/env Rscript
# Freeze the R `gmm` reference for the multi-asset Euler / Hansen-Singleton
# cross-check (Tier 2 of docs/validation/r-reference-crosschecks.org).
#
# Reads tests/data/gmm_euler.csv (from gen_euler_data.py, the bundled
# emu_gmm.examples.euler DGP) and fits the SAME nonlinear moment
#   psi_j(theta) = beta * (c_next/c_t)^(-gamma) * (1 + r_j) - 1,   j = 1..3
# (M=3, K=2 (beta, gamma), one over-identifying restriction) via R's gmm,
# writing tests/data/gmm_euler_reference.json + provenance. The Python test
# consumes the JSON (no R in CI). Regenerate:
#
#     Rscript scripts/gmm_reference/reference_euler.R
#
# R + gmm install from the Ubuntu r-cran-* debs (CRAN is blocked by the agent
# egress policy) -- see docs/validation/r-reference-crosschecks.org.

suppressMessages({
  library(gmm)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
this_file <- sub("^--file=", "", args[grep("^--file=", args)])
root <- normalizePath(file.path(dirname(this_file), "..", ".."))
data_path <- file.path(root, "tests", "data", "gmm_euler.csv")
out_path <- file.path(root, "tests", "data", "gmm_euler_reference.json")

x <- as.matrix(read.csv(data_path))
data_sha256 <- digest::digest(file = data_path, algo = "sha256")

# The bundled Euler moment, columns (c_t, c_next, r0, r1, r2).
g <- function(theta, x) {
  sdf <- theta[1] * (x[, 2] / x[, 1])^(-theta[2])
  cbind(sdf * (1 + x[, 3]) - 1, sdf * (1 + x[, 4]) - 1, sdf * (1 + x[, 5]) - 1)
}

num <- function(v) as.numeric(v)
fit <- function(type) {
  res <- gmm(g, x, t0 = c(beta = 0.95, gamma = 1.5), type = type, vcov = "iid")
  cf <- num(coef(res))
  se <- num(sqrt(diag(vcov(res))))
  st <- num(specTest(res)$test)
  list(
    coef = list(beta = cf[1], gamma = cf[2]),
    se = list(beta = se[1], gamma = se[2]),
    J = st[1], J_pvalue = st[2], J_df = 1L
  )
}

reference <- list(
  provenance = list(
    description = "R gmm reference for the emu-gmm multi-asset Euler (Hansen-Singleton) cross-check",
    generated_by = "scripts/gmm_reference/reference_euler.R",
    data_file = "tests/data/gmm_euler.csv",
    data_sha256 = data_sha256,
    r_version = as.character(getRversion()),
    packages = list(gmm = as.character(packageVersion("gmm"))),
    model = "psi_j = beta*(c_next/c_t)^(-gamma)*(1+r_j)-1, j=1..3; M=3, K=2, over_id=1",
    truth = list(beta = 0.96, gamma = 2.0),
    note = paste(
      "emu-gmm and gmm agree to ~5-6 significant figures on this nonlinear fit",
      "(tighter than the linear-IV case: at the over-identified optimum the",
      "moment mean is ~0, so the centered-vs-uncentered covariance gap nearly",
      "vanishes). Both reference J to chi^2_1."
    )
  ),
  gmm_cue = fit("cue"),
  gmm_iterative = fit("iterative")
)

write_json(reference, out_path, auto_unbox = TRUE, digits = 17, pretty = TRUE)
cat("wrote", out_path, "\n")
