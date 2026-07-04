#!/usr/bin/env Rscript
# Freeze the R reference estimates for the emu-gmm linear-IV cross-check.
#
# Reads the committed dataset tests/data/gmm_linear_iv.csv (produced by
# gen_data.py) and fits the SAME over-identified IV moment conditions three
# ways, writing the results + full provenance to
# tests/data/gmm_linear_iv_reference.json. The Python acceptance test
# (tests/validation/test_r_reference_linear_iv.py) consumes that JSON; it does
# NOT run R, so CI needs no R toolchain. Regenerate after changing the data:
#
#     Rscript scripts/gmm_reference/reference.R
#
# R + packages are installable in a Debian/Ubuntu container without CRAN via:
#     apt-get install -y r-base-core r-cran-gmm r-cran-aer r-cran-sandwich
# (CRAN itself is blocked by the agent egress policy; the Ubuntu r-cran-* debs
# are the reachable route -- see docs/validation/r-reference-crosschecks.org.)

suppressMessages({
  library(gmm)
  library(AER)
  library(sandwich)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
this_file <- sub("^--file=", "", args[grep("^--file=", args)])
root <- normalizePath(file.path(dirname(this_file), "..", ".."))
data_path <- file.path(root, "tests", "data", "gmm_linear_iv.csv")
out_path <- file.path(root, "tests", "data", "gmm_linear_iv_reference.json")

d <- read.csv(data_path)
data_sha256 <- digest::digest(file = data_path, algo = "sha256")

# Linear IV: regression y ~ xe, instruments ~ z1 + z2 + z3 (constant added to
# both by default) -> M = 4 moments, K = 2 params, 2 over-identifying dof.
# specTest()$test can carry a `noquote` print class; as.numeric() strips it so
# jsonlite serialises plain doubles.
num <- function(x) as.numeric(x)

# Use gmm's FUNCTION interface g(theta, x) -- the same general nonlinear entry
# point as the Euler cross-check -- NOT the formula interface gmm(y ~ x, ~ z).
# For LINEAR models the two disagree: the formula interface runs a different
# internal path (its iterated even collapses to exact 2SLS), giving a ~0.1% CUE
# difference, whereas the function interface is the standard general GMM that
# emu-gmm implements and that an independent numpy CUE confirms. The moments are
# identical: instruments (1, z1, z2, z3) times the residual (y - b0 - b1*xe).
# See docs/validation/r-reference-crosschecks.org.
g_lin <- function(theta, x) {
  r <- x[, "y"] - theta[1] - theta[2] * x[, "xe"]
  cbind(r, r * x[, "z1"], r * x[, "z2"], r * x[, "z3"])
}
xm <- as.matrix(d)

fit_gmm <- function(type) {
  res <- gmm(g_lin, xm, t0 = c(b0 = 0.0, b1 = 0.0), type = type, vcov = "iid")
  st <- num(specTest(res)$test)  # c(J, p) ; df = M - K is implicit (= 2 here)
  cf <- num(coef(res))
  se <- num(sqrt(diag(vcov(res))))
  list(
    coef = list(b0 = cf[1], b1 = cf[2]),
    se = list(b0 = se[1], b1 = se[2]),
    J = st[1], J_pvalue = st[2], J_df = 2L
  )
}

# AER 2SLS: the canonical, weight-fixed (Z'Z)^{-1} estimator -- the machine-
# precision anchor for the emu-gmm solver (matched-weight reproduction).
iv <- ivreg(y ~ xe | z1 + z2 + z3, data = d)
iv_cf <- num(coef(iv))
sarg <- summary(iv, diagnostics = TRUE)$diagnostics["Sargan", ]  # keep names

reference <- list(
  provenance = list(
    description = "R reference estimates for the emu-gmm linear over-identified IV cross-check",
    generated_by = "scripts/gmm_reference/reference.R",
    data_file = "tests/data/gmm_linear_iv.csv",
    data_sha256 = data_sha256,
    r_version = as.character(getRversion()),
    packages = list(
      gmm = as.character(packageVersion("gmm")),
      AER = as.character(packageVersion("AER")),
      sandwich = as.character(packageVersion("sandwich"))
    ),
    model = "y ~ xe, instruments (1, z1, z2, z3); M=4, K=2, over_id=2",
    note = paste(
      "gmm uses vcov='iid' via the FUNCTION interface g(theta, x); on this the",
      "gmm CUE/iterated estimates match emu-gmm to ~5-6 significant figures, and",
      "an independent numpy CUE confirms both. gmm's FORMULA interface",
      "(gmm(y~xe, ~z)) instead gives ~0.1% different CUE for linear models (its",
      "iterated collapses to exact 2SLS) -- a gmm interface quirk, not an emu",
      "difference. The 2SLS point (fixed (Z'Z)^-1 weight) is the exact",
      "machine-precision anchor. See docs/validation/r-reference-crosschecks.org."
    )
  ),
  twoStageLeastSquares = list(
    coef = list(b0 = iv_cf[1], b1 = iv_cf[2]),
    sargan_J = num(sarg["statistic"]),
    sargan_pvalue = num(sarg["p-value"]),
    J_df = 2L
  ),
  gmm_cue = fit_gmm("cue"),
  gmm_iterative = fit_gmm("iterative")
)

write_json(reference, out_path, auto_unbox = TRUE, digits = 17, pretty = TRUE)
cat("wrote", out_path, "\n")
cat("data_sha256:", data_sha256, "\n")
