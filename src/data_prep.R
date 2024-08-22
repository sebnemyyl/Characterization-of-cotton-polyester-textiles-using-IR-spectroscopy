#TODO improve this, make configurable
#setwd("C:/Users/sebne/Documents/FHWN_Tulln/DataAnalysis/repo")
setwd(".")

library(pls)
library(plotly)
library(dplyr)
source("src/util/02_data_prep_load_CSV.R")
source("src/util/03_data_prep_limit_spectra.R")
#source("src/util/04_data_prep_baseline_correction.R")
source("src/util/05_plot_spectra.R")


run_pls <- function(spectra_df, rds_path) {
  options(digits = 4)
  pls.options(plsalg = "oscorespls")
  ref_cotton <- spectra_df$reference.cotton
  spectra <- spectra_df_clean %>% dplyr::select(starts_with("spectra"))
  names(spectra) <- as.numeric(sub("spectra.", "", names(spectra)))
  spectra_matrix <- data.matrix(spectra)
  pls_result <- plsr(ref_cotton ~ spectra_matrix, data = spectra_df, ncomp = 10, val = "LOO")
  if (!missing(rds_path)) {
    saveRDS(pls_result, rds_path)
  }
  return(pls_result)
}


plot_pls <- function(pls) {
  summary(pls)

  plot(RMSEP(pls), legendpos = "topright", main = "RMSEP vs. Factors PLS")

  plot(pls, ncomp = 5, line = TRUE, main = "Predicted vs. Measured PLS")
}

plot_predicted_vs_measured <- function(pls, spectra_df) {
  factor <- 10
  ref_cotton <- spectra_df$reference.cotton
  len <- length(ref_cotton)
  end <- factor * len
  start <- (end - len) + 1
  predicted <- pls$validation$pred[start:end]
  model <- lm(predicted ~ ref_cotton)
  print(summary(model))
  plot(ref_cotton, predicted,
       xlab = "Measured Cotton Content [%]", ylab = "Predicted Cotton Content [%]",
       main = "Predicted Linear Model")
  abline(model, col = "black", lwd = 2)
}

plot_loading <- function(pls) {
  plot(pls, "loadings", comps = 1:3, legendpos = "topleft",
       labels = "numbers", xlab = "Wavenumber [1/cm]", lty = c(1, 3, 5), col = "black")
  abline(h = 0)

  loadings <- pls$loadings
  x <- as.numeric(names(loadings[, 1]))
  plot(x,
       loadings[, 1],
       xlim = rev(range(x)),
       ylim = c(-0.10, 0.25),
       lty = 1,
       type = "l",
       xlab = "Wavenumber [1/cm]",
       ylab = "Loading value [-]",
       lwd = 1)
  lines(x,
        loadings[, 2] + 0.05,
        lty = 3,
        type = "l",
        lwd = 1)
  lines(x,
        loadings[, 3] + 0.10,
        lty = 5,
        type = "l",
        lwd = 1)
  text(x = 2400, y = 0.01,
       labels = "Loading 1",
       col = "black", cex = 1.2)
  text(x = 2400, y = 0.01 + 0.05,
       labels = "Loading 2",
       col = "black", cex = 1.2)
  text(x = 2400, y = 0.01 + 0.10,
       labels = "Loading 3",
       col = "black", cex = 1.2)
}

#myprep <- list(prep("snv"))

# TODO doesnt work yet
preprocessing <- function(originalspectra, prep_set) {
  print("Hello world!")
  attr(originalspectra, "xaxis.values") <- as.numeric(colnames(originalspectra))
  attr(originalspectra, "xaxis.name") <- "Wavenumber"

  # apply combined methods
  pspectra <- employ.prep(prep_set, originalspectra)
  print("What aup!!")

  #par(mfrow = c(2, 1))
  mdaplot(originalspectra, type = "l", main = "Original")
  #mdaplot(pspectra, type = "l", main = "after treatment")
  return(pspectra)
}

# TODO doesnt work yet
run_mda <- function(spectra_df) {
  ref_cotton <- spectra_df$reference.cotton
  Xpv <- pcvpls(spectra_df, ref_cotton, 20, cv = list("ven", 10))
  pls_model_pcv <- pls(spectra_df, ref_cotton, 10, x.test = Xpv, y.test = ref_cotton)
  summary(pls_model_pcv)
}


library(pcv)
library(mdatools)

par(mfrow = c(1, 1))

TEMP_DIR <- "temp"
csv_path <- "input/spectra_mir_240806.csv"
spectra_df_full <- load_csv(csv_path)
spectra_df_clean <- clean_up_spectra(spectra_df_full)
plot_spectra(spectra_df_clean)
#baseline_correction_df <- stdnormalvariate(spectra_df_clean)
#plot_spectra(baseline_correction_df)

spectra_rds_path <- file.path(TEMP_DIR, "spectra_treated.RDS")
#saveRDS(baseline_correction_df, spectra_rds_path)

pls_rds_path <- file.path(TEMP_DIR, "pls.RDS")
#pls <- run_pls(spectra_df_clean, pls_rds_path)
#pls <- readRDS(pls_rds_path)

#plot_pls(pls)
#plot_predicted_vs_measured(pls, spectra_df_clean)
#plot_loading(pls)