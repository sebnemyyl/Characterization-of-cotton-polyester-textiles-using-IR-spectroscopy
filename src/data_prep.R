library(pls)
library(plotly)
library(dplyr)
library(prospectr)

load_csv <- function(csv_path) {
  # Load CSV file
  csv_raw <- read.csv(csv_path, header = TRUE, sep = ";", dec = ",", row.names = 1)
  # Extract spectra and features
  feature_names <- c("pet", "cotton", "specimen", "area", "spot", "measuring_date")
  feature_col <- which(names(csv_raw) %in% feature_names)
  spectra_raw <- csv_raw[, -feature_col]
  features <- csv_raw[, feature_col]
  # Fix wave number naming
  names(spectra_raw) <- as.numeric(sub("X", "0", names(spectra_raw)))
  # Convert spectra to matrix
  spectra_matrix <- data.matrix(spectra_raw)
  # Create data frame (full spectral area and with outliers)
  spectra_df <- data.frame(reference = features, spectra = spectra_matrix)
  return(spectra_df)
}

clean_up_spectra <- function(spectra_df) {
  reference <- spectra_df %>% select(starts_with("reference"))
  spectra <- spectra_df %>% select(starts_with("spectra"))
  # SNV
  # TODO make SNV configurable
  spectra <- standardNormalVariate(spectra)
  # Outlier Removal (4th sample is the outlier for here and mdatools)
  # TODO check what this 4th sample means.. -> Extract function and make it configurable
  spectra <- spectra[-c(4), ]
  reference <- reference[-c(4), ]
  # Limit spectral area
  # TODO make it configurable for NIR
  spectra[, c(1:200, 610:1140, 1715:1762)] <- 0
  # Recreate data frame
  clean_df <- data.frame(reference, spectra)
  return(clean_df)
}

run_pls <- function(spectra_df, rds_path) {
  options(digits = 4)
  pls.options(plsalg = "oscorespls")
  ref_cotton <- spectra_df$reference.cotton
  spectra <- spectra_df_clean %>% select(starts_with("spectra"))
  names(spectra) <- as.numeric(sub("spectra.", "", names(spectra)))
  spectra_matrix <- data.matrix(spectra)
  pls_result <- plsr(ref_cotton ~ spectra_matrix, data = spectra_df, ncomp = 10, val = "LOO")
  if (!missing(rds_path)) {
    saveRDS(pls_result, rds_path)
  }
  return(pls_result)
}

plot_spectra <- function(spectra_df) {
  spectra <- spectra_df %>% select(starts_with("spectra"))
  wave_numbers <- as.integer(sub("spectra.", "", names(spectra)))
  matplot(wave_numbers, t(spectra), lty = 1, type = "l", ylab = "Absorbance")
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
       xlab = "Measured Cotton Content [%]", ylab = "Predicted Cotton Content [%]")
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

TEMP_DIR <- "temp"
setwd(".")
csv_path <- "input/spectra_mir_240806.csv"
spectra_df_full <- load_csv(csv_path)
spectra_df_clean <- clean_up_spectra(spectra_df_full)
#plot_spectra(spectra_df_clean)

pls_rds_path <- file.path(TEMP_DIR, "pls.RDS")
#pls <- run_pls(spectra_df_clean, pls_rds_path)
pls <- readRDS(pls_rds_path)
#plot_pls(pls)
#plot_predicted_vs_measured(pls, spectra_df_clean)
plot_loading(pls)