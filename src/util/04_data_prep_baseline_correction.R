# Load packages
Packages <- c("plyr","dplyr","IDPmisc","prospectr","dendextend","baseline",
              "pls","plotrix","knitr","ggplot2", "gridExtra",
              "ChemoSpec", "matrixStats", "stringr", "MASS", "caret", "reshape2")

for (p in Packages) {
  library(p, character.only = TRUE)
}

# Standard Normal Variate
stdnormalvariate <- function(spectra_df) {
  spectra_matrix <- data.matrix(spectra_df %>% dplyr::select(starts_with("spectra")))
  reference_matrix <- data.matrix(spectra_df %>% dplyr::select(starts_with("reference")))
  spectra_SNV <- standardNormalVariate(spectra_matrix)
  SNV_df <- data.frame(reference_matrix, spectra_SNV)
  return(SNV_df)
}


# prospectr Detrend
detrend <- function(spectra_df, polynomial_order = 3) {
  spectra_matrix <- data.matrix(spectra_df %>% dplyr::select(starts_with("spectra")))
  reference_matrix <- data.matrix(spectra_df %>% dplyr::select(starts_with("reference")))
  wavenumbers <- as.numeric(sub("spectra.", "", colnames(spectra_matrix)))
  spectra_detrend <- prospectr::detrend(spectra_matrix, wav = wavenumbers, p = polynomial_order)
  detrend_df <- data.frame(reference_matrix, spectra_detrend)
  return(detrend_df)
}


# Asymmetric Least Squares
als <- function(spectra_df, lambda_als = 2, p_als = 0.05, maxit_als = 20) {
  spectra_matrix <- data.matrix(spectra_df %>% dplyr::select(starts_with("spectra")))
  reference_matrix <- data.matrix(spectra_df %>% dplyr::select(starts_with("reference")))
  spectra_als <- baseline.als(spectra = spectra_matrix, 
                        lambda = lambda_als,         # 2nd derivative constraint
                        p = p_als,           # Weighting of positive residuals
                        maxit = maxit_als)         # Maximum number of iterations
  als_df <- data.frame(reference_matrix, spectra_als$corrected)
  colnames(als_df) <- colnames(spectra_df)
  return(als_df)
}


# FillPeaks                                                          
fillpeaks <- function(spectra_df, lambda_fp = 1, hwi_fp = 10 , it_fp = 6, int_fp = 400) {
  spectra_matrix <- data.matrix(spectra_df %>% dplyr::select(starts_with("spectra")))
  reference_matrix <- data.matrix(spectra_df %>% dplyr::select(starts_with("reference")))
  spectra_fillpeaks <- baseline.fillPeaks(spectra = spectra_matrix, 
                        lambda = lambda_fp,        # 2nd derivative penalty for primary smoothing
                        hwi = hwi_fp,           # Half width of local windows
                        it = it_fp,            # Number of iterations in suppression loop
                        int = int_fp)         # Number of buckets to divide spectra into
  fillpeaks_df <- data.frame(reference_matrix, spectra_fillpeaks$corrected)
  colnames(fillpeaks_df) <- colnames(spectra_df)
  return(fillpeaks_df)
}


# Multiplicative Scatter Correction                                                          
msc <- function(spectra_df) {
  spectra_matrix <- data.matrix(spectra_df %>% dplyr::select(starts_with("spectra")))
  reference_matrix <- data.matrix(spectra_df %>% dplyr::select(starts_with("reference")))
  spectra_msc <- prospectr::msc(spectra_matrix, ref_spectrum = colMeans(spectra_matrix))
  msc_df <- data.frame(reference_matrix, spectra_msc)
  colnames(msc_df) <- colnames(spectra_df)
  return(msc_df)
}

# Savitzky-Golay Smoothing
savitzky_golay <- function(spectra_df, m = 0, p = 3, w =21) {
  spectra_matrix <- data.matrix(spectra_df %>% dplyr::select(starts_with("spectra")))
  reference_matrix <- data.matrix(spectra_df %>% dplyr::select(starts_with("reference")))
  spectra_SG <- savitzkyGolay(spectra_matrix, m, p, w)
  savgol_df <- data.frame(reference_matrix, spectra_SG)
  return(savgol_df)
}