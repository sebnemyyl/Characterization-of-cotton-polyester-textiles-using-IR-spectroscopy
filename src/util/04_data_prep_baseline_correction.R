# Load packages
Packages <- c("plyr","dplyr","IDPmisc","prospectr","dendextend","baseline",
              "pls","plotrix","knitr","ggplot2","gridExtra","ggpubr","ggpmisc",
              "ChemoSpec", "matrixStats", "stringr", "caret", 
              "ROCR", "binom", "cvAUC","reshape2")

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
detrend <- function(spectra_df, polynomial_order) {
  spectra_matrix <- data.matrix(spectra_df %>% dplyr::select(starts_with("spectra")))
  wavenumbers <- as.numeric(sub("spectra.", "", colnames(spectra_matrix)))
  spectra_detrend <- detrend(spectra_matrix, wav = wavenumbers, p = polynomial_order)
  return(spectra_detrend)
}


# Asymmetric Least Squares
als <- function(spectra_df, lambda, p , maxit) {
  spectra_matrix <- data.matrix(spectra_df %>% dplyr::select(starts_with("spectra")))
  spectra_als <- baseline(spectra = as.matrix(spectra_matrix), 
                        method = "als",  
                        lambda = 3,         # 2nd derivative constraint
                        p = 0.05,           # Weighting of positive residuals
                        maxit = 20)         # Maximum number of iterations
  return(spectra_als)
}


# FillPeaks                                                          
fillpeaks <- function(spectra_df, lambda, hwi , it, int) {
  spectra_fillpeaks <- baseline(spectra = data.matrix(spectra_df %>% dplyr::select(starts_with("spectra"))), 
                        method = "fillPeaks",  
                        lambda=1,        # 2nd derivative penalty for primary smoothing
                        hwi=10,           # Half width of local windows
                        it=6,            # Number of iterations in suppression loop
                        int=400)         # Number of buckets to divide spectra into
  return(spectra_fillpeaks)
}