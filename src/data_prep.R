library(pls)
library(plotly)
library(dplyr)
library(prospectr)

load_csv <- function(csv_path) {
  # Load CSV file
  csv_raw <- read.csv(csv_path, header = TRUE, sep = ";", dec = ",", row.names = 1)
  # TODO maybe we have to remove two first columns like in original
  # Extract spectra and features
  spectra_raw <- csv_raw[, -c(1763:1768)]
  features <- csv_raw[, c(1763:1768)]
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
  spectra <- standardNormalVariate(spectra)
  # Outlier Removal (4th sample is the outlier for here and mdatools)
  # TODO check what this 4th sample means..
  spectra <- spectra[-c(4), ]
  reference <- reference[-c(4), ]
  # Spektralen Bereich einschraenken
  spectra[, c(1:200, 610:1140, 1715:1760)] <- 0
  # Recreate data frame
  clean_df <- data.frame(reference, spectra)
  return(clean_df)
}

run_pls <- function(spectra_df) {
  options(digits = 4)
  pls.options(plsalg = "oscorespls")
  ref <- spectra_df$reference.cotton
  spectra <- spectra_df_clean %>% select(starts_with("spectra"))
  spectra_matrix <- data.matrix(spectra)
  pls_result <- plsr(ref ~ spectra_matrix, data = spectra_df, ncomp = 10, val = "LOO")
  summary(pls_result)
}

setwd(".")
csv_path <- "input/spectra_mir_240806.csv"
spectra_df_full <- load_csv(csv_path)
spectra_df_clean <- clean_up_spectra(spectra_df_full)
#print(spectra_df_clean[20])
#print(spectra_df_full[20])
run_pls(spectra_df_clean)

# TODO doesnt work well (but in original it is also bad)
#matplot(names(spectra), t(spectra), lty = 1, type = "l", ylab = "Absorbance")