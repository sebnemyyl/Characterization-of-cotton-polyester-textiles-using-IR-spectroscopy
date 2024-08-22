library(dplyr)

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
  reference <- spectra_df %>% dplyr::select(starts_with("reference"))
  spectra <- spectra_df %>% dplyr::select(starts_with("spectra"))
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
