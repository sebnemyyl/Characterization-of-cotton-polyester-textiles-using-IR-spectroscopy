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

save_csv <- function(spectra_df, csv_path, digits = 5) {
  df_decimals_truncated <- format(spectra_df, digits)
  write.csv(df_decimals_truncated, file=csv_path)
}

load_saved_csv <- function(csv_path) {
  return(read.csv(csv_path, row.names = 1))
}