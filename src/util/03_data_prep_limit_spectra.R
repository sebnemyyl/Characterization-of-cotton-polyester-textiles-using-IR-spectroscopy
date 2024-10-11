library(dplyr)

clean_up_spectra <- function(spectra_df, type = "nir") {
  reference <- spectra_df %>% dplyr::select(starts_with("reference"))
  spectra <- spectra_df %>% dplyr::select(starts_with("spectra"))
  # TODO check what this 4th sample means.. -> Extract function and make it configurable
  spectra <- spectra[-c(4), ]
  reference <- reference[-c(4), ]
  wavenumbers <- as.numeric(sub("spectra\\.", "", colnames(spectra)))
  # Limit spectral area
  # TODO make it configurable for water band
  if (type == "nir") {
    columns_to_keep <- (wavenumbers >= 3787.81 & wavenumbers <= 7876.49)
  } else if (type == "mir") {
    columns_to_keep <- (wavenumbers >= 1801.44478 & wavenumbers <= 2798.60426)
  } else {
    stop("This type is not supported")
  }
  filtered_spectra <- spectra[, columns_to_keep]
  # Recreate data frame
  clean_df <- data.frame(reference, filtered_spectra)
  return(clean_df)
}