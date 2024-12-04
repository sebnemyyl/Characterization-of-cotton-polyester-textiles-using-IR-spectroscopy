library(dplyr)
library(stringr)

clean_up_spectra <- function(spectra_df, type = "nir", remove_waterband = FALSE) {
  reference <- spectra_df %>% dplyr::select(starts_with("reference"))
  spectra <- spectra_df %>% dplyr::select(starts_with("spectra"))

  wavenumbers <- as.numeric(sub("spectra\\.", "", colnames(spectra)))
  # Limit spectral area
  if (type == "nir") {
    columns_to_keep <- (wavenumbers >= 3787.81 & wavenumbers <= 7876.49)
  } else if (type == "mir") {
    columns_to_keep <- (wavenumbers >= 1801.44478 & wavenumbers <= 2798.60426)
  } else {
    stop(str_glue("Type {type} is not supported"))
  }
  if (remove_waterband) {
    if (type == "nir") {
      water_band <- (wavenumbers >= 4798.40825 & wavenumbers <= 5396.28066)
    } else if (type == "mir") {
      water_band <- (wavenumbers >= 1600.85564 & wavenumbers <= 1651.00293)
    }
    columns_to_keep <- columns_to_keep & !water_band
  }
  filtered_spectra <- spectra[, columns_to_keep]
  # Recreate data frame
  clean_df <- data.frame(reference, filtered_spectra)
  return(clean_df)
}