library(dplyr)

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