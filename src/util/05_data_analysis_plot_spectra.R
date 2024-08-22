library(pls)
library(plotly)

plot_spectra <- function(spectra_df) {
  spectra <- spectra_df %>% dplyr::select(starts_with("spectra"))
  wave_numbers <- as.integer(sub("spectra.", "", names(spectra)))
  matplot(wave_numbers, t(spectra), lty = 1, type = "l", ylab = "Absorbance")
}