library(pls)
library(plotly)
library(dplyr)

plot_spectra <- function(spectra_df, lim = c(8000, 4000), plt_title = "") {
  spectra <- spectra_df %>% dplyr::select(starts_with("spectra"))
  
  # Reverse wave numbers (decreasing order of wavenumbers)
  wave_numbers <- as.integer(sub("spectra.", "", names(spectra)))
  wave_numbers <- rev(wave_numbers)
  
  # Reverse the spectra matrix to match the reversed wave numbers
  spectra <- spectra[, rev(seq_along(wave_numbers))]
  
  color_palette <- rainbow(nrow(spectra))
  
  # Specify xlim so that x axis is also reversed
  matplot(wave_numbers, t(spectra), lty = 1, type = "l", xlim = lim ,
          ylab = "Absorbance (%)", xlab = "Wavenumber (1/cm)", main = plt_title, col = color_palette)
}

