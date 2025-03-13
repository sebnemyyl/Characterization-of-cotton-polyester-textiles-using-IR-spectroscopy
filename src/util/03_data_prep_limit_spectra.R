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

fix_cotton_content <- function(spectra_df_clean) {
  cotton_frame <- data.frame(
    old_cotton = c(16.5, 35, 25, 23, 30, 45),
    new_cotton = c(27.01, 40.93, 34.01, 31.57, 38.25, 47.23)
  )
  print(cotton_frame)
  for (i in seq_len(nrow(cotton_frame))) {
    old_cotton <- cotton_frame[i, ]$old_cotton
    new_cotton <- cotton_frame[i, ]$new_cotton
    selected_rows <- which(spectra_df_clean$reference.cotton == old_cotton)
    spectra_df_clean$reference.cotton[selected_rows] <- new_cotton
    spectra_df_clean$reference.pet[selected_rows] <- 100 - new_cotton
    msg <- str_glue("{length(selected_rows)} rows with cotton content {old_cotton} were changed to {new_cotton}") # nolint
    print(msg)
  }
  return(spectra_df_clean)
}