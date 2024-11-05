# Before running script always set the work directory to top-level of the repository.
#setwd(".")
library(stringr)

source("src/util/02_data_prep_load_CSV.R")
source("src/util/03_data_prep_limit_spectra.R")
source("src/util/04_data_prep_baseline_correction.R")
source("src/util/05_data_analysis_plot_spectra.R")

## TODO water band configurable
## TODO filter per cotton content

save_csv_with_baseline_corr <- function(spectra_df_clean, output_dir, type = "nir", baseline_corr = "snv") {
  if (baseline_corr == "snv") {
    baseline_correction_df <- stdnormalvariate(spectra_df_clean)
  } else if (baseline_corr == "detrend") {
    baseline_correction_df <- detrend(spectra_df_clean, 2)
  } else if (baseline_corr == "als") {
    baseline_correction_df <- als(spectra_df_clean)
  } else if (baseline_corr == "fillpeaks") {
    baseline_correction_df <- fillpeaks(spectra_df_clean)
  } else if (baseline_corr == "msc") {
    baseline_correction_df <- msc(spectra_df_clean)
  } else {
    stop(str_glue("Baseline correction type {baseline_corr} not supported!"))
  }
  #plot_spectra(baseline_correction_df)
  file_name <- str_glue("spectra_treated_{type}_{baseline_corr}.csv")
  baseline_csv_path <- file.path(output_dir, file_name)
  save_csv(baseline_correction_df, baseline_csv_path, 5)
  print(str_glue("CSV file saved to {baseline_csv_path}"))
}


csv_path <- "input/spectra_nir_240827.csv"
type <- "nir"
spectra_df_full <- load_csv(csv_path)
spectra_df_clean <- clean_up_spectra(spectra_df_full, type, remove_waterband = FALSE)

output_dir <- "temp"
baseline_corr_types <- list("snv", "detrend", "als", "fillpeaks", "msc")
for (baseline_corr in baseline_corr_types) {
  save_csv_with_baseline_corr(spectra_df_clean, output_dir, type, baseline_corr)
}