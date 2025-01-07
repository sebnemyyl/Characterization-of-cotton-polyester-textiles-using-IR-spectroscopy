# Before running script always set the work directory to top-level of the repository.
#setwd(".")
library(stringr)

source("src/util/02_data_prep_load_CSV.R")
source("src/util/04_data_prep_baseline_correction.R")
source("src/util/05_data_analysis_plot_spectra.R")

save_csv_with_baseline_corr <- function(spectra_df_clean, output_dir, file_naming = "", baseline_corr = "snv") {
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
  } else if (baseline_corr == "savgol") {
    baseline_correction_df <- savitzky_golay(spectra_df_clean, 2, 3, 11)
  } else {
    stop(str_glue("Baseline correction type {baseline_corr} not supported!"))
  }
  file_name <- str_glue("spectra_{file_naming}_{baseline_corr}.csv")
  baseline_csv_path <- file.path(output_dir, file_name)
  save_csv(baseline_correction_df, baseline_csv_path)
  print(str_glue("CSV file saved to {baseline_csv_path}"))
}


csv_path <- "input/clean_csv/spectra_nir_all.csv"
file_naming <- "nir_regression"
spectra_df <- load_saved_csv(csv_path)

output_dir <- "temp/spectra_treated/nir"
baseline_corr_types <- list("snv", "detrend", "als", "fillpeaks", "msc", "savgol")
for (baseline_corr in baseline_corr_types) {
  save_csv_with_baseline_corr(spectra_df, output_dir, file_naming, baseline_corr)
}

# csv_path_snv <- "temp/spectra_treated/nir_new/spectra_nir_regression_snv.csv"
# snv_df <- load_saved_csv(csv_path)
# baseline_corr_types <- list("detrend", "als", "fillpeaks", "msc", "savgol")
# for (baseline_corr in baseline_corr_types) {
#   save_csv_with_baseline_corr(spectra_df, output_dir, file_naming, baseline_corr)
# }
# 
# csv_path <- "temp/spectra_treated/nir_new/spectra_nir_regression_detrend.csv"
# spectra_df <- load_saved_csv(csv_path)
# plot_spectra(spectra_df)






