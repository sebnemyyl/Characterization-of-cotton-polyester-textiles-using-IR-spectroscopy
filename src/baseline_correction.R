# Before running script always set the work directory to top-level of the repository.
setwd("/Users/andi/Documents/Coding/spectroscopy/")
#setwd(".")

source("src/util/02_data_prep_load_CSV.R")
source("src/util/03_data_prep_limit_spectra.R")
source("src/util/04_data_prep_baseline_correction.R")
source("src/util/05_data_analysis_plot_spectra.R")


## TODO make method to get CSV file with baseline correction
## TODO make it possible to run multiple times (with different baseline corrections)
## TODO water band configurable
## TODO filter per cotton content

TEMP_DIR <- "temp"
csv_path1 <- "input/spectra_nir_240827.csv"

spectra_df_full1 <- load_csv(csv_path1)
spectra_df_clean <- clean_up_spectra(spectra_df_full1)

## Baseline Correction
baseline_correction_df_nir <- stdnormalvariate(spectra_df_clean)
#baseline_correction_df <- detrend(spectra_reproducibility, 2)
#baseline_correction_df <- als(spectra_reproducibility)
#baseline_correction_df <- fillpeaks(spectra_reproducibility)
#baseline_correction_df <- msc(spectra_reproducibility)

baseline_csv_path <- file.path(TEMP_DIR, "spectra_treated_snv_nir.csv")
save_csv(baseline_correction_df_nir, baseline_csv_path, 5)
plot_spectra(baseline_correction_df_nir)

print("CSV saved!")