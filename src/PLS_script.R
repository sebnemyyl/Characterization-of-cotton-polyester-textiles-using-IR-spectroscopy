# Before running script always set the work directory to top-level of the repository.
setwd("C:/Users/sebne/Documents/FHWN_Tulln/DataAnalysis/repo")
#setwd(".")

source("src/util/02_data_prep_load_CSV.R")
source("src/util/03_data_prep_limit_spectra.R")
debugSource("src/util/04_data_prep_baseline_correction.R")
source("src/util/05_data_analysis_plot_spectra.R")
source("src/util/06_model_pls.R")


par(mfrow = c(1, 1))

TEMP_DIR <- "temp"
csv_path <- "input/spectra_nir_240812.csv"
spectra_df_full <- load_csv(csv_path)

spectra_reproducibility <- spectra_df_full %>% filter(reference.pet == 70 & reference.measuring_date == 240812)

#spectra_df_clean <- clean_up_spectra(spectra_df_full)
plot_spectra(spectra_reproducibility)

#baseline_correction_df <- stdnormalvariate(spectra_reproducibility)
#baseline_correction_df <- detrend(spectra_reproducibility, 2)
#baseline_correction_df <- als(spectra_reproducibility)
baseline_correction_df <- fillpeaks(spectra_reproducibility)

baseline_csv_path <- file.path(TEMP_DIR, "spectra_treated_fillpeaks_30_cotton_nir.csv")
save_csv(baseline_correction_df, baseline_csv_path, 5)
baseline_correction_df_read <- load_saved_csv(baseline_csv_path)
#baseline_correction_df <- load_csv(baseline_csv_path)
plot_spectra(baseline_correction_df_read)

#spectra_rds_path <- file.path(TEMP_DIR, "spectra_treated.RDS")
#saveRDS(baseline_correction_df, spectra_rds_path)

#pls_rds_path <- file.path(TEMP_DIR, "pls.RDS")
#pls <- run_pls(spectra_df_clean, pls_rds_path)
#pls <- readRDS(pls_rds_path)

#plot_pls(pls)
#plot_predicted_vs_measured(pls, spectra_df_clean)
#plot_loading(pls)