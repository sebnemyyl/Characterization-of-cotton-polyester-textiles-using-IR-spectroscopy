#TODO improve this, make configurable
#setwd("C:/Users/sebne/Documents/FHWN_Tulln/DataAnalysis/repo")
setwd(".")

source("src/util/02_data_prep_load_CSV.R")
source("src/util/03_data_prep_limit_spectra.R")
#source("src/util/04_data_prep_baseline_correction.R")
source("src/util/05_plot_spectra.R")
source("src/util/06_model_pls.R")


par(mfrow = c(1, 1))

TEMP_DIR <- "temp"
csv_path <- "input/spectra_mir_240806.csv"
spectra_df_full <- load_csv(csv_path)
spectra_df_clean <- clean_up_spectra(spectra_df_full)
#plot_spectra(spectra_df_clean)
#baseline_correction_df <- stdnormalvariate(spectra_df_clean)
#plot_spectra(baseline_correction_df)

spectra_rds_path <- file.path(TEMP_DIR, "spectra_treated.RDS")
#saveRDS(baseline_correction_df, spectra_rds_path)

pls_rds_path <- file.path(TEMP_DIR, "pls.RDS")
pls <- run_pls(spectra_df_clean, pls_rds_path)
#pls <- readRDS(pls_rds_path)

plot_pls(pls)
#plot_predicted_vs_measured(pls, spectra_df_clean)
#plot_loading(pls)