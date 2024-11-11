# Before running script always set the work directory to top-level of the repository.
#setwd(".")
library(stringr)

source("src/util/02_data_prep_load_CSV.R")
source("src/util/03_data_prep_limit_spectra.R")
source("src/util/05_data_analysis_plot_spectra.R")

input_path <- "input/raw_csv/spectra_nir_241011.csv"
output_path <- "input/clean_csv/spectra_nir_resampling.csv"
type <- "nir"

spectra_df_full <- load_csv(input_path)
spectra_df_clean <- clean_up_spectra(spectra_df_full, type, remove_waterband = FALSE)
## Filter as needed (cotton content, date etc)
spectra_df_clean <- spectra_df_clean %>% filter((reference.measuring_date == 240807)|(reference.measuring_date == 240812)) # NIR resampling
save_csv(spectra_df_clean, output_path)