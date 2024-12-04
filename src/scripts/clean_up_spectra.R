# Before running script always set the work directory to top-level of the repository.
#setwd(".")
library(stringr)

source("src/util/02_data_prep_load_CSV.R")
source("src/util/03_data_prep_limit_spectra.R")
source("src/util/05_data_analysis_plot_spectra.R")
source("src/util/04_data_prep_baseline_correction.R")

input_path <- "input/raw_csv/spectra_nir_241011.csv"
output_path <- "input/clean_csv/spectra_nir_all.csv"
type <- "nir"

spectra_df_full <- load_csv(input_path)
spectra_df_clean <- clean_up_spectra(spectra_df_full, type, remove_waterband = TRUE)
## Filter as needed (cotton content, date etc)
#spectra_df_clean <- spectra_df_clean %>% filter(reference.measuring_date == 240827 | reference.measuring_date == 240812 ) # NIR resampling

spectra_df_clean$reference.pet<-as.numeric(spectra_df_clean$reference.pet)
spectra_df_clean$reference.cotton<-as.numeric(spectra_df_clean$reference.cotton)

save_csv(spectra_df_clean, output_path)



