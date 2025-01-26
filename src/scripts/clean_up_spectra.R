# Before running script always set the work directory to top-level of the repository.
#setwd(".")
library(stringr)

source("src/util/02_data_prep_load_CSV.R")
source("src/util/03_data_prep_limit_spectra.R")
source("src/util/05_data_analysis_plot_spectra.R")
source("src/util/04_data_prep_baseline_correction.R")

input_path <- "input/raw_csv/spectra_nir_250124.csv"
output_path <- "input/clean_csv/spectra_nir_balanced.csv"
type <- "nir"

spectra_df_full <- load_csv(input_path)
spectra_df_clean <- clean_up_spectra(spectra_df_full, type, remove_waterband = TRUE)

## Filter as needed (cotton content, date etc)
# 17% cotton has been revised to 16.5% and remeasured.  39.47% is also remasured in 250122 and it was redundant.
#spectra_df_clean <- spectra_df_clean %>% filter(!(reference.cotton == 50 & reference.specimen >= 5))
spectra_df_clean <- spectra_df_clean %>% filter(!(reference.cotton == 17) & 
                                                !(reference.cotton == 39.47 & reference.measuring_date == 250122) &
                                                  !(reference.cotton == 30 & reference.specimen == 3)&
                                                  !(reference.cotton == 50 & reference.specimen > 5))

spectra_df_clean$reference.pet<-as.numeric(spectra_df_clean$reference.pet)
spectra_df_clean$reference.cotton<-as.numeric(spectra_df_clean$reference.cotton)

save_csv(spectra_df_clean, output_path)



