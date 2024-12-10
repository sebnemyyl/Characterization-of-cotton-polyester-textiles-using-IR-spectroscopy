source("src/util/01_raw_data_prep_convert_TXT_to_CSV.R")
source("src/util/02_data_prep_load_CSV.R")
source("src/util/06_model_pls.R")
library(jsonlite)

output_dir <- "temp/multiblock"
# TODO Update paths
nir_path <- "temp/nir/spectra_nir_msc.csv"
mir_path <- "temp/nir/spectra_nir_msc.csv" #"temp/spectra_treated/mir/spectra_mir_regression_msc.csv")
filtered_data_nir <- load_saved_csv(nir_path)
filtered_data_mir <- load_saved_csv(mir_path)
filtered_data_nir <- filtered_data_nir[order(filtered_data_nir$reference.cotton),]
filtered_data_mir <- filtered_data_mir[order(filtered_data_mir$reference.cotton),]

metrics <- perform_multiblock_pls_regression(filtered_data_nir, filtered_data_mir)

splitted <- split_file_name(nir_path)
baseline_corr <- tail(splitted, n = 1)
metrics$baseline_corr <- baseline_corr

json_file_name <- paste("multiblock_output_", baseline_corr, ".json", sep = "")
json_path <- file.path(output_dir, json_file_name)
jsonlite::write_json(metrics, json_path, pretty = TRUE, auto_unbox = TRUE)
cat("Metrics have been saved to", json_path, "\n")