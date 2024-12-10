source("src/util/01_raw_data_prep_convert_TXT_to_CSV.R")
source("src/util/02_data_prep_load_CSV.R")
source("src/util/06_model_pls.R")
library(jsonlite)

input_dir <- "temp/test"
output_path <- "temp/pls_metrics.json"
results <- list()
i <- 1
csv_files <- list.files(path = input_dir, recursive = TRUE, include.dirs = TRUE, pattern = ".csv")
for (csv_file in csv_files) {
  splitted <- split_file_name(csv_file)
  baseline_corr <- tail(splitted, n = 1)
  csv_path <- file.path(input_dir, csv_file)
  spectra <- load_saved_csv(csv_path)
  metrics <- perform_pls_regression(spectra)
  metrics$baseline_corr <- baseline_corr
  print(metrics)
  results[[i]] <- metrics
  i <- i + 1
}
jsonlite::write_json(results, output_path, pretty = TRUE, auto_unbox = TRUE)
cat("Metrics have been saved to", output_path, "\n")




