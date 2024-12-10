source("src/util/02_data_prep_load_CSV.R")
source("src/util/06_model_pls.R")
library(jsonlite)
library(dplyr)
library(microbenchmark)



perform_pls_regression <- function(filtered_data_nir,filtered_data_mir, json_path) {
  #### Conventional PLS
  pls_rds_path <- file.path(TEMP_DIR, "pls.RDS")
  
  train_time_pls <- system.time({
    pls <- run_pls(filtered_data_nir, pls_rds_path)
  })[["elapsed"]]
  
  #pred_time_pls <- microbenchmark(predict(pls, filtered_data_nir), times = 10)$time / 1e9
  
  msep_pls <- MSEP(pls)
  adjCV_pls <- msep_pls$val[2, , 11]
  rmsep_pls <- sqrt(adjCV_pls)
  r2_pls <- R2(pls)
  
  pls_metrics <- list(
    model_name = "pls",
    baseline_corr = bsln,
    RMSEP = rmsep_pls,
    R2 = r2_pls$val[1, , 11],
    training_time = train_time_pls
    #prediction_time = mean(pred_time_pls)
  )
  
  plot_pls(pls)
  plot_predicted_vs_measured(pls, filtered_data_nir)
  plot_loading(pls)
  plot(pls$residuals)
  plot(RMSEP(pls), legendpos = "topright",main="RMSEP vs. Faktoren PLS_101")
  
  
  #### Multiblock PLS
  multiblock_pls_rds_path <- file.path(TEMP_DIR, "mbpls.RDS")
  train_time_multiblock <- system.time({
    multiblock_pls <- run_multiblock_pls(filtered_data_nir, filtered_data_mir, multiblock_pls_rds_path)
  })[["elapsed"]]
  
  #pred_time_multiblock <- microbenchmark(predict(multiblock_pls, filtered_data_nir), times = 10)$time / 1e9
  
  msep_multiblock <- multiblock::MSEP(multiblock_pls)
  adjCV_multiblock  <- msep_multiblock$val[2, , 11]
  rmsep_multiblock <- sqrt(adjCV_multiblock )
  r2_multiblock <- R2(multiblock_pls)
  
  #plot_pls(multiblock_pls)
  
  multiblock_metrics <- list(
    model_name = "multiblock_pls",
    baseline_corr = bsln,
    RMSEP = rmsep_multiblock,
    R2 = r2_multiblock$val[1, , 11],
    training_time = train_time_multiblock
    #prediction_time = mean(pred_time_multiblock)
  )
  
  results <- list(
    models = list(pls_metrics, multiblock_metrics)
  )
  
  output_path <- "pls_msc_metrics.json"
  #jsonlite::write_json(results, output_path, pretty = TRUE)
  
  cat("Metrics have been saved to", output_path, "\n")
}


  
filtered_data_nir <- load_saved_csv("temp/spectra_treated/nir/spectra_nir_regression_msc.csv")
filtered_data_mir <- load_saved_csv("temp/spectra_treated/mir/spectra_mir_regression_msc.csv")

filtered_data_nir <- filtered_data_nir[order(filtered_data_nir$reference.cotton),]
filtered_data_mir <- filtered_data_mir[order(filtered_data_mir$reference.cotton),]

TEMP_DIR <- "temp/"
bsln="msc"
perform_pls_regression(filtered_data_nir,filtered_data_mir, json_path=TEMP_DIR)
  
