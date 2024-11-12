# Before running script always set the work directory to top-level of the repository.
#setwd("C:/Users/sebne/Documents/FHWN_Tulln/DataAnalysis/repo")
setwd(".")

source("src/util/02_data_prep_load_CSV.R")
source("src/util/03_data_prep_limit_spectra.R")
source("src/util/04_data_prep_baseline_correction.R")
source("src/util/05_data_analysis_plot_spectra.R")
source("src/util/06_model_pls.R")


par(mfrow = c(1, 1))

TEMP_DIR <- "temp"
csv_path1 <- "input/raw_csv/spectra_nir_240827.csv"
csv_path2 <- "input/raw_csv/spectra_mir_240814.csv"

spectra_df_full1 <- load_csv(csv_path1)
spectra_df_full2 <- load_csv(csv_path2)

spectra_nir <- spectra_df_full1 #%>% filter((reference.measuring_date == 240506)|(reference.measuring_date == 240417))
spectra_mir <- spectra_df_full2 #%>% filter(( reference.measuring_date == 240404)|(reference.measuring_date == 240814))

#spectra_df_clean <- clean_up_spectra(spectra_df_full)
#plot_spectra(spectra_mir)

## Baseline Correction
baseline_correction_df_nir <- stdnormalvariate(spectra_nir)
baseline_correction_df_mir <- stdnormalvariate(spectra_mir)



spectra_columns <- baseline_correction_df_mir %>% dplyr::select(starts_with("spectra"))
wavenumbers <- as.numeric(sub("spectra\\.", "", colnames(spectra_columns)))
# Identify the columns to keep, i.e., wavenumbers not in the range 1801.44478 to 2798.60426
columns_to_keep <- !((wavenumbers >= 1801.44478 & wavenumbers <= 2798.60426)|(wavenumbers >= 3701.25539 & wavenumbers <= 601.76742))

filtered_spectra <- spectra_columns[, columns_to_keep]
filtered_data_mir <- bind_cols(baseline_correction_df_mir %>% dplyr::select(-starts_with("spectra")), filtered_spectra)



spectra_columns <- baseline_correction_df_nir %>% dplyr::select(starts_with("spectra"))
wavenumbers <- as.numeric(sub("spectra\\.", "", colnames(spectra_columns)))
# Identify the columns to keep, i.e., wavenumbers not in the range 1801.44478 to 2798.60426
columns_to_keep <- (wavenumbers >= 3787.81 & wavenumbers <= 7876.49)

filtered_spectra <- spectra_columns[, columns_to_keep]
filtered_data_nir <- bind_cols(baseline_correction_df_nir %>% dplyr::select(-starts_with("spectra")), filtered_spectra)


#baseline_correction_df <- detrend(spectra_reproducibility, 2)
#baseline_correction_df <- als(spectra_reproducibility)
#baseline_correction_df <- fillpeaks(spectra_reproducibility)
#baseline_correction_df <- msc(spectra_reproducibility)

baseline_csv_path <- file.path(TEMP_DIR, "spectra_treated_snv_nir.csv")
save_csv(filtered_data_nir, baseline_csv_path, 5)
#baseline_correction_df_read <- load_saved_csv(baseline_csv_path)
#baseline_correction_df <- load_csv(baseline_csv_path)
#plot_spectra(baseline_correction_df_read)

#spectra_rds_path <- file.path(TEMP_DIR, "spectra_treated.RDS")
#saveRDS(baseline_correction_df, spectra_rds_path)



filtered_data_nir <- filtered_data_nir[order(filtered_data_nir$reference.cotton),]
filtered_data_mir <- filtered_data_mir[order(filtered_data_mir$reference.cotton),]

baseline_correction_df_nir_ <- filtered_data_nir %>%
  rename_with(~ paste0("NIR_", .), starts_with("spectra"))

baseline_correction_df_mir_ <- filtered_data_mir %>%
  rename_with(~ paste0("MIR_", .), starts_with("spectra"))

baseline_correction_df_nir_$num <- sequence(rle(as.character(baseline_correction_df_nir_$reference.cotton))$lengths)
baseline_correction_df_mir_$num <- sequence(rle(as.character(baseline_correction_df_mir_$reference.cotton))$lengths)

joined_data <- inner_join(baseline_correction_df_mir_, baseline_correction_df_nir_, by = c("num", "reference.cotton"))


nir_reference_data <- joined_data %>% dplyr::select(-starts_with("MIR"))

## Conventional PLS
pls_rds_path <- file.path(TEMP_DIR, "pls.RDS")
pls <- run_pls(filtered_data_nir, pls_rds_path)

#pls <- readRDS(pls_rds_path)

plot_pls(pls)
plot_predicted_vs_measured(pls, filtered_data_nir)
plot_loading(pls)
plot(pls$residuals)
plot(RMSEP(pls), legendpos = "topright",main="RMSEP vs. Faktoren PLS_101")

## Multiblock PLS
pls_rds_path <- file.path(TEMP_DIR, "mbpls.RDS")
multiblock_pls <- run_multiblock_pls(filtered_data_nir,filtered_data_mir,pls_rds_path)
plot(multiblock_pls$residuals)
plot_multiblockpls_predicted_vs_measured(multiblock_pls, joined_data )
plot(MSEP(pls), legendpos = "topright",main="RMSEP PLS")

#loadingplot(pls, block = 1, labels = "names", scatter = TRUE)
#scoreplot(pls, labels = "names")

#rmsep_result <- multiblock::RMSEP(multiblock_pls,multiblock_pls$fitted.values)
print("multiblock MSEP result")
print(multiblock::MSEP(multiblock_pls))
print("pls MSEP result")
print(RMSEP(pls))

plot(multiblock::MSEP(multiblock_pls),legendpos = "topright",main="MSEP Multiblock PLS")

