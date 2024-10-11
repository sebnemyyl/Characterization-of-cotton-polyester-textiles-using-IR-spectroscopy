source("src/util/02_data_prep_load_CSV.R")
source("src/util/06_model_pls.R")

filtered_data_nir <- load_saved_csv("/Users/andi/Documents/Coding/spectroscopy/temp/spectra_treated_nir_detrend.csv")
# TODO just to make it run -> delete later!
filtered_data_mir <- filtered_data_nir

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

TEMP_DIR <- "temp/"
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

