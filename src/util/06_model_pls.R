library(pls)
library(plotly)
library(dplyr)
library(ggplot2)
library(multiblock)


test_train_split <- function(spectra_df){
  set.seed(123)
  # Define the splitting ratio
  train_size <- 0.8
  
  train_index <- sample(1:nrow(spectra_df), size = round(train_size * nrow(spectra_df)))
  
  train_data <- spectra_df[train_index, ]
  test_data <- spectra_df[-train_index, ]
  
  print(dim(train_data))
  print(dim(test_data))
  return(train_data, test_data)
  
}
run_pls <- function(spectra_df, rds_path) {
  options(digits = 4)
  pls.options(plsalg = "oscorespls")
  ref_cotton <- spectra_df$reference.cotton
  spectra <- spectra_df %>% dplyr::select(starts_with("spectra"))
  names(spectra) <- as.numeric(sub("spectra.", "", names(spectra)))
  spectra_matrix <- data.matrix(spectra)
  print("pls data")
  print(dim(spectra_matrix))
  pls_result <- plsr(ref_cotton ~ spectra_matrix, data = spectra_df, ncomp = 10, val = "LOO")
  if (!missing(rds_path)) {
    saveRDS(pls_result, rds_path)
  }
  return(pls_result)
}


run_multiblock_pls <- function(block1, block2, rds_path) {
  
  baseline_correction_df_nir_ <- block1 %>%
    rename_with(~ paste0("NIR_", .), starts_with("spectra"))
  
  baseline_correction_df_mir_ <- block2 %>%
    rename_with(~ paste0("MIR_", .), starts_with("spectra"))
  
  baseline_correction_df_nir_$num <- sequence(rle(as.character(baseline_correction_df_nir_$reference.cotton))$lengths)
  baseline_correction_df_mir_$num <- sequence(rle(as.character(baseline_correction_df_mir_$reference.cotton))$lengths)
  
  joined_data <- inner_join(baseline_correction_df_mir_, baseline_correction_df_nir_, by = c("num", "reference.cotton"))
  joined_data <- select(joined_data, -c(reference.pet.x, reference.specimen.x, reference.area.x, reference.spot.x, reference.measuring_date.x,
                                        reference.pet.y, reference.specimen.y, reference.area.y, reference.spot.y, reference.measuring_date.y,
                                        num))  
  
  # Separate the predictors (X) and response (Y)
  #X_NIR <- joined_data %>% select(starts_with("NIR"))
  #X_MIR <- joined_data %>% select(starts_with("MIR"))
  #Y <- joined_data %>% select(starts_with("reference.cotton"))
  print("multiblock data")
  print(dim(joined_data))
  
  pls_result <- mbpls(X = list(NIR = joined_data %>% select(starts_with("NIR")),
                               MIR = joined_data %>% select(starts_with("MIR"))),
                      Y = joined_data %>% select(starts_with("reference.cotton")), ncomp = 10, validation="CV")
  if (!missing(rds_path)) {
    saveRDS(pls_result, rds_path)
  }
  return(pls_result)
}



plot_pls <- function(pls) {
  summary(pls)

  plot(RMSEP(pls), legendpos = "topright", main = "RMSEP vs. Factors PLS")

  plot(pls, ncomp = 5, line = TRUE, main = "Predicted vs. Measured PLS")
}

plot_predicted_vs_measured <- function(pls, spectra_df) {
  factor <- 10
  ref_cotton <- spectra_df$reference.cotton
  len <- length(ref_cotton)
  end <- factor * len
  start <- (end - len) + 1
  predicted <- pls$validation$pred[start:end]
  model <- lm(predicted ~ ref_cotton)
  print(summary(model))
  plot(ref_cotton, predicted,
       xlab = "Measured Cotton Content [%]", ylab = "Predicted Cotton Content [%]",
       main = "Predicted Linear Model")
  abline(model, col = "red", lwd = 2)
}

plot_loading <- function(pls) {
  plot(pls, "loadings", comps = 1:3, legendpos = "topleft",
       labels = "numbers", xlab = "Wavenumber [1/cm]", lty = c(1, 3, 5), col = "black")
  abline(h = 0)

  loadings <- pls$loadings
  x <- as.numeric(names(loadings[, 1]))
  plot(x,
       loadings[, 1],
       xlim = rev(range(x)),
       ylim = c(-0.10, 0.25),
       lty = 1,
       type = "l",
       xlab = "Wavenumber [1/cm]",
       ylab = "Loading value [-]",
       lwd = 1)
  lines(x,
        loadings[, 2] + 0.05,
        lty = 3,
        type = "l",
        lwd = 1)
  lines(x,
        loadings[, 3] + 0.10,
        lty = 5,
        type = "l",
        lwd = 1)
  text(x = 2400, y = 0.01,
       labels = "Loading 1",
       col = "black", cex = 1.2)
  text(x = 2400, y = 0.01 + 0.05,
       labels = "Loading 2",
       col = "black", cex = 1.2)
  text(x = 2400, y = 0.01 + 0.10,
       labels = "Loading 3",
       col = "black", cex = 1.2)
}


plot_multiblockpls_predicted_vs_measured <- function(pls, joined_data) {
  Y = joined_data %>% select(starts_with("reference.cotton"))
  
  pred_vs_actual <- data.frame(
    Actual = Y,
    Predicted = pls$fitted.values[,,3]
  )
  
  ggplot(pred_vs_actual, aes(x = reference.cotton, y = Predicted)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +  # Reference line (y = x)
    labs(title = "Predicted vs Actual",
         x = "Actual Values",
         y = "Predicted Values") +
    theme_minimal()
  
}