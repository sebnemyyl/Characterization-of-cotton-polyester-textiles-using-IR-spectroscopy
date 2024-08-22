library(pls)
library(plotly)
library(dplyr)

run_pls <- function(spectra_df, rds_path) {
  options(digits = 4)
  pls.options(plsalg = "oscorespls")
  ref_cotton <- spectra_df$reference.cotton
  spectra <- spectra_df_clean %>% dplyr::select(starts_with("spectra"))
  names(spectra) <- as.numeric(sub("spectra.", "", names(spectra)))
  spectra_matrix <- data.matrix(spectra)
  pls_result <- plsr(ref_cotton ~ spectra_matrix, data = spectra_df, ncomp = 10, val = "LOO")
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
  abline(model, col = "black", lwd = 2)
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