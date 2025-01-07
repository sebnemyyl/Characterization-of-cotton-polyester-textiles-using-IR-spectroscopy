
data_snv <- load_saved_csv("temp/spectra_treated/nir/spectra_nir_regression_snv.csv")
pls <- run_pls(data_snv, rds_path = "temp")

vip <- function(model, ncomp) {
  # Extract necessary components from the model
  W <- model$loading.weights[, 1:ncomp]
  T <- model$scores[, 1:ncomp]
  Q <- model$Yloadings[, 1:ncomp]
  
  # Calculate the explained variance for each component
  explained_variance <- model$Xvar / sum(model$Xvar)
  
  # Compute the sum of squares for each component
  SS <- colSums((T %*% diag(Q))^2)
  
  # Normalize weights
  W_norm <- sweep(W, 2, sqrt(colSums(W^2)), FUN = "/")
  
  # Calculate VIP scores
  vip_scores <- sqrt(ncol(W) * rowSums((W_norm^2) * SS / sum(SS)))
  
  return(vip_scores)
}

vip_scores <- vip(pls, ncomp = 5)

important_vars <- which(vip_scores > 0.1)
print(important_vars)

spectra_matrix <- data.matrix(data_snv %>% dplyr::select(starts_with("spectra")))
reference_matrix <- data.matrix(data_snv %>% dplyr::select(starts_with("reference")))

# Subset the data to include only important wavelengths
data_snv_selected <- spectra_matrix[, important_vars]

df <- data.frame(reference_matrix, data_snv_selected)
#colnames(df) <- colnames(data_snv)


write.csv(df,"temp/data_snv_selected.csv")