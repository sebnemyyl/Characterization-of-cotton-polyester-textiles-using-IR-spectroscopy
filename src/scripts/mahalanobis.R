library(dplyr)

input_path <- "input/raw_csv/spectra_nir_241011.csv"
output_path <- "input/clean_csv/spectra_nir_all.csv"
type <- "nir"

spectral_data <- spectra_df_clean %>% dplyr::select(starts_with("spectra"))
# Display a summary of the data
summary(spectral_data)

spectra_30 <- spectra_df_clean %>% filter(reference.cotton == 30)
spectra_30 <- dplyr::select(spectra_30, -c("reference.specimen", "reference.pet", "reference.area", "reference.spot", "reference.measuring_date"))

# Perform PCA
norm.factors <- apply(spectra_30, 1, function(x){sqrt(sum(x^2))})
spectra_30 <- sweep(spectra_30, 1, norm.factors, "/")
pca_result <- prcomp(spectra_30, scale. = TRUE)

# Choose a subset of principal components explaining most of the variance
pca_data <- as.data.frame(pca_result$x[, 1:3]) # Adjust the number of components as needed
# Assuming pca_data has the row names or an index column with specimen information

pca_data$Specimen <- gsub("_area.*", "", rownames(pca_data))

# Group and summarize PCA scores
aggregated_pca <- pca_data %>%
  group_by(Specimen) %>%
  dplyr::summarise(dplyr::across(starts_with("PC"), ~ mean(.x, na.rm = TRUE)))


# Check and remove zero-variance columns
aggregated_pca <- aggregated_pca %>%
  select_if(~ var(.) > 1e-6)

# Regularize covariance matrix
cov_matrix <- cov(aggregated_pca[-1]) + diag(1e-6, ncol(aggregated_pca[-1]))

# Calculate mean vector
mean_vector <- colMeans(aggregated_pca[-1])

# Calculate Mahalanobis distances
aggregated_pca$Mahalanobis_Distance <- mahalanobis(aggregated_pca[-1], 
                                                   center = mean_vector, 
                                                   cov = cov_matrix)

# Set threshold
df <- ncol(aggregated_pca) - 1
threshold <- qchisq(0.95, df = df)

# Flag outliers
aggregated_pca$Outlier <- aggregated_pca$Mahalanobis_Distance > threshold

# View results
print(aggregated_pca)
