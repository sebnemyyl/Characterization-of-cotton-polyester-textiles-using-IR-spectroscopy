# Function to extract pet content from sample name
extract_pet_content <- function(sample_file_name) {
  splitted <- strsplit(sample_file_name, "_")
  pet_content <- as.numeric(splitted[[1]][[2]])
  pet_content
}

### Create spectra dataframe with labeled columns (wave numbers) and rows (sample names)
load_spectra <- function(file_path) {
  spectra <- read.csv(file_path, header = TRUE, row.names = 1, sep = ";", dec = ".")
  sample_file_names <- rownames(spectra)
  sample_info <- sapply(sample_file_names, extract_pet_content)
  spectra$pet_content <- sample_info
  spectra
}

setwd(".")
spectra <- load_spectra("spectra_textiles_mir.csv")