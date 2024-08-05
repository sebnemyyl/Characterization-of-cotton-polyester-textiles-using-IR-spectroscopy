########################
#
#Get several OPUS files to R
#
#David Lilek 220328
#######################

# Before using script:
#### use Converter-FolderFiles.mtb file and anleitung N:\Spektroskopie\Geräte\RAMAN\Bedienung\ExportOpus_alsTxt.file_V001.docx
#### to convert OPUS files to several TXT files

load_txt_files <- function(file_name, dir) {
  path <- file.path(dir, file_name)
  df <- try(read.csv(path, header = FALSE, sep = ",", dec = "."))
  return(df)
}

split_file_name <- function(file_path) {
  file_name_with_ending <- basename(file_path)
  file_name <- strsplit(file_name_with_ending, split = "\\.")[[1]][1]
  feature_list <- strsplit(file_name, split = "_")[[1]]
  return(feature_list)
}

extract_number <- function(feature_list, index) {
  text <- sapply(feature_list, `[[`, index)
  number <- regmatches(text, regexpr("[0-9]+", text))
  return(number)
}

#########################
##Import Function
#########################
# dir: Path, where the TXT.files are located
convert_txt_to_dataframe <- function(dir) {
  # List the TXT.files in the directory
  file_paths <- list.files(path = dir, recursive = TRUE, include.dirs = TRUE, pattern = ".txt")
  # Load TXT files
  raw_dfs <- lapply(file_paths, load_txt_files, dir = dir)
  # Combine list of data frames
  combined_dfs <- do.call("cbind", raw_dfs)

  # Remove wavenumbers
  spectra <- as.data.frame(t(combined_dfs[, -c(which(colnames(combined_dfs) == "V1"))]))
  # Extract wavenumbers and set as column names
  wave_numbers <- round(as.numeric(combined_dfs[, 1]), 2)
  colnames(spectra) <- wave_numbers

  # File names as row name/unique id
  rownames(spectra) <- lapply(file_paths, basename)
  # Extracting features/meta data from the file name
  features <- lapply(file_paths, split_file_name)
  spectra$pet <- sapply(features, `[[`, 2)
  spectra$cotton <- sapply(features, `[[`, 3)
  spectra$specimen <- extract_number(features, 4)
  spectra$area <- extract_number(features, 5)
  spectra$spot <- extract_number(features, 6)
  spectra$measuring_date <- sapply(features, `[[`, 7)
  return(spectra)
}

spectra <- convert_txt_to_dataframe(dir = "input/clean_txt/NIR")
#write to csv file
write.csv2(spectra, "spectra_nir_240523.csv", row.names = TRUE)
