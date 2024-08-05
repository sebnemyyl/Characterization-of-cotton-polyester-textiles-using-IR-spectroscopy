########################
#
#Get several OPUS files to R
#
#David Lilek 220328
#######################

#First step
####use Converter-FolderFiles.mtb file and anleitung N:\Spektroskopie\Geräte\RAMAN\Bedienung\ExportOpus_alsTxt.file_V001.docx
####to convert OPUS files to several TXT files

#Second step
####load Import Function

import <- function(data, Location) {
  path <- file.path(Location, data)
  df <- try(read.csv(path, header = FALSE,sep = ",", dec = "."))
  return(df)
}

split_file_name <- function(file_path) {
  file_name_with_ending <- basename((file_path))
  file_name <- strsplit(file_name_with_ending, split = "\\.")[[1]][1]
  feature_list <- strsplit(file_name, split = "_")[[1]]
  return(feature_list)
}

extract_number <- function(feature_list, index) {
  text <- sapply(feature_list, `[[`, index)
  number <- regmatches(text, regexpr("[0-9]+", text))
  return(number)
} 
# TODO rename location to folder something

#########################
##Import Function
#########################
# Location: Path, where the TXT.files are located
My.Import <- function(Location) {
  # List of the TXT.files in location
  Files1 <- list.files(path = Location, recursive = TRUE, include.dirs = TRUE, pattern = ".txt")
  # Import of the files
  raw_dfs <- lapply(Files1, import, Location = Location)
  # combine list of data frames 
  Raw.data <- do.call("cbind", raw_dfs)
  

  # Remove wavenumbers
  Spectra <- as.data.frame(t(Raw.data[, -c(which(colnames(Raw.data) == "V1"))]))
  
  # Extract wavenumbers and set as column names
  Wavenumber <- round(as.numeric(Raw.data[,1]),2)
  colnames(Spectra) <- Wavenumber
  
  rownames(Spectra) <- Files1
  features <- lapply(Files1, split_file_name)
  Spectra$pet <- sapply(features, `[[`, 2) #TODO could be cotton instead
  Spectra$cotton <- sapply(features, `[[`, 3)
  Spectra$specimen <- extract_number(features, 4)
  Spectra$area <- extract_number(features, 5)
  Spectra$spot <- extract_number(features, 6)
  Spectra$measuring_date <- sapply(features, `[[`, 7)
  return(Spectra)
}

spectra <- My.Import(Location = "input/clean_txt/NIR")
#write to csv file
write.csv2(spectra, "spectra_nir_240523.csv", row.names = TRUE)
