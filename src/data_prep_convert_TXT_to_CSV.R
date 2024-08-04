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

#########################
##Import Function
#########################
# Location: Path, where the TXT.files are located
# Gruppen: string vector of the used groups, must be part of file name and be explicit
# A maximum of 8 groups is possible based on the available colours

My.Import <- function(Location, Gruppen = Files) {
  # List of the TXT.files in location
  Files <- list.files(path = Location, recursive=TRUE, include.dirs=TRUE, pattern = ".txt")
  Files1 <- c()
  print(Files)
  for (x in 1:length(Gruppen)) {
    Files1 <- c(Files1,Files[grep(pattern = as.character(Gruppen[x]), x = Files)])
  }
  
  import <- function(data) {
    start <- getwd() ##store Start wd
    setwd(Location) ### setwd to Data file
    
    df <- try(read.csv(data, header = FALSE, 
                       sep = ",",              
                       dec = "." ))         
    setwd(start) ## back to normal
    return(df)  ### Output = df
  }

  # Import of the files
  Raw.list <- lapply(Files1,import)
  
  # combine list of data frames 
  Raw.data <- do.call("cbind", Raw.list)
  
  # Remove wavenumbers
  Spectra <- as.data.frame(t(Raw.data[,-c(which(colnames(Raw.data) == "V1"))]))
  
  # Extract wavenumbers and set as column names
  Wavenumber <- round(as.numeric(Raw.data[,1]),2)
  colnames(Spectra) <- Wavenumber

  ### Test for a Gruppen vector, else the files are organised according to files vector
  if(Gruppen[1] != Files1[1]){
    ##### construct Groups vector
    Groups <- c(1:length(Files1))
    for (i in 1:length(Gruppen)){
      Pos <- c(grep(pattern = as.character(Gruppen[i]), x = Files1)) ##grep recognizes patterns in file names
      Groups[Pos] <- Gruppen[i] # appends vector
    }
  }else{
    Groups <- sub(pattern = ".TXT",x= Gruppen, replacement = "")
  }
  
  rownames(Spectra) <- Files1

  ##### Creation of a list containig spectra, wavenumbers and Groups 
  ###Zur Besseren Organisation
  OriginalData <- list(Wavenumber,Spectra,Files1)
  
  names(OriginalData) <- c("Wavenumber","Spectra","Filenames")
  
  # Output = list
  return(OriginalData)
}

#Third Step
#to save spectra take part of spectra name
data_raw <- My.Import(Location = "C:/Users/sebne/Documents/FHWN_Tulln/DataAnalysis/repo/input/clean_txt/NIR")
#data <- cbind(data_raw$Filenames,data_raw$Spectra)
#colnames(data)[names(data) == "data_raw$Filenames"] <- ""
#write to csv file
write.csv2(data_raw$Spectra,"spectra_nir_240523.csv", row.names = TRUE)

