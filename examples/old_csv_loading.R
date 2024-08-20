
library(pls)
library(plotly)
library(dplyr)
library(prospectr)


setwd(".")
Data_Raw <- read.csv("input/spectra_mir_240626.csv",header=TRUE,sep=";",dec=",", row.names = 1)


### assign variables
Spectra_Original <- Data_Raw[,-c(1:2)]
Probenbezeichnung <- rownames(Data_Raw)

# Function to extract the first number from a string
extract_number <- function(string) {
  number <- as.numeric(sub("\\D*(\\d+).*", "\\1", string))
  ifelse(is.na(number), 50, number)
}

# Extracting numbers from sample names
numbers <- sapply(Probenbezeichnung, extract_number)
numbers[numbers < 10] <- 50

Reference_000 <- numbers
###Wellenzahl erzeugen
Wellenzahl_000 <- as.numeric(sub("X","0",names(Spectra_Original)))
names(Spectra_Original) <- as.numeric(sub("X","0",names(Spectra_Original)))

#Spektren als Matrix abspeichern
Spectra_000 <- data.matrix(Spectra_Original) 
#Referenzwerte definieren
Reference_000 <- Reference_000
#Dataframe erstellen
Data_000 <- data.frame(Reference=I(Reference_000),Spektren=(Spectra_000))

# SNV
Spectra_100 <- standardNormalVariate(Spectra_Original)
Reference_100 <- 100-Reference_000
# Outlier Removal
# 4th sample is the outlier for both here and mdatools 

Spectra_101 <- Spectra_100
Spectra_101 <- Spectra_101[-c(4),]

Reference_101 <- Reference_100
Reference_101 <- Reference_101[-c(4)]

#Spektralen Bereich einschraenken
Spectra_101[,c(1:200,610:1140,1715:1760)] <- 0
Wellenzahl_101 <- Wellenzahl_000

#Spektren als Matrix abspeichern
Spectra_101 <- data.matrix(Spectra_101) 
#Dataframe erstellen
Data_101 <- data.frame(Reference=I(Reference_101),Spektren=(Spectra_101))


#matplot(Wellenzahl_101,t(Spectra_101),lty=1,type="l",ylab="Absorbance")

options(digits = 4)
pls.options(plsalg="oscorespls")
PLS_101 <- plsr(Reference_101 ~ Spectra_101, data=Data_101, ncomp=10, val="LOO")
# summary(PLS_101)

# plot(PLS_101, "loadings", comps = 1:3, legendpos = "topleft",
#      labels = "numbers", xlab = "Wavenumber [1/cm]", lty = c(1,3,5), col = "black")
# abline(h = 0)


x <- as.numeric(names(PLS_101$loadings[,1]))
plot(x,
     PLS_101$loadings[,1],
     xlim = rev(range(x)),
     ylim = c(-0.10,0.25),
     lty = 1,
     type = "l",
     xlab = "Wavenumber [1/cm]",
     ylab = "Loading value [-]",
     lwd = 1)
lines(x,
      PLS_101$loadings[,2]+0.05,
      lty = 3,
      type = "l",
      lwd = 1)
lines(x,
      PLS_101$loadings[,3]+0.10,
      lty = 5,
      type = "l",
      lwd = 1)
text(x = 2400, y = 0.01, 
     labels = "Loading 1",
     col = "black", cex = 1.2)
text(x = 2400, y = 0.01+0.05, 
     labels = "Loading 2",
     col = "black", cex = 1.2)
text(x = 2400, y = 0.01+0.10, 
     labels = "Loading 3",
     col = "black", cex = 1.2)