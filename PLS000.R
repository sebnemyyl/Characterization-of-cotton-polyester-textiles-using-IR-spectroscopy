##########################################
#
#Einlesen der x- und y-Daten
#
##########################################

###Paket installieren
#install.packages("pls")
library(pls)
library(plotly)
library(dplyr)

###Daten laden

Data_Raw <- read.csv("C:\\Users\\sebne\\Documents\\FHWN_Tulln\\DataAnalysis\\DataAnalysis\\JAF\\JAF\\spectra_mir.csv",header=TRUE,sep=",",dec=".")

### assign variables
Spectra_Original <- Data_Raw[,-1]
rownames(Spectra_Original) <- Data_Raw[,1]
Probenbezeichnung <- colnames(Spectra_Original)#Data_Raw[,1]

# Function to extract the first number from a string
extract_number <- function(string) {
  number <- as.numeric(sub("\\D*(\\d+).*", "\\1", string))
  ifelse(is.na(number), 50, number)
}

# Extracting numbers from sample names
numbers <- sapply(Probenbezeichnung, extract_number)
numbers[numbers < 10] <- 50

Reference_000 <- numbers
###Generate wavenumber
#Wellenzahl_000 <- as.numeric(sub("X","0",names(Spectra_Original)))
Wellenzahl_000<- Data_Raw[,c(1:1)]
#names(Spectra_Original) <- as.numeric(sub("X","0",names(Spectra_Original)))

#Spektren als Matrix abspeichern
Spectra_000 <- data.matrix(Spectra_Original) 
#Referenzwerte definieren
Reference_000 <- Reference_000
#Dataframe erstellen
Data_000 <- data.frame(Reference=I(Reference_000),Spektren=(t(Spectra_000)))
#Data_000 <- subset(Data_000, Reference == 55)

contains_na <- any(is.na(Spectra_000))
print(contains_na)

##################################################
#
#Daten visualisieren
#
###################################################

#load function plot_spectra
#load function plot_spectra
plot_spectra <- function(Spectra_000, Wellenzahl_000, Probenbezeichnung) {
  spc <- as.data.frame(t(Spectra_000))
  spc$wavenumber <- Wellenzahl_000
  spc_plot <- plot_ly(data = spc) %>%
    layout(
      xaxis = list(
        range=c(4000,600),
        title = "Wavenumber"),
      yaxis = list(
        title = "Absorbance")
    )
  i=1
  sample_count = ncol(spc)-1
  for (i in 1:sample_count) {
    y_values = spc[,i]
    na_count <- sum(sapply(y_values, is.na))
    print(i)
    print(na_count)
    spc_plot <- add_trace(spc_plot,
                          x=spc[,ncol(spc)],
                          y=y_values,
                          data=spc,
                          type="scatter",
                          mode="lines",
                          name=Probenbezeichnung[i]) 
  }
  spc_plot
}

#call function plot_spectra
spc_plot <- plot_spectra(t(Spectra_000),Wellenzahl_000, Probenbezeichnung)
spc_plot

##################################################
#
#PLS_000 Modell erstellen
#
###################################################

### PLS Modell erstellen
#Algorithmus einstellen
number_factors_pls <- 10

run_pls_step1 <- function(Reference_000, Spectra_000, Data_000, number_factors_pls) {
  options(digits = 4)
  pls.options(plsalg="oscorespls")
  PLS_000 <- plsr(Reference_000 ~ Spectra_000, data=Data_000, ncomp=number_factors_pls, val="LOO")
  
  
  #############
  #
  #Ergebnisse PLS_000 Modell
  #
  #############
  
  ###Zusammenfassung
  summary(PLS_000)
  
  ###RMSEP
  RMSE <- as.data.frame(RMSEP(PLS_000)[[1]])
  RMSE_PLOT <- plot_ly(x=1:number_factors_pls,
          y=~as.numeric(RMSE[1,-1]),
          mode="lines") %>%
    layout(
      xaxis = list(
        title = "Factor"),
      yaxis = list(
        title = "RMSE"),
      title = "RMSE vs. Factors"
    )
  
  
  
  ###Erklaerte Varianz der Komponenten
  VARIANCE_PLOT <- plot_ly(x=1:number_factors_pls,
                           y=explvar(PLS_000),
                           mode="line")   %>%
    layout(title = "Explained Variance vs. Factors",
           xaxis = list(
             title = "Factors"),
           yaxis = list(
             title = "Explained variance [%]")
    )
  
  return(list(PLS_000,RMSE_PLOT,VARIANCE_PLOT))

}

PLS_000 <- run_pls_step1(Reference_000, Spectra_000, Data_000, number_factors_pls)
#RMSE plot
PLS_000[[2]]
#explained variance
PLS_000[[3]]

##############################################
Faktor <- 5
##############################################

run_pls_part2 <- function(Reference_000, Faktor, PLS_000, Wellenzahl_000, number_factors_pls) {
  i=1
  #Predicted vs. Measured inkl. Geradengleichung und Bestimmtheitsmasz; Faktor eingeben und ausfuehren
  Coord <- par('usr')
  YY <- length(Reference_000)
  B <- Faktor*YY
  A <- (B-YY)+1  
  predicted_000 <- PLS_000$validation$pred[A:B]
  reference_000 <- Reference_000
  Modell_000 <- lm (predicted_000 ~ reference_000)
  summary(Modell_000)
  
  PRED_MEASURED_PLOT <- plot_ly(x=reference_000,y=predicted_000,text=1:length(reference_000),type = "scatter") %>%
    add_trace(x = ~reference_000, y = Modell_000$fitted.values,mode="lines") %>%
    layout(title = 'Measured vs. Predicted',
           showlegend = FALSE,
           annotations = list(
             x = Coord[1]*1.1,
             y = Coord[4]*0.95,
             text = paste("R^2=",round(summary(Modell_000)$r.squared,4),"\n",
                          "Achsenabschnitt=",round(summary(Modell_000)$coefficients[1],4),"\n",
                          "Steigung=",round(summary(Modell_000)$coefficients[2],4)),
             showarrow = F,
             xanchor="left")
    )
  
  ###Scoreplot
  SCOREPLOT <- plot_ly(x=PLS_000$scores[,1],
          y=PLS_000$scores[,2],
          text=1:length(PLS_000$scores[,1]),
          type="scatter") %>%
    layout(
      xaxis = list(
        title = "PC1"),
      yaxis = list(
        title = "PC2"),
      title = "Scoreplot"
    )
  
  
  ###Loadingplot
  #Welche Loadings sollten geplottet werden?
  n_loadings <- c(1:Faktor)
  
  ldg <- seq(1,length(Wellenzahl_000)*number_factors_pls,by=length(Wellenzahl_000))
  ldg <- c(ldg,length(Wellenzahl_000)*number_factors_pls)
  lwz_000 <- length(Wellenzahl_000)
  LOADING_PLOT <- plot_ly(x=~Wellenzahl_000)
  
  for (i in n_loadings){
    LOADING_PLOT <- add_trace(LOADING_PLOT,
                              x=Wellenzahl_000,
                              y=PLS_000$loadings[ldg[i]:(ldg[i+1]-1)],
                              text=1:length(Wellenzahl_000),
                              type="scatter",
                              mode="lines",
                              name=paste("Loading",i)) %>%
      layout(title="Loadingplot",
        xaxis = list(
          range=c(4000, 600),
          tickformat = "digit"),
        yaxis = list(
          title = "Loadings")
      )
  }
  
  ###Residuen
  
  YYY <- length(Reference_000[!is.na(Reference_000)])
  BB <- Faktor*YYY
  AA <- (BB-YYY)+1 
  x <- 1:(BB-AA+1)
  
  RESIDUENPLOT <- plot_ly(x=x,
          y=PLS_000$residuals[AA:BB]) %>%
    layout(title = "Residuenplot",
           yaxis = list(
             title = "Residuen"),
           xaxis = list(
             title = "Sample Number")
           )
  
  return(list(PLS_000,PRED_MEASURED_PLOT,SCOREPLOT,LOADING_PLOT,RESIDUENPLOT))
}

run_pls_part2(Reference_000, Faktor, PLS_000[[1]], Wellenzahl_000, number_factors_pls)  
