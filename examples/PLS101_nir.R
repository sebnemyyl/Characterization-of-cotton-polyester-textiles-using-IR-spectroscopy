###################################################
#
#Ausreiszer entfernen
#Spektralen Bereich einschraenken
#
###################################################

#Ausreiszer entfernen
Spectra_101 <- Spectra_100
Spectra_101 <- Spectra_101[-c(31:40),]

Reference_101 <- Reference_100
Reference_101 <- Reference_101[-c(31:40)]

#Spektralen Bereich einschraenken
Spectra_101[,c(1:850,2110:2307)] <- 0
Wellenzahl_101 <- Wellenzahl_100

#Spektren als Matrix abspeichern
Spectra_101 <- data.matrix(Spectra_101) 
#Dataframe erstellen
Data_101 <- data.frame(Reference=I(Reference_101),Spektren=(Spectra_101))


##################################################
#
#Daten visualsieren
#
###################################################

matplot(Wellenzahl_000,t(Spectra_000),lty=1,type="l",ylab="Absorbance")

##################################################
#
#PLS_101 Modell erstellen
#
###################################################

### PLS Modell erstellen
#Algorithmus einstellen
options(digits = 4)
pls.options(plsalg="oscorespls")
PLS_101 <- plsr(Reference_101 ~ Spectra_101, data=Data_101, ncomp=10, val="LOO")


#############
#
#Ergebnisse PLS_101 Modell
#
#############

###Zusammenfassung
summary(PLS_101)

###RMSEP
plot(RMSEP(PLS_101), legendpos = "topright",main="RMSEP vs. Faktoren PLS_101")

###Predicted vs. Measured ncomp=Anzahl der Kompnenten
plot(PLS_101, ncomp = 3, line = TRUE,main="Predicted vs. Measured PLS_101")

#Predicted vs. Measured inkl. Geradengleichung und Bestimmtheitsmasz; Faktor eingeben und ausfuehren
Faktor <- 8

YY <- length(Reference_101[!is.na(Reference_101)])
B <- Faktor*YY
A <- (B-YY)+1  
predicted_101 <- PLS_101$validation$pred[A:B]
reference_101 <- Reference_101[!is.na(Reference_101)]
Modell_101 <- lm (predicted_101 ~ reference_101)
summary(Modell_101)
plot(reference_101, predicted_101,main="Predicted vs. Measured PLS_101")
Coord <- par('usr') #xmin xmax ymin ymax
x <- Coord[1]*1.1
y <- Coord[4]*0.95
y1 <- y*0.95
y2 <- y1*0.95
text(x,y,paste("R^2=",round(summary(Modell_101)$r.squared,4)), cex=0.8, adj=0) 
text(x,y1,paste("Achsenabschnitt=",round(summary(Modell_101)$coefficients[1],4)), cex=0.8,adj=0)
text(x,y2,paste("Steigung=",round(summary(Modell_101)$coefficients[2],4)), cex=0.8,adj=0)
abline(lm(predicted_101 ~ reference_101),col="red")

#Identify
#identify(reference_101,predicted_101)

###Scoreplot
plot(PLS_101, plottype = "scores", comps = 1:2,main="Scoreplot PLS_101")
abline(h=0)
#Identify
#identify(PLS_101$scores[,1],PLS_101$scores[,2])

###Erklaerte Varianz der Komponenten
explvar(PLS_101)

###Loadingplot
plot(PLS_101, "loadings", comps = 1:3, legendpos = "topleft",
     labels = "numbers", xlab = "Wellenzahl",main="Loading-Plot PLS_101")
abline(h = 0)
#Identify
#lwz_101 <- length(Wellenzahl_101)
#identify(Wellenzahl_101,PLS_101$loadings[1:lwz_101])

###Regressionskoeffizient
plot(PLS_101, plottype = "coef", ncomp = 1:3, legendpos = "bottomleft",
     labels = "numbers", xlab = "Wellenzahl",main="Regressionskoeffizient PLS_101")
#Identify
#lwz_101 <- length(Wellenzahl_101)
#identify(Wellenzahl_101,PLS_101$coefficients[1:lwz_101])

###Residuen

FFaktor <- 2

  YYY <- length(Reference_101[!is.na(Reference_101)])
  BB <- FFaktor*YYY
  AA <- (BB-YYY)+1  
  plot(PLS_101$residuals[AA:BB],main="Residuenplot PLS_101")
  abline(0,0)
  
  #Identify
  #Index <- seq(from=1,to=YYY, by=1)
  #identify(Index,PLS_101$residuals[AA:BB]) 

  