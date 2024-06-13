###Datenvorbehandlungen
##Pkg prospectr https://cran.r-project.org/web/packages/prospectr/vignettes/prospectr-intro.pdf

#install.packages("mdatools")
#install.packages("prospectr")
library(prospectr)
library(mdatools)

#############################################
#Spektrenvorbehandlung mit SNV
#############################################

Spectra_100 <- standardNormalVariate(Spectra_Original)

Wellenzahl_100 <- Wellenzahl_000
Reference_100 <- Reference_000
Reference_100 <- 100-Reference_100
#Spektren als Matrix abspeichern
Spectra_100 <- data.matrix(Spectra_100) 
#Dataframe erstellen
Data_100 <- data.frame(Reference=I(Reference_100),Spektren=(Spectra_100))


##################################################
#
#Daten visualsieren
#
###################################################

matplot(Wellenzahl_100,t(Spectra_100),lty=1,type="l",ylab="Absorbance")

##################################################
#
#PLS_100 Modell erstellen
#
###################################################

### PLS Modell erstellen
#Algorithmus einstellen
options(digits = 4)
pls.options(plsalg="oscorespls")
PLS_100 <- plsr(Reference_100 ~ Spectra_100, data=Data_100, ncomp=7, val="LOO")


#############
#
#Ergebnisse PLS_100 Modell
#
#############

###Zusammenfassung
summary(PLS_100)

###RMSEP
plot(RMSEP(PLS_100),legendpos="topright",main="RMSEP vs. Faktoren PLS_100")

###Predicted vs. Measured ncomp=Anzahl der Kompnenten
plot(PLS_100, ncomp = 3, line = TRUE,main="Predicted vs. Measured PLS_100")

#Predicted vs. Measured inkl. Geradengleichung und Bestimmtheitsmasz; Faktor eingeben und ausfuehren
Faktor <- 3

YY <- length(Reference_100)
B <- Faktor*YY
A <- (B-YY)+1  
predicted_100 <- PLS_100$validation$pred[A:B]
reference_100 <- Reference_100
Modell_100 <- lm (predicted_100 ~ reference_100)
summary(Modell_100)
plot(reference_100, predicted_100,main="Predicted vs. Measured PLS_100")
Coord <- par('usr') #xmin xmax ymin ymax
x <- Coord[1]*1.1
y <- Coord[4]*0.95
y1 <- y*0.95
y2 <- y1*0.95
text(x,y,paste("R^2=",round(summary(Modell_100)$r.squared,4)), cex=0.8, adj=0) 
text(x,y1,paste("Achsenabschnitt=",round(summary(Modell_100)$coefficients[1],4)), cex=0.8,adj=0)
text(x,y2,paste("Steigung=",round(summary(Modell_100)$coefficients[2],4)), cex=0.8,adj=0)
abline(lm(predicted_100 ~ reference_100),col="red")

#Identify
#identify(reference_100,predicted_100)

###Scoreplot
plot(PLS_100, plottype = "scores", comps = 1:2,main="Scoreplot PLS_100")
abline(h=0)
#Identify
#identify(PLS_100$scores[,1],PLS_100$scores[,2])

###Erklaerte Varianz der Komponenten
explvar(PLS_100)

###Loadingplot
plot(PLS_100, "loadings", comps = 1:3, legendpos = "topleft",
     labels = "numbers", xlab = "Wellenzahl",main="Loading-Plot PLS_100")
abline(h = 0)
#Identify
#lwz_100 <- length(Wellenzahl_100)
#identify(Wellenzahl_100,PLS_100$loadings[1:lwz_100])

###Regressionskoeffizient
plot(PLS_100, plottype = "coef", ncomp = 1:3, legendpos = "bottomleft",
     labels = "numbers", xlab = "Wellenzahl",main="Regressionskoeffizient PLS_100")
#Identify
#lwz_100 <- length(Wellenzahl_100)
#identify(Wellenzahl_100,PLS_100$coefficients[1:lwz_100])

###Residuen

FFaktor <- 2

  YYY <- length(Reference_100[!is.na(Reference_100)])
  BB <- FFaktor*YYY
  AA <- (BB-YYY)+1  
  plot(PLS_100$residuals[AA:BB],main="Residuenplot PLS_100")
  abline(0,0)
  
  #Identify
  #Index <- seq(from=1,to=YYY, by=1)
  #identify(Index,PLS_100$residuals[AA:BB]) 

