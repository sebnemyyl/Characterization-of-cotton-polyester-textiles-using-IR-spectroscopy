library(pcv)
library(mdatools)


# TODO doesnt work yet
preprocessing <- function(originalspectra, prep_set) {
  print("Hello world!")
  attr(originalspectra, "xaxis.values") <- as.numeric(colnames(originalspectra))
  attr(originalspectra, "xaxis.name") <- "Wavenumber"

  # apply combined methods
  pspectra <- employ.prep(prep_set, originalspectra)
  print("What aup!!")

  #par(mfrow = c(2, 1))
  mdaplot(originalspectra, type = "l", main = "Original")
  #mdaplot(pspectra, type = "l", main = "after treatment")
  return(pspectra)
}

# TODO doesnt work yet
run_mda <- function(spectra_df) {
  ref_cotton <- spectra_df$reference.cotton
  Xpv <- pcvpls(spectra_df, ref_cotton, 20, cv = list("ven", 10))
  pls_model_pcv <- pls(spectra_df, ref_cotton, 10, x.test = Xpv, y.test = ref_cotton)
  summary(pls_model_pcv)
}


myprep <- list(prep("snv"))