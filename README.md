# Characterization of cotton-polyester textiles using IR-spectroscopy
In this Github project you find util functions and scripts 
to work with IR-spectral data. The project aims to quantify cotton and polyester 
ratios in textiles using Near_Infrared (NIR) and Mid-Infrared (MIR) spectral data. 
It provides options for data exploration and functions to find the 
best performing regression models and the optimum hyperparameter space.

## Prerequisites
The codebase is in R and Python. To use the scripts Python 3.13+ and R 4.4+ should be installed.
Notebooks can be run with Jupyter.


## Structure
The project is structured via its folder structure. The data is in the `input` folder and the 
source code in `src`. Within these folders, the files have the following structure:

### input folder
- **renamed_txt:** This folder contains the raw spectroscopic data received from the spectroscopic machines. Each txt-file is one measurement.
- **raw_csv:** This folder combines the different txt-files into csv-files. In a csv-file one row is one measurement.
- **clean_csv:** This folder contains csv-files, which are already preprocessed and can be used for further data exploration or in machine learning models.

### src folder
- **util:** The core modules used for data preparation, analysis and regression models.
The majority of the models are in python, but some data preparation modules are in R. These
files can be included in any script or notebook and the functions can be used directly.  
- **scripts:** Scripts include the core modules and call their functions. Scripts can be
called from the command line or within an IDE.
- **notebooks:** Sample Jupyter notebooks are included in the thesis, which do 
data exploration and regression model evaluation. They utilize the functions 
provided in the util folder, similar to scripts.


## Sample Workflows
By using the modules in `util` and scripts in `scripts` in conjunction with
the data provided in the `input` many different data exploration and machine learning 
workflows can be enabled. Some sample workflows are described here:

### Preprocess raw spectra data
1. Run the R script [clean_up_spectra.R](src/scripts/preprocessing/clean_up_spectra.R): Set a csv-file from `input/raw_csv` 
as `input_path` and define the desired `output_path`. Define whether the waterband
should be removed and further filtering of the data.
2. Run the R script [baseline_correction.R](src/scripts/preprocessing/baseline_correction.R): Use the resulting csv_file from the 
previous step as input `csv_path`. Define an existing empty folder as `output_dir` 
and define a list of baseline correction methods in `baseline_corr_types`. After
running the script a list of csv-files with baseline corrected spectra is generated
in the `output_dir`.
3. The resulting csv files can be used for further data analysis as well as 
in regression models.

### Evaluate regression models
1. Run the script [run_hyper_param_search.py](src/scripts/models/run_hyper_param_search.py): 
A regression model pipeline is provided in the script . A folder with preprocessed 
spectral data (csv-files) should be used as input (e.g. by following the steps above). 
Settings of the pipeline can be set, e.g. whether to use PCA or to scale the data 
or which train test split. After running
the script, a JSON file with the best performing results and parameters is generated. 
2. The resulting JSON file can be plotted using the helper functions of 
[m06_model_plotting.py](src/util/m06_model_plotting.py). This can be done in a Jupyter notebook 
(e.g. [model_evaluation.ipynb](src/notebooks/model_evaluation.ipynb)) or via custom scripts.
3. For further analysis on how model parameters behave, 
the script [run_model_evaluation_over_param.py](src/scripts/models/run_model_evaluation_over_param.py) 
can be run. It runs a specific model over a self-defined list of parameters and
plots how the error changes based on the parameter.
