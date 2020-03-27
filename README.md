# Custom-molded headcases have limited efficacy in reducing head motion for fMRI

The data and code in this repository allow users to reproduce all figures, tables, and estimates reported in the main text of the manuscript. 

The python environment used during this project (also contains non-essential libraries) can be reproduced with the following command:  

`conda create --name headcase --file requirements.txt`

This requirements file was generated using the following command:  
`conda list --export > requirements.txt`. 

The [Officer](https://davidgohel.github.io/officer/) and [Flextable](https://davidgohel.github.io/flextable/) R packages are also dependencies that require manual installation. They are used to generate `.docx` files of semi-formatted tables and in-text statistical reporting.

# Organization of this repository

- **code:** scripts and helper functions ordered and named by analysis steps
- **data:** raw data (realignment parameters)
- **analysis:** generated data files, i.e. processed/analyzed/summarized data
- **figures:** figures in .png, .pdf and tables in .docx files


