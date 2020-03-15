# Headcase Evaluation

The data and code in this repository allow users to reproduce all figures, tables, and estimates reported in the main text of the manuscript. 

The python environment used during this project (also contains non-essential libraries) can be reproduced with the following command:  

`conda create --name headcase --file requirements.txt`

This requirements file was generated using the following command:  
`conda list --export > requirements.txt`. 

# Organization of this repository

- **code:** scripts and libraries ordered and named by analysis steps
- **data:** raw data (realignment parameters)
- **analysis:** generated data files, i.e. processed/analyzed/summarized data
- **figures:** figures in .png, .pdf and tables in .docx files


