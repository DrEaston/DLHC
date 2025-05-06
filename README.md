# DLHC
DLHC final

Our code may be run through the 2 files in the main directory:
  - run_models.ipynb
  - CollectStats.ipynb

run_models:
  - change dataset variable to run different models in cell: in[10]
  - run model in cell: in[15]
  - calls experiment_competing_risk.py to run the models

Experiment_Competing_Risk:
  - comment out different models to run or not run them
  - set hyperparameters/grid search at line 30

CollectStats:
  - reads csvs from DLHC/results/{dataset}
  - creates and prints summary information
  - there are different cells that can be used to collect stats for 1 dataset, or loop through all datasets
