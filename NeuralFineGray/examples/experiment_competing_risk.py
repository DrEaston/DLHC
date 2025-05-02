import sys
import os
import time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/content/drive/MyDrive/DLHC'))) 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/content/drive/MyDrive/DLHC/NeuralFineGray')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/content/drive/MyDrive/DLHC/DeepSurvivalMachines')))


from nfg import datasets
from experiment import *

random_seed = 0

# Open dataset
dataset = sys.argv[1]  # FRAMINGHAM, SYNTHETIC_COMPETING, PBC, SEER
sys.path.append('/content/drive/MyDrive/DLHC')
# Specific fold selection
fold = None

# if len(sys.argv) == 3:
#     fold = int(sys.argv[2])

print("Script running experiments on", dataset)
x, t, e, covariates = datasets.load_dataset(dataset, competing=True)

# Hyperparameters
max_epochs = 500
grid_search = 3
layers = [[i] * (j + 1) for i in [25, 50] for j in range(4)]
layers_large = [[i] * (j + 1) for i in [25, 50] for j in range(8)]

batch = [100, 250] if dataset != 'SEER' else [1000, 5000]

timing_results = []  # List to store timing info

# --- DSM
param_grid = {
    'epochs': [max_epochs],
    'learning_rate': [1e-3, 1e-4],
    'batch': batch,
    'k': [2, 3, 4, 5],
    'distribution': ['LogNormal', 'Weibull'],
    'layers': layers_large,
}

for name, exp_class, target, extra_args in [
    ('dsm', DSMExperiment, (e == 1), {}),
    #('dsmnc', DSMExperiment, (e == 1), {})
]:
    start_time = time.time()
    exp_class.create(param_grid,n_iter=grid_search, path=f'Results/{dataset}/{dataset}_{name}', random_seed=random_seed, fold=fold).train(x, t, target, **extra_args)
    end_time = time.time()
    timing_results.append((name, end_time - start_time))

# --- NFG
param_grid = {
    'epochs': [max_epochs],
    'learning_rate': [1e-3, 1e-4],
    'batch': batch,
    'dropout': [0., 0.25, 0.5, 0.75],
    'layers_surv': layers,
    'layers': layers,
    'act': ['Tanh'],
}

for name, exp_class, target, extra_args in [
    ('nfg', NFGExperiment, (e == 1), {}),
    #('nfgnc', NFGExperiment, (e == 1), {}),
    #('nfgcs', NFGExperiment, e, {'cause_specific': True})
]:
    start_time = time.time()
    exp_class.create(param_grid,n_iter=grid_search, path=f'Results/{dataset}/{dataset}_{name}', random_seed=random_seed, fold=fold).train(x, t, target, **extra_args)
    end_time = time.time()
    timing_results.append((name, end_time - start_time))

# # --- NFG Monohead
# param_grid['multihead'] = [False]
# start_time = time.time()
# NFGExperiment.create(param_grid, n_iter=grid_search, path=f'Results/{dataset}/{dataset}_nfgmono', random_seed=random_seed, fold=fold).train(x, t, e)
# end_time = time.time()
# timing_results.append(('nfgmono', end_time - start_time))

# --- DeSurv
param_grid = {
    'epochs': [max_epochs],
    'learning_rate': [1e-3, 1e-4],
    'batch': batch,
    'embedding': [True],
    'layers_surv': layers,
    'layers': layers,
    'act': ['Tanh'],
}

for name, exp_class, target, extra_args in [
    ('ds', DeSurvExperiment, (e == 1), {}),
    #('dsnc', DeSurvExperiment, (e == 1), {})
]:
    start_time = time.time()
    exp_class.create(param_grid,n_iter=grid_search, path=f'Results/{dataset}/{dataset}_{name}', random_seed=random_seed, fold=fold).train(x, t, target, **extra_args)
    end_time = time.time()
    timing_results.append((name, end_time - start_time))

# --- DeepHit
param_grid = {
    'epochs': [max_epochs],
    'learning_rate': [1e-3, 1e-4],
    'batch': batch,
    'nodes': layers,
    'shared': layers,
}

for name, exp_class, target, extra_args in [
    ('dh', DeepHitExperiment, (e == 1), {}),
    #('dhnc', DeepHitExperiment, (e == 1), {})
]:
    start_time = time.time()
    exp_class.create(param_grid,n_iter=grid_search, path=f'Results/{dataset}/{dataset}_{name}', random_seed=random_seed, fold=fold).train(x, t, target, **extra_args)
    end_time = time.time()
    timing_results.append((name, end_time - start_time))

# --- Save timing results
timing_df = pd.DataFrame(timing_results, columns=['Model', 'Seconds'])
timing_save_path = f'{dataset}_training_times.csv'
timing_df.to_csv(timing_save_path, index=False)

print(f"Timing results saved to {timing_save_path}")
print(timing_df)