# Modelling-and-Simulation-of-battery-grouping-algorithm
Project Thesis

This repository contains the full research implementation for **"Modelling and Optimization of Large-Scale Battery Cell Grouping Using Multiple Algorithmic Frameworks"**. The Battery Cell Grouping Problem (BCGP) is a combinatorial optimization problem arising in lithium-ion battery pack assembly: given a population of *N* freshly produced cells, the goal is to sort them into *K* matched modules such that the within-group variance of key electrochemical parameters — capacity (C), internal resistance (DCIR), and open-circuit voltage (OCV) — is minimized.
 
Poor cell matching leads to accelerated degradation, capacity imbalance, and thermal runaway risk at the pack level. Optimal grouping is therefore critical for gigafactory end-of-line quality assurance. Because the problem is **NP-hard** (reducible from 3-PARTITION), exact solvers become intractable at production scale (N > 10⁴), motivating the suite of heuristic, metaheuristic, and machine-learning approaches implemented here.
 
The framework covers the full pipeline:
 
```
Raw / Synthetic Cell Data
        │
        ▼
   Data Loading & Preprocessing
        │
        ▼
  Clustering / Grouping Algorithms
  (FCM · K-Means · CDM+KNN · PCA+SOM)
        │
        ▼
   Benchmarking & Comparative Analysis
```


## 1 · Synthetic Data Generator
 
> **Directory:** `synthetic_data_generator/`
 
Real cell-parameter datasets suitable for large-scale grouping research are scarce — most public repositories (MATR, CALCE, NASA PCoE) contain fewer than 200 cells with incomplete parameter coverage. The synthetic data generator addresses this gap by producing statistically valid, physically consistent cell populations of arbitrary size (N up to 10⁵+).
config.py -----> contains all the hyperparameters and prerequisites.
data_loader.py------> this loads the data to the timegan.
generate.py-----> pre-functional and static parameters 
timegan.py------> process the timeseries data and trains through the Embedor, supervisor, recover, and Joint adversarial training saves the output files in a                         different location.
preprocessor.py---> Cleans and preprocesses the raw data.

 
---
