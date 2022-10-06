# Validating Logistic Q-Learning
This is the code for my research internship at the ALR Institute at KIT.
The goal was to reimplement the experiments from Logistic Q-Learning and validate their results.


## Results
The repository contains an implementation of both REPS and QREPS.
We successfully reproduced the reported results and found an impact of the discount factor on REPS.
For a more in-depth discussion please refer to the report.


## Implementation Details
The ``qreps`` module contains the implementations of the algorithm and necessary extras.
The ``experiments`` module contains the script for running of the experiments.
Results are saved to W&B.


## Dependencies
Use Python3.7
Install all requirements into a virtual env
```bash
python -m python37 venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```
