# PIDDPM-for-ACOPF

## Overview

This repository contains a simplified implementation accompanying the paper:  
**"A Physics-Informed and Score-Based Diffusion Approach for Enhanced AC-OPF Solving"**.  

Due to ongoing updates on TechRxiv, we are unable to update our previous version, *"Learning the Direction Fields of the ACOPF Solution Space: A Score-Based Diffusion Approach"*. Therefore, we have published the latest version on GitHub and will keep it updated until TechRxiv permits further updates.

Because of GitHub’s 25 MB file size limit, the dataset is not included in this repository. However, users can generate their own data by running the `Data_Generator.jl` script.

## Repository Contents

- **`Data_Generator.jl`** – Generates the training and test datasets for the AC Optimal Power Flow (ACOPF) problem.  
- **`Distribution_Display.py`** – Visualizes the distribution of the generated data.  
- **`Check_ACPF_Balance.py`** – Verifies the correctness of the generated data by checking power flow balance.  
- **`PIDDPM-ACOPF_Solver-torch.py`** – Main training and evaluation script. Running this will produce results for the IEEE 118-bus system similar to those reported in the paper.  
  > **Note:** The data distribution used in this code may differ slightly from the original paper, so the obtained ACOPF cost may vary marginally.

## Requirements

The code is written in Python (with PyTorch) and Julia (for data generation). Please ensure you have the necessary dependencies installed (e.g., PyTorch, NumPy, Matplotlib, and Julia with appropriate packages).
