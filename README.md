# SAVER: Stochastic Adaptive Variance-Driven Exploration and Reconstruction

This repository contains the official Python implementation of the SAVER framework for low-dose Computed Tomography (CT).

## Overview
SAVER is an adaptive acquisition framework that selects projection angles in real-time based on the statistical variance of acquired data to improve reconstruction fidelity per unit of radiation dose.

## Project Structure
- `saver_env.py`: Core math engines, Woodbury update, and ROI-SSIM evaluation.
- `phantoms.py`: Generation functions for 8 types of synthetic phantoms.
- `ct_methods.py`: Implementation of 8 sampling strategies (SAVER, Random, AIRS, Oracle-based).
- `run_all_experiments.py`: Script to reproduce the full experiment matrix (Noise levels x Annealing rates).
- `save_visualizations.py`: Script to generate SSIM curves and AUC boxplots.

## Requirements
- Python 3.8+
- numpy, scipy, scikit-image, matplotlib

## Usage
1. Run the experiments:
   ```bash
   python run_all_experiments.py
