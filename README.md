# CS 273A Final Project – QM9 Atomization Energy

This repo contains the code for our CS 273A final project (Group 23), where we predict molecular atomization energies for the QM9 dataset using several machine-learning models and feature representations.

We:

- Build **Morgan fingerprints** and **RDKit 2D descriptors** from QM9 SMILES.
- Train and tune:
  - **Random Forests** (fingerprints)
  - **Support Vector Regression (RBF)** (fingerprints, on subsets)
  - **Feed-forward neural networks** (fingerprints and descriptors)
  - **Symbolic regression** with **PySR** (RF-selected fingerprint bits)
- Evaluate models with RMSE/MAE and parity plots, and compare fingerprints vs. descriptors.

High level result: descriptor-based neural networks achieve the best performance by a large margin, while symbolic regression provides interpretable—but much less accurate—fragment-level formulas.

See `report/report.pdf` for full details, and the `src/` directory for the training scripts for each model.
