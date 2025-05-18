# LSTM-MLP-nnLASSO
An L2O (Learning to Optimize) method based on LSTM-MLP for solving the non-negative LASSO problem.

## Overview
The core LSTM-MLP algorithm is adapted from the paper *"Towards Constituting Mathematical Structures for Learning to Optimize"* (Liu et al., 2023). This implementation extends the original unconstrained optimization framework to handle non-negative constrained LASSO problems, as detailed in the undergraduate thesis *"Extending L2O to Constrained Optimization: A Case Study on Non-Negative LASSO"* by Zhuoxuan Wu.

## Quick Start
1. Run the corresponding scripts in the `scripts` folder to execute experiments.
2. **Important Notes**:
   - Some code blocks (e.g., synthetic data generation, ground truth computation, model training) in different scripts may not need to be re-run. You can manually comment them out to avoid redundant computations.
   - The code uses **GBK encoding** by default. If you encounter garbled characters in printed outputs, consider resaving the files with **UTF-8 encoding**.
   - Adjust hyperparameters and experimental configurations by modifying or creating new `.yaml` files in the `configs` folder.
   - To run experiments on the **BSDS500 dataset**, download and extract the `BSDS500` folder to the project root directory.

## Contributions
We welcome contributions, improvements, and extensions to this codebase. Feel free to fork the repository, submit pull requests, or share your findings!

## Citation
If you use this code or the associated methodology in your research, please cite the original papers:
1. Liu et al. (2023). *Towards Constituting Mathematical Structures for Learning to Optimize*.
2. Wu, Z. (202X). *Extending L2O to Constrained Optimization: A Case Study on Non-Negative LASSO* [Undergraduate Thesis].
