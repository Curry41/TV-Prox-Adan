# TV Prox-Adan: Robust Tensor Completion under Extremely Low Sampling Ratios

This repository contains the MATLAB implementation for the paper **“TV Prox-Adan: An Accelerated Algorithm for Robust Tensor Completion under Extremely Low Sampling Ratios.”** The proposed method combines CP decomposition, total variation (TV) regularization, and the Adan optimizer to achieve robust and efficient tensor completion, especially under extremely low sampling conditions.

The core implementation is provided in `TV_Prox_Adan.m`. For evaluation, the repository includes two types of testing scripts. The `*_single.m` scripts are designed for testing a single tensor at a single sampling ratio, which is useful for quick verification or debugging. In contrast, the `*_dir.m` scripts are used to process multiple tensors from a directory, repeat experiments multiple times, and compute averaged performance metrics. The latter is recommended for reproducing the experimental results reported in the paper.

Due to size limitations, the datasets are not included in this repository. Please download the data from [the provided Google Drive link](https://drive.google.com/drive/folders/1tCaPUI3V84cjpq5VMttn7zAIF-amFMfJ?usp=share_link) and place them in your local directory before running the experiments. You may need to update the dataset paths in the test scripts accordingly.

This code is intended for academic research purposes.
