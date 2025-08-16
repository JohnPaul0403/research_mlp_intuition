# Options Mispricing

## Introduction

This project investigates the prediction of mispricing in options markets using synthetic data and a Multi-Layer Perceptron (MLP) model. Options mispricing can lead to significant financial opportunities or risks, making its accurate detection critical for traders, risk managers, and financial institutions. However, real-world financial data is often noisy, incomplete, or influenced by unobservable factors, which complicates the study of mispricing. To address these challenges, we leverage synthetic datasets that simulate underlying asset prices and option prices under controlled conditions. This approach allows us to systematically analyze the behavior of pricing models and the effectiveness of machine learning techniques. Machine learning, particularly neural networks like MLPs, has shown promise in capturing complex, nonlinear relationships in financial data, making it a suitable candidate for identifying subtle patterns indicative of mispricing.

## Project Structure

The repository is organized into several key components to facilitate experimentation and reproducibility:

- `generate_dataset.py`: Responsible for generating synthetic datasets by simulating underlying asset prices using geometric Brownian motion (GBM) and calculating corresponding option prices with the Black-Scholes formula.
- `get_dataset.py`: Provides utilities for loading, preprocessing, and preparing datasets for training and evaluation.
- `engine.py`: Contains the main training loop and the implementation of the Multi-Layer Perceptron (MLP) model used for mispricing prediction.
- `utils.py`: Includes helper functions for data preprocessing, feature engineering, and evaluation metrics.
- `readme.MD`: This documentation file, providing an overview of the project, methodology, and results.

## Abstract

In this study, we explore the application of a Multi-Layer Perceptron (MLP) for predicting mispricing in options markets. To facilitate controlled experimentation, we generate synthetic data: underlying asset prices are simulated using geometric Brownian motion (GBM), and corresponding option prices are calculated using the Black-Scholes formula. The resulting dataset is used to train and evaluate the MLP model on the binary classification task of detecting mispriced options. Our approach achieves an accuracy of 0.64 and an Area Under the Curve (AUC) of 0.597, demonstrating the potential of neural networks in financial mispricing detection tasks. Despite these promising results, the model's performance indicates room for improvement, possibly through more sophisticated architectures or richer feature sets. Future work could involve extending the synthetic data generation to include more realistic market phenomena, incorporating alternative machine learning models, and validating findings on real-world financial datasets.
