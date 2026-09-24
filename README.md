# League of Legends Early-Game Outcome Prediction

Machine learning models for predicting League of Legends match outcomes from early-game gameplay data.

## Overview

This project investigates how well early-game information can predict the eventual outcome of a League of Legends match.

The dataset contains approximately 25,000 ranked matches and over 240,000 game frames, with 55 features describing the state of the game.

Several machine learning approaches were implemented and compared, including logistic regression, support vector machines (SVM), and neural networks.

Data courtesy of Y. L. Ma, “League of legends solo-Q ranked games,” Kaggle, https://www.kaggle.com/datasets/bobbyscience/league-of-legends-soloq-ranked-games (accessed Nov. 19, 2024).

## Results

- 83% accuracy
- 0.92 AUC-ROC for mid-game outcome prediction

Model performance was evaluated at different points in the game to examine how predictive information develops throughout a match.

## Methods

- Data preprocessing and feature engineering
- Logistic regression with L1 regularization
- Support Vector Machines
- Neural networks
- 5-fold cross-validation
- AUC-ROC and accuracy evaluation
- Feature and odds-ratio analysis
- Data visualization

## Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Matplotlib
