# League of Legends Early-Game Outcome Prediction

A machine learning project investigating how early match data can predict League of Legends outcomes and which in-game features are most influential in determining a win.

## Overview

This project analyzes approximately **25,000 ranked matches and 240,000 gameplay frames** to investigate how the ability to predict match outcomes changes as a game progresses.

Using gameplay data from Platinum-tier solo queue matches, the project evaluates models at different time frames to identify when predictions become more reliable. It also examines feature importance to understand which gameplay metrics contribute most to predicted outcomes.

The analysis focuses on the blue team's probability of winning, using `hasWon` as the binary target variable.

## Key Results

* **83% accuracy** and **0.91–0.92 ROC-AUC** in later-game frame groups.
* Approximately **72% accuracy** and **0.80 ROC-AUC** at the 10-minute frame.
* Predictive performance generally improved as more gameplay information became available.
* Gold difference (`goldDiff`) and champion-level difference (`champLevelDiff`) consistently ranked among the most influential predictors.

These results demonstrate how the availability of in-game information affects predictive performance. Later-game results should not be interpreted as equivalent to early-game prediction performance.

## Methodology

### 1. Data Preprocessing and Feature Engineering

* Processed a dataset containing approximately 55 gameplay features across roughly 240,000 frames.
* Removed irrelevant identifiers and fields, including `gameId` and `gameDuration`.
* Grouped observations by game time and combined sparse late-game frames into broader intervals.
* Combined selected related metrics to reduce feature dimensionality.
* Standardized numerical features using `StandardScaler`.
* Examined feature correlations to assess potential multicollinearity.

### 2. Model Development

The project began with **L1-regularized logistic regression**, selected for binary classification and interpretable coefficients. The approach was subsequently extended to a neural network to capture nonlinear relationships between gameplay features and match outcomes.

The neural network used three hidden layers of 128, 64, and 32 units with ReLU activations, followed by a sigmoid output layer. Training used binary cross-entropy loss and the Adam optimizer.

### 3. Model Evaluation

* Used an 80/20 training-test split and five-fold cross-validation.
* Evaluated classification performance using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC.
* Monitored training and validation loss to assess convergence and potential overfitting.
* Compared model performance across frame groups to study how predictive ability changed over the course of a match.

### 4. Feature Importance and Interpretation

Permutation importance was used to identify influential features in the nonlinear model. Gold difference and champion-level difference were consistently important, while deaths, turret destruction, and neutral objectives became more prominent in later-game predictions.

The project also explored preprocessing and model-design alternatives, including removing redundant features, comparing scaling methods, increasing training epochs, and tuning learning rates and batch sizes.

## Technologies

* **Language:** Python
* **Data Analysis:** NumPy, Pandas
* **Machine Learning:** Scikit-learn, PyTorch
* **Visualization:** Matplotlib, Seaborn
* **Methods:** Feature Engineering, Logistic Regression, Neural Networks, L1 Regularization, Cross-Validation, Permutation Importance, ROC-AUC Analysis

## Dataset

Y. L. Ma, [League of Legends SoloQ Ranked Games Dataset](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-soloq-ranked-games), Kaggle.

The dataset contains ranked solo queue matches from the Platinum tier, collected from Riot Games API data. The project uses gameplay frames recorded at regular intervals beginning at approximately 10 minutes into each match.

## Limitations

Predictive performance was lower in early-game frames, when less information about the match was available. The model was also developed using a specific League of Legends dataset and feature set, limiting its direct applicability to other games or competitive environments.

This project was developed as an individual academic project covering data preprocessing, feature engineering, model development, evaluation, and interpretation.
