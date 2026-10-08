import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
import matplotlib.pyplot as plt


# define the Logistic Regression Model with hidden layers
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[]):
        super(LogisticRegressionModel, self).__init__()
        layers = []

        # input Layer
        prev_dim = input_dim

        # hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # output Layer
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.network(x))


# implement Permutation Importance
def compute_permutation_importance(model, X_val, y_val, metric_fn, device, n_repeats=10, random_state=42):
    rng = np.random.RandomState(random_state)
    baseline_score = metric_fn(y_val, model(torch.tensor(X_val, dtype=torch.float32).to(device)).detach().cpu().numpy().flatten())
    feature_importances = np.zeros(X_val.shape[1])

    for col in range(X_val.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X_val.copy()
            rng.shuffle(X_permuted[:, col])  # Permute feature
            score = metric_fn(y_val, model(torch.tensor(X_permuted, dtype=torch.float32).to(device)).detach().cpu().numpy().flatten())
            scores.append(baseline_score - score)
        feature_importances[col] = np.mean(scores)

    return feature_importances


# train with K-Fold Cross-Validation
def train_kfold_logistic_model(data, target_col, frame_group, model_dir="models/", k=5, lr=0.01, epochs=20, hidden_dims=[]):
    # separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # save the scaler
    scaler_path = os.path.join(model_dir, f"logistic_model_frame_{frame_group}_scaler.pkl")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # convert data to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(device)

    # K-Fold Cross-Validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    training_losses = []
    validation_losses = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_tensor)):
        # split data
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

        # initialize the model
        input_dim = X_train.shape[1]
        model = LogisticRegressionModel(input_dim, hidden_dims).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        fold_training_losses = []
        fold_validation_losses = []

        # training loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
            fold_training_losses.append(loss.item())

            # validation loop
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val)
                val_loss = criterion(val_predictions, y_val)
                fold_validation_losses.append(val_loss.item())

        training_losses.append(fold_training_losses)
        validation_losses.append(fold_validation_losses)

        # compute permutation importance only for the last fold
        if fold == k - 1:  # Last fold
            X_val_np = X_scaled[val_idx]
            y_val_np = y.iloc[val_idx].values

            feature_importances = compute_permutation_importance(
                model=model,
                X_val=X_val_np,
                y_val=y_val_np,
                metric_fn=roc_auc_score,
                device=device,
                n_repeats=10,
                random_state=42
            )

            # save feature importance
            feature_importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": feature_importances
            }).sort_values(by="Importance", ascending=False)

            feature_importance_path = os.path.join(model_dir, f"feature_importance_frame_{frame_group}.csv")
            feature_importance_df.to_csv(feature_importance_path, index=False)
            print(f"Feature importance for the last fold saved to {feature_importance_path}")

    # plot the loss curves
    plt.figure(figsize=(10, 6))
    for i, (train_loss, val_loss) in enumerate(zip(training_losses, validation_losses)):
        plt.plot(train_loss, label=f"Fold {i + 1} - Training Loss")
        plt.plot(val_loss, label=f"Fold {i + 1} - Validation Loss")
    plt.title(f"Training and Validation Loss for Frame Group {frame_group}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(model_dir, f"loss_curve_frame_{frame_group}.png"))
    print(f"Loss curve saved to {os.path.join(model_dir, f'loss_curve_frame_{frame_group}.png')}")

    # save the final model
    model_path = os.path.join(model_dir, f"logistic_model_frame_{frame_group}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return training_losses, validation_losses


if __name__ == "__main__":
    file_path = "lol_ranked_games.csv"
    data = pd.read_csv(file_path)

    # drop unnecessary columns
    data = data.drop(columns=['gameId', 'gameDuration','expDiff'])

    # define frame groups
    frame_groups = {
        "10": [10],
        "12": [12],
        "14": [14],
        "16": [16],
        "18": [18],
        "20": [20],
        "22": [22],
        "24": [24],
        "26": [26],
        "28_30": [28, 30],
        "32+": list(range(32, data['frame'].max() + 1))
    }

    for group_name, frames in frame_groups.items():
        frame_data = data[data['frame'].isin(frames)]

        if frame_data.empty:
            print(f"No data for frame group = {group_name}")
            continue

        print(f"\nProcessing frame group = {group_name}, Rows: {len(frame_data)}")

        model_dir = "models/"
        training_losses, validation_losses = train_kfold_logistic_model(
            frame_data, target_col="hasWon", frame_group=group_name, model_dir=model_dir, k=5, lr=0.01, epochs=20, hidden_dims=[128, 64, 32]
        )
