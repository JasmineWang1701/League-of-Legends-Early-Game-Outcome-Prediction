import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
import joblib
from Training import LogisticRegressionModel
import os
import numpy as np
import matplotlib.pyplot as plt


def compute_permutation_importance(model, X, y, device, metric_fn=roc_auc_score, n_repeats=10):
    baseline_score = metric_fn(y, model(torch.tensor(X, dtype=torch.float32).to(device)).detach().cpu().numpy().flatten())
    feature_importances = []

    for col in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, col])
            permuted_score = metric_fn(y, model(torch.tensor(X_permuted, dtype=torch.float32).to(device)).detach().cpu().numpy().flatten())
            scores.append(baseline_score - permuted_score)
        feature_importances.append(np.mean(scores))

    return feature_importances


def t_logistic_model(data, frame_group, model_dir="models/", target_col="hasWon", hidden_dims=[]):
    # file paths for model and scaler
    model_path = os.path.join(model_dir, f"logistic_model_frame_{frame_group}.pth")
    scaler_path = os.path.join(model_dir, f"logistic_model_frame_{frame_group}_scaler.pkl")

    # load the scaler
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print(f"Scaler not found for frame group {frame_group} at {scaler_path}")
        return None, None, None

    # prepare data
    X = data.drop(columns=[target_col])
    y = data[target_col]

    try:
        X = scaler.transform(X)
    except Exception as e:
        print(f"Error scaling data for frame group {frame_group}: {e}")
        return None, None, None

    # convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(device)

    # load the trained model
    input_dim = X.shape[1]
    model = LogisticRegressionModel(input_dim, hidden_dims).to(device)
    try:
        model.load_state_dict(torch.load(model_path,weights_only=True))
    except FileNotFoundError:
        print(f"Model not found for frame group {frame_group} at {model_path}")
        return None, None, None

    model.eval()
    print(f"Model loaded from {model_path}")

    # perform inference
    y_pred = model(X_tensor).detach().cpu().numpy().flatten()
    y_pred_class = (y_pred > 0.5).astype(int)
    y_true = y_tensor.cpu().numpy().flatten()

    # metrics
    cm = confusion_matrix(y_true, y_pred_class)
    cr = classification_report(y_true, y_pred_class)
    auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred_class)

    print(f"\nResults for frame group {frame_group}:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{cr}")
    print(f"AUC-ROC: {auc:.2f}")

    # compute feature importance using permutation importance
    feature_importances = compute_permutation_importance(model, X, y, device, metric_fn=roc_auc_score)

    # get top 3 features
    feature_importance_df = pd.DataFrame({
        "Feature": data.drop(columns=[target_col]).columns,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    print(f"Top 3 features for frame group {frame_group}:\n{feature_importance_df.head(3)}")

    return feature_importance_df, auc, accuracy


if __name__ == "__main__":
    file_path = "lol_ranked_games.csv"
    data = pd.read_csv(file_path)

    # drop unnecessary columns
    data = data.drop(columns=['gameId', 'gameDuration', 'expDiff'])

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
        "28_30": [28, 30],  # Combine frames 28 and 30
        "32+": list(range(32, data['frame'].max() + 1))  # Combine everything past frame 32
    }

    # metrics across all frame groups
    aucs = []
    accuracies = []
    frame_group_labels = []

    # analyze each frame group
    for frame_group, frames in frame_groups.items():
        frame_data = data[data['frame'].isin(frames)]

        if frame_data.empty:
            print(f"No data for frame group = {frame_group}")
            continue

        print(f"\nTesting and analyzing frame group = {frame_group}, Rows: {len(frame_data)}")
        _, auc, accuracy = t_logistic_model(
            frame_data,
            frame_group=frame_group,
            model_dir="models/",
            hidden_dims=[128, 64, 32]
        )

        if auc is not None and accuracy is not None:
            aucs.append(auc)
            accuracies.append(accuracy)
            frame_group_labels.append(frame_group)

    # plot AUC-ROC and Accuracy across frame groups
    plt.figure(figsize=(10, 6))
    plt.plot(frame_group_labels, aucs, label="AUC-ROC", color="red", marker='o')
    plt.plot(frame_group_labels, accuracies, label="Accuracy", color="blue", marker='o')
    plt.title("AUC-ROC and Accuracy Across Frame Groups")
    plt.xlabel("Frame Groups")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid()
    plt.savefig("metrics_across_frame_groups.png")
    plt.show()

    print("Metrics plot saved as metrics_across_frame_groups.png")
