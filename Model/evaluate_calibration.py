
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score
from Training import LogisticRegressionModel


RANDOM_STATE = 42
TEST_SIZE = 0.20
EPOCHS = 20
LEARNING_RATE = 0.01
HIDDEN_DIMS = [128, 64, 32]
N_BINS = 10

DATA_PATH = "lol_ranked_games.csv"
OUTPUT_DIR = "calibration_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

data = pd.read_csv(DATA_PATH)

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
    "32+": list(range(32, int(data["frame"].max()) + 1)),
}

results = []

for group_name, frames in frame_groups.items():
    frame_data = data[data["frame"].isin(frames)].copy()
    frame_data = frame_data.dropna(subset=["hasWon", "gameId"])
    frame_data = frame_data.reset_index(drop=True)

    if frame_data.empty:
        continue

    # Keep every snapshot from a match in the same split.
    X = frame_data.drop(
        columns=["hasWon", "gameId", "gameDuration", "expDiff"],
        errors="ignore",
    )
    y = frame_data["hasWon"].astype(int)
    match_ids = frame_data["gameId"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(
        splitter.split(X, y, groups=match_ids)
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Fit preprocessing on training data only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(
        X_train, dtype=torch.float32, device=device
    )
    y_train_t = torch.tensor(
        y_train.to_numpy(), dtype=torch.float32,
        device=device
    ).view(-1, 1)

    X_test_t = torch.tensor(
        X_test, dtype=torch.float32, device=device
    )

    model = LogisticRegressionModel(
        input_dim=X_train.shape[1],
        hidden_dims=HIDDEN_DIMS,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        probabilities = model(X_train_t)
        loss = criterion(probabilities, y_train_t)

        loss.backward()
        optimizer.step()

    # Get probabilities for held-out matches.
    model.eval()
    with torch.no_grad():
        y_prob = model(X_test_t).cpu().numpy().ravel()

    y_true = y_test.to_numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    # Probability and classification metrics.
    brier = brier_score_loss(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    # Quantile bins aim to give each bin a similar sample count.
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=N_BINS,
        strategy="quantile",
    )

    # Reliability diagram.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        prob_pred, prob_true,
        marker="o", label="Model",
    )
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", label="Perfect calibration",
    )
    ax.set(
        xlabel="Mean predicted win probability",
        ylabel="Observed win rate",
        title=f"Reliability Diagram — Frame Group {group_name}",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            OUTPUT_DIR, f"reliability_{group_name}.png"
        ),
        dpi=150,
    )
    plt.close(fig)

    results.append({
        "frame_group": group_name,
        "test_matches": match_ids.iloc[test_idx].nunique(),
        "test_rows": len(test_idx),
        "accuracy": accuracy,
        "roc_auc": auc,
        "brier_score": brier,
    })

    print(
        f"Frame {group_name}: "
        f"Accuracy={accuracy:.3f}, "
        f"ROC-AUC={auc:.3f}, "
        f"Brier={brier:.4f}"
    )

results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "calibration_metrics.csv"),
    index=False,
)
print(results_df)