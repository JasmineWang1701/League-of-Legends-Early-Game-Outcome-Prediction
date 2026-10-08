import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


# Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Train and Evaluate Logistic Regression
def train_evaluate_logistic(data, target_col='hasWon', test_size=0.2, lr=0.01, epochs=10):
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    device = torch.device("cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    # Initialize the model
    input_dim = X_train.shape[1]
    model = LogisticRegressionModel(input_dim).to(device)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    y_pred = model(X_test_tensor).detach().cpu().numpy().flatten()
    y_pred_class = (y_pred > 0.5).astype(int)
    y_test_np = y_test_tensor.cpu().numpy().flatten()

    # Metrics
    cm = confusion_matrix(y_test_np, y_pred_class)
    cr = classification_report(y_test_np, y_pred_class, output_dict=True)
    auc = roc_auc_score(y_test_np, y_pred)

    # Feature analysis
    feature_coefficients = model.linear.weight.detach().cpu().numpy().flatten()
    feature_odds_ratios = np.exp(feature_coefficients)
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": feature_coefficients,
        "Odds Ratio": feature_odds_ratios
    }).sort_values(by="Coefficient", key=lambda x: abs(x), ascending=False)

    return cm, cr, auc, feature_importance


# Main function to process frames and perform feature analysis
def process_frames_with_logistic_and_analysis(file_path, target_col='hasWon', frame_col='frame', test_size=0.2, lr=0.01,
                                              epochs=10):
    results = {}

    # Read the dataset
    data = pd.read_csv(file_path)

    # Drop unnecessary columns
    data = data.drop(columns=['gameId', 'gameDuration'])

    # Define frame groups
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
        "32+": list(range(32, data[frame_col].max() + 1))  # Combine everything past frame 32
    }

    for group_name, frames in frame_groups.items():
        # Filter data for the current group
        frame_data = data[data[frame_col].isin(frames)]

        if frame_data.empty:
            print(f"No data for frame group = {group_name}")
            continue

        print(f"\nProcessing frame group = {group_name}, Rows: {len(frame_data)}")

        # Train and evaluate the logistic regression model
        cm, cr, auc, feature_importance = train_evaluate_logistic(frame_data, target_col=target_col,
                                                                  test_size=test_size, lr=lr, epochs=epochs)

        results[group_name] = {
            "Confusion Matrix": cm,
            "Classification Report": cr,
            "AUC-ROC": auc,
            "Feature Importance": feature_importance
        }

    return results


# Run the process and generate the summarized report
file_path = "data.csv"
results_logistic = process_frames_with_logistic_and_analysis(file_path, target_col='hasWon', frame_col='frame', lr=0.01,
                                                             epochs=10)

# Display the summarized report
print("\nFinal Report for Logistic Regression with Feature Analysis:")
for frame_group, metrics in results_logistic.items():
    print(f"\nFrame Group {frame_group}:")
    print(f"AUC-ROC: {metrics['AUC-ROC']:.2f}")
    print(f"Confusion Matrix:\n{metrics['Confusion Matrix']}")
    print("Classification Report:")
    for key, value in metrics['Classification Report'].items():
        if isinstance(value, dict):  # For class-specific metrics
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value:.2f}")
        else:  # For overall metrics
            print(f"  {key}: {value:.2f}")
    print("\nFeature Importance:")
    print(metrics["Feature Importance"])
