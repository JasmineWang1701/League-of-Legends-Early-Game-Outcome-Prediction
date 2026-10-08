import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


# Function to train and evaluate the SVM model
def train_evaluate_svm(data, target_col='hasWon', test_size=0.2, random_state=42):
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize SVM model
    model = SVC(kernel='rbf', probability=True, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    return cm, cr, auc


# Main function to process frames and generate summarized reports for SVM
def process_frames_with_svm(file_path, target_col='hasWon', frame_col='frame', test_size=0.2, random_state=42):
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

        # Train and evaluate the SVM model on the current frame group
        cm, cr, auc = train_evaluate_svm(frame_data, target_col=target_col, test_size=test_size,
                                         random_state=random_state)
        results[group_name] = {
            "Confusion Matrix": cm,
            "Classification Report": cr,
            "AUC-ROC": auc
        }

    return results


# Run the process for SVM and generate the summarized report
file_path = "data.csv"
results_svm = process_frames_with_svm(file_path, target_col='hasWon', frame_col='frame', test_size=0.2, random_state=42)

# Display the summarized report
print("\nFinal Report for SVM:")
for frame_group, metrics in results_svm.items():
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
