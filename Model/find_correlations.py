import pandas as pd

# Load the dataset
file_path = "lol_ranked_games.csv"
data = pd.read_csv(file_path)

# Preprocessing: Drop unnecessary columns
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
    "32+": list(range(32, data['frame'].max() + 1))  # Combine everything past frame 32
}

# Generate and store statistics for each frame group
frame_descriptions = {}
for group_name, frames in frame_groups.items():
    frame_data = data[data['frame'].isin(frames)]
    if frame_data.empty:
        print(f"No data for frame group {group_name}")
        continue

    # Drop the 'frame' column since it's not a feature
    features_only = frame_data.drop(columns=['frame', 'hasWon'])
    stats = features_only.describe()  # Summary statistics for each feature
    frame_descriptions[group_name] = stats

    # Save results as CSV for review if needed
    stats.to_csv(f"frame_{group_name}_description.csv")
    print(f"Summary statistics for frame group {group_name} saved to frame_{group_name}_description.csv.")