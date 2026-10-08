import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "lol_ranked_games.csv"
data = pd.read_csv(file_path)

# Drop unnecessary columns
data = data.drop(columns=['gameId', 'gameDuration'], errors='ignore')  # Adjust as needed

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(20, 15))  # Adjust size as needed
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Feature Correlation Heatmap")

# Save the heatmap to a file
output_path = "correlation_heatmap.png"  # Change file extension to .pdf for PDF
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save as high-quality image
print(f"Heatmap saved as {output_path}")
