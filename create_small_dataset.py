import pandas as pd

# Load the full dataset
data = pd.read_csv('data/sentiment140.csv', encoding='latin-1', header=None)
data.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Ensure the target column is binary (0 for negative, 1 for positive)
data['target'] = data['target'].replace(4, 1)  # Convert 4 to 1 (positive sentiment)

# Create a balanced smaller subset
# Take equal numbers of positive and negative samples
positive_samples = data[data['target'] == 1].head(5000)  # First 5000 positive samples
negative_samples = data[data['target'] == 0].head(5000)  # First 5000 negative samples

# Combine the samples
small_data = pd.concat([positive_samples, negative_samples])

# Shuffle the dataset to mix positive and negative samples
small_data = small_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the smaller dataset
small_data.to_csv('data/sentiment140_small.csv', index=False)

print("Balanced smaller dataset created and saved as 'data/sentiment140_small.csv'.")