import numpy as np

# NOTE: as you increase the threshold paramter it decreases sensitivity to the outliers (and vice versa)

# Generate a sample dataset with some outliers
data = np.random.normal(loc=50, scale=10, size=100)
data[0] = 100
data[1] = 20
data[2] = 10

# Define a function to remove outliers using Z-Score method
def remove_outliers_zscore(data, threshold=2):
    z_scores = (data - np.mean(data)) / np.std(data)
    abs_z_scores = np.abs(z_scores)
    filtered_data = data[abs_z_scores < threshold]
    removed_data = data[abs_z_scores >= threshold]
    return filtered_data, removed_data

# Print the original dataset and its mean and standard deviation
print("Original Data: ", data)
print("Mean: ", np.mean(data))
print("Standard Deviation: ", np.std(data))

# Remove outliers using Z-Score method
filtered_data, removed_data = remove_outliers_zscore(data)

# Print the filtered dataset and its mean and standard deviation
print("Filtered Data: ", filtered_data)
print("Mean: ", np.mean(filtered_data))
print("Standard Deviation: ", np.std(filtered_data))

# Print the removed data
print("Removed Data: ", removed_data)

