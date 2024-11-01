import pandas as pd
import matplotlib.pyplot as plt

# Load the data with zombie forest proportions
data = pd.read_csv('outputs/zombie_forest_cooccurrence_proportion_frequencies.csv')  # Replace with your file path

# Sort data by zombie forest proportion in descending order
data_sorted = data.nlargest(50, 'zombie_forest_proportion')

# Plot the proportions
plt.figure(figsize=(12, 8))
plt.barh(data_sorted['species'], data_sorted['zombie_forest_proportion'], color='teal')
plt.xlabel('Proportion of Observations in Zombie Forests')
plt.ylabel('Species')
plt.title('Species Proportion of Observations in Zombie Forests (Descending)')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.tight_layout()
plt.show()