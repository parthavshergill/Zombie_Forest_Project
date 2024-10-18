from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from tqdm import tqdm

def visualize_clusters(dataset):
  # Scatter plot of species by clusters
    plt.scatter(dataset['decimallongitude'], dataset['decimallatitude'], c=dataset['cluster'], cmap='viridis')
    plt.title('Geographical Clusters of Species')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()  

# Returns a pandas series that can be used as a new 'cluster' col in the  dataset
def k_means(dataset, k):
    # Apply KMeans to the lat/long data to create clusters
    kmeans = KMeans(n_clusters=5)
    return kmeans.fit_predict(dataset[['decimallatitude', 'decimallongitude']])

# Method to do k-means clustering on latitude and longitude 
def identify_best_k(dataset):
    # Define a range of k values to try
    k_values = range(2, 11)  # Try k values from 2 to 10

    # Initialize lists to store the inertia and silhouette scores for each k
    inertia_values = []
    silhouette_scores = []

    # Loop through each value of k
    print("Trying different values...")
    for k in tqdm(k_values):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(dataset[['decimallatitude', 'decimallongitude']])
        
        # Store the inertia (sum of squared distances to nearest cluster center)
        inertia_values.append(kmeans.inertia_)
        
        # Compute silhouette score (how similar points are within a cluster)
        silhouette_avg = silhouette_score(dataset[['decimallatitude', 'decimallongitude']], cluster_labels)
        silhouette_scores.append(silhouette_avg)

    print("Plotting elbow method results")
    # Plot the Elbow Method results (inertia vs k)
    plt.figure(figsize=(20, 15))
    plt.plot(k_values, inertia_values, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (within-cluster sum of squares)')
    plt.show()

    print("Plotting silhouette score results")
    # Plot Silhouette Scores vs k
    plt.figure(figsize=(20, 15))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title('Silhouette Score for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.show()
