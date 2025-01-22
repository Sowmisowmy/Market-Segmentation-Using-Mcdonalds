# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Load the dataset
mcdonalds = pd.read_csv("mcdonalds.csv")

# Inspect the data
print(mcdonalds.head())
print(mcdonalds.info())

# Check for missing values
print(mcdonalds.isnull().sum())

# Convert categorical variables to numerical using one-hot encoding
categorical_columns = mcdonalds.select_dtypes(include=['object']).columns
mcdonalds_encoded = pd.get_dummies(mcdonalds, columns=categorical_columns, drop_first=True)

# Standardize the data
scaler = StandardScaler()
mcdonalds_scaled = scaler.fit_transform(mcdonalds_encoded)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
mcdonalds['Cluster'] = kmeans.fit_predict(mcdonalds_scaled)

# Visualize cluster centers
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=mcdonalds_encoded.columns)
print(cluster_centers)

# Reduce dimensionality for visualization using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(mcdonalds_scaled)
mcdonalds['PCA1'] = pca_result[:, 0]
mcdonalds['PCA2'] = pca_result[:, 1]

# Scatter plot of clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(data=mcdonalds, x='PCA1', y='PCA2', hue='Cluster', palette='viridis')
plt.title("Clusters visualized in 2D using PCA")
plt.show()

# Hierarchical clustering
dendro_data = linkage(mcdonalds_scaled, method='ward')
plt.figure(figsize=(12, 8))
dendrogram(dendro_data)
plt.title("Dendrogram for Hierarchical Clustering")
plt.show()

# Mosaic plot for clusters and the "Like" variable
# Handle the "Like" column with additional parsing logic
def parse_like(value):
    if isinstance(value, str) and "!" in value:
        # Extract the numeric part after the exclamation mark
        num = value.split("!")[-1]
        try:
            return int(num)
        except ValueError:
            return None
    try:
        return int(value)  # Handle numeric values directly
    except ValueError:
        return None

# Apply the parsing function and compute Like_Num
like_mapped = mcdonalds['Like'].apply(parse_like)
mcdonalds['Like_Num'] = 6 - like_mapped  # Convert to the required scale

# Visualize the relationship between "Like" and clusters
sns.histplot(data=mcdonalds, x='Like_Num', hue='Cluster', multiple='stack', palette='viridis')
plt.title("Distribution of 'Like' across Clusters")
plt.xlabel("Like (Scaled)")
plt.ylabel("Count")
plt.show()
