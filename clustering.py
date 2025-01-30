from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA, StringIndexer
from pyspark.ml.clustering import KMeans
import numpy as np
from pyspark.sql.functions import col, count, round, sum

# Initialize Spark Session
spark = SparkSession.builder.appName("SpeciesClusteringOptimized").getOrCreate()

# Load dataset
file_path = "inat-data-matrix-gdf.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Fill null values with 0
df = df.fillna(0)

# Convert veg_class to a categorical numerical variable
indexer = StringIndexer(inputCol="veg_class", outputCol="veg_class_index")
df = indexer.fit(df).transform(df)

# Select species-related columns
species_columns = [col for col in df.columns if col.startswith("species_")]

# Assemble species features into a single vector
vector_assembler = VectorAssembler(inputCols=species_columns, outputCol="features")
df_vectorized = vector_assembler.transform(df).select("vcm_label", "features")

# Randomly sample 10% of the data for faster clustering
df_sampled = df_vectorized.sample(fraction=0.1, seed=1).cache()

# Standardize the features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
scaler_model = scaler.fit(df_sampled)
df_scaled_sampled = scaler_model.transform(df_sampled).cache()

# Perform PCA with fewer components
pca = PCA(k=min(len(species_columns), 50), inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(df_scaled_sampled)

# Compute explained variance
explained_variance = np.cumsum(pca_model.explainedVariance.toArray())

# Find the number of components needed for 80% variance
num_components = np.argmax(explained_variance >= 0.8) + 1

# Re-run PCA with the optimal number of components
pca_optimal = PCA(k=num_components, inputCol="scaled_features", outputCol="pca_optimal_features")
df_reduced = pca_optimal.fit(df_scaled_sampled).transform(df_scaled_sampled)

# Cluster using K-Means with 2 clusters
kmeans = KMeans(k=2, seed=1, featuresCol="pca_optimal_features", predictionCol="cluster")
model = kmeans.fit(df_reduced)
df_clustered = model.transform(df_reduced)

# Compare clusters with vcm_label
df_comparison = df_clustered.select("vcm_label", "cluster")

# Compute summary statistics
print("\n=== Cluster Counts by VCM Label ===")
df_summary = df_comparison.groupBy("vcm_label", "cluster").agg(count("*").alias("count"))
df_summary.show()

# Compute normalized percentages within each vcm_label
print("\n=== Normalized Cluster Distribution by VCM Label ===")
df_total = df_comparison.groupBy("vcm_label").agg(count("*").alias("total"))
df_percent = df_summary.join(df_total, "vcm_label").withColumn("percentage", round(col("count") / col("total") * 100, 2))
df_percent.select("vcm_label", "cluster", "count", "percentage").show()

# Optionally, calculate entropy per cluster (measures label mixing)
from pyspark.sql.functions import log2
df_entropy = df_percent.withColumn("entropy_term", -(col("percentage") / 100) * log2(col("percentage") / 100))
df_entropy = df_entropy.groupBy("cluster").agg(sum("entropy_term").alias("entropy"))
print("\n=== Entropy Score (Lower is Better) ===")
df_entropy.show()
