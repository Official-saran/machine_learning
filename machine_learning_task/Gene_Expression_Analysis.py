10. Gene Expression Analysis  
Scenario: In a biological study, researchers have collected gene expression data from thousands of genes across multiple samples.  
Question: How would PCA help in identifying key patterns or variations in gene expression? What challenges might arise when interpreting the principal components?


import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the CSV file with gene expression data
file_path = "gene_expression_data.csv"
gene_expression_df = pd.read_csv(file_path, index_col=0)

# Extract the numerical data for PCA
data = gene_expression_df.values

# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components
principal_components = pca.fit_transform(data)

# Create a DataFrame for the principal components
pca_df = pd.DataFrame(principal_components, columns=["PC1", "PC2"], index=gene_expression_df.index)

# Variance explained by each principal component
explained_variance = pca.explained_variance_ratio_

# Print explained variance for PC1 and PC2
print(f"Explained Variance by PC1: {explained_variance[0]:.2f}")
print(f"Explained Variance by PC2: {explained_variance[1]:.2f}")

# Plot PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_df["PC1"], pca_df["PC2"], color="skyblue", alpha=0.7)
plt.title("PCA of Gene Expression Data", fontsize=16)
plt.xlabel(f"PC1 ({explained_variance[0]*100:.2f}% Variance)", fontsize=12)
plt.ylabel(f"PC2 ({explained_variance[1]*100:.2f}% Variance)", fontsize=12)
plt.grid(alpha=0.3)
plt.show()
