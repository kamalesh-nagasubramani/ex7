import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

ROLL_NUMBERS = "24BAD054"

sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

df = pd.read_csv('Mall_Customers.csv')

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- SCENARIO 1: K-MEANS CLUSTERING ---")

wcss = [] 
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='royalblue', linewidth=2)
plt.title('Elbow Method for Optimal K', fontsize=16)
plt.xlabel('Number of Clusters (K)', fontsize=14)
plt.ylabel('Inertia (WCSS)', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('kmeans_elbow_curve.png', dpi=300, bbox_inches='tight')
print("Elbow Curve saved as 'kmeans_elbow_curve.png'")

K_OPTIMAL = 5
kmeans = KMeans(n_clusters=K_OPTIMAL, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

inertia = kmeans.inertia_
sil_score_kmeans = silhouette_score(X_scaled, y_kmeans)
print(f"Optimal K selected: {K_OPTIMAL}")
print(f"Inertia (WCSS): {inertia:.2f}")
print(f"Silhouette Score (K-Means): {sil_score_kmeans:.4f}")

plt.figure()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
for i in range(K_OPTIMAL):
    plt.scatter(X_scaled[y_kmeans == i, 0], X_scaled[y_kmeans == i, 1], 
                s=100, c=colors[i], label=f'Cluster {i+1}', edgecolors='k', alpha=0.7)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='yellow', marker='*', label='Centroids', edgecolors='black')

plt.title('Customer Segments using K-Means', fontsize=16)
plt.xlabel('Annual Income (Scaled)', fontsize=14)
plt.ylabel('Spending Score (Scaled)', fontsize=14)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.4)
plt.savefig('kmeans_clusters.png', dpi=300, bbox_inches='tight')
print("K-Means Clusters visualization saved as 'kmeans_clusters.png'")

print("\n--- Cluster Interpretation ---")
df['KMeans_Label'] = y_kmeans
summary = df.groupby('KMeans_Label')[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']].mean().reset_index()
print(summary)
print("\nFinal Roll Numbers: ", ROLL_NUMBERS)
