import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
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

print("\n--- SCENARIO 2: GAUSSIAN MIXTURE MODELS ---")

n_components = np.arange(1, 11)
models = [GaussianMixture(n, covariance_type='full', random_state=42).fit(X_scaled) for n in n_components]

plt.figure()
plt.plot(n_components, [m.bic(X_scaled) for m in models], label='BIC', marker='s', color='#E67E22')
plt.plot(n_components, [m.aic(X_scaled) for m in models], label='AIC', marker='^', color='#2E86C1')
plt.legend(loc='best')
plt.xlabel('Number of Components', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.title('Model Selection (AIC & BIC)', fontsize=16)
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('gmm_aic_bic.png', dpi=300, bbox_inches='tight')
print("GMM Model Selection saved as 'gmm_aic_bic.png'")

K_OPTIMAL = 5
gmm = GaussianMixture(n_components=K_OPTIMAL, covariance_type='full', random_state=42)
gmm.fit(X_scaled)
y_gmm = gmm.predict(X_scaled)
probs = gmm.predict_proba(X_scaled) 

sil_score_gmm = silhouette_score(X_scaled, y_gmm)
log_likelihood = gmm.score(X_scaled) * len(X_scaled)
print(f"Log-Likelihood: {log_likelihood:.2f}")
print(f"AIC: {gmm.aic(X_scaled):.2f}")
print(f"BIC: {gmm.bic(X_scaled):.2f}")
print(f"Silhouette Score (GMM): {sil_score_gmm:.4f}")

plt.figure()
for i in range(K_OPTIMAL):
    plt.scatter(X_scaled[y_gmm == i, 0], X_scaled[y_gmm == i, 1], 
                s=100, label=f'Cluster {i+1}', alpha=0.6, edgecolors='w')

x = np.linspace(X_scaled[:, 0].min()-1, X_scaled[:, 0].max()+1, 100)
y = np.linspace(X_scaled[:, 1].min()-1, X_scaled[:, 1].max()+1, 100)
XX, YY = np.meshgrid(x, y)
ZZ = -gmm.score_samples(np.stack([XX.ravel(), YY.ravel()], axis=1))
ZZ = ZZ.reshape(XX.shape)

plt.contour(XX, YY, ZZ, levels=10, linewidths=1, colors='gray', alpha=0.5)
plt.title('Customer Segments using GMM (with Probability Contours)', fontsize=16)
plt.xlabel('Annual Income (Scaled)', fontsize=14)
plt.ylabel('Spending Score (Scaled)', fontsize=14)
plt.legend()
plt.savefig('gmm_clusters.png', dpi=300, bbox_inches='tight')
print("GMM Clusters visualization saved as 'gmm_clusters.png'")

print("\nFinal Roll Numbers: ", ROLL_NUMBERS)
