SCENARIO 1 – CLUSTERING USING K-MEANS

Problem Statement

Group customers/data points into clusters based on similarity using K-Means clustering.

Dataset (Kaggle – Public)

Mall Customer Segmentation Dataset Dataset Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python


Target Variable: Cluster labels (no predefined labels – unsupervised learning)

Input Feature:

• Annual Income

• Spending Score

• Age (optional)


IN-LAB TASKS

1. Import required Python libraries (NumPy, Pandas, Matplotlib, Scikit-learn)

2. Load the dataset

3. Perform data preprocessing (handling missing values, scaling)

4. Select relevant features

5. Use Elbow Method to determine optimal K

6. Apply K-Means clustering

7. Assign cluster labels

8. Visualize clusters

9. Interpret cluster characteristics


Evaluation Metrics

• Inertia (Within-cluster sum of squares) • Silhouette Score


Analysis Tasks

• Observe how different K values affect clustering • Analyze compactness and separation of clusters • Identify customer segments (e.g., high income–high spending) • Study sensitivity to initialization


Visualization

• Elbow curve (K vs inertia) • Scatter plot of clusters • Cluster centroids


SCENARIO 2 – CLUSTERING USING GMM

Problem Statement

Cluster data using Gaussian Mixture Models to capture probabilistic cluster membership.

Dataset (Same / Alternative Dataset)

Same dataset (or any numerical dataset)

Target Variable: Cluster probabilities and labels

Input Features

• Annual Income

• Spending Score

• Age (optional)

IN-LAB TASKS

1. Load dataset

2. Perform preprocessing and scaling

3. Apply Gaussian Mixture Model (GMM)

4. Choose number of components (clusters)

5. Fit model using Expectation-Maximization (EM)

6. Predict cluster probabilities

7. Assign clusters based on highest probability

8. Compare with K-Means clustering


Evaluation Metrics

• Log-Likelihood

• AIC (Akaike Information Criterion)

• BIC (Bayesian Information Criterion)

• Silhouette Score

Analysis Tasks

• Compare soft vs hard clustering

• Analyze overlapping clusters

• Observe cluster shapes (elliptical vs spherical)

• Compare flexibility with K-Means

Visualization

• Cluster probability distribution

• GMM contour plots

• Comparison plot (K-Means vs GMM)

SUBMISSION REQUIREMENTS

• Python code (with student roll numbers) • Screenshots of execution • Output graphs and visualizations • GitHub repository link
