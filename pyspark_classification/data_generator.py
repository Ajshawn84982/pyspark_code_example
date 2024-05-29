# -*- coding: utf-8 -*-
"""
Created on Wed May 29 22:08:47 2024

@author: localadmin
"""

from sklearn.datasets import make_classification
import pandas as pd

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000000,    # Number of samples
                           n_features=20,     # Number of features
                           n_informative=15,  # Number of informative features
                           n_redundant=5,     # Number of redundant features
                           n_classes=5,       # Number of classes
                           random_state=42)   # Seed for reproducibility

# Convert to pandas DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
df['target'] = y

# Save to CSV
df.to_csv('classification_dataset.csv', index=False)

print("Dataset generated and saved to 'classification_dataset.csv'")