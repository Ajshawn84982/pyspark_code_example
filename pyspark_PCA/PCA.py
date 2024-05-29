from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, PCA
import pandas as pd
import matplotlib.pyplot as plt
# Start a Spark session
spark = SparkSession.builder \
    .appName("PySpark PCA Example") \
    .getOrCreate()

# Load the dataset from CSV
data = spark.read.csv('C:/kaggle/pyspark_code_example/pyspark_classification/classification_dataset.csv', header=True, inferSchema=True)
data.show()

# Preprocess the data: Assume the last column is the target, and the rest are features
feature_columns = [col for col in data.columns if col != 'target']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

# Define the PCA model: Choose the number of principal components
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")

# Fit the PCA model
pca_model = pca.fit(data)

# Transform the data
pca_result = pca_model.transform(data)


# Stop the Spark session
spark.stop()