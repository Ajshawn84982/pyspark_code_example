# -*- coding: utf-8 -*-
"""
Created on Wed May 29 22:56:11 2024

@author: localadmin
"""
from pyspark.sql import SparkSession

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
spark = SparkSession.builder \
    .appName("PySpark Random Forest Classification") \
    .getOrCreate()
spark.sparkContext.setLogLevel("INFO")
# Load the dataset from CSV
data = spark.read.csv('C:/kaggle/pyspark_code_example/pyspark_classification/classification_dataset.csv', header=True, inferSchema=True)

# Assemble feature columns into a single vector column
feature_columns = [col for col in data.columns if col != 'target']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

# Convert the target column to numeric if it's not already
indexer = StringIndexer(inputCol="target", outputCol="label")
data = indexer.fit(data).transform(data)

# Select the columns needed for modeling
data = data.select("features", "label")
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)


layers = [len(feature_columns), 64, 32,5]

# Initialize the Multilayer Perceptron Classifier
mlp = MultilayerPerceptronClassifier(featuresCol="features", labelCol="label", maxIter=100, layers=layers, blockSize=128, seed=1234)

# Train the model
mlp_model = mlp.fit(train_data)

# Make predictions on the test data
predictions = mlp_model.transform(test_data)

# Show predictions
predictions.select("features", "label", "prediction").show()

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Evaluate the model
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy = {accuracy}")

# Stop the Spark session
spark.stop()