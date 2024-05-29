# -*- coding: utf-8 -*-
"""
Created on Wed May 29 22:14:40 2024

@author: localadmin
"""
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Start a Spark session
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
data.show()

# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# Initialize the Random Forest classifier
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10,maxDepth=5)

# Train the model
rf_model = rf.fit(train_data)

# Make predictions
predictions = rf_model.transform(test_data)
predictions.select("features", "label", "prediction", "probability").show()

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy = {accuracy}")

# Stop the Spark session
spark.stop()