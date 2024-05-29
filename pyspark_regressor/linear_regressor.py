# -*- coding: utf-8 -*-
"""
Created on Wed May 29 21:32:28 2024

@author: localadmin
"""


from pyspark.sql import SparkSession
from pyspark import SparkFiles
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator


#spark = SparkSession.builder.appName("Datacamp Pyspark Tutorial").config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size","10g").getOrCreate()



# Step 1: Create a Spark session
spark = SparkSession.builder.getOrCreate()
# Step 2: Read data from a CSV file
data = spark.read.csv('C:/kaggle/test_csv speed/arrays.csv', header=True, inferSchema=True)

assembler = VectorAssembler(
    inputCols=[f'col{i}' for i in range(1, 11)],
    outputCol="features")

data = assembler.transform(data)
final_data = data.select("features", "col11")

train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

lr = LinearRegression(featuresCol="features", labelCol="col11", predictionCol="predicted_col")
lr_model = lr.fit(train_data)
predictions = lr_model.transform(test_data)

evaluator = RegressionEvaluator(labelCol="col11", predictionCol="predicted_col", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data: {:.3f}".format(rmse))

evaluator_r2 = RegressionEvaluator(labelCol="col11", predictionCol="predicted_col", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)
print("R-squared (R2) on test data: {:.3f}".format(r2))