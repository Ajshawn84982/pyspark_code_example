# pyspark_code_example
version: spark 3.3.1 hadoop 2.7 python 3.9 Java 17 pyspark 3.3.1
## pyspark regressor 
linear regression in pyspark LinearRegression model
## pyspark_classification 
classification by using pyspark random forest tree. The test data has 20 input feature and 5 output class. The test dataset is synthesized by "make_classification" from sklearn. The maximum number of tree iis 10 and the maximum depth is 10. It reports a error when using larger number of tree and increase the depth.<br>

<div style="display: flex; justify-content: space-around;">
<img src="pyspark_classification/con_matrix_random_forest.png" style="width: 45%;"/>
<img src="pyspark_classification/roc.png" style="width: 45%;"/>
</div>

## py_spark_MultilayerPerceptronClassifier 
multi-layer neural network by pyspark. The test data has 20 input feature and 5 output class with 1000000 rows. The test dataset is synthesized by "make_classification" from sklearn. The pyspark doesn't support the GPU learn on MultilayerPerceptronClassifier yet and it is hard to perform large NN.

<div style="display: flex; justify-content: space-around;">
<img src="py_spark_MultilayerPerceptronClassifier/con_matrix.png" style="width: 45%;"/>
<img src="py_spark_MultilayerPerceptronClassifier/roc.png" style="width: 45%;"/>
</div>

