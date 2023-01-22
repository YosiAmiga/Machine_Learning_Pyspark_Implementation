from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer,IndexToString
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from pyspark.sql.functions import cast
import matplotlib.pyplot as plt

################### DATASET CLEANING SECTION ###################

# Start a Spark session
spark = SparkSession.builder.appName("Random Forest Example").getOrCreate()

# Load the dataset
df = spark.read.text("adult.data")
test_data = spark.read.text("adult.test")


def dataframeCleaning(df):
    # Split the values in the first column based on the "," delimiter
    df = df.select(split(df.value, ", ").alias("features"))
    # Get the number of columns
    num_of_cols = len(df.select("features").first()[0])
    # Create the new columns' names
    col_names = ["col" + str(i) for i in range(num_of_cols)]
    # Create the new columns
    for i, col_name in enumerate(col_names):
        df = df.withColumn(col_name, df["features"][i])
    # drop the old column
    df = df.drop("features")
    # Rename the columns to the desired names
    df = df.withColumnRenamed("col0", "age") \
        .withColumnRenamed("col1", "workclass") \
        .withColumnRenamed("col2", "fnlwgt") \
        .withColumnRenamed("col3", "education") \
        .withColumnRenamed("col4", "education-num") \
        .withColumnRenamed("col5", "marital-status") \
        .withColumnRenamed("col6", "occupation") \
        .withColumnRenamed("col7", "relationship") \
        .withColumnRenamed("col8", "race") \
        .withColumnRenamed("col9", "sex") \
        .withColumnRenamed("col10", "capital-gain") \
        .withColumnRenamed("col11", "capital-loss") \
        .withColumnRenamed("col12", "hours-per-week") \
        .withColumnRenamed("col13", "native-country") \
        .withColumnRenamed("col14", "Yearly-income(Label)")

    # Set all the values of '?' to be None, it is a garbage value
    df = df.replace("?", None)

    # Each non numeric column value should be represented as a double value in the final dataset
    features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
                "native-country"]
    for feature in features:
        # Create a StringIndexer object
        indexer = StringIndexer(inputCol=feature, outputCol=feature + "_index", handleInvalid='keep')
        # Fit the indexer on your dataset and transform the data
        indexer_model = indexer.fit(df)
        df = indexer_model.transform(df)
        df = df.drop(feature)
        df = df.withColumnRenamed(feature + "_index", feature)

    # convert the value column from string to double
    df = df.withColumn("age", col("age").cast('double'))
    df = df.withColumn("fnlwgt", col("fnlwgt").cast('double'))
    df = df.withColumn("education-num", col("education-num").cast('double'))
    df = df.withColumn("capital-gain", col("capital-gain").cast('double'))
    df = df.withColumn("capital-loss", col("capital-loss").cast('double'))
    df = df.withColumn("hours-per-week", col("hours-per-week").cast('double'))

    indexer = StringIndexer(inputCol="Yearly-income(Label)", outputCol="label")
    indexer_model = indexer.fit(df)
    df = indexer_model.transform(df)
    df = df.drop("Yearly-income(Label)")
    df.show(30)
    return df


# Clean and set up data for model creation
df = dataframeCleaning(df)
test_data = dataframeCleaning(test_data)

# pyspark.sql.utils.IllegalArgumentException: requirement failed:
# DecisionTree requires maxBins (= 32) to be at least as large as the number of values in each categorical feature,
# but categorical feature 13 has 42 values.
# Consider removing this and other categorical features with a large number of values, or add more training examples.
# df = df.drop("native-country")
# df = df.drop("native-country")
feature_names = df.columns

################### MODEL CREATION SECTION ###################


# A. Use RandomForestClassifier to build a classification model on the training data.
# Tune the hyperparameters numTrees, subsamplingRate, and featureSubsetStrategy.

#################### Create the model and hyperparameters #####################

# # Split the data into training and test sets
# (trainingData, testData) = data.randomSplit([0.7, 0.3])
trainingData = df
testData = test_data
# # Create a vector assembler to combine all feature columns into a single vector column
assembler = VectorAssembler(inputCols=trainingData.columns[:-1], outputCol="features")

# Create the Random Forest Classifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features",maxBins=100)

# Create a pipeline to combine the vector assembler and the classifier
pipeline = Pipeline(stages=[assembler, rf])

# Create a param grid builder to specify the range of values for the hyperparameters
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 50, 100]) \
    .addGrid(rf.subsamplingRate, [0.5, 0.7, 1.0]) \
    .addGrid(rf.featureSubsetStrategy, ["auto", "sqrt", "log2"]) \
    .build()


# Create a cross-validator to tune the hyperparameters
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=MulticlassClassificationEvaluator(), numFolds=3)
# cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=BinaryClassificationEvaluator(), numFolds=3)

# Fit the model to the training data
model = cv.fit(trainingData)

#################### Evaluate the model performance #####################

# Use the best model to predict on the test data
predictions = model.transform(testData)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

# Find the best combination of hyperparameters
bestModel = model.bestModel
best_param = bestModel.stages[1].extractParamMap()

# print(best_param)
with open("best_param.txt", "w") as file:
    content = str(best_param)
    content = content.replace("name=", "\nname=")
    file.write(content)



# B. By checking featureImportances, which features are the most important? Try to
# give an analysis on your results


# Access the feature importances
importances = bestModel.stages[-1].featureImportances
print("Feature Importance, " + str(importances))
# Extract the feature names
feature_names = trainingData.columns

# Create a list of (feature, importance) tuples
feature_importances = [(feature_names[i], importances[i]) for i in range(len(feature_names)-1)]

# Sort the feature importances by descending importance
feature_importances.sort(key=lambda x: x[1], reverse=True)

# Print the feature importances
for feature, importance in feature_importances:
    print("Feature:", feature, "Importance:", importance)

# Plot the feature importances
plt.bar(range(len(feature_importances)), [importance for _, importance in feature_importances], align='center')
plt.xticks(range(len(feature_importances)), [feature for feature, _ in feature_importances], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()

### CREATING GRADIENT-BOOST TREE CLASSIFIERS ###

# Create the Gradient Descent Classifier
gbt = GBTClassifier(labelCol="label", featuresCol="features",maxBins=100)
# Repeat the same process for GBTClassifier
pipeline_gbt = Pipeline(stages=[assembler, gbt])
# Repeat the same process for GBTClassifier
paramGrid_gbt = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [5, 10, 15]) \
    .addGrid(gbt.maxDepth, [2, 4, 6]) \
    .addGrid(gbt.featureSubsetStrategy, ["auto", "sqrt", "log2"]) \
    .build()

# Create a cross-validator to tune the hyperparameters
cv_gbt = CrossValidator(estimator=pipeline_gbt, estimatorParamMaps=paramGrid_gbt, evaluator=MulticlassClassificationEvaluator(), numFolds=3)
# cv_gbt = CrossValidator(estimator=pipeline_gbt, estimatorParamMaps=paramGrid_gbt, evaluator=BinaryClassificationEvaluator(), numFolds=3)

# Fit the model to the training data
model_gbt = cv_gbt.fit(trainingData)

# Create predictions for each model
predictions_rf = model.transform(testData)
predictions_gbt = model_gbt.transform(testData)

# Create the evaluator object
evaluator = MulticlassClassificationEvaluator()

# Set the metric for accuracy
evaluator.setMetricName("accuracy")
# Evaluate the accuracy of the RandomForestClassifier model
accuracy_rf = evaluator.evaluate(predictions_rf)
accuracy_gbt = evaluator.evaluate(predictions_gbt)

# Set the metric for F1 score
evaluator.setMetricName("f1")
# Evaluate the F1 score of the RandomForestClassifier model
f1_rf = evaluator.evaluate(predictions_rf)
f1_gbt = evaluator.evaluate(predictions_gbt)

# Set the metric for AUC
evaluator.setMetricName("weightedPrecision")
# Evaluate the AUC of the RandomForestClassifier model
auc_rf = evaluator.evaluate(predictions_rf)
auc_gbt = evaluator.evaluate(predictions_gbt)


# Compare the results
print("Accuracy of Random Forest Classifier: ", accuracy_rf)
print("Accuracy of GBT Classifier: ", accuracy_gbt)

print("F1 score of Random Forest Classifier: ", f1_rf)
print("F1 score of GBT Classifier: ", f1_gbt)

print("AUC of Random Forest Classifier: ", auc_rf)
print("AUC of GBT Classifier: ", auc_gbt)


# # Convert the predictions DataFrame to an RDD
# predictions_rdd_rf = predictions_rf.select("rawPrediction", "label").rdd
# predictions_rdd_gbt = predictions_gbt.select("rawPrediction", "label").rdd
#
# # Instantiate a BinaryClassificationMetrics object
# metrics_rf = BinaryClassificationMetrics(predictions_rdd_rf)
# metrics_gbt = BinaryClassificationMetrics(predictions_rdd_gbt)
#
# # Compute the ROC curve
# roc_rf = metrics_rf.roc()
# roc_gbt = metrics_gbt.roc()
#
# # Convert the ROC curve to an RDD and collect it as a list of (FPR, TPR) tuples
# roc_points_rf = roc_rf.collect()
# roc_points_gbt = roc_gbt.collect()
#
#
# fpr = [x[0] for x in roc_points_rf]
# tpr = [x[1] for x in roc_points_rf]
#
# plt.figure()
# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for Random Forest')
# plt.show()
#
# fpr_gbt = [x[0] for x in roc_points_gbt]
# tpr_gbt = [x[1] for x in roc_points_gbt]
#
# plt.figure()
# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for Gradient Boost Trees')
# plt.show()
