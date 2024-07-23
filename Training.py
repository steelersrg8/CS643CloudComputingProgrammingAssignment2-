import findspark
findspark.init()
findspark.find()

#Loading the libraries
import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#Starting the spark session
con1 = pyspark.SparkConf().setAppName('winequality').setMaster('local')
sc2 = pyspark.SparkContext(conf=con1)
spa3 = SparkSession(sc2)

#Loading the dataset
df4 = spa3.read.format("csv").load("TrainingDataset.csv", header = True, sep =";")
df4.printSchema()
df4.show()

#changing the 'quality' column name to 'label'
for col5 in df4.columns[1:-1] + ['""""quality"""""']:
    df4 = df4.withColumn(col5, col(col5).cast('float'))
df4 = df4.withColumnRenamed('""""quality"""""', "label")


#getting the features and label seperately and converting it to numpy array
features =np.array(df4.select(df4.columns[1:-1]).collect())
label = np.array(df4.select('label').collect())

#creating the feature vector
VectorAssembler = VectorAssembler(inputCols =df4.columns[1:-1], outputCol ='features')
df5 = VectorAssembler.transform(df4)
df5 = df5.select(['features', 'label'])

#The following function creates the labeledpoint and parallelize it to convert it into RDD
def t6(s8, f7, l9, categorical=False):
    lp10 = []
    for x, y in zip(f7, l9):
        lp11 = LabeledPoint(y, x)
        lp10.append(lp11)
    return s8.parallelize(lp10)

#rdd converted dataset
d12 = t6(sc2, features, label)

#Splitting the dataset into train and test
t13, t14 = d12.randomSplit([0.7, 0.3], seed =11)


#Creating a random forest training classifier
rf15 = RandomForest.trainClassifier(t13, numClasses=10, categoricalFeaturesInfo={},
                                    numTrees=21, featureSubsetStrategy="auto",
                                    impurity='gini', maxDepth=30, maxBins=32)

#predictions
p16 = rf15.predict(t14.map(lambda x: x.features))
#predictionAndLabels = test.map(lambda x: (float(model.predict(x.features)), x.label))

#getting a RDD of label and predictions
lap17 = t14.map(lambda lp: lp.label).zip(p16)

lap18 = lap17.toDF()
#cpnverting rdd ==> spark dataframe ==> pandas dataframe 
l19 = lap17.toDF(["label", "Prediction"])
l19.show()
l20 = l19.toPandas()


#Calculating the F1score
f21 = f1_score(l20['label'], l20['Prediction'], average='micro')
print("F1- score: ", f21)
print(confusion_matrix(l20['label'], l20['Prediction']))
print(classification_report(l20['label'], l20['Prediction']))
print("Accuracy", accuracy_score(l20['label'], l20['Prediction']))

#calculating the test error
t22 = lap17.filter(
    lambda lp23: lp23[0] != lp23[1]).count() / float(t14.count())
print('Test Error = ' + str(t22))

#save training model
rf15.save(sc2, 's3://winequal/trainingmodel.model')



