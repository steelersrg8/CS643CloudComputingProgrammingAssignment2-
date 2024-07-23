#import findspark
#findspark.init()
#findspark.find()

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
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#Starting the spark session
c1 = pyspark.SparkConf().setAppName('winequality').setMaster('local')
sc22 = pyspark.SparkContext(conf=c1)
s3 = SparkSession(sc22)

p4  = sys.argv[1]
#loading the validation dataset
v5 = s3.read.format("csv").load(p4, header = True, sep=";")
v5.printSchema()
v5.show()

    
#changing the 'quality' column name to 'label'
for c6 in v5.columns[1:-1] + ['""""quality"""""']:
    v5 = v5.withColumn(c6, col(c6).cast('float'))
v5 = v5.withColumnRenamed('""""quality"""""', "label")

#getting the features and label seperately and converting it to numpy array
features =np.array(v5.select(v5.columns[1:-1]).collect())
label = np.array(v5.select('label').collect())

#creating the feature vector
VectorAssembler = VectorAssembler(inputCols =v5.columns[1:-1], outputCol ='features')
d7 = VectorAssembler.transform(v5)
d7 = d7.select(['features', 'label'])

#The following function creates the labeledpoint and parallelize it to convert it into RDD
def t8(sc, features, labels, categorical=False):
    labeled_points = []
    for x, y in zip(features, labels):        
        lp = LabeledPoint(y, x)
        labeled_points.append(lp)
    return sc.parallelize(labeled_points) 

#rdd converted dataset
d9 = t8(sc22, features, label)

#loading the model from s3
r10 = RandomForestModel.load(sc22, "/winepredict/trainingmodel.model/")

print("model loaded successfully")
p11 = r10.predict(d9.map(lambda x20: x20.f24))

#getting a RDD of label and predictions
l12 = d9.map(lambda l18: l18.l23).zip(p11)
 
labelsAndPredictions_df = l12.toDF()
#cpnverting rdd ==> spark dataframe ==> pandas dataframe 
l13 = l12.toDF(["label", "Prediction"])
l13.show()
l14 = l13.toPandas()


#Calculating the F1score
f16 = f1_score(l14['label'], l14['Prediction'], average='micro')
print("F1- score: ", f16)
print(confusion_matrix(l14['label'], l14['Prediction']))
print(classification_report(l14['label'], l14['Prediction']))
print("Accuracy", accuracy_score(l14['label'], l14['Prediction']))

#calculating the test error
t15 = l12.filter(
    lambda l17: l17[0] != l17[1]).count() / float(d9.count())
print('Test Error = ' + str(t15))

