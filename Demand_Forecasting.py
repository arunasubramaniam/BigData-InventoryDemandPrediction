
###### Please note that this project is executed in the google collab environment 
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q https://www-us.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz
!tar xf spark-2.4.3-bin-hadoop2.7.tgz
!pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.3-bin-hadoop2.7"

import findspark
findspark.init()
from pyspark.sql import SparkSession
sc = SparkSession.builder.master("local[*]").getOrCreate()

import pandas as pd
import numpy as np
import re
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

from pyspark import SparkContext
from pyspark import HiveContext
from pyspark.sql import functions as sf
from pyspark.sql.functions import udf,rank,unix_timestamp,round,pandas_udf, PandasUDFType
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import RandomForestRegressor,GBTRegressor
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

### Function for Feature Engineering
def extract_weight(x,flag):
    x = x.lower()
    if flag==1:
        extract = re.findall(r'[0-9]+(?=g)',x)
        if len(extract)>0:
            return int(extract[0])
        else:
            return 0
    elif flag==2:
        extract = re.findall(r'[0-9]+(?=kg)',x)
        if len(extract)>0:
            return int(extract[0])
        else:
            return 0
    elif flag==3:
        extract = re.findall(r'[0-9]+(?=ml)',x)
        if len(extract)>0:
            return int(extract[0])
        else:
            return 0
    else:
        extract = re.findall(r'[0-9]+(?=l)',x)
        if len(extract)>0:
            return int(extract[0])
        else:
            return 0
##### Since product details and town file is small we performed the feature engineering using pandas 
product = pd.read_csv("/content/gdrive/My Drive/grupo-bimbo-inventory-demand/producto_tabla.csv")
product["grams"] = product.NombreProducto.apply(lambda x : extract_weight(x,1))
product["kilograms"] = product.NombreProducto.apply(lambda x : extract_weight(x,2))
product["millilitre"] = product.NombreProducto.apply(lambda x : extract_weight(x,3))
product["litre"] = product.NombreProducto.apply(lambda x : extract_weight(x,4))
product["pbrand"] = product.NombreProducto.apply(lambda x : x[:re.search("\d",x).start()-1])
pbrand = pd.DataFrame(product.pbrand.drop_duplicates())

####Generating makeshift id for the newly generated sub brand
pbrand["pbrand_Id"] = list(range(50001,50001+len(pbrand)))
product = pd.merge(product,pbrand,on=['pbrand'])
product.to_csv("/content/gdrive/My Drive/grupo-bimbo-inventory-demand/product_modified.csv")

####Feature Engineer for town_state
client = pd.read_csv("/content/gdrive/My Drive/grupo-bimbo-inventory-demand/town_state.csv")
state = pd.DataFrame(client.State.drop_duplicates())
state["state_Id"] = list(range(30001,30001+len(state)))
town = pd.DataFrame(client.Town.drop_duplicates())
town["town_Id"] = list(range(40001,40001+len(town)))
client = client.merge(state,on=['State']).merge(town,on=['Town'])
client.to_csv('/content/gdrive/My Drive/grupo-bimbo-inventory-demand/state_modified.csv')

#### Defining the sqlcontext
sqlContext = HiveContext(sc)

#### Reading the actual input File
df = sc.read.csv("/content/gdrive/My Drive/grupo-bimbo-inventory-demand/train.csv",header='true', inferSchema='true')
names = ['week_number','Depot_Id','Sales_Channel_Id','Route_Id','Client_Id','Product_Id','Unit_Sales','Sales','Return_Units','Returns','Adjusted_Demand']
df = df.toDF(*names)
df = df.cache()

#### Finding Descriptive Statistics of a column 
df.select("Unit_Sales").describe().show()
df = df.select(['week_number','Depot_Id','Sales_Channel_Id','Route_Id','Client_Id','Product_Id','Adjusted_Demand'])


#### reading_product_informations from the file into spark
product = sc.read.csv("/content/gdrive/My Drive/grupo-bimbo-inventory-demand/product_modified.csv",header='true', inferSchema='true')
names1 = ['id','Product_Id','Product_Name','grams','kilograms','millilitre','litre','pbrand','pbrand_Id']
product = product.toDF(*names1)
product = product.select(["Product_Id",'grams','kilograms','millilitre','litre','pbrand_Id'])

### Joining the modified product information with actual file using spark
df = df.join(product,on="Product_Id",how="left")

#### reading_town_informations from the file into spark
town = sc.read.csv("/content/gdrive/My Drive/grupo-bimbo-inventory-demand/state_modified.csv",header='true', inferSchema='true')
names2 = ['Id','Depot_Id','Town','State','state_Id','town_Id']
town = town.toDF(*names2)
town = town.select(["Depot_Id",'state_Id','town_Id'])

#### Joining the modified town information with actual file using spark
df = df.join(town,on="Depot_Id",how="left")

#### Filtering the data for 100 products 
pfilter = product.select('Product_Id').limit(100)
pfilter = pfilter.toDF("Product_Id")

#### Filtering the sales data for filtered products
df_new = df.join(pfilter,on="Product_Id")

#### Splitting the data into train and test using weeknumber 
## since data is time sensitive any sales before the week 9 is train and week 9th data is test
train = df_new.where("week_number<9")
test = df_new.where("week_number=9")

#### Constructing the one hot encoding estimator using spark mlib
cols = df.columns
stages = []

Categorical = ['Product_Id','Client_Id','pbrand_Id']
for categoricalCol in Categorical:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols =[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer,encoder]

#### Vector Assembler for converting the dependent data into features
numericCols = [ 'grams','kilograms','millilitre','litre']
assemblerInputs = [c + "classVec" for c in Categorical] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


#### Constructing the spark mlib pipeline with one hot encoder and 
pipeline = Pipeline(stages = stages)

#### Transform train data
pipelineModel = pipeline.fit(train)
train = pipelineModel.transform(train)
selectedCols = ['Adjusted_Demand', 'features']
model_data = train.select(selectedCols)

#### Transform train data
pipelineModel = pipeline.fit(test)
test = pipelineModel.transform(test)
selectedCols = ['Adjusted_Demand', 'features']
test = test.select(selectedCols)

#### Defining the random forest regressor 
dt = RandomForestRegressor(featuresCol='features',labelCol="Adjusted_Demand")
rf = dt.fit(model_data)

#### Prediction and RMSE for train data
predictions = rf.transform(model_data)
evaluator = RegressionEvaluator(labelCol="Adjusted_Demand", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on train data = %g" % rmse)

#### Prediction and RMSE for test data
predictions = rf.transform(test)
evaluator = RegressionEvaluator(labelCol="Adjusted_Demand", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


#### Defining the Gradient Boosted Regressor 
gbt = GBTRegressor(featuresCol='features',labelCol="Adjusted_Demand")
model = gbt.fit(model_data)

#### Prediction and RMSE for train data
predictions = model.transform(model_data)
evaluator = RegressionEvaluator(labelCol="Adjusted_Demand", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on train data = %g" % rmse)

#### Prediction and RMSE for test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(labelCol="Adjusted_Demand", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

#### Gradient Boosted Trees Regression is the best model and we take its prediction dataset
#### Combining the prediction to the test data and write to csv
test=test.withColumn('row_index', sf.monotonically_increasing_id())
predictions=predictions.withColumn('row_index', sf.monotonically_increasing_id())
test = test.join(predictions.select("prediction","row_index"),on=["row_index"])
test.write.format("csv").save("/content/gdrive/My Drive/grupo-bimbo-inventory-demand/output/")