from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.clustering import KMeans
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
import pyspark

spark = SparkSession.builder \
    .appName("FlightPrediction") \
    .master("local[*]") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

try:
    df = spark.read.load('clean_dataset',format='csv',sep=',',inferSchema=True, header=True)
    df.printSchema()

    #  numeric columns
    input_cols_num = [
        "Year",
        "Quarter",
        "Month",
        "DayOfWeek",
        "Flight_Number_Operating_Airline",
        "CRSDepTime",
        "DepTime",
        "DepDelay",
        "TaxiOut",
        "WheelsOff",
        "Distance",
        "DistanceGroup",
        "AirTime",
        "CRSArrTime",
        "ArrTime",
        "ArrDelay",
        "WheelsOn",
        "TaxiIn",
        "CRSElapsedTime",
        "ActualElapsedTime"
    ]


    # categoric columns
    input_cols_str = [
        "Operated_or_Branded_Code_Share_Partners",
        "Operating_Airline",
        "Origin",
        "Dest",
        "OriginCityNameState",
        "DestCityNameState"
    ]
    #=======================Correlation Matrix============================

    assembler = VectorAssembler(inputCols=input_cols_num, outputCol="numeric_features")
    numeric_df = assembler.transform(df).select("numeric_features")
    correlation_matrix = Correlation.corr(numeric_df, "numeric_features", "pearson").head()[0].toArray()
    corr_df = pd.DataFrame(correlation_matrix, columns=input_cols_num, index=input_cols_num)

    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()
    plt.savefig("plots/matrix.png")
    plt.show()

    #=======================Delay by DayWeek============================
    df = df.toPandas()
    df_filtrado = df[df['ArrDelay'] > 15]
    contagem = df_filtrado.groupby('DayOfWeek').size().reset_index(name='QtdAtrasos')
    
    day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    contagem['DayOfWeek'] = contagem['DayOfWeek'].map(day_names)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=contagem, x='DayOfWeek', y='QtdAtrasos', palette='Blues_d')
    plt.title('Delay by Day of Week')
    plt.savefig("plots/delayDayweek.png")
    plt.show()

finally:
    print("Encerrando a Spark Session...")
    spark.stop()