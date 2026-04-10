from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.clustering import KMeans
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col, when
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import floor
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
import pyspark
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



spark = SparkSession.builder \
    .appName("FlightPrediction") \
    .master("local[*]") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

try:
    df = spark.read.load('clean_dataset',format='csv',sep=',',inferSchema=True, header=True)

    #===========================TARGET===================================
    df = df.withColumn("delayed", when(col("ArrDelay") > 25, 1).otherwise(0))
    df.groupBy("delayed").count().show()

    #===========================FEATURE SELECTION===================================
    df = df.withColumn("hour", floor(col("CRSDepTime") / 100))
    # Época do ano (férias tendem a ter mais atrasos)
    df = df.withColumn("is_summer", when((col("Month") >= 6) & (col("Month") <= 8), 1).otherwise(0))
    df = df.withColumn("is_holiday_season", when((col("Month") == 11) | (col("Month") == 12), 1).otherwise(0))

    # Período do dia
    df = df.withColumn("time_of_day", 
        when(col("hour") < 6, 0)       # madrugada
        .when(col("hour") < 12, 1)     # manhã
        .when(col("hour") < 18, 2)     # tarde
        .otherwise(3))                  # noite

    cat_cols = ["Operating_Airline", "Origin", "Dest"]
    num_cols = ["Month", "DayOfWeek", "CRSDepTime", "Distance", "DepDelay", "hour"]
    target = "delayed"

    # Remover colunas de leakage e manter só o necessário
    cols_to_keep = num_cols + cat_cols + [target]
    df = df.select(cols_to_keep)

    
    #===========================PRE PROCESS===================================
    #The `StringIndexer` converts categorical columns like airline, 
    # origin and destination into numbers, since ML models can't work with raw strings.

    #The `VectorAssembler` merges all columns into a single vector called `features`, 
    # which is required by MLlib.

    #The `StandardScaler` normalizes the vector so all features are on the same scale, 
    #preventing large-valued columns like distance from dominating smaller ones like day of week.

    #The `Pipeline` chains all these steps together, 
    # ensuring the same transformations are applied consistently during both training and inference.

    indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") 
                for c in cat_cols]

    cat_indexed = [c + "_idx" for c in cat_cols]
    assembler = VectorAssembler(inputCols=num_cols + cat_indexed, outputCol="features_raw")

    scaler = StandardScaler(inputCol="features_raw", outputCol="features")

    pipeline = Pipeline(stages=indexers + [assembler, scaler])

    pipeline_model = pipeline.fit(df)
    df_prepared = pipeline_model.transform(df)

    df_prepared.select("features", "delayed").show(5)

    #===========================MODEL TRAINING==============================
    # 1 — Split treino/teste
    train, test = df_prepared.randomSplit([0.8, 0.2], seed=42)

    # 2 — Modelo
    gbt = GBTClassifier(
    labelCol="delayed", 
    featuresCol="features", 
    maxIter=100,
    maxDepth=6,
    stepSize=0.1
    )
    # 3 — Treinar
    model = gbt.fit(train)

    # 4 — Avaliar
    predictions = model.transform(test)

    evaluator = BinaryClassificationEvaluator(labelCol="delayed", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print(f"AUC-ROC: {auc:.4f}")

    # 5 — Precision, Recall, F1
    for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
        val = MulticlassClassificationEvaluator(labelCol="delayed", metricName=metric).evaluate(predictions)
        print(f"{metric}: {val:.4f}")

    #===========================MODEL SAVE==============================

    # Guardar o pipeline de pré-processamento
    pipeline_model.save("models/pipeline")

    # GBT
    model.save("models/gbt_flights")

finally:
    print("Encerrando a Spark Session...")
    spark.stop()