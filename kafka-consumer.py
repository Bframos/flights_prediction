import json
from kafka import KafkaConsumer
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import GBTClassificationModel
from pyspark.sql.functions import col, floor, when
from pyspark.sql.types import *

spark = SparkSession.builder.appName("flights_consumer").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Carregar o pipeline e o modelo
pipeline_model = PipelineModel.load("models/pipeline")
model = GBTClassificationModel.load("models/gbt_flights")

# Definir o schema dos dados
schema = StructType([
    StructField("Month", IntegerType()),
    StructField("DayOfWeek", IntegerType()),
    StructField("CRSDepTime", IntegerType()),
    StructField("Distance", DoubleType()),
    StructField("DepDelay", IntegerType()),
    StructField("Operating_Airline", StringType()),
    StructField("Origin", StringType()),
    StructField("Dest", StringType()),
])

# Ligar ao Kafka
consumer = KafkaConsumer(
    "flights",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8"))
)

print("A aguardar mensagens do Kafka...\n")

for message in consumer:
    data = message.value

    # Converter tipos antes de criar o DataFrame
    parsed = {
        "Month": int(data.get("Month", 0) or 0),
        "DayOfWeek": int(data.get("DayOfWeek", 0) or 0),
        "CRSDepTime": int(data.get("CRSDepTime", 0) or 0),
        "Distance": float(data.get("Distance", 0.0) or 0.0),
        "DepDelay": int(data.get("DepDelay", 0) or 0),
        "Operating_Airline": str(data.get("Operating_Airline", "") or ""),
        "Origin": str(data.get("Origin", "") or ""),
        "Dest": str(data.get("Dest", "") or ""),
    }

    row = spark.createDataFrame([parsed], schema=schema)


    # Adicionar feature hour
    row = row.withColumn("hour", floor(col("CRSDepTime") / 100))

    # Aplicar pipeline e modelo
    transformed = pipeline_model.transform(row)
    prediction = model.transform(transformed)

    result = prediction.select("prediction").collect()[0][0]
    label = "ATRASADO ✈ " if result == 1.0 else "A TEMPO ✅"

    print(f"{data.get('Origin')} → {data.get('Dest')} | {data.get('Operating_Airline')} | Previsão: {label}")