# producer.py
import time
import json
from kafka import KafkaProducer
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("producer").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.option("header", "true").csv("streaming_dataset/")

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Enviar linha a linha
rows = df.toLocalIterator()

for row in rows:
    data = row.asDict()
    producer.send("flights", value=data)
    print(f"Enviado: {data['Origin']} → {data['Dest']} | ArrDelay: {data.get('ArrDelay', 'N/A')}")
    time.sleep(10)  # simular streaming — 1 voo a cada 0.5 segundos

producer.flush()
print("Todos os dados enviados.")