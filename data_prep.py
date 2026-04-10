from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
import pyspark.sql.functions as F



spark = SparkSession.builder \
    .appName("FlightPrediction") \
    .master("local[*]") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

try:
    df = spark.read.load('flights.csv',format='csv',sep=',',inferSchema=True, header=True)
    df.printSchema()



    #Check for duplicates
    print(f'Number of rows before drop duplicates:{df.count()}')
    df_dropduplicates=df.dropDuplicates()
    print(f'Number of rows after drop duplicates:{df_dropduplicates.count()}')

    #Remove useless columns
    cols_to_dismiss = [
    'DepDelayMinutes', 'DepDel15', 'DepartureDelayGroups',
    'ArrDelayMinutes', 'ArrDel15', 'ArrivalDelayGroups',
    'DepTimeBlk','ArrTimeBlk','DestAirportID','DestAirportSeqID',
    'DestCityMarketID','OriginAirportID','OriginAirportSeqID',
    'OriginCityMarketID','Marketing_Airline_Network',
    'Flight_Number_Marketing_Airline',
    'FlightDate','OriginWac','DestWac','Flights',
    'Duplicate',"OriginStateFips","DestStateFips",'OriginState',
    "DestState",

    # Diverted flights 
    'DivAirportLandings', 'DivReachedDest', 'DivActualElapsedTime',
    'DivArrDelay', 'DivDistance',

   
    'Div1Airport', 'Div1AirportID', 'Div1AirportSeqID', 'Div1WheelsOn',
    'Div1TotalGTime', 'Div1LongestGTime', 'Div1WheelsOff', 'Div1TailNum',
    'Div2Airport', 'Div2AirportID', 'Div2AirportSeqID', 'Div2WheelsOn',
    'Div2TotalGTime', 'Div2LongestGTime', 'Div2WheelsOff', 'Div2TailNum',
    'Div3Airport', 'Div3AirportID', 'Div3AirportSeqID', 'Div3WheelsOn',
    'Div3TotalGTime', 'Div3LongestGTime', 'Div3WheelsOff', 'Div3TailNum',
    'Div4Airport', 'Div4AirportID', 'Div4AirportSeqID', 'Div4WheelsOn',
    'Div4TotalGTime', 'Div4LongestGTime', 'Div4WheelsOff', 'Div4TailNum',
    'Div5Airport', 'Div5AirportID', 'Div5AirportSeqID', 'Div5WheelsOn',
    'Div5TpartidaotalGTime', 'Div5LongestGTime', 'Div5WheelsOff', 'Div5TailNum',

    
    'DOT_ID_Marketing_Airline', 'DOT_ID_Operating_Airline','DayofMonth',
    'IATA_Code_Marketing_Airline', 'IATA_Code_Operating_Airline','Tail_Number',
    "Cancelled", "Diverted",


    
    'CarrierDelay','WeatherDelay','NASDelay','SecurityDelay'
    'CancellationCode','Originally_Scheduled_Code_Share_Airline','DOT_ID_Originally_Scheduled_Code_Share_Airline',
    'IATA_Code_Originally_Scheduled_Code_Share_Airline','Flight_Num_Originally_Scheduled_Code_Share_Airline',
    'FirstDepTime','TotalAddGTime','LongestAddGTime','_c119','LateAircraftDelay', "CancellationCode", "SecurityDelay"
    ]

    df_clean = df_dropduplicates.drop(*cols_to_dismiss)


    #Data Transformation
    
    def hhmm_to_minutes(col):
        return (F.floor(col / 100) * 60) + (col % 100)

    #Departure Time
    df_clean = df_clean.withColumn("DepTime", hhmm_to_minutes(F.col("DepTime")))
    df_clean = df_clean.withColumn("CRSDepTime", hhmm_to_minutes(F.col("CRSDepTime")))
    df_clean = df_clean.withColumn("DepDelay",F.col("DepTime") - F.col("CRSDepTime"))

    #Arrive Time
    df_clean = df_clean.withColumn("ArrTime", hhmm_to_minutes(F.col("ArrTime")))
    df_clean = df_clean.withColumn("CRSArrTime", hhmm_to_minutes(F.col("CRSArrTime")))
    df_clean = df_clean.withColumn("ArrDelay", F.col("ArrTime") - F.col("CRSArrTime"))

    #=============================
    #Missing Values
    
    df_clean = df_clean.dropna(how='any')
    print(f'Dados Finais: {df_clean.count()}')
    df_clean.coalesce(1).write.option("header",True).csv("clean_dataset",header=True, mode='overwrite')

    df_clean=df_clean.limit(1000)
    df_clean.write.csv('streaming_dataset', header=True, mode='overwrite')
finally:
    print("Encerrando a Spark Session...")
    spark.stop()