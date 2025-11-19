```python
# Setup: Initialize Spark Session for ML Models
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Create Spark Session
spark = SparkSession.builder \
    .appName("TLC Gold Layer ML Models") \
    .config("spark.driver.memory", "10g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark Session initialized for ML Models.")
print(f"Spark version: {spark.version}")

```

    Spark Session initialized for ML Models.
    Spark version: 3.5.0



```python
# Configuration
SILVER_PATH = "/home/ubuntu/project/silver_layer_data_v2"
ML_MODELS_PATH = "/home/ubuntu/project/gold_layer_data/ml_models"
ML_PREDICTIONS_PATH = "/home/ubuntu/project/gold_layer_data/ml_predictions"

print(f"Reading from: {SILVER_PATH}")
print(f"Models will be saved to: {ML_MODELS_PATH}")
print(f"Predictions will be saved to: {ML_PREDICTIONS_PATH}")
```

    Reading from: /home/ubuntu/project/silver_layer_data_v2
    Models will be saved to: /home/ubuntu/project/gold_layer_data/ml_models
    Predictions will be saved to: /home/ubuntu/project/gold_layer_data/ml_predictions



```python
# Load Silver Layer Data with Quality Filtering
silver_df = spark.read.parquet(SILVER_PATH)

# Filter for high-quality records only
silver_df_clean = silver_df.filter(
    (F.col("has_parse_errors") == False) & 
    (F.col("has_missing_values") == False)
)

print(f"Total Silver records: {silver_df.count():,}")
print(f"Clean records (no parse errors/missing values): {silver_df_clean.count():,}")
print(f"\nSchema:")
silver_df_clean.printSchema()
```

    Total Silver records: 179,779,179


    [Stage 20:====================================================> (106 + 3) / 109]

    Clean records (no parse errors/missing values): 169,610,695
    
    Schema:
    root
     |-- record_id: string (nullable = true)
     |-- pickup_datetime: timestamp (nullable = true)
     |-- pickup_month: integer (nullable = true)
     |-- pickup_day: integer (nullable = true)
     |-- pickup_hour: integer (nullable = true)
     |-- pickup_dayofweek: integer (nullable = true)
     |-- passenger_count: double (nullable = true)
     |-- trip_distance: double (nullable = true)
     |-- payment_type: integer (nullable = true)
     |-- fare_amount: double (nullable = true)
     |-- tip_amount: double (nullable = true)
     |-- tolls_amount: double (nullable = true)
     |-- total_amount: double (nullable = true)
     |-- has_missing_values: boolean (nullable = true)
     |-- has_parse_errors: boolean (nullable = true)
     |-- parse_errors: array (nullable = true)
     |    |-- element: string (containsNull = true)
     |-- ingestion_timestamp: timestamp (nullable = true)
     |-- source_file: string (nullable = true)
     |-- processing_batch_id: string (nullable = true)
     |-- pickup_year: integer (nullable = true)
     |-- taxi_type: string (nullable = true)
    


                                                                                    

---
## Model 1: Demand Forecasting (Time Series)

**Business Value**: Predict hourly taxi demand to optimize driver allocation and reduce wait times.

**Features**: Hour of day, day of week, month, taxi type, historical demand patterns


```python
# Prepare Demand Forecasting Dataset
demand_df = silver_df_clean.groupBy(
    F.col("pickup_year"),
    F.col("pickup_month"),
    F.col("pickup_day"),
    F.col("pickup_hour"),
    F.dayofweek(F.col("pickup_datetime")).alias("day_of_week"),
    F.col("taxi_type")
).agg(
    F.count("*").alias("trip_count")
)

# Add temporal features
demand_df = demand_df.withColumn(
    "is_weekend", 
    F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0)
).withColumn(
    "is_rush_hour",
    F.when(F.col("pickup_hour").between(7, 9) | F.col("pickup_hour").between(17, 19), 1).otherwise(0)
)

print(f"Demand forecasting records: {demand_df.count():,}")
demand_df.show(10)
```

                                                                                    

    Demand forecasting records: 87,587


    [Stage 29:=================================================>     (99 + 8) / 109]

    +-----------+------------+----------+-----------+-----------+---------+----------+----------+------------+
    |pickup_year|pickup_month|pickup_day|pickup_hour|day_of_week|taxi_type|trip_count|is_weekend|is_rush_hour|
    +-----------+------------+----------+-----------+-----------+---------+----------+----------+------------+
    |       2024|           5|        31|          5|          6|   yellow|       608|         0|           0|
    |       2024|          12|         8|          3|          1|   yellow|      2309|         1|           0|
    |       2024|          12|         5|          4|          5|   yellow|       282|         0|           0|
    |       2024|           5|        31|          6|          6|   yellow|      1483|         0|           0|
    |       2024|          12|         9|          5|          2|   yellow|       710|         0|           0|
    |       2024|          12|        12|         23|          5|   yellow|      7695|         0|           0|
    |       2024|          12|        12|         13|          5|   yellow|      7046|         0|           0|
    |       2024|          12|         8|         20|          1|   yellow|      5094|         1|           0|
    |       2024|          12|        10|          9|          3|   yellow|      5936|         0|           1|
    |       2024|          12|        16|         12|          2|   yellow|      6118|         0|           0|
    +-----------+------------+----------+-----------+-----------+---------+----------+----------+------------+
    only showing top 10 rows
    


                                                                                    


```python
# Build Demand Forecasting Model
# Index categorical features
taxi_indexer = StringIndexer(inputCol="taxi_type", outputCol="taxi_type_idx")

# Feature vector
demand_assembler = VectorAssembler(
    inputCols=["pickup_hour", "day_of_week", "pickup_month", "taxi_type_idx", 
               "is_weekend", "is_rush_hour"],
    outputCol="features"
)

# Random Forest Regressor
demand_rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="trip_count",
    numTrees=50,
    maxDepth=10,
    seed=42
)

# Pipeline
demand_pipeline = Pipeline(stages=[taxi_indexer, demand_assembler, demand_rf])

# Split data
demand_train, demand_test = demand_df.randomSplit([0.8, 0.2], seed=42)

print(f"Training records: {demand_train.count():,}")
print(f"Test records: {demand_test.count():,}")

# Train model
print("\nTraining Demand Forecasting model...")
demand_model = demand_pipeline.fit(demand_train)
print("Model trained successfully!")
```

                                                                                    

    Training records: 70,179


                                                                                    

    Test records: 17,408
    
    Training Demand Forecasting model...


    25/11/18 18:11:04 WARN DAGScheduler: Broadcasting large task binary with size 1292.4 KiB
    25/11/18 18:11:04 WARN DAGScheduler: Broadcasting large task binary with size 1292.4 KiB
    25/11/18 18:11:04 WARN DAGScheduler: Broadcasting large task binary with size 1872.3 KiB
    25/11/18 18:11:04 WARN DAGScheduler: Broadcasting large task binary with size 1872.3 KiB
    25/11/18 18:11:05 WARN DAGScheduler: Broadcasting large task binary with size 2.5 MiB
    25/11/18 18:11:05 WARN DAGScheduler: Broadcasting large task binary with size 2.5 MiB


    Model trained successfully!



```python
# Evaluate Demand Forecasting Model
demand_predictions = demand_model.transform(demand_test)

evaluator = RegressionEvaluator(labelCol="trip_count", predictionCol="prediction")

rmse = evaluator.evaluate(demand_predictions, {evaluator.metricName: "rmse"})
mae = evaluator.evaluate(demand_predictions, {evaluator.metricName: "mae"})
r2 = evaluator.evaluate(demand_predictions, {evaluator.metricName: "r2"})

print("=" * 60)
print("DEMAND FORECASTING MODEL PERFORMANCE")
print("=" * 60)
print(f"RMSE: {rmse:,.2f}")
print(f"MAE:  {mae:,.2f}")
print(f"R²:   {r2:.4f}")
print("=" * 60)

# Show sample predictions
demand_predictions.select(
    "pickup_hour", "day_of_week", "pickup_month", "taxi_type",
    "trip_count", "prediction"
).show(20)
```

                                                                                    

    ============================================================
    DEMAND FORECASTING MODEL PERFORMANCE
    ============================================================
    RMSE: 1,355.40
    MAE:  755.67
    R²:   0.7494
    ============================================================


    [Stage 97:=====================================================>(108 + 1) / 109]

    +-----------+-----------+------------+---------+----------+------------------+
    |pickup_hour|day_of_week|pickup_month|taxi_type|trip_count|        prediction|
    +-----------+-----------+------------+---------+----------+------------------+
    |          3|          4|           1|   yellow|      9808| 1344.318034647472|
    |          8|          4|           1|    green|       141|193.76338474057388|
    |         10|          4|           1|   yellow|      4510|5372.4573841692045|
    |         23|          4|           1|    green|       198|230.53270814454388|
    |          8|          5|           1|   yellow|      7438|  5705.47640039626|
    |         14|          5|           1|    green|       592|248.81973343239622|
    |         20|          5|           1|    green|       478| 239.9515514579356|
    |          7|          6|           1|    green|       281|333.62585854761636|
    |          3|          7|           1|   yellow|      2837|2220.6963526923823|
    |          4|          7|           1|    green|       106|  73.6436660331028|
    |          5|          7|           1|   yellow|      1134|2084.0485679680805|
    |         14|          7|           1|    green|       628|174.19618801199397|
    |         16|          7|           1|   yellow|     10729|5314.2766206711985|
    |          0|          1|           1|    green|       434|148.88749151675813|
    |         16|          1|           1|    green|       557|191.99129873471958|
    |          4|          2|           1|    green|        88| 63.24600750670431|
    |         17|          2|           1|    green|       800|495.14717745596363|
    |         16|          4|           1|   yellow|     11642|5885.7699908770555|
    |         13|          5|           1|    green|       565|238.93607641070997|
    |         20|          5|           1|   yellow|     13737| 5563.429252487739|
    +-----------+-----------+------------+---------+----------+------------------+
    only showing top 20 rows
    


                                                                                    


```python
# Save Demand Forecasting Model and Predictions
demand_model.write().overwrite().save(f"{ML_MODELS_PATH}/demand_forecasting")
demand_predictions.write.mode("overwrite").parquet(f"{ML_PREDICTIONS_PATH}/demand_forecasting")

print("Demand forecasting model and predictions saved!")
```

    [Stage 113:=================================================>   (102 + 7) / 109]

    Demand forecasting model and predictions saved!


                                                                                    

---
## Model 2: Fare Prediction

**Business Value**: Estimate fare amount based on trip characteristics for pricing transparency and surge detection.

**Features**: Trip distance, hour, day of week, passenger count, taxi type


```python
# Prepare Fare Prediction Dataset
fare_df = silver_df_clean.select(
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    F.dayofweek("pickup_datetime").alias("day_of_week"),
    "taxi_type",
    "fare_amount"
).filter(
    # Remove outliers
    (F.col("fare_amount") > 0) & (F.col("fare_amount") < 500) &
    (F.col("trip_distance") > 0) & (F.col("trip_distance") < 100) &
    (F.col("passenger_count") > 0) & (F.col("passenger_count") <= 6)
)

# Add engineered features
fare_df = fare_df.withColumn(
    "is_weekend",
    F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0)
).withColumn(
    "is_rush_hour",
    F.when(F.col("pickup_hour").between(7, 9) | F.col("pickup_hour").between(17, 19), 1).otherwise(0)
)

print(f"Fare prediction records: {fare_df.count():,}")
fare_df.describe().show()
```

    25/11/18 18:11:38 WARN SparkStringUtils: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.
    25/11/18 18:11:38 WARN SparkStringUtils: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.


    Fare prediction records: 162,986,730


    [Stage 119:====================================================>(108 + 1) / 109]

    +-------+-----------------+------------------+------------------+------------------+---------+------------------+------------------+-------------------+
    |summary|    trip_distance|   passenger_count|       pickup_hour|       day_of_week|taxi_type|       fare_amount|        is_weekend|       is_rush_hour|
    +-------+-----------------+------------------+------------------+------------------+---------+------------------+------------------+-------------------+
    |  count|        162986730|         162986730|         162986730|         162986730|162986730|         162986730|         162986730|          162986730|
    |   mean|3.312325191750113|1.4175876342816376|14.226686773825083| 4.135018083987574|     NULL|16.190379620660785|0.2638686720078377|0.31491607322878373|
    | stddev|4.292475085408215|0.9471097262173267| 5.649264049693675|1.9448050629247293|     NULL|15.331290811562072|0.4407289383877852| 0.4644824446355716|
    |    min|             0.01|               1.0|                 0|                 1|    green|              0.01|                 0|                  0|
    |    max|            99.96|               6.0|                23|                 7|   yellow|             499.7|                 1|                  1|
    +-------+-----------------+------------------+------------------+------------------+---------+------------------+------------------+-------------------+
    


                                                                                    


```python
# Build Fare Prediction Model
# Index categorical features
fare_taxi_indexer = StringIndexer(inputCol="taxi_type", outputCol="taxi_type_idx")

# Feature vector
fare_assembler = VectorAssembler(
    inputCols=["trip_distance", "passenger_count", 
               "pickup_hour", "day_of_week", "taxi_type_idx", "is_weekend", "is_rush_hour"],
    outputCol="raw_features"
)

# Scale features
scaler = StandardScaler(inputCol="raw_features", outputCol="features")

# Random Forest Regressor
fare_rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="fare_amount",
    numTrees=100,
    maxDepth=12,
    seed=42
)

# Pipeline
fare_pipeline = Pipeline(stages=[fare_taxi_indexer, fare_assembler, scaler, fare_rf])

# Sample data for faster training (optional - adjust sample rate as needed)
fare_df_sample = fare_df.sample(fraction=0.1, seed=42)

# Split data
fare_train, fare_test = fare_df_sample.randomSplit([0.8, 0.2], seed=42)

print(f"Training records: {fare_train.count():,}")
print(f"Test records: {fare_test.count():,}")

# Train model
print("\nTraining Fare Prediction model...")
fare_model = fare_pipeline.fit(fare_train)
print("Model trained successfully!")
```

                                                                                    

    Training records: 13,039,334


                                                                                    

    Test records: 3,260,382
    
    Training Fare Prediction model...


    25/11/18 18:14:10 WARN MemoryStore: Not enough space to cache rdd_283_78 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:10 WARN BlockManager: Persisting block rdd_283_78 to disk instead.
    25/11/18 18:14:10 WARN MemoryStore: Not enough space to cache rdd_283_78 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:10 WARN BlockManager: Persisting block rdd_283_78 to disk instead.
    25/11/18 18:14:11 WARN MemoryStore: Not enough space to cache rdd_283_78 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:11 WARN MemoryStore: Not enough space to cache rdd_283_82 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:11 WARN BlockManager: Persisting block rdd_283_82 to disk instead.
    25/11/18 18:14:11 WARN MemoryStore: Not enough space to cache rdd_283_78 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:11 WARN MemoryStore: Not enough space to cache rdd_283_82 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:11 WARN BlockManager: Persisting block rdd_283_82 to disk instead.
    25/11/18 18:14:13 WARN MemoryStore: Not enough space to cache rdd_283_82 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:13 WARN MemoryStore: Not enough space to cache rdd_283_86 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:13 WARN BlockManager: Persisting block rdd_283_86 to disk instead.
    25/11/18 18:14:13 WARN MemoryStore: Not enough space to cache rdd_283_84 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:13 WARN BlockManager: Persisting block rdd_283_84 to disk instead.
    25/11/18 18:14:13 WARN MemoryStore: Not enough space to cache rdd_283_87 in memory! (computed 5.2 MiB so far)
    25/11/18 18:14:13 WARN BlockManager: Persisting block rdd_283_87 to disk instead.
    25/11/18 18:14:13 WARN MemoryStore: Not enough space to cache rdd_283_82 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:13 WARN MemoryStore: Not enough space to cache rdd_283_86 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:13 WARN BlockManager: Persisting block rdd_283_86 to disk instead.
    25/11/18 18:14:13 WARN MemoryStore: Not enough space to cache rdd_283_84 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:13 WARN BlockManager: Persisting block rdd_283_84 to disk instead.
    25/11/18 18:14:13 WARN MemoryStore: Not enough space to cache rdd_283_87 in memory! (computed 5.2 MiB so far)
    25/11/18 18:14:13 WARN BlockManager: Persisting block rdd_283_87 to disk instead.
    25/11/18 18:14:14 WARN MemoryStore: Not enough space to cache rdd_283_84 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:14 WARN MemoryStore: Not enough space to cache rdd_283_89 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:14 WARN BlockManager: Persisting block rdd_283_89 to disk instead.
    25/11/18 18:14:14 WARN MemoryStore: Not enough space to cache rdd_283_84 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:14 WARN MemoryStore: Not enough space to cache rdd_283_89 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:14 WARN BlockManager: Persisting block rdd_283_89 to disk instead.
    25/11/18 18:14:14 WARN MemoryStore: Not enough space to cache rdd_283_90 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:14 WARN BlockManager: Persisting block rdd_283_90 to disk instead.
    25/11/18 18:14:14 WARN MemoryStore: Not enough space to cache rdd_283_90 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:14 WARN BlockManager: Persisting block rdd_283_90 to disk instead.
    25/11/18 18:14:15 WARN MemoryStore: Not enough space to cache rdd_283_92 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:15 WARN BlockManager: Persisting block rdd_283_92 to disk instead.
    25/11/18 18:14:15 WARN MemoryStore: Not enough space to cache rdd_283_90 in memory! (computed 7.8 MiB so far)
    25/11/18 18:14:15 WARN MemoryStore: Not enough space to cache rdd_283_94 in memory! (computed 3.5 MiB so far)
    25/11/18 18:14:15 WARN BlockManager: Persisting block rdd_283_94 to disk instead.
    25/11/18 18:14:15 WARN MemoryStore: Not enough space to cache rdd_283_92 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:15 WARN BlockManager: Persisting block rdd_283_92 to disk instead.
    25/11/18 18:14:15 WARN MemoryStore: Not enough space to cache rdd_283_90 in memory! (computed 7.8 MiB so far)
    25/11/18 18:14:15 WARN MemoryStore: Not enough space to cache rdd_283_94 in memory! (computed 3.5 MiB so far)
    25/11/18 18:14:15 WARN BlockManager: Persisting block rdd_283_94 to disk instead.
    25/11/18 18:14:16 WARN MemoryStore: Not enough space to cache rdd_283_95 in memory! (computed 2.3 MiB so far)
    25/11/18 18:14:16 WARN BlockManager: Persisting block rdd_283_95 to disk instead.
    25/11/18 18:14:16 WARN MemoryStore: Not enough space to cache rdd_283_95 in memory! (computed 2.3 MiB so far)
    25/11/18 18:14:16 WARN BlockManager: Persisting block rdd_283_95 to disk instead.
    25/11/18 18:14:16 WARN MemoryStore: Not enough space to cache rdd_283_93 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:16 WARN BlockManager: Persisting block rdd_283_93 to disk instead.
    25/11/18 18:14:16 WARN MemoryStore: Not enough space to cache rdd_283_93 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:16 WARN BlockManager: Persisting block rdd_283_93 to disk instead.
    25/11/18 18:14:17 WARN MemoryStore: Not enough space to cache rdd_283_93 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:17 WARN MemoryStore: Not enough space to cache rdd_283_95 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:17 WARN MemoryStore: Not enough space to cache rdd_283_93 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:17 WARN MemoryStore: Not enough space to cache rdd_283_95 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:17 WARN MemoryStore: Not enough space to cache rdd_283_96 in memory! (computed 7.8 MiB so far)
    25/11/18 18:14:17 WARN BlockManager: Persisting block rdd_283_96 to disk instead.
    25/11/18 18:14:17 WARN MemoryStore: Not enough space to cache rdd_283_98 in memory! (computed 2.3 MiB so far)
    25/11/18 18:14:17 WARN BlockManager: Persisting block rdd_283_98 to disk instead.
    25/11/18 18:14:17 WARN MemoryStore: Not enough space to cache rdd_283_99 in memory! (computed 3.5 MiB so far)
    25/11/18 18:14:17 WARN BlockManager: Persisting block rdd_283_99 to disk instead.
    25/11/18 18:14:18 WARN MemoryStore: Not enough space to cache rdd_283_97 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:18 WARN BlockManager: Persisting block rdd_283_97 to disk instead.
    25/11/18 18:14:17 WARN MemoryStore: Not enough space to cache rdd_283_96 in memory! (computed 7.8 MiB so far)
    25/11/18 18:14:17 WARN BlockManager: Persisting block rdd_283_96 to disk instead.
    25/11/18 18:14:17 WARN MemoryStore: Not enough space to cache rdd_283_98 in memory! (computed 2.3 MiB so far)
    25/11/18 18:14:17 WARN BlockManager: Persisting block rdd_283_98 to disk instead.
    25/11/18 18:14:17 WARN MemoryStore: Not enough space to cache rdd_283_99 in memory! (computed 3.5 MiB so far)
    25/11/18 18:14:17 WARN BlockManager: Persisting block rdd_283_99 to disk instead.
    25/11/18 18:14:18 WARN MemoryStore: Not enough space to cache rdd_283_97 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:18 WARN BlockManager: Persisting block rdd_283_97 to disk instead.
    25/11/18 18:14:18 WARN MemoryStore: Not enough space to cache rdd_283_98 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:18 WARN MemoryStore: Not enough space to cache rdd_283_96 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:18 WARN MemoryStore: Not enough space to cache rdd_283_99 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:18 WARN MemoryStore: Not enough space to cache rdd_283_98 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:18 WARN MemoryStore: Not enough space to cache rdd_283_96 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:18 WARN MemoryStore: Not enough space to cache rdd_283_99 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:19 WARN MemoryStore: Not enough space to cache rdd_283_97 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:19 WARN MemoryStore: Not enough space to cache rdd_283_97 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:21 WARN MemoryStore: Not enough space to cache rdd_283_100 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:21 WARN BlockManager: Persisting block rdd_283_100 to disk instead.
    25/11/18 18:14:21 WARN MemoryStore: Not enough space to cache rdd_283_101 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:21 WARN BlockManager: Persisting block rdd_283_101 to disk instead.
    25/11/18 18:14:21 WARN MemoryStore: Not enough space to cache rdd_283_100 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:21 WARN BlockManager: Persisting block rdd_283_100 to disk instead.
    25/11/18 18:14:21 WARN MemoryStore: Not enough space to cache rdd_283_101 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:21 WARN BlockManager: Persisting block rdd_283_101 to disk instead.
    25/11/18 18:14:22 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 5.2 MiB so far)
    25/11/18 18:14:22 WARN BlockManager: Persisting block rdd_283_104 to disk instead.
    25/11/18 18:14:22 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 5.2 MiB so far)
    25/11/18 18:14:22 WARN BlockManager: Persisting block rdd_283_104 to disk instead.
    25/11/18 18:14:23 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:23 WARN BlockManager: Persisting block rdd_283_103 to disk instead.
    25/11/18 18:14:23 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 7.8 MiB so far)
    25/11/18 18:14:23 WARN BlockManager: Persisting block rdd_283_105 to disk instead.
    25/11/18 18:14:23 WARN MemoryStore: Not enough space to cache rdd_283_108 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:23 WARN BlockManager: Persisting block rdd_283_108 to disk instead.
    25/11/18 18:14:23 WARN MemoryStore: Not enough space to cache rdd_283_107 in memory! (computed 1030.6 KiB so far)
    25/11/18 18:14:23 WARN BlockManager: Persisting block rdd_283_107 to disk instead.
    25/11/18 18:14:23 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:23 WARN BlockManager: Persisting block rdd_283_103 to disk instead.
    25/11/18 18:14:23 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 7.8 MiB so far)
    25/11/18 18:14:23 WARN BlockManager: Persisting block rdd_283_105 to disk instead.
    25/11/18 18:14:23 WARN MemoryStore: Not enough space to cache rdd_283_108 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:23 WARN BlockManager: Persisting block rdd_283_108 to disk instead.
    25/11/18 18:14:23 WARN MemoryStore: Not enough space to cache rdd_283_107 in memory! (computed 1030.6 KiB so far)
    25/11/18 18:14:23 WARN BlockManager: Persisting block rdd_283_107 to disk instead.
    25/11/18 18:14:23 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 3.5 MiB so far)
    25/11/18 18:14:23 WARN BlockManager: Persisting block rdd_283_106 to disk instead.
    25/11/18 18:14:23 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 3.5 MiB so far)
    25/11/18 18:14:23 WARN BlockManager: Persisting block rdd_283_106 to disk instead.
    25/11/18 18:14:24 WARN MemoryStore: Not enough space to cache rdd_283_102 in memory! (computed 94.7 MiB so far)
    25/11/18 18:14:24 WARN BlockManager: Persisting block rdd_283_102 to disk instead.
    25/11/18 18:14:24 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:24 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:24 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:24 WARN MemoryStore: Not enough space to cache rdd_283_102 in memory! (computed 94.7 MiB so far)
    25/11/18 18:14:24 WARN BlockManager: Persisting block rdd_283_102 to disk instead.
    25/11/18 18:14:24 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:24 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:24 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:24 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:24 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 41.9 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:27 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:29 WARN MemoryStore: Not enough space to cache rdd_283_8 in memory! (computed 62.9 MiB so far)
    25/11/18 18:14:29 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 62.9 MiB so far)
    25/11/18 18:14:29 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 62.9 MiB so far)
    25/11/18 18:14:29 WARN MemoryStore: Not enough space to cache rdd_283_8 in memory! (computed 62.9 MiB so far)
    25/11/18 18:14:29 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 62.9 MiB so far)
    25/11/18 18:14:29 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 62.9 MiB so far)
    25/11/18 18:14:41 WARN MemoryStore: Not enough space to cache rdd_283_90 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:41 WARN MemoryStore: Not enough space to cache rdd_283_93 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:14:41 WARN MemoryStore: Not enough space to cache rdd_283_90 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:41 WARN MemoryStore: Not enough space to cache rdd_283_93 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:14:41 WARN MemoryStore: Not enough space to cache rdd_283_95 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:14:42 WARN MemoryStore: Not enough space to cache rdd_283_96 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:41 WARN MemoryStore: Not enough space to cache rdd_283_95 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:14:42 WARN MemoryStore: Not enough space to cache rdd_283_96 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:42 WARN MemoryStore: Not enough space to cache rdd_283_97 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:14:42 WARN MemoryStore: Not enough space to cache rdd_283_97 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:14:42 WARN MemoryStore: Not enough space to cache rdd_283_98 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:42 WARN MemoryStore: Not enough space to cache rdd_283_99 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:14:42 WARN MemoryStore: Not enough space to cache rdd_283_98 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:42 WARN MemoryStore: Not enough space to cache rdd_283_99 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:14:43 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:43 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:43 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:14:43 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:14:43 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 7.8 MiB so far)
    25/11/18 18:14:43 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 7.8 MiB so far)
    25/11/18 18:14:43 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 7.8 MiB so far)
    25/11/18 18:14:43 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 7.8 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 3.5 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 3.5 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 5.2 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 5.2 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 18.4 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:45 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 11.8 MiB so far)
    25/11/18 18:14:47 WARN MemoryStore: Not enough space to cache rdd_283_8 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:47 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:47 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:47 WARN MemoryStore: Not enough space to cache rdd_283_11 in memory! (computed 7.8 MiB so far)
    25/11/18 18:14:47 WARN MemoryStore: Not enough space to cache rdd_283_8 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:47 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:47 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 28.0 MiB so far)
    25/11/18 18:14:47 WARN MemoryStore: Not enough space to cache rdd_283_11 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:02 WARN MemoryStore: Not enough space to cache rdd_283_93 in memory! (computed 41.9 MiB so far)
    25/11/18 18:15:02 WARN MemoryStore: Not enough space to cache rdd_283_95 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:02 WARN MemoryStore: Not enough space to cache rdd_283_93 in memory! (computed 41.9 MiB so far)
    25/11/18 18:15:02 WARN MemoryStore: Not enough space to cache rdd_283_95 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:02 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_96 in memory.
    25/11/18 18:15:03 WARN MemoryStore: Not enough space to cache rdd_283_96 in memory! (computed 384.0 B so far)
    25/11/18 18:15:02 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_96 in memory.
    25/11/18 18:15:03 WARN MemoryStore: Not enough space to cache rdd_283_96 in memory! (computed 384.0 B so far)
    25/11/18 18:15:03 WARN MemoryStore: Not enough space to cache rdd_283_97 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:03 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_98 in memory.
    25/11/18 18:15:03 WARN MemoryStore: Not enough space to cache rdd_283_98 in memory! (computed 384.0 B so far)
    25/11/18 18:15:03 WARN MemoryStore: Not enough space to cache rdd_283_97 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:03 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_98 in memory.
    25/11/18 18:15:03 WARN MemoryStore: Not enough space to cache rdd_283_98 in memory! (computed 384.0 B so far)
    25/11/18 18:15:03 WARN MemoryStore: Not enough space to cache rdd_283_99 in memory! (computed 41.9 MiB so far)
    25/11/18 18:15:03 WARN MemoryStore: Not enough space to cache rdd_283_99 in memory! (computed 41.9 MiB so far)
    25/11/18 18:15:04 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:04 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:04 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:04 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_105 in memory.
    25/11/18 18:15:04 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 384.0 B so far)
    25/11/18 18:15:04 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:04 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_105 in memory.
    25/11/18 18:15:04 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 384.0 B so far)
    25/11/18 18:15:05 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 41.9 MiB so far)
    25/11/18 18:15:05 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 41.9 MiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 5.2 MiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 3.5 MiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 3.5 MiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 11.8 MiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 11.8 MiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 5.2 MiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 5.2 MiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 3.5 MiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 3.5 MiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 11.8 MiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 11.8 MiB so far)
    25/11/18 18:15:06 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 5.2 MiB so far)
    25/11/18 18:15:09 WARN MemoryStore: Not enough space to cache rdd_283_8 in memory! (computed 41.9 MiB so far)
    25/11/18 18:15:09 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:09 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_10 in memory.
    25/11/18 18:15:09 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 384.0 B so far)
    25/11/18 18:15:09 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_11 in memory.
    25/11/18 18:15:09 WARN MemoryStore: Not enough space to cache rdd_283_11 in memory! (computed 384.0 B so far)
    25/11/18 18:15:09 WARN MemoryStore: Not enough space to cache rdd_283_8 in memory! (computed 41.9 MiB so far)
    25/11/18 18:15:09 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:09 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_10 in memory.
    25/11/18 18:15:09 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 384.0 B so far)
    25/11/18 18:15:09 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_11 in memory.
    25/11/18 18:15:09 WARN MemoryStore: Not enough space to cache rdd_283_11 in memory! (computed 384.0 B so far)
    25/11/18 18:15:30 WARN MemoryStore: Not enough space to cache rdd_283_93 in memory! (computed 41.9 MiB so far)
    25/11/18 18:15:30 WARN MemoryStore: Not enough space to cache rdd_283_95 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:30 WARN MemoryStore: Not enough space to cache rdd_283_93 in memory! (computed 41.9 MiB so far)
    25/11/18 18:15:30 WARN MemoryStore: Not enough space to cache rdd_283_95 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:30 WARN MemoryStore: Not enough space to cache rdd_283_96 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:30 WARN MemoryStore: Not enough space to cache rdd_283_96 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:31 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_97 in memory.
    25/11/18 18:15:31 WARN MemoryStore: Not enough space to cache rdd_283_97 in memory! (computed 384.0 B so far)
    25/11/18 18:15:31 WARN MemoryStore: Not enough space to cache rdd_283_98 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:31 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_97 in memory.
    25/11/18 18:15:31 WARN MemoryStore: Not enough space to cache rdd_283_97 in memory! (computed 384.0 B so far)
    25/11/18 18:15:31 WARN MemoryStore: Not enough space to cache rdd_283_98 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:31 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_99 in memory.
    25/11/18 18:15:31 WARN MemoryStore: Not enough space to cache rdd_283_99 in memory! (computed 384.0 B so far)
    25/11/18 18:15:31 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_99 in memory.
    25/11/18 18:15:31 WARN MemoryStore: Not enough space to cache rdd_283_99 in memory! (computed 384.0 B so far)
    25/11/18 18:15:32 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 41.9 MiB so far)
    25/11/18 18:15:32 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 41.9 MiB so far)
    25/11/18 18:15:33 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:33 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_105 in memory.
    25/11/18 18:15:33 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 384.0 B so far)
    25/11/18 18:15:33 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:33 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_105 in memory.
    25/11/18 18:15:33 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 384.0 B so far)
    25/11/18 18:15:33 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_106 in memory.
    25/11/18 18:15:33 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 384.0 B so far)
    25/11/18 18:15:33 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_106 in memory.
    25/11/18 18:15:33 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 384.0 B so far)
    25/11/18 18:15:35 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 2.3 MiB so far)
    25/11/18 18:15:35 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 2.3 MiB so far)
    25/11/18 18:15:35 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 2.3 MiB so far)
    25/11/18 18:15:35 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 2.3 MiB so far)
    25/11/18 18:15:35 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:35 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 7.8 MiB so far)
    25/11/18 18:15:36 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 18.4 MiB so far)
    25/11/18 18:15:36 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 18.4 MiB so far)
    25/11/18 18:15:36 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 28.0 MiB so far)
    25/11/18 18:15:36 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 18.4 MiB so far)
    25/11/18 18:15:36 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 28.0 MiB so far)
    25/11/18 18:15:36 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 18.4 MiB so far)
    25/11/18 18:15:36 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 18.4 MiB so far)
    25/11/18 18:15:36 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 28.0 MiB so far)
    25/11/18 18:15:36 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 18.4 MiB so far)
    25/11/18 18:15:36 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 28.0 MiB so far)
    25/11/18 18:15:38 WARN MemoryStore: Not enough space to cache rdd_283_8 in memory! (computed 62.9 MiB so far)
    25/11/18 18:15:38 WARN MemoryStore: Not enough space to cache rdd_283_8 in memory! (computed 62.9 MiB so far)
    25/11/18 18:15:38 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 18.4 MiB so far)
    25/11/18 18:15:38 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 28.0 MiB so far)
    25/11/18 18:15:38 WARN MemoryStore: Not enough space to cache rdd_283_11 in memory! (computed 11.8 MiB so far)
    25/11/18 18:15:38 WARN MemoryStore: Not enough space to cache rdd_283_12 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:15:38 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 18.4 MiB so far)
    25/11/18 18:15:38 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 28.0 MiB so far)
    25/11/18 18:15:38 WARN MemoryStore: Not enough space to cache rdd_283_11 in memory! (computed 11.8 MiB so far)
    25/11/18 18:15:38 WARN MemoryStore: Not enough space to cache rdd_283_12 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:16:04 WARN MemoryStore: Not enough space to cache rdd_283_96 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:04 WARN MemoryStore: Not enough space to cache rdd_283_96 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:05 WARN MemoryStore: Not enough space to cache rdd_283_97 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:05 WARN MemoryStore: Not enough space to cache rdd_283_97 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:05 WARN MemoryStore: Not enough space to cache rdd_283_98 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:05 WARN MemoryStore: Not enough space to cache rdd_283_98 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:05 WARN MemoryStore: Not enough space to cache rdd_283_99 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:05 WARN MemoryStore: Not enough space to cache rdd_283_99 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:06 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:06 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:07 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:07 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_105 in memory.
    25/11/18 18:16:07 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 384.0 B so far)
    25/11/18 18:16:07 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_106 in memory.
    25/11/18 18:16:07 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 384.0 B so far)
    25/11/18 18:16:07 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:07 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_105 in memory.
    25/11/18 18:16:07 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 384.0 B so far)
    25/11/18 18:16:07 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_106 in memory.
    25/11/18 18:16:07 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 384.0 B so far)
    25/11/18 18:16:10 WARN DAGScheduler: Broadcasting large task binary with size 1057.8 KiB
    25/11/18 18:16:10 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 5.2 MiB so far)
    25/11/18 18:16:10 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:10 WARN DAGScheduler: Broadcasting large task binary with size 1057.8 KiB
    25/11/18 18:16:10 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 5.2 MiB so far)
    25/11/18 18:16:10 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:11 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 7.8 MiB so far)
    25/11/18 18:16:11 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 11.8 MiB so far)
    25/11/18 18:16:11 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 5.2 MiB so far)
    25/11/18 18:16:11 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 11.8 MiB so far)
    25/11/18 18:16:11 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 11.8 MiB so far)
    25/11/18 18:16:11 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 28.0 MiB so far)
    25/11/18 18:16:11 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 7.8 MiB so far)
    25/11/18 18:16:11 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 11.8 MiB so far)
    25/11/18 18:16:11 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 5.2 MiB so far)
    25/11/18 18:16:11 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 11.8 MiB so far)
    25/11/18 18:16:11 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 11.8 MiB so far)
    25/11/18 18:16:11 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 28.0 MiB so far)
    25/11/18 18:16:14 WARN MemoryStore: Not enough space to cache rdd_283_8 in memory! (computed 41.9 MiB so far)
    25/11/18 18:16:14 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:16:14 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_11 in memory.
    25/11/18 18:16:14 WARN MemoryStore: Not enough space to cache rdd_283_11 in memory! (computed 384.0 B so far)
    25/11/18 18:16:14 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 41.9 MiB so far)
    25/11/18 18:16:14 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_12 in memory.
    25/11/18 18:16:14 WARN MemoryStore: Not enough space to cache rdd_283_12 in memory! (computed 384.0 B so far)
    25/11/18 18:16:14 WARN MemoryStore: Not enough space to cache rdd_283_8 in memory! (computed 41.9 MiB so far)
    25/11/18 18:16:14 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:16:14 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_11 in memory.
    25/11/18 18:16:14 WARN MemoryStore: Not enough space to cache rdd_283_11 in memory! (computed 384.0 B so far)
    25/11/18 18:16:14 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 41.9 MiB so far)
    25/11/18 18:16:14 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_12 in memory.
    25/11/18 18:16:14 WARN MemoryStore: Not enough space to cache rdd_283_12 in memory! (computed 384.0 B so far)
    25/11/18 18:16:14 WARN MemoryStore: Not enough space to cache rdd_283_13 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:16:14 WARN MemoryStore: Not enough space to cache rdd_283_13 in memory! (computed 1549.7 KiB so far)
    25/11/18 18:16:48 WARN MemoryStore: Not enough space to cache rdd_283_97 in memory! (computed 28.0 MiB so far)
    25/11/18 18:16:48 WARN MemoryStore: Not enough space to cache rdd_283_97 in memory! (computed 28.0 MiB so far)
    25/11/18 18:16:48 WARN MemoryStore: Not enough space to cache rdd_283_98 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:48 WARN MemoryStore: Not enough space to cache rdd_283_98 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:48 WARN MemoryStore: Not enough space to cache rdd_283_99 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:48 WARN MemoryStore: Not enough space to cache rdd_283_99 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:49 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:49 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:49 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 28.0 MiB so far)
    25/11/18 18:16:49 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 28.0 MiB so far)
    25/11/18 18:16:50 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:50 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_106 in memory.
    25/11/18 18:16:50 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 384.0 B so far)
    25/11/18 18:16:50 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 2.3 MiB so far)
    25/11/18 18:16:50 WARN MemoryStore: Failed to reserve initial memory threshold of 1024.0 KiB for computing block rdd_283_106 in memory.
    25/11/18 18:16:50 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 384.0 B so far)
    25/11/18 18:16:55 WARN DAGScheduler: Broadcasting large task binary with size 2002.2 KiB
    25/11/18 18:16:55 WARN DAGScheduler: Broadcasting large task binary with size 2002.2 KiB
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 5.2 MiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 5.2 MiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 3.5 MiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 1030.6 KiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 1030.6 KiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 5.2 MiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 5.2 MiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 3.5 MiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 1030.6 KiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 1030.6 KiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 28.0 MiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 28.0 MiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 28.0 MiB so far)
    25/11/18 18:16:55 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 28.0 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_8 in memory! (computed 41.9 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 28.0 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_8 in memory! (computed 41.9 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 28.0 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_11 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_14 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_12 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_15 in memory! (computed 7.8 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_13 in memory! (computed 11.8 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_11 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_14 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_12 in memory! (computed 18.4 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_15 in memory! (computed 7.8 MiB so far)
    25/11/18 18:16:59 WARN MemoryStore: Not enough space to cache rdd_283_13 in memory! (computed 11.8 MiB so far)
    25/11/18 18:17:40 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 7.8 MiB so far)
    25/11/18 18:17:40 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 7.8 MiB so far)
    25/11/18 18:17:41 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 41.9 MiB so far)
    25/11/18 18:17:41 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 41.9 MiB so far)
    25/11/18 18:17:42 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 18.4 MiB so far)
    25/11/18 18:17:42 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 18.4 MiB so far)
    25/11/18 18:17:42 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 5.2 MiB so far)
    25/11/18 18:17:42 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 5.2 MiB so far)
    25/11/18 18:17:47 WARN DAGScheduler: Broadcasting large task binary with size 3.8 MiB
    25/11/18 18:17:47 WARN DAGScheduler: Broadcasting large task binary with size 3.8 MiB
    25/11/18 18:17:47 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 5.2 MiB so far)
    25/11/18 18:17:47 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 11.8 MiB so far)
    25/11/18 18:17:47 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 5.2 MiB so far)
    25/11/18 18:17:47 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 7.8 MiB so far)
    25/11/18 18:17:47 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 11.8 MiB so far)
    25/11/18 18:17:47 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 11.8 MiB so far)
    25/11/18 18:17:47 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 18.4 MiB so far)
    25/11/18 18:17:47 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 5.2 MiB so far)
    25/11/18 18:17:47 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 11.8 MiB so far)
    25/11/18 18:17:47 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 5.2 MiB so far)
    25/11/18 18:17:47 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 7.8 MiB so far)
    25/11/18 18:17:47 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 11.8 MiB so far)
    25/11/18 18:17:47 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 11.8 MiB so far)
    25/11/18 18:17:47 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 18.4 MiB so far)
    25/11/18 18:17:48 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 62.9 MiB so far)
    25/11/18 18:17:48 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 62.9 MiB so far)
    25/11/18 18:17:53 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 18.4 MiB so far)
    25/11/18 18:17:53 WARN MemoryStore: Not enough space to cache rdd_283_11 in memory! (computed 7.8 MiB so far)
    25/11/18 18:17:53 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 41.9 MiB so far)
    25/11/18 18:17:53 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 18.4 MiB so far)
    25/11/18 18:17:53 WARN MemoryStore: Not enough space to cache rdd_283_11 in memory! (computed 7.8 MiB so far)
    25/11/18 18:17:53 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 41.9 MiB so far)
    25/11/18 18:17:55 WARN MemoryStore: Not enough space to cache rdd_283_13 in memory! (computed 41.9 MiB so far)
    25/11/18 18:17:55 WARN MemoryStore: Not enough space to cache rdd_283_15 in memory! (computed 5.2 MiB so far)
    25/11/18 18:17:55 WARN MemoryStore: Not enough space to cache rdd_283_14 in memory! (computed 7.8 MiB so far)
    25/11/18 18:17:55 WARN MemoryStore: Not enough space to cache rdd_283_12 in memory! (computed 41.9 MiB so far)
    25/11/18 18:17:55 WARN MemoryStore: Not enough space to cache rdd_283_13 in memory! (computed 41.9 MiB so far)
    25/11/18 18:17:55 WARN MemoryStore: Not enough space to cache rdd_283_15 in memory! (computed 5.2 MiB so far)
    25/11/18 18:17:55 WARN MemoryStore: Not enough space to cache rdd_283_14 in memory! (computed 7.8 MiB so far)
    25/11/18 18:17:55 WARN MemoryStore: Not enough space to cache rdd_283_12 in memory! (computed 41.9 MiB so far)
    25/11/18 18:17:59 WARN MemoryStore: Not enough space to cache rdd_283_18 in memory! (computed 18.4 MiB so far)
    25/11/18 18:17:59 WARN MemoryStore: Not enough space to cache rdd_283_18 in memory! (computed 18.4 MiB so far)
    25/11/18 18:17:59 WARN MemoryStore: Not enough space to cache rdd_283_17 in memory! (computed 62.9 MiB so far)
    25/11/18 18:17:59 WARN MemoryStore: Not enough space to cache rdd_283_17 in memory! (computed 62.9 MiB so far)
    25/11/18 18:18:00 WARN MemoryStore: Not enough space to cache rdd_283_20 in memory! (computed 62.9 MiB so far)
    25/11/18 18:18:00 WARN MemoryStore: Not enough space to cache rdd_283_20 in memory! (computed 62.9 MiB so far)
    25/11/18 18:18:41 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 7.8 MiB so far)
    25/11/18 18:18:41 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 62.9 MiB so far)
    25/11/18 18:18:41 WARN MemoryStore: Not enough space to cache rdd_283_105 in memory! (computed 7.8 MiB so far)
    25/11/18 18:18:41 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 62.9 MiB so far)
    25/11/18 18:18:41 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 18.4 MiB so far)
    25/11/18 18:18:41 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 18.4 MiB so far)
    25/11/18 18:18:42 WARN MemoryStore: Not enough space to cache rdd_283_108 in memory! (computed 7.8 MiB so far)
    25/11/18 18:18:42 WARN MemoryStore: Not enough space to cache rdd_283_108 in memory! (computed 7.8 MiB so far)
    25/11/18 18:18:45 WARN DAGScheduler: Broadcasting large task binary with size 1186.2 KiB
    25/11/18 18:18:45 WARN DAGScheduler: Broadcasting large task binary with size 1186.2 KiB
    25/11/18 18:18:47 WARN DAGScheduler: Broadcasting large task binary with size 7.2 MiB
    25/11/18 18:18:47 WARN DAGScheduler: Broadcasting large task binary with size 7.2 MiB
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 18.4 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 11.8 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 11.8 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 7.8 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 11.8 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 5.2 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 7.8 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 7.8 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 18.4 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 11.8 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 11.8 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 7.8 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 11.8 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 5.2 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 7.8 MiB so far)
    25/11/18 18:18:47 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 7.8 MiB so far)
    25/11/18 18:18:53 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 41.9 MiB so far)
    25/11/18 18:18:54 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 28.0 MiB so far)
    25/11/18 18:18:53 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 41.9 MiB so far)
    25/11/18 18:18:54 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 28.0 MiB so far)
    25/11/18 18:18:54 WARN MemoryStore: Not enough space to cache rdd_283_14 in memory! (computed 18.4 MiB so far)
    25/11/18 18:18:54 WARN MemoryStore: Not enough space to cache rdd_283_15 in memory! (computed 18.4 MiB so far)
    25/11/18 18:18:54 WARN MemoryStore: Not enough space to cache rdd_283_11 in memory! (computed 41.9 MiB so far)
    25/11/18 18:18:54 WARN MemoryStore: Not enough space to cache rdd_283_13 in memory! (computed 18.4 MiB so far)
    25/11/18 18:18:54 WARN MemoryStore: Not enough space to cache rdd_283_12 in memory! (computed 62.9 MiB so far)
    25/11/18 18:18:54 WARN MemoryStore: Not enough space to cache rdd_283_14 in memory! (computed 18.4 MiB so far)
    25/11/18 18:18:54 WARN MemoryStore: Not enough space to cache rdd_283_15 in memory! (computed 18.4 MiB so far)
    25/11/18 18:18:54 WARN MemoryStore: Not enough space to cache rdd_283_11 in memory! (computed 41.9 MiB so far)
    25/11/18 18:18:54 WARN MemoryStore: Not enough space to cache rdd_283_13 in memory! (computed 18.4 MiB so far)
    25/11/18 18:18:54 WARN MemoryStore: Not enough space to cache rdd_283_12 in memory! (computed 62.9 MiB so far)
    25/11/18 18:19:00 WARN MemoryStore: Not enough space to cache rdd_283_17 in memory! (computed 41.9 MiB so far)
    25/11/18 18:19:00 WARN MemoryStore: Not enough space to cache rdd_283_17 in memory! (computed 41.9 MiB so far)
    25/11/18 18:19:01 WARN MemoryStore: Not enough space to cache rdd_283_21 in memory! (computed 41.9 MiB so far)
    25/11/18 18:19:01 WARN MemoryStore: Not enough space to cache rdd_283_22 in memory! (computed 41.9 MiB so far)
    25/11/18 18:19:01 WARN MemoryStore: Not enough space to cache rdd_283_21 in memory! (computed 41.9 MiB so far)
    25/11/18 18:19:01 WARN MemoryStore: Not enough space to cache rdd_283_22 in memory! (computed 41.9 MiB so far)
    25/11/18 18:19:01 WARN MemoryStore: Not enough space to cache rdd_283_23 in memory! (computed 18.4 MiB so far)
    25/11/18 18:19:01 WARN MemoryStore: Not enough space to cache rdd_283_20 in memory! (computed 62.9 MiB so far)
    25/11/18 18:19:01 WARN MemoryStore: Not enough space to cache rdd_283_23 in memory! (computed 18.4 MiB so far)
    25/11/18 18:19:01 WARN MemoryStore: Not enough space to cache rdd_283_20 in memory! (computed 62.9 MiB so far)
    25/11/18 18:19:55 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 41.9 MiB so far)
    25/11/18 18:19:55 WARN MemoryStore: Not enough space to cache rdd_283_106 in memory! (computed 41.9 MiB so far)
    25/11/18 18:20:00 WARN DAGScheduler: Broadcasting large task binary with size 2.2 MiB
    25/11/18 18:20:00 WARN DAGScheduler: Broadcasting large task binary with size 2.2 MiB
    25/11/18 18:20:03 WARN DAGScheduler: Broadcasting large task binary with size 13.4 MiB
    25/11/18 18:20:03 WARN DAGScheduler: Broadcasting large task binary with size 13.4 MiB
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 18.4 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 7.8 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 11.8 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 18.4 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 11.8 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 7.8 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 28.0 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 18.4 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 18.4 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 7.8 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 11.8 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 18.4 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 11.8 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 7.8 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 28.0 MiB so far)
    25/11/18 18:20:03 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 18.4 MiB so far)
    25/11/18 18:20:11 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 62.9 MiB so far)
    25/11/18 18:20:11 WARN MemoryStore: Not enough space to cache rdd_283_9 in memory! (computed 62.9 MiB so far)
    25/11/18 18:20:12 WARN MemoryStore: Not enough space to cache rdd_283_12 in memory! (computed 1030.6 KiB so far)
    25/11/18 18:20:12 WARN MemoryStore: Not enough space to cache rdd_283_13 in memory! (computed 1030.6 KiB so far)
    25/11/18 18:20:12 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 62.9 MiB so far)
    25/11/18 18:20:12 WARN MemoryStore: Not enough space to cache rdd_283_12 in memory! (computed 1030.6 KiB so far)
    25/11/18 18:20:12 WARN MemoryStore: Not enough space to cache rdd_283_13 in memory! (computed 1030.6 KiB so far)
    25/11/18 18:20:12 WARN MemoryStore: Not enough space to cache rdd_283_10 in memory! (computed 62.9 MiB so far)
    25/11/18 18:20:12 WARN MemoryStore: Not enough space to cache rdd_283_15 in memory! (computed 62.9 MiB so far)
    25/11/18 18:20:12 WARN MemoryStore: Not enough space to cache rdd_283_15 in memory! (computed 62.9 MiB so far)
    25/11/18 18:20:20 WARN MemoryStore: Not enough space to cache rdd_283_19 in memory! (computed 62.9 MiB so far)
    25/11/18 18:20:20 WARN MemoryStore: Not enough space to cache rdd_283_18 in memory! (computed 62.9 MiB so far)
    25/11/18 18:20:20 WARN MemoryStore: Not enough space to cache rdd_283_21 in memory! (computed 28.0 MiB so far)
    25/11/18 18:20:20 WARN MemoryStore: Not enough space to cache rdd_283_22 in memory! (computed 41.9 MiB so far)
    25/11/18 18:20:20 WARN MemoryStore: Not enough space to cache rdd_283_19 in memory! (computed 62.9 MiB so far)
    25/11/18 18:20:20 WARN MemoryStore: Not enough space to cache rdd_283_18 in memory! (computed 62.9 MiB so far)
    25/11/18 18:20:20 WARN MemoryStore: Not enough space to cache rdd_283_21 in memory! (computed 28.0 MiB so far)
    25/11/18 18:20:20 WARN MemoryStore: Not enough space to cache rdd_283_22 in memory! (computed 41.9 MiB so far)
    25/11/18 18:20:29 WARN MemoryStore: Not enough space to cache rdd_283_31 in memory! (computed 18.4 MiB so far)
    25/11/18 18:20:29 WARN MemoryStore: Not enough space to cache rdd_283_28 in memory! (computed 62.9 MiB so far)
    25/11/18 18:20:29 WARN MemoryStore: Not enough space to cache rdd_283_30 in memory! (computed 62.9 MiB so far)
    25/11/18 18:20:29 WARN MemoryStore: Not enough space to cache rdd_283_31 in memory! (computed 18.4 MiB so far)
    25/11/18 18:20:29 WARN MemoryStore: Not enough space to cache rdd_283_28 in memory! (computed 62.9 MiB so far)
    25/11/18 18:20:29 WARN MemoryStore: Not enough space to cache rdd_283_30 in memory! (computed 62.9 MiB so far)
    25/11/18 18:21:28 WARN MemoryStore: Not enough space to cache rdd_283_86 in memory! (computed 41.9 MiB so far)
    25/11/18 18:21:28 WARN MemoryStore: Not enough space to cache rdd_283_86 in memory! (computed 41.9 MiB so far)
    25/11/18 18:21:51 WARN DAGScheduler: Broadcasting large task binary with size 3.9 MiB
    25/11/18 18:21:51 WARN DAGScheduler: Broadcasting large task binary with size 3.9 MiB
    25/11/18 18:21:56 WARN DAGScheduler: Broadcasting large task binary with size 23.9 MiB
    25/11/18 18:21:56 WARN DAGScheduler: Broadcasting large task binary with size 23.9 MiB
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 5.2 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 5.2 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 18.4 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 7.8 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 11.8 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 28.0 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 5.2 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 5.2 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 18.4 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 7.8 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 11.8 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 28.0 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 41.9 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 62.9 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 41.9 MiB so far)
    25/11/18 18:21:57 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 62.9 MiB so far)
    25/11/18 18:22:20 WARN MemoryStore: Not enough space to cache rdd_283_21 in memory! (computed 41.9 MiB so far)
    25/11/18 18:22:20 WARN MemoryStore: Not enough space to cache rdd_283_21 in memory! (computed 41.9 MiB so far)
    25/11/18 18:22:22 WARN MemoryStore: Not enough space to cache rdd_283_22 in memory! (computed 62.9 MiB so far)
    25/11/18 18:22:22 WARN MemoryStore: Not enough space to cache rdd_283_22 in memory! (computed 62.9 MiB so far)
    25/11/18 18:22:43 WARN MemoryStore: Not enough space to cache rdd_283_37 in memory! (computed 28.0 MiB so far)
    25/11/18 18:22:43 WARN MemoryStore: Not enough space to cache rdd_283_37 in memory! (computed 28.0 MiB so far)
    25/11/18 18:22:44 WARN MemoryStore: Not enough space to cache rdd_283_38 in memory! (computed 28.0 MiB so far)
    25/11/18 18:22:44 WARN MemoryStore: Not enough space to cache rdd_283_38 in memory! (computed 28.0 MiB so far)
    25/11/18 18:23:21 WARN MemoryStore: Not enough space to cache rdd_283_63 in memory! (computed 7.8 MiB so far)
    25/11/18 18:23:21 WARN MemoryStore: Not enough space to cache rdd_283_63 in memory! (computed 7.8 MiB so far)
    25/11/18 18:23:29 WARN MemoryStore: Not enough space to cache rdd_283_68 in memory! (computed 41.9 MiB so far)
    25/11/18 18:23:29 WARN MemoryStore: Not enough space to cache rdd_283_68 in memory! (computed 41.9 MiB so far)
    25/11/18 18:23:31 WARN MemoryStore: Not enough space to cache rdd_283_70 in memory! (computed 41.9 MiB so far)
    25/11/18 18:23:31 WARN MemoryStore: Not enough space to cache rdd_283_70 in memory! (computed 41.9 MiB so far)
    25/11/18 18:23:36 WARN MemoryStore: Not enough space to cache rdd_283_73 in memory! (computed 18.4 MiB so far)
    25/11/18 18:23:36 WARN MemoryStore: Not enough space to cache rdd_283_73 in memory! (computed 18.4 MiB so far)
    25/11/18 18:23:41 WARN MemoryStore: Not enough space to cache rdd_283_77 in memory! (computed 41.9 MiB so far)
    25/11/18 18:23:41 WARN MemoryStore: Not enough space to cache rdd_283_77 in memory! (computed 41.9 MiB so far)
    25/11/18 18:23:42 WARN MemoryStore: Not enough space to cache rdd_283_78 in memory! (computed 18.4 MiB so far)
    25/11/18 18:23:42 WARN MemoryStore: Not enough space to cache rdd_283_78 in memory! (computed 18.4 MiB so far)
    25/11/18 18:23:48 WARN MemoryStore: Not enough space to cache rdd_283_82 in memory! (computed 7.8 MiB so far)
    25/11/18 18:23:48 WARN MemoryStore: Not enough space to cache rdd_283_82 in memory! (computed 7.8 MiB so far)
    25/11/18 18:23:52 WARN MemoryStore: Not enough space to cache rdd_283_85 in memory! (computed 18.4 MiB so far)
    25/11/18 18:23:52 WARN MemoryStore: Not enough space to cache rdd_283_85 in memory! (computed 18.4 MiB so far)
    25/11/18 18:24:29 WARN DAGScheduler: Broadcasting large task binary with size 6.8 MiB
    25/11/18 18:24:29 WARN DAGScheduler: Broadcasting large task binary with size 6.8 MiB
    25/11/18 18:24:37 WARN DAGScheduler: Broadcasting large task binary with size 40.8 MiB
    25/11/18 18:24:37 WARN DAGScheduler: Broadcasting large task binary with size 40.8 MiB
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 28.0 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 28.0 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 41.9 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 41.9 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 41.9 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 62.9 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 62.9 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 62.9 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_2 in memory! (computed 28.0 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_3 in memory! (computed 28.0 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_1 in memory! (computed 41.9 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_4 in memory! (computed 41.9 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_7 in memory! (computed 41.9 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_6 in memory! (computed 62.9 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_0 in memory! (computed 62.9 MiB so far)
    25/11/18 18:24:38 WARN MemoryStore: Not enough space to cache rdd_283_5 in memory! (computed 62.9 MiB so far)
    25/11/18 18:25:25 WARN MemoryStore: Not enough space to cache rdd_283_26 in memory! (computed 11.8 MiB so far)
    25/11/18 18:25:25 WARN MemoryStore: Not enough space to cache rdd_283_26 in memory! (computed 11.8 MiB so far)
    25/11/18 18:25:37 WARN MemoryStore: Not enough space to cache rdd_283_32 in memory! (computed 62.9 MiB so far)
    25/11/18 18:25:37 WARN MemoryStore: Not enough space to cache rdd_283_32 in memory! (computed 62.9 MiB so far)
    25/11/18 18:26:21 WARN MemoryStore: Not enough space to cache rdd_283_52 in memory! (computed 62.9 MiB so far)
    25/11/18 18:26:21 WARN MemoryStore: Not enough space to cache rdd_283_52 in memory! (computed 62.9 MiB so far)
    25/11/18 18:26:23 WARN MemoryStore: Not enough space to cache rdd_283_53 in memory! (computed 28.0 MiB so far)
    25/11/18 18:26:23 WARN MemoryStore: Not enough space to cache rdd_283_53 in memory! (computed 28.0 MiB so far)
    25/11/18 18:26:40 WARN MemoryStore: Not enough space to cache rdd_283_61 in memory! (computed 7.8 MiB so far)
    25/11/18 18:26:40 WARN MemoryStore: Not enough space to cache rdd_283_61 in memory! (computed 7.8 MiB so far)
    25/11/18 18:27:09 WARN MemoryStore: Not enough space to cache rdd_283_74 in memory! (computed 41.9 MiB so far)
    25/11/18 18:27:09 WARN MemoryStore: Not enough space to cache rdd_283_74 in memory! (computed 41.9 MiB so far)
    25/11/18 18:28:06 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 18.4 MiB so far)
    25/11/18 18:28:06 WARN MemoryStore: Not enough space to cache rdd_283_103 in memory! (computed 18.4 MiB so far)
    25/11/18 18:28:08 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 41.9 MiB so far)
    25/11/18 18:28:08 WARN MemoryStore: Not enough space to cache rdd_283_104 in memory! (computed 41.9 MiB so far)
    25/11/18 18:28:23 WARN DAGScheduler: Broadcasting large task binary with size 10.8 MiB
    25/11/18 18:28:23 WARN DAGScheduler: Broadcasting large task binary with size 10.8 MiB
                                                                                    

    Model trained successfully!



```python
# Evaluate Fare Prediction Model
fare_predictions = fare_model.transform(fare_test)

fare_evaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction")

fare_rmse = fare_evaluator.evaluate(fare_predictions, {fare_evaluator.metricName: "rmse"})
fare_mae = fare_evaluator.evaluate(fare_predictions, {fare_evaluator.metricName: "mae"})
fare_r2 = fare_evaluator.evaluate(fare_predictions, {fare_evaluator.metricName: "r2"})

print("=" * 60)
print("FARE PREDICTION MODEL PERFORMANCE")
print("=" * 60)
print(f"RMSE: ${fare_rmse:.2f}")
print(f"MAE:  ${fare_mae:.2f}")
print(f"R²:   {fare_r2:.4f}")
print("=" * 60)

# Show sample predictions
fare_predictions.select(
    "trip_distance", "passenger_count", "pickup_hour", "taxi_type",
    "fare_amount", "prediction",
    (F.abs(F.col("fare_amount") - F.col("prediction"))).alias("error")
).orderBy("error", ascending=False).show(20)
```

                                                                                    

    ============================================================
    FARE PREDICTION MODEL PERFORMANCE
    ============================================================
    RMSE: $6.55
    MAE:  $3.44
    R²:   0.8181
    ============================================================


    [Stage 168:====================================================>(108 + 1) / 109]

    +-------------+---------------+-----------+---------+-----------+------------------+------------------+
    |trip_distance|passenger_count|pickup_hour|taxi_type|fare_amount|        prediction|             error|
    +-------------+---------------+-----------+---------+-----------+------------------+------------------+
    |         0.01|            2.0|         19|   yellow|      431.0| 9.329521759298053| 421.6704782407019|
    |         0.07|            1.0|         14|   yellow|      425.0| 7.151910594184289| 417.8480894058157|
    |        81.47|            1.0|         10|   yellow|      498.6|101.08530677097895| 397.5146932290211|
    |         0.13|            1.0|         13|   yellow|      400.0| 7.247407853830907| 392.7525921461691|
    |        84.77|            1.0|          6|   yellow|      493.0|100.57156486733098|392.42843513266905|
    |         0.04|            1.0|         21|   yellow|      400.0| 7.836953342160368|392.16304665783963|
    |         0.11|            2.0|         20|    green|      400.0|   9.8534753796777| 390.1465246203223|
    |        68.17|            1.0|         15|   yellow|      495.1| 108.1451097212876| 386.9548902787124|
    |        73.36|            1.0|         14|   yellow|      492.3|107.48065818758815|384.81934181241184|
    |         3.15|            4.0|         19|   yellow|      400.0|16.996918154092683|383.00308184590733|
    |         0.01|            3.0|         13|   yellow|      385.0| 9.080650104932515| 375.9193498950675|
    |        72.24|            2.0|         16|   yellow|      479.7|105.97456949321669| 373.7254305067833|
    |         4.22|            1.0|         17|   yellow|      390.0| 19.26231055582387| 370.7376894441761|
    |         0.13|            4.0|         21|   yellow|      380.0| 10.28157904823158| 369.7184209517684|
    |         96.3|            1.0|         16|   yellow|      480.0|110.43125762733155|369.56874237266845|
    |        66.87|            1.0|         10|   yellow|      469.9|101.08530677097895|368.81469322902103|
    |        70.99|            1.0|         18|   yellow|      474.8|108.55027715647489| 366.2497228435251|
    |        83.21|            1.0|         20|   yellow|      472.7|108.04346631314576|364.65653368685423|
    |         4.93|            4.0|          0|   yellow|      375.0|20.412896436965404| 354.5871035630346|
    |        73.43|            1.0|         17|   yellow|      462.0|  107.988088284601|  354.011911715399|
    +-------------+---------------+-----------+---------+-----------+------------------+------------------+
    only showing top 20 rows
    


                                                                                    


```python
# Save Fare Prediction Model and Predictions
fare_model.write().overwrite().save(f"{ML_MODELS_PATH}/fare_prediction")
fare_predictions.write.mode("overwrite").parquet(f"{ML_PREDICTIONS_PATH}/fare_prediction")

print("Fare prediction model and predictions saved!")
```

    25/11/18 18:30:31 WARN TaskSetManager: Stage 183 contains a task of very large size (4459 KiB). The maximum recommended task size is 1000 KiB.
    [Stage 186:====================================================>(108 + 1) / 109]

    Fare prediction model and predictions saved!


                                                                                    

---
## Model 3: Payment Type Classification

**Business Value**: Predict payment method to optimize payment processing and detect fraud patterns.

**Features**: Fare amount, trip distance, hour, taxi type, tip amount patterns


```python
# Prepare Payment Classification Dataset
payment_df = silver_df_clean.select(
    "fare_amount",
    "trip_distance",
    "tip_amount",
    "tolls_amount",
    "total_amount",
    "passenger_count",
    "pickup_hour",
    F.dayofweek("pickup_datetime").alias("day_of_week"),
    "taxi_type",
    F.col("payment_type").cast("string").alias("payment_type")
).filter(
    # Remove outliers and ensure valid payment types
    (F.col("fare_amount") > 0) & (F.col("fare_amount") < 500) &
    (F.col("trip_distance") > 0) & (F.col("trip_distance") < 100) &
    (F.col("payment_type").isNotNull()) &
    (F.col("tip_amount").isNotNull()) &
    (F.col("tolls_amount").isNotNull()) &
    (F.col("total_amount").isNotNull())
)

# Add engineered features with additional safety checks
payment_df = payment_df.withColumn(
    "tip_ratio",
    F.when(F.col("fare_amount") > 0, F.col("tip_amount") / F.col("fare_amount")).otherwise(0)
).withColumn(
    "is_weekend",
    F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0)
)

# Filter out any NaN or infinite values that might have been created
payment_df = payment_df.filter(
    (F.col("tip_ratio").isNotNull()) & 
    (~F.isnan(F.col("tip_ratio"))) &
    (F.col("tip_ratio") >= 0) & 
    (F.col("tip_ratio") < 10)  # Cap extreme tip ratios
)

print(f"Payment classification records: {payment_df.count():,}")
print("\nPayment type distribution:")
payment_df.groupBy("payment_type").count().orderBy("count", ascending=False).show()
print("\nSample payment data:")
payment_df.select("payment_type", "fare_amount", "tip_amount", "tip_ratio").show(10)
```

                                                                                    

    Payment classification records: 165,843,499
    
    Payment type distribution:


                                                                                    

    +------------+---------+
    |payment_type|    count|
    +------------+---------+
    |           1|131387419|
    |           2| 33058290|
    |           4|   795644|
    |           3|   602052|
    |           5|       94|
    +------------+---------+
    
    
    Sample payment data:
    +------------+-----------+----------+-------------------+
    |payment_type|fare_amount|tip_amount|          tip_ratio|
    +------------+-----------+----------+-------------------+
    |           2|        5.1|       0.0|                0.0|
    |           1|       19.8|       2.0|0.10101010101010101|
    |           1|       10.0|       0.0|                0.0|
    |           1|       31.0|       7.2|0.23225806451612904|
    |           1|        7.9|      2.38| 0.3012658227848101|
    |           1|       28.2|       8.3| 0.2943262411347518|
    |           1|       14.2|      3.64|0.25633802816901413|
    |           1|        6.5|       2.9| 0.4461538461538461|
    |           1|        5.8|      1.96|0.33793103448275863|
    |           1|        7.2|      2.45| 0.3402777777777778|
    +------------+-----------+----------+-------------------+
    only showing top 10 rows
    



```python
# Build Payment Type Classification Model
# Index categorical features
payment_taxi_indexer = StringIndexer(inputCol="taxi_type", outputCol="taxi_type_idx", handleInvalid="keep")
payment_label_indexer = StringIndexer(inputCol="payment_type", outputCol="label", handleInvalid="keep")

# Feature vector - using simpler features to avoid issues
payment_assembler = VectorAssembler(
    inputCols=["fare_amount", "trip_distance", "tip_amount", "passenger_count", 
               "taxi_type_idx", "tip_ratio", "is_weekend"],
    outputCol="raw_features",
    handleInvalid="skip"  # Skip rows with invalid values
)

# Scale features
payment_scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=False, withStd=True)

# Logistic Regression with simpler parameters
payment_lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=50,  # Reduced iterations
    regParam=0.1,  # Increased regularization
    elasticNetParam=0.0,
    family="auto"
)

# Pipeline
payment_pipeline = Pipeline(stages=[
    payment_taxi_indexer, payment_label_indexer, payment_assembler, payment_scaler, payment_lr
])

# Sample data for faster training - use smaller sample
payment_df_sample = payment_df.sample(fraction=0.05, seed=42)

# Split data
payment_train, payment_test = payment_df_sample.randomSplit([0.8, 0.2], seed=42)

print(f"Training records: {payment_train.count():,}")
print(f"Test records: {payment_test.count():,}")

# Train model
print("\nTraining Payment Type Classification model...")
try:
    payment_model = payment_pipeline.fit(payment_train)
    print("Model trained successfully!")
except Exception as e:
    print(f"Error training model: {e}")
    print("Attempting with even simpler configuration...")
    raise
```

                                                                                    

    Training records: 6,633,743


                                                                                    

    Test records: 1,659,572
    
    Training Payment Type Classification model...


                                                                                    

    Model trained successfully!



```python
# Evaluate Payment Classification Model
payment_predictions = payment_model.transform(payment_test)

payment_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

payment_accuracy = payment_evaluator.evaluate(payment_predictions, {payment_evaluator.metricName: "accuracy"})
payment_f1 = payment_evaluator.evaluate(payment_predictions, {payment_evaluator.metricName: "f1"})
payment_precision = payment_evaluator.evaluate(payment_predictions, {payment_evaluator.metricName: "weightedPrecision"})
payment_recall = payment_evaluator.evaluate(payment_predictions, {payment_evaluator.metricName: "weightedRecall"})

print("=" * 60)
print("PAYMENT TYPE CLASSIFICATION MODEL PERFORMANCE")
print("=" * 60)
print(f"Accuracy:  {payment_accuracy:.4f}")
print(f"F1 Score:  {payment_f1:.4f}")
print(f"Precision: {payment_precision:.4f}")
print(f"Recall:    {payment_recall:.4f}")
print("=" * 60)

# Show sample predictions
payment_predictions.select(
    "fare_amount", "tip_amount", "tip_ratio", "taxi_type", "payment_type", "prediction"
).show(20)
```

                                                                                    

    ============================================================
    PAYMENT TYPE CLASSIFICATION MODEL PERFORMANCE
    ============================================================
    Accuracy:  0.9543
    F1 Score:  0.9520
    Precision: 0.9543
    Recall:    0.9543
    ============================================================


    [Stage 1176:>                                                       (0 + 1) / 1]

    +-----------+----------+------------------+---------+------------+----------+
    |fare_amount|tip_amount|         tip_ratio|taxi_type|payment_type|prediction|
    +-----------+----------+------------------+---------+------------+----------+
    |       0.01|       0.0|               0.0|   yellow|           2|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           4|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           2|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           1|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           2|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           2|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           2|       1.0|
    |        3.0|       1.4|0.4666666666666666|   yellow|           1|       0.0|
    |        3.0|       0.0|               0.0|   yellow|           4|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           2|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           2|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           2|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           4|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           4|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           2|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           2|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           1|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           2|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           4|       1.0|
    |        3.0|       0.0|               0.0|   yellow|           4|       1.0|
    +-----------+----------+------------------+---------+------------+----------+
    only showing top 20 rows
    


                                                                                    


```python
# Save Payment Classification Model and Predictions
payment_model.write().overwrite().save(f"{ML_MODELS_PATH}/payment_classification")
payment_predictions.write.mode("overwrite").parquet(f"{ML_PREDICTIONS_PATH}/payment_classification")

print("Payment classification model and predictions saved!")
```

                                                                                    

    Payment classification model and predictions saved!


---
## Model 4: Trip Pattern Clustering (K-Means)

**Business Value**: Segment customers into behavioral groups for targeted marketing and service optimization.

**Features**: Average trip distance, trip duration, fare amount, tip ratio, time of day patterns


```python
# Prepare Clustering Dataset - Aggregate by behavioral patterns
clustering_df = silver_df_clean.select(
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "pickup_hour",
    F.dayofweek("pickup_datetime").alias("day_of_week"),
    "passenger_count",
    "taxi_type"
).filter(
    # Remove outliers
    (F.col("trip_distance") > 0) & (F.col("trip_distance") < 50) &
    (F.col("fare_amount") > 0) & (F.col("fare_amount") < 300)
)

# Add engineered features
clustering_df = clustering_df.withColumn(
    "tip_ratio",
    F.when(F.col("fare_amount") > 0, F.col("tip_amount") / F.col("fare_amount")).otherwise(0)
).withColumn(
    "fare_per_mile",
    F.when(F.col("trip_distance") > 0, F.col("fare_amount") / F.col("trip_distance")).otherwise(0)
).withColumn(
    "is_weekend",
    F.when(F.col("day_of_week").isin([1, 7]), 1.0).otherwise(0.0)
).withColumn(
    "is_night",
    F.when((F.col("pickup_hour") >= 22) | (F.col("pickup_hour") <= 5), 1.0).otherwise(0.0)
)

print(f"Clustering records: {clustering_df.count():,}")
clustering_df.describe().show()
```

                                                                                    

    Clustering records: 165,824,718


    [Stage 1199:===================================================>(108 + 1) / 109]

    +-------+------------------+------------------+-----------------+------------------+-----------------+------------------+------------------+---------+-------------------+--------------------+-------------------+-------------------+
    |summary|     trip_distance|       fare_amount|       tip_amount|      total_amount|      pickup_hour|       day_of_week|   passenger_count|taxi_type|          tip_ratio|       fare_per_mile|         is_weekend|           is_night|
    +-------+------------------+------------------+-----------------+------------------+-----------------+------------------+------------------+---------+-------------------+--------------------+-------------------+-------------------+
    |  count|         165824718|         165824718|        165824718|         165824718|        165824718|         165824718|         165824718|165824718|          165824718|           165824718|          165824718|          165824718|
    |   mean|3.2971511385894474| 16.12450474367848|2.940205416481041|23.686981927669834|14.22397812852002|4.1349320612141796|1.3931800927289983|     NULL|0.20664409453004043|   9.935845189382524|0.26387078191804914|0.15583472302362064|
    | stddev| 4.231080020503172|15.019155369733465|3.487461577013029|19.132271173828627|5.643337533971289|1.9448876480018598|0.9570827358539905|     NULL| 6.5981822549864075|  121.82043369716224|0.44073006879424653|0.36269858411232064|
    |    min|              0.01|              0.01|              0.0|              0.01|                0|                 1|               0.0|    green|                0.0|2.024291497975708...|                0.0|                0.0|
    |    max|             49.99|            299.99|          1400.16|           1737.18|               23|                 7|             112.0|   yellow|            27500.0|             29700.0|                1.0|                1.0|
    +-------+------------------+------------------+-----------------+------------------+-----------------+------------------+------------------+---------+-------------------+--------------------+-------------------+-------------------+
    


                                                                                    


```python
# Build K-Means Clustering Model
# Index categorical features
clustering_taxi_indexer = StringIndexer(inputCol="taxi_type", outputCol="taxi_type_idx")

# Feature vector
clustering_assembler = VectorAssembler(
    inputCols=["trip_distance", "fare_amount", 
               "tip_ratio", "fare_per_mile", "passenger_count", "pickup_hour",
               "taxi_type_idx", "is_weekend", "is_night"],
    outputCol="raw_features"
)

# Scale features (important for K-Means)
clustering_scaler = StandardScaler(inputCol="raw_features", outputCol="features")

# K-Means Clustering
kmeans = KMeans(
    featuresCol="features",
    k=5,  # 5 customer segments
    seed=42,
    maxIter=50
)

# Pipeline
clustering_pipeline = Pipeline(stages=[clustering_taxi_indexer, clustering_assembler, clustering_scaler, kmeans])

# Sample data for faster training
clustering_df_sample = clustering_df.sample(fraction=0.05, seed=42)

print(f"Training records: {clustering_df_sample.count():,}")

# Train model
print("\nTraining K-Means Clustering model...")
clustering_model = clustering_pipeline.fit(clustering_df_sample)
print("Model trained successfully!")
```

                                                                                    

    Training records: 8,292,335
    
    Training K-Means Clustering model...


    [Stage 1278:================================================>   (102 + 7) / 109]

    Model trained successfully!


                                                                                    


```python
# Evaluate Clustering Model
clustering_predictions = clustering_model.transform(clustering_df_sample)

# Compute cluster statistics
cluster_stats = clustering_predictions.groupBy("prediction").agg(
    F.count("*").alias("cluster_size"),
    F.avg("trip_distance").alias("avg_trip_distance"),
    F.avg("fare_amount").alias("avg_fare"),
    F.avg("tip_ratio").alias("avg_tip_ratio"),
    F.avg("fare_per_mile").alias("avg_fare_per_mile"),
    F.avg("is_weekend").alias("pct_weekend"),
    F.avg("is_night").alias("pct_night")
).orderBy("prediction")

print("=" * 100)
print("K-MEANS CLUSTERING - CUSTOMER SEGMENTS")
print("=" * 100)
cluster_stats.show(truncate=False)

# Extract cluster centers
kmeans_model = clustering_model.stages[-1]
print("\nCluster Centers:")
for i, center in enumerate(kmeans_model.clusterCenters()):
    print(f"Cluster {i}: {center}")
```

    ====================================================================================================
    K-MEANS CLUSTERING - CUSTOMER SEGMENTS
    ====================================================================================================


    [Stage 1281:===================================================>(108 + 1) / 109]

    +----------+------------+------------------+------------------+-------------------+-------------------+-------------------+-------------------+
    |prediction|cluster_size|avg_trip_distance |avg_fare          |avg_tip_ratio      |avg_fare_per_mile  |pct_weekend        |pct_night          |
    +----------+------------+------------------+------------------+-------------------+-------------------+-------------------+-------------------+
    |0         |1147131     |2.8549952533755967|13.642464382882165|0.2097316036774764 |7.237146308805454  |0.40348922660097236|1.0                |
    |1         |1           |0.1               |0.01              |3300.0             |0.09999999999999999|0.0                |0.0                |
    |2         |600186      |2.4015926895995565|12.992440793354062|0.1974450662017761 |7.998235299109651  |0.30260119362997473|0.09446238332783503|
    |3         |5816799     |2.0451740519141213|12.025862353160205|0.20744492091669167|7.692672627124558  |0.23196710080578684|0.0                |
    |4         |728218      |14.715268024135625|55.32631387853638 |0.16132906669911562|33.34981532434796  |0.2658901592654947 |0.12565605354440565|
    +----------+------------+------------------+------------------+-------------------+-------------------+-------------------+-------------------+
    
    
    Cluster Centers:
    Cluster 0: [0.67539218 0.90847042 0.17950953 0.06042539 1.32218865 2.26242497
     0.11790384 0.9155948  2.75442132]
    Cluster 1: [2.36564998e-02 6.65913653e-04 2.82447140e+03 8.34933959e-04
     0.00000000e+00 1.94810298e+00 0.00000000e+00 0.00000000e+00
     0.00000000e+00]
    Cluster 2: [0.5681357  0.86518633 0.16899315 0.0667799  4.37225796 2.60876836
     0.11851151 0.68666657 0.26019293]
    Cluster 3: [0.48381686 0.80081885 0.1775522  0.06422873 1.18232591 2.56067044
     0.17013742 0.52637863 0.        ]
    Cluster 4: [3.48112074 3.68425886 0.1380814  0.27844962 1.45751667 2.51018384
     0.09006651 0.60335532 0.34611066]


                                                                                    


```python
# Interpret Clusters (Business Insights)
print("\n" + "=" * 100)
print("CLUSTER INTERPRETATION (Based on Statistics)")
print("=" * 100)

# Let's add business names to clusters based on their characteristics
cluster_profiles = cluster_stats.collect()

for cluster in cluster_profiles:
    cluster_id = cluster['prediction']
    size = cluster['cluster_size']
    avg_dist = cluster['avg_trip_distance']
    avg_fare = cluster['avg_fare']
    tip_ratio = cluster['avg_tip_ratio']
    fare_per_mile = cluster['avg_fare_per_mile']
    pct_night = cluster['pct_night'] * 100
    pct_weekend = cluster['pct_weekend'] * 100
    
    print(f"\nCluster {cluster_id} ({size:,} trips):")
    print(f"  - Avg Distance: {avg_dist:.2f} miles")
    print(f"  - Avg Fare: ${avg_fare:.2f}")
    print(f"  - Fare per Mile: ${fare_per_mile:.2f}")
    print(f"  - Tip Ratio: {tip_ratio:.2%}")
    print(f"  - Night Trips: {pct_night:.1f}%")
    print(f"  - Weekend Trips: {pct_weekend:.1f}%")
    
    # Simple heuristic interpretation
    if avg_dist > 10:
        segment = "Long-Distance Travelers"
    elif pct_night > 40:
        segment = "Night Riders"
    elif tip_ratio > 0.20:
        segment = "Premium Customers"
    elif avg_fare < 15:
        segment = "Budget-Conscious Riders"
    else:
        segment = "Standard Commuters"
    
    print(f"  → Segment: {segment}")
```

    
    ====================================================================================================
    CLUSTER INTERPRETATION (Based on Statistics)
    ====================================================================================================


    [Stage 1284:==================================================> (105 + 4) / 109]

    
    Cluster 0 (1,147,131 trips):
      - Avg Distance: 2.85 miles
      - Avg Fare: $13.64
      - Fare per Mile: $7.24
      - Tip Ratio: 20.97%
      - Night Trips: 100.0%
      - Weekend Trips: 40.3%
      → Segment: Night Riders
    
    Cluster 1 (1 trips):
      - Avg Distance: 0.10 miles
      - Avg Fare: $0.01
      - Fare per Mile: $0.10
      - Tip Ratio: 330000.00%
      - Night Trips: 0.0%
      - Weekend Trips: 0.0%
      → Segment: Premium Customers
    
    Cluster 2 (600,186 trips):
      - Avg Distance: 2.40 miles
      - Avg Fare: $12.99
      - Fare per Mile: $8.00
      - Tip Ratio: 19.74%
      - Night Trips: 9.4%
      - Weekend Trips: 30.3%
      → Segment: Budget-Conscious Riders
    
    Cluster 3 (5,816,799 trips):
      - Avg Distance: 2.05 miles
      - Avg Fare: $12.03
      - Fare per Mile: $7.69
      - Tip Ratio: 20.74%
      - Night Trips: 0.0%
      - Weekend Trips: 23.2%
      → Segment: Premium Customers
    
    Cluster 4 (728,218 trips):
      - Avg Distance: 14.72 miles
      - Avg Fare: $55.33
      - Fare per Mile: $33.35
      - Tip Ratio: 16.13%
      - Night Trips: 12.6%
      - Weekend Trips: 26.6%
      → Segment: Long-Distance Travelers


                                                                                    


```python
# Save Clustering Model and Predictions
clustering_model.write().overwrite().save(f"{ML_MODELS_PATH}/trip_clustering")
clustering_predictions.write.mode("overwrite").parquet(f"{ML_PREDICTIONS_PATH}/trip_clustering")

print("Clustering model and predictions saved!")
```

    [Stage 1306:===================================================>(107 + 2) / 109]

    Clustering model and predictions saved!


                                                                                    

---
## Summary: All Models Trained and Evaluated

### Model Performance Summary:

1. **Demand Forecasting** - Random Forest regressor for hourly trip volume prediction
2. **Fare Prediction** - Random Forest regressor for trip fare estimation based on distance and time
3. **Tip Amount Prediction** - Gradient Boosted Trees for tip amount estimation
4. **Payment Type Classification** - Logistic Regression for payment method prediction
5. **Trip Pattern Clustering** - K-Means for customer segmentation (5 behavioral groups)

### Saved Artifacts:
- **Models**: `/home/ubuntu/project/gold_layer_data/ml_models/`
- **Predictions**: `/home/ubuntu/project/gold_layer_data/ml_predictions/`

### Business Applications:
- **Operations**: Optimize driver allocation using demand forecasting
- **Pricing**: Detect surge pricing opportunities with fare prediction
- **Customer Experience**: Understand tipping patterns and customer behavior
- **Fraud Detection**: Identify anomalies with payment classification
- **Marketing**: Target specific segments identified by clustering

### Note:
Models are built using available features from the silver layer. Trip duration was not available in the scrambled dataset, so models focus on distance-based and temporal features.
