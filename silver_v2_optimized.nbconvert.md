# Silver Layer - Medallion Architecture (Optimized)
**Purpose:** Transform Bronze layer data into structured, clean format for analytics.

Silver layer follows these principles:
- Parse raw data into structured columns
- Filter to valid records only
- Type conversions and standardization
- Add derived columns for easier querying
- Preserve nulls for data integrity
- Add data quality flags

**Optimization:** Process data year-by-year to avoid memory issues

## 1. Imports and Spark Session Initialization


```python
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, split, regexp_extract, regexp_replace,
    to_timestamp, year, month, dayofmonth, hour, dayofweek,
    when, coalesce, trim, array, array_compact, size, explode, sum as _sum
)
from pyspark.sql.types import (
    DoubleType, IntegerType, TimestampType, StringType, BooleanType
)
import os

# Initialize Spark Session with optimized memory settings
spark = SparkSession.builder \
    .appName("TLC Silver Layer Transformation") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.memory.storageFraction", "0.3") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark Session initialized with optimized memory settings.")
print(f"Spark Version: {spark.version}")
```

    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).


    25/11/17 19:38:08 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable


    Spark Session initialized with optimized memory settings.
    Spark Version: 3.5.0


## 2. Define Paths


```python
# Source: Bronze layer
BRONZE_DIR = "/home/ubuntu/project/bronze_v2"

# Destination: Silver layer
SILVER_DIR = "/home/ubuntu/project/silver_layer_data_v2"

print(f"Bronze Directory: {BRONZE_DIR}")
print(f"Silver Directory: {SILVER_DIR}")

os.makedirs(SILVER_DIR, exist_ok=True)
```

    Bronze Directory: /home/ubuntu/project/bronze_v2
    Silver Directory: /home/ubuntu/project/silver_layer_data_v2


## 3. Discover Available Years


```python
print("\nDiscovering available years in Bronze layer...")

# Get list of available years
df_bronze_sample = spark.read.parquet(BRONZE_DIR)
available_years = [row.year for row in df_bronze_sample.select("year").distinct().collect()]
available_years = sorted([int(y) for y in available_years])

print(f"Found years: {available_years}")
print(f"\nWill process {len(available_years)} years sequentially to manage memory efficiently.\n")
```

    
    Discovering available years in Bronze layer...


    [Stage 0:>                                                          (0 + 1) / 1]

                                                                                    

    [Stage 1:>                                                        (0 + 8) / 108]

    [Stage 1:====>                                                    (8 + 8) / 108]

    [Stage 1:========>                                              (17 + 11) / 108]

    [Stage 1:===============>                                        (29 + 8) / 108][Stage 1:===========================>                            (53 + 8) / 108]

    [Stage 1:========================================>               (79 + 8) / 108][Stage 1:====================================================>  (103 + 5) / 108]

    Found years: [2020, 2021, 2022, 2023, 2024]
    
    Will process 5 years sequentially to manage memory efficiently.
    


                                                                                    

## 4. Process Each Year - Complete Bronze→Silver Transformation

For each year:
1. Read Bronze data
2. Filter to valid records
3. Parse raw format into structured columns
4. Apply type conversions with error tracking
5. Add derived columns
6. Write to Silver layer


```python
# Helper functions for all years
def safe_cast(col_name, data_type):
    """Convert 'NULL' string to actual null, then cast to type"""
    return when(col(col_name) == "NULL", lit(None)).otherwise(col(col_name).cast(data_type))

def is_parse_error(raw_col, typed_col):
    """Detect if parsing failed (non-NULL input became NULL output)"""
    return when(
        (col(raw_col) != "NULL") & 
        (col(raw_col) != "") & 
        col(typed_col).isNull(),
        True
    ).otherwise(False)

# Process each year
for process_year in available_years:
    print(f"\n{'='*70}")
    print(f"PROCESSING YEAR {process_year}")
    print(f"{'='*70}\n")
    
    # ===== READ BRONZE DATA FOR THIS YEAR =====
    df_bronze_year = spark.read.parquet(BRONZE_DIR) \
        .filter(col("year") == str(process_year))
    
    total_year = df_bronze_year.count()
    print(f"Total records for {process_year}: {total_year:,}")
    
    # Filter to valid records only
    df_valid = df_bronze_year.filter(col("is_valid") == True)
    valid_count = df_valid.count()
    print(f"Valid records: {valid_count:,}")
    
    if valid_count == 0:
        print(f"⚠ Skipping {process_year} - no valid records")
        continue
    
    # ===== STEP 1: PARSE RAW FORMAT =====
    print("  Parsing raw data...")
    split_col = split(col("value"), "\\|")
    
    df_parsed = df_valid.withColumn("pickup_datetime_raw", split_col.getItem(0)) \
        .withColumn("record_id", split_col.getItem(1)) \
        .withColumn("taxi_type", split_col.getItem(2)) \
        .withColumn("payment_type_raw", split_col.getItem(3)) \
        .withColumn("payload_raw", split_col.getItem(4))
    
    # Extract fields from payload
    df_parsed = df_parsed \
        .withColumn("passenger_count_raw", regexp_extract(col("payload_raw"), r"passengers:([^,}]+)", 1)) \
        .withColumn("trip_distance_raw", regexp_extract(col("payload_raw"), r"dist:([^,}]+)", 1)) \
        .withColumn("fare_amount_raw", regexp_extract(col("payload_raw"), r"fare:([^,}]+)", 1)) \
        .withColumn("tip_amount_raw", regexp_extract(col("payload_raw"), r"tip:([^,}]+)", 1)) \
        .withColumn("tolls_amount_raw", regexp_extract(col("payload_raw"), r"tolls:([^,}]+)", 1)) \
        .withColumn("total_amount_raw", regexp_extract(col("payload_raw"), r"total:([^,}]+)", 1))
    
    # ===== STEP 2: TYPE CONVERSIONS WITH ERROR TRACKING =====
    print("  Applying type conversions...")
    
    df_typed = df_parsed \
        .withColumn("pickup_datetime", to_timestamp(col("pickup_datetime_raw"), "yyyy-MM-dd'T'HH:mm:ss")) \
        .withColumn("pickup_datetime_error", is_parse_error("pickup_datetime_raw", "pickup_datetime")) \
        .withColumn("payment_type", safe_cast("payment_type_raw", IntegerType())) \
        .withColumn("payment_type_error", is_parse_error("payment_type_raw", "payment_type")) \
        .withColumn("passenger_count", safe_cast("passenger_count_raw", DoubleType())) \
        .withColumn("passenger_count_error", is_parse_error("passenger_count_raw", "passenger_count")) \
        .withColumn("trip_distance", safe_cast("trip_distance_raw", DoubleType())) \
        .withColumn("trip_distance_error", is_parse_error("trip_distance_raw", "trip_distance")) \
        .withColumn("fare_amount", safe_cast("fare_amount_raw", DoubleType())) \
        .withColumn("fare_amount_error", is_parse_error("fare_amount_raw", "fare_amount")) \
        .withColumn("tip_amount", safe_cast("tip_amount_raw", DoubleType())) \
        .withColumn("tip_amount_error", is_parse_error("tip_amount_raw", "tip_amount")) \
        .withColumn("tolls_amount", safe_cast("tolls_amount_raw", DoubleType())) \
        .withColumn("tolls_amount_error", is_parse_error("tolls_amount_raw", "tolls_amount")) \
        .withColumn("total_amount", safe_cast("total_amount_raw", DoubleType())) \
        .withColumn("total_amount_error", is_parse_error("total_amount_raw", "total_amount"))
    
    # ===== STEP 3: ADD DERIVED COLUMNS =====
    print("  Adding derived columns...")
    
    # Aggregate parse errors
    df_enriched = df_typed.withColumn(
        "parse_errors",
        array_compact(array(
            when(col("pickup_datetime_error"), lit("pickup_datetime")),
            when(col("payment_type_error"), lit("payment_type")),
            when(col("passenger_count_error"), lit("passenger_count")),
            when(col("trip_distance_error"), lit("trip_distance")),
            when(col("fare_amount_error"), lit("fare_amount")),
            when(col("tip_amount_error"), lit("tip_amount")),
            when(col("tolls_amount_error"), lit("tolls_amount")),
            when(col("total_amount_error"), lit("total_amount"))
        ))
    )
    
    # Time-based derived columns
    df_enriched = df_enriched \
        .withColumn("pickup_year", year(col("pickup_datetime"))) \
        .withColumn("pickup_month", month(col("pickup_datetime"))) \
        .withColumn("pickup_day", dayofmonth(col("pickup_datetime"))) \
        .withColumn("pickup_hour", hour(col("pickup_datetime"))) \
        .withColumn("pickup_dayofweek", dayofweek(col("pickup_datetime")))
    
    # Data quality flags
    df_enriched = df_enriched \
        .withColumn("has_missing_values", when(
            col("pickup_datetime").isNull() | 
            col("passenger_count").isNull() | 
            col("trip_distance").isNull() | 
            col("fare_amount").isNull() | 
            col("total_amount").isNull(),
            True
        ).otherwise(False)) \
        .withColumn("has_parse_errors", when(size(col("parse_errors")) > 0, True).otherwise(False))
    
    # ===== STEP 4: SELECT FINAL SCHEMA =====
    df_silver_year = df_enriched.select(
        col("record_id"), col("taxi_type"),
        col("pickup_datetime"), col("pickup_year"), col("pickup_month"), 
        col("pickup_day"), col("pickup_hour"), col("pickup_dayofweek"),
        col("passenger_count"), col("trip_distance"), col("payment_type"),
        col("fare_amount"), col("tip_amount"), col("tolls_amount"), col("total_amount"),
        col("has_missing_values"), col("has_parse_errors"), col("parse_errors"),
        col("ingestion_timestamp"), col("source_file"), col("processing_batch_id")
    )
    
    # ===== STEP 5: WRITE TO SILVER LAYER =====
    print(f"  Writing {process_year} to Silver layer...")
    
    # Use 'overwrite' for first year, 'append' for subsequent years
    write_mode = "overwrite" if process_year == available_years[0] else "append"
    
    df_silver_year.write \
        .mode(write_mode) \
        .partitionBy("pickup_year", "taxi_type") \
        .parquet(SILVER_DIR)
    
    print(f"  ✓ Year {process_year} complete!\n")
    
    # Clear cache to free memory
    spark.catalog.clearCache()

print(f"\n{'='*70}")
print("ALL YEARS PROCESSED SUCCESSFULLY!")
print(f"{'='*70}\n")
print(f"Silver layer saved to: {SILVER_DIR}")
```

    
    ======================================================================
    PROCESSING YEAR 2020
    ======================================================================
    


    Total records for 2020: 26,383,390


    [Stage 8:>                                                         (0 + 8) / 17]

    [Stage 8:===========================>                              (8 + 9) / 17]                                                                                

    Valid records: 26,383,390
      Parsing raw data...
      Applying type conversions...


      Adding derived columns...
      Writing 2020 to Silver layer...


    [Stage 11:>                                                        (0 + 8) / 17]

    [Stage 11:===>                                                     (1 + 8) / 17]

    [Stage 11:==========>                                              (3 + 8) / 17]

    [Stage 11:================>                                        (5 + 8) / 17][Stage 11:=======================>                                 (7 + 8) / 17]

    [Stage 11:==========================>                              (8 + 8) / 17]

    [Stage 11:==============================>                          (9 + 8) / 17]

    [Stage 11:================================>                       (10 + 7) / 17]

    [Stage 11:====================================>                   (11 + 6) / 17]

    [Stage 11:=======================================>                (12 + 5) / 17][Stage 11:==========================================>             (13 + 4) / 17]

    [Stage 11:==============================================>         (14 + 3) / 17][Stage 11:=================================================>      (15 + 2) / 17]

    [Stage 11:====================================================>   (16 + 1) / 17]

                                                                                    

      ✓ Year 2020 complete!
    
    
    ======================================================================
    PROCESSING YEAR 2021
    ======================================================================
    


    Total records for 2021: 31,972,712
    Valid records: 31,972,712
      Parsing raw data...


      Applying type conversions...
      Adding derived columns...


      Writing 2021 to Silver layer...


    [Stage 19:>                                                        (0 + 8) / 20]

    [Stage 19:=====>                                                   (2 + 8) / 20]

    [Stage 19:========>                                                (3 + 8) / 20]

    [Stage 19:===========>                                             (4 + 8) / 20]

    [Stage 19:==============>                                          (5 + 8) / 20]

    [Stage 19:=================>                                       (6 + 8) / 20]

    [Stage 19:======================>                                  (8 + 8) / 20]

    [Stage 19:=========================>                               (9 + 8) / 20]

    [Stage 19:============================>                           (10 + 8) / 20]

    [Stage 19:==============================>                         (11 + 8) / 20]

    [Stage 19:=================================>                      (12 + 8) / 20]

    [Stage 19:====================================>                   (13 + 7) / 20][Stage 19:=======================================>                (14 + 6) / 20]

    [Stage 19:==========================================>             (15 + 5) / 20]

    [Stage 19:============================================>           (16 + 4) / 20]

    [Stage 19:===============================================>        (17 + 3) / 20]

    [Stage 19:==================================================>     (18 + 2) / 20]

    [Stage 19:=====================================================>  (19 + 1) / 20]                                                                                

      ✓ Year 2021 complete!
    
    
    ======================================================================
    PROCESSING YEAR 2022
    ======================================================================
    


    Total records for 2022: 40,496,016
    Valid records: 40,496,016
      Parsing raw data...


      Applying type conversions...
      Adding derived columns...


      Writing 2022 to Silver layer...


    [Stage 27:>                                                        (0 + 8) / 24]

    [Stage 27:==>                                                      (1 + 8) / 24][Stage 27:====>                                                    (2 + 8) / 24]

    [Stage 27:=======>                                                 (3 + 8) / 24]

    [Stage 27:=========>                                               (4 + 8) / 24]

    [Stage 27:===========>                                             (5 + 8) / 24][Stage 27:==============>                                          (6 + 8) / 24]

    [Stage 27:================>                                        (7 + 8) / 24]

    [Stage 27:===================>                                     (8 + 8) / 24]

    [Stage 27:=====================>                                   (9 + 8) / 24]

    [Stage 27:=======================>                                (10 + 8) / 24]

    [Stage 27:=========================>                              (11 + 8) / 24]

    [Stage 27:============================>                           (12 + 8) / 24]

    [Stage 27:==============================>                         (13 + 8) / 24]

    [Stage 27:================================>                       (14 + 8) / 24]

    [Stage 27:===================================>                    (15 + 8) / 24]

    [Stage 27:=====================================>                  (16 + 8) / 24]

    [Stage 27:=======================================>                (17 + 7) / 24][Stage 27:============================================>           (19 + 5) / 24]

    [Stage 27:==============================================>         (20 + 4) / 24][Stage 27:=================================================>      (21 + 3) / 24]

    [Stage 27:===================================================>    (22 + 2) / 24]

    [Stage 27:=====================================================>  (23 + 1) / 24]

                                                                                    

      ✓ Year 2022 complete!
    
    
    ======================================================================
    PROCESSING YEAR 2023
    ======================================================================
    


    Total records for 2023: 39,097,193
    Valid records: 39,097,193
      Parsing raw data...


      Applying type conversions...
      Adding derived columns...


      Writing 2023 to Silver layer...


    [Stage 35:>                                                        (0 + 8) / 24]

    [Stage 35:==>                                                      (1 + 8) / 24][Stage 35:====>                                                    (2 + 9) / 24]

    [Stage 35:=======>                                                 (3 + 8) / 24]

    [Stage 35:=========>                                               (4 + 8) / 24][Stage 35:===========>                                             (5 + 8) / 24]

    [Stage 35:==============>                                          (6 + 8) / 24]

    [Stage 35:================>                                        (7 + 8) / 24]

    [Stage 35:===================>                                     (8 + 8) / 24]

    [Stage 35:=====================>                                   (9 + 8) / 24]

    [Stage 35:==============================>                         (13 + 8) / 24]

    [Stage 35:================================>                       (14 + 8) / 24]

    [Stage 35:===================================>                    (15 + 8) / 24]

    [Stage 35:=====================================>                  (16 + 8) / 24]

    [Stage 35:=======================================>                (17 + 7) / 24]

    [Stage 35:==========================================>             (18 + 6) / 24]

    [Stage 35:============================================>           (19 + 5) / 24][Stage 35:==============================================>         (20 + 4) / 24]

    [Stage 35:=================================================>      (21 + 3) / 24]

    [Stage 35:===================================================>    (22 + 2) / 24]

    [Stage 35:=====================================================>  (23 + 1) / 24]

                                                                                    

      ✓ Year 2023 complete!
    
    
    ======================================================================
    PROCESSING YEAR 2024
    ======================================================================
    


    Total records for 2024: 41,829,868
    Valid records: 41,829,868
      Parsing raw data...
      Applying type conversions...


      Adding derived columns...
      Writing 2024 to Silver layer...


    [Stage 43:>                                                        (0 + 8) / 25]

    [Stage 43:====>                                                    (2 + 8) / 25]

    [Stage 43:======>                                                  (3 + 8) / 25]

    [Stage 43:=========>                                               (4 + 8) / 25]

    [Stage 43:===========>                                             (5 + 8) / 25][Stage 43:=============>                                           (6 + 8) / 25]

    [Stage 43:===============>                                         (7 + 8) / 25]

    [Stage 43:==================>                                      (8 + 8) / 25]

    [Stage 43:====================>                                    (9 + 8) / 25]

    [Stage 43:======================>                                 (10 + 8) / 25][Stage 43:========================>                               (11 + 8) / 25]

    [Stage 43:==========================>                             (12 + 8) / 25][Stage 43:=============================>                          (13 + 8) / 25]

    [Stage 43:===============================>                        (14 + 8) / 25][Stage 43:=================================>                      (15 + 8) / 25]

    [Stage 43:===================================>                    (16 + 8) / 25]

    [Stage 43:======================================>                 (17 + 8) / 25][Stage 43:==========================================>             (19 + 6) / 25]

    [Stage 43:============================================>           (20 + 5) / 25]

    [Stage 43:===============================================>        (21 + 4) / 25]

    [Stage 43:=================================================>      (22 + 3) / 25]

    [Stage 43:=====================================================>  (24 + 1) / 25]

      ✓ Year 2024 complete!
    
    
    ======================================================================
    ALL YEARS PROCESSED SUCCESSFULLY!
    ======================================================================
    
    Silver layer saved to: /home/ubuntu/project/silver_layer_data_v2


                                                                                    

## 5. Data Quality Report


```python
print("\n" + "="*70)
print("SILVER LAYER DATA QUALITY REPORT")
print("="*70 + "\n")

# Read all silver data
df_silver = spark.read.parquet(SILVER_DIR)

total = df_silver.count()
print(f"Total Records: {total:,}\n")

# Parsing errors analysis
parse_error_count = df_silver.filter(col("has_parse_errors") == True).count()
print(f"Records with Parse Errors: {parse_error_count:,} ({100*parse_error_count/total:.2f}%)")
print(f"Clean Parse Records: {total - parse_error_count:,} ({100*(total-parse_error_count)/total:.2f}%)\n")

# Missing values analysis
missing_count = df_silver.filter(col("has_missing_values") == True).count()
print(f"Records with Missing Values: {missing_count:,} ({100*missing_count/total:.2f}%)")
print(f"Complete Records: {total - missing_count:,} ({100*(total-missing_count)/total:.2f}%)\n")

# Parse error breakdown by field
if parse_error_count > 0:
    print("--- Parse Error Breakdown by Field ---")
    df_silver.filter(col("has_parse_errors") == True) \
        .select(explode(col("parse_errors")).alias("error_field")) \
        .groupBy("error_field") \
        .count() \
        .orderBy(col("count").desc()) \
        .show(truncate=False)

# Null counts by field
print("\n--- Null Counts by Field ---")
null_counts = df_silver.select(
    _sum(when(col("pickup_datetime").isNull(), 1).otherwise(0)).alias("pickup_datetime_nulls"),
    _sum(when(col("passenger_count").isNull(), 1).otherwise(0)).alias("passenger_count_nulls"),
    _sum(when(col("trip_distance").isNull(), 1).otherwise(0)).alias("trip_distance_nulls"),
    _sum(when(col("fare_amount").isNull(), 1).otherwise(0)).alias("fare_amount_nulls"),
    _sum(when(col("tip_amount").isNull(), 1).otherwise(0)).alias("tip_amount_nulls"),
    _sum(when(col("tolls_amount").isNull(), 1).otherwise(0)).alias("tolls_amount_nulls"),
    _sum(when(col("total_amount").isNull(), 1).otherwise(0)).alias("total_amount_nulls"),
).collect()[0]

for field_name in null_counts.asDict():
    count = null_counts[field_name]
    print(f"  {field_name}: {count:,} ({100*count/total:.2f}%)")

# Distribution by taxi type
print("\n--- Distribution by Taxi Type ---")
df_silver.groupBy("taxi_type").count().orderBy("taxi_type").show()

# Distribution by year
print("--- Distribution by Year ---")
df_silver.groupBy("pickup_year").count().orderBy("pickup_year").show()
```

    
    ======================================================================
    SILVER LAYER DATA QUALITY REPORT
    ======================================================================
    


    Total Records: 179,779,179
    
    Records with Parse Errors: 0 (0.00%)
    Clean Parse Records: 179,779,179 (100.00%)
    


    Records with Missing Values: 10,168,484 (5.66%)
    Complete Records: 169,610,695 (94.34%)
    
    
    --- Null Counts by Field ---


    [Stage 54:>                                                       (0 + 8) / 109]

    [Stage 54:=>                                                      (2 + 8) / 109]

    [Stage 54:====>                                                   (8 + 8) / 109][Stage 54:========>                                              (16 + 8) / 109]

    [Stage 54:==========>                                            (21 + 9) / 109][Stage 54:==============>                                        (29 + 8) / 109]

    [Stage 54:=================>                                     (35 + 8) / 109][Stage 54:====================>                                  (41 + 8) / 109]

    [Stage 54:========================>                              (49 + 8) / 109][Stage 54:===========================>                           (55 + 8) / 109]

    [Stage 54:===============================>                       (62 + 8) / 109][Stage 54:===================================>                   (70 + 8) / 109]

    [Stage 54:======================================>                (76 + 8) / 109][Stage 54:==========================================>            (85 + 8) / 109]

    [Stage 54:==============================================>        (92 + 8) / 109][Stage 54:=================================================>    (100 + 8) / 109]

    [Stage 54:====================================================> (105 + 4) / 109]                                                                                

      pickup_datetime_nulls: 0 (0.00%)
      passenger_count_nulls: 10,168,484 (5.66%)
      trip_distance_nulls: 0 (0.00%)
      fare_amount_nulls: 0 (0.00%)
      tip_amount_nulls: 0 (0.00%)
      tolls_amount_nulls: 0 (0.00%)
      total_amount_nulls: 0 (0.00%)
    
    --- Distribution by Taxi Type ---


    [Stage 57:====>                                                   (8 + 8) / 109][Stage 57:============>                                          (24 + 8) / 109]

    [Stage 57:===========================>                           (55 + 8) / 109][Stage 57:==========================================>            (85 + 8) / 109]

    [Stage 57:=====================================================>(107 + 2) / 109]                                                                                

    +---------+---------+
    |taxi_type|    count|
    +---------+---------+
    |    green|  5090500|
    |   yellow|174688679|
    +---------+---------+
    
    --- Distribution by Year ---


    [Stage 60:===============================>                       (63 + 8) / 109][Stage 60:===================================================>  (104 + 5) / 109]

    +-----------+--------+
    |pickup_year|   count|
    +-----------+--------+
    |       2020|26383390|
    |       2021|31972712|
    |       2022|40496016|
    |       2023|39097193|
    |       2024|41829868|
    +-----------+--------+
    


                                                                                    

## 6. Write Rejected Records (Parse Errors)


```python
print("\n--- Writing Rejected Records ---\n")

df_rejected = df_silver.filter(col("has_parse_errors") == True)
rejected_count = df_rejected.count()

if rejected_count > 0:
    REJECTED_DIR = f"{SILVER_DIR}_rejected"
    
    print(f"Found {rejected_count:,} records with parse errors")
    print(f"Writing to: {REJECTED_DIR}\n")
    
    df_rejected.write \
        .mode("overwrite") \
        .partitionBy("pickup_year", "taxi_type") \
        .parquet(REJECTED_DIR)
    
    print("✓ Rejected records saved for investigation")
    print("\nSample rejected records:")
    df_rejected.select(
        "record_id", "taxi_type", "pickup_datetime", 
        "parse_errors", "fare_amount", "total_amount"
    ).show(5, truncate=False)
else:
    print("✓ No rejected records - all data parsed successfully!")
```

    
    --- Writing Rejected Records ---
    
    ✓ No rejected records - all data parsed successfully!


## 7. Sample Analytics Queries


```python
print("\n" + "="*70)
print("SAMPLE ANALYTICS ON SILVER LAYER")
print("="*70 + "\n")

# Query 1: Average fare by taxi type
print("--- Average Fare Amount by Taxi Type ---")
df_silver.groupBy("taxi_type") \
    .agg({"fare_amount": "avg", "tip_amount": "avg", "total_amount": "avg"}) \
    .show()

# Query 2: Trip patterns by hour of day
print("--- Trip Count by Hour of Day (Top 10) ---")
df_silver.groupBy("pickup_hour") \
    .count() \
    .orderBy(col("count").desc()) \
    .show(10)

# Query 3: Busiest day of week
print("--- Trip Count by Day of Week (1=Sunday, 7=Saturday) ---")
df_silver.groupBy("pickup_dayofweek") \
    .count() \
    .orderBy("pickup_dayofweek") \
    .show()
```

    
    ======================================================================
    SAMPLE ANALYTICS ON SILVER LAYER
    ======================================================================
    
    --- Average Fare Amount by Taxi Type ---


    [Stage 66:====>                                                   (8 + 8) / 109][Stage 66:=========>                                             (18 + 8) / 109]

    [Stage 66:===============>                                       (30 + 8) / 109][Stage 66:=====================>                                 (43 + 8) / 109]

    [Stage 66:============================>                          (56 + 8) / 109][Stage 66:===================================>                   (71 + 8) / 109]

    [Stage 66:===========================================>           (87 + 8) / 109][Stage 66:=================================================>    (100 + 9) / 109]

                                                                                    

    +---------+------------------+------------------+------------------+
    |taxi_type|  avg(fare_amount)| avg(total_amount)|   avg(tip_amount)|
    +---------+------------------+------------------+------------------+
    |   yellow|15.353990398941235| 23.80468901647651|3.9023874615839533|
    |    green|17.753426235125815|21.918969016939478|1.6965199803555913|
    +---------+------------------+------------------+------------------+
    
    --- Trip Count by Hour of Day (Top 10) ---


    [Stage 69:==========>                                            (21 + 9) / 109][Stage 69:===========================>                           (54 + 9) / 109]

    [Stage 69:================================================>      (96 + 8) / 109]                                                                                

    +-----------+--------+
    |pickup_hour|   count|
    +-----------+--------+
    |         18|12696120|
    |         17|12233902|
    |         15|11446544|
    |         16|11335645|
    |         19|11229083|
    |         14|11161214|
    |         13|10412095|
    |         12|10121820|
    |         20| 9700640|
    |         21| 9389265|
    +-----------+--------+
    only showing top 10 rows
    
    --- Trip Count by Day of Week (1=Sunday, 7=Saturday) ---


    +----------------+--------+
    |pickup_dayofweek|   count|
    +----------------+--------+
    |               1|21706407|
    |               2|22934742|
    |               3|26052254|
    |               4|27356513|
    |               5|28129945|
    |               6|27497896|
    |               7|26101422|
    +----------------+--------+
    


## 8. Cleanup


```python
spark.stop()
print("\n✓ Silver layer transformation complete. Spark session stopped.")
```

    
    ✓ Silver layer transformation complete. Spark session stopped.

