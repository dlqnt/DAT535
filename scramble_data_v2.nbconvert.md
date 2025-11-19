# TLC Data Scrambler - Full & Fast (Enriched)
**Purpose:** Transform ALL structured TLC taxi data into messy, unstructured single-line format with enriched fields for realistic data processing challenges.

## 1. Imports and Spark Session Initialization


```python
import findspark
findspark.init()
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit
from pyspark.sql.types import TimestampType, IntegerType, DoubleType, StringType
import os
import glob
from functools import reduce

spark = SparkSession.builder \
    .appName("TLC Full & Fast Scrambler (Enriched)") \
    .config("spark.driver.memory", "10g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark Session initialized.")
```

    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    25/11/17 16:08:14 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable


    Spark Session initialized.


## 2. Define Paths


```python
BRONZE_LAYER_PATH = "/home/ubuntu/project/raw_tlc_data"
OUTPUT_PATH_FULL = "/home/ubuntu/project/scrambled_tlc_data/scrambled_tlc_full_enriched.txt"
os.makedirs(os.path.dirname(OUTPUT_PATH_FULL), exist_ok=True)

print(f"Reading from: {BRONZE_LAYER_PATH}")
print(f"Writing to: {OUTPUT_PATH_FULL}")
```

    Reading from: /home/ubuntu/project/raw_tlc_data
    Writing to: /home/ubuntu/project/scrambled_tlc_data/scrambled_tlc_full_enriched.txt


## 3. Robust Schema Standardization Function


```python
def standardize_schema(df, taxi_type):
    """
    Standardizes columns using a comprehensive mapping and casts to a common type.
    """
    schema_mappings = {
        "tpep_pickup_datetime": "pickup_datetime", "lpep_pickup_datetime": "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime", "lpep_dropoff_datetime": "dropoff_datetime",
        "RatecodeID": "rate_code", "PULocationID": "pu_location_id", "DOLocationID": "do_location_id",
        "VendorID": "vendor_id", "fare_amount": "fare_amount", "extra": "extra", "mta_tax": "mta_tax",
        "tip_amount": "tip_amount", "tolls_amount": "tolls_amount", "improvement_surcharge": "improvement_surcharge",
        "total_amount": "total_amount", "payment_type": "payment_type", "trip_type": "trip_type",
        "passenger_count": "passenger_count", "trip_distance": "trip_distance"
    }

    for old_name, new_name in schema_mappings.items():
        if old_name in df.columns:
            df = df.withColumnRenamed(old_name, new_name)

    df = df.withColumn("taxi_type", lit(taxi_type))

    # EXPANDED FINAL SCHEMA - using DoubleType for everything numeric to avoid casting issues
    final_schema = {
        "pickup_datetime": TimestampType(),
        "passenger_count": DoubleType(),
        "trip_distance": DoubleType(),
        "rate_code": DoubleType(),  # Changed from IntegerType
        "payment_type": DoubleType(),  # Changed from IntegerType
        "fare_amount": DoubleType(),
        "tip_amount": DoubleType(),
        "tolls_amount": DoubleType(),
        "total_amount": DoubleType(),
        "taxi_type": StringType()
    }

    select_exprs = [
        col(c).cast(final_schema[c]) if c in df.columns else lit(None).cast(final_schema[c]).alias(c)
        for c in final_schema.keys()
    ]
    return df.select(*select_exprs)
```

## 4. Memory-Optimized Ingestion of ALL Files


```python
def read_and_convert_to_rdd(taxi_type):
    """
    Read parquet files and convert directly to RDD to avoid DataFrame union issues
    """
    path = os.path.join(BRONZE_LAYER_PATH, taxi_type)
    all_files = glob.glob(os.path.join(path, "*.parquet"))
    if not all_files: 
        print(f"No files found for {taxi_type}")
        return None
    
    print(f"Found {len(all_files)} {taxi_type} files")
    
    all_rdds = []
    for i, fp in enumerate(all_files, 1):
        try:
            # Read parquet file
            df = spark.read.parquet(f"file://{fp}")
            # Standardize schema
            df_std = standardize_schema(df, taxi_type)
            # Convert to RDD immediately (avoid DataFrame unions)
            rdd = df_std.rdd
            all_rdds.append(rdd)
            
            if i % 10 == 0:
                print(f"  Processed {i}/{len(all_files)} files...")
        except Exception as e:
            print(f"  Warning: Failed to process {fp}: {e}")
            continue
    
    if not all_rdds:
        return None
    
    # Union RDDs instead of DataFrames (more reliable)
    combined_rdd = all_rdds[0]
    for rdd in all_rdds[1:]:
        combined_rdd = combined_rdd.union(rdd)
    
    return combined_rdd

print("\n--- Processing yellow and green taxi files ---")
yellow_rdd = read_and_convert_to_rdd("yellow")
green_rdd = read_and_convert_to_rdd("green")

# Combine RDDs
if yellow_rdd is not None and green_rdd is not None:
    combined_rdd = yellow_rdd.union(green_rdd)
elif yellow_rdd is not None:
    combined_rdd = yellow_rdd
elif green_rdd is not None:
    combined_rdd = green_rdd
else:
    raise ValueError("No data found!")

print("\nData ingestion and schema standardization complete.")
print("  Ready for scrambling...")
```

    
    --- Processing yellow and green taxi files ---
    Found 132 yellow files


    [Stage 0:>                                                          (0 + 1) / 1]                                                                                

      Processed 10/132 files...


      Processed 20/132 files...


      Processed 30/132 files...


      Processed 40/132 files...


      Processed 50/132 files...


      Processed 60/132 files...


      Processed 70/132 files...


      Processed 80/132 files...


      Processed 90/132 files...


      Processed 100/132 files...


      Processed 110/132 files...


      Processed 120/132 files...


      Processed 130/132 files...


    Found 94 green files


      Processed 10/94 files...


      Processed 20/94 files...


      Processed 30/94 files...


      Processed 40/94 files...


      Processed 50/94 files...


      Processed 60/94 files...


      Processed 70/94 files...


      Processed 80/94 files...


      Processed 90/94 files...


    
    Data ingestion and schema standardization complete.
      Ready for scrambling...


## 5. The "Messy" Single-Line Scrambling (Enriched)


```python
def to_messy_single_line_enriched(row):
    import uuid
    timestamp = row.pickup_datetime.isoformat() if row.pickup_datetime else "NULL"
    
    # Format payment_type as integer if possible, otherwise NULL
    payment = int(row.payment_type) if row.payment_type is not None else "NULL"
    
    # Create the nested payload with all the new, interesting fields
    payload = (
        f"passengers:{row.passenger_count if row.passenger_count is not None else 'NULL'},"
        f"dist:{row.trip_distance if row.trip_distance is not None else 'NULL'},"
        f"fare:{row.fare_amount if row.fare_amount is not None else 'NULL'},"
        f"tip:{row.tip_amount if row.tip_amount is not None else 'NULL'},"
        f"tolls:{row.tolls_amount if row.tolls_amount is not None else 'NULL'},"
        f"total:{row.total_amount if row.total_amount is not None else 'NULL'}"
    )
    
    # Main line includes fields we might want to filter on directly
    return f"{timestamp}|{str(uuid.uuid4())}|{row.taxi_type}|{payment}|payload={{{payload}}}"

print("\n--- Converting to ENRICHED messy single-line format ---")
from datetime import datetime
import shutil

from pyspark.sql import functions as F
START_YEAR = 2020
END_YEAR = 2024
BASE_OUTPUT_DIR = "/home/ubuntu/project/scrambled_tlc_data"
TMP_PARQUET_DIR = os.path.join(BASE_OUTPUT_DIR, "_tmp_parquet")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

print(f"\nProcessing years {START_YEAR}–{END_YEAR} into monthly text files...")

# Convert RDD back to DataFrame (still using schema)
df = combined_rdd.toDF()

# Add month/year columns
df = df.withColumn("pickup_year", F.year("pickup_datetime"))
df = df.withColumn("pickup_month", F.date_format("pickup_datetime", "yyyy-MM"))

# Filter by year range
df = df.filter((F.col("pickup_year") >= START_YEAR) & (F.col("pickup_year") <= END_YEAR))

# Build the messy text format directly in Spark
df_text = df.withColumn(
    "line",
    F.concat_ws(
        "|",
        F.coalesce(F.date_format("pickup_datetime", "yyyy-MM-dd'T'HH:mm:ss"), F.lit("NULL")),
        F.expr("uuid()"),
        F.col("taxi_type"),
        F.coalesce(F.col("payment_type").cast("int").cast("string"), F.lit("NULL")),
        F.concat(
            F.lit("payload={"),
            F.concat_ws(
                ",",
                F.concat(F.lit("passengers:"), F.coalesce(F.col("passenger_count").cast("string"), F.lit("NULL"))),
                F.concat(F.lit("dist:"), F.coalesce(F.col("trip_distance").cast("string"), F.lit("NULL"))),
                F.concat(F.lit("fare:"), F.coalesce(F.col("fare_amount").cast("string"), F.lit("NULL"))),
                F.concat(F.lit("tip:"), F.coalesce(F.col("tip_amount").cast("string"), F.lit("NULL"))),
                F.concat(F.lit("tolls:"), F.coalesce(F.col("tolls_amount").cast("string"), F.lit("NULL"))),
                F.concat(F.lit("total:"), F.coalesce(F.col("total_amount").cast("string"), F.lit("NULL"))),
            ),
            F.lit("}")
        )
    )
)

# Save all in parallel partitioned by pickup_month
print("\nWriting data partitioned by pickup_month")

if os.path.exists(TMP_PARQUET_DIR):
    shutil.rmtree(TMP_PARQUET_DIR)

df_text.select("pickup_month", "line") \
    .write \
    .mode("overwrite") \
    .partitionBy("pickup_month") \
    .text(TMP_PARQUET_DIR)

print("All months written to partitioned directories.\n")

# Optionally flatten: move each partition to a single flat .txt file
for part_dir in glob.glob(os.path.join(TMP_PARQUET_DIR, "pickup_month=*")):
    month = os.path.basename(part_dir).split("=")[1]
    dest_file = os.path.join(BASE_OUTPUT_DIR, f"{month}.txt")

    print(f"Flattening {month} → {dest_file}")
    with open(dest_file, "wb") as outfile:
        for part in glob.glob(os.path.join(part_dir, "part-*")):
            with open(part, "rb") as infile:
                shutil.copyfileobj(infile, outfile)
    
    # DELETE the partition folder immediately after flattening
    shutil.rmtree(part_dir)
    print(f"  Cleaned up partition: {month}")

# Final cleanup of tmp directory
if os.path.exists(TMP_PARQUET_DIR):
    shutil.rmtree(TMP_PARQUET_DIR)
    print("\nCleaned up all temporary files")

print("\nAll months written as flat text files.")


```

    
    --- Converting to ENRICHED messy single-line format ---
    
    Processing years 2020–2024 into monthly text files...


    25/11/17 16:08:56 WARN DAGScheduler: Broadcasting large task binary with size 2.4 MiB


    [Stage 226:>                                                        (0 + 1) / 1]                                                                                

    25/11/17 16:08:57 WARN DAGScheduler: Broadcasting large task binary with size 2.4 MiB


    [Stage 227:============================>                            (2 + 2) / 4]                                                                                

    
    Writing data partitioned by pickup_month


    25/11/17 16:08:59 WARN DAGScheduler: Broadcasting large task binary with size 2.7 MiB


    [Stage 228:>                                                     (0 + 8) / 1272][Stage 228:>                                                     (4 + 8) / 1272]

    [Stage 228:>                                                     (7 + 8) / 1272]

    [Stage 228:>                                                     (8 + 8) / 1272][Stage 228:>                                                    (11 + 8) / 1272]

    [Stage 228:>                                                    (13 + 8) / 1272][Stage 228:>                                                    (14 + 8) / 1272]

    [Stage 228:>                                                    (15 + 8) / 1272]

    [Stage 228:>                                                    (16 + 8) / 1272]

    [Stage 228:>                                                    (18 + 8) / 1272][Stage 228:>                                                    (20 + 9) / 1272]

    [Stage 228:>                                                    (22 + 8) / 1272]

    [Stage 228:=>                                                   (24 + 8) / 1272][Stage 228:=>                                                   (26 + 8) / 1272]

    [Stage 228:=>                                                   (29 + 8) / 1272]

    [Stage 228:=>                                                   (32 + 8) / 1272][Stage 228:=>                                                   (36 + 8) / 1272]

    [Stage 228:=>                                                   (40 + 8) / 1272][Stage 228:=>                                                   (43 + 8) / 1272]

    [Stage 228:=>                                                   (47 + 8) / 1272][Stage 228:==>                                                  (51 + 8) / 1272]

    [Stage 228:==>                                                  (53 + 8) / 1272][Stage 228:==>                                                  (54 + 8) / 1272]

    [Stage 228:==>                                                  (55 + 8) / 1272]

    [Stage 228:==>                                                  (56 + 8) / 1272]

    [Stage 228:==>                                                  (57 + 8) / 1272][Stage 228:==>                                                  (59 + 8) / 1272]

    [Stage 228:==>                                                  (60 + 8) / 1272]

    [Stage 228:==>                                                  (61 + 8) / 1272][Stage 228:==>                                                  (62 + 8) / 1272]

    [Stage 228:==>                                                  (63 + 8) / 1272]

    [Stage 228:==>                                                  (64 + 8) / 1272][Stage 228:==>                                                  (65 + 8) / 1272]

    [Stage 228:==>                                                  (67 + 8) / 1272][Stage 228:==>                                                  (68 + 8) / 1272]

    [Stage 228:==>                                                  (69 + 8) / 1272][Stage 228:==>                                                  (70 + 8) / 1272]

    [Stage 228:==>                                                  (71 + 8) / 1272][Stage 228:===>                                                 (72 + 8) / 1272]

    [Stage 228:===>                                                 (73 + 8) / 1272][Stage 228:===>                                                 (74 + 8) / 1272]

    [Stage 228:===>                                                 (75 + 8) / 1272][Stage 228:===>                                                 (76 + 8) / 1272]

    [Stage 228:===>                                                 (77 + 8) / 1272][Stage 228:===>                                                 (78 + 8) / 1272]

    [Stage 228:===>                                                 (79 + 8) / 1272][Stage 228:===>                                                 (80 + 8) / 1272]

    [Stage 228:===>                                                 (81 + 8) / 1272][Stage 228:===>                                                 (83 + 8) / 1272]

    [Stage 228:===>                                                 (85 + 8) / 1272][Stage 228:===>                                                 (86 + 8) / 1272]

    [Stage 228:===>                                                 (87 + 8) / 1272][Stage 228:===>                                                 (88 + 8) / 1272]

    [Stage 228:===>                                                 (89 + 8) / 1272][Stage 228:===>                                                 (91 + 8) / 1272]

    [Stage 228:===>                                                 (93 + 8) / 1272][Stage 228:===>                                                 (94 + 8) / 1272]

    [Stage 228:====>                                                (96 + 8) / 1272]

    [Stage 228:====>                                                (97 + 8) / 1272][Stage 228:====>                                                (99 + 8) / 1272]

    [Stage 228:====>                                               (100 + 8) / 1272][Stage 228:====>                                               (102 + 8) / 1272]

    [Stage 228:====>                                               (103 + 8) / 1272]

    [Stage 228:====>                                               (104 + 8) / 1272]

    [Stage 228:====>                                               (105 + 8) / 1272][Stage 228:====>                                               (106 + 8) / 1272]

    [Stage 228:====>                                               (107 + 8) / 1272][Stage 228:====>                                               (108 + 8) / 1272]

    [Stage 228:====>                                               (109 + 8) / 1272][Stage 228:====>                                               (110 + 8) / 1272]

    [Stage 228:====>                                               (111 + 8) / 1272][Stage 228:====>                                               (112 + 8) / 1272]

    [Stage 228:====>                                               (114 + 8) / 1272][Stage 228:====>                                               (115 + 8) / 1272]

    [Stage 228:====>                                               (117 + 8) / 1272][Stage 228:====>                                               (119 + 8) / 1272]

    [Stage 228:====>                                               (120 + 8) / 1272][Stage 228:====>                                               (122 + 8) / 1272]

    [Stage 228:=====>                                              (124 + 8) / 1272]

    [Stage 228:=====>                                              (128 + 8) / 1272][Stage 228:=====>                                              (130 + 8) / 1272]

    [Stage 228:=====>                                              (132 + 8) / 1272][Stage 228:=====>                                              (135 + 8) / 1272]

    [Stage 228:=====>                                              (138 + 8) / 1272][Stage 228:=====>                                              (139 + 8) / 1272]

    [Stage 228:=====>                                              (141 + 8) / 1272]

    [Stage 228:=====>                                              (143 + 8) / 1272][Stage 228:=====>                                              (144 + 8) / 1272]

    [Stage 228:=====>                                              (145 + 8) / 1272][Stage 228:=====>                                              (146 + 8) / 1272]

    [Stage 228:======>                                             (147 + 8) / 1272][Stage 228:======>                                             (148 + 8) / 1272]

    [Stage 228:======>                                             (150 + 8) / 1272][Stage 228:======>                                             (151 + 8) / 1272]

    [Stage 228:======>                                             (152 + 8) / 1272][Stage 228:======>                                             (155 + 8) / 1272]

    [Stage 228:======>                                             (156 + 8) / 1272]

    [Stage 228:======>                                             (157 + 8) / 1272]

    [Stage 228:======>                                             (159 + 8) / 1272][Stage 228:======>                                             (160 + 8) / 1272]

    [Stage 228:======>                                             (161 + 8) / 1272][Stage 228:======>                                             (162 + 8) / 1272]

    [Stage 228:======>                                             (163 + 8) / 1272][Stage 228:======>                                             (164 + 8) / 1272]

    [Stage 228:======>                                             (167 + 8) / 1272]

    [Stage 228:======>                                             (170 + 8) / 1272]

    [Stage 228:======>                                             (171 + 8) / 1272][Stage 228:=======>                                            (172 + 8) / 1272]

    [Stage 228:=======>                                            (174 + 8) / 1272][Stage 228:=======>                                            (175 + 8) / 1272]

    [Stage 228:=======>                                            (176 + 8) / 1272][Stage 228:=======>                                            (177 + 8) / 1272]

    [Stage 228:=======>                                            (178 + 8) / 1272]

    [Stage 228:=======>                                            (179 + 8) / 1272][Stage 228:=======>                                            (181 + 8) / 1272]

    [Stage 228:=======>                                            (183 + 8) / 1272]

    [Stage 228:=======>                                            (184 + 8) / 1272][Stage 228:=======>                                            (185 + 8) / 1272]

    [Stage 228:=======>                                            (186 + 8) / 1272]

    [Stage 228:=======>                                            (187 + 8) / 1272]

    [Stage 228:=======>                                            (188 + 8) / 1272][Stage 228:=======>                                            (190 + 8) / 1272]

    [Stage 228:=======>                                            (191 + 8) / 1272][Stage 228:=======>                                            (192 + 8) / 1272]

    [Stage 228:=======>                                            (193 + 8) / 1272][Stage 228:=======>                                            (194 + 8) / 1272]

    [Stage 228:=======>                                            (195 + 8) / 1272]

    [Stage 228:========>                                           (196 + 8) / 1272]

    [Stage 228:========>                                           (198 + 8) / 1272][Stage 228:========>                                           (199 + 8) / 1272]

    [Stage 228:========>                                           (200 + 8) / 1272]

    [Stage 228:========>                                           (201 + 8) / 1272][Stage 228:========>                                           (202 + 8) / 1272]

    [Stage 228:========>                                           (203 + 8) / 1272][Stage 228:========>                                           (204 + 8) / 1272]

    [Stage 228:========>                                           (206 + 8) / 1272][Stage 228:========>                                           (208 + 8) / 1272]

    [Stage 228:========>                                           (209 + 8) / 1272][Stage 228:========>                                           (210 + 8) / 1272]

    [Stage 228:========>                                           (211 + 8) / 1272]

    [Stage 228:========>                                           (213 + 8) / 1272][Stage 228:========>                                           (214 + 8) / 1272]

    [Stage 228:========>                                           (216 + 8) / 1272]

    [Stage 228:========>                                           (217 + 8) / 1272][Stage 228:========>                                           (218 + 8) / 1272]

    [Stage 228:========>                                           (219 + 8) / 1272][Stage 228:========>                                           (220 + 8) / 1272]

    [Stage 228:=========>                                          (221 + 8) / 1272]

    [Stage 228:=========>                                          (222 + 8) / 1272]

    [Stage 228:=========>                                          (224 + 8) / 1272]

    [Stage 228:=========>                                          (225 + 8) / 1272]

    [Stage 228:=========>                                          (226 + 8) / 1272][Stage 228:=========>                                          (227 + 8) / 1272]

    [Stage 228:=========>                                          (228 + 8) / 1272][Stage 228:=========>                                          (231 + 8) / 1272]

    [Stage 228:=========>                                          (232 + 8) / 1272][Stage 228:=========>                                          (233 + 8) / 1272]

    [Stage 228:=========>                                          (234 + 8) / 1272]

    [Stage 228:=========>                                          (235 + 8) / 1272]

    [Stage 228:=========>                                          (238 + 8) / 1272][Stage 228:=========>                                          (240 + 8) / 1272]

    [Stage 228:=========>                                          (242 + 8) / 1272][Stage 228:=========>                                          (244 + 8) / 1272]

    [Stage 228:==========>                                         (245 + 8) / 1272][Stage 228:==========>                                         (247 + 8) / 1272]

    [Stage 228:==========>                                         (249 + 8) / 1272]

    [Stage 228:==========>                                         (250 + 8) / 1272][Stage 228:==========>                                         (251 + 8) / 1272]

    [Stage 228:==========>                                         (252 + 8) / 1272][Stage 228:==========>                                         (254 + 8) / 1272]

    [Stage 228:==========>                                         (255 + 8) / 1272][Stage 228:==========>                                         (256 + 8) / 1272]

    [Stage 228:==========>                                         (258 + 8) / 1272]

    [Stage 228:==========>                                         (260 + 8) / 1272][Stage 228:==========>                                         (262 + 8) / 1272]

    [Stage 228:==========>                                         (264 + 8) / 1272]

    [Stage 228:==========>                                         (265 + 8) / 1272][Stage 228:==========>                                         (267 + 8) / 1272]

    [Stage 228:==========>                                         (268 + 8) / 1272][Stage 228:==========>                                         (269 + 8) / 1272]

    [Stage 228:===========>                                        (270 + 8) / 1272]

    [Stage 228:===========>                                        (271 + 9) / 1272][Stage 228:===========>                                        (272 + 8) / 1272]

    [Stage 228:===========>                                        (273 + 8) / 1272][Stage 228:===========>                                        (274 + 8) / 1272]

    [Stage 228:===========>                                        (275 + 8) / 1272]

    [Stage 228:===========>                                        (277 + 8) / 1272][Stage 228:===========>                                        (279 + 8) / 1272]

    [Stage 228:===========>                                        (280 + 8) / 1272]

    [Stage 228:===========>                                        (281 + 8) / 1272][Stage 228:===========>                                        (283 + 8) / 1272]

    [Stage 228:===========>                                        (284 + 8) / 1272][Stage 228:===========>                                        (285 + 8) / 1272]

    [Stage 228:===========>                                        (286 + 8) / 1272][Stage 228:===========>                                        (288 + 8) / 1272]

    [Stage 228:===========>                                        (289 + 8) / 1272]

    [Stage 228:===========>                                        (290 + 8) / 1272][Stage 228:===========>                                        (291 + 8) / 1272]

    [Stage 228:===========>                                        (293 + 8) / 1272][Stage 228:============>                                       (294 + 8) / 1272]

    [Stage 228:============>                                       (296 + 8) / 1272]

    [Stage 228:============>                                       (298 + 8) / 1272]

    [Stage 228:============>                                       (299 + 8) / 1272][Stage 228:============>                                       (300 + 8) / 1272]

    [Stage 228:============>                                       (301 + 8) / 1272][Stage 228:============>                                       (303 + 8) / 1272]

    [Stage 228:============>                                       (304 + 8) / 1272]

    [Stage 228:============>                                       (306 + 8) / 1272][Stage 228:============>                                       (307 + 9) / 1272]

    [Stage 228:============>                                       (309 + 8) / 1272]

    [Stage 228:============>                                       (310 + 8) / 1272][Stage 228:============>                                       (312 + 8) / 1272]

    [Stage 228:============>                                       (313 + 8) / 1272][Stage 228:============>                                       (314 + 8) / 1272]

    [Stage 228:============>                                       (317 + 8) / 1272][Stage 228:=============>                                      (319 + 8) / 1272]

    [Stage 228:=============>                                      (320 + 8) / 1272]

    [Stage 228:=============>                                      (322 + 8) / 1272][Stage 228:=============>                                      (323 + 9) / 1272]

    [Stage 228:=============>                                      (325 + 8) / 1272]

    [Stage 228:=============>                                      (326 + 8) / 1272]

    [Stage 228:=============>                                      (327 + 8) / 1272][Stage 228:=============>                                      (328 + 8) / 1272]

    [Stage 228:=============>                                      (329 + 9) / 1272]

    [Stage 228:=============>                                      (330 + 8) / 1272]

    [Stage 228:=============>                                      (331 + 8) / 1272]

    [Stage 228:=============>                                      (332 + 8) / 1272][Stage 228:=============>                                      (333 + 8) / 1272]

    [Stage 228:=============>                                      (335 + 8) / 1272][Stage 228:=============>                                      (337 + 8) / 1272]

    [Stage 228:=============>                                      (338 + 8) / 1272][Stage 228:=============>                                      (340 + 8) / 1272]

    [Stage 228:=============>                                      (342 + 8) / 1272]

    [Stage 228:==============>                                     (343 + 8) / 1272][Stage 228:==============>                                     (344 + 8) / 1272]

    [Stage 228:==============>                                     (345 + 8) / 1272][Stage 228:==============>                                     (346 + 8) / 1272]

    [Stage 228:==============>                                     (348 + 8) / 1272]

    [Stage 228:==============>                                     (351 + 8) / 1272][Stage 228:==============>                                     (352 + 8) / 1272]

    [Stage 228:==============>                                     (353 + 8) / 1272][Stage 228:==============>                                     (354 + 8) / 1272]

    [Stage 228:==============>                                     (355 + 8) / 1272]

    [Stage 228:==============>                                     (357 + 8) / 1272]

    [Stage 228:==============>                                     (359 + 8) / 1272][Stage 228:==============>                                     (360 + 8) / 1272]

    [Stage 228:==============>                                     (362 + 8) / 1272][Stage 228:==============>                                     (363 + 8) / 1272]

    [Stage 228:==============>                                     (365 + 8) / 1272]

    [Stage 228:==============>                                     (366 + 8) / 1272]

    [Stage 228:===============>                                    (367 + 8) / 1272][Stage 228:===============>                                    (369 + 8) / 1272]

    [Stage 228:===============>                                    (370 + 8) / 1272][Stage 228:===============>                                    (371 + 8) / 1272]

    [Stage 228:===============>                                    (374 + 8) / 1272]

    [Stage 228:===============>                                    (376 + 8) / 1272]

    [Stage 228:===============>                                    (380 + 8) / 1272][Stage 228:===============>                                    (382 + 8) / 1272]

    [Stage 228:===============>                                    (384 + 8) / 1272][Stage 228:===============>                                    (386 + 8) / 1272]

    [Stage 228:===============>                                    (387 + 8) / 1272]

    [Stage 228:===============>                                    (389 + 8) / 1272][Stage 228:===============>                                    (390 + 8) / 1272]

    [Stage 228:================>                                   (392 + 8) / 1272][Stage 228:================>                                   (394 + 8) / 1272]

    [Stage 228:================>                                   (395 + 8) / 1272]

    [Stage 228:================>                                   (397 + 8) / 1272][Stage 228:================>                                   (398 + 8) / 1272]

    [Stage 228:================>                                   (399 + 8) / 1272][Stage 228:================>                                   (400 + 8) / 1272]

    [Stage 228:================>                                   (402 + 8) / 1272]

    [Stage 228:================>                                   (403 + 8) / 1272][Stage 228:================>                                   (404 + 8) / 1272]

    [Stage 228:================>                                   (405 + 8) / 1272]

    [Stage 228:================>                                   (407 + 8) / 1272][Stage 228:================>                                   (408 + 8) / 1272]

    [Stage 228:================>                                   (409 + 8) / 1272]

    [Stage 228:================>                                   (410 + 8) / 1272][Stage 228:================>                                   (411 + 8) / 1272]

    [Stage 228:================>                                   (412 + 8) / 1272][Stage 228:================>                                   (413 + 8) / 1272]

    [Stage 228:================>                                   (414 + 8) / 1272][Stage 228:=================>                                  (416 + 8) / 1272]

    [Stage 228:=================>                                  (417 + 8) / 1272][Stage 228:=================>                                  (419 + 8) / 1272]

    [Stage 228:=================>                                  (420 + 8) / 1272][Stage 228:=================>                                  (421 + 8) / 1272]

    [Stage 228:=================>                                  (422 + 8) / 1272][Stage 228:=================>                                  (424 + 8) / 1272]

    [Stage 228:=================>                                  (426 + 8) / 1272][Stage 228:=================>                                  (427 + 8) / 1272]

    [Stage 228:=================>                                  (428 + 8) / 1272]

    [Stage 228:=================>                                  (429 + 8) / 1272]

    [Stage 228:=================>                                  (430 + 8) / 1272]

    [Stage 228:=================>                                  (431 + 8) / 1272][Stage 228:=================>                                  (432 + 8) / 1272]

    [Stage 228:=================>                                  (433 + 8) / 1272][Stage 228:=================>                                  (434 + 8) / 1272]

    [Stage 228:=================>                                  (436 + 8) / 1272][Stage 228:=================>                                  (437 + 8) / 1272]

    [Stage 228:=================>                                  (438 + 8) / 1272][Stage 228:=================>                                  (439 + 8) / 1272]

    [Stage 228:==================>                                 (441 + 8) / 1272][Stage 228:==================>                                 (442 + 8) / 1272]

    [Stage 228:==================>                                 (445 + 8) / 1272][Stage 228:==================>                                 (449 + 8) / 1272]

    [Stage 228:==================>                                 (453 + 8) / 1272]

    [Stage 228:==================>                                 (454 + 8) / 1272][Stage 228:==================>                                 (455 + 8) / 1272]

    [Stage 228:==================>                                 (456 + 8) / 1272][Stage 228:==================>                                 (457 + 9) / 1272]

    [Stage 228:==================>                                 (459 + 8) / 1272]

    [Stage 228:==================>                                 (460 + 8) / 1272]

    [Stage 228:==================>                                 (461 + 8) / 1272]

    [Stage 228:==================>                                 (462 + 8) / 1272][Stage 228:==================>                                 (464 + 8) / 1272]

    [Stage 228:===================>                                (465 + 8) / 1272][Stage 228:===================>                                (467 + 8) / 1272]

    [Stage 228:===================>                                (468 + 8) / 1272]

    [Stage 228:===================>                                (469 + 8) / 1272]

    [Stage 228:===================>                                (470 + 8) / 1272]

    [Stage 228:===================>                                (472 + 8) / 1272][Stage 228:===================>                                (473 + 8) / 1272]

    [Stage 228:===================>                                (475 + 8) / 1272]

    [Stage 228:===================>                                (477 + 8) / 1272]

    [Stage 228:===================>                                (478 + 8) / 1272][Stage 228:===================>                                (479 + 8) / 1272]

    [Stage 228:===================>                                (480 + 8) / 1272]

    [Stage 228:===================>                                (481 + 8) / 1272]

    [Stage 228:===================>                                (482 + 8) / 1272]

    [Stage 228:===================>                                (483 + 8) / 1272][Stage 228:===================>                                (485 + 8) / 1272]

    [Stage 228:===================>                                (486 + 8) / 1272][Stage 228:===================>                                (489 + 8) / 1272]

    [Stage 228:====================>                               (492 + 8) / 1272][Stage 228:====================>                               (493 + 8) / 1272]

    [Stage 228:====================>                               (495 + 8) / 1272]

    [Stage 228:====================>                               (497 + 8) / 1272]

    [Stage 228:====================>                               (498 + 8) / 1272][Stage 228:====================>                               (499 + 8) / 1272]

    [Stage 228:====================>                               (500 + 8) / 1272][Stage 228:====================>                               (501 + 8) / 1272]

    [Stage 228:====================>                               (502 + 8) / 1272][Stage 228:====================>                               (503 + 8) / 1272]

    [Stage 228:====================>                               (504 + 8) / 1272][Stage 228:====================>                               (506 + 8) / 1272]

    [Stage 228:====================>                               (508 + 8) / 1272]

    [Stage 228:====================>                               (509 + 8) / 1272]

    [Stage 228:====================>                               (510 + 8) / 1272][Stage 228:====================>                               (511 + 8) / 1272]

    [Stage 228:====================>                               (513 + 8) / 1272]

    [Stage 228:=====================>                              (514 + 8) / 1272][Stage 228:=====================>                              (515 + 8) / 1272]

    [Stage 228:=====================>                              (517 + 8) / 1272]

    [Stage 228:=====================>                              (518 + 8) / 1272][Stage 228:=====================>                              (519 + 8) / 1272]

    [Stage 228:=====================>                              (521 + 8) / 1272][Stage 228:=====================>                              (522 + 8) / 1272]

    [Stage 228:=====================>                              (523 + 8) / 1272][Stage 228:=====================>                              (524 + 8) / 1272]

    [Stage 228:=====================>                              (525 + 8) / 1272]

    [Stage 228:=====================>                              (527 + 8) / 1272][Stage 228:=====================>                              (529 + 8) / 1272]

    [Stage 228:=====================>                              (531 + 8) / 1272]

    [Stage 228:=====================>                              (533 + 8) / 1272]

    [Stage 228:=====================>                              (534 + 8) / 1272]

    [Stage 228:=====================>                              (535 + 8) / 1272][Stage 228:=====================>                              (536 + 8) / 1272]

    [Stage 228:=====================>                              (538 + 8) / 1272][Stage 228:======================>                             (539 + 8) / 1272]

    [Stage 228:======================>                             (540 + 8) / 1272][Stage 228:======================>                             (541 + 8) / 1272]

    [Stage 228:======================>                             (542 + 8) / 1272]

    [Stage 228:======================>                             (543 + 8) / 1272]

    [Stage 228:======================>                             (544 + 8) / 1272]

    [Stage 228:======================>                             (545 + 8) / 1272]

    [Stage 228:======================>                             (546 + 8) / 1272]

    [Stage 228:======================>                             (549 + 8) / 1272][Stage 228:======================>                             (552 + 8) / 1272]

    [Stage 228:======================>                             (556 + 8) / 1272][Stage 228:======================>                             (560 + 8) / 1272]

    [Stage 228:=======================>                            (563 + 8) / 1272][Stage 228:=======================>                            (565 + 8) / 1272]

    [Stage 228:=======================>                            (567 + 8) / 1272][Stage 228:=======================>                            (569 + 8) / 1272]

    [Stage 228:=======================>                            (571 + 8) / 1272]

    [Stage 228:=======================>                            (572 + 9) / 1272]

    [Stage 228:=======================>                            (573 + 8) / 1272]

    [Stage 228:=======================>                            (574 + 8) / 1272]

    [Stage 228:=======================>                            (576 + 8) / 1272][Stage 228:=======================>                            (577 + 8) / 1272]

    [Stage 228:=======================>                            (579 + 8) / 1272][Stage 228:=======================>                            (582 + 8) / 1272]

    [Stage 228:=======================>                            (583 + 8) / 1272][Stage 228:=======================>                            (584 + 8) / 1272]

    [Stage 228:=======================>                            (586 + 8) / 1272]

    [Stage 228:========================>                           (588 + 8) / 1272][Stage 228:========================>                           (589 + 8) / 1272]

    [Stage 228:========================>                           (589 + 8) / 1272]

    [Stage 228:========================>                           (590 + 8) / 1272]

    [Stage 228:========================>                           (591 + 8) / 1272][Stage 228:========================>                           (593 + 8) / 1272]

    [Stage 228:========================>                           (595 + 8) / 1272]

    [Stage 228:========================>                           (597 + 8) / 1272]

    [Stage 228:========================>                           (599 + 8) / 1272][Stage 228:========================>                           (600 + 8) / 1272]

    [Stage 228:========================>                           (601 + 8) / 1272][Stage 228:========================>                           (602 + 8) / 1272]

    [Stage 228:========================>                           (604 + 8) / 1272][Stage 228:========================>                           (605 + 8) / 1272]

    [Stage 228:========================>                           (606 + 8) / 1272]

    [Stage 228:========================>                           (607 + 8) / 1272][Stage 228:========================>                           (608 + 8) / 1272]

    [Stage 228:========================>                           (610 + 8) / 1272][Stage 228:========================>                           (611 + 8) / 1272]

    [Stage 228:=========================>                          (613 + 8) / 1272]

    [Stage 228:=========================>                          (615 + 8) / 1272][Stage 228:=========================>                          (616 + 8) / 1272]

    [Stage 228:=========================>                          (618 + 8) / 1272][Stage 228:=========================>                          (619 + 9) / 1272]

    [Stage 228:=========================>                          (621 + 8) / 1272]

    [Stage 228:=========================>                          (622 + 8) / 1272]

    [Stage 228:=========================>                          (623 + 8) / 1272][Stage 228:=========================>                          (624 + 8) / 1272]

    [Stage 228:=========================>                          (625 + 8) / 1272][Stage 228:=========================>                          (626 + 8) / 1272]

    [Stage 228:=========================>                          (628 + 8) / 1272]

    [Stage 228:=========================>                          (629 + 8) / 1272]

    [Stage 228:=========================>                          (630 + 8) / 1272][Stage 228:=========================>                          (631 + 8) / 1272]

    [Stage 228:=========================>                          (632 + 8) / 1272][Stage 228:=========================>                          (633 + 8) / 1272]

    [Stage 228:=========================>                          (634 + 8) / 1272][Stage 228:==========================>                         (636 + 8) / 1272]

    [Stage 228:==========================>                         (637 + 8) / 1272]

    [Stage 228:==========================>                         (638 + 8) / 1272][Stage 228:==========================>                         (639 + 8) / 1272]

    [Stage 228:==========================>                         (641 + 8) / 1272][Stage 228:==========================>                         (642 + 8) / 1272]

    [Stage 228:==========================>                         (644 + 8) / 1272][Stage 228:==========================>                         (645 + 8) / 1272]

    [Stage 228:==========================>                         (647 + 8) / 1272][Stage 228:==========================>                         (648 + 8) / 1272]

    [Stage 228:==========================>                         (650 + 8) / 1272][Stage 228:==========================>                         (651 + 8) / 1272]

    [Stage 228:==========================>                         (653 + 8) / 1272]

    [Stage 228:==========================>                         (654 + 8) / 1272][Stage 228:==========================>                         (655 + 9) / 1272]

    [Stage 228:==========================>                         (657 + 8) / 1272]

    [Stage 228:==========================>                         (658 + 8) / 1272]

    [Stage 228:==========================>                         (660 + 8) / 1272][Stage 228:===========================>                        (661 + 8) / 1272]

    [Stage 228:===========================>                        (662 + 8) / 1272][Stage 228:===========================>                        (663 + 9) / 1272]

    [Stage 228:===========================>                        (665 + 8) / 1272]

    [Stage 228:===========================>                        (667 + 8) / 1272][Stage 228:===========================>                        (669 + 8) / 1272]

    [Stage 228:===========================>                        (670 + 8) / 1272]

    [Stage 228:===========================>                        (671 + 8) / 1272]

    [Stage 228:===========================>                        (672 + 8) / 1272][Stage 228:===========================>                        (674 + 8) / 1272]

    [Stage 228:===========================>                        (676 + 8) / 1272]

    [Stage 228:===========================>                        (677 + 8) / 1272]

    [Stage 228:===========================>                        (678 + 8) / 1272]

    [Stage 228:===========================>                        (679 + 8) / 1272][Stage 228:===========================>                        (681 + 8) / 1272]

    [Stage 228:===========================>                        (682 + 8) / 1272]

    [Stage 228:===========================>                        (683 + 8) / 1272][Stage 228:===========================>                        (684 + 8) / 1272]

    [Stage 228:============================>                       (685 + 8) / 1272]

    [Stage 228:============================>                       (687 + 8) / 1272]

    [Stage 228:============================>                       (688 + 8) / 1272][Stage 228:============================>                       (690 + 8) / 1272]

    [Stage 228:============================>                       (691 + 8) / 1272]

    [Stage 228:============================>                       (692 + 8) / 1272]

    [Stage 228:============================>                       (694 + 8) / 1272]

    [Stage 228:============================>                       (696 + 8) / 1272][Stage 228:============================>                       (698 + 8) / 1272]

    [Stage 228:============================>                       (701 + 8) / 1272]

    [Stage 228:============================>                       (703 + 8) / 1272][Stage 228:============================>                       (704 + 8) / 1272]

    [Stage 228:============================>                       (706 + 8) / 1272][Stage 228:============================>                       (708 + 8) / 1272]

    [Stage 228:=============================>                      (710 + 8) / 1272][Stage 228:=============================>                      (711 + 8) / 1272]

    [Stage 228:=============================>                      (712 + 8) / 1272][Stage 228:=============================>                      (713 + 8) / 1272]

    [Stage 228:=============================>                      (714 + 8) / 1272][Stage 228:=============================>                      (715 + 8) / 1272]

    [Stage 228:=============================>                      (716 + 8) / 1272]

    [Stage 228:=============================>                      (717 + 8) / 1272][Stage 228:=============================>                      (718 + 8) / 1272]

    [Stage 228:=============================>                      (719 + 8) / 1272]

    [Stage 228:=============================>                      (720 + 8) / 1272][Stage 228:=============================>                      (722 + 8) / 1272]

    [Stage 228:=============================>                      (723 + 8) / 1272][Stage 228:=============================>                      (724 + 8) / 1272]

    [Stage 228:=============================>                      (725 + 8) / 1272][Stage 228:=============================>                      (726 + 8) / 1272]

    [Stage 228:=============================>                      (728 + 9) / 1272]

    [Stage 228:=============================>                      (730 + 8) / 1272][Stage 228:=============================>                      (732 + 8) / 1272]

    [Stage 228:=============================>                      (733 + 8) / 1272]

    [Stage 228:==============================>                     (734 + 8) / 1272]

    [Stage 228:==============================>                     (735 + 8) / 1272]

    [Stage 228:==============================>                     (736 + 8) / 1272]

    [Stage 228:==============================>                     (737 + 8) / 1272][Stage 228:==============================>                     (739 + 8) / 1272]

    [Stage 228:==============================>                     (740 + 8) / 1272]

    [Stage 228:==============================>                     (742 + 8) / 1272]

    [Stage 228:==============================>                     (743 + 8) / 1272][Stage 228:==============================>                     (744 + 8) / 1272]

    [Stage 228:==============================>                     (746 + 8) / 1272]

    [Stage 228:==============================>                     (748 + 8) / 1272]

    [Stage 228:==============================>                     (749 + 8) / 1272][Stage 228:==============================>                     (750 + 8) / 1272]

    [Stage 228:==============================>                     (751 + 8) / 1272][Stage 228:==============================>                     (752 + 8) / 1272]

    [Stage 228:==============================>                     (754 + 8) / 1272]

    [Stage 228:==============================>                     (755 + 8) / 1272]

    [Stage 228:==============================>                     (756 + 8) / 1272]

    [Stage 228:==============================>                     (757 + 8) / 1272][Stage 228:===============================>                    (759 + 8) / 1272]

    [Stage 228:===============================>                    (761 + 8) / 1272][Stage 228:===============================>                    (762 + 8) / 1272]

    [Stage 228:===============================>                    (764 + 8) / 1272][Stage 228:===============================>                    (766 + 8) / 1272]

    [Stage 228:===============================>                    (767 + 8) / 1272]

    [Stage 228:===============================>                    (769 + 8) / 1272]

    [Stage 228:===============================>                    (770 + 8) / 1272][Stage 228:===============================>                    (771 + 8) / 1272]

    [Stage 228:===============================>                    (773 + 8) / 1272]

    [Stage 228:===============================>                    (775 + 9) / 1272][Stage 228:===============================>                    (777 + 8) / 1272]

    [Stage 228:===============================>                    (778 + 8) / 1272]

    [Stage 228:===============================>                    (780 + 8) / 1272][Stage 228:===============================>                    (781 + 8) / 1272]

    [Stage 228:================================>                   (783 + 8) / 1272][Stage 228:================================>                   (784 + 8) / 1272]

    [Stage 228:================================>                   (786 + 8) / 1272]

    [Stage 228:================================>                   (788 + 8) / 1272][Stage 228:================================>                   (789 + 8) / 1272]

    [Stage 228:================================>                   (790 + 8) / 1272][Stage 228:================================>                   (792 + 8) / 1272]

    [Stage 228:================================>                   (793 + 8) / 1272][Stage 228:================================>                   (794 + 8) / 1272]

    [Stage 228:================================>                   (795 + 8) / 1272]

    [Stage 228:================================>                   (796 + 8) / 1272][Stage 228:================================>                   (798 + 8) / 1272]

    [Stage 228:================================>                   (800 + 8) / 1272][Stage 228:================================>                   (802 + 8) / 1272]

    [Stage 228:================================>                   (803 + 8) / 1272]

    [Stage 228:================================>                   (804 + 8) / 1272][Stage 228:================================>                   (806 + 8) / 1272]

    [Stage 228:================================>                   (807 + 8) / 1272][Stage 228:=================================>                  (808 + 8) / 1272]

    [Stage 228:=================================>                  (809 + 8) / 1272]

    [Stage 228:=================================>                  (810 + 8) / 1272]

    [Stage 228:=================================>                  (811 + 9) / 1272]

    [Stage 228:=================================>                  (812 + 8) / 1272]

    [Stage 228:=================================>                  (813 + 8) / 1272]

    [Stage 228:=================================>                  (814 + 8) / 1272][Stage 228:=================================>                  (815 + 8) / 1272]

    [Stage 228:=================================>                  (818 + 8) / 1272][Stage 228:=================================>                  (819 + 8) / 1272]

    [Stage 228:=================================>                  (821 + 8) / 1272]

    [Stage 228:=================================>                  (824 + 8) / 1272][Stage 228:=================================>                  (826 + 8) / 1272]

    [Stage 228:=================================>                  (827 + 8) / 1272][Stage 228:=================================>                  (828 + 8) / 1272]

    [Stage 228:=================================>                  (829 + 8) / 1272]

    [Stage 228:=================================>                  (830 + 8) / 1272][Stage 228:=================================>                  (831 + 8) / 1272]

    [Stage 228:==================================>                 (832 + 8) / 1272]

    [Stage 228:==================================>                 (833 + 8) / 1272]

    [Stage 228:==================================>                 (834 + 8) / 1272][Stage 228:==================================>                 (835 + 8) / 1272]

    [Stage 228:==================================>                 (836 + 8) / 1272]

    [Stage 228:==================================>                 (837 + 8) / 1272][Stage 228:==================================>                 (838 + 8) / 1272]

    [Stage 228:==================================>                 (839 + 8) / 1272][Stage 228:==================================>                 (840 + 8) / 1272]

    [Stage 228:==================================>                 (841 + 8) / 1272][Stage 228:==================================>                 (842 + 8) / 1272]

    [Stage 228:==================================>                 (843 + 8) / 1272][Stage 228:==================================>                 (844 + 8) / 1272]

    [Stage 228:==================================>                 (845 + 8) / 1272][Stage 228:==================================>                 (846 + 8) / 1272]

    [Stage 228:==================================>                 (847 + 8) / 1272][Stage 228:==================================>                 (848 + 8) / 1272]

    [Stage 228:==================================>                 (849 + 8) / 1272][Stage 228:==================================>                 (850 + 8) / 1272]

    [Stage 228:==================================>                 (851 + 8) / 1272][Stage 228:==================================>                 (852 + 8) / 1272]

    [Stage 228:==================================>                 (853 + 8) / 1272]

    [Stage 228:==================================>                 (854 + 8) / 1272]

    [Stage 228:==================================>                 (855 + 8) / 1272][Stage 228:==================================>                 (856 + 8) / 1272]

    [Stage 228:===================================>                (858 + 8) / 1272]

    [Stage 228:===================================>                (859 + 8) / 1272]

    [Stage 228:===================================>                (860 + 8) / 1272]

    [Stage 228:===================================>                (861 + 8) / 1272]

    [Stage 228:===================================>                (862 + 8) / 1272][Stage 228:===================================>                (863 + 8) / 1272]

    [Stage 228:===================================>                (864 + 8) / 1272][Stage 228:===================================>                (865 + 8) / 1272]

    [Stage 228:===================================>                (866 + 8) / 1272][Stage 228:===================================>                (868 + 8) / 1272]

    [Stage 228:===================================>                (870 + 9) / 1272][Stage 228:===================================>                (872 + 8) / 1272]

    [Stage 228:===================================>                (873 + 8) / 1272][Stage 228:===================================>                (874 + 8) / 1272]

    [Stage 228:===================================>                (875 + 8) / 1272][Stage 228:===================================>                (876 + 8) / 1272]

    [Stage 228:===================================>                (877 + 8) / 1272][Stage 228:===================================>                (878 + 8) / 1272]

    [Stage 228:===================================>                (879 + 8) / 1272]

    [Stage 228:====================================>               (881 + 8) / 1272]

    [Stage 228:====================================>               (882 + 8) / 1272][Stage 228:====================================>               (883 + 8) / 1272]

    [Stage 228:====================================>               (884 + 8) / 1272][Stage 228:====================================>               (885 + 8) / 1272]

    [Stage 228:====================================>               (886 + 8) / 1272]

    [Stage 228:====================================>               (887 + 8) / 1272]

    [Stage 228:====================================>               (888 + 8) / 1272][Stage 228:====================================>               (890 + 8) / 1272]

    [Stage 228:====================================>               (891 + 8) / 1272][Stage 228:====================================>               (892 + 8) / 1272]

    [Stage 228:====================================>               (893 + 8) / 1272][Stage 228:====================================>               (894 + 8) / 1272]

    [Stage 228:====================================>               (895 + 8) / 1272]

    [Stage 228:====================================>               (897 + 8) / 1272]

    [Stage 228:====================================>               (898 + 8) / 1272]

    [Stage 228:====================================>               (899 + 8) / 1272][Stage 228:====================================>               (900 + 8) / 1272]

    [Stage 228:====================================>               (901 + 8) / 1272][Stage 228:====================================>               (902 + 8) / 1272]

    [Stage 228:====================================>               (905 + 8) / 1272]

    [Stage 228:=====================================>              (907 + 8) / 1272][Stage 228:=====================================>              (909 + 8) / 1272]

    [Stage 228:=====================================>              (911 + 8) / 1272][Stage 228:=====================================>              (912 + 8) / 1272]

    [Stage 228:=====================================>              (913 + 8) / 1272][Stage 228:=====================================>              (915 + 8) / 1272]

    [Stage 228:=====================================>              (919 + 8) / 1272][Stage 228:=====================================>              (920 + 8) / 1272]

    [Stage 228:=====================================>              (921 + 8) / 1272]

    [Stage 228:=====================================>              (922 + 8) / 1272]

    [Stage 228:=====================================>              (924 + 8) / 1272]

    [Stage 228:=====================================>              (925 + 8) / 1272][Stage 228:=====================================>              (926 + 8) / 1272]

    [Stage 228:=====================================>              (927 + 8) / 1272][Stage 228:=====================================>              (928 + 8) / 1272]

    [Stage 228:=====================================>              (929 + 8) / 1272][Stage 228:======================================>             (931 + 8) / 1272]

    [Stage 228:======================================>             (933 + 8) / 1272][Stage 228:======================================>             (934 + 8) / 1272]

    [Stage 228:======================================>             (935 + 8) / 1272][Stage 228:======================================>             (936 + 8) / 1272]

    [Stage 228:======================================>             (938 + 8) / 1272]

    [Stage 228:======================================>             (939 + 8) / 1272][Stage 228:======================================>             (940 + 8) / 1272]

    [Stage 228:======================================>             (942 + 8) / 1272]

    [Stage 228:======================================>             (943 + 8) / 1272][Stage 228:======================================>             (945 + 8) / 1272]

    [Stage 228:======================================>             (946 + 8) / 1272][Stage 228:======================================>             (947 + 8) / 1272]

    [Stage 228:======================================>             (948 + 8) / 1272][Stage 228:======================================>             (950 + 8) / 1272]

    [Stage 228:======================================>             (952 + 8) / 1272]

    [Stage 228:======================================>             (953 + 8) / 1272][Stage 228:=======================================>            (954 + 8) / 1272]

    [Stage 228:=======================================>            (955 + 8) / 1272][Stage 228:=======================================>            (956 + 8) / 1272]

    [Stage 228:=======================================>            (957 + 8) / 1272]

    [Stage 228:=======================================>            (959 + 8) / 1272]

    [Stage 228:=======================================>            (960 + 8) / 1272]

    [Stage 228:=======================================>            (961 + 8) / 1272][Stage 228:=======================================>            (962 + 8) / 1272]

    [Stage 228:=======================================>            (963 + 8) / 1272]

    [Stage 228:=======================================>            (964 + 8) / 1272][Stage 228:=======================================>            (965 + 8) / 1272]

    [Stage 228:=======================================>            (967 + 8) / 1272]

    [Stage 228:=======================================>            (968 + 8) / 1272]

    [Stage 228:=======================================>            (969 + 8) / 1272]

    [Stage 228:=======================================>            (970 + 8) / 1272][Stage 228:=======================================>            (971 + 8) / 1272]

    [Stage 228:=======================================>            (972 + 8) / 1272][Stage 228:=======================================>            (973 + 8) / 1272]

    [Stage 228:=======================================>            (975 + 8) / 1272]

    [Stage 228:=======================================>            (976 + 8) / 1272]

    [Stage 228:=======================================>            (978 + 8) / 1272]

    [Stage 228:========================================>           (980 + 8) / 1272][Stage 228:========================================>           (981 + 8) / 1272]

    [Stage 228:========================================>           (982 + 8) / 1272][Stage 228:========================================>           (984 + 8) / 1272]

    [Stage 228:========================================>           (986 + 8) / 1272]

    [Stage 228:========================================>           (987 + 8) / 1272][Stage 228:========================================>           (988 + 8) / 1272]

    [Stage 228:========================================>           (990 + 8) / 1272]

    [Stage 228:========================================>           (991 + 8) / 1272][Stage 228:========================================>           (992 + 8) / 1272]

    [Stage 228:========================================>           (993 + 8) / 1272][Stage 228:========================================>           (994 + 8) / 1272]

    [Stage 228:========================================>           (996 + 8) / 1272]

    [Stage 228:========================================>           (998 + 8) / 1272][Stage 228:========================================>          (1000 + 8) / 1272]

    [Stage 228:========================================>          (1001 + 8) / 1272]

    [Stage 228:========================================>          (1002 + 8) / 1272][Stage 228:========================================>          (1003 + 8) / 1272]

    [Stage 228:========================================>          (1005 + 8) / 1272][Stage 228:========================================>          (1006 + 8) / 1272]

    [Stage 228:========================================>          (1008 + 8) / 1272]

    [Stage 228:========================================>          (1009 + 8) / 1272]

    [Stage 228:========================================>          (1010 + 8) / 1272]

    [Stage 228:========================================>          (1011 + 8) / 1272]

    [Stage 228:========================================>          (1012 + 8) / 1272]

    [Stage 228:========================================>          (1013 + 8) / 1272][Stage 228:========================================>          (1014 + 8) / 1272]

    [Stage 228:========================================>          (1016 + 8) / 1272]

    [Stage 228:========================================>          (1017 + 8) / 1272][Stage 228:========================================>          (1019 + 8) / 1272]

    [Stage 228:========================================>          (1020 + 8) / 1272][Stage 228:========================================>          (1021 + 8) / 1272]

    [Stage 228:========================================>          (1022 + 8) / 1272]

    [Stage 228:=========================================>         (1024 + 8) / 1272][Stage 228:=========================================>         (1026 + 8) / 1272]

    [Stage 228:=========================================>         (1028 + 8) / 1272]

    [Stage 228:=========================================>         (1029 + 8) / 1272][Stage 228:=========================================>         (1030 + 8) / 1272]

    [Stage 228:=========================================>         (1032 + 8) / 1272][Stage 228:=========================================>         (1033 + 8) / 1272]

    [Stage 228:=========================================>         (1034 + 8) / 1272][Stage 228:=========================================>         (1035 + 8) / 1272]

    [Stage 228:=========================================>         (1036 + 8) / 1272]

    [Stage 228:=========================================>         (1037 + 8) / 1272]

    [Stage 228:=========================================>         (1039 + 8) / 1272]

    [Stage 228:=========================================>         (1040 + 8) / 1272]

    [Stage 228:=========================================>         (1041 + 8) / 1272][Stage 228:=========================================>         (1042 + 8) / 1272]

    [Stage 228:=========================================>         (1043 + 8) / 1272][Stage 228:=========================================>         (1044 + 8) / 1272]

    [Stage 228:=========================================>         (1045 + 8) / 1272][Stage 228:=========================================>         (1046 + 8) / 1272]

    [Stage 228:==========================================>        (1048 + 8) / 1272][Stage 228:==========================================>        (1049 + 8) / 1272]

    [Stage 228:==========================================>        (1050 + 8) / 1272][Stage 228:==========================================>        (1051 + 8) / 1272]

    [Stage 228:==========================================>        (1052 + 8) / 1272][Stage 228:==========================================>        (1054 + 8) / 1272]

    [Stage 228:==========================================>        (1055 + 8) / 1272][Stage 228:==========================================>        (1056 + 8) / 1272]

    [Stage 228:==========================================>        (1057 + 8) / 1272][Stage 228:==========================================>        (1058 + 8) / 1272]

    [Stage 228:==========================================>        (1060 + 8) / 1272]

    [Stage 228:==========================================>        (1061 + 8) / 1272]

    [Stage 228:==========================================>        (1063 + 8) / 1272][Stage 228:==========================================>        (1064 + 8) / 1272]

    [Stage 228:==========================================>        (1065 + 8) / 1272]

    [Stage 228:==========================================>        (1066 + 8) / 1272][Stage 228:==========================================>        (1067 + 8) / 1272]

    [Stage 228:==========================================>        (1069 + 8) / 1272]

    [Stage 228:==========================================>        (1070 + 8) / 1272][Stage 228:==========================================>        (1071 + 8) / 1272]

    [Stage 228:===========================================>       (1073 + 8) / 1272][Stage 228:===========================================>       (1074 + 8) / 1272]

    [Stage 228:===========================================>       (1075 + 8) / 1272]

    [Stage 228:===========================================>       (1076 + 8) / 1272]

    [Stage 228:===========================================>       (1077 + 8) / 1272][Stage 228:===========================================>       (1078 + 8) / 1272]

    [Stage 228:===========================================>       (1079 + 8) / 1272]

    [Stage 228:===========================================>       (1082 + 8) / 1272]

    [Stage 228:===========================================>       (1085 + 8) / 1272][Stage 228:===========================================>       (1086 + 8) / 1272]

    [Stage 228:===========================================>       (1087 + 8) / 1272][Stage 228:===========================================>       (1088 + 8) / 1272]

    [Stage 228:===========================================>       (1089 + 8) / 1272][Stage 228:===========================================>       (1090 + 8) / 1272]

    [Stage 228:===========================================>       (1091 + 8) / 1272]

    [Stage 228:===========================================>       (1092 + 8) / 1272][Stage 228:===========================================>       (1093 + 8) / 1272]

    [Stage 228:===========================================>       (1094 + 8) / 1272][Stage 228:===========================================>       (1095 + 8) / 1272]

    [Stage 228:===========================================>       (1096 + 8) / 1272]

    [Stage 228:============================================>      (1098 + 8) / 1272][Stage 228:============================================>      (1101 + 8) / 1272]

    [Stage 228:============================================>      (1102 + 8) / 1272][Stage 228:============================================>      (1104 + 8) / 1272]

    [Stage 228:============================================>      (1105 + 8) / 1272][Stage 228:============================================>      (1106 + 8) / 1272]

    [Stage 228:============================================>      (1108 + 8) / 1272][Stage 228:============================================>      (1109 + 8) / 1272]

    [Stage 228:============================================>      (1110 + 8) / 1272][Stage 228:============================================>      (1111 + 8) / 1272]

    [Stage 228:============================================>      (1113 + 8) / 1272][Stage 228:============================================>      (1114 + 8) / 1272]

    [Stage 228:============================================>      (1115 + 8) / 1272]

    [Stage 228:============================================>      (1116 + 8) / 1272]

    [Stage 228:============================================>      (1117 + 8) / 1272][Stage 228:============================================>      (1118 + 8) / 1272]

    [Stage 228:============================================>      (1119 + 8) / 1272][Stage 228:============================================>      (1121 + 8) / 1272]

    [Stage 228:============================================>      (1122 + 8) / 1272]

    [Stage 228:=============================================>     (1124 + 8) / 1272]

    [Stage 228:=============================================>     (1125 + 8) / 1272]

    [Stage 228:=============================================>     (1128 + 8) / 1272][Stage 228:=============================================>     (1131 + 8) / 1272]

    [Stage 228:=============================================>     (1135 + 8) / 1272][Stage 228:=============================================>     (1137 + 8) / 1272]

    [Stage 228:=============================================>     (1139 + 8) / 1272][Stage 228:=============================================>     (1141 + 8) / 1272]

    [Stage 228:=============================================>     (1143 + 8) / 1272][Stage 228:=============================================>     (1144 + 8) / 1272]

    [Stage 228:=============================================>     (1145 + 8) / 1272][Stage 228:=============================================>     (1146 + 8) / 1272]

    [Stage 228:==============================================>    (1148 + 8) / 1272][Stage 228:==============================================>    (1149 + 8) / 1272]

    [Stage 228:==============================================>    (1151 + 8) / 1272][Stage 228:==============================================>    (1152 + 8) / 1272]

    [Stage 228:==============================================>    (1153 + 8) / 1272]

    [Stage 228:==============================================>    (1155 + 8) / 1272][Stage 228:==============================================>    (1156 + 8) / 1272]

    [Stage 228:==============================================>    (1156 + 9) / 1272]

    [Stage 228:==============================================>    (1158 + 8) / 1272][Stage 228:==============================================>    (1159 + 8) / 1272]

    [Stage 228:==============================================>    (1161 + 8) / 1272][Stage 228:==============================================>    (1163 + 8) / 1272]

    [Stage 228:==============================================>    (1165 + 8) / 1272]

    [Stage 228:==============================================>    (1167 + 8) / 1272][Stage 228:==============================================>    (1169 + 8) / 1272]

    [Stage 228:==============================================>    (1170 + 8) / 1272][Stage 228:==============================================>    (1171 + 8) / 1272]

    [Stage 228:==============================================>    (1171 + 9) / 1272][Stage 228:==============================================>    (1172 + 8) / 1272]

    [Stage 228:===============================================>   (1175 + 8) / 1272][Stage 228:===============================================>   (1176 + 8) / 1272]

    [Stage 228:===============================================>   (1178 + 8) / 1272]

    [Stage 228:===============================================>   (1182 + 8) / 1272]

    [Stage 228:===============================================>   (1183 + 8) / 1272]

    [Stage 228:===============================================>   (1186 + 8) / 1272]

    [Stage 228:===============================================>   (1187 + 8) / 1272][Stage 228:===============================================>   (1188 + 8) / 1272]

    [Stage 228:===============================================>   (1190 + 8) / 1272][Stage 228:===============================================>   (1191 + 8) / 1272]

    [Stage 228:===============================================>   (1192 + 8) / 1272][Stage 228:===============================================>   (1193 + 8) / 1272]

    [Stage 228:================================================>  (1198 + 8) / 1272]

    [Stage 228:================================================>  (1200 + 8) / 1272][Stage 228:================================================>  (1202 + 8) / 1272]

    [Stage 228:================================================>  (1204 + 8) / 1272][Stage 228:================================================>  (1206 + 8) / 1272]

    [Stage 228:================================================>  (1207 + 8) / 1272][Stage 228:================================================>  (1208 + 8) / 1272]

    [Stage 228:================================================>  (1209 + 8) / 1272]

    [Stage 228:================================================>  (1210 + 8) / 1272]

    [Stage 228:================================================>  (1212 + 8) / 1272][Stage 228:================================================>  (1213 + 8) / 1272]

    [Stage 228:================================================>  (1214 + 8) / 1272][Stage 228:================================================>  (1215 + 8) / 1272]

    [Stage 228:================================================>  (1216 + 8) / 1272][Stage 228:================================================>  (1217 + 8) / 1272]

    [Stage 228:================================================>  (1218 + 8) / 1272]

    [Stage 228:================================================>  (1219 + 8) / 1272]

    [Stage 228:================================================>  (1220 + 8) / 1272]

    [Stage 228:================================================>  (1221 + 8) / 1272][Stage 228:=================================================> (1223 + 8) / 1272]

    [Stage 228:=================================================> (1224 + 8) / 1272][Stage 228:=================================================> (1226 + 8) / 1272]

    [Stage 228:=================================================> (1227 + 8) / 1272][Stage 228:=================================================> (1229 + 8) / 1272]

    [Stage 228:=================================================> (1230 + 8) / 1272]

    [Stage 228:=================================================> (1231 + 8) / 1272][Stage 228:=================================================> (1232 + 8) / 1272]

    [Stage 228:=================================================> (1234 + 8) / 1272]

    [Stage 228:=================================================> (1235 + 8) / 1272]

    [Stage 228:=================================================> (1236 + 8) / 1272]

    [Stage 228:=================================================> (1237 + 8) / 1272]

    [Stage 228:=================================================> (1238 + 8) / 1272][Stage 228:=================================================> (1240 + 8) / 1272]

    [Stage 228:=================================================> (1241 + 8) / 1272]

    [Stage 228:=================================================> (1243 + 8) / 1272][Stage 228:=================================================> (1244 + 8) / 1272]

    [Stage 228:=================================================> (1247 + 8) / 1272][Stage 228:==================================================>(1250 + 8) / 1272]

    [Stage 228:==================================================>(1251 + 8) / 1272]

    [Stage 228:==================================================>(1252 + 8) / 1272][Stage 228:==================================================>(1253 + 9) / 1272]

    [Stage 228:==================================================>(1255 + 8) / 1272]

    [Stage 228:==================================================>(1259 + 8) / 1272][Stage 228:==================================================>(1260 + 8) / 1272]

    [Stage 228:==================================================>(1261 + 8) / 1272]

    [Stage 228:==================================================>(1262 + 8) / 1272]

    [Stage 228:==================================================>(1263 + 8) / 1272]

    [Stage 228:==================================================>(1264 + 8) / 1272]

    [Stage 228:==================================================>(1265 + 7) / 1272]

    [Stage 228:==================================================>(1266 + 6) / 1272]

    [Stage 228:==================================================>(1267 + 5) / 1272]

    [Stage 228:==================================================>(1268 + 4) / 1272]

    [Stage 228:==================================================>(1269 + 3) / 1272]

    [Stage 228:==================================================>(1271 + 1) / 1272]

                                                                                    

    All months written to partitioned directories.
    
    Flattening 2021-02 → /home/ubuntu/project/scrambled_tlc_data/2021-02.txt


      Cleaned up partition: 2021-02
    Flattening 2024-11 → /home/ubuntu/project/scrambled_tlc_data/2024-11.txt


      Cleaned up partition: 2024-11
    Flattening 2024-06 → /home/ubuntu/project/scrambled_tlc_data/2024-06.txt


      Cleaned up partition: 2024-06
    Flattening 2024-05 → /home/ubuntu/project/scrambled_tlc_data/2024-05.txt


      Cleaned up partition: 2024-05
    Flattening 2021-12 → /home/ubuntu/project/scrambled_tlc_data/2021-12.txt


      Cleaned up partition: 2021-12
    Flattening 2023-04 → /home/ubuntu/project/scrambled_tlc_data/2023-04.txt


      Cleaned up partition: 2023-04
    Flattening 2020-08 → /home/ubuntu/project/scrambled_tlc_data/2020-08.txt


      Cleaned up partition: 2020-08
    Flattening 2022-12 → /home/ubuntu/project/scrambled_tlc_data/2022-12.txt


      Cleaned up partition: 2022-12
    Flattening 2024-12 → /home/ubuntu/project/scrambled_tlc_data/2024-12.txt


      Cleaned up partition: 2024-12
    Flattening 2020-05 → /home/ubuntu/project/scrambled_tlc_data/2020-05.txt


      Cleaned up partition: 2020-05
    Flattening 2022-08 → /home/ubuntu/project/scrambled_tlc_data/2022-08.txt


      Cleaned up partition: 2022-08
    Flattening 2020-11 → /home/ubuntu/project/scrambled_tlc_data/2020-11.txt


      Cleaned up partition: 2020-11
    Flattening 2021-01 → /home/ubuntu/project/scrambled_tlc_data/2021-01.txt


      Cleaned up partition: 2021-01
    Flattening 2020-03 → /home/ubuntu/project/scrambled_tlc_data/2020-03.txt


      Cleaned up partition: 2020-03
    Flattening 2023-05 → /home/ubuntu/project/scrambled_tlc_data/2023-05.txt


      Cleaned up partition: 2023-05
    Flattening 2022-05 → /home/ubuntu/project/scrambled_tlc_data/2022-05.txt


      Cleaned up partition: 2022-05
    Flattening 2021-06 → /home/ubuntu/project/scrambled_tlc_data/2021-06.txt


      Cleaned up partition: 2021-06
    Flattening 2023-12 → /home/ubuntu/project/scrambled_tlc_data/2023-12.txt


      Cleaned up partition: 2023-12
    Flattening 2020-07 → /home/ubuntu/project/scrambled_tlc_data/2020-07.txt


      Cleaned up partition: 2020-07
    Flattening 2023-02 → /home/ubuntu/project/scrambled_tlc_data/2023-02.txt


      Cleaned up partition: 2023-02
    Flattening 2022-09 → /home/ubuntu/project/scrambled_tlc_data/2022-09.txt


      Cleaned up partition: 2022-09
    Flattening 2024-03 → /home/ubuntu/project/scrambled_tlc_data/2024-03.txt


      Cleaned up partition: 2024-03
    Flattening 2021-05 → /home/ubuntu/project/scrambled_tlc_data/2021-05.txt


      Cleaned up partition: 2021-05
    Flattening 2022-03 → /home/ubuntu/project/scrambled_tlc_data/2022-03.txt


      Cleaned up partition: 2022-03
    Flattening 2021-04 → /home/ubuntu/project/scrambled_tlc_data/2021-04.txt


      Cleaned up partition: 2021-04
    Flattening 2024-10 → /home/ubuntu/project/scrambled_tlc_data/2024-10.txt


      Cleaned up partition: 2024-10
    Flattening 2021-10 → /home/ubuntu/project/scrambled_tlc_data/2021-10.txt


      Cleaned up partition: 2021-10
    Flattening 2021-09 → /home/ubuntu/project/scrambled_tlc_data/2021-09.txt


      Cleaned up partition: 2021-09
    Flattening 2024-09 → /home/ubuntu/project/scrambled_tlc_data/2024-09.txt


      Cleaned up partition: 2024-09
    Flattening 2023-01 → /home/ubuntu/project/scrambled_tlc_data/2023-01.txt


      Cleaned up partition: 2023-01
    Flattening 2020-01 → /home/ubuntu/project/scrambled_tlc_data/2020-01.txt


      Cleaned up partition: 2020-01
    Flattening 2024-08 → /home/ubuntu/project/scrambled_tlc_data/2024-08.txt


      Cleaned up partition: 2024-08
    Flattening 2024-07 → /home/ubuntu/project/scrambled_tlc_data/2024-07.txt


      Cleaned up partition: 2024-07
    Flattening 2023-08 → /home/ubuntu/project/scrambled_tlc_data/2023-08.txt


      Cleaned up partition: 2023-08
    Flattening 2023-03 → /home/ubuntu/project/scrambled_tlc_data/2023-03.txt


      Cleaned up partition: 2023-03
    Flattening 2021-03 → /home/ubuntu/project/scrambled_tlc_data/2021-03.txt


      Cleaned up partition: 2021-03
    Flattening 2022-01 → /home/ubuntu/project/scrambled_tlc_data/2022-01.txt


      Cleaned up partition: 2022-01
    Flattening 2022-02 → /home/ubuntu/project/scrambled_tlc_data/2022-02.txt


      Cleaned up partition: 2022-02
    Flattening 2022-07 → /home/ubuntu/project/scrambled_tlc_data/2022-07.txt


      Cleaned up partition: 2022-07
    Flattening 2023-09 → /home/ubuntu/project/scrambled_tlc_data/2023-09.txt


      Cleaned up partition: 2023-09
    Flattening 2020-09 → /home/ubuntu/project/scrambled_tlc_data/2020-09.txt


      Cleaned up partition: 2020-09
    Flattening 2020-04 → /home/ubuntu/project/scrambled_tlc_data/2020-04.txt


      Cleaned up partition: 2020-04
    Flattening 2021-08 → /home/ubuntu/project/scrambled_tlc_data/2021-08.txt


      Cleaned up partition: 2021-08
    Flattening 2020-12 → /home/ubuntu/project/scrambled_tlc_data/2020-12.txt


      Cleaned up partition: 2020-12
    Flattening 2020-06 → /home/ubuntu/project/scrambled_tlc_data/2020-06.txt


      Cleaned up partition: 2020-06
    Flattening 2023-07 → /home/ubuntu/project/scrambled_tlc_data/2023-07.txt


      Cleaned up partition: 2023-07
    Flattening 2022-10 → /home/ubuntu/project/scrambled_tlc_data/2022-10.txt


      Cleaned up partition: 2022-10
    Flattening 2020-10 → /home/ubuntu/project/scrambled_tlc_data/2020-10.txt


      Cleaned up partition: 2020-10
    Flattening 2023-06 → /home/ubuntu/project/scrambled_tlc_data/2023-06.txt


      Cleaned up partition: 2023-06
    Flattening 2020-02 → /home/ubuntu/project/scrambled_tlc_data/2020-02.txt


      Cleaned up partition: 2020-02
    Flattening 2021-11 → /home/ubuntu/project/scrambled_tlc_data/2021-11.txt


      Cleaned up partition: 2021-11
    Flattening 2024-04 → /home/ubuntu/project/scrambled_tlc_data/2024-04.txt


      Cleaned up partition: 2024-04
    Flattening 2023-11 → /home/ubuntu/project/scrambled_tlc_data/2023-11.txt


      Cleaned up partition: 2023-11
    Flattening 2022-04 → /home/ubuntu/project/scrambled_tlc_data/2022-04.txt


      Cleaned up partition: 2022-04
    Flattening 2022-11 → /home/ubuntu/project/scrambled_tlc_data/2022-11.txt


      Cleaned up partition: 2022-11
    Flattening 2024-01 → /home/ubuntu/project/scrambled_tlc_data/2024-01.txt


      Cleaned up partition: 2024-01
    Flattening 2022-06 → /home/ubuntu/project/scrambled_tlc_data/2022-06.txt


      Cleaned up partition: 2022-06
    Flattening 2021-07 → /home/ubuntu/project/scrambled_tlc_data/2021-07.txt


      Cleaned up partition: 2021-07
    Flattening 2023-10 → /home/ubuntu/project/scrambled_tlc_data/2023-10.txt


      Cleaned up partition: 2023-10
    Flattening 2024-02 → /home/ubuntu/project/scrambled_tlc_data/2024-02.txt


      Cleaned up partition: 2024-02
    
    Cleaned up all temporary files
    
    All months written as flat text files.


<!-- ## 6. Save the Full Scrambled Dataset (IN PARALLEL) -->
## 6. Organize into folders


```python
import os
import shutil
import calendar

SOURCE_DIR = "/home/ubuntu/project/scrambled_tlc_data"
DEST_DIR = "/home/ubuntu/project/scrambled_tlc_data_organized"

os.makedirs(DEST_DIR, exist_ok=True)

for filename in os.listdir(SOURCE_DIR):
    if not filename.endswith(".txt"):
        continue

    # 2016-01.txt, 2020-12.txt, etc
    try:
        year_str, month_str = filename.replace(".txt", "").split("-")
        month_name = calendar.month_name[int(month_str)]
    except Exception:
        print(f"Skipping {filename} (not in expected YYYY-MM.txt format)")
        continue

    year_dir = os.path.join(DEST_DIR, year_str)
    month_dir = os.path.join(year_dir, f"{month_str}_{month_name}")
    os.makedirs(month_dir, exist_ok=True)

    src_path = os.path.join(SOURCE_DIR, filename)
    dest_path = os.path.join(month_dir, filename)
    print(f"Moving {filename} → {dest_path}")
    shutil.move(src_path, dest_path)

print("\nReorganization complete.")
```

    Moving 2020-10.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2020/10_October/2020-10.txt
    Moving 2022-11.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2022/11_November/2022-11.txt
    Moving 2022-10.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2022/10_October/2022-10.txt
    Moving 2023-09.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2023/09_September/2023-09.txt
    Moving 2021-05.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2021/05_May/2021-05.txt
    Moving 2022-09.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2022/09_September/2022-09.txt
    Moving 2021-09.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2021/09_September/2021-09.txt
    Moving 2024-11.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2024/11_November/2024-11.txt
    Moving 2023-06.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2023/06_June/2023-06.txt
    Moving 2024-02.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2024/02_February/2024-02.txt
    Moving 2021-03.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2021/03_March/2021-03.txt
    Moving 2022-12.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2022/12_December/2022-12.txt
    Moving 2020-02.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2020/02_February/2020-02.txt
    Moving 2023-01.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2023/01_January/2023-01.txt
    Moving 2024-12.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2024/12_December/2024-12.txt
    Moving 2023-02.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2023/02_February/2023-02.txt
    Moving 2021-08.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2021/08_August/2021-08.txt
    Moving 2023-03.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2023/03_March/2023-03.txt
    Moving 2020-05.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2020/05_May/2020-05.txt
    Moving 2022-01.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2022/01_January/2022-01.txt
    Moving 2024-06.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2024/06_June/2024-06.txt
    Moving 2020-12.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2020/12_December/2020-12.txt
    Moving 2020-11.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2020/11_November/2020-11.txt
    Moving 2023-05.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2023/05_May/2023-05.txt
    Moving 2022-03.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2022/03_March/2022-03.txt
    Moving 2020-03.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2020/03_March/2020-03.txt
    Moving 2024-07.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2024/07_July/2024-07.txt
    Moving 2024-05.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2024/05_May/2024-05.txt
    Moving 2024-10.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2024/10_October/2024-10.txt
    Moving 2024-01.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2024/01_January/2024-01.txt
    Moving 2023-10.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2023/10_October/2023-10.txt
    Moving 2023-08.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2023/08_August/2023-08.txt
    Moving 2023-12.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2023/12_December/2023-12.txt
    Moving 2022-07.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2022/07_July/2022-07.txt
    Moving 2023-11.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2023/11_November/2023-11.txt
    Moving 2021-11.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2021/11_November/2021-11.txt
    Moving 2021-04.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2021/04_April/2021-04.txt
    Moving 2024-09.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2024/09_September/2024-09.txt
    Moving 2024-08.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2024/08_August/2024-08.txt
    Moving 2024-04.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2024/04_April/2024-04.txt
    Moving 2020-04.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2020/04_April/2020-04.txt
    Moving 2024-03.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2024/03_March/2024-03.txt
    Moving 2022-06.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2022/06_June/2022-06.txt
    Moving 2020-07.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2020/07_July/2020-07.txt
    Moving 2022-08.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2022/08_August/2022-08.txt
    Moving 2021-07.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2021/07_July/2021-07.txt
    Moving 2022-05.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2022/05_May/2022-05.txt
    Moving 2020-08.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2020/08_August/2020-08.txt
    Moving 2021-12.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2021/12_December/2021-12.txt
    Moving 2020-09.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2020/09_September/2020-09.txt
    Moving 2022-04.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2022/04_April/2022-04.txt
    Moving 2020-06.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2020/06_June/2020-06.txt
    Moving 2022-02.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2022/02_February/2022-02.txt
    Moving 2023-04.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2023/04_April/2023-04.txt
    Moving 2021-10.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2021/10_October/2021-10.txt
    Moving 2021-02.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2021/02_February/2021-02.txt
    Moving 2021-06.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2021/06_June/2021-06.txt
    Moving 2023-07.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2023/07_July/2023-07.txt
    Moving 2021-01.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2021/01_January/2021-01.txt
    Moving 2020-01.txt → /home/ubuntu/project/scrambled_tlc_data_organized/2020/01_January/2020-01.txt
    
    Reorganization complete.


## 7. Clean Up


```python
spark.stop()
print("\nSpark session stopped.")
```

    
    Spark session stopped.

