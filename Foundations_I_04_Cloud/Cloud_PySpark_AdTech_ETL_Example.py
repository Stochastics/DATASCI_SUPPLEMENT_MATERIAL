from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType


# === Step 0: Initialize SparkSession (auto-includes SparkContext) ===
spark = SparkSession.builder \
    .appName("Ad_Impression_ETL_Cloud") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# === STEP 1: Load impression logs from cloud (S3) using wildcard paths ===
# Assume data is bucketed by hour or region — wildcards allow flexible input
# Example: impressions/2025/04/01/hour=*/ OR impressions/2025/04/01/region=US*
impressions = spark.read.parquet("s3a://ad-company-data/impressions/2025/04/01/*")

# Sample schema:
# impression_id | timestamp           | ip_address   | geo_code | publisher_id | user_agent                              | bid_price | supply_vendor | ssp_id
# --------------|---------------------|--------------|----------|--------------|------------------------------------------|-----------|----------------|--------
# abc123        | 2025-04-01 12:05:01 | 12.34.56.78  | US-NY    | pub001       | Mozilla/5.0 (iPhone; CPU iPhone OS 15...)| 0.54      | Magnite        | 132
# def456        | 2025-04-01 12:07:44 | 98.76.54.32  | US-CA    | pub002       | Mozilla/5.0 (Windows NT 10.0; Win64...)  | 0.61      | IndexExchange  | 845

# === STEP 2: Repartition for parallelism on EMR cluster ===
# Repartitioning prepares the data for wide operations like joins and aggregations
impressions = impressions.repartition(100, col("geo_code"))

# Why 100?
# Assume EMR cluster has 8 nodes × 16 vCPUs = 128 total slots
# 100 partitions ensures even work distribution without over-fragmentation

# === STEP 3: Broadcast join for metadata enrichment ===
# Broadcast joins avoid expensive shuffles by replicating small lookup tables to all workers

geo_lookup = spark.read.csv("s3a://ad-company-meta/geo.csv", header=True, inferSchema=True)
# geo_lookup sample: geo_code | country | region
#                    US-NY    | US      | Northeast

publisher_meta = spark.read.parquet("s3a://ad-company-meta/publisher_metadata.parquet")
# publisher_meta sample: publisher_id | publisher_name   | vertical
#                        pub001       | SportsDaily      | Sports
#                        pub002       | NewsNow          | News

from pyspark.sql.functions import broadcast

# Safe to broadcast since lookup tables are small (<10MB each)
# Puts the smaller table in each worker node for faster joining 
enriched = impressions \
    .join(broadcast(geo_lookup), on="geo_code", how="left") \
    .join(broadcast(publisher_meta), on="publisher_id", how="left")

# operations are lazy(not executed) if you don' want to re-run the logic can do like
enriched.cache()
# but need to actually do a .show(), or .collect or .write to execute the "pipeline" above


# === STEP 4: UDF — parse user agent → device type ===
# UDF used here since there's no built-in device parser in PySpark

@udf(returnType=StringType())
def parse_device_type(ua):
    if not ua:
        return "unknown"
    ua = ua.lower()
    if "mobile" in ua:
        return "mobile"
    elif "tablet" in ua:
        return "tablet"
    elif "mac" in ua or "windows" in ua:
        return "desktop"
    return "other"

# Apply the transformation
enriched = enriched.withColumn("device_type", parse_device_type(col("user_agent")))

# === STEP 5: Write out to cloud partitioned by `country` and `device_type` for downstream speed ===
# Partitioning enables efficient filtering in future analytics or ML queries

enriched.select(
    "impression_id", "timestamp", "device_type", "country", "region",
    "publisher_name", "bid_price", "supply_vendor", "ssp_id"
).write.mode("overwrite") \
 .partitionBy("country", "device_type") \
 .parquet("s3a://ad-company-processed/impressions/2025/04/01/")

s3://ad-company-processed/impressions/2025/04/01/
└── country=US/
    └── device_type=mobile/
        └── part-00000-....snappy.parquet
    └── device_type=desktop/
        └── part-00001-....snappy.parquet
└── country=CA/
    └── device_type=mobile/
        └── part-00002-....snappy.parquet

spark.stop()



##############################################################################################
##############################################################################################

import pandas as pd
import numpy as np
import os
from faker import Faker
import uuid
import pyarrow.parquet as pq
import pyarrow as pa

faker = Faker()
output_base = "/tmp/impressions_synthetic"

os.makedirs(output_base, exist_ok=True)

def generate_batch(n_rows=1_000_000, part_id=0):
    df = pd.DataFrame({
        "impression_id": [str(uuid.uuid4()) for _ in range(n_rows)],
        "timestamp": pd.date_range("2025-04-01", periods=n_rows, freq='S'),
        "ip_address": [faker.ipv4() for _ in range(n_rows)],
        "geo_code": np.random.choice(["US-NY", "US-CA", "US-TX", "CA-ON", "UK-LON"], size=n_rows),
        "publisher_id": np.random.choice(["pub001", "pub002", "pub003"], size=n_rows),
        "user_agent": np.random.choice([
            "Mozilla/5.0 (iPhone; CPU iPhone OS 15_2)", 
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", 
            "Mozilla/5.0 (Linux; Android 12)",
            "Mozilla/5.0 (iPad; CPU OS 15_1)"
        ], size=n_rows),
        "bid_price": np.round(np.random.uniform(0.01, 1.25, n_rows), 2),
        "supply_vendor": np.random.choice(["Magnite", "IndexExchange", "OpenX"], size=n_rows),
        "ssp_id": np.random.randint(100, 999, size=n_rows)
    })

    table = pa.Table.from_pandas(df)
    output_path = f"{output_base}/hour={part_id:02d}"
    os.makedirs(output_path, exist_ok=True)
    pq.write_table(table, f"{output_path}/impressions.parquet")

# Generate ~100 million rows = ~100 GB (estimate ~1GB per million rows in Parquet)
for i in range(100):  # You can go 500+ if you want 500 GB
    generate_batch(n_rows=1_000_000, part_id=i)
