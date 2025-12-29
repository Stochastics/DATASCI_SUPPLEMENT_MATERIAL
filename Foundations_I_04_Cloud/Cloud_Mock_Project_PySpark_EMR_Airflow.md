# Mock PySpark Project with Airflow and AWS EMR

This document outlines a mock project that processes data from AWS S3 using a PySpark job, performs queries and basic modeling, and orchestrates the workflow on AWS EMR using Apache Airflow. The project includes a PySpark script, an Airflow DAG, and instructions for packaging and execution.

## 📌 Project Overview

- **Objective**: Process a sample dataset (customer transactions) stored in S3, perform queries (e.g., aggregations), and train a simple K-means clustering model to segment customers.
- **Components**:
  - `spark_job.py`: Loads data, performs queries, and modeling.
  - `emr_spark_dag.py`: Submits the PySpark job to EMR.
  - Packaging: Zip and upload PySpark script for EMR submission.
- **Assumptions**:
  - S3 bucket: `s3://my-mock-bucket/`
  - Input data: `s3://my-mock-bucket/input/transactions.csv`
  - Output path: `s3://my-mock-bucket/output/`
  - EMR cluster is dynamically created by Airflow.
  - Airflow is set up (e.g., MWAA or EC2) with AWS connections.

## 1️⃣ PySpark Job (`spark_job.py`)

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as sum_
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

def main():
    spark = SparkSession.builder.appName("CustomerTransactionProcessing").getOrCreate()

    input_path = "s3://my-mock-bucket/input/transactions.csv"
    output_path = "s3://my-mock-bucket/output/"

    df = spark.read.option("header", "true").csv(input_path)
    df = df.withColumn("amount", col("amount").cast("double")) \
           .withColumn("customer_id", col("customer_id").cast("string")) \
           .withColumn("transaction_date", col("transaction_date").cast("date"))

    agg_df = df.groupBy("customer_id").agg(sum_("amount").alias("total_spend"))
    agg_df.write.mode("overwrite").parquet(output_path + "aggregated/")

    assembler = VectorAssembler(inputCols=["total_spend"], outputCol="features")
    feature_df = assembler.transform(agg_df)

    kmeans = KMeans(k=3, seed=42)
    model = kmeans.fit(feature_df)
    clustered_df = model.transform(feature_df).select("customer_id", "total_spend", "prediction")

    clustered_df.write.mode("overwrite").parquet(output_path + "clustered/")
    spark.stop()

if __name__ == "__main__":
    main()
```

### 📝 Notes

- **Input format**:
  ```
  customer_id,amount,transaction_date
  C001,100.50,2023-01-01
  C002,200.75,2023-01-02
  C001,50.25,2023-01-03
  ```
- **Output**:
  - Aggregated: `s3://my-mock-bucket/output/aggregated/`
  - Clustered: `s3://my-mock-bucket/output/clustered/`

## 2️⃣ Airflow DAG (`emr_spark_dag.py`)

```python
from datetime import datetime
from airflow import DAG
from airflow.providers.amazon.aws.operators.emr import (
    EmrCreateJobFlowOperator,
    EmrAddStepsOperator,
    EmrTerminateJobFlowOperator,
)
from airflow.providers.amazon.aws.sensors.emr import EmrStepSensor

JOB_FLOW_OVERRIDES = {
    "Name": "spark-processing-cluster",
    "ReleaseLabel": "emr-6.9.0",
    "Applications": [{"Name": "Spark"}],
    "Instances": {
        "InstanceGroups": [
            {
                "Name": "Master",
                "Market": "ON_DEMAND",
                "InstanceRole": "MASTER",
                "InstanceType": "m5.xlarge",
                "InstanceCount": 1,
            },
            {
                "Name": "Core",
                "Market": "ON_DEMAND",
                "InstanceRole": "CORE",
                "InstanceType": "m5.xlarge",
                "InstanceCount": 2,
            },
        ],
        "KeepJobFlowAliveWhenNoSteps": True,
        "TerminationProtected": False,
    },
    "JobFlowRole": "EMR_EC2_DefaultRole",
    "ServiceRole": "EMR_DefaultRole",
}

SPARK_STEP = [
    {
        "Name": "Run Spark Job",
        "ActionOnFailure": "TERMINATE_CLUSTER",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": [
                "spark-submit",
                "--deploy-mode", "cluster",
                "s3://my-mock-bucket/scripts/spark_job.py",
            ],
        },
    }
]

with DAG(
    dag_id="emr_spark_processing",
    start_date=datetime(2025, 5, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    create_cluster = EmrCreateJobFlowOperator(
        task_id="create_emr_cluster",
        job_flow_overrides=JOB_FLOW_OVERRIDES,
        aws_conn_id="aws_default",
    )

    add_step = EmrAddStepsOperator(
        task_id="add_spark_step",
        job_flow_id="{{ task_instance.xcom_pull('create_emr_cluster')['JobFlowId'] }}",
        steps=SPARK_STEP,
        aws_conn_id="aws_default",
    )

    monitor_step = EmrStepSensor(
        task_id="monitor_spark_step",
        job_flow_id="{{ task_instance.xcom_pull('create_emr_cluster')['JobFlowId'] }}",
        step_id="{{ task_instance.xcom_pull('add_spark_step')[0] }}",
        aws_conn_id="aws_default",
    )

    terminate_cluster = EmrTerminateJobFlowOperator(
        task_id="terminate_emr_cluster",
        job_flow_id="{{ task_instance.xcom_pull('create_emr_cluster')['JobFlowId'] }}",
        aws_conn_id="aws_default",
    )

    create_cluster >> add_step >> monitor_step >> terminate_cluster
```

## 3️⃣ Packaging & Setup

### Upload files to S3:
```bash
aws s3 cp spark_job.py s3://my-mock-bucket/scripts/
aws s3 cp transactions.csv s3://my-mock-bucket/input/
```

### Airflow Setup:
- Place `emr_spark_dag.py` in `~/airflow/dags/`
- Install AWS provider:
  ```bash
  pip install apache-airflow-providers-amazon
  ```
- Configure AWS connection (`aws_default`) in Airflow UI

## 4️⃣ Execution Flow

1. Airflow creates an EMR cluster
2. Submits PySpark job
3. Monitors completion
4. Terminates cluster

## 📁 Sample Structure

```
mock_project/
├── spark_job.py
├── emr_spark_dag.py
└── input/
    └── transactions.csv
```

**S3 Layout:**

```
s3://my-mock-bucket/
├── input/transactions.csv
├── scripts/spark_job.py
└── output/
    ├── aggregated/
    └── clustered/
```

## ✅ Next Steps

- Replace `my-mock-bucket` with your actual bucket.
- Test PySpark job locally:
  ```bash
  spark-submit spark_job.py
  ```
- Monitor Airflow and EMR logs.
- Extend project with additional models or Airflow tasks.
