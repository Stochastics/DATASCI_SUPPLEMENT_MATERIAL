# Simple ML Pipeline with Airflow - Complete Guide

Everything you need in one file. Copy-paste each section to build the complete pipeline.

## Overview

A minimal ML pipeline showing:
- **Airflow** for orchestration
- **Docker Compose** for containerization
- **Simple ML** (RandomForest on synthetic data)
- **Task dependencies** and parallel execution
- **XCom** for data passing between tasks

**Pipeline Flow:**
```
generate_data
     │
     ├─────────────────┐
     │                 │
     ▼                 ▼
train_model    compute_statistics
     │                 │
     └────────┬────────┘
              ▼
        save_results
```

---

## Project Structure

Create this directory structure:
```bash
mkdir -p simple-ml-pipeline/{dags,logs,plugins,config}
cd simple-ml-pipeline
```

---

## File 1: `docker-compose.yml`

The core Airflow setup with Postgres, webserver, and scheduler.

```yaml
version: '3'

x-airflow-common:
  &airflow-common
  image: apache/airflow:2.7.3-python3.10
  environment:
    &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CORE__FERNET_KEY: ''
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    AIRFLOW__API__AUTH_BACKENDS: 'airflow.api.auth.backend.basic_auth'
    AIRFLOW__WEBSERVER__EXPOSE_CONFIG: 'true'
  volumes:
    - ./dags:/opt/airflow/dags
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins
    - ./requirements.txt:/requirements.txt
  user: "${AIRFLOW_UID:-50000}:0"
  depends_on:
    postgres:
      condition: service_healthy

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 5s
      retries: 5
    restart: always

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - 8080:8080
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 10s
      timeout: 10s
      retries: 5
    restart: always
    depends_on:
      airflow-init:
        condition: service_completed_successfully

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    healthcheck:
      test: ["CMD-SHELL", 'airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}"']
      interval: 10s
      timeout: 10s
      retries: 5
    restart: always
    depends_on:
      airflow-init:
        condition: service_completed_successfully

  airflow-init:
    <<: *airflow-common
    entrypoint: /bin/bash
    command:
      - -c
      - |
        mkdir -p /sources/logs /sources/dags /sources/plugins
        chown -R "${AIRFLOW_UID:-50000}:0" /sources/{logs,dags,plugins}
        exec /entrypoint airflow version
    environment:
      <<: *airflow-common-env
      _AIRFLOW_DB_UPGRADE: 'true'
      _AIRFLOW_WWW_USER_CREATE: 'true'
      _AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-airflow}
      _AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-airflow}
    user: "0:0"
    volumes:
      - .:/sources

volumes:
  postgres-db-volume:
```

---

## File 2: `requirements.txt`

Python dependencies for ML tasks.

```txt
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
boto3==1.28.25
```

---

## File 3: `.env`

Environment variables for Airflow.

```bash
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow
```

---

## File 4: `dags/ml_pipeline.py`

The main DAG with 4 tasks: generate → train/stats → save.

```python
"""
Simple ML Pipeline DAG
Generates data -> Trains model -> Evaluates -> Saves results
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import logging

logger = logging.getLogger(__name__)

# Default args
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def generate_data(**context):
    """Generate synthetic ML data"""
    logger.info("Generating synthetic data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Features: age, income, hours_online
    X = np.random.randn(n_samples, 3)
    X[:, 0] = X[:, 0] * 10 + 40  # age
    X[:, 1] = X[:, 1] * 20000 + 60000  # income
    X[:, 2] = np.abs(X[:, 2] * 5 + 10)  # hours online
    
    # Target: customer segment (0, 1, 2)
    y = ((X[:, 0] > 45).astype(int) + 
         (X[:, 1] > 70000).astype(int))
    
    df = pd.DataFrame(X, columns=['age', 'income', 'hours_online'])
    df['segment'] = y
    
    # Save to temporary location
    output_path = f"/tmp/ml_data_{context['ds']}.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"Generated {len(df)} records")
    logger.info(f"Saved to {output_path}")
    
    # Push to XCom
    context['task_instance'].xcom_push(key='data_path', value=output_path)
    context['task_instance'].xcom_push(key='n_records', value=len(df))
    
    return output_path


def train_model(**context):
    """Train ML model"""
    logger.info("Training model...")
    
    # Get data path from previous task
    data_path = context['task_instance'].xcom_pull(
        task_ids='generate_data', 
        key='data_path'
    )
    
    # Load data
    df = pd.read_csv(data_path)
    X = df[['age', 'income', 'hours_online']]
    y = df['segment']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.3f}")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'date': context['ds']
    }
    
    results_path = f"/tmp/ml_results_{context['ds']}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Push to XCom
    context['task_instance'].xcom_push(key='accuracy', value=accuracy)
    context['task_instance'].xcom_push(key='results_path', value=results_path)
    
    return results_path


def compute_statistics(**context):
    """Compute data statistics"""
    logger.info("Computing statistics...")
    
    # Get data path
    data_path = context['task_instance'].xcom_pull(
        task_ids='generate_data', 
        key='data_path'
    )
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Compute stats
    stats = {
        'total_records': len(df),
        'segment_distribution': df['segment'].value_counts().to_dict(),
        'avg_age': float(df['age'].mean()),
        'avg_income': float(df['income'].mean()),
        'avg_hours': float(df['hours_online'].mean()),
        'date': context['ds']
    }
    
    logger.info(f"Stats: {stats}")
    
    stats_path = f"/tmp/ml_stats_{context['ds']}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats_path


def save_results(**context):
    """Save final results (could upload to S3 here)"""
    logger.info("Saving results...")
    
    # Get all previous results
    results_path = context['task_instance'].xcom_pull(
        task_ids='train_model', 
        key='results_path'
    )
    stats_path = context['task_instance'].xcom_pull(
        task_ids='compute_statistics'
    )
    
    logger.info(f"✅ Pipeline complete for {context['ds']}")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Stats saved to: {stats_path}")
    
    # In production, you'd upload to S3 here:
    # import boto3
    # s3 = boto3.client('s3')
    # s3.upload_file(results_path, 'my-bucket', f'results/date={context["ds"]}/results.json')
    
    return "Success"


# Create DAG
with DAG(
    'simple_ml_pipeline',
    default_args=default_args,
    description='Simple ML pipeline with data generation and training',
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'example'],
) as dag:
    
    # Task 1: Generate data
    generate_data_task = PythonOperator(
        task_id='generate_data',
        python_callable=generate_data,
    )
    
    # Task 2: Train model
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )
    
    # Task 3: Compute statistics
    compute_stats_task = PythonOperator(
        task_id='compute_statistics',
        python_callable=compute_statistics,
    )
    
    # Task 4: Save results
    save_results_task = PythonOperator(
        task_id='save_results',
        python_callable=save_results,
    )
    
    # Define dependencies
    generate_data_task >> [train_model_task, compute_stats_task] >> save_results_task
```

---

## File 5: `config/pipeline_config.yaml`

Configuration for pipeline parameters.

```yaml
# ML Pipeline Configuration

pipeline:
  name: "simple_ml_pipeline"
  schedule: "@daily"
  
data:
  n_samples: 1000
  random_seed: 42
  
model:
  type: "RandomForestClassifier"
  n_estimators: 10
  random_state: 42
  test_size: 0.2

output:
  # Local output (development)
  local_path: "/tmp"
  
  # S3 output (production)
  # s3_bucket: "my-ml-bucket"
  # s3_prefix: "ml-results"
```

---

## File 6: `Makefile`

Convenience commands for managing Airflow.

```makefile
.PHONY: help init up down logs restart clean

help:
	@echo "Simple ML Pipeline - Airflow"
	@echo ""
	@echo "Commands:"
	@echo "  make init      - Initialize Airflow (first time)"
	@echo "  make up        - Start Airflow services"
	@echo "  make down      - Stop Airflow services"
	@echo "  make logs      - View logs"
	@echo "  make restart   - Restart services"
	@echo "  make clean     - Remove all containers and volumes"
	@echo "  make install   - Install Python dependencies"

init:
	@echo "Initializing Airflow..."
	docker-compose up airflow-init

up:
	@echo "Starting Airflow..."
	docker-compose up -d
	@echo ""
	@echo "✅ Airflow running at http://localhost:8080"
	@echo "   Username: airflow"
	@echo "   Password: airflow"

down:
	@echo "Stopping Airflow..."
	docker-compose down

logs:
	docker-compose logs -f

restart: down up

clean:
	@echo "Removing all containers and volumes..."
	docker-compose down -v

install:
	@echo "Installing Python dependencies..."
	docker-compose exec airflow-scheduler pip install -r /requirements.txt
```

---

## Setup Instructions

### 1. Create Project Structure

```bash
mkdir -p simple-ml-pipeline/{dags,logs,plugins,config}
cd simple-ml-pipeline
```

### 2. Copy All Files

Create each file above in the appropriate location:
- `docker-compose.yml` → root
- `requirements.txt` → root
- `.env` → root
- `Makefile` → root
- `dags/ml_pipeline.py` → dags/
- `config/pipeline_config.yaml` → config/

### 3. Start Airflow

```bash
# Initialize (first time only)
make init

# Start services
make up

# Install ML dependencies
make install
```

### 4. Access Airflow UI

Open http://localhost:8080

- **Username:** `airflow`
- **Password:** `airflow`

### 5. Run the Pipeline

1. Find `simple_ml_pipeline` in the DAGs list
2. Toggle it **ON** (unpause)
3. Click **"Trigger DAG"** to run manually

---

## Usage

### View Logs

```bash
# All services
make logs

# Just scheduler
docker-compose logs -f airflow-scheduler

# Or in UI: DAG → Graph → Click task → Log tab
```

### Check Results

```bash
# Results are saved to /tmp inside the container
docker-compose exec airflow-scheduler ls -la /tmp/ml_*

# View results
docker-compose exec airflow-scheduler cat /tmp/ml_results_2024-01-01.json
docker-compose exec airflow-scheduler cat /tmp/ml_stats_2024-01-01.json
```

### Stop Airflow

```bash
make down
```

### Clean Everything

```bash
make clean  # Removes all containers and volumes
```

---

## Add S3 Upload

To upload results to S3, modify the `save_results()` function in `dags/ml_pipeline.py`:

```python
def save_results(**context):
    """Save final results"""
    import boto3
    
    logger.info("Uploading to S3...")
    
    results_path = context['task_instance'].xcom_pull(
        task_ids='train_model', 
        key='results_path'
    )
    stats_path = context['task_instance'].xcom_pull(
        task_ids='compute_statistics'
    )
    
    # Upload to S3
    s3 = boto3.client('s3')
    date = context['ds']
    
    s3.upload_file(
        results_path,
        'my-ml-bucket',
        f'ml-results/date={date}/results.json'
    )
    
    s3.upload_file(
        stats_path,
        'my-ml-bucket', 
        f'ml-results/date={date}/stats.json'
    )
    
    logger.info(f"✅ Uploaded to S3: s3://my-ml-bucket/ml-results/date={date}/")
    return "Success"
```

---

## Extending the Pipeline

### Add New Task

```python
def feature_engineering(**context):
    """Create new features"""
    data_path = context['task_instance'].xcom_pull(
        task_ids='generate_data',
        key='data_path'
    )
    
    df = pd.read_csv(data_path)
    
    # Add new features
    df['age_income_ratio'] = df['age'] / df['income']
    df['hours_per_dollar'] = df['hours_online'] / df['income']
    
    # Save
    output_path = f"/tmp/ml_data_engineered_{context['ds']}.csv"
    df.to_csv(output_path, index=False)
    
    context['task_instance'].xcom_push(key='engineered_path', value=output_path)
    return output_path

# Add to DAG
feature_eng_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
)

# Update dependencies
generate_data_task >> feature_eng_task >> [train_model_task, compute_stats_task]
```

### Add Email Alerts

```python
default_args = {
    'owner': 'mlops',
    'email': ['your-email@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    # ... rest of args
}
```

### Process Date Range

Run for multiple dates:

```python
from airflow.operators.bash import BashOperator

backfill_task = BashOperator(
    task_id='backfill',
    bash_command='airflow dags backfill simple_ml_pipeline -s 2024-01-01 -e 2024-01-31',
)
```

---

## Production Deployment

For production, you'd want to:

1. **Use CeleryExecutor or KubernetesExecutor**
   ```yaml
   AIRFLOW__CORE__EXECUTOR: CeleryExecutor
   ```

2. **Use RDS for Postgres**
   ```yaml
   AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql://user:pass@rds-endpoint/airflow
   ```

3. **Store DAGs in S3**
   - Sync DAGs to workers
   - Use Git-sync sidecar

4. **Add Monitoring**
   - CloudWatch
   - Datadog
   - Prometheus + Grafana

5. **Use Secrets Manager**
   ```python
   from airflow.hooks.base import BaseHook
   conn = BaseHook.get_connection('my_aws_conn')
   ```

6. **Deploy to ECS/EKS**
   - Fargate for webserver/scheduler
   - Auto-scaling worker pools

---

## Troubleshooting

### DAG not showing up
```bash
# Check DAG file
docker-compose exec airflow-scheduler python /opt/airflow/dags/ml_pipeline.py

# Check scheduler logs
docker-compose logs airflow-scheduler | grep ml_pipeline

# Restart scheduler
docker-compose restart airflow-scheduler
```

### Dependencies not installed
```bash
make install
# or
docker-compose exec airflow-scheduler pip install pandas numpy scikit-learn
```

### Port 8080 in use
```yaml
# In docker-compose.yml, change:
ports:
  - 8081:8080  # Use 8081 instead
```

### Permission issues
```bash
# Set correct UID in .env
echo "AIRFLOW_UID=$(id -u)" > .env
make down
make init
make up
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Compose                          │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Postgres   │  │   Airflow    │  │   Airflow    │      │
│  │   (Metadata) │  │  Webserver   │  │  Scheduler   │      │
│  │              │  │              │  │              │      │
│  │  Port: 5432  │  │  Port: 8080  │  │  Runs tasks  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
│                     ┌──────┴──────┐                         │
│                     │  DAGs + Logs │                         │
│                     │   (Volumes)  │                         │
│                     └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘

Task Flow:
generate_data (1s)
     │
     ├──────────────────┐
     │                  │
     ▼                  ▼
train_model (2s)  compute_stats (1s)
     │                  │
     └────────┬─────────┘
              ▼
        save_results (<1s)
```

---

## Summary

You now have a complete, minimal ML pipeline with:
- ✅ Airflow orchestration
- ✅ Docker containerization
- ✅ Task dependencies
- ✅ Parallel execution
- ✅ Data passing with XCom
- ✅ Simple ML (RandomForest)
- ✅ Config-driven setup
- ✅ Production-ready structure

**Just like your multi-domain-nlp-pipeline but simpler!**

Start with `make init && make up` and you're ready to go.
