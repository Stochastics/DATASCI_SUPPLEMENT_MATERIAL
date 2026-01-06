# Part 2: Productionizing the D&O Insurance Claims Model

Now that we understand the statistical foundations from Part 1, let's transform our analysis into a **production-ready ML pipeline**. This guide covers containerization, orchestration, modularization, and deployment patterns used in real MLOps workflows.

---

## 📁 Project Structure

A well-organized codebase is the foundation of maintainable ML systems:

```
insurance-claims-model/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Data ingestion
│   │   └── preprocessor.py    # CPI adjustment, cleaning
│   ├── models/
│   │   ├── __init__.py
│   │   ├── distributions.py   # Lognormal, Gamma, Weibull fitting
│   │   ├── bayesian.py        # ABC posterior estimation
│   │   └── evaluation.py      # KS tests, Q-Q plots
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── train.py           # Training orchestration
│   │   └── predict.py         # Inference logic
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py
│       └── s3_helpers.py
├── cli/
│   └── main.py                # Click CLI entrypoint
├── dags/
│   └── insurance_pipeline_dag.py
├── tests/
│   ├── test_distributions.py
│   ├── test_preprocessor.py
│   └── conftest.py
├── config/
│   ├── base.yaml
│   ├── dev.yaml
│   └── prod.yaml
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🔧 Configuration Management

Use YAML configs with environment-specific overrides. This separates code from configuration and makes deployments predictable.

### `config/base.yaml`

```yaml
# Base configuration - inherited by all environments
project:
  name: "do-insurance-claims"
  version: "1.0.0"

data:
  raw_path: "data/raw/claims.csv"
  processed_path: "data/processed/"
  cpi_adjustment_base_year: 2023

model:
  distributions:
    - lognormal
    - gamma
    - weibull
  claim_probability: 0.25
  bayesian:
    n_samples: 10000
    acceptance_threshold: 0.05

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### `config/prod.yaml`

```yaml
# Production overrides
data:
  raw_path: "s3://insurance-data-prod/raw/claims.csv"
  processed_path: "s3://insurance-data-prod/processed/"

aws:
  region: "us-east-1"
  sagemaker_role: "arn:aws:iam::123456789:role/SageMakerExecutionRole"

logging:
  level: "WARNING"
```

### `src/config.py`

```python
"""
Configuration loader with environment-specific overrides.
"""
from pathlib import Path
from typing import Any
import yaml
from functools import lru_cache


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache(maxsize=1)
def load_config(environment: str = "dev") -> dict[str, Any]:
    """
    Load configuration with environment-specific overrides.
    
    Args:
        environment: One of 'dev', 'staging', 'prod'
        
    Returns:
        Merged configuration dictionary
    """
    config_dir = Path(__file__).parent.parent / "config"
    
    # Load base config
    with open(config_dir / "base.yaml") as f:
        config = yaml.safe_load(f)
    
    # Apply environment overrides
    env_file = config_dir / f"{environment}.yaml"
    if env_file.exists():
        with open(env_file) as f:
            overrides = yaml.safe_load(f) or {}
            config = deep_merge(config, overrides)
    
    return config


class Config:
    """Typed configuration access."""
    
    def __init__(self, environment: str = "dev"):
        self._config = load_config(environment)
    
    @property
    def data_raw_path(self) -> str:
        return self._config["data"]["raw_path"]
    
    @property
    def distributions(self) -> list[str]:
        return self._config["model"]["distributions"]
    
    @property
    def claim_probability(self) -> float:
        return self._config["model"]["claim_probability"]
    
    @property
    def bayesian_samples(self) -> int:
        return self._config["model"]["bayesian"]["n_samples"]
```

---

## 📦 Modularized Code

### `src/data/preprocessor.py`

```python
"""
Data preprocessing with CPI adjustment and validation.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CPIAdjuster:
    """Adjust historical claims for inflation."""
    
    base_year: int = 2023
    cpi_table: Optional[dict[int, float]] = None
    
    def __post_init__(self):
        # Default CPI values (would load from external source in prod)
        if self.cpi_table is None:
            self.cpi_table = {
                2018: 251.1,
                2019: 255.7,
                2020: 258.8,
                2021: 271.0,
                2022: 292.7,
                2023: 304.7,
                2024: 314.5,
            }
    
    def adjust(self, amount: float, year: int) -> float:
        """Adjust a dollar amount from `year` to base_year dollars."""
        if year not in self.cpi_table or self.base_year not in self.cpi_table:
            logger.warning(f"CPI data missing for year {year}, returning unadjusted")
            return amount
        
        adjustment_factor = self.cpi_table[self.base_year] / self.cpi_table[year]
        return amount * adjustment_factor


class ClaimsPreprocessor:
    """Preprocess raw claims data for modeling."""
    
    def __init__(self, cpi_adjuster: CPIAdjuster):
        self.cpi_adjuster = cpi_adjuster
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize claims data.
        
        Args:
            df: Raw claims DataFrame with columns [claim_id, year, amount, ...]
            
        Returns:
            Processed DataFrame with adjusted amounts
        """
        logger.info(f"Processing {len(df)} raw claims")
        
        # Validate required columns
        required = {"claim_id", "year", "amount"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Remove invalid records
        df = df.dropna(subset=["amount"])
        df = df[df["amount"] > 0]
        
        # Apply CPI adjustment
        df = df.copy()
        df["amount_adjusted"] = df.apply(
            lambda row: self.cpi_adjuster.adjust(row["amount"], row["year"]),
            axis=1
        )
        
        logger.info(f"Processed {len(df)} valid claims")
        return df
```

### `src/models/distributions.py`

```python
"""
Distribution fitting and evaluation for claim severities.
"""
from dataclasses import dataclass
from typing import Protocol, Tuple
import numpy as np
from scipy import stats
from scipy.stats import kstest
import logging

logger = logging.getLogger(__name__)


class DistributionFitter(Protocol):
    """Protocol for distribution fitters."""
    
    name: str
    
    def fit(self, data: np.ndarray) -> Tuple[float, ...]:
        """Fit distribution to data, return parameters."""
        ...
    
    def expected_value(self, params: Tuple[float, ...]) -> float:
        """Calculate expected value given parameters."""
        ...
    
    def ks_test(self, data: np.ndarray, params: Tuple[float, ...]) -> Tuple[float, float]:
        """Perform KS test, return (statistic, p-value)."""
        ...


@dataclass
class LognormalFitter:
    """Lognormal distribution fitting."""
    
    name: str = "lognormal"
    
    def fit(self, data: np.ndarray) -> Tuple[float, float, float]:
        """Fit lognormal, return (shape, loc, scale)."""
        shape, loc, scale = stats.lognorm.fit(data, floc=0)
        logger.info(f"Lognormal fit: shape={shape:.4f}, scale={scale:.4f}")
        return (shape, loc, scale)
    
    def expected_value(self, params: Tuple[float, float, float]) -> float:
        shape, loc, scale = params
        return stats.lognorm.mean(shape, loc=loc, scale=scale)
    
    def ks_test(self, data: np.ndarray, params: Tuple[float, float, float]) -> Tuple[float, float]:
        shape, loc, scale = params
        stat, pval = kstest(data, lambda x: stats.lognorm.cdf(x, shape, loc=loc, scale=scale))
        return (stat, pval)


@dataclass
class GammaFitter:
    """Gamma distribution fitting."""
    
    name: str = "gamma"
    
    def fit(self, data: np.ndarray) -> Tuple[float, float, float]:
        """Fit gamma, return (shape, loc, scale)."""
        shape, loc, scale = stats.gamma.fit(data, floc=0)
        logger.info(f"Gamma fit: shape={shape:.4f}, scale={scale:.4f}")
        return (shape, loc, scale)
    
    def expected_value(self, params: Tuple[float, float, float]) -> float:
        shape, loc, scale = params
        return stats.gamma.mean(shape, loc=loc, scale=scale)
    
    def ks_test(self, data: np.ndarray, params: Tuple[float, float, float]) -> Tuple[float, float]:
        shape, loc, scale = params
        stat, pval = kstest(data, lambda x: stats.gamma.cdf(x, shape, loc=loc, scale=scale))
        return (stat, pval)


@dataclass  
class WeibullFitter:
    """Weibull distribution fitting."""
    
    name: str = "weibull"
    
    def fit(self, data: np.ndarray) -> Tuple[float, float, float]:
        """Fit Weibull min, return (c, loc, scale)."""
        c, loc, scale = stats.weibull_min.fit(data, floc=0)
        logger.info(f"Weibull fit: c={c:.4f}, scale={scale:.4f}")
        return (c, loc, scale)
    
    def expected_value(self, params: Tuple[float, float, float]) -> float:
        c, loc, scale = params
        return stats.weibull_min.mean(c, loc=loc, scale=scale)
    
    def ks_test(self, data: np.ndarray, params: Tuple[float, float, float]) -> Tuple[float, float]:
        c, loc, scale = params
        stat, pval = kstest(data, lambda x: stats.weibull_min.cdf(x, c, loc=loc, scale=scale))
        return (stat, pval)


# Factory pattern for distribution selection
DISTRIBUTION_REGISTRY: dict[str, type] = {
    "lognormal": LognormalFitter,
    "gamma": GammaFitter,
    "weibull": WeibullFitter,
}


def get_fitter(name: str) -> DistributionFitter:
    """Get a distribution fitter by name."""
    if name not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {name}. Available: {list(DISTRIBUTION_REGISTRY.keys())}")
    return DISTRIBUTION_REGISTRY[name]()
```

### `src/pipeline/train.py`

```python
"""
Training pipeline orchestration.
"""
from dataclasses import dataclass, field
from typing import Any
import json
from pathlib import Path
import numpy as np
import pandas as pd
import logging

from src.config import Config
from src.data.preprocessor import ClaimsPreprocessor, CPIAdjuster
from src.models.distributions import get_fitter, DistributionFitter

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifact:
    """Container for trained model artifacts."""
    
    distribution_name: str
    parameters: tuple
    expected_loss: float
    ks_statistic: float
    ks_pvalue: float
    claim_probability: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": self.distribution_name,
            "parameters": list(self.parameters),
            "expected_loss": self.expected_loss,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "claim_probability": self.claim_probability,
        }
    
    def save(self, path: Path) -> None:
        """Serialize artifact to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved model artifact to {path}")


@dataclass
class TrainingPipeline:
    """End-to-end training pipeline."""
    
    config: Config
    preprocessor: ClaimsPreprocessor = field(init=False)
    
    def __post_init__(self):
        cpi_adjuster = CPIAdjuster(base_year=2023)
        self.preprocessor = ClaimsPreprocessor(cpi_adjuster)
    
    def run(self, raw_data: pd.DataFrame, output_dir: Path) -> list[ModelArtifact]:
        """
        Execute full training pipeline.
        
        Args:
            raw_data: Raw claims DataFrame
            output_dir: Directory for model artifacts
            
        Returns:
            List of fitted model artifacts
        """
        logger.info("Starting training pipeline")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Preprocess
        processed = self.preprocessor.process(raw_data)
        severities = processed["amount_adjusted"].values
        
        # Step 2: Fit distributions
        artifacts = []
        for dist_name in self.config.distributions:
            logger.info(f"Fitting {dist_name} distribution")
            
            fitter = get_fitter(dist_name)
            params = fitter.fit(severities)
            
            # Calculate metrics
            expected_severity = fitter.expected_value(params)
            expected_loss = self.config.claim_probability * expected_severity
            ks_stat, ks_pval = fitter.ks_test(severities, params)
            
            artifact = ModelArtifact(
                distribution_name=dist_name,
                parameters=params,
                expected_loss=expected_loss,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                claim_probability=self.config.claim_probability,
            )
            
            # Save artifact
            artifact.save(output_dir / f"{dist_name}_model.json")
            artifacts.append(artifact)
        
        # Step 3: Select best model (highest p-value)
        best = max(artifacts, key=lambda a: a.ks_pvalue)
        logger.info(f"Best fit: {best.distribution_name} (p={best.ks_pvalue:.4f})")
        
        # Save summary
        summary = {
            "best_model": best.distribution_name,
            "models": [a.to_dict() for a in artifacts],
        }
        with open(output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        return artifacts
```

---

## 🖥️ CLI Interface with Click

### `cli/main.py`

```python
"""
Command-line interface for insurance claims modeling.
"""
import click
import pandas as pd
from pathlib import Path
import logging
import sys

from src.config import Config
from src.pipeline.train import TrainingPipeline
from src.utils.logging_config import setup_logging


@click.group()
@click.option("--env", default="dev", type=click.Choice(["dev", "staging", "prod"]))
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, env: str, verbose: bool):
    """Insurance Claims Modeling Pipeline CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config(environment=env)
    
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_dir", required=True, type=click.Path())
@click.pass_context
def train(ctx, input_path: str, output_dir: str):
    """
    Train distribution models on claims data.
    
    Example:
        python -m cli.main --env prod train -i data/claims.csv -o models/
    """
    config = ctx.obj["config"]
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading data from {input_path}")
    raw_data = pd.read_csv(input_path)
    
    pipeline = TrainingPipeline(config=config)
    artifacts = pipeline.run(raw_data, Path(output_dir))
    
    click.echo(f"\n✅ Training complete. {len(artifacts)} models saved to {output_dir}")
    
    # Print summary table
    click.echo("\nModel Comparison:")
    click.echo("-" * 60)
    click.echo(f"{'Distribution':<15} {'Expected Loss':>15} {'KS p-value':>15}")
    click.echo("-" * 60)
    for a in sorted(artifacts, key=lambda x: x.ks_pvalue, reverse=True):
        click.echo(f"{a.distribution_name:<15} ${a.expected_loss:>14,.0f} {a.ks_pvalue:>15.4f}")


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True))
@click.option("--samples", "-n", default=1000, help="Number of samples to generate")
@click.pass_context
def simulate(ctx, model: str, samples: int):
    """
    Simulate claims from a fitted model.
    
    Example:
        python -m cli.main simulate -m models/weibull_model.json -n 5000
    """
    import json
    from scipy import stats
    
    with open(model) as f:
        artifact = json.load(f)
    
    dist_name = artifact["distribution"]
    params = tuple(artifact["parameters"])
    
    # Generate samples based on distribution type
    if dist_name == "lognormal":
        samples_arr = stats.lognorm.rvs(params[0], loc=params[1], scale=params[2], size=samples)
    elif dist_name == "gamma":
        samples_arr = stats.gamma.rvs(params[0], loc=params[1], scale=params[2], size=samples)
    elif dist_name == "weibull":
        samples_arr = stats.weibull_min.rvs(params[0], loc=params[1], scale=params[2], size=samples)
    
    click.echo(f"\nSimulated {samples} claims from {dist_name} distribution:")
    click.echo(f"  Mean:   ${samples_arr.mean():,.0f}")
    click.echo(f"  Median: ${np.median(samples_arr):,.0f}")
    click.echo(f"  Std:    ${samples_arr.std():,.0f}")
    click.echo(f"  95th %: ${np.percentile(samples_arr, 95):,.0f}")


@cli.command()
@click.pass_context
def validate_config(ctx):
    """Validate configuration files."""
    config = ctx.obj["config"]
    click.echo(f"✅ Configuration valid")
    click.echo(f"   Distributions: {config.distributions}")
    click.echo(f"   Claim probability: {config.claim_probability}")


if __name__ == "__main__":
    cli()
```

---

## 🐳 Docker Containerization

### `Dockerfile`

```dockerfile
# Multi-stage build for smaller production image
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Production stage
FROM python:3.11-slim as production

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser cli/ ./cli/
COPY --chown=appuser:appuser config/ ./config/

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
ENTRYPOINT ["python", "-m", "cli.main"]
CMD ["--help"]
```

### `docker-compose.yaml`

```yaml
version: "3.8"

services:
  # Main application container
  insurance-model:
    build:
      context: .
      target: production
    image: insurance-claims-model:latest
    volumes:
      - ./data:/app/data:ro
      - ./models:/app/models
    environment:
      - ENV=dev
      - AWS_DEFAULT_REGION=us-east-1
    command: ["train", "-i", "/app/data/claims.csv", "-o", "/app/models"]

  # Development container with mounted source
  dev:
    build:
      context: .
      target: production
    volumes:
      - ./src:/app/src
      - ./cli:/app/cli
      - ./config:/app/config
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - ENV=dev
    command: ["--help"]

  # Jupyter for exploration
  jupyter:
    build:
      context: .
      target: production
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/app/notebooks
      - ./data:/app/data
      - ./models:/app/models
    command: >
      jupyter lab --ip=0.0.0.0 --port=8888 --no-browser 
      --NotebookApp.token='' --NotebookApp.password=''
```

### `requirements.txt`

```
# Core dependencies
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
pyyaml>=6.0

# CLI
click>=8.1.0

# Visualization (optional, for notebooks)
matplotlib>=3.7.0
seaborn>=0.12.0

# AWS integration
boto3>=1.28.0

# Workflow orchestration
apache-airflow>=2.7.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Development
ipykernel>=6.25.0
jupyterlab>=4.0.0
```

---

## ☁️ AWS ECR: Container Registry

### Push to ECR Script: `scripts/push_to_ecr.sh`

```bash
#!/bin/bash
set -euo pipefail

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="insurance-claims-model"
IMAGE_TAG="${IMAGE_TAG:-latest}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "🔐 Authenticating with ECR..."
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $ECR_URI

echo "🏗️  Building Docker image..."
docker build -t ${ECR_REPO}:${IMAGE_TAG} --target production .

echo "🏷️  Tagging image for ECR..."
docker tag ${ECR_REPO}:${IMAGE_TAG} ${ECR_URI}:${IMAGE_TAG}

# Also tag with git commit SHA for traceability
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
docker tag ${ECR_REPO}:${IMAGE_TAG} ${ECR_URI}:${GIT_SHA}

echo "📤 Pushing to ECR..."
docker push ${ECR_URI}:${IMAGE_TAG}
docker push ${ECR_URI}:${GIT_SHA}

echo "✅ Successfully pushed:"
echo "   ${ECR_URI}:${IMAGE_TAG}"
echo "   ${ECR_URI}:${GIT_SHA}"
```

### Create ECR Repository: `scripts/setup_ecr.sh`

```bash
#!/bin/bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO="insurance-claims-model"

echo "📦 Creating ECR repository: ${ECR_REPO}"

# Create repository with lifecycle policy
aws ecr create-repository \
    --repository-name ${ECR_REPO} \
    --region ${AWS_REGION} \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 \
    || echo "Repository may already exist"

# Set lifecycle policy to clean up old images
aws ecr put-lifecycle-policy \
    --repository-name ${ECR_REPO} \
    --region ${AWS_REGION} \
    --lifecycle-policy-text '{
        "rules": [
            {
                "rulePriority": 1,
                "description": "Keep last 10 images",
                "selection": {
                    "tagStatus": "any",
                    "countType": "imageCountMoreThan",
                    "countNumber": 10
                },
                "action": {
                    "type": "expire"
                }
            }
        ]
    }'

echo "✅ ECR repository configured"
```

---

## 🌬️ Apache Airflow DAG

### `dags/insurance_pipeline_dag.py`

```python
"""
Airflow DAG for Insurance Claims Model Training Pipeline.

Schedule: Weekly retraining with fresh claims data.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.amazon.aws.operators.s3 import S3CopyObjectOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.utils.task_group import TaskGroup


# DAG default arguments
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email": ["mlops-alerts@company.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def validate_data(**context):
    """Validate incoming claims data before training."""
    import pandas as pd
    import logging
    
    logger = logging.getLogger(__name__)
    
    # In production, this would pull from S3
    data_path = context["params"]["data_path"]
    df = pd.read_csv(data_path)
    
    # Validation checks
    assert len(df) > 100, f"Insufficient data: {len(df)} rows"
    assert "amount" in df.columns, "Missing 'amount' column"
    assert df["amount"].min() > 0, "Found non-positive amounts"
    
    # Push validated row count to XCom
    context["ti"].xcom_push(key="row_count", value=len(df))
    logger.info(f"Validation passed: {len(df)} valid claims")


def run_training(**context):
    """Execute model training pipeline."""
    import sys
    sys.path.insert(0, "/opt/airflow/dags/repo")
    
    from src.config import Config
    from src.pipeline.train import TrainingPipeline
    import pandas as pd
    from pathlib import Path
    
    config = Config(environment="prod")
    data_path = context["params"]["data_path"]
    output_dir = Path(context["params"]["output_dir"])
    
    raw_data = pd.read_csv(data_path)
    pipeline = TrainingPipeline(config=config)
    artifacts = pipeline.run(raw_data, output_dir)
    
    # Push best model info to XCom
    best = max(artifacts, key=lambda a: a.ks_pvalue)
    context["ti"].xcom_push(key="best_model", value=best.distribution_name)
    context["ti"].xcom_push(key="expected_loss", value=best.expected_loss)


def notify_completion(**context):
    """Send notification on successful completion."""
    ti = context["ti"]
    best_model = ti.xcom_pull(task_ids="training.run_training", key="best_model")
    expected_loss = ti.xcom_pull(task_ids="training.run_training", key="expected_loss")
    
    message = f"""
    ✅ Insurance Claims Model Training Complete
    
    Best Model: {best_model}
    Expected Loss: ${expected_loss:,.0f}
    Run Date: {context['ds']}
    """
    
    # In production: send to Slack, email, etc.
    print(message)


# DAG definition
with DAG(
    dag_id="insurance_claims_training",
    default_args=default_args,
    description="Weekly training pipeline for D&O insurance claims model",
    schedule_interval="0 6 * * 0",  # Every Sunday at 6 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "insurance", "training"],
    params={
        "data_path": "/data/claims.csv",
        "output_dir": "/models/{{ ds }}",
    },
) as dag:
    
    # Task: Wait for fresh data
    wait_for_data = S3KeySensor(
        task_id="wait_for_data",
        bucket_name="insurance-data-prod",
        bucket_key="raw/claims_{{ ds }}.csv",
        aws_conn_id="aws_default",
        timeout=3600,
        poke_interval=300,
    )
    
    # Task: Download data from S3
    download_data = BashOperator(
        task_id="download_data",
        bash_command="""
            aws s3 cp s3://insurance-data-prod/raw/claims_{{ ds }}.csv /data/claims.csv
        """,
    )
    
    # Task Group: Training pipeline
    with TaskGroup(group_id="training") as training_group:
        
        validate = PythonOperator(
            task_id="validate_data",
            python_callable=validate_data,
        )
        
        train = PythonOperator(
            task_id="run_training",
            python_callable=run_training,
        )
        
        validate >> train
    
    # Task: Upload models to S3
    upload_models = BashOperator(
        task_id="upload_models",
        bash_command="""
            aws s3 sync /models/{{ ds }}/ s3://insurance-models-prod/{{ ds }}/
        """,
    )
    
    # Task: Notify completion
    notify = PythonOperator(
        task_id="notify_completion",
        python_callable=notify_completion,
    )
    
    # DAG dependencies
    wait_for_data >> download_data >> training_group >> upload_models >> notify
```

### Airflow DAG Visualization

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────────────┐
│  wait_for_data  │────▶│ download_data│────▶│       training          │
│  (S3 Sensor)    │     │   (Bash)     │     │  ┌─────────────────┐    │
└─────────────────┘     └──────────────┘     │  │ validate_data   │    │
                                              │  └────────┬────────┘    │
                                              │           ▼             │
                                              │  ┌─────────────────┐    │
                                              │  │  run_training   │    │
                                              │  └─────────────────┘    │
                                              └───────────┬─────────────┘
                                                          ▼
                                              ┌───────────────────┐
                                              │   upload_models   │
                                              │      (Bash)       │
                                              └─────────┬─────────┘
                                                        ▼
                                              ┌───────────────────┐
                                              │ notify_completion │
                                              │     (Python)      │
                                              └───────────────────┘
```

---

## 🧪 Testing

### `tests/test_distributions.py`

```python
"""Tests for distribution fitting module."""
import pytest
import numpy as np
from scipy import stats

from src.models.distributions import (
    LognormalFitter,
    GammaFitter,
    WeibullFitter,
    get_fitter,
)


@pytest.fixture
def sample_claims():
    """Generate synthetic claims data for testing."""
    np.random.seed(42)
    # Generate data from known gamma distribution
    return stats.gamma.rvs(a=2.0, scale=10000, size=500)


class TestLognormalFitter:
    
    def test_fit_returns_three_params(self, sample_claims):
        fitter = LognormalFitter()
        params = fitter.fit(sample_claims)
        assert len(params) == 3
    
    def test_expected_value_positive(self, sample_claims):
        fitter = LognormalFitter()
        params = fitter.fit(sample_claims)
        ev = fitter.expected_value(params)
        assert ev > 0
    
    def test_ks_test_returns_tuple(self, sample_claims):
        fitter = LognormalFitter()
        params = fitter.fit(sample_claims)
        stat, pval = fitter.ks_test(sample_claims, params)
        assert 0 <= stat <= 1
        assert 0 <= pval <= 1


class TestGammaFitter:
    
    def test_fit_recovers_approximate_params(self, sample_claims):
        """Test that fitting recovers approximately correct shape."""
        fitter = GammaFitter()
        params = fitter.fit(sample_claims)
        shape, loc, scale = params
        
        # Original data was gamma(2.0, scale=10000)
        # Should recover shape close to 2.0
        assert 1.5 < shape < 2.5


class TestGetFitter:
    
    def test_valid_distribution(self):
        fitter = get_fitter("gamma")
        assert fitter.name == "gamma"
    
    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            get_fitter("invalid_dist")


class TestDistributionComparison:
    """Integration tests comparing distributions."""
    
    def test_all_fitters_produce_valid_results(self, sample_claims):
        for dist_name in ["lognormal", "gamma", "weibull"]:
            fitter = get_fitter(dist_name)
            params = fitter.fit(sample_claims)
            ev = fitter.expected_value(params)
            stat, pval = fitter.ks_test(sample_claims, params)
            
            assert ev > 0, f"{dist_name} expected value should be positive"
            assert pval > 0, f"{dist_name} should have valid p-value"
```

### `tests/conftest.py`

```python
"""Pytest configuration and fixtures."""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_claims_df():
    """Generate synthetic claims DataFrame."""
    np.random.seed(42)
    n = 200
    
    return pd.DataFrame({
        "claim_id": range(1, n + 1),
        "year": np.random.choice([2020, 2021, 2022, 2023], size=n),
        "amount": np.random.gamma(shape=2.0, scale=15000, size=n),
        "company_id": np.random.randint(1, 50, size=n),
    })
```

### Run Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_distributions.py -v

# Run with Docker
docker-compose run --rm dev pytest tests/ -v
```

---

## 🔄 CI/CD with GitHub Actions

### `.github/workflows/ci.yaml`

```yaml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"
  AWS_REGION: us-east-1

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Login to ECR
        id: ecr-login
        uses: aws-actions/amazon-ecr-login@v2
      
      - name: Build and push
        env:
          ECR_REGISTRY: ${{ steps.ecr-login.outputs.registry }}
          ECR_REPO: insurance-claims-model
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG --target production .
          docker tag $ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPO:latest
          docker push $ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPO:latest
```

---

## 📊 Logging and Monitoring

### `src/utils/logging_config.py`

```python
"""Centralized logging configuration."""
import logging
import sys
from typing import Optional
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure application logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON formatting (for production)
        log_file: Optional file path for file logging
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
    
    root_logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
```

---

## 🎯 Quick Reference Commands

```bash
# Local development
docker-compose build
docker-compose run --rm dev train -i data/claims.csv -o models/

# Run tests
docker-compose run --rm dev pytest tests/ -v

# Push to ECR
./scripts/push_to_ecr.sh

# Run with Airflow (local)
airflow standalone  # Initialize Airflow
airflow dags trigger insurance_claims_training

# CLI commands
python -m cli.main --env prod train -i data/claims.csv -o models/
python -m cli.main simulate -m models/weibull_model.json -n 10000
python -m cli.main validate-config
```

---

## 📚 Key Takeaways

| Concept | Why It Matters |
|---------|----------------|
| **Modular code** | Testable, reusable components that scale with team size |
| **Config-driven** | Same code, different environments—no hardcoded paths |
| **Docker** | Reproducible environments from dev laptop to production |
| **ECR** | Secure, versioned container storage with lifecycle policies |
| **Airflow** | Reliable scheduling, retries, monitoring, and alerting |
| **CLI with Click** | Scriptable interface for automation and testing |
| **Structured logging** | Queryable logs for debugging production issues |
| **CI/CD** | Automated testing and deployment on every commit |

---

## 🔗 Further Reading

- [12-Factor App Methodology](https://12factor.net/)
- [AWS Well-Architected ML Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/machine-learning-lens.html)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Click Documentation](https://click.palletsprojects.com/)
