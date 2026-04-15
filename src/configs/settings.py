from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = "fraud_detection"
    environment: str = "dev"

    kafka_brokers: str = Field(default="localhost:9092")
    kafka_raw_topic: str = Field(default="transactions")
    kafka_processed_topic: str = Field(default="processed-transactions")
    kafka_dlq_topic: str = Field(default="transactions-dlq")
    kafka_group_id: str = Field(default="fraud-feature-consumer")

    mlflow_tracking_uri: str = Field(default="http://mlflow:5000")
    mlflow_model_name: str = Field(default="fraud-detection-model")
    mlflow_model_stage: str = Field(default="Production")

    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)

    feast_repo_path: str = Field(default="src/feature_engineering/feature_store")

    serving_host: str = Field(default="0.0.0.0")
    serving_port: int = Field(default=8000)

    api_rate_limit: str = Field(default="30/minute")

    airflow_base_url: str = Field(default="http://localhost:8080/api/v1")
    airflow_username: str = Field(default="admin")
    airflow_password: str = Field(default="admin")

    prediction_threshold: float = Field(default=0.5)
    promotion_margin: float = Field(default=0.005)
    drift_threshold: float = Field(default=0.3)
    performance_threshold: float = Field(default=0.7)

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


settings = Settings()