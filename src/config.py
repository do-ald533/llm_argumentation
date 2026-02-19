"""Configuration management using Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""
    
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    
    deepseek_api_key: str = Field(default="", description="DeepSeek API key")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", description="DeepSeek base URL")
    
    max_retries: int = Field(default=3, description="Maximum number of retries for LLM calls")
    timeout_seconds: int = Field(default=120, description="Timeout for LLM calls in seconds")
    temperature: float = Field(default=0.0, description="Temperature for LLM generation")
    
    input_data_dir: Path = Field(default=Path("data/Input"), description="Input data directory")
    output_data_dir: Path = Field(default=Path("output"), description="Output directory")
    golden_standard_dir: Path = Field(default=Path("data/Golden Standard"), description="Golden standard directory")
    
    mlflow_tracking_uri: str = Field(default="./mlruns", description="MLflow tracking URI")
    mlflow_experiment_name: str = Field(default="argumentation-structuring", description="MLflow experiment name")
    enable_mlflow: bool = Field(default=True, description="Enable MLflow experiment tracking")
    
    prompt_version: str = Field(default="1.0", description="Version of prompts being used")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_data_dir.mkdir(parents=True, exist_ok=True)


config = Config()
