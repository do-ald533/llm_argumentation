#!/usr/bin/env python3
"""Quick test script to verify MLflow integration."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import mlflow
        print("✓ MLflow imported successfully")
    except ImportError:
        print("✗ MLflow not installed. Run: pip install mlflow")
        return False
    
    try:
        from src.evaluation import EvaluationMetrics, evaluate_against_golden_standard
        print("✓ Evaluation module imported successfully")
    except ImportError as e:
        print(f"✗ Evaluation module import failed: {e}")
        return False
    
    try:
        from src.pipeline import ArgumentationPipeline
        print("✓ Pipeline imported successfully")
    except ImportError as e:
        print(f"✗ Pipeline import failed: {e}")
        return False
    
    try:
        from src.config import config
        print("✓ Config imported successfully")
        print(f"  - MLflow enabled: {config.enable_mlflow}")
        print(f"  - Tracking URI: {config.mlflow_tracking_uri}")
        print(f"  - Experiment name: {config.mlflow_experiment_name}")
        print(f"  - Prompt version: {config.prompt_version}")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    return True


def test_mlflow_setup():
    """Test MLflow configuration."""
    print("\nTesting MLflow setup...")
    
    try:
        import mlflow
        
        # Set tracking URI
        mlflow.set_tracking_uri("./mlruns")
        print(f"✓ Tracking URI set: {mlflow.get_tracking_uri()}")
        
        # Create/get experiment
        experiment_name = "test-experiment"
        experiment = mlflow.set_experiment(experiment_name)
        print(f"✓ Experiment created/retrieved: {experiment_name}")
        
        # Test a simple run
        with mlflow.start_run(run_name="test-run") as run:
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.95)
            print(f"✓ Test run created: {run.info.run_id}")
        
        print("✓ MLflow is working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ MLflow setup failed: {e}")
        return False


def test_evaluation():
    """Test evaluation metrics (without actual data)."""
    print("\nTesting evaluation metrics...")
    
    try:
        from src.evaluation import EvaluationMetrics
        
        metrics = EvaluationMetrics(
            component_precision=0.85,
            component_recall=0.80,
            component_f1=0.825,
            relation_precision=0.75,
            relation_recall=0.70,
            relation_f1=0.724,
            total_texts=10
        )
        
        print("✓ EvaluationMetrics created")
        print(f"  Sample metrics: {metrics.to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("MLflow Integration Test Suite")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(test_imports())
    
    # Test MLflow setup
    results.append(test_mlflow_setup())
    
    # Test evaluation
    results.append(test_evaluation())
    
    # Summary
    print("\n" + "="*60)
    if all(results):
        print("✅ All tests passed! Experiment tracking is ready to use.")
        print("\nNext steps:")
        print("1. Set up your .env file with API keys")
        print("2. Run the pipeline with: python -m src.main --input <file> --output-prefix <name>")
        print("3. View results with: mlflow ui --port 5000")
        print("\nSee MLFLOW_GUIDE.md for detailed usage instructions.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()
