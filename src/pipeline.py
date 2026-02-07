"""Main pipeline for processing multiple texts."""
import polars as pl
from pathlib import Path
from typing import List
from src.config import Config
from src.graph import ArgumentationWorkflow
from src.models import ArgumentGraph
from src.export import export_to_golden_standard


class ArgumentationPipeline:
    """Main pipeline for processing texts through argumentation analysis."""
    
    def __init__(self, config: Config):
        """Initialize pipeline.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.workflow = ArgumentationWorkflow(config)
    
    def process_csv(self, input_file: Path, output_prefix: str) -> List[ArgumentGraph]:
        """Process texts from CSV file.
        
        Args:
            input_file: Path to input CSV file with columns: text_id, text_tokens
            output_prefix: Prefix for output files
            
        Returns:
            List of processed ArgumentGraph objects
        """
        # Load input data
        print(f"\nLoading data from: {input_file}")
        df = pl.read_csv(input_file)
        
        if "text_id" not in df.columns or "text_tokens" not in df.columns:
            raise ValueError("CSV must have 'text_id' and 'text_tokens' columns")
        
        print(f"Found {len(df)} texts to process\n")
        
        # Process each text
        graphs = []
        for row in df.iter_rows(named=True):
            text_id = row["text_id"]
            text = row["text_tokens"]
            
            try:
                graph = self.workflow.run(text, text_id)
                graphs.append(graph)
            except Exception as e:
                self.logger.error("text_processing_failed", text_id=text_id, error=str(e))
                continue
        
        # Export results
        self.logger.info("exporting_results", prefix=output_prefix)
        
        export_to_golden_standard(
            graphs,
            self.config.output_data_dir,
            prefix=output_prefix
        )
        
        self.logger.info("pipeline_complete", 
                        processed_count=len(graphs),
                        output_dir=str(self.config.output_data_dir))
        
        return graphs
    
    def process_single_text(self, text: str, text_id: str) -> ArgumentGraph:
        """Process a single text.
        
        Args:
            text: Input text to analyze
            text_id: Identifier for the text
            
        Returns:
            ArgumentGraph
        """
        return self.workflow.run(text, text_id)
