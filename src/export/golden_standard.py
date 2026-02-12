"""Export argumentation graphs to golden standard CSV format."""
import polars as pl
from pathlib import Path
from typing import List
from src.models import ArgumentGraph
from src.logging_config import get_logger

logger = get_logger("export")

SEPARATOR = ";"


def export_to_golden_standard(
    graphs: List[ArgumentGraph],
    output_dir: Path,
    prefix: str = "output"
) -> None:
    """Export multiple argumentation graphs to golden standard CSV files.
    
    Uses semicolon as CSV separator.
    
    Args:
        graphs: List of ArgumentGraph objects to export
        output_dir: Output directory path
        prefix: Prefix for output files (default: "output")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all components and relations
    all_components = []
    all_relations = []
    
    for graph in graphs:
        all_components.extend(graph.to_golden_standard_components())
        all_relations.extend(graph.to_golden_standard_relations())
    
    # Export components
    components_path = output_dir / f"components_{prefix}.csv"
    if all_components:
        components_df = pl.DataFrame(all_components)
        components_df.write_csv(components_path, separator=SEPARATOR)
        logger.info("components_exported", count=len(all_components), path=str(components_path))
    else:
        # Write empty CSV with headers
        pl.DataFrame({
            "text_id": [], "component_tokens": [], "labels": []
        }).write_csv(components_path, separator=SEPARATOR)
        logger.warning("no_components_to_export", path=str(components_path))
    
    # Export relations
    relations_path = output_dir / f"relations_{prefix}.csv"
    if all_relations:
        relations_df = pl.DataFrame(all_relations)
        relations_df.write_csv(relations_path, separator=SEPARATOR)
        logger.info("relations_exported", count=len(all_relations), path=str(relations_path))
    else:
        # Write empty CSV with headers
        pl.DataFrame({
            "text_id": [], "source_tokens": [], "target_tokens": [], "labels": []
        }).write_csv(relations_path, separator=SEPARATOR)
        logger.warning("no_relations_to_export", path=str(relations_path))
