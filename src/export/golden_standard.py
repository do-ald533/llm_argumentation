"""Export argumentation graphs to golden standard CSV format."""
import polars as pl
from pathlib import Path
from typing import List
from src.models import ArgumentGraph
from src.logging_config import get_logger

logger = get_logger("export")


def export_to_golden_standard(
    graphs: List[ArgumentGraph],
    output_dir: Path,
    prefix: str = "output"
) -> None:
    """Export multiple argumentation graphs to golden standard CSV files.
    
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
    if all_components:
        components_df = pl.DataFrame(all_components)
        components_path = output_dir / f"components_{prefix}.csv"
        components_df.write_csv(components_path)
        logger.info("components_exported", count=len(all_components), path=str(components_path))
    
    # Export relations
    if all_relations:
        relations_df = pl.DataFrame(all_relations)
        relations_path = output_dir / f"relations_{prefix}.csv"
        relations_df.write_csv(relations_path)
        logger.info("relations_exported", count=len(all_relations), path=str(relations_path))


def export_single_graph(
    graph: ArgumentGraph,
    output_dir: Path,
    text_id: str
) -> None:
    """Export a single argumentation graph to CSV files.
    
    Args:
        graph: ArgumentGraph to export
        output_dir: Output directory path
        text_id: Text identifier for filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export components
    components = graph.to_golden_standard_components()
    if components:
        components_df = pl.DataFrame(components)
        components_path = output_dir / f"components_{text_id}.csv"
        components_df.write_csv(components_path)
        logger.info("graph_components_exported", text_id=text_id, count=len(components), path=str(components_path))
    
    # Export relations
    relations = graph.to_golden_standard_relations()
    if relations:
        relations_df = pl.DataFrame(relations)
        relations_path = output_dir / f"relations_{text_id}.csv"
        relations_df.write_csv(relations_path)
        logger.info("graph_relations_exported", text_id=text_id, count=len(relations), path=str(relations_path))
