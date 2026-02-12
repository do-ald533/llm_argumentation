"""Argumentation graph data model."""
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import networkx as nx
from .components import ArgumentComponent, ArgumentRelation


class ArgumentGraph(BaseModel):
    """Complete argumentation graph for a single text."""
    
    text_id: str = Field(..., description="Source document identifier")
    text: str = Field(..., description="Original text")
    components: dict[int, ArgumentComponent] = Field(
        default_factory=dict,
        description="Components indexed by ID"
    )
    relations: list[ArgumentRelation] = Field(
        default_factory=list,
        description="Relations between components"
    )
    conclusion_id: Optional[int] = Field(
        default=None,
        description="ID of the main conclusion/major claim"
    )
    
    def add_component(self, component: ArgumentComponent) -> None:
        """Add a component to the graph."""
        self.components[component.id] = component
    
    def add_relation(self, relation: ArgumentRelation) -> None:
        """Add a relation to the graph."""
        self.relations.append(relation)
    
    def get_component_by_id(self, component_id: int) -> Optional[ArgumentComponent]:
        """Get a component by its ID."""
        return self.components.get(component_id)
    
    def derive_labels(self) -> None:
        """Derive component labels from graph structure.
        
        - MajorClaim: the main conclusion node
        - Claim: any node that has children (is a target in some relation AND
          is also a source in another) — i.e., intermediate nodes
        - Premise: leaf nodes (only appear as source, never as target with
          children of their own)
        """
        if not self.components or self.conclusion_id is None:
            return
        
        # Build sets of sources and targets
        sources = {r.source_id for r in self.relations}
        targets = {r.target_id for r in self.relations}
        
        for comp_id, comp in self.components.items():
            if comp_id == self.conclusion_id:
                comp.label = "MajorClaim"
            elif comp_id in targets:
                # This component is a target of some relation, meaning
                # other components point to it → it's an intermediate node
                comp.label = "Claim"
            else:
                # Leaf node: only a source, never a target (or disconnected)
                comp.label = "Premise"
    
    def validate_relations(self) -> None:
        """Remove relations that reference non-existent components."""
        valid = [
            r for r in self.relations
            if r.source_id in self.components and r.target_id in self.components
        ]
        removed = len(self.relations) - len(valid)
        if removed > 0:
            self.relations = valid
    
    def ensure_dag(self) -> None:
        """Remove edges that create cycles, ensuring the graph is a DAG.
        
        Uses DFS to find back-edges and removes them.
        """
        G = self.to_networkx()
        
        if nx.is_directed_acyclic_graph(G):
            return
        
        # Iteratively remove edges until acyclic
        while not nx.is_directed_acyclic_graph(G):
            try:
                cycle = nx.find_cycle(G, orientation="original")
                # Remove the last edge in the cycle (most likely the spurious one)
                u, v, _ = cycle[-1]
                G.remove_edge(u, v)
            except nx.NetworkXNoCycle:
                break
        
        # Rebuild relations to match the cleaned graph
        remaining_edges = set(G.edges())
        self.relations = [
            r for r in self.relations
            if (r.source_id, r.target_id) in remaining_edges
        ]
    
    def transitive_reduction(self) -> None:
        """Apply transitive reduction to remove redundant edges.
        
        Removes direct edges between nodes when an indirect path already
        exists, clarifying the true logical structure of the argument.
        Requires the graph to be a DAG.
        """
        G = self.to_networkx()
        
        if len(G.nodes) == 0 or len(G.edges) == 0:
            return
        
        if not nx.is_directed_acyclic_graph(G):
            return
        
        reduced = nx.transitive_reduction(G)
        reduced_edges = set(reduced.edges())
        
        original_count = len(self.relations)
        self.relations = [
            r for r in self.relations
            if (r.source_id, r.target_id) in reduced_edges
        ]
        removed = original_count - len(self.relations)
        if removed > 0:
            pass  # Logging handled by caller
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph.
        
        Returns:
            NetworkX DiGraph with components as nodes and relations as edges
        """
        G = nx.DiGraph()
        
        for comp_id, comp in self.components.items():
            G.add_node(comp_id, text=comp.text, label=comp.label)
        
        for rel in self.relations:
            G.add_edge(
                rel.source_id,
                rel.target_id,
                relation=rel.relation_type,
                is_convergent=rel.is_convergent
            )
        
        return G
    
    def visualize(self, output_path: Path) -> None:
        """Save a tree-view visualization of the argument graph as a PNG image.
        
        The conclusion (MajorClaim) is at the root/top, Claims are
        intermediate, and Premises are leaves at the bottom.  Arrows point
        upward from premise → conclusion.  Disconnected nodes are shown in
        grey at the bottom.
        
        Args:
            output_path: Path to save the PNG file
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.lines import Line2D
        except ImportError:
            raise ImportError(
                "matplotlib is required for graph visualization. "
                "Install it with: pip install matplotlib"
            )
        
        G = self.to_networkx()
        
        if len(G.nodes) == 0:
            return
        
        # --- Tree layout ---
        R = G.reverse()  # reverse for BFS layering (root → leaves)
        root = self.conclusion_id
        if root is None or root not in G.nodes:
            roots = [n for n in G.nodes if G.in_degree(n) == 0]
            root = roots[0] if roots else list(G.nodes)[0]
        
        pos = self._hierarchy_pos(R, root)
        
        # Identify disconnected nodes (zero degree in original graph)
        disconnected = {n for n in G.nodes if G.degree(n) == 0}
        
        # --- Colors ---
        node_color_map = {
            "MajorClaim": "#E74C3C",   # Red
            "Claim":      "#3498DB",   # Blue
            "Premise":    "#2ECC71",   # Green
        }
        disconnected_color = "#BDC3C7"  # Grey
        
        node_colors = []
        node_edge_colors = []
        node_line_styles = []
        for n in G.nodes:
            if n in disconnected:
                node_colors.append(disconnected_color)
                node_edge_colors.append("#95A5A6")
            else:
                label = G.nodes[n].get("label", "Premise")
                node_colors.append(node_color_map.get(label, "#BDC3C7"))
                node_edge_colors.append("#2C3E50")
        
        # --- Node labels: ID + truncated text ---
        node_labels = {}
        for n in G.nodes:
            text = G.nodes[n].get("text", "")
            short = (text[:22] + "…") if len(text) > 22 else text
            status = " ⊘" if n in disconnected else ""
            node_labels[n] = f"{n}{status}\n{short}"
        
        # --- Figure ---
        fig, ax = plt.subplots(1, 1, figsize=(18, 11))
        fig.patch.set_facecolor("white")
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=2200,
            alpha=0.92,
            ax=ax,
            edgecolors=node_edge_colors,
            linewidths=2.0,
        )
        nx.draw_networkx_labels(
            G, pos, labels=node_labels,
            font_size=7, font_weight="bold", ax=ax,
        )
        
        # --- Draw edges with varying curvature to reduce overlap ---
        support_edges = [e for e in G.edges if G.edges[e].get("relation") == "support"]
        attack_edges = [e for e in G.edges if G.edges[e].get("relation") == "attack"]
        
        # Group edges by target to fan them out
        from collections import defaultdict
        edges_by_target: dict[int, list] = defaultdict(list)
        for u, v in G.edges:
            edges_by_target[v].append((u, v))
        
        def _draw_edge_group(edgelist, color, style, ax):
            """Draw edges with alternating curvature to avoid overlap."""
            if not edgelist:
                return
            for idx, (u, v) in enumerate(edgelist):
                # Alternate curvature direction and magnitude
                rad = 0.08 + 0.04 * (idx % 3)
                if idx % 2 == 1:
                    rad = -rad
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    edge_color=color,
                    style=style,
                    arrows=True,
                    arrowstyle="-|>",
                    arrowsize=15,
                    connectionstyle=f"arc3,rad={rad}",
                    ax=ax,
                    width=2.0,
                    min_source_margin=22,
                    min_target_margin=22,
                )
        
        _draw_edge_group(support_edges, "#27AE60", "solid", ax)
        _draw_edge_group(attack_edges, "#C0392B", (0, (5, 3)), ax)
        
        # --- Legend ---
        legend_elements = [
            mpatches.Patch(facecolor="#E74C3C", edgecolor="#2C3E50", linewidth=1.5, label="MajorClaim"),
            mpatches.Patch(facecolor="#3498DB", edgecolor="#2C3E50", linewidth=1.5, label="Claim"),
            mpatches.Patch(facecolor="#2ECC71", edgecolor="#2C3E50", linewidth=1.5, label="Premise"),
            mpatches.Patch(facecolor="#BDC3C7", edgecolor="#95A5A6", linewidth=1.5, label="Disconnected"),
            Line2D([0], [0], color="#27AE60", linewidth=2.5, label="Support"),
            Line2D([0], [0], color="#C0392B", linewidth=2.5, linestyle="dashed", label="Attack"),
        ]
        ax.legend(
            handles=legend_elements, loc="upper left",
            fontsize=9, framealpha=0.9, edgecolor="#CCCCCC",
        )
        
        ax.set_title(
            f"Argument Graph: {self.text_id}",
            fontsize=15, fontweight="bold", pad=15,
        )
        ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    
    @staticmethod
    def _hierarchy_pos(
        G: nx.DiGraph,
        root,
        width: float = 1.0,
        vert_gap: float = 0.35,
        xcenter: float = 0.5,
    ) -> dict:
        """Compute hierarchical tree positions via BFS from root.
        
        Root at top, children spread horizontally at each depth.
        Disconnected nodes placed in an extra row at the bottom.
        
        Args:
            G: Directed graph (edges point root → leaves)
            root: Root node
            width: Horizontal space
            vert_gap: Vertical gap between levels
            xcenter: X-centre of root
        """
        from collections import deque
        
        pos = {}
        visited = set()
        
        # BFS to assign layers
        layers: dict[int, list] = {}
        queue = deque([(root, 0)])
        visited.add(root)
        
        while queue:
            node, depth = queue.popleft()
            layers.setdefault(depth, []).append(node)
            for child in G.successors(node):
                if child not in visited:
                    visited.add(child)
                    queue.append((child, depth + 1))
        
        # Disconnected nodes → extra bottom row
        disconnected = set(G.nodes) - visited
        if disconnected:
            max_depth = max(layers.keys(), default=-1) + 1
            layers[max_depth] = sorted(disconnected)
        
        # Positions
        for depth, nodes in layers.items():
            n = len(nodes)
            for i, node in enumerate(nodes):
                x = xcenter - width / 2 + (i + 1) * width / (n + 1)
                y = -depth * vert_gap
                pos[node] = (x, y)
        
        return pos
    
    def to_golden_standard_components(self) -> list[dict]:
        """Export components in golden standard CSV format."""
        return [comp.to_golden_standard() for comp in self.components.values()]
    
    def to_golden_standard_relations(self) -> list[dict]:
        """Export relations in golden standard CSV format."""
        return [rel.to_golden_standard(self.components) for rel in self.relations]
    
    class Config:
        arbitrary_types_allowed = True
