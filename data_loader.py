import os
import urllib.request
import gzip
import random
from graph_engine import GraphDatabase

class DataLoader:
    """Handles data ingestion for both synthetic testing and real benchmarking."""

    @staticmethod
    def generate_synthetic_graph(num_nodes: int = 500,
                                 edges_per_node: int = 3,
                                 seed: int = 42) -> GraphDatabase:
        """Random directed graph for fast debugging. Reproducible via seed."""
        print(f"Generating synthetic test graph "
              f"({num_nodes} nodes, ~{edges_per_node} edges/node, seed={seed})...")
        rng = random.Random(seed)
        db = GraphDatabase()

        for i in range(num_nodes):
            db.add_node(i, {'name': f'Node_{i}', 'type': 'synthetic'})

        for i in range(num_nodes):
            for _ in range(edges_per_node):
                target = rng.randint(0, num_nodes - 1)
                if target != i:
                    db.add_edge(i, target, {'weight': rng.random()})

        return db

    @staticmethod
    def load_snap_facebook_graph(
            local_path: str = "facebook_combined.txt.gz",
            url: str = "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    ) -> GraphDatabase:
        """
        Load the SNAP Facebook combined ego-network as an undirected graph.

        The file lists each undirected edge exactly once; we insert both
        (u, v) and (v, u) so adjacency-list lookups and pandas joins
        correctly reflect undirected semantics.
        """
        print("Loading SNAP Facebook ego-network dataset...")

        if not os.path.exists(local_path):
            print(f"  Downloading {url} -> {local_path}")
            urllib.request.urlretrieve(url, local_path)

        print("  Parsing 88,234 undirected edges (materializing both directions)...")
        db = GraphDatabase()
        added_nodes = set()
        edge_lines = 0

        with gzip.open(local_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                source, target = map(int, line.strip().split())
                edge_lines += 1

                if source not in added_nodes:
                    db.add_node(source, {'type': 'user'})
                    added_nodes.add(source)
                if target not in added_nodes:
                    db.add_node(target, {'type': 'user'})
                    added_nodes.add(target)

                # Undirected
                db.add_edge(source, target, {'relationship': 'friend'})
                db.add_edge(target, source, {'relationship': 'friend'})

        print(f"  Loaded: {db.get_node_count():,} nodes, "
              f"{db.get_edge_count():,} directed edges "
              f"({edge_lines:,} undirected pairs)")
        return db