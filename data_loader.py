import os
import urllib.request
import gzip
import random
from graph_engine import GraphDatabase

class DataLoader:
    """Handles data ingestion for both synthetic testing and real-world benchmarking."""
    
    @staticmethod
    def generate_synthetic_graph(num_nodes=500, edges_per_node=3) -> GraphDatabase:
        print(f"Generating synthetic test graph with {num_nodes} nodes...")
        db = GraphDatabase()
        
        for i in range(num_nodes):
            db.add_node(i, {'name': f'Node_{i}', 'type': 'synthetic'})
            
        for i in range(num_nodes):
            for _ in range(edges_per_node):
                target = random.randint(0, num_nodes - 1)
                if target != i:
                    db.add_edge(i, target, {'weight': random.random()})
                    
        return db

    @staticmethod
    def load_snap_facebook_graph() -> GraphDatabase:
        print("Fetching large-scale dataset from Stanford Network Analysis Project (SNAP)...")
        db = GraphDatabase()
        url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
        filename = "facebook_combined.txt.gz"
        
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            
        print("Processing 88,000+ real-world social connections...")
        added_nodes = set()
        
        with gzip.open(filename, 'rt') as f:
            for line in f:
                if line.startswith('#'): continue
                source, target = map(int, line.strip().split())
                
                if source not in added_nodes:
                    db.add_node(source, {'type': 'user'})
                    added_nodes.add(source)
                if target not in added_nodes:
                    db.add_node(target, {'type': 'user'})
                    added_nodes.add(target)
                    
                db.add_edge(source, target, {'relationship': 'friend'})
                
        return db