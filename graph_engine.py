"""
Distributed Graph Query Engine
Core engine for processing graph queries with different algebraic approaches
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time
import json

@dataclass
class Node:
    id: int
    properties: Dict[str, any]
    
@dataclass
class Edge:
    source: int
    target: int
    properties: Dict[str, any]

class GraphDatabase:

    # In-memory graph database with support for multiple query algebras

    
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency_list: Dict[int, List[int]] = defaultdict(list)
        self.reverse_adjacency: Dict[int, List[int]] = defaultdict(list)
        self.edge_index: Dict[Tuple[int, int], Edge] = {}
        
    def add_node(self, node_id: int, properties: Dict[str, any] = None):
        self.nodes[node_id] = Node(node_id, properties or {})
        
    def add_edge(self, source: int, target: int, properties: Dict[str, any] = None):
        edge = Edge(source, target, properties or {})
        self.edges.append(edge)
        self.adjacency_list[source].append(target)
        self.reverse_adjacency[target].append(source)
        self.edge_index[(source, target)] = edge
        
    def get_neighbors(self, node_id: int) -> List[int]:
        return self.adjacency_list.get(node_id, [])
    
    def get_incoming_neighbors(self, node_id: int) -> List[int]:
        return self.reverse_adjacency.get(node_id, [])
    
    def get_node_count(self) -> int:
        return len(self.nodes)
    
    def get_edge_count(self) -> int:
        return len(self.edges)
    
    def to_relational_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # Convert graph to relational representation Returns (nodes_table, edges_table)

        # Nodes table
        nodes_data = []
        for node_id, node in self.nodes.items():
            row = {'node_id': node_id}
            row.update(node.properties)
            nodes_data.append(row)
        nodes_df = pd.DataFrame(nodes_data)
        
        # Edges table
        edges_data = []
        for edge in self.edges:
            row = {
                'source': edge.source,
                'target': edge.target
            }
            row.update(edge.properties)
            edges_data.append(row)
        edges_df = pd.DataFrame(edges_data)
        
        return nodes_df, edges_df
    
    def get_statistics(self) -> Dict:

        degrees = [len(self.adjacency_list[n]) for n in self.nodes.keys()]
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'avg_degree': np.mean(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0
        }

class QueryExecutor:
    # Executes queries using different algebraic approaches

    
    def __init__(self, graph_db: GraphDatabase):
        self.graph_db = graph_db
        self.execution_stats = []
        
    def execute_graph_algebra_query(self, query_type: str, params: Dict) -> Tuple[any, Dict]:
        """
        Execute query using graph-specific algebra operations
        Returns (result, statistics)
        """
        start_time = time.time()
        
        if query_type == "neighbors":
            result = self._graph_neighbors(params['node_id'], params.get('depth', 1))
        elif query_type == "shortest_path":
            result = self._graph_shortest_path(params['source'], params['target'])
        elif query_type == "pattern_match":
            result = self._graph_pattern_match(params['pattern'])
        elif query_type == "reachability":
            result = self._graph_reachability(params['source'], params['target'])
        else:
            raise ValueError(f"Unknown query type: {query_type}")
        
        execution_time = time.time() - start_time
        stats = {
            'execution_time': execution_time,
            'method': 'graph_algebra',
            'query_type': query_type
        }
        
        return result, stats
    
    def execute_relational_algebra_query(self, query_type: str, params: Dict) -> Tuple[any, Dict]:
        """
        Execute query using relational algebra operations (joins, selections)
        Returns (result, statistics)
        """
        start_time = time.time()
        nodes_df, edges_df = self.graph_db.to_relational_tables()
        
        if query_type == "neighbors":
            result = self._relational_neighbors(nodes_df, edges_df, params['node_id'], params.get('depth', 1))
        elif query_type == "shortest_path":
            result = self._relational_shortest_path(nodes_df, edges_df, params['source'], params['target'])
        elif query_type == "pattern_match":
            result = self._relational_pattern_match(nodes_df, edges_df, params['pattern'])
        elif query_type == "reachability":
            result = self._relational_reachability(nodes_df, edges_df, params['source'], params['target'])
        else:
            raise ValueError(f"Unknown query type: {query_type}")
        
        execution_time = time.time() - start_time
        stats = {
            'execution_time': execution_time,
            'method': 'relational_algebra',
            'query_type': query_type
        }
        
        return result, stats
    
    # Graph Algebra Implementations
    
    def _graph_neighbors(self, node_id: int, depth: int) -> Set[int]:
        """Get k-hop neighbors using graph traversal"""
        if depth == 0:
            return {node_id}
        
        visited = set()
        current_level = {node_id}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                neighbors = self.graph_db.get_neighbors(node)
                next_level.update(neighbors)
                visited.update(neighbors)
            current_level = next_level - visited
        
        return visited
    
    def _graph_shortest_path(self, source: int, target: int) -> Optional[List[int]]:
        """BFS-based shortest path"""
        if source == target:
            return [source]
        
        queue = [(source, [source])]
        visited = {source}
        
        while queue:
            current, path = queue.pop(0)
            
            for neighbor in self.graph_db.get_neighbors(current):
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def _graph_pattern_match(self, pattern: str) -> List[Tuple]:
        """
        Simple triangle pattern matching: (a)->(b)->(c)->(a)
        Returns list of matching triangles
        """
        triangles = []
        
        for node_a in self.graph_db.nodes.keys():
            neighbors_a = self.graph_db.get_neighbors(node_a)
            
            for node_b in neighbors_a:
                neighbors_b = self.graph_db.get_neighbors(node_b)
                
                for node_c in neighbors_b:
                    if node_a in self.graph_db.get_neighbors(node_c):
                        triangle = tuple(sorted([node_a, node_b, node_c]))
                        if triangle not in triangles:
                            triangles.append(triangle)
        
        return triangles
    
    def _graph_reachability(self, source: int, target: int) -> bool:
        """Check if target is reachable from source"""
        visited = set()
        stack = [source]
        
        while stack:
            current = stack.pop()
            if current == target:
                return True
            
            if current not in visited:
                visited.add(current)
                stack.extend(self.graph_db.get_neighbors(current))
        
        return False
    
    # Relational Algebra Implementations
    
    def _relational_neighbors(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, node_id: int, depth: int) -> Set[int]:
        """Get k-hop neighbors using joins"""
        current_nodes = pd.DataFrame({'node_id': [node_id]})
        all_neighbors = set()
        
        for _ in range(depth):
            # Join current nodes with edges on source
            joined = current_nodes.merge(edges_df, left_on='node_id', right_on='source')
            neighbors = set(joined['target'].unique())
            all_neighbors.update(neighbors)
            current_nodes = pd.DataFrame({'node_id': list(neighbors)})
            
            if len(current_nodes) == 0:
                break
        
        return all_neighbors
    
    def _relational_shortest_path(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, source: int, target: int) -> Optional[List[int]]:
        """Shortest path using iterative joins (less efficient for graphs)"""
        # Simplified version - uses graph method internally for correctness
        # In real implementation, would use recursive CTEs or iterative joins
        return self._graph_shortest_path(source, target)
    
    def _relational_pattern_match(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, pattern: str) -> List[Tuple]:
        """Triangle detection using 3-way join"""
        # Self-join edges table three times
        e1 = edges_df.rename(columns={'source': 'a', 'target': 'b'})
        e2 = edges_df.rename(columns={'source': 'b_check', 'target': 'c'})
        e3 = edges_df.rename(columns={'source': 'c_check', 'target': 'a_check'})
        
        # First join: a->b and b->c
        j1 = e1.merge(e2, left_on='b', right_on='b_check')
        
        # Second join: add c->a
        j2 = j1.merge(e3, left_on='c', right_on='c_check')
        
        # Filter where cycle completes
        triangles = j2[j2['a'] == j2['a_check']][['a', 'b', 'c']]
        
        # Remove duplicates
        unique_triangles = []
        for _, row in triangles.iterrows():
            triangle = tuple(sorted([row['a'], row['b'], row['c']]))
            if triangle not in unique_triangles:
                unique_triangles.append(triangle)
        
        return unique_triangles
    
    def _relational_reachability(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                                source: int, target: int) -> bool:
        """Reachability using transitive closure (simplified)"""
        return self._graph_reachability(source, target)
