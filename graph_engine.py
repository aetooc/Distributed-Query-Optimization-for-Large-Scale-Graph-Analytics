import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time


@dataclass
class Node:
    id: int
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: int
    target: int
    properties: Dict[str, Any] = field(default_factory=dict)

class GraphDatabase:
    """In-memory graph database supporting both graph-native and relational queries."""

    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency_list: Dict[int, List[int]] = defaultdict(list)
        self.reverse_adjacency: Dict[int, List[int]] = defaultdict(list)
        self.edge_index: Dict[Tuple[int, int], Edge] = {}
        self._nodes_df: Optional[pd.DataFrame] = None
        self._edges_df: Optional[pd.DataFrame] = None

    def add_node(self, node_id: int, properties: Dict[str, Any] = None):
        if not isinstance(node_id, int) or isinstance(node_id, bool):
            raise TypeError(f"node_id must be int, got {type(node_id).__name__}")
        self.nodes[node_id] = Node(node_id, properties or {})
        self._invalidate_tables()

    def add_edge(self, source: int, target: int, properties: Dict[str, Any] = None):
        if not isinstance(source, int) or not isinstance(target, int):
            raise TypeError("source and target must be int")
        edge = Edge(source, target, properties or {})
        self.edges.append(edge)
        self.adjacency_list[source].append(target)
        self.reverse_adjacency[target].append(source)
        self.edge_index[(source, target)] = edge
        self._invalidate_tables()

    def _invalidate_tables(self):
        self._nodes_df = None
        self._edges_df = None

    def get_neighbors(self, node_id: int) -> List[int]:
        return self.adjacency_list.get(node_id, [])

    def get_incoming_neighbors(self, node_id: int) -> List[int]:
        return self.reverse_adjacency.get(node_id, [])

    def get_node_count(self) -> int:
        return len(self.nodes)

    def get_edge_count(self) -> int:
        return len(self.edges)

    def materialize_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if self._nodes_df is not None and self._edges_df is not None:
            return self._nodes_df, self._edges_df

        node_ids = list(self.nodes.keys())
        self._nodes_df = pd.DataFrame({'node_id': node_ids})

        if self.edges:
            sources = np.fromiter((e.source for e in self.edges), dtype=np.int64,
                                  count=len(self.edges))
            targets = np.fromiter((e.target for e in self.edges), dtype=np.int64,
                                  count=len(self.edges))
            self._edges_df = pd.DataFrame({'source': sources, 'target': targets})
        else:
            self._edges_df = pd.DataFrame({'source': pd.Series(dtype=np.int64),
                                           'target': pd.Series(dtype=np.int64)})

        return self._nodes_df, self._edges_df

    # Backwards-compatible alias.
    def to_relational_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.materialize_tables()

    def get_statistics(self) -> Dict:
        degrees = [len(self.adjacency_list[n]) for n in self.nodes.keys()]
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'avg_degree': float(np.mean(degrees)) if degrees else 0.0,
            'max_degree': int(max(degrees)) if degrees else 0,
            'min_degree': int(min(degrees)) if degrees else 0,
        }


class QueryExecutor:
    """Executes queries using either graph-native or relational algebra."""

    def __init__(self, graph_db: GraphDatabase):
        self.graph_db = graph_db
        self.graph_db.materialize_tables()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def execute_graph_algebra_query(self, query_type: str, params: Dict) -> Tuple[Any, Dict]:
        self._validate_params(query_type, params)
        start = time.perf_counter()
        if query_type == "neighbors":
            result = self._graph_neighbors(params['node_id'], params.get('depth', 1))
        elif query_type == "shortest_path":
            result = self._graph_shortest_path(params['source'], params['target'])
        elif query_type == "pattern_match":
            result = self._graph_pattern_match(params.get('pattern', 'triangle'))
        elif query_type == "reachability":
            result = self._graph_reachability(params['source'], params['target'])
        else:
            raise ValueError(f"Unknown query type: {query_type}")
        return result, {'execution_time': time.perf_counter() - start,
                        'method': 'graph_algebra', 'query_type': query_type}

    def execute_relational_algebra_query(self, query_type: str, params: Dict) -> Tuple[Any, Dict]:
        self._validate_params(query_type, params)
        # Tables already materialized in __init__; this call is a cache hit.
        nodes_df, edges_df = self.graph_db.materialize_tables()
        start = time.perf_counter()
        if query_type == "neighbors":
            result = self._relational_neighbors(edges_df, params['node_id'], params.get('depth', 1))
        elif query_type == "shortest_path":
            result = self._relational_shortest_path(edges_df, params['source'], params['target'])
        elif query_type == "pattern_match":
            result = self._relational_pattern_match(edges_df, params.get('pattern', 'triangle'))
        elif query_type == "reachability":
            result = self._relational_reachability(edges_df, params['source'], params['target'])
        else:
            raise ValueError(f"Unknown query type: {query_type}")
        return result, {'execution_time': time.perf_counter() - start,
                        'method': 'relational_algebra', 'query_type': query_type}

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_params(query_type: str, params: Dict):
        if not isinstance(params, dict):
            raise TypeError("params must be a dict")
        if query_type == "neighbors":
            nid = params.get('node_id')
            if not isinstance(nid, int) or isinstance(nid, bool):
                raise TypeError("node_id must be int")
            d = params.get('depth', 1)
            if not isinstance(d, int) or isinstance(d, bool) or d < 0:
                raise ValueError("depth must be a non-negative int")
            if d > 20:
                raise ValueError("depth > 20 rejected to prevent resource exhaustion")
        elif query_type in ("shortest_path", "reachability"):
            s, t = params.get('source'), params.get('target')
            if (not isinstance(s, int) or isinstance(s, bool)
                    or not isinstance(t, int) or isinstance(t, bool)):
                raise TypeError("source and target must be int")

    # ------------------------------------------------------------------
    # Graph-native algebra
    # ------------------------------------------------------------------
    def _graph_neighbors(self, node_id: int, depth: int) -> Set[int]:
        """
        Return the set of nodes reachable in 1..depth hops (excluding the
        start node itself). Standard BFS frontier expansion.
        """
        if depth == 0:
            return set()
        visited: Set[int] = {node_id}
        frontier: Set[int] = {node_id}
        result: Set[int] = set()
        for _ in range(depth):
            next_frontier: Set[int] = set()
            for u in frontier:
                for v in self.graph_db.get_neighbors(u):
                    if v not in visited:
                        visited.add(v)
                        next_frontier.add(v)
            if not next_frontier:
                break
            result |= next_frontier
            frontier = next_frontier
        return result

    def _graph_shortest_path(self, source: int, target: int) -> Optional[List[int]]:
        """BFS shortest path using parent pointers (no per-node path copying)."""
        if source == target:
            return [source]
        parent: Dict[int, int] = {source: source}
        q = deque([source])
        while q:
            u = q.popleft()
            for v in self.graph_db.get_neighbors(u):
                if v in parent:
                    continue
                parent[v] = u
                if v == target:
                    path = [v]
                    while path[-1] != source:
                        path.append(parent[path[-1]])
                    return list(reversed(path))
                q.append(v)
        return None

    def _graph_pattern_match(self, pattern: str) -> List[Tuple[int, int, int]]:

        if pattern != 'triangle':
            raise ValueError(f"Unsupported pattern: {pattern}")
        adj = self.graph_db.adjacency_list
        nbr_sets: Dict[int, Set[int]] = {u: set(adj[u]) for u in self.graph_db.nodes}
        triangles: List[Tuple[int, int, int]] = []
        for u in self.graph_db.nodes:
            u_nbrs = nbr_sets[u]
            for v in u_nbrs:
                if v <= u:
                    continue
                common = u_nbrs & nbr_sets.get(v, set())
                for w in common:
                    if w > v:
                        triangles.append((u, v, w))
        return triangles

    def _graph_reachability(self, source: int, target: int) -> bool:
        if source == target:
            return True
        visited: Set[int] = {source}
        stack = [source]
        while stack:
            u = stack.pop()
            for v in self.graph_db.get_neighbors(u):
                if v == target:
                    return True
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        return False

    # ------------------------------------------------------------------
    # Relational algebra
    # ------------------------------------------------------------------


    def _relational_neighbors(self, edges_df: pd.DataFrame,
                              node_id: int, depth: int) -> Set[int]:
        """k-hop neighbours via iterated semi-join (SQL recursive-CTE style)."""
        if depth == 0:
            return set()
        visited: Set[int] = {node_id}
        frontier = pd.DataFrame({'node_id': pd.Series([node_id], dtype=np.int64)})
        result: Set[int] = set()
        for _ in range(depth):
            joined = frontier.merge(edges_df, left_on='node_id', right_on='source')
            if joined.empty:
                break
            new_neighbors = set(joined['target'].to_numpy().tolist()) - visited
            if not new_neighbors:
                break
            result |= new_neighbors
            visited |= new_neighbors
            frontier = pd.DataFrame({'node_id': pd.Series(list(new_neighbors), dtype=np.int64)})
        return result

    def _relational_reachability(self, edges_df: pd.DataFrame,
                                 source: int, target: int) -> bool:
        """Reachability via iterated joins to fix-point (recursive-CTE style)."""
        if source == target:
            return True
        visited: Set[int] = {source}
        frontier = pd.DataFrame({'node_id': pd.Series([source], dtype=np.int64)})
        while len(frontier) > 0:
            joined = frontier.merge(edges_df, left_on='node_id', right_on='source')
            if joined.empty:
                return False
            reached = set(joined['target'].to_numpy().tolist())
            if target in reached:
                return True
            new_nodes = reached - visited
            if not new_nodes:
                return False
            visited |= new_nodes
            frontier = pd.DataFrame({'node_id': pd.Series(list(new_nodes), dtype=np.int64)})
        return False

    def _relational_shortest_path(self, edges_df: pd.DataFrame,
                                  source: int, target: int) -> Optional[List[int]]:
        """Shortest path via level-by-level joins with parent tracking."""
        if source == target:
            return [source]
        parent: Dict[int, int] = {source: source}
        frontier = pd.DataFrame({'node_id': pd.Series([source], dtype=np.int64)})
        while len(frontier) > 0:
            joined = frontier.merge(edges_df, left_on='node_id', right_on='source')
            if joined.empty:
                return None
            sources_arr = joined['source'].to_numpy()
            targets_arr = joined['target'].to_numpy()
            next_nodes: List[int] = []
            for s, t in zip(sources_arr, targets_arr):
                t_i = int(t)
                if t_i not in parent:
                    parent[t_i] = int(s)
                    if t_i == target:
                        path = [t_i]
                        while path[-1] != source:
                            path.append(parent[path[-1]])
                        return list(reversed(path))
                    next_nodes.append(t_i)
            if not next_nodes:
                return None
            frontier = pd.DataFrame({'node_id': pd.Series(next_nodes, dtype=np.int64)})
        return None

    def _relational_pattern_match(self, edges_df: pd.DataFrame,
                                  pattern: str) -> List[Tuple[int, int, int]]:

        if pattern != 'triangle':
            raise ValueError(f"Unsupported pattern: {pattern}")

        e1 = edges_df.rename(columns={'source': 'a', 'target': 'b'})
        e2 = edges_df.rename(columns={'source': 'b2', 'target': 'c'})
        e3 = edges_df.rename(columns={'source': 'c2', 'target': 'a2'})

        # Selection pushed before the first join to keep intermediates small.
        e1 = e1[e1['a'] < e1['b']]

        j1 = e1.merge(e2, left_on='b', right_on='b2')
        j1 = j1[j1['b'] < j1['c']]

        j2 = j1.merge(e3, left_on='c', right_on='c2')
        triangles = j2[j2['a2'] == j2['a']][['a', 'b', 'c']]

        return list(map(tuple, triangles.to_numpy().tolist()))