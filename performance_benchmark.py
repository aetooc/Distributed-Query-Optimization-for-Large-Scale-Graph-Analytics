"""
Performance Benchmarking Module
Comprehensive performance measurement and comparison framework
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json

@dataclass
class BenchmarkResult:
    """Single benchmark measurement"""
    query_type: str
    method: str
    dataset_size: int
    execution_time: float
    memory_usage: float
    throughput: float  # queries per second

class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking system
    """
    
    def __init__(self, graph_db, query_executor):
        self.graph_db = graph_db
        self.query_executor = query_executor
        self.results = []
    
    def benchmark_query_type(self, query_type: str, params: Dict, 
                            iterations: int = 10) -> Dict:
        """
        Benchmark a specific query type with both algebra approaches
        """
        print(f"Benchmarking {query_type} query...")
        
        graph_times = []
        relational_times = []
        
        # Warm-up runs
        for _ in range(2):
            self.query_executor.execute_graph_algebra_query(query_type, params)
            self.query_executor.execute_relational_algebra_query(query_type, params)
        
        # Actual benchmark runs
        for i in range(iterations):
            # Graph algebra
            _, graph_stats = self.query_executor.execute_graph_algebra_query(
                query_type, params
            )
            graph_times.append(graph_stats['execution_time'])
            
            # Relational algebra
            _, rel_stats = self.query_executor.execute_relational_algebra_query(
                query_type, params
            )
            relational_times.append(rel_stats['execution_time'])
        
        # Statistical analysis
        results = {
            'query_type': query_type,
            'iterations': iterations,
            'graph_algebra': {
                'mean': np.mean(graph_times),
                'median': np.median(graph_times),
                'std': np.std(graph_times),
                'min': np.min(graph_times),
                'max': np.max(graph_times),
                'p95': np.percentile(graph_times, 95),
                'p99': np.percentile(graph_times, 99)
            },
            'relational_algebra': {
                'mean': np.mean(relational_times),
                'median': np.median(relational_times),
                'std': np.std(relational_times),
                'min': np.min(relational_times),
                'max': np.max(relational_times),
                'p95': np.percentile(relational_times, 95),
                'p99': np.percentile(relational_times, 99)
            }
        }
        
        # Calculate speedup
        speedup = np.mean(relational_times) / np.mean(graph_times)
        results['speedup_graph_vs_relational'] = speedup
        results['winner'] = 'graph_algebra' if speedup > 1 else 'relational_algebra'
        
        return results
    
    def scalability_test(self, query_type: str, base_params: Dict,
                        size_multipliers: List[int] = [1, 2, 5, 10]) -> Dict:
        """
        Test how performance scales with dataset size
        """
        print(f"\nScalability testing for {query_type}...")
        
        results = {
            'query_type': query_type,
            'size_multipliers': size_multipliers,
            'graph_algebra': [],
            'relational_algebra': []
        }
        
        original_node_count = self.graph_db.get_node_count()
        
        for multiplier in size_multipliers:
            print(f"  Testing with {multiplier}x data size...")
            
            # For scalability, we'll use depth as proxy for data size
            params = base_params.copy()
            if 'depth' in params:
                params['depth'] = min(params['depth'] * multiplier, 5)  # Cap at 5
            
            # Run benchmark
            _, graph_stats = self.query_executor.execute_graph_algebra_query(
                query_type, params
            )
            _, rel_stats = self.query_executor.execute_relational_algebra_query(
                query_type, params
            )
            
            results['graph_algebra'].append({
                'multiplier': multiplier,
                'time': graph_stats['execution_time']
            })
            
            results['relational_algebra'].append({
                'multiplier': multiplier,
                'time': rel_stats['execution_time']
            })
        
        # Calculate scaling factors
        graph_times = [r['time'] for r in results['graph_algebra']]
        rel_times = [r['time'] for r in results['relational_algebra']]
        
        # Linear regression to determine scaling complexity
        log_multipliers = np.log(size_multipliers)
        log_graph_times = np.log(graph_times)
        log_rel_times = np.log(rel_times)
        
        # Fit: log(time) = a + b*log(size) => time = size^b
        graph_slope = np.polyfit(log_multipliers, log_graph_times, 1)[0]
        rel_slope = np.polyfit(log_multipliers, log_rel_times, 1)[0]
        
        results['complexity_analysis'] = {
            'graph_algebra_exponent': graph_slope,
            'relational_algebra_exponent': rel_slope,
            'graph_complexity': self._classify_complexity(graph_slope),
            'relational_complexity': self._classify_complexity(rel_slope)
        }
        
        return results
    
    def _classify_complexity(self, exponent: float) -> str:
        """Classify algorithmic complexity based on scaling exponent"""
        if exponent < 1.2:
            return "O(n) - Linear"
        elif exponent < 1.7:
            return "O(n log n) - Log-linear"
        elif exponent < 2.3:
            return "O(n²) - Quadratic"
        else:
            return "O(n³+) - Cubic or worse"
    
    def compare_join_algorithms(self, table_sizes: List[int] = [100, 500, 1000, 5000]) -> Dict:
        """
        Compare performance of different join algorithms
        """
        print("\nComparing join algorithms...")
        
        from join_algorithms import JoinAlgorithms
        
        results = {
            'table_sizes': table_sizes,
            'nested_loop': [],
            'hash_join': [],
            'sort_merge': []
        }
        
        for size in table_sizes:
            print(f"  Testing with {size} rows...")
            
            # Create test tables
            left_df = pd.DataFrame({
                'id': np.arange(size),
                'value': np.random.randint(0, 100, size)
            })
            
            right_df = pd.DataFrame({
                'id': np.random.randint(0, size, size),  # Random joins
                'data': np.random.randn(size)
            })
            
            # Run all join algorithms
            comparison = JoinAlgorithms.compare_all_joins(
                left_df, right_df, 'id', 'id'
            )
            
            results['nested_loop'].append({
                'size': size,
                'time': comparison['nested_loop']['execution_time'],
                'comparisons': comparison['nested_loop']['comparisons']
            })
            
            results['hash_join'].append({
                'size': size,
                'time': comparison['hash']['execution_time'],
                'comparisons': comparison['hash']['comparisons']
            })
            
            results['sort_merge'].append({
                'size': size,
                'time': comparison['sort_merge']['execution_time'],
                'comparisons': comparison['sort_merge']['comparisons']
            })
        
        # Calculate speedups
        results['speedup_analysis'] = {
            'hash_vs_nested': [
                results['nested_loop'][i]['time'] / results['hash_join'][i]['time']
                for i in range(len(table_sizes))
            ],
            'sortmerge_vs_nested': [
                results['nested_loop'][i]['time'] / results['sort_merge'][i]['time']
                for i in range(len(table_sizes))
            ]
        }
        
        return results
    
    def comprehensive_benchmark(self) -> Dict:
        """
        Run comprehensive benchmark suite
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE PERFORMANCE BENCHMARK")
        print("="*60 + "\n")
        
        # Get random nodes for testing
        nodes = list(self.graph_db.nodes.keys())
        if len(nodes) < 2:
            raise ValueError("Need at least 2 nodes for benchmarking")
        
        results = {
            'graph_statistics': self.graph_db.get_statistics(),
            'benchmarks': {}
        }
        
        # 1. Neighbor queries
        results['benchmarks']['neighbors'] = self.benchmark_query_type(
            'neighbors',
            {'node_id': nodes[0], 'depth': 2},
            iterations=10
        )
        
        # 2. Reachability queries
        results['benchmarks']['reachability'] = self.benchmark_query_type(
            'reachability',
            {'source': nodes[0], 'target': nodes[min(10, len(nodes)-1)]},
            iterations=10
        )
        
        # 3. Pattern matching
        results['benchmarks']['pattern_match'] = self.benchmark_query_type(
            'pattern_match',
            {'pattern': 'triangle'},
            iterations=5
        )
        
        # 4. Scalability tests
        results['scalability'] = {}
        results['scalability']['neighbors'] = self.scalability_test(
            'neighbors',
            {'node_id': nodes[0], 'depth': 1},
            size_multipliers=[1, 2, 3, 4]
        )
        
        # 5. Join algorithm comparison
        results['join_comparison'] = self.compare_join_algorithms(
            table_sizes=[100, 500, 1000]
        )
        
        return results
    
    def generate_performance_report(self, results: Dict) -> str:
        """
        Generate human-readable performance report
        """
        report_lines = []
        report_lines.append("\n" + "="*70)
        report_lines.append("PERFORMANCE BENCHMARK REPORT")
        report_lines.append("="*70 + "\n")
        
        # Graph statistics
        stats = results['graph_statistics']
        report_lines.append("Graph Statistics:")
        report_lines.append(f"  Nodes: {stats['num_nodes']:,}")
        report_lines.append(f"  Edges: {stats['num_edges']:,}")
        report_lines.append(f"  Avg Degree: {stats['avg_degree']:.2f}")
        report_lines.append(f"  Max Degree: {stats['max_degree']}")
        report_lines.append("")
        
        # Query benchmarks
        report_lines.append("Query Performance Comparison:")
        report_lines.append("-" * 70)
        
        for query_type, bench in results['benchmarks'].items():
            report_lines.append(f"\n{query_type.upper()} Query:")
            
            graph_time = bench['graph_algebra']['mean'] * 1000  # ms
            rel_time = bench['relational_algebra']['mean'] * 1000  # ms
            speedup = bench['speedup_graph_vs_relational']
            
            report_lines.append(f"  Graph Algebra:      {graph_time:.3f} ms (±{bench['graph_algebra']['std']*1000:.3f})")
            report_lines.append(f"  Relational Algebra: {rel_time:.3f} ms (±{bench['relational_algebra']['std']*1000:.3f})")
            report_lines.append(f"  Speedup:            {speedup:.2f}x ({'Graph' if speedup > 1 else 'Relational'} is faster)")
        
        report_lines.append("\n" + "-" * 70)
        
        # Scalability analysis
        if 'scalability' in results:
            report_lines.append("\nScalability Analysis:")
            report_lines.append("-" * 70)
            
            for query_type, scale_data in results['scalability'].items():
                if 'complexity_analysis' in scale_data:
                    comp = scale_data['complexity_analysis']
                    report_lines.append(f"\n{query_type.upper()}:")
                    report_lines.append(f"  Graph Algebra:      {comp['graph_complexity']}")
                    report_lines.append(f"  Relational Algebra: {comp['relational_complexity']}")
        
        report_lines.append("\n" + "-" * 70)
        
        # Join comparison
        if 'join_comparison' in results:
            report_lines.append("\nJoin Algorithm Performance:")
            report_lines.append("-" * 70)
            
            sizes = results['join_comparison']['table_sizes']
            speedups_hash = results['join_comparison']['speedup_analysis']['hash_vs_nested']
            speedups_sm = results['join_comparison']['speedup_analysis']['sortmerge_vs_nested']
            
            report_lines.append(f"\nAverage Speedups (vs Nested Loop):")
            report_lines.append(f"  Hash Join:       {np.mean(speedups_hash):.2f}x faster")
            report_lines.append(f"  Sort-Merge Join: {np.mean(speedups_sm):.2f}x faster")
        
        report_lines.append("\n" + "="*70 + "\n")
        
        return "\n".join(report_lines)
