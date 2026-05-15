import time
import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    query_type: str
    method: str
    dataset_size: int
    execution_time: float


class PerformanceBenchmark:
    """Comprehensive performance benchmarking system."""

    def __init__(self, graph_db, query_executor):
        self.graph_db = graph_db
        self.query_executor = query_executor

    # ------------------------------------------------------------------
    def benchmark_query_type(self, query_type: str, params: Dict,
                             iterations: int = 10,
                             warmup: int = 2) -> Dict:
        """Benchmark a single query under both execution paths."""
        print(f"  Benchmarking {query_type} ...")

        for _ in range(warmup):
            self.query_executor.execute_graph_algebra_query(query_type, params)
            self.query_executor.execute_relational_algebra_query(query_type, params)

        graph_times, rel_times = [], []
        last_graph_result, last_rel_result = None, None
        for _ in range(iterations):
            g_res, g_stats = self.query_executor.execute_graph_algebra_query(query_type, params)
            r_res, r_stats = self.query_executor.execute_relational_algebra_query(query_type, params)
            graph_times.append(g_stats['execution_time'])
            rel_times.append(r_stats['execution_time'])
            last_graph_result, last_rel_result = g_res, r_res

        # Correctness cross-check: both paths must agree on the result.
        same_result = self._results_equal(last_graph_result, last_rel_result)

        gmean, rmean = float(np.mean(graph_times)), float(np.mean(rel_times))
        # Guard against degenerate near-zero means before computing speedup.
        eps = 1e-9
        speedup = rmean / max(gmean, eps)

        return {
            'query_type': query_type,
            'iterations': iterations,
            'results_match': same_result,
            'graph_algebra': self._stats(graph_times),
            'relational_algebra': self._stats(rel_times),
            'speedup_graph_vs_relational': speedup,
            'winner': 'graph_algebra' if speedup > 1 else 'relational_algebra',
        }

    @staticmethod
    def _stats(times: List[float]) -> Dict:
        a = np.array(times)
        return {
            'mean': float(a.mean()),
            'median': float(np.median(a)),
            'std': float(a.std(ddof=0)),
            'min': float(a.min()),
            'max': float(a.max()),
            'p95': float(np.percentile(a, 95)),
            'p99': float(np.percentile(a, 99)),
        }

    @staticmethod
    def _results_equal(a, b) -> bool:
        """Loose equality covering the result types our queries return."""
        if type(a) != type(b):
            # set vs list-of-tuples can both be valid; compare as sets.
            try:
                return set(map(tuple, a)) == set(map(tuple, b)) if hasattr(a, '__iter__') else a == b
            except Exception:
                return False
        if isinstance(a, set):
            return a == b
        if isinstance(a, list):
            try:
                return set(map(tuple, a)) == set(map(tuple, b))
            except TypeError:
                return a == b
        return a == b

    # ------------------------------------------------------------------
    def scalability_test(self, base_params: Dict,
                         depths: List[int] = (1, 2, 3, 4)) -> Dict:
        """
        Measure how neighbour-query cost scales with traversal depth.

        Depth is a meaningful workload-size knob: each additional hop
        expands the frontier roughly multiplicatively, so this probes
        the engines' behaviour on growing intermediate sizes.
        """
        print(f"  Scalability over depths {list(depths)} ...")
        graph_pts, rel_pts = [], []
        for d in depths:
            p = dict(base_params); p['depth'] = d
            _, g = self.query_executor.execute_graph_algebra_query('neighbors', p)
            _, r = self.query_executor.execute_relational_algebra_query('neighbors', p)
            graph_pts.append({'depth': d, 'time': g['execution_time']})
            rel_pts.append({'depth': d, 'time': r['execution_time']})

        return {
            'query_type': 'neighbors',
            'depths': list(depths),
            'graph_algebra': graph_pts,
            'relational_algebra': rel_pts,
        }

    # ------------------------------------------------------------------
    def compare_join_algorithms(self, table_sizes: List[int] = (100, 500, 1000, 2000),
                                seed: int = 42) -> Dict:
        """Benchmark the three from-scratch join algorithms across sizes."""
        from join_algorithms import JoinAlgorithms
        print(f"  Join algorithms across sizes {list(table_sizes)} ...")

        results = {
            'table_sizes': list(table_sizes),
            'nested_loop': [],
            'hash_join': [],
            'sort_merge': [],
            'result_sizes_consistent_per_size': [],
        }
        for size in table_sizes:
            rng = np.random.default_rng(seed)
            left = pd.DataFrame({'id': np.arange(size),
                                 'lv': rng.integers(0, 100, size)})
            right = pd.DataFrame({'id': rng.integers(0, size, size),
                                  'rv': rng.standard_normal(size)})
            cmp = JoinAlgorithms.compare_all_joins(left, right, 'id', 'id')
            results['nested_loop'].append({'size': size,
                                           'time': cmp['nested_loop']['execution_time'],
                                           'comparisons': cmp['nested_loop']['comparisons']})
            results['hash_join'].append({'size': size,
                                         'time': cmp['hash']['execution_time'],
                                         'probes': cmp['hash']['probes']})
            results['sort_merge'].append({'size': size,
                                          'time': cmp['sort_merge']['execution_time'],
                                          'comparisons': cmp['sort_merge']['comparisons']})
            results['result_sizes_consistent_per_size'].append(cmp['result_size_consistent'])

        results['speedup_analysis'] = {
            'hash_vs_nested': [
                results['nested_loop'][i]['time'] / results['hash_join'][i]['time']
                for i in range(len(table_sizes))
            ],
            'sortmerge_vs_nested': [
                results['nested_loop'][i]['time'] / results['sort_merge'][i]['time']
                for i in range(len(table_sizes))
            ],
        }
        return results

    # ------------------------------------------------------------------
    def _benchmark_triangles_on_subgraph(self, subgraph_size: int = 500,
                                         iterations: int = 3) -> Dict:
        """
        Triangle enumeration on a deterministic induced subgraph.

        Both execution paths run against an identical, smaller graph so
        their results agree and the relational 3-way self-join fits in
        memory.
        """
        from graph_engine import GraphDatabase, QueryExecutor

        # Build the induced subgraph deterministically.
        full_nodes = list(self.graph_db.nodes.keys())
        keep = set(full_nodes[:subgraph_size])
        sub = GraphDatabase()
        for n in keep:
            sub.add_node(n)
        for e in self.graph_db.edges:
            if e.source in keep and e.target in keep:
                sub.add_edge(e.source, e.target)
        sub_exec = QueryExecutor(sub)

        # Warm-up.
        sub_exec.execute_graph_algebra_query('pattern_match', {'pattern': 'triangle'})
        sub_exec.execute_relational_algebra_query('pattern_match', {'pattern': 'triangle'})

        gt, rt = [], []
        gr_res, re_res = None, None
        for _ in range(iterations):
            r1, s1 = sub_exec.execute_graph_algebra_query('pattern_match',
                                                          {'pattern': 'triangle'})
            r2, s2 = sub_exec.execute_relational_algebra_query('pattern_match',
                                                               {'pattern': 'triangle'})
            gt.append(s1['execution_time']); rt.append(s2['execution_time'])
            gr_res, re_res = r1, r2

        # Both should produce the same set of canonical triangles.
        gset = {tuple(sorted(t)) for t in gr_res}
        rset = {tuple(sorted(t)) for t in re_res}
        same = (gset == rset)

        gmean, rmean = float(np.mean(gt)), float(np.mean(rt))
        eps = 1e-9
        speedup = rmean / max(gmean, eps)
        return {
            'query_type': 'pattern_match',
            'iterations': iterations,
            'subgraph_size': subgraph_size,
            'subgraph_edges': sub.get_edge_count(),
            'triangles_found': len(gset),
            'results_match': same,
            'graph_algebra': self._stats(gt),
            'relational_algebra': self._stats(rt),
            'speedup_graph_vs_relational': speedup,
            'winner': 'graph_algebra' if speedup > 1 else 'relational_algebra',
            'note': ('Run on 500-node induced subgraph because relational '
                     '3-way self-join exhausts memory on the full 4,039-node graph.'),
        }

    # ------------------------------------------------------------------
    def comprehensive_benchmark(self) -> Dict:
        """Run the full benchmark suite."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK")
        print("=" * 60)

        nodes = list(self.graph_db.nodes.keys())
        if len(nodes) < 2:
            raise ValueError("need >=2 nodes")

        # Pick anchor node with non-trivial degree so neighbour queries
        # do meaningful work.
        anchor = max(nodes, key=lambda n: len(self.graph_db.get_neighbors(n)))
        far_target = nodes[min(100, len(nodes) - 1)]

        out = {
            'graph_statistics': self.graph_db.get_statistics(),
            'benchmarks': {},
        }

        out['benchmarks']['neighbors'] = self.benchmark_query_type(
            'neighbors', {'node_id': anchor, 'depth': 2}, iterations=10)

        out['benchmarks']['reachability'] = self.benchmark_query_type(
            'reachability', {'source': anchor, 'target': far_target}, iterations=10)

        out['benchmarks']['shortest_path'] = self.benchmark_query_type(
            'shortest_path', {'source': anchor, 'target': far_target}, iterations=10)

        out['benchmarks']['pattern_match'] = self._benchmark_triangles_on_subgraph(
            subgraph_size=500, iterations=3)

        out['scalability'] = self.scalability_test({'node_id': anchor},
                                                   depths=(1, 2, 3, 4))
        out['join_comparison'] = self.compare_join_algorithms(
            table_sizes=(100, 500, 1000, 2000))
        return out

    # ------------------------------------------------------------------
    def generate_performance_report(self, results: Dict) -> str:
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("PERFORMANCE BENCHMARK REPORT")
        lines.append("=" * 70)

        s = results['graph_statistics']
        lines.append("\nGraph Statistics")
        lines.append(f"  Nodes:        {s['num_nodes']:,}")
        lines.append(f"  Edges:        {s['num_edges']:,}")
        lines.append(f"  Avg Degree:   {s['avg_degree']:.2f}")
        lines.append(f"  Max Degree:   {s['max_degree']}")

        lines.append("\nQuery Performance (Graph vs Relational)")
        lines.append("-" * 70)
        for name, b in results['benchmarks'].items():
            gt = b['graph_algebra']['mean'] * 1000
            rt = b['relational_algebra']['mean'] * 1000
            sp = b['speedup_graph_vs_relational']
            winner = b['winner'].replace('_', ' ')
            match = "OK" if b['results_match'] else "MISMATCH"
            lines.append(f"\n  {name.upper()}  (results agree: {match})")
            lines.append(f"    Graph algebra      : {gt:9.3f} ms  (std {b['graph_algebra']['std']*1000:.3f})")
            lines.append(f"    Relational algebra : {rt:9.3f} ms  (std {b['relational_algebra']['std']*1000:.3f})")
            lines.append(f"    Speedup            : {sp:9.2f}x  ({winner} faster)")

        if 'scalability' in results:
            lines.append("\nScalability (neighbours, varying depth)")
            lines.append("-" * 70)
            sc = results['scalability']
            for g, r in zip(sc['graph_algebra'], sc['relational_algebra']):
                lines.append(f"  depth={g['depth']}: "
                             f"graph={g['time']*1000:7.2f} ms   "
                             f"relational={r['time']*1000:7.2f} ms")

        if 'join_comparison' in results:
            lines.append("\nJoin Algorithms")
            lines.append("-" * 70)
            jc = results['join_comparison']
            for i, size in enumerate(jc['table_sizes']):
                nl = jc['nested_loop'][i]['time'] * 1000
                hj = jc['hash_join'][i]['time'] * 1000
                sm = jc['sort_merge'][i]['time'] * 1000
                sp_h = jc['speedup_analysis']['hash_vs_nested'][i]
                sp_s = jc['speedup_analysis']['sortmerge_vs_nested'][i]
                lines.append(f"\n  size={size}")
                lines.append(f"    nested-loop : {nl:9.2f} ms  (1.00x baseline)")
                lines.append(f"    hash        : {hj:9.2f} ms  ({sp_h:6.2f}x)")
                lines.append(f"    sort-merge  : {sm:9.2f} ms  ({sp_s:6.2f}x)")
            mean_h = float(np.mean(jc['speedup_analysis']['hash_vs_nested']))
            mean_s = float(np.mean(jc['speedup_analysis']['sortmerge_vs_nested']))
            lines.append(f"\n  Mean speedup vs nested-loop:")
            lines.append(f"    hash       : {mean_h:.2f}x")
            lines.append(f"    sort-merge : {mean_s:.2f}x")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)