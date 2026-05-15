
import json
from data_loader import DataLoader
from graph_engine import QueryExecutor
from performance_benchmark import PerformanceBenchmark
from vulnerability_analysis import VulnerabilityAnalyzer


def main(use_synthetic: bool = False):
    # 1. Load data
    if use_synthetic:
        db = DataLoader.generate_synthetic_graph(num_nodes=500)
    else:
        db = DataLoader.load_snap_facebook_graph()

    # 2. Build executor (materializes relational tables once)
    executor = QueryExecutor(db)

    # 3. Vulnerability suite
    print("\n" + "*" * 50)
    print("VULNERABILITY ANALYSIS")
    print("*" * 50)
    analyzer = VulnerabilityAnalyzer(db, executor)
    analyzer.run_all_tests()
    vuln_summary = analyzer.generate_report()
    print(f"\n  Tests passed: {vuln_summary['passed']}/{vuln_summary['total_tests']}")

    # 4. Performance benchmark
    print("\n" + "*" * 50)
    print("PERFORMANCE BENCHMARK")
    print("*" * 50)
    benchmark = PerformanceBenchmark(db, executor)
    results = benchmark.comprehensive_benchmark()
    print(benchmark.generate_performance_report(results))

    # 5. Persist machine-readable summary for the write-up
    with open("benchmark_results.json", "w") as f:
        json.dump({
            'graph_statistics': results['graph_statistics'],
            'benchmarks': {k: {kk: vv for kk, vv in v.items()
                               if kk != 'results_match'} | {'results_match': v['results_match']}
                           for k, v in results['benchmarks'].items()},
            'scalability': results['scalability'],
            'join_comparison': results['join_comparison'],
            'vulnerability': vuln_summary,
        }, f, indent=2, default=str)
    print("\n  Wrote benchmark_results.json")


if __name__ == "__main__":
    main(use_synthetic=False)