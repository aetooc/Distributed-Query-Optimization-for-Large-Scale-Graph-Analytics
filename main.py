from data_loader import DataLoader
from graph_engine import QueryExecutor
from performance_benchmark import PerformanceBenchmark
from vulnerability_analysis import VulnerabilityAnalyzer

def main():
    # ==========================================
    # 1. DATASET
    # ==========================================
    
    # For fast testing and debugging:
    # db = DataLoader.generate_synthetic_graph(num_nodes=500)
    
    # For production benchmarking:
    db = DataLoader.load_snap_facebook_graph()

    # ==========================================
    # 2. INITIALIZE SYSTEM
    # ==========================================
    executor = QueryExecutor(db)
    
    # ==========================================
    # 3. RUN SECURITY SCANNER
    # ==========================================
    print("\n" + "*"*50)
    print("STARTING VULNERABILITY ANALYSIS")
    print("*"*50)
    analyzer = VulnerabilityAnalyzer(db, executor)
    analyzer.run_all_tests()
    
    # ==========================================
    # 4. RUN PERFORMANCE BENCHMARKS
    # ==========================================
    print("\n" + "*"*50)
    print("STARTING PERFORMANCE BENCHMARKS")
    print("*"*50)
    benchmark = PerformanceBenchmark(db, executor)
    results = benchmark.comprehensive_benchmark()
    
    # Print the final report
    print(benchmark.generate_performance_report(results))

if __name__ == "__main__":
    main()