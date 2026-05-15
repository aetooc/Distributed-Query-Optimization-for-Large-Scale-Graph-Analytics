# Dual-Execution Graph Query Engine & Relational Benchmark

## Overview
This repository contains a custom, in-memory database execution engine engineered to benchmark the performance of graph-native traversals against traditional relational algebra operations (Joins, Selections). It was developed to analyze algorithmic scalability, execution bottlenecks, and system vulnerabilities within large-scale data processing pipelines.

The system features a modular architecture capable of processing both synthetically generated data for unit testing and large-scale, real-world datasets for stress testing and algorithmic complexity analysis.

## System Architecture
The project is built entirely in Python (utilizing Pandas for relational operations) to isolate and measure pure algorithmic complexity without the overhead of disk I/O or network latency from external databases. 

* `main.py`: The entry point that orchestrates data loading, security scanning, and benchmarking.
* `data_loader.py`: Data ingestion pipeline supporting both random synthetic generation and automated fetching/parsing of the SNAP Facebook dataset.
* `graph_engine.py`: Core in-memory graph database and dual-query executor.
* `join_algorithms.py`: Custom, from-scratch implementations of **Nested-Loop**, **Hash**, and **Sort-Merge** joins.
* `performance_benchmark.py`: Statistical measurement suite for execution times, standard deviations, and scalability (O(n) vs O(n³)).
* `vulnerability_analysis.py`: Automated security testing framework simulating adversarial queries.

## Performance Benchmarks 
The engine was stress-tested against the **Stanford Network Analysis Project (SNAP) Facebook Ego Network dataset**, successfully parsing and processing **4,039 nodes and 88,234 real-world edges** (Avg Degree: 21.85, Max Degree: 1043). 

Testing revealed massive performance advantages for graph-native algebra when handling highly connected data, primarily by bypassing the heavy cross-product computations required by relational engines on "super-nodes".


### Relational Join Algorithm Comparison
Compared to a baseline O(n²) Nested-Loop join, the optimized algebraic implementations yielded significant efficiency gains on large tabular data representations:
* **Hash Join:** **217.07x faster** than Nested-Loop.
* **Sort-Merge Join:** **67.54x faster** than Nested-Loop.

## Vulnerability & Security Analysis
The engine includes an automated security suite to evaluate system behavior under adversarial conditions:
* **Resource Exhaustion [PASS]:** System successfully mitigated intentional DoS attempts via deep-traversal memory limits.
* **Algorithmic Complexity Attacks [PASS]:** The engine remained stable under heavy data load during O(n³) queries without crashing the memory heap.
* **Cache Poisoning & Concurrency [PASS]:** Cache performance remained stable under simulated load.
* **Injection Attacks [FAIL]:** The framework successfully identified a medium-severity vulnerability regarding missing type-checking and input validation on node IDs. This intentional baseline test provides the exact requirements for the next iteration of input sanitization and security patching.

## Installation & Usage

### Prerequisites
* Python 3.8+
* `pandas`, `numpy`, `psutil`, `matplotlib`

### Setup
1. Clone the repository

2. Install dependencies:
`
pip install pandas numpy psutil matplotlib
`
3. Run the complete benchmark and security suite:
`
python main.py
`

Note: By default, main.py is configured to download and benchmark the SNAP Facebook dataset. To run faster debugging tests, swap the dataset configuration to DataLoader.generate_synthetic_graph() inside main.py.