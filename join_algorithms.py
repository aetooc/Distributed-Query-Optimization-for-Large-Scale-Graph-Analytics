"""
Join Algorithm Implementations
Comparison of different join strategies for relational algebra operations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import time
from collections import defaultdict

class JoinAlgorithms:
    """
    Implementation of different join algorithms for performance comparison
    """
    
    @staticmethod
    def nested_loop_join(left_df: pd.DataFrame, right_df: pd.DataFrame,
                        left_key: str, right_key: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Nested Loop Join - O(n*m) complexity
        Simple but inefficient for large datasets
        """
        start_time = time.time()
        result_rows = []
        comparisons = 0
        
        for _, left_row in left_df.iterrows():
            for _, right_row in right_df.iterrows():
                comparisons += 1
                if left_row[left_key] == right_row[right_key]:
                    merged_row = {**left_row.to_dict(), **right_row.to_dict()}
                    result_rows.append(merged_row)
        
        result_df = pd.DataFrame(result_rows)
        execution_time = time.time() - start_time
        
        stats = {
            'algorithm': 'nested_loop_join',
            'execution_time': execution_time,
            'comparisons': comparisons,
            'left_size': len(left_df),
            'right_size': len(right_df),
            'result_size': len(result_df)
        }
        
        return result_df, stats
    
    @staticmethod
    def hash_join(left_df: pd.DataFrame, right_df: pd.DataFrame,
                 left_key: str, right_key: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Hash Join - O(n+m) average complexity
        Build hash table on smaller relation, probe with larger
        """
        start_time = time.time()
        
        # Determine which table to hash (typically the smaller one)
        if len(left_df) <= len(right_df):
            build_df, probe_df = left_df, right_df
            build_key, probe_key = left_key, right_key
            build_is_left = True
        else:
            build_df, probe_df = right_df, left_df
            build_key, probe_key = right_key, left_key
            build_is_left = False
        
        # Build phase: create hash table
        hash_table = defaultdict(list)
        for idx, row in build_df.iterrows():
            hash_table[row[build_key]].append(row.to_dict())
        
        build_time = time.time() - start_time
        
        # Probe phase: look up in hash table
        result_rows = []
        comparisons = 0
        
        for _, probe_row in probe_df.iterrows():
            key_value = probe_row[probe_key]
            if key_value in hash_table:
                for build_row in hash_table[key_value]:
                    comparisons += 1
                    if build_is_left:
                        merged_row = {**build_row, **probe_row.to_dict()}
                    else:
                        merged_row = {**probe_row.to_dict(), **build_row}
                    result_rows.append(merged_row)
        
        result_df = pd.DataFrame(result_rows)
        total_time = time.time() - start_time
        
        stats = {
            'algorithm': 'hash_join',
            'execution_time': total_time,
            'build_time': build_time,
            'probe_time': total_time - build_time,
            'comparisons': comparisons,
            'hash_table_size': len(hash_table),
            'left_size': len(left_df),
            'right_size': len(right_df),
            'result_size': len(result_df)
        }
        
        return result_df, stats
    
    @staticmethod
    def sort_merge_join(left_df: pd.DataFrame, right_df: pd.DataFrame,
                       left_key: str, right_key: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Sort-Merge Join - O(n log n + m log m) complexity
        Sort both relations, then merge
        """
        start_time = time.time()
        
        # Sort phase
        left_sorted = left_df.sort_values(by=left_key).reset_index(drop=True)
        right_sorted = right_df.sort_values(by=right_key).reset_index(drop=True)
        sort_time = time.time() - start_time
        
        # Merge phase
        result_rows = []
        i, j = 0, 0
        comparisons = 0
        
        while i < len(left_sorted) and j < len(right_sorted):
            comparisons += 1
            left_val = left_sorted.iloc[i][left_key]
            right_val = right_sorted.iloc[j][right_key]
            
            if left_val == right_val:
                # Handle duplicates: all matching rows
                j_start = j
                while j < len(right_sorted) and right_sorted.iloc[j][right_key] == left_val:
                    merged_row = {**left_sorted.iloc[i].to_dict(), 
                                 **right_sorted.iloc[j].to_dict()}
                    result_rows.append(merged_row)
                    j += 1
                
                # Check if next left row also matches
                if i + 1 < len(left_sorted) and left_sorted.iloc[i + 1][left_key] == left_val:
                    i += 1
                    j = j_start  # Reset right pointer for next left match
                else:
                    i += 1
            elif left_val < right_val:
                i += 1
            else:
                j += 1
        
        result_df = pd.DataFrame(result_rows)
        total_time = time.time() - start_time
        
        stats = {
            'algorithm': 'sort_merge_join',
            'execution_time': total_time,
            'sort_time': sort_time,
            'merge_time': total_time - sort_time,
            'comparisons': comparisons,
            'left_size': len(left_df),
            'right_size': len(right_df),
            'result_size': len(result_df)
        }
        
        return result_df, stats
    
    @staticmethod
    def compare_all_joins(left_df: pd.DataFrame, right_df: pd.DataFrame,
                         left_key: str, right_key: str) -> Dict[str, Dict]:
        """
        Run all three join algorithms and compare performance
        """
        results = {}
        
        # Nested Loop Join
        _, nested_stats = JoinAlgorithms.nested_loop_join(
            left_df.copy(), right_df.copy(), left_key, right_key
        )
        results['nested_loop'] = nested_stats
        
        # Hash Join
        _, hash_stats = JoinAlgorithms.hash_join(
            left_df.copy(), right_df.copy(), left_key, right_key
        )
        results['hash'] = hash_stats
        
        # Sort-Merge Join
        _, sort_merge_stats = JoinAlgorithms.sort_merge_join(
            left_df.copy(), right_df.copy(), left_key, right_key
        )
        results['sort_merge'] = sort_merge_stats
        
        # Calculate speedups
        baseline_time = nested_stats['execution_time']
        results['speedup_analysis'] = {
            'hash_vs_nested': baseline_time / hash_stats['execution_time'],
            'sortmerge_vs_nested': baseline_time / sort_merge_stats['execution_time'],
            'hash_vs_sortmerge': sort_merge_stats['execution_time'] / hash_stats['execution_time']
        }
        
        return results

class IndexedJoin:
    """
    Index-based join optimization
    """
    
    def __init__(self):
        self.indexes = {}
    
    def build_index(self, df: pd.DataFrame, key_column: str) -> Dict:
        """Build B-tree style index on a column"""
        start_time = time.time()
        index = defaultdict(list)
        
        for idx, row in df.iterrows():
            index[row[key_column]].append(idx)
        
        build_time = time.time() - start_time
        self.indexes[key_column] = index
        
        return {
            'build_time': build_time,
            'index_size': len(index),
            'total_entries': sum(len(v) for v in index.values())
        }
    
    def indexed_join(self, left_df: pd.DataFrame, right_df: pd.DataFrame,
                    left_key: str, right_key: str, 
                    use_left_index: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Join using pre-built index
        """
        start_time = time.time()
        
        if use_left_index:
            if left_key not in self.indexes:
                raise ValueError(f"No index built for {left_key}")
            index = self.indexes[left_key]
            indexed_df, probe_df = left_df, right_df
            index_key, probe_key = left_key, right_key
        else:
            if right_key not in self.indexes:
                raise ValueError(f"No index built for {right_key}")
            index = self.indexes[right_key]
            indexed_df, probe_df = right_df, left_df
            index_key, probe_key = right_key, left_key
        
        result_rows = []
        lookups = 0
        
        for _, probe_row in probe_df.iterrows():
            key_value = probe_row[probe_key]
            if key_value in index:
                lookups += 1
                for idx in index[key_value]:
                    indexed_row = indexed_df.iloc[idx]
                    if use_left_index:
                        merged_row = {**indexed_row.to_dict(), **probe_row.to_dict()}
                    else:
                        merged_row = {**probe_row.to_dict(), **indexed_row.to_dict()}
                    result_rows.append(merged_row)
        
        result_df = pd.DataFrame(result_rows)
        execution_time = time.time() - start_time
        
        stats = {
            'algorithm': 'indexed_join',
            'execution_time': execution_time,
            'index_lookups': lookups,
            'result_size': len(result_df)
        }
        
        return result_df, stats
