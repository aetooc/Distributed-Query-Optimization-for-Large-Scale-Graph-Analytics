import pandas as pd
import numpy as np
from typing import Dict, Tuple
import time
from collections import defaultdict


class JoinAlgorithms:
    """Three equi-join algorithms implemented from scratch."""

    # ------------------------------------------------------------------
    @staticmethod
    def nested_loop_join(left_df: pd.DataFrame, right_df: pd.DataFrame,
                         left_key: str, right_key: str) -> Tuple[pd.DataFrame, Dict]:
        """Textbook O(n*m) nested-loop join."""
        left_cols = list(left_df.columns)
        right_cols = list(right_df.columns)
        left_arr = left_df.to_numpy()
        right_arr = right_df.to_numpy()
        lkey_idx = left_cols.index(left_key)
        rkey_idx = right_cols.index(right_key)

        start = time.perf_counter()
        result_rows = []
        comparisons = 0
        for i in range(len(left_arr)):
            lk = left_arr[i, lkey_idx]
            for j in range(len(right_arr)):
                comparisons += 1
                if lk == right_arr[j, rkey_idx]:
                    result_rows.append(np.concatenate([left_arr[i], right_arr[j]]))
        elapsed = time.perf_counter() - start

        cols = left_cols + [c + '_r' if c in left_cols else c for c in right_cols]
        result_df = (pd.DataFrame(result_rows, columns=cols) if result_rows
                     else pd.DataFrame(columns=cols))

        return result_df, {
            'algorithm': 'nested_loop_join',
            'execution_time': elapsed,
            'comparisons': comparisons,
            'left_size': len(left_df),
            'right_size': len(right_df),
            'result_size': len(result_df),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def hash_join(left_df: pd.DataFrame, right_df: pd.DataFrame,
                  left_key: str, right_key: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Classic in-memory hash join. Build phase on the smaller relation,
        probe phase on the larger.
        """
        # Choose build vs probe.
        if len(left_df) <= len(right_df):
            build_df, probe_df = left_df, right_df
            build_key, probe_key = left_key, right_key
            build_is_left = True
        else:
            build_df, probe_df = right_df, left_df
            build_key, probe_key = right_key, left_key
            build_is_left = False

        build_cols = list(build_df.columns)
        probe_cols = list(probe_df.columns)
        build_arr = build_df.to_numpy()
        probe_arr = probe_df.to_numpy()
        b_key_idx = build_cols.index(build_key)
        p_key_idx = probe_cols.index(probe_key)

        start = time.perf_counter()

        # Build phase: hash table maps key -> list of row indices into build_arr.
        hash_table: Dict = defaultdict(list)
        for i in range(len(build_arr)):
            hash_table[build_arr[i, b_key_idx]].append(i)
        build_time = time.perf_counter() - start

        # Probe phase.
        result_rows = []
        probes = 0
        matches = 0
        for j in range(len(probe_arr)):
            probes += 1
            bucket = hash_table.get(probe_arr[j, p_key_idx])
            if bucket:
                for i in bucket:
                    matches += 1
                    if build_is_left:
                        result_rows.append(np.concatenate([build_arr[i], probe_arr[j]]))
                    else:
                        result_rows.append(np.concatenate([probe_arr[j], build_arr[i]]))
        elapsed = time.perf_counter() - start

        # Column naming: preserve left/right order regardless of build/probe.
        if build_is_left:
            cols = build_cols + [c + '_r' if c in build_cols else c for c in probe_cols]
        else:
            cols = probe_cols + [c + '_r' if c in probe_cols else c for c in build_cols]
        result_df = (pd.DataFrame(result_rows, columns=cols) if result_rows
                     else pd.DataFrame(columns=cols))

        return result_df, {
            'algorithm': 'hash_join',
            'execution_time': elapsed,
            'build_time': build_time,
            'probe_time': elapsed - build_time,
            'probes': probes,
            'matches': matches,
            'hash_table_size': len(hash_table),
            'left_size': len(left_df),
            'right_size': len(right_df),
            'result_size': len(result_df),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def sort_merge_join(left_df: pd.DataFrame, right_df: pd.DataFrame,
                        left_key: str, right_key: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Sort-merge join. Sorts both relations on the join key, then performs
        a single linear merge pass. Correctly handles many-to-many duplicate
        keys by emitting the Cartesian product of each matching group.
        """
        left_cols = list(left_df.columns)
        right_cols = list(right_df.columns)
        lkey_idx = left_cols.index(left_key)
        rkey_idx = right_cols.index(right_key)

        start = time.perf_counter()

        # Sort phase: argsort on the join key, then reorder rows.
        left_arr = left_df.to_numpy()
        right_arr = right_df.to_numpy()
        left_order = np.argsort(left_arr[:, lkey_idx], kind='mergesort')
        right_order = np.argsort(right_arr[:, rkey_idx], kind='mergesort')
        left_sorted = left_arr[left_order]
        right_sorted = right_arr[right_order]
        sort_time = time.perf_counter() - start

        # Merge phase with proper duplicate-group handling.
        result_rows = []
        comparisons = 0
        i, j = 0, 0
        n_l, n_r = len(left_sorted), len(right_sorted)
        while i < n_l and j < n_r:
            comparisons += 1
            lv = left_sorted[i, lkey_idx]
            rv = right_sorted[j, rkey_idx]
            if lv < rv:
                i += 1
            elif lv > rv:
                j += 1
            else:
                # Equal -- find full run on both sides, emit Cartesian product.
                i_end = i
                while i_end < n_l and left_sorted[i_end, lkey_idx] == lv:
                    i_end += 1
                j_end = j
                while j_end < n_r and right_sorted[j_end, rkey_idx] == rv:
                    j_end += 1
                for ii in range(i, i_end):
                    for jj in range(j, j_end):
                        result_rows.append(
                            np.concatenate([left_sorted[ii], right_sorted[jj]])
                        )
                i, j = i_end, j_end

        elapsed = time.perf_counter() - start
        cols = left_cols + [c + '_r' if c in left_cols else c for c in right_cols]
        result_df = (pd.DataFrame(result_rows, columns=cols) if result_rows
                     else pd.DataFrame(columns=cols))

        return result_df, {
            'algorithm': 'sort_merge_join',
            'execution_time': elapsed,
            'sort_time': sort_time,
            'merge_time': elapsed - sort_time,
            'comparisons': comparisons,
            'left_size': len(left_df),
            'right_size': len(right_df),
            'result_size': len(result_df),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def compare_all_joins(left_df: pd.DataFrame, right_df: pd.DataFrame,
                          left_key: str, right_key: str) -> Dict[str, Dict]:
        """Run all three join algorithms and report speedups vs. nested-loop."""
        results = {}
        _, results['nested_loop'] = JoinAlgorithms.nested_loop_join(
            left_df.copy(), right_df.copy(), left_key, right_key)
        _, results['hash'] = JoinAlgorithms.hash_join(
            left_df.copy(), right_df.copy(), left_key, right_key)
        _, results['sort_merge'] = JoinAlgorithms.sort_merge_join(
            left_df.copy(), right_df.copy(), left_key, right_key)

        # Sanity check: all three should produce the same number of output rows
        # for the same input. Surface mismatches loudly rather than silently.
        sizes = (results['nested_loop']['result_size'],
                 results['hash']['result_size'],
                 results['sort_merge']['result_size'])
        results['result_size_consistent'] = (sizes[0] == sizes[1] == sizes[2])
        results['result_sizes'] = sizes

        baseline = results['nested_loop']['execution_time']
        results['speedup_analysis'] = {
            'hash_vs_nested': baseline / results['hash']['execution_time'],
            'sortmerge_vs_nested': baseline / results['sort_merge']['execution_time'],
            'hash_vs_sortmerge': (results['sort_merge']['execution_time']
                                  / results['hash']['execution_time']),
        }
        return results


class IndexedJoin:
    """Index-nested-loop join (hash index built once, reused across joins)."""

    def __init__(self):
        self.indexes: Dict[str, Dict] = {}

    def build_index(self, df: pd.DataFrame, key_column: str) -> Dict:
        """Build a hash index on a column."""
        arr = df[key_column].to_numpy()
        start = time.perf_counter()
        index: Dict = defaultdict(list)
        for i, k in enumerate(arr):
            index[k].append(i)
        build_time = time.perf_counter() - start
        self.indexes[key_column] = index
        return {
            'build_time': build_time,
            'distinct_keys': len(index),
            'total_entries': sum(len(v) for v in index.values()),
        }

    def indexed_join(self, left_df: pd.DataFrame, right_df: pd.DataFrame,
                     left_key: str, right_key: str,
                     use_left_index: bool = True) -> Tuple[pd.DataFrame, Dict]:
        if use_left_index:
            if left_key not in self.indexes:
                raise ValueError(f"No index built for {left_key}")
            index = self.indexes[left_key]
            indexed_df, probe_df = left_df, right_df
            probe_key = right_key
        else:
            if right_key not in self.indexes:
                raise ValueError(f"No index built for {right_key}")
            index = self.indexes[right_key]
            indexed_df, probe_df = right_df, left_df
            probe_key = left_key

        idx_cols = list(indexed_df.columns)
        probe_cols = list(probe_df.columns)
        idx_arr = indexed_df.to_numpy()
        probe_arr = probe_df.to_numpy()
        p_key_idx = probe_cols.index(probe_key)

        start = time.perf_counter()
        result_rows = []
        lookups = 0
        for j in range(len(probe_arr)):
            lookups += 1
            bucket = index.get(probe_arr[j, p_key_idx])
            if bucket:
                for i in bucket:
                    if use_left_index:
                        result_rows.append(np.concatenate([idx_arr[i], probe_arr[j]]))
                    else:
                        result_rows.append(np.concatenate([probe_arr[j], idx_arr[i]]))
        elapsed = time.perf_counter() - start

        if use_left_index:
            cols = idx_cols + [c + '_r' if c in idx_cols else c for c in probe_cols]
        else:
            cols = probe_cols + [c + '_r' if c in probe_cols else c for c in idx_cols]
        result_df = (pd.DataFrame(result_rows, columns=cols) if result_rows
                     else pd.DataFrame(columns=cols))

        return result_df, {
            'algorithm': 'indexed_join',
            'execution_time': elapsed,
            'lookups': lookups,
            'result_size': len(result_df),
        }