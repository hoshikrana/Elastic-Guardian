# ADR-001: DSA Structure Selection
Status: ACCEPTED

| # | Structure | File | Key Operation | Justification |
|---|-----------|------|---------------|---------------|
| 1 | Fibonacci Heap | intelligence/strategy/selector.py | O(1) decrease-key | Live re-score under pressure |
| 2 | Red-Black Tree | intelligence/estimator/calibration/store.py | O(log n) range | Similar-hardware lookup |
| 3 | Skip List | orchestration/pressure/monitor.py | O(log n) lock-free | Concurrent event log |
| 4 | Segment Tree | intelligence/estimator/dryrun.py | O(log n) range-max | Measurement window peaks |
| 5 | Dijkstra | intelligence/graph/topology_graph.py | O((V+E)logV) | Min-latency tensor routing |
| 6 | Kahn's BFS | intelligence/graph/dependency_dag.py | O(V+E) | Import cycle detection |
| 7 | Binary Search | intelligence/strategy/batch_optimizer.py | O(log n) | Monotone VRAM predicate |
| 8 | Trie | runtime/config_loader.py | O(k) | Prefix queries + autocomplete |
