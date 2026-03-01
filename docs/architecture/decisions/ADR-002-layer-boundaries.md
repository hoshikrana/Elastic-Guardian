# ADR-002: Layer Import Boundaries
Status: ACCEPTED
Rule: Layer N may ONLY import from layers < N.
Enforced at startup by Kahn's algorithm in dependency_dag.py.

| Layer | Package | Allowed imports |
|-------|---------|-----------------|
| 1 | core/ | stdlib only |
| 2 | infrastructure/ | L1 + torch + pynvml |
| 3 | intelligence/ | L1-2, no torch |
| 4 | orchestration/, resilience/ | L1-3 |
| 5 | training/, peft/, models/, export/, data/, monitoring/ | L1-4 + torch |
| 6 | runtime/ | L1-5 |
| 7 | api/, cli/ | L1-6 |
