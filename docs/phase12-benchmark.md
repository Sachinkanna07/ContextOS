# Phase 12 terminal latency

Measured on Windows 11, Python 3.13.5, using `python -m contextos.benchmarks.terminal`.
The run used a temporary SQLite database, deterministic local embedding, one
ingested memory, and five sequential samples per operation. HTTP requests used
an in-process ASGI transport; CLI startup used a new Python process each time.
This is local development evidence, not a production or provider benchmark.

| Operation | Median | Maximum |
| --- | ---: | ---: |
| CLI startup (`contextos version`) | 1587.205 ms | 1594.773 ms |
| Dashboard refresh | 5.817 ms | 2019.819 ms |
| Telemetry summary query | 2.475 ms | 7.070 ms |
| Memory search | 5.631 ms | 8.801 ms |
| Monitor request overhead above telemetry query | 3.441 ms | 2012.748 ms |

The cold dashboard sample includes model discovery; model availability is cached
for up to ten seconds during subsequent refreshes. The overhead figure is the
dashboard request duration minus the paired telemetry query duration; it is
not CPU utilization. No token savings or answer quality figures are inferred
from these latency measurements.
