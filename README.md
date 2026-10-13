# perturbation-status-metrics

Omnibenchmark collector for the complete data-only perturbation-status
benchmark. It compares Mixscape, Mixscale and PS, computes the target-expression
reference transiently inside the collector, and reproduces the current coverage,
direct-expression, transcriptome-effect, implementation-reproducibility,
agreement and resource analyses. The frozen NTC-free direct-expression cell
table is versioned in this repository. Figure generation is intentionally kept
outside the benchmark module.

The complete benchmark framework, target-expression reference implementation,
and their dataset configuration are owned by this repository. At runtime the
collector uses only upstream Omnibenchmark artifacts and its active software
environment; it does not call code or environments from the development tree.
