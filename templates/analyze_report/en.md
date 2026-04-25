<!-- BLOCK: EXECUTIVE_SUMMARY -->
This report captures the comparison of selected runs across quality and performance metrics. The objective is to provide a reproducible basis for model selection and deployment trade-offs.

<!-- BLOCK: INTRO -->
The analysis is built on a consistent artifact and metric set. The report is organized into engineering-focused blocks: overall quality, computational cost, per-class stability, and implementation-level recommendations.

<!-- BLOCK: QUALITY -->
This section describes which model maximizes the target metric, where degradation vs baseline occurs, and how stable results are across compared runs.

<!-- BLOCK: SPEED -->
This section describes the compute cost of quality: latency/FPS, efficiency per quality unit, and model positioning on the speed-vs-quality plane.

<!-- BLOCK: PER_CLASS -->
This section describes class-level quality heterogeneity and identifies AP/PR bottlenecks with the highest model-to-model variance.

<!-- BLOCK: CONCLUSION -->
Conclusion: model choice should follow the target operating profile and SLA constraints. For latency-critical production paths, prioritize predictable speed with acceptable quality; for quality-critical paths, prioritize metric maximum with controlled per-class degradation and bounded compute cost.
