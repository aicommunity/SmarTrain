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

<!-- BLOCK: SUB_CONTEXT_DATASET -->
This subsection lists datasets involved in the comparison using the report’s abbreviated legend. It provides a quick anchor for which data underlies the metrics below.

<!-- BLOCK: SUB_CONTEXT_MODELS -->
Next we enumerate the baseline run and candidates, plus tables for format aliases and eval settings. This defines the artifact namespace referenced in later sections.

<!-- BLOCK: SUB_QUALITY_GENERAL -->
This subsection collects quality summary tables: per-run aggregates, recomputed metrics, and optional baseline deltas. It lets you compare runs without digging into raw logs.

<!-- BLOCK: SUB_QUALITY_ENV -->
Here we compare training and test hardware/software context: CPU, GPU, memory, OS. This matters when interpreting speed differences and reproducibility across runs.

<!-- BLOCK: SUB_QUALITY_RUN_CARD -->
Below is a compact per-run card (model, formats, CPU/GPU/RAM/OS). Wide tables are omitted when fields are too sparse.

<!-- BLOCK: SUB_FORMAT_QUALITY -->
This subsection compares quality across exported formats (pt/onnx/engine, etc.) on aligned splits and metrics. The tables show which format is strongest on the selected fields.

<!-- BLOCK: SUB_FORMAT_DEEP_DIAG -->
A link to the baseline deep-diagnostics report: extra plots and checks go beyond the main table but help explain anomalies.

<!-- BLOCK: SUB_FORMAT_ISSUES -->
A compact summary of metric computation issues by split/format: reason codes, affected runs, and short notes. This separates pipeline problems from true model differences.

<!-- BLOCK: SUB_FORMAT_PERF -->
This subsection captures per-format performance on the test split: pure inference, full pipeline, and optional diagnostic columns. Measurements follow a consistent benchmarking methodology.

<!-- BLOCK: NARR_PERF_NOT_COLLECTED -->
If PT rows in the performance tables show “no data”, the corresponding test artifacts have no `perf_*.json` from model testing. Run `smartrain model test --collect-performance --run <path>` for each run. The inference benchmark from `analyze all` (speed comparison figure, `benchmark.csv`) measures PT on CPU only and does not populate format performance tables.

<!-- BLOCK: SUB_FORMAT_SPEED -->
Speed-related figures and companion tables (e.g., speed–quality trade-off). Visuals complement numeric tables and help pick a point on the empirical Pareto frontier.

<!-- BLOCK: SUB_PER_CLASS -->
Next we analyze per-class quality: where models disagree most and which classes have the lowest AP. Use it to prioritize labeling and targeted improvements.

<!-- BLOCK: SUB_ULTRA_RUN -->
This subsection lists Ultralytics test artifacts and images for a specific run: configuration, machine info, and CSV/PR links. It is a quick completeness audit of the test pass.

<!-- BLOCK: SUB_ULTRA_COMPLETENESS -->
Below is how complete the Ultralytics test artifact set is for this run and where files were resolved from (test-split vs train-ultralytics fallback).

<!-- BLOCK: SUB_ULTRA_PER_CLASS_TABLE -->
The table summarizes per-class AP from pr_per_class.csv for quick review without opening the source CSV.

<!-- BLOCK: SUB_CONCLUSION_MISSING -->
Below are missing artifacts and structured reasons from the session manifest. This is an explicit record of what could not be collected or recomputed—not a reporting error.

<!-- BLOCK: NARR_TAKEAWAY_NO_DATA -->
Not enough data in the table for numeric comparison or spread-based takeaways.

<!-- BLOCK: NARR_PREAMBLE_GENERIC -->
The table below contains numeric and text fields for the current artifact; column layout depends on the session profile.

<!-- BLOCK: NARR_PREAMBLE_ALIAS -->
This table maps a short format alias to a run and artifact path, helping you reference rows in later summaries.

<!-- BLOCK: NARR_PREAMBLE_EVAL -->
This table lists metric recomputation parameters (input size, conf/IoU thresholds, split) for each compared variant.

<!-- BLOCK: NARR_FIG_COMPARE -->
The plot compares key quality metrics across runs and formats; read it together with the metrics tables above for a consistent interpretation.

<!-- BLOCK: NARR_FIG_BENCHMARK -->
The bar chart shows relative inference speed across runs/models; see performance tables for exact numbers when available.

<!-- BLOCK: NARR_FIG_SPEED_MAP -->
The scatter plot shows the speed–quality trade-off between the chosen axes; see the speed–quality table when present.

<!-- BLOCK: NARR_FIG_PR -->
A PR or per-class visualization; use PR/per-class tables in the corresponding sections for quantitative comparisons.

<!-- BLOCK: NARR_FIG_DEFAULT -->
A session artifact figure; align with the caption and neighboring tables.
