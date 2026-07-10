# ALLClear Batch Evaluation

- Runs: `1`
- Root: `outputs/allclear`
- Branch eval: `True`
- Split visuals: `True`

## Run Summary

| run | framework | epochs | checkpoint | latest_val_recon_total | anomalies | visuals |
|---|---|---:|---|---:|---:|---:|
| 2026-07-05T15-10-13_stage1__dadigan_lama_ffc_omni_lossbalanced_v1 | dadigan_baseline | 50 | yes | 4.911534601991827 | 5 | 0/50 |

## Output Files

- `runs_summary.csv`: one row per run.
- `metric_summary_all.csv`: train/val loss statistics from every run.
- `loss_contributions_all.csv`: weighted contribution ratios for every loss term.
- `generalization_gaps_all.csv`: train/val gap table.
- `visualization_inventory_all.csv`: saved low/medium/high/heavy visualization coverage.
- `visual_candidate_comparison_all.csv`: image-panel proxy metrics extracted from saved visual grids.
- `branch_metrics_all.csv`: checkpoint-based val/test metrics by candidate and region when branch eval is enabled.
- `<run>/log_visual/report.md`: detailed per-run log and visualization diagnosis.
