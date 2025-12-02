Noise Windows Diagnostics

Inputs:
- windows_dir: exact_data/anomaly
- labels_csv : outputs/dbscan_eps2.1_run/labels_per_window.csv
- noise_label: -1

Key outputs (in outputs/noise_diagnostics):
- figs/topk_noise_bar.png
- figs/noise_heatmap_topk.png
- figs/windows_pca.png
- figs/windows_umap.png (if umap available)
- figs/top_feature_distributions.png
- tables/noise_windows_summary.csv
- matrices/window_features.parquet
- matrices/window_features_z.parquet
