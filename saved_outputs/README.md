# Saved Outputs

This directory is an archive of selected run folders that were originally written under `outputs/` and later copied here for reporting/reference.

Because of that, the inner `config.json`, `run_summary.json`, and `run.log` files still record their original `outputs/...` paths. The runs were not regenerated inside `saved_outputs`; this folder is a preserved snapshot.

## How the data files were generated

All archived runs ultimately depend on the same preprocessed inputs created by `data_processing/data_processing.ipynb` and saved under `data_processing/data/`:

- `mhc_class1.csv`: cleaned combined MHC class-I table used by the predictor.
- `mhc_class1_ms_balanced.npz`: AR-ready balanced NPZ derived from the MHC data; used by the explainer and generator training.
- `iedb.npz`: cleaned held-out IEDB class-I NPZ used for external generation tests.

The notebook generates these systematically by:

1. Cleaning and standardizing the combined MHC / IEDB sources.
2. Filtering to the class-I workflow and valid 8-11mer peptide inputs downstream.
3. Removing duplicates and normalizing allele labels.
4. Removing overlaps between the held-out IEDB set and the MHC train/validation pool.
5. Exporting the reusable CSV/NPZ files consumed by the shell workflows.

The executable workflows do not choose arbitrary inputs at runtime. They read fixed repo-default paths from `scripts/common.sh`:

```bash
MHC_CSV="data_processing/data/mhc_class1.csv"
MHC_NPZ="data_processing/data/mhc_class1_ms_balanced.npz"
IEDB_NPZ="data_processing/data/iedb.npz"
```

So the `saved_outputs` folders were generated systematically from those three prepared inputs, not from hand-picked files each time.

## Naming Convention

- Top-level saved folder: original `outputs/<stage>_<run_id>` directory copied into this archive.
- Timestamped child folder: generator execution timestamp created by `generator/pipeline.py`.
- `alleles: "all"` in a generator config means "take the top `max_alleles` most frequent alleles from the input dataset"; the default here is `10`.

Example:

- `generator_full_10alleles_20260328_1545` is the workflow-level output directory.
- `generator_full_10alleles_20260328_1545/20260329_064608` is the actual generator run created inside it.

## Workflow 1: Full Pipeline 

Driver script:

```bash
bash scripts/run_full_pipeline.sh --run-id <run_id>
```

Systematic generation chain:

1. Predictor: `prediction-model/run_netmhcpan_mhc1.py` scores `mhc_class1.csv` with NetMHCpan and writes predictor artifacts.
2. Explainer: `explainer/shap.py` reads `mhc_class1_ms_balanced.npz` and exports `shap_results.json` / `shap_heatmaps.npz`.
3. Generator training: `generator/generate.py` trains on `mhc_class1_ms_balanced.npz` using the explainer SHAP JSON.
4. External test: `generator/generate.py` reuses the trained `ar_model.pt` and evaluates on `iedb.npz` using the default 10-allele IEDB panel.

Archived bundle for this workflow:

- `predictor_full_10alleles_20260328_1545`
- `explainer_full_10alleles_20260328_1545`
- `generator_full_10alleles_20260328_1545`
- `iedb_test_full_10alleles_20260328_1545`

What the saved configs show:

- Generator training used `data_processing/data/mhc_class1_ms_balanced.npz`.
- IEDB testing used `data_processing/data/iedb.npz`.
- The generator consumed `outputs/explainer_full_10alleles_20260328_1545/shap_results.json`.
- The IEDB test reused the trained checkpoint from the matching generator run.

Alleles used in the archived full-pipeline bundle:

- Generator training (`generator_full_10alleles_20260328_1545` with `alleles=all`, resolved from `run.log`):
  `HLA-B*07:02`, `HLA-A*03:01`, `HLA-B*40:01`, `HLA-B*57:01`, `HLA-B*15:02`, `HLA-A*11:01`, `HLA-B*51:01`, `HLA-A*01:01`, `HLA-B*40:02`, `HLA-A*24:02`.
- External IEDB test (`iedb_test_full_10alleles_20260328_1545`):
  `HLA-A*02:01`, `HLA-A*03:01`, `HLA-A*24:02`, `HLA-B*07:02`, `HLA-B*44:02`, `HLA-B*57:01`, `HLA-A*01:01`, `HLA-B*08:01`, `HLA-B*27:05`, `HLA-B*58:01`.

## Workflow 2: No-XAI Ablation Bundle

Driver script:

```bash
bash scripts/run_no_xai_ablation.sh --run-id <run_id>
```

Systematic generation chain:

1. Train the generator on `mhc_class1_ms_balanced.npz`.
2. Disable SHAP guidance by running with `--refine_mode without_shap` and no `shap_json`.
3. Reuse the resulting `ar_model.pt`.
4. Test on `iedb.npz`.
5. When the test config says `alleles: "all"` with `max_alleles: 10`, the generator evaluates the top 10 most frequent alleles in `iedb.npz`.

Archived folders from this workflow:

- `generator_no_xai_ablation_20260323_0637`
- `iedb_test_no_xai_ablation_20260323_0637`
- `generator_no_xai_ablation_20260328_095726`

Notes:

- `generator_no_xai_ablation_20260328_095726` is a training snapshot only; it contains the AR checkpoint and training metadata but no paired `iedb_test_*` folder in this archive.

Alleles used in the archived no-XAI runs:

- Generator training (`generator_no_xai_ablation_20260323_0637`, resolved from `run.log`):
  `HLA-A*02:01`, `HLA-B*27:05`, `HLA-B*07:02`, `HLA-A*03:01`, `HLA-B*57:01`, `HLA-B*15:02`, `HLA-B*40:01`, `HLA-B*51:01`, `HLA-B*40:02`, `HLA-A*11:01`.
- IEDB test (`iedb_test_no_xai_ablation_20260323_0637`, `alleles=all` resolved from `run.log`):
  `HLA-A*02:01`, `HLA-C*04:01`, `HLA-A*01:01`, `HLA-A*24:02`, `HLA-B*57:01`, `HLA-A*02:02`, `HLA-A*02:03`, `HLA-A*02:04`, `HLA-A*02:05`, `HLA-A*02:06`.
- Partial training snapshot (`generator_no_xai_ablation_20260328_095726`, resolved from `run.log`):
  `HLA-A*02:01`, `HLA-B*27:05`, `HLA-B*07:02`, `HLA-A*03:01`, `HLA-B*57:01`, `HLA-B*15:02`, `HLA-B*40:01`, `HLA-B*51:01`, `HLA-B*40:02`, `HLA-A*11:01`.

## Follow-On No-SHAP Evaluation Folders

These folders were generated systematically from an existing no-SHAP checkpoint plus `iedb.npz`. They are not separate data-preparation pipelines.

- `iedb_benchmark_refine_without_shap_20260323`
  Uses `iedb.npz` + checkpoint `generator_no_xai_ablation_20260323_0637/.../ar_model.pt` + allele `HLA-A*02:01` + `candidate_pool_size=150`.
- `iedb_expanded_panel_without_shap_20260323`
  Uses `iedb.npz` + the same checkpoint + the 10-allele panel `HLA-A*02:01,HLA-A*03:01,HLA-A*24:02,HLA-B*07:02,HLA-B*44:02,HLA-B*57:01,HLA-A*01:01,HLA-B*08:01,HLA-B*27:05,HLA-B*58:01` + `candidate_pool_size=300` + `length_weights=8:1,9:6,10:2,11:1`.
- `iedb_test_selected_panel`
  Uses `iedb.npz` + the newer checkpoint `generator_no_xai_ablation_20260328_095726/.../ar_model.pt` + the same 10-allele panel, still with `refine_mode=without_shap`.

In other words, these folders are parameter variations of the same generator inference stage, using saved checkpoints as inputs.

Exact alleles in these follow-on evaluation folders:

- `iedb_benchmark_refine_without_shap_20260323`:
  `HLA-A*02:01`.
- `iedb_expanded_panel_without_shap_20260323`:
  `HLA-A*02:01`, `HLA-A*03:01`, `HLA-A*24:02`, `HLA-B*07:02`, `HLA-B*44:02`, `HLA-B*57:01`, `HLA-A*01:01`, `HLA-B*08:01`, `HLA-B*27:05`, `HLA-B*58:01`.
- `iedb_test_selected_panel`:
  `HLA-A*02:01`, `HLA-A*03:01`, `HLA-A*24:02`, `HLA-B*07:02`, `HLA-B*44:02`, `HLA-B*57:01`, `HLA-A*01:01`, `HLA-B*08:01`, `HLA-B*27:05`, `HLA-B*58:01`.

## Folder Contents

- Predictor folders:
  `mhc1_netmhcpan_predictions.csv`, `mhc1_netmhcpan_summary.csv`, `run.log`.
- Some predictor archives may also contain `shap_results.json` and `shap_heatmaps.npz`, because `prediction-model/run_netmhcpan_mhc1.py` includes its own SHAP export step.
- Explainer folders:
  `shap_results.json`, `shap_heatmaps.npz`, `run.log`.
- Generator / IEDB folders:
  `config.json`, `run_summary.json`, `run.log`, per-allele `final_peptides_div.csv`, `final_peptides_nodiv.csv`, `summary_div.json`, `summary_nodiv.json`, and `ar_model.pt` for training runs.

## Historical Extra Folder
`predictor_full_mhcflurry_20260323_0350` is a standalone historical predictor run log. It reflects a direct `prediction-model/run_netmhcpan_mhc1.py` execution on `data_processing/data/mhc_class1.csv`, separate from the two main archived workflow bundles above.

## How To Read A Run Quickly

1. Open `config.json` to see the exact input files and parameters used.
2. Open `run_summary.json` for aggregate results.
3. Inspect the per-allele `final_peptides_div.csv` files for final candidates.
4. Use `run.log` for runtime details and failure/debug context.
