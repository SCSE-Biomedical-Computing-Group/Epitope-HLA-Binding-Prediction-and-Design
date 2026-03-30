# HLA Class I Peptide Generator

Main CLI: `python generator/generate.py`

## Usage

Train an AR model and generate peptides for specific alleles:

```bash
python generator/generate.py \
    --data data/mhc_ab.npz \
    --shap_json explainer/shap_results.json \
    --out_dir outputs/run1 \
    --alleles HLA-A*02:01,HLA-B*07:02 \
    --num_final 50 \
    --device mps
```

Run with a fuller configuration:

```bash
python generator/generate.py \
    --data data/mhc_ab.npz \
    --shap_json explainer/shap_results.json \
    --out_dir outputs/run1 \
    --alleles HLA-A*02:01 \
    --num_final 50 \
    --ar_epochs 50 \
    --temperature 1.0 \
    --top_p 0.9 \
    --enable_refinement \
    --refine_steps 15 \
    --length_quotas \
    --seed 42 \
    --device mps
```

Skip AR training and reuse an existing checkpoint:

```bash
python generator/generate.py \
    --data data/mhc_ab.npz \
    --ar_ckpt generator/checkpoints/ar_transformer_latest.pt \
    --out_dir outputs/run2 \
    --alleles all \
    --num_final 50
```

## Features

1. Transformer-based autoregressive (AR) peptide generator
2. NetMHCpan 4.2 based scoring with no surrogate model
3. SHAP-guided local refinement using position importance weights
4. Diversity filtering with edit distance and k-mer Jaccard
5. Length quotas (`8:10%`, `9:40%`, `10:30%`, `11:20%`) or proportional selection

## Outputs

Per-allele outputs:

```text
outputs/<run_id>/<allele>/
    final_peptides.csv  - Generated peptides with scores
    summary.json        - Statistics and motif analysis
```

Run-level outputs:

```text
outputs/<run_id>/
    run_summary.json    - Overall run configuration and results
    ar_model.pt         - Trained AR model (if training was performed)
```

Reusable checkpoints:

```text
generator/checkpoints/
    ar_transformer_latest.pt         - Reusable checkpoint preset
    ar_transformer_latest.meta.json  - Metadata for the reusable checkpoint
```
