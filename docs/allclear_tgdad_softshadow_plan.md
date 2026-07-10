# ALLClear Stage1 Restoration Plan

## Data Flow

Inputs:

- `s2_toa`: cloudy Sentinel-2 TOA tensor, normalized by `optical_scale`.
- `s1`: Sentinel-1 SAR tensor, expected VV/VH.
- `cld_shdw`: ALLClear original cloud/shadow probability map.  The loader
  prefers `data/.../cld_shdw/*.tif` over `visible_masks/*.png` because the PNG
  masks are binary visible masks and do not preserve cloud/shadow channels.
  The official TIFF mask is five-channel: channel 0 is cloud probability,
  channel 1 is binary cloud, and channels 2/3/4 are binary shadow masks from
  different dark-pixel thresholds.  Stage1 uses channel 1 as `M_cloud` and
  channel 3 as `M_shadow`.
- `target`: clear Sentinel-2 TOA tensor.

Region masks:

- `M_clear`
- `M_shadow`
- `M_cloud`

There is no `P_uncertainty` and no thin/thick split.

Stage 1:

```text
CLEAR:
  I_clear = s2_toa

SHADOW:
  SoftShadow mask predictor -> M_shadow_soft
  SoftShadow removal head -> I_shadow

CLOUDY:
  S2 stem + S1 stem
  DDIN PGDA-unfolded shared/private updates
  PDAFM: CAB(P,S) -> CAB(V,FM)
  bottleneck context: FFC by default in the current mainline
  DADIGAN RB reconstruction -> I_cloud_raw
  I_cloud = (1-M_cloud)*s2_toa + M_cloud*I_cloud_raw

I_hat(stage1) = I_stage1
              = M_clear*I_clear + M_shadow*I_shadow + M_cloud*I_cloud

Optional boundary refinement:
  [I_stage1, s2_toa, M_clear, M_shadow, M_cloud]
  -> none / local conv / LaMa-style FFC boundary refiner
  -> I_hat
```

Stage 1 is now the main model, not a pretraining stage.  The complete restored
image is formed from three task-specific candidates.  Stage 2 TG-style
Transformer/MoE fusion is disabled by default and should be treated as an
ablation, because ALLClear already provides explicit clear/shadow/cloud routing.

The boundary refiner is optional.  The default is `stage1_refiner: none` because
a full-resolution FFC residual after hard routing can be overkill and adds
memory/time.  If visible seams appear, use `stage1_refiner: conv` first; use
`stage1_refiner: ffc` only as a LaMa-style local/global seam ablation.

## Losses

Stage 1:

```text
L = L1(I_hat, target; shadow/cloud/boundary mask)
  + lambda_grad * Lgrad
  + lambda_shadow * (Lrem + Lmask + Lpen)
  + lambda_cloud_l1 * L1_cloud
  + lambda_known_l1 * L1_known
  + lambda_kl * LKL_cloud
  + lambda_adv * Ladv_cloud
```

There is no MoE balance loss in the Stage1-only mainline.  `final_l1` and
`grad` default to `final_mask_mode: degraded`, so they supervise shadow/cloud
and optional boundary pixels rather than clear pixels.

DADIGAN contributes the CLOUDY branch objective: cGAN + L1 + KL.  LaMa is used
in two bounded ways: known-region L1/adversarial scheduling is referenced for
large-mask inpainting, and FFC is used as the CLOUDY bottleneck context in the
current CAB+FFC mainline.  The DADIGAN MSAB path remains available only as an
ablation.

The cloudy branch keeps both `I_cloud_raw` and `I_cloud`.  `I_cloud_raw` is the
full generator prediction and receives cloud L1, known-region L1, and
adversarial supervision.  `I_cloud` is the mask-composited candidate used by
Stage1 hard routing before boundary refinement.  This follows LaMa's distinction
between predicted image and inpainted/composited image while preserving the
DADIGAN-style SAR-optical generator.

## CLOUDY Bottleneck Ablations

Current mainline:

```yaml
cloud_bottleneck_context: ffc
```

Optional ablations:

```yaml
cloud_bottleneck_context: identity  # remove global context
cloud_bottleneck_context: msab      # DADIGAN MSAB bottleneck
cloud_bottleneck_context: msab_ffc  # keep MSAB and append LaMa FFC
cloud_ffc_ratio: 0.75               # LaMa ffc_resnet_075 ratio_gin/gout
cloud_ffc_blocks: 1
```

These settings should stay fixed within one experiment so ablations isolate one
factor at a time.

## Commands

Stage1-only training:

```bash
python -m src.allclear.train \
  --config configs/allclear_tgdad_softshadow_stage1.yaml
```

Optional Stage2 ablation:

```bash
python -m src.allclear.train \
  --config configs/allclear_tgdad_softshadow_stage2.yaml \
  --stage stage2 \
  --stage1-checkpoint outputs/allclear/<stage1-run>/checkpoints/best_epoch_XXXX_loss_YYYYYY.pt
```

Evaluate Stage1:

```bash
python scripts/evaluate_allclear_two_stage.py \
  --config configs/allclear_tgdad_softshadow_stage1.yaml \
  --checkpoint outputs/allclear/<stage1-run>/checkpoints/best_epoch_XXXX_loss_YYYYYY.pt \
  --stage stage1 \
  --split val
```

## Checkpoints and Logs

Each run creates:

```text
outputs/allclear/<date>_<stage>_<run_name>/
  config.resolved.json
  train_log.csv
  checkpoints/
    last.pt
    best_epoch_*.pt   # only keep_best retained
  metrics/
    latest.json
  visualizations/
    epoch_XXXX_stage_low.png
    epoch_XXXX_stage_medium.png
    epoch_XXXX_stage_high.png
```
