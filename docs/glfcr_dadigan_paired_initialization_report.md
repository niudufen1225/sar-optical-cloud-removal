# GLF-CR x DADIGAN Paired Initialization Report

## Scope

This report covers the paired common initialization mechanism for the GLF-CR x
DADIGAN S1/S2 ablation path. This phase implemented code, tests, and
documentation only.

Explicit declarations for this phase:

- No tests were run.
- No existing weights were loaded.
- No GPU or CUDA path was used.
- No training was started.
- No evaluation was started.
- No dataset, manifest, preprocessing, model mathematics, or loss definition was
  changed for this phase.

Repository state recorded when the implementation work began:

- Branch: `main`
- Commit: `22bde2ce2ad958ff752a2c5404d52c6156631960`
- Last commit summary: `22bde2c refactor: decouple lowres DDIN and GLF-CR fusion`

## Root Cause

The Phase 3 structure check showed that S1 and S2 differed only by the post-DDIN
SAR dynamic filter:

- `model.cloud_post_ddin_sar_filter`: `none` vs `glfcr_dynamic`
- `model.cloud_post_ddin_sar_filter_kernel_size`: `null` vs `5`

The structural difference is intentionally small, but the raw same-seed shared
parameter initialization still diverged. The observed mismatch was expected:
S2 instantiates extra post-DDIN filter parameters, so the PyTorch RNG stream is
advanced before later shared modules are initialized. Same seed only fixes the
initial RNG state; it does not guarantee equality when constructor paths consume
a different number or order of random values.

Therefore, direct same-seed construction is not a valid fairness mechanism for
this ablation.

## Why Explicit Paired Initialization

The implementation uses explicit paired model-only initialization instead of
reordering constructors. This is the safer choice for the current project
because:

- It preserves existing model construction order and runtime behavior.
- It avoids changing model mathematics just to satisfy an experimental control.
- It makes the common-state equality explicit, inspectable, and repeatable.
- It allows S2-only parameters to remain real S2 parameters rather than dummy
  placeholders in S1.
- It creates a strict checkpoint artifact that can be used by training without
  optimizer, scheduler, epoch, or best-metric state.

The paired initializer copies S1 shared tensors into S2 exactly and leaves S2-only
post-filter tensors untouched.

## Files Added Or Modified

Added:

- `src/allclear/paired_initialization.py`
- `scripts/create_glfcr_s1_s2_paired_init.py`
- `tests/test_glfcr_dadigan_paired_initialization.py`
- `docs/glfcr_dadigan_paired_initialization_report.md`

Modified:

- `src/allclear/train.py`
- `scripts/inspect_glfcr_s1_s2_configs.py`

No outputs, checkpoints, manifests, datasets, or training logs were intentionally
modified by this phase.

## Paired Init Format

The paired initializer writes:

- `s1_model_init.pt`
- `s2_model_init.pt`
- `paired_init_manifest.json`

Each model init file is a model-only payload with:

- `format_version`
- `kind = model_initialization`
- `role = s1 | s2`
- `pair_id`
- `seed`
- `created_at`
- `git_commit`
- `config_path`
- `config_sha256`
- `structure_signature`
- `state_dict`

The payload intentionally excludes:

- optimizer state
- scheduler state
- AMP scaler state
- epoch
- best metric
- training history

This makes `--init-model-checkpoint` a fresh-training initializer, not a resume
mechanism.

## Shared Synchronization Rule

Let:

- `K1` be S1 state-dict keys.
- `K2` be S2 state-dict keys.
- `Kshared = K1 intersect K2`.
- `K2only = K2 - K1`.
- `K1only = K1 - K2`.

The paired initialization rule is:

- `K1only` must be empty.
- Every key in `K2only` must start with
  `cloud_branch.post_ddin_sar_filter.`.
- Every shared tensor must have identical shape and dtype.
- S1 keeps its original tensors.
- S2 shared tensors are replaced by exact clones of the matching S1 tensors.
- S2-only tensors are preserved as initialized in S2.
- After synchronization, every shared tensor difference must be exactly zero.

Allowed S2-only prefix:

```text
cloud_branch.post_ddin_sar_filter.
```

Any extra S2-only key outside this prefix is treated as an incompatibility.

## Strict Model-Only Load

The new training path adds:

```text
--init-model-checkpoint PATH
```

This path is separate from `--resume` and `--resume-in-place`.

Rules:

- `--init-model-checkpoint` cannot be combined with either resume flag.
- The payload must have `kind = model_initialization`.
- The payload config hash must match the loaded training config.
- If the full config hash differs, the loader may still accept the init only when
  the recorded reference config is available and every difference is a runtime
  screening field.
- The payload structure signature must match the instantiated model.
- The payload seed must match the config seed.
- The payload state dict is loaded with `strict=True`.
- The training state remains fresh: optimizer, scheduler, epoch, and best metric
  are newly initialized.

The legacy resume path remains unchanged. It continues to load training state and
is not used for paired initialization.

Runtime screening config differences currently allowed for model-only init:

- `run_name`
- `output_root`
- `train.epochs`
- `train.val_every`
- `train.keep_best`

All data, model, loss, optimizer, scheduler, augmentation, manifest, and seed
differences remain rejected unless a new paired init is generated for that exact
configuration.

## Resume Vs Init-Model

`--resume`:

- Restores model parameters.
- Restores optimizer state.
- Restores scheduler state if present.
- Restores scaler state if present.
- Restores epoch and best metric.
- Continues an interrupted run.

`--init-model-checkpoint`:

- Loads only model parameters.
- Requires strict config and structure compatibility.
- Starts from epoch 1 in the existing one-based training loop.
- Uses fresh optimizer, scheduler, scaler, and best metric.
- Is intended for controlled ablation initialization.

These two modes are mutually exclusive by CLI validation.

## DataLoader And Runtime RNG

The paired initialization path also isolates runtime randomness so S1/S2 training
can start from comparable non-model random states.

Legacy behavior:

- If `--init-model-checkpoint` is not used, DataLoader construction remains on
  the existing path.
- No paired generator or paired worker initialization hook is installed.

Paired init behavior:

- A deterministic train DataLoader generator is created from the config seed and
  rank.
- A deterministic worker init function seeds Python, NumPy if available, and
  PyTorch per worker from `torch.initial_seed()`.
- For DDP, `DistributedSampler` receives the common config seed across ranks,
  while DataLoader worker/runtime seeds still include rank.
- Runtime RNG is reset before auxiliary/discriminator construction so S1/S2
  non-model trainable modules are not accidentally shifted by model-constructor
  differences.
- Runtime RNG is reset again immediately before the first training iteration so
  dropout and other runtime random streams start comparably.

This does not claim bitwise equality for a whole training run under nondeterministic
CUDA kernels. It only removes the known constructor-order and loader-order
confounders.

## Inspect Script Extension

`scripts/inspect_glfcr_s1_s2_configs.py` now accepts:

```text
--s1-init PATH
--s2-init PATH
```

Both must be supplied together.

With these arguments, the script:

- Builds both models from the supplied configs.
- Strict-loads the corresponding model-only initialization payloads.
- Validates pair metadata.
- Verifies S1-only, S2-only, shared key, shape, dtype, and shared-value rules.
- Reports raw initialization and paired initialization separately.

Without these arguments, the script preserves the existing raw same-seed behavior.

## Required User Commands

Run unit tests:

```bash
cd /home/students/sushaoqi/CR/main
python -m unittest -q tests.test_glfcr_dadigan_paired_initialization
python -m unittest -q tests.test_glfcr_dadigan_phase1 tests.test_glfcr_dadigan_phase3_configs
```

Generate the paired model-only initialization files:

```bash
cd /home/students/sushaoqi/CR/main
python scripts/create_glfcr_s1_s2_paired_init.py \
  --s1-config configs/allclear_dadigan_glfcr_ablation_s1_clean_core.yaml \
  --s2-config configs/allclear_dadigan_glfcr_ablation_s2_post_sar_filter.yaml \
  --seed 2026 \
  --output-dir pretrained/glfcr_dadigan_phase3_seed2026 \
  --device cpu
```

Inspect paired initialization on CPU:

```bash
cd /home/students/sushaoqi/CR/main
python scripts/inspect_glfcr_s1_s2_configs.py \
  --s1-config configs/allclear_dadigan_glfcr_ablation_s1_clean_core.yaml \
  --s2-config configs/allclear_dadigan_glfcr_ablation_s2_post_sar_filter.yaml \
  --seed 2026 \
  --device cpu \
  --s1-init pretrained/glfcr_dadigan_phase3_seed2026/s1_model_init.pt \
  --s2-init pretrained/glfcr_dadigan_phase3_seed2026/s2_model_init.pt \
  --output-json pretrained/glfcr_dadigan_phase3_seed2026/cpu_inspection.json
```

Optional GPU smoke inspection after CPU checks pass:

```bash
cd /home/students/sushaoqi/CR/main
python scripts/inspect_glfcr_s1_s2_configs.py \
  --s1-config configs/allclear_dadigan_glfcr_ablation_s1_clean_core.yaml \
  --s2-config configs/allclear_dadigan_glfcr_ablation_s2_post_sar_filter.yaml \
  --seed 2026 \
  --device cuda:1 \
  --smoke \
  --batch-size 1 \
  --height 128 \
  --width 128 \
  --s1-init pretrained/glfcr_dadigan_phase3_seed2026/s1_model_init.pt \
  --s2-init pretrained/glfcr_dadigan_phase3_seed2026/s2_model_init.pt \
  --output-json pretrained/glfcr_dadigan_phase3_seed2026/gpu_smoke.json
```

## Expected Training Interface

After the tests and inspections above pass, S1/S2 training should use the strict
model-only init path, not `--resume`.

Example S1 command:

```bash
python -m src.allclear.train \
  --config configs/allclear_dadigan_glfcr_ablation_s1_clean_core.yaml \
  --stage stage1 \
  --gpu 1 \
  --output-dir ./outputs/allclear \
  --run-name _glfcr_s1_paired_seed2026 \
  --init-model-checkpoint pretrained/glfcr_dadigan_phase3_seed2026/s1_model_init.pt
```

Example S2 command:

```bash
python -m src.allclear.train \
  --config configs/allclear_dadigan_glfcr_ablation_s2_post_sar_filter.yaml \
  --stage stage1 \
  --gpu 1 \
  --output-dir ./outputs/allclear \
  --run-name _glfcr_s2_paired_seed2026 \
  --init-model-checkpoint pretrained/glfcr_dadigan_phase3_seed2026/s2_model_init.pt
```

These commands are documented for the next phase only. They were not executed in
this phase.

## Unverified Risks

Because this phase intentionally did not run tests, load weights, use GPU, train,
or evaluate, the following must be checked by the user commands above:

- Python import and syntax validity in the target environment.
- Strict payload generation and strict loading behavior on the real configs.
- CPU inspection shared-difference value after load.
- Optional GPU smoke behavior.
- Whether existing dirty user modifications in `src/allclear/train.py` interact
  with future branches.
- Whether any equivalent-but-not-identical config path spelling changes alter the
  config hash. Use the exact same config files for generation and training.

## Stop Items

Do not start S1/S2 training if any of the following happens:

- `tests.test_glfcr_dadigan_paired_initialization` fails.
- Pair generation reports any S1-only key.
- Pair generation reports any S2-only key outside
  `cloud_branch.post_ddin_sar_filter.`.
- Paired inspection reports nonzero shared tensor differences.
- Strict load requires `strict=False`.
- The training config differs from the paired-init config hash.
- The model structure signature differs from the paired-init structure signature.

## Conclusion

The paired initialization path is now designed to remove the known constructor RNG
confounder between S1 and S2 without changing the model math. It gives S1 and S2
identical shared tensors, preserves S2-only post-filter tensors, and exposes a
strict model-only training initializer that remains separate from resume.

The next recommended action is to run the paired initialization unit test and CPU
inspection commands above. If they pass, it is reasonable to proceed to S1/S2
training with `--init-model-checkpoint`.
