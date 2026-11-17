# Image trainer (Gradients image tournament)

A LoRA trainer for the tournament's image tasks. Text-to-image: flux, ideogram4, z-image, krea2, qwen-image, flux2-klein-4b,
hidream-o1, moss-image-base. Image editing: flux2-klein-4b, qwen-image-edit-2511, hidream-o1, moss-image-base. Every family is
trained on the training stack the evaluator scores it with.

## Layout

- `comfytrain/g56_comfy_train.py` - the trainer (the container entrypoint); the other modules in `comfytrain/` are its helpers
  (`g56_newfam.py` drives the families of the newer tournament format).
- `ops/docker/` - the two Dockerfiles the validator builds (flux: `standalone-image-trainer.dockerfile`; the other image models:
  `standalone-image-toolkit-trainer.dockerfile`), their locked package lists and install scripts.

## Running

```
python comfytrain/g56_comfy_train.py --task-id <id> --model <repo> --dataset-zip <zip> --model-type <family> \
    --expected-repo-name <name> --hours-to-complete <h> [--trigger-word <w>] [--training-mode edit]
```

The base model and the dataset are read from the validator-staged `/cache`; the container runs offline. The trained adapter is
written to `/app/checkpoints/<task-id>/<expected-repo-name>/last.safetensors`.

## New since the 9 Nov entry (2f31b62)

Mechanisms this release runs by default that our earlier entry did not contain. Each line names what it
does and its call site.

- **Evaluator-native training (new image families).** Trains each adapter with the evaluator's own training loop and loss. call site `NEWFAM_TYPES` (`comfytrain/g56_comfy_train.py:9642`), `run` (`comfytrain/g56_newfam.py:3119`).
- **Loader-mirror gate (new image families).** Checks every shipped file against a mirror of the evaluator's loader. call site `shipped_form` (`comfytrain/g56_newfam.py:2054`).
- **Kill-safe snapshots (new image families).** A checked trained snapshot replaces the shipped file throughout the run. call site `snapshot_due` (`comfytrain/g56_aitk_runner.py:1787`).
- **Out-of-memory fallback (new image families).** Relaunches once after an out-of-memory stop, at lower resolution or with checkpointing. call site `relaunch_aitk` (`comfytrain/g56_newfam.py:3311`).
- **Seeded-member mean (new image families, cells).** Members train in turn as measured time allows; their exact mean ships. call site `member_plan` (`comfytrain/g56_aitk_runner.py:1601`), `ship_mean` (`comfytrain/g56_newfam.py:2224`).
- **Member length by task size (new image families, cells).** Sets each member's full length from the task's number of training items. call site `step_cap` (`comfytrain/g56_newfam.py:3199`).
- **Export scale (new image families, cells).** Scales the finished adapter's weight change by a per-cell factor. call site `scaled_for_export` (`comfytrain/g56_newfam.py:2059`).
- **Scored noise band (hidream-o1 edit).** Trains in a noise band that contains the one the evaluator scores. call site `musubi_band` (`comfytrain/g56_newfam.py:2978`).
- **Empty-instruction share (hidream-o1 edit).** Trains a share of the steps on pairs with empty instructions. call site `prepare_notext` (`comfytrain/g56_newfam.py:2986`).
- **Sequence-length memory rule (moss-image-base).** Chooses gradient checkpointing from the task's longest token sequence. call site `gc_for_task` (`comfytrain/g56_newfam.py:1730`).

## Live mechanisms (on by default in code)

Run as the image's ENTRYPOINT the trainer turns these on by itself; no environment variable is needed. Each line names the
default and the call site.

- **Release profile (every family).** Selects the release profile itself when run as the entrypoint. default `G56CT_PROFILE` (`comfytrain/g56_comfy_train.py:43`).
- **Seeded-member ensemble (every family).** Independently seeded members train on all images; their exact average ships. default `seeds_auto` (`comfytrain/g56_comfy_train.py:214`); call site `_SafeEnsemble` (`comfytrain/g56_comfy_train.py:9403`).
- **Member planner (every family).** Plans member count and lengths from the run's measured step times. default `T159_PROFILES` (`comfytrain/g56_comfy_train.py:12186`); call site `_t159_train_loop` (`comfytrain/g56_comfy_train.py:13372`).
- **Save-reserve planner (qwen-image).** Prices the final write from measured speed and replans the members. default `T302_QJ` (`comfytrain/g56_comfy_train.py:24515`); call site `_t302_fit` (`comfytrain/g56_comfy_train.py:24772`).
- **Fused block kernels (flux, ideogram4, z-image, qwen-image).** Fused normalisation and modulation kernels behind a per-operation parity gate. default `T248_F_G10` (`comfytrain/g56_comfy_train.py:19161`), `T251_I_FUSE` (`comfytrain/g56_comfy_train.py:19204`), `T248_Z_FUSE` (`comfytrain/g56_comfy_train.py:19146`), `T248_Q_FUSE` (`comfytrain/g56_comfy_train.py:19152`); call site `_t241_train_loop` (`comfytrain/g56_comfy_train.py:18106`), `g56fused` (`comfytrain/g56_comfy_train.py:17933`).
- **CUDA-graph training step (flux, qwen-image).** Replays each training step as a captured CUDA graph. default `graph_step` (`comfytrain/g56_comfy_train.py:19163`); call site `_t241g_train_loop` (`comfytrain/g56_comfy_train.py:18671`).
- **Fused training kernels (krea2).** Fused norm, attention-input and gated-MLP pieces, self-checked, eager fallback. default `T251_K2_FUSE` (`comfytrain/g56_comfy_train.py:19362`); call site `_t251_train_loop` (`comfytrain/g56_comfy_train.py:19360`), `g56fused_k2i4` (`comfytrain/g56_comfy_train.py:19316`).
- **Per-cell recipe layer (every family).** Matches the task's cell and switches on the cell mechanisms below. default `REL_C9N` (`comfytrain/g56_comfy_train.py:22819`); call site `_t287_run_cell` (`comfytrain/g56_comfy_train.py:23054`).
- **K-law members (cells).** Member count and lengths follow a law of the image count. call site `_t277_train_loop` (`comfytrain/g56_comfy_train.py:21152`).
- **Twin and mirror views (cells).** Adds shifted-crop and mirrored twins of each image in chosen bands. call site `_t280f_twins_build` (`comfytrain/g56_comfy_train.py:21579`), `t280f_mirror_cells` (`comfytrain/g56_comfy_train.py:21582`).
- **View bank (cells).** Shares a bank of twin and mirror views across members. call site `_t300_twins_build` (`comfytrain/g56_comfy_train.py:22559`), `_t300_member_run` (`comfytrain/g56_comfy_train.py:22685`).
- **Rotated mirror views (cells).** Rotates tagged mirror cases through upside-down and half-turned image views. call site `_t321_encode_latent` (`comfytrain/g56_comfy_train.py:24470`).
- **Warmup-stable-decay schedule (krea2 cell).** Holds the peak learning rate, then decays over the final fraction. call site `_t307_run_one` (`comfytrain/g56_comfy_train.py:24066`).
- **Exposure-count schedule (every family).** Fixed per-band exposure counts choose every step's prompt half and band. default `exposure` (`comfytrain/g56_comfy_train.py:265`); call site `exposure_setup` (`comfytrain/g56_comfy_train.py:9391`).
- **Generator-aware recipe cells (flux, z-image).** Picks the recipe cell from the training images' generator signature. call site `gencond_setup` (`comfytrain/g56_comfy_train.py:9710`).
- **Base-precision restoration (qwen-image).** Every file carries exact corrections of the base's low-precision rounding. default `qf8` (`comfytrain/g56_comfy_train.py:376`); call site `qf8_save_file` (`comfytrain/g56_comfy_train.py:5727`).
- **Training on the restored base (qwen-image).** Merges that restoration into the in-memory base before training. default `qf8_train` (`comfytrain/g56_comfy_train.py:377`); call site `qf8_train_setup` (`comfytrain/g56_comfy_train.py:9723`).
- **Jib-Mix base correction (qwen-image).** Corrects the low-precision Jib-Mix root toward its source weights. default `T167_C26S_D15` (`comfytrain/g56_comfy_train.py:13682`); call site `qf8_d15_ship` (`comfytrain/g56_comfy_train.py:13907`).
- **Extended base restoration (qwen-image, long Jib cell).** Extends the base correction to more blocks when time remains. default `t308_d60q` (`comfytrain/g56_comfy_train.py:22846`); call site `t308_extend` (`comfytrain/g56_comfy_train.py:24201`).
- **Final-file transforms (flux, z-image).** Reshapes the shipped factors per module group or noise band at export. default `T167_C26S_FC` (`comfytrain/g56_comfy_train.py:13683`), `T167_C26S_P1` (`comfytrain/g56_comfy_train.py:13684`); call site `_t157_fc` (`comfytrain/g56_comfy_train.py:11522`).
- **Unconditional-branch component (ideogram4).** Adds the base repo's unconditional-branch difference to every file. default `unc` (`comfytrain/g56_comfy_train.py:264`); call site `unc_prepare` (`comfytrain/g56_comfy_train.py:9777`).
- **Non-block layer correction (ideogram4).** Ships the non-block layers' low-precision corrections with every file. default `io_fix` (`comfytrain/g56_comfy_train.py:266`); call site `io_fix_apply` (`comfytrain/g56_comfy_train.py:1465`).
- **Base anchor (ideogram4, smaller tasks).** Periodically pulls the adapter toward its start on smaller tasks. default `T225_I4_ANC` (`comfytrain/g56_comfy_train.py:433`); call site `anchor_every` (`comfytrain/g56_comfy_train.py:7876`).
- **Rao-Blackwellised empty-prompt target (ideogram4, z-image).** Lowers the empty-prompt target's variance in the highest noise bands. default `rb_empty` (`comfytrain/g56_comfy_train.py:265`); call site `RBEmpty` (`comfytrain/g56_comfy_train.py:7414`).
- **Loss-aware band sampling (flux, small openai tasks).** Draws noise bands by recent loss with unbiased step weights. default `band_is` (`comfytrain/g56_comfy_train.py:339`); call site `BandIS` (`comfytrain/g56_comfy_train.py:7404`).
- **Short-member exclusion (every family).** Leaves members that ended well short of plan out of the average. default `T208_C26S_F1` (`comfytrain/g56_comfy_train.py:17608`); call site `_t208d_run_one` (`comfytrain/g56_comfy_train.py:17600`).
- **Canonical factor form (flux, ideogram4, z-image, qwen-image).** Writes the averaged adapter in canonical SVD form; product unchanged. default `canon` (`comfytrain/g56_comfy_train.py:208`); call site `canon_sd` (`comfytrain/g56_comfy_train.py:5713`).
- **Evaluator data path and loss target (every family).** Reads images and computes the flow-matching target as the evaluator does. call site `read_dataset` (`comfytrain/g56_comfy_train.py:1201`), `flow_prediction_mse` (`comfytrain/g56_comfy_train.py:6246`).
- **Evaluator-native training (new image families).** Trains each adapter with the evaluator's own training loop and loss. default `NEWFAM_TYPES` (`comfytrain/g56_comfy_train.py:181`); call site `NEWFAM_TYPES` (`comfytrain/g56_comfy_train.py:9642`), `run` (`comfytrain/g56_newfam.py:3119`).
- **Loader-mirror gate (new image families).** Checks every shipped file against a mirror of the evaluator's loader. call site `shipped_form` (`comfytrain/g56_newfam.py:2054`).
- **Kill-safe snapshots (new image families).** A checked trained snapshot replaces the shipped file throughout the run. default `SNAPSHOT_EVERY_S` (`comfytrain/g56_newfam.py:156`); call site `snapshot_due` (`comfytrain/g56_aitk_runner.py:1787`).
- **Out-of-memory fallback (new image families).** Relaunches once after an out-of-memory stop, at lower resolution or with checkpointing. default `RES_RUNGS` (`comfytrain/g56_newfam.py:278`); call site `relaunch_aitk` (`comfytrain/g56_newfam.py:3311`).
- **Seeded-member mean (new image families, cells).** Members train in turn as measured time allows; their exact mean ships. default `MEMBERS_MAX` (`comfytrain/g56_newfam.py:138`); call site `member_plan` (`comfytrain/g56_aitk_runner.py:1601`), `ship_mean` (`comfytrain/g56_newfam.py:2224`).
- **Member length by task size (new image families, cells).** Sets each member's full length from the task's number of training items. default `STEPS_PER_IMAGE` (`comfytrain/g56_newfam.py:142`); call site `step_cap` (`comfytrain/g56_newfam.py:3199`).
- **Export scale (new image families, cells).** Scales the finished adapter's weight change by a per-cell factor. default `EXPORT_SCALE` (`comfytrain/g56_newfam.py:152`); call site `scaled_for_export` (`comfytrain/g56_newfam.py:2059`).
- **Scored noise band (hidream-o1 edit).** Trains in a noise band that contains the one the evaluator scores. default `MUSUBI_TBAND` (`comfytrain/g56_newfam.py:177`); call site `musubi_band` (`comfytrain/g56_newfam.py:2978`).
- **Empty-instruction share (hidream-o1 edit).** Trains a share of the steps on pairs with empty instructions. default `MUSUBI_NOTEXT` (`comfytrain/g56_newfam.py:178`); call site `prepare_notext` (`comfytrain/g56_newfam.py:2986`).
- **Sequence-length memory rule (moss-image-base).** Chooses gradient checkpointing from the task's longest token sequence. default `GC_OFF_MAX_TOKENS` (`comfytrain/g56_newfam.py:101`); call site `gc_for_task` (`comfytrain/g56_newfam.py:1730`).

## License

Apache-2.0 (`LICENSE.md`, `NOTICE`).
