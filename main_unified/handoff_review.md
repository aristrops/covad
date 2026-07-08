# Handoff — Independent Review of the Unified VAD+CBM Work

You are an independent reviewer. Your job: **verify that the "unified" and
"unified++" pipeline added to this repo is correct and that its results make
sense.** Be skeptical. Read the code, don't trust this document's claims —
confirm or refute them. Report concrete findings (file:line), not vibes.

Repo: `/home/borsattifr/code-files/VAD/CBM/CONVAD/covad` (git branch `unified`).
Python: use the repo venv `.venv/bin/python`. GPUs available (a benchmark may be
running on one; check `nvidia-smi` and use a free device for any spot-checks).
Data/model symlinks: `cbm_data/`, `cbm_models/`.

## 1. What the work claims to do

The original repo has **two separate branches** fused only at the score level
(`main_scripts/combined_branches.py`): a visual STFPM heatmap branch and a
concept-bottleneck (CBM) branch. Reviewers of the paper disliked the separation.

**Unified** = a single model that keeps the *original joint-CBM structure*, with
ONE change: the concept classifier is fed the **STFPM teacher–student feature
difference** instead of raw-image features. The final anomaly score still comes
from the concept→MLP head, exactly as in the original joint CBM. Trained jointly:
STFPM feature-matching loss (masked to normal samples) + the joint-CBM loss.

**Unified++** = same, but additionally the two *deeper* feature differences are
injected (added) into the truncated concept CNN at the internal blocks whose
output shape matches. This is the intended **final** model.

Setting under test: **fully-supervised** (`cbm_data/{mvtec,visa}/*_dataset_automated.csv`,
full real anomalies + concept labels, no subsampling/generated anomalies).
Backbone: mobilenet_v2 only.

## 2. Files to review

All unified code lives under `main_unified/` (self-contained package):
- `main_unified/models_unified.py` — `UnifiedModel`, `ConceptNetFromDiff`, and the
  `unified_model(...)` factory. Reuses `BackboneModelFeatures`, `FC`, `MLP` imported
  from the shared `models/model_backbones.py`.
- `main_unified/trainer_unified.py` — `UnifiedTrainer` (loss).
- `main_unified/evaluator_unified.py` — `UnifiedEvaluator` (metrics).
- `main_unified/unified.py` — CLI entrypoint (train/eval) + results-table writer.
  Run as `python -m main_unified.unified` from the repo root.
- `main_unified/run_unified_*.sh` — runners (`run_unified_pp_all.sh` is the final
  full benchmark; `run_unified_ppmasked_all.sh` is the ablation).
- `main_unified/visualize_unified.py` (heatmaps), `main_unified/compare_ablation.py`
  (before/after diff table).

Modified shared file (the ONLY unified change outside `main_unified/`):
- `datasets/concept_dataset.py` `__getitem__` — added a branch returning
  `(image, attr_label, label, mask)` when `use_attr and load_mask`. Backward
  compatible (no existing caller sets both).

Reference (original, for comparison): `models/model_backbones.py`
(`BackboneModel`, `End2EndModel`, `FC`, `MLP`, `BackboneModelFeatures`),
`models/full_models.py` (`joint_model`), `trainers/trainer_cbm.py`,
`trainers/trainer_stfpm.py`, `evaluators/evaluator_stfpm.py`.

## 3. Specific claims to verify (the important part)

**A. Concept input is really the feature difference, same shape as expected.**
- `UnifiedModel.forward`: concept input = `normalize(t_f3,dim=1) - normalize(s_f3,dim=1)`,
  fed to `ConceptNetFromDiff`. Confirm mobilenet block-3 output is [B,24,56,56] and
  that `ConceptNetFromDiff` = `mobilenet_v2.features[4:]` (which expects 24-ch input)
  → AvgPool7 → 1280 → N per-concept `FC` heads. Confirm teacher is **frozen**
  (`requires_grad=False`, `.eval()` even in train) and student is trained.

**B. Final score = concept head, structurally identical to `joint_model`.**
- Confirm `UnifiedModel` builds `main_model = MLP(num_attr, expand_dim)`, applies
  ReLU to concept logits, concatenates, → main logit. Compare against
  `End2EndModel.second_forward_stage` / `joint_model`. It should be the same head.

**C. unified++ injection is correct.**
- `UnifiedModel.INJECT_MAP = {1:4, 2:10}` and `ConceptNetFromDiff.forward(x, injects)`
  add `normalize(t_fk)-normalize(s_fk)` after truncated element 4 and 10.
- Verify shapes: truncated element 4 output must equal f8 diff shape [B,64,14,14];
  element 10 output must equal f14 diff shape [B,160,7,7]. (There is a smoke check
  you can reproduce — run a dummy tensor through `model.concept_net.features` and
  print shapes at indices 4 and 10.) A shape mismatch would crash, so the fact it
  runs is weak evidence; still confirm the *mapping is the intended one* (STFPM
  comparison feature indices are [3,8,14]; truncated index = original_index − 4).

**D. Loss is right.** In `trainer_unified.py::compute_losses`:
- STFPM term: per-sample `Σ_L ‖n(t_L)−n(s_L)‖²`, **masked so only label==0 (normal)
  samples contribute** (mean over normal samples). Check the masking math — if a
  batch has 0 normal samples it must not divide by zero.
- CBM term: `(main_BCE + λ·Σ concept_BCE) / (1 + λ·num_attr)`, λ=0.55, with
  `pos_weight` on main + per-concept BCE. Confirm this matches the original joint
  `CBMTrainer.run_epoch` normalization (not bottleneck branch).
- Total = STFPM + CBM. Question worth raising: **is summing an unweighted STFPM
  term (magnitude ~O(1) here) with the normalized CBM term a sensible balance?**
  Check the loss logs (train prints stfpm/main/concept separately).

**E. Metrics.** In `evaluator_unified.py`:
- `CBM I-AUC` = ROC-AUC of `sigmoid(main_logit)` vs `label_index` (primary).
- `STFPM I-AUC` = ROC-AUC of max-pooled heatmap vs image label; `P-AUC` = pixel
  ROC-AUC. `concept AUC/F1` = per-concept.
- **Mask binarization**: masks are binarized with `> 0` (NOT `> 0.5`). Reason: VisA
  encodes anomaly as pixel value 5 (→ ~0.02 after ToTensor), MVTec uses 255 (→1.0);
  a 0.5 threshold silently zeroed all VisA positives → P-AUC was `nan`. Confirm the
  fix is correct and that MVTec is unaffected. (This was a real bug we found & fixed.)

**F. Data correctness / leakage.**
- Train uses the full `train` split (normal + anomalous); eval uses `test`.
  Concept columns = all df columns minus the metadata set. Confirm no train/test
  overlap and that `attr_cols` are the intended concepts.
- Teacher init: `unified_model` loads `feature_extractor.*` from a fine-tuned
  backbone (`fine-tuned-mobilenet.pth` for MVTec, `-visa.pth` for VisA), else
  ImageNet. Confirm the right teacher is used per dataset and that it is frozen.

## 4. Results to sanity-check

- Baseline unified table: `results/unified_results.md` (+ `.csv`). unified++ final
  table: `results/unified_pp_results.md` (being written by the running benchmark).
- Checkpoints: `cbm_models/{mvtec,visa}/unified{,_pp}/<cat>/mobilenet_v2.pth`.
- Heatmap figures: `plots/unified_pp/` (capsule, grid, hazelnut) and
  `plots/unified_visa/` (candle, macaroni2). Column 3 is the model heatmap; check it
  localizes the GT (column 2).

Known, already-explained observations (verify they're what we say):
- Hard categories capsule & grid had baseline **CBM I-AUC 0.348 / 0.278** (< random)
  with near-random concept AUC but fine P-AUC → the *concept branch* failed there,
  not localization. unified++ raised them to **0.900 / 0.990**. Confirm this on the
  checkpoints; make sure it's a genuine improvement and not, e.g., a metric/label
  quirk.
- On texture classes (grid) the heatmap fires on unlabeled dirt/specks (background),
  overshadowing faint scratches — a standard STFPM texture artifact; it depresses
  pixel metrics but not the image-level concept score. Confirm this interpretation
  from the figures.

## 5. Things we did NOT do (flag if you think they're needed)

- No head-to-head vs the **original joint-CBM** on the same splits — so "as good as
  before" is asserted, not measured. A reviewer may want the original numbers.
- No matched standalone-STFPM baseline (the existing `cbm_models/stfpm_models/*`
  were trained against an unknown teacher; pairing them gave cable P-AUC ~0.51).
- `make_dataloader` in `main_unified/unified.py` uses `num_workers=8`; the rest of
  the repo uses 0 (a known GPU-starvation bottleneck) — only unified was changed.
- `utils/metrics.py::min_max_norm` has no epsilon (`(x-min)/(max-min)`); a constant
  score map would produce nan. Not currently triggered — flag as latent risk.

## 6. How to spot-check quickly

Re-run one eval and confirm the printed metrics match the table:
```
CUDA_VISIBLE_DEVICES=<free> .venv/bin/python -m main_unified.unified --mode eval \
  --dataframe_path cbm_data/mvtec/grid_dataset_automated.csv --category grid \
  --backbone mobilenet_v2 --device cuda --inject_diffs \
  --save_path cbm_models/mvtec/unified_pp/grid/mobilenet_v2.pth
```
(omit `--inject_diffs` and use the `unified/` checkpoint for the baseline model.)

Deliver: a short report of what checks out, any real bugs (with file:line and a
failing scenario), and whether the headline claims (unified == joint-CBM but on the
feature difference; unified++ rescues the hard categories) are supported by the code
and the numbers.
```
Journal of what was done: `main_unified/journal.md`.
```
