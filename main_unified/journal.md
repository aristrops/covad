# Unified STFPM + Concept Branch — Journal

Single-branch VAD+CBM. Fully-supervised setting (MVTec + VisA, `*_dataset_automated.csv`).

## Architecture (current, correct)
Same structure as the original joint CBM — the ONLY change is that the concept
classifier reads the STFPM teacher-student **feature difference** instead of
raw-image features.

- Teacher (frozen) + Student (trained): `BackboneModelFeatures`, mobilenet_v2 →
  features at blocks [3,8,14], channels [24,64,160].
- STFPM heatmap = product of per-layer normalized squared diffs (localization).
- Concept classifier = truncated mobilenet `features[4:]`, fed the first diff
  `n(t_f3)-n(s_f3)` [B,24,56,56] → pool → 1280 → per-concept FC heads.
- Concept bottleneck → ReLU → MLP main head → **final anomaly score** (as before).
- Loss = STFPM(normal-masked) + (main_BCE + λ·Σ concept_BCE)/(1+λ·N), λ=0.55.
- Metrics: **CBM I-AUC** (primary, from concepts) + STFPM I-AUC & P-AUC (heatmap
  bonus) + concept AUC/F1. Auto-updating table `results/unified_results.{md,csv}`.

Files: `models/model_backbones.py` (`UnifiedModel`, `ConceptNetFromDiff`),
`models/full_models.py` (`unified_model`), `trainers/trainer_unified.py`,
`evaluators/evaluator_unified.py`, `main_scripts/unified.py`,
`run_unified_mvtec.sh`, `run_unified_all.sh`, `visualize_unified.py`.
Dataset tweak: `ConceptDataset` returns `(image,concepts,label,mask)` when
`use_attr and load_mask` (eval only). Perf: `make_dataloader` uses num_workers=8
(rest of repo still 0 → GPU-starve; flagged, not changed).

## Status
- Sanity (fixed arch, 50 ep): hazelnut CBM I-AUC 1.000 / P-AUC 0.993;
  cable 0.954 / 0.898. Heatmaps in plots/unified/ localize well.
- Full MVTec+VisA run in progress on GPU1 → results/unified_results.md.
- Partial outliers: capsule CBM I-AUC 0.348, grid 0.278 (< random); their concept
  AUC also ~random (0.58, 0.53) but heatmap P-AUC fine → concept branch fails
  there, not localization.

## unified++ (feature-diff injection) — running GPU2
- Also ADD the deeper diffs into the truncated concept net at matching blocks
  (residual add, shapes exact): f8 diff [64x14^2] after trunc elem 4 (orig
  features[8]); f14 diff [160x7^2] after trunc elem 10 (orig features[14]).
- Impl: `ConceptNetFromDiff.forward(x, injects={elem:diff})`,
  `UnifiedModel.inject_diffs` + `INJECT_MAP={1:4,2:10}`, CLI `--inject_diffs`.
  Separate ckpts `cbm_models/mvtec/unified_pp/`, table `results/unified_pp_results.*`.
- Test cats: hazelnut, leather (sanity) + capsule, grid (hard). 50 ep.

## [2026-07-08] Repo reorg for merge to main
- All unified code moved into a self-contained `main_unified/` package:
  models_unified.py (UnifiedModel, ConceptNetFromDiff, unified_model — reuses the
  shared BackboneModelFeatures/FC/MLP), trainer_unified.py, evaluator_unified.py,
  unified.py, visualize_unified.py, compare_ablation.py, run_unified_*.sh,
  journal.md, handoff_review.md. Run as `python -m main_unified.unified` from repo root.
- Reverted the unified additions in models/model_backbones.py and models/full_models.py
  (original CBM pipeline untouched; verified it still imports). The ONLY shared-file
  change kept is the backward-compatible `datasets/concept_dataset.py` 4-tuple return.
- Checkpoints unaffected by the move (state_dict keys are attribute-based); verified
  by re-eval (grid unified_pp CBM I-AUC 1.000).

## [2026-07-08] GPU-starvation fix in ORIGINAL code (merge prep)
- Added num_workers=8, pin_memory=True (+persistent_workers on training loaders) to
  the original DataLoaders: main_scripts/cbm.py (make_dataloader), stfpm.py
  (train/val/test), combined_branches.py. Was num_workers=0 -> GPU-starved.

## Full ablation result (unified++ vs unified++masked, 27 cats)
- Masking the student from anomalies is slightly worse overall (ALL CBM I-AUC
  0.912 -> 0.898), driven by VisA (-0.049); MVTec ~flat (+0.013). Large per-cat
  swings in both directions (transistor 0.47->1.00, capsules 0.85->0.51) => single-
  seed instability; would need multi-seed to conclude. See results/compare_pp_vs_ppmasked.md.

## Final model = unified++ (results/unified_pp_results.md). Next
- (optional) multi-seed on volatile cats; matched standalone-STFPM baseline (open).
