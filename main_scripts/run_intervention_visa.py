"""
Intervention experiment on VisA dataset — fully supervised joint model.
Averages results across 3 seeds per category.

Run from covad/ directory:
    uv run main_scripts/run_intervention_visa.py --device cuda --save_path results/intervention_visa_supervised.pkl
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.concept_dataset import ConceptDataset
from evaluators.evaluator_cbm import CBMEvaluator
from models.full_models import joint_model
from utils.intervention_utils import compute_intervention_order, modify_concepts
from utils.model_utils import generate_concept_logits


VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]

BACKBONE = "mobilenet_v2"
SEEDS = [0, 1, 2]
CHECKPOINT_NAME = f"{BACKBONE}_1.0ratio_0MLP_automated.pth"


def get_paths(base_dir, category, seed):
    df_path = os.path.join(
        base_dir, "cbm_data", "visa",
        f"{category}_dataset_automated.csv",
    )
    model_path = os.path.join(
        base_dir, "cbm_models", "visa",
        f"{category}_models", f"seeds_{seed}",
        "original_anomalies", "joint",
        CHECKPOINT_NAME,
    )
    return df_path, model_path


def run_intervention_single_seed(df_path, model_path, device, batch_size):
    dataframe = pd.read_csv(df_path)
    gt_test_df = dataframe[dataframe["split"] == "test"].reset_index(drop=True)

    state_dict = torch.load(model_path, weights_only=False)

    test_dataset = ConceptDataset(dataframe, "test", use_attr=True)
    num_attr = len(test_dataset.attr_cols)
    attr_cols = test_dataset.attr_cols

    concept_model, main_task_model = joint_model(
        num_attr=num_attr,
        freeze_parameters=True,
        expand_dim=0,
        backbone=BACKBONE,
        model_state_dict=state_dict,
        mode="test",
        concept_intervention=True,
    )
    concept_model.to(device)

    pred_df = generate_concept_logits(concept_model, dataframe, save_path=None, device=device)
    pred_df_test = pred_df[pred_df["split"] == "test"].reset_index(drop=True)

    train_logits = pred_df[attr_cols][pred_df["split"] == "train"].values
    ptl_5 = dict(zip(attr_cols, np.percentile(train_logits, 5, axis=0)))
    ptl_95 = dict(zip(attr_cols, np.percentile(train_logits, 95, axis=0)))

    intervention_order = compute_intervention_order(pred_df_test, attr_cols)

    auc_scores, f1_scores = [], []
    main_task_model.to(device)

    for n in range(0, num_attr + 1):
        modified_df = modify_concepts(
            intervention_order, gt_test_df, pred_df_test, attr_cols, ptl_5, ptl_95, n
        )
        new_test_dataset = ConceptDataset(modified_df, "test", use_attr=True, load_image=False)
        new_attr_cols = new_test_dataset.attr_cols
        loader = torch.utils.data.DataLoader(new_test_dataset, batch_size, shuffle=False)

        evaluator = CBMEvaluator(
            main_task_model, num_attr, new_attr_cols, loader, device, main_only=True
        )
        auc, f1 = evaluator.evaluate()
        auc_scores.append(auc)
        f1_scores.append(f1)

    return {"auc": auc_scores, "f1": f1_scores, "num_concepts": num_attr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_path", type=str, default="results/intervention_visa_supervised.pkl")
    parser.add_argument("--categories", type=str, nargs="+", default=VISA_CATEGORIES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    results = {}
    if os.path.exists(args.save_path):
        with open(args.save_path, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded existing results for: {list(results.keys())}")

    for category in args.categories:
        if category in results and "error" not in results[category]:
            print(f"\nSKIP {category} — already computed.")
            continue

        print(f"\n{'='*60}\nCategory: {category}\n{'='*60}")

        df_path, _ = get_paths(args.base_dir, category, 0)
        if not os.path.exists(df_path):
            print(f"  SKIP — dataframe not found: {df_path}")
            continue

        seed_results = []
        for seed in args.seeds:
            _, model_path = get_paths(args.base_dir, category, seed)
            if not os.path.exists(model_path):
                print(f"  SKIP seed {seed} — checkpoint not found: {model_path}")
                continue
            print(f"  Seed {seed}...")
            try:
                r = run_intervention_single_seed(df_path, model_path, device, args.batch_size)
                seed_results.append(r)
            except Exception as e:
                print(f"  ERROR seed {seed}: {e}")
                import traceback; traceback.print_exc()

        if not seed_results:
            results[category] = {"error": "all seeds failed"}
        else:
            # average across seeds
            num_concepts = seed_results[0]["num_concepts"]
            avg_auc = np.mean([r["auc"] for r in seed_results], axis=0).tolist()
            std_auc = np.std([r["auc"] for r in seed_results], axis=0).tolist()
            avg_f1  = np.mean([r["f1"]  for r in seed_results], axis=0).tolist()
            std_f1  = np.std([r["f1"]  for r in seed_results], axis=0).tolist()
            results[category] = {
                "auc": avg_auc, "auc_std": std_auc,
                "f1":  avg_f1,  "f1_std":  std_f1,
                "num_concepts": num_concepts,
                "n_seeds": len(seed_results),
            }
            print(f"  Done ({len(seed_results)} seeds). "
                  f"AUC[0]={avg_auc[0]:.3f} -> AUC[-1]={avg_auc[-1]:.3f}")

        with open(args.save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"  Saved to {args.save_path}")

    print(f"\nAll done. Results saved to {args.save_path}")


if __name__ == "__main__":
    main()
