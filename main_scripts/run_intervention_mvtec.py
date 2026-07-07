"""
Batch concept intervention experiment across all MVTec categories.

Two modes:
  sag        — SAG model (trained on generated anomalies). Tests on ALL real MVTec
               anomaly images (any CSV split) + CSV-test normals.
  supervised — Fully supervised model (trained on real anomalies). Tests on the
               standard CSV test split only (training anomalies excluded).

Run from covad/ directory:
    uv run main_scripts/run_intervention_mvtec.py --mode sag        --device cuda --save_path results/intervention_mvtec_sag_real.pkl
    uv run main_scripts/run_intervention_mvtec.py --mode supervised --device cuda --save_path results/intervention_mvtec_supervised.pkl
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from datasets.concept_dataset import ConceptDataset
from evaluators.evaluator_cbm import CBMEvaluator
from models.full_models import joint_model
from utils.intervention_utils import compute_intervention_order, modify_concepts
from utils.model_utils import generate_concept_logits


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

BACKBONE = "mobilenet_v2"
CHECKPOINT_NAME = f"{BACKBONE}_1.0ratio_0MLP_automated.pth"


def get_paths(base_dir: str, category: str, mode: str):
    if mode == "sag":
        df_path = os.path.join(
            base_dir, "cbm_data", "mvtec",
            f"{category}_dataset_automated_gen_concepts.csv",
        )
        model_path = os.path.join(
            base_dir, "cbm_models", "mvtec",
            f"{category}_models", "gen_anomalies", "joint",
            CHECKPOINT_NAME,
        )
    else:  # supervised
        df_path = os.path.join(
            base_dir, "cbm_data", "mvtec",
            f"{category}_dataset_automated.csv",
        )
        model_path = os.path.join(
            base_dir, "cbm_models", "mvtec",
            f"{category}_models", "original_anomalies", "joint",
            CHECKPOINT_NAME,
        )
    return df_path, model_path


def build_sag_test_df(dataframe):
    """
    SAG mode: test on ALL real anomaly images (any CSV split, no generated) +
              normal images labeled 'test' in the CSV.
    """
    real_anomalies = dataframe[
        (dataframe["label_index"] == 1) &
        (~dataframe["image_path"].str.contains("generated_anomalies"))
    ].copy()
    test_normals = dataframe[
        (dataframe["split"] == "test") &
        (dataframe["label_index"] == 0)
    ].copy()
    test_df = pd.concat([real_anomalies, test_normals], ignore_index=True)
    test_df["split"] = "test"
    return test_df


def build_eval_dataframe(dataframe, mode):
    """
    Returns (eval_dataframe, gt_test_df):
      eval_dataframe — full df passed to generate_concept_logits (train for percentiles + test)
      gt_test_df     — ground-truth rows for the test set used in intervention
    """
    if mode == "sag":
        test_df = build_sag_test_df(dataframe)
        # train = generated anomaly rows (for percentile computation)
        train_df = dataframe[
            dataframe["image_path"].str.contains("generated_anomalies") |
            ((dataframe["label_index"] == 0) & (dataframe["split"] != "test"))
        ].copy()
        eval_df = pd.concat([train_df, test_df], ignore_index=True)
    else:  # supervised
        eval_df = dataframe
        test_df = dataframe[dataframe["split"] == "test"].reset_index(drop=True)

    return eval_df, test_df.reset_index(drop=True)


def run_intervention(df_path, model_path, device, batch_size, mode, backbone=BACKBONE):
    dataframe = pd.read_csv(df_path)
    eval_dataframe, gt_test_df = build_eval_dataframe(dataframe, mode)

    print(f"  Test set: {len(gt_test_df)} images "
          f"({(gt_test_df['label_index']==1).sum()} anomalies, "
          f"{(gt_test_df['label_index']==0).sum()} normals)")

    state_dict = torch.load(model_path, weights_only=False)

    test_dataset = ConceptDataset(eval_dataframe, "test", use_attr=True)
    num_attr = len(test_dataset.attr_cols)
    attr_cols = test_dataset.attr_cols

    print("  Extracting concept logits...")
    concept_model, main_task_model = joint_model(
        num_attr=num_attr,
        freeze_parameters=True,
        expand_dim=0,
        backbone=backbone,
        model_state_dict=state_dict,
        mode="test",
        concept_intervention=True,
    )
    concept_model.to(device)

    pred_df = generate_concept_logits(concept_model, eval_dataframe, save_path=None, device=device)
    pred_df_test = pred_df[pred_df["split"] == "test"].reset_index(drop=True)

    train_logits = pred_df[attr_cols][pred_df["split"] == "train"].values
    ptl_5 = dict(zip(attr_cols, np.percentile(train_logits, 5, axis=0)))
    ptl_95 = dict(zip(attr_cols, np.percentile(train_logits, 95, axis=0)))

    print("  Computing intervention order...")
    intervention_order = compute_intervention_order(pred_df_test, attr_cols)

    auc_scores, f1_scores = [], []
    main_task_model.to(device)

    for n in range(0, num_attr + 1):
        print(f"  Intervening on {n}/{num_attr} concept(s)...")
        modified_df = modify_concepts(
            intervention_order, gt_test_df, pred_df_test, attr_cols, ptl_5, ptl_95, n
        )
        new_test_dataset = ConceptDataset(modified_df, "test", use_attr=True, load_image=False)
        new_attr_cols = new_test_dataset.attr_cols
        loader = torch.utils.data.DataLoader(new_test_dataset, batch_size, shuffle=False)

        evaluator = CBMEvaluator(
            main_task_model, num_attr, new_attr_cols, loader, device, main_only=True,
        )
        auc, f1 = evaluator.evaluate()
        auc_scores.append(auc)
        f1_scores.append(f1)

    return {"auc": auc_scores, "f1": f1_scores, "num_concepts": num_attr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["sag", "supervised"],
                        help="'sag': test on all real anomalies; 'supervised': use CSV test split")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--backbone", type=str, default=BACKBONE)
    parser.add_argument("--categories", type=str, nargs="+", default=MVTEC_CATEGORIES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    results = {}
    if os.path.exists(args.save_path):
        with open(args.save_path, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded existing results for {list(results.keys())}")

    for category in args.categories:
        print(f"\n{'='*60}\nCategory: {category} [{args.mode}]\n{'='*60}")

        if category in results and "error" not in results[category]:
            print(f"  SKIP — already computed.")
            continue

        df_path, model_path = get_paths(args.base_dir, category, args.mode)

        if not os.path.exists(df_path):
            print(f"  SKIP — dataframe not found: {df_path}")
            continue
        if not os.path.exists(model_path):
            print(f"  SKIP — checkpoint not found: {model_path}")
            continue

        try:
            results[category] = run_intervention(
                df_path, model_path, device, args.batch_size, args.mode, args.backbone
            )
            print(f"  Done. AUC[0]={results[category]['auc'][0]:.3f} -> AUC[-1]={results[category]['auc'][-1]:.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results[category] = {"error": str(e)}

        with open(args.save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"  Saved to {args.save_path}")

    print(f"\nAll done. Results saved to {args.save_path}")


if __name__ == "__main__":
    main()
