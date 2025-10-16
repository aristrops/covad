import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse

from matplotlib.ticker import MaxNLocator

from utils.intervention_utils import compute_intervention_order, modify_concepts
from datasets.concept_dataset import ConceptDataset
from models.full_models import joint_model, concepts_model, main_model
from evaluators.evaluator_cbm import CBMEvaluator
from utils.model_utils import generate_concept_logits


def simulate_concept_intervention(category: str,
                                  dataset: str, 
                                  model_type: str,
                                  device: torch.device,
                                  backbone: str,
                                  batch_size: int = 8,
                                  use_fusion: bool = False,
                                fusion_mode: str = "concat"):
    
    dataframe_path = f"/mnt/disk1/arianna_stropeni/cbm_data/{dataset}/{category}_dataset_automated.csv"
    model_path_student = f"/mnt/disk1/arianna_stropeni/cbm_models/stfpm_models/{category}_{backbone}.pth" if use_fusion else None

    if use_fusion:
        save_path = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/{model_type}_{backbone}_main_automated_fused_{fusion_mode}.pth"
    else:
        save_path = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/{model_type}_{backbone}_main_automated.pth" 

    dataframe = pd.read_csv(dataframe_path)
    gt_test_df = dataframe[dataframe["split"] == "test"]

    state_dict = torch.load(save_path) if save_path else None
    student_state_dict = torch.load(model_path_student) if use_fusion else None

    if model_type in ["independent", "sequential"]:
        if use_fusion:
            save_path_concepts = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/{model_type}_{backbone}_concepts_automated_fused_{fusion_mode}.pth"
        else:
            save_path_concepts = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/{model_type}_{backbone}_concepts_automated.pth"
            
        state_dict_concepts = torch.load(save_path_concepts) if save_path_concepts else None

    test_dataset = ConceptDataset(dataframe, "test", use_attr=True)
    num_attr = len(test_dataset.attr_cols) 
    attr_cols = test_dataset.attr_cols

    f1_scores = []

    #step 1: extract predicted concepts
    print(f"Extracting predicted concepts...")

    if model_type in ["sequential", "independent"]:
        concept_model = concepts_model(num_attr=num_attr, freeze_parameters=True, 
                                    expand_dim=0, model_state_dict=state_dict_concepts, backbone=backbone, mode = "test", 
                                    use_fusion=use_fusion, student_state_dict=student_state_dict, fusion_mode=fusion_mode)
    elif model_type == "joint":
        concept_model, main_task_model = joint_model(num_attr=num_attr, freeze_parameters=True, expand_dim=0, model_state_dict=state_dict, mode = "test", concept_intervention=True,
                                                      use_fusion=use_fusion, fusion_mode=fusion_mode, student_state_dict=student_state_dict)
        
    concept_model.to(device)

    pred_df = generate_concept_logits(concept_model, dataframe, save_path = None, device = device)
    pred_df_test = pred_df[pred_df["split"] == "test"]

    #compute 5th and 95th percentile over the training distribution
    logits_array = pred_df[attr_cols][pred_df["split"] == "train"].values
    ptl_5 = np.percentile(logits_array, 5, axis = 0)
    ptl_95 = np.percentile(logits_array, 95, axis = 0)
    ptl_5, ptl_95 = dict(zip(attr_cols, ptl_5)), dict(zip(attr_cols, ptl_95))

    #step 2: modify concepts based on intervention order
    print(f"Computing intervention order...")
    intervention_order = compute_intervention_order(pred_df_test, attr_cols)

    for i in range(num_attr):
        print(f"\nInervenening on the first {i + 1} concept(s)...")
        modified_df = modify_concepts(intervention_order, gt_test_df, pred_df_test, attr_cols, ptl_5, ptl_95, i+1)

        #step 3: perform inference over the main task using the new df
        new_test_dataset = ConceptDataset(modified_df, "test", use_attr=True, load_image=False)
        attr_cols = new_test_dataset.attr_cols
        new_test_dataloader = torch.utils.data.DataLoader(new_test_dataset, batch_size, shuffle = False)

        print(f"Performing inference of {model_type} model using the new concepts...")
        if model_type in ["sequential", "independent"]:
            main_task_model = main_model(num_attr=num_attr, expand_dim = 8, model_state_dict=state_dict)

        main_task_model.to(device)

        main_evaluator = CBMEvaluator(main_task_model, num_attr, attr_cols, new_test_dataloader, device, main_only = True)
        f1_main = main_evaluator.evaluate()

        f1_scores.append(f1_main)
    
    return f1_scores


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="Dataset to use (MvTec or Real-IAD)")
    parser.add_argument("--model_types", type = str, nargs = "+", help="Which model to train, e.g. 'independent', 'joint', ...")
    parser.add_argument("--categories", type = str, nargs="+", help="Which categories to train/test")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--backbone", type = str, default="resnet18", help = "Which pre-trained network to use for concept extraction")
    parser.add_argument("--batch_size", type=int, default = 16, help="Batch size to use")
    parser.add_argument("--n_replaced", type=int, default = 1, help="How many concepts to intervene on")
    parser.add_argument("--save_plot", action = "store_true", help="Whether to save the plot")
    parser.add_argument("--standard_f1", type=float, default = None, help="F1 Score achieved by the standard model")
    parser.add_argument("--seed", type=int, default=42, help="Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    all_scores = {model_type: [] for model_type in args.model_types}

    for category in args.categories:
        for model_type in args.model_types:
            f1_scores = simulate_concept_intervention(category, args.dataset, model_type, device, args.backbone, args.batch_size)
            all_scores[model_type] = f1_scores
        
        if args.save_plot:
            x = list(range(1, len(f1_scores) + 1))

            colors = {
                        "joint": "darkviolet",
                        "sequential": "dodgerblue",
                        "independent": "plum"
                    }

            for model_type, f1_list in all_scores.items():
                plt.plot(x, f1_list, marker='o', label=model_type, color = colors[model_type], markerfacecolor='none', markeredgewidth=1.5)
            
            plt.axhline(y=args.standard_f1, color='lightcoral', linestyle='--', label='standard')
            
            plt.xlabel("Number of Intervened Concepts")
            plt.ylabel("AD F1 Score")
            plt.legend()
            plt.grid(True)

            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

            plt.tight_layout()
            plt.savefig(f"plots/f1_vs_intervention_{category}_{args.backbone}.png")

if __name__ == "__main__":
    main()

    

