import numpy as np
import torch

def compute_intervention_order(pred_df, n_trials = 1):

    attr_preds = pred_df[pred_df.attr_cols]

    all_intervention_order_list = []

    for _ in range(n_trials):
        intervention_order_list = []
        for image_idx in range(len(pred_df)):
            attr_pred = np.array(attr_preds[image_idx])

            entropy = 1 / (np.abs(attr_pred - 0.5) ** 2 + 1e-8)
            intervention_order = np.argsort(entropy)[::-1]

            intervention_order_list.append(intervention_order)
        
        all_intervention_order_list.append(intervention_order_list)
    
    return all_intervention_order_list


def modify_concepts(intervention_order, gt_df, pred_df, ptl_5, ptl_95, n_replaced, trial):

    attr_cols = pred_df.attr_cols

    modified_df = pred_df.copy()

    for i in range(len(pred_df)):
        attr_idxs = intervention_order[trial][i][:n_replaced]

        for attr_idx in attr_idxs:
            attr_name = attr_cols[attr_idx]
            print(f"Modifying concept {attr_name}...")

            binary_val = gt_df.iloc[i][attr_name]

            new_val = (1 - binary_val) * ptl_5[attr_idx] + binary_val * ptl_95[attr_idx]
            modified_df.at[i, attr_name] = new_val
    
    return modified_df


