import numpy as np
import pandas as pd

def compute_intervention_order(pred_df, attr_cols):

    attr_preds = pred_df[attr_cols]

    intervention_order_list = []
    all_entropies = []

    for image_idx in range(len(pred_df)):
        logits = np.array(attr_preds.iloc[image_idx])

        p = 1 / (1 + np.exp(-logits)) #convert to probabilities
        entropy = -(p * np.log(p + 1e-8) + (1 - p) * np.log(1 - p + 1e-8)) #compute Shannon entropy

        #entropy = 1 / (np.abs(attr_pred - 0.5) ** 2 + 1e-8)
        all_entropies.append(entropy)

        intervention_order = np.argsort(entropy)[::-1]
        intervention_order_list.append(intervention_order)
    
    all_entropies = np.stack(all_entropies, axis = 0)
    mean_entropy_per_attr = pd.Series(np.mean(all_entropies, axis=0), index=attr_cols)
    entropy_order = mean_entropy_per_attr.sort_values(ascending=False)

    print("\nAttribute uncertainty ranking:")
    for rank, (attr, uncertainty) in enumerate(entropy_order.items(), 1):
        print(f"{rank}. {attr} (mean uncertainty: {uncertainty:.4f})")
    
    return intervention_order_list


def modify_concepts(intervention_order, gt_df, pred_df, attr_cols, ptl_5, ptl_95, n_replaced):

    modified_df = pred_df.copy()

    for i in range(len(pred_df)):
        if n_replaced > 0:
            attr_idxs = intervention_order[i][:n_replaced]

            for attr_idx in attr_idxs:
                attr_name = attr_cols[attr_idx]

                binary_val = gt_df.iloc[i][attr_name]

                new_val = (1 - binary_val) * ptl_5[attr_name] + binary_val * ptl_95[attr_name]
                modified_df.at[i, attr_name] = new_val
    
    return modified_df


