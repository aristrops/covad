import torch
import os
import numpy as np
import pandas as pd

from datasets.mvtec_concept_dataset import MvTecConceptDataset

def inference(image, model, use_relu = False, use_sigmoid = False):
    model.eval()

    outputs = model(image)

    if use_relu:
        attr_outputs = [torch.relu(o) for o in outputs]
    elif use_sigmoid:
        attr_outputs = [torch.sigmoid(o) for o in outputs]
    else:
        attr_outputs = outputs
    
    attr_outputs = torch.cat([o.unsqueeze(1) for o in attr_outputs], dim = 1).squeeze()

    return list(attr_outputs.data.cpu().numpy())


def generate_concept_logits(model, dataframe, save_path, splits = ["train", "val"], use_relu = False, use_sigmoid = False):
    
    updated_rows = []

    for split in splits:
        dataset = MvTecConceptDataset(dataframe, split = split, load_image = True, apply_transformation=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        for i, sample in enumerate(loader):
            image, label, _ = sample

            logits = inference(image, model, use_relu, use_sigmoid)

            #copy row and replace attribute label with logits
            row = dataset.df.iloc[i].copy()
            for j, attr in enumerate(dataset.attr_cols):
                row[attr] = logits[j] if isinstance(logits, (list, np.ndarray)) else logits

            updated_rows.append(row)

    #combine rows
    new_df = pd.DataFrame(updated_rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    new_df.to_csv(save_path, index = False)
    print(f"Saved new logits dataset to {save_path}")