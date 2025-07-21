import numpy as np
import pandas as pd
import torch

from concept_intervention import compute_intervention_order, modify_concepts
from datasets.concept_dataset import ConceptDataset
from models.full_models import joint_model, concepts_model, main_model
from evaluators.evaluator_cbm import CBMEvaluator
from utils.model_utils import generate_concept_logits

# def extract_concept_labels(dataset):
#     all_attr_labels = []

#     for i in range(len(dataset)):
#         attr_label, _ = dataset[i][1:3] #ignore label
#         all_attr_labels.append(attr_label)
    
#     all_attr_labels = torch.stack(all_attr_labels).numpy() #shape [n_test, num_attr]

#     return all_attr_labels

def simulate_concept_intervention(category: str,
                                  dataset: str, 
                                  model_type: str,
                                  device: torch.device,
                                  backbone: str,
                                  batch_size: int = 8,
                                  n_replaced: int = 1):
    
    dataframe_path = f"/mnt/disk1/arianna_stropeni/cbm_data/{dataset}/{category}_dataset_automated.csv"
    save_path = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/{model_type}_{backbone}_main_automated.pth"

    dataframe = pd.read_csv(dataframe_path)

    state_dict = torch.load(save_path) if save_path else None

    if model_type in ["independent", "sequential"]:
        save_path_concepts = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/{model_type}_{backbone}_concepts_automated.pth"
        state_dict_concepts = torch.load(save_path_concepts) if save_path_concepts else None

    test_dataset = ConceptDataset(dataframe, "test", use_attr=True)
    num_attr = len(test_dataset.attr_cols) 


    #step 1: extract predicted concepts
    print(f"Extracting predicted concepts...")

    concept_model = concepts_model(num_attr=num_attr, freeze_parameters=True, 
                                expand_dim=0, model_state_dict=state_dict_concepts, backbone=backbone, mode = "test")
        
    concept_model.to(device)

    pred_df = generate_concept_logits(concept_model, dataframe, save_path = None)

    pred_df_test = ConceptDataset(pred_df, "test", load_image=False)
    pred_df_train = ConceptDataset(pred_df, "train", load_image=False)

    #compute 5th and 95th percentile over the training distribution
    logits_array = pred_df_train[pred_df_train.attr_cols].values
    ptl_5 = np.percentile(logits_array, 5, axis = 0)
    ptl_95 = np.percentile(logits_array, 95, axis = 0)
    ptl_5, ptl_95 = dict(zip(dataset.attr_cols, ptl_5)), dict(zip(dataset.attr_cols, ptl_95))

    #step 2: modify concepts based on intervention order
    intervention_order = compute_intervention_order(pred_df)
    modified_df = modify_concepts(intervention_order, test_dataset, pred_df, ptl_5, ptl_95, n_replaced)

    #step 3: perform inference over the main task using the new df
    test_dataloader = torch.utils.data.DataLoader(modified_df, batch_size, shuffle = False)

    main_task_model = main_model(num_attr=num_attr, expand_dim = 8, model_state_dict=state_dict)
    main_task_model.to(device)

    main_evaluator = CBMEvaluator(main_task_model, num_attr, test_dataloader, device, main_only = True)
    main_evaluator.evaluate()

    

