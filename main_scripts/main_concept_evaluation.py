import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.concept_dataset import ConceptDataset
from utils.metrics import compute_pearson_correlation, compute_leakage, compute_dci, compute_ois

def compute_metrics(categories: str,
                    backbone: str,
                    automated: bool,
                    metrics: list):
    
    for category in categories:
        print(f"Computing concept quality for {category} category...")
        if automated:
            dataframe = pd.read_csv(f"/mnt/disk1/arianna_stropeni/cbm_data/realiad/{category}_dataset_automated.csv")
            predicted_dataframe = pd.read_csv(f"/mnt/disk1/arianna_stropeni/cbm_data/predicted_concepts/{category}/sequential_{backbone}_logits_automated.csv")
        else:
            dataframe = pd.read_csv(f"/mnt/disk1/arianna_stropeni/cbm_data/realiad/{category}_dataset.csv")
            predicted_dataframe = pd.read_csv(f"/mnt/disk1/arianna_stropeni/cbm_data/predicted_concepts/{category}/sequential_{backbone}_logits.csv")

        train_dataset = ConceptDataset(dataframe, split="train", use_attr=True, load_image=False)
        gt_concepts_train = train_dataset.df[train_dataset.attr_cols]
        y_train = train_dataset.df["label_index"].values

        test_dataset = ConceptDataset(dataframe, split="test", use_attr=True, load_image=False)
        gt_concepts_test = test_dataset.df[test_dataset.attr_cols]
        y_test = test_dataset.df["label_index"].values

        train_dataset_predicted = ConceptDataset(predicted_dataframe, split="train", use_attr=True, load_image=False)
        predicted_concepts_train = train_dataset_predicted.df[train_dataset_predicted.attr_cols]

        test_dataset_predicted = ConceptDataset(predicted_dataframe, split="test", use_attr=True, load_image=False)
        predicted_concepts_test = test_dataset_predicted.df[test_dataset_predicted.attr_cols]

        sorted_concept_corr = compute_pearson_correlation(train_dataset)
        
        if "leakage" in metrics:
            leakage = compute_leakage(sorted_concept_corr, gt_concepts_train, gt_concepts_test, predicted_concepts_train, predicted_concepts_test, y_train, y_test)
            print(f"Concept leakage for {category} category: {leakage:.2f}")
        
        if "disentanglement" in metrics:
            disentanglement = compute_dci(predicted_concepts_train, gt_concepts_train)
            print(f"Concept disentanglement for {category} category: {disentanglement:.2f}")
        
        if "impurity" in metrics:
            ois = compute_ois(predicted_concepts_train, gt_concepts_train)
            print(f"OIS for {category} category: {ois:.2f}\n")

compute_metrics(["mint"], "mobilenet_v2", automated = True, metrics = ["leakage", "disentanglement", "impurity"])