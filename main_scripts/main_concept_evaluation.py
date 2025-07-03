import pandas as pd
import numpy as np
import torch

from datasets.mvtec_concept_dataset import MvTecConceptDataset
from utils.concept_evaluation import compute_pearson_correlation, compute_leakage

def compute_metrics(category: str,
                    dataframe_path: str,
                    metrics: list,
                    predicted_dataframe_path: str = None):
    
    dataframe = pd.read_csv(dataframe_path)
    predicted_dataframe = pd.read_csv(predicted_dataframe_path)

    train_dataset = MvTecConceptDataset(dataframe, split="train", use_attr=True, load_image=False)
    gt_concepts_train = train_dataset.df[train_dataset.attr_cols]
    y_train = train_dataset.df["label_index"].values

    test_dataset = MvTecConceptDataset(dataframe, split="test", use_attr=True, load_image=False)
    gt_concepts_test = test_dataset.df[test_dataset.attr_cols]
    y_test = test_dataset.df["label_index"].values

    train_dataset_predicted = MvTecConceptDataset(predicted_dataframe, split="train", use_attr=True, load_image=False)
    predicted_concepts_train = train_dataset_predicted.df[train_dataset_predicted.attr_cols]

    test_dataset_predicted = MvTecConceptDataset(predicted_dataframe, split="test", use_attr=True, load_image=False)
    predicted_concepts_test = test_dataset_predicted.df[test_dataset_predicted.attr_cols]

    sorted_concept_corr = compute_pearson_correlation(train_dataset)
    
    if "leakage" in metrics:
        leakage = compute_leakage(sorted_concept_corr, gt_concepts_train, gt_concepts_test, predicted_concepts_train, predicted_concepts_test, y_train, y_test)
        print(f"Concept for {category} category: {leakage:.2f}")

compute_metrics("hazelnut", "/mnt/disk1/arianna_stropeni/cbm_data/mvtec/hazelnut_dataset.csv", metrics = ["leakage"], predicted_dataframe_path= "/mnt/disk1/arianna_stropeni/cbm_data/predicted_concepts/hazelnut/independent_logits.csv")