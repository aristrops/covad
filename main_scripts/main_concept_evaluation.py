import pandas as pd

from datasets.mvtec_concept_dataset import MvTecConceptDataset
from utils.metrics import compute_pearson_correlation, compute_leakage, compute_dci, compute_ois

def compute_metrics(categories: str,
                    automated: bool,
                    metrics: list):
    
    for category in categories:
        print(f"Computing concept quality for category {category}...")
        if automated:
            dataframe = pd.read_csv(f"/mnt/disk1/arianna_stropeni/cbm_data/mvtec/{category}_dataset.csv")
            predicted_dataframe = pd.read_csv(f"/mnt/disk1/arianna_stropeni/cbm_data/predicted_concepts/{category}/independent_logits_automated.csv")
        else:
            dataframe = pd.read_csv(f"/mnt/disk1/arianna_stropeni/cbm_data/mvtec/{category}_dataset.csv")
            predicted_dataframe = pd.read_csv(f"/mnt/disk1/arianna_stropeni/cbm_data/predicted_concepts/{category}/independent_logits.csv")

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
            print(f"Concept leakage for {category} category: {leakage:.2f}")
        
        if "disentanglement" in metrics:
            disentanglement = compute_dci(predicted_concepts_train, gt_concepts_train)
            print(f"Concept disentanglement for {category} category: {disentanglement:.2f}")
        
        if "impurity" in metrics:
            ois = compute_ois(predicted_concepts_train, gt_concepts_train)
            print(f"OIS for {category} category: {ois:.2f}\n")

compute_metrics(["screw"], automated = True, metrics = ["leakage", "disentanglement", "impurity"])