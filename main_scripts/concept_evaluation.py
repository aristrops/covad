import pandas as pd
import torch 
import argparse

from datasets.concept_dataset import ConceptDataset
from utils.model_utils import generate_concept_logits
from utils.metrics import compute_pearson_correlation, compute_leakage, compute_dci, compute_ois
from models.full_models import concepts_model

def compute_metrics(category: str,
                    dataframe_path: str,
                    model_path_concepts: str,
                    backbone: str,
                    metrics: list,
                    device: torch.device):

    device = torch.device(device)
    
    print(f"Computing concept quality for {category} category...")
    dataframe = pd.read_csv(dataframe_path)

    state_dict_concepts = torch.load(model_path_concepts)

    train_dataset = ConceptDataset(dataframe, split="train", use_attr=True, load_image=False)
    gt_concepts_train = train_dataset.df[train_dataset.attr_cols]
    num_attr = len(train_dataset.attr_cols) 
    y_train = train_dataset.df["label_index"].values

    test_dataset = ConceptDataset(dataframe, split="test", use_attr=True, load_image=False)
    gt_concepts_test = test_dataset.df[test_dataset.attr_cols]
    y_test = test_dataset.df["label_index"].values

    concept_model = concepts_model(num_attr=num_attr, freeze_parameters=True, 
                                expand_dim=0, model_state_dict=state_dict_concepts, backbone=backbone, mode = "test")
    
    predicted_dataframe = generate_concept_logits(concept_model, dataframe, save_path=None, device = device)

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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataframe_path", type=str, help="Path to dataframe")
    parser.add_argument("--categories", type = str, nargs="+", help="Which categories to consider")
    parser.add_argument("--model_path_concepts", type=str, help="Path to trained concept model")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--backbone", type = str, default="resnet18", help = "Which pre-trained network to use for concept extraction")
    parser.add_argument("--metrics", type = str, nargs="+", help = "Which metrics to compute")

    args = parser.parse_args()

    for category in args.categories:
        compute_metrics(category, args.dataframe_path, args.model_path_concepts, args.backbone, args.metrics, args.device)

if __name__ == "__main__":
    main()