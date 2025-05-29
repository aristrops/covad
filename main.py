import argparse
import torch
import pandas as pd
import gc

from datasets.mvtec_concept_dataset import MvTecConceptDataset
from models.final_models import joint_model, standard_model
from trainers.resnet_trainer import ResNetTrainer

def train_model(dataframe_path: str, model: str, device: torch.device, lambda_: float, batch_size: int = 32, optimizer: str = "adam", lr: float = 1e-3, epochs: int = 100, use_concepts = True):

    dataframe = pd.read_csv(dataframe_path)
    concepts = [col for col in dataframe.columns if col not in ["image_path", "label_index", "mask_path", "split", "anomaly_type"]]
    num_attr = len(concepts)

    train_dataset = MvTecConceptDataset(dataframe, split = "train")
    print(f"Number of training images: {len(train_dataset)}")
    imbalance_ratio, label_counts = train_dataset.find_class_imbalance()
    weight_tensor = torch.tensor([imbalance_ratio], dtype=torch.float32).to(device)
    print("Imbalance Ratio (negatives per positive) in training set:", imbalance_ratio)
    print("Label Counts in training set:", label_counts)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    val_dataset = MvTecConceptDataset(dataframe, split = "val")
    print(f"Number of validation images: {len(val_dataset)}")
    imbalance_ratio, label_counts = val_dataset.find_class_imbalance()
    print("Imbalance Ratio (negatives per positive) in validation set:", imbalance_ratio)
    print("Label Counts in validation set:", label_counts)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle = False)

    #define the model
    if model == "resnet-18" and use_concepts:
        model = joint_model(num_attr=num_attr, expand_dim=1, use_relu = True, use_sigmoid=False)
    if model == "resnet-18" and not use_concepts:
        model = standard_model()
    model.to(device)
    model.train()

    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    trainer = ResNetTrainer(model, num_attr, train_dataloader, val_dataloader, weight_tensor, optimizer, device, lambda_ = lambda_, num_epochs=epochs, concepts=use_concepts)  
    trainer.train() 

    del model
    del train_dataset
    del val_dataset
    del train_dataloader
    del val_dataloader
    torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataframe_path", type=str, help="Path of the directory that contains the dataframe")
    parser.add_argument("--model", type = str, help="Which model to train, e.g. 'resnet_18'")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--batch_size", type=int, default = 8, help="Batch size to use")
    parser.add_argument("--lambda_", type=float, help="How much weight to give to the attributes")
    parser.add_argument("--optimizer", type=str, default = "adam", help="Which optimizer to use")
    parser.add_argument("--lr", type = float, default = 1e-3, help="Learning rate to use")
    parser.add_argument("--epochs", type=int, default=100, help="How many epochs to run the training for")
    parser.add_argument("--use_concepts", action="store_true", help="Whether to use concepts or not")
    parser.add_argument("--seed", type=int, default=42, help="Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    train_model(args.dataframe_path, args.model, device, args.lambda_, args.batch_size, args.optimizer, args.lr, args.epochs, args.use_concepts)


if __name__ == "__main__":
    main()