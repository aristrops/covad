import argparse
import torch
import pandas as pd
import gc

from datasets.mvtec_concept_dataset import MvTecConceptDataset
from models.resnet_18 import ResNet18model
from trainers.resnet_trainer import ResNetTrainer

def train_model(dataframe_path: str, model: str, device: torch.device, batch_size: int = 32, optimizer: str = "adam", lr: float = 1e-3, epochs: int = 100):

    dataframe = pd.read_csv(dataframe_path)
    concepts = [col for col in dataframe.columns if col not in ["image_path", "label_index", "mask_path", "split", "path"]]
    num_attr = len(concepts)

    train_dataset = MvTecConceptDataset(dataframe, split = "train")
    print(f"Number of training images: {len(train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    val_dataset = MvTecConceptDataset(dataframe, split = "val")
    print(f"Number of validation images: {len(val_dataset)}")
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle = False)

    #define the model
    if model == "resnet-18":
        model = ResNet18model(num_attr=num_attr)
    model.to(device)
    model.train()

    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    trainer = ResNetTrainer(model, train_dataloader, val_dataloader, optimizer, device, lambda_ = 0.1, num_epochs=epochs)  
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
    parser.add_argument("--batch_size", type=int, default = 32, help="Batch size to use")
    parser.add_argument("--optimizer", type=str, default = "adam", help="Which optimizer to use")
    parser.add_argument("--lr", type = float, default = 1e-3, help="Learning rate to use")
    parser.add_argument("--epochs", type=int, default=100, help="How many epochs to run the training for")
    parser.add_argument("--seed", type=int, default=42, help="Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    train_model(args.dataframe_path, args.model, device, args.batch_size, args.optimizer, args.lr, args.epochs)


if __name__ == "__main__":
    main()