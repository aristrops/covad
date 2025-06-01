import argparse
import torch
import pandas as pd
import gc

from datasets.mvtec_concept_dataset import MvTecConceptDataset
from models.final_models import joint_model, standard_model, concepts_model, main_model
from trainers.resnet_trainer import ResNetTrainer

def train_model(dataframe_path: str, model_type: str, device: torch.device, lambda_: float, batch_size: int = 32, optimizer: str = "adam", lr: float = 1e-3, epochs: int = 100, use_concepts = True, freeze_parameters = True):

    dataframe = pd.read_csv(dataframe_path)
    concepts = [col for col in dataframe.columns if col not in ["image_path", "label_index", "mask_path", "split", "anomaly_type"]]
    num_attr = len(concepts)

    train_dataset = MvTecConceptDataset(dataframe, split = "train")
    if model_type == "independent":
        train_dataset_no_img = MvTecConceptDataset(dataframe, split = "train", load_image=False)
    print(f"Number of training images: {len(train_dataset)}")
    imbalance_ratio, label_counts = train_dataset.find_class_imbalance("main")
    imabalance_ratio_attr, label_counts_attr = train_dataset.find_class_imbalance("attributes")
    print("Imbalance Ratio (negatives per positive) in training set:", imbalance_ratio)
    print("Label Counts in training set:", label_counts)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    if model_type == "independent":
        train_dataloader_no_img = torch.utils.data.DataLoader(train_dataset_no_img, batch_size = batch_size, shuffle = True)

    val_dataset = MvTecConceptDataset(dataframe, split = "val")
    if model_type == "independent":
        val_dataset_no_img = MvTecConceptDataset(dataframe, split = "val", load_image=False)
    print(f"Number of validation images: {len(val_dataset)}")
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle = False)
    if model_type == "independent":
        val_dataloader_no_img = torch.utils.data.DataLoader(val_dataset_no_img, batch_size = batch_size, shuffle = True)

    #define the model
    if model_type == "joint":
        model = joint_model(num_attr=num_attr, expand_dim=1, use_relu = True, use_sigmoid=False, freeze_parameters=freeze_parameters)
        model.to(device)
        model.train()
    elif model_type == "independent" or model_type == "sequential":
        concept_model = concepts_model(num_attr=num_attr, expand_dim=1)
        concept_model.to(device)
        concept_model.train()
        main_task_model = main_model(num_attr=num_attr, expand_dim = 1)
        main_task_model.to(device)
        main_task_model.train()
    elif model_type == "standard":
        model = standard_model()
        model.to(device)
        model.train()
    

    if optimizer == "adam":
        if model_type == "standard" or model_type == "joint":
            optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        elif model_type == "independent":
            concept_optimizer = torch.optim.Adam(concept_model.parameters(), lr = lr)
            main_optimizer = torch.optim.Adam(main_task_model.parameters(), lr = lr)

    if model_type == "joint":
        trainer = ResNetTrainer(model, num_attr, train_dataloader, val_dataloader, optimizer, device, lambda_ = lambda_, num_epochs=epochs, concepts=use_concepts, weight_main=imbalance_ratio, weight_attr=imabalance_ratio_attr) 
        trainer.train() 
    elif model_type == "independent":
        # trainer_concepts = ResNetTrainer(concept_model, num_attr, train_dataloader, val_dataloader, concept_optimizer, device, lambda_ = lambda_, num_epochs=epochs, concepts=True, bottleneck=True, weight_attr=imabalance_ratio_attr)
        # trainer_concepts.train()
        trainer_main =  ResNetTrainer(main_task_model, num_attr, train_dataloader_no_img, val_dataloader_no_img, main_optimizer, device, lambda_ = lambda_, num_epochs=epochs, concepts=False, no_img=True, weight_main=imbalance_ratio)
        trainer_main.train()

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
    parser.add_argument("--model_type", type = str, help="Which model to train, e.g. 'independent', 'joint', ...")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--batch_size", type=int, default = 8, help="Batch size to use")
    parser.add_argument("--lambda_", type=float, help="How much weight to give to the attributes")
    parser.add_argument("--optimizer", type=str, default = "adam", help="Which optimizer to use")
    parser.add_argument("--lr", type = float, default = 1e-3, help="Learning rate to use")
    parser.add_argument("--epochs", type=int, default=100, help="How many epochs to run the training for")
    parser.add_argument("--use_concepts", action="store_true", help="Whether to use concepts or not")
    parser.add_argument("--freeze_parameters", action="store_true", help="Whether to freeze the parameters of the network for concept prediction")
    parser.add_argument("--seed", type=int, default=42, help="Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    train_model(args.dataframe_path, args.model_type, device, args.lambda_, args.batch_size, args.optimizer, args.lr, args.epochs, args.use_concepts, args.freeze_parameters)


if __name__ == "__main__":
    main()