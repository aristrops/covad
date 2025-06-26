import argparse
import torch
import pandas as pd
import gc
import os

from utils.model_utils import generate_concept_logits
from datasets.mvtec_concept_dataset import MvTecConceptDataset
from models.full_models import joint_model, standard_model, concepts_model, main_model
from trainer import ResNetTrainer

def train_model(dataframe_path: str, 
                model_type: str, 
                device: torch.device, 
                backbone: str,
                lambda_: float,
                batch_size: int = 32, 
                optimizer: str = "adam", 
                lr: float = 1e-3, 
                epochs: int = 100, 
                use_concepts: bool = True, 
                freeze_parameters: bool = True, 
                save_path: str = None,
                save_path_concepts: str = None,
                save_path_new_df: str = None, 
                model_path: str = None):
    
    def load_dataset(df, split, use_attr = True, load_image = True):
        return MvTecConceptDataset(df, split=split, use_attr=use_attr, load_image=load_image)

    def make_dataloader(dataset, shuffle = True):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def init_optimizer(params):
        if optimizer == "adam":
            return torch.optim.Adam(params, lr = lr)
        
    dataframe = pd.read_csv(dataframe_path)
    concepts = [col for col in dataframe.columns if col not in ["image_path", "label_index", "mask_path", "split", "anomaly_type"]]
    num_attr = len(concepts) if use_concepts else None

    state_dict = torch.load(model_path) if model_path else None

    train_dataset = load_dataset(dataframe, "train", use_attr=use_concepts)
    val_dataset = load_dataset(dataframe, "val", use_attr=use_concepts)
    train_dataloader = make_dataloader(train_dataset)
    val_dataloader = make_dataloader(val_dataset, shuffle = False)

    if model_type == "independent":
        train_dataset_no_img = load_dataset(dataframe, "train", load_image=False)
        val_dataset_no_img = load_dataset(dataframe, "val", load_image=False)
        train_dataloader_no_img = make_dataloader(train_dataset_no_img)
        val_dataloader_no_img = make_dataloader(val_dataset_no_img)

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    imbalance_ratio, label_counts = train_dataset.find_class_imbalance("main")
    print("Imbalance Ratio (negatives per positive) in training set:", imbalance_ratio)
    print("Label Counts in training set:", label_counts)

    imbalance_ratio_attr = None
    if use_concepts:
        imbalance_ratio_attr, label_counts_attr = train_dataset.find_class_imbalance("attributes")

    #initialize the model
    if model_type == "joint":
        model = joint_model(num_attr=num_attr, expand_dim=0, use_relu = True, use_sigmoid=False, 
                            freeze_parameters=freeze_parameters, model_state_dict=state_dict, backbone=backbone)
        model.to(device)
        optimizer =init_optimizer(model.parameters())
        trainer = ResNetTrainer(model, num_attr, train_dataloader, val_dataloader, optimizer, 
                                device, lambda_ = lambda_, num_epochs=epochs, concepts=use_concepts, 
                                weight_main=imbalance_ratio, weight_attr=imbalance_ratio_attr, save_path=save_path) 
        trainer.train() 
    
    elif model_type == "standard":
        model = standard_model(freeze_parameters=freeze_parameters, model_state_dict=state_dict, backbone=backbone)
        model.to(device)
        optimizer = init_optimizer(model.parameters())
        trainer = ResNetTrainer(model, num_attr, train_dataloader, val_dataloader, optimizer, 
                                device, lambda_ = lambda_, num_epochs=epochs, concepts=False, 
                                main_only=True, weight_main=imbalance_ratio, save_path=save_path) 
        trainer.train() 

    elif model_type == "independent" or model_type == "sequential":
        #common first step: attribute prediction
        concept_model = concepts_model(num_attr=num_attr, freeze_parameters=freeze_parameters, 
                                       expand_dim=0, model_state_dict=state_dict, backbone=backbone)
        concept_model.to(device)
        concept_optimizer = init_optimizer(concept_model.parameters())

        trainer_concepts = ResNetTrainer(concept_model, num_attr, train_dataloader, val_dataloader, concept_optimizer, 
                                         device, lambda_ = lambda_, num_epochs=epochs, concepts=True, 
                                         bottleneck=True, weight_attr=imbalance_ratio_attr, save_path=save_path_concepts)
        trainer_concepts.train()

        #main model: common
        main_task_model = main_model(num_attr=num_attr, expand_dim = 1)
        main_task_model.to(device)
        main_optimizer = init_optimizer(main_task_model.parameters())

        if model_type == "independent":
            trainer_main =  ResNetTrainer(main_task_model, num_attr, train_dataloader_no_img, val_dataloader_no_img, 
                                        main_optimizer, device, lambda_ = lambda_, num_epochs=epochs, 
                                        concepts=False, main_only=True, weight_main=imbalance_ratio, save_path=save_path)
            trainer_main.train()
        
        else:
            #generate concept logits
            generate_concept_logits(concept_model, dataframe, save_path = save_path_new_df)
            dataframe_new = pd.read_csv(save_path_new_df)

            train_dataset_no_img = load_dataset(dataframe_new, "train", load_image=False)
            val_dataset_no_img = load_dataset(dataframe_new, "val", load_image=False)

            train_dataloader_no_img = make_dataloader(train_dataset_no_img)
            val_dataloader_no_img = make_dataloader(val_dataset_no_img)

            trainer_main =  ResNetTrainer(main_task_model, num_attr, train_dataloader_no_img, val_dataloader_no_img, 
                                        main_optimizer, device, lambda_ = lambda_, num_epochs=epochs, 
                                        concepts=False, main_only=True, weight_main=imbalance_ratio, save_path=save_path)
            trainer_main.train()


    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataframe_path", type=str, help="Path of the directory that contains the dataframe")
    parser.add_argument("--model_type", type = str, help="Which model to train, e.g. 'independent', 'joint', ...")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--backbone", type = str, default="resnet18", help = "Which pre-trained network to use for concept extraction")
    parser.add_argument("--batch_size", type=int, default = 8, help="Batch size to use")
    parser.add_argument("--lambda_", type=float, help="How much weight to give to the attributes")
    parser.add_argument("--optimizer", type=str, default = "adam", help="Which optimizer to use")
    parser.add_argument("--lr", type = float, default = 1e-3, help="Learning rate to use")
    parser.add_argument("--epochs", type=int, default=100, help="How many epochs to run the training for")
    parser.add_argument("--use_concepts", action="store_true", help="Whether to use concepts or not")
    parser.add_argument("--freeze_parameters", action="store_true", help="Whether to freeze the parameters of the network for concept prediction")
    parser.add_argument("--save_path", type=str, default = None, help="Where to save the model")
    parser.add_argument("--save_path_concepts", type=str, default = None, help="Where to save the concepts model")
    parser.add_argument("--save_path_new_df", type=str, default = None, help="Where to save dataframe with new logits (sequential model)")
    parser.add_argument("--model_path", type=str, default = None, help="If specified, loads the state dict of a chosen model")
    parser.add_argument("--seed", type=int, default=42, help="Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    train_model(args.dataframe_path, args.model_type, device, args.backbone, args.lambda_, args.batch_size, args.optimizer, args.lr, args.epochs, args.use_concepts, args.freeze_parameters, args.save_path, args.save_path_concepts, args.save_path_new_df, args.model_path)


if __name__ == "__main__":
    main()