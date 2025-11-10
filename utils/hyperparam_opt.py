import torch
import pandas as pd
import gc
import optuna

from utils.model_utils import generate_concept_logits
from datasets.concept_dataset import MvTecConceptDataset
from models.full_models import joint_model, standard_model, concepts_model, main_model
from trainers.trainer_cbm import ResNetTrainer

def load_dataset(df, split, use_attr = True, load_image = True, multiclass = False):
    return MvTecConceptDataset(df, split=split, use_attr=use_attr, load_image=load_image, multiclass=multiclass)

def make_dataloader(dataset, batch_size, shuffle = True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
                multiclass: bool = False,
                freeze_parameters: bool = True, 
                save_path: str = None,
                save_path_concepts: str = None,
                save_path_new_df: str = None, 
                model_path: str = None):
    
    def init_optimizer(params):
        if optimizer == "adam":
            opt = torch.optim.Adam(params, lr = lr)
        elif optimizer == "sgd":
            opt = torch.optim.SGD(params, lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode = "min", factor = 0.1, patience = 5)
        return opt, scheduler
        
    dataframe = pd.read_csv(dataframe_path)

    state_dict = torch.load(model_path) if model_path else None

    train_dataset = load_dataset(dataframe, "train", use_attr=use_concepts, multiclass=multiclass)
    val_dataset = load_dataset(dataframe, "val", use_attr=use_concepts, multiclass=multiclass)
    train_dataloader = make_dataloader(train_dataset, batch_size)
    val_dataloader = make_dataloader(val_dataset, batch_size, shuffle = False)

    num_classes = train_dataset.num_classes if multiclass else None
    if multiclass:
        print(f"Number of labels: {num_classes}")
    num_attr = len(train_dataset.attr_cols) if use_concepts else None

    if model_type == "independent":
        train_dataset_no_img = load_dataset(dataframe, "train", load_image=False)
        val_dataset_no_img = load_dataset(dataframe, "val", load_image=False)
        train_dataloader_no_img = make_dataloader(train_dataset_no_img, batch_size)
        val_dataloader_no_img = make_dataloader(val_dataset_no_img, batch_size)

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    if not multiclass:
        imbalance_ratio, contamination_ratio = train_dataset.find_class_imbalance("main") 
        print("Imbalance Ratio (negatives per positive) in training set:", imbalance_ratio)
        print("Contamination ratio in training set:", contamination_ratio)
    else:
        imbalance_ratio = None

    imbalance_ratio_attr = None
    if use_concepts:
        imbalance_ratio_attr = train_dataset.find_class_imbalance("attributes")

    print(f"Training {model_type} model...")

    #initialize the model
    if model_type == "joint":
        model = joint_model(num_attr=num_attr, expand_dim=0, use_relu = True, use_sigmoid=False, 
                            freeze_parameters=freeze_parameters, model_state_dict=state_dict, backbone=backbone)
        model.to(device)
        optimizer, scheduler = init_optimizer(model.parameters())
        trainer = ResNetTrainer(model, num_attr, train_dataloader, val_dataloader, optimizer, scheduler, 
                                device, lambda_ = lambda_, num_epochs=epochs, concepts=use_concepts, 
                                weight_main=imbalance_ratio, weight_attr=imbalance_ratio_attr, save_path=save_path) 
        val_main_f1, val_attr_f1 = trainer.train() 
    
    elif model_type == "standard":
        if multiclass:
            model = standard_model(freeze_parameters=freeze_parameters, model_state_dict=state_dict, backbone=backbone, num_classes=num_classes)
        else:
            model = standard_model(freeze_parameters=freeze_parameters, model_state_dict=state_dict, backbone=backbone, num_classes=1)
        model.to(device)
        optimizer, scheduler = init_optimizer(model.parameters())
        trainer = ResNetTrainer(model, num_attr, train_dataloader, val_dataloader, optimizer, scheduler,
                                device, lambda_ = lambda_, num_epochs=epochs, concepts=False, 
                                main_only=True, multiclass=multiclass, weight_main=imbalance_ratio, save_path=save_path) 
        val_main_f1 = trainer.train() 

    elif model_type == "independent" or model_type == "sequential":
        # common first step: attribute prediction
        concept_model = concepts_model(num_attr=num_attr, freeze_parameters=freeze_parameters, 
                                       expand_dim=0, model_state_dict=state_dict, backbone=backbone)
        concept_model.to(device)
        concept_optimizer, concept_scheduler = init_optimizer(concept_model.parameters())

        trainer_concepts = ResNetTrainer(concept_model, num_attr, train_dataloader, val_dataloader, concept_optimizer, concept_scheduler,
                                         device, lambda_ = lambda_, num_epochs=epochs, concepts=True, 
                                         bottleneck=True, weight_attr=imbalance_ratio_attr, save_path=save_path_concepts)
        val_attr_f1 = trainer_concepts.train()

        #possibly save df with concept logits
        dataframe_new = generate_concept_logits(concept_model, dataframe, save_path = save_path_new_df)

        #main model: common
        main_task_model = main_model(num_attr=num_attr, expand_dim = 1)
        main_task_model.to(device)
        main_optimizer, main_scheduler = init_optimizer(main_task_model.parameters())

        if model_type == "independent":
            trainer_main =  ResNetTrainer(main_task_model, num_attr, train_dataloader_no_img, val_dataloader_no_img, 
                                        main_optimizer, main_scheduler, device, lambda_ = lambda_, num_epochs=epochs, 
                                        concepts=False, main_only=True, weight_main=imbalance_ratio, save_path=save_path)
            val_main_f1 = trainer_main.train()
        
        else:
            #generate concept logits
            train_dataset_no_img = load_dataset(dataframe_new, "train", load_image=False)
            val_dataset_no_img = load_dataset(dataframe_new, "val", load_image=False)

            train_dataloader_no_img = make_dataloader(train_dataset_no_img, batch_size)
            val_dataloader_no_img = make_dataloader(val_dataset_no_img, batch_size)

            trainer_main =  ResNetTrainer(main_task_model, num_attr, train_dataloader_no_img, val_dataloader_no_img, 
                                        main_optimizer, main_scheduler, device, lambda_ = lambda_, num_epochs=epochs, 
                                        concepts=False, main_only=True, weight_main=imbalance_ratio, save_path=save_path)
            val_main_f1 = trainer_main.train()


    torch.cuda.empty_cache()
    gc.collect()
    
    return val_main_f1, val_attr_f1


def objective(trial, dataframe_path):
    model_type = "joint"

    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    if model_type == "joint":
        lambda_ = trial.suggest_uniform("lambda", 0.0, 1.0)
    else:
        lambda_ = 1

    val_main_f1, val_attr_f1 = train_model(dataframe_path = dataframe_path,
                                           model_type=model_type,
                                           device = "cpu",
                                           backbone = "resnet18",
                                           lambda_ = lambda_,
                                           batch_size=batch_size,
                                           optimizer=optimizer,
                                           lr = lr,
                                           epochs = 100,
                                           use_concepts=True,
                                           freeze_parameters=False)
    
    return val_main_f1, val_attr_f1

data_path = "path/to/df.csv"

study = optuna.create_study(directions=["maximize", "maximize"])
study.optimize(lambda trial: objective(trial, data_path), n_trials=20)
pareto_front = study.best_trials
for trial in pareto_front:
    print(f"Main F1: {trial.values[0]:.3f}, Concept F1: {trial.values[1]:.3f}")
    print(f"Params: {trial.params}")


