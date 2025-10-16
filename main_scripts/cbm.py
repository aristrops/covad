import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import pandas as pd
import gc

from ptflops import get_model_complexity_info

from utils.model_utils import generate_concept_logits
from datasets.concept_dataset import ConceptDataset
from models.full_models import joint_model, standard_model, concepts_model, main_model
from trainers.trainer_cbm import CBMTrainer
from evaluators.evaluator_cbm import CBMEvaluator

def load_dataset(df, split, use_attr = True, load_image = True, multiclass = False):
    return ConceptDataset(df, split=split, use_attr=use_attr, load_image=load_image, multiclass=multiclass)

def make_dataloader(dataset, batch_size, shuffle = True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_model(category: str, 
                dataset: str,
                model_type: str, 
                device: torch.device, 
                backbone: str,
                lambda_: float,
                batch_size: int = 16, 
                optimizer: str = "adam", 
                lr: float = 1e-3, 
                epochs: int = 100, 
                use_concepts: bool = True, 
                multiclass: bool = False,
                use_fusion: bool = False,
                fusion_mode: str = "concat",
                freeze_parameters: bool = False, 
                model_path: str = None,
                save_concepts: bool = False):
    
    def init_optimizer(params):
        if optimizer == "adam":
            opt = torch.optim.Adam(params, lr = lr)
        elif optimizer == "sgd":
            opt = torch.optim.SGD(params, lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode = "min", factor = 0.1, patience = 5)
        return opt, scheduler
    
        
    dataframe_path = f"/mnt/disk1/arianna_stropeni/cbm_data/{dataset}/{category}_dataset_automated.csv"
    model_path_student = f"/mnt/disk1/arianna_stropeni/cbm_models/stfpm_models/{category}_{backbone}.pth" if use_fusion else None

    if use_fusion:
        save_path = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/{model_type}_{backbone}_main_automated_fused_{fusion_mode}.pth"
        save_path_concepts = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/{model_type}_{backbone}_concepts_automated_fused_{fusion_mode}.pth"
        save_path_new_df = f"/mnt/disk1/arianna_stropeni/cbm_data/predicted_concepts/{category}/{model_type}_{backbone}_logits_automated_fused_{fusion_mode}.csv" if save_concepts else None
    else:
        save_path = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/{model_type}_{backbone}_main_automated.pth"
        save_path_concepts = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/{model_type}_{backbone}_concepts_automated.pth"
        save_path_new_df = f"/mnt/disk1/arianna_stropeni/cbm_data/predicted_concepts/{category}/{model_type}_{backbone}_logits_automated.csv" if save_concepts else None

    dataframe = pd.read_csv(dataframe_path)

    state_dict = torch.load(model_path) if model_path else None
    student_state_dict = torch.load(model_path_student) if use_fusion else None

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
        imbalance_ratio, label_counts = train_dataset.find_class_imbalance("main") 
        print("Imbalance Ratio (negatives per positive) in training set:", imbalance_ratio)
        print("Label Counts in training set:", label_counts)
    else:
        imbalance_ratio = None

    imbalance_ratio_attr = None
    if use_concepts:
        imbalance_ratio_attr, label_counts_attr = train_dataset.find_class_imbalance("attributes")

    print(f"Training {model_type} model for {category} category...")
    #initialize the model
    if model_type == "joint":
        model = joint_model(num_attr=num_attr, expand_dim=0, use_relu = True, use_sigmoid=False, 
                            freeze_parameters=freeze_parameters, model_state_dict=state_dict, backbone=backbone, use_fusion=use_fusion, fusion_mode=fusion_mode, student_state_dict=student_state_dict)
        model.to(device)
        optimizer, scheduler = init_optimizer(model.parameters())
        trainer = CBMTrainer(model, num_attr, train_dataloader, val_dataloader, optimizer, scheduler, 
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
        trainer = CBMTrainer(model, num_attr, train_dataloader, val_dataloader, optimizer, scheduler,
                                device, lambda_ = lambda_, num_epochs=epochs, concepts=False, 
                                main_only=True, multiclass=multiclass, weight_main=imbalance_ratio, save_path=save_path) 
        val_main_f1 = trainer.train() 

    elif model_type == "independent" or model_type == "sequential":
        # common first step: attribute prediction
        concept_model = concepts_model(num_attr=num_attr, freeze_parameters=freeze_parameters, 
                                       expand_dim=0, model_state_dict=state_dict, backbone=backbone, use_fusion=use_fusion, student_state_dict=student_state_dict, fusion_mode=fusion_mode)
        concept_model.to(device)
        concept_optimizer, concept_scheduler = init_optimizer(concept_model.parameters())

        trainer_concepts = CBMTrainer(concept_model, num_attr, train_dataloader, val_dataloader, concept_optimizer, concept_scheduler,
                                         device, lambda_ = lambda_, num_epochs=epochs, concepts=True, 
                                         bottleneck=True, weight_attr=imbalance_ratio_attr, save_path=save_path_concepts)
        val_attr_f1 = trainer_concepts.train()

        #main model: common
        main_task_model = main_model(num_attr=num_attr, expand_dim = 8)
        main_task_model.to(device)
        main_optimizer, main_scheduler = init_optimizer(main_task_model.parameters())

        if model_type == "sequential":
            #possibly save df with concept logits
            dataframe_new = generate_concept_logits(concept_model, dataframe, save_path = save_path_new_df, device = device)

            #generate concept logits
            train_dataset_no_img = load_dataset(dataframe_new, "train", load_image=False)
            val_dataset_no_img = load_dataset(dataframe_new, "val", load_image=False)

            train_dataloader_no_img = make_dataloader(train_dataset_no_img, batch_size)
            val_dataloader_no_img = make_dataloader(val_dataset_no_img, batch_size)

        trainer_main =  CBMTrainer(main_task_model, num_attr, train_dataloader_no_img, val_dataloader_no_img, 
                                    main_optimizer, main_scheduler, device, lambda_ = lambda_, num_epochs=epochs, 
                                    concepts=False, main_only=True, weight_main=imbalance_ratio, save_path=save_path)
        val_main_f1 = trainer_main.train()


    torch.cuda.empty_cache()
    gc.collect()


def test_model(category: str,
               dataset: str, 
               model_type: str, 
               device: torch.device, 
               backbone: str,
               batch_size: int = 16, 
               use_concepts: bool = True,
               use_fusion: bool = False,
               fusion_mode: str = "concat",
               save_concepts: bool = False):

    dataframe_path = f"/mnt/disk1/arianna_stropeni/cbm_data/{dataset}/{category}_dataset_automated.csv"
    model_path_student = f"/mnt/disk1/arianna_stropeni/cbm_models/stfpm_models/{category}_{backbone}.pth" if use_fusion else None

    if use_fusion:
        save_path = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/{model_type}_{backbone}_main_automated_fused_{fusion_mode}.pth"
    else:
        save_path = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/{model_type}_{backbone}_main_automated.pth"

    dataframe = pd.read_csv(dataframe_path)

    state_dict = torch.load(save_path) if save_path else None
    student_state_dict = torch.load(model_path_student) if use_fusion else None

    if model_type in ["independent", "sequential"]:
        if use_fusion:
            save_path_concepts = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/{model_type}_{backbone}_concepts_automated_fused_{fusion_mode}.pth"
            save_path_new_df = f"/mnt/disk1/arianna_stropeni/cbm_data/predicted_concepts/{category}/{model_type}_{backbone}_logits_automated_fused_{fusion_mode}.csv" if save_concepts else None
        else:
            save_path_concepts = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/{model_type}_{backbone}_concepts_automated.pth"
            save_path_new_df = f"/mnt/disk1/arianna_stropeni/cbm_data/predicted_concepts/{category}/{model_type}_{backbone}_logits_automated.csv" if save_concepts else None

        state_dict_concepts = torch.load(save_path_concepts) if save_path_concepts else None

    test_dataset = load_dataset(dataframe, "test", use_attr=use_concepts)
    test_dataloader = make_dataloader(test_dataset, batch_size, shuffle = False)
    num_attr = len(test_dataset.attr_cols) if use_concepts else None
    attr_cols = test_dataset.attr_cols

    print(f"Number of test images: {len(test_dataset)}")

    print(f"Testing {model_type} model for {category} category...")

    if model_type == "joint":
        model = joint_model(num_attr=num_attr, expand_dim=0, use_relu = True, use_sigmoid=False, 
                            freeze_parameters=True, model_state_dict=state_dict, backbone=backbone, mode = "test", use_fusion=use_fusion, fusion_mode=fusion_mode, student_state_dict=student_state_dict)
        
        model.to(device)
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=False)
        params = sum(p.numel() for p in model.parameters())
        print(f"MACs of the joint {backbone} model: {macs}")
        print(f"Parameters of the joint {backbone} model: {params/1e6:.2f} M")
        print(f"Size of the joint {backbone} model: {(os.path.getsize(save_path) / (1024**2)):.2f} MB")

        evaluator = CBMEvaluator(model, num_attr, attr_cols, test_dataloader, device, concepts=use_concepts)
        evaluator.evaluate()
    
    elif model_type == "standard":
        model = standard_model(freeze_parameters=True, model_state_dict=state_dict, backbone=backbone, mode="test")
        model.to(device)
        evaluator = CBMEvaluator(model, num_attr, attr_cols, test_dataloader, device, concepts=False, main_only=True)
        evaluator.evaluate()
    
    elif model_type == "independent" or model_type == "sequential":
        #first step: attribute prediction
        concept_model = concepts_model(num_attr=num_attr, freeze_parameters=True, 
                                    expand_dim=0, model_state_dict=state_dict_concepts, backbone=backbone, mode = "test", use_fusion=use_fusion, student_state_dict=student_state_dict, fusion_mode=fusion_mode)
        concept_model.to(device)

        macs_concept, params_concept = get_model_complexity_info(concept_model, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=False)
        params_concept = sum(p.numel() for p in concept_model.parameters())
        print(f"MACs of the concept extraction model: {macs_concept}")
        print(f"Parameters of the concept extraction model: {params_concept/1e6:.2f} M")
        print(f"Size of the concept extraction model: {(os.path.getsize(save_path_concepts) / (1024**2)):.2f} MB")

        concept_evaluator = CBMEvaluator(concept_model, num_attr, attr_cols, test_dataloader, device, concepts = True, bottleneck = True)
        concept_evaluator.evaluate()

        #second step: main prediction
        main_task_model = main_model(num_attr=num_attr, expand_dim = 8, model_state_dict=state_dict)
        main_task_model.to(device)

        #generate concept logits
        dataframe_new = generate_concept_logits(concept_model, dataframe, splits=["test"], save_path = save_path_new_df, device = device)

        test_dataset_no_img = load_dataset(dataframe_new, "test", load_image=False)
        test_dataloader_no_img = make_dataloader(test_dataset_no_img, batch_size)

        main_evaluator = CBMEvaluator(main_task_model, num_attr, attr_cols, test_dataloader_no_img, device, main_only = True)
        main_evaluator.evaluate()



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, help="Whether to train or test")
    parser.add_argument("--dataset", type=str, help="Dataset to use (MvTec or Real-IAD)")
    parser.add_argument("--model_type", type = str, nargs="+", help="Which model to train, e.g. 'independent', 'joint', ...")
    parser.add_argument("--categories", type = str, nargs="+", help="Which categories to train/test")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--backbone", type = str, default="resnet18", help = "Which pre-trained network to use for concept extraction")
    parser.add_argument("--batch_size", type=int, default = 16, help="Batch size to use")
    parser.add_argument("--lambda_", type=float, help="How much weight to give to the attributes")
    parser.add_argument("--optimizer", type=str, default = "adam", help="Which optimizer to use")
    parser.add_argument("--lr", type = float, default = 1e-3, help="Learning rate to use")
    parser.add_argument("--epochs", type=int, default=100, help="How many epochs to run the training for")
    parser.add_argument("--use_concepts", action="store_true", help="Whether to use concepts or not")
    parser.add_argument("--multiclass", action="store_true", help="Train the model to perform multiclass classification")
    parser.add_argument("--use_fusion", action="store_true", help="Whether to use fused teacher-student features")
    parser.add_argument("--fusion_mode", type=str, default="concat", help="Which fusion mode to use")
    parser.add_argument("--freeze_parameters", action="store_true", help="Whether to freeze the parameters of the network for concept prediction")
    parser.add_argument("--model_path", type=str, default = None, help="If specified, loads the state dict of a chosen model")
    parser.add_argument("--save_concepts", action="store_true", help="Whether to save the predicted concepts dataframe")
    parser.add_argument("--seed", type=int, default=42, help="Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    for category in args.categories:
        for model_type in args.model_type:
            if args.mode == "train":
                train_model(category, args.dataset, model_type, device, args.backbone, args.lambda_, args.batch_size, args.optimizer, args.lr, args.epochs, args.use_concepts, args.multiclass, args.use_fusion, args.fusion_mode, args.freeze_parameters, args.model_path, args.save_concepts)
            elif args.mode == "test":
                test_model(category, args.dataset, model_type, device, args.backbone, args.batch_size, args.use_concepts, args.use_fusion, args.fusion_mode, args.save_concepts)
            else:
                raise ValueError("Invalid mode specified. Use 'train' or 'test'.")

if __name__ == "__main__":
    main()