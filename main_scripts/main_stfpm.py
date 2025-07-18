import argparse
import torch
import pandas as pd

from datasets.mvtec_concept_dataset import MvTecConceptDataset
from trainers.trainer_stfpm import STFPMTrainer
from models.model_backbones import BackboneModelFeatures

def train_model(category: str,
                dataset: str,
                device: torch.device,
                backbone: str,
                batch_size: int = 8,
                optimizer: str = "adam",
                lr: float = 1e-3,
                epochs: int = 100):
    
    def init_optimizer(params):
        if optimizer == "adam":
            opt = torch.optim.Adam(params, lr = lr)
        elif optimizer == "sgd":
            opt = torch.optim.SGD(params, lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode = "min", factor = 0.1, patience = 5)
        return opt, scheduler

    dataframe_path = f"/mnt/disk1/arianna_stropeni/cbm_data/{dataset}/{category}_dataset_automated.csv"
    save_path = f"/mnt/disk1/arianna_stropeni/cbm_models/stfpm_models/{category}_{backbone}.pth"
    model_path = f"/mnt/disk1/arianna_stropeni/cbm_models/{category}_models/joint_{backbone}_main_automated.pth"

    dataframe = pd.read_csv(dataframe_path)

    state_dict = torch.load(model_path) if model_path else None
    state_dict = {
        k.replace("first_model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("first_model.")
    }

    normal_images = dataframe[(dataframe["split"] == "train") & (dataframe["label_index"] == 0)]
    train_dataset = MvTecConceptDataset(normal_images, split="train", use_attr=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    val_dataset = MvTecConceptDataset(dataframe, split = "val", use_attr=False, load_mask=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    print(f"Training student model for {category} category...")

    teacher_model = BackboneModelFeatures(pretrained = True, backbone = backbone)
    teacher_model.load_state_dict(state_dict, strict=False)

    student_model = BackboneModelFeatures(pretrained=False, backbone=backbone)

    optimizer, scheduler = init_optimizer(student_model.parameters())

    trainer = STFPMTrainer(teacher_model, student_model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs)
    trainer.train()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, help="Whether to train or test")
    parser.add_argument("--dataset", type=str, help="Dataset to use (MvTec or Real-IAD)")
    parser.add_argument("--categories", type = str, nargs="+", help="Which categories to train/test")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--backbone", type = str, default="resnet18", help = "Which pre-trained network to use for concept extraction")
    parser.add_argument("--batch_size", type=int, default = 16, help="Batch size to use")
    parser.add_argument("--optimizer", type=str, default = "adam", help="Which optimizer to use")
    parser.add_argument("--lr", type = float, default = 1e-3, help="Learning rate to use")
    parser.add_argument("--epochs", type=int, default=100, help="How many epochs to run the training for")
    parser.add_argument("--seed", type=int, default=42, help="Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    for category in args.categories:
        if args.mode == "train":
            train_model(category, args.dataset, device, args.backbone, args.batch_size, args.optimizer, args.lr, args.epochs)
        # elif args.mode == "test":
        #     test_model(category, args.dataset, args.model_type, device, args.backbone, args.batch_size, args.use_concepts)
        # else:
        #     raise ValueError("Invalid mode specified. Use 'train' or 'test'.")

if __name__ == "__main__":
    main()