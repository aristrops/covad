import argparse
import torch
import pandas as pd

from datasets.concept_dataset import ConceptDataset
from trainers.trainer_stfpm import STFPMTrainer
from evaluators.evaluator_stfpm import STFPMEvaluator
from models.model_backbones import BackboneModelFeatures

def train_model(category: str,
                dataframe_path: str,
                teacher_path: str,
                device: torch.device,
                backbone: str,
                batch_size: int = 8,
                optimizer: str = "adam",
                lr: float = 1e-3,
                epochs: int = 100,
                student_path: str = None):
    
    def init_optimizer(params):
        if optimizer == "adam":
            opt = torch.optim.Adam(params, lr = lr)
        elif optimizer == "sgd":
            opt = torch.optim.SGD(params, lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode = "min", factor = 0.1, patience = 5)
        return opt, scheduler

    dataframe = pd.read_csv(dataframe_path)

    state_dict = torch.load(teacher_path)
    state_dict = {
        k.replace("first_model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("first_model.")
    }

    normal_images = dataframe[(dataframe["split"] == "train") & (dataframe["label_index"] == 0)]
    train_dataset = ConceptDataset(normal_images, split="train", use_attr=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    val_dataset = ConceptDataset(dataframe, split = "val", use_attr=False, load_mask=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    print(f"Training student model for {category} category...")

    teacher_model = BackboneModelFeatures(pretrained = True, backbone = backbone)
    teacher_model.load_state_dict(state_dict, strict=False)

    student_model = BackboneModelFeatures(pretrained=False, backbone=backbone)

    optimizer, scheduler = init_optimizer(student_model.parameters())

    trainer = STFPMTrainer(teacher_model, student_model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs, save_path = student_path)
    trainer.train()

def test_model(category: str,
               dataframe_path: str,
                teacher_path: str,
                student_path: str,
                device: torch.device,
                backbone: str,
                batch_size: int = 8,
                evaluate_one: bool = False,
                save_path_image: str = None):

    dataframe = pd.read_csv(dataframe_path)

    state_dict = torch.load(teacher_path) 
    state_dict = {
        k.replace("first_model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("first_model.")
    }

    state_dict_student = torch.load(student_path)
    
    teacher_model = BackboneModelFeatures(pretrained=True, backbone=backbone)
    teacher_model.load_state_dict(state_dict, strict=False)

    student_model = BackboneModelFeatures(pretrained=True, backbone=backbone)
    student_model.load_state_dict(state_dict_student, strict = False)

    test_dataset = ConceptDataset(dataframe, split = "test", use_attr=False, load_mask=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    evaluator = STFPMEvaluator(teacher_model, student_model, test_dataloader, device)

    if evaluate_one:
        anomalous_indices = test_dataset.df.index[test_dataset.df["label_index"] == 1].tolist()
        for idx in anomalous_indices:
            print(f"Creating heatmap for image {idx}...")
            image, label, mask = test_dataset[idx]
            evaluator.visualize(image, mask, save_path=save_path_image)
    
    else:
        print(f"Number of test images: {len(test_dataset)}")
        print(f"Evaluating STFPM model for {category} category...")
        evaluator.evaluate()
    


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, help="Whether to train or test")
    parser.add_argument("--dataframe_path", type=str, help="Path to dataframe")
    parser.add_argument("--categories", type = str, nargs="+", help="Which categories to train/test")
    parser.add_argument("--teacher_path", type=str, help="Path to trained model to use as teacher")
    parser.add_argument("--student_path", type=str, help="Path where to save trained student model")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--backbone", type = str, default="resnet18", help = "Which pre-trained network to use for concept extraction")
    parser.add_argument("--batch_size", type=int, default = 8, help="Batch size to use")
    parser.add_argument("--optimizer", type=str, default = "adam", help="Which optimizer to use")
    parser.add_argument("--lr", type = float, default = 1e-3, help="Learning rate to use")
    parser.add_argument("--epochs", type=int, default=100, help="How many epochs to run the training for")
    parser.add_argument("--save_path_image", type=str, help="Path where to save the generated heatmap")
    parser.add_argument("--seed", type=int, default=42, help="Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    for category in args.categories:
        if args.mode == "train":
            train_model(category, args.dataframe_path, args.teacher_path, device, args.backbone, args.batch_size, args.optimizer, args.lr, args.epochs, args.student_path)
        elif args.mode == "test":
            test_model(category, args.dataframe_path, args.teacher_path, args.student_path, device, args.backbone, args.batch_size)
        elif args.mode == "visualize":
            test_model(category, args.dataframe_path, args.teacher_path, args.student_path, device, args.backbone, args.batch_size, evaluate_one=True, save_path_image = args.save_path_image)

if __name__ == "__main__":
    main()