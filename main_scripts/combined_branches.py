import argparse
import os
import torch
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from utils.model_utils import generate_concept_logits
from datasets.concept_dataset import ConceptDataset
from models.full_models import joint_model, standard_model, concepts_model, main_model
from evaluators.evaluator_cbm import CBMEvaluator
from evaluators.evaluator_stfpm import STFPMEvaluator
from models.model_backbones import BackboneModelFeatures

def load_dataset(df, split, use_attr = True, load_image = True):
    return ConceptDataset(df, split=split, use_attr=use_attr, load_image=load_image)

def make_dataloader(dataset, batch_size, shuffle = True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=8, pin_memory=True)


def test_cbm(dataloder,
               save_path: str,
               num_attr: int,
               attr_cols,
               device: torch.device, 
               backbone: str,
               expand_dim: int):

    print(f"Loading state dict from {save_path}")

    state_dict = torch.load(save_path) if save_path else None

    model = joint_model(num_attr=num_attr, expand_dim=expand_dim, use_relu=True, use_sigmoid=False,
                        freeze_parameters=True, model_state_dict=state_dict, backbone=backbone, mode="test")

    evaluator = CBMEvaluator(model, num_attr, attr_cols, dataloder, device, concepts=True)

    auc_main, f1_main, auc_attr, f1_attr, image_preds = evaluator.evaluate()
    
    return auc_main, auc_attr, f1_main, f1_attr, image_preds


def test_stfpm(dataloader,
                teacher_path: str,
                student_path: str,
                device: torch.device,
                backbone: str):

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

    evaluator = STFPMEvaluator(teacher_model, student_model, dataloader, device)
    image_auc, image_f1_max, image_preds = evaluator.evaluate()

    return image_auc, image_f1_max, image_preds


def evaluate_combined_performance(category: str,
                                  model_type: str,
                                  dataframe_path: str,
                                  save_dir_cbm: str,
                                  teacher_path: str,
                                  student_path: str,
                                  device: torch.device,
                                  backbone: str,
                                  anomaly_ratio: float = 1.0,
                                  expand_dim: int = 0,
                                  batch_size: int = 8):
    
    sub_dir = "original_anomalies"
    save_dir = os.path.join(save_dir_cbm, sub_dir, model_type)
    os.makedirs(save_dir, exist_ok=True)
    save_path_cbm = os.path.join(save_dir, f"{backbone}_{anomaly_ratio}ratio_{expand_dim}MLP_automated.pth")

    dataframe = pd.read_csv(dataframe_path)

    test_dataset = ConceptDataset(dataframe, split = "test", use_attr=False, load_mask=True)
    num_attr = len(test_dataset.attr_cols) 
    attr_cols = test_dataset.attr_cols
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False,
                                                  num_workers=8, pin_memory=True)

    print(f"\nNumber of test images: {len(test_dataset)}")

    print(f"Evaluating STFPM model for {category} category...")
    auc_stfpm, f1_stfpm, image_preds_stfpm = test_stfpm(test_dataloader, teacher_path, student_path, device, backbone, batch_size)

    print(f"\nEvaluating joint model for {category} category...")
    auc_cbm, auc_cbm_attr, f1_cbm, f1_cbm_attr, image_preds_cbm = test_cbm(test_dataloader, save_path_cbm, num_attr, attr_cols, device, backbone, expand_dim)

    y_preds = np.where(image_preds_cbm == 0, image_preds_stfpm, image_preds_cbm)
    y_true = dataframe[dataframe["split"] == "test"]["label_index"].to_numpy().astype(int)

    assert len(y_true) == len(y_preds), \
       f"Length mismatch: {len(y_true)} labels vs {len(y_preds)} predictions"
    
    f1_combined = f1_score(y_true, y_preds)
    auc_combined = roc_auc_score(y_true, y_preds)

    print("F1 Score from the two combined branches:", f1_combined)
    print("AUROC from the two combined branches:", auc_combined)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataframe_path", type=str, help="Path to dataframe")
    parser.add_argument("--category", type = str, nargs="+", help="Which categories to test")
    parser.add_argument("--model_type", type = str, nargs="+", help="Which model to train, e.g. 'independent', 'joint', ...")
    parser.add_argument("--save_dir_cbm", type=str, help="Directory to save the models in")
    parser.add_argument("--teacher_path", type=str, help="Path to trained model to use as teacher")
    parser.add_argument("--student_path", type=str, help="Path where to save trained student model")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--backbone", type = str, default="resnet18", help = "Which pre-trained network to use for concept extraction")
    parser.add_argument("--anomaly_ratio", type=int, default = 1.0, help="Anomaly ratios to keep in training")
    parser.add_argument("--expand_dim", type=int, default = 0, help="How many neurons to use in FC layers")
    parser.add_argument("--batch_size", type=int, default = 8, help="Batch size to use")


    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    evaluate_combined_performance(args.category, args.model_type, args.dataframe_path, args.save_dir_cbm, args.teacher_path, args.student_path, device, args.backbone, args.anomaly_ratio, args.expand_dim, args.batch_size)

if __name__ == "__main__":
    main()









