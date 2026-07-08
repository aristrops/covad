import argparse
import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import time
from PIL import Image
from torchvision import transforms
from scipy.stats import rankdata

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve

from utils.model_utils import generate_concept_logits
from datasets.concept_dataset import ConceptDataset
from models.full_models import joint_model, standard_model, concepts_model, main_model
from evaluators.evaluator_cbm import CBMEvaluator
from evaluators.evaluator_stfpm import STFPMEvaluator
from models.model_backbones import BackboneModelFeatures
from utils.metrics import compute_pixel_f1, min_max_norm, compute_pixel_pro, compute_pixel_pr, compute_image_f1, AverageMeter, binary_accuracy

def load_dataset(df,
                 split,
                 use_attr=True,
                 load_image=True,
                 load_mask=False):
    return ConceptDataset(df, split=split, use_attr=use_attr, load_image=load_image, load_mask=load_mask)
 
 
def make_dataloader(dataset, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
 

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def best_f1_threshold(scores: np.ndarray, y_true: np.ndarray):
    """
    Returns the score threshold that maximises the F1 on (scores, y_true),
    together with that maximum F1 value.
 
    Parameters
    ----------
    scores : ndarray[N]  – continuous anomaly scores in [0, 1]
    y_true : ndarray[N]  – binary ground-truth labels {0, 1}
 
    Returns
    -------
    threshold : float
    f1_max    : float
    """
    prec, rec, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve appends a sentinel point with no paired threshold
    denom = prec[:-1] + rec[:-1]
    f1 = np.where(denom > 0, 2 * prec[:-1] * rec[:-1] / denom, 0.0)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])

def compute_model_size_mb(*model_paths: str) -> dict:
    """
    Computes the size in MB of each model checkpoint and the total,
    counting shared paths only once.

    Parameters
    ----------
    *model_paths : str  – paths to .pth files; duplicates are counted once.

    Returns
    -------
    dict with keys: per-path sizes and 'total_mb'
    """
    seen   = {}   # path -> size in MB
    result = {}

    for path in model_paths:
        if path in seen:
            result[path] = seen[path]
            continue
        try:
            size_mb = os.path.getsize(path) / (1024 ** 2)
        except FileNotFoundError:
            print(f"  [WARNING] File not found for size check: {path}")
            size_mb = 0.0
        seen[path]   = size_mb
        result[path] = size_mb

    result["total_mb"] = sum(seen.values())  # each unique path counted once
    return result
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CBM branch
# ─────────────────────────────────────────────────────────────────────────────
 
def test_cbm(dataloader,
             save_path: str,
             num_attr: int,
             attr_cols,
             device: torch.device,
             backbone: str,
             expand_dim: int):
    """
    Returns
    -------
    auc_main     : float   – image-level AUROC (main task)
    auc_attr     : float   – mean concept AUROC
    image_scores : ndarray[N] – raw anomaly scores (sigmoid probabilities)
    """
    print(f"Loading CBM state dict from {save_path}")
    state_dict = torch.load(save_path, weights_only=False) if save_path else None
 
    model = joint_model(
        num_attr=num_attr,
        expand_dim=expand_dim,
        use_relu=True,
        use_sigmoid=False,
        freeze_parameters=True,
        model_state_dict=state_dict,
        backbone=backbone,
        mode="test",
    )
 
    evaluator = CBMEvaluator(model, num_attr, attr_cols, dataloader, device, concepts=True)
    auc_main, auc_attr, image_scores = evaluator.evaluate()
 
    return auc_main, auc_attr, image_scores
 
 
# ─────────────────────────────────────────────────────────────────────────────
# STFPM branch
# ─────────────────────────────────────────────────────────────────────────────
 
def test_stfpm(dataloader,
               teacher_path: str,
               student_path: str,
               device: torch.device,
               backbone: str):
    """
    Returns
    -------
    image_auc    : float
    pixel_auc    : float  – pixel-level AUROC
    image_scores : ndarray[N] – raw image-level anomaly scores
    """
    state_dict = torch.load(teacher_path, weights_only=False)
    state_dict = {
        k.replace("first_model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("first_model.")
    }
    state_dict_student = torch.load(student_path)
 
    teacher_model = BackboneModelFeatures(pretrained=True, backbone=backbone)
    teacher_model.load_state_dict(state_dict, strict=False)
 
    student_model = BackboneModelFeatures(pretrained=True, backbone=backbone)
    student_model.load_state_dict(state_dict_student, strict=False)
 
    evaluator = STFPMEvaluator(teacher_model, student_model, dataloader, device)
    image_auc, pixel_auc, image_scores = evaluator.evaluate()
 
    return image_auc, pixel_auc, image_scores
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Dataloader factory  (avoids duplicating the dataset-setup boilerplate)
# ─────────────────────────────────────────────────────────────────────────────
 
def make_dataloaders(dataframe: pd.DataFrame, split: str, batch_size: int):
    """
    Build and return (stfpm_dataloader, cbm_dataloader, num_attr, attr_cols).
 
    Parameters
    ----------
    dataframe  : the full CSV loaded as a DataFrame
    split      : "val" or "test"
    batch_size : int
    """
    stfpm_dataset = ConceptDataset(
        dataframe,
        split=split,
        use_attr=False,
        load_mask=True,
    )
    stfpm_dataloader = torch.utils.data.DataLoader(
        stfpm_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
 
    cbm_dataset = load_dataset(
        dataframe,
        split,
        use_attr=True,
        load_mask=False,
    )
    cbm_dataloader = torch.utils.data.DataLoader(
        cbm_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
 
    num_attr  = len(cbm_dataset.attr_cols)
    attr_cols = cbm_dataset.attr_cols
 
    return stfpm_dataloader, cbm_dataloader, num_attr, attr_cols
 
 
def get_labels(dataframe: pd.DataFrame, split: str) -> np.ndarray:
    return (
        dataframe[dataframe["split"] == split]["label_index"]
        .to_numpy()
        .astype(int)
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Combined evaluation for one category
# ─────────────────────────────────────────────────────────────────────────────
 
def evaluate_combined_performance(category: str,
                                  dataset: str,
                                  model_type: str,
                                  device: torch.device,
                                  backbone: str = "mobilenet_v2",
                                  anomaly_ratio: float = 1.0,
                                  expand_dim: int = 0,
                                  batch_size: int = 8,
                                  student_path_override: str = None):
    # ── resolve paths ─────────────────────────────────────────────────────────
    # cbm_path = teacher_path = (
    #     f"/mnt/disk1/arianna_stropeni/cbm_models/{dataset}/{category}_models/"
    #     f"gen_anomalies/joint/"
    #     f"mobilenet_v2_1.0ratio_0MLP_automated.pth"
    # )
    cbm_path = teacher_path = f"/mnt/disk1/borsattifr/covad_disk_old/cbm_models/{dataset}_weakly/{category}_models/cont1/seed_0/gen_anomalies_weakly_sup_1/joint/mobilenet_v2_1.0ratio_0MLP_automated.pth"

    student_path = student_path_override or (
        f"/mnt/disk1/arianna_stropeni/cbm_models/stfpm_models/"
        f"{dataset}/{category}_stfpm_model.pth"
    )
    dataframe_path = (
        f"/mnt/disk1/arianna_stropeni/cbm_data/"
        f"{dataset}/{category}_dataset_automated_gen_concepts.csv"
    )

    size_info = compute_model_size_mb(cbm_path, student_path)

    print(f"\n── Model sizes ───────────────────────────────────────────────")
    print(f"  CBM / Teacher : {size_info[cbm_path]:.2f} MB  (shared, counted once)")
    print(f"  Student       : {size_info[student_path]:.2f} MB")
    print(f"  Total         : {size_info['total_mb']:.2f} MB")
 
    print(f"\n{'='*60}")
    print(f"  Category : {category}")
    print(f"  Dataset  : {dataset}")
    print(f"{'='*60}")
 
    dataframe = pd.read_csv(dataframe_path)
 
    # ── validation dataloaders & labels ───────────────────────────────────────
    print("\nBuilding validation dataloaders…")
    (stfpm_val_dl,
     cbm_val_dl,
     num_attr,
     attr_cols) = make_dataloaders(dataframe, "val", batch_size)
 
    y_val = get_labels(dataframe, "val")
    print(f"  Val images : {len(y_val)}  (anomalous: {y_val.sum()})")
 
    # ── test dataloaders & labels ─────────────────────────────────────────────
    print("\nBuilding test dataloaders…")
    (stfpm_test_dl,
     cbm_test_dl,
     num_attr,
     attr_cols) = make_dataloaders(dataframe, "test", batch_size)
 
    y_true = get_labels(dataframe, "test")
    print(f"  Test images: {len(y_true)}  (anomalous: {y_true.sum()})")
 
    # ── STFPM – val ───────────────────────────────────────────────────────────
    print("\nEvaluating STFPM model on VAL set…")
    _, _, scores_stfpm_val = test_stfpm(
        stfpm_val_dl, teacher_path, student_path, device, backbone
    )
 
    # ── CBM – val ─────────────────────────────────────────────────────────────
    print("\nEvaluating CBM model on VAL set…")
    _, _, scores_cbm_val = test_cbm(
        cbm_val_dl, cbm_path, num_attr, attr_cols, device, backbone, expand_dim
    )
 
    # ── STFPM – test ──────────────────────────────────────────────────────────
    print("\nEvaluating STFPM model on TEST set…")
    auc_stfpm, pixel_auc_stfpm, scores_stfpm = test_stfpm(
        stfpm_test_dl, teacher_path, student_path, device, backbone
    )
 
    # ── CBM – test ────────────────────────────────────────────────────────────
    print("\nEvaluating CBM model on TEST set…")
    auc_cbm, concept_auc_cbm, scores_cbm = test_cbm(
        cbm_test_dl, cbm_path, num_attr, attr_cols, device, backbone, expand_dim
    )
 
    # ── Sanity check lengths ──────────────────────────────────────────────────
    assert len(scores_stfpm_val) == len(scores_cbm_val) == len(y_val), (
        f"Val length mismatch: y_val={len(y_val)}, "
        f"stfpm={len(scores_stfpm_val)}, cbm={len(scores_cbm_val)}"
    )
    assert len(scores_stfpm) == len(scores_cbm) == len(y_true), (
        f"Test length mismatch: y_true={len(y_true)}, "
        f"stfpm={len(scores_stfpm)}, cbm={len(scores_cbm)}"
    )
 
    # ── Rank-normalise (val) ──────────────────────────────────────────────────
    n_val = len(scores_stfpm_val)
    scores_stfpm_val_rank = rankdata(scores_stfpm_val, method="average") / n_val
    scores_cbm_val_rank   = rankdata(scores_cbm_val,   method="average") / n_val
 
    # ── Rank-normalise (test) ─────────────────────────────────────────────────
    n_test = len(scores_stfpm)
    scores_stfpm_rank = rankdata(scores_stfpm, method="average") / n_test
    scores_cbm_rank   = rankdata(scores_cbm,   method="average") / n_test
 
    # ── Compute thresholds on VAL, then apply to TEST ─────────────────────────
    thr_cbm,   f1_cbm_val   = best_f1_threshold(scores_cbm_val_rank,   y_val)
    thr_stfpm, f1_stfpm_val = best_f1_threshold(scores_stfpm_val_rank, y_val)
 
    # print(f"\n── Max-F1 thresholds (calibrated on VAL) ────────────────────")
    # print(f"  CBM   threshold : {thr_cbm:.4f}  (val F1 = {f1_cbm_val:.4f})")
    # print(f"  STFPM threshold : {thr_stfpm:.4f}  (val F1 = {f1_stfpm_val:.4f})")
 
    # Binary predictions on TEST using VAL-calibrated thresholds
    pred_cbm_test   = (scores_cbm_rank   >= thr_cbm).astype(int)
    pred_stfpm_test = (scores_stfpm_rank >= thr_stfpm).astype(int)
 
    # ── CBM-prioritised fusion ────────────────────────────────────────────────
    #
    #  Case                        | Action
    #  ----------------------------|------------------------------------------
    #  Both normal                 | CBM continuous score  (agreement: normal)
    #  Both anomaly                | CBM continuous score  (agreement: anomaly)
    #  CBM anomaly, STFPM normal   | CBM continuous score  (trust CBM)
    #  CBM normal,  STFPM anomaly  | force score = 1.0     (STFPM escalates)
    #
    stfpm_only_anomaly = (pred_stfpm_test == 1) & (pred_cbm_test == 0)
 
    combined_scores = scores_cbm_rank.copy()
    combined_scores[stfpm_only_anomaly] = 1.0
 
    # stfpm_anomaly = pred_stfpm_test == 1
    # combined_scores = scores_cbm_rank.copy()
    # combined_scores[stfpm_anomaly] = 1.0
    # ── Agreement statistics ──────────────────────────────────────────────────
    agree          = pred_cbm_test == pred_stfpm_test
    cbm_only_anom  = (pred_cbm_test == 1) & (pred_stfpm_test == 0)
 
    print(f"\n── Branch agreement on TEST (val-calibrated thresholds) ─────")
    print(f"  Agree (both normal)   : "
          f"{((pred_cbm_test==0)&(pred_stfpm_test==0)).sum():>4d}  "
          f"({((pred_cbm_test==0)&(pred_stfpm_test==0)).mean()*100:.1f}%)")
    print(f"  Agree (both anomaly)  : "
          f"{((pred_cbm_test==1)&(pred_stfpm_test==1)).sum():>4d}  "
          f"({((pred_cbm_test==1)&(pred_stfpm_test==1)).mean()*100:.1f}%)")
    print(f"  CBM-only anomaly      : "
          f"{cbm_only_anom.sum():>4d}  ({cbm_only_anom.mean()*100:.1f}%)")
    print(f"  STFPM-only anomaly    : "
          f"{stfpm_only_anomaly.sum():>4d}  ({stfpm_only_anomaly.mean()*100:.1f}%)")
 
    print(f"\n── CBM dominance breakdown by ground-truth class ────────────")
    for lbl, name in [(0, "Normal"), (1, "Anomalous")]:
        mask = y_true == lbl
        if mask.sum() == 0:
            continue
        n_cbm_driven = (~stfpm_only_anomaly[mask]).sum()
        pct          = n_cbm_driven / mask.sum() * 100
        print(f"  {name:10s}: CBM drives score in "
              f"{n_cbm_driven}/{mask.sum()} images ({pct:.1f}%)")
 
    # ── Score statistics ──────────────────────────────────────────────────────
    print(f"\n── Score statistics ─────────────────────────────────────────")
    print(f"  STFPM raw   : [{scores_stfpm.min():.4f}, {scores_stfpm.max():.4f}]")
    print(f"  CBM raw     : [{scores_cbm.min():.4f},   {scores_cbm.max():.4f}]")
    print(f"  STFPM rank  : [{scores_stfpm_rank.min():.4f}, {scores_stfpm_rank.max():.4f}]")
    print(f"  CBM rank    : [{scores_cbm_rank.min():.4f},   {scores_cbm_rank.max():.4f}]")
    print(f"  Combined    : [{combined_scores.min():.4f},   {combined_scores.max():.4f}]")
 
    # ── AUC metrics ───────────────────────────────────────────────────────────
    auc_combined = roc_auc_score(y_true, combined_scores)
 
    print(f"\n── STFPM branch ──────────────────────────────────────────────")
    print(f"  Image AUC  : {auc_stfpm:.4f}")
    print(f"  Pixel AUC  : {pixel_auc_stfpm:.4f}")
 
    print(f"\n── CBM branch ────────────────────────────────────────────────")
    print(f"  Image AUC  : {auc_cbm:.4f}")
    print(f"  Concept AUC: {concept_auc_cbm:.4f}")
 
    print(f"\n── Combined (CBM-prioritised fusion) ─────────────────────────")
    print(f"  Image AUC  : {auc_combined:.4f}")
 
    return {
        "category":           category,
        "stfpm_auc":          auc_stfpm,
        "stfpm_pixel_auc":    pixel_auc_stfpm,
        "cbm_auc":            auc_cbm,
        "cbm_concept_auc":    concept_auc_cbm,
        "combined_auc":       auc_combined,
        "pct_agree":          agree.mean() * 100,
        "pct_cbm_only_anom":  cbm_only_anom.mean() * 100,
        "pct_stfpm_only_anom": stfpm_only_anomaly.mean() * 100,
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Evaluators
# ─────────────────────────────────────────────────────────────────────────────
 
class STFPMEvaluator:
    def __init__(self, teacher_model, student_model, test_dataloader, device):
        self.teacher_model   = teacher_model.to(device)
        self.student_model   = student_model.to(device)
        self.test_dataloader = test_dataloader
        self.device          = device
 
    def unnormalize_image(self, img_tensor,
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]):
        for t, m, s in zip(img_tensor, mean, std):
            t.mul_(s).add_(m)
        return img_tensor
 
    def visualize(self, image, mask, save_path=None):
        self.teacher_model.eval()
        self.student_model.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0).to(self.device)
        else:
            image = image.to(self.device)
        with torch.no_grad():
            t_features = self.teacher_model(image)
            s_features = self.student_model(image)
        score_map = 1.
        for j in range(len(t_features)):
            t_features[j] = F.normalize(t_features[j], dim=1)
            s_features[j] = F.normalize(s_features[j], dim=1)
            sm = torch.sum((t_features[j] - s_features[j]) ** 2, 1, keepdim=True)
            sm = F.interpolate(sm, size=(64, 64), mode="bilinear", align_corners=False)
            score_map = score_map * sm
        score_map = score_map.squeeze(0).squeeze(0).cpu().numpy()
        norm_score_map = (score_map - score_map.min()) / (
            score_map.max() - score_map.min() + 1e-8
        )
        original_size = image.shape[-2:]
        score_map_resized = F.interpolate(
            torch.tensor(norm_score_map).unsqueeze(0).unsqueeze(0),
            size=original_size, mode="bilinear", align_corners=False
        ).squeeze().numpy()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        import matplotlib.pyplot as plt
        plt.imsave(save_path, score_map_resized, cmap="jet")
        plt.close()
 
    def evaluate(self):
        """
        Returns
        -------
        image_auc    : float
        pixel_auc    : float
        image_scores : ndarray[N]
        """
        self.teacher_model.eval()
        self.student_model.eval()
 
        n = len(self.test_dataloader.dataset)
        loss_map = np.zeros((n, 64, 64))
        gt_masks  = []
        i = 0
 
        for image, label, mask in self.test_dataloader:
            image = image.to(self.device)
            with torch.no_grad():
                t_features = self.teacher_model(image)
                s_features = self.student_model(image)
 
            score_map = 1.
            for j in range(len(t_features)):
                t_features[j] = F.normalize(t_features[j], dim=1)
                s_features[j] = F.normalize(s_features[j], dim=1)
                sm = torch.sum((t_features[j] - s_features[j]) ** 2, 1, keepdim=True)
                sm = F.interpolate(sm, size=(64, 64), mode="bilinear", align_corners=False)
                score_map = score_map * sm
 
            loss_map[i: i + image.size(0)] = score_map.squeeze().cpu().data.numpy()
            resized_masks = F.interpolate(mask.float(), size=(64, 64), mode="nearest")
            gt_masks.append(resized_masks.squeeze(1).cpu().numpy())
            i += image.size(0)
 
        gt_masks   = np.concatenate(gt_masks, axis=0)
        pred_masks = min_max_norm(loss_map)
 
        if isinstance(gt_masks, np.ndarray):
            y_true_pixel = (gt_masks > 0.5).astype(int)
        else:
            y_true_pixel = (gt_masks > 0.5).int()
 
        pixel_auc = roc_auc_score(y_true_pixel.flatten(), pred_masks.flatten())
        pixel_pr  = compute_pixel_pr(pred_masks, y_true_pixel)
        pixel_pro = compute_pixel_pro(pred_masks, y_true_pixel)
 
        image_labels = (np.sum(gt_masks, axis=(1, 2)) > 0).astype(int)
        image_scores = np.percentile(
            pred_masks.reshape(pred_masks.shape[0], -1),
            99,
            axis=1,
        )
        image_auc = roc_auc_score(image_labels, image_scores)
 
        print(f"  Pixel PRO = {pixel_pro:.4f}, Pixel PR = {pixel_pr:.4f}")
        print(f"  Image AUC = {image_auc:.4f}")
 
        return image_auc, pixel_auc, image_scores
 
 
class CBMEvaluator:
    def __init__(self, model, num_attr, attr_cols, test_dataloader,
                 device, bottleneck=False, concepts=False, main_only=False):
        self.model           = model
        self.num_attr        = num_attr
        self.attr_cols       = attr_cols
        self.test_dataloader = test_dataloader
        self.device          = device
        self.bottleneck      = bottleneck
        self.concepts        = concepts
        self.main_only       = main_only
 
    def evaluate(self):
        """
        Returns (when not bottleneck and not main_only)
        -------
        auc_main     : float
        mean_auc     : float   – mean concept AUROC
        image_scores : ndarray[N]
        """
        self.model.eval()
 
        accuracy_meter_main = AverageMeter()
        accuracy_meter_attr = AverageMeter()
 
        all_main_targets, all_main_probs = [], []
        all_attr_probs, all_attr_targets = [], []
 
        total_inference_time = 0.0
        total_instances      = 0
 
        for sample in self.test_dataloader:
            if self.main_only:
                inputs, labels = sample
                inputs = inputs.to(self.device)
                if isinstance(inputs, list):
                    inputs = torch.stack(inputs).t().float()
            else:
                inputs, concepts, labels = sample
                inputs, concepts = inputs.to(self.device), concepts.to(self.device)
 
            with torch.no_grad():
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start       = time.time()
                predictions = self.model(inputs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.time()
 
            total_inference_time += end - start
            total_instances      += inputs.size(0)
 
            if not self.bottleneck:
                logits_main = (
                    predictions[0].squeeze(1)
                    if isinstance(predictions, list)
                    else predictions.squeeze(1)
                )
                probs_main = torch.sigmoid(logits_main)
                all_main_targets.append(labels.cpu().int())
                all_main_probs.append(probs_main.cpu())
 
            if not self.main_only:
                attr_logits = (
                    torch.cat(predictions[1:], dim=1)
                    if not self.bottleneck
                    else torch.cat(predictions, dim=1)
                )
                probs_attr = torch.sigmoid(attr_logits)
                all_attr_probs.append(probs_attr.cpu())
                all_attr_targets.append(concepts.cpu().int())
 
        # ── main task ─────────────────────────────────────────────────────────
        if not self.bottleneck and all_main_probs:
            all_main_targets = torch.cat(all_main_targets).numpy()
            all_main_probs   = torch.cat(all_main_probs).numpy()
            auc_main     = roc_auc_score(all_main_targets, all_main_probs)
            image_scores = all_main_probs
        else:
            auc_main     = 0
            image_scores = np.array([])
 
        # ── concept task ──────────────────────────────────────────────────────
        if all_attr_probs:
            all_attr_probs   = torch.cat(all_attr_probs).numpy()
            all_attr_targets = torch.cat(all_attr_targets).numpy()
 
            aucs = []
            for i in range(all_attr_targets.shape[1]):
                try:
                    aucs.append(roc_auc_score(all_attr_targets[:, i],
                                              all_attr_probs[:, i]))
                except ValueError:
                    aucs.append(np.nan)
            mean_auc = float(np.nanmean(aucs))
        else:
            mean_auc = 0.0
 
        # ── print ─────────────────────────────────────────────────────────────
        if self.bottleneck:
            print(f"  Concept AUC : {mean_auc:.4f}")
        else:
            print(f"  Main AUC    : {auc_main:.4f}")
            if self.concepts:
                print(f"  Concept AUC : {mean_auc:.4f}")
 
        if self.bottleneck:
            return mean_auc
        if self.main_only:
            return (auc_main,)
        return auc_main, mean_auc, image_scores
 
    def inference(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        img          = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(self.device)
 
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(input_tensor)
 
        if not self.bottleneck:
            if isinstance(predictions, list):
                main_logits = predictions[0].squeeze(1)
                attr_logits = (
                    torch.cat(predictions[1:], dim=1)
                    if len(predictions) > 1
                    else None
                )
            else:
                main_logits = predictions.squeeze(1)
                attr_logits = None
        else:
            main_logits = None
            attr_logits = torch.cat(predictions, dim=1)
 
        main_pred = main_prob = None
        if main_logits is not None:
            main_prob = torch.sigmoid(main_logits)
            main_pred = (main_prob >= 0.5).int()
 
        attr_probs = attr_preds = []
        if attr_logits is not None:
            attr_probs = torch.sigmoid(attr_logits).squeeze(0).cpu().numpy()
            attr_preds = (attr_probs >= 0.5).astype(int)
 
        if main_pred is not None:
            print(f"Main Task Prediction: {'Anomalous' if main_pred else 'Normal'}")
        if len(self.attr_cols) > 0 and len(attr_preds) > 0:
            print("\nConcept Predictions:")
            for name, pred in zip(self.attr_cols, attr_preds):
                print(f"  - {name}: {'Present' if pred else 'Absent'}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str, required=True,
                        help="Dataset name (used to resolve all paths)")
    parser.add_argument("--model_type", type=str, nargs="+",
                        help="Model type label(s), e.g. 'joint'")
    parser.add_argument("--device",     type=str, default="cpu",
                        help="Torch device string, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--backbone",   type=str, default="mobilenet_v2")
    parser.add_argument("--anomaly_ratio", type=float, default=1.0)
    parser.add_argument("--expand_dim", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--student_path_override", type=str, default=None,
                        help="Override the student model path (shared across categories)")
    args = parser.parse_args()
 
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
 
    # categories = [
    #     "bottle", "cable", "capsule", "carpet", "grid",
    #     "hazelnut", "leather", "metal_nut", "pill", "screw", "toothbrush", "transistor", "wood", "zipper", "tile"
    # ]
    categories = ["tile"]
 
    all_results = []
    for category in categories:
        result = evaluate_combined_performance(
            category=category,
            dataset=args.dataset,
            model_type=args.model_type[0] if args.model_type else "joint",
            device=device,
            backbone=args.backbone,
            anomaly_ratio=args.anomaly_ratio,
            expand_dim=args.expand_dim,
            batch_size=args.batch_size,
            student_path_override=args.student_path_override,
        )
        all_results.append(result)
 
    # ── aggregate summary ─────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*70}")
        print("  AGGREGATE RESULTS")
        print(f"{'='*70}")
 
        metric_keys = [
            "stfpm_auc", "stfpm_pixel_auc",
            "cbm_auc", "cbm_concept_auc",
            "combined_auc",
        ]
        stat_keys = [
            "pct_agree", "pct_cbm_only_anom", "pct_stfpm_only_anom",
        ]
        all_keys = metric_keys + stat_keys
 
        header = f"{'Category':<20}" + "".join(f"{k:>22}" for k in all_keys)
        print(header)
        print("-" * len(header))
        for r in all_results:
            row = f"{r['category']:<20}" + "".join(f"{r[k]:>22.4f}" for k in all_keys)
            print(row)
 
        print(f"\n{'Means':}")
        for k in all_keys:
            vals = [r[k] for r in all_results]
            print(f"  {k:<28}: {np.mean(vals):.4f}")
 
 
if __name__ == "__main__":
    main()
 







