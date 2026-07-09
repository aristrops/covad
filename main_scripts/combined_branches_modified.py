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


from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

from datasets.concept_dataset import ConceptDataset
from models.full_models import joint_model
from evaluators.evaluator_cbm import CBMEvaluator
from evaluators.evaluator_stfpm import STFPMEvaluator
from models.model_backbones import BackboneModelFeatures
from utils.metrics import min_max_norm, compute_pixel_pro, compute_pixel_pr, AverageMeter

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def load_dataset(df, split, use_attr=True, load_image=True, load_mask=False): 
    return ConceptDataset(df, split=split, use_attr=use_attr, load_image=load_image, load_mask=load_mask)

def make_dataloader(dataset, batch_size, shuffle=True): 
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def best_f1_threshold(scores: np.ndarray, y_true: np.ndarray): 
    """
    Compute the score threshold that maximizes F1(=2*prec*rec/(prec+rec)) on (scores, y_true). 

    Parameters
    ----------
    scores : ndarray[N]  - continuous anomaly scores in [0, 1]
    y_true : ndarray[N]  - binary ground-truth labels {0, 1}
    
    Returns
    -------
    threshold : float
    f1_max    : float
    """
    prec, rec, thresholds = precision_recall_curve(y_true, scores)
    # the last values does not correspond to a threshold, so we remove them.
    denom = prec[:-1] + rec[:-1] 
    f1 = np.where(denom > 0, 2*prec[:-1]*rec[:-1]/denom, 0)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])
    
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
    auc_main     : float      - image-level AUROC (main task)
    auc_attr     : float      - mean concept AUROC 
    image_scores : ndarray[N] - raw anomaly scores (sigmoid probabilities)
    all_attr_probs : ndarray[N, M] - all concept probabilities
    """
    print(f"Loading CBM state dict from {save_path}")
    state_dict = torch.load(save_path, weights_only=False) if save_path else None

    model = joint_model(
        num_attr = num_attr,
        expand_dim = expand_dim,
        use_relu = True, 
        use_sigmoid = False, 
        freeze_parameters = True, 
        model_state_dict = state_dict, 
        backbone = backbone,
        mode = "test",
    )

    evaluator = CBMEvaluator(model, num_attr, attr_cols, dataloader, device, concepts=True)
    auc_main, auc_attr, image_scores, all_attr_probs = evaluator.evaluate()
    return auc_main, auc_attr, image_scores, all_attr_probs

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
    pixel_auc    : float      - pixel-level AUROC
    image_scores : ndarray[N] - raw anomaly scores
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
def fit_calibrator(scores_val: np.ndarray, y_val: np.ndarray, method:str): 
    """
    Fit a calibration model on VAL, to apply to both VAL and TEST scores. 
    
    method == "rank":       Rank-based calibration, unsupervised.
    method == "isotonic":   Isotononic regression, supervised on val.
    """
    if method == "rank": 
        def transforms(s):
            n=len(s) 
            return rankdata(s, method="average") / n
        return transforms
    elif method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip") 
        ir.fit(scores_val, y_val)
        return lambda s: ir.predict(s)

    raise ValueError(f"Unknown calibration method: {method}")


def evaluate_combined_performance(
        category: str,
        dataset: str, 
        model_type: str,
        device: torch.device, 
        backbone: str = "mobilenet_v2", 
        expand_dim: int = 0, 
        anomaly_ratio: float = 0.0,
        batch_size: int = 8, 
        student_path_override: str = None,
        calibration_strategy: str = "rank",
        fusion_strategy: str = "max"
):
    """
    Parameters
    ----------
    category              : str   - category name (used to resolve paths)
    dataset               : str   - dataset name (used to resolve paths)
    backbone              : str   - ["mobilenet_v2", "resnet18"]. Backbone architecture for both branches.
    expand_dim            : int   - 0 or 1. Whether to expand the feature dimension in the CBM branch. 
    batch_size            : int   - batch size for dataloaders. Default: 8.
    student_path_override : str   - optional override for the student model path (shared across categories)
    calibration_strategy  : str   - ["rank", "isotonic"]. Calibration strategy for the continuous scores. 
    fusion_strategy       : str   - ["cbm_prioritized", "mean", "learned"]. Fusion strategy for combining the two branches.
    """

    # ── Resolve paths ─────────────────────────────────────────────────────────────
    cbm_path = teacher_path = (
        f"/mnt/disk1/arianna_stropeni/cbm_models/{dataset}/{category}_models/"
        f"gen_anomalies/joint/"
        f"mobilenet_v2_1.0ratio_0MLP_automated.pth"
    )
    # cbm_path = teacher_path = (
    #     f"/mnt/disk1/arianna_stropeni/cbm_models/{dataset}/{category}_models/"
    #     f"original_anomalies/joint/"
    #     f"mobilenet_v2_1.0ratio_0MLP_automated.pth"
    # )
    student_path = student_path_override or (
        f"/mnt/disk1/arianna_stropeni/cbm_models/stfpm_models/"
        f"{dataset}/{category}_stfpm_model.pth"
    )
    dataframe_path = (
        f"/mnt/disk1/arianna_stropeni/cbm_data/"
        f"{dataset}/{category}_dataset_automated_gen_concepts.csv"
    )
    # dataframe_path = (
    #     f"/mnt/disk1/arianna_stropeni/cbm_data/"
    #     f"{dataset}/{category}_dataset_automated.csv"
    # )
    print(f"\n{'='*60}")
    print(f"  Category : {category}")
    print(f"  Dataset  : {dataset}")
    print(f"{'='*60}")

    dataframe = pd.read_csv(dataframe_path)

    # ── validation dataloaders & labels ───────────────────────────────────────
    print("\nBuilding validation dataloaders…")
    (stfpm_val_dl, cbm_val_dl, num_attr, attr_cols) = make_dataloaders(dataframe, "val", batch_size)
    y_val = get_labels(dataframe, "val")
    print(f"  Val images : {len(y_val)}  (anomalous: {y_val.sum()})")

    # ── test dataloaders & labels ─────────────────────────────────────────────
    print("\nBuilding test dataloaders…")
    (stfpm_test_dl, cbm_test_dl, num_attr, attr_cols) = make_dataloaders(dataframe, "test", batch_size)
    y_true = get_labels(dataframe, "test")
    print(f"  Test images: {len(y_true)}  (anomalous: {y_true.sum()})")

    # ── STFPM – val ───────────────────────────────────────────────────────────
    print("\nEvaluating STFPM model on VAL set…")
    _, _, scores_stfpm_val = test_stfpm(stfpm_val_dl, teacher_path, student_path, device, backbone)
 
    # ── CBM – val ─────────────────────────────────────────────────────────────
    print("\nEvaluating CBM model on VAL set…")
    _, _, scores_cbm_val, all_attr_probs_val = test_cbm(cbm_val_dl, cbm_path, num_attr, attr_cols, device, backbone, expand_dim)

    # ── STFPM – test ──────────────────────────────────────────────────────────
    print("\nEvaluating STFPM model on TEST set…")
    auc_stfpm, pixel_auc_stfpm, scores_stfpm = test_stfpm(stfpm_test_dl, teacher_path, student_path, device, backbone)
 
    # ── CBM – test ────────────────────────────────────────────────────────────
    print("\nEvaluating CBM model on TEST set…")
    auc_cbm, concept_auc_cbm, scores_cbm, all_attr_probs = test_cbm(cbm_test_dl, cbm_path, num_attr, attr_cols, device, backbone, expand_dim)
 
    # ── Sanity check lengths ──────────────────────────────────────────────────
    assert len(scores_stfpm_val) == len(scores_cbm_val) == len(y_val), (
        f"Val length mismatch: y_val={len(y_val)}, "
        f"stfpm={len(scores_stfpm_val)}, cbm={len(scores_cbm_val)}"
    )
    assert len(scores_stfpm) == len(scores_cbm) == len(y_true), (
        f"Test length mismatch: y_true={len(y_true)}, "
        f"stfpm={len(scores_stfpm)}, cbm={len(scores_cbm)}"
    )

    # ── Calibration ─────────────────────────────────────────────────────────────
    calibrator_cbm = fit_calibrator(scores_cbm_val, y_val, method=calibration_strategy)
    calibrator_stfpm = fit_calibrator(scores_stfpm_val, y_val, method=calibration_strategy)

    p_cbm_val    =   calibrator_cbm(scores_cbm_val)
    p_stfpm_val  =   calibrator_stfpm(scores_stfpm_val)
    p_cbm_test   =   calibrator_cbm(scores_cbm)
    p_stfpm_test =   calibrator_stfpm(scores_stfpm)

    # ── Compute thresholds on VAL, then apply to TEST ─────────────────────────
    # (needed for agreement statistics)
    thr_cbm,   f1_cbm_val   = best_f1_threshold(p_cbm_val,   y_val)
    thr_stfpm, f1_stfpm_val = best_f1_threshold(p_stfpm_val, y_val)
    pred_cbm_test   = (p_cbm_test   >= thr_cbm).astype(int)
    pred_stfpm_test = (p_stfpm_test >= thr_stfpm).astype(int)

    agree = pred_cbm_test == pred_stfpm_test
    stfpm_only_anomaly  = (pred_stfpm_test == 1) & (pred_cbm_test == 0)
    cbm_only_anom       = (pred_cbm_test == 1) & (pred_stfpm_test == 0)

    # ── Agreement statistics ──────────────────────────────────────────────────
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

   
    #  ── FUSION STRATEGY ───────────────────────────────────────────────────────────────
    if fusion_strategy == "cbm_prioritized":
        # Both normal / both anomaly / CBM anomaly-only -> CBM score 
        # CBM normal, STFPM anomaly                     -> force score = 1.0 (STFPM escalates)
        combined_scores = p_cbm_test.copy()
        combined_scores[stfpm_only_anomaly] = 1.0

        combined_scores_val = p_cbm_val.copy()
        combined_scores_val[(p_cbm_val < thr_cbm) & (p_stfpm_val >= thr_stfpm)] = 1.0
        
        print(f"\n── CBM dominance breakdown by ground-truth class ────────────")
        for lbl, name in [(0, "Normal"), (1, "Anomalous")]:
            mask = y_true == lbl
            if mask.sum() == 0:
                continue
            n_cbm_driven = (~stfpm_only_anomaly[mask]).sum()
            pct          = n_cbm_driven / mask.sum() * 100
            print(f"  {name:10s}: CBM drives score in "
                f"{n_cbm_driven}/{mask.sum()} images ({pct:.1f}%)")


    elif fusion_strategy == "mean":
        combined_scores = 0.5 * p_cbm_test + 0.5 * p_stfpm_test
        combined_scores_val = 0.5 * p_cbm_val + 0.5 * p_stfpm_val

    elif fusion_strategy == "learned": 
        # meta-model that depend on the input: besides the two calibrated scores, 
        # it receives concept probs of the CBM as context
        X_val = np.column_stack([p_cbm_val, p_stfpm_val, all_attr_probs_val])
        X_test = np.column_stack([p_cbm_test, p_stfpm_test, all_attr_probs])

        meta = LogisticRegression(max_iter=1000).fit(X_val, y_val)
        combined_scores = meta.predict_proba(X_test)[:, 1]

        combined_scores_val = meta.predict_proba(X_val)[:, 1]

        n_score_feats = 2
        print(f"\n── Learned fusion (logistic regression, con concept gating) ──")
        print(f"  Weight CBM   : {meta.coef_[0][0]:.3f}")
        print(f"  Weight STFPM : {meta.coef_[0][1]:.3f}")
        print(f"  Weight concept (primi 5): {meta.coef_[0][n_score_feats:n_score_feats+5]}")
        print(f"  Bias       : {meta.intercept_[0]:.3f}")

    elif fusion_strategy == "learned_weights": 
        err_cbm = np.abs(p_cbm_val - y_val)
        err_stfpm = np.abs(p_stfpm_val - y_val)
        target = (err_cbm < err_stfpm).astype(int) # 1: CBM is better, 0: STFPM is better

        clf = LogisticRegression(max_iter=1000).fit(all_attr_probs_val, target)
        alpha_test = clf.predict_proba(all_attr_probs)[:, 1] # probability that CBM is better than STFPM based on concepts
        combined_scores = alpha_test * p_cbm_test + (1 - alpha_test) * p_stfpm_test

        alpha_val = clf.predict_proba(all_attr_probs_val)[:, 1]
        combined_scores_val = alpha_val * p_cbm_val + (1 - alpha_val) * p_stfpm_val

        print(f"\n── Learned gated fusion ──────────────────────────────")
        print(f"  Alpha (weight CBM) average on test : {alpha_test.mean():.3f}")
        print(f"  Alpha min/max                  : [{alpha_test.min():.3f}, {alpha_test.max():.3f}]")

    elif fusion_strategy == "learned_simple": 
        X_val = np.column_stack([p_cbm_val, p_stfpm_val])
        X_test = np.column_stack([p_cbm_test, p_stfpm_test])

        meta = LogisticRegression(max_iter=1000).fit(X_val, y_val)
        combined_scores = meta.predict_proba(X_test)[:, 1]

        combined_scores_val = meta.predict_proba(X_val)[:, 1]

        print(f"\n── Learned fusion (logistic regression, no concept gating) ──")
        print(f"  Weight CBM   : {meta.coef_[0][0]:.3f}")
        print(f"  Weight STFPM : {meta.coef_[0][1]:.3f}")
        print(f"  Bias       : {meta.intercept_[0]:.3f}")
    else: 
        raise ValueError(f"Unknown fusion method: {fusion_strategy}")

    # ── Final agreement statistics ─────────────────────────────────────────────
    thr_combined, _ = best_f1_threshold(combined_scores_val, y_val)
    pred_combined_test = (combined_scores >= thr_combined).astype(int)

    same_as_cbm = pred_combined_test == pred_cbm_test
    same_as_stfpm = pred_combined_test == pred_stfpm_test
    print("\n── Final prediction alignment ─────────────────")
    print(f"Aligned with CBM   : {same_as_cbm.mean()*100:.1f}%")
    print(f"Aligned with STFPM : {same_as_stfpm.mean()*100:.1f}%")
            
    
        
          
    # ── Score statistics ──────────────────────────────────────────────────────
    print(f"\n── Score statistics ─────────────────────────────────────────")
    print(f"  STFPM raw         : [{scores_stfpm.min():.4f}, {scores_stfpm.max():.4f}]")
    print(f"  CBM raw           : [{scores_cbm.min():.4f},   {scores_cbm.max():.4f}]")
    print(f"  STFPM calibrated  : [{p_stfpm_test.min():.4f}, {p_stfpm_test.max():.4f}]")
    print(f"  CBM calibrated    : [{p_cbm_test.min():.4f},   {p_cbm_test.max():.4f}]")
    print(f"  Combined          : [{combined_scores.min():.4f},   {combined_scores.max():.4f}]")

    
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
        "calibration_strategy": calibration_strategy,
        "fusion_strategy": fusion_strategy,
        "pct_aligned_with_cbm": same_as_cbm.mean() * 100,
        "pct_aligned_with_stfpm": same_as_stfpm.mean() * 100
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
        return auc_main, mean_auc, image_scores, all_attr_probs
 
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
    parser.add_argument("--calibration_strategy", type=str, default="rank",
                        help="Calibration strategy: 'rank', 'isotonic'")
    parser.add_argument("--fusion_strategy", type=str, default="cbm_prioritized",
                        help="Fusion strategy: 'cbm_prioritized', 'mean', 'learned'")
    args = parser.parse_args()
 
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
 
    categories = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw", "toothbrush", "transistor", "wood", "zipper", "tile"
    ]
    # categories = ["tile"]
 
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
            fusion_strategy=args.fusion_strategy,
            calibration_strategy=args.calibration_strategy,
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
            "pct_aligned_with_cbm", "pct_aligned_with_stfpm"
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
 







