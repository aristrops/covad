import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, f1_score

from utils.metrics import (
    min_max_norm,
    compute_pixel_f1,
    compute_image_f1,
    compute_pixel_pro,
)


class UnifiedEvaluator:
    """Evaluates a UnifiedModel: heatmap metrics (I-AUC, P-AUC) from the
    STFPM branch, and concept metrics (AUC, weighted F1) from the concept net.
    """

    def __init__(self, model, num_attr, attr_cols, test_dataloader, device):
        self.model = model.to(device)
        self.num_attr = num_attr
        self.attr_cols = attr_cols
        self.test_dataloader = test_dataloader
        self.device = device

    def evaluate(self):
        self.model.eval()

        n = len(self.test_dataloader.dataset)
        loss_map = np.zeros((n, 64, 64))
        gt_masks = []
        all_concept_logits = []
        all_concept_gt = []
        all_main_probs = []
        all_labels = []

        i = 0
        for images, concepts, labels, mask in self.test_dataloader:
            images = images.to(self.device)
            with torch.no_grad():
                t_features, s_features, concept_logits, main_logit = self.model(images)

            all_main_probs.append(torch.sigmoid(main_logit.squeeze(1)).cpu().numpy())
            all_labels.append(labels.numpy())

            # ---- heatmap: product of per-layer normalized squared diffs ----
            score_map = 1.0
            for j in range(len(t_features)):
                tn = F.normalize(t_features[j], dim=1)
                sn = F.normalize(s_features[j], dim=1)
                sm = torch.sum((tn - sn) ** 2, dim=1, keepdim=True)
                sm = F.interpolate(sm, size=(64, 64), mode="bilinear", align_corners=False)
                score_map = score_map * sm

            loss_map[i: i + images.size(0)] = score_map.squeeze(1).cpu().numpy()

            resized_masks = F.interpolate(mask.float(), size=(64, 64), mode="nearest")
            gt_masks.append(resized_masks.squeeze(1).cpu().numpy())

            probs = torch.sigmoid(torch.cat(concept_logits, dim=1))
            all_concept_logits.append(probs.cpu().numpy())
            all_concept_gt.append(concepts.numpy())

            i += images.size(0)

        gt_masks = np.concatenate(gt_masks, axis=0)

        # ---------------- Pixel-level metrics ----------------
        # Binarize masks as > 0 (any positive). MVTec encodes anomaly as 255 (->1.0),
        # but VisA encodes it as a small label value (e.g. 5 -> ~0.02), so a 0.5
        # threshold would erase every VisA positive pixel and make P-AUC nan.
        y_pred = loss_map
        pred_masks = min_max_norm(y_pred)
        y_true = (gt_masks > 0).astype(int)

        if len(np.unique(y_true)) < 2:
            pixel_auc = float("nan")  # no positive pixels at all (e.g. no GT masks)
        else:
            pixel_auc = roc_auc_score(y_true.flatten(), pred_masks.flatten())
        pixel_f1 = compute_pixel_f1(y_true, y_pred)
        pixel_pro = compute_pixel_pro(pred_masks, y_true)

        # ---------------- Image-level metrics ----------------
        # (a) STFPM heatmap image score = max over the anomaly map
        image_labels = (np.sum(gt_masks, axis=(1, 2)) > 0).astype(int)
        image_scores = np.max(pred_masks.reshape(pred_masks.shape[0], -1), axis=1)
        stfpm_image_auc = roc_auc_score(image_labels, image_scores)
        stfpm_image_f1_max, _ = compute_image_f1(image_labels, image_scores)

        # (b) CBM main-head image score (final anomaly prediction from concepts)
        main_probs = np.concatenate(all_main_probs, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0).astype(int)
        cbm_image_auc = roc_auc_score(labels_arr, main_probs)
        cbm_image_f1_max, cbm_best_thresh = compute_image_f1(labels_arr, main_probs)

        # ---------------- Concept metrics ----------------
        concept_probs = np.concatenate(all_concept_logits, axis=0)  # [N, num_attr]
        concept_gt = np.concatenate(all_concept_gt, axis=0).astype(int)
        concept_preds = (concept_probs >= 0.5).astype(int)

        # per-concept AUC averaged over concepts that have both classes present
        aucs = []
        for c in range(concept_gt.shape[1]):
            if len(np.unique(concept_gt[:, c])) > 1:
                aucs.append(roc_auc_score(concept_gt[:, c], concept_probs[:, c]))
        concept_auc = float(np.mean(aucs)) if aucs else float("nan")
        concept_f1 = f1_score(concept_gt, concept_preds, average="weighted", zero_division=0)

        print(f"[CBM]      I-AUC = {cbm_image_auc:.4f}, Image F1-max = {cbm_image_f1_max:.4f} "
              f"(thr={cbm_best_thresh:.4f})   <- final anomaly score from concepts")
        print(f"[STFPM]    I-AUC = {stfpm_image_auc:.4f}, P-AUC = {pixel_auc:.4f}, "
              f"Pixel F1 = {pixel_f1:.4f}, Pixel PRO = {pixel_pro:.4f}")
        print(f"[Concepts] AUC = {concept_auc:.4f}, weighted F1 = {concept_f1:.4f}")

        return {
            "cbm_image_auc": cbm_image_auc,
            "cbm_image_f1": cbm_image_f1_max,
            "stfpm_image_auc": stfpm_image_auc,
            "pixel_auc": pixel_auc,
            "pixel_f1": pixel_f1,
            "pixel_pro": pixel_pro,
            "concept_auc": concept_auc,
            "concept_f1": concept_f1,
        }
