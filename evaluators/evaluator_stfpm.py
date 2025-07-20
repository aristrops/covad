import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import os

import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.measure import label, regionprops

class STFPMEvaluator:
    def __init__(self,
                teacher_model,
                student_model,
                test_dataloader,
                device: torch.device):

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.test_dataloader = test_dataloader
        self.device = device

    def unnormalize_image(self, img_tensor, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        for t, m, s in zip(img_tensor, mean, std):
            t.mul_(s).add_(m)
        return img_tensor
    
    def visualize(self, image, mask, save_path = None):

        self.teacher_model.eval()
        self.student_model.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0).to(self.device)
        else:
            image = image.to(self.device)

        mask = mask.unsqueeze(0)

        with torch.no_grad():
            t_features = self.teacher_model(image)
            s_features = self.student_model(image)

        score_map = 1.

        for j in range(len(t_features)):
            t_features[j] = F.normalize(t_features[j], dim = 1)
            s_features[j] = F.normalize(s_features[j], dim = 1)

            sm = torch.sum((t_features[j] - s_features[j])**2, 1, keepdim = True)
            sm = F.interpolate(sm, size = (64, 64), mode = "bilinear", align_corners = False)

            score_map = score_map * sm
        
        score_map = score_map.squeeze(0).squeeze(0).cpu().numpy()
        print("Shape of score map:", score_map.shape)
        norm_score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)

        #resize score map
        original_size = image.shape[-2:]
        score_map_resized = F.interpolate(torch.tensor(norm_score_map).unsqueeze(0).unsqueeze(0), 
                                            size = original_size, mode = "bilinear", align_corners=False).squeeze().numpy()
        
        #convert tensors to image
        unnorm_tensor = self.unnormalize_image(image.cpu().squeeze().clone())
        input_image = unnorm_tensor.permute(1, 2, 0).numpy()
        input_image = np.clip(input_image, 0, 1)

        smoothed_mask = gaussian_filter(mask.float(), sigma=2).squeeze(0).squeeze(0)
        gt_mask = F.interpolate(torch.tensor(smoothed_mask).unsqueeze(0).unsqueeze(0), 
                                            size = original_size, mode = "bilinear", align_corners=False).squeeze().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(input_image)
        axs[0].set_title("Input Image")
        axs[1].imshow(input_image)
        axs[1].imshow(score_map_resized, cmap="jet", alpha=0.5)
        axs[1].set_title("Predicted Heatmap")
        axs[2].imshow(input_image)
        axs[2].imshow(gt_mask, cmap="jet", alpha = 0.5)
        axs[2].set_title("Ground Truth Heatmap")

        for ax in axs:
            ax.axis("off")
    
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

        plt.close()
        

    # def compute_pro(self, y_true, y_pred, num_thresholds = 100):
    #     pro_scores = []
    #     fpr_list = []
    #     thresholds = np.linspace(0, y_pred.max(), num_thresholds)

    #     total_pos = y_true.sum()

    #     for thresh in thresholds:
    #         binary_pred = (y_pred >= thresh).astype(np.uint8)

    #         pro_numerator = 0
    #         false_positive = 0

    #         for gt_mask, pred_mask in zip(y_true, binary_pred):
    #             gt_labels = label(gt_mask)
    #             pred_mask_filled = binary_fill_holes(pred_mask)

    #             false_positive += np.sum(pred_mask_filled * (1-gt_mask))

    #             for region_label in np.unique(gt_labels):
    #                 if region_label == 0:
    #                     continue
    #                 region = (gt_labels == region_label).astype(np.uint8)

    #                 intersection = np.sum(region * pred_mask_filled)
    #                 region_size = np.sum(region)

    #                 if region_size > 0:
    #                     pro_numerator += intersection / region_size

    #         num_regions = np.sum([len(np.unique(label(m))) - 1 for m in y_true])
    #         if num_regions == 0:
    #             pro_score = 0
    #         else:
    #             pro_score = pro_numerator / num_regions
            
    #         pro_scores.append(pro_score)

    #         fpr = false_positive / (y_true.size - total_pos)
    #         fpr_list.append(fpr)

    #     fpr_list, pro_scores = zip(*sorted(zip(fpr_list, pro_scores)))

    #     max_fpr = 0.3
    #     filtered = [(f, p) for f, p in zip(fpr_list, pro_scores) if f <= max_fpr]
    #     if len(filtered) > 1:
    #         fpr_arr, pro_arr = zip(*filtered)
    #         pro_auc = np.trapz(pro_arr, fpr_arr) / max_fpr
    #     else:
    #         pro_auc = 0
        
    #     return pro_auc

    def cal_pro_auc_pxl(self, scores: np.ndarray, gt_masks: np.ndarray) -> float:
        def rescale(x):
            return (x - x.min()) / (x.max() - x.min())

        """
        Calculate pixel-level pro auc score

        Args:
            scores (np.array)  : numpy array of predicted masks
            gt_mask (np.array) : numpy array of ground truth masks

        Returns:
            per_pixel_roc_auc (float) : pro_auc pixel level score
        """

        # remove the channel dimension
        gt = np.squeeze(gt_masks, axis=1)

        gt[gt <= 0.5] = 0
        gt[gt > 0.5] = 1
        gt = gt.astype(np.bool_)

        max_step = 200
        expect_fpr = 0.3

        # set the max and min scores and the delta step
        max_th = scores.max()
        min_th = scores.min()
        delta = (max_th - min_th) / max_step

        pros_mean = []
        threds = []
        fprs = []

        binary_score_maps = np.zeros_like(scores, dtype=np.bool_)

        for step in range(max_step):
            thred = max_th - step * delta

            # segment the scores with different thresholds
            binary_score_maps[scores <= thred] = 0
            binary_score_maps[scores > thred] = 1

            pro = []
            for i in range(len(binary_score_maps)):

                # label the regions in the ground truth
                label_map = label(gt[i], connectivity=2)

                # calculate some properties for every corresponding region
                props = regionprops(label_map, binary_score_maps[i])

                # calculate the per-regione overlap
                for prop in props:
                    pro.append(prop.intensity_image.sum() / prop.area)

            # append the per-region overlap
            pros_mean.append(np.array(pro).mean())

            # calculate the false positive rate
            gt_neg = ~gt
            fpr = np.logical_and(gt_neg, binary_score_maps).sum() / gt_neg.sum()
            fprs.append(fpr)
            threds.append(thred)

        threds = np.array(threds)
        pros_mean = np.array(pros_mean)
        fprs = np.array(fprs)

        # select the case when the false positive rates are under the expected fpr
        idx = fprs <= expect_fpr

        fprs_selected = fprs[idx]
        fprs_selected = rescale(fprs_selected)
        pros_mean_selected = rescale(pros_mean[idx])
        per_pixel_roc_auc = auc(fprs_selected, pros_mean_selected)

        return per_pixel_roc_auc
        
    def compute_pixel_f1(self, y_true, y_pred):
        precision, recall, _ = precision_recall_curve(y_true.flatten(), y_pred.flatten())

        a = 2 * precision * recall
        b = precision + recall

        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

        return np.max(f1)
    

    def min_max_norm(self, x):
        return (x - x.min()) / (x.max() - x.min())



    def evaluate(self):

        self.teacher_model.eval()
        self.student_model.eval()

        loss_map = np.zeros((len(self.test_dataloader.dataset), 64, 64))
        gt_masks = []

        i = 0

        for image, label, mask in self.test_dataloader:
            image = image.to(self.device)
            with torch.no_grad():
                t_features = self.teacher_model(image)
                s_features = self.student_model(image)

            score_map = 1.

            for j in range(len(t_features)):
                t_features[j] = F.normalize(t_features[j], dim = 1)
                s_features[j] = F.normalize(s_features[j], dim = 1)

                sm = torch.sum((t_features[j] - s_features[j])**2, 1, keepdim = True)
                sm = F.interpolate(sm, size = (64, 64), mode = "bilinear", align_corners = False)

                score_map = score_map * sm
            
            loss_map[i: i + image.size(0)] = score_map.squeeze().cpu().data.numpy()

            #collect and resize GT masks
            resized_masks = F.interpolate(mask.float(), size = (64, 64), mode = "nearest")
            gt_masks.append(resized_masks.squeeze().cpu().numpy())

            i += image.size(0)
        
        gt_masks = np.concatenate(gt_masks, axis = 0)

        y_pred = loss_map
        pred_masks = self.min_max_norm(y_pred)
        y_true = gt_masks
        
        pixel_auc = roc_auc_score(y_true.flatten(), pred_masks.flatten())

        f1_score = self.compute_pixel_f1(y_true, y_pred)

        # pixel_pro = self.compute_pro(y_true, y_pred)

        pixel_pro = self.cal_pro_auc_pxl(np.squeeze(pred_masks, axis=1), y_true)

        print(f"Pixel AUC = {pixel_auc:.4f}, Pixel-level F1 score = {f1_score:.4f}, Pixel PRO = {0:.4f}")


    

