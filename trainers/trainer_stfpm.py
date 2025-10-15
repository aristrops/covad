import numpy as np
import torch
import copy
import os

import torch.nn.functional as F

from utils.metrics import AverageMeter
from sklearn.metrics import roc_auc_score

from utils.metrics import compute_pixel_f1, min_max_norm, compute_pixel_pro, compute_pixel_pr

class STFPMTrainer:
    def __init__(self,
                teacher_model,
                student_model,
                train_dataloader,
                val_dataloader,
                optimizer, 
                scheduler,
                device: torch.device, 
                num_epochs: int, 
                patience: int = 10, 
                save_path = None):
        
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.save_path = save_path

        self.best_val_epoch = -1
        self.best_val_loss = float("inf")
        self.monitored_epochs = 0
        self.best_model_wts = copy.deepcopy(self.student_model.state_dict())

    def run_epoch(self, dataloader, loss_meter):

        self.student_model.train() 
        self.teacher_model.eval()

        for image, _ in dataloader:
            image = image.to(self.device)

            with torch.no_grad():
                t_features = self.teacher_model(image)
            s_features = self.student_model(image)

            loss = 0
            for i in range(len(t_features)):
                t_features[i] = F.normalize(t_features[i], dim = 1) #normalize along the channel dimension
                s_features[i] = F.normalize(s_features[i], dim = 1)
                loss += torch.sum((t_features[i] - s_features[i])**2, 1).mean()
            
            loss_meter.update(loss.item(), image.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss_meter
        
    def run_epoch_eval(self, dataloader):

        self.teacher_model.eval()
        self.student_model.eval()

        loss_map = np.zeros((len(dataloader.dataset), 64, 64))
        gt_masks = []

        i = 0

        for image, label, mask in dataloader:
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
        
        return loss_map, gt_masks
    
        
    def train(self):
        for epoch in range(self.num_epochs):

            loss_meter = AverageMeter()

            train_loss = self.run_epoch(self.train_dataloader, loss_meter)
            val_loss_map, gt_masks,  = self.run_epoch_eval(self.val_dataloader)
            val_loss = val_loss_map.mean()

            pred_masks = min_max_norm(val_loss_map)

            if isinstance(gt_masks, np.ndarray):
                y_true = (gt_masks > 0.5).astype(int)
            else:  
                y_true = (gt_masks > 0.5).int()

            y_pred = val_loss_map

            pixel_auc = roc_auc_score(y_true.flatten(), pred_masks.flatten())
            f1_score = compute_pixel_f1(y_true, y_pred)
            pixel_pro = compute_pixel_pro(pred_masks, y_true)

            print(f"Epoch [{epoch + 1}/{self.num_epochs}]: Train Loss = {train_loss.avg:.4f}, "
                  f"Val Loss = {val_loss:.4f}, Pixel AUC = {pixel_auc:.4f}, Pixel F1 Score = {f1_score:.4f}, Pixel PRO = {pixel_pro:.4f}")

            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_epoch = epoch
                self.best_val_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.student_model.state_dict())
                self.monitored_epochs = 0
            else:
                self.monitored_epochs += 1
                print(f"No improvement for {self.monitored_epochs} epochs")
            
            if self.monitored_epochs > self.patience:
                print("Early stopping triggered")
                break
        
        self.student_model.load_state_dict(self.best_model_wts)
        print("Best model restored.")

        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.student_model.state_dict(), self.save_path)
            print(f"Model saved to {self.save_path}")
