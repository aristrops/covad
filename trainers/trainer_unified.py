import copy
import os

import torch
import torch.nn.functional as F

from utils.metrics import AverageMeter


class UnifiedTrainer:
    """Jointly trains the STFPM student and the concept net of a UnifiedModel.

    Loss = STFPM feature-matching loss (masked to normal samples only)
           + lambda_concept * mean per-concept BCE (all samples).
    """

    def __init__(self,
                 model,
                 num_attr: int,
                 train_dataloader,
                 val_dataloader,
                 optimizer,
                 scheduler,
                 device: torch.device,
                 num_epochs: int,
                 lambda_: float = 0.55,
                 weight_attr=None,
                 weight_main=None,
                 patience: int = 10,
                 save_path: str = None):

        self.model = model.to(device)
        self.num_attr = num_attr
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.lambda_ = lambda_
        self.patience = patience
        self.save_path = save_path

        # main-task (anomaly label) criterion, optionally class-weighted
        if weight_main is not None:
            pos_weight = torch.tensor(weight_main, dtype=torch.float32).to(device)
            self.main_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.main_criterion = torch.nn.BCEWithLogitsLoss()

        self.attr_criterion = []
        for i in range(num_attr):
            if weight_attr is not None:
                pos_weight = torch.tensor(weight_attr[i], dtype=torch.float32).to(device)
                self.attr_criterion.append(torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight))
            else:
                self.attr_criterion.append(torch.nn.BCEWithLogitsLoss())

        self.best_val_epoch = -1
        self.best_val_loss = float("inf")
        self.monitored_epochs = 0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def compute_losses(self, t_features, s_features, concept_logits, main_logit, labels, concepts):
        # ---- STFPM loss, masked to normal samples (label == 0) ----
        normal_mask = (labels == 0).float()  # [B]
        n_normal = normal_mask.sum()

        stfpm_loss = torch.zeros((), device=self.device)
        for i in range(len(t_features)):
            tn = F.normalize(t_features[i], dim=1)
            sn = F.normalize(s_features[i], dim=1)
            per_sample = torch.sum((tn - sn) ** 2, dim=1).mean(dim=(1, 2))  # [B]
            if n_normal > 0:
                stfpm_loss = stfpm_loss + (per_sample * normal_mask).sum() / n_normal

        # ---- CBM loss (same normalization as the original joint model) ----
        main_loss = self.main_criterion(main_logit.squeeze(1).float(), labels)

        concept_sum = torch.zeros((), device=self.device)
        for i, crit in enumerate(self.attr_criterion):
            concept_sum = concept_sum + crit(
                concept_logits[i].squeeze(1).float(), concepts[:, i]
            )
        cbm_loss = (main_loss + self.lambda_ * concept_sum) / (1 + self.lambda_ * self.num_attr)

        total = stfpm_loss + cbm_loss
        return total, stfpm_loss, main_loss, concept_sum / self.num_attr

    def run_epoch(self, dataloader, is_training):
        loss_meter = AverageMeter()
        stfpm_meter = AverageMeter()
        main_meter = AverageMeter()
        concept_meter = AverageMeter()

        self.model.train() if is_training else self.model.eval()

        for images, concepts, labels in dataloader:
            images = images.to(self.device)
            concepts = concepts.to(self.device).float()
            labels = labels.to(self.device).float()

            t_features, s_features, concept_logits, main_logit = self.model(images)
            total, stfpm_loss, main_loss, concept_loss = self.compute_losses(
                t_features, s_features, concept_logits, main_logit, labels, concepts
            )

            loss_meter.update(total.item(), images.size(0))
            stfpm_meter.update(stfpm_loss.item(), images.size(0))
            main_meter.update(main_loss.item(), images.size(0))
            concept_meter.update(concept_loss.item(), images.size(0))

            if is_training:
                self.optimizer.zero_grad()
                total.backward()
                self.optimizer.step()

        return loss_meter, stfpm_meter, main_meter, concept_meter

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss, train_stfpm, train_main, train_concept = self.run_epoch(
                self.train_dataloader, is_training=True
            )
            with torch.no_grad():
                val_loss, val_stfpm, val_main, val_concept = self.run_epoch(
                    self.val_dataloader, is_training=False
                )

            print(f"Epoch [{epoch + 1}/{self.num_epochs}]: "
                  f"Train Loss = {train_loss.avg:.4f} (stfpm {train_stfpm.avg:.4f}, "
                  f"main {train_main.avg:.4f}, concept {train_concept.avg:.4f}), "
                  f"Val Loss = {val_loss.avg:.4f} (stfpm {val_stfpm.avg:.4f}, "
                  f"main {val_main.avg:.4f}, concept {val_concept.avg:.4f})")

            if self.scheduler is not None:
                self.scheduler.step(val_loss.avg)

            if val_loss.avg < self.best_val_loss:
                self.best_val_epoch = epoch
                self.best_val_loss = val_loss.avg
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.monitored_epochs = 0
            else:
                self.monitored_epochs += 1
                print(f"No improvement for {self.monitored_epochs} epochs")

            if self.monitored_epochs > self.patience:
                print("Early stopping triggered")
                break

        self.model.load_state_dict(self.best_model_wts)
        print("Best model restored.")

        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.save_path)
            print(f"Model saved to {self.save_path}")
