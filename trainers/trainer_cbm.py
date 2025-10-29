import numpy as np
import torch
import copy
import os
import wandb

from utils.metrics import AverageMeter, binary_accuracy
from sklearn.metrics import f1_score

class CBMTrainer:
    def __init__(self, 
                 model, 
                 num_attr: int, 
                 train_dataloader, 
                 val_dataloader, 
                 optimizer, 
                 scheduler,
                 device: torch.device, 
                 lambda_: float, 
                 num_epochs: int, 
                 patience: int = 10, 
                 bottleneck: bool = False, 
                 concepts: bool = True, 
                 main_only: bool = False, 
                 multiclass: bool = False,
                 weight_main: bool = None, 
                 weight_attr: bool = None,
                 save_path: str = None,
                 use_wandb: bool = None):
        
        self.model = model.to(device)
        self.num_attr = num_attr
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.lambda_ = lambda_
        self.num_epochs = num_epochs
        self.patience = patience
        self.bottleneck = bottleneck
        self.concepts = concepts
        self.main_only = main_only
        self.multiclass = multiclass
        self.save_path = save_path
        self.use_wandb = use_wandb

        if weight_main is not None:
            pos_weight = torch.tensor(weight_main, dtype=torch.float32).to(device)
            self.main_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            if self.multiclass:
                self.main_criterion = torch.nn.CrossEntropyLoss()
            else:
                self.main_criterion = torch.nn.BCEWithLogitsLoss()
        
        if concepts:
            self.attr_criterion = []
            for i in range(self.num_attr):
                if weight_attr is not None:
                    pos_weight = torch.tensor(weight_attr[i], dtype = torch.float32)
                    self.attr_criterion.append(torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight))
                else:
                    self.attr_criterion.append(torch.nn.BCEWithLogitsLoss())
        else:
            self.attr_criterion = None

        self.best_val_epoch = -1
        self.best_val_loss = float("inf")
        self.monitored_epochs = 0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())

        self.convergence_epochs = 0
        self.convergence_patience = 10
    

    def run_epoch_main(self, dataloader, loss_meter, accuracy_meter, is_training): #prediction of A -> Y (independent and sequential) or prediction of X -> Y

        all_main_preds, all_main_targets = [], []

        self.model.train() if is_training else self.model.eval()

        for inputs, labels in dataloader:
            if self.multiclass:
                inputs, labels = inputs.to(self.device), labels.long().to(self.device)
            else:
                inputs, labels = inputs.to(self.device), labels.float().to(self.device)

            if isinstance(inputs, list):
                inputs = torch.stack(inputs).t().float()

            outputs = self.model(inputs)
            logits = outputs[0].squeeze(1) if isinstance(outputs, list) else outputs.squeeze(1)

            loss = self.main_criterion(logits, labels)

            if self.multiclass:
                preds = torch.argmax(logits, dim = 1)
            else:
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).int()

            loss_meter.update(loss.item(), inputs.size(0))
            if self.multiclass:
                accuracy_meter.update((preds == labels).float().mean().item(), inputs.size(0))
            else:
                accuracy_meter.update(binary_accuracy(probs, labels).item(), inputs.size(0))
            
            all_main_preds.append(preds.cpu())
            all_main_targets.append(labels.cpu().int())

            if is_training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        all_main_preds = torch.cat(all_main_preds).numpy()
        all_main_targets = torch.cat(all_main_targets).numpy()

        if self.multiclass:
            f1_main = f1_score(all_main_targets, all_main_preds, average="weighted")
        else:
            f1_main = f1_score(all_main_targets, all_main_preds, average="binary")
        
        return loss_meter, accuracy_meter, f1_main


    def run_epoch(self, dataloader, loss_meter, accuracy_meter_main, accuracy_meter_attr, is_training):

        self.model.train() if is_training else self.model.eval()

        if not self.bottleneck:
            all_main_preds, all_main_targets = [], []
        else:
            all_main_preds = all_main_targets = None

        if self.concepts:
            all_attr_preds, all_attr_targets = [], []
        else:
            all_attr_preds = all_attr_targets = None
            accuracy_meter_attr = None

        for images, concepts, labels in dataloader:
            images, labels = images.to(self.device), labels.float().to(self.device)
            concepts = concepts.to(self.device) if self.concepts else None

            predictions = self.model(images)
            losses = []
            output_start = 0 #where attribute outputs start

            if not self.bottleneck:
                #compute main task loss
                main_loss = self.main_criterion(predictions[0].squeeze(1), labels)
                losses.append(main_loss)
                output_start = 1

            #compute attribute loss (always)
            for i, attr_criterion in enumerate(self.attr_criterion):
                ground_truths = concepts[:, i]
                predicted_attributes = predictions[i + output_start]
                losses.append(self.lambda_ * attr_criterion(predicted_attributes.squeeze(1).float(), ground_truths))

            #compute main task accuracy
            if not self.bottleneck:
                logits_main = predictions[0].squeeze(1)
                probs_main = torch.sigmoid(logits_main)
                main_preds = (probs_main >= 0.5).int()

                all_main_preds.append(main_preds.cpu())
                all_main_targets.append(labels.cpu().int())

                accuracy_main = binary_accuracy(probs_main, labels)
                accuracy_meter_main.update(accuracy_main.item(), images.size(0))
            
            #if attributes exist, compute attribute accuracy
            if len(predictions) > 1:
                if not self.bottleneck:
                    attr_logits = torch.cat(predictions[1:], dim = 1)
                else:
                    attr_logits = torch.cat(predictions, dim = 1)
                
                probs_attr = torch.sigmoid(attr_logits)
                attr_preds = (probs_attr >= 0.5).int()

                all_attr_preds.append(attr_preds.cpu())
                all_attr_targets.append(concepts.cpu().int())

                accuracy_attr = binary_accuracy(probs_attr, concepts)
                accuracy_meter_attr.update(accuracy_attr.numpy(), images.size(0))

            if self.concepts:
                if self.bottleneck:
                    total_loss = sum(losses) / self.num_attr
                else:
                    total_loss = (losses[0] + sum(losses[1:])) / (1 + self.lambda_ * self.num_attr) #normalize loss
            else: 
                total_loss = sum(losses)

            loss_meter.update(total_loss.item(), images.size(0))

            if is_training:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        
        if not self.bottleneck and all_main_preds is not None:
            all_main_preds = torch.cat(all_main_preds).numpy()
            all_main_targets = torch.cat(all_main_targets).numpy()

            f1_main = f1_score(all_main_targets, all_main_preds, average="binary")
        else:
            f1_main = 0

        if all_attr_preds:
            all_attr_preds = torch.cat(all_attr_preds).numpy()
            all_attr_targets = torch.cat(all_attr_targets).numpy()

            f1_attr = f1_score(all_attr_targets, all_attr_preds, average = "weighted")
        else:
            f1_attr = 0
        
        if self.concepts:
            return loss_meter, accuracy_meter_main, accuracy_meter_attr, f1_main, f1_attr
        else:
            return loss_meter, accuracy_meter_main, f1_main
    

    def helper_train(self, epoch, dataloader, is_training):
        loss_meter = AverageMeter()
        acc_main = AverageMeter()
        acc_attr = AverageMeter() if self.concepts else None
    
        if self.main_only:
            loss_meter, acc_main, f1_main = self.run_epoch_main(dataloader, loss_meter, acc_main, is_training=is_training)

            return loss_meter.avg, acc_main.avg, 0, f1_main, 0
        
        loss_meter, acc_main, acc_attr, f1_main, f1_attr = self.run_epoch(dataloader, loss_meter, acc_main, acc_attr, is_training=is_training)

        return loss_meter.avg, acc_main.avg, acc_attr.avg.item(), f1_main, f1_attr


    def train(self):

        if self.use_wandb:
            if self.concepts and not self.main_only:
                wandb.watch(self.model, [self.main_criterion] + self.attr_criterion, log = "all", log_freq = 100)
            elif self.main_only:
                wandb.watch(self.model, self.main_criterion, log = "all", log_freq=100)
            elif self.bottleneck:
                wandb.watch(self.model, self.attr_criterion, log = "all", log_freq=100)

        for epoch in range(self.num_epochs):

            train_loss, train_acc_main, train_acc_attr, train_f1_main, train_f1_attr = self.helper_train(epoch, self.train_dataloader, is_training=True)

            with torch.no_grad():
                val_loss, val_acc_main, val_acc_attr, val_f1_main, val_f1_attr = self.helper_train(epoch, self.val_dataloader, is_training=False)
            
            log = f"Epoch [{epoch}/{self.num_epochs}]: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"

            if self.main_only:
                log += (f", Train Main Acc = {train_acc_main:.4f}, Train Main F1 = {train_f1_main:.4f}"
                        f", Val Main Acc = {val_acc_main:.4f}, Val Main F1 = {val_f1_main:.4f}")
            elif self.bottleneck:
                log += (f", Train Attr Acc = {train_acc_attr:.4f}, Train Attr F1 = {train_f1_attr:.4f}"
                    f", Val Attr Acc = {val_acc_attr:.4f}, Val Attr F1 = {val_f1_attr:.4f}")
            else:
                log += (f", Train Main Acc = {train_acc_main:.4f}, Train Attr Acc = {train_acc_attr:.4f}, "
                    f"Train Main F1 = {train_f1_main:.4f}, Train Attr F1 = {train_f1_attr:.4f}"
                    f", Val Main Acc = {val_acc_main:.4f}, Val Attr Acc = {val_acc_attr:.4f}, "
                    f"Val Main F1 = {val_f1_main:.4f}, Val Attr F1 = {val_f1_attr:.4f}")
                
            print(log)

            if self.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            
                if self.main_only:
                    log_dict.update({
                        "train_main_acc": train_acc_main,
                        "train_main_f1": train_f1_main,
                        "val_main_acc": val_acc_main,
                        "val_main_f1": val_f1_main,
                    })
                elif self.bottleneck:
                    log_dict.update({
                        "train_attr_acc": train_acc_attr,
                        "train_attr_f1": train_f1_attr,
                        "val_attr_acc": val_acc_attr,
                        "val_attr_f1": val_f1_attr,
                    })
                else:
                    log_dict.update({
                        "train_main_acc": train_acc_main,
                        "train_attr_acc": train_acc_attr,
                        "train_main_f1": train_f1_main,
                        "train_attr_f1": train_f1_attr,
                        "val_main_acc": val_acc_main,
                        "val_attr_acc": val_acc_attr,
                        "val_main_f1": val_f1_main,
                        "val_attr_f1": val_f1_attr,
                    })
                
                wandb.log(log_dict)

            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_epoch = epoch
                self.best_val_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.monitored_epochs = 0
            else:
                self.monitored_epochs += 1
                print(f"No improvement for {self.monitored_epochs} epochs")
            
            if self.main_only:
                val_f1 = val_f1_main
            else:
                val_f1 = val_f1_attr
            
            if val_f1 == 1:
                self.convergence_epochs += 1
            else:
                self.convergence_epochs = 0
            
            if self.convergence_epochs >= self.convergence_patience:
                print(f"Convergence achieved (F1=1.0 for {self.convergence_patience} epochs). Stopping early.")
                break

            if self.monitored_epochs > self.patience:
                print("Early stopping triggered")
                break
        
        self.model.load_state_dict(self.best_model_wts)
        print("\nBest model restored.")

        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.save_path)
            print(f"Model saved to {self.save_path}")
        
        if self.main_only:
            return val_f1_main
        
        elif self.bottleneck:
            return val_f1_attr
        
        else:
            return val_f1_main, val_f1_attr
    

        
