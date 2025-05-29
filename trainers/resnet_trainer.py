import numpy as np
import torch
import torch.nn as nn
import copy

from metrics import AverageMeter, binary_accuracy
from sklearn.metrics import f1_score

class ResNetTrainer:
    def __init__(self, model, num_attr, train_dataloader, val_dataloader, weight, optimizer, device, lambda_, num_epochs, patience = 5, bottleneck = False, concepts = True):
        self.model = model.to(device)
        self.num_attr = num_attr
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.weight = weight
        self.optimizer = optimizer
        self.device = device
        self.lambda_ = lambda_
        self.num_epochs = num_epochs
        self.patience = patience
        self.bottleneck = bottleneck
        self.concepts = concepts

        if self.weight is not None:
            self.main_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.weight)
        else:
            self.main_criterion = torch.nn.BCEWithLogitsLoss()
        if concepts:
            self.attr_criterion = []
            for attr in range(self.num_attr):
                self.attr_criterion.append(torch.nn.BCEWithLogitsLoss())

        self.best_val_epoch = -1
        self.best_val_loss = float("inf")
        self.monitored_epochs = 0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
    

    def run_epoch(self, dataloader, loss_meter, accuracy_meter_main, accuracy_meter_attr, is_training):

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        all_main_preds, all_main_targets = [], []

        if self.concepts:
            all_attr_preds, all_attr_targets = [], []
        else:
            all_attr_preds, all_attr_targets = None, None
            
        if not self.concepts:
            accuracy_meter_attr = None

        for images, labels, concepts in dataloader:

            images, labels, concepts = images.to(self.device), labels.to(self.device), concepts.to(self.device)

            predictions = self.model(images)
            
            losses = []
            output_start = 0 #where attribute outputs start
            if not self.bottleneck:
                main_loss = self.main_criterion(predictions[0].squeeze(1), labels)
                losses.append(main_loss)
                output_start = 1
            if self.concepts:
                for i in range(len(self.attr_criterion)):
                    ground_truths = concepts[:, i]
                    predicted_attributes = predictions[i + output_start]
                    losses.append(self.lambda_ * self.attr_criterion[i](predicted_attributes.squeeze(1).type(torch.FloatTensor), ground_truths))

            #compute accuracy on the main task
            sigmoid_outputs = nn.Sigmoid()(predictions[0].squeeze(1))
            main_predictions = (sigmoid_outputs >= 0.5).int()
            all_main_preds.append(main_predictions)
            all_main_targets.append(labels.detach().cpu().int())

            accuracy_main = binary_accuracy(sigmoid_outputs, labels)
            accuracy_meter_main.update(accuracy_main.item(), images.size(0))
            
            #if attributes exist, compute attribute accuracy
            if len(predictions) > 1:
                sigmoid_outputs = nn.Sigmoid()(torch.cat(predictions[1:], dim = 1))
                attr_predictions = (sigmoid_outputs >= 0.5).int()
                all_attr_preds.append(attr_predictions)
                all_attr_targets.append(concepts.detach().cpu().int())

                accuracy = binary_accuracy(sigmoid_outputs, concepts)
                accuracy_meter_attr.update(accuracy.data.cpu().numpy(), images.size(0))

            if self.concepts:
                if self.bottleneck:
                    total_loss = sum(losses) / self.num_attr
                else:
                    total_loss = losses[0] + sum(losses[1:])
                    total_loss = total_loss / (1 + self.lambda_ * self.num_attr) #normalize loss
            else: 
                total_loss = sum(losses)
            loss_meter.update(total_loss.item(), images.size(0))

            if is_training:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        
        all_main_preds = torch.cat(all_main_preds).numpy()
        all_main_targets = torch.cat(all_main_targets).numpy()

        if is_training:
            print("Count of label=0:", np.sum(all_main_targets == 0))
            print("Count of label=1:", np.sum(all_main_targets == 1))
            print("Count of predictions=0:", np.sum(all_main_preds == 0))
            print("Count of predictions=1:", np.sum(all_main_preds == 1))

        f1_main = f1_score(all_main_targets, all_main_preds, average="binary")

        if all_attr_preds:
            all_attr_preds = torch.cat(all_attr_preds).numpy()
            all_attr_targets = torch.cat(all_attr_targets).numpy()
            f1_attr = f1_score(all_attr_targets, all_attr_preds, average = "macro")
        else:
            f1_attr = None
            
        return loss_meter, accuracy_meter_main, accuracy_meter_attr, f1_main, f1_attr
    

    def train(self):

        for epoch in range(self.num_epochs):
            train_loss = AverageMeter()
            train_acc_main = AverageMeter()
            train_acc_attr = AverageMeter()

            train_loss, train_acc_main, train_acc_attr, train_f1_main, train_f1_attr = self.run_epoch(self.train_dataloader, train_loss, train_acc_main, train_acc_attr, is_training=True)

            #evaluate on validation set
            val_loss = AverageMeter()
            val_acc_main = AverageMeter()
            val_acc_attr = AverageMeter()


            with torch.no_grad():
                val_loss, val_acc_main, val_acc_attr, val_f1_main, val_f1_attr = self.run_epoch(self.val_dataloader, val_loss, val_acc_main, val_acc_attr, is_training=False)
            
            train_loss_avg = train_loss.avg
            val_loss_avg = val_loss.avg

            if self.concepts:
                print(f"Epoch [{epoch}/{self.num_epochs}]: Train Loss = {train_loss_avg:.4f}, Train Main Accuracy = {train_acc_main.avg:.4f}, Train Attribute Accuracy = {train_acc_attr.avg.item():.4f}, Train Main F1 = {train_f1_main:.4f}, Train Attribute F1 = {train_f1_attr:.4f}"
                  f"| Val Loss = {val_loss_avg:.4f}, Val Main Accuracy = {val_acc_main.avg:.4f}, Val Attribute Accuracy = {val_acc_attr.avg.item():.4f}, Val Main F1 = {val_f1_main:.4f}, Val Attribute F1 = {val_f1_attr:.4f}")
            else:
                print(f"Epoch [{epoch}/{self.num_epochs}]: Train Loss = {train_loss_avg:.4f}, Train Main Accuracy = {train_acc_main.avg:.4f}, Train Main F1 = {train_f1_main:.4f}"
                  f"| Val Loss = {val_loss_avg:.4f}, Val Main Accuracy = {val_acc_main.avg:.4f}, Val Main F1 = {val_f1_main:.4f}")

            if val_loss_avg < self.best_val_loss:
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
