import torch
import copy

class ResNetTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, device, lambda_, num_epochs, patience = 5):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.lambda_ = lambda_
        self.num_epochs = num_epochs
        self.patience = patience

        self.main_criterion = torch.nn.BCEWithLogitsLoss()
        self.attr_criterion = torch.nn.BCEWithLogitsLoss()

        self.best_val_loss = float("inf")
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.monitored_epochs = 0


    def joint_loss(self, main_pred, attr_preds, labels, concepts):

        labels = labels.float().unsqueeze(1)
        main_loss = self.main_criterion(main_pred, labels)

        #sum of concept losses
        attr_loss = sum(self.attr_criterion(pred.squeeze(1), concepts[:, j].float()) for j, pred in enumerate(attr_preds))

        return main_loss + self.lambda_ * attr_loss
    

    def compute_metrics(self, main_pred, attr_preds, labels, concepts):
        main_pred_labels = main_pred.argmax(dim=1)
        main_acc = (main_pred_labels == labels).float().mean().item()

        #average attribute accuracy
        attr_accs = []
        for j, pred in enumerate(attr_preds):
            pred_binary = (torch.sigmoid(pred.squeeze(1)) > 0.5).float()
            acc = (pred_binary == concepts[:, j].float()).float().mean().item()
            attr_accs.append(acc)
        attr_acc = sum(attr_accs)/len(attr_accs)

        return main_acc, attr_acc
    
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_main_acc = 0
        total_attr_acc = 0
        n_batches = len(self.train_dataloader)

        for images, labels, concepts in self.train_dataloader:
            images, labels, concepts = images.to(self.device), labels.to(self.device), concepts.to(self.device)
            self.optimizer.zero_grad()
            main_pred, attr_preds = self.model(images)

            loss = self.joint_loss(main_pred, attr_preds, labels, concepts)
            loss.backward()
            self.optimizer.step()

            main_acc, attr_acc = self.compute_metrics(main_pred, attr_preds, labels, concepts)
            total_loss += loss.item()
            total_main_acc += main_acc
            total_attr_acc += attr_acc

        return (total_loss / n_batches, total_main_acc / n_batches, total_attr_acc / n_batches)


    def val_epoch(self):
        self.model.eval()
        total_loss = 0
        total_main_acc = 0
        total_attr_acc = 0
        n_batches = len(self.val_dataloader)

        for images, labels, concepts in self.val_dataloader:
            images, labels, concepts = images.to(self.device), labels.to(self.device), concepts.to(self.device)
            main_pred, attr_preds = self.model(images)
            loss = self.joint_loss(main_pred, attr_preds, labels, concepts)

            main_acc, attr_acc = self.compute_metrics(main_pred, attr_preds, labels, concepts)
            total_loss += loss.item()
            total_main_acc += main_acc
            total_attr_acc += attr_acc

        return (total_loss / n_batches, total_main_acc / n_batches, total_attr_acc / n_batches)
    

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss, train_main_acc, train_attr_acc = self.train_epoch()
            val_loss, val_main_acc, val_attr_acc = self.val_epoch()

            print(
            f"Epoch {epoch+1}/{self.num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Main Train Accuracy: {train_main_acc:.4f}, Concept Train Accuracy: {train_attr_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Main Val Accuracy: {val_main_acc:.4f}, Concept Val Accuracy: {val_attr_acc:.4f}"
        )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
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
