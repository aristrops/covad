import torch
import time
import numpy as np

from utils.metrics import AverageMeter, binary_accuracy
from sklearn.metrics import f1_score, roc_auc_score

class CBMEvaluator:
    def __init__(self,
                 model,
                 num_attr: int,
                 attr_cols, 
                 test_dataloader,
                 device: torch.device, 
                 bottleneck: bool = False,
                 concepts: bool = False,
                 main_only: bool = False):

        self.model = model
        self.num_attr = num_attr
        self.attr_cols = attr_cols
        self.test_dataloader = test_dataloader
        self.device = device
        self.bottleneck = bottleneck
        self.concepts = concepts
        self.main_only = main_only
    
    def evaluate(self):
        self.model.eval()

        accuracy_meter_main = AverageMeter()
        accuracy_meter_attr = AverageMeter()

        if not self.bottleneck:
            all_main_preds, all_main_targets, all_main_probs = [], [], []
        else:
            all_main_preds = all_main_targets = all_main_probs = None

        if not self.main_only:
            all_attr_probs, all_attr_preds, all_attr_targets = [], [], []
        else:
            all_attr_probs = all_attr_preds = all_attr_targets = None
            accuracy_meter_attr = None
        
        total_inference_time = 0.0
        total_instances = 0
        
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
                start = time.time()
                predictions = self.model(inputs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.time()
            
            batch_time = end - start
            total_inference_time += batch_time
            total_instances += inputs.size(0)

            #compute main task accuracy
            if not self.bottleneck:
                logits_main = predictions[0].squeeze(1) if isinstance(predictions, list) else predictions.squeeze(1)
                probs_main = torch.sigmoid(logits_main)
                main_preds = (probs_main >= 0.5).int()

                all_main_preds.append(main_preds.cpu())
                all_main_targets.append(labels.cpu().int())
                all_main_probs.append(probs_main.cpu())

                accuracy_main = binary_accuracy(probs_main, labels)
                accuracy_meter_main.update(accuracy_main.item(), inputs.size(0))
            
            #if attributes exist, compute attribute accuracy
            if not self.main_only:
                if not self.bottleneck:
                    attr_logits = torch.cat(predictions[1:], dim = 1)
                else:
                    attr_logits = torch.cat(predictions, dim = 1)
                
                probs_attr = torch.sigmoid(attr_logits)
                attr_preds = (probs_attr >= 0.5).int()

                all_attr_probs.append(probs_attr.cpu())
                all_attr_preds.append(attr_preds.cpu())
                all_attr_targets.append(concepts.cpu().int())

                accuracy_attr = binary_accuracy(probs_attr.cpu(), concepts.cpu())
                accuracy_meter_attr.update(accuracy_attr.numpy(), inputs.size(0))
        
        if not self.bottleneck and all_main_preds is not None:
            all_main_preds = torch.cat(all_main_preds).numpy()
            all_main_targets = torch.cat(all_main_targets).numpy()
            all_main_probs = torch.cat(all_main_probs).numpy()

            #compute main F1 score
            f1_main = f1_score(all_main_targets, all_main_preds, average="binary")
            auc_main = roc_auc_score(all_main_targets, all_main_probs)
        else:
            f1_main = 0
            auc_main = 0

        if all_attr_probs:
            all_attr_probs = torch.cat(all_attr_probs).numpy()
            all_attr_preds = torch.cat(all_attr_preds).numpy()
            all_attr_targets = torch.cat(all_attr_targets).numpy()

            #compute AUC
            aucs = []
            for i in range(all_attr_targets.shape[1]):
                try:
                    auc = roc_auc_score(all_attr_targets[:, i], all_attr_probs[:, i])
                    aucs.append(auc)
                except ValueError:
                    aucs.append(np.nan)

            mean_auc = np.nanmean(aucs)

            #compute F1 score
            f1_attr = f1_score(all_attr_targets, all_attr_preds, average = "weighted")

            #compute attribute-specific f1-score
            f1_scores_attr = []
            for i, attr_name in enumerate(self.attr_cols):
                f1 = f1_score(all_attr_targets[:, i], all_attr_preds[:, i])
                f1_scores_attr.append((attr_name, f1))

            f1_scores_attr = sorted(f1_scores_attr, key=lambda x: (np.isnan(x[1]), x[1]))

            # print("\nPer-Attribute F1 Score:")
            # for name, score in f1_scores_attr:
            #     print(f"{name}: F1 Score = {score:.4f}")

        else:
            mean_auc = 0
            f1_attr = 0
        
        avg_inference_time = total_inference_time / total_instances

        if self.bottleneck:
            print(f"\nAccuracy of the concept prediction task: {accuracy_meter_attr.avg.item():.2f}")
            print(f"AUC of the concept prediction task: {mean_auc:.2f}")
            print(f"F1 Score of the concept prediction task: {f1_attr:.2f}")
        else:
            print(f"F1 Score of the main task: {f1_main:.2f}")
            print(f"AUC Score of the main task: {auc_main:.2f}")
            if self.concepts:
                print(f"\nAccuracy of the concept prediction task: {accuracy_meter_attr.avg.item():.2f}")
                print(f"AUC of the concept prediction task: {mean_auc:.2f}")
                print(f"F1 Score of the concept prediction task: {f1_attr:.2f}")
        
        #print(f"\nAverage inference time per instance: {avg_inference_time*1000:.4f} ms")

        if f1_main:
                return f1_main