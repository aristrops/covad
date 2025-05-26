import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18model(nn.Module):
    def __init__(self,
                 num_attr: int,
                 expand_dim: int = 0,
                 connect_CY: bool = True):
        
        super(ResNet18model, self).__init__()

        #load pretrained resnet
        base_model = models.resnet18(pretrained = True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1]) #remove last FC layer
        feature_dim = base_model.fc.in_features

        self.num_attr = num_attr
        self.connect_CY = connect_CY

        #FC heads for attribute prediction
        self.attr_heads = nn.ModuleList()
        for _ in range(num_attr):
            output_dim = 1
            self.attr_heads.append(self._make_fc(feature_dim, output_dim, expand_dim))

        self.main_fc = self._make_fc(feature_dim, 1, expand_dim)

        #optional: connect concepts to label (joint model)
        if connect_CY:
            self.cy_fc = self._make_fc(num_attr, 1, expand_dim)
        else:
            self.cy_fc = None
        
    
    #create FC layers
    def _make_fc(self, input_dim, output_dim, expand_dim):
        if expand_dim > 0: #create MLP rather than standard FC layer
            return nn.Sequential(
                nn.Linear(input_dim, expand_dim),
                nn.ReLU(),
                nn.Linear(expand_dim, output_dim))
        else:
            return nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)

        #predict concepts
        attr_preds = []
        for head in self.attr_heads:
            attr_preds.append(head(x))
        if self.num_attr > 0:
            attr_preds_concat = torch.cat(attr_preds, dim = 1)
        else:
            attr_preds_concat = None
        
        #predict main label
        main_pred = self.main_fc(x)

        #optional: connect concepts to main task
        if self.connect_CY and attr_preds_concat is not None:
            main_pred += self.cy_fc(attr_preds_concat)
        
        if self.num_attr > 0:
            return main_pred, attr_preds
        else:
            return main_pred