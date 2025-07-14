import torch
import torch.nn as nn
import torchvision.models as models


class End2EndModel(nn.Module): #joint bottleneck
    def __init__(self, model_1, model_2, use_relu = False, use_sigmoid = False):
        super(End2EndModel, self).__init__()
        self.first_model = model_1
        self.second_model = model_2
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid

    def second_forward_stage(self, output_stage_1):
        if self.use_relu:
            attr_outputs = [nn.ReLU()(o) for o in output_stage_1]
        elif self.use_sigmoid:
            attr_outputs = [nn.Sigmoid()(o) for o in output_stage_1]
        else:
            attr_outputs = output_stage_1
        
        input_stage_2 = attr_outputs
        input_stage_2 = torch.cat(input_stage_2, dim = 1)
        all_outputs = [self.second_model(input_stage_2)] #main task prediction
        all_outputs.extend(output_stage_1) #concept predictions
        return all_outputs
    
    def forward(self, x):
        outputs = self.first_model(x)
        return self.second_forward_stage(outputs)
    

class MLP(nn.Module):
    def __init__(self, input_dim, expand_dim):
        super(MLP, self).__init__()
        self.expand_dim = expand_dim

        if self.expand_dim:
            self.linear = nn.Linear(input_dim, expand_dim)
            self.activation = nn.ReLU()
            self.linear_2 = nn.Linear(expand_dim, 1)
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        x = self.linear(x)
        if self.expand_dim:
            x = self.activation(x)
            x = self.linear_2(x)
        return x


class BackboneModel(nn.Module):
    def __init__(self,
                 num_attr: int,
                 num_classes: int,
                 freeze_parameters: bool,
                 expand_dim: int = 0,
                 bottleneck: bool = True,
                 backbone: str = "resnet18"):
        
        super(BackboneModel, self).__init__()
        
        self.num_attr = num_attr
        self.num_classes = num_classes
        self.bottleneck = bottleneck

        #load pretrained model
        if backbone == "resnet18":
            base_model = models.resnet18(pretrained = True)
            feature_dim = base_model.fc.in_features
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1]) #remove last FC layer
        elif backbone == "mobilenet_v2":
            base_model = models.mobilenet_v2(pretrained=True)
            feature_dim = base_model.last_channel
            self.feature_extractor = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1) #add pooling layer since mobilenet ends with conv
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        #freeze layers by default
        for name, param in base_model.named_parameters():
            param.requires_grad = False
        if not freeze_parameters: 
            #fine-tune last layer
            for name, param in base_model.named_parameters():
                if ("layer4" in name or #resnet18
                    any(f"features.{i}" in name for i in range(14, 19))): #mobilenet_v2
                    param.requires_grad = True

        self.fc_layers = nn.ModuleList() #list of fc layers for each prediction; main task is always the first fc layer

        if self.num_attr is not None:
            if not self.bottleneck:
                self.fc_layers.append(FC(feature_dim, 1, expand_dim))
            for _ in range(self.num_attr):
                self.fc_layers.append(FC(feature_dim, 1, expand_dim))
        else:
            self.fc_layers.append(FC(feature_dim, num_classes, expand_dim))
    
    def forward(self, x):
        x = self.feature_extractor(x)
        if hasattr(self, "pool"):
            x = self.pool(x)
        x = torch.flatten(x, 1)

        predictions = []

        for fc in self.fc_layers:
            predictions.append(fc(x))
        
        return predictions
  

#fc layers for prediction
class FC(nn.Module):
    def __init__(self, input_dim, output_dim, expand_dim):
        super(FC, self).__init__()

        self.expand_dim = expand_dim

        if self.expand_dim > 0: #create MLP rather than standard FC layer
            self.relu = nn.ReLU()
            self.fc_new = nn.Linear(input_dim, expand_dim)
            self.fc = nn.Linear(expand_dim, output_dim)

        else:
            self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        if self.expand_dim > 0:
            x = self.fc_new(x)
            x = self.relu(x)
        x = self.fc(x)
        return x