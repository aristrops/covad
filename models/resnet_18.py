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


class ResNet18model(nn.Module):
    def __init__(self,
                 num_attr: int,
                 expand_dim: int = 0,
                 bottleneck: bool = True,
                 freeze_parameters: bool = True):
        
        super(ResNet18model, self).__init__()

        #load pretrained resnet
        base_model = models.resnet18(pretrained = True)
        if freeze_parameters:
            for param in base_model.parameters():
                param.requires_grad = False

        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1]) #remove last FC layer
        feature_dim = base_model.fc.in_features

        self.num_attr = num_attr
        self.bottleneck = bottleneck

        self.fc_layers = nn.ModuleList() #list of fc layers for each prediction; main task is always the first fc layer

        if self.num_attr is not None:
            if not self.bottleneck:
                self.fc_layers.append(FC(feature_dim, 1, expand_dim))
            for _ in range(self.num_attr):
                self.fc_layers.append(FC(feature_dim, 1, expand_dim))
        else:
            self.fc_layers.append(FC(feature_dim, 1, expand_dim))
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)

        predictions = []

        for fc in self.fc_layers:
            predictions.append(fc(x))
        # if self.num_attr > 0 and self.cy_fc is not None:
        #     attr_preds = torch.cat(predictions[1:], dim = 1)
        #     predictions[0] += self.cy_fc(attr_preds) #add attribute prediction to the main task
        
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