from models.resnet_18 import ResNet18model, MLP, End2EndModel

def joint_model(num_attr, expand_dim, use_relu, use_sigmoid, freeze_parameters):
    model_1 = ResNet18model(num_attr=num_attr, expand_dim=expand_dim, bottleneck=True, freeze_parameters=freeze_parameters)
    model_2 = MLP(input_dim=num_attr, expand_dim=expand_dim)

    return End2EndModel(model_1, model_2, use_relu, use_sigmoid)

def standard_model():
    return ResNet18model(num_attr=None)

def concepts_model(num_attr, expand_dim): #for concept prediction in independent and sequential model
    return ResNet18model(num_attr=num_attr, expand_dim=expand_dim, bottleneck=True)

def main_model(num_attr, expand_dim): #for main prediction in independent and sequential model
    return MLP(input_dim=num_attr, expand_dim=expand_dim)
