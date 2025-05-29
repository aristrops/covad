from models.resnet_18 import ResNet18model, MLP, End2EndModel

def joint_model(num_attr, expand_dim, use_relu, use_sigmoid):
    model_1 = ResNet18model(num_attr=num_attr, expand_dim=expand_dim, bottleneck=True)
    model_2 = MLP(input_dim=num_attr, expand_dim=expand_dim)

    return End2EndModel(model_1, model_2, use_relu, use_sigmoid)

def standard_model():
    return ResNet18model(num_attr=None)