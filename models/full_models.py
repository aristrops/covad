from models.model_backbones import ResNet18model, MLP, End2EndModel

def joint_model(num_attr, expand_dim, use_relu, use_sigmoid, freeze_parameters, model_state_dict = None):
    model_1 = ResNet18model(num_attr=num_attr, freeze_parameters=freeze_parameters, expand_dim=expand_dim, bottleneck=True)
    if model_state_dict is not None:
        print(f"Using weights of pre-trained model for attribute prediction...")
        model_1.load_state_dict(model_state_dict, strict = False)
    if freeze_parameters:
        print("Training with frozen parameters...")
    else:
        print("Fine-tuning layer 4...")
    model_2 = MLP(input_dim=num_attr, expand_dim=expand_dim)

    return End2EndModel(model_1, model_2, use_relu, use_sigmoid)

def standard_model(freeze_parameters, model_state_dict): #for label prediction without using attributes
    model = ResNet18model(num_attr=None, freeze_parameters=freeze_parameters)
    if model_state_dict is not None:
        print(f"Using weights of pre-trained model...")
        model.load_state_dict(model_state_dict)
    if freeze_parameters:
        print("Training with frozen parameters...")
    else:
        print("Fine-tuning layer 4...")
    return model

def concepts_model(num_attr, freeze_parameters, expand_dim, model_state_dict = None): #for concept prediction in independent and sequential model
    model = ResNet18model(num_attr=num_attr, freeze_parameters=freeze_parameters, expand_dim=expand_dim, bottleneck=True)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict, strict=False)
        print(f"Using weights of pre-trained model...")
    if freeze_parameters:
        print("Training with frozen parameters...")
    else:
        print("Fine-tuning layer 4...")
    return model

def main_model(num_attr, expand_dim): #for main prediction in independent and sequential model
    return MLP(input_dim=num_attr, expand_dim=expand_dim)
