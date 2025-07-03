from models.model_backbones import BackboneModel, MLP, End2EndModel

def joint_model(num_attr, expand_dim, use_relu, use_sigmoid, freeze_parameters, backbone = "resnet18", model_state_dict = None):
    model_1 = BackboneModel(num_attr=num_attr, freeze_parameters=freeze_parameters, expand_dim=expand_dim, bottleneck=True, backbone = backbone)
    print(f"Using {backbone} for concept extraction...")
    if model_state_dict is not None:
        print(f"Using weights of pre-trained model for attribute prediction...")
        model_1.load_state_dict(model_state_dict, strict = False)
    if freeze_parameters:
        print("Using frozen parameters...")
    else:
        print("Fine-tuning last layer...")
    model_2 = MLP(input_dim=num_attr, expand_dim=expand_dim)

    return End2EndModel(model_1, model_2, use_relu, use_sigmoid)

def standard_model(freeze_parameters, backbone = "resnet18", model_state_dict = None): #for label prediction without using attributes
    model = BackboneModel(num_attr=None, freeze_parameters=freeze_parameters, backbone = backbone)
    print(f"Using {backbone} for prediction...")
    if model_state_dict is not None:
        print(f"Using weights of pre-trained model...")
        model.load_state_dict(model_state_dict)
    if freeze_parameters:
        print("Using frozen parameters...")
    else:
        print("Fine-tuning last layer...")
    return model

def concepts_model(num_attr, freeze_parameters, expand_dim, backbone = "resnet18", model_state_dict = None): #for concept prediction in independent and sequential model
    model = BackboneModel(num_attr=num_attr, freeze_parameters=freeze_parameters, expand_dim=expand_dim, bottleneck=True, backbone=backbone)
    print(f"Using {backbone} for concept extraction...")
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict, strict=False)
        print(f"Using weights of pre-trained model...")
    if freeze_parameters:
        print("Using frozen parameters...")
    else:
        print("Fine-tuning last layer...")
    return model

def main_model(num_attr, expand_dim, model_state_dict = None): #for main prediction in independent and sequential model
    model = MLP(input_dim=num_attr, expand_dim=expand_dim)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict) 
    return model
