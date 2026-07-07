import torch
import torch.nn as nn
import torchvision.models as models


class End2EndModel(nn.Module):  # joint bottleneck
    def __init__(self, model_1, model_2, use_relu=False, use_sigmoid=False):
        super(End2EndModel, self).__init__()
        self.first_model = model_1
        self.second_model = model_2
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid
        self.train_df = None  # Placeholder for train dataframe if needed

    def second_forward_stage(self, output_stage_1):
        if self.use_relu:
            attr_outputs = [nn.ReLU()(o) for o in output_stage_1]
        elif self.use_sigmoid:
            attr_outputs = [nn.Sigmoid()(o) for o in output_stage_1]
        else:
            attr_outputs = output_stage_1

        input_stage_2 = attr_outputs
        input_stage_2 = torch.cat(input_stage_2, dim=1)
        all_outputs = [self.second_model(input_stage_2)]  # main task prediction
        all_outputs.extend(output_stage_1)  # concept predictions
        return all_outputs

    def forward(self, x):
        outputs = self.first_model(x)
        return self.second_forward_stage(outputs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["train_df"] = getattr(self, "train_df", None)
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        setattr(self, "train_df", state_dict.pop("train_df", None))
        return super().load_state_dict(state_dict, strict=False)


class MLP(nn.Module):
    def __init__(self, input_dim, expand_dim):
        super(MLP, self).__init__()
        self.expand_dim = expand_dim

        if self.expand_dim:
            self.linear = nn.Linear(input_dim, expand_dim)
            self.activation = nn.ReLU()
            self.linear_2 = nn.Linear(expand_dim, 1)
        else:
            self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.linear(x)
        if self.expand_dim:
            x = self.activation(x)
            x = self.linear_2(x)
        return x


class BackboneModel(nn.Module):
    def __init__(
        self,
        num_attr: int,
        num_classes: int,
        freeze_parameters: bool,
        pretrained: bool = True,
        expand_dim: int = 0,
        bottleneck: bool = True,
        backbone: str = "resnet18",
    ):

        super(BackboneModel, self).__init__()

        self.num_attr = num_attr
        self.num_classes = num_classes
        self.bottleneck = bottleneck
        self.train_df = None  # Placeholder for train dataframe if needed

        # load pretrained model
        if backbone == "resnet18":
            base_model = models.resnet18(pretrained=pretrained)
            feature_dim = base_model.fc.in_features
            self.feature_extractor = nn.Sequential(
                *list(base_model.children())[:-1]
            )  # remove last FC layer
        elif backbone == "mobilenet_v2":
            base_model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT
            )
            feature_dim = base_model.last_channel
            self.feature_extractor = base_model.features
            # self.pool = nn.AdaptiveAvgPool2d(1) #add pooling layer since mobilenet ends with conv
            self.pool = nn.AvgPool2d(kernel_size=7)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # freeze layers by default
        for name, param in base_model.named_parameters():
            param.requires_grad = False
        if not freeze_parameters:
            # fine-tune last layer
            for name, param in base_model.named_parameters():
                if "layer4" in name or any(  # resnet18
                    f"features.{i}" in name for i in range(14, 19)
                ):  # mobilenet_v2
                    param.requires_grad = True

        self.fc_layers = (
            nn.ModuleList()
        )  # list of fc layers for each prediction; main task is always the first fc layer

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

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["train_df"] = getattr(self, "train_df", None)
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        setattr(self, "train_df", state_dict.pop("train_df", None))
        return super().load_state_dict(state_dict, strict=False)


# fc layers for prediction
class FC(nn.Module):
    def __init__(self, input_dim, output_dim, expand_dim):
        super(FC, self).__init__()

        self.expand_dim = expand_dim

        if self.expand_dim > 0:  # create MLP rather than standard FC layer
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


class ConceptNetFromDiff(nn.Module):
    """Truncated MobileNetV2 that predicts concepts from the first STFPM
    teacher-student feature difference.

    The difference at mobilenet block 3 has 24 channels, which is exactly the
    input expected by mobilenet ``features[4]``. We therefore reuse
    ``features[start:]`` (pretrained, trainable) as the concept sub-network,
    followed by the same pool + per-concept FC heads used by ``BackboneModel``.
    """

    def __init__(self, num_attr: int, expand_dim: int = 0, start: int = 4,
                 pretrained: bool = True):
        super(ConceptNetFromDiff, self).__init__()

        base_model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        feature_dim = base_model.last_channel  # 1280
        # truncated conv stack; consumes the 24-channel diff at block-3 resolution
        self.features = nn.Sequential(*list(base_model.features)[start:])
        self.pool = nn.AvgPool2d(kernel_size=7)

        self.fc_layers = nn.ModuleList()
        for _ in range(num_attr):
            self.fc_layers.append(FC(feature_dim, 1, expand_dim))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return [fc(x) for fc in self.fc_layers]


class UnifiedModel(nn.Module):
    """Single branch producing the STFPM heatmap, the concepts, and the final
    anomaly-score prediction from the concepts.

    Same structure as the original joint CBM (concept bottleneck -> MLP head),
    except the concept classifier consumes the first STFPM teacher-student
    feature difference instead of raw-image features.

    forward returns ``(t_features, s_features, concept_logits, main_logit)``:
    - ``t_features``/``s_features``: raw teacher/student feature lists (heatmap +
      STFPM loss),
    - ``concept_logits``: per-concept logits from the first normalized diff,
    - ``main_logit``: image-level anomaly logit from the concept bottleneck.
    """

    def __init__(self, num_attr: int, expand_dim: int = 0,
                 backbone: str = "mobilenet_v2", concept_layer_idx: int = 0,
                 use_relu: bool = True, use_sigmoid: bool = False):
        super(UnifiedModel, self).__init__()
        if backbone != "mobilenet_v2":
            raise ValueError("UnifiedModel only supports mobilenet_v2")

        self.concept_layer_idx = concept_layer_idx
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid

        self.teacher = BackboneModelFeatures(pretrained=True, backbone=backbone)
        self.student = BackboneModelFeatures(pretrained=False, backbone=backbone)

        # freeze teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.concept_net = ConceptNetFromDiff(num_attr=num_attr, expand_dim=expand_dim)
        # main-task head over the concept bottleneck (same as joint CBM)
        self.main_model = MLP(input_dim=num_attr, expand_dim=expand_dim)

    def load_teacher(self, state_dict):
        """Load teacher feature_extractor weights (strict=False)."""
        self.teacher.load_state_dict(state_dict, strict=False)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super(UnifiedModel, self).train(mode)
        self.teacher.eval()  # teacher always frozen/eval
        return self

    def forward(self, x):
        with torch.no_grad():
            t_features = self.teacher(x)
        s_features = self.student(x)

        i = self.concept_layer_idx
        diff = torch.nn.functional.normalize(t_features[i], dim=1) - \
            torch.nn.functional.normalize(s_features[i], dim=1)
        concept_logits = self.concept_net(diff)

        # concept bottleneck -> main task head (same as joint CBM End2EndModel)
        if self.use_relu:
            attr_outputs = [torch.relu(o) for o in concept_logits]
        elif self.use_sigmoid:
            attr_outputs = [torch.sigmoid(o) for o in concept_logits]
        else:
            attr_outputs = concept_logits
        main_logit = self.main_model(torch.cat(attr_outputs, dim=1))

        return t_features, s_features, concept_logits, main_logit


# feature extractor model for STFPM
class BackboneModelFeatures(nn.Module):
    def __init__(self, pretrained: bool = True, backbone: str = "resnet18"):
        super(BackboneModelFeatures, self).__init__()

        self.backbone = backbone

        # load pretrained model
        if backbone == "resnet18":
            base_model = models.resnet18(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(
                *list(base_model.children())[:-2]
            )  # remove last AvgPool and FC layer
        elif backbone == "mobilenet_v2":
            base_model = models.mobilenet_v2(pretrained=pretrained)
            self.feature_extractor = base_model.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # freeze layers by default
        if pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x):
        res = []
        if self.backbone == "resnet18":
            for name, module in self.feature_extractor._modules.items():
                x = module(x)
                if name in ["4", "5", "6"]:
                    res.append(x)

        elif self.backbone == "mobilenet_v2":
            for idx, module in enumerate(self.feature_extractor):
                x = module(x)
                if idx in [3, 8, 14]:
                    res.append(x)

        return res
