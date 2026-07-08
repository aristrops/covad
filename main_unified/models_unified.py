"""Model definitions for the unified STFPM + concept pipeline.

Reuses the original repo's backbone pieces (``BackboneModelFeatures``, ``FC``,
``MLP`` from ``models/model_backbones.py``); adds the unified single-branch model
and its factory here so the whole unified pipeline lives under ``main_unified/``.
"""
import os

import torch
import torch.nn as nn
import torchvision.models as models

from models.model_backbones import BackboneModelFeatures, FC, MLP


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

    def forward(self, x, injects=None):
        # injects: optional {element_index: diff_tensor}; the diff is added to the
        # running activation right after that block (unified++). Shapes must match
        # the block output (they do, because the block IS the corresponding STFPM
        # comparison layer of the same mobilenet).
        if injects is None:
            x = self.features(x)
        else:
            for i, block in enumerate(self.features):
                x = block(x)
                if i in injects:
                    x = x + injects[i]
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

    # unified++: map STFPM comparison layer -> element index inside the truncated
    # net (features[4:]) whose output has matching channels/resolution.
    # comparison feature indices [3, 8, 14] -> truncated element (idx-4): 8->4, 14->10.
    INJECT_MAP = {1: 4, 2: 10}  # {t/s feature-list index: truncated element index}

    def __init__(self, num_attr: int, expand_dim: int = 0,
                 backbone: str = "mobilenet_v2", concept_layer_idx: int = 0,
                 use_relu: bool = True, use_sigmoid: bool = False,
                 inject_diffs: bool = False):
        super(UnifiedModel, self).__init__()
        if backbone != "mobilenet_v2":
            raise ValueError("UnifiedModel only supports mobilenet_v2")

        self.concept_layer_idx = concept_layer_idx
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid
        self.inject_diffs = inject_diffs  # unified++ toggle

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

    def _student_norm(self, feat, mask_b):
        # channel-wise normalized student feature; if mask_b is given, per-sample
        # gate the gradient so masked rows (mask_b==0) are detached (value kept,
        # gradient blocked) — used to stop anomalous samples from updating the
        # student through the concept path (unified++masked ablation).
        sn = torch.nn.functional.normalize(feat, dim=1)
        if mask_b is not None:
            sn = sn * mask_b + sn.detach() * (1 - mask_b)
        return sn

    def forward(self, x, student_grad_mask=None):
        with torch.no_grad():
            t_features = self.teacher(x)
        s_features = self.student(x)

        mask_b = None
        if student_grad_mask is not None:
            mask_b = student_grad_mask.view(-1, 1, 1, 1).float()

        i = self.concept_layer_idx
        diff = torch.nn.functional.normalize(t_features[i], dim=1) - \
            self._student_norm(s_features[i], mask_b)

        injects = None
        if self.inject_diffs:
            # add the normalized diffs of the deeper comparison layers to the
            # matching internal activations of the truncated concept net.
            injects = {}
            for feat_idx, elem_idx in self.INJECT_MAP.items():
                injects[elem_idx] = (
                    torch.nn.functional.normalize(t_features[feat_idx], dim=1)
                    - self._student_norm(s_features[feat_idx], mask_b)
                )
        concept_logits = self.concept_net(diff, injects=injects)

        # concept bottleneck -> main task head (same as joint CBM End2EndModel)
        if self.use_relu:
            attr_outputs = [torch.relu(o) for o in concept_logits]
        elif self.use_sigmoid:
            attr_outputs = [torch.sigmoid(o) for o in concept_logits]
        else:
            attr_outputs = concept_logits
        main_logit = self.main_model(torch.cat(attr_outputs, dim=1))

        return t_features, s_features, concept_logits, main_logit


def unified_model(num_attr, expand_dim=0, backbone="mobilenet_v2",
                  teacher_path=None, model_state_dict=None, mode="train",
                  inject_diffs=False):
    """Build the unified STFPM+concept model.

    inject_diffs=True -> unified++: also add the deeper feature differences to the
    matching internal activations of the truncated concept net.

    train: optionally load teacher feature_extractor weights from ``teacher_path``
    (falls back to ImageNet weights if the file is missing / keys mismatch).
    test: load the full trained ``model_state_dict``.
    """
    model = UnifiedModel(num_attr=num_attr, expand_dim=expand_dim, backbone=backbone,
                         inject_diffs=inject_diffs)

    if mode == "train":
        if teacher_path is not None and os.path.exists(teacher_path):
            raw = torch.load(teacher_path, map_location="cpu")
            # keep only feature_extractor.* keys (BackboneModel / BackboneModelFeatures layout)
            teacher_sd = {k: v for k, v in raw.items() if k.startswith("feature_extractor.")}
            if not teacher_sd:
                # maybe wrapped under first_model.*
                teacher_sd = {
                    k.replace("first_model.", ""): v
                    for k, v in raw.items()
                    if k.startswith("first_model.feature_extractor.")
                }
            if teacher_sd:
                model.load_teacher(teacher_sd)
                print(f"Loaded fine-tuned teacher feature_extractor from {teacher_path}")
            else:
                print(f"No feature_extractor.* keys in {teacher_path}; using ImageNet teacher.")
        else:
            print("Using ImageNet-pretrained teacher (no teacher_path provided/found).")

    elif mode == "test":
        model.load_state_dict(model_state_dict, strict=False)

    return model
