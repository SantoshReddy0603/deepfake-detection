"""Shared model and preprocessing contract. Inference never downloads initialization."""
import copy
import hashlib
import math
from pathlib import Path
import torch
from torch import nn
from torchvision.models import (efficientnet_b0, convnext_tiny, convnext_small, resnet50, swin_t,
                                efficientnet_v2_s, maxvit_t,
                                EfficientNet_B0_Weights, ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights,
                                ResNet50_Weights, Swin_T_Weights, EfficientNet_V2_S_Weights, MaxVit_T_Weights)

NORMALIZATION = {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
LEGACY_ARCHITECTURES = ("efficientnet_b0", "convnext_tiny")
TIMM_ARCHITECTURES = {
    "vit_base_in21k": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "deit3_base_in22k": "deit3_base_patch16_224.fb_in22k_ft_in1k",
    "convnext_small_in22k": "convnext_small.fb_in22k_ft_in1k",
    "convnextv2_tiny_in22k": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "swin_small_in22k": "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
}
ARCHITECTURES = (LEGACY_ARCHITECTURES + ("convnext_small", "resnet50", "swin_t", "efficientnet_v2_s", "maxvit_t")
                 + tuple(TIMM_ARCHITECTURES))
TTA_VIEWS = ["identity", "hflip"]


def build_model(architecture, pretrained=False, dropout=0.2, initialization_path=None):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if initialization_path is not None and pretrained:
        raise ValueError("Choose an attached initialization file or a download, never both")
    if initialization_path is not None and architecture not in LEGACY_ARCHITECTURES:
        raise ValueError("Attached initialization files exist only for ConvNeXt-Tiny and EfficientNet-B0")
    if architecture in TIMM_ARCHITECTURES:
        try:
            import timm
        except ImportError as error:
            raise ImportError(f"{architecture} needs the timm package: pip install -r requirements.txt") from error
        return timm.create_model(TIMM_ARCHITECTURES[architecture], pretrained=pretrained, num_classes=1,
                                 drop_rate=dropout)
    if architecture == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[1].in_features, 1))
    elif architecture == "convnext_small":
        model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[2].in_features, 1))
    elif architecture == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 1))
    elif architecture == "swin_t":
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        model.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.head.in_features, 1))
    elif architecture == "efficientnet_v2_s":
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[1].in_features, 1))
    elif architecture == "maxvit_t":
        model = maxvit_t(weights=MaxVit_T_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[5] = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[5].in_features, 1))
    else:
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        model.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[2].in_features, 1))
    if initialization_path is not None:
        path = Path(initialization_path)
        expected = "7f5810bc" if architecture == "efficientnet_b0" else "983f1562"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not digest.startswith(expected):
            raise ValueError("Attached ImageNet initialization checksum mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        old_head = "classifier.1" if architecture == "efficientnet_b0" else "classifier.2"
        if state[old_head + ".weight"].shape[0] != 1000:
            raise ValueError("Expected the permitted ImageNet 1000-class initialization")
        state = {k: v for k, v in state.items() if k not in {old_head + ".weight", old_head + ".bias"}}
        new_head = "classifier.1" if architecture == "efficientnet_b0" else "classifier.2.1"
        missing, unexpected = model.load_state_dict(state, strict=False)
        if set(missing) != {new_head + ".weight", new_head + ".bias"} or unexpected:
            raise ValueError("Initialization backbone does not match the requested architecture")
    return model


def head_parameters(model):
    """Parameters of the replaced binary head, whatever the backbone family."""
    if hasattr(model, "get_classifier") and hasattr(model, "default_cfg"):
        return list(model.get_classifier().parameters())
    for name in ("classifier", "fc", "head"):
        if hasattr(model, name):
            return list(getattr(model, name).parameters())
    raise ValueError("Model has no recognised classification head")


PRETRAINED_WEIGHTS = {"efficientnet_b0": EfficientNet_B0_Weights.IMAGENET1K_V1,
                      "convnext_tiny": ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                      "convnext_small": ConvNeXt_Small_Weights.IMAGENET1K_V1,
                      "resnet50": ResNet50_Weights.IMAGENET1K_V2,
                      "swin_t": Swin_T_Weights.IMAGENET1K_V1,
                      "efficientnet_v2_s": EfficientNet_V2_S_Weights.IMAGENET1K_V1,
                      "maxvit_t": MaxVit_T_Weights.IMAGENET1K_V1}


def initialization_provenance(architecture, path=None):
    weights = PRETRAINED_WEIGHTS[architecture]
    filename = weights.url.rsplit("/", 1)[-1]
    attached = path is not None
    path = Path(path) if attached else Path(torch.hub.get_dir()) / "checkpoints" / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {"source": "torchvision ImageNet-1K classification initialization", "enum": str(weights),
            "url": weights.url, "sha256": digest,
            "loaded_from": "attached persistent input" if attached else "torch hub initialization cache",
            "terms_reference": "https://github.com/pytorch/vision#pre-trained-model-license",
            "fine_tuning_source": "provided official train partition only"}


def normalize(x, device):
    x = x.to(device, non_blocking=True).float().div_(255.)
    mean = x.new_tensor(NORMALIZATION["mean"]).view(1, 3, 1, 1)
    std = x.new_tensor(NORMALIZATION["std"]).view(1, 3, 1, 1)
    return (x - mean) / std


class EMA:
    def __init__(self, model, decay=0.99):
        if not 0.99 <= decay <= 0.995:
            raise ValueError("EMA target decay must be between 0.99 and 0.995")
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, live):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for ema, current in zip(self.model.parameters(), live.parameters()):
            ema.lerp_(current.detach(), 1 - decay)
        # Preserve training-only BN statistics rather than recalibrating on validation/test.
        for ema, current in zip(self.model.buffers(), live.buffers()):
            ema.copy_(current)
        return decay

    def state_dict(self):
        return {"model": self.model.state_dict(), "updates": self.updates, "decay": self.decay}

    def load_state_dict(self, state):
        if state["decay"] != self.decay:
            raise ValueError("EMA decay changed on resume")
        self.model.load_state_dict(state["model"], strict=True)
        self.updates = state["updates"]


class ProbabilityEnsemble(nn.ModuleList):
    outputs_probabilities = True

    def __init__(self, members):
        members = list(members)
        if len(members) != 2 or any(not isinstance(model, nn.Module) or getattr(model, "outputs_probabilities", False)
                                    for model in members):
            raise ValueError("Probability ensemble requires exactly two logit-output models")
        super().__init__(members)

    def forward(self, x):
        first = self[0](x).reshape(-1).float().sigmoid()
        second = self[1](x).reshape(-1).float().sigmoid()
        return .5 * (first + second)


class LogitEnsemble(nn.ModuleList):
    """Any number of members; mean of member logits over the original and mirrored view."""
    outputs_probabilities = True

    def __init__(self, members, tta=True):
        members = list(members)
        if not members or any(getattr(model, "outputs_probabilities", False) for model in members):
            raise ValueError("Logit ensemble requires logit-output members")
        super().__init__(members)
        self.tta = bool(tta)

    def forward(self, x):
        views = [x, x.flip(3)] if self.tta else [x]
        total = 0
        for model in self:
            for view in views:
                total = total + model(view).reshape(-1).float()
        return (total / (len(self) * len(views))).sigmoid()


def _validate_state(state):
    if not isinstance(state, dict) or not state or any(not isinstance(key, str) or not key for key in state):
        raise ValueError("Ensemble member state must contain named tensors")
    floating = set()
    for value in state.values():
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.device.type != "cpu":
            raise ValueError("Ensemble member states must be dense CPU tensors")
        if value.is_floating_point():
            if value.dtype not in (torch.float16, torch.float32) or not torch.isfinite(value).all():
                raise ValueError("Ensemble floating tensors must be finite FP16 or FP32")
            floating.add(value.dtype)
        elif value.dtype != torch.int64:
            raise ValueError("Unsupported ensemble buffer dtype")
    if len(floating) != 1:
        raise ValueError("Ensemble member floating tensors must use one storage precision")
    return next(iter(floating))


def _validate_ensemble(payload):
    expected = {"format_version", "architecture", "aggregation", "members", "preprocessing", "label_mapping", "threshold", "purpose"}
    if (not isinstance(payload, dict) or set(payload) != expected or type(payload.get("format_version")) is not int or payload.get("format_version") != 2
            or payload.get("architecture") != "probability_ensemble" or payload.get("aggregation") != "uniform_probability"
            or payload.get("preprocessing") != NORMALIZATION
            or payload.get("label_mapping") != {"0": "real", "1": "fake"}
            or isinstance(payload.get("threshold"), bool) or not isinstance(payload.get("threshold"), (int, float))
            or not math.isfinite(payload["threshold"]) or not 0 <= payload["threshold"] <= 1
            or not isinstance(payload.get("purpose"), str) or not payload["purpose"]):
        raise ValueError("Incompatible probability-ensemble export contract")
    members = payload["members"]
    if not isinstance(members, list) or len(members) != 2:
        raise ValueError("Probability ensemble requires exactly two members")
    architectures, precisions = [], set()
    for member in members:
        if (not isinstance(member, dict) or set(member) != {"architecture", "dropout", "state_dict"}
                or member["architecture"] not in LEGACY_ARCHITECTURES
                or isinstance(member["dropout"], bool) or not isinstance(member["dropout"], (int, float))
                or not math.isfinite(member["dropout"]) or not 0 <= member["dropout"] <= 1):
            raise ValueError("Incompatible ensemble member metadata")
        architectures.append(member["architecture"])
        precisions.add(_validate_state(member["state_dict"]))
    if set(architectures) != set(LEGACY_ARCHITECTURES) or len(precisions) != 1:
        raise ValueError("Ensemble requires distinct ConvNeXt-Tiny/B0 members with common storage precision")


def export_ensemble(memberpayloads, path, threshold, purpose, half=True):
    if not isinstance(memberpayloads, (list, tuple)) or len(memberpayloads) != 2 or type(half) is not bool:
        raise ValueError("Export requires exactly two single-model payloads and a boolean half flag")
    members = []
    keys = {"format_version", "architecture", "dropout", "preprocessing", "label_mapping", "threshold", "purpose", "state_dict"}
    for member in memberpayloads:
        if (not isinstance(member, dict) or set(member) != keys or type(member["format_version"]) is not int or member["format_version"] != 1
                or member["architecture"] not in LEGACY_ARCHITECTURES or member["preprocessing"] != NORMALIZATION
                or member["label_mapping"] != {"0": "real", "1": "fake"}
                or isinstance(member["threshold"], bool) or not isinstance(member["threshold"], (int, float))
                or not math.isfinite(member["threshold"]) or not 0 <= member["threshold"] <= 1
                or not isinstance(member["purpose"], str) or not member["purpose"]):
            raise ValueError("Expected lean format1 inference members with compatible metadata")
        _validate_state(member["state_dict"])
        state = {key: value.detach().to(dtype=(torch.float16 if half else torch.float32)).clone()
                 if value.is_floating_point() else value.detach().clone() for key, value in member["state_dict"].items()}
        members.append({"architecture": member["architecture"], "dropout": member["dropout"], "state_dict": state})
    payload = {"format_version": 2, "architecture": "probability_ensemble", "aggregation": "uniform_probability",
               "members": members, "preprocessing": NORMALIZATION, "label_mapping": {"0": "real", "1": "fake"},
               "threshold": threshold, "purpose": purpose}
    _validate_ensemble(payload)
    for member in members:
        model = build_model(member["architecture"], pretrained=False, dropout=member["dropout"])
        model.load_state_dict(member["state_dict"], strict=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


V3_KEYS = {"format_version", "architecture", "aggregation", "tta", "members", "preprocessing",
           "label_mapping", "threshold", "purpose"}


def _validate_logit_ensemble(payload):
    if (not isinstance(payload, dict) or set(payload) != V3_KEYS or payload["format_version"] != 3
            or payload["architecture"] != "logit_ensemble" or payload["aggregation"] != "mean_logit"
            or payload["tta"] not in (TTA_VIEWS, ["identity"])
            or payload["preprocessing"] != NORMALIZATION or payload["label_mapping"] != {"0": "real", "1": "fake"}
            or isinstance(payload["threshold"], bool) or not isinstance(payload["threshold"], (int, float))
            or not math.isfinite(payload["threshold"]) or not 0 <= payload["threshold"] <= 1
            or not isinstance(payload["purpose"], str) or not payload["purpose"]):
        raise ValueError("Incompatible logit-ensemble export contract")
    members = payload["members"]
    if not isinstance(members, list) or not members:
        raise ValueError("Logit ensemble needs at least one member")
    precisions = set()
    for member in members:
        if (not isinstance(member, dict) or set(member) != {"architecture", "dropout", "state_dict"}
                or member["architecture"] not in ARCHITECTURES
                or isinstance(member["dropout"], bool) or not isinstance(member["dropout"], (int, float))
                or not 0 <= member["dropout"] <= 1):
            raise ValueError("Incompatible logit-ensemble member metadata")
        precisions.add(_validate_state(member["state_dict"]))
    if len(precisions) != 1:
        raise ValueError("Logit-ensemble members must share one storage precision")


def export_logit_ensemble(members, path, threshold, purpose, tta=True, half=True):
    """members: list of (architecture, dropout, state_dict)."""
    packed = []
    for architecture, dropout, state in members:
        state = {key: value.detach().cpu().to(torch.float16 if half else torch.float32).clone()
                 if value.is_floating_point() else value.detach().cpu().clone() for key, value in state.items()}
        model = build_model(architecture, pretrained=False, dropout=dropout)
        model.load_state_dict(state, strict=True)
        packed.append({"architecture": architecture, "dropout": float(dropout), "state_dict": state})
    payload = {"format_version": 3, "architecture": "logit_ensemble", "aggregation": "mean_logit",
               "tta": list(TTA_VIEWS) if tta else ["identity"], "members": packed,
               "preprocessing": NORMALIZATION, "label_mapping": {"0": "real", "1": "fake"},
               "threshold": float(threshold), "purpose": purpose}
    _validate_logit_ensemble(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def export_model(model, architecture, path, threshold=0.5, purpose="stage_a_validation", half=False):
    state = {k: v.detach().cpu().half() if half and v.is_floating_point() else v.detach().cpu()
             for k, v in model.state_dict().items()}
    payload = {"format_version": 1, "architecture": architecture, "dropout": 0.2,
               "preprocessing": NORMALIZATION, "label_mapping": {"0": "real", "1": "fake"},
               "threshold": float(threshold), "purpose": purpose, "state_dict": state}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_export(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") == 3:
        _validate_logit_ensemble(payload)
        members = []
        for member in payload["members"]:
            model = build_model(member["architecture"], pretrained=False, dropout=member["dropout"])
            model.load_state_dict({k: v.float() if v.is_floating_point() else v
                                   for k, v in member["state_dict"].items()}, strict=True)
            members.append(model)
        return LogitEnsemble(members, tta=payload["tta"] == TTA_VIEWS).to(device).eval(), payload
    if payload.get("format_version") == 2:
        _validate_ensemble(payload)
        members = []
        for member in payload["members"]:
            model = build_model(member["architecture"], pretrained=False, dropout=member["dropout"])
            model.load_state_dict(member["state_dict"], strict=True)
            members.append(model)
        return ProbabilityEnsemble(members).to(device).eval(), payload
    if (payload.get("format_version") != 1 or payload.get("preprocessing") != NORMALIZATION or
            payload.get("label_mapping") != {"0": "real", "1": "fake"}):
        raise ValueError("Incompatible model/preprocessing/label contract")
    model = build_model(payload["architecture"], pretrained=False, dropout=payload["dropout"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload
