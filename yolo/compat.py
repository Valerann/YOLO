"""Drop-in-signature compatibility shim for yolov7detect's `yolov7.load` /
`yolov7.detect_cpu` API, backed by this (MIT-licensed) engine.

Usage in calling code, e.g. sod-classifier.py:

    import yolo.compat as yolov7   # was: import yolov7

    self.model = yolov7.load(weights_path, autoshape=True)
    detections = yolov7.detect_cpu(self.model, image_data, conf_thres=...,
                                    iou_thres=..., nms_agnostic=True, rgb=True)

`weights_path` must point at a *converted* checkpoint (see ../../convert_weights.py)
in this engine's state-dict format, not the original yolov7x.pt directly.

Only the CPU, single-image, yolov7x code path actually exercised by
sod-classifier.py is implemented:
  - load(): device must be "cpu"; trace/half/hf_model are not implemented.
  - detect_cpu(): annotate=True is not implemented.

Output format matches the original exactly: an (N, 6) tensor of
[x1, y1, x2, y2, conf, cls] rows in the input image's own pixel coordinates
-- note this is a DIFFERENT column order than this engine's own PostProcess,
which returns [cls, x1, y1, x2, y2, conf]. Getting this reversed is exactly
the kind of silent-corruption bug this shim exists to prevent.
"""
import os
from dataclasses import dataclass
from typing import List, Optional

import hydra
import numpy as np
import torch
from PIL import Image

from yolo import create_converter, create_model
from yolo.config.config import NMSConfig
from yolo.tools.data_augmentation import AugmentationComposer
from yolo.utils.bounding_box_utils import bbox_nms

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

# Same order as the original yolov7/coco_labels.py -- index *is* the class id,
# so this must match exactly for valid_classes name lookups to be correct.
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Same default fallback as the original yolov7.helpers.detect_cpu.
DEFAULT_VALID_CLASSES = [
    "car", "bicycle", "motorbike", "bus", "truck", "person",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
]


@dataclass
class _LoadedModel:
    model: torch.nn.Module
    converter: object
    device: torch.device


def load(
    model_path,
    autoshape=True,
    device="cpu",
    trace=False,
    size=640,
    half=False,
    hf_model=False,
    model_variant="v7x",
):
    """Signature-compatible with yolov7.helpers.load_model.

    autoshape is accepted but is a no-op: the original's autoShape wrapper
    already short-circuits to plain inference for tensor input, which is the
    only input type detect_cpu ever passes it, so it never did anything for
    this call pattern anyway.
    """
    if device != "cpu":
        raise NotImplementedError("yolo.compat only supports device='cpu'")
    if trace or half or hf_model:
        raise NotImplementedError("trace/half/hf_model are not implemented in yolo.compat")

    with hydra.initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = hydra.compose(
            config_name="config",
            overrides=[f"model={model_variant}", "dataset=coco", f"image_size=[{size},{size}]"],
        )

    torch_device = torch.device("cpu")
    model = create_model(cfg.model, weight_path=model_path, class_num=cfg.dataset.class_num)
    model = model.to(torch_device).eval()
    converter = create_converter(cfg.model.name, model, cfg.model.anchor, cfg.image_size, torch_device)

    return _LoadedModel(model=model, converter=converter, device=torch_device)


def detect_cpu(
    yolov7_model: _LoadedModel,
    image: np.ndarray,
    rgb: bool = True,
    img_size: int = 640,
    letterbox_stride: int = 32,  # accepted for signature compatibility; unused (see note below)
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    nms_agnostic: bool = False,
    valid_classes: Optional[List[str]] = None,
    annotate: bool = False,
) -> torch.Tensor:
    """Signature- and output-format-compatible with yolov7.helpers.detect_cpu.

    Returns an (N, 6) tensor of [x1, y1, x2, y2, conf, cls] rows in `image`'s
    own pixel coordinates -- same column order as the original, NOT this
    engine's native [cls, x1, y1, x2, y2, conf] order.

    Note: letterbox_stride is accepted but unused. The original pads to the
    nearest multiple of `letterbox_stride` >= img_size; this engine's
    PadAndResize always pads to exactly img_size x img_size. Identical for
    any img_size that's already a stride multiple (e.g. 416, 640), which
    covers every value this is actually called with.
    """
    if annotate:
        raise NotImplementedError("annotate=True is not implemented in yolo.compat")

    if valid_classes is None:
        valid_classes = DEFAULT_VALID_CLASSES
    valid_class_ids = {COCO_CLASS_NAMES.index(name) for name in valid_classes}

    model = yolov7_model.model
    converter = yolov7_model.converter
    device = yolov7_model.device

    # Match the original's exact (unusual) semantics: rgb=True means the
    # array is ALREADY RGB-ordered (no channel swap); rgb=False means BGR.
    img_rgb_np = image if rgb else image[:, :, ::-1]
    # The original never goes through PIL -- it feeds the raw array straight
    # into cv2-based letterbox()/torch, so it's dtype-agnostic (uint8 or
    # float32 alike). PIL's fromarray() rejects 3-channel float arrays
    # outright, so normalize to uint8 first to accept the same inputs.
    if img_rgb_np.dtype != np.uint8:
        img_rgb_np = np.clip(img_rgb_np, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(np.ascontiguousarray(img_rgb_np))

    transform = AugmentationComposer([], [img_size, img_size])
    frame, _, rev_tensor = transform(pil_image, torch.zeros(0, 5))
    frame = frame[None].to(device)
    rev_tensor = rev_tensor[None].to(device)

    converter.update((img_size, img_size))
    with torch.no_grad():
        predict = model(frame)
        pred_class, _, pred_bbox, pred_conf = converter(predict["Main"])
        # Un-letterbox: model-space (padded img_size x img_size) -> original pixel coords.
        pred_bbox = (pred_bbox - rev_tensor[:, None, 1:]) / rev_tensor[:, 0:1, None]

        # Mask excluded classes' logits to -inf BEFORE NMS, matching the
        # original's pre-NMS class filter (`classes=` in non_max_suppression).
        # Doing this after NMS instead would give different results whenever
        # nms_agnostic=True, since an excluded-class box could otherwise have
        # suppressed a wanted-class box during agnostic suppression.
        mask = torch.zeros(pred_class.shape[-1])
        for cid in range(pred_class.shape[-1]):
            if cid not in valid_class_ids:
                mask[cid] = -1e4
        pred_class = pred_class + mask

        nms_cfg = NMSConfig(min_confidence=conf_thres, min_iou=iou_thres, max_bbox=300, agnostic=nms_agnostic)
        boxes = bbox_nms(pred_class, pred_bbox, nms_cfg, confidence=pred_conf)[0]

    if boxes.numel() == 0:
        return torch.zeros((0, 6))
    # Reorder [cls, x1, y1, x2, y2, conf] -> [x1, y1, x2, y2, conf, cls].
    return torch.cat([boxes[:, 1:5], boxes[:, 5:6], boxes[:, 0:1]], dim=1)
