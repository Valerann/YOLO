import sys, time, json
sys.path.insert(0, ".")
import torch
import numpy as np
import hydra
from PIL import Image
from openvino import Core
from yolo import create_converter, PostProcess, AugmentationComposer
from yolo.config.config import NMSConfig
from yolo.model.yolo import create_model

img_path = sys.argv[1]
xml_path = sys.argv[2]
img_size = int(sys.argv[3])
model_name = sys.argv[4]
n_iters = int(sys.argv[5]) if len(sys.argv) > 5 else 8

with hydra.initialize(config_path="yolo/config", version_base=None):
    cfg = hydra.compose(config_name="config", overrides=[f"model={model_name}", "dataset=coco", f"image_size=[{img_size},{img_size}]"])

device = torch.device("cpu")

# converter/postprocess need a real (tiny, weight-free) model instance only for anchor/stride bookkeeping
dummy_model = create_model(cfg.model, weight_path=False, class_num=cfg.dataset.class_num).eval()

t0 = time.time()
core = Core()
compiled = core.compile_model(core.read_model(xml_path), "CPU")
t_load = time.time() - t0

converter = create_converter(cfg.model.name, dummy_model, cfg.model.anchor, cfg.image_size, device)
nms_cfg = NMSConfig(min_confidence=0.25, min_iou=0.45, max_bbox=300, agnostic=True)
post_process = PostProcess(converter, nms_cfg)

image = Image.open(img_path).convert("RGB")
transform = AugmentationComposer([], cfg.image_size)
frame, _, rev_tensor = transform(image, torch.zeros(0, 5))
frame_np = frame[None].numpy()
rev_tensor = rev_tensor[None]

VEHICLE_IDS = {1, 2, 3, 5, 7}

times = []
boxes = None
for i in range(n_iters):
    t0 = time.time()
    outputs = compiled([frame_np])
    predict = {"Main": [torch.from_numpy(outputs[compiled.output(j)]) for j in range(3)]}
    boxes = post_process(predict, rev_tensor=rev_tensor)[0]
    times.append(time.time() - t0)

vehicle_boxes = [row for row in boxes.tolist() if int(row[0]) in VEHICLE_IDS]

result = {
    "engine": "openvino",
    "model": model_name,
    "img_size": img_size,
    "load_s": t_load,
    "infer_s_all": times,
    "infer_s_mean": sum(times) / len(times),
    "infer_s_min": min(times),
    "detections_all_classes": len(boxes),
    "detections_vehicle": len(vehicle_boxes),
    "scores_vehicle": sorted([round(float(r[5]), 3) for r in vehicle_boxes], reverse=True),
}
print(json.dumps(result))
