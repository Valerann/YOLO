import sys, time, json
sys.path.insert(0, ".")
import torch
import hydra
from PIL import Image
from yolo import create_model, create_converter, PostProcess, AugmentationComposer
from yolo.config.config import NMSConfig

img_path = sys.argv[1]
weight_path = sys.argv[2]
img_size = int(sys.argv[3])
model_name = sys.argv[4]
n_iters = int(sys.argv[5]) if len(sys.argv) > 5 else 8

print(f"torch threads: {torch.get_num_threads()}, torch version: {torch.__version__}", file=sys.stderr)

with hydra.initialize(config_path="yolo/config", version_base=None):
    cfg = hydra.compose(config_name="config", overrides=[f"model={model_name}", "dataset=coco", f"image_size=[{img_size},{img_size}]"])

device = torch.device("cpu")

t0 = time.time()
model = create_model(cfg.model, weight_path=weight_path, class_num=cfg.dataset.class_num)
model = model.to(device).eval()
converter = create_converter(cfg.model.name, model, cfg.model.anchor, cfg.image_size, device)
nms_cfg = NMSConfig(min_confidence=0.25, min_iou=0.45, max_bbox=300, agnostic=True)
post_process = PostProcess(converter, nms_cfg)
t_load = time.time() - t0

image = Image.open(img_path).convert("RGB")
transform = AugmentationComposer([], cfg.image_size)
frame, _, rev_tensor = transform(image, torch.zeros(0, 5))
frame = frame[None].to(device)
rev_tensor = rev_tensor[None].to(device)

VEHICLE_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorbike, bus, truck (0-indexed coco)

times = []
boxes = None
for i in range(n_iters):
    t0 = time.time()
    with torch.no_grad():
        predict = model(frame)
        boxes = post_process(predict, rev_tensor=rev_tensor)[0]
    times.append(time.time() - t0)

vehicle_boxes = [row for row in boxes.tolist() if int(row[0]) in VEHICLE_IDS]

result = {
    "engine": "mtl",
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
