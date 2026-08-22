import sys, time
sys.path.insert(0, ".")
import hydra
from omegaconf import OmegaConf
import torch
from PIL import Image

from yolo import create_model, create_converter, PostProcess, draw_bboxes, AugmentationComposer
from yolo.config.config import NMSConfig

img_path = sys.argv[1]
weight_path = sys.argv[2]
out_path = sys.argv[3]
model_name = sys.argv[4] if len(sys.argv) > 4 else "v7"
img_size = int(sys.argv[5]) if len(sys.argv) > 5 else 640

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

t0 = time.time()
with torch.no_grad():
    predict = model(frame)
    pred_bbox = post_process(predict, rev_tensor=rev_tensor)
t_infer = time.time() - t0

out_img = draw_bboxes(image, pred_bbox, idx2label=cfg.dataset.class_list)
out_img.save(out_path)

boxes = pred_bbox[0] if isinstance(pred_bbox, list) else pred_bbox
print(f"load={t_load:.2f}s infer={t_infer:.2f}s detections={len(boxes)}")
class_list = cfg.dataset.class_list
for row in boxes.tolist():
    cls_id, x1, y1, x2, y2, conf = row
    print(f"  cls={int(cls_id)}({class_list[int(cls_id)]}) score={conf:.3f} box={[round(v,1) for v in (x1,y1,x2,y2)]}")
