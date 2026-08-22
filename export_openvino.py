import sys, torch, hydra
sys.path.insert(0, ".")
from yolo import create_model

weight_path = sys.argv[1]
model_name = sys.argv[2]
img_size = int(sys.argv[3])
out_stem = sys.argv[4]  # e.g. weights/v7x_640

with hydra.initialize(config_path="yolo/config", version_base=None):
    cfg = hydra.compose(config_name="config", overrides=[f"model={model_name}", "dataset=coco"])

model = create_model(cfg.model, weight_path=weight_path, class_num=cfg.dataset.class_num).eval()


class ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)["Main"]
        return out[0], out[1], out[2]


wrapper = ONNXWrapper(model)
dummy = torch.zeros(1, 3, img_size, img_size)

onnx_path = f"{out_stem}.onnx"
torch.onnx.export(
    wrapper, dummy, onnx_path,
    input_names=["images"], output_names=["p3", "p4", "p5"],
    opset_version=13, dynamic_axes=None,
)
print("wrote", onnx_path)

from openvino import convert_model, save_model

ov_model = convert_model(onnx_path)
xml_path = f"{out_stem}.xml"
save_model(ov_model, xml_path)
print("wrote", xml_path)
