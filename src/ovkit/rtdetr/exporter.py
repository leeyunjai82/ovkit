# Apache-2.0
"""Export: RTDETRNet -> ONNX -> OpenVINO IR (static shape, FP16 optional)."""

import json
from pathlib import Path


def export_openvino(net, names, imgsz=640, out_dir=".", fname="rtdetr", half=False):
    """Returns path to the generated .xml. Writes <fname>.names.json alongside."""
    import openvino as ov
    import torch

    from .nn.rtdetr_net import DeployWrapper

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{fname}.onnx"
    xml_path = out_dir / f"{fname}.xml"

    wrapper = DeployWrapper(net).eval().cpu()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["images"],
        output_names=["boxes", "scores"],
        opset_version=17,
        dynamo=False,
    )

    model = ov.convert_model(str(onnx_path))
    ov.save_model(model, str(xml_path), compress_to_fp16=half)

    sidecar = out_dir / f"{fname}.names.json"
    table = {int(k): v for k, v in (names or {}).items()}
    sidecar.write_text(json.dumps(table, ensure_ascii=False))
    # labels.txt beside the IR is how ovkit's Model discovers class names, so a
    # freshly trained model answers "my-class 0.91" with zero extra wiring.
    if table:
        lines = [table.get(i, f"class_{i}") for i in range(max(table) + 1)]
        (out_dir / "labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ovkit] exported: {xml_path}")
    return xml_path
