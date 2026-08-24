"""RT-DETR, trainable — an Ultralytics-style workflow without the AGPL.

    from ovkit import RTDETR

    model = RTDETR("rtdetr_r50")                  # pretrained from the mirror
    model = RTDETR("best.pt")                     # or your own checkpoint
    model.train(data="data.yaml", epochs=100)     # YOLO-format labels
    model.val(data="data.yaml")                   # mAP50 / mAP50-95
    xml = model.export(half=True)                 # -> IR + labels.txt

    r = model("bus.jpg", conf=0.5)                # ovkit Results — print(r),
    print(r.found)                                # r.found, r.save(), ...

The network, loss, trainer and exporter are original Apache-2.0
implementations (no Ultralytics code or weights anywhere). Inference always
runs through ovkit's own engine, so a trained model answers exactly like
every other ovkit model. Training needs ``pip install "ovkit[train]"``;
inference needs nothing beyond ovkit itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.errors import ModelNotFoundError
from ..core.i18n import lang

_TORCH_HINT = (
    '학습에는 [train] 추가 설치가 필요해요:  pip install "ovkit[train]"'
    if lang() == "ko"
    else "training requires the [train] extra: pip install 'ovkit[train]'"
)

#: Hyphenated spellings accepted for the mirror's registry names.
_REGISTRY_NAMES = {
    "rtdetr-r18": "rtdetr_r18",
    "rtdetr-r34": "rtdetr_r34",
    "rtdetr-r50": "rtdetr_r50",
}


class RTDETR:
    """Train, validate, export, and run RT-DETR with one object."""

    def __init__(self, model: str | Path = "rtdetr_r50", device: str = "AUTO") -> None:
        self._src = str(model)
        self._device = device
        self.names: dict[int, str] = {}
        self.net = None  # torch net (lazy; only after .pt load or train())
        self.ckpt: dict | None = None
        self.variant: str | None = None
        self._runner = None  # ovkit Model over the IR (lazy)
        self._ir_path: Path | None = None

        p = Path(self._src)
        if p.suffix == ".pt":
            if not p.exists():
                raise FileNotFoundError(f"checkpoint not found: {p}")
            self._load_checkpoint(p)
        elif p.suffix in (".xml", ".onnx") and p.exists():
            self._ir_path = p
        elif p.suffix in (".xml", ".onnx", ".pt"):
            raise FileNotFoundError(f"'{model}' does not exist.")
        else:
            # a registry name — resolved (and downloaded) on first predict
            self.variant = self._src.rsplit("-", 1)[-1].rsplit("_", 1)[-1]

    # -- inference (always through ovkit's engine) --------------------------

    def __call__(self, source: Any, **kwargs: Any) -> Any:
        return self.predict(source, **kwargs)

    def predict(self, source: Any, **kwargs: Any) -> Any:
        """Run on any ovkit source; answers with ovkit ``Results``."""
        from ..core.model import Model, _immediate

        if self._runner is None:
            if self._ir_path is None and self.net is not None:
                self._ir_path = self.export(out_dir=self._export_dir())
            target = self._ir_path if self._ir_path is not None else self._registry_name()
            self._runner = Model.network(target, task="detect", device=self._device)
        run_opts = {k: kwargs.pop(k) for k in ("conf", "imgsz") if k in kwargs}
        return _immediate(self._runner, source, **run_opts)

    def _registry_name(self) -> str:
        from ..core.registry import resolve

        name = _REGISTRY_NAMES.get(self._src, self._src)
        if resolve(name) is None:
            known = ", ".join(sorted(_REGISTRY_NAMES.values()))
            raise ModelNotFoundError(
                f"'{self._src}' is not on the mirror yet. Use a mirrored name "
                f"({known} — availability depends on the mirror), pass a .pt/.xml "
                f"path, or train your own: RTDETR('rtdetr-r18').train(data=...)."
            )
        return name

    def _export_dir(self) -> Path:
        from ..core.constants import cache_root

        out = cache_root() / "trained" / f"rtdetr-{self.variant or 'r18'}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    # -- checkpoint ---------------------------------------------------------

    def _load_checkpoint(self, path: Path) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(_TORCH_HINT) from exc
        from .nn.rtdetr_net import RTDETRNet

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.ckpt = ckpt
        self.variant = ckpt["variant"]
        self.names = {int(k): v for k, v in ckpt.get("names", {}).items()}
        self.net = RTDETRNet(ckpt["variant"], ckpt["num_classes"], pretrained_backbone=False)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()

    # -- training -----------------------------------------------------------

    def train(
        self,
        data: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 8,
        lr: float = 1e-4,
        device: str | None = None,
        workers: int = 4,
        project: str = "runs",
        name: str = "train",
        val: bool = True,
        **kwargs: Any,
    ) -> Path:
        """Train on YOLO-format labels; returns the best checkpoint path."""
        try:
            from .trainer import Trainer
        except ImportError as exc:
            raise ImportError(_TORCH_HINT) from exc
        from .data.dataset import load_data_yaml
        from .nn.rtdetr_net import RTDETRNet
        from .validator import validate_torch

        cfg = load_data_yaml(data)
        self.names = cfg["names"]
        if self.net is None:
            self.net = RTDETRNet(self.variant or "r18", cfg["nc"])
        elif self.net.num_classes != cfg["nc"]:
            raise ValueError(
                f"checkpoint has {self.net.num_classes} classes but data.yaml has {cfg['nc']}"
            )
        trainer = Trainer(
            self.net,
            data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr=lr,
            device=device,
            workers=workers,
            project=project,
            name=name,
            **kwargs,
        )
        val_fn = (
            (lambda n: validate_torch(n, data, imgsz=imgsz, batch=batch, device=device)["map"])
            if val
            else None
        )
        best = trainer.train(val_fn=val_fn)
        self._load_checkpoint(best)
        self._runner = None  # re-export on next predict
        self._ir_path = None
        return best

    def val(
        self,
        data: str,
        imgsz: int = 640,
        batch: int = 8,
        conf: float = 0.01,
        device: str | None = None,
    ) -> dict:
        """COCO-style mAP on the data.yaml val split (needs torch weights)."""
        if self.net is None:
            raise RuntimeError("val() needs torch weights — load a .pt checkpoint first.")
        from .validator import validate_torch

        metrics = validate_torch(self.net, data, imgsz=imgsz, batch=batch, conf=conf, device=device)
        print(f"[ovkit] mAP50 {metrics['map50']:.4f}  mAP50-95 {metrics['map']:.4f}")
        return metrics

    def export(
        self, format: str = "openvino", imgsz: int = 640, half: bool = False, out_dir: str = "."
    ) -> Path:
        """Export the trained net to OpenVINO IR (plus labels.txt for ovkit)."""
        if format.lower() not in ("openvino", "onnx"):
            raise ValueError("supported formats: 'openvino', 'onnx'")
        if self.net is None:
            raise RuntimeError("export() needs torch weights — load a .pt checkpoint first.")
        from .exporter import export_openvino

        xml = export_openvino(
            self.net,
            self.names,
            imgsz=self.ckpt.get("imgsz", imgsz) if self.ckpt else imgsz,
            out_dir=out_dir,
            fname=f"rtdetr-{self.variant}",
            half=half,
        )
        return xml if format.lower() == "openvino" else xml.with_suffix(".onnx")

    def __repr__(self) -> str:
        return f"RTDETR({self._src!r}, device={self._device!r})"
