# Apache-2.0
"""Training loop (PyTorch). Kept deliberately small and readable."""

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.dataset import DetDataset, load_data_yaml
from .utils.loss import SetCriterion


class Trainer:
    def __init__(
        self,
        net,
        data,
        epochs=100,
        imgsz=640,
        batch=8,
        lr=1e-4,
        lr_backbone_mult=0.1,
        weight_decay=1e-4,
        warmup_epochs=1,
        device=None,
        workers=4,
        project="runs",
        name="train",
        amp=True,
    ):
        self.net = net
        self.data_yaml = data
        self.cfg = load_data_yaml(data)
        self.epochs, self.imgsz, self.batch = epochs, imgsz, batch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.workers = workers
        self.amp = amp and self.device.type == "cuda"

        self.save_dir = self._unique_dir(Path(project) / name)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "weights").mkdir(exist_ok=True)

        backbone_params, other_params = [], []
        for n, p in net.named_parameters():
            (backbone_params if n.startswith("backbone") else other_params).append(p)
        self.opt = torch.optim.AdamW(
            [
                {"params": other_params, "lr": lr},
                {"params": backbone_params, "lr": lr * lr_backbone_mult},
            ],
            lr=lr,
            weight_decay=weight_decay,
        )
        self.criterion = SetCriterion(net.num_classes)
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [g["lr"] for g in self.opt.param_groups]
        self.scaler = torch.amp.GradScaler(enabled=self.amp)

    @staticmethod
    def _unique_dir(p: Path) -> Path:
        if not p.exists():
            return p
        i = 2
        while (q := p.with_name(f"{p.name}{i}")).exists():
            i += 1
        return q

    def _set_lr(self, epoch, step, steps_per_epoch):
        t = epoch + step / max(steps_per_epoch, 1)
        if t < self.warmup_epochs:
            scale = t / max(self.warmup_epochs, 1e-8)
        else:
            prog = (t - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1e-8)
            scale = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
            scale = max(scale, 0.01)
        for g, base in zip(self.opt.param_groups, self.base_lrs, strict=False):
            g["lr"] = base * scale

    def train(self, val_fn=None):
        ds = DetDataset(self.data_yaml, "train", self.imgsz, augment=True)
        dl = DataLoader(
            ds,
            batch_size=self.batch,
            shuffle=True,
            drop_last=len(ds) > self.batch,
            num_workers=self.workers,
            collate_fn=DetDataset.collate,
            pin_memory=self.device.type == "cuda",
        )
        net = self.net.to(self.device)
        best = -1.0
        print(
            f"[ovkit] training on {self.device}, {len(ds)} images, "
            f"{self.epochs} epochs -> {self.save_dir}"
        )

        for epoch in range(self.epochs):
            net.train()
            t0, running = time.time(), 0.0
            for step, (imgs, targets) in enumerate(dl):
                self._set_lr(epoch, step, len(dl))
                imgs = imgs.to(self.device, non_blocking=True)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                with torch.amp.autocast(self.device.type, enabled=self.amp):
                    out = net(imgs)
                    loss, logs = self.criterion(out, targets)

                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                self.scaler.step(self.opt)
                self.scaler.update()
                running += float(loss.detach())

            avg = running / max(len(dl), 1)
            msg = (
                f"epoch {epoch + 1}/{self.epochs}  loss {avg:.3f}  "
                f"vfl {logs['vfl']:.3f} l1 {logs['l1']:.3f} giou {logs['giou']:.3f}  "
                f"{time.time() - t0:.1f}s"
            )

            metric = None
            if val_fn is not None:
                metric = val_fn(net)
                msg += f"  mAP50-95 {metric:.4f}"
            print("[ovkit] " + msg)

            self._save(net, epoch, "last.pt")
            if metric is None or metric > best:
                best = metric if metric is not None else best
                self._save(net, epoch, "best.pt")

        return self.save_dir / "weights" / "best.pt"

    def _save(self, net, epoch, fname):
        torch.save(
            {
                "model": net.state_dict(),
                "variant": net.variant,
                "num_classes": net.num_classes,
                "names": self.cfg["names"],
                "epoch": epoch,
                "imgsz": self.imgsz,
            },
            self.save_dir / "weights" / fname,
        )
