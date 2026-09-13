"""``ovkit`` CLI: ``gui``, ``run``, ``pull``, ``train``, ``val``, ``export``, ``list``, ``info``, ``download``, ``devices``."""

from __future__ import annotations

import argparse
import contextlib
import sys

from . import __version__
from .core.convert import to_ir
from .core.download import fetch
from .core.errors import OVKitError
from .core.registry import list_models, resolve


def _cmd_list(args: argparse.Namespace) -> int:
    """What you can run, named the way you would type it.

    A beginner deciding what to try needs the *names they can use*, not the
    model zoo's catalogue numbers: ``detect``, not ``rtdetr_r50``. The raw
    names are still there under ``--all``.
    """
    from .core.i18n import KO_CAPS
    from .pipelines import list_pipelines

    show_all = bool(getattr(args, "all", False))
    korean = {target: ko for ko, target in KO_CAPS.items()}

    caps = list_pipelines()
    print(f"\ncapabilities — several models chained into one answer ({len(caps)}):")
    for name, desc in caps.items():
        print(f"  {name:21s} {korean.get(name, ''):10s} {_clip(desc, 52)}")

    names = list_models(tier=None if show_all else "core")
    friendly: list[tuple[str, str, str]] = []
    raw: list[tuple[str, str, str]] = []
    for name in names:
        entry = resolve(name)
        if entry is None:
            continue
        # A capability of the same name shadows the alias — Model("gaze") is
        # the pipeline — so listing the alias too would offer a dead end.
        if entry.name != name and name in caps:
            continue
        row = (name, korean.get(name, ""), _clip(entry.description or str(entry.task), 52))
        (friendly if entry.name != name else raw).append(row)

    if friendly:
        print(f"\nmodels — one network each ({len(friendly)}):")
        for name, ko, desc in friendly:
            print(f"  {name:21s} {ko:10s} {desc}")
    if show_all and raw:
        print(f"\nby their registry name ({len(raw)}):")
        for name, _ko, desc in raw:
            print(f"  {name:46s} {desc}")

    from .hub import pulled_models

    pulled = pulled_models()
    if pulled:
        print(f"\npulled from Hugging Face ({len(pulled)}):")
        for name in pulled:
            print(f"  {name}")

    if not show_all:
        hidden = len(list_models(tier=None)) - len(list_models())
        print(
            f'\nRun one:  Model("detect", "photo.jpg")   ·   ovkit run detect photo.jpg'
            f"\nEverything else ({hidden} archived zoo entries, plus registry names):"
            f"  ovkit list --all"
            f"\nAnything on the Hub:  ovkit pull <owner/model>"
        )
    return 0


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _cmd_capabilities(_: argparse.Namespace) -> int:
    """What ``Model(name)`` can answer beyond a single network."""
    from .pipelines import ALIASES, list_pipelines

    print("capabilities — Model(name) chains the models the answer needs:\n")
    for name, description in list_pipelines().items():
        print(f"  {name:16s} {description}")
    print("\naliases:")
    for alias, target in sorted(ALIASES.items()):
        print(f"  {alias:16s} -> {target}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    entry = resolve(args.name)
    if entry is None:
        print(f"'{args.name}' is not a registered model.", file=sys.stderr)
        return 1
    print(f"name       : {entry.name}")
    print(f"task       : {entry.task}")
    if entry.description:
        print(f"description: {entry.description}")
    print(f"license    : {entry.license}")
    print(f"source     : {entry.src} ({entry.repo or entry.url})")
    print(f"precision  : {entry.precision}")
    if entry.filename:
        print(f"filename   : {entry.filename}")
    if entry.imgsz:
        print(f"imgsz      : {entry.imgsz}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    entry = resolve(args.name)
    if entry is None:
        print(f"'{args.name}' is not a registered model.", file=sys.stderr)
        return 1
    print(f"Fetching {entry.name} from {entry.src}...")
    source = fetch(entry)
    print(f"Downloaded source: {source}")
    if not args.no_convert:
        ir = to_ir(source, entry.name, entry.precision)
        print(f"IR ready: {ir}")
    return 0


def _cmd_devices(_: argparse.Namespace) -> int:
    from .core.backend import available_devices

    for dev in available_devices():
        print(dev)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """One-shot inference from the shell: ``ovkit run detect img.jpg``."""
    from pathlib import Path

    from .core.model import Model

    model = Model(args.model, device=args.device)
    results = model.predict(args.source, conf=args.conf)
    if not isinstance(results, list):  # raw (.npy/.wav) input -> tensor dict
        for name, arr in results.items():
            print(f"{name}: shape={tuple(arr.shape)} dtype={arr.dtype}")
        return 0

    for r in results:
        print(r.summary())
        if r.boxes is not None:
            for x1, y1, x2, y2, c, cl in r.boxes.data[:20]:
                print(
                    f"  {r.name_for(int(cl)):16s} {c:.2f} [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"
                )

    save = args.save
    if save is None and results and Path(str(args.source)).is_file():
        save = f"{Path(str(args.source)).stem}_out.jpg"
    if save and results:
        results[0].save(save)
        print(f"saved -> {save}")
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    """Open the desktop window: ``ovkit gui``."""
    from .gui import main as gui_main

    return gui_main(device=args.device, camera=args.camera)


def _cmd_pull(args: argparse.Namespace) -> int:
    """Convert a Hugging Face model to IR: ``ovkit pull google/vit-base...``."""
    from .hub import pull

    pull(args.model_id, task=args.task)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Train RT-DETR on YOLO-format data: ``ovkit train --data data.yaml``."""
    from ovkit import RTDETR

    model = RTDETR(args.model, device=args.device)
    best = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, lr=args.lr
    )
    print(f"best checkpoint: {best}")
    return 0


def _cmd_val(args: argparse.Namespace) -> int:
    from ovkit import RTDETR

    RTDETR(args.model, device=args.device).val(data=args.data, imgsz=args.imgsz)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from ovkit import RTDETR

    out = RTDETR(args.model).export(imgsz=args.imgsz, half=args.half, out_dir=args.out)
    print(f"exported: {out} (+ labels.txt)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog="ovkit", description="ovkit model utilities")
    parser.add_argument("--version", action="version", version=f"ovkit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gui = sub.add_parser("gui", help="open the desktop window (easiest way to try ovkit)")
    p_gui.add_argument("--device", default="AUTO", help="AUTO | CPU | GPU | NPU")
    p_gui.add_argument("--camera", type=int, default=0, help="camera index for the webcam button")
    p_gui.set_defaults(func=_cmd_gui)

    p_list = sub.add_parser("list", help="list registered models")
    p_list.add_argument(
        "--all", action="store_true", help="include the archived Open Model Zoo entries"
    )
    p_list.set_defaults(func=_cmd_list)

    p_caps = sub.add_parser("capabilities", help="list composed capabilities (Model(name))")
    p_caps.set_defaults(func=_cmd_capabilities)

    p_info = sub.add_parser("info", help="show details for a model")
    p_info.add_argument("name")
    p_info.set_defaults(func=_cmd_info)

    p_dl = sub.add_parser("download", help="download (and convert) a model")
    p_dl.add_argument("name")
    p_dl.add_argument("--no-convert", action="store_true", help="skip IR conversion")
    p_dl.set_defaults(func=_cmd_download)

    p_train = sub.add_parser("train", help="train RT-DETR on your own data (needs ovkit[train])")
    p_train.add_argument("--data", required=True, help="data.yaml (YOLO format)")
    p_train.add_argument("--model", default="rtdetr-r18", help="variant or a .pt to resume")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--device", default=None)
    p_train.set_defaults(func=_cmd_train)

    p_val = sub.add_parser("val", help="mAP on the val split of a data.yaml")
    p_val.add_argument("--model", required=True, help="a trained .pt")
    p_val.add_argument("--data", required=True)
    p_val.add_argument("--imgsz", type=int, default=640)
    p_val.add_argument("--device", default=None)
    p_val.set_defaults(func=_cmd_val)

    p_exp = sub.add_parser("export", help="export a trained .pt to OpenVINO IR + labels.txt")
    p_exp.add_argument("--model", required=True, help="a trained .pt")
    p_exp.add_argument("--imgsz", type=int, default=640)
    p_exp.add_argument("--half", action="store_true", help="FP16 IR")
    p_exp.add_argument("--out", default=".", help="output directory")
    p_exp.set_defaults(func=_cmd_export)

    p_pull = sub.add_parser(
        "pull", help="convert a Hugging Face model to OpenVINO IR (needs ovkit[hf])"
    )
    p_pull.add_argument("model_id", help="a Hub id, e.g. google/vit-base-patch16-224")
    p_pull.add_argument(
        "--task", default=None, help="override the task read from the model's config"
    )
    p_pull.set_defaults(func=_cmd_pull)

    p_dev = sub.add_parser("devices", help="list OpenVINO devices")
    p_dev.set_defaults(func=_cmd_devices)

    p_run = sub.add_parser("run", help="run a model on an image/folder/video from the shell")
    p_run.add_argument("model", help="alias, registered name, or model path")
    p_run.add_argument("source", help="image / folder / video path (or .npy/.wav)")
    p_run.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p_run.add_argument("--device", default="AUTO", help="AUTO | CPU | GPU | NPU")
    p_run.add_argument("--save", metavar="PATH", help="annotated output (default: <src>_out.jpg)")
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except OVKitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        # `ovkit list | head` closes the pipe under us. Python would otherwise
        # print a traceback while flushing stdout at exit, which reads as a
        # crash for what the user meant as "show me the first few".
        with contextlib.suppress(OSError):
            sys.stdout.close()
        return 0
    except KeyboardInterrupt:
        print("\ncancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
