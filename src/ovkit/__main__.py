"""``ovkit`` command-line interface: ``gui``, ``run``, ``list``, ``info``, ``download``, ``devices``."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .core.convert import to_ir
from .core.download import fetch
from .core.errors import OVKitError
from .core.registry import list_models, resolve


def _cmd_list(_: argparse.Namespace) -> int:
    names = list_models()
    if not names:
        print("No models registered.")
        return 0
    aliases: list[tuple[str, str]] = []
    models: list[tuple[str, str, str]] = []
    for name in names:
        entry = resolve(name)
        if entry is None:
            continue
        if entry.name != name:  # capability alias -> its target
            aliases.append((name, entry.name))
            continue
        desc = entry.description or ""
        if len(desc) > 60:
            desc = desc[:57] + "..."
        models.append((name, str(entry.task), desc))
    if aliases:
        print("aliases (capability -> model):")
        for alias, target in aliases:
            print(f"  {alias:24s} -> {target}")
        print()
    print(f"models ({len(models)}):")
    for name, task, desc in models:
        print(f"  {name:44s} {task:18s} {desc}")
    return 0


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
    results = model(args.source, conf=args.conf)
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


if __name__ == "__main__":
    raise SystemExit(main())
