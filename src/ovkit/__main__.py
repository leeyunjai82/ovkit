"""``ovkit`` command-line interface: ``download``, ``info``, ``list``, ``devices``."""

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
    for name in names:
        entry = resolve(name)
        task = entry.task if entry else "?"
        lic = entry.license if entry else "?"
        print(f"{name:24s} task={task:9s} license={lic}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    entry = resolve(args.name)
    if entry is None:
        print(f"'{args.name}' is not a registered model.", file=sys.stderr)
        return 1
    print(f"name       : {entry.name}")
    print(f"task       : {entry.task}")
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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog="ovkit", description="ovkit model utilities")
    parser.add_argument("--version", action="version", version=f"ovkit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list registered models")
    p_list.set_defaults(func=_cmd_list)

    p_info = sub.add_parser("info", help="show details for a model")
    p_info.add_argument("name")
    p_info.set_defaults(func=_cmd_info)

    p_dl = sub.add_parser("download", help="download (and convert) a model")
    p_dl.add_argument("name")
    p_dl.add_argument("--no-convert", action="store_true", help="skip IR conversion")
    p_dl.set_defaults(func=_cmd_download)

    p_dev = sub.add_parser("devices", help="list OpenVINO devices")
    p_dev.set_defaults(func=_cmd_devices)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except OVKitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
