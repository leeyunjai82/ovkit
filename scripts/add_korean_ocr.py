#!/usr/bin/env python3
"""Convert PaddleOCR's Korean recogniser to OpenVINO IR and mirror it.

    python scripts/add_korean_ocr.py             # fetch + convert, print the shapes
    python scripts/add_korean_ocr.py --upload    # ... and push it to the mirror

Or run the **Korean OCR** workflow on GitHub, where ``HF_TOKEN`` lives as a
secret and the network is open.

Why this model
--------------
ovkit could find Korean text and not read it: ``text_recognition_0014`` knows
0-9 and a-z. PaddleOCR's ``korean_PP-OCRv3_rec`` is Apache-2.0, about 10 MB,
and CTC — so it drops straight into ovkit's existing greedy-CTC decoder and
runs in real time on a classroom laptop.

And the conversion is one step, not three: OpenVINO ships a **paddle**
frontend, so ``ov.convert_model`` reads ``inference.pdmodel`` directly. No
PaddlePaddle install, no ONNX detour.

What lands in the mirror
------------------------
``optical_character_recognition/korean_ppocrv3_rec/``

* ``model.xml`` / ``model.bin`` — the IR, input pinned to ``[1, 3, 48, 320]``
  (the exported graph leaves width dynamic; ovkit resizes each crop to the
  model's own input size, which has to be a number).
* ``labels.txt`` — the character table, one symbol per line, from PaddleOCR's
  ``korean_dict.txt``. The manifest reads it via ``postprocess.charset:
  labels``. The dictionary contains a **space** entry, and a space on a line
  of its own does not survive a round trip through a text file that anything
  strips — so it is written as the token ``<space>``, which the decoder maps
  back. Losing that one line would shift every class after it by one and the
  model would read fluent nonsense.
* ``LICENSE.md`` — where it came from and under what licence.
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

TARGET_REPO = "leeyunjai/ovkit-models"
DEST = "optical_character_recognition/korean_ppocrv3_rec"

MODEL_TAR = (
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar"
)
DICT_URL = (
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/"
    "ppocr/utils/dict/korean_dict.txt"
)

#: PP-OCRv3 recognisers take 48-pixel-high lines; 320 wide fits a phrase.
INPUT_SHAPE = [1, 3, 48, 320]

#: How a space is written in labels.txt (a bare space line gets stripped away).
SPACE_TOKEN = "<space>"

LICENSE_NOTE = """\
Korean text recognition — PaddleOCR PP-OCRv3 (`korean_PP-OCRv3_rec`).

Converted to OpenVINO IR with the `paddle` frontend from
{tar}
and mirrored here so ovkit downloads from one repository.

`labels.txt` is PaddleOCR's `korean_dict.txt` (one symbol per line). The CTC
blank is class 0, so dictionary entry *i* is class *i + 1*, and the space
character is appended by the decoder rather than stored in the file.

PaddleOCR is Apache-2.0: https://github.com/PaddlePaddle/PaddleOCR
"""


def _get(url: str) -> bytes:
    print(f"fetching {url}")
    with urllib.request.urlopen(url, timeout=180) as response:  # noqa: S310 - fixed URLs
        return response.read()


def fetch(work: Path) -> tuple[Path, list[str]]:
    """Download the inference model and the character dictionary."""
    with tarfile.open(fileobj=io.BytesIO(_get(MODEL_TAR))) as tar:
        for member in tar.getmembers():
            if member.isfile():
                name = Path(member.name).name
                (work / name).write_bytes(tar.extractfile(member).read())  # type: ignore[union-attr]
    model = next((work / n for n in ("inference.pdmodel", "model.pdmodel") if (work / n).is_file()), None)
    if model is None:
        raise SystemExit(f"no .pdmodel in {MODEL_TAR} — contents: {sorted(p.name for p in work.iterdir())}")

    # Every line is a symbol, including the one that is a space. Only the
    # trailing newline at the end of the file is not an entry.
    raw = _get(DICT_URL).decode("utf-8").split("\n")
    if raw and raw[-1] == "":
        raw.pop()
    chars = [SPACE_TOKEN if not ln.strip() else ln for ln in raw]
    print(f"  model: {model.name} ({model.stat().st_size / 1e6:.1f} MB)  dictionary: {len(chars)} symbols")
    return model, chars


def check_classes(xml: Path, chars: list[str]) -> None:
    """The class count has to match the alphabet, or the reading is garbage.

    PaddleOCR numbers classes ``[blank] + dictionary + [space]``. If the two
    sides disagree by even one, decoding still "works" — every symbol after
    the gap is simply the wrong one, and nothing raises. So it is checked here,
    where it can stop the upload.
    """
    import openvino as ov

    model = ov.Core().read_model(xml)
    classes = int(model.outputs[0].partial_shape[-1].get_length())
    expected = 1 + len(chars) + 1
    if classes != expected:
        raise SystemExit(
            f"the model has {classes} classes but the alphabet gives {expected} "
            f"(1 blank + {len(chars)} dictionary + 1 space). Refusing to upload "
            f"a recogniser that would read the wrong symbol for every class "
            f"after the mismatch."
        )
    print(f"  classes: {classes} = 1 blank + {len(chars)} dictionary + 1 space  OK")


def convert(pdmodel: Path, out_dir: Path) -> Path:
    """Read the Paddle graph, pin the input shape, save FP16 IR."""
    import openvino as ov

    print(f"converting with the paddle frontend ({ov.__version__}) ...")
    model = ov.convert_model(pdmodel)
    model.reshape({model.inputs[0]: ov.PartialShape(INPUT_SHAPE)})

    out_dir.mkdir(parents=True, exist_ok=True)
    xml = out_dir / "model.xml"
    ov.save_model(model, xml, compress_to_fp16=True)

    # Print what the graph actually says: the manifest has to match it, and a
    # wrong guess here is a model that "loads" and reads nothing.
    for port in model.inputs:
        print(f"  input  {port.any_name:20s} {port.partial_shape}")
    for port in model.outputs:
        print(f"  output {port.any_name:20s} {port.partial_shape}")
    print(f"  -> {xml} ({xml.with_suffix('.bin').stat().st_size / 1e6:.1f} MB)")
    return xml


def upload(out_dir: Path) -> int:
    import os

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print(
            "HF_TOKEN is not set — uploading needs a token with write access to\n"
            f"{TARGET_REPO}. Either export it here, or run the \"Korean OCR\"\n"
            "workflow on GitHub, where it lives as a secret.",
            file=sys.stderr,
        )
        return 2

    from huggingface_hub import CommitOperationAdd, HfApi

    operations = [
        CommitOperationAdd(path_in_repo=f"{DEST}/{name}", path_or_fileobj=str(out_dir / name))
        for name in ("model.xml", "model.bin", "labels.txt")
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo=f"{DEST}/LICENSE.md",
            path_or_fileobj=LICENSE_NOTE.format(tar=MODEL_TAR).encode("utf-8"),
        )
    )
    print(f"\ncommitting {len(operations)} file(s) to {TARGET_REPO} ...")
    HfApi().create_commit(
        repo_id=TARGET_REPO,
        operations=operations,
        commit_message="Add the Korean PP-OCRv3 text recogniser",
    )
    print("done. ovkit already has the manifest entry:")
    print('  python -c "from ovkit import Model; print(Model(\'read_text\', \'sign.jpg\'))"')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--upload", action="store_true", help="push to the mirror")
    parser.add_argument("--out", default="", help="where to write the IR (default: a temp dir)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        out_dir = Path(args.out) if args.out else work / "ir"
        pdmodel, chars = fetch(work)
        xml = convert(pdmodel, out_dir)
        (out_dir / "labels.txt").write_text("\n".join(chars) + "\n", encoding="utf-8")
        print(f"  -> {out_dir / 'labels.txt'} ({len(chars)} symbols)")
        check_classes(xml, chars)
        if not args.upload:
            print("\nlooks right? re-run with --upload.")
            return 0
        return upload(out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
