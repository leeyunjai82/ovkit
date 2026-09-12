#!/usr/bin/env python3
"""Convert Supertonic-3 to OpenVINO IR and put it on ovkit's mirror.

    python scripts/mirror_supertonic.py             # convert + check, upload nothing
    python scripts/mirror_supertonic.py --upload    # and copy it in

Downloads ``Supertone/supertonic-3``, converts its four ONNX graphs to IR,
compiles each one to prove the IR is usable, and uploads the lot — plus the
tables that turn text into ids, the voice styles, and **the licence** — to
``leeyunjai/ovkit-models``.

The licence is not an afterthought here. Supertonic-3 is OpenRAIL-M: ovkit may
serve it, redistribute it and use it commercially, on the condition that the
same use-based restrictions travel with every copy. A mirrored copy without the
licence beside it would break that condition, so ``LICENSE`` is uploaded with
the weights and ``license_url`` is set on every manifest entry — without which
``registry.resolve`` refuses to load the model at all.

Layout on the mirror::

    tts/supertonic3/
      text_encoder/model.xml + .bin
      duration_predictor/model.xml + .bin
      vector_estimator/model.xml + .bin
      vocoder/model.xml + .bin
      data/tts.json                 sample rate, chunk sizes, latent dim
      data/unicode_indexer.json     codepoint -> id, 31 languages
      data/voices/F1.json .. M5.json
      LICENSE
      README.md

Needs ``HF_TOKEN`` with write access to the mirror. Runs in Actions
(**Supertonic** workflow, ``upload`` input); this container cannot reach
huggingface.co.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SOURCE = "Supertone/supertonic-3"
MIRROR = "leeyunjai/ovkit-models"
PREFIX = "tts/supertonic3"
LICENSE_URL = "https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE"

#: ONNX graph -> the folder it lands in on the mirror.
GRAPHS = {
    "onnx/text_encoder.onnx": "text_encoder",
    "onnx/duration_predictor.onnx": "duration_predictor",
    "onnx/vector_estimator.onnx": "vector_estimator",
    "onnx/vocoder.onnx": "vocoder",
}

#: Data files, and where they go. Without these the graphs are unusable: the
#: indexer is the only thing that turns a sentence into ids the model knows.
TABLES = {
    "onnx/tts.json": "data/tts.json",
    "onnx/unicode_indexer.json": "data/unicode_indexer.json",
}

VOICES = tuple(f"{sex}{n}" for sex in "FM" for n in range(1, 6))


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} GB"


def _card() -> str:
    return f"""---
license: openrail
---

# Supertonic 3 (OpenVINO IR)

`{SOURCE}` converted to OpenVINO IR for [ovkit](https://github.com/leeyunjai82/ovkit).
Four graphs, ~99M parameters in total, 44.1 kHz output, 31 languages including
Korean.

| file | what it does |
|---|---|
| `duration_predictor/` | text + voice style -> how many seconds the sentence takes |
| `text_encoder/` | text + voice style -> text embedding |
| `vector_estimator/` | flow matching; run it `total_step` times, output feeds back in |
| `vocoder/` | latent -> waveform |
| `data/unicode_indexer.json` | unicode codepoint -> character id |
| `data/tts.json` | sample rate, chunk sizes, latent dim |
| `data/voices/` | ten preset voice styles |

## Licence

The model is **OpenRAIL-M**, © Supertone Inc. It permits commercial use,
redistribution and modification, and requires that the same use-based
restrictions are passed on to everyone you pass the model to. `LICENSE` in this
folder is the copy that travels with it — read it before using the model, and
keep it with any copy you make.

Conversion and the surrounding code are ovkit's; Supertone's own sample code is
MIT.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true", help="실제로 미러에 올린다")
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--mirror", default=MIRROR)
    parser.add_argument("--prefix", default="", help="원본 저장소 안의 경로 앞부분")
    parser.add_argument("--work", default="supertonic_ir", help="변환 결과를 둘 폴더")
    args = parser.parse_args()

    import openvino as ov
    from huggingface_hub import hf_hub_download

    def at(path: str) -> str:
        return f"{args.prefix.strip('/')}/{path}" if args.prefix.strip("/") else path

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    print(f"=== {args.source} -> IR\n")
    total = 0
    for name, folder in GRAPHS.items():
        src = Path(hf_hub_download(args.source, at(name)))
        out_dir = work / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        xml = out_dir / "model.xml"

        model = ov.convert_model(str(src))
        ov.save_model(model, str(xml), compress_to_fp16=True)

        # An IR that saves but will not compile is worse than no IR: it fails
        # later, on a student's machine, with a message about weights.
        compiled = ov.Core().compile_model(str(xml), "CPU")
        ports = ", ".join(p.any_name for p in compiled.inputs)
        size = xml.stat().st_size + xml.with_suffix(".bin").stat().st_size
        total += size
        print(f"  {folder:20s} {_human(size):>10s}   in: {ports}")

    data_dir = work / "data"
    (data_dir / "voices").mkdir(parents=True, exist_ok=True)
    for name, target in TABLES.items():
        shutil.copy(hf_hub_download(args.source, at(name)), work / target)
    for voice in VOICES:
        try:
            shutil.copy(
                hf_hub_download(args.source, at(f"voice_styles/{voice}.json")),
                data_dir / "voices" / f"{voice}.json",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  목소리 {voice} 없음: {type(exc).__name__}")

    # The obligation travels with the weights or the model does not ship.
    try:
        shutil.copy(hf_hub_download(args.source, at("LICENSE")), work / "LICENSE")
    except Exception as exc:  # noqa: BLE001
        print(f"\n!! LICENSE를 못 받았다: {exc}")
        print("OpenRAIL-M은 사본마다 같은 제한이 따라가야 한다. 올리지 않는다.")
        return 1
    (work / "README.md").write_text(_card(), encoding="utf-8")

    cfg = json.loads((work / "data" / "tts.json").read_text(encoding="utf-8"))
    print(f"\n  IR 합계 {_human(total)}")
    print(f"  샘플레이트 {cfg['ae']['sample_rate']} Hz")
    print(f"  목소리 {len(list((data_dir / 'voices').glob('*.json')))}개")
    print(f"  LICENSE {(work / 'LICENSE').stat().st_size:,}바이트")

    print("\n=== 매니페스트 (src/ovkit/manifests/tts.yaml)\n")
    print(_manifest_snippet(args.mirror))

    if not args.upload:
        print("\n--upload 없이 돌렸다. 올린 것은 없다.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.mirror, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(work),
        path_in_repo=PREFIX,
        repo_id=args.mirror,
        commit_message="Supertonic 3 (OpenVINO IR) — 라이선스 포함",
    )
    print(f"\n{args.mirror}/{PREFIX} 에 올렸다.")
    return 0


def _manifest_snippet(mirror: str) -> str:
    lines = []
    for folder in GRAPHS.values():
        lines.append(
            f"supertonic3_{folder}:\n"
            f"  src: hf\n"
            f"  repo: {mirror}\n"
            f"  filename: {PREFIX}/{folder}/model.xml\n"
            f"  task: generic\n"
            f"  license: openrail\n"
            f"  license_url: {LICENSE_URL}\n"
            f"  tier: zoo\n"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
