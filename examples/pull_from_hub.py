"""Use any Hugging Face model — convert once, then run with plain ovkit.

pip install "ovkit[hf]"
python examples/pull_from_hub.py google/vit-base-patch16-224 photo.jpg
"""

from __future__ import annotations

import sys

from ovkit import Model, hub


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        print("\nAlready converted:", ", ".join(hub.pulled_models()) or "(none yet)")
        return
    model_id = sys.argv[1]

    if hub.ir_path(model_id) is None:
        hub.pull(model_id)  # one-time: downloads, converts to IR, writes labels

    if len(sys.argv) > 2:
        r = Model(model_id, sys.argv[2])
        print(r)  # tabby 0.94
        for row in r.found:
            print(f"  {row['name']} ({row['name_en']}) {row['score']}")
        print(f"  {r.elapsed_ms:.0f} ms on {r.device}")


if __name__ == "__main__":
    main()
