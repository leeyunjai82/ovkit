"""Teach your own AI — no GPU, no training loop, just example photos.

python examples/teach_ai.py cans/ bottles/ test_photo.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

from ovkit import Model


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__.strip())
        print("\nCollect examples with your webcam first:")
        print("  python -c \"from ovkit.pipelines.teach import collect; collect('can', 30)\"")
        return
    *folders, test = sys.argv[1:]

    ai = Model("teach")  # modes: photo(default) · face · hand · upper · body
    for folder in folders:
        n = ai.learn(Path(folder).name.rstrip("/"), folder)
        print(f"learned {Path(folder).name}: {n} examples")

    label, confidence = ai.guess(test)
    print(f"{Path(test).name}: {label} ({confidence:.2f})")

    saved = ai.save("my-ai")
    print(f"saved -> {saved}  (reload with Model('teach', load='my-ai'))")


if __name__ == "__main__":
    main()
