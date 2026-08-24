"""Remove the people from a picture before sharing it.

python examples/anonymize.py street.jpg          # faces
python examples/anonymize.py street.jpg --plates # faces and number plates
"""

from __future__ import annotations

import argparse

from ovkit import Model


def main() -> None:
    ap = argparse.ArgumentParser(description="Blur faces (and plates) in a picture.")
    ap.add_argument("image")
    ap.add_argument("out", nargs="?", default="anonymized.jpg")
    ap.add_argument("--plates", action="store_true", help="also redact number plates")
    ap.add_argument("--method", default="pixelate", choices=["pixelate", "blur"])
    args = ap.parse_args()

    r = Model("anonymize", plates=args.plates, method=args.method)(args.image)[0]
    print(r.summary())  # 'pixelated 3 faces and 1 plate'
    r.save(args.out)  # the redacted picture, never the original
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
