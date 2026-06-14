#!/usr/bin/env python3
"""Chat with an LLM via openvino-genai.

    pip install -e ".[genai]"
    python examples/llm.py "Explain OpenVINO in one sentence."

The model is downloaded from Hugging Face on first use.
"""

from __future__ import annotations

import argparse

from ovkit.genai import pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="Explain OpenVINO in one sentence.")
    ap.add_argument("--model", default="tinyllama_chat")
    ap.add_argument("--device", default="AUTO")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    llm = pipeline(args.model, device=args.device)
    print(llm.generate(args.prompt, max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
