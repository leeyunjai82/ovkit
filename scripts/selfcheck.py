#!/usr/bin/env python3
"""ovkit self-check — run this where Hugging Face is reachable.

One command to confirm that ovkit can pull models straight from the mirror
(``leeyunjai/ovkit-models``) and actually run them on your machine::

    python scripts/selfcheck.py

What it checks, in order:

1. Environment      — Python / ovkit / OpenVINO / huggingface_hub + devices.
2. HF reachability  — can this machine reach huggingface.co at all.
3. Mirror complete  — every ``<task>/<name>/`` has model.xml + model.bin + LICENSE.
4. Download + infer  — every registered IR model is fetched from the mirror and
   run on a dummy frame (this is the real "does it work" test).
5. GenAI (optional) — if ``ovkit[genai]`` is installed, build one LLM pipeline.

A private mirror needs a token: ``export HF_TOKEN=hf_...`` (or
``huggingface-cli login``) before running. A public mirror needs nothing.

Exit code is 0 only if every required check passes.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path

# Make sibling scripts/ and the package importable when run from anywhere.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

REPO = os.environ.get("OVKIT_MIRROR", "leeyunjai/ovkit-models")

OK = "✅"
NO = "❌"
SKIP = "•"


def _hr(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check_env() -> bool:
    _hr("[1/5] 환경 (Environment)")
    ok = True
    try:
        import ovkit

        print(f"  {OK} ovkit            {ovkit.__version__}")
    except Exception as e:  # pragma: no cover
        print(f"  {NO} ovkit import 실패: {e}  (pip install -e . 했는지 확인)")
        return False
    try:
        import openvino as ov

        print(f"  {OK} openvino         {ov.__version__}")
    except Exception as e:
        print(f"  {NO} openvino         {e}")
        ok = False
    try:
        import numpy as np

        print(f"  {OK} numpy            {np.__version__}")
    except Exception as e:
        print(f"  {NO} numpy            {e}")
        ok = False
    try:
        import huggingface_hub as hub

        print(f"  {OK} huggingface_hub  {hub.__version__}")
    except Exception as e:
        print(f"  {NO} huggingface_hub  {e}  (pip install -e .)")
        ok = False
    try:
        from ovkit.core.backend import available_devices

        print(f"  {OK} OpenVINO devices {available_devices()}")
    except Exception as e:
        print(f"  {SKIP} devices 조회 실패: {e}")
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"  {SKIP} HF_TOKEN         {'set' if tok else 'not set (public repo면 불필요)'}")
    return ok


def check_reachable() -> bool:
    _hr("[2/5] Hugging Face 도달성 (Reachability)")
    import urllib.request

    url = f"https://huggingface.co/api/models/{REPO}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:  # noqa: S310
            print(f"  {OK} {url} -> HTTP {r.status}")
        return True
    except Exception as e:
        print(f"  {NO} huggingface.co 도달 실패: {type(e).__name__}: {e}")
        print("     -> 이 머신/환경에서 HF로 나갈 수 없습니다. 네트워크 정책/방화벽 확인.")
        return False


def check_mirror() -> bool:
    _hr(f"[3/5] 미러 완전성 (Mirror completeness) — {REPO}")
    spec = importlib.util.spec_from_file_location("verify_mirror", HERE / "verify_mirror.py")
    vm = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    try:
        spec.loader.exec_module(vm)
        return vm.verify(REPO) == 0
    except Exception as e:
        print(f"  {NO} 미러 점검 실패: {type(e).__name__}: {e}")
        return False


def check_download_infer() -> bool:
    _hr("[4/5] 다운로드 + 추론 (Download from mirror & run)")
    import numpy as np

    from ovkit import Model
    from ovkit.core.registry import list_models, resolve

    # Registered IR models (genai handled separately in step 5).
    names = [n for n in list_models() if getattr(resolve(n), "src", None) != "genai"]
    if not names:
        print(f"  {SKIP} 등록된 IR 모델 없음 (build_mirror.py --emit-manifest 로 배선 필요)")
        return True

    all_ok = True
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    for name in names:
        try:
            model = Model(name)
            r = model(img)[0]
            if r.boxes is not None:
                detail = f"boxes={len(r.boxes)}"
            elif r.probs is not None:
                detail = f"top1={r.probs.top1}"
            elif r.masks is not None:
                detail = f"masks={r.masks.data.shape}"
            elif r.text is not None:
                detail = f"text={r.text!r}"
            else:
                detail = f"tensors={list(getattr(r, 'tensors', {}) or {})}"
            print(f"  {OK} {name:30s} task={model.task} {detail}")
        except Exception as e:
            all_ok = False
            print(f"  {NO} {name:30s} {type(e).__name__}: {str(e)[:120]}")
    return all_ok


def check_genai() -> bool:
    _hr("[5/5] GenAI (optional)")
    if importlib.util.find_spec("openvino_genai") is None:
        print(f'  {SKIP} openvino-genai 미설치 — 건너뜀 (pip install -e ".[genai]")')
        return True
    try:
        from ovkit.genai import pipeline

        print("  ... tinyllama_chat 다운로드 + 파이프라인 빌드 (시간이 걸릴 수 있음)")
        llm = pipeline("tinyllama_chat")
        out = llm.generate("Say hello in one word.", max_new_tokens=8)
        print(f"  {OK} tinyllama_chat -> {str(out)[:60]!r}")
        return True
    except Exception as e:
        print(f"  {NO} genai 파이프라인 실패: {type(e).__name__}: {str(e)[:160]}")
        return False


def main() -> int:
    print("ovkit self-check —", f"mirror={REPO}")
    results: dict[str, bool] = {}

    results["환경"] = check_env()
    if not results["환경"]:
        print(f"\n{NO} 환경이 준비되지 않았습니다. 먼저 `pip install -e .` 후 다시 실행하세요.")
        return 1

    reachable = check_reachable()
    results["HF 도달"] = reachable
    if reachable:
        results["미러 완전성"] = check_mirror()
        results["다운로드+추론"] = check_download_infer()
        try:
            results["GenAI"] = check_genai()
        except Exception:
            traceback.print_exc()
            results["GenAI"] = False
    else:
        print("\nHF에 못 나가서 미러/다운로드 검증을 건너뜁니다.")

    _hr("요약 (Summary)")
    for k, v in results.items():
        print(f"  {OK if v else NO} {k}")
    # GenAI is optional; everything else is required.
    required = {k: v for k, v in results.items() if k != "GenAI"}
    passed = all(required.values())
    print(f"\nRESULT: {'ALL OK ✅' if passed else 'FAILED ❌'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
