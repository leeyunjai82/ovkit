# 가이드 (전체 흐름)

설치부터 모델 실행까지, 그리고 미러를 관리한다면 모델 세트를 빌드·검증·배포하는 전 과정을
담았습니다. 위에서 아래로 읽으면 되고, 문제가 생기면 [문제 해결](#가이드-문제해결)로 가세요.

## 1. ovkit이란

ovkit은 OpenVINO를 {class}`~ovkit.Model` 객체 하나 뒤로 감쌉니다. 모델 이름이나 파일을 주고
이미지에 호출하면 깔끔한 {class}`~ovkit.Results`가 나옵니다. 다운로드, OpenVINO IR 변환, 캐시,
디바이스 컴파일, 태스크 감지, 전·후처리는 전부 자동입니다. 모델은 Hugging Face 미러
(`leeyunjai/ovkit-models`)에서 받고, GenAI(LLM/STT)는 openvino-genai로 처리합니다.

## 2. 설치

소스에서 가상환경에 설치합니다:

```bash
git clone https://github.com/leeyunjai82/ovkit.git
cd ovkit
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e .                     # 코어 (가벼움)
pip install -e ".[quant]"            # + NNCF INT8 양자화
pip install -e ".[genai]"            # + openvino-genai / optimum-intel
pip install -e ".[all]"              # 전부
```

`-e`는 editable 설치라 `git pull`만 하면 재설치 없이 갱신됩니다. Python 3.10+.

## 3. 빠른 시작

```python
from ovkit import Model

model = Model("rtdetr_r50")              # 이름 -> 자동 다운로드 / 변환 / 캐시
for r in model("street.jpg", conf=0.25): # __call__ == predict
    print(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)
    r.save("out.jpg")
```

## 4. 모델 고르기

모델을 지정하는 세 가지:

- **기능 별칭** — 친숙한 기본값: `Model("face_detection")`, `Model("pose")`,
  `Model("llm")`. 전체 별칭 표는 {doc}`models` 카탈로그 참고.
- **등록된 이름** — 예: `Model("rtdetr_r50")`. YAML 매니페스트에서 찾아 미러에서 받고 변환·캐시.
- **파일 경로** — `Model("model.xml")`(IR) 또는 `Model("model.onnx")`(최초 사용 시 변환).

CLI로 탐색:

```bash
ovkit list                 # 등록 모델: 이름 / 태스크 / 설명
ovkit info face_detection  # 소스·태스크·라이선스·정밀도 (별칭도 따라감)
```

## 5. 추론 실행

모델 호출(`model(x)`)은 `model.predict(x)`와 같고, **입력 종류가 자동 감지**됩니다:

```python
model("img.jpg")                       # 이미지 파일
model(cv2.imread("img.jpg"))           # HWC BGR ndarray
model("frames/")                       # 이미지 폴더
model("clip.mp4")                      # 비디오 파일
for r in model.predict(0, stream=True):# 웹캠(카메라 인덱스) — 지연 제너레이터
    annotated = r.plot()
```

- `conf` — 검출/인스턴스 태스크의 신뢰도 임계값.
- `stream=True` — 지연 **제너레이터** 반환(프레임 하나씩 처리); 아니면
  {class}`~ovkit.Results`의 `list`.
- 비이미지 입력(`.npy`, `.wav`, 비이미지 ndarray)은 자동으로 원시 추론으로 가서
  `{이름: ndarray}`를 반환 — [§7](#가이드-저수준) 참고.

### Results

```python
r = model("img.jpg")[0]
r.boxes.xyxy      # (N,4) 픽셀 박스; .xywh .conf .cls 도
r.name_for(2)     # "car"  (클래스 id -> 이름)
annotated = r.plot()   # -> 주석 ndarray (박스/마스크/키포인트/텍스트)
r.save("out.jpg")
```

| 속성 | 태스크 | 내용 |
| ---- | ------ | ---- |
| `r.boxes` | detect | `xyxy`, `xywh`, `conf`, `cls` |
| `r.masks` | segment | `(N,H,W)` 마스크 (또는 클래스맵) |
| `r.keypoints` | pose | `(N,K,3)` `[x,y,conf]` |
| `r.probs` | classify | `top1`, `top5`, 확률 |
| `r.text` | ocr | 디코드된 문자열 |
| `r.tensors` | generic | 원시 `{이름: ndarray}` |

## 6. 태스크 자동 감지

순서대로 결정합니다: 매니페스트 `task` → IR `rt_info` → 출력 모양 휴리스틱. 직접 지정도 가능:

```python
Model("some.xml", task="detect")   # detect | classify | segment | pose | ocr
```

비전 태스크는 전용 디코더, 그 외는 원시 텐서를 돌려주는 제너릭 어댑터.

(가이드-저수준)=
## 7. 저수준 — 모든 모델 (NLP / 오디오 / 다중입력)

이미지를 받지 않는 모델(BERT, 번역, 다중입력 얼굴 파이프라인)은 텐서를 직접 만들어 넣습니다:

```python
m = Model("bert_small_uncased_whole_word_masking_squad_0002")
print(m.inputs)                                 # [(이름, shape, dtype), ...]
out = m.infer({"input_ids": ids, "attention_mask": mask})   # {이름: ndarray}
```

회색조(1채널) 모델은 자동 처리됩니다 — 전처리가 3채널을 만들면 백엔드가 모델 채널 수에 맞춰
정합하므로 OCR·회색조 분류기도 그냥 동작합니다.

## 8. 디바이스

`device="AUTO"`(기본)는 OpenVINO가 고르고, `"CPU"`/`"GPU"`/`"NPU"`는 명시 지정. `Model`에
설정하거나 호출마다 덮어쓰기. 단일 이미지는 동기, `stream=True`는 `AsyncInferQueue` 처리량 모드.

```python
from ovkit.core.backend import available_devices
print(available_devices())             # 예: ['CPU', 'GPU', 'NPU']
Model("rtdetr_r50")("img.jpg", device="GPU")
```

## 9. 양자화 (INT8)

```python
m = Model("rtdetr_r50")
m.quantize(["calib1.jpg", "calib2.jpg", ...], preset="int8")   # NNCF PTQ
r = m("img.jpg")                                               # 이제 INT8
```

`pip install -e ".[quant]"` 필요.

## 10. GenAI (LLM / STT)

```python
from ovkit.genai import pipeline

llm = pipeline("tinyllama_chat")               # 또는 Model 별칭: "llm"
print(llm.generate("OpenVINO를 한 문장으로.", max_new_tokens=64))

stt = pipeline("whisper_base")                 # 음성->텍스트 ("stt")
print(stt.generate(audio_16k_mono_float32))
```

genai 모델은 미러(서브폴더)에서 받고, 원본 OpenVINO repo를 폴백으로 둡니다.
`pip install -e ".[genai]"` 필요.

(가이드-미러)=
## 11. 모델 미러 (메인테이너)

ovkit은 본인이 관리하는 HF repo(`leeyunjai/ovkit-models`)에서 모델을 받습니다. 최종 사용자는
아래 스크립트를 절대 실행하지 않습니다 — 그냥 `Model("이름")`만 씁니다. 이 절은 미러를
채우거나 갱신하는 사람을 위한 것입니다.

### 한눈에 보는 파이프라인

```
build_mirror.py  → 다운로드 + 변환 + 검증 + 업로드  (미러 기록)
verify_mirror.py → 모든 모델이 완전하고 크기 정상인 IR인지 점검
selfcheck.py     → 미러에서 받아 각 모델을 실제로 실행
                   (--prune-manifest = 로드 안 되는 것 제거)
--emit-manifest  → src/ovkit/manifests/omz.yaml (런타임 레지스트리) 생성
```

### 빌드 / 갱신

```bash
export HF_TOKEN=...                # write 토큰 (또는 huggingface-cli login)
                                   # Windows PowerShell: $env:HF_TOKEN = "..."

# 큐레이트 + Apache-2.0 OMZ 전체 미러:
python scripts/build_mirror.py --omz-intel

# genai만 미러(openvino-genai 디렉토리 전체 -> genai/<name>/):
python scripts/build_mirror.py --models tinyllama_chat whisper_base
```

각 모델은 다운로드 → IR 변환 → **컴파일 검증** → 모델카드·LICENSE와 함께 업로드됩니다.
가중치가 비었거나 소스 URL이 죽은 모델은 실패로 보고되고 **업로드되지 않습니다.**

### 런타임 매니페스트 생성

```bash
python scripts/build_mirror.py --omz-intel \
    --emit-manifest src/ovkit/manifests/omz.yaml
```

OMZ 모델마다 미러를 가리키는 항목을 쓰되, 미러와 **교차검증**합니다(너무 작거나 없는 `.bin`은
제외). genai는 `genai.yaml`에서만 관리하고 여기엔 넣지 않습니다.

### 검증 + 프룬

```bash
python scripts/verify_mirror.py            # 태스크별 개수; 작은/없는 .bin 표시
python scripts/selfcheck.py --prune-manifest src/ovkit/manifests/omz.yaml \
    --load-only --no-genai
```

`selfcheck`는 등록된 모든 모델을 미러에서 받아 컴파일합니다. `--prune-manifest`는
**로드되는 모델만** 남기고 다운로드/컴파일 실패를 omz.yaml에서 제거합니다. `--load-only`를
쓰면 빈 프레임 더미추론(모든 모델에 맞진 않음)을 실패로 오인하지 않습니다.

### 배포

`omz.yaml`은 커밋 전까지 본인 PC에만 있습니다:

```bash
git add src/ovkit/manifests/omz.yaml
git commit -m "Update OMZ runtime manifest"
git push
```

커밋하면 모든 사용자가 그 모델들을 이름으로 받고, 그걸 가리키는 기능 별칭도 활성화됩니다.

## 12. 모델 추가

모델은 코드가 아니라 데이터 — 매니페스트에 한 줄:

```yaml
my_model:
  src: hf                          # hf | url | genai
  repo: leeyunjai/ovkit-models
  filename: detect/my_model/model.xml
  task: detect
  description: ovkit list에 표시될 한 줄 설명.
  license: apache-2.0             # 필수; permissive 여야 함
  fallback: { src: hf, repo: onnx-community/..., filename: onnx/model.onnx }
```

`aliases.yaml`에 별칭 추가: `my_alias: { alias: my_model }`.

(가이드-문제해결)=
## 13. 문제 해결

| 증상 | 원인 & 해결 |
| ---- | ----------- |
| huggingface.co에서 `Failed to download ... 403` | 네트워크/환경이 HF에 못 나감. Claude Code 웹이면 huggingface.co를 허용하는 네트워크 정책으로 세션 생성. |
| 로드 시 `Empty weights data in bin file` / `core.cpp:135` | 미러 `.bin`이 비었거나 손상. 재미러(`build_mirror.py --omz-intel`); 소스가 죽었으면 실패로 보고되니 `selfcheck --prune-manifest`로 제거. |
| OCR / 회색조 모델이 추론에서 실패 | 수정됨: 백엔드가 3↔1 채널을 자동 정합. `git pull`로 갱신. |
| `This model takes N inputs ...` | 다중입력 모델(예: gaze). 단일 이미지 대신 `model.infer({...})`로 모든 입력 제공. |
| public repo인데 `... repo is gated` (401) | HF "Gated access"는 공개여부와 별개 — 모델 페이지에서 access requests 비활성화하거나 인증. |
| `omz.yaml` 재생성 시 깨진 모델이 계속 다시 들어감 | 오래된 `omz.yaml`이 `genai.yaml`/큐레이트를 덮어씀. `omz.yaml` 삭제 → genai 먼저 미러 → 재생성. |
| `ovkit list`에 모델이 `?`로 보임 | 대상이 `omz.yaml`을 필요로 하는 별칭. 생성 후 커밋([§11](#가이드-미러)). |

HF가 닿는 곳에서 언제든 전체 자가점검:

```bash
python scripts/selfcheck.py            # 환경, HF, 미러, 다운로드+실행, genai
```

## 14. CLI 레퍼런스

```bash
ovkit list                 # 등록 모델: 이름 / 태스크 / 설명
ovkit info <name>          # 소스, 태스크, 라이선스, 정밀도 (별칭도 따라감)
ovkit download <name>      # 다운로드 + IR 변환 (캐시 워밍업)
ovkit devices              # 사용 가능한 OpenVINO 디바이스
```

## 15. 환경 변수

| 변수 | 의미 |
| ---- | ---- |
| `OVKIT_HOME` | 캐시 루트 (기본 `~/.cache/ovkit`) |
| `OVKIT_OFFLINE` | `1` = 캐시 전용, 네트워크 차단 |
| `OVKIT_MANIFESTS` | 추가 매니페스트 경로 (`os.pathsep` 구분) |
| `OVKIT_MIRROR` | 미러 repo id 덮어쓰기 (스크립트) |
| `HF_TOKEN` | 비공개 repo / 업로드용 HF 토큰 |
