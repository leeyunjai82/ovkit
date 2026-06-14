# 사용법

이 문서는 ovkit이 어떻게 동작하고 각 부분을 어떻게 쓰는지 설명합니다. ovkit은 OpenVINO를
{class}`~ovkit.Model` 객체 하나 뒤로 감쌉니다 — 모델 이름이나 파일을 주고 이미지에 호출하면
깔끔한 {class}`~ovkit.Results` 컨테이너가 나옵니다. 그 사이의 다운로드, OpenVINO IR 변환,
캐시, 디바이스 컴파일, 태스크 감지, 전·후처리는 전부 자동으로 처리됩니다.

## 설치

ovkit은 아직 PyPI에 없어서 소스에서 설치합니다. 가상환경으로 의존성을 시스템과 분리하세요:

```bash
git clone https://github.com/leeyunjai82/ovkit.git
cd ovkit

python -m venv .venv                     # 가상환경 생성
source .venv/bin/activate                # Windows: .venv\Scripts\activate

pip install -e .                         # 코어 (가벼움)

pip install -e ".[quant]"                # + NNCF INT8 양자화
pip install -e ".[genai]"                # + openvino-genai / optimum-intel
pip install -e ".[anomaly]"              # + anomalib
pip install -e ".[all]"                  # 전부
```

`-e`는 *editable*(개발) 설치라 패키지가 체크아웃을 가리켜서, `git pull`만 하면 재설치 없이
갱신됩니다. Python 3.10+ 필요. 코어 의존성은 가볍게(`openvino`, `numpy`,
`opencv-python-headless`, `pillow`, `pyyaml`, `huggingface_hub`) 유지하고, 무거운 것(NNCF,
openvino-genai, anomalib)은 `[...]` 옵션 extra로 선택 설치합니다.

## 모델 로드

```python
from ovkit import Model

model = Model("rtdetr_r50")          # 등록된 이름 -> 자동 다운로드 + 변환 + 캐시
model = Model("path/to/model.xml")   # OpenVINO IR 파일
model = Model("path/to/model.onnx")  # ONNX -> 최초 사용 시 IR 변환
```

모델을 지정하는 방법은 세 가지예요:

- **등록된 이름** (예: `"rtdetr_r50"`) — 모델 레지스트리(YAML 매니페스트)에서 찾습니다. 최초
  사용 시 소스에서 다운로드 → 필요하면 OpenVINO IR로 변환 → `~/.cache/ovkit`에 캐시. 다음
  실행부턴 캐시에서 바로 로드됩니다.
- **IR 경로** (`.xml`) — 그대로 사용.
- **ONNX 경로** (`.onnx`) — 로드 시 OpenVINO가 읽어 변환.

### 태스크 자동 감지

ovkit은 모델의 **태스크**를 알아내서 알맞은 디코더를 붙입니다. 순서대로 시도해요: 매니페스트의
`task` 필드 → IR `rt_info` 메타데이터 → 출력 텐서 모양 휴리스틱. 못 정하면 직접 지정하세요:

```python
model = Model("some_model.xml", task="detect")   # detect | classify | segment | pose | ocr
```

비전 태스크(`detect`, `classify`, `segment`, `pose`, `optical_character_recognition`)는
전용 디코더가 붙고, 그 외 태스크는 원시 출력 텐서를 돌려주는 제너릭 어댑터로 처리됩니다.

## 추론

모델 호출(`model(x)`)은 `model.predict(x)`와 같습니다. **입력 종류가 자동 감지**돼요:

```python
results = model("img.jpg", device="NPU", conf=0.25)   # 이미지 파일
results = model.predict("frames/", imgsz=640)         # 이미지 폴더
results = model.predict("clip.mp4")                   # 비디오 파일
for r in model.predict(0, stream=True):               # 웹캠 (카메라 인덱스)
    annotated = r.plot()
```

- `source`는 이미지 경로, `numpy` 배열(HWC BGR), 폴더, 비디오 파일, 카메라 인덱스(`int`)일 수
  있어요.
- `conf`는 검출/인스턴스 태스크의 신뢰도 임계값.
- `stream=True`는 지연 **제너레이터**를 반환(비디오·대용량 폴더에서 프레임을 하나씩 처리);
  아니면 {class}`~ovkit.Results`의 `list`를 받습니다.

비이미지 입력은 자동으로 원시 추론으로 라우팅됩니다 — `.npy` 텐서, `.wav` 파일, 비이미지
`ndarray`는 모델에 바로 들어가고 원시 `{이름: ndarray}` 출력이 반환돼요
([저수준](#저수준-모든-모델) 참고).

### `Results` 다루기

`Results`는 원본 이미지, 태스크, 그리고 태스크가 만든 출력을 담습니다. `r.plot()`으로 주석이
그려진 이미지를, `r.save(path)`로 저장하세요.

```python
r = results[0]
r.boxes.xyxy      # (N, 4) 픽셀 박스 [x1, y1, x2, y2]
r.boxes.conf      # (N,) 신뢰도
r.boxes.cls       # (N,) 클래스 id
r.name_for(2)     # "car"  (id -> 이름)
annotated = r.plot()   # -> 주석 ndarray (박스/마스크/키포인트/텍스트)
r.save("out.jpg")
```

| 속성           | 태스크      | 내용                                |
| -------------- | ----------- | ----------------------------------- |
| `r.boxes`      | detect      | `xyxy`, `xywh`, `conf`, `cls`       |
| `r.masks`      | segment     | `(N, H, W)` 마스크 (또는 클래스맵 1) |
| `r.keypoints`  | pose        | `(N, K, 3)` `[x, y, conf]`          |
| `r.probs`      | classify    | `top1`, `top5`, 확률                |
| `r.text`       | ocr         | 디코드된 문자열                     |
| `r.tensors`    | generic     | 원시 `{이름: ndarray}` 출력         |

(저수준-모든-모델)=
## 저수준 — 모든 모델

이미지를 받지 않는 모델(NLP / 오디오 / 시계열, 다중 입력)은 입력 텐서를 직접 만들어
{meth}`~ovkit.Model.infer`로 호출하세요:

```python
m = Model("bert_small_uncased_whole_word_masking_squad_0002")
print(m.inputs)                              # [(이름, shape, dtype), ...]
out = m.infer({"input_ids": ids, "attention_mask": mask})   # {이름: ndarray}
```

`m.inputs`가 모델이 기대하는 텐서를 정확히 알려줍니다.

## 디바이스

`device="AUTO"`(기본)는 OpenVINO가 최적 디바이스를 고르게 합니다. `"CPU"`, `"GPU"`,
`"NPU"`(Intel® Core™ Ultra 등)는 명시적 지정. `Model`에 설정하거나 호출마다 덮어쓸 수 있어요.
단일 이미지는 동기 실행, `stream=True`는 `AsyncInferQueue`로 처리량 모드입니다.

```python
from ovkit.core.backend import available_devices
print(available_devices())             # 예: ['CPU', 'GPU', 'NPU']
```

## 양자화 (INT8)

PTQ(학습 후 양자화)는 가중치/활성을 INT8로 바꿔 모델을 작고 빠르게 만들며, 대표 이미지 몇
장으로 보정합니다:

```python
model.quantize(calib_images, preset="int8")   # NNCF PTQ; INT8 IR 캐시됨
r = model("img.jpg")                           # 이제 INT8 모델로 추론
```

`pip install -e ".[quant]"` 필요.

## CLI

```bash
ovkit list                 # 등록된 모델 (이름 / 태스크 / 라이선스)
ovkit info rtdetr_r50      # 소스, 태스크, 라이선스, 정밀도
ovkit download rtdetr_r50  # 다운로드 + IR 변환 (캐시 워밍업)
ovkit devices              # 사용 가능한 OpenVINO 디바이스
```

## 모델 레지스트리

모델은 코드와 분리된 YAML 매니페스트(`src/ovkit/manifests/*.yaml`)에 있어서, **모델 추가는
한 줄 편집**이면 됩니다 — 파이썬 수정 불필요:

```yaml
rtdetr_r50:
  src: hf                         # hf | url | genai
  repo: leeyunjai/ovkit-models
  filename: detect/rtdetr_r50/model.xml
  task: detect
  precision: fp16
  license: apache-2.0             # 필수; permissive여야 함
  fallback:                       # 선택: 1순위 실패 시 시도
    src: hf
    repo: onnx-community/rtdetr_r50vd
    filename: onnx/model.onnx
```

`Model("name")` 해석 순서: 로컬 경로 → `$OVKIT_HOME` 캐시 IR → 매니페스트 소스에서 다운로드 →
IR 변환 → 캐시. 견고하게 설계됐어요 — **원자적 저장**(임시파일로 받고 성공 시 rename), 선택적
**`sha256` 무결성** 검사, `(name, precision)` 키 기반 **변환-1회 캐시**, 1순위가 죽으면
**업스트림 폴백**, 그리고 **오프라인 모드**.

### 환경 변수

| 변수              | 의미                                                |
| ----------------- | --------------------------------------------------- |
| `OVKIT_HOME`      | 캐시 루트 (기본 `~/.cache/ovkit`)                   |
| `OVKIT_OFFLINE`   | `1`이면 네트워크 차단, 캐시만 사용                  |
| `OVKIT_MANIFESTS` | 추가 매니페스트 경로 (`os.pathsep` 구분)            |

## GenAI (LLM / STT / TTS)

최신 OpenVINO 생성 모델은 **openvino-genai**로 처리하며, `ovkit.genai`로 재노출돼요(비전
파이프라인과 별개). extra를 설치하고 `pipeline(name)`을 호출하세요:

```python
from ovkit.genai import pipeline

llm = pipeline("tinyllama_chat")          # 다운로드 + 파이프라인 빌드
print(llm.generate("OpenVINO를 한 문장으로 설명해줘.", max_new_tokens=64))

stt = pipeline("whisper_base")            # 음성 -> 텍스트
print(stt.generate(audio_16k_mono_float32))
```

등록된 genai 모델은 `src/ovkit/manifests/genai.yaml`에 있어요. `pip install -e ".[genai]"`
필요.

## 라이선스 정책

ovkit은 Apache-2.0이고 라이선스가 깨끗합니다:

- 의존성·기본 모델에 **AGPL-3.0 모델 스택 없음**.
- 검출 기본은 DETR 계열(RT-DETR, RT-DETRv2, D-FINE, RF-DETR) — 전부 Apache-2.0.
- 얼굴 모델은 ovkit HF 미러 `leeyunjai/ovkit-models`의 Apache-2.0 OMZ 가중치 사용(폐기된
  `omz_downloader` 미사용).
- **InsightFace 사전학습 가중치 미사용** — 비상업 라이선스; 아키텍처 참고만.
- 모든 매니페스트 항목은 permissive `license`를 선언해야 하며, 비-permissive는 로드 시 거부됩니다.
