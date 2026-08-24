---
sd_hide_title: true
---

# ovkit

:::{div} ovk-hero
# ovkit

```{div} ovk-tagline
OpenVINO를 위한 간단한 파이썬 추론 API — `import` 하나, `Model` 클래스 하나, 호출 가능한
객체, 깔끔한 `Results`. 여기에 `AUTO`/`NPU` 디바이스, async 처리량, INT8 양자화까지.
```

```{div} ovk-badges
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![OpenVINO](https://img.shields.io/badge/OpenVINO-IR%20%7C%20genai-0a7d8c)
```
:::

```python
from ovkit import Model

model = Model("rtdetr_r50")            # 이름 -> 자동 다운로드 / 변환 / 캐시
for r in model("img.jpg", conf=0.25):  # __call__ == predict
    print(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)
    r.save("out.jpg")
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🚀 시작하기
:link: usage
:link-type: doc
설치, 모델 로드, 추론 실행, 디바이스 선택.
:::

:::{grid-item-card} 📖 쿡북
:link: cookbook
:link-type: doc
모든 기능의 복붙 호출 예시.
:::

:::{grid-item-card} 🧩 API 레퍼런스
:link: api
:link-type: doc
`Model`, `Results`, 레지스트리, 어댑터, 이미지 유틸.
:::
::::

## 기능(capability) 한 줄 호출

`Model()`에 기능 이름을 주면 필요한 모델을 알아서 이어 붙여 **답**을 돌려줍니다.
검출하고, 자르고, 속성 모델을 돌리고, 합치는 코드를 직접 쓸 필요가 없습니다.

```python
from ovkit import Model, list_pipelines

list_pipelines()                                   # 사용 가능한 기능 목록
Model("face_analyze")("group.jpg")[0].summary()    # '2 faces: age 31 · male 98% · happy 92%, ...'
Model("read_text")("sign.jpg")[0].text             # 'STOP AHEAD'
Model("track")(0)                                  # 웹캠, 프레임이 바뀌어도 같은 id 유지
```

| `Model(...)` | 결과 | 내부 구성 |
| ------------ | ---- | --------- |
| `face_analyze` | 얼굴마다 나이·성별·감정 | 얼굴 검출 + age/gender + emotion |
| `person_analyze` | 사람마다 착용/소지 속성 | 사람 검출 + 속성 |
| `vehicle_analyze` | 차량마다 종류·색상 | 차량 검출 + 속성 |
| `read_text` | 읽기 순서대로 모든 단어 | 텍스트 검출 + 인식 |
| `track` | 프레임 간 유지되는 객체 id | 검출 + IoU 연결 |
| `gaze` | 어디를 보고 있는지 | 얼굴 검출 + 랜드마크 + 머리 자세 + 시선 |
| `face_match` | 직접 만든 갤러리에서 누구인지 | 임베딩 + 코사인 매칭 |

## 무엇을 하나

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 🎯 비전 태스크
검출(DETR / SSD / YOLO), 분류, 분할(시맨틱+인스턴스), 포즈, OCR — `model(img)` →
타입이 정해진 `Results`.
:::

:::{grid-item-card} ⚙️ 모든 모델
타입 디코더가 없는 모델은 원시 텐서로 반환 + 저수준 `model.infer(tensors)`로 NLP /
오디오 / 시계열까지.
:::

:::{grid-item-card} 💬 GenAI
LLM / Whisper(STT) / TTS를 `ovkit.genai.pipeline(...)`로 (openvino-genai).
:::

:::{grid-item-card} 📦 자동 처리
자동 다운로드 + IR 변환 + 캐시, 태스크 자동 감지, 입력 자동 라우팅(이미지 / `.npy` /
`.wav`), INT8 양자화.
:::
::::

ovkit은 **Apache-2.0**이고 라이선스가 깨끗합니다 — AGPL 모델 스택이나 비상업 가중치를
번들/다운로드하지 않아요. [라이선스 정책](usage.md#라이선스-정책) 참고.

```{toctree}
:hidden:

usage
guide
cookbook
models
api
```

```{toctree}
:hidden:
:caption: 프로젝트

genindex
GitHub <https://github.com/leeyunjai82/ovkit>
```
