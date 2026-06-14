# 쿡북 — 모든 기능 호출 예시

각 ovkit 기능의 복붙용 예시입니다. (소스+venv 설치: `git clone ... && cd ovkit &&
python -m venv .venv && . .venv/bin/activate && pip install -e .`; genai는
`pip install -e ".[genai]"`, 양자화는 `pip install -e ".[quant]"`.)

## 모델 로드

```python
from ovkit import Model

m = Model("rtdetr_r50")             # 등록 이름 -> 자동 다운로드/변환/캐시
m = Model("path/to/model.xml")      # OpenVINO IR
m = Model("path/to/model.onnx")     # ONNX -> 즉석 IR 변환
m = Model("rtdetr_r50", device="NPU")            # AUTO | CPU | GPU | NPU
m = Model("some.xml", task="detect")             # 태스크 자동감지 덮어쓰기
m = Model("rtdetr_r50", precision="int8")        # IR 정밀도 지정
```

## 비전 태스크 (이미지 입력 → `Results`)

`model(x)`는 `list[Results]`(이미지당 하나)를 반환합니다. `r.plot()`은 주석이 그려진
`ndarray`, `r.save("out.jpg")`는 파일 저장.

```python
import cv2

# 검출 -> boxes
r = Model("rtdetr_r50")("street.jpg", conf=0.25)[0]
for x1, y1, x2, y2, conf, cls in r.boxes.data:
    print(r.name_for(int(cls)), float(conf), [int(x1), int(y1), int(x2), int(y2)])
print(r.boxes.xyxy, r.boxes.xywh, r.boxes.conf, r.boxes.cls)
r.save("det.jpg")

# 분류 -> probs
r = Model("resnet_18")("cat.jpg")[0]
print("top1:", r.name_for(r.probs.top1))
print("top5:", [r.name_for(int(i)) for i in r.probs.top5])

# 시맨틱 분할 -> masks (1, H, W) 클래스맵
r = Model("road_segmentation_adas_0001")("road.jpg")[0]
print(r.masks.data.shape)
cv2.imwrite("seg.jpg", r.plot())          # 컬러 오버레이

# 인스턴스 분할 -> boxes + 인스턴스별 마스크 (N, H, W)
r = Model("instance_segmentation_security_0002")("people.jpg")[0]
print(len(r.boxes), r.masks.data.shape)

# 포즈 -> keypoints (N, K, 3) = [x, y, conf]
r = Model("human_pose_estimation_0001")("person.jpg")[0]
print(r.keypoints.xy, r.keypoints.conf)

# OCR -> 디코드 텍스트
r = Model("text_recognition_0012")("word.png")[0]
print(r.text)

# 제너릭(초해상도, 임베딩, 액션 등) -> 원시 출력 텐서
r = Model("single_image_super_resolution_1032")("small.png")[0]
for name, arr in r.tensors.items():
    print(name, arr.shape, arr.dtype)
```

## 입력 (자동 감지)

```python
m = Model("rtdetr_r50")
m("img.jpg")                 # 파일 경로
m(cv2.imread("img.jpg"))     # HWC BGR ndarray
m("folder/")                 # 폴더 안 모든 이미지
m("clip.mp4")                # 비디오 파일
for r in m.predict(0, stream=True):   # 웹캠(카메라 인덱스), 지연 제너레이터
    annotated = r.plot()

# 비이미지 입력은 자동으로 원시 추론(dict 반환):
m("features.npy")            # 저장된 텐서
m("speech.wav")              # .wav (모델 입력 길이에 맞춤)
```

## 저수준 (모든 모델, 직접 텐서)

이미지를 받지 않는 NLP / 오디오 / 시계열 모델은 텐서를 직접 만들어 넣습니다:

```python
m = Model("bert_small_uncased_whole_word_masking_squad_0002")
print(m.inputs)                          # [(이름, shape, dtype), ...]
out = m.infer({"input_ids": ids, "attention_mask": mask})   # {이름: ndarray}
```

## 양자화 (INT8, NNCF)

```python
m = Model("rtdetr_r50")
m.quantize(["calib1.jpg", "calib2.jpg", ...], preset="int8")   # INT8 IR 캐시
r = m("img.jpg")             # 이제 INT8 모델로 추론
```

## 디바이스

```python
from ovkit.core.backend import available_devices
print(available_devices())                # ['CPU', 'GPU', 'NPU', ...]
Model("rtdetr_r50")("img.jpg", device="GPU")
```

## GenAI (LLM / STT / TTS) — `ovkit[genai]`

```python
from ovkit.genai import pipeline

llm = pipeline("tinyllama_chat")                       # 다운로드 + 빌드
print(llm.generate("OpenVINO를 한 문장으로.", max_new_tokens=64))

stt = pipeline("whisper_base")                         # 음성 -> 텍스트
print(stt.generate(audio_16k_mono_float32))

# 로컬 openvino-genai 모델 디렉토리(파이프라인 타입 지정):
tts = pipeline("/path/to/tts-ov", pipeline_type="text2speech")
```

## CLI

```bash
ovkit list                 # 등록 모델 (이름 / 태스크 / 라이선스)
ovkit info rtdetr_r50      # 소스, 태스크, 라이선스, 정밀도
ovkit download rtdetr_r50  # 다운로드 + IR 변환
ovkit devices              # OpenVINO 디바이스
```

## 레지스트리 / 매니페스트

```python
from ovkit.core.registry import list_models, resolve
print(list_models())
e = resolve("rtdetr_r50")
print(e.task, e.license, e.repo)
```

`src/ovkit/manifests/*.yaml`에 한 줄로 모델 추가:

```yaml
my_model:
  src: hf                       # hf | url | genai
  repo: leeyunjai/ovkit-models
  filename: detect/my_model/model.xml
  task: detect
  license: apache-2.0
  fallback: { src: hf, repo: onnx-community/..., filename: onnx/model.onnx }
```

환경 변수: `OVKIT_HOME`(캐시), `OVKIT_OFFLINE=1`(캐시 전용), `OVKIT_MANIFESTS`(추가 매니페스트).

## 미러 도구 (scripts/)

```bash
# 모델을 HF repo로 미러(OMZ + curated) 후 검증:
python scripts/build_mirror.py --repo leeyunjai/ovkit-models --omz-intel
python scripts/verify_mirror.py
# 미러를 가리키는 런타임 매니페스트 생성:
python scripts/build_mirror.py --omz-intel --emit-manifest src/ovkit/manifests/omz.yaml
```

## 예제 앱 (examples/)

```bash
pip install -r examples/requirements.txt
python examples/web_app.py                 # 이미지 업로드 / 웹캠 / 오디오 / 텍스트
python examples/predict.py rtdetr_r50 img.jpg --save out.jpg
python examples/denoise_audio.py noise_suppression_poconetlike_0001 in.wav out.wav
python examples/llm.py "Hello"             # ovkit[genai] 필요
python examples/stt.py speech.wav
```
