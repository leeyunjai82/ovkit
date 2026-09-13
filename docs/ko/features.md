# 기능

ovkit이 지금 할 수 있는 일 전부입니다. 조합 기능 19개, 모델 68개(추천 36개),
한국어 이름 44개. 각 항목이 무엇을 돌려주는지까지 함께 적었습니다.

```bash
pip install ovkit
```

추가 설치: `ovkit[genai]` (LLM/STT/VLM) · `ovkit[hf]` (허깅페이스 변환) ·
`ovkit[train]` (RT-DETR 학습) · `ovkit[quant]` (INT8) · `ovkit[all]`.

---

## `Model()` 하나로 들어가고 나온다

모델을 고르고, 내려받고, 변환하고, 전처리하고, 출력 텐서를 해석하는 일을 한 줄이 대신합니다.

```python
from ovkit import Model

r = Model("detect", "image.jpg")   # 내려받기 → 변환 → 캐시 → 실행
print(r)                           # 사람 2명, 자동차
r.save("out.jpg")                  # 상자 그려진 사진
```

### 받는 것

입력이 무엇이든 호출은 같습니다. 사진 하나면 결과 하나, 폴더면 결과 목록,
영상·웹캠·마이크면 결과가 흘러나옵니다.

| 입력 | 예 | 돌려주는 것 |
| ---- | -- | ----------- |
| 사진 경로 | `"a.jpg"` | `Results` 하나 |
| 넘파이 배열 | `ndarray` | `Results` 하나 |
| 폴더 | `"photos/"` | `Results` 목록 |
| 동영상 | `"clip.mp4"` | `Results` 흐름 |
| 웹캠 | `0` | `Results` 흐름 |
| 마이크 | `"mic"` | `Results` 흐름 |

```python
for r in Model("detect", "photos/"):     # 폴더 한 번에
    print(r.path, r)

for r in Model("track", 0):              # 웹캠은 흐름으로
    print(r, r.elapsed_ms, r.device)     # 사람 2명 (#1, #4)  14.2  GPU
    if not r.show("추적"):                # 창에 띄우고, q를 누르면 끝
        break
```

### 돌려주는 것 — `Results`

| | 무엇 |
| --- | --- |
| `print(r)` | **사람 2명, 자동차** — 한 줄 요약 |
| `r.found` | 찾은 것마다 `name` · `name_en` · `score` · `box` · `pos`(왼쪽 위 같은 9칸 위치) |
| `r.save(...)` | 그려서 저장. 배경 제거 결과면 투명 PNG로 |
| `r.plot()` | 겹치지 않는 자막 한 줄이 얹힌 그림 |
| `r.crop(i)` | i번째로 찾은 것만 잘라내기 |
| `r.show()` | 창에 띄우기. 창을 못 열면 파일로 저장하고 이유를 말합니다. `q`/`Esc`면 `False` |
| `r.to_dict()` / `r.to_json()` | 그대로 파일이나 서버로 |
| `r.boxes` `r.masks` `r.keypoints` `r.probs` `r.text` `r.tensors` | 숫자가 필요할 때의 원본 값 |
| `r.elapsed_ms` · `r.device` | 몇 ms 걸렸고 어디서 돌았는지 |

`r.elapsed_ms`와 `r.device`는 AI PC에서 그대로 수업 자료가 됩니다 — 같은 사진을
CPU와 NPU에 한 번씩 넣어보면 됩니다.

세밀하게 쓰고 싶으면 모델을 쥐고 있어도 됩니다.

```python
m = Model("detect", device="GPU")
m.predict("a.jpg", conf=0.4, imgsz=960)
m.infer(x)                # 원본 출력 텐서 그대로
```

---

## 조합 기능 19개

검출한 뒤 잘라서 다른 모델에 넣고 결과를 합치는 일 — 초급자가 막히는 그 구간을
기능 이름 하나가 대신합니다. 입력 종류도 `Results`도 위와 똑같습니다.

| `Model(...)` | 한국어 | 돌려주는 답 | 엮는 것 |
| ------------ | ------ | ----------- | ------- |
| `scene` | 장면설명 | 사람 2명(1명 웃음) · 노트북과 컵 · 바닥 47% | 검출 + 영역분할 + 얼굴 |
| `face_analyze` | 얼굴분석 | 얼굴 2개: 31세 · 남 98% · 기쁨 92% … | 얼굴검출 + 나이/성별 + 감정 |
| `person_analyze` | 사람분석 | 사람 3명: 남 0.98 · 긴바지 0.95 · 가방 0.71 … | 사람검출 + 속성 |
| `vehicle_analyze` | 차량분석 | 차량 2대: 승용차(0.98) · 검정(0.83) … | 차량검출 + 종류/색 |
| `read_text` | 글자읽기 | `'여기서 기다리세요'` — 읽는 순서대로 | 글자검출 + 글자인식(언어에 맞는 것으로) |
| `read_plate` | 번호판읽기 | 차량 2대: 검정 승용차 — 12GA3456 … | 번호판검출 + 인식 + 차량속성 |
| `speak` | 읽어주기 | 글을 사람 목소리로 — `r.save('안녕.wav')` | 길이예측 + 글자인코딩 + 흐름정합 + 보코더 |
| `track` | 따라가기 | 사람 2명 (#1, #4) — 프레임이 바뀌어도 번호 유지 | 검출 + IoU 연결 |
| `count` | 개수세기 | 연필 3 · 컵 1 (한 종류만 셀 수도) | 검출 + 종류별 집계 |
| `drowsiness` | 졸음감지 | 눈 감김 1.4초 — 졸음 | 얼굴 + 랜드마크 + 눈상태 + 머리방향 · 시간축 |
| `posture` | 거북목알림 | 목 32° — 허리 펴세요 (6초) | 자세 + 목 각도 · 시간축 |
| `exercise` | 운동횟수 | 스쿼트 12회 (내려가는 중) | 자세 + 관절각 히스테리시스 |
| `gesture` | 동작알아보기 | 엄지척 0.94 | 수어 모델 + 8프레임 묶음 |
| `gaze` | 시선 | 얼굴 1개: 오른쪽 위를 보는 중 | 얼굴 + 랜드마크 + 머리방향 + 시선 |
| `attention` | 뭘보나 | 사람 1명이 보는 것: 노트북 | 시선 + 물체검출(상자로 광선 쏘기) |
| `anonymize` | 모자이크 | 얼굴(과 번호판)이 전부 가려진 사진 | 얼굴·번호판 검출 + 가림 |
| `face_match` | 누구지 | `('윤재', 0.81)` — 내가 만든 명단에서 | 임베딩 + 코사인 유사도 |
| `attendance` | 출석체크 | 출석 24/26 + `roll.csv` | 얼굴검출 + 명단 대조 |
| `teach` | 가르치기 | 내가 정한 이름으로 분류 | 임베딩 + 최근접 이웃 |
| `anomaly` | — | 이 부품이 정상인지 이상인지 점수로 | anomalib 모델 |

모델 하나로 끝나지만 답이 문장으로 나오는 것도 같은 자리에 있습니다.

| `Model(...)` | 한국어 | 돌려주는 답 |
| ------------ | ------ | ----------- |
| `depth` | 거리재기 | 가장 가까운 곳: 왼쪽 아래 · 화면의 34%가 가까움 + 색지도 |
| `remove_background` | 배경지우기 | 피사체만 남은 투명 PNG |


### 읽어주기 — 글을 소리로

```python
r = Model("읽어주기", "안녕하세요. 오늘은 기계 학습을 배웁니다.")
r.save("인사.wav")          # 44.1 kHz WAV
r.plot()                    # 파형 그림
```

한국어를 포함해 **31개 언어**를 읽습니다. 언어는 글에서 알아서 고르고,
`lang="en"`처럼 직접 정할 수도 있습니다.

```python
Model("읽어주기", "Good morning.", voice="M3")   # 목소리 열 개: F1~F5, M1~M5
Model("읽어주기", "안녕하세요", speed=1.3)        # 빠르게
Model("읽어주기", "안녕하세요", steps=2)          # 거칠고 빠르게 (기본 8)
Model("읽어주기", "원고.txt")                     # 파일도 읽어 줍니다
```

네트워크 **네 개**가 이어 달립니다. 문장이 몇 초짜리인지 예측하고
(`duration_predictor`), 글을 인코딩하고(`text_encoder`), 그 길이만큼의
잡음을 문장 쪽으로 몇 번 밀어내고(`vector_estimator`, `steps`번 반복),
마지막에 파형으로 바꿉니다(`vocoder`). 노트북 CPU에서 문장 하나가 1초
아래입니다.

같은 문장은 같은 소리가 납니다(`seed=0`이 기본). 매번 다르게 하려면
`seed=None`.

`ovkit gui` 창에서도 왼쪽 목록에 **읽어주기**가 있습니다. 글을 쓰고 목소리를
고른 뒤 `읽기`를 누르면 파형이 나오고, `Save`로 `.wav`을 씁니다.

> **라이선스**: 모델은 Supertone Inc.의 **OpenRAIL-M**입니다. 상업적 이용과
> 재배포가 되지만, 사본을 넘길 때 **같은 사용 제한이 함께 가야 합니다**.
> 미러에 `LICENSE`가 같이 올라가 있고, 쓰기 전에 한 번 읽어 보세요.

`gesture`, `drowsiness`, `posture`, `exercise`는 **시간이 필요합니다** — 사진 한 장이
아니라 웹캠이나 동영상을 넣어야 답이 나옵니다.

줄임 이름도 받습니다: `ocr` `anpr` `blur` `driver` `describe` `faces` `people`
`vehicle` `tracking` `reid` `tts` `say`. 무엇이 있는지는 `list_pipelines()` 또는
`ovkit capabilities`.

```python
from ovkit import Model, list_pipelines

list_pipelines()                                    # 전부, 설명과 함께
Model("read_text")("sign.jpg").text              # 'STOP AHEAD'
Model("face_analyze", attributes=("age_gender",))   # 무엇까지 돌릴지 고르기
```

---

## 모델

허용 라이선스(Apache-2.0/MIT 계열)만 등록합니다 — AGPL이나 비상업 가중치는 불러오는
순간 거부됩니다. 전부 한 저장소 `leeyunjai/ovkit-models`에서 내려옵니다.
아래는 추천 35개이고, 나머지 32개는 `ovkit list --all`로 이름을 불러 씁니다.

### 물체 검출

| 모델 | 무엇 |
| ---- | ---- |
| `rtdetr_r18` | COCO 80종, NMS 없음. 가장 빠름 — `detect` 기본값 |
| `rtdetr_r34` | r18이 작은 물체를 놓칠 때 |
| `rtdetr_r50` | 정지 사진 · 정확도 작업 |
| `rtdetr_r101` | 가장 큼 |
| `face_detection_0205` | 일반 장면 얼굴 |
| `person_detection_0202` | 감시 장면 사람 |
| `vehicle_detection_0200` | 감시 장면 차량 |
| `text_detection_0004` | 간판·문서의 글자 영역 |
| `vehicle_license_plate_detection_barrier_0106` | 차량 + 번호판 |

### 얼굴

| 모델 | 무엇 |
| ---- | ---- |
| `age_gender_recognition_retail_0013` | 나이(0–100)와 성별 |
| `emotions_recognition_retail_0003` | 무표정 · 기쁨 · 슬픔 · 놀람 · 화남 |
| `facial_landmarks_98_detection_0001` | 98점 — 눈꺼풀 · 입술 윤곽 · 턱선 |
| `landmarks_regression_retail_0009` | 5점 — 눈 · 코끝 · 입꼬리 |
| `head_pose_estimation_adas_0001` | 고개 각도 yaw / pitch / roll |
| `face_reidentification_retail_0095` | 얼굴 256차원 임베딩 |

### 분류 · 임베딩

| 모델 | 무엇 |
| ---- | ---- |
| `nfnet_f0` | 이미지넷 1000종 분류 |
| `person_reidentification_retail_0287` | 사람 256차원 — 얼굴이 안 보여도 같은 사람 |
| `image_retrieval_0001` | 장면 임베딩(유사 이미지 찾기) |
| `person_attributes_recognition_crossroad_0234` | 가방 · 모자 · 소매 같은 보행자 속성 |
| `vehicle_attributes_recognition_barrier_0042` | 차종과 색 |
| `gaze_estimation_adas_0002` | 눈 크롭 + 고개 각도 → 시선 |
| `open_closed_eye_0001` | 눈 떴는지 감았는지 |

### 영역 · 자세 · 깊이 · 이미지 처리

| 모델 | 무엇 |
| ---- | ---- |
| `pspnet_pytorch` | 21종 의미 분할(Pascal VOC) |
| `instance_segmentation_person_0007` | 사람별 마스크 |
| `human_pose_estimation_0007` | 여러 명 17개 관절 |
| `depth_anything_v2_small` | 한 장에서 거리 추정 |
| `u2net` | 피사체와 배경 분리 |
| `single_image_super_resolution_1033` | 3배 확대 (4배는 `super_resolution_4x`) |
| `fast_neural_style_mosaic_onnx` | 모자이크풍 화풍 변환 |

### 글자 · 소리 · 동작

| 모델 | 무엇 |
| ---- | ---- |
| `korean_text_recognition` | **한국어 글자 인식** (PP-OCRv3, CTC) — `read_text`가 한국어일 때 쓰는 것 |
| `text_recognition_0014` | 잘라낸 영문·숫자 읽기(CTC) |
| `common_sign_language_0002` | 손동작 12종 (영상 클립) |
| `noise_suppression_poconetlike_0001` | 16 kHz 음성 잡음 제거 |

보관(`zoo`) 쪽에 `aclnet`(환경음 53종), `bert_small…squad_0002`(지문에서 답 찾기),
`machine_translation_nar_en_de_0002`(번역), `time_series_forecasting_electricity_0001`
같은 것들이 이름으로 그대로 남아 있습니다.

---

## 내가 가르치는 AI

임베딩 모델이 예시를 벡터로 바꾸고, 새 입력은 가장 가까운 것으로 답합니다.
학습이 아니라 비교라서 GPU 없이 노트북에서 몇 초면 끝나고, 결과는 JSON 한 장입니다.

```python
ai = Model("teach")                  # 또는 "가르치기"
ai.learn("캔", "photos/cans/")        # 종류마다 폴더 하나
ai.learn("병", "photos/bottles/")

ai.guess("new.jpg")                  # ('캔', 0.93)
ai.score("test/")                    # 정확도 + 헷갈리는 쌍
ai.save("recycling")                 # 문서/ovkit/recycling.json

for r in ai.predict(0, stream=True): # 웹캠에서 내 분류로
    print(r)
```

모드 5가지: `photo`(사진, 기본) · `face`(표정) · `hand`(손모양, `ovkit[hand]` 필요) ·
`upper`(상반신) · `body`(전신). 메서드도 한국어로 부를 수 있습니다 —
`배우기` `맞혀봐` `점수` `저장`.

예시는 웹캠으로 모읍니다:

```python
from ovkit.pipelines.teach import collect
collect("가위", 30)
```

---

## 한국어

기능 이름 44개와 클래스 이름 약 200종이 한글로 번역되어 있습니다. 결과에는 언제나
영어 원본 `name_en`이 함께 들어 있어, 코드를 한글에 묶어두지 않습니다.
`OVKIT_LANG=en`이면 전부 영어로 바뀝니다.

```python
r = Model("얼굴분석", "group.jpg")
for row in r.found:
    print(row["name"], row["name_en"], row["pos"])
    # 사람 person 왼쪽 위
```

| 한국어 | → | 한국어 | → |
| ------ | - | ------ | - |
| 물체찾기 | `detect` | 장면설명 | `scene` |
| 얼굴찾기 | `face_detection` | 얼굴분석 | `face_analyze` |
| 사람찾기 | `person_detection` | 사람분석 | `person_analyze` |
| 차찾기 | `vehicle_detection` | 차량분석 | `vehicle_analyze` |
| 글자찾기 | `text_detection` | 글자읽기 | `read_text` |
| 번호판읽기 | `read_plate` | 개수세기 | `count` |
| 따라가기 · 추적 | `track` | 누구지 | `face_match` |
| 졸음감지 | `drowsiness` | 거북목 · 거북목알림 | `posture` |
| 운동횟수 | `exercise` | 동작알아보기 · 제스처 | `gesture` |
| 시선 | `gaze` | 뭘보나 | `attention` |
| 얼굴가리기 · 모자이크 | `anonymize` | 출석 · 출석체크 | `attendance` |
| 가르치기 · 내가가르치기 | `teach` | 이건뭐야 · 분류 | `classify` |
| 거리재기 · 깊이 | `depth` | 배경지우기 | `remove_background` |
| 영역나누기 | `segment` | 자세 | `pose` |
| 나이성별 | `age_gender` | 감정 | `emotion` |
| 화질좋게 | `super_resolution` | 그림풍바꾸기 | `style_transfer` |
| 무슨소리 | `sound_classification` | 잡음제거 | `noise_suppression` |
| 받아쓰기 | `stt` | 대화 | `llm` |

---

## GenAI — 말하고 듣고 설명하는 쪽

`pip install "ovkit[genai]"`. 이쪽은 `Model()`이 아니라 `ovkit.genai`를 씁니다 —
OpenVINO IR이 아니라 openvino-genai 파이프라인이기 때문입니다.

```python
from ovkit.genai import pipeline, transcribe, describe

llm = pipeline("llm")                       # Qwen2.5 1.5B INT4
llm.generate("안녕", max_new_tokens=50)

transcribe("수업녹음.wav")                   # Whisper base
describe("사진.jpg", "이 사진 설명해줘")      # Qwen3-VL 4B
```

---

## 등록되지 않은 모델도

### 허깅페이스에서 바로

`optimum-intel`이 거의 모든 허브 모델을 OpenVINO IR로 바꿉니다. 무거운 건 변환
한 번뿐이고, 그 뒤로는 **ovkit 말고 아무것도 필요 없습니다** — 교실 컴퓨터에
캐시만 건네주면 PyTorch를 볼 일이 없습니다. 모델의 클래스 이름과 전처리
정규화 값도 함께 받아 적으므로, 결과가 엉뚱한 자신감을 갖지 않습니다.

```bash
pip install "ovkit[hf]"
ovkit pull google/vit-base-patch16-224
```

```python
Model("google/vit-base-patch16-224", "photo.jpg")   # tabby 0.94
```

되는 작업은 이미지 분류, 제로샷(CLIP) 분류, 특징 추출입니다.
**검출과 분할은 안 됩니다** — optimum-intel이 내보내지 못해서, 그쪽은 등록된
모델(`detect`, `segment`)로 갑니다.

### 내 데이터로 검출기 학습

RT-DETR을 직접 학습시켜 같은 `Model()`로 씁니다. YOLO 형식 `data.yaml`을 그대로 읽습니다.

```bash
pip install "ovkit[train]"

ovkit train --data data.yaml --epochs 100
ovkit val   --data data.yaml --weights best.pt
ovkit export best.pt                    # → OpenVINO IR + labels.txt
```

```python
Model("best_openvino/model.xml", "test.jpg")
```

### INT8 양자화

```python
Model("detect").quantize(calib_data)     # ovkit[quant]
```

NPU에서 특히 값이 큽니다.

---

## 터미널

| 명령 | 하는 일 |
| ---- | ------- |
| `ovkit gui` | 창이 열립니다. 기능을 고르고 웹캠을 겨누면 끝 — 가장 쉬운 시작 |
| `ovkit run detect image.jpg` | 결과를 찍고 `image_out.jpg`로 저장 |
| `ovkit capabilities` | `Model(이름)`으로 부를 수 있는 조합 기능 목록 |
| `ovkit list` / `ovkit list --all` | 추천 모델 / 전체 등록 이름 |
| `ovkit info <모델>` | 출처 · 라이선스 · 입력 크기 · 전처리 |
| `ovkit download <모델>` | 미리 받아두기 (수업 전날에) |
| `ovkit pull <허브ID>` | 허깅페이스 모델을 IR로 변환 |
| `ovkit devices` | 이 컴퓨터에서 쓸 수 있는 장치 |
| `ovkit train` / `val` / `export` | RT-DETR 학습 · 평가 · 내보내기 |

---

## 실행 환경

| | |
| --- | --- |
| `device="AUTO"` | 기본값. `CPU` · `GPU` · `NPU` 직접 지정 가능. 결과마다 어디서 돌았는지(`r.device`)와 몇 ms인지(`r.elapsed_ms`)가 붙습니다 |
| 흐름 처리 | 폴더 · 영상 · 웹캠은 `AsyncInferQueue`로 겹쳐 돌립니다 — 한 장씩 기다리지 않습니다 |
| `OVKIT_HOME` | 캐시 위치. 기본 `~/.cache/ovkit`. 변환 결과는 `(이름, 정밀도)`로 한 번만 만들어 재사용 |
| `OVKIT_OFFLINE=1` | 네트워크를 아예 쓰지 않습니다. 캐시를 통째로 복사해 넣은 교실 컴퓨터용 |
| `OVKIT_LANG` | `ko`(기본) / `en` |
| 모델 저장소 | 전부 `leeyunjai/ovkit-models` 한 곳. 원본 저장소는 예비로만 — 오프라인 학교가 복제할 곳이 하나여야 하니까 |
| 무결성 | `sha256` 검사, IR과 가중치(`.bin`)를 짝으로 받고 한쪽이 없으면 컴파일 전에 거부 |

---

## 글자 읽기와 언어

`read_text`는 표시 언어에 맞는 인식기를 고릅니다 — 한국어면 PP-OCRv3 한국어
모델, 아니면 기존 라틴 모델입니다. 직접 지정하면 그게 이깁니다.

```python
Model("read_text")("간판.jpg")                        # 한국어 (OVKIT_LANG=ko)
Model("read_text", recognizer="text_recognition")     # 영문·숫자로 강제
```

한국어 모델을 못 불러오면(오프라인, 캐시 없음) 한 번 경고하고 라틴 모델로
내려갑니다 — 조용히 빈 문자열을 돌려주지 않습니다.

## 아직 없는 것

| | |
| --- | --- |
| 손 랜드마크 | `teach`의 `hand` 모드는 별도 설치(`ovkit[hand]`)가 필요합니다 |
| 검출 · 분할 변환 | 허브에서 가져올 수 있는 건 분류 · 임베딩까지입니다 |
| 최근 등록 7종 | 미러에 묻혀 있던 모델을 되살려 등록했지만 매니페스트로만 확인했습니다. `rtdetr_r101`과 `human_pose_estimation_0005`는 먼저 한 장 넣어보고 쓰세요 |
