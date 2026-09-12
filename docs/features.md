# Features

Everything ovkit can answer today: 19 composed capabilities, 67 models (35
curated), 44 Korean names — and what each one hands back.

```bash
pip install ovkit
```

Extras: `ovkit[genai]` (LLM/STT/VLM) · `ovkit[hf]` (Hugging Face conversion) ·
`ovkit[train]` (RT-DETR training) · `ovkit[quant]` (INT8) · `ovkit[all]`.

---

## One call in, one answer out

Choosing a model, downloading it, converting it, preprocessing, and decoding
output tensors — one line does all of it.

```python
from ovkit import Model

r = Model("detect", "image.jpg")   # download -> convert -> cache -> run
print(r)                           # 2x person, car
r.save("out.jpg")                  # the boxes drawn on the photo
```

### What it takes

The call is the same whatever you point it at. A photo gives one result, a
folder a list, a video or camera or microphone a stream.

| Source | Example | You get |
| ------ | ------- | ------- |
| image path | `"a.jpg"` | one `Results` |
| numpy array | `ndarray` | one `Results` |
| folder | `"photos/"` | a list of `Results` |
| video | `"clip.mp4"` | a stream |
| webcam | `0` | a stream |
| microphone | `"mic"` | a stream |

```python
for r in Model("detect", "photos/"):     # a whole folder
    print(r.path, r)

for r in Model("track", 0):              # webcam, as a stream
    print(r, r.elapsed_ms, r.device)     # 2x person (#1, #4)  14.2  GPU
```

### What it hands back — `Results`

| | What |
| --- | --- |
| `print(r)` | **2x person, car** — one readable line |
| `r.found` | per detection: `name` · `name_en` · `score` · `box` · `pos` (a 9-cell position such as "top left") |
| `r.save(...)` | draw and write. A background-removal result saves as a transparent PNG |
| `r.plot()` | the picture with one wrapped caption that never overlaps |
| `r.crop(i)` | just the i-th thing it found |
| `r.to_dict()` / `r.to_json()` | straight to a file or a server |
| `r.boxes` `r.masks` `r.keypoints` `r.probs` `r.text` `r.tensors` | the raw values, when you want numbers |
| `r.elapsed_ms` · `r.device` | how long it took and where it ran |

On an AI PC that last pair is teaching material on its own — run the same photo
on CPU and on NPU and read the two numbers.

Hold the model when you want the detail:

```python
m = Model("detect", device="GPU")
m.predict("a.jpg", conf=0.4, imgsz=960)
m.infer(x)                # raw output tensors
```

---

## 19 composed capabilities

Detect, crop, feed the crop to another model, stitch the answers together —
the part beginners get stuck on is what a capability name replaces. Same
sources, same `Results` as above.

| `Model(...)` | Korean | Answers | Chains |
| ------------ | ------ | ------- | ------ |
| `scene` | 장면설명 | 2 people (1 happy) · a laptop and a cup · floor 47% | detection + segmentation + faces |
| `face_analyze` | 얼굴분석 | 2 faces: age 31 · male 98% · happy 92% … | face detection + age/gender + emotion |
| `person_analyze` | 사람분석 | 3 people: male 0.98 · long pants 0.95 · bag 0.71 … | person detection + attributes |
| `vehicle_analyze` | 차량분석 | 2 vehicles: car (0.98) · black (0.83) … | vehicle detection + type/colour |
| `read_text` | 글자읽기 | `'STOP AHEAD'` — every word, in reading order | text detection + recognition |
| `read_plate` | 번호판읽기 | 2 vehicles: black car — 12GA3456 … | plate detection + recognition + attributes |
| `track` | 따라가기 | 2x person (#1, #4) — ids stable across frames | detection + IoU association |
| `count` | 개수세기 | pencil 3 · cup 1 (optionally one kind only) | detection + per-kind tally |
| `drowsiness` | 졸음감지 | EYES CLOSED 1.4s — drowsy | face + landmarks + eye state + head pose, over time |
| `posture` | 거북목알림 | neck 32° — sit up (6s) | pose + neck angle, over time |
| `exercise` | 운동횟수 | squat x 12 (down) | pose + joint-angle hysteresis |
| `gesture` | 동작알아보기 | thumb up 0.94 | sign-language model over a rolling 8-frame clip |
| `gaze` | 시선 | 1 face: looking right and slightly up | face + landmarks + head pose + gaze |
| `attention` | 뭘보나 | 1 person looking at: laptop | gaze + detection (ray-cast into the boxes) |
| `anonymize` | 모자이크 | the picture with every face pixelated | face (and plate) detection + redaction |
| `face_match` | 누구지 | `('yunjai', 0.81)` — from a gallery you build | embedding + cosine matching |
| `attendance` | 출석체크 | present 24/26 + `roll.csv` | face detection + roster matching |
| `teach` | 가르치기 | your own classes, from example photos | embedding + nearest neighbour |
| `anomaly` | — | this part scored normal or anomalous | anomalib model |

Two more are a single model, but they answer in sentences the same way:

| `Model(...)` | Korean | Answers |
| ------------ | ------ | ------- |
| `depth` | 거리재기 | nearest: bottom-left · 34% of the frame is close, plus a colour map |
| `remove_background` | 배경지우기 | the subject, as a transparent PNG |

`gesture`, `drowsiness`, `posture` and `exercise` **need time** — point them at
a webcam or a video, not a still.

Aliases: `ocr` `anpr` `blur` `driver` `describe` `faces` `people` `vehicle`
`tracking` `reid`. To see what exists: `list_pipelines()` or
`ovkit capabilities`.

```python
from ovkit import Model, list_pipelines

list_pipelines()                                    # every capability, described
Model("read_text")("sign.jpg")[0].text              # 'STOP AHEAD'
Model("face_analyze", attributes=("age_gender",))   # configure what runs
```

---

## Models

Only permissive licences (Apache-2.0 / MIT family) are registered — AGPL and
non-commercial weights are refused at load time. Everything comes from one
repository, `leeyunjai/ovkit-models`. Below are the 35 curated ones; the other
32 load by name and are listed with `ovkit list --all`.

### Detection

| Model | What |
| ----- | ---- |
| `rtdetr_r18` | COCO-80, NMS-free. Fastest — the `detect` default |
| `rtdetr_r34` | when r18 misses small objects |
| `rtdetr_r50` | stills and accuracy work |
| `rtdetr_r101` | the largest |
| `face_detection_0205` | faces in general scenes |
| `person_detection_0202` | people, surveillance scenes |
| `vehicle_detection_0200` | vehicles, surveillance scenes |
| `text_detection_0004` | text regions on signs and documents |
| `vehicle_license_plate_detection_barrier_0106` | vehicle + plate |

### Faces

| Model | What |
| ----- | ---- |
| `age_gender_recognition_retail_0013` | age (0–100) and gender |
| `emotions_recognition_retail_0003` | neutral · happy · sad · surprise · anger |
| `facial_landmarks_98_detection_0001` | 98 points — eyelid and lip contours, jawline |
| `landmarks_regression_retail_0009` | 5 points — eyes, nose tip, mouth corners |
| `head_pose_estimation_adas_0001` | yaw / pitch / roll |
| `face_reidentification_retail_0095` | 256-d face embedding |

### Classification and embeddings

| Model | What |
| ----- | ---- |
| `nfnet_f0` | ImageNet 1000-class classifier |
| `person_reidentification_retail_0287` | 256-d person embedding — the same person with their face turned away |
| `image_retrieval_0001` | scene embedding (find similar images) |
| `person_attributes_recognition_crossroad_0234` | bag, hat, sleeves and the rest |
| `vehicle_attributes_recognition_barrier_0042` | type and colour |
| `gaze_estimation_adas_0002` | eye crops + head pose → gaze |
| `open_closed_eye_0001` | eye open or closed |

### Segmentation, pose, depth, image processing

| Model | What |
| ----- | ---- |
| `pspnet_pytorch` | 21-class semantic segmentation (Pascal VOC) |
| `instance_segmentation_person_0007` | per-person masks |
| `human_pose_estimation_0007` | multi-person, 17 keypoints |
| `depth_anything_v2_small` | depth from a single photo |
| `u2net` | subject cut from its background |
| `single_image_super_resolution_1033` | 3x upscale (4x is `super_resolution_4x`) |
| `fast_neural_style_mosaic_onnx` | mosaic style transfer |

### Text, sound, motion

| Model | What |
| ----- | ---- |
| `text_recognition_0014` | read a cropped word (CTC) |
| `common_sign_language_0002` | 12 hand gestures (video clip) |
| `noise_suppression_poconetlike_0001` | 16 kHz speech denoising |

The `zoo` tier keeps `aclnet` (53 environmental sounds),
`bert_small…squad_0002` (extractive QA),
`machine_translation_nar_en_de_0002` and
`time_series_forecasting_electricity_0001` loadable by name.

---

## Teach your own AI

An embedding model turns examples into vectors; a new input matches the nearest
one. It compares rather than trains, so it finishes in seconds on a laptop with
no GPU, and what it saves is a single JSON file.

```python
ai = Model("teach")                  # or Model("가르치기")
ai.learn("can", "photos/cans/")      # a folder per thing to recognise
ai.learn("bottle", "photos/bottles/")

ai.guess("new.jpg")                  # ('can', 0.93)
ai.score("test/")                    # accuracy + what it confuses
ai.save("recycling")                 # -> Documents/ovkit/recycling.json

for r in ai.predict(0, stream=True): # webcam, over YOUR classes
    print(r)
```

Five modes: `photo` (default) · `face` (expressions) · `hand` (needs
`ovkit[hand]`) · `upper` · `body`. Korean method names work too:
`배우기` `맞혀봐` `점수` `저장`.

Collect examples with the webcam:

```python
from ovkit.pipelines.teach import collect
collect("scissors", 30)
```

---

## Korean

44 capability names and around 200 class names are translated. Every result
also carries the stable English `name_en`, so your code is never tied to the
Korean. `OVKIT_LANG=en` switches the display back to English.

```python
r = Model("얼굴분석", "group.jpg")
for row in r.found:
    print(row["name"], row["name_en"], row["pos"])
    # 사람 person 왼쪽 위
```

See the Korean page for the full table of names.

---

## GenAI — talking, listening, describing

`pip install "ovkit[genai]"`. This side uses `ovkit.genai`, not `Model()`:
these are openvino-genai pipelines rather than OpenVINO IR.

```python
from ovkit.genai import pipeline, transcribe, describe

llm = pipeline("llm")                     # Qwen2.5 1.5B INT4
llm.generate("Hello", max_new_tokens=50)

transcribe("lecture.wav")                 # Whisper base
describe("photo.jpg", "what is happening here?")   # Qwen3-VL 4B
```

---

## Models that aren't registered

### Straight from the Hugging Face Hub

`optimum-intel` converts almost any Hub model to OpenVINO IR. Converting is the
heavy half and happens once; what lands in the cache is a plain OpenVINO model,
so **running it afterwards needs nothing but ovkit** — a classroom machine can
be handed the cache and never see PyTorch. The pull also writes the model's own
class names and the normalisation its image processor declares, so results read
as answers and not as confident nonsense.

```bash
pip install "ovkit[hf]"
ovkit pull google/vit-base-patch16-224
```

```python
Model("google/vit-base-patch16-224", "photo.jpg")   # tabby 0.94
```

Tasks: image classification, zero-shot (CLIP) classification, feature
extraction. **Detection and segmentation are not available** — optimum-intel
does not export them, so those come from the registry (`detect`, `segment`).

### Train your own detector

Train RT-DETR on your own data and load it through the same `Model()`. It reads
a YOLO-format `data.yaml` as is.

```bash
pip install "ovkit[train]"

ovkit train --data data.yaml --epochs 100
ovkit val   --data data.yaml --weights best.pt
ovkit export best.pt                    # -> OpenVINO IR + labels.txt
```

```python
Model("best_openvino/model.xml", "test.jpg")
```

### INT8 quantization

```python
Model("detect").quantize(calib_data)     # ovkit[quant]
```

Worth the most on NPU.

---

## From the shell

| Command | What it does |
| ------- | ------------ |
| `ovkit gui` | a window: pick a capability, point it at your webcam — the easiest start |
| `ovkit run detect image.jpg` | prints results, saves `image_out.jpg` |
| `ovkit capabilities` | what `Model(name)` can answer |
| `ovkit list` / `ovkit list --all` | curated models / every registered name |
| `ovkit info <model>` | source, licence, input size, preprocessing |
| `ovkit download <model>` | fetch ahead of time (the night before class) |
| `ovkit pull <hub-id>` | convert a Hugging Face model to IR |
| `ovkit devices` | what this machine can run on |
| `ovkit train` / `val` / `export` | RT-DETR training, evaluation, export |

---

## Runtime

| | |
| --- | --- |
| `device="AUTO"` | the default. `CPU` · `GPU` · `NPU` can be named directly; every result carries where it ran (`r.device`) and how long it took (`r.elapsed_ms`) |
| streams | folders, video and webcam run through `AsyncInferQueue` — frames overlap instead of queueing |
| `OVKIT_HOME` | cache location, default `~/.cache/ovkit`. A conversion is built once per `(name, precision)` and reused |
| `OVKIT_OFFLINE=1` | no network at all — for a classroom machine handed a copied cache |
| `OVKIT_LANG` | `ko` (default) / `en` |
| model mirror | everything from `leeyunjai/ovkit-models`; the upstream repository stays registered as the fallback, so an offline site has one copy to keep |
| integrity | `sha256` checks, IR fetched together with its `.bin` weights, and a missing half refused before compile |

---

## Not there yet

| | |
| --- | --- |
| Korean OCR | text recognition covers Latin letters and digits; Korean signage is not readable yet |
| hand landmarks | `teach`'s `hand` mode needs `ovkit[hand]` |
| converting detectors | what the Hub path can bring over stops at classification and embeddings |
| the 7 newest models | rescued from the mirror and registered, but verified as manifest entries only. Run `rtdetr_r101` and `human_pose_estimation_0005` on one photo before relying on them |
