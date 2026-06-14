# Model catalog

ovkit serves a large set of permissively-licensed OpenVINO models from its
mirror. Many are the **same capability in several variants** (different
accuracy/speed, input resolution, INT8 quantization, or pruning), so this page
is organized **by function**. Run `ovkit list` for the live list with
descriptions, and `ovkit info <name>` for one model.

## Capability aliases (recommended defaults)

Don't want to pick among variants? Use a **capability alias** — a friendly name
that points at a well-tested default. `Model("face_detection")` just works.

| alias | → default model | what it does |
| ----- | --------------- | ------------ |
| `detect` | `rtdetr_r50` | general object detection (COCO-80) |
| `face_detection` | `face_detection_0205` | detect faces |
| `person_detection` | `person_detection_0202` | detect people |
| `pedestrian_detection` | `pedestrian_detection_adas_0002` | detect pedestrians (driving) |
| `vehicle_detection` | `vehicle_detection_0200` | detect vehicles |
| `text_detection` | `text_detection_0004` | detect text regions |
| `segment` | `road_segmentation_adas_0001` | semantic segmentation |
| `instance_segmentation` | `instance_segmentation_person_0007` | per-instance masks |
| `pose` | `human_pose_estimation_0007` | human body keypoints |
| `face_landmarks` | `landmarks_regression_retail_0009` | 5-point face landmarks |
| `head_pose` | `head_pose_estimation_adas_0001` | head yaw/pitch/roll |
| `gaze` | `gaze_estimation_adas_0002` | gaze direction |
| `age_gender` | `age_gender_recognition_retail_0013` | age + gender |
| `emotion` | `emotions_recognition_retail_0003` | facial emotion |
| `face_reid` | `face_reidentification_retail_0095` | face embedding |
| `person_attributes` | `person_attributes_recognition_crossroad_0234` | clothing/attributes |
| `vehicle_attributes` | `vehicle_attributes_recognition_barrier_0042` | type + color |
| `classify` | `resnet50_binary_0001` | image classification |
| `image_retrieval` | `image_retrieval_0001` | image embedding |
| `super_resolution` | `single_image_super_resolution_1033` | 3-4x upscaling |
| `text_recognition` | `text_recognition_0014` | read cropped text |
| `license_plate` | `vehicle_license_plate_detection_barrier_0106` | detect license plates |
| `qa` | `bert_small_uncased_whole_word_masking_squad_0002` | extractive Q&A |
| `translation` | `machine_translation_nar_en_de_0002` | EN→DE translation |
| `noise_suppression` | `noise_suppression_poconetlike_0001` | denoise speech |
| `time_series` | `time_series_forecasting_electricity_0001` | forecasting |
| `llm` | `tinyllama_chat` | chat LLM (genai) |
| `stt` | `whisper_base` | speech-to-text (genai) |

```python
from ovkit import Model
Model("face_detection")("photo.jpg")     # uses face_detection_0205
```

Aliases live in `src/ovkit/manifests/aliases.yaml` — re-point any one in a line.

## Full catalog by function

Variants are collapsed; the suffix conventions are: numbered IDs = accuracy/speed
tiers, `int8` = quantized, `sparse_NN` = pruned, `adas`/`retail` = target domain.

### Detection
- **General objects (COCO):** `rtdetr_r50`, `rtdetr_r101` (DETR, NMS-free);
  `faster_rcnn_resnet101_coco_sparse_60_0001`; `yolo_v2_ava_0001`
  (+`sparse_35`/`sparse_70`), `yolo_v2_tiny_ava_0001` (+`sparse_30`/`sparse_60`)
- **Face:** `face_detection_0200/0202/0204/0205/0206`, `face_detection_adas_0001`,
  `face_detection_retail_0004/0005`
- **Person / pedestrian:** `person_detection_0106/0200/0201/0202/0203/0301/0302/0303`,
  `person_detection_retail_0002/0013`, `pedestrian_detection_adas_0002`,
  `pedestrian_and_vehicle_detector_adas_0001`
- **Person+vehicle+bike (traffic):** `person_vehicle_bike_detection_2000…2004`,
  `…_crossroad_0078/1016/yolov3_1020`
- **Vehicle / plate:** `vehicle_detection_0200/0201/0202`, `vehicle_detection_adas_0002`,
  `vehicle_license_plate_detection_barrier_0106`
- **Text regions:** `horizontal_text_detection_0001`, `text_detection_0003/0004`
- **Specialized:** `product_detection_0001`, `smartlab_object_detection_0001…0004`,
  `person_detection_asl_0001`, `person_detection_raisinghand_recognition_0001`,
  `person_detection_action_recognition_0005/0006`

### Segmentation
- **Semantic:** `road_segmentation_adas_0001`, `semantic_segmentation_adas_0001`,
  `icnet_camvid_ava_0001` (+`sparse_30`/`sparse_60`), `unet_camvid_onnx_0001`
- **Instance:** `instance_segmentation_person_0007`

### Pose, landmarks, gaze
- **Human pose:** `human_pose_estimation_0001/0005/0006/0007`
- **Face landmarks:** `landmarks_regression_retail_0009` (5-pt),
  `facial_landmarks_35_adas_0002` (35-pt), `facial_landmarks_98_detection_0001` (98-pt)
- **Head pose:** `head_pose_estimation_adas_0001` · **Gaze:** `gaze_estimation_adas_0002`

### Attributes & re-identification
- **Face:** `age_gender_recognition_retail_0013`, `emotions_recognition_retail_0003`,
  `face_reidentification_retail_0095`
- **Person:** `person_attributes_recognition_crossroad_0230/0234/0238`
- **Vehicle:** `vehicle_attributes_recognition_barrier_0039/0042`

### Classification, embedding, super-resolution
- **Classify:** `resnet50_binary_0001`, `resnet18_xnor_binary_onnx_0001`
- **Embedding / retrieval:** `image_retrieval_0001`
- **Super-resolution:** `single_image_super_resolution_1032/1033`

### Action recognition
`asl_recognition_0004`, `common_sign_language_0002` (sign language);
`smartlab_sequence_modelling_0001/0002`; `weld_porosity_detection_0001` (industrial)

### OCR (text recognition)
`text_recognition_0012/0014` (scene text); `handwritten_score_recognition_0003`,
`handwritten_simplified_chinese_recognition_0001`

### NLP
- **Question answering (BERT/SQuAD):** `bert_large_*` and `bert_small_*` —
  large/small × plain/embedding/`int8` variants
- **Machine translation (non-autoregressive):**
  `machine_translation_nar_{de_en, en_de, en_ru, ru_en}_0002`

### Audio, time series, GenAI
- **Noise suppression:** `noise_suppression_denseunet_ll_0001`, `noise_suppression_poconetlike_0001`
- **Time series:** `time_series_forecasting_electricity_0001`
- **GenAI:** `tinyllama_chat` (LLM), `whisper_base` (speech-to-text)

:::{note}
The full OMZ set becomes available once you generate the mirror manifest
(`python scripts/build_mirror.py --omz-intel --emit-manifest
src/ovkit/manifests/omz.yaml`); `detect`, `llm`, `stt` and the curated DETR/face
models work without it. A handful of legacy OMZ models that don't compile on
current OpenVINO are dropped automatically by
`python scripts/selfcheck.py --prune-manifest src/ovkit/manifests/omz.yaml --load-only`.
:::
