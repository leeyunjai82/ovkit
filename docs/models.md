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
| `pedestrian_detection` | `person_detection_0202` | detect pedestrians (driving) |
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

## Registered models (one representative per capability)

The registry deliberately exposes **one well-tested model per capability** so
pickers stay readable. Every variant (other accuracy/speed tiers, `int8`,
`sparse_NN`, other resolutions) is still hosted on the mirror — see
[Variants](#model-variants) below to surface one.

| function | model |
| -------- | ----- |
| general object detection | `rtdetr_r50` |
| face detection | `face_detection_0205` |
| person detection | `person_detection_0202` |
| vehicle detection | `vehicle_detection_0200` |
| traffic combo (person+vehicle+bike) | `person_vehicle_bike_detection_2000` |
| text region detection | `text_detection_0004` |
| product detection | `product_detection_0001` |
| license plate detection | `vehicle_license_plate_detection_barrier_0106` |
| age + gender | `age_gender_recognition_retail_0013` |
| emotion | `emotions_recognition_retail_0003` |
| head pose | `head_pose_estimation_adas_0001` |
| face landmarks (5-pt) | `landmarks_regression_retail_0009` |
| face re-identification | `face_reidentification_retail_0095` |
| image classification | `resnet50_binary_0001` |
| person attributes | `person_attributes_recognition_crossroad_0234` |
| vehicle attributes | `vehicle_attributes_recognition_barrier_0042` |
| image embedding / retrieval | `image_retrieval_0001` |
| semantic segmentation | `road_segmentation_adas_0001` |
| instance segmentation | `instance_segmentation_person_0007` |
| human pose | `human_pose_estimation_0007` |
| gaze | `gaze_estimation_adas_0002` |
| scene text recognition | `text_recognition_0014` |
| handwritten Chinese OCR | `handwritten_simplified_chinese_recognition_0001` |
| sign language | `common_sign_language_0002` |
| weld defect detection | `weld_porosity_detection_0001` |
| question answering | `bert_small_uncased_whole_word_masking_squad_0002` |
| translation (en↔de, en↔ru) | `machine_translation_nar_{en_de, de_en, en_ru, ru_en}_0002` |
| speech noise suppression | `noise_suppression_poconetlike_0001` |
| time-series forecasting | `time_series_forecasting_electricity_0001` |
| super-resolution | `single_image_super_resolution_1033` |
| chat LLM | `tinyllama_chat` |
| speech-to-text | `whisper_base` |

(model-variants)=
## Variants

Suffix conventions on the mirror: numbered IDs = accuracy/speed tiers,
`int8` = quantized, `sparse_NN` = pruned, `adas`/`retail` = target domain.
The mirror (`leeyunjai/ovkit-models`) hosts the full Apache-2.0 OMZ set; to
surface a variant by name, add it to `scripts/representatives.yaml` and
regenerate, or paste its entry into `src/ovkit/manifests/omz.yaml`:

```bash
python scripts/build_mirror.py --omz-intel --representatives \
    --emit-manifest src/ovkit/manifests/omz.yaml   # omit --representatives for ALL
```
