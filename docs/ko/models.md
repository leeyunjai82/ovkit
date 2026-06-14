# 모델 카탈로그

ovkit은 permissive 라이선스의 OpenVINO 모델을 미러에서 다량 제공합니다. 상당수는 **같은
기능의 여러 변형**(정확도/속도, 입력 해상도, INT8 양자화, 가지치기)이라, 이 페이지는 **기능
중심**으로 정리했습니다. 실시간 목록은 `ovkit list`, 개별 정보는 `ovkit info <이름>`.

## 기능 별칭 (추천 기본값)

변형 중에 고르기 싫다면 **기능 별칭**을 쓰세요 — 검증된 기본 모델을 가리키는 친숙한 이름입니다.
`Model("face_detection")`이 바로 동작합니다.

| 별칭 | → 기본 모델 | 기능 |
| ---- | ----------- | ---- |
| `detect` | `rtdetr_r50` | 범용 객체 검출 (COCO-80) |
| `face_detection` | `face_detection_0205` | 얼굴 검출 |
| `person_detection` | `person_detection_0202` | 사람 검출 |
| `pedestrian_detection` | `pedestrian_detection_adas_0002` | 보행자 검출 (주행) |
| `vehicle_detection` | `vehicle_detection_0200` | 차량 검출 |
| `text_detection` | `text_detection_0004` | 텍스트 영역 검출 |
| `segment` | `road_segmentation_adas_0001` | 시맨틱 분할 |
| `instance_segmentation` | `instance_segmentation_person_0007` | 인스턴스 마스크 |
| `pose` | `human_pose_estimation_0007` | 사람 키포인트 |
| `face_landmarks` | `landmarks_regression_retail_0009` | 5점 얼굴 랜드마크 |
| `head_pose` | `head_pose_estimation_adas_0001` | 머리 자세 |
| `gaze` | `gaze_estimation_adas_0002` | 시선 방향 |
| `age_gender` | `age_gender_recognition_retail_0013` | 나이 + 성별 |
| `emotion` | `emotions_recognition_retail_0003` | 표정 |
| `face_reid` | `face_reidentification_retail_0095` | 얼굴 임베딩 |
| `person_attributes` | `person_attributes_recognition_crossroad_0234` | 복장/속성 |
| `vehicle_attributes` | `vehicle_attributes_recognition_barrier_0042` | 차종 + 색 |
| `classify` | `resnet50_binary_0001` | 이미지 분류 |
| `image_retrieval` | `image_retrieval_0001` | 이미지 임베딩 |
| `super_resolution` | `single_image_super_resolution_1033` | 3~4배 업스케일 |
| `text_recognition` | `text_recognition_0014` | 잘린 텍스트 인식 |
| `license_plate` | `vehicle_license_plate_detection_barrier_0106` | 번호판 검출 |
| `qa` | `bert_small_uncased_whole_word_masking_squad_0002` | 추출형 QA |
| `translation` | `machine_translation_nar_en_de_0002` | EN→DE 번역 |
| `noise_suppression` | `noise_suppression_poconetlike_0001` | 음성 노이즈 제거 |
| `time_series` | `time_series_forecasting_electricity_0001` | 시계열 예측 |
| `llm` | `tinyllama_chat` | 챗 LLM (genai) |
| `stt` | `whisper_base` | 음성→텍스트 (genai) |

```python
from ovkit import Model
Model("face_detection")("photo.jpg")     # face_detection_0205 사용
```

별칭은 `src/ovkit/manifests/aliases.yaml`에 있어 한 줄로 바꿀 수 있습니다.

## 기능별 전체 카탈로그

변형은 합쳤습니다. 접미사 규칙: 숫자 ID = 정확도/속도 등급, `int8` = 양자화,
`sparse_NN` = 가지치기, `adas`/`retail` = 대상 도메인.

### 검출
- **범용 객체(COCO):** `rtdetr_r50`, `rtdetr_r101` (DETR, NMS 없음);
  `faster_rcnn_resnet101_coco_sparse_60_0001`; `yolo_v2_ava_0001`
  (+`sparse_35`/`sparse_70`), `yolo_v2_tiny_ava_0001` (+`sparse_30`/`sparse_60`)
- **얼굴:** `face_detection_0200/0202/0204/0205/0206`, `face_detection_adas_0001`,
  `face_detection_retail_0004/0005`
- **사람/보행자:** `person_detection_0106/0200/0201/0202/0203/0301/0302/0303`,
  `person_detection_retail_0002/0013`, `pedestrian_detection_adas_0002`,
  `pedestrian_and_vehicle_detector_adas_0001`
- **사람+차량+자전거(교통):** `person_vehicle_bike_detection_2000…2004`,
  `…_crossroad_0078/1016/yolov3_1020`
- **차량/번호판:** `vehicle_detection_0200/0201/0202`, `vehicle_detection_adas_0002`,
  `vehicle_license_plate_detection_barrier_0106`
- **텍스트 영역:** `horizontal_text_detection_0001`, `text_detection_0003/0004`
- **특수:** `product_detection_0001`, `smartlab_object_detection_0001…0004`,
  `person_detection_asl_0001`, `person_detection_raisinghand_recognition_0001`,
  `person_detection_action_recognition_0005/0006`

### 분할
- **시맨틱:** `road_segmentation_adas_0001`, `semantic_segmentation_adas_0001`,
  `icnet_camvid_ava_0001` (+`sparse_30`/`sparse_60`), `unet_camvid_onnx_0001`
- **인스턴스:** `instance_segmentation_person_0007`

### 포즈·랜드마크·시선
- **사람 포즈:** `human_pose_estimation_0001/0005/0006/0007`
- **얼굴 랜드마크:** `landmarks_regression_retail_0009` (5점),
  `facial_landmarks_35_adas_0002` (35점), `facial_landmarks_98_detection_0001` (98점)
- **머리 자세:** `head_pose_estimation_adas_0001` · **시선:** `gaze_estimation_adas_0002`

### 속성·재식별
- **얼굴:** `age_gender_recognition_retail_0013`, `emotions_recognition_retail_0003`,
  `face_reidentification_retail_0095`
- **사람:** `person_attributes_recognition_crossroad_0230/0234/0238`
- **차량:** `vehicle_attributes_recognition_barrier_0039/0042`

### 분류·임베딩·초해상도
- **분류:** `resnet50_binary_0001`, `resnet18_xnor_binary_onnx_0001`
- **임베딩/검색:** `image_retrieval_0001`
- **초해상도:** `single_image_super_resolution_1032/1033`

### 행동 인식
`asl_recognition_0004`, `common_sign_language_0002` (수어);
`smartlab_sequence_modelling_0001/0002`; `weld_porosity_detection_0001` (산업)

### OCR (텍스트 인식)
`text_recognition_0012/0014` (장면 텍스트); `handwritten_score_recognition_0003`,
`handwritten_simplified_chinese_recognition_0001`

### NLP
- **질의응답(BERT/SQuAD):** `bert_large_*`, `bert_small_*` —
  large/small × 일반/임베딩/`int8` 변형
- **기계번역(비자기회귀):**
  `machine_translation_nar_{de_en, en_de, en_ru, ru_en}_0002`

### 오디오·시계열·GenAI
- **노이즈 제거:** `noise_suppression_denseunet_ll_0001`, `noise_suppression_poconetlike_0001`
- **시계열:** `time_series_forecasting_electricity_0001`
- **GenAI:** `tinyllama_chat` (LLM), `whisper_base` (음성→텍스트)

:::{note}
전체 OMZ 세트는 미러 매니페스트를 생성하면 사용 가능합니다
(`python scripts/build_mirror.py --omz-intel --emit-manifest
src/ovkit/manifests/omz.yaml`). `detect`, `llm`, `stt`와 큐레이트된 DETR/얼굴 모델은
그 없이도 동작합니다. 현재 OpenVINO에서 컴파일 안 되는 일부 구형 OMZ 모델은
`python scripts/selfcheck.py --prune-manifest src/ovkit/manifests/omz.yaml --load-only`
로 자동 제거됩니다.
:::
