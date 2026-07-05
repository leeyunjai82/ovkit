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
| `pedestrian_detection` | `person_detection_0202` | 보행자 검출 (주행) |
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

## 등록 모델 (기능당 대표 1개)

레지스트리는 일부러 **기능당 검증된 모델 1개**만 노출해 목록을 읽기 쉽게 유지합니다.
변형들(다른 정확도/속도 등급, `int8`, `sparse_NN`, 다른 해상도)은 전부 미러에 그대로
있습니다 — 되살리는 방법은 아래 [변형](#모델-변형) 참고.

| 기능 | 모델 |
| ---- | ---- |
| 범용 객체 검출 | `rtdetr_r50` |
| 얼굴 검출 | `face_detection_0205` |
| 사람 검출 | `person_detection_0202` |
| 차량 검출 | `vehicle_detection_0200` |
| 교통 통합(사람+차량+자전거) | `person_vehicle_bike_detection_2000` |
| 텍스트 영역 검출 | `text_detection_0004` |
| 상품 검출 | `product_detection_0001` |
| 번호판 검출 | `vehicle_license_plate_detection_barrier_0106` |
| 나이 + 성별 | `age_gender_recognition_retail_0013` |
| 감정 | `emotions_recognition_retail_0003` |
| 머리 자세 | `head_pose_estimation_adas_0001` |
| 얼굴 랜드마크(5점) | `landmarks_regression_retail_0009` |
| 얼굴 재식별 | `face_reidentification_retail_0095` |
| 이미지 분류 | `resnet50_binary_0001` |
| 사람 속성 | `person_attributes_recognition_crossroad_0234` |
| 차량 속성 | `vehicle_attributes_recognition_barrier_0042` |
| 이미지 임베딩/검색 | `image_retrieval_0001` |
| 시맨틱 분할 | `road_segmentation_adas_0001` |
| 인스턴스 분할 | `instance_segmentation_person_0007` |
| 사람 포즈 | `human_pose_estimation_0007` |
| 시선 | `gaze_estimation_adas_0002` |
| 장면 텍스트 인식 | `text_recognition_0014` |
| 중국어 손글씨 OCR | `handwritten_simplified_chinese_recognition_0001` |
| 수어 인식 | `common_sign_language_0002` |
| 용접 결함 검출 | `weld_porosity_detection_0001` |
| 질의응답 | `bert_small_uncased_whole_word_masking_squad_0002` |
| 번역 (en↔de, en↔ru) | `machine_translation_nar_{en_de, de_en, en_ru, ru_en}_0002` |
| 음성 노이즈 제거 | `noise_suppression_poconetlike_0001` |
| 시계열 예측 | `time_series_forecasting_electricity_0001` |
| 초해상도 | `single_image_super_resolution_1033` |
| 챗 LLM | `tinyllama_chat` |
| 음성→텍스트 | `whisper_base` |

(모델-변형)=
## 변형

미러의 접미사 규칙: 숫자 ID = 정확도/속도 등급, `int8` = 양자화, `sparse_NN` = 가지치기,
`adas`/`retail` = 대상 도메인. 미러(`leeyunjai/ovkit-models`)에는 Apache-2.0 OMZ 전체가
있으며, 변형을 이름으로 쓰려면 `scripts/representatives.yaml`에 추가 후 재생성하거나
`src/ovkit/manifests/omz.yaml`에 항목을 붙여넣으면 됩니다:

```bash
python scripts/build_mirror.py --omz-intel --representatives \
    --emit-manifest src/ovkit/manifests/omz.yaml   # --representatives 빼면 전체
```
