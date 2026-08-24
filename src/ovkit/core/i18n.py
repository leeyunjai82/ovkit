"""Korean names — capability aliases, class-name translations, positions.

The library's audience is Korean students, so what they *type* (capability
names) and what they *read* (class names, error messages, positions) both
work in Korean. English stays the canonical layer underneath: every Korean
name maps onto one English identifier, and ``name_en`` never changes with
the display language — that is what code compares against.

The display language comes from ``$OVKIT_LANG`` (``ko`` default, ``en`` to
switch); it is read per call so tests and long-running apps can change it.
"""

from __future__ import annotations

import os

ENV_LANG = "OVKIT_LANG"


def lang() -> str:
    """Current display language: ``"ko"`` (default) or ``"en"``."""
    value = os.environ.get(ENV_LANG, "ko").strip().lower()
    return "en" if value.startswith("en") else "ko"


# --- what students type: Korean capability / model aliases -----------------

#: Korean names accepted by ``Model()``, mapped to the canonical identifier
#: (a capability name or a registry alias). One direction only — results and
#: docs always show the canonical name too, so students meet the English name.
KO_CAPS: dict[str, str] = {
    # capabilities (composed pipelines)
    "얼굴분석": "face_analyze",
    "사람분석": "person_analyze",
    "차량분석": "vehicle_analyze",
    "장면설명": "scene",
    "글자읽기": "read_text",
    "번호판읽기": "read_plate",
    "따라가기": "track",
    "추적": "track",
    "졸음감지": "drowsiness",
    "동작알아보기": "gesture",
    "제스처": "gesture",
    "시선": "gaze",
    "뭘보나": "attention",
    "얼굴가리기": "anonymize",
    "모자이크": "anonymize",
    "누구지": "face_match",
    "가르치기": "teach",
    "내가가르치기": "teach",
    # single-model aliases (registry)
    "물체찾기": "detect",
    "이건뭐야": "classify",
    "분류": "classify",
    "영역나누기": "segment",
    "자세": "pose",
    "얼굴찾기": "face_detection",
    "사람찾기": "person_detection",
    "차찾기": "vehicle_detection",
    "글자찾기": "text_detection",
    "나이성별": "age_gender",
    "감정": "emotion",
    "화질좋게": "super_resolution",
    "그림풍바꾸기": "style_transfer",
    "받아쓰기": "stt",
    "무슨소리": "sound_classification",
    "잡음제거": "noise_suppression",
    "대화": "llm",
}


def canonical(name: str) -> str:
    """Map a (possibly Korean) name onto its canonical identifier."""
    return KO_CAPS.get(str(name).strip(), str(name).strip())


# --- what students read: class-name translations ---------------------------

#: English class name -> Korean, flat across every class table. English is
#: the key because ``Results.names`` stores English; a missing entry simply
#: falls back to the English name.
KO_NAMES: dict[str, str] = {
    # coco80
    "person": "사람",
    "bicycle": "자전거",
    "car": "자동차",
    "motorcycle": "오토바이",
    "airplane": "비행기",
    "bus": "버스",
    "train": "기차",
    "truck": "트럭",
    "boat": "배",
    "traffic light": "신호등",
    "fire hydrant": "소화전",
    "stop sign": "정지 표지판",
    "parking meter": "주차 미터기",
    "bench": "벤치",
    "bird": "새",
    "cat": "고양이",
    "dog": "개",
    "horse": "말",
    "sheep": "양",
    "cow": "소",
    "elephant": "코끼리",
    "bear": "곰",
    "zebra": "얼룩말",
    "giraffe": "기린",
    "backpack": "배낭",
    "umbrella": "우산",
    "handbag": "핸드백",
    "tie": "넥타이",
    "suitcase": "여행가방",
    "frisbee": "프리스비",
    "skis": "스키",
    "snowboard": "스노보드",
    "sports ball": "공",
    "kite": "연",
    "baseball bat": "야구 방망이",
    "baseball glove": "야구 글러브",
    "skateboard": "스케이트보드",
    "surfboard": "서핑보드",
    "tennis racket": "테니스 라켓",
    "bottle": "병",
    "wine glass": "와인잔",
    "cup": "컵",
    "fork": "포크",
    "knife": "칼",
    "spoon": "숟가락",
    "bowl": "그릇",
    "banana": "바나나",
    "apple": "사과",
    "sandwich": "샌드위치",
    "orange": "오렌지",
    "broccoli": "브로콜리",
    "carrot": "당근",
    "hot dog": "핫도그",
    "pizza": "피자",
    "donut": "도넛",
    "cake": "케이크",
    "chair": "의자",
    "couch": "소파",
    "potted plant": "화분",
    "bed": "침대",
    "dining table": "식탁",
    "toilet": "변기",
    "tv": "티브이",
    "laptop": "노트북",
    "mouse": "마우스",
    "remote": "리모컨",
    "keyboard": "키보드",
    "cell phone": "휴대폰",
    "microwave": "전자레인지",
    "oven": "오븐",
    "toaster": "토스터",
    "sink": "싱크대",
    "refrigerator": "냉장고",
    "book": "책",
    "clock": "시계",
    "vase": "꽃병",
    "scissors": "가위",
    "teddy bear": "곰인형",
    "hair drier": "드라이기",
    "toothbrush": "칫솔",
    # voc21 extras
    "background": "배경",
    "aeroplane": "비행기",
    "diningtable": "식탁",
    "motorbike": "오토바이",
    "pottedplant": "화분",
    "sofa": "소파",
    "tvmonitor": "모니터",
    # detectors / segmentation
    "face": "얼굴",
    "vehicle": "차량",
    "text": "글자",
    "object": "물체",
    "bike": "자전거",
    "license plate": "번호판",
    "road": "도로",
    "curb": "연석",
    "lane mark": "차선",
    # attribute heads
    "female": "여성",
    "male": "남성",
    "neutral": "무표정",
    "happy": "행복",
    "sad": "슬픔",
    "surprise": "놀람",
    "anger": "화남",
    "real": "실제",
    "spoof": "위조",
    "van": "밴",
    "white": "흰색",
    "gray": "회색",
    "yellow": "노란색",
    "red": "빨간색",
    "green": "초록색",
    "blue": "파란색",
    "black": "검은색",
    "bag": "가방",
    "hat": "모자",
    "long sleeves": "긴소매",
    "long pants": "긴바지",
    "long hair": "긴머리",
    "coat/jacket": "외투",
    "open": "눈 뜸",
    "closed": "눈 감음",
    "no weld": "용접 없음",
    "normal weld": "정상 용접",
    "porosity": "기공",
    "undefined": "미확인",
    # sign language 12
    "digit 0": "숫자 0",
    "digit 1": "숫자 1",
    "digit 2": "숫자 2",
    "digit 3": "숫자 3",
    "digit 4": "숫자 4",
    "digit 5": "숫자 5",
    "thumb up": "엄지 위",
    "thumb down": "엄지 아래",
    "sliding two fingers up": "두 손가락 위로",
    "sliding two fingers down": "두 손가락 아래로",
    "sliding two fingers left": "두 손가락 왼쪽으로",
    "sliding two fingers right": "두 손가락 오른쪽으로",
    # aclnet 53 (only ones not already above)
    "rooster": "수탉",
    "pig": "돼지",
    "frog": "개구리",
    "hen": "암탉",
    "insects (flying)": "날벌레",
    "crow": "까마귀",
    "rain": "비",
    "sea waves": "파도",
    "crackling fire": "장작불",
    "crickets": "귀뚜라미",
    "chirping birds": "새소리",
    "water drops": "물방울",
    "wind": "바람",
    "pouring water": "물 붓는 소리",
    "toilet flush": "변기 물 내림",
    "thunderstorm": "천둥",
    "crying baby": "아기 울음",
    "sneezing": "재채기",
    "clapping": "박수",
    "breathing": "숨소리",
    "coughing": "기침",
    "footsteps": "발소리",
    "laughing": "웃음",
    "brushing teeth": "양치질",
    "snoring": "코골이",
    "drinking sipping": "마시는 소리",
    "door knock": "노크",
    "mouse click": "마우스 클릭",
    "keyboard typing": "키보드 소리",
    "door wood creaks": "문 삐걱임",
    "can opening": "캔 따는 소리",
    "washing machine": "세탁기",
    "vacuum cleaner": "청소기",
    "clock alarm": "알람",
    "clock tick": "시계 초침",
    "glass breaking": "유리 깨짐",
    "helicopter": "헬리콥터",
    "chainsaw": "전기톱",
    "siren": "사이렌",
    "car horn": "자동차 경적",
    "engine": "엔진",
    "church bells": "교회 종",
    "fireworks": "불꽃놀이",
    "hand saw": "톱질",
    "gunshot": "총소리",
    "crowd": "군중",
    "speech": "말소리",
}


def display_name(name_en: str) -> str:
    """The name to show for ``name_en`` in the current language."""
    if lang() == "ko":
        return KO_NAMES.get(name_en, name_en)
    return name_en


def name_key(name_en: str) -> str:
    """The comparison form of an English name: spaces become hyphens.

    ``if row["name_en"] == "cell-phone"`` must keep working whatever the
    display language is, and a value used as a key should not carry spaces.
    """
    return name_en.replace(" ", "-")


# --- where in the frame ----------------------------------------------------

_COLS = {"ko": ("왼쪽", "가운데", "오른쪽"), "en": ("left", "center", "right")}
_ROWS = {"ko": ("위", "중간", "아래"), "en": ("top", "middle", "bottom")}


def position(cx: float, cy: float, width: int, height: int) -> str:
    """A 9-grid position label ("왼쪽 위" / "top-left") for a point.

    Coordinates are a later lesson; "the cat is at the top-left" is day one.
    """
    col = min(2, int(3 * cx / max(width, 1)))
    row = min(2, int(3 * cy / max(height, 1)))
    if lang() == "ko":
        if (row, col) == (1, 1):
            return "가운데"
        return f"{_COLS['ko'][col]} {_ROWS['ko'][row]}".replace("가운데 중간", "가운데")
    if (row, col) == (1, 1):
        return "center"
    return f"{_ROWS['en'][row]}-{_COLS['en'][col]}"
