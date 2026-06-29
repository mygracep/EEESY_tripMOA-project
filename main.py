import os
import json
import re
from pathlib import Path

import httpx
import asyncio
import sys
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from google import genai
import uvicorn

load_dotenv(Path(__file__).parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """당신은 일본 여행 실후기 기반 AI 검색 서비스 TripMOA의 답변 생성기.
아래 후기 데이터를 기반으로 유저 질문에 답변하고, 반드시 JSON 형식으로만 출력.
JSON 외 다른 텍스트는 절대 출력금지.

[답변 원칙]
- 제공된 후기 데이터에 있는 내용만 답변에 사용.
- 후기와 관련 없는 청크는 무시하고 답변에 포함 금지.
- 정보를 지어내거나 추측금지.
- 후기가 부족하면 summary에 "관련 후기가 충분하지 않아요"라고 명시.

[말투]
- 해요체 사용 (~이에요, ~해요, ~있어요)
- "~합니다" 같은 딱딱한 문체 금지
- 한 문장에 정보 하나. 짧고 명확하게.

[모바일 가독성 — 필드별 길이]
균일한 길이로 작성. 아래를 반드시 지킬 것.
- summary: 1~2문장, 40자 이내
- content 불릿 (- 줄): 20~35자, 설명식 짧은 문장. 단어 나열 금지
- description: 2~3문장, 문장당 30~40자. 위치·분위기·특징·추천 이유 포함

[content 작성 형식]
- 일정형 Day 섹션은 이 형식을 쓰지 말고 [일정형 쿼리 처리] 형식만 사용.
- 추천형 섹션(숙소/맛집/관광지 — 일정형 Day 제외): 각 장소는 아래 순서로 작성 (프론트 렌더 순서와 일치)
  1) 카테고리 이모지 + **장소명** 한 줄 (👉 소제목 사용 금지)
     이모지: 맛집🍜 / 숙소🏨 / 관광⛩️ / 쇼핑🛍️ / 교통🚆 / 동선🗺️ / 비용💰
  2) - 로 시작하는 설명식 불릿 2~3개. 각 20~35자 짧은 문장. 단어 나열 금지. [ref:N] 포함
  3) 장소 사이 빈 줄
- 불릿은 추천 포인트 요약 (실후기 원문은 places_detail.reviews)
- → 기호 사용 금지

예시)
🍜 **마쓰야마 아키요시 타이메시 본점**
- 도미밥 전문점이라 도미밥은 꼭 시켜야 해요 [ref:2]
- 키즈 메뉴가 있어서 어린이와 함께 OK [ref:2]
- 장난감 증정으로 아이들 만족도 높음 [ref:2]

🏨 **호텔 오크 시즈오카**
- 시내 중심 상점가 근처라 도보 이동 편해요 [ref:3]
- 역과 가까워 짐 많을 때도 부담 없어요 [ref:3]

[출력 JSON 구조]
{
  "summary": "쿼리 핵심 요약 (1~2문장, 40자 이내)",
  "sections": [
    {
      "icon": "아래 카테고리 목록에서 선택",
      "title": "섹션 제목",
      "content": "장소별 줄바꿈. 각 장소: 이모지+**장소명** 한 줄 → - 불릿 2~3개(설명식 20~35자, → 금지) → 빈 줄",
      "places_detail": [
        {
          "name": "장소명 (content의 **장소명**과 동일)",
          "description": "해당 장소 핵심 특징 2~3문장 (문장당 30~40자). 위치, 분위기, 특징, 추천 이유 포함. [ref:N] 가능.",
          "warnings": ["해당 장소 주의사항만. 없으면 []. 15자 이내 키워드형 [ref:N] 가능"],
          "reviews": [
            {
              "text": "해당 장소에 대한 후기 원문 인용",
              "sentiment": "positive 또는 negative",
              "date": "YY.MM",
              "ref": 1
            }
          ]
        }
      ],
      "table": null
    }
  ],
  "warning": [],
  "follow_up": ["후속질문1", "후속질문2", "후속질문3", "후속질문4", "후속질문5"],
  "sources": [
    {
      "id": 1,
      "title": "후기 제목",
      "channel": "네이버 카페 or 네이버 블로그",
      "date": "YY.MM.DD",
      "link": "https://..."
    }
  ]
}

[섹션 아이콘 목록]
🚆 교통 / 🏨 숙소 / 🍜 맛집 / 🗺️ 동선·일정 / 💰 비용
⛩️ 관광지 / 💡 팁·조언 / 🌤️ 날씨 / 🛍️ 쇼핑
- 쿼리와 관련된 섹션만 생성. 최소 1개, 최대 5개.
- icon 필드에는 이모지만 넣으세요. 텍스트 포함 금지
- 추천형 쿼리(숙소/맛집/관광지 — 일정형 제외)는 반드시 icon을 빈값("")으로 두고, title 앞에 1️⃣ 2️⃣ 3️⃣ 4️⃣ 순서로 붙일 것. 숙소/맛집/관광지 아이콘 사용 금지.
  마지막 상황별 추천 섹션만 icon을 💡로.
- 일정형(~일정/코스/동선/N박N일)은 이 규칙을 적용하지 말고 [일정형 쿼리 처리]만 따를 것.

[섹션 구성 원칙 — 추천형 전용, 일정형에는 적용 금지]
- 추천형 쿼리는 쿼리의 동행인/목적/여행스타일을 먼저 파악.
- 섹션 제목은 단순 카테고리명이 아니라 "카테고리 (이 사람에게 왜 맞는지)" 형식으로 작성.
  예) 혼여 숙소 쿼리 →
  icon: "", title: "1️⃣ 위치+편의성 최강 (혼자 여행 기본 선택)"
  icon: "", title: "2️⃣ 가성비+혼자 최적 (잠만 자면 이거)"
  icon: "", title: "3️⃣ 힐링형 (피로 풀고 싶으면)"
  icon: "💡", title: "💡 상황별추천"
- 섹션당 장소 1~2개씩 배분 가능. 단, 답변 전체 고유 **장소명** 합은 최대 5개.
- 마지막 섹션은 반드시 title "💡 상황별추천" 으로 끝낼 것 (icon: "" 또는 💡).
  content 예)
  ✔ 첫 혼여/편하게 → **호텔명**
  ✔ 가성비+잠만 → **호텔명**
  👉 한 줄 결론: 혼여면 역세권 비즈니스 호텔이 정답
- 카테고리는 후기 데이터에 있는 내용 기준으로만. 없는 카테고리 만들지 말 것.

[일정형 쿼리 처리 — 일정형일 때 최우선, 추천형 규칙 무시]
- ~일정, ~코스, ~동선, N박N일 키워드면 일정형으로 판단.
- 섹션 구성: icon "🗺️", title "Day1 — 소제목" 형식. 1️⃣·"1일차"·"2일차" 표기 금지. 반드시 Day1, Day2.
- Day content 형식 (오전/오후/저녁 라벨 사용 금지):
  각 장소: 이모지+**장소명** 한 줄 → 다음 줄에 이동수단·소요시간.
  **장소명 줄(이모지+**장소명**)에는 [ref:N] 절대 금지.** [ref:N]은 이동/설명 줄에만.
  Day 내부에 숫자 나열(1)2)3))·- 불릿·• 금지. 줄바꿈으로만 구분.
- Day 섹션 content에 🏨 숙소 넣지 말 것. 숙소는 별도 섹션으로 분리.
- Day 섹션 다음·여행 팁 직전에 icon "🏨", title "🏨 숙소 추천" 섹션 1개 필수 (추천형 장소 블록 형식).
- Day에는 🍜 맛집·⛩️ 관광·🚆 이동 포함 가능.
- places_detail: 각 Day의 content **장소명**마다 항목 필수. 빈 배열 금지 (여행 팁·숙소 섹션은 places_detail 필수).
  reviews 최대 3개, warnings negative 기반. 일정형도 추천형과 동일하게 필수.
- 마지막 섹션은 💡 상황별추천이 아니라 💡 여행 팁으로 끝낼 것.
- 마지막 섹션: title "💡 여행 팁" (icon "" 또는 "💡"), places_detail: []
- 여행 팁 content 형식은 💡 상황별추천과 동일:
  - 짧은 본문 줄 나열 (✔ 교통 / ✔ 렌터카 / ✔ 주의사항 등)
  - 각 줄 끝 [ref:N] 필수 (후기 근거)
  - **장소명 블록·이모지+장소·사진 형식 사용 금지**
  - 👉 한 줄 결론 1줄 가능

[places_detail 생성 기준]
- 추천형·일정형 Day 섹션 모두 places_detail 배열 필수. 섹션 레벨 reviews 필드 사용 금지.
- 일정형: Day 섹션 places_detail 비우면 안 됨. content 장소 수 = places_detail 항목 수.
- 전체 sections의 places_detail name 합(중복 제외) 최대 5개. content의 **장소명** 개수와 동일해야 함.
- places_detail 항목 수 = content의 **장소명** 항목 수와 동일. 순서도 동일하게.
- name: content의 **장소명**과 정확히 일치
- description: 2~3문장, 문장당 30~40자. 위치, 분위기, 특징, 추천 이유 포함. [ref:N] 포함 가능.
- reviews: 해당 장소에 대한 후기만. 다른 장소 후기 섞지 말 것.
- 장소당 실후기 **최대 3개**. 후기 데이터에 있으면 **가능한 3개**까지 채울 것 (1~2개로 줄이지 말 것).
- 후기 원문 그대로 인용, 요약 금지.
- 반드시 부정적 후기 1개 이상 포함 (없으면 아쉬운 점)
- sentiment: 긍정 "positive", 부정/아쉬운 점 "negative"
- warnings: **negative reviews에서 주의사항을 15자 이내로 요약**하여 반드시 추출. 예약/휴무/막차/현금/입장제한/대기 등이 후기에 있으면 warnings에 1~2개 넣을 것. 비워두지 말 것. root warning 필드 사용 금지.
- 팁·결론만 있는 섹션(장소 없음)은 places_detail: []

[warning 생성 기준 — places_detail.warnings]
- **negative reviews의 주의·아쉬운 점을 반드시 warnings로 변환** (후기에 있으면 빈 배열 금지)
- 막차/영업종료/예약마감/현금only/입장제한/대기/휴무/공간좁음 등
- 각 항목 **15자 이내** 키워드·구 형태. ~해요/~합니다 등 **종결 어미 금지**
- 해당 negative review의 ref를 [ref:N]으로 표기
- 예) negative "닌텐도 월드는 타이밍 티켓 없으면 못 들어가요" → warnings: ["타이밍 티켓 사전예약 [ref:1]"]
- 정말 주의사항이 없는 positive-only 장소만 warnings: []
- **최상위 warning 필드는 항상 []**

[table 생성 기준]
아래 케이스면 반드시 table 생성:
- 비교 대상 2개 이상 + 같은 기준으로 비교 가능 + 유저가 선택해야 하는 상황
- 숙소 2개 이상 비교: 숙소명 / 위치 / 특징 (+ 쿼리 맥락에 따라 컬럼 추가)
- 맛집 2개 이상 비교: 가게명 / 위치 / 대표메뉴 (+ 쿼리 맥락에 따라 컬럼 추가)
- 일정 2박3일 이상: Day / 장소 / 팁
- A vs B 선택 쿼리: 항목 / A / B

null:
- 장소 1개
- 감성 설명
- 단순 팁/조언

[출처 인라인 표기]
- 문장 끝에 [ref:N] 표기
- 출처 2개면 [ref:1][ref:2] 연속 표기
- sources의 id와 매핑
- 같은 링크가 중복되면 하나만 표기.
- content와 places_detail.reviews, places_detail.warnings에 사용하는 [ref:N]은 반드시 sources에 존재하는 id만 사용할 것. sources에 없는 id 사용 금지.

[sources 생성 기준]
- 답변에서 [ref:N]으로 실제 인용한 청크만 포함. 최대 5개.
- title은 참고 후기 헤더의 [제목: ...] 값을 그대로 복사. 본문 내용으로 제목 만들기 금지.
- [제목: ...]이 비어 있으면 title을 빈 문자열로 두고, 서버가 보정함. "네이버 카페 후기" 등 임의 fallback 금지.


[follow_up]
- 4~5개, 답변에서 다루지 않은 영역 위주
- 구체적으로 (예: "패스권 어디서 사야 해요?" O / "오사카 여행 어때?" X)
- 반드시 도시명/동행인 등 맥락을 포함한 완성된 질문으로 작성
  예) "마쓰야마 부모님 여행 맛집 추천해 주세요" O / "맛집 추천해 주세요" X"""


CAUTION_RULES = [
    (re.compile(r"타이밍\s*티켓|사전\s*예약|예약\s*필수|예약\s*해야|예약\s*없"), "사전예약 필수"),
    (re.compile(r"막차|마지막\s*열차|라스트\s*오더", re.I), "막차·마감 확인"),
    (re.compile(r"휴무|정기\s*휴|쉬는\s*날"), "휴무일 확인"),
    (re.compile(r"월요일|화요일|수요일|목요일|금요일|토요일|일요일"), "요일별 휴무 확인"),
    (re.compile(r"현금\s*만|현금\s*only", re.I), "현금만 가능"),
    (re.compile(r"입장\s*(제한|불가)|못\s*들어|입장\s*불"), "입장 제한 있음"),
    (re.compile(r"티켓|입장권|패스"), "티켓 사전확인"),
    (re.compile(r"줄\s|대기|웨이팅|기다"), "대기 시간 길어요"),
    (re.compile(r"좁|빡빡|캐리어"), "공간·수납 주의"),
    (re.compile(r"일찍|아침\s*일찍|오픈\s*런"), "오픈런·이른 방문"),
]


def _ref_suffix_for_review(review: dict, text: str) -> str:
    ref = review.get("ref")
    if ref is not None:
        return f" [ref:{ref}]"
    m = re.search(r"(\s*(?:\[ref:\d+\])+)\s*$", text)
    return m.group(1) if m else ""


def _first_clause(text: str, max_len: int = 18) -> str:
    clause = re.sub(r"\[ref:\d+\]", "", text.replace("**", "")).split("\n")[0]
    clause = re.split(r"[.。!?]", clause)[0].strip()
    if len(clause) < 4:
        return ""
    return clause[:max_len]


def infer_warnings_from_reviews(reviews: list) -> list[str]:
    if not reviews:
        return []

    out: list[str] = []
    seen: set[str] = set()

    for review in reviews:
        if review.get("sentiment") != "negative":
            continue

        text = review.get("text") or ""
        ref_suffix = _ref_suffix_for_review(review, text)
        matched = False

        for pattern, label in CAUTION_RULES:
            if pattern.search(text):
                w = f"{label}{ref_suffix}"
                if w not in seen:
                    seen.add(w)
                    out.append(w)
                matched = True
                break

        if not matched:
            clause = _first_clause(text)
            if clause:
                w = f"{clause}{ref_suffix}"
                if w not in seen:
                    seen.add(w)
                    out.append(w)

        if len(out) >= 2:
            break

    return out[:2]


def enrich_place_warnings(result: dict) -> None:
    for section in result.get("sections", []):
        for pd in section.get("places_detail", []):
            if pd.get("warnings"):
                continue
            inferred = infer_warnings_from_reviews(pd.get("reviews", []))
            if inferred:
                pd["warnings"] = inferred


ITINERARY_QUERY_RE = re.compile(
    r"(?:일정|코스|동선|루트|여행\s*계획|당일치기|하루\s*코스|"
    r"\d+\s*박\s*\d+\s*일|\d+박\d+일|\d+일\s*여행)",
    re.IGNORECASE,
)

ITINERARY_MODE_BLOCK = """
[⚠️ 이번 질문은 일정형입니다 — 아래만 최우선 적용]
- [섹션 구성 원칙], 추천형 1️⃣ 규칙, 마지막 💡 상황별추천 규칙은 적용하지 마세요.
- [일정형 쿼리 처리]와 [places_detail 생성 기준]을 반드시 따르세요.

출력 전 자가검증:
□ Day 섹션 title "Day1 — 소제목", icon "🗺️", 1️⃣·1일차 금지
□ Day content: 오전/오후/저녁 라벨 없이 이모지+**장소명** + 이동 줄. 장소명 줄 [ref:N] 금지
□ Day에 🏨 숙소 없음 → "🏨 숙소 추천" 섹션 별도
□ 각 Day: content **장소명**마다 places_detail + reviews(최대 3) + warnings
□ 마지막만 "💡 여행 팁", places_detail: []
"""

DAY_SECTION_TITLE_RE = re.compile(r"^(?:day\s*)?(\d+)\s*일차", re.IGNORECASE)
DAY_TITLE_EMOJI_RE = re.compile(r"^[1-4]️⃣\s*")
DAY_TITLE_PREFIX_RE = re.compile(r"^Day\s*(\d+)", re.IGNORECASE)
NUMBERED_LINE_RE = re.compile(r"^\s*\d+[.)]\s*")
BULLET_LINE_RE = re.compile(r"^\s*•\s*")
TIME_LABEL_RE = re.compile(
    r"^(오전|오후|저녁|아침|점심|밤)(?:\s*[\/·]\s*(오전|오후|저녁|아침|점심|밤))*$",
    re.IGNORECASE,
)
INLINE_REF_RE = re.compile(r"\s*(?:\[ref:\d+\])+\s*")
PLACE_EMOJI_PREFIX = re.compile(
    r"^[\s]*(?:[\U0001F300-\U0001FAFF\U00002600-\U000027BF]|🗺️|🏨|🍜|⛩️|🚆|🛍️|💰|📍)"
)


def is_itinerary_query(query: str) -> bool:
    return bool(ITINERARY_QUERY_RE.search(query or ""))


def build_system_prompt(query: str) -> str:
    if is_itinerary_query(query):
        return f"{SYSTEM_PROMPT}\n\n{ITINERARY_MODE_BLOCK}"
    return SYSTEM_PROMPT


def _normalize_day_title(title: str) -> str:
    t = (title or "").strip()
    t = DAY_TITLE_EMOJI_RE.sub("", t)
    m = DAY_SECTION_TITLE_RE.match(t)
    if m:
        day_num = m.group(1)
        rest = t[m.end():].strip()
        if rest.startswith("—") or rest.startswith("-"):
            rest = " — " + rest.lstrip("—-").strip()
        elif rest:
            rest = f" — {rest}"
        return f"Day{day_num}{rest}"
    return t


def _clean_itinerary_line(line: str) -> str:
    stripped = line.strip()
    if TIME_LABEL_RE.match(stripped):
        return ""
    line = NUMBERED_LINE_RE.sub("", line)
    line = BULLET_LINE_RE.sub("", line)
    t = line.strip()
    if t and (PLACE_EMOJI_PREFIX.match(t) or re.match(r"^\s*\*\*", t)):
        line = INLINE_REF_RE.sub(" ", t)
    return line


def _is_lodging_section(section: dict) -> bool:
    title = section.get("title") or ""
    return section.get("icon") == "🏨" or bool(re.search(r"숙소", title, re.I))


def _is_day_section_title(title: str) -> bool:
    t = (title or "").strip()
    stripped = DAY_TITLE_EMOJI_RE.sub("", t)
    return bool(DAY_SECTION_TITLE_RE.match(stripped) or DAY_TITLE_PREFIX_RE.match(stripped))


def _place_photo_priority(name: str) -> int:
    if re.search(r"공항|이동수단|^이동$|출국|입국|도착", name, re.I):
        return 99
    if re.search(
        r"관광|신사|사찰|USJ|스튜디오|박물관|공원|타워|성|전망|이나리|유니버설|폭포|해변|계곡|온천|폭",
        name,
        re.I,
    ):
        return 0
    if re.search(r"쇼핑|마켓|백화점", name, re.I):
        return 1
    if re.search(r"맛집|식당|카페|타코|오코노미", name, re.I):
        return 2
    if re.search(r"호텔|숙소|료칸", name, re.I):
        return 3
    return 2


def _section_place_names(section: dict) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        n = (name or "").strip()
        if n and n not in seen:
            seen.add(n)
            names.append(n)

    for pd in section.get("places_detail", []):
        add(pd.get("name") or "")
    for m in re.findall(r"\*\*(.+?)\*\*", section.get("content", "")):
        add(m)
    return names


def normalize_itinerary_response(result: dict) -> None:
    for section in result.get("sections", []):
        title = (section.get("title") or "").strip()
        if re.search(r"여행\s*팁", title, re.IGNORECASE):
            section["icon"] = section.get("icon") or "💡"
            section["places_detail"] = []
            continue

        stripped = DAY_TITLE_EMOJI_RE.sub("", title)
        is_day = DAY_SECTION_TITLE_RE.match(stripped) or DAY_TITLE_PREFIX_RE.match(stripped)
        if is_day:
            section["title"] = _normalize_day_title(title)
            section["icon"] = "🗺️"

        content = section.get("content")
        if content:
            cleaned = [_clean_itinerary_line(line) for line in content.split("\n")]
            section["content"] = "\n".join(line for line in cleaned if line.strip())

        for pd in section.get("places_detail", []):
            warnings = pd.get("warnings") or []
            pd["warnings"] = [
                NUMBERED_LINE_RE.sub("", w).strip() for w in warnings
            ]


def collect_place_names_for_api(
    result: dict, limit: int = 5, itinerary: bool = False
) -> list[str]:
    """content **장소명** + places_detail.name 수집 (최대 limit개)."""
    if itinerary:
        day_attractions: list[str] = []
        rest_pool: list[str] = []
        lodging_pool: list[str] = []
        seen_sections: set[str] = set()

        for section in result.get("sections", []):
            title = section.get("title") or ""
            if re.search(r"여행\s*팁", title, re.I):
                continue
            names = _section_place_names(section)
            if _is_lodging_section(section):
                lodging_pool.extend(names)
                continue
            if _is_day_section_title(title):
                primary = None
                for prio in (0, 1, 2):
                    candidates = [n for n in names if _place_photo_priority(n) == prio]
                    if candidates:
                        primary = candidates[0]
                        break
                if primary:
                    day_attractions.append(primary)
                for n in names:
                    if _place_photo_priority(n) < 99 and n not in seen_sections:
                        seen_sections.add(n)
                        rest_pool.append(n)
            else:
                for n in names:
                    if _place_photo_priority(n) < 99 and n not in seen_sections:
                        seen_sections.add(n)
                        rest_pool.append(n)

        picked: list[str] = []
        seen: set[str] = set()

        def pick(name: str) -> None:
            n = (name or "").strip()
            if n and n not in seen and len(picked) < limit:
                seen.add(n)
                picked.append(n)

        for n in day_attractions:
            pick(n)
        rest_pool.sort(key=_place_photo_priority)
        for n in rest_pool:
            pick(n)
        lodging_pool.sort(key=_place_photo_priority)
        for n in lodging_pool:
            pick(n)
        return picked[:limit]

    seen: set[str] = set()
    names: list[str] = []

    def add(name: str) -> None:
        n = (name or "").strip()
        if not n or n in seen:
            return
        seen.add(n)
        names.append(n)

    for section in result.get("sections", []):
        if re.search(r"여행\s*팁", section.get("title", ""), re.I):
            continue
        for pd in section.get("places_detail", []):
            add(pd.get("name") or "")
        for m in re.findall(r"\*\*(.+?)\*\*", section.get("content", "")):
            add(m)

    if len(names) > limit:
        names.sort(key=_place_photo_priority)

    return names[:limit]


class SearchRequest(BaseModel):
    query: str
    city: str = None
    category: str = None
    travel_style: str = None
    match_threshold: float = 0.65
    match_count: int = 20


async def get_place_details(place_name: str, city: str = None) -> dict:
    query = f"{place_name} {city}" if city else place_name

    async with httpx.AsyncClient(timeout=10.0) as client:
        search_res = await client.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
                "X-Goog-FieldMask": "places.displayName,places.location,places.photos"
            },
            json={
                "textQuery": query,
                "languageCode": "ko"
            }
        )
        data = search_res.json()

        if not data.get("places"):
            return None

        place = data["places"][0]
        lat = place["location"]["latitude"]
        lng = place["location"]["longitude"]

        photo_urls = []
        if place.get("photos"):
            for photo in place["photos"][:3]:
                photo_urls.append(
                    f"https://places.googleapis.com/v1/{photo['name']}/media"
                    f"?maxWidthPx=800&key={GOOGLE_PLACES_API_KEY}"
                )

        return {
            "lat": lat,
            "lng": lng,
            "photo_urls": photo_urls
        }


def youtube_video_id(url: str) -> str | None:
    m = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})", url or "")
    return m.group(1) if m else None


def format_youtube_item(v: dict) -> dict:
    """youtube_videos.title 컬럼 값 사용."""
    title = (v.get("title") or "").strip()
    url = (v.get("url") or "").strip()
    return {
        "title": title or "YouTube 영상",
        "url": url,
    }


async def enrich_youtube_titles(videos: list[dict]) -> list[dict]:
    """RPC 결과에 title이 없으면 youtube_videos 테이블에서 url/영상ID로 보강."""
    if not videos:
        return videos

    urls = list(dict.fromkeys(
        (v.get("url") or "").strip() for v in videos if (v.get("url") or "").strip()
    ))
    if not urls:
        return videos

    title_by_url: dict[str, str] = {}
    title_by_id: dict[str, str] = {}

    def register_title(url: str, title: str) -> None:
        if not url or not title:
            return
        title_by_url[url.strip()] = title
        vid = youtube_video_id(url)
        if vid:
            title_by_id[vid] = title

    def lookup_title(url: str, current: str) -> str:
        if current:
            return current
        if url in title_by_url:
            return title_by_url[url]
        vid = youtube_video_id(url)
        if vid and vid in title_by_id:
            return title_by_id[vid]
        return ""

    try:
        for i in range(0, len(urls), 40):
            batch = urls[i:i + 40]
            db_res = await asyncio.to_thread(
                lambda links=batch: supabase.table("youtube_videos")
                .select("url,title")
                .in_("url", links)
                .execute()
            )
            for row in db_res.data or []:
                register_title(row.get("url") or "", (row.get("title") or "").strip())

        # url 문자열이 조금 달라도(youtu.be 등) video id로 한 번 더 조회
        missing_ids = list(dict.fromkeys(
            youtube_video_id(u) for u in urls
            if youtube_video_id(u) and not lookup_title(u, "")
        ))
        for vid in missing_ids[:10]:
            db_res = await asyncio.to_thread(
                lambda video_id=vid: supabase.table("youtube_videos")
                .select("url,title")
                .ilike("url", f"%{video_id}%")
                .limit(1)
                .execute()
            )
            for row in db_res.data or []:
                register_title(row.get("url") or "", (row.get("title") or "").strip())
    except Exception as e:
        print(
            f"youtube_videos title 조회 실패: {e}",
            flush=True,
            file=sys.stderr,
        )

    enriched: list[dict] = []
    for v in videos:
        url = (v.get("url") or "").strip()
        title = lookup_title(url, (v.get("title") or "").strip())
        enriched.append({**v, "title": title, "url": url})
    return enriched


@app.post("/search")
async def search(req: SearchRequest):
    # 1. 쿼리 임베딩
    result = await gemini_client.aio.models.embed_content(
        model="gemini-embedding-2",
        contents=req.query,
        config={"output_dimensionality": 768}
    )
    query_vector = result.embeddings[0].values

    # 2. 벡터 검색 (유튜브는 RPC 없거나 실패 시 빈 배열 — 후기 검색은 계속)
    res = await asyncio.to_thread(
        lambda: supabase.rpc("match_travel_chunks", {
            "query_embedding": query_vector,
            "match_threshold": req.match_threshold,
            "match_count": req.match_count,
            "filter_city": req.city,
            "filter_category": req.category,
            "filter_travel_style": req.travel_style
        }).execute()
    )

    chunks = res.data

    youtube_videos = []
    try:
        youtube_res = await asyncio.to_thread(
            lambda: supabase.rpc("match_youtube_videos", {
                "query_embedding": query_vector,
                "match_threshold": 0.6,
                "match_count": 3,
                "filter_city": req.city
            }).execute()
        )
        youtube_videos = youtube_res.data or []
        youtube_videos = await enrich_youtube_titles(youtube_videos)
    except Exception as e:
        print(
            f"match_youtube_videos 실패 (후기 검색은 계속): {e}",
            flush=True,
            file=sys.stderr,
        )

    if not chunks:
        return {
            "summary": "관련 후기가 충분하지 않아요.",
            "sections": [],
            "warning": [],
            "places": None,
            "follow_up": [],
            "sources": [],
            "youtube_videos": [
                format_youtube_item(v)
                for v in youtube_videos
                if (v.get("url") or "").strip()
            ],
        }

    # 3. 컨텍스트 구성
    print(f"\n=== 검색된 청크 {len(chunks)}개 ===")
    for i, c in enumerate(chunks):
        print(f"[{i+1}] similarity={c.get('similarity', '?'):.3f} | {c.get('title','')[:30]} | {c['text'][:60]}")

    GENERIC_TITLES = {
        "",
        "네이버 카페 후기",
        "네이버 카페",
        "네이버 블로그 후기",
        "네이버후기",
        "네이버 후기",
    }

    def is_generic_title(title: str | None) -> bool:
        t = (title or "").strip()
        if not t or t in GENERIC_TITLES:
            return True
        if "네이버" in t and "후기" in t and len(t) <= 24:
            return True
        if t.endswith("?") or t.endswith("？"):
            return True
        return False

    def extract_title(text: str) -> str | None:
        """text에서 '제목: ...' 추출 (청크 분할돼도 본문 어디든 검색)."""
        if not text:
            return None
        text = text.strip().lstrip("\ufeff").replace("\r\n", "\n")

        for line in text.split("\n")[:20]:
            line = line.strip()
            m = re.match(r"^제목\s*[:：]\s*(.+)$", line, re.IGNORECASE)
            if m:
                return m.group(1).strip() or None

        m = re.search(
            r"(?:^|\n)제목\s*[:：]\s*(.+?)(?:\n|$)",
            text[:4000],
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip() or None

        m = re.search(r"\[제목:\s*(.+?)\]", text[:800])
        if m:
            return m.group(1).strip() or None

        m = re.search(r"\[제목\s*[:：]\s*(.+?)\]", text[:800])
        if m:
            return m.group(1).strip() or None

        return None

    def title_from_text_fallback(text: str, max_len: int = 56) -> str:
        """제목: 줄이 없을 때 본문 첫 유의미 줄을 제목 후보로."""
        if not text:
            return ""
        text = text.strip().replace("\r\n", "\n")
        skip = re.compile(
            r"^(제목|작성자|날짜|출처|링크|url|http|www\.|ref:|\[ref:)",
            re.I,
        )
        for line in text.split("\n")[:15]:
            line = line.strip()
            line = re.sub(r"^[#*\-\d.)\s]+", "", line).strip()
            if len(line) < 8 or len(line) > 100:
                continue
            if skip.match(line):
                continue
            return line[:max_len].strip()
        return ""

    def resolve_chunk_title(chunk: dict) -> str:
        db_title = (chunk.get("title") or "").strip()
        text = chunk.get("text", "") or ""
        extracted = extract_title(text)
        fallback = title_from_text_fallback(text)

        for candidate in (extracted, db_title, fallback):
            if candidate and not is_generic_title(candidate):
                return candidate.strip()

        for candidate in (extracted, db_title, fallback):
            if candidate and candidate.strip():
                return candidate.strip()

        return ""

    chunk_titles_by_id = {i + 1: resolve_chunk_title(c) for i, c in enumerate(chunks)}

    # 같은 링크 청크 중 제목이 있는 조각을 우선 사용 (벡터 분할 대응)
    link_best_titles: dict[str, str] = {}
    for c in chunks:
        link = c.get("link") or ""
        if not link:
            continue
        title = resolve_chunk_title(c)
        prev = link_best_titles.get(link)
        if prev is None or (is_generic_title(prev) and not is_generic_title(title)):
            link_best_titles[link] = title

    # 같은 링크의 다른 청크(제목 줄만 있는 조각)에서 제목 보강
    chunk_links = list(dict.fromkeys([c.get("link") or "" for c in chunks if c.get("link")]))
    for i in range(0, len(chunk_links), 40):
        batch = chunk_links[i:i + 40]
        db_res = await asyncio.to_thread(
            lambda links=batch: supabase.table("travel_chunks")
            .select("link,text")
            .in_("link", links)
            .execute()
        )
        for row in db_res.data or []:
            link = row.get("link") or ""
            if not link:
                continue
            title = resolve_chunk_title(row)
            prev = link_best_titles.get(link)
            if prev is None or (is_generic_title(prev) and not is_generic_title(title)):
                link_best_titles[link] = title

    chunk_titles_by_link = {
        c["link"]: link_best_titles.get(c["link"], chunk_titles_by_id[i + 1])
        for i, c in enumerate(chunks)
    }

    def best_title_for_chunk(ref_id: int, chunk: dict) -> str:
        by_id = chunk_titles_by_id.get(ref_id, resolve_chunk_title(chunk))
        link = chunk.get("link") or ""
        by_link = link_best_titles.get(link) if link else None
        if by_link and not is_generic_title(by_link):
            return by_link
        if not is_generic_title(by_id):
            return by_id
        return by_link or by_id

    context = "\n\n".join([
        f"[id:{i + 1}] [출처: {c['link']}] [날짜: {c.get('date', '')}] [제목: {best_title_for_chunk(i + 1, c)}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

    itinerary_query = is_itinerary_query(req.query)
    system_prompt = build_system_prompt(req.query)

    # 4. Gemini 답변 생성
    response = await gemini_client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{system_prompt}\n\n질문: {req.query}\n\n참고 후기:\n{context}",
        config={"thinking_config": {"thinking_budget": 0}}
    )

    # 5. JSON 파싱
    try:
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
    except Exception as e:
        print(f"JSON 파싱 실패: {e}")
        print(f"응답 텍스트: {response.text[:500]}")
        result = {
            "summary": response.text,
            "sections": [],
            "warning": [],
            "places": None,
            "follow_up": [],
            "sources": []
        }

    print(f"\n=== LLM 응답 ===", flush=True, file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True, file=sys.stderr)

    enrich_place_warnings(result)
    if itinerary_query:
        normalize_itinerary_response(result)

    # 6. content·places_detail 장소명 → Places API (최대 5개)
    place_names = collect_place_names_for_api(result, limit=5, itinerary=itinerary_query)

    places = []
    if place_names:
        tasks = [get_place_details(name, req.city) for name in place_names]
        details_list = await asyncio.gather(*tasks)
        for name, details in zip(place_names, details_list):
            if details:
                places.append({
                    "day": None,
                    "name": name,
                    "lat": details["lat"],
                    "lng": details["lng"],
                    "photo_urls": details["photo_urls"],
                    "description": ""
                })

    result["places"] = places if places else None

    def collect_cited_ref_ids(payload: dict) -> set[int]:
        refs: set[int] = set()

        def scan(text: str | None) -> None:
            if not text:
                return
            for m in re.findall(r"\[ref:(\d+)\]", text):
                refs.add(int(m))

        scan(payload.get("summary"))
        for section in payload.get("sections", []):
            scan(section.get("content"))
            table = section.get("table")
            if table and isinstance(table.get("rows"), list):
                for row in table["rows"]:
                    if isinstance(row, list):
                        for cell in row:
                            scan(cell)
            for pd in section.get("places_detail", []):
                scan(pd.get("description"))
                for warning in pd.get("warnings", []):
                    scan(warning)
                for review in pd.get("reviews", []):
                    scan(review.get("text"))
                    ref = review.get("ref")
                    if ref is not None:
                        try:
                            refs.add(int(ref))
                        except (TypeError, ValueError):
                            pass
            # 구 스키마 호환
            for review in section.get("reviews", []):
                ref = review.get("ref")
                if ref is not None:
                    try:
                        refs.add(int(ref))
                    except (TypeError, ValueError):
                        pass
        return refs

    def chunk_to_source(ref_id: int, chunk: dict, title: str) -> dict:
        link = chunk.get("link", "")
        channel = "네이버 블로그" if "blog.naver.com" in link else "네이버 카페"
        return {
            "id": ref_id,
            "title": title,
            "channel": channel,
            "date": chunk.get("date", ""),
            "link": link,
        }

    # 7. 본문 [ref:N] ↔ sources 동기화 + 제목 보정 + 중복 제거 + 모바일 URL
    cited_refs = collect_cited_ref_ids(result)
    sources_by_id: dict[int, dict] = {}

    for source in result.get("sources", []):
        try:
            sid = int(source.get("id"))
        except (TypeError, ValueError):
            continue
        sources_by_id[sid] = source

    for ref_id in cited_refs:
        if ref_id < 1 or ref_id > len(chunks):
            continue
        if ref_id in sources_by_id:
            continue
        chunk = chunks[ref_id - 1]
        sources_by_id[ref_id] = chunk_to_source(
            ref_id,
            chunk,
            best_title_for_chunk(ref_id, chunk),
        )

    if sources_by_id:
        result["sources"] = sorted(sources_by_id.values(), key=lambda s: int(s["id"]))
    elif result.get("sources") is None:
        result["sources"] = []

    if result.get("sources"):
        seen_links = set()
        unique_sources = []
        for source in result["sources"]:
            link = source.get("link")
            try:
                sid = int(source.get("id"))
            except (TypeError, ValueError):
                sid = None

            if link and link in seen_links and sid not in cited_refs:
                continue
            if link:
                seen_links.add(link)
            unique_sources.append(source)
        result["sources"] = unique_sources

        for source in result["sources"]:
            ref_id = source.get("id")
            if ref_id is not None:
                try:
                    ref_id = int(ref_id)
                except (TypeError, ValueError):
                    ref_id = None

            link = source.get("link")
            resolved = None
            if isinstance(ref_id, int) and 1 <= ref_id <= len(chunks):
                resolved = best_title_for_chunk(ref_id, chunks[ref_id - 1])
            elif link and link in link_best_titles:
                resolved = link_best_titles[link]
            elif isinstance(ref_id, int) and ref_id in chunk_titles_by_id:
                resolved = chunk_titles_by_id[ref_id]

            # LLM 제목 → 서버 추출 제목으로 교체 (non-generic 우선)
            if resolved:
                if not is_generic_title(resolved):
                    source["title"] = resolved
                elif is_generic_title(source.get("title")):
                    source["title"] = resolved

            if is_generic_title(source.get("title")):
                link = source.get("link") or ""
                if link and link in link_best_titles:
                    alt = link_best_titles[link]
                    if alt and not is_generic_title(alt):
                        source["title"] = alt

            if is_generic_title(source.get("title")) and isinstance(ref_id, int) and 1 <= ref_id <= len(chunks):
                fb = title_from_text_fallback(chunks[ref_id - 1].get("text", ""))
                if fb and not is_generic_title(fb):
                    source["title"] = fb

            if link and "blog.naver.com" in link and "m.blog.naver.com" not in link:
                source["link"] = link.replace("https://blog.naver.com", "https://m.blog.naver.com")

    # 유튜브 링크 추가
    result["youtube_videos"] = [
        format_youtube_item(v)
        for v in youtube_videos
        if (v.get("url") or "").strip()
    ]

    return result


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4)