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
- summary: 1~2문장, 40자 이내. **[ref:N] 표기 금지** (출처 없이 요약만)
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
- 모든 쿼리에서 icon을 빈값("")으로 두고, title 앞에 1️⃣ 2️⃣ 3️⃣ 4️⃣ 순서로 붙일 것.
- 단, 여행 팁은 넘버링 금지. [일정형 쿼리 처리] 규칙 따를 것.
- 마지막 여행 팁 섹션은 icon 💡, 넘버링 없음.
- 일정형(~일정/코스/동선/N박N일)은 이 규칙을 적용하지 말고 [일정형 쿼리 처리]만 따를 것.

[섹션 구성 원칙 — 추천형 전용, 일정형에는 적용 금지]
- 추천형 쿼리는 쿼리의 동행인/목적/여행스타일을 먼저 파악.
- 섹션 제목은 단순 카테고리명이 아니라 "카테고리 (이 사람에게 왜 맞는지)" 형식으로 작성.
  예) 혼여 숙소 쿼리 →
  icon: "", title: "1️⃣ 위치+편의성 최강 (혼자 여행 기본 선택)"
  icon: "", title: "2️⃣ 가성비+혼자 최적 (잠만 자면 이거)"
  icon: "", title: "3️⃣ 힐링형 (피로 풀고 싶으면)"
  icon: "💡", title: "상황별추천"
- 섹션당 장소 1~2개씩 배분 가능. 단, 답변 전체 고유 **장소명** 합은 최대 5개.
- 마지막 섹션은 반드시 title "상황별추천" 으로 끝낼 것 (icon: "💡").
  content 예)
  ✔ 첫 혼여/편하게 → **호텔명**
  ✔ 가성비+잠만 → **호텔명**
  👉 한 줄 결론: 혼여면 역세권 비즈니스 호텔이 정답
- 카테고리는 후기 데이터에 있는 내용 기준으로만. 없는 카테고리 만들지 말 것.

[일정형 쿼리 처리 — 일정형일 때 최우선, 추천형 규칙 무시]
- ~일정, ~코스, ~동선, N박N일 키워드면 일정형으로 판단.
- 섹션 구성: icon "" (빈값), title "DAY1 — 소제목" 형식. 1️⃣·"1일차"·Day1 소문자 금지. 반드시 DAY1, DAY2.
- Day content 형식 (오전/오후/저녁 라벨 사용 금지):
  각 장소: 이모지+**실제 장소명** 한 줄 → 다음 줄에 이동수단·소요시간(약 N분/N시간 필수).
  **카테고리만 있는 줄 금지** (예: "점심 및 쇼핑", "숙소 체크인" — 반드시 **후지노미야 OO호텔** 등 실명).
  **장소명 줄(이모지+**장소명**)에는 [ref:N] 절대 금지.** [ref:N]은 이동/설명 줄에만.
  **같은 Day에서 동일 **장소명** 반복 금지.** 재방문·이동은 첫 블록 설명 줄에 합칠 것 (🚆 렌터카 반납 후 출국, 약 30분).
  Day 내부에 숫자 나열(1)2)3))·- 불릿·• 금지. 줄바꿈으로만 구분.
- Day 섹션 content에 🏨 숙소 넣지 말 것. 숙소는 별도 섹션으로 분리.
- Day 섹션 다음·여행 팁 직전에 icon "🏨", title "숙소 추천" (title에 🏨 이모지 중복 금지, icon만 사용).
- Day에는 🍜 맛집·⛩️ 관광·🚆 이동 포함 가능.
- **전체 sections의 places_detail name 합(숙소 섹션 포함) 최대 5개 엄수.**
  Day가 3개면 Day당 1~2개만. 숙소 2개 + Day 장소 3개 = 5개가 한계.
  6개 이상 places_detail 생성 금지 — JSON 출력이 깨질 수 있음.
- places_detail: 각 Day의 content **장소명**마다 항목 필수. 빈 배열 금지 (숙소 섹션은 places_detail 필수).
  reviews 최대 3개, warnings negative 기반. 일정형도 추천형과 동일하게 필수.
- 마지막 섹션은 상황별추천이 아니라 여행 팁으로 끝낼 것.
- 마지막 섹션: title "여행 팁" (이모지 없이), icon "💡", places_detail: []
- 여행 팁 content 형식은 상황별추천과 동일:
  - 짧은 본문 줄 나열 (✔ 교통 / ✔ 렌터카 / ✔ 주의사항 등)
  - 각 줄 끝 [ref:N] 필수 (후기 근거)
  - **장소명 블록·이모지+장소·사진 형식 사용 금지**
  - 👉 한 줄 결론 1줄 가능
- content 불릿은 "이동해요 / 식사해요" 같은 행동 나열 금지.
  반드시 해당 장소의 특징·추천 이유·팁을 담을 것.
  후기 데이터에 정보 없으면 해당 장소 불릿 생략.
- reviews는 실제 경험 서술 문장만. 
  시간표형("8시 저녁ㅡ 돈요시"), 질문형, 일정 나열형 금지.


[places_detail 생성 기준]
- 추천형·일정형 Day 섹션 모두 places_detail 배열 필수. 섹션 레벨 reviews 필드 사용 금지.
- 일정형: Day 섹션 places_detail 비우면 안 됨. content 장소 수 = places_detail 항목 수.
- 전체 sections의 places_detail name 합(중복 제외) 최대 5개. content의 **장소명** 개수와 동일해야 함.
- places_detail 항목 수 = content의 **장소명** 항목 수와 동일. 순서도 동일하게.
- name: content의 **장소명**과 정확히 일치
- description: 2~3문장, 문장당 30~40자. 위치, 분위기, 특징, 추천 이유 포함. [ref:N] 포함 가능.
- reviews: 해당 장소 **직접 방문·체험 후기**만. 다른 장소·다른 메뉴 후기 섞지 말 것.
  ✗ 예: "아키요시 타이메시 본점" places_detail에 "말차 모찌" 후기 (다른 가게/디저트) 넣기 금지
  ✓ 후기는 장소명·대표 메뉴(도미밥/타이메시 등)와 직접 관련된 경험만
- **reviews 제외:** 질문·문의(?/궁금/할까요), 일정·동선 나열(/, ->), 의견·제안만(~포기하면, ~넣고 싶)
- **reviews 포함:** 방문 소감, 맛·분위기·동선 팁, 일정·동선 조언, 추천/비추, 아쉬운 **경험**
- 장소당 실후기 **2~3개 필수** (참고 후기에 2개 이상 있으면 반드시 2개 이상). **1개만 넣거나 reviews 빈 배열 금지.**
- reviews.text: 참고 후기 **원문 전체**를 줄임·요약 없이 복사. 첫 문장만 잘라내지 말 것. 2~4문장·줄바꿈 포함 가능.
- 후기 원문 그대로 인용, 요약·한 줄 압축 금지.
- 부정 후기 1개 이상 포함 (질문형·의견형 negative 금지, 실제 아쉬운 **경험**만)
- sentiment: 긍정 "positive", 부정/아쉬운 점 "negative"
- warnings: **negative reviews에서 주의사항을 15자 이내로 요약**하여 반드시 추출. 예약/휴무/막차/현금/입장제한/대기 등이 후기에 있으면 warnings에 1~2개 넣을 것. 비워두지 말 것. root warning 필드 사용 금지.
- 팁·결론만 있는 섹션(여행 팁)은 places_detail: []


[reviews 생성 기준]
- 실제 경험 서술 문장만 인용. (~했어요, ~였어요, ~좋았어요, ~별로였어요)
- 아래는 reviews에 넣지 말 것:
  ✗ 질문형 (~까요?, ~나요?, ~죠?, ~할까요?)
  ✗ 시간표형 ("8시 저녁ㅡ 돈요시", "1)2)3)" 형태)
  ✗ 일정 나열형 (장소명만 나열)
  ✗ 타인 의견 인용 ("~하라고 하더라구요")
  ✗ 계획/의도 서술 ("~할 예정이에요", "~넣고 싶어요")
  ✗ 한 문장만 잘라낸 단편·중간에서 끊긴 문장 ("움직이시"처럼 미완성). 원문 **끝까지** 복사
- 퀄리티 좋은 후기 기준:
  ✔ 구체적 경험 ("웨이팅 30분 기다렸는데 그만한 가치 있었어요")
  ✔ 감정/느낌 포함 ("부모님이 너무 좋아하셨어요")
  ✔ 비교/대조 ("다른 곳보다 여기가 훨씬 나았어요")
  ✔ 구체적 디테일 ("2층 좌식 테이블에 앉았는데 아이가 편해했어요")
- sentiment 판단 기준:
  positive:
    ✔ 만족/추천 표현 ("좋았어요", "추천해요", "또 가고 싶어요")
    ✔ 구체적 장점 ("뷰가 정말 좋았어요", "직원이 친절했어요")
    ✔ 부모님/동행 만족 ("부모님이 너무 좋아하셨어요")
  negative:
    ✔ 실망/비추 표현 ("별로였어요", "다시는 안 갈 것 같아요")
    ✔ 구체적 단점 ("계단이 너무 많아서 힘들었어요")
    ✔ 아쉬운 경험 ("웨이팅이 너무 길었어요", "가격 대비 아쉬웠어요")
  negative 금지:
    ✗ 질문형 우려 ("~힘들지 않을까요?")
    ✗ 타인 전달 ("~별로라고 하더라구요")
    ✗ 단순 조건 제시 ("~하면 괜찮을 것 같아요")


[warning 생성 기준 — places_detail.warnings]
- **negative reviews의 주의·아쉬운 점을 warnings로 변환** (아래 유형 우선, 실제 방문 경험 기반)
- 아래 케이스 위주로 warning 생성. **실질적 주의·아쉬운 점**이면 15자 이내로 요약하여 포함.
  ✔ 예약/티켓 필수
  ✔ 영업시간/휴무일
  ✔ 교통/이동 실질적 주의
  ✔ 신체 부담 (계단/경사)
  ✔ 현금only
  ✔ 직원 서비스 아쉬움
  ✔ 위생 주의
  ✔ 혼잡 주의
  ✔ 가격 대비 아쉬움
- 개인 의견, 일정 부족 느낌, 질문형 문장 → warning 생성 금지
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
- 반드시 도시명을 포함한 완성된 질문으로 작성. 도시명 없는 follow_up 생성 금지.
  예) "마쓰야마 부모님 여행 맛집 추천해 주세요" O
      "맛집 추천해 주세요" X (도시명 없음 → 금지)
- 쿼리에 동행인/여행스타일이 있으면 그것도 포함.
  예) "마쓰야마 부모님과 가기 좋은 관광지 알려주세요" O"""


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
    (re.compile(r"불친절|불쾌|무뚝뚝|직원.*별로"), "직원 서비스 아쉬움"),
    (re.compile(r"끈적|눅눅|위생|더럽|지저분"), "위생 주의"),
    (re.compile(r"맵|짜|달|느끼|비려"), "맛 호불호 있음"),
    (re.compile(r"비싸|가성비.*별로|가격.*아깝"), "가격 대비 아쉬움"),
    (re.compile(r"시끄|복잡|사람.*많|붐비"), "혼잡할 수 있음"),
    (re.compile(r"교통|이동|주차|운전|운행|셔틀|shuttle", re.I), "교통·이동 주의"),
    (re.compile(r"계단|경사|힘들|체력|몸\s*아"), "신체 부담 주의"),
]

SCHEDULE_FEELING_RE = re.compile(
    r"시간.*(없|부족|짧|너무)|체류.*짧|너무\s*없|일정.*부족|촉박|포기하면|넣고\s*싶",
    re.I,
)

QUESTION_RE = re.compile(
    r"[?？]|궁금합니다|궁금해요|궁금한|할까요|될까요|을까요|를까요|인지\s*궁금|할까\?|될까\?|어떻게\s*해야|알려주세요",
    re.I,
)
OPINION_RE = re.compile(r"포기하면|이견|넣고\s*싶은데|넣고\s*싶어", re.I)
ITINERARY_DUMP_RE = re.compile(
    r"(?:/|->|→).*(?:/|->|→)|주차장-|복귀.*취침|저녁식사후|하부\s*무료",
    re.I,
)

SKIP_MATCH_TOKENS = {
    "본점", "지점", "점", "마쓰야마", "오사카", "교토", "도쿄", "후쿠오카", "나고야",
    "삿포로", "오키나와", "일본", "여행", "식당", "카페", "레스토랑", "호텔", "숙소",
}

DESSERT_MARKERS_RE = re.compile(
    r"(?:말차|모찌|아이스크림|케이크|디저트|마카롱|와플|빙수|パフェ|パンケーキ)",
    re.I,
)
SAVORY_MARKERS_RE = re.compile(
    r"(?:도미밥|타이메시|타마고|냉면|라멘|스시|초밥|회|우동|소바|丼|焼き|定食|고기|삼겹|갈비)",
    re.I,
)


def extract_place_match_terms(place_name: str, description: str = "") -> list[str]:
    terms: set[str] = set()
    name = re.sub(r"\*\*", "", place_name or "").strip()
    desc = re.sub(r"\s*\[ref:\d+\]", "", description or "")
    desc = re.sub(r"\*\*", "", desc).strip()

    def add(raw: str) -> None:
        cleaned = re.sub(r"(?:본점|지점|점)$", "", raw).strip()
        if len(cleaned) >= 2 and cleaned not in SKIP_MATCH_TOKENS:
            terms.add(cleaned)

    for part in re.split(r"[\s·・/]+", name):
        add(part)

    for w in re.findall(r"[가-힣]{2,}|[a-zA-Z]{3,}|[ぁ-んァ-ン一-龯]{2,}", f"{name} {desc}"):
        add(w)

    return list(terms)


def is_review_relevant_to_place(text: str, place_name: str, description: str = "") -> bool:
    review = (text or "").strip()
    if not review:
        return False

    terms = extract_place_match_terms(place_name, description)
    if not terms:
        return True

    if any(t in review for t in terms):
        return True

    context = f"{place_name} {description}"
    place_savory = bool(SAVORY_MARKERS_RE.search(context))
    place_dessert = bool(DESSERT_MARKERS_RE.search(context))

    if place_savory and DESSERT_MARKERS_RE.search(review) and not SAVORY_MARKERS_RE.search(review):
        return False
    if place_dessert and SAVORY_MARKERS_RE.search(review) and not DESSERT_MARKERS_RE.search(review):
        return False

    return False


def is_relaxed_review_text(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 8:
        return False
    if ITINERARY_DUMP_RE.search(t):
        return False
    if t.count("/") >= 3:
        return False
    if re.search(r"폭포.*/.*호수|호수.*/.*폭포", t, re.I):
        return False
    if re.search(r"[?？]", t):
        return False
    return True


def is_valid_review_text(text: str) -> bool:
    t = (text or "").strip()
    if not is_relaxed_review_text(t):
        return False
    if QUESTION_RE.search(t):
        return False
    if OPINION_RE.search(t):
        return False
    return True


def pick_place_reviews(
    reviews: list,
    min_count: int = 2,
    max_count: int = 3,
    place_name: str = "",
    description: str = "",
) -> list:
    strict: list = []
    relaxed_pool: list = []
    raw_pool: list = []
    has_place_context = bool((place_name or "").strip())

    def is_relevant(r: dict) -> bool:
        if not has_place_context:
            return True
        return is_review_relevant_to_place(r.get("text") or "", place_name, description)

    for r in reviews or []:
        if not isinstance(r, dict):
            continue
        text = (r.get("text") or "").strip()
        if not text or not is_relevant(r):
            continue
        raw_pool.append(r)
        if is_valid_review_text(text):
            strict.append(r)
        elif is_relaxed_review_text(text):
            relaxed_pool.append(r)

    out = strict[:max_count]
    seen = {(r.get("text") or "").strip() for r in out}

    def append(r: dict) -> None:
        text = (r.get("text") or "").strip()
        if not text or text in seen:
            return
        out.append(r)
        seen.add(text)

    for r in relaxed_pool:
        if len(out) >= max_count:
            break
        append(r)

    if len(out) < min_count:
        for r in raw_pool:
            if len(out) >= min_count:
                break
            text = (r.get("text") or "").strip()
            if len(text) < 8 or text in seen:
                continue
            if QUESTION_RE.search(text):
                continue
            if ITINERARY_DUMP_RE.search(text):
                continue
            append(r)

    return out[:max_count]


def filter_valid_reviews(reviews: list) -> list:
    return pick_place_reviews(reviews)


def _strip_warning_endings(body: str) -> str:
    body = re.sub(
        r"(?:입니다|습니다|해요|돼요|있어요|없어요|주세요|에요|예요|네요|같아요)[.!]?$",
        "",
        body,
    )
    body = re.sub(r"[.!?…]+$", "", body).strip()
    return re.sub(r"(?:없는\s*건?|너무\s*없)$", "", body).strip()


def _warning_clause_from_review(text: str, max_len: int = 15) -> str:
    clause = re.sub(r"\[ref:\d+\]", "", text.replace("**", "")).split("\n")[0]
    clause = re.split(r"[.。!?]", clause)[0].strip()
    if len(clause) < 4:
        return ""
    if QUESTION_RE.search(clause) or SCHEDULE_FEELING_RE.search(clause):
        return ""
    clause = _strip_warning_endings(clause)
    if len(clause) < 4:
        return ""
    return clause[:max_len]


def sanitize_warning_text(text: str) -> str:
    suffix_m = re.search(r"(\s*(?:\[ref:\d+\])+)\s*$", text or "")
    suffix = suffix_m.group(1) if suffix_m else ""
    body = INLINE_REF_RE.sub(" ", (text or "").replace("**", "")).strip()
    body = re.sub(r"^⚠️\s*", "", body).strip()
    if not body:
        return suffix.strip()

    for pattern, label in CAUTION_RULES:
        if pattern.search(body):
            return f"{label}{suffix}"

    body = _strip_warning_endings(body)

    if QUESTION_RE.search(body) or SCHEDULE_FEELING_RE.search(body):
        return f"주의사항 확인{suffix}" if suffix else ""

    if len(body) > 16:
        body = body[:15].rstrip()

    return f"{body}{suffix}" if body else suffix.strip()


def _ref_suffix_for_review(review: dict, text: str) -> str:
    ref = review.get("ref")
    if ref is not None:
        return f" [ref:{ref}]"
    m = re.search(r"(\s*(?:\[ref:\d+\])+)\s*$", text)
    return m.group(1) if m else ""


def infer_warnings_from_reviews(reviews: list) -> list[str]:
    if not reviews:
        return []

    out: list[str] = []
    seen: set[str] = set()

    for review in reviews:
        if review.get("sentiment") != "negative":
            continue
        text = review.get("text") or ""
        if not is_valid_review_text(text):
            continue

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
            clause = _warning_clause_from_review(text)
            if clause:
                w = f"{clause}{ref_suffix}"
                if w not in seen:
                    seen.add(w)
                    out.append(w)

        if len(out) >= 2:
            break

    return out[:2]


MAX_REVIEW_CHUNK_HOPS = 3

KOREAN_SENTENCE_END_RE = re.compile(
    r"(?:요|다|니다|습니다|해요|돼요|있어요|없어요|네요|죠|래요|이에요|예요|구요|군요|게요|"
    r"세요|시요|시죠|습니까|까요)[.!?…]?\s*$"
)


def is_review_text_truncated(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if re.search(r'[.!?。]"?\s*$', t):
        return False
    if KOREAN_SENTENCE_END_RE.search(t):
        return False
    return True


def _fetch_travel_chunk_text(article_id, chunk_index: int) -> str | None:
    res = (
        supabase.table("travel_chunks")
        .select("text")
        .eq("article_id", article_id)
        .eq("chunk_index", chunk_index)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return None
    return (rows[0].get("text") or "").strip() or None


async def extend_truncated_review(text: str, chunk: dict) -> str:
    """reviews 후처리: 문장이 끊겼으면 같은 article_id의 다음 청크를 이어붙임."""
    article_id = chunk.get("article_id")
    chunk_index = chunk.get("chunk_index")
    if article_id is None or chunk_index is None:
        return text

    out = (text or "").strip()
    next_index = int(chunk_index)

    for _ in range(MAX_REVIEW_CHUNK_HOPS):
        if not is_review_text_truncated(out):
            break
        next_index += 1
        next_text = await asyncio.to_thread(
            _fetch_travel_chunk_text, article_id, next_index
        )
        if not next_text:
            break
        out = out + next_text

    return out


def _review_ref_id(review: dict) -> int | None:
    ref = review.get("ref")
    if ref is not None:
        try:
            return int(ref)
        except (TypeError, ValueError):
            pass
    text = review.get("text") or ""
    m = re.search(r"\[ref:(\d+)\]", text)
    if m:
        return int(m.group(1))
    return None


async def extend_result_reviews(result: dict, chunks: list) -> None:
    for section in result.get("sections", []):
        for pd in section.get("places_detail", []):
            for review in pd.get("reviews", []):
                if not isinstance(review, dict):
                    continue
                text = (review.get("text") or "").strip()
                if not text or not is_review_text_truncated(text):
                    continue
                ref_id = _review_ref_id(review)
                if ref_id is None or ref_id < 1 or ref_id > len(chunks):
                    continue
                extended = await extend_truncated_review(text, chunks[ref_id - 1])
                if extended != text:
                    review["text"] = extended


def postprocess_place_detail(pd: dict) -> None:
    pd["reviews"] = pick_place_reviews(
        pd.get("reviews", []),
        place_name=pd.get("name") or "",
        description=pd.get("description") or "",
    )
    raw_warnings = pd.get("warnings") or []
    pd["warnings"] = [
        w
        for w in (
            sanitize_warning_text(NUMBERED_LINE_RE.sub("", w).strip())
            for w in raw_warnings
            if w
        )
        if w
    ]
    if not pd["warnings"]:
        inferred = infer_warnings_from_reviews(pd.get("reviews", []))
        if inferred:
            pd["warnings"] = inferred


def enrich_place_warnings(result: dict) -> None:
    for section in result.get("sections", []):
        for pd in section.get("places_detail", []):
            postprocess_place_detail(pd)


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
□ Day 섹션 title "DAY1 — 소제목", icon "" (이모지 없음)
□ Day content: 실제 **장소명**만. 카테고리 줄·동일 장소명 중복 금지. 이동 줄에 약 N분/N시간 필수
□ Day에 🏨 숙소 없음 → icon 🏨 + title "숙소 추천" 섹션 별도 (places_detail 필수)
□ 전체 places_detail name 합(숙소 포함) **5개 이하** — Day당 1~2개만
□ 각 Day: content **장소명**마다 places_detail + reviews(최대 3) + warnings
□ 마지막만 title "여행 팁", icon "💡", places_detail: []
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
    else:
        m = DAY_TITLE_PREFIX_RE.match(t)
        if m:
            day_num = m.group(1)
            rest = t[m.end():].strip()
        else:
            return t

    if rest.startswith("—") or rest.startswith("-"):
        rest = " — " + rest.lstrip("—-").strip()
    elif rest:
        rest = f" — {rest}"
    else:
        rest = ""
    return f"DAY{day_num}{rest}"


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
    """사진 API 우선순위: 명소(0) → 숙소(1) → 맛집(2). 쇼핑·이동은 99(제외)."""
    if re.search(r"공항|이동수단|^이동$|출국|입국|도착", name, re.I):
        return 99
    if re.search(r"쇼핑|마켓|백화점|아울렛|면세", name, re.I):
        return 99
    if re.search(
        r"관광|신사|사찰|USJ|스튜디오|박물관|공원|타워|성|전망|이나리|유니버설|폭포|해변|계곡|온천|폭",
        name,
        re.I,
    ):
        return 0
    if re.search(r"호텔|숙소|료칸|게스트|민박|펜션|inn", name, re.I):
        return 1
    if re.search(r"맛집|식당|카페|타코|오코노미|라멘|스시", name, re.I):
        return 2
    return 0


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

        if _is_lodging_section(section):
            section["icon"] = "🏨"
            section["title"] = re.sub(r"^🏨\s*", "", title).strip() or "숙소 추천"
            if not section["title"].startswith("숙소"):
                section["title"] = "숙소 추천"
            continue

        stripped = DAY_TITLE_EMOJI_RE.sub("", title)
        is_day = DAY_SECTION_TITLE_RE.match(stripped) or DAY_TITLE_PREFIX_RE.match(stripped)
        if is_day:
            section["title"] = _normalize_day_title(title)
            section["icon"] = ""

        content = section.get("content")
        if content:
            cleaned = [_clean_itinerary_line(line) for line in content.split("\n")]
            section["content"] = "\n".join(line for line in cleaned if line.strip())

        for pd in section.get("places_detail", []):
            postprocess_place_detail(pd)


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
                if not primary:
                    eligible = sorted(
                        [n for n in names if _place_photo_priority(n) < 99],
                        key=_place_photo_priority,
                    )
                    if eligible:
                        primary = eligible[0]
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
        # Day마다 사진 1장 이상 → Day 수만큼 API 호출 확보
        effective_limit = max(limit, len(day_attractions))

        def pick(name: str, *, required: bool = False) -> None:
            n = (name or "").strip()
            if not n or n in seen:
                return
            if required or len(picked) < effective_limit:
                seen.add(n)
                picked.append(n)

        for n in day_attractions:
            pick(n, required=True)
        lodging_pool.sort(key=_place_photo_priority)
        for n in lodging_pool:
            pick(n)
        rest_pool.sort(key=_place_photo_priority)
        for n in rest_pool:
            pick(n)
        return picked

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


def extract_map_title(query: str, city: str = None) -> str:
    keywords = ["2박3일", "3박4일", "4박5일", "1박2일", "일정", "숙소", "맛집", "코스"]

    if not city:
        cities = ["오사카", "마쓰야마", "시즈오카", "교토", "도쿄", "후쿠오카", "나라", "고베"]
        for c in cities:
            if c in query:
                city = c
                break

    title = city or ""
    for kw in keywords:
        if kw in query:
            title = f"{title} {kw}".strip()
            break

    return title or query[:15]


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

    itinerary_query = is_itinerary_query(req.query)
    match_count = 15 if itinerary_query else req.match_count

    # 2. 벡터 검색 (유튜브는 RPC 없거나 실패 시 빈 배열 — 후기 검색은 계속)
    res = await asyncio.to_thread(
        lambda: supabase.rpc("match_travel_chunks", {
            "query_embedding": query_vector,
            "match_threshold": req.match_threshold,
            "match_count": match_count,
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
    print(f"\n=== 검색된 청크 {len(chunks)}개 ===", flush=True, file=sys.stderr)

    def resolve_chunk_title(chunk: dict) -> str:
        return (chunk.get("title") or "").strip() or "네이버 블로그 후기"

    def format_chunk_log(i: int, chunk: dict) -> str:
        sim = chunk.get("similarity")
        try:
            sim_str = f"{float(sim):.3f}" if sim is not None else "?"
        except (TypeError, ValueError):
            sim_str = "?"
        title_preview = (chunk.get("title") or "")[:30]
        text_preview = (chunk.get("text") or "")[:60]
        return f"[{i + 1}] similarity={sim_str} | {title_preview} | {text_preview}"

    for i, c in enumerate(chunks):
        print(format_chunk_log(i, c), flush=True, file=sys.stderr)

    context = "\n\n".join([
        f"[id:{i + 1}] [출처: {c.get('link', '')}] [날짜: {c.get('date', '')}] [제목: {resolve_chunk_title(c)}]\n{c.get('text') or ''}"
        for i, c in enumerate(chunks)
    ])

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

    if itinerary_query:
        normalize_itinerary_response(result)
        await extend_result_reviews(result, chunks)
    enrich_place_warnings(result)

    # 6. content·places_detail 장소명 → Places API (일정형: Day 수만큼 최소 확보)
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
        text = chunk.get("text", "") or ""
        return {
            "id": ref_id,
            "title": title,
            "channel": channel,
            "date": chunk.get("date", ""),
            "link": link,
            "text_preview": text[:1200],
            "is_ad": chunk.get("is_ad", False),
        }

    # 7. 본문 [ref:N] ↔ sources 동기화 + 중복 제거 + 모바일 URL
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
            resolve_chunk_title(chunk),
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
            if link and link in seen_links:
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
            chunk_for_ref = (
                chunks[ref_id - 1]
                if isinstance(ref_id, int) and 1 <= ref_id <= len(chunks)
                else None
            )
            if chunk_for_ref:
                source["text_preview"] = (chunk_for_ref.get("text") or "")[:1200]
                source["is_ad"] = bool(chunk_for_ref.get("is_ad"))
                source["title"] = resolve_chunk_title(chunk_for_ref)

            if link and "blog.naver.com" in link and "m.blog.naver.com" not in link:
                source["link"] = link.replace("https://blog.naver.com", "https://m.blog.naver.com")

    # 유튜브 링크 추가
    result["youtube_videos"] = [
        format_youtube_item(v)
        for v in youtube_videos
        if (v.get("url") or "").strip()
    ]

    result["map_title"] = extract_map_title(req.query, req.city)

    if result.get("summary"):
        result["summary"] = INLINE_REF_RE.sub(" ", str(result["summary"])).strip()

    renumber_source_refs(result)

    return result


REF_TAG_RE = re.compile(r"\[ref:(\d+)\]")


def _remap_ref_text(text: str, old_to_new: dict[int, int]) -> str:
    if not text:
        return text

    def repl(m: re.Match) -> str:
        old = int(m.group(1))
        new = old_to_new.get(old)
        return f"[ref:{new}]" if new is not None else ""

    return REF_TAG_RE.sub(repl, text)


def renumber_source_refs(result: dict) -> None:
    sources = result.get("sources") or []
    if not sources:
        return

    sorted_sources = sorted(sources, key=lambda s: int(s.get("id", 0)))
    old_to_new: dict[int, int] = {}
    for i, source in enumerate(sorted_sources):
        try:
            old_id = int(source.get("id"))
        except (TypeError, ValueError):
            continue
        new_id = i + 1
        old_to_new[old_id] = new_id
        source["id"] = new_id

    result["sources"] = sorted_sources

    result["summary"] = _remap_ref_text(result.get("summary") or "", old_to_new)

    for section in result.get("sections", []):
        section["content"] = _remap_ref_text(section.get("content") or "", old_to_new)
        table = section.get("table")
        if table and isinstance(table.get("rows"), list):
            for row in table["rows"]:
                if isinstance(row, list):
                    for j, cell in enumerate(row):
                        row[j] = _remap_ref_text(str(cell or ""), old_to_new)
        for pd in section.get("places_detail", []):
            pd["description"] = _remap_ref_text(pd.get("description") or "", old_to_new)
            pd["warnings"] = [
                w
                for w in (
                    _remap_ref_text(w, old_to_new)
                    for w in (pd.get("warnings") or [])
                    if w
                )
                if w
            ]
            for review in pd.get("reviews", []):
                if not isinstance(review, dict):
                    continue
                review["text"] = _remap_ref_text(review.get("text") or "", old_to_new)
                ref = review.get("ref")
                if ref is not None:
                    try:
                        mapped = old_to_new.get(int(ref))
                    except (TypeError, ValueError):
                        mapped = None
                    if mapped is not None:
                        review["ref"] = mapped
                    else:
                        review.pop("ref", None)
        for review in section.get("reviews", []):
            if not isinstance(review, dict):
                continue
            review["text"] = _remap_ref_text(review.get("text") or "", old_to_new)
            ref = review.get("ref")
            if ref is not None:
                try:
                    mapped = old_to_new.get(int(ref))
                except (TypeError, ValueError):
                    mapped = None
                if mapped is not None:
                    review["ref"] = mapped
                else:
                    review.pop("ref", None)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4)