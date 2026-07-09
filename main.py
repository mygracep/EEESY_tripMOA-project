import os
import json
import re
import time
from pathlib import Path

import httpx
import asyncio
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, Response
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
BACKEND_BASE_URL = "https://eeesytripmoa-project-production.up.railway.app"
PLACE_PHOTOS_ENABLED = False
PLACES_API_ENABLED = True
FETCH_PLACE_REVIEWS_ENABLED = False  # 임시 — 새 RPC로도 충분한지 테스트
MAX_TOTAL_CHUNKS = 20  # LLM 컨텍스트 상한 (30→20, 토큰·지연 절감)
CONTEXT_CHUNK_MAX_CHARS = 800  # 청크 본문 truncate (Gemini 입력 축소)
QUALITY_FLOOR = 0.4
ITINERARY_MIN_GUARANTEE = {
    "일정/동선": 4,
    "음식/맛집": 3,
    "관광/체험": 3,
    "숙소": 2,
    "교통/이동": 2,
}

CITY_ALIASES = {
    "마쓰야마": ["마쓰야마", "마츠야마", "松山", "도고온천", "시코쿠"],
    "오사카": ["오사카", "大阪", "간사이", "난바", "우메다", "도톤보리"],
    "시즈오카": ["시즈오카", "静岡", "후지산", "아타미", "이즈"],
}


def is_city_relevant(chunk: dict, city: str | None) -> bool:
    if not city:
        return True
    aliases = CITY_ALIASES.get(city, [city])
    text = f"{chunk.get('title','')} {chunk.get('text','')}"
    return any(a in text for a in aliases)

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
- 장소 2개 이상이면 절대 한 블록에 합치지 말 것. 각 장소는 반드시 별도의 이모지+**장소명** 줄로 시작.
- description/highlights 안에 다른 장소의 🏨/🍜 등 이모지+**장소명**이 등장하면 안 됨 (분리된 블록이어야 함).


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
          "warnings": ["해당 장소 주의사항만. 없으면 []. 개조식 명사형 종결(~안됨/~없음/~주의), [ref:N] 가능"],
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
- 장소가 3개 이상이면 최소 2개 섹션으로 나눌 것. 모든 장소를 하나의 섹션에 몰아넣지 말 것.
- 나누는 기준은 후기 데이터에서 실제로 드러나는 차이를 기준으로 잡을 것. 
    예시:
  · 위치/접근성 (역세권 vs 외곽, 도보 거리)
  · 가격대 (가성비 vs 프리미엄)
  · 객실 구성 (싱글룸/패밀리룸/대욕장 유무)
  · 분위기 (전통 료칸 vs 모던 호텔)
  · 동행인 적합도 (혼자/친구/가족)
- 뚜렷한 차이가 없으면 가격대(저가/중가)나 위치(역세권/외곽) 기준을 기본으로 사용.
- 섹션당 장소 2~3개씩 추천. 참고 후기에 다양한 정보가 있으면 답변 전체 장소 수를 5개로 제한하지 말고 충분히 추천할 것.
- 장소당 reviews 최소 2개, 가능하면 3개. 참고 후기에 해당 장소 관련 내용이 충분히 있다면 2개 이하로 끝내지 말 것.
- 같은 [ref:N]을 모든 장소에 반복 사용 금지. 참고 후기에 다양한 ref가 있다면 장소마다 다른 출처를 적극 활용할 것.
- 마지막 섹션은 반드시 title "상황별추천" 으로 끝낼 것 (icon: "💡").
  content 예)
  ✔ 첫 혼여/편하게 → **호텔명**
  ✔ 가성비+잠만 → **호텔명**
  👉 한 줄 결론: 혼여면 역세권 비즈니스 호텔이 정답
  - 상황별추천 content에는 [ref:N] 표기 금지 (출처는 places_detail·reviews에만)
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
- **Day(섹션)당 places_detail은 최대 3개.** 언급된 장소가 3개보다 많으면,
  참고 후기에서 해당 장소의 실경험 리뷰가 2개 이상 확보되는 장소를 우선 선정. 1개뿐이어도 places_detail 가능.
- 리뷰가 **아예 없는** 장소만 places_detail을 생략하고 content에 이름만 언급.
- 정말 인기 명소라 후기 자체가 없는 경우가 아니면, 리뷰 부족을 이유로 장소 자체를
  일정에서 빼지는 말 것 — 리뷰만 생략하고 이름은 유지.
- places_detail: 선정된 장소(위 기준)마다 항목 작성. Day당 최대 3개. 숙소 섹션은 places_detail 필수.
  reviews 2개를 우선 목표로 하되, 참고 후기에 관련 리뷰가 1개뿐이면 1개만 넣을 것. warnings negative 기반.
  - **reviews는 2개를 시도하고, 관련 리뷰가 1개뿐이면 1개로 충분함. 절대 3개 이상은 넣지 말 것.**
  - 리뷰가 아예 없는 장소는 앞선 규칙대로 places_detail 자체를 생략(이름만 content 유지).
  - 같은 [ref:N]을 모든 장소에 반복 사용 금지. 장소마다 다른 출처 우선 사용. 참고 후기에 다양한 ref가 있다면 적극 활용할 것.
- 마지막 섹션은 상황별추천이 아니라 여행 팁으로 끝낼 것.
- 마지막 섹션: title "여행 팁" (이모지 없이), icon "💡", places_detail: []
- 여행 팁 content 형식은 상황별추천과 동일:
  - 짧은 본문 줄 나열 (✔ 교통 / ✔ 렌터카 / ✔ 주의사항 등)
  - 여행 팁 content에는 [ref:N] 표기 금지 (출처는 places_detail·reviews에만)
  - **장소명 블록·이모지+장소·사진 형식 사용 금지**
  - 👉 한 줄 결론 1줄 가능
- content 불릿은 "이동해요 / 식사해요" 같은 행동 나열 금지.
  반드시 해당 장소의 특징·추천 이유·팁을 담을 것.
  후기 데이터에 정보 없으면 해당 장소 불릿 생략.
- reviews는 실제 경험 서술 문장만. 
  시간표형("8시 저녁ㅡ 돈요시"), 질문형, 일정 나열형 금지.


[places_detail 생성 기준]
- 추천형·일정형 Day 섹션 모두 places_detail 배열 필수. 섹션 레벨 reviews 필드 사용 금지.
- 일정형: 선정된 장소만 places_detail 작성. Day당 최대 3개. 리뷰 부족 장소는 생략.
- places_detail 대상 선정: 참고 후기에서 해당 장소 리뷰(경험 서술)가 2개 이상 나오는
  장소를 우선. 후기가 1개뿐이거나 없으면 해당 장소는 places_detail 생성 생략
  (content 언급은 유지 가능).
- 추천형: places_detail 항목 수 = content **장소명** 수·순서와 동일.
- name: content의 **장소명**과 정확히 일치
- description: 2~3문장, 문장당 30~40자. 위치, 분위기, 특징, 추천 이유 포함. [ref:N] 포함 가능.
- **추천형:** content와 places_detail 항목 수·순서 동일. 장소마다 🏨 **장소명** 줄을 따로 쓸 것.
- **한 places_detail.description에 숙소 2개 이상 넣지 말 것.** 다른 호텔은 별도 places_detail 항목 + content 🏨 줄.
- reviews: 해당 장소 **직접 방문·체험 후기**만. 다른 장소·다른 메뉴 후기 섞지 말 것.
  ✗ 예: "아키요시 타이메시 본점" places_detail에 "말차 모찌" 후기 (다른 가게/디저트) 넣기 금지
  ✓ 후기는 장소명·대표 메뉴(도미밥/타이메시 등)와 직접 관련된 경험만
- **reviews 제외:** 질문·문의(?/궁금/할까요), 일정·동선 나열(/, ->), 의견·제안만(~포기하면, ~넣고 싶)
- **reviews 포함:** 방문 소감, 맛·분위기·동선 팁, 일정·동선 조언, 추천/비추, 아쉬운 **경험**
- **추천형**: 장소당 실후기 최소 2개, 가능하면 3개. 참고 후기에 충분히 있다면 2개 이하로 끝내지 말 것.
- **일정형**: 장소당 실후기 2개를 우선 시도하되, 관련 리뷰가 1개뿐이면 1개도 허용. 3개 이상은 금지.
- 둘 다 reviews 빈 배열은 금지 — 리뷰가 아예 없으면 그 장소는 places_detail 자체를 생성하지 말 것.
- **reviews.ref 필수** — 각 review마다 ref(숫자)와 text 본문 [ref:N] 중 하나 이상 반드시 포함. ref 없는 review 금지.
- 한 장소의 reviews 안에서도 같은 [ref:N]을 2개 이상 review에 쓰지 말 것. review마다 반드시 다른 ref 사용.
- reviews.text: 후기 작성자의 **경험·감상 문장**만 원문 그대로 인용. 2~4문장·줄바꿈 포함 가능. 요약·한 줄 압축·청크 전체 복사 금지.
- 부정 후기 1개 이상 포함 (질문형·의견형 negative 금지, 실제 아쉬운 **경험**만)
- sentiment: 긍정 "positive", 부정/아쉬운 점 "negative"
- warnings: **negative reviews에서 주의사항을 개조식 명사형으로 추출**하여 반드시 포함. 예약/휴무/막차/현금/입장제한/대기 등이 후기에 있으면 warnings에 1~2개 넣을 것. 비워두지 말 것. root warning 필드 사용 금지.
- 팁·결론만 있는 섹션(여행 팁)은 places_detail: []


[reviews 생성 기준]
- reviews.text 끝에 반드시 [ref:N] 포함. ref 없는 review 생성 금지.
- reviews.text는 후기 작성자가 실제로 쓴 감상/경험 문장만. "제목:", "1. 내가주는 추천점수", "상점명:", "지역:" 같은 글 메타데이터·목록 형식·청크 전체 복사 절대 금지.
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
- 아래 케이스 위주로 warning 생성. **실질적 주의·아쉬운 점**이면 개조식 명사형으로 포함.
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
- 각 항목은 **개조식 명사형 종결**로 작성 (~안됨, ~없음, ~필요, ~주의, ~불가 등). 해요체·완결된 문장 금지.
  예) "타이밍 티켓 없으면 입장 안됨", "현금만 가능, 카드 안됨", "저녁 6시 이후 상점가 문 닫음"
- **의미가 끝나지 않은 상태로 절대 자르지 말 것.** 아래는 금지되는 형태:
  ✗ "가이드 입장료, 유료 관광지" (무슨 문제인지 불명확한 채 끊김)
  ✗ "오전에 후지산이 잘 보이는" (동사 끝맺음 없이 끊김)
  ✓ "가이드 입장료·유료 관광지 입장료 별도 부담", "오전 방문 시 후지산이 더 잘 보임"
- 글자수 제한 없음. 위 예시(12~15자)는 참고용 길이일 뿐이며, 내용이 다 안 들어가면 20자 이상으로 늘려서 완결된 형태로 작성할 것. 짧게 맞추려고 문장을 끊는 것보다 길더라도 의미가 완전한 것이 우선.
- 해당 negative review의 ref를 [ref:N]으로 표기
- 예) negative "닌텐도 월드는 타이밍 티켓 없으면 못 들어가요" → warnings: ["타이밍 티켓 없으면 입장 안됨 [ref:1]"]
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
TITLE_PREFIX_RE = re.compile(r"^제목\s*[:：]\s*.+?(?=\n\n|\Z)", re.DOTALL)
NUMBERED_FIELD_RE = re.compile(
    r"\d\.\s*(?:내가주는\s*추천점수|상점명|지역|상점위치|분위기)\s*[:：]"
)
QNA_SPLIT_RE = re.compile(r"질문\s*[:：].*?댓글\s*[:：]\s*", re.S)


def clean_qna_text(text: str) -> str:
    """qna 청크 전용: 질문 블록 제거 + 남은 질문성 줄 제거."""
    t = text or ""
    m = QNA_SPLIT_RE.search(t)
    if m:
        t = t[m.end():].strip()
    lines = [l for l in t.split("\n") if not QUESTION_RE.search(l)]
    return "\n".join(lines).strip()

SKIP_MATCH_TOKENS = {
    "본점", "지점", "점", "마쓰야마", "오사카", "교토", "도쿄", "후쿠오카", "나고야",
    "삿포로", "오키나와", "일본", "여행", "식당", "카페", "레스토랑", "호텔", "숙소",
    "도고", "온센", "온천",
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


def trim_review_to_place_sentences(text: str, place_name: str, description: str = "") -> str:
    """여러 장소가 섞인 리뷰에서, 이 장소와 직접 관련된 문장만 남김."""
    sentences = re.split(r"(?<=[.!?요])\s*\n?", text)
    terms = extract_place_match_terms(place_name, description)
    if not terms:
        return ""
    relevant = [s for s in sentences if any(t in s for t in terms)]
    return " ".join(relevant).strip() if relevant else ""


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


TEMPLATE_SCORE_RE = re.compile(r"추천점수\s*[:：]\s*([1-5])")


def parse_template_score(text: str) -> int | None:
    """맛집/숙박 정형 템플릿의 '추천점수' 필드 파싱 (없으면 None)."""
    m = TEMPLATE_SCORE_RE.search(text or "")
    return int(m.group(1)) if m else None


def build_place_candidates(
    chunks: list,
    max_places: int = 8,
    reviews_per_place: int = 3,
) -> list[dict]:
    """
    place_name 컬럼 기준으로 chunks를 장소별 그룹핑.
    비광고 우선 → 관련성 검증(is_review_relevant_to_place) →
    리뷰 개수·다양성 기준 랭킹까지 코드가 확정한다.
    (참고용 로그 목적으로만 사용 — 실제 파이프라인에는 연결하지 않음)
    """
    place_groups: dict[str, list[dict]] = {}

    for i, chunk in enumerate(chunks):
        raw_names = (chunk.get("place_name") or "").strip()
        text = (chunk.get("text") or "").strip()
        if not raw_names or not text:
            continue
        for name in raw_names.split(","):
            name = name.strip()
            if not name:
                continue
            place_groups.setdefault(name, []).append({**chunk, "_ref_id": i + 1})

    candidates = []
    for name, group_chunks in place_groups.items():
        non_ad = [c for c in group_chunks if not c.get("is_ad")]
        ad = [c for c in group_chunks if c.get("is_ad")]
        ordered = non_ad + ad

        relevant = [
            c for c in ordered
            if is_review_relevant_to_place(c.get("text") or "", name)
        ]
        if not relevant:
            continue

        article_ids = {c.get("article_id") for c in relevant if c.get("article_id") is not None}
        scores = [s for c in relevant if (s := parse_template_score(c.get("text") or "")) is not None]

        candidates.append({
            "name": name,
            "review_count": len(relevant),
            "diverse": len(article_ids) >= 2,
            "avg_template_score": (sum(scores) / len(scores)) if scores else None,
            "chunk_ref_ids": [c["_ref_id"] for c in relevant[:reviews_per_place]],
        })

    candidates.sort(key=lambda c: (c["review_count"], c["diverse"]), reverse=True)
    return candidates[:max_places]


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


def clean_review_text(text: str, max_len: int = 500) -> str:
    """청크 전체가 review로 들어온 경우 정리."""
    t = (text or "").strip()
    if not t:
        return t

    t = clean_qna_text(t)
    if NUMBERED_FIELD_RE.search(t):
        return ""

    if t.startswith("제목"):
        return ""

    if len(t) > max_len:
        return ""

    return t


# ============================================================
# 정렬/필터링 설계 (오늘 확정분)
#
# 1) match_travel_chunks RPC: 유사도로 후보 100개 추린 뒤,
#    is_ad(게이트) → place_name 집중도 → quality_score →
#    category 일치 → travel_style 일치 → date 순으로 재정렬.
#    city만 하드 필터, category/travel_style은 소프트 필터.
#
# 2) is_ad 게이트(prioritize_non_ad)는 "비교/평가형 질문"일 때만 True.
#    카테고리 무관 — 실측 결과 카테고리별 광고비율 차이가 크지 않아
#    카테고리 기반 게이트는 근거 부족으로 기각.
#
# 3) 생성 단계(pick_place_reviews 등)에도 같은 게이트 값을 전달해서
#    retrieval에서 걸러도 LLM이 다시 광고를 골라 쓰는 걸 방지.
#
# 4) fetch_place_reviews()는 벡터검색이 아니라 장소명 텍스트 매칭이라
#    match_travel_chunks와 별도로 match_place_reviews RPC를 새로 만듦.
#    (같은 정렬 철학 공유, 조회 방식은 별도)
#
# 5) 일정형은 카테고리별 쿼터로 여러 번 RPC 호출 (build_retrieval_plan).
#    일정형 호출 시 hard_category=True로 category 하드 필터.
# ============================================================

EVALUATIVE_QUERY_RE = re.compile(
    r"나은가요|나을까요|나아요|어디가\s*좋|뭐가\s*좋|비교|vs|어느\s*쪽|중\s*(?:어디|뭐)|둘\s*중",
    re.I,
)


def should_prioritize_non_ad(query: str) -> bool:
    """비교/평가형 질문일 때만 광고 배제를 최우선으로 둔다. 카테고리 무관(실측 근거)."""
    return bool(EVALUATIVE_QUERY_RE.search(query or ""))


ITINERARY_QUOTAS = {
    "일정/동선": 8,
    "음식/맛집": 6,
    "관광/체험": 6,
    "숙소": 4,
    "교통/이동": 3,
}


def build_retrieval_plan(
    req: "SearchRequest", itinerary_query: bool, detail_query: bool
) -> list[dict]:
    """match_travel_chunks RPC를 몇 번, 어떤 category·count로 부를지 결정.
    (정렬 기준 자체는 RPC 안에서 항상 고정 — 여기는 '구성'만 결정)
    """
    if itinerary_query:
        return [
            {"filter_category": cat, "match_count": count}
            for cat, count in ITINERARY_QUOTAS.items()
        ]

    if detail_query:
        return [{"filter_category": req.category, "match_count": 5}]

    # 추천형(기본): 단일 카테고리, 넉넉히
    return [{"filter_category": req.category, "match_count": 20}]


async def fetch_place_reviews(
    place_name: str,
    city: str | None,
    prioritize_non_ad: bool,
    limit: int = 3,
) -> list[dict]:
    """장소명 텍스트 매칭 기반 보강 조회. match_place_reviews RPC 사용
    (is_ad 게이트 → 집중도 → quality_score → date 순 정렬은 RPC 안에서 처리)."""
    terms = extract_place_match_terms(place_name)
    if not terms:
        return []
    core_term = max(terms, key=len)
    res = await asyncio.to_thread(
        lambda: supabase.rpc("match_place_reviews", {
            "search_term": core_term,
            "filter_city": city,
            "prioritize_non_ad": prioritize_non_ad,
            "match_count": limit,
        }).execute()
    )
    return res.data or []


def pick_place_reviews(
    reviews: list,
    min_count: int = 2,
    max_count: int = 3,
    place_name: str = "",
    description: str = "",
    exclude_ad: bool = False,
) -> list:
    strict: list = []
    relaxed_pool: list = []
    raw_pool: list = []
    seen_texts: set[str] = set()
    has_place_context = bool((place_name or "").strip())

    def is_relevant(r: dict) -> bool:
        if not has_place_context:
            return True
        return is_review_relevant_to_place(r.get("text") or "", place_name, description)

    for r in reviews or []:
        if not isinstance(r, dict):
            continue
        if exclude_ad and r.get("is_ad"):
            continue
        text = (r.get("text") or "").strip()
        if not text or not is_relevant(r):
            continue
        if text in seen_texts:
            continue
        ref_id = _review_ref_id(r)
        if ref_id is None:
            continue
        r["ref"] = ref_id
        raw_pool.append(r)
        seen_texts.add(text)
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


def backfill_reviews_from_chunks(
    pd: dict, chunks: list, min_count: int = 2, max_count: int = 3
) -> None:
    """places_detail.reviews가 min_count 미달이면 원본 chunks에서 관련 텍스트를 추가로 찾아 보강."""
    reviews = pd.get("reviews", [])
    if len(reviews) >= min_count:
        return

    place_name = pd.get("name") or ""
    description = pd.get("description") or ""
    existing_texts = {(r.get("text") or "").strip() for r in reviews}
    existing_refs = {r.get("ref") for r in reviews if r.get("ref") is not None}

    for i, chunk in enumerate(chunks):
        if len(reviews) >= max_count:
            break
        ref_id = i + 1
        if ref_id in existing_refs:
            continue
        text = (chunk.get("text") or "").strip()
        text = clean_review_text(text)
        if not text or text in existing_texts:
            continue
        if not is_relaxed_review_text(text):
            continue

        if not is_review_relevant_to_place(text, place_name, description):
            continue

        reviews.append({
            "text": text,
            "sentiment": "positive",
            "date": chunk.get("date", ""),
            "ref": ref_id,
        })
        existing_texts.add(text)
        existing_refs.add(ref_id)

    pd["reviews"] = reviews


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


ENDING_CONVERSIONS = [
    (re.compile(r"안\s*돼요\.?$"), "안됨"),
    (re.compile(r"없어요\.?$"), "없음"),
    (re.compile(r"있어요\.?$"), "있음"),
    (re.compile(r"(?:해야|하셔야)\s*해요\.?$"), "필요"),
    (re.compile(r"필요해요\.?$"), "필요"),
    (re.compile(r"(?:주의|조심)하세요\.?$"), "주의"),
]


def _to_terse_ending(text: str) -> str:
    for pattern, label in ENDING_CONVERSIONS:
        if pattern.search(text):
            return pattern.sub(label, text)
    return text


def _warning_clause_from_review(text: str, max_len: int = 15) -> str:
    clause = re.sub(r"\[ref:\d+\]", "", text.replace("**", "")).split("\n")[0]
    clause = re.split(r"[.。!?]", clause)[0].strip()
    if len(clause) < 4:
        return ""
    if QUESTION_RE.search(clause) or SCHEDULE_FEELING_RE.search(clause):
        return ""
    return _to_terse_ending(clause)


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

    if QUESTION_RE.search(body) or SCHEDULE_FEELING_RE.search(body):
        return f"주의사항 확인{suffix}" if suffix else ""

    return f"{body}{suffix}" if body else suffix.strip()


VALID_WARNING_ENDING_RE = re.compile(
    r"(?:안됨|없음|있음|필요|주의|불가|확인|방문|가능|휴무|마감|매진|제한|혼잡|대기"
    r"|저렴함|비쌈|늦음|빠름|필수|한정|추천|비추|아쉬움|불편|편리|중요|권장|참고|발생)"
    r"[.!]?$",
)


def is_warning_complete(text: str) -> bool:
    """의미가 끝난 상태로 종결됐는지 확인 — 중간에 끊긴 워닝(예: '조기', '관광지') 걸러냄."""
    t = INLINE_REF_RE.sub("", text or "").strip()
    if not t:
        return False
    return bool(VALID_WARNING_ENDING_RE.search(t))


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
TITLE_HEADER_RE = re.compile(r"^제목\s*[:：]")

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
        # 다음 청크가 새 글의 제목으로 시작하면 별개 맥락 → 이어붙이지 않고 중단
        if TITLE_HEADER_RE.match(next_text.strip()):
            break
        out = out + clean_review_text(next_text, max_len=1000)

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
    pending: list[tuple[dict, str, dict]] = []
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
                pending.append((review, text, chunks[ref_id - 1]))

    if not pending:
        return

    extended_texts = await asyncio.gather(
        *[extend_truncated_review(text, chunk) for _, text, chunk in pending]
    )
    for (review, text, _), extended in zip(pending, extended_texts):
        if extended != text:
            review["text"] = extended


def validate_warning_ref(
    warning: str, place_name: str, description: str, chunks: list | None
) -> bool:
    """warning의 ref가 실제로 해당 장소와 관련된 청크인지 확인."""
    if not chunks:
        return True
    m = re.search(r"\[ref:(\d+)\]", warning)
    if not m:
        return True
    ref_id = int(m.group(1))
    if ref_id < 1 or ref_id > len(chunks):
        return False
    chunk_text = chunks[ref_id - 1].get("text", "")
    return is_review_relevant_to_place(chunk_text, place_name, description)


def postprocess_place_detail(
    pd: dict,
    chunks: list | None = None,
    prioritize_non_ad: bool = False,
    itinerary: bool = False,
) -> None:
    raw_reviews = pd.get("reviews", []) or []
    llm_reviews = []
    for r in raw_reviews:
        if not isinstance(r, dict):
            continue
        text = clean_review_text(r.get("text", ""), max_len=1000)
        if not text:
            continue
        if prioritize_non_ad and r.get("is_ad"):
            continue
        ref_id = _review_ref_id(r)
        if ref_id is None:
            continue
        r["text"] = text
        r["ref"] = ref_id
        llm_reviews.append(r)

    place_name = pd.get("name") or ""
    if itinerary:
        min_reviews = 1
        max_reviews = 2
    else:
        min_reviews = 1 if _is_attraction_name(place_name) else 2
        max_reviews = 3

    if len(llm_reviews) >= min_reviews:
        pd["reviews"] = llm_reviews[:max_reviews]
    else:
        pd["reviews"] = pick_place_reviews(
            llm_reviews,
            min_count=min_reviews,
            max_count=max_reviews,
            place_name=place_name,
            description=pd.get("description") or "",
            exclude_ad=prioritize_non_ad,
        )
        if chunks and len(pd["reviews"]) < min_reviews:
            backfill_reviews_from_chunks(
                pd, chunks, min_count=min_reviews, max_count=max_reviews
            )

    for r in pd["reviews"]:
        trimmed = trim_review_to_place_sentences(
            r["text"], pd.get("name", ""), pd.get("description", "")
        )
        r["text"] = trimmed if trimmed else r["text"]
    pd["reviews"] = [r for r in pd["reviews"] if (r.get("text") or "").strip()]
    raw_warnings = pd.get("warnings") or []
    description = pd.get("description") or ""
    sanitized = [
        sanitize_warning_text(NUMBERED_LINE_RE.sub("", w).strip())
        for w in raw_warnings
        if w
    ]
    pd["warnings"] = [
        w
        for w in sanitized
        if w
        and validate_warning_ref(w, place_name, description, chunks)
        and is_warning_complete(w)
    ]
    if not pd["warnings"]:
        inferred = infer_warnings_from_reviews(pd.get("reviews", []))
        if inferred:
            pd["warnings"] = [
                w for w in inferred if is_warning_complete(w)
            ]


def enrich_place_warnings(
    result: dict,
    chunks: list | None = None,
    prioritize_non_ad: bool = False,
    itinerary: bool = False,
) -> None:
    for section in result.get("sections", []):
        for pd in section.get("places_detail", []):
            postprocess_place_detail(pd, chunks, prioritize_non_ad, itinerary=itinerary)


def _place_detail_rank(pd: dict, chunks: list | None) -> tuple[int, int]:
    """(순위, 리뷰수) — 순위 높을수록 우선. 0리뷰는 호출 전에 걸러짐."""
    reviews = pd.get("reviews", [])
    n = len(reviews)
    if n == 0:
        return (0, 0)

    article_ids = set()
    if chunks:
        for r in reviews:
            ref = _review_ref_id(r)
            if ref and 1 <= ref <= len(chunks):
                aid = chunks[ref - 1].get("article_id")
                if aid is not None:
                    article_ids.add(aid)
    diverse = len(article_ids) >= 2

    if n >= 3 and diverse:
        rank = 4
    elif n >= 2 and diverse:
        rank = 3
    elif n >= 2:
        rank = 2
    else:
        rank = 1
    return (rank, n)


def rank_and_trim_places_detail(
    result: dict, chunks: list | None, max_per_section: int = 3
) -> None:
    for section in result.get("sections", []):
        candidates = [
            pd for pd in section.get("places_detail", [])
            if pd.get("reviews")
        ]
        candidates.sort(key=lambda pd: _place_detail_rank(pd, chunks), reverse=True)
        section["places_detail"] = candidates[:max_per_section]


ITINERARY_KEYWORDS_RE = re.compile(r"일정|코스|동선|루트|여행\s*계획|당일치기|하루\s*코스")
DURATION_RE = re.compile(r"\d+\s*박\s*\d+\s*일|\d+박\d+일|\d+일\s*여행")
TRIP_DURATION_RE = re.compile(r"(\d+)\s*박\s*(\d+)\s*일")
SINGLE_CATEGORY_ASK_RE = re.compile(
    r"(숙소|호텔|료칸|숙박|맛집|식당|카페|관광지)[^.?!\n]{0,30}추천"
)
DETAIL_QUESTION_RE = re.compile(
    r"(영업시간|휴무|예약|가격|얼마|몇\s*시|몇\s*분|주차|현금|카드|가능한가요|되나요|드나요)"
)

ITINERARY_MODE_BLOCK = """
[⚠️ 이번 질문은 일정형입니다 — 아래만 최우선 적용]
- [섹션 구성 원칙], 추천형 1️⃣ 규칙, 마지막 💡 상황별추천 규칙은 적용하지 마세요.
- [일정형 쿼리 처리]와 [places_detail 생성 기준]을 반드시 따르세요.

출력 전 자가검증:
□ Day 섹션 title "DAY1 — 소제목", icon "" (이모지 없음)
□ Day content: 실제 **장소명**만. 카테고리 줄·동일 장소명 중복 금지. 이동 줄에 약 N분/N시간 필수
□ Day에 🏨 숙소 없음 → icon 🏨 + title "숙소 추천" 섹션 별도 (places_detail 필수)
□ Day(섹션)당 places_detail 최대 3개, 리뷰 2개 이상 확보 가능한 장소 우선 선별
□ 리뷰가 아예 없는 장소만 places_detail 생략(이름만 content 유지), 억지로 채우지 않음
□ 각 Day: 선정된 places_detail마다 reviews(2개 우선·1개 허용·최대 2) + warnings
□ 장소마다 서로 다른 [ref:N] 우선 — 같은 ref를 모든 장소에 반복 금지
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
NO_REF_SECTION_TITLE_RE = re.compile(r"여행\s*팁|상황별")


def is_no_ref_section(section: dict) -> bool:
    title = (section.get("title") or "").strip()
    return bool(NO_REF_SECTION_TITLE_RE.search(title))


def strip_refs_from_tip_sections(payload: dict) -> None:
    for section in payload.get("sections", []):
        if not is_no_ref_section(section):
            continue
        content = section.get("content")
        if content:
            section["content"] = INLINE_REF_RE.sub(" ", str(content)).strip()


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
        for review in section.get("reviews", []):
            ref = review.get("ref")
            if ref is not None:
                try:
                    refs.add(int(ref))
                except (TypeError, ValueError):
                    pass
    return refs


PLACE_EMOJI_PREFIX = re.compile(
    r"^[\s]*(?:[\U0001F300-\U0001FAFF\U00002600-\U000027BF]|🗺️|🏨|🍜|⛩️|🚆|🛍️|💰|📍)"
)
PLACE_INLINE_SPLIT_RE = re.compile(
    r"([^\n])(\s*)(🍜|🏨|⛩️|🛍️|🚆|💰)(\s*\*\*)"
)


def split_inline_place_blocks(content: str) -> str:
    """한 줄에 여러 장소(이모지+**장소명**)가 붙어있으면 강제로 분리."""
    if not content:
        return content
    return PLACE_INLINE_SPLIT_RE.sub(r"\1\n\n\3\4", content)


def is_itinerary_query(query: str) -> bool:
    q = query or ""
    if ITINERARY_KEYWORDS_RE.search(q):
        return True
    if SINGLE_CATEGORY_ASK_RE.search(q):
        return False
    return bool(DURATION_RE.search(q))


def extract_trip_day_count(query: str) -> int | None:
    """'2박3일' → 3, '3박4일' → 4. 못 찾으면 None."""
    m = TRIP_DURATION_RE.search(query or "")
    return int(m.group(2)) if m else None


def is_detail_query(query: str) -> bool:
    q = query or ""
    if is_itinerary_query(q):
        return False
    if SINGLE_CATEGORY_ASK_RE.search(q):
        return False
    return bool(DETAIL_QUESTION_RE.search(q)) and bool(QUESTION_RE.search(q))


def build_system_prompt(query: str) -> str:
    if is_itinerary_query(query):
        block = ITINERARY_MODE_BLOCK
        day_count = extract_trip_day_count(query)
        if day_count:
            block += (
                f"\n\n[⚠️ 필수] 이 여행은 총 {day_count}일 일정입니다. "
                f"반드시 DAY1부터 DAY{day_count}까지 {day_count}개의 Day 섹션을 빠짐없이 만드세요. "
                f"참고 후기가 일부 Day에 부족하더라도 생략하지 말고, "
                f"있는 데이터로라도 채워서 반드시 {day_count}개를 유지하세요."
            )
        return f"{SYSTEM_PROMPT}\n\n{block}"
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


ATTRACTION_NAME_RE = re.compile(
    r"관광|신사|사찰|USJ|스튜디오|박물관|공원|타워|성|전망|이나리|유니버설|폭포|해변|계곡|온천|"
    r"마츠바라|동물원|정원|신궁|다리|대교|저택|온센",
    re.I,
)


def _is_attraction_name(name: str) -> bool:
    return bool(ATTRACTION_NAME_RE.search(name or ""))


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


def normalize_itinerary_response(
    result: dict, chunks: list | None = None, prioritize_non_ad: bool = False
) -> None:
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
            postprocess_place_detail(pd, chunks, prioritize_non_ad, itinerary=True)


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

    section_primaries: list[str] = []
    rest_pool: list[str] = []
    seen: set[str] = set()

    for section in result.get("sections", []):
        if re.search(r"여행\s*팁", section.get("title", ""), re.I):
            continue
        names = _section_place_names(section)
        if not names:
            continue
        primary = min(names, key=_place_photo_priority)
        if primary not in seen:
            seen.add(primary)
            section_primaries.append(primary)
        for n in names:
            if n not in seen:
                seen.add(n)
                rest_pool.append(n)

    effective_limit = max(limit, len(section_primaries))
    picked: list[str] = list(section_primaries)
    for n in sorted(rest_pool, key=_place_photo_priority):
        if len(picked) >= effective_limit:
            break
        picked.append(n)

    return picked[:effective_limit]


def select_itinerary_photo_places(result: dict, max_places: int = 3) -> list[str]:
    """Day 순서대로, 최대 max_places(=3)개 Day에서 대표 장소 1곳씩 선정.
    Day 개수가 3보다 많으면 뒤쪽 Day는 사진 없이 넘어감(고정 3곳 정책)."""
    picked: list[str] = []
    seen: set[str] = set()

    day_sections = [
        s for s in result.get("sections", [])
        if _is_day_section_title(s.get("title") or "")
    ]

    for section in day_sections:
        if len(picked) >= max_places:
            break

        places_detail = section.get("places_detail", [])
        candidates = [pd for pd in places_detail if len(pd.get("reviews", [])) >= 2]
        if not candidates:
            candidates = places_detail
        if not candidates:
            continue

        candidates.sort(key=lambda pd: _place_photo_priority(pd.get("name", "")))
        chosen = candidates[0]["name"]
        if chosen not in seen:
            seen.add(chosen)
            picked.append(chosen)

    return picked


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


def _places_search_queries(place_name: str, city: str | None) -> list[str]:
    """Google Places textQuery 후보 — 한국어·일본어 도시명·장소명만 순서대로 시도."""
    queries: list[str] = []
    name = (place_name or "").strip()
    if not name:
        return queries

    if city:
        queries.append(f"{name} {city}")
        for alias in CITY_ALIASES.get(city, []):
            if alias != city:
                queries.append(f"{name} {alias}")
    queries.append(name)

    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped


async def _google_places_text_search(client: httpx.AsyncClient, text_query: str) -> dict | None:
    field_mask = "places.displayName,places.location"
    if PLACE_PHOTOS_ENABLED:
        field_mask += ",places.photos"

    try:
        search_res = await client.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
                "X-Goog-FieldMask": field_mask,
            },
            json={
                "textQuery": text_query,
                "languageCode": "ja",
                "regionCode": "JP",
            },
        )
        if search_res.status_code != 200:
            print(
                f"[Places API] HTTP {search_res.status_code} query={text_query!r} "
                f"body={search_res.text[:200]}",
                flush=True,
                file=sys.stderr,
            )
            return None

        data = search_res.json()
        if data.get("error"):
            print(
                f"[Places API] error query={text_query!r}: {data['error']}",
                flush=True,
                file=sys.stderr,
            )
            return None
        places = data.get("places") or []
        if not places:
            print(
                f"[Places API] '{text_query}' 검색 결과 없음",
                flush=True,
                file=sys.stderr,
            )
            return None
        return places[0]
    except Exception as e:
        print(
            f"[Places API] '{text_query}' 호출 실패: {e}",
            flush=True,
            file=sys.stderr,
        )
        return None


async def get_place_details(place_name: str, city: str = None) -> dict:
    cache_key = f"{place_name}|{city or ''}"

    try:
        cached = await asyncio.to_thread(
            lambda: supabase.table("place_cache")
            .select("lat, lng, photo_urls, photos_checked")
            .eq("place_key", cache_key)
            .limit(1)
            .execute()
        )
    except Exception as e:
        print(f"place_cache 조회 실패: {e}", flush=True, file=sys.stderr)
        cached = None

    if cached and cached.data:
        row = cached.data[0]
        cached_photos = row.get("photo_urls") or []
        already_checked = row.get("photos_checked", False)
        has_coords = row.get("lat") is not None and row.get("lng") is not None
        if not PLACE_PHOTOS_ENABLED and has_coords:
            return {"lat": row["lat"], "lng": row["lng"], "photo_urls": []}
        if already_checked and has_coords and (cached_photos or not PLACE_PHOTOS_ENABLED):
            return {
                "lat": row["lat"],
                "lng": row["lng"],
                "photo_urls": cached_photos if PLACE_PHOTOS_ENABLED else [],
            }

    if not PLACES_API_ENABLED:
        print(f"[Places] API 비활성 → {place_name!r} skip", flush=True, file=sys.stderr)
        return None

    try:
        place = None
        matched_query = None
        async with httpx.AsyncClient(timeout=10.0) as client:
            for text_query in _places_search_queries(place_name, city):
                place = await _google_places_text_search(client, text_query)
                if place:
                    matched_query = text_query
                    break

        if not place:
            print(f"[Places] 미매칭 place={place_name!r} city={city!r}", flush=True, file=sys.stderr)
            return None

        lat = place["location"]["latitude"]
        lng = place["location"]["longitude"]
        display = (place.get("displayName") or {}).get("text", "")

        photo_urls = []
        if PLACE_PHOTOS_ENABLED and place.get("photos"):
            for photo in place["photos"][:2]:
                photo_urls.append(
                    f"{BACKEND_BASE_URL}/photo/{photo['name']}?maxWidthPx=800"
                )

        print(
            f"[Places] 매칭 place={place_name!r} query={matched_query!r} "
            f"display={display!r} photos={len(photo_urls)}",
            flush=True,
            file=sys.stderr,
        )

        result = {"lat": lat, "lng": lng, "photo_urls": photo_urls}
    except Exception as e:
        print(
            f"[Places API] '{place_name}' 호출 실패: {e}",
            flush=True,
            file=sys.stderr,
        )
        return None

    try:
        await asyncio.to_thread(
            lambda: supabase.table("place_cache")
            .upsert({
                "place_key": cache_key,
                **result,
                "photos_checked": PLACE_PHOTOS_ENABLED,
            })
            .execute()
        )
    except Exception as e:
        print(f"place_cache 저장 실패: {e}", flush=True, file=sys.stderr)

    return result


async def refresh_result_places(result: dict, city: str | None) -> None:
    """places[].photo_urls 갱신 — answer_cache hit·stale place_cache 대응."""
    if not PLACE_PHOTOS_ENABLED:
        return
    places = result.get("places")
    if not places:
        return
    names = [p.get("name") for p in places if p.get("name")]
    if not names:
        return
    details_list = await asyncio.gather(
        *[get_place_details(name, city) for name in names]
    )
    details_by_name = {
        name: details
        for name, details in zip(names, details_list)
        if details
    }
    for p in places:
        details = details_by_name.get(p.get("name"))
        if not details:
            continue
        p["lat"] = details["lat"]
        p["lng"] = details["lng"]
        p["photo_urls"] = details["photo_urls"]


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


async def fetch_youtube_for_search(query_vector: list, city: str | None) -> list[dict]:
    try:
        youtube_res = await asyncio.to_thread(
            lambda: supabase.rpc("match_youtube_videos", {
                "query_embedding": query_vector,
                "match_threshold": 0.6,
                "match_count": 3,
                "filter_city": city,
            }).execute()
        )
        videos = youtube_res.data or []
        return await enrich_youtube_titles(videos)
    except Exception as e:
        print(
            f"match_youtube_videos 실패 (후기 검색은 계속): {e}",
            flush=True,
            file=sys.stderr,
        )
        return []


def chunk_context_text(chunk: dict) -> str:
    raw = (
        clean_qna_text(chunk.get("text") or "")
        if chunk.get("content_type") == "qna"
        else (chunk.get("text") or "")
    )
    if len(raw) > CONTEXT_CHUNK_MAX_CHARS:
        return raw[:CONTEXT_CHUNK_MAX_CHARS] + "…"
    return raw


@app.get("/photo/{photo_name:path}")
async def photo_proxy(photo_name: str, maxWidthPx: int = 800):
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"https://places.googleapis.com/v1/{photo_name}/media",
            params={"maxWidthPx": maxWidthPx, "key": GOOGLE_PLACES_API_KEY},
            follow_redirects=True,
        )
    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "image/jpeg"),
    )


@app.post("/search")
async def search(req: SearchRequest):
    # 1. 쿼리 임베딩
    embed_text = f"{req.city} {req.query}" if req.city else req.query
    result = await gemini_client.aio.models.embed_content(
        model="gemini-embedding-2",
        contents=embed_text,
        config={"output_dimensionality": 768}
    )
    query_vector = result.embeddings[0].values
    embed_t0 = time.monotonic()
    print(f"[임베딩] 타입: {type(query_vector)}, 차원: {len(query_vector)}, 앞 5개 값: {query_vector[:5]}", flush=True, file=sys.stderr)

    non_ad_count = 0
    ad_count = 0
    qna_filtered_count = 0
    fallback_used = False

    try:
        cache_res = await asyncio.to_thread(
            lambda: supabase.rpc("match_answer_cache", {
                "query_embedding": query_vector,
                "match_threshold": 0.95,
                "filter_city": req.city,
                "filter_category": req.category,
                "filter_travel_style": req.travel_style,
            }).execute()
        )
        if cache_res.data:
            try:
                await asyncio.to_thread(
                    lambda: supabase.table("search_logs").insert({
                        "query": req.query,
                        "city": req.city,
                        "category": req.category,
                        "travel_style": req.travel_style,
                        "chunk_count": None,
                        "had_result": True,
                        "cache_hit": True,
                    }).execute()
                )
            except Exception as e:
                print(f"search_logs 저장 실패(캐시 히트): {e}", flush=True, file=sys.stderr)
            cached_result = cache_res.data[0]["result"]
            strip_refs_from_tip_sections(cached_result)
            await refresh_result_places(cached_result, req.city)
            return cached_result
    except Exception as e:
        print(f"answer_cache 조회 실패: {e}", flush=True, file=sys.stderr)

    print(f"[timing] 임베딩+캐시조회: {time.monotonic() - embed_t0:.1f}s", flush=True, file=sys.stderr)

    itinerary_query = is_itinerary_query(req.query)
    detail_query = is_detail_query(req.query)
    prioritize_non_ad = should_prioritize_non_ad(req.query)
    match_count = req.match_count
    print(f"[요청확인] city={req.city!r}, category={req.category!r}, travel_style={req.travel_style!r}, match_threshold={req.match_threshold}", flush=True, file=sys.stderr)
    print(f"[인코딩] city={req.city!r}, utf8_bytes={len(req.city.encode('utf-8')) if req.city else 0}", flush=True, file=sys.stderr)
    print(f"[게이트] itinerary={itinerary_query}, detail={detail_query}, prioritize_non_ad={prioritize_non_ad}", flush=True, file=sys.stderr)

    search_t0 = time.monotonic()
    youtube_task = asyncio.create_task(fetch_youtube_for_search(query_vector, req.city))

    # 2. 벡터 검색 — build_retrieval_plan()에 따라 RPC를 1번 또는 카테고리별 여러 번 호출.
    #    정렬(is_ad 게이트 → 집중도 → quality_score → category → style → date)은
    #    RPC 내부에서 항상 고정으로 처리됨.
    async def fetch_plan_call(call: dict) -> list[dict]:
        res = await asyncio.to_thread(
            lambda: supabase.rpc("match_travel_chunks", {
                "query_embedding": query_vector,
                "match_threshold": req.match_threshold,
                "match_count": call["match_count"],
                "filter_city": req.city,
                "filter_category": call["filter_category"],
                "filter_travel_style": req.travel_style,
                "prioritize_non_ad": prioritize_non_ad,
                "hard_category": itinerary_query,
            }).execute()
        )
        call_chunks = res.data or []
        print(
            f"[검색] category={call['filter_category']!r} 요청={call['match_count']} 확보={len(call_chunks)}",
            flush=True, file=sys.stderr,
        )
        return call_chunks

    plan = build_retrieval_plan(req, itinerary_query, detail_query)
    plan_results = await asyncio.gather(*[fetch_plan_call(call) for call in plan])
    chunks: list[dict] = [c for sub in plan_results for c in sub]
    print(f"[timing] RPC 검색: {time.monotonic() - search_t0:.1f}s", flush=True, file=sys.stderr)

    non_ad_count = sum(1 for c in chunks if not c.get("is_ad"))
    ad_count = len(chunks) - non_ad_count

    chunks_before_qna = len(chunks)
    chunks = [c for c in chunks if is_city_relevant(c, req.city)]
    qna_filtered_count = chunks_before_qna - len(chunks)
    print(f"[필터] qna 필터 후: {len(chunks)}개", flush=True, file=sys.stderr)

    if len(chunks) < 5:
        fallback_used = True
        fallback_threshold = max(0.5, req.match_threshold - 0.15)
        print(f"[fallback] threshold {req.match_threshold} → {fallback_threshold}로 재검색", flush=True, file=sys.stderr)
        res_fallback = await asyncio.to_thread(
            lambda: supabase.rpc("match_travel_chunks", {
                "query_embedding": query_vector,
                "match_threshold": fallback_threshold,
                "match_count": match_count,
                "filter_city": req.city,
                "filter_category": req.category,
                "filter_travel_style": req.travel_style,
                "prioritize_non_ad": prioritize_non_ad,
            }).execute()
        )
        fallback_chunks = [
            c for c in (res_fallback.data or [])
            if is_city_relevant(c, req.city)
        ]
        print(f"[fallback] 원본 {len(res_fallback.data or [])}개 → qna필터 후 {len(fallback_chunks)}개", flush=True, file=sys.stderr)
        existing_ids = {c.get("id") for c in chunks}
        for c in fallback_chunks:
            if c.get("id") not in existing_ids:
                chunks.append(c)
                existing_ids.add(c.get("id"))
        print(f"[fallback] 최종 합산: {len(chunks)}개", flush=True, file=sys.stderr)

    if len(chunks) > MAX_TOTAL_CHUNKS:
        if itinerary_query:
            kept: list[dict] = []
            kept_ids: set[int] = set()

            for cat, min_n in ITINERARY_MIN_GUARANTEE.items():
                cat_chunks = [c for c in chunks if c.get("category") == cat]
                for c in cat_chunks[:min_n]:
                    kept.append(c)
                    kept_ids.add(id(c))

            remaining_slots = MAX_TOTAL_CHUNKS - len(kept)
            for c in chunks:
                if remaining_slots <= 0:
                    break
                if id(c) in kept_ids:
                    continue
                if (c.get("quality_score") or 0) <= QUALITY_FLOOR:
                    continue
                kept.append(c)
                kept_ids.add(id(c))
                remaining_slots -= 1

            guaranteed = sum(
                min(len([c for c in chunks if c.get("category") == cat]), n)
                for cat, n in ITINERARY_MIN_GUARANTEE.items()
            )
            print(
                f"[상한] 일정형 {len(chunks)}개 → {len(kept)}개 "
                f"(최소보장 {guaranteed}개 + "
                f"quality>{QUALITY_FLOOR} 채움 {len(kept) - guaranteed}개)",
                flush=True,
                file=sys.stderr,
            )
            chunks = kept
        else:
            chunks.sort(key=lambda c: float(c.get("similarity") or 0), reverse=True)
            print(f"[상한] {len(chunks)}개 → {MAX_TOTAL_CHUNKS}개로 자름", flush=True, file=sys.stderr)
            chunks = chunks[:MAX_TOTAL_CHUNKS]

    place_names_in_chunks = set()
    for c in chunks:
        if c.get("place_name"):
            for p in c["place_name"].split(","):
                place_names_in_chunks.add(p.strip())

    place_names_in_chunks = list(place_names_in_chunks)[:15]

    if FETCH_PLACE_REVIEWS_ENABLED and place_names_in_chunks:
        extra_tasks = [
            fetch_place_reviews(p, req.city, prioritize_non_ad)
            for p in place_names_in_chunks
        ]
        extra_results = await asyncio.gather(*extra_tasks, return_exceptions=True)
        extra_chunks = []
        for r in extra_results:
            if isinstance(r, Exception):
                print(f"fetch_place_reviews 개별 실패: {r}", flush=True, file=sys.stderr)
                continue
            extra_chunks.extend(r)

        existing_links = {c.get("link") for c in chunks}
        for c in extra_chunks:
            if c.get("link") not in existing_links:
                chunks.append(c)
                existing_links.add(c.get("link"))

    youtube_videos = await youtube_task
    print(f"[timing] 검색+유튜브: {time.monotonic() - search_t0:.1f}s", flush=True, file=sys.stderr)

    if not chunks:
        try:
            await asyncio.to_thread(
                lambda: supabase.table("search_logs").insert({
                    "query": req.query,
                    "city": req.city,
                    "category": req.category,
                    "travel_style": req.travel_style,
                    "chunk_count": 0,
                    "had_result": False,
                    "cache_hit": False,
                }).execute()
            )
        except Exception as e:
            print(f"search_logs 저장 실패(결과없음): {e}", flush=True, file=sys.stderr)
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

    context = "\n\n".join([
        f"[id:{i + 1}] [출처: {c.get('link', '')}] [날짜: {c.get('date', '')}] [제목: {resolve_chunk_title(c)}]\n"
        f"{chunk_context_text(c)}"
        for i, c in enumerate(chunks)
    ])

    place_candidates = build_place_candidates(chunks)
    print(f"[코드후보] {json.dumps(place_candidates, ensure_ascii=False, default=str)}", flush=True, file=sys.stderr)

    system_prompt = build_system_prompt(req.query)

    # 4. Gemini 답변 생성
    llm_t0 = time.monotonic()
    response = await gemini_client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{system_prompt}\n\n질문: {req.query}\n\n참고 후기:\n{context}",
        config={
            "thinking_config": {"thinking_budget": 0},
            "response_mime_type": "application/json",
        }
    )
    usage = getattr(response, "usage_metadata", None)
    if usage:
        print(
            f"[토큰] 입력={usage.prompt_token_count}, 출력={usage.candidates_token_count}, "
            f"전체={usage.total_token_count}",
            flush=True, file=sys.stderr,
        )

    print(f"[timing] Gemini LLM: {time.monotonic() - llm_t0:.1f}s", flush=True, file=sys.stderr)

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
    for section in result.get("sections", []):
        pd_names = [pd.get("name") for pd in section.get("places_detail", [])]
        content_places = re.findall(r"\*\*(.+?)\*\*", section.get("content", ""))
        print(
            f"[{section.get('title')}] content장소={content_places} | places_detail={pd_names}",
            flush=True, file=sys.stderr,
        )

    for section in result.get("sections", []):
        content = section.get("content")
        if content:
            section["content"] = split_inline_place_blocks(content)

    if itinerary_query:
        normalize_itinerary_response(result, chunks, prioritize_non_ad)
    enrich_place_warnings(result, chunks, prioritize_non_ad, itinerary=itinerary_query)
    post_t0 = time.monotonic()
    await extend_result_reviews(result, chunks)
    rank_and_trim_places_detail(result, chunks, max_per_section=3)
    print(f"[timing] 후처리: {time.monotonic() - post_t0:.1f}s", flush=True, file=sys.stderr)

    all_places_detail = [
        pd for section in result.get("sections", [])
        for pd in section.get("places_detail", [])
    ]
    places_detail_count = len(all_places_detail)
    avg_reviews_per_place = (
        sum(len(pd.get("reviews", [])) for pd in all_places_detail) / places_detail_count
        if places_detail_count else 0
    )

    # 6. content·places_detail 장소명 → Places API (일정형: Day당 1곳, 최대 3곳)
    place_names = (
        []
        if detail_query
        else (
            select_itinerary_photo_places(result, max_places=3) if itinerary_query
            else collect_place_names_for_api(result, limit=3, itinerary=False)
        )
    )

    places = []
    places_t0 = time.monotonic()
    if PLACES_API_ENABLED and place_names:
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

    if place_names:
        print(f"[timing] Places API: {time.monotonic() - places_t0:.1f}s", flush=True, file=sys.stderr)
    
    print(
        f"[사진] place_names={place_names}, 확보된 places={len(places)}개, "
        f"사진있음={sum(1 for p in places if p.get('photo_urls'))}개",
        flush=True,
        file=sys.stderr,
    )
    result["places"] = places if places else None

    all_cited_refs = collect_cited_ref_ids(result)
    strip_refs_from_tip_sections(result)

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
    sources_by_id: dict[int, dict] = {}

    for source in result.get("sources", []):
        try:
            sid = int(source.get("id"))
        except (TypeError, ValueError):
            continue
        sources_by_id[sid] = source

    for ref_id in all_cited_refs:
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

    def _insert_search_log():
        try:
            supabase.table("search_logs").insert({
                "query": req.query,
                "city": req.city,
                "category": req.category,
                "travel_style": req.travel_style,
                "chunk_count": len(chunks),
                "had_result": bool(chunks),
                "cache_hit": False,
                "non_ad_count": non_ad_count,
                "ad_count": ad_count,
                "qna_filtered_count": qna_filtered_count,
                "fallback_used": fallback_used,
                "places_detail_count": places_detail_count,
                "avg_reviews_per_place": round(avg_reviews_per_place, 2),
            }).execute()
        except Exception as e:
            print(f"search_logs 저장 실패: {e}", flush=True, file=sys.stderr)

    def _insert_answer_cache():
        try:
            supabase.table("answer_cache").insert({
                "query": req.query,
                "query_embedding": query_vector,
                "city": req.city,
                "category": req.category,
                "travel_style": req.travel_style,
                "result": result,
            }).execute()
        except Exception as e:
            print(f"answer_cache 저장 실패: {e}", flush=True, file=sys.stderr)

    print(f"[timing] 총 소요: {time.monotonic() - embed_t0:.1f}s", flush=True, file=sys.stderr)

    asyncio.create_task(asyncio.to_thread(_insert_search_log))
    if result.get("sections"):
        asyncio.create_task(asyncio.to_thread(_insert_answer_cache))

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
                ref_id = _review_ref_id(review)
                if ref_id is not None:
                    mapped = old_to_new.get(ref_id)
                    if mapped is not None:
                        review["ref"] = mapped
                    else:
                        review.pop("ref", None)
                else:
                    review.pop("ref", None)
        for review in section.get("reviews", []):
            if not isinstance(review, dict):
                continue
            review["text"] = _remap_ref_text(review.get("text") or "", old_to_new)
            ref_id = _review_ref_id(review)
            if ref_id is not None:
                mapped = old_to_new.get(ref_id)
                if mapped is not None:
                    review["ref"] = mapped
                else:
                    review.pop("ref", None)
            else:
                review.pop("ref", None)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4)