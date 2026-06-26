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

[content 작성 형식]
- 장소/항목별로 줄바꿈 구분
- 각 항목은 반드시 • **장소명** → 핵심 특징 1~2개 형식으로 작성
- 소제목 필요하면 👉 소제목 형식 사용

예시)
👉 위치 중심
• **호텔 오크 시즈오카** → 시즈오카 시내, 상점가 근처, 도보 편리 [ref:3]
• **유메구리 노 야도** → 석식·조식 포함, 부담 없는 가격 [ref:2]

[출력 JSON 구조]
{
  "summary": "쿼리 핵심을 한 문장으로 요약",
  "sections": [
    {
      "icon": "아래 카테고리 목록에서 선택",
      "title": "섹션 제목",
      "content": "장소/항목별로 줄바꿈 구분. 소제목 필요하면 👉 소제목 형식 사용. 각 항목은 • **장소명** → 핵심 특징 1~2개 형식으로 작성.\n예시)\n👉 위치 중심\n• **호텔 오크 시즈오카** → 시즈오카 시내, 상점가 근처, 도보 편리 [ref:3]\n• **유메구리 노 야도** → 석식·조식 포함, 부담 없는 가격 [ref:2]",
      "reviews": [
      {
        "text": "후기 원문에서 가장 생생한 문장 그대로 인용",
        "sentiment": "positive 또는 negative",
        "date": "YY.MM",
        "ref": 1
      }
    ],
      "table": null
    }
  ],
  "warning": [],
  "follow_up": ["후속질문1", "후속질문2"],
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
- 추천형 쿼리(숙소/맛집/관광지 추천)는 icon을 빈값("")으로 두고, title 앞에 1️⃣ 2️⃣ 3️⃣ 순서로 붙일 것.
  마지막 상황별 추천 섹션은 icon을 💡로.


[섹션 구성 원칙]
- 추천형 쿼리는 쿼리의 동행인/목적/여행스타일을 먼저 파악.
- 섹션 제목은 단순 카테고리명이 아니라 "카테고리 (이 사람에게 왜 맞는지)" 형식으로 작성.
 예) 혼여 숙소 쿼리 →
  icon: "", title: "1️⃣ 위치+편의성 최강 (혼자 여행 기본 선택)"
  icon: "", title: "2️⃣ 가성비+혼자 최적 (잠만 자면 이거)"
  icon: "", title: "3️⃣ 힐링형 (피로 풀고 싶으면)"
  icon: "💡", title: "상황별 추천 + 한 줄 결론"
- 섹션당 장소 최대 3~4개. 핵심만 추려낼 것.
- 마지막 섹션은 반드시 "✔ 상황별 추천 + 한 줄 결론" 형식으로 끝낼 것.
  예) ✔ 첫 혼여/편하게 → 호텔명
      ✔ 가성비+잠만 → 호텔명
      👉 한 줄 결론: 혼여면 역세권 비즈니스 호텔이 정답
- 카테고리는 후기 데이터에 있는 내용 기준으로만. 없는 카테고리 만들지 말 것.

[reviews 생성 기준]
- 장소별로 후기 원문에서 생생한 문장 2~3개 직접 인용. 요약 금지.
- 반드시 부정적 후기 1개 이상 포함. 없으면 아쉬운 점 포함.
- text는 후기 원문 말투 그대로. LLM이 재가공 금지.
- sentiment: 긍정이면 "positive", 부정/아쉬운 점이면 "negative"
- 장소가 없는 섹션(팁·조언 등)은 reviews: []


[table 생성 기준]
생성: 비교 대상 2개 이상 + 같은 기준으로 비교 가능 + 유저가 선택해야 하는 상황
null: 순서형 동선 정보 / 감성 설명 / 단일 정보

[warning 생성 기준]
해당하면 배열에 추가, 없으면 []:
- 막차/영업종료/예약마감
- 현금only/현장발권 불가
- 날씨·계절로 헛걸음 가능성
- 사전예약 필수

[출처 인라인 표기]
- 문장 끝에 [ref:N] 표기
- 출처 2개면 [ref:1][ref:2] 연속 표기
- sources의 id와 매핑
- 같은 링크가 중복되면 하나만 표기.

[sources 생성 기준]
- 답변에서 [ref:N]으로 실제 인용한 청크만 포함. 최대 5개.
- title은 반드시 [제목: ...] 에서 가져올 것. 본문 내용 절대 금지.

[follow_up]
- 2~3개, 답변에서 다루지 않은 영역 위주
- 구체적으로 (예: "패스권 어디서 사야 해요?" O / "오사카 여행 어때?" X)"""


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

        photo_url = None
        if place.get("photos"):
            photo_name = place["photos"][0]["name"]
            photo_url = (
                f"https://places.googleapis.com/v1/{photo_name}/media"
                f"?maxWidthPx=800&key={GOOGLE_PLACES_API_KEY}"
            )

        return {
            "lat": lat,
            "lng": lng,
            "photo_url": photo_url
        }


@app.post("/search")
async def search(req: SearchRequest):
    # 1. 쿼리 임베딩
    result = await gemini_client.aio.models.embed_content(
        model="gemini-embedding-2",
        contents=req.query,
        config={"output_dimensionality": 768}
    )
    query_vector = result.embeddings[0].values

    # 2. 벡터 검색
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

    if not chunks:
        return {
            "summary": "관련 후기가 충분하지 않아요.",
            "sections": [],
            "warning": [],
            "places": None,
            "follow_up": [],
            "sources": []
        }

    # 3. 컨텍스트 구성
    print(f"\n=== 검색된 청크 {len(chunks)}개 ===")
    for i, c in enumerate(chunks):
        print(f"[{i+1}] similarity={c.get('similarity', '?'):.3f} | {c.get('title','')[:30]} | {c['text'][:60]}")
    
    context = "\n\n".join([
        f"[id:{i+1}] [출처: {c['link']}] [날짜: {c.get('date', '')}] [제목: {c.get('title', '')}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

    # 4. Gemini 답변 생성
    response = await gemini_client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{SYSTEM_PROMPT}\n\n질문: {req.query}\n\n참고 후기:\n{context}",
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

    # 6. content에서 장소명 추출 → Places API 호출
    place_names = []
    for section in result.get("sections", []):
        matches = re.findall(r'\*\*(.+?)\*\*', section.get("content", ""))
        place_names.extend(matches)
    place_names = list(dict.fromkeys(place_names))

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
                    "photo_url": details["photo_url"],
                    "description": ""
                })

    result["places"] = places if places else None

    # 7. 중복 소스 제거
    if result.get("sources"):
        seen_links = set()
        unique_sources = []
        for source in result["sources"]:
            if source["link"] not in seen_links:
                seen_links.add(source["link"])
                unique_sources.append(source)
        result["sources"] = unique_sources

        # 블로그 모바일 URL 치환
        for source in result["sources"]:
            if "blog.naver.com" in source["link"] and "m.blog.naver.com" not in source["link"]:
                source["link"] = source["link"].replace("https://blog.naver.com", "https://m.blog.naver.com")

    return result

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4)

