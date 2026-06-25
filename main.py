import os
import json
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
- places는 핵심 장소만 최대 5개만 추출

[말투]
- 해요체 사용 (~이에요, ~해요, ~있어요)
- "~합니다" 같은 딱딱한 문체 금지
- 한 문장에 정보 하나. 짧고 명확하게.

[출력 JSON 구조]
{
  "summary": "쿼리 핵심을 한 문장으로 요약",
  "sections": [
    {
      "icon": "아래 카테고리 목록에서 선택",
      "title": "섹션 제목",
      "content": "내용. 출처 있으면 [ref:N] 표기.",
      "table": null
    }
  ],
  "warning": [],
  "places": null,
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

[table 생성 기준]
생성: 비교 대상 2개 이상 + 같은 기준으로 비교 가능 + 유저가 선택해야 하는 상황
null: 순서형 동선 정보 / 감성 설명 / 단일 정보

[warning 생성 기준]
해당하면 배열에 추가, 없으면 []:
- 막차/영업종료/예약마감
- 현금only/현장발권 불가
- 날씨·계절로 헛걸음 가능성
- 사전예약 필수

[places 출력 형식]
장소명만 뽑으세요. lat/lng/photo_url은 빈값으로 두세요.
생성: 숙소/맛집/관광지/역/온천/쇼핑
null: 항공편/패스권/비용/날씨/준비물
일정 쿼리면 day에 숫자, 일반이면 day: null
장소 없는 쿼리면 places: null
{
  "day": null,
  "name": "장소명",
  "lat": null,
  "lng": null,
  "photo_url": null,
  "description": "한 줄 설명"
}

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
    match_threshold: float = 0.7
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
    result = gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=req.query,
        config={"output_dimensionality": 768}
    )
    query_vector = result.embeddings[0].values

    # 2. 벡터 검색
    res = supabase.rpc("match_travel_chunks", {
        "query_embedding": query_vector,
        "match_threshold": req.match_threshold,
        "match_count": req.match_count,
        "filter_city": req.city,
        "filter_category": req.category,
        "filter_travel_style": req.travel_style
    }).execute()

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
    response = gemini_client.models.generate_content(
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

    # 6. Places API로 좌표/사진 채우기
    if result.get("places"):
        tasks = [get_place_details(place["name"], req.city) for place in result["places"]]
        details_list = await asyncio.gather(*tasks)
        print(f"details_list: {details_list}")

        for place, details in zip(result["places"], details_list):
            if details:
                place["lat"] = details["lat"]
                place["lng"] = details["lng"]
                place["photo_url"] = details["photo_url"]

    if result.get("sources"):
        seen_links = set()
        unique_sources = []
        for source in result["sources"]:
            if source["link"] not in seen_links:
                seen_links.add(source["link"])
                unique_sources.append(source)
        result["sources"] = unique_sources

    return result


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

