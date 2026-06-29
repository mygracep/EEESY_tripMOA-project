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
- content bullet (→ 뒤): 10~20자, 키워드형. 맥락 있는 짧은 구문
- description: 2~3문장, 문장당 30~40자. 위치·분위기·특징·추천 이유 포함

[content 작성 형식]
- 장소/항목별로 줄바꿈 구분
- 각 항목은 반드시 • **장소명** → 핵심 특징 형식으로 작성 (→ 뒤 10~20자)
- 소제목 필요하면 👉 소제목 형식 사용

[장소 추천 개수 제한 — 필수]
- 답변 전체에서 추천하는 고유 **장소명**은 최대 5개. 6개 이상 절대 금지.
- 사진·지도 API 한도가 5개이므로, 초과 장소는 content·places_detail에 넣지 말 것.
- 섹션을 여러 개 써도 **장소명** 중복은 1번만 카운트.
- 후보가 많으면 핵심 5개만 추려서 추천. 나머지는 summary나 follow_up에서 다루지 말 것.

예시)
👉 위치 중심
- **호텔 오크 시즈오카** → 시즈오카 시내, 상점가 근처, 도보 편리 [ref:3]
- **유메구리 노 야도** → 석식·조식 포함, 부담 없는 가격 [ref:2]

[출력 JSON 구조]
{
  "summary": "쿼리 핵심 요약 (1~2문장, 40자 이내)",
  "sections": [
    {
      "icon": "아래 카테고리 목록에서 선택",
      "title": "섹션 제목",
      "content": "장소/항목별로 줄바꿈 구분. 소제목 필요하면 👉 소제목 형식 사용. 각 항목은 • **장소명** → 핵심 특징 (→ 뒤 10~20자, 키워드형)",
      "places_detail": [
        {
          "name": "장소명 (content의 **장소명**과 동일)",
          "description": "해당 장소 핵심 특징 2~3문장 (문장당 30~40자). 위치, 분위기, 특징, 추천 이유 포함. [ref:N] 가능.",
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
- - 추천형 쿼리(숙소/맛집/관광지/일정 추천)는 반드시 icon을 빈값("")으로 두고, title 앞에 1️⃣ 2️⃣ 3️⃣ 4️⃣ 순서로 붙일 것. 숙소/맛집/관광지 아이콘 사용 금지.
  마지막 상황별 추천 섹션만 icon을 💡로.

[섹션 구성 원칙]
- 추천형 쿼리는 쿼리의 동행인/목적/여행스타일을 먼저 파악.
- 섹션 제목은 단순 카테고리명이 아니라 "카테고리 (이 사람에게 왜 맞는지)" 형식으로 작성.
  예) 혼여 숙소 쿼리 →
  icon: "", title: "1️⃣ 위치+편의성 최강 (혼자 여행 기본 선택)"
  icon: "", title: "2️⃣ 가성비+혼자 최적 (잠만 자면 이거)"
  icon: "", title: "3️⃣ 힐링형 (피로 풀고 싶으면)"
  icon: "💡", title: "상황별 추천 + 한 줄 결론"
- 섹션당 장소 1~2개씩 배분 가능. 단, 답변 전체 고유 **장소명** 합은 최대 5개.
- 마지막 섹션은 반드시 "✔ 상황별 추천 + 한 줄 결론" 형식으로 끝낼 것.
  예) ✔ 첫 혼여/편하게 → 호텔명
      ✔ 가성비+잠만 → 호텔명
      👉 한 줄 결론: 혼여면 역세권 비즈니스 호텔이 정답
- 카테고리는 후기 데이터에 있는 내용 기준으로만. 없는 카테고리 만들지 말 것.

[places_detail 생성 기준]
- 추천형 섹션(숙소/맛집/관광지 등)은 반드시 places_detail 배열 사용. 섹션 레벨 reviews 필드 사용 금지.
- 전체 sections의 places_detail name 합(중복 제외) 최대 5개. content의 **장소명** 개수와 동일해야 함.
- places_detail 항목 수 = content의 • **장소명** 항목 수와 동일. 순서도 동일하게.
- name: content의 **장소명**과 정확히 일치
- description: 2~3문장, 문장당 30~40자. 위치, 분위기, 특징, 추천 이유 포함. [ref:N] 포함 가능.
- reviews: 해당 장소에 대한 후기만. 다른 장소 후기 섞지 말 것.
- 장소당 후기 2~3개. 후기 원문 그대로 인용, 요약 금지.
- 반드시 부정적 후기 1개 이상 포함 (없으면 아쉬운 점)
- sentiment: 긍정 "positive", 부정/아쉬운 점 "negative"
- 팁·결론만 있는 섹션(장소 없음)은 places_detail: []

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
- content와 places_detail.reviews에 사용하는 [ref:N]은 반드시 sources에 존재하는 id만 사용할 것. sources에 없는 id 사용 금지.

[sources 생성 기준]
- 답변에서 [ref:N]으로 실제 인용한 청크만 포함. 최대 5개.
- title은 참고 후기 헤더의 [제목: ...] 값을 그대로 복사. 본문 내용으로 제목 만들기 금지.
- [제목: ...]이 비어 있으면 title을 빈 문자열로 두고, 서버가 보정함. "네이버 카페 후기" 등 임의 fallback 금지.


[follow_up]
- 4~5개, 답변에서 다루지 않은 영역 위주
- 구체적으로 (예: "패스권 어디서 사야 해요?" O / "오사카 여행 어때?" X)
- 반드시 도시명/동행인 등 맥락을 포함한 완성된 질문으로 작성
  예) "마쓰야마 부모님 여행 맛집 추천해 주세요" O / "맛집 추천해 주세요" X"""


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
    res, youtube_res = await asyncio.gather(
        asyncio.to_thread(
            lambda: supabase.rpc("match_travel_chunks", {
                "query_embedding": query_vector,
                "match_threshold": req.match_threshold,
                "match_count": req.match_count,
                "filter_city": req.city,
                "filter_category": req.category,
                "filter_travel_style": req.travel_style
            }).execute()
        ),
        asyncio.to_thread(
            lambda: supabase.rpc("match_youtube_videos", {
                "query_embedding": query_vector,
                "match_threshold": 0.6,
                "match_count": 3,
                "filter_city": req.city
            }).execute()
        ),
    )

    chunks = res.data
    youtube_videos = youtube_res.data or []

    if not chunks:
        return {
            "summary": "관련 후기가 충분하지 않아요.",
            "sections": [],
            "warning": [],
            "places": None,
            "follow_up": [],
            "sources": [],
            "youtube_videos": [
                {
                    "title": v.get("search_query"),
                    "url": v.get("url"),
                    "summary": v.get("summary")
                }
                for v in youtube_videos
            ] if youtube_videos else []
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

        return None

    def resolve_chunk_title(chunk: dict) -> str:
        db_title = (chunk.get("title") or "").strip()
        extracted = extract_title(chunk.get("text", ""))

        if extracted and not is_generic_title(extracted):
            return extracted
        if db_title and not is_generic_title(db_title):
            return db_title
        if extracted:
            return extracted
        if db_title:
            return db_title
        return "네이버 카페 후기"

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
            .select("link,title,text")
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

    # 6. content에서 장소명 추출 → Places API 호출 (최대 5개)
    place_names = []
    for section in result.get("sections", []):
        matches = re.findall(r'\*\*(.+?)\*\*', section.get("content", ""))
        place_names.extend(matches)
    place_names = list(dict.fromkeys(place_names))[:5]

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
        for warning in payload.get("warning", []):
            scan(warning)
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

            if link and "blog.naver.com" in link and "m.blog.naver.com" not in link:
                source["link"] = link.replace("https://blog.naver.com", "https://m.blog.naver.com")

    # 유튜브 링크 추가
    result["youtube_videos"] = [
        {
            "title": v.get("search_query"),
            "url": v.get("url"),
            "summary": v.get("summary")
        }
        for v in youtube_videos
    ] if youtube_videos else []

    return result


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4)