import os
from pathlib import Path

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

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    city: str = None
    category: str = None
    travel_style: str = None
    match_threshold: float = 0.7
    match_count: int = 20

@app.post("/search")
async def search(req: SearchRequest):
    result = gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=req.query,
        config={"output_dimensionality": 768} 
    )
    query_vector = result.embeddings[0].values

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
        return {"answer": "관련 정보를 찾을 수 없어요.", "sources": []}

    context = "\n\n".join([f"[출처: {c['link']}]\n{c['text']}" for c in chunks])

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"질문: {req.query}\n\n참고 후기:\n{context}\n\n위 후기들을 바탕으로 질문에 친절하고 구체적으로 답변해주세요. 출처 링크도 함께 제공해주세요.",
        config={"thinking_config": {"thinking_budget": 0}}
    )

    answer = response.text
    sources = [{"link": c["link"], "text": c["text"][:100], "similarity": c["similarity"]} for c in chunks]

    return {"answer": answer, "sources": sources}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 