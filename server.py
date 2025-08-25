#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, BackgroundTasks, Response, status
from pydantic import BaseModel
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
import os, re, json, uuid, math

# ---------- 설정(.env) ----------
load_dotenv()
MONGO_URL = os.getenv("MONGO_URL")
DB_NAME    = os.getenv("DB_NAME", "newsdatas")
COL_NAME   = os.getenv("COL_NAME", "articles")
LLM_KEY    = os.getenv("LLM_API_KEY")
LLM_MODEL  = os.getenv("LLM_MODEL", "gpt-5-mini")

if not MONGO_URL:
    raise SystemExit("환경변수 MONGO_URL 필요")
if not LLM_KEY:
    raise SystemExit("환경변수 LLM_API_KEY 필요")

mongo = MongoClient(MONGO_URL)
db    = mongo[DB_NAME]
col   = db[COL_NAME]
client = OpenAI(api_key=LLM_KEY)

# ---------- 앱/메모리 잡스토어 ----------
app = FastAPI(title="News Review API")
JOBS: dict[str, dict] = {}   # { job_id: {status, progress, result, ...} }

# ---------- 프롬프트 ----------
SYSTEM = (
    "너는 한국어 뉴스 기사 준수 검수원이다. 다음을 검사하라.\n"
    "1) 성별·인종·종교·성적지향·장애 등 차별/혐오 표현\n"
    "2) 특정 인물에 대한 근거 없는 비방/인신공격\n"
    "3) 광고성(협찬/스폰서/구매 유도)\n"
    "JSON만 반환: "
    '{"discrimination":{"flag":bool,"evidence":[string,...]},'
    '"defamation":{"flag":bool,"evidence":[string,...]},'
    '"advertisement":{"flag":bool,"evidence":[string,...]}}'
)
USER_TPL = "제목: {title}\n본문(일부): {text}\n맥락(인용/비판보도)은 감안하라."
PRETTY = {
    "discrimination": "차별적 용어",
    "defamation": "특정인물 비방",
    "advertisement": "광고성",
}

def trunc(s: str | None, n: int) -> str:
    return re.sub(r"\s+", " ", s or "")[:n]

def analyze_with_llm(title: str, body: str) -> dict:
    """OpenAI Chat Completions 호출. 일부 모델은 response_format 미지원일 수 있어 fallback."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TPL.format(title=trunc(title, 300), text=trunc(body, 4700))}
    ]
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            # 일부 모델은 temperature/response_format 제한 → temperature 생략, 아래가 실패하면 재시도
            response_format={"type": "json_object"},
            # timeout 등은 간단화를 위해 생략
        )
    except Exception:
        # response_format 미지원 같은 경우 → 일반 호출로 재시도
        r = client.chat.completions.create(model=LLM_MODEL, messages=messages)

    content = r.choices[0].message.content
    # 사용량 로그(선택)
    try:
        print(f"[OpenAI] req_id={r.id} model={r.model} usage={getattr(r,'usage',None)}")
    except Exception:
        pass

    try:
        data = json.loads(content)
    except Exception:
        # 모델이 JSON 아닌 텍스트를 주는 경우에 대비해 기본값
        data = {}
    for k in ("discrimination","defamation","advertisement"):
        data.setdefault(k, {"flag": False, "evidence": []})
        data[k].setdefault("flag", False)
        data[k].setdefault("evidence", [])
    return data

def fetch_articles_by_date(date_str: str) -> list[dict]:
    return list(col.find(
        {"published_date": date_str},
        {"url":1, "title":1, "text":1, "published_date":1}
    ))

def run_review(job_id: str, date_str: str):
    JOBS[job_id] = {"status": "running", "progress": 0}
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
            raise ValueError("날짜는 YYYY-MM-DD 형식")
        articles = fetch_articles_by_date(date_str)
        total = max(1, len(articles))
        results = []

        for i, a in enumerate(articles, start=1):
            res = analyze_with_llm(a.get("title",""), a.get("text",""))
            # 위반 항목만 추려서 저장
            viols = [PRETTY[k] for k,v in res.items() if k in PRETTY and v.get("flag")]
            if viols:
                results.append({
                    "url": a.get("url",""),
                    "title": a.get("title",""),
                    "published_date": a.get("published_date",""),
                    "violations": viols,
                    "evidence": {k: (v.get("evidence") or [])[:5] for k,v in res.items() if v.get("flag")}
                })
            # 진행률 업데이트
            JOBS[job_id]["progress"] = math.floor(i * 100 / total)

        JOBS[job_id] = {
            "status": "done",
            "progress": 100,
            "count": len(results),
            "result": results,
        }
    except Exception as e:
        JOBS[job_id] = {"status": "failed", "error": str(e)}

# ---------- API ----------

class ReviewReq(BaseModel):
    date: str  # "YYYY-MM-DD"

@app.post("/review", status_code=202)
def start_review(req: ReviewReq, response: Response, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "queued", "progress": 0}
    bg.add_task(run_review, job_id, req.date)  # 백그라운드 실행
    response.headers["Location"] = f"/review/{job_id}"
    return {"job_id": job_id, "status": "queued"}

@app.get("/review/{job_id}")
def get_review(job_id: str):
    return JOBS.get(job_id, {"status": "not_found"})
