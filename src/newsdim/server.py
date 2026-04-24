from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from newsdim import Tagger

app = FastAPI(title="newsdim", version="0.9.1")

_tagger = Tagger()


class ScoreRequest(BaseModel):
    text: str


class ScoreResponse(BaseModel):
    scores: dict[str, int]


class BatchRequest(BaseModel):
    texts: list[str]


class BatchResponse(BaseModel):
    results: list[dict[str, int]]


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    s = _tagger.score(req.text)
    return ScoreResponse(scores=s.to_dict())


@app.post("/score/batch", response_model=BatchResponse)
def score_batch(req: BatchRequest):
    results = _tagger.score_batch(req.texts)
    return BatchResponse(results=[s.to_dict() for s in results])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("newsdim.server:app", host="127.0.0.1", port=8427)
