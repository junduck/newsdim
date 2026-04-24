from __future__ import annotations

import json
import asyncio

from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI, AsyncOpenAI

DIMS = ("mom", "stab", "horz", "eng", "hype", "sent", "sec", "pol")

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    p = PROMPTS_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


@dataclass
class ScorerConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    prompt_name: str = "news_scorer_zh_CN_v2.txt"
    temperature: float = 0.0
    max_retries: int = 3
    thinking: bool = False

    def __post_init__(self):
        if not self.api_key:
            import os

            from dotenv import load_dotenv

            for env_file in (".env.local", ".env"):
                if Path(env_file).exists():
                    load_dotenv(env_file)
                    break
            self.api_key = os.environ.get("LLM_API_KEY", "")
            self.base_url = self.base_url or os.environ.get("LLM_BASE_URL", "")
            self.model = self.model or os.environ.get("LLM_MODEL", "")

        if not self.api_key:
            raise ValueError("LLM_API_KEY not set (env or config)")
        if not self.base_url:
            raise ValueError("LLM_BASE_URL not set (env or config)")
        if not self.model:
            raise ValueError("LLM_MODEL not set (env or config)")


@dataclass
class DimScores:
    mom: int = 0
    stab: int = 0
    horz: int = 0
    eng: int = 0
    hype: int = 0
    sent: int = 0
    sec: int = 0
    pol: int = 0

    def to_dict(self) -> dict[str, int]:
        return {d: getattr(self, d) for d in DIMS}

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> DimScores:
        return cls(**{k: int(d[k]) for k in DIMS if k in d})


class NewsScorer:
    def __init__(self, config: ScorerConfig | None = None):
        self.config = config or ScorerConfig()
        self._prompt: str | None = None
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            max_retries=self.config.max_retries,
        )

    @property
    def prompt(self) -> str:
        if self._prompt is None:
            self._prompt = _load_prompt(self.config.prompt_name)
        return self._prompt

    @property
    def prompt_version(self) -> str:
        return self.config.prompt_name.replace(".txt", "")

    def _api_kwargs(self) -> dict:
        kw: dict = {"model": self.config.model}
        if self.config.thinking:
            kw["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            kw["temperature"] = self.config.temperature
            kw["extra_body"] = {"thinking": {"type": "disabled"}}
        return kw

    def score(self, text: str) -> DimScores:
        resp = self._client.chat.completions.create(
            **self._api_kwargs(),
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": text},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        scores = DimScores.from_dict(parsed)
        return scores

    def score_batch(self, texts: list[str], concurrency: int = 5) -> list[DimScores]:
        async_client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            max_retries=self.config.max_retries,
        )
        api_kw = self._api_kwargs()
        semaphore = asyncio.Semaphore(concurrency)

        async def _score_one(t: str) -> DimScores:
            async with semaphore:
                resp = await async_client.chat.completions.create(
                    **api_kw,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": t},
                    ],
                )
                raw = resp.choices[0].message.content.strip()
                raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                return DimScores.from_dict(json.loads(raw))

        async def _run():
            try:
                return await asyncio.gather(*[_score_one(t) for t in texts])
            finally:
                await async_client.close()

        results = asyncio.run(_run())
        return list(results)
