"""LLM evidence extractor for any OpenAI-compatible endpoint (OpenAI, Ollama, LM Studio) with an on-disk cache.

The API key is read by the SDK from OPENAI_API_KEY; it is never written anywhere by this module.
"""
import hashlib
import json
from pathlib import Path

from .evidence import CLAIM_FIELDS, KINDS, LEVELS, Claim, Evidence, valid

PROMPT = """You extract negotiation claims from one message sent by a tenant negotiating radio resources.
Issues: "q" = number of PRBs, "d" = lease duration, "kappa" = credit price.
Return every claim actually stated, nothing inferred:
- weight_order: the sender says issue a matters more than issue b (a, b in q, d, kappa). a is always the MORE
  important issue, also for phrasings like "b is less important than a".
- reservation_level: the sender describes its walk-away point as "low" (flexible, modest terms), "mid" (reasonable deal) or "high" (needs a strong deal).
- kpi_need: the sender forecasts its PRB need next epoch will exceed value.
Unused fields: a, b, level = "" and value = 0. confidence in [0, 1] is how sure you are the claim is stated."""

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["claims"],
    "properties": {"claims": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["kind", "a", "b", "level", "value", "confidence"],
        "properties": {"kind": {"type": "string", "enum": list(KINDS)},
                       "a": {"type": "string", "enum": ["", "q", "d", "kappa"]},
                       "b": {"type": "string", "enum": ["", "q", "d", "kappa"]},
                       "level": {"type": "string", "enum": ["", *LEVELS]},
                       "value": {"type": "number"}, "confidence": {"type": "number"}}}}},
}


class LLMExtractor:
    def __init__(self, model, base_url=None, client=None, cache_dir=".llm_cache"):
        if client is None:
            from openai import OpenAI  # optional dependency
            client = OpenAI(base_url=base_url)
        self.client, self.model, self.base_url = client, model, base_url
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.usage = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _complete(self, text):
        key = hashlib.sha256(json.dumps([self.model, self.base_url, PROMPT, SCHEMA, text]).encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if path.exists():
            self.usage["cache_hits"] += 1
            return json.loads(path.read_text())
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": text}],
            response_format={"type": "json_schema", "json_schema": {"name": "claims", "strict": True, "schema": SCHEMA}})
        rec = {"content": resp.choices[0].message.content, "prompt_tokens": resp.usage.prompt_tokens,
               "completion_tokens": resp.usage.completion_tokens}
        path.write_text(json.dumps(rec))
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += rec["prompt_tokens"]
        self.usage["completion_tokens"] += rec["completion_tokens"]
        return rec

    def __call__(self, msg):
        try:
            items = json.loads(self._complete(msg.text)["content"])["claims"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []  # malformed output yields no evidence rather than guessed evidence
        out = []
        for k, it in enumerate(items):
            try:
                # keep only the fields the claim kind uses; a stray "a" on a KPI claim is noise, not content
                cl = Claim(**{f: float(it[f]) if f == "value" else it[f] for f in CLAIM_FIELDS[it["kind"]]})
                c = min(max(float(it["confidence"]), 0.0), 1.0)
            except (KeyError, TypeError, ValueError):
                continue
            out.append(Evidence(cl, c, float(valid(cl)), msg.sender, msg.cell, msg.t, f"{msg.msg_id}:{k}"))
        return out
