"""Benchmark the LLM extractor on generated messages: extraction micro-F1 against stated claims, tokens and cost."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import sample_preferences
from .evidence import MockExtractor, compose, f1, match_counts
from .llm import PROMPT, LLMExtractor

PROFILES = ("honest", "noise", "contradictory", "strategic_lie")


def messages(seed, n):
    rng = np.random.default_rng(seed)
    prefs = sample_preferences(rng, 3, 1, "severe")
    return [compose(rng, PROFILES[i % 4], prefs[i % 3, 0], "RD"[i % 2], float(rng.uniform(20, 60)), i % 3, 0, i,
                    f"bench{i}", PROFILES[i % 4] == "strategic_lie") for i in range(n)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt-4.1-nano")
    p.add_argument("--base-url", default=None)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--price-in", type=float, help="USD per 1M input tokens")
    p.add_argument("--price-out", type=float, help="USD per 1M output tokens")
    p.add_argument("--out", help="also write the results as JSON to this file")
    p.add_argument("--tuning-seed", type=int,
                   help="also score only messages whose text differs from the 20 prompt-tuning messages of this seed")
    a = p.parse_args(argv)

    msgs = messages(a.seed, a.n)
    tuning = {m.text for m in messages(a.tuning_seed, 20)} if a.tuning_seed is not None else set()
    groups = (*PROFILES, "all", *(("not_in_tuning",) if a.tuning_seed is not None else ()))
    extractors = {"llm": LLMExtractor(a.model, a.base_url), "mock": MockExtractor(np.random.default_rng(a.seed))}
    counts = {(name, g): np.zeros(3, int) for name in extractors for g in groups}
    for i, m in enumerate(msgs):
        for name, ext in extractors.items():
            c = np.array(match_counts(m.claims, ext(m)))
            for g in (PROFILES[i % 4], "all", *(("not_in_tuning",) if "not_in_tuning" in groups and m.text not in tuning
                                                else ())):
                counts[name, g] += c

    for name in extractors:
        print(f"{name:5s} micro-F1 " + "  ".join(f"{g}={f1(*counts[name, g]):.3f}" for g in groups))
    overlap = {"in_tuning_set": sum(m.text in tuning for m in msgs), "distinct_texts": len({m.text for m in msgs})}
    if a.tuning_seed is not None:
        print(f"messages whose text is in the tuning set: {overlap['in_tuning_set']} of {a.n}; "
              f"distinct texts: {overlap['distinct_texts']}")
    u = extractors["llm"].usage
    print(f"model={a.model} calls={u['calls']} cache_hits={u['cache_hits']} "
          f"prompt_tokens={u['prompt_tokens']} completion_tokens={u['completion_tokens']}")
    cost = None
    if a.price_in is not None and a.price_out is not None:
        cost = (u['prompt_tokens'] * a.price_in + u['completion_tokens'] * a.price_out) / 1e6
        print(f"cost_usd={cost:.5f}")
    if a.out:
        rec = {"model": a.model, "base_url": a.base_url, "seed": a.seed, "n": a.n, "tuning_seed": a.tuning_seed,
               "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(), "usage": u, "cost_usd": cost,
               "overlap": overlap,
               "results": {name: {g: {"tp_fp_fn": counts[name, g].tolist(), "micro_f1": f1(*counts[name, g])}
                                  for g in groups} for name in extractors}}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
