"""Benchmark the LLM extractor on generated messages: extraction micro-F1 against stated claims, tokens and cost."""
import argparse

import numpy as np

from .contracts import sample_preferences
from .evidence import MockExtractor, compose, f1, match_counts
from .llm import LLMExtractor

PROFILES = ("honest", "noise", "contradictory", "strategic_lie")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt-4.1-nano")
    p.add_argument("--base-url", default=None)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--price-in", type=float, help="USD per 1M input tokens")
    p.add_argument("--price-out", type=float, help="USD per 1M output tokens")
    a = p.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    prefs = sample_preferences(rng, 3, 1, "severe")
    msgs = [compose(rng, PROFILES[i % 4], prefs[i % 3, 0], "RD"[i % 2], float(rng.uniform(20, 60)), i % 3, 0, i,
                    f"bench{i}", PROFILES[i % 4] == "strategic_lie") for i in range(a.n)]
    extractors = {"llm": LLMExtractor(a.model, a.base_url), "mock": MockExtractor(np.random.default_rng(a.seed))}
    counts = {(name, prof): np.zeros(3, int) for name in extractors for prof in (*PROFILES, "all")}
    for i, m in enumerate(msgs):
        for name, ext in extractors.items():
            c = np.array(match_counts(m.claims, ext(m)))
            counts[name, PROFILES[i % 4]] += c
            counts[name, "all"] += c

    for name in extractors:
        print(f"{name:5s} micro-F1 " + "  ".join(f"{prof}={f1(*counts[name, prof]):.3f}" for prof in (*PROFILES, "all")))
    u = extractors["llm"].usage
    print(f"model={a.model} calls={u['calls']} cache_hits={u['cache_hits']} "
          f"prompt_tokens={u['prompt_tokens']} completion_tokens={u['completion_tokens']}")
    if a.price_in is not None and a.price_out is not None:
        print(f"cost_usd={(u['prompt_tokens'] * a.price_in + u['completion_tokens'] * a.price_out) / 1e6:.5f}")


if __name__ == "__main__":
    main()
