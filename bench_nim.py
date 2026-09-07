#!/usr/bin/env python3
"""
Serving cost of the Polish token tax, measured on a running NIM.

Sends the same content in Polish and in English to an OpenAI-compatible
endpoint and reports, per language: prompt tokens, time to first token,
end-to-end latency, output tokens per second, and the KV-cache footprint the
prompt occupies. The point is to turn "+36% tokens" from a property of a
tokenizer into a measured property of a deployment.

    python bench_nim.py --url http://localhost:8000/v1 \
        --model nvidia/nvidia-nemotron-nano-9b-v2 --runs 5

Requires only `requests`. Run it against the NIM container, not against a
hosted API, so the numbers belong to a GPU you can name.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import requests

# Same content, two languages. Keep them semantically matched: the comparison
# is "this meaning costs N tokens in Polish and M in English", so a translation
# that drifts in content invalidates the measurement.
PROMPTS = {
    "pl": (
        "Wyjaśnij w trzech akapitach, dlaczego tokenizacja wpływa na koszt "
        "inferencji modeli językowych. Opisz, czym jest tokenizer typu BPE, "
        "dlaczego języki fleksyjne z bogatą morfologią są w nim kosztowniejsze "
        "od angielskiego, i jakie to ma konsekwencje dla okna kontekstowego "
        "oraz przepustowości serwowania modelu."
    ),
    "en": (
        "Explain in three paragraphs why tokenization affects the inference "
        "cost of language models. Describe what a BPE tokenizer is, why "
        "inflected languages with rich morphology are more expensive in one "
        "than English is, and what this means for the context window and for "
        "model serving throughput."
    ),
}


def stream_once(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    """One streamed completion. Returns timings and token counts."""
    started = time.perf_counter()
    first = None
    chunks = 0
    usage = {}

    resp = requests.post(
        f"{url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            break
        event = json.loads(payload)
        if event.get("usage"):
            usage = event["usage"]
        if event.get("choices") and event["choices"][0].get("delta", {}).get("content"):
            if first is None:
                first = time.perf_counter()
            chunks += 1

    total = time.perf_counter() - started
    out_tokens = usage.get("completion_tokens") or chunks
    return {
        "ttft_s": (first - started) if first else float("nan"),
        "total_s": total,
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": out_tokens,
        "output_tps": out_tokens / total if total else float("nan"),
    }


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    """KV cache bytes for one token: 2 (K and V) x layers x kv_heads x head_dim."""
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True, help="model name the NIM reports at /v1/models")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1)
    # Nemotron Nano 9B v2 defaults; check config.json for the model you deploy.
    ap.add_argument("--layers", type=int, default=56)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype-bytes", type=int, default=2, help="2 for bf16, 1 for fp8")
    ap.add_argument("--json", metavar="FILE", help="write raw timings")
    args = ap.parse_args()

    try:
        models = requests.get(f"{args.url.rstrip('/')}/models", timeout=10).json()
        print("endpoint models:", [m["id"] for m in models.get("data", [])], "\n")
    except Exception as exc:
        print(f"cannot reach {args.url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    per_token = kv_bytes_per_token(args.layers, args.kv_heads, args.head_dim,
                                   args.dtype_bytes)
    results = {}

    for lang, prompt in PROMPTS.items():
        for _ in range(args.warmup):
            stream_once(args.url, args.model, prompt, 16)

        runs = [stream_once(args.url, args.model, prompt, args.max_tokens)
                for _ in range(args.runs)]
        results[lang] = runs

        pt = runs[0]["prompt_tokens"]
        print(f"--- {lang} ---")
        print(f"  prompt tokens        {pt}")
        print(f"  TTFT       median    {statistics.median(r['ttft_s'] for r in runs):.3f} s")
        print(f"  end-to-end median    {statistics.median(r['total_s'] for r in runs):.3f} s")
        print(f"  output tok/s median  {statistics.median(r['output_tps'] for r in runs):.1f}")
        if pt:
            print(f"  prompt KV cache      {pt * per_token / 2**20:.2f} MiB")
        print()

    pl, en = results["pl"][0]["prompt_tokens"], results["en"][0]["prompt_tokens"]
    if pl and en:
        print(f"Same content: {pl} tokens in Polish vs {en} in English "
              f"(+{pl / en - 1:.0%}).")
        print(f"Prompt KV cache: {(pl - en) * per_token / 2**20:.2f} MiB more per "
              f"session, on every session, for the same meaning.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"args": vars(args), "kv_bytes_per_token": per_token,
                       "runs": results}, fh, indent=2)
        print(f"\nwritten: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
