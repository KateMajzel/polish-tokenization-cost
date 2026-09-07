#!/usr/bin/env python3
"""
Polish Tokenization Cost
========================

How much does Polish text cost in different tokenizers?

Measures tokenization density (bytes per token) of a Polish text across several
tokenizers and converts it to a reference inference cost. The metric is the one
used by the GoLLeM-45M-PL model card: bytes/token rather than perplexity —
models with different vocabularies predict units of different difficulty, so
perplexity is not comparable between them.

Usage:
    python polish_token_tax.py                         # built-in sample text
    python polish_token_tax.py --input corpus.txt      # your own file
    python polish_token_tax.py --json results.json     # write raw results
    python polish_token_tax.py --price 0.50            # USD per 1M tokens
    python polish_token_tax.py --doc-sep $'\\n\\n\\n'    # per-document percentiles

Requires:
    pip install -r requirements.txt

Access note: some repositories (Gemma, Llama, Mistral) are gated — accept the
licence on Hugging Face and authenticate with `hf auth login` or the HF_TOKEN
environment variable. Tokenizers that cannot be downloaded are skipped with a
warning; the rest are measured normally.

Provenance: the JSON output records the SHA-256 of the input text, the
`transformers` version, and the Hub commit hash of every tokenizer measured,
so a later re-run can be compared against a published results.json exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass, asdict, field
from datetime import date

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Hugging Face repository IDs. Labels are used verbatim in the README table.
# Verify IDs before running — variant names change between releases.
TOKENIZERS: dict[str, str] = {
    "GoLLeM-PL (Polish, 32k)": "KateMajzel/tokenizer-pl-32k",
    "Bielik-PL v3 / APT4 (Polish)": "speakleash/Bielik-PL-11B-v3.0-Instruct",
    "PLLuM 12B (Polish)": "CYFRAGOVPL/PLLuM-12B-base-2412",
    "Bielik 11B v3 (Mistral 32k)": "speakleash/Bielik-11B-v3.0-Instruct",
    "Gemma 2": "google/gemma-2-2b",
    "Nemotron 3 Nano 30B": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "Qwen 2.5": "Qwen/Qwen2.5-7B",
    "Mistral Nemo / Tekken (131k)": "mistralai/Mistral-Nemo-Base-2407",
    "Mistral 7B v0.1 (32k)": "mistralai/Mistral-7B-v0.1",
    "Llama 3": "meta-llama/Meta-Llama-3-8B",
    "GPT-2": "gpt2",
}

# Fallback text: several registers of Polish, heavy on diacritics and inflection.
SAMPLE_TEXT = """
Wczesnym rankiem mgła osiadała na łąkach nad Wisłą, a wieś budziła się powoli.
Gospodarze wychodzili do obejścia, żeby sprawdzić, czy nocna wichura nie zerwała
gontów z dachu stodoły.

Zgodnie z rozporządzeniem ministra właściwego do spraw informatyzacji, podmiot
przetwarzający dane osobowe obowiązany jest wdrożyć środki techniczne
i organizacyjne zapewniające bezpieczeństwo przetwarzania, uwzględniając stan
wiedzy technicznej oraz koszt wdrażania.

no i wiecie co, zamówiłam ten sprzęt trzy tygodnie temu i dalej nic, kurier
dzwonił raz, nie odebrałam bo byłam u lekarza, a teraz nikt nie wie gdzie
paczka jest. ktoś miał podobnie?

Przekształcenie Fouriera pozwala rozłożyć sygnał na składowe częstotliwościowe,
co w praktyce oznacza, że skomplikowany przebieg czasowy da się opisać skończoną
liczbą współczynników.

Śnieżnobiałe źdźbła trawy, chrząszcz brzmiący w trzcinie, gęś, jeż, łódź, ćma —
polskie diakrytyki bywają dla tokenizerów kosztowne.
""".strip()


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class Result:
    """Measurement for one tokenizer."""
    label: str
    repo: str
    revision: str | None          # Hub commit hash of the tokenizer files
    vocab_size: int               # len(tokenizer), including added tokens
    n_tokens: int
    bytes_per_token: float
    tokens_per_word: float        # word = whitespace-separated (str.split())
    roundtrip_ok: bool
    per_doc: dict[str, float] = field(default_factory=dict)  # bytes/token percentiles

    def cost(self, price_per_million: float) -> float:
        """Cost of tokenizing this text at the given price per 1M tokens."""
        return self.n_tokens / 1_000_000 * price_per_million


def repo_revision(repo: str) -> str | None:
    """Current commit hash of a Hub repository, or None if unavailable."""
    try:
        from huggingface_hub import model_info
        return model_info(repo).sha
    except Exception:
        return None


def percentiles(values: list[float]) -> dict[str, float]:
    """Median and 10th/25th/75th/90th percentiles of a list."""
    if len(values) < 2:
        return {}
    q = statistics.quantiles(values, n=100, method="inclusive")
    return {
        "p10": q[9], "p25": q[24], "median": statistics.median(values),
        "p75": q[74], "p90": q[89], "n_docs": len(values),
    }


def measure(label: str, repo: str, text: str, docs: list[str] | None) -> Result | None:
    """Metrics for one tokenizer, or None if it cannot be loaded."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
    except Exception as exc:  # gated repo, no network, bad ID
        print(f"  skipping {label} ({repo}): {type(exc).__name__}", file=sys.stderr)
        return None

    ids = tok.encode(text, add_special_tokens=False)
    n_bytes = len(text.encode("utf-8"))
    n_words = len(text.split())

    # Round-trip: does decoding reproduce the text byte for byte?
    try:
        roundtrip_ok = tok.decode(ids, skip_special_tokens=True) == text
    except Exception:
        roundtrip_ok = False

    per_doc: dict[str, float] = {}
    if docs:
        ratios = []
        for d in docs:
            n = len(tok.encode(d, add_special_tokens=False))
            if n:
                ratios.append(len(d.encode("utf-8")) / n)
        per_doc = percentiles(ratios)

    return Result(
        label=label,
        repo=repo,
        revision=repo_revision(repo),
        vocab_size=len(tok),
        n_tokens=len(ids),
        bytes_per_token=n_bytes / len(ids),
        tokens_per_word=len(ids) / n_words,
        roundtrip_ok=roundtrip_ok,
        per_doc=per_doc,
    )


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

def print_table(results: list[Result], n_bytes: int, price: float) -> None:
    """Prints a markdown table ready to paste into the README."""
    if not results:
        print("No tokenizer could be loaded.", file=sys.stderr)
        return

    baseline = max(results, key=lambda r: r.bytes_per_token)
    show_docs = any(r.per_doc for r in results)

    header = (
        "| Tokenizer | Vocab | Tokens | Bytes/token | Density | "
        "Tokens/word | Cost (USD) | Round-trip |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|:---:|"
    if show_docs:
        header += " Bytes/token p10 / median / p90 |"
        sep += "---:|"

    print(f"\nText: {n_bytes:,} bytes, reference price: {price:.2f} USD / 1M tokens\n")
    print(header)
    print(sep)

    for r in sorted(results, key=lambda r: r.bytes_per_token, reverse=True):
        density = r.bytes_per_token / baseline.bytes_per_token
        row = (
            f"| {r.label} | {r.vocab_size:,} | {r.n_tokens:,} | "
            f"{r.bytes_per_token:.3f} | {density:.2f}× | "
            f"{r.tokens_per_word:.2f} | {r.cost(price):.4f} | "
            f"{'✓' if r.roundtrip_ok else '✗'} |"
        )
        if show_docs:
            p = r.per_doc
            row += (f" {p['p10']:.2f} / {p['median']:.2f} / {p['p90']:.2f} |"
                    if p else " – |")
        print(row)

    worst = min(results, key=lambda r: r.bytes_per_token)
    if worst is not baseline:
        overhead = worst.n_tokens / baseline.n_tokens
        print(
            f"\nThe same Polish text costs {overhead:.2f}× more tokens in "
            f"\u201c{worst.label}\u201d than in \u201c{baseline.label}\u201d."
        )


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare tokenization density of Polish text across tokenizers."
    )
    parser.add_argument(
        "--input", metavar="FILE",
        help="UTF-8 text file to measure (default: built-in sample)",
    )
    parser.add_argument(
        "--doc-sep", metavar="SEP",
        help="document separator; if given, per-document bytes/token percentiles are reported",
    )
    parser.add_argument(
        "--price", type=float, default=0.50, metavar="USD",
        help="price per 1M tokens for the cost column (default 0.50)",
    )
    parser.add_argument(
        "--json", metavar="FILE",
        help="write raw results to a JSON file",
    )
    args = parser.parse_args()

    if args.input:
        with open(args.input, encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = SAMPLE_TEXT

    if not text.strip():
        print("Empty input text.", file=sys.stderr)
        return 1

    docs = None
    if args.doc_sep:
        docs = [d for d in text.split(args.doc_sep) if d.strip()]
        print(f"{len(docs)} documents found.", file=sys.stderr)

    raw = text.encode("utf-8")
    input_sha256 = hashlib.sha256(raw).hexdigest()
    print(f"Input: {len(raw):,} bytes, sha256 {input_sha256}", file=sys.stderr)

    print("Loading tokenizers...", file=sys.stderr)
    results = [
        r for label, repo in TOKENIZERS.items()
        if (r := measure(label, repo, text, docs)) is not None
    ]

    print_table(results, len(raw), args.price)

    if args.json:
        import transformers
        payload = {
            "measured_on": date.today().isoformat(),
            "transformers_version": transformers.__version__,
            "input_file": args.input,
            "input_sha256": input_sha256,
            "n_bytes": len(raw),
            "n_words": len(text.split()),
            "n_docs": len(docs) if docs else None,
            "price_per_million_usd": args.price,
            "results": [asdict(r) for r in results],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nWritten: {args.json}", file=sys.stderr)

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
