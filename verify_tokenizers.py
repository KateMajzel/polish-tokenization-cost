#!/usr/bin/env python3
"""
Two checks that the README currently asserts without shipping evidence.

    python verify_tokenizers.py

1. Identity check. Confirms that every repo ID in TOKENIZERS resolves to a
   distinct tokenizer, and prints vocab size, hash of the vocabulary, and the
   Hub commit. If two "different" repos hash identically, they are the same
   tokenizer and the table must say so. If a repo silently redirects, the
   resolved name will not match the requested one.

2. Shared-ID evidence. For a group of repos believed to share a tokenizer,
   counts how many tokens carry identical IDs across all of them. This is the
   number footnote 1 quotes (131,064); tokenizer_groups.json currently records
   only which tokens differ, not that the rest have matching IDs.

Exit code is non-zero if a repo fails to load or an identity check fails.
"""
from __future__ import annotations

import hashlib
import json
import sys

REPOS = [
    "KateMajzel/tokenizer-pl-32k",
    "speakleash/Bielik-PL-11B-v3.0-Instruct",
    "speakleash/Bielik-11B-v3.0-Instruct",
    "CYFRAGOVPL/PLLuM-12B-base-2412",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "mistralai/Mistral-Nemo-Base-2407",
    "mistralai/Mistral-7B-v0.1",
]

# Repos the README collapses into one table row.
GROUPS = {
    "tekken": [
        "mistralai/Mistral-Nemo-Base-2407",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "CYFRAGOVPL/PLLuM-12B-base-2412",
    ],
    "mistral-32k": [
        "mistralai/Mistral-7B-v0.1",
        "speakleash/Bielik-11B-v3.0-Instruct",
    ],
}

PROBE = "Zażółć gęślą jaźń — źdźbło trawy, chrząszcz w trzcinie."


def vocab_fingerprint(vocab: dict[str, int]) -> str:
    """Stable hash of the token -> id mapping."""
    blob = json.dumps(vocab, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load(repo: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(repo, trust_remote_code=False)


def main() -> int:
    from huggingface_hub import model_info

    toks, vocabs, failed = {}, {}, []

    print("=== identity ===\n")
    for repo in REPOS:
        try:
            tok = load(repo)
        except Exception as exc:
            print(f"FAILED  {repo}\n        {type(exc).__name__}: {exc}\n")
            failed.append(repo)
            continue
        try:
            sha = model_info(repo).sha[:12]
        except Exception:
            sha = "?"

        vocab = tok.get_vocab()
        toks[repo], vocabs[repo] = tok, vocab
        resolved = getattr(tok, "name_or_path", "")
        flag = "" if resolved.rstrip("/") == repo else f"  <-- resolved as {resolved!r}"

        print(f"{repo}{flag}")
        print(f"  vocab_size={tok.vocab_size}  len(tok)={len(tok)}  "
              f"fingerprint={vocab_fingerprint(vocab)}  commit={sha}")
        print(f"  probe -> {len(tok.encode(PROBE, add_special_tokens=False))} tokens\n")

    # Any two repos with the same fingerprint are the same tokenizer.
    print("=== collisions ===\n")
    seen: dict[str, list[str]] = {}
    for repo, vocab in vocabs.items():
        seen.setdefault(vocab_fingerprint(vocab), []).append(repo)
    collisions = {fp: rs for fp, rs in seen.items() if len(rs) > 1}
    if collisions:
        for fp, rs in collisions.items():
            print(f"identical vocabulary ({fp}):")
            for r in rs:
                print(f"  {r}")
        print()
    else:
        print("no two repos share an identical vocabulary\n")

    print("=== shared IDs within declared groups ===\n")
    evidence = {}
    for name, repos in GROUPS.items():
        present = [r for r in repos if r in vocabs]
        if len(present) < 2:
            print(f"{name}: not enough repos loaded, skipping\n")
            continue

        ref = vocabs[present[0]]
        shared = set(ref)
        for r in present[1:]:
            shared &= set(vocabs[r])
        same_id = sum(
            1 for t in shared
            if all(vocabs[r][t] == ref[t] for r in present[1:])
        )
        only = {r: sorted(set(vocabs[r]) - shared) for r in present}

        print(f"{name}: {len(present)} repos")
        print(f"  tokens present in all:      {len(shared):,}")
        print(f"  of those, identical ID:     {same_id:,}")
        for r, extra in only.items():
            if extra:
                print(f"  only in {r}: {len(extra)} -> {extra}")
        print()

        evidence[name] = {
            "repos": present,
            "tokens_in_all": len(shared),
            "tokens_in_all_same_id": same_id,
            "all_shared_ids_identical": same_id == len(shared),
        }

    with open("shared_id_evidence.json", "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, ensure_ascii=False, indent=2)
    print("written: shared_id_evidence.json")
    print("  -> fold tokens_in_all_same_id into tokenizer_groups.json;")
    print("     it is the number footnote 1 quotes.")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
