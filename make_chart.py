#!/usr/bin/env python3
"""Bar chart of tokenization density, generated from results.json.

    python make_chart.py results.json density.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Table rows, in README order. Repos sharing a tokenizer are collapsed here.
ROWS = [
    ("GoLLeM-PL\n32k, Polish",        "KateMajzel/tokenizer-pl-32k"),
    ("Bielik-PL\n32k, Polish",        "speakleash/Bielik-PL-11B-v3.0-Instruct"),
    ("Gemma 2\n256k",                 "google/gemma-2-2b"),
    ("Tekken\n131k",                  "mistralai/Mistral-Nemo-Base-2407"),
    ("Qwen 2.5\n152k",                "Qwen/Qwen2.5-7B"),
    ("Llama 3\n128k",                 "meta-llama/Meta-Llama-3-8B"),
    ("Mistral 7B\n32k",               "mistralai/Mistral-7B-v0.1"),
    ("GPT-2\n50k",                    "gpt2"),
]

POLISH = {"KateMajzel/tokenizer-pl-32k", "speakleash/Bielik-PL-11B-v3.0-Instruct"}


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "density.png"

    data = json.load(open(src, encoding="utf-8"))
    by_repo = {r["repo"]: r for r in data["results"]}

    missing = [repo for _, repo in ROWS if repo not in by_repo]
    if missing:
        print("missing from results.json:", *missing, sep="\n  ")
        return 1

    labels = [label for label, _ in ROWS]
    values = [by_repo[repo]["bytes_per_token"] for _, repo in ROWS]
    base = values[0]
    colours = ["#76b900" if repo in POLISH else "#9aa0a6" for _, repo in ROWS]

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    bars = ax.bar(labels, values, color=colours, width=0.68)

    for bar, value in zip(bars, values):
        overhead = base / value - 1
        tag = f"{value:.2f}" if overhead < 1e-9 else f"{value:.2f}\n+{overhead:.0%} tokens"
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.06, tag,
                ha="center", va="bottom", fontsize=8.5, linespacing=1.35)

    ax.axhline(base, color="#76b900", lw=0.9, ls="--", alpha=0.55, zorder=0)
    ax.set_ylabel("bytes per token  (higher is denser)")
    ax.set_title("Polish text: how much each tokenizer packs into one token",
                 fontsize=12, pad=12)
    ax.set_ylim(0, max(values) * 1.28)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)

    fig.text(0.5, 0.005,
             f"8.98 MB held out from SpeakLeash · {data['n_bytes']:,} bytes · "
             "green = trained on Polish",
             ha="center", fontsize=7.5, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
