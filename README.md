# Polish Tokenization Cost

**How much extra does Polish text cost in open models — and what does a language-specific tokenizer buy you?**

The same 9 MB of Polish text costs between **24% more tokens in Gemma 2** and **106% more in GPT-2** than in GoLLeM-PL, a 32k tokenizer trained on Polish. Mistral, whose vocabulary is the same size as GoLLeM-PL's, costs 72% more. Tokens are the billing unit, the context-window unit, and the training-compute unit — so this overhead is paid three times over, on every request.

**Disclosure.** GoLLeM-PL is the author's own tokenizer, published as [tokenizer-pl-32k](https://huggingface.co/KateMajzel/tokenizer-pl-32k). This repository exists so that the comparison can be checked and reproduced by anyone, and so that additional tokenizers can be dropped in with one line.

---

## Results

Measured on a held-out slice of the [SpeakLeash](https://speakleash.org/) Polish corpus: 1,999 documents, 8,981,681 bytes, ~1.25M whitespace-separated words, spanning encyclopaedic, legal, colloquial, and technical registers. "Held-out" means the slice was not used to train the GoLLeM-PL tokenizer; see [Limitations](#limitations) for what it does not mean.

| Tokenizer | Vocab | Tokens | Bytes/token | Density | Tokens/word | Overhead | Round-trip |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **GoLLeM-PL** (Polish, 32k) | 32,768 | 2,117,649 | **4.241** | 1.00× | 1.69 | — | ✓ |
| Gemma 2 | 256,000 | 2,633,816 | 3.410 | 0.80× | 2.11 | +24% | ✓ |
| Nemotron Nano 9B v2 | 131,072 | 2,880,858 | 3.118 | 0.74× | 2.30 | +36% | ✓ |
| Nemotron 3 Nano 30B | 131,072 | 2,880,858 | 3.118 | 0.74× | 2.30 | +36% | ✓ |
| Qwen 2.5 | 151,665 | 3,103,076 | 2.894 | 0.68× | 2.48 | +47% | ✓ |
| Mistral | 32,000 | 3,636,086 | 2.470 | 0.58× | 2.91 | +72% | ✓ |
| GPT-2 | 50,257 | 4,355,238 | 2.062 | 0.49× | 3.48 | +106% | ✓ |

Higher bytes/token means more text packed into each token, which means fewer tokens for the same content. Density is bytes/token relative to the densest tokenizer in the run; overhead is the extra tokens needed for the same text.

Raw output, including the Hub commit hash of every tokenizer measured, is in [`results.json`](results.json).

### What stands out

**Vocabulary size explains almost nothing.** Mistral's vocabulary (32,000) is essentially the same size as GoLLeM-PL's (32,768) and produces 72% more tokens on the same text. Gemma 2's is 7.8× larger and produces 24% more. Capacity is not what separates them — what that capacity was trained on is.

**The overhead compounds across three separate budgets.** A 36% token overhead means 36% higher inference cost at a fixed per-token price, 36% less text fitting in a fixed context window, and 36% more tokens to process when training on the same corpus.

**The overhead is a family trait, not a release artefact.** The Nemotron tokenizer is shared across the family: Nemotron 3 Nano 30B (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`) and the retired Nemotron Nano 9B v2 (`nvidia/NVIDIA-Nemotron-Nano-9B-v2`) produce exactly 2,880,858 tokens on this corpus, despite a generation and a size class between them.

**The measurement is stable across sample sizes.** The GoLLeM-PL model card reports 4.103 bytes/token on a 2,767,440-byte slice; this run measures 4.241 on the full 8.98 MB held-out set — a 3.4% difference across samples differing in size by more than 3×.

**Nothing here is lossy.** All seven tokenizers reconstruct the text byte for byte, so none of the overhead comes from dropping or normalising Polish diacritics. The difference is purely how finely each one cuts.

---

## Limitations

**Held-out is not out-of-distribution.** The test slice was not used to train the GoLLeM-PL tokenizer, but it comes from the same corpus family (SpeakLeash) as the training data. A tokenizer trained on similar text has a natural advantage on it. The numbers should be read as "Polish text of the kind the tokenizer was built for", not as a universal constant. Measuring on a genuinely different source — Polish Wikipedia, court rulings, forum posts from another collection — is the obvious next step; the script accepts any UTF-8 file.

**Specialisation cuts both ways.** A 32k vocabulary spent entirely on Polish is *worse* than Gemma, Nemotron or Qwen on English, code, and every other language. GoLLeM-PL is not a better tokenizer; it is a better tokenizer *for Polish*. The claim here is about the cost of Polish in general-purpose vocabularies, not about which tokenizer to use overall.

**The reference tokenizer is the author's.** See the disclosure above. Everything needed to check the comparison is in this repository.

**One sample, one number.** The table reports a single aggregate over 8.98 MB. The corpus is large enough that the estimate is stable (see above), but it is a point estimate. Run the script with `--doc-sep` on a file with explicit document separators to get per-document percentiles.

**Tokenizer repositories change.** Hugging Face repositories are updated in place. The script records the commit hash of every tokenizer it measures and the `transformers` version, so a re-run can be compared against `results.json` exactly.

---

## Reproducing

```bash
pip install -r requirements.txt
python polish_token_tax.py --input heldout.txt --json results.json
```

Or measure your own language — any UTF-8 file works:

```bash
python polish_token_tax.py --input your_corpus.txt
```

**Input file.** `heldout.txt` is a UTF-8 concatenation of 1,999 documents held out from the SpeakLeash corpus used to train the GoLLeM-PL tokenizer. The set is private and is not redistributed here; the hashes below identify it.

SHA-256 as read by the script (8,981,681 bytes, after Python normalises CRLF to LF):
`f6df196e74b4de596d1f5815adaa4eb567a7aac64b84f1300e02c78345780f54`

SHA-256 of the file on disk (8,981,940 bytes, Windows line endings intact):
`579eeb4c072662639a6ffb1ff02c7a0ba1d2fa26b23da116b0ab73744233ef21`

**Gated repositories.** Gemma 2 and Mistral require accepting the licence on their Hugging Face model pages, then authenticating:

```bash
export HF_TOKEN=hf_your_token
```

Tokenizers that fail to download are skipped with a warning; the rest are measured normally.

**Options**

| Flag | Meaning |
|---|---|
| `--input FILE` | UTF-8 text to measure (falls back to a built-in Polish sample) |
| `--doc-sep SEP` | Treat `SEP` as a document separator and report per-document percentiles |
| `--price USD` | Reference price per 1M tokens for the cost column (default 0.50) |
| `--json FILE` | Write raw results, tokenizer revisions and input hash as JSON |

Tokenizers are configured in the `TOKENIZERS` dictionary at the top of `polish_token_tax.py`; any tokenizer on the Hugging Face Hub can be added.

---

## Method

**Metric: bytes per token.** Perplexity cannot be compared across models with different vocabularies — each predicts units of different difficulty, so the numbers are not on the same scale. Bytes per token measures compression directly and is comparable by construction.

**One text, many tokenizers.** All tokenizers see byte-identical input, so no normalisation is needed. Special tokens are excluded (`add_special_tokens=False`) to measure the tokenizer rather than the chat template.

**Words.** "Word" means a whitespace-separated token (`str.split()`), so tokens/word counts punctuation attached to words as part of the word. It is a convenience figure; bytes/token is the primary metric and does not depend on this definition.

**Vocabulary size** is `len(tokenizer)`, which includes added special tokens. This is the size of the output space the model actually predicts over.

**Round-trip verification.** Every tokenizer is checked for exact `decode(encode(text)) == text` reconstruction.

**Cost column.** The optional cost column applies one reference price to every model, which is a normalisation rather than a market quote — real per-model pricing differs by provider. It shows how token overhead translates to spend when the rate is held constant. The density figures stand on their own without it.

---

## Files

| Path | Contents |
|---|---|
| `polish_token_tax.py` | Measurement script |
| `results.json` | Raw output from the run reported above |
| `requirements.txt` | Python dependencies |
| `k8s/nim-nemotron.yaml` | GKE deployment manifest for the Nemotron NIM |
| `docs/nim-on-gke.md` | Notes from deploying Nemotron NIM on Google Kubernetes Engine (separate from the tokenizer measurement) |

## Background

GoLLeM-PL comes from [GoLLeM-45M-PL](https://huggingface.co/KateMajzel/GoLLeM-45M-PL), a 45M-parameter Polish language model trained from scratch on a single consumer GPU as a controlled tokenizer ablation. That experiment asked whether a dedicated tokenizer produces a better model; the answer was that it produces a comparable one 2.5× faster. This repository isolates the narrower question: how much of the cost of Polish in general-purpose models is attributable to tokenization alone.

The [Bielik v3](https://arxiv.org/abs/2604.10799) team reached the same conclusion independently at 11B scale.

## Licence

Code and text in this repository: MIT. The SpeakLeash corpus is distributed under its own terms and is not included here.
