# Phase 1 - Baseline

## Purpose

Build a clean baseline around `EleutherAI/pythia-70m-deduped` before touching Medusa logic.

This phase should establish:
- model loading
- tokenizer loading
- text generation
- language-modeling evaluation
- shared prompt sets

If this phase is unstable, all later Medusa comparisons will be noisy or misleading.

## Scope

Implement:
- baseline generation script
- baseline language-modeling evaluation script
- prompt-set loader
- result export helpers

Do not implement yet:
- Medusa heads
- Medusa cache logic
- tree mask
- speculative decoding

## Required files

Recommended files:
- `src/pythia_medusa/generation/base_generate.py`
- `src/pythia_medusa/eval/eval_language_modeling.py`
- `src/pythia_medusa/data/prompt_sets.py`
- `src/pythia_medusa/utils/io.py`
- `tests/test_generation_smoke.py`

## Functional requirements

### 1. Baseline generation

Support:
- loading `EleutherAI/pythia-70m-deduped`
- prompt input from CLI or jsonl
- configurable `max_new_tokens`
- configurable `temperature`
- optional greedy mode

Output:
- generated text
- prompt text
- token counts
- generation latency

### 2. Baseline LM evaluation

Support:
- text dataset input
- tokenization
- teacher-forcing loss
- perplexity
- total evaluated tokens

Output:
- `json` summary
- optional `csv` row

### 3. Shared prompt sets

Create a small reusable prompt collection for later baseline-vs-medusa comparison.

Suggested prompt buckets:
- short factual prompts
- short continuation prompts
- medium reasoning prompts
- medium instruction prompts

## Suggested CLI examples

```bash
python -m pythia_medusa.generation.base_generate \
  --model EleutherAI/pythia-70m-deduped \
  --prompt "The capital of France is"
```

```bash
python -m pythia_medusa.eval.eval_language_modeling \
  --model EleutherAI/pythia-70m-deduped \
  --dataset path/to/valid.jsonl \
  --output outputs/baseline_eval.json
```

## Metrics to produce

- `loss`
- `perplexity`
- `num_tokens`
- `num_examples`
- `latency_sec` for generation
- `tokens_per_sec` for generation

## Tests

Add smoke tests that verify:
- model loads
- generation returns non-empty text
- evaluation returns finite loss and perplexity

## Deliverables

Phase 1 is complete when:
- baseline generation works on a single prompt
- baseline evaluation works on a small dataset
- results are saved to disk
- at least one smoke test passes

## Notes for Codex

- Do not introduce Medusa abstractions yet.
- Keep this phase simple and dependable.
- Baseline outputs here will become the reference for later comparison.
