# Phase 3 - Medusa Generation

## Purpose

Implement a minimal Medusa generation path for Pythia-70M.

This phase is where Medusa stops being just a multi-head predictor and becomes a decoding system.

## Scope

Implement:
- candidate generation from baseline and Medusa logits
- tree or tree-like verification helpers
- acceptance logic
- iterative Medusa decoding loop

At this stage, correctness matters more than speed.

## Important migration note

Pythia uses the GPTNeoX family, not LLaMA or Mistral.

Therefore:
- do not blindly copy `modeling_llama_kv.py`
- inspect GPTNeoX attention-mask handling and cache format
- keep the design modular so a simplified verification loop can be used first

## Required files

Recommended files:
- `src/pythia_medusa/generation/tree_utils.py`
- `src/pythia_medusa/generation/candidate_utils.py`
- `src/pythia_medusa/generation/posterior_utils.py`
- `src/pythia_medusa/generation/medusa_generate.py`
- optional `src/pythia_medusa/models/modeling_gpt_neox_medusa.py`
- `tests/test_generation_smoke.py`

## Required concepts

### 1. Tree mask

Document locally what tree mask means.

The implementation should define:
- root node semantics
- ancestor visibility
- sibling invisibility
- mask shape

Suggested rule:
- each node attends to itself
- each node attends to the root
- each node attends to its ancestors
- unrelated branches are masked

### 2. Candidate generation

Given:
- baseline logits
- Medusa logits

Produce:
- next-step candidates
- tree candidates or an equivalent verification structure

Support at least one sampling mode:
- greedy, or
- low-temperature typical sampling

### 3. Acceptance logic

The decoder should:
- verify candidates with the base model
- compute accepted prefix length
- append the accepted prefix to the running sequence

Record:
- accept length per round
- number of rounds
- total generated tokens

## Two valid implementation paths

### Path A: simplified first version

Use a simpler verification loop without full custom GPTNeoX tree attention.

Pros:
- much easier to get correct
- good for experiments and debugging

Cons:
- may not provide speedup

### Path B: fuller Medusa-style tree verification

Add GPTNeoX-specific support for:
- Medusa mask injection
- Medusa-aware cache handling
- tree decoding in a single verification step

Pros:
- closer to the original Medusa idea

Cons:
- more engineering risk

Recommendation:
- implement Path A first
- keep function boundaries ready for Path B later

Repository status update:
- the repository now defaults to a linear `tree_verify` path
- Medusa candidate chains are verified in a single base-model forward with an explicit tree mask
- a `serial` verification fallback is still kept for debugging
- multi-branch tree expansion and Medusa-aware KV cache are still future work

## Metrics to log during generation

- prompt length
- output length
- total rounds
- average accept length
- accept length histogram
- latency
- tokens per second

## Tests

Add smoke tests for:
- Medusa generation returns text
- generation loop terminates
- accept length is always valid
- greedy generation on a tiny prompt does not crash

## Deliverables

Phase 3 is complete when:
- Medusa generation produces text
- acceptance statistics are recorded
- generation works on fixed prompt sets

## Notes for Codex

- If full GPTNeoX tree-mask support is difficult, implement a correctness-first version first.
- Keep tree-mask logic isolated and well documented.
- Do not mix evaluation logic into generation code.
