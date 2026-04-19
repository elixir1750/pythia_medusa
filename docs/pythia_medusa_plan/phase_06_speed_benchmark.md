# Phase 6 - Speed Benchmark

## Purpose

Measure optional generation-speed differences between baseline Pythia-70M and the Medusa implementation.

This phase is optional because early Medusa implementations may be correct but not yet faster.

## Scope

Implement:
- prompt-based benchmark runner
- repeated measurement
- summary aggregation
- export of latency and throughput statistics

## Required files

Recommended files:
- `src/pythia_medusa/eval/benchmark_generation.py`
- `src/pythia_medusa/utils/profiling.py`
- `configs/eval/benchmark_gen.yaml`
- `scripts/benchmark_compare.sh`

## Benchmark modes

### 1. Text mode

Use when you want to inspect outputs while also recording timing.

Record:
- prompt
- output text
- latency
- output token count

### 2. Bench mode

Use when you only care about timing and throughput.

Record:
- prompt length
- output length
- total latency
- tokens per second
- average over repeated runs

## Suggested benchmark settings

Prompt length buckets:
- short: around `32` tokens
- medium: around `128` tokens
- long: around `256` tokens

Output lengths:
- `32`
- `64`
- `128`

Repeat count:
- `5` runs per setting if possible

Sampling:
- prefer greedy or low temperature for stable comparison

## Metrics to record

Required:
- `avg_latency_sec`
- `avg_tokens_per_sec`
- `output_tokens`

Optional:
- `prefill_latency_sec`
- `decode_latency_sec`
- `ttft_sec`
- `avg_accept_length`
- `accept_p50`
- `accept_p90`

## Important interpretation note

For a first Pythia Medusa prototype, it is acceptable if:
- Medusa is only slightly faster, or
- Medusa is not faster yet

The first benchmark goal is:
- reproducible measurement
- comparable logging
- visibility into where time is spent

## Output formats

Suggested files:
- `generation_compare.jsonl`
- `benchmark_summary.json`
- optional `benchmark_summary.csv`

Example summary:

```json
{
  "baseline": {
    "avg_latency_sec": 0.82,
    "avg_tokens_per_sec": 78.4
  },
  "medusa": {
    "avg_latency_sec": 0.71,
    "avg_tokens_per_sec": 89.1,
    "avg_accept_length": 1.63
  }
}
```

## Benchmark hygiene

Try to keep the following fixed:
- same machine
- same dtype
- same prompt set
- same generation settings
- same warmup policy

If possible:
- run a short warmup before timing
- separate compile/warmup time from steady-state timing

## Deliverables

Phase 6 is complete when:
- the benchmark script runs on shared prompts
- baseline and Medusa summaries are exported
- repeated runs can be averaged

## Notes for Codex

- Treat speed benchmarking as a measurement tool, not a proof of superiority.
- Keep benchmark code separate from core generation code.
- Record enough metadata so results are reproducible later.
