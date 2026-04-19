# Phase 5 - Evaluation

## Purpose

Compare the trained Medusa model against the baseline model on the same data.

This phase is required to tell whether the Medusa heads learned anything useful and whether the base model behavior stayed reasonable.

## Scope

Implement:
- baseline vs Medusa LM evaluation
- head-wise future-token accuracy evaluation
- generation comparison on shared prompts
- exportable result tables

## Required files

Recommended files:
- `src/pythia_medusa/eval/eval_language_modeling.py`
- `src/pythia_medusa/eval/eval_heads.py`
- `src/pythia_medusa/eval/compare_models.py`
- `src/pythia_medusa/eval/metrics.py`
- `configs/eval/eval_lm.yaml`
- `tests/test_eval_pipeline.py`

## Evaluation categories

### 1. Static language-modeling evaluation

Compare on the same validation set:
- baseline loss
- baseline perplexity
- medusa-wrapper baseline-path loss
- medusa-wrapper baseline-path perplexity

Purpose:
- verify the Medusa wrapper does not break base inference

### 2. Head-wise Medusa evaluation

For each Medusa head, evaluate:
- future-token accuracy
- top-k hit rate, optional
- per-head loss

Suggested outputs:
- `head1_acc`
- `head2_acc`
- `head3_acc`

### 3. Generation comparison

Using a fixed prompt set, collect:
- prompt
- baseline generated text
- medusa generated text
- output lengths
- generation latency
- average accept length, if available

## Output formats

Write at least:
- `json`
- `csv`

Suggested summary fields:
- model name
- checkpoint path
- dataset name
- split name
- number of examples
- loss
- perplexity
- per-head metrics

## Example comparison table

```text
split,model,loss,ppl,head1_acc,head2_acc,head3_acc
valid,baseline,3.42,30.6,,,
valid,medusa,3.45,31.5,0.62,0.41,0.29
```

## Quality checks

Flag any of the following:
- NaN or Inf metrics
- Medusa base-path loss drastically worse than baseline
- empty outputs during generation
- acceptance statistics outside valid ranges

## Tests

Add tests for:
- metric computation
- end-to-end comparison on a tiny dataset
- export of json or csv results

## Deliverables

Phase 5 is complete when:
- baseline and Medusa can be evaluated on the same data
- head-wise metrics are reported
- prompt-based text comparison works
- outputs are saved in machine-readable form

## Notes for Codex

- Reuse the same tokenizer and prompt formatting between baseline and Medusa.
- Keep generation comparison deterministic when possible.
- Avoid mixing speed results into this phase; treat speed as a separate benchmark.
