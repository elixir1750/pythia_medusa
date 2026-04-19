# Phase 2 - Medusa Structure

## Purpose

Add the Medusa-1 model structure on top of Pythia-70M without yet requiring full Medusa decoding.

The goal of this phase is to make the model produce:
- baseline logits
- Medusa-head logits for future-token prediction

## Scope

Implement:
- Medusa config
- residual block
- Medusa heads
- Medusa wrapper model for Pythia / GPTNeoX
- checkpoint save/load conventions for Medusa heads

Do not require yet:
- full tree verification
- speed optimization
- final decoding parity with the Medusa paper

## Required files

Recommended files:
- `src/pythia_medusa/models/medusa_config.py`
- `src/pythia_medusa/models/medusa_heads.py`
- `src/pythia_medusa/models/medusa_model.py`
- `tests/test_model_shapes.py`
- `tests/test_medusa_forward.py`

## Required concepts

### 1. MedusaConfig

Should include at least:
- `base_model_name_or_path`
- `medusa_num_heads`
- `medusa_num_layers`
- `hidden_size`
- `vocab_size`

Optional fields:
- `medusa_hidden_size`
- `version`
- `tree_name`

### 2. Medusa heads

Initial implementation recommendation:
- one head per future offset
- each head = `N * ResBlock + Linear(hidden_size -> vocab_size)`

For a minimal setup:
- `medusa_num_heads = 3`
- `medusa_num_layers = 1`

### 3. Medusa wrapper model

Implement a wrapper that:
- loads the base GPTNeoX causal LM
- exposes tokenizer access
- supports baseline forward
- supports medusa forward

Recommended forward modes:
- `medusa_forward=False`: behave like baseline causal LM
- `medusa_forward=True`: return Medusa logits
- `output_orig=True`: also return original model logits

## Interface expectations

Expected medusa forward output:
- `medusa_logits`: shape `[num_heads, batch, seq, vocab]`
- optionally original logits and raw base outputs

The wrapper must be explicit about where it gets:
- hidden states
- lm head
- tokenizer

## Recommended implementation strategy

1. Start from a plain wrapper around `GPTNeoXForCausalLM`.
2. Extract final hidden states from the base model.
3. Run each Medusa head on the same hidden states.
4. Return stacked logits.

## Tests

Add tests for:
- Medusa head stack output shape
- Medusa wrapper output shape
- `output_orig=True` path
- save/load roundtrip for Medusa config or checkpoint metadata

## Deliverables

Phase 2 is complete when:
- a prompt can be passed through the model
- original logits are available
- Medusa logits are available
- shape tests pass

## Notes for Codex

- Prioritize clear interfaces.
- Keep the Medusa wrapper independent from the generation algorithm.
- Avoid baking tree-decoding logic into the model class at this stage.
