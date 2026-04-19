# Pythia-70M Medusa-1 Staged Plan

This folder contains a staged implementation plan for building a minimal Medusa-1 system on top of `EleutherAI/pythia-70m-deduped`.

Audience:
- Human readers who want a concrete implementation roadmap
- Codex-style coding agents that need local, explicit specs

Goals:
- Build a correctness-first Medusa-1 implementation
- Compare against the baseline model on the same data
- Optionally benchmark generation speed

Recommended reading order:
1. [Phase 1 - Baseline](./phase_01_baseline.md)
2. [Phase 2 - Medusa Structure](./phase_02_medusa_structure.md)
3. [Phase 3 - Medusa Generation](./phase_03_medusa_generation.md)
4. [Phase 4 - Training](./phase_04_training.md)
5. [Phase 5 - Evaluation](./phase_05_evaluation.md)
6. [Phase 6 - Speed Benchmark](./phase_06_speed_benchmark.md)

Global rules:
- Prefer minimal working implementations over premature optimization.
- Default to batch size `1` for generation.
- Keep baseline and medusa evaluation on the same prompts and datasets.
- Save all experiment outputs as `json`, `jsonl`, or `csv`.
- Add smoke tests as soon as each stage becomes runnable.

Suggested target project layout:

```text
pythia_medusa/
  README.md
  requirements.txt
  pyproject.toml

  configs/
    model/
    train/
    eval/

  src/
    pythia_medusa/
      models/
      generation/
      training/
      eval/
      data/
      utils/

  scripts/
  tests/
  outputs/
```

Suggested implementation order:
1. Finish baseline generation and baseline LM evaluation.
2. Add Medusa config and Medusa heads.
3. Make Medusa forward pass work.
4. Add training for heads only.
5. Add baseline vs Medusa evaluation.
6. Add optional generation-speed benchmark.

Definition of done for the whole plan:
- Baseline generation works
- Medusa generation works
- Medusa heads can be trained
- Baseline vs Medusa metrics can be exported
- Optional speed benchmark can be run on fixed prompts
