from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CandidateBundle:
    baseline_token_id: int
    medusa_token_ids: list[int]

    @property
    def candidate_token_ids(self) -> list[int]:
        return list(self.medusa_token_ids)


def select_token_from_logits(
    logits: torch.Tensor,
    *,
    greedy: bool = True,
    temperature: float = 1.0,
) -> int:
    if logits.dim() != 1:
        raise ValueError(f"Expected 1D logits for token selection, got shape {tuple(logits.shape)}")

    if greedy or temperature <= 0.0:
        return int(torch.argmax(logits).item())

    scaled_logits = logits / temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


def build_candidate_bundle(
    baseline_logits: torch.Tensor,
    medusa_logits: torch.Tensor,
    *,
    greedy: bool = True,
    temperature: float = 1.0,
) -> CandidateBundle:
    if baseline_logits.dim() != 1:
        raise ValueError(
            f"Expected baseline logits for one position with shape [vocab], got {tuple(baseline_logits.shape)}"
        )
    if medusa_logits.dim() != 2:
        raise ValueError(
            f"Expected medusa logits for one position with shape [num_heads, vocab], got {tuple(medusa_logits.shape)}"
        )

    baseline_token_id = select_token_from_logits(
        baseline_logits,
        greedy=greedy,
        temperature=temperature,
    )
    medusa_token_ids = [
        select_token_from_logits(head_logits, greedy=greedy, temperature=temperature)
        for head_logits in medusa_logits
    ]
    return CandidateBundle(
        baseline_token_id=baseline_token_id,
        medusa_token_ids=medusa_token_ids,
    )
