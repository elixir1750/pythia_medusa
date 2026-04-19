from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcceptanceResult:
    candidate_token_ids: list[int]
    verified_token_ids: list[int]
    accepted_token_ids: list[int]
    tokens_to_append: list[int]
    fallback_token_id: int
    accepted_length: int


def compute_acceptance(
    candidate_token_ids: list[int],
    verified_token_ids: list[int],
    fallback_token_id: int,
) -> AcceptanceResult:
    accepted_length = 0
    for candidate_token_id, verified_token_id in zip(candidate_token_ids, verified_token_ids):
        if candidate_token_id != verified_token_id:
            break
        accepted_length += 1

    accepted_token_ids = candidate_token_ids[:accepted_length]
    tokens_to_append = accepted_token_ids if accepted_token_ids else [fallback_token_id]
    return AcceptanceResult(
        candidate_token_ids=list(candidate_token_ids),
        verified_token_ids=list(verified_token_ids),
        accepted_token_ids=accepted_token_ids,
        tokens_to_append=tokens_to_append,
        fallback_token_id=fallback_token_id,
        accepted_length=accepted_length,
    )
