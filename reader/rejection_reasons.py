from __future__ import annotations


REJECTION_MESSAGES = {
    "LOW_READER_SCORE": "Reader evidence was below the configured minimum.",
    "SPAN_BOUNDARY_INCOMPLETE": "The span starts or ends inside a semantic phrase or named entity.",
    "ANSWER_TYPE_MISMATCH": "The candidate did not match the expected answer type.",
    "LOW_RANKING_SCORE": "The candidate ranking score was below the configured minimum.",
    "NO_VALID_SPAN": "Reader did not produce a valid answer span.",
    "EVIDENCE_UNSUPPORTED": "The proposed answer was not supported by the source passage.",
    "INSUFFICIENT_FALLBACK_EVIDENCE": "Fallback did not provide strong typed evidence.",
    "LOCATION_RELATION_MISMATCH": "The candidate does not fill the requested location relation.",
    "RELATION_MISMATCH": "The candidate mentions the topic but does not express the requested relation.",
    "CAUSE_RELATION_MISMATCH": "The evidence has a causal form but not the requested direction.",
    "CAUSE_TARGET_MISMATCH": "The causal evidence does not explain the requested target.",
    "CAUSE_SUBJECT_MISMATCH": "The causal evidence concerns a different subject.",
    "CAUSE_PHRASE_NOT_FOUND": "No standalone cause phrase for the requested relation was found.",
    "CAUSE_RELATION_NOT_FOUND": "No cause/effect relation for the requested target was found.",
    "CAUSE_EFFECT_REPETITION": "The candidate repeats the effect instead of its cause.",
    "TIME_SUBJECT_MISMATCH": "The time expression belongs to a different subject.",
    "BIRTH_TIME_RELATION_MISMATCH": "The candidate time is not connected to the subject's birth.",
    "DEATH_TIME_RELATION_MISMATCH": "The candidate time is not connected to the subject's death.",
    "PURPOSE_SUBJECT_MISMATCH": "The purpose belongs to a different subject or action.",
    "PURPOSE_RELATION_NOT_FOUND": "No purpose relation for the requested action was found.",
    "SUBJECT_MISMATCH": "The evidence concerns a different subject.",
    "LOCATION_SUBJECT_MISMATCH": "The location evidence concerns a different subject.",
    "EVENT_TIME_RELATION_MISMATCH": "The candidate time is not connected to the requested event.",
    "RELATION_UNSUPPORTED": "No semantic validator is registered for the requested relation.",
    "DANGLING_CONNECTOR": "The answer ends with an incomplete connector.",
    "INCOMPLETE_CLAUSE": "The proposed answer is not a complete standalone clause.",
    "INCOMPLETE_RELATION": "The answer contains only part of the requested relation.",
    "RETRIEVAL_MISS": "Retriever returned no passage with a positive score.",
    "LOWER_RANKING_SCORE": "A stronger valid candidate was selected.",
    "NO_ANSWER": "No candidate satisfied the answer acceptance gates.",
}


REJECTION_DEBUG_PRIORITY = {
    "LOW_RANKING_SCORE": 6,
    "SPAN_BOUNDARY_INCOMPLETE": 5,
    "INCOMPLETE_RELATION": 5,
    "LOCATION_RELATION_MISMATCH": 5,
    "RELATION_MISMATCH": 5,
    "CAUSE_RELATION_MISMATCH": 5,
    "CAUSE_TARGET_MISMATCH": 5,
    "CAUSE_SUBJECT_MISMATCH": 5,
    "CAUSE_PHRASE_NOT_FOUND": 5,
    "CAUSE_RELATION_NOT_FOUND": 5,
    "CAUSE_EFFECT_REPETITION": 5,
    "TIME_SUBJECT_MISMATCH": 5,
    "BIRTH_TIME_RELATION_MISMATCH": 5,
    "DEATH_TIME_RELATION_MISMATCH": 5,
    "PURPOSE_SUBJECT_MISMATCH": 5,
    "PURPOSE_RELATION_NOT_FOUND": 5,
    "DANGLING_CONNECTOR": 5,
    "INCOMPLETE_CLAUSE": 5,
    "ANSWER_TYPE_MISMATCH": 4,
    "INSUFFICIENT_FALLBACK_EVIDENCE": 3,
    "EVIDENCE_UNSUPPORTED": 2,
    "NO_VALID_SPAN": 1,
}


__all__ = ["REJECTION_DEBUG_PRIORITY", "REJECTION_MESSAGES"]
