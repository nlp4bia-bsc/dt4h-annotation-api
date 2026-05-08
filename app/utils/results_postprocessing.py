import numpy as np

# =============================================================================
# V1 INFERENCE
# =============================================================================

# --- Entity alignment --------------------------------------------------------

def align_results(results_pre: list[dict], added_spaces: list[int], start_sent_offset: int) -> list[dict]:
    """
    Realign NER results produced on pretokenized text back to the original text.
    - Fixes character offsets (start, end)
    - Removes artificially added spaces inside entity spans
    - Normalizes output field names
    """

    def spaces_before(pos: int) -> int:
        """Count artificial spaces that appear before a given position."""
        return sum(1 for space_pos in added_spaces if space_pos < pos)

    def spaces_inside_span(start: int, end: int) -> set[int]:
        """Return the set of artificial space positions strictly inside an entity span."""
        return {space_pos for space_pos in added_spaces if start < space_pos < end}

    def clean_surface_form(word: str, span_start: int, artificial_spaces: set[int]) -> str:
        """Strip leading/trailing whitespace and remove any artificial spaces from the word."""
        return "".join(
            char
            for i, char in enumerate(word.strip())
            if (i + span_start) not in artificial_spaces
        )

    aligned_results = []
    for ent in results_pre:
        start, end = ent["start"], ent["end"]

        artificial_spaces = spaces_inside_span(start, end)
        cleaned_word = clean_surface_form(ent["word"], start, artificial_spaces)

        aligned = ent.copy()
        aligned["span"] = cleaned_word
        aligned["ner_class"] = aligned.pop("entity_group")
        aligned["start"] = start_sent_offset + start - spaces_before(start)
        aligned["end"]   = start_sent_offset + end   - spaces_before(end)
        aligned.pop("word", None)

        aligned_results.append(aligned)

    return aligned_results

# =============================================================================
# V2 INFERENCE  
# =============================================================================

# --- Entity merging ----------------------------------------------------------


def merge_contiguous_entities(
    entities: list[dict],
    text: str,
    allow_space: bool = True,
    score_mode: str = "mean",
) -> list[dict]:
    """
    Merge adjacent predicted entities that share the same label into a single
    entity, updating the span text and aggregating scores.

    Two entities are considered mergeable when they belong to the same document,
    carry the same label, and are either directly contiguous or separated by a
    single space (if *allow_space* is True).

    Args:
        entities:    Flat list of entity dicts as produced by :func:`_predict_chunks`.
                     Must contain keys: ``filename``, ``label``, ``start``, ``end``, ``score``.
        text:        Original input text, used to recompute ``span`` after merging.
        allow_space: If True, entities separated by exactly one space character
                     are also merged. Defaults to True.
        score_mode:  How to aggregate scores of merged entities.
                     One of ``"mean"`` (default), ``"max"``, or ``"min"``.

    Returns:
        A new list of entity dicts with contiguous same-label entities fused.
    """
    if not entities:
        return []

    entities = sorted(entities, key=lambda e: (e["filename"], e["start"], e["end"]))

    def _aggregate(scores: list[float]) -> float:
        if score_mode == "max":
            return float(np.max(scores))
        if score_mode == "min":
            return float(np.min(scores))
        return float(np.mean(scores))

    merged = []
    current = {**entities[0], "_scores": [entities[0]["ner_score"]]}

    for entity in entities[1:]:
        same_doc   = entity["filename"] == current["filename"]
        same_label = entity["ner_class"]    == current["ner_class"]
        contiguous = entity["start"]    == current["end"]
        space_gap  = allow_space and entity["start"] == current["end"] + 1

        if same_doc and same_label and (contiguous or space_gap):
            current["end"] = entity["end"]
            current["span"] = text[current["start"]:current["end"]]
            current["_scores"].append(entity["ner_score"])
        else:
            current["ner_score"] = _aggregate(current.pop("_scores"))
            merged.append(current)
            current = {**entity, "_scores": [entity["ner_score"]]}

    current["ner_score"] = _aggregate(current.pop("_scores"))
    merged.append(current)
    return merged


# =============================================================================
# ALL VERSIONS
# =============================================================================

# --- Entity aggregation ------------------------------------------------------

def join_all_entities(results: list[list[list[dict]]]) -> list[list[dict]]:
    num_texts = len(results[0])  # number of documents
    entities_all = []

    for text_idx in range(num_texts):
        entities_file = []
        for model_idx in range(len(results)):
            entities_file.extend(results[model_idx][text_idx])
        entities_file = sorted(entities_file, key=lambda x: (x['start'], -x['end']))
        entities_all.append(entities_file)
    return entities_all

