"""Apply entity linkers to NER output."""

from __future__ import annotations

from app.src.nel.linker import EntityLinker


def biencoder_inference(
    ner_results: list[list[list[dict]]],
    linkers: list[EntityLinker],
) -> list[list[list[dict]]]:
    """Add ``code``, ``term``, ``nel_score`` and ``nel_method`` to every annotation.

    Parameters
    ----------
    ner_results:
        Nested per entity type → per document → per annotation::

            [ // entity type level (one per NER model, aligned with `linkers`)
                [ // document level
                    {'start': 136, 'end': 159, 'ner_score': 0.9999,
                     'span': 'varicela con meningitis', 'ner_class': 'ENFERMEDAD'}
                ],
                ...
            ]

    linkers:
        One ``EntityLinker`` per entity type, in the same order as
        ``ner_results``.

    Returns
    -------
    The same nested structure, annotated in place. Annotations whose span found
    no candidate keep no linking keys at all; the CDM formatter emits those as
    null rather than inventing a code.
    """
    if len(ner_results) != len(linkers):
        raise ValueError(
            f"Got {len(ner_results)} entity-type result groups but {len(linkers)} "
            "linkers — they must correspond one to one."
        )

    for docs, linker in zip(ner_results, linkers):
        mentions = [ann["span"] for doc in docs for ann in doc]
        if not mentions:
            continue  # no mentions of this entity type in this batch

        linked = linker.link_texts(mentions)

        for doc in docs:
            for ann in doc:
                candidate = linked.get(ann["span"])
                if candidate is not None:
                    ann["code"] = candidate.code
                    ann["term"] = candidate.term
                    ann["nel_score"] = round(candidate.score, 4)
                    # Which retriever won, not which ones ran: with the methods
                    # now selectable per run, a code no longer implies the
                    # bi-encoder produced it. Becomes nel_component_type.
                    ann["nel_method"] = candidate.effective_method

    return ner_results
